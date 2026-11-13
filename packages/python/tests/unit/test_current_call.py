import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import CallInfo, Engine, EngineTools, Stop, current_call
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import (
    Decision,
    Event,
    Generation,
    Run,
    RunStarted,
    ToolCallStarted,
    ToolResultReady,
)
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Answering, Scripted, text_reply
from .trees import add_agent, provided


def tree(
    root: Path, *, run_tools: Mapping[str, list[str]] | None = None, **delegates: list[str]
) -> Path:
    for name, targets in delegates.items():
        entry: dict[str, Any] = {
            "kind": "generation",
            "model": "strong",
            "description": name.title(),
            "instructions": f"{name}.md",
        }
        if targets:
            entry["orchestration"] = {"delegate_to": targets}
        if run_tools and name in run_tools:
            entry["tools"] = {"allow": run_tools[name]}
        add_agent(root, name, entry, files={f"{name}.md": f"You are {name}.\n"})
    return root


def use(*calls: ToolCall) -> Reply:
    return Reply(text="", tool_calls=list(calls), stop_reason="tool_use")


class Seen:
    def __init__(self) -> None:
        self.calls: list[CallInfo | None] = []
        self.later: list[asyncio.Task[CallInfo | None]] = []

    async def body(self, args: dict[str, Any]) -> str:
        self.calls.append(current_call())

        async def afterwards() -> CallInfo | None:
            await asyncio.sleep(0)
            return current_call()

        self.later.append(asyncio.create_task(afterwards()))
        return "seen"

    def tools(self, name: str = "probe") -> Tools:
        return Tools([Tool(name, "Look", body=self.body)])


async def test_a_body_learns_the_call_it_serves(tmp_path: Path) -> None:
    seen = Seen()
    asked: list[CallInfo | None] = []
    started: list[CallInfo | None] = []

    async def approve(agent: str, call: ToolCall) -> bool:
        asked.append(current_call())
        return True

    async def watch(event: Event) -> None:
        if isinstance(event, ToolCallStarted):
            started.append(current_call())

    llm = Scripted(
        use(ToolCall(id="p1", name="probe", args={}), ToolCall(id="p2", name="probe", args={})),
        text_reply("Done."),
    )
    engine = Engine(
        tree(tmp_path, desk=[], run_tools={"desk": ["probe"]}),
        providers=provided({"strong": llm, "judge": None}),
    )
    run = await engine.run("desk", "acme", "x", tools=seen.tools(), approve=approve, on_event=watch)
    assert isinstance(run, Generation)
    expected = {
        CallInfo(run_id=run.run_id, agent="desk", scope="acme", tool="probe", call_id=call_id)
        for call_id in ("p1", "p2")
    }
    assert set(seen.calls) == set(asked) == set(started) == expected
    assert {await task for task in seen.later} == expected


async def test_a_delegated_run_s_call_names_the_delegated_run(tmp_path: Path) -> None:
    seen = Seen()
    llm = Scripted(
        use(ToolCall(id="d1", name="Delegate", args={"agent": "child", "input": "go"})),
        use(ToolCall(id="c1", name="probe", args={})),
        text_reply("child done"),
        text_reply("lead done"),
    )
    engine = Engine(
        tree(tmp_path, lead=["child"], child=[], run_tools={"child": ["probe"]}),
        providers=provided({"strong": llm, "judge": None}),
    )
    run = await engine.run("lead", "acme", "x", tools=seen.tools())
    assert isinstance(run, Generation)
    [child] = run.children
    assert seen.calls == [
        CallInfo(run_id=child.run_id, agent="child", scope="acme", tool="probe", call_id="c1")
    ]


async def test_a_consult_s_call_names_the_consult_and_its_body_the_caller(tmp_path: Path) -> None:
    inner = Seen()
    back = Engine(
        tree(tmp_path / "back", billing=[], run_tools={"billing": ["probe"]}),
        providers=provided(
            {
                "strong": Scripted(
                    use(ToolCall(id="i1", name="probe", args={})), text_reply("Paid.")
                ),
                "judge": None,
            }
        ),
    )
    consult = back.tool("billing", scope="acme", tools=inner.tools())
    outer: list[CallInfo | None] = []

    async def asking(args: dict[str, Any]) -> Any:
        outer.append(current_call())
        return await consult.dispatch("billing", {"input": "Was it paid?"})

    front = Engine(
        tree(tmp_path / "front", support=[], run_tools={"support": ["ask"]}),
        providers=provided(
            {
                "strong": Scripted(
                    use(ToolCall(id="o1", name="ask", args={})), text_reply("Done.")
                ),
                "judge": None,
            }
        ),
    )
    run = await front.run("support", "acme", "x", tools=Tools([Tool("ask", "Ask", body=asking)]))
    assert isinstance(run, Generation)
    [mine] = outer
    assert mine == CallInfo(
        run_id=run.run_id, agent="support", scope="acme", tool="ask", call_id="o1"
    )
    [theirs] = inner.calls
    assert theirs is not None and theirs.agent == "billing" and theirs.call_id == "i1"
    assert theirs.run_id != run.run_id


async def test_outside_a_run_there_is_no_call(tmp_path: Path) -> None:
    seen = Seen()
    assert current_call() is None
    await seen.tools().dispatch("probe", {})
    engine = Engine(
        tree(tmp_path, desk=["desk"], run_tools={"desk": ["probe"]}),
        providers=provided({"strong": Scripted(text_reply("x")), "judge": None}),
    )
    surface = EngineTools(engine, agent="desk", scope="acme", tools=seen.tools())
    await surface.dispatch("probe", {})
    assert seen.calls == [None, None]


class Slow:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls = 0
        self.entered = asyncio.Event()

    async def generate(self, system: str, messages: list[Any], tools: list[Any]) -> Reply:
        self.calls += 1
        if self.calls > 1:
            return text_reply("ok")
        self.entered.set()
        await asyncio.sleep(self.seconds)
        return use(ToolCall(id="n1", name="ping", args={}))


async def ping(args: dict[str, Any]) -> str:
    return "pong"


async def test_a_consult_from_approve_stays_out_of_the_run_s_stop(tmp_path: Path) -> None:
    notifier = Slow(0.3)
    notify = Engine(
        tree(tmp_path / "back", notifier=[], run_tools={"notifier": ["ping"]}),
        providers=provided({"strong": notifier, "judge": None}),
    ).tool("notifier", scope="acme", tools=Tools([Tool("ping", "Ping", body=ping)]))

    async def approve(agent: str, call: ToolCall) -> bool:
        await notify.dispatch("notifier", {"input": "May it refund?"})
        return True

    async def refund(args: dict[str, Any]) -> dict[str, str]:
        return {"refunded": "R-1"}

    llm = Scripted(use(ToolCall(id="r1", name="refund", args={})), text_reply("never"))
    engine = Engine(
        tree(tmp_path / "front", desk=[], run_tools={"desk": ["refund"]}),
        providers=provided({"strong": llm, "judge": None}),
    )
    stop = Stop()
    task = asyncio.create_task(
        engine.run(
            "desk",
            "acme",
            "x",
            tools=Tools([Tool("refund", "Refund", body=refund)]),
            approve=approve,
            stop=stop,
        )
    )
    await asyncio.wait_for(notifier.entered.wait(), 2)
    stop.after_step()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.stopped == "after_step"
    step = run.steps[0]
    assert (step.approved, step.stopped, step.is_error) == (True, False, False)
    assert "R-1" in str(step.result)


async def test_a_cut_call_s_result_handler_sees_the_call(tmp_path: Path) -> None:
    started = asyncio.Event()
    finished: list[CallInfo | None] = []

    async def slow(args: dict[str, Any]) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    async def watch(event: Event) -> None:
        if isinstance(event, ToolResultReady):
            finished.append(current_call())

    llm = Scripted(use(ToolCall(id="s1", name="slow", args={})), text_reply("never"))
    engine = Engine(
        tree(tmp_path, desk=[], run_tools={"desk": ["slow"]}),
        providers=provided({"strong": llm, "judge": None}),
    )
    stop = Stop()
    task = asyncio.create_task(
        engine.run(
            "desk",
            "acme",
            "x",
            tools=Tools([Tool("slow", "Slow", body=slow)]),
            on_event=watch,
            stop=stop,
        )
    )
    await asyncio.wait_for(started.wait(), 2)
    stop.now()
    run = await asyncio.wait_for(task, 2)
    assert isinstance(run, Generation) and run.steps[0].stopped
    assert finished == [
        CallInfo(run_id=run.run_id, agent="desk", scope="acme", tool="slow", call_id="s1")
    ]


class Keeping:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)

    def of(self, agent: str) -> Generation | Decision:
        [found] = [r for r in self.runs if r.agent == agent]
        assert isinstance(found, Generation | Decision)
        return found


def router(root: Path, name: str, target: str) -> None:
    add_agent(
        root,
        name,
        {
            "kind": "decision",
            "model": "judge",
            "instructions": f"{name}.md",
            "questions": [f"{name}/*.json"],
            "orchestration": {"route": {"by": "go", "to": {"next": target}}},
        },
        files={
            f"{name}.md": "",
            f"{name}/go.json": '{"type": "choice", "instructions": "Where?", "options": {"stay": "Keep it", "next": "Pass it on"}}',
        },
    )


def delegate(agent: str, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="Delegate", args={"agent": agent, "input": "go"})


async def test_a_run_outside_any_call_records_none(tmp_path: Path) -> None:
    keeping = Keeping()
    engine = Engine(
        tree(tmp_path, desk=[]),
        refinery=keeping,
        providers=provided({"strong": Scripted(text_reply("x")), "judge": None}),
    )
    run = await engine.run("desk", "acme", "x")
    assert run.call is None


async def parallel(tmp_path: Path, events: list[Event] | None = None) -> Generation:
    async def watch(event: Event) -> None:
        if events is not None:
            events.append(event)

    llm = Scripted(
        use(delegate("child", "d1"), delegate("child", "d2")),
        text_reply("one"),
        text_reply("two"),
        text_reply("done"),
    )
    engine = Engine(
        tree(tmp_path, lead=["child"], child=[]), providers=provided({"strong": llm, "judge": None})
    )
    run = await engine.run("lead", "acme", "x", on_event=watch)
    assert isinstance(run, Generation)
    return run


async def test_parallel_delegated_runs_record_their_own_delegate_calls(tmp_path: Path) -> None:
    run = await parallel(tmp_path)
    assert {c.call.call_id for c in run.children if c.call} == {"d1", "d2"}
    assert all(
        c.call is not None and c.call.run_id == run.run_id and c.call.tool == "Delegate"
        for c in run.children
    )


async def test_call_matches_run_started_wherever_both_exist(tmp_path: Path) -> None:
    events: list[Event] = []
    run = await parallel(tmp_path, events)
    started = [e for e in events if isinstance(e, RunStarted)]
    by_id = {r.run_id: r for r in [run, *run.children]}
    assert len(started) == 3
    assert all(by_id[e.run_id].call == e.call for e in started)


async def test_a_routed_run_records_the_call_its_router_worked_for(tmp_path: Path) -> None:
    tree(tmp_path, lead=["r"], target=[])
    router(tmp_path, "r", "target")
    llm = Scripted(use(delegate("r", "d1")), text_reply("routed"), text_reply("done"))
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": Answering()}))
    run = await engine.run("lead", "acme", "x")
    assert isinstance(run, Generation)
    [routing] = run.children
    assert isinstance(routing, Decision) and routing.routed_to is not None
    assert routing.call is not None and routing.call.call_id == "d1"
    assert routing.routed_to.call == routing.call


def consulting(
    tmp_path: Path, keeping: Keeping, *replies: Reply, tool: str | None = "billing"
) -> Engine:
    return Engine(
        tree(tmp_path, support=[], billing=[], run_tools={"support": [tool]} if tool else None),
        refinery=keeping,
        providers=provided({"strong": Scripted(*replies), "judge": None}),
    )


async def test_a_consult_inside_a_run_records_the_callers_call(tmp_path: Path) -> None:
    keeping = Keeping()
    engine = consulting(
        tmp_path,
        keeping,
        use(ToolCall(id="b1", name="billing", args={"input": "paid?"})),
        text_reply("Paid."),
        text_reply("Done."),
    )
    await engine.run("support", "acme", "x", tools=engine.tool("billing", scope="acme"))
    outer, consult = keeping.of("support"), keeping.of("billing")
    assert consult.call is not None
    assert consult.call.run_id == outer.run_id and consult.call.call_id == "b1"


async def test_a_consult_inside_a_run_that_is_not_recorded_is_not_recorded(
    tmp_path: Path,
) -> None:
    keeping = Keeping()
    engine = consulting(
        tmp_path,
        keeping,
        use(ToolCall(id="b1", name="billing", args={"input": "paid?"})),
        text_reply("Paid."),
        text_reply("Done."),
    )
    run = await engine.run(
        "support", "acme", "x", tools=engine.tool("billing", scope="acme"), record=False
    )
    assert isinstance(run, Generation) and run.steps[0].result == "Paid."
    assert keeping.runs == []


async def test_a_delegate_through_engine_tools_inside_a_run_that_is_not_recorded_is_not_recorded(
    tmp_path: Path,
) -> None:
    keeping = Keeping()
    engine = Engine(
        tree(tmp_path, support=["billing"], billing=[], run_tools={"support": ["ask"]}),
        refinery=keeping,
        providers=provided(
            {
                "strong": Scripted(
                    use(ToolCall(id="a1", name="ask", args={})),
                    text_reply("Paid."),
                    text_reply("Done."),
                ),
                "judge": None,
            }
        ),
    )

    async def ask(args: dict[str, Any]) -> Any:
        return await surface.dispatch("Delegate", {"agent": "billing", "input": "paid?"})

    tools = Tools([Tool("ask", "Asks billing", body=ask)])
    surface = EngineTools(engine, agent="support", scope="acme", tools=tools)
    run = await engine.run("support", "acme", "x", tools=tools, record=False)
    assert isinstance(run, Generation) and run.steps[0].result == "Paid."
    assert keeping.runs == []


async def test_a_run_asked_to_record_inside_one_that_is_not_records(tmp_path: Path) -> None:
    keeping = Keeping()
    engine = consulting(
        tmp_path,
        keeping,
        use(ToolCall(id="b1", name="recorded", args={})),
        text_reply("Paid."),
        text_reply("Done."),
        tool="recorded",
    )

    async def recorded(args: dict[str, Any]) -> str:
        run = await engine.run("billing", "acme", "paid?", record=True)
        assert isinstance(run, Generation)
        return run.reply

    tools = Tools([Tool("recorded", "Runs billing, recorded", body=recorded)])
    await engine.run("support", "acme", "x", tools=tools, record=False)
    assert [run.agent for run in keeping.runs] == ["billing"]


async def test_a_consult_outside_any_run_records_none(tmp_path: Path) -> None:
    keeping = Keeping()
    engine = consulting(tmp_path, keeping, text_reply("Paid."), tool=None)
    await engine.tool("billing", scope="acme").dispatch("billing", {"input": "x"})
    assert keeping.of("billing").call is None


async def test_a_consult_in_another_scope_records_none_while_its_event_names_the_call(
    tmp_path: Path,
) -> None:
    keeping = Keeping()
    engine = consulting(
        tmp_path,
        keeping,
        use(ToolCall(id="b1", name="billing", args={"input": "paid?"})),
        text_reply("Paid."),
        text_reply("Done."),
    )
    events: list[Event] = []

    async def watch(event: Event) -> None:
        events.append(event)

    tools = engine.tool("billing", scope="other", on_event=watch)
    outer = await engine.run("support", "acme", "x", tools=tools)
    consult = keeping.of("billing")
    assert consult.scope == "other" and consult.call is None
    [started] = [e for e in events if isinstance(e, RunStarted) and e.run_id == consult.run_id]
    assert started.call is not None
    assert started.call.run_id == outer.run_id and started.call.call_id == "b1"


async def test_a_run_a_tool_body_starts_after_the_call_ended_records_that_call(
    tmp_path: Path,
) -> None:
    keeping = Keeping()
    engine = consulting(
        tmp_path,
        keeping,
        use(ToolCall(id="t1", name="spawn", args={})),
        text_reply("Done."),
        text_reply("Later."),
        tool="spawn",
    )
    spawned: list[asyncio.Task[Generation | Decision]] = []

    async def spawn(args: dict[str, Any]) -> str:
        spawned.append(asyncio.create_task(engine.run("billing", "acme", "later")))
        return "started"

    outer = await engine.run("support", "acme", "x", tools=Tools([Tool("spawn", "Go", body=spawn)]))
    later = await spawned[0]
    assert later.call is not None
    assert later.call.call_id == "t1" and later.call.run_id == outer.run_id


class Failing:
    async def decide(self, state: Any, questions: Any) -> Any:
        raise RuntimeError("down")


async def test_a_failed_decision_run_keeps_its_call(tmp_path: Path) -> None:
    add_agent(
        tmp_path,
        "judge",
        {
            "kind": "decision",
            "model": "judge",
            "instructions": "judge.md",
            "questions": ["judge/*.json"],
        },
        files={"judge.md": "", "judge/ok.json": '{"type": "check", "instructions": "Ok?"}'},
    )
    tree(tmp_path, lead=["judge"])
    llm = Scripted(use(delegate("judge", "d1")), text_reply("done"))
    engine = Engine(tmp_path, providers=provided({"strong": llm, "judge": Failing()}))
    run = await engine.run("lead", "acme", "x")
    assert isinstance(run, Generation)
    [child] = run.children
    assert child.error is not None and child.call is not None and child.call.call_id == "d1"


async def test_a_call_carries_the_tags_of_the_run_it_serves(tmp_path: Path) -> None:
    seen = Seen()
    started: list[CallInfo | None] = []

    async def watch(event: Event) -> None:
        if isinstance(event, RunStarted):
            started.append(event.call)

    llm = Scripted(
        use(ToolCall(id="p1", name="probe", args={})),
        use(ToolCall(id="d1", name="Delegate", args={"agent": "child", "input": "go"})),
        use(ToolCall(id="c1", name="probe", args={})),
        text_reply("child done"),
        text_reply("lead done"),
    )
    engine = Engine(
        tree(tmp_path, lead=["child"], child=[], run_tools={"lead": ["probe"], "child": ["probe"]}),
        providers=provided({"strong": llm, "judge": None}),
    )
    run = await engine.run(
        "lead", "acme", "x", tools=seen.tools(), tags={"ticket": "T-7"}, on_event=watch
    )
    assert isinstance(run, Generation)
    [child] = run.children
    assert seen.calls == [
        CallInfo(
            run_id=run.run_id,
            agent="lead",
            scope="acme",
            tool="probe",
            call_id="p1",
            tags={"ticket": "T-7"},
        ),
        CallInfo(
            run_id=child.run_id,
            agent="child",
            scope="acme",
            tool="probe",
            call_id="c1",
            tags={"ticket": "T-7"},
        ),
    ]
    assert [None if call is None else call.tags for call in started] == [None, {"ticket": "T-7"}]
    assert child.call is not None and child.call.tags == {"ticket": "T-7"}


def test_a_call_s_tags_cannot_be_changed_and_it_can_be_kept_in_a_set() -> None:
    held = CallInfo(
        run_id="r1", agent="desk", scope="acme", tool="probe", call_id="p1", tags={"ticket": "T-7"}
    )
    with pytest.raises(TypeError):
        held.tags["ticket"] = "T-8"  # type: ignore[index]
    assert held.tags == {"ticket": "T-7"}
    same = CallInfo(
        run_id="r1", agent="desk", scope="acme", tool="probe", call_id="p1", tags={"ticket": "T-7"}
    )
    other = CallInfo(
        run_id="r1", agent="desk", scope="acme", tool="probe", call_id="p1", tags={"ticket": "T-8"}
    )
    assert {held, same, other} == {held, other}
    assert held.model_dump()["tags"] == {"ticket": "T-7"}
    assert CallInfo.model_validate(held.model_dump()) == held
