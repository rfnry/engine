import asyncio
from pathlib import Path
from typing import Any

from rfnry_engine import CallInfo, Engine, EngineTools, Stop, current_call
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Event, Generation, ToolCallStarted, ToolResultReady
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Scripted, text_reply


def tree(root: Path, **agents: str) -> Path:
    for name, front in agents.items():
        (root / name).mkdir(parents=True)
        (root / name / "AGENT.md").write_text(
            f"---\nname: {name}\nkind: generation\ndescription: {name.title()}\n{front}---\n"
            f"You are {name}.\n",
            encoding="utf-8",
        )
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
    engine = Engine(tree(tmp_path, desk=""), generation=llm)
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
    engine = Engine(tree(tmp_path, lead="agents: [child]\n", child=""), generation=llm)
    run = await engine.run("lead", "acme", "x", tools=seen.tools())
    assert isinstance(run, Generation)
    [child] = run.children
    assert seen.calls == [
        CallInfo(run_id=child.run_id, agent="child", scope="acme", tool="probe", call_id="c1")
    ]


async def test_a_consult_s_call_names_the_consult_and_its_body_the_caller(tmp_path: Path) -> None:
    inner = Seen()
    back = Engine(
        tree(tmp_path / "back", billing=""),
        generation=Scripted(use(ToolCall(id="i1", name="probe", args={})), text_reply("Paid.")),
    )
    consult = back.tool("billing", scope="acme", tools=inner.tools())
    outer: list[CallInfo | None] = []

    async def asking(args: dict[str, Any]) -> Any:
        outer.append(current_call())
        return await consult.dispatch("billing", {"input": "Was it paid?"})

    front = Engine(
        tree(tmp_path / "front", support=""),
        generation=Scripted(use(ToolCall(id="o1", name="ask", args={})), text_reply("Done.")),
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
    engine = Engine(tree(tmp_path, desk="agents: [desk]\n"), generation=Scripted(text_reply("x")))
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
    notify = Engine(tree(tmp_path / "back", notifier=""), generation=notifier).tool(
        "notifier", scope="acme", tools=Tools([Tool("ping", "Ping", body=ping)])
    )

    async def approve(agent: str, call: ToolCall) -> bool:
        await notify.dispatch("notifier", {"input": "May it refund?"})
        return True

    async def refund(args: dict[str, Any]) -> dict[str, str]:
        return {"refunded": "R-1"}

    llm = Scripted(use(ToolCall(id="r1", name="refund", args={})), text_reply("never"))
    engine = Engine(tree(tmp_path / "front", desk=""), generation=llm)
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
    engine = Engine(tree(tmp_path, desk=""), generation=llm)
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
