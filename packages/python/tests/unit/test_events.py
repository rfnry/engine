import asyncio
import contextlib
import gc
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine import (
    ChoiceAnswer,
    Event,
    ReplyReady,
    RunEnded,
    RunStarted,
    Stop,
    TextDelta,
    ToolCallStarted,
    ToolResultReady,
)
from rfnry_engine.engine import Engine
from rfnry_engine.errors import RunawayError
from rfnry_engine.models import Message, Reply, ToolCall, Usage
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Fixed, MemoryStore, Scripted, Streaming, text_reply
from .trees import provided, runtime, write_folder

STOPPED = "stopped by the user"
ROUTER_TEAM = {
    "router.md": "",
    "router/team.json": '{"type": "choice", "instructions": "Which team?", "options": {"billing": "Charges", "other": "Anything else"}}',
}


class Verdict(BaseModel):
    ok: bool


def make_tree(tmp_path: Path, run_tools: dict[str, list[str]] | None = None) -> Path:
    root = tmp_path / "agents"
    root.mkdir(parents=True)
    agents: dict[str, dict[str, Any]] = {
        "lead": {
            "kind": "generation",
            "model": "strong",
            "description": "Leads",
            "instructions": "lead.md",
            "orchestration": {"delegate_to": ["chat"]},
        },
        "chat": {
            "kind": "generation",
            "model": "strong",
            "description": "Talks",
            "instructions": "chat.md",
        },
    }
    for name, tools in (run_tools or {}).items():
        agents[name] = {**agents[name], "tools": {"allow": tools}}
    return write_folder(
        root,
        agents,
        runtime_tools=runtime(name for names in (run_tools or {}).values() for name in names),
        files={f"{name}.md": f"You are {name}.\n" for name in agents},
    )


async def lookup(args: dict[str, Any]) -> dict[str, str]:
    return {"order": args["order_id"], "status": "shipped"}


async def broken(args: dict[str, Any]) -> str:
    raise RuntimeError("the ledger is down")


async def slow(args: dict[str, Any]) -> str:
    await asyncio.sleep(30)
    return "never"


async def refund(args: dict[str, Any]) -> str:
    raise AssertionError("a refused call must not run")


TOOLS = Tools(
    [
        Tool("lookup", body=lookup),
        Tool("broken", body=broken),
        Tool("slow", body=slow),
        Tool("refund", body=refund),
    ]
)


class Collect:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)

    def of(self, run_id: str) -> list[ToolCallStarted | ToolResultReady]:
        return [
            e
            for e in self.events
            if isinstance(e, ToolCallStarted | ToolResultReady) and e.run_id == run_id
        ]


def check(run: Generation, calls: list[ToolCall], events: list[Any]) -> None:
    started = [e for e in events if isinstance(e, ToolCallStarted)]
    ready = [e for e in events if isinstance(e, ToolResultReady)]
    assert {e.call_id: (e.tool, e.args) for e in started} == {c.id: (c.name, c.args) for c in calls}
    assert len(ready) == len(calls) and all(e.agent == run.agent for e in events)
    steps = {e.call_id: e.step for e in ready}
    assert [steps[c.id] for c in calls] == run.steps
    for c in calls:
        assert events.index(next(e for e in started if e.call_id == c.id)) < events.index(
            next(e for e in ready if e.call_id == c.id)
        )


async def test_every_finished_call_sends_the_step_the_run_records(tmp_path: Path) -> None:
    calls = [
        ToolCall(id="c1", name="lookup", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="broken", args={"why": "x"}),
        ToolCall(id="c3", name="slow", args={}),
        ToolCall(id="c4", name="refund", args={"order_id": "A-1", "amount": 30}),
        ToolCall(id="c5", name="Delegate", args={"agent": "chat", "input": "Say hi"}),
    ]
    llm = Scripted(
        Reply(tool_calls=calls, stop_reason="tool_use"),
        text_reply("Hi from chat."),
        text_reply("Done."),
    )

    async def approve(agent: str, call: ToolCall) -> bool:
        return call.name != "refund"

    collect = Collect()
    engine = Engine(
        make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
        tool_timeout=0.05,
        providers=provided({"strong": llm, "judge": None}),
    )
    run = await engine.run(
        "lead", "acme", "Handle A-1", tools=TOOLS, approve=approve, on_event=collect
    )
    assert isinstance(run, Generation) and run.reply == "Done."
    check(run, calls, collect.of(run.run_id))
    ok, failed, late, refused, delegated = run.steps
    assert ok.result == '{"order": "A-1", "status": "shipped"}' and not ok.is_error
    assert failed.result == "RuntimeError: the ledger is down" and failed.is_error
    assert late.result == "TimeoutError: engine: slow timed out after 0.05s" and late.is_error
    assert refused.result == "NotApproved: refund was not approved" and refused.approved is False
    [child] = run.children
    assert delegated.child_run_id == child.run_id and delegated.result == "Hi from chat."
    assert [s.approved for s in (ok, failed, late, delegated)] == [True] * 4


async def test_a_call_cut_off_by_stop_sends_its_stopped_step(tmp_path: Path) -> None:
    calls = [
        ToolCall(id="c1", name="slow", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="refund", args={"order_id": "A-2"}),
    ]
    started = asyncio.Event()

    async def tracked(args: dict[str, Any]) -> str:
        started.set()
        return await slow(args)

    asked = asyncio.Event()

    async def approve(agent: str, call: ToolCall) -> bool:
        if call.name == "refund":
            asked.set()
            await asyncio.sleep(30)
        return True

    collect = Collect()
    stop = Stop()
    tools = Tools([Tool("slow", body=tracked), Tool("refund", body=refund)])
    engine = Engine(
        make_tree(tmp_path, run_tools={"lead": ["slow", "refund"]}),
        providers=provided(
            {"strong": Scripted(Reply(tool_calls=calls, stop_reason="tool_use")), "judge": None}
        ),
    )
    running = asyncio.create_task(
        engine.run(
            "lead", "acme", "Refund", tools=tools, approve=approve, stop=stop, on_event=collect
        )
    )
    await started.wait()
    await asked.wait()
    stop.now()
    run = await asyncio.wait_for(running, 2)
    assert isinstance(run, Generation) and run.stopped == "now"
    check(run, calls, collect.of(run.run_id))
    assert [(s.result, s.is_error, s.stopped) for s in run.steps] == [(STOPPED, True, True)] * 2


class StopOn(Collect):
    def __init__(self, stop: Stop, kind: type[BaseModel], pause: float) -> None:
        super().__init__()
        self.stop, self.kind, self.pause = stop, kind, pause
        self.handled: list[Event] = []

    async def __call__(self, event: Event) -> None:
        await super().__call__(event)
        if isinstance(event, self.kind) and not self.handled:
            self.stop.now()
            await asyncio.sleep(self.pause)
            self.handled.append(event)


@pytest.mark.parametrize("pause", [0, 0.01])
async def test_a_stop_from_a_result_handler_sends_each_result_once(
    tmp_path: Path, pause: float
) -> None:
    calls = [
        ToolCall(id="c1", name="slow", args={}),
        ToolCall(id="c2", name="lookup", args={"order_id": "A-1"}),
    ]
    stop = Stop()
    handler = StopOn(stop, ToolResultReady, pause)
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=handler),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    events = handler.of(run.run_id)
    check(run, calls, events)
    assert [(s.result, s.stopped) for s in run.steps] == [
        (STOPPED, True),
        ('{"order": "A-1", "status": "shipped"}', False),
    ]
    first = [e for e in events if isinstance(e, ToolResultReady) and e.call_id == "c2"]
    assert handler.handled == first


async def test_a_stop_from_a_result_handler_answers_a_pending_output_once(
    tmp_path: Path,
) -> None:
    calls = [
        ToolCall(id="c1", name="lookup", args={"order_id": "A-1"}),
        ToolCall(id="o1", name="Output", args={"ok": True}),
    ]
    stop = Stop()
    handler = StopOn(stop, ToolResultReady, 0.01)
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run(
            "lead", "acme", "Check A-1", tools=TOOLS, expect=Verdict, stop=stop, on_event=handler
        ),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    ready = [e for e in handler.of(run.run_id) if isinstance(e, ToolResultReady)]
    assert [e.call_id for e in ready] == ["c1", "o1"]
    assert [e.step for e in ready] == run.steps
    assert [(s.tool, s.stopped) for s in run.steps] == [("lookup", False), ("Output", True)]


async def test_a_stop_from_a_start_handler_sends_each_event_once(tmp_path: Path) -> None:
    calls = [
        ToolCall(id="c1", name="lookup", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="slow", args={}),
    ]
    stop = Stop()
    handler = StopOn(stop, ToolCallStarted, 0.01)
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=handler),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    check(run, calls, handler.of(run.run_id))
    assert [(s.result, s.stopped) for s in run.steps] == [(STOPPED, True)] * 2


async def test_a_stop_from_a_text_handler_sends_each_delta_once(tmp_path: Path) -> None:
    looked = ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})
    stop = Stop()
    handler = StopOn(stop, TextDelta, 0.01)
    llm = Streaming(Reply(text="Let me look.", tool_calls=[looked], stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=handler),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now" and run.reply == "Let "
    assert [e.text for e in handler.events if isinstance(e, TextDelta)] == ["Let "]
    assert handler.of(run.run_id) == [] and run.steps == []


class Destroy:
    def __init__(self) -> None:
        self.destroyed: list[str] = []

    async def __call__(self, args: dict[str, Any]) -> str:
        await asyncio.sleep(0)
        self.destroyed.append(args["order_id"])
        return "destroyed"


def answered(run: Generation) -> list[str | None]:
    return [m.tool_call_id for m in run.conversation if m.role == "tool"]


async def test_a_stop_from_a_calls_start_handler_stops_the_tool_before_it_runs(
    tmp_path: Path,
) -> None:
    destroy, stop, collect = Destroy(), Stop(), Collect()

    async def kill(event: Event) -> None:
        await collect(event)
        if isinstance(event, ToolCallStarted) and event.call_id == "c1":
            stop.now()

    calls = [ToolCall(id="c1", name="destroy", args={"order_id": "A-1"})]
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["destroy"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run(
            "lead",
            "acme",
            "Destroy A-1",
            tools=Tools([Tool("destroy", body=destroy)]),
            stop=stop,
            on_event=kill,
        ),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    assert destroy.destroyed == []
    check(run, calls, collect.of(run.run_id))
    assert [(s.result, s.is_error, s.stopped) for s in run.steps] == [(STOPPED, True, True)]
    assert answered(run) == ["c1"]


async def test_a_stop_from_a_text_handler_stops_the_batch_before_it_runs(tmp_path: Path) -> None:
    destroy, stop, collect = Destroy(), Stop(), Collect()

    async def kill(event: Event) -> None:
        await collect(event)
        if isinstance(event, TextDelta):
            stop.now()

    calls = [
        ToolCall(id="c1", name="destroy", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="destroy", args={"order_id": "A-2"}),
    ]
    llm = Scripted(Reply(text="Destroying both.", tool_calls=calls, stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["destroy"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run(
            "lead",
            "acme",
            "Destroy A-1 and A-2",
            tools=Tools([Tool("destroy", body=destroy)]),
            stop=stop,
            on_event=kill,
        ),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    assert run.reply == "Destroying both." and destroy.destroyed == []
    events = collect.of(run.run_id)
    assert [e.call_id for e in events if isinstance(e, ToolResultReady)] == ["c1", "c2"]
    assert not any(isinstance(e, ToolCallStarted) for e in events)
    assert [e.step for e in events if isinstance(e, ToolResultReady)] == run.steps
    assert [(s.result, s.is_error, s.stopped) for s in run.steps] == [(STOPPED, True, True)] * 2
    assert answered(run) == ["c1", "c2"]


async def test_a_stop_from_a_text_handler_answers_an_output_beside_it_as_stopped(
    tmp_path: Path,
) -> None:
    stop, collect = Stop(), Collect()

    async def kill(event: Event) -> None:
        await collect(event)
        if isinstance(event, TextDelta):
            stop.now()

    output = ToolCall(id="o1", name="Output", args={"ok": True})
    llm = Scripted(Reply(text="Verdict.", tool_calls=[output], stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(make_tree(tmp_path), providers=provided({"strong": llm, "judge": None})).run(
            "lead", "acme", "Judge", expect=Verdict, stop=stop, on_event=kill
        ),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now" and run.reply == "Verdict."
    assert [(s.tool, s.result, s.stopped) for s in run.steps] == [("Output", STOPPED, True)]
    events = collect.of(run.run_id)
    assert [(type(e), e.call_id) for e in events] == [(ToolResultReady, "o1")]
    assert [e.step for e in events if isinstance(e, ToolResultReady)] == run.steps
    assert answered(run) == ["o1"]


@pytest.mark.parametrize("dead", [False, True])
async def test_cancelling_a_run_a_stop_holds_on_a_result_handler_cancels_the_handler(
    tmp_path: Path, dead: bool
) -> None:
    stop, entered, left = Stop(), asyncio.Event(), asyncio.Event()
    dying, release, died = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancels = 0

    async def stuck(event: Event) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == "c2":
            stop.now()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                left.set()

    async def slow_to_die(args: dict[str, Any]) -> str:
        nonlocal cancels
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancels += 1
            dying.set()
            cancels += await outlast(asyncio.ensure_future(release.wait()))
            died.set()
            raise
        return "never"

    before = asyncio.all_tasks()
    calls = [
        ToolCall(id="c1", name="dying", args={}),
        ToolCall(id="c2", name="lookup", args={"order_id": "A-1"}),
    ]
    tools = Tools([Tool("dying", body=slow_to_die), Tool("lookup", body=lookup)])
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["dying", "lookup"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=tools, stop=stop, on_event=stuck)
    )
    async with asyncio.timeout(2):
        await entered.wait()
        await dying.wait()
        if dead:
            release.set()
            await died.wait()
            await asyncio.sleep(0.01)
    running.cancel()
    async with asyncio.timeout(2):
        await left.wait()
    if not dead:
        assert not died.is_set() and not running.done()
        release.set()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(2):
            await running
    assert died.is_set() and cancels == (1 if dead else 2)
    assert asyncio.all_tasks() <= before


@pytest.mark.parametrize("stopper", ["a holding handler", "a handler that returned", "the caller"])
async def test_cancelling_a_run_cancels_the_handler_of_a_call_that_swallowed_the_stop(
    tmp_path: Path, stopper: str
) -> None:
    stop, entered = Stop(), asyncio.Event()
    held = {"c1": asyncio.Event(), "c2": asyncio.Event()}
    left = {"c1": asyncio.Event(), "c2": asyncio.Event()}

    async def holding(event: Event) -> None:
        if not isinstance(event, ToolResultReady):
            return
        if event.call_id == "c2":
            stop.now()
            if stopper == "a handler that returned":
                return
        held[event.call_id].set()
        try:
            await asyncio.Event().wait()
        finally:
            left[event.call_id].set()

    async def swallows(args: dict[str, Any]) -> str:
        entered.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(30)
        return "kept going"

    before = asyncio.all_tasks()
    calls = [ToolCall(id="c1", name="swallows", args={})]
    if stopper != "the caller":
        calls.append(ToolCall(id="c2", name="lookup", args={"order_id": "A-1"}))
    tools = Tools([Tool("swallows", body=swallows), Tool("lookup", body=lookup)])
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["swallows", "lookup"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=tools, stop=stop, on_event=holding)
    )
    async with asyncio.timeout(2):
        await entered.wait()
        if stopper == "the caller":
            stop.now()
        await held["c1"].wait()
    await asyncio.sleep(0.05)
    running.cancel()
    async with asyncio.timeout(2):
        for call_id in ["c1", "c2"] if stopper == "a holding handler" else ["c1"]:
            await left[call_id].wait()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(2):
            await running
    assert asyncio.all_tasks() <= before


async def outlast(ending: asyncio.Future[Any]) -> int:
    cancels = 0
    while not ending.done():
        try:
            await asyncio.wait([ending])
        except asyncio.CancelledError:
            cancels += 1
    return cancels


class Dying:
    def __init__(self, kept: str | None = None) -> None:
        self.entered, self.died = asyncio.Event(), asyncio.Event()
        self.kept = kept
        self.cancels = 0

    async def __call__(self, *args: Any) -> Any:
        self.entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancels += 1 + await outlast(asyncio.ensure_future(asyncio.sleep(0.2)))
            self.died.set()
            if self.kept is None:
                raise
            return self.kept
        raise AssertionError("a cancelled call must not end on its own")


class Absorbing:
    def __init__(self) -> None:
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.cancels = 0

    async def __call__(self, *args: Any) -> Any:
        self.entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancels += 1
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancels += 1
            raise
        raise AssertionError("a cancelled call must not end on its own")


class DyingModel:
    def __init__(self) -> None:
        self.generate = Dying()


async def cancel_while_dying(running: asyncio.Task[Any], dying: Dying, cancels: int) -> None:
    async with asyncio.timeout(2):
        await dying.entered.wait()
    running.cancel()
    for _ in range(cancels - 1):
        await asyncio.sleep(0.05)
        assert not running.done()
        running.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(2):
            await running
    assert dying.died.is_set() and dying.cancels == cancels


@pytest.mark.parametrize("kept", [None, "kept going"])
@pytest.mark.parametrize("cancels", [1, 2])
async def test_a_cancelled_run_returns_once_its_tool_calls_have_ended(
    tmp_path: Path, cancels: int, kept: str | None
) -> None:
    dying = Dying(kept)
    calls = [
        ToolCall(id="c1", name="lookup", args={"order_id": "A-1"}),
        ToolCall(id="c2", name="dying", args={}),
    ]
    tools = Tools([Tool("lookup", body=lookup), Tool("dying", body=dying)])
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    before = asyncio.all_tasks()
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["lookup", "dying"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=tools)
    )
    await cancel_while_dying(running, dying, cancels)
    assert asyncio.all_tasks() <= before


@pytest.mark.parametrize("cancels", [1, 2])
async def test_a_cancelled_run_returns_once_a_delegated_run_has_ended(
    tmp_path: Path, cancels: int
) -> None:
    dying = Dying()
    llm = Scripted(
        Reply(
            tool_calls=[
                ToolCall(id="d", name="Delegate", args={"agent": "chat", "input": "Check A-1"})
            ],
            stop_reason="tool_use",
        ),
        Reply(tool_calls=[ToolCall(id="c1", name="dying", args={})], stop_reason="tool_use"),
    )
    before = asyncio.all_tasks()
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"chat": ["dying"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=Tools([Tool("dying", body=dying)]))
    )
    await cancel_while_dying(running, dying, cancels)
    assert asyncio.all_tasks() <= before


@pytest.mark.parametrize("where", ["model", "approve"])
async def test_a_cancelled_run_returns_once_its_model_call_or_approval_has_ended(
    tmp_path: Path, where: str
) -> None:
    calls = [ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})]
    model = DyingModel()
    llm = model if where == "model" else Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    dying = model.generate if where == "model" else Dying()
    approve = dying if where == "approve" else None
    before = asyncio.all_tasks()
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=TOOLS, approve=approve)
    )
    await cancel_while_dying(running, dying, 2)
    assert asyncio.all_tasks() <= before


@pytest.mark.parametrize("apart", [True, False])
@pytest.mark.parametrize("presses", [2, 3])
async def test_each_further_stop_cancels_a_call_still_running_once_more(
    tmp_path: Path, presses: int, apart: bool
) -> None:
    stop, dying, collect = Stop(), Absorbing(), Collect()
    held, release = asyncio.Event(), asyncio.Event()
    handler_cancelled = False

    async def holding(event: Event) -> None:
        nonlocal handler_cancelled
        await collect(event)
        if isinstance(event, ToolResultReady) and event.call_id == "c2":
            held.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                handler_cancelled = True
                raise

    before = asyncio.all_tasks()
    calls = [
        ToolCall(id="c1", name="dying", args={}),
        ToolCall(id="c2", name="lookup", args={"order_id": "A-1"}),
    ]
    tools = Tools([Tool("dying", body=dying), Tool("lookup", body=lookup)])
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["dying", "lookup"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=tools, stop=stop, on_event=holding)
    )
    try:
        async with asyncio.timeout(2):
            await dying.entered.wait()
            await held.wait()
            for _ in range(presses):
                stop.now()
                if apart:
                    await asyncio.sleep(0.05)
            while dying.cancels < presses:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert dying.cancels == presses and not handler_cancelled and not running.done()
    finally:
        release.set()
        dying.release.set()
    async with asyncio.timeout(2):
        run = await running
    assert isinstance(run, Generation) and run.stopped == "now"
    assert [(s.result, s.stopped) for s in run.steps] == [
        (STOPPED, True),
        ('{"order": "A-1", "status": "shipped"}', False),
    ]
    check(run, calls, collect.of(run.run_id))
    assert not handler_cancelled
    assert asyncio.all_tasks() <= before


async def test_two_cancels_of_a_run_at_once_cancel_a_call_still_running_twice(
    tmp_path: Path,
) -> None:
    dying = Absorbing()
    calls = [ToolCall(id="c1", name="dying", args={})]
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    before = asyncio.all_tasks()
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["dying"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=Tools([Tool("dying", body=dying)]))
    )
    try:
        async with asyncio.timeout(2):
            await dying.entered.wait()
            running.cancel()
            running.cancel()
            while dying.cancels < 2:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert dying.cancels == 2 and not running.done()
    finally:
        dying.release.set()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(2):
            await running
    assert asyncio.all_tasks() <= before


async def test_a_further_stop_reaches_a_delegated_runs_call_and_keeps_the_delegation(
    tmp_path: Path,
) -> None:
    stop, dying = Stop(), Absorbing()
    llm = Scripted(
        Reply(
            tool_calls=[
                ToolCall(id="d", name="Delegate", args={"agent": "chat", "input": "Check A-1"})
            ],
            stop_reason="tool_use",
        ),
        Reply(tool_calls=[ToolCall(id="c1", name="dying", args={})], stop_reason="tool_use"),
    )
    before = asyncio.all_tasks()
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"chat": ["dying"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=Tools([Tool("dying", body=dying)]), stop=stop)
    )
    try:
        async with asyncio.timeout(2):
            await dying.entered.wait()
            stop.now()
            await asyncio.sleep(0.05)
            stop.now()
            while dying.cancels < 2:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert dying.cancels == 2 and not running.done()
    finally:
        dying.release.set()
    async with asyncio.timeout(2):
        run = await running
    assert isinstance(run, Generation) and run.stopped == "now"
    [child] = run.children
    assert isinstance(child, Generation) and child.stopped == "now"
    assert [(s.tool, s.stopped, s.child_run_id) for s in run.steps] == [
        ("Delegate", True, child.run_id)
    ]
    assert [(s.tool, s.result, s.stopped) for s in child.steps] == [("dying", STOPPED, True)]
    assert asyncio.all_tasks() <= before


@pytest.mark.parametrize("stopped", [False, True])
async def test_a_call_whose_tool_returns_after_the_run_is_cancelled_never_reaches_its_handler(
    tmp_path: Path, stopped: bool
) -> None:
    stop, collect = Stop(), Collect()
    entered, finish, c2_held = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancels = 0

    async def holding(event: Event) -> None:
        await collect(event)
        if isinstance(event, ToolResultReady):
            if event.call_id == "c2":
                stop.now()
                c2_held.set()
            await asyncio.Event().wait()

    async def outlasting(args: dict[str, Any]) -> str:
        nonlocal cancels
        entered.set()
        while not finish.is_set():
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancels += 1
        return "kept going"

    before = asyncio.all_tasks()
    calls = [ToolCall(id="c1", name="outlasting", args={})]
    if stopped:
        calls.append(ToolCall(id="c2", name="lookup", args={"order_id": "A-1"}))
    tools = Tools([Tool("outlasting", body=outlasting), Tool("lookup", body=lookup)])
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["outlasting", "lookup"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=tools, stop=stop, on_event=holding)
    )
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            if stopped:
                await c2_held.wait()
        await asyncio.sleep(0.05)
        running.cancel()
        await asyncio.sleep(0.05)
        assert not running.done() and cancels == (2 if stopped else 1)
        finish.set()
        done, _ = await asyncio.wait({running}, timeout=1)
        assert running in done and running.cancelled()
    finally:
        finish.set()
        running.cancel()
        await asyncio.wait({running}, timeout=2)
    ready = [e.call_id for e in collect.events if isinstance(e, ToolResultReady)]
    assert ready == (["c2"] if stopped else [])
    started = [e.call_id for e in collect.events if isinstance(e, ToolCallStarted)]
    assert started == [c.id for c in calls]
    assert asyncio.all_tasks() <= before


async def test_a_further_cancel_reaches_a_result_handler_that_ignored_the_first(
    tmp_path: Path,
) -> None:
    held, left = asyncio.Event(), asyncio.Event()
    cancels = 0

    async def stubborn(event: Event) -> None:
        nonlocal cancels
        if not isinstance(event, ToolResultReady):
            return
        held.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancels += 1
        try:
            await asyncio.Event().wait()
        finally:
            cancels += 1
            left.set()

    before = asyncio.all_tasks()
    calls = [ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})]
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    running = asyncio.create_task(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=TOOLS, on_event=stubborn)
    )
    try:
        async with asyncio.timeout(2):
            await held.wait()
        running.cancel()
        await asyncio.sleep(0.05)
        assert cancels == 1 and not running.done()
        running.cancel()
        async with asyncio.timeout(2):
            await left.wait()
        done, _ = await asyncio.wait({running}, timeout=1)
        assert running in done and running.cancelled() and cancels == 2
    finally:
        running.cancel()
        await asyncio.wait({running}, timeout=2)
    assert asyncio.all_tasks() <= before


@pytest.mark.parametrize("cancelled", [False, True])
async def test_a_run_with_a_stop_leaves_no_task_behind(tmp_path: Path, cancelled: bool) -> None:
    dying = Dying()
    reply = (
        Reply(tool_calls=[ToolCall(id="c1", name="dying", args={})], stop_reason="tool_use")
        if cancelled
        else text_reply("Done.")
    )
    engine = Engine(
        make_tree(tmp_path, run_tools={"lead": ["dying"]}),
        providers=provided({"strong": Scripted(reply), "judge": None}),
    )
    me = asyncio.current_task()
    assert me is not None

    async def cancel_me() -> None:
        await dying.entered.wait()
        me.cancel()

    helper = asyncio.create_task(cancel_me())
    before = asyncio.all_tasks()
    raised = False
    try:
        async with asyncio.timeout(2):
            await engine.run(
                "lead", "acme", "Hi", tools=Tools([Tool("dying", body=dying)]), stop=Stop()
            )
    except asyncio.CancelledError:
        me.uncancel()
        raised = True
    try:
        assert raised is cancelled
        assert asyncio.all_tasks() <= before
    finally:
        helper.cancel()
        await asyncio.wait([helper])


async def test_a_batch_whose_calls_all_fail_raises_the_first_and_retrieves_the_rest(
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()
    seen: list[dict[str, Any]] = []
    gc.collect()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: seen.append(context))

    async def approve(agent: str, call: ToolCall) -> bool:
        raise RuntimeError(f"approval of {call.id} is down")

    calls = [ToolCall(id=f"c{n}", name="lookup", args={"order_id": "A-1"}) for n in (1, 2, 3)]
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    try:
        with pytest.raises(RuntimeError, match="^approval of c1 is down$"):
            await Engine(
                make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
                providers=provided({"strong": llm, "judge": None}),
            ).run("lead", "acme", "Check A-1", tools=TOOLS, approve=approve)
        gc.collect()
    finally:
        loop.set_exception_handler(previous)
    assert seen == []


@pytest.mark.parametrize(
    "reply",
    [
        text_reply("All done."),
        Reply(
            text="All done.",
            tool_calls=[ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})],
            stop_reason="max_tokens",
        ),
    ],
)
async def test_a_stop_from_a_text_handler_stops_a_reply_that_ends_the_run(
    tmp_path: Path, reply: Reply
) -> None:
    stop = Stop()

    async def kill(event: Event) -> None:
        if isinstance(event, TextDelta):
            stop.now()

    run = await asyncio.wait_for(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
            providers=provided({"strong": Scripted(reply), "judge": None}),
        ).run("lead", "acme", "Finish", tools=TOOLS, stop=stop, on_event=kill),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now" and run.reply == "All done."
    assert [s.tool for s in run.steps] == [c.name for c in reply.tool_calls]


@pytest.mark.parametrize("kind", [ToolCallStarted, ToolResultReady])
async def test_a_stop_on_the_last_allowed_turn_returns_the_run(
    tmp_path: Path, kind: type[BaseModel]
) -> None:
    stop, collect = Stop(), Collect()

    async def kill(event: Event) -> None:
        await collect(event)
        if isinstance(event, kind):
            stop.now()

    calls = [ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})]
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    run = await asyncio.wait_for(
        Engine(
            make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
            max_model_calls=1,
            providers=provided({"strong": llm, "judge": None}),
        ).run("lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=kill),
        2,
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    check(run, calls, collect.of(run.run_id))
    assert [s.stopped for s in run.steps] == [kind is ToolCallStarted]
    assert answered(run) == ["c1"]


async def test_a_childs_text_carries_its_own_run_and_its_own_turns(tmp_path: Path) -> None:
    delegate = ToolCall(id="d", name="Delegate", args={"agent": "chat", "input": "Say hi"})
    llm = Scripted(
        Reply(text="Asking chat.", tool_calls=[delegate], stop_reason="tool_use"),
        text_reply("Hi from chat."),
        text_reply("Chat says hi."),
    )
    collect = Collect()
    run = await Engine(make_tree(tmp_path), providers=provided({"strong": llm, "judge": None})).run(
        "lead", "acme", "Greet", on_event=collect
    )
    assert isinstance(run, Generation) and run.reply == "Chat says hi."
    [child] = run.children
    said = [(e.run_id, e.agent, e.turn, e.text) for e in collect.events if isinstance(e, TextDelta)]
    assert said == [
        (run.run_id, "lead", 0, "Asking chat."),
        (child.run_id, "chat", 0, "Hi from chat."),
        (run.run_id, "lead", 1, "Chat says hi."),
    ]


class ByAgent:
    def __init__(self, lead: list[Reply], **others: Any) -> None:
        self.lead = lead
        self.others = others

    async def generate(self, system: str, messages: list[Message], tools: list[Any]) -> Reply:
        for name, answer in self.others.items():
            if f"You are {name}." in system:
                reply: Reply = answer(messages) if callable(answer) else answer.pop(0)
                return reply
        return self.lead.pop(0)


def looking(messages: list[Message]) -> Reply:
    if len(messages) == 1:
        call = ToolCall(id=f"r-{messages[0].content}", name="lookup", args={"order_id": "A-1"})
        return Reply(tool_calls=[call], stop_reason="tool_use")
    return text_reply(f"Found {messages[0].content}.")


def tree(tmp_path: Path, **delegates: list[str]) -> Path:
    root = tmp_path / "agents"
    root.mkdir(parents=True)
    agents: dict[str, dict[str, Any]] = {}
    for name, targets in delegates.items():
        agents[name] = {
            "kind": "generation",
            "model": "strong",
            "description": name.title(),
            "instructions": f"{name}.md",
        }
        if targets:
            agents[name]["orchestration"] = {"delegate_to": targets}
    return write_folder(root, agents, files={f"{name}.md": f"You are {name}.\n" for name in agents})


def started(collect: Collect) -> dict[str, RunStarted]:
    return {e.run_id: e for e in collect.events if isinstance(e, RunStarted)}


async def test_a_run_says_it_started_before_anything_else(tmp_path: Path) -> None:
    collect = Collect()
    run = await Engine(
        make_tree(tmp_path),
        providers=provided({"strong": Scripted(text_reply("Hi.")), "judge": None}),
    ).run("chat", "acme", "Hi", on_event=collect)
    assert collect.events[0] == RunStarted(
        run_id=run.run_id, agent="chat", call=None, input="Hi", tools=run.tools
    )
    assert [type(e) for e in collect.events] == [
        RunStarted,
        ReplyReady,
        TextDelta,
        RunEnded,
    ]


async def test_run_started_carries_the_input_and_the_tools_offered(tmp_path: Path) -> None:
    delegate = ToolCall(id="d", name="Delegate", args={"agent": "chat", "input": "Look up A-1"})
    collect = Collect()
    run = await Engine(
        make_tree(tmp_path, run_tools={"lead": ["lookup", "refund"]}),
        providers=provided(
            {
                "strong": ByAgent(
                    [Reply(tool_calls=[delegate], stop_reason="tool_use"), text_reply("Done.")],
                    chat=looking,
                ),
                "judge": None,
            }
        ),
    ).run("lead", "acme", {"order": "A-1"}, tools=TOOLS, on_event=collect)
    assert isinstance(run, Generation) and run.tools
    runs = started(collect)
    assert runs[run.run_id].input == run.input == {"order": "A-1"}
    assert runs[run.run_id].tools == run.tools
    assert {spec.name for spec in runs[run.run_id].tools} >= {"lookup", "refund", "Delegate"}
    [child] = run.children
    assert isinstance(child, Generation)
    assert runs[child.run_id].input == child.input == "Look up A-1"
    assert runs[child.run_id].tools == child.tools


async def test_a_run_started_built_by_hand_needs_no_input_or_tools() -> None:
    event = RunStarted(run_id="r", agent="a", call=None)
    assert event.input == "" and event.tools == []


async def test_a_handler_changing_run_started_changes_nothing_sent(tmp_path: Path) -> None:
    async def meddle(event: Event) -> None:
        if isinstance(event, RunStarted):
            event.tools.append(event.tools[0].model_copy(update={"name": "extra"}))
            event.tools[0].description = "changed"
            if isinstance(event.input, dict):
                event.input["order"]["id"] = "B-2"

    llm = Scripted(text_reply("Done."))
    run = await Engine(
        make_tree(tmp_path, run_tools={"lead": ["lookup", "refund"]}),
        providers=provided({"strong": llm, "judge": None}),
    ).run("lead", "acme", {"order": {"id": "A-1"}}, tools=TOOLS, on_event=meddle)
    assert isinstance(run, Generation)
    names = [spec.name for spec in run.tools]
    assert "extra" not in names and run.input == {"order": {"id": "A-1"}}
    [(_, messages, sent)] = llm.calls
    assert [spec.name for spec in sent] == names
    assert all(spec.description != "changed" for spec in sent + run.tools)
    assert "B-2" not in str(messages) and "A-1" in str(messages)


async def test_each_sub_agent_names_the_call_it_works_for_before_its_first_tool(
    tmp_path: Path,
) -> None:
    split = [
        ToolCall(id="d1", name="Delegate", args={"agent": "chat", "input": "A"}),
        ToolCall(id="d2", name="Delegate", args={"agent": "chat", "input": "B"}),
    ]
    llm = ByAgent(
        [Reply(tool_calls=split, stop_reason="tool_use"), text_reply("Both.")], chat=looking
    )
    collect = Collect()
    run = await Engine(
        make_tree(tmp_path, run_tools={"lead": ["lookup", "broken", "slow", "refund"]}),
        providers=provided({"strong": llm, "judge": None}),
    ).run("lead", "acme", "Split", tools=TOOLS, on_event=collect)
    assert isinstance(run, Generation) and run.reply == "Both."
    runs = started(collect)
    assert runs[run.run_id].call is None and len(runs) == 3
    for child in run.children:
        assert isinstance(child, Generation)
        call = runs[child.run_id].call
        assert call is not None
        assert (call.run_id, call.agent, call.scope, call.tool) == (
            run.run_id,
            "lead",
            "acme",
            "Delegate",
        )
        assert call.call_id == {"A": "d1", "B": "d2"}[str(child.input)]
        own = [e for e in collect.events if e.run_id == child.run_id]
        assert [type(e) for e in own] == [
            RunStarted,
            ReplyReady,
            ToolCallStarted,
            ToolResultReady,
            ReplyReady,
            TextDelta,
            RunEnded,
        ]


async def test_a_grandchild_names_its_parents_call(tmp_path: Path) -> None:
    root = tree(tmp_path, lead=["mid"], mid=["leaf"], leaf=[])
    llm = ByAgent(
        [
            Reply(
                tool_calls=[
                    ToolCall(id="d1", name="Delegate", args={"agent": "mid", "input": "go"})
                ],
                stop_reason="tool_use",
            ),
            text_reply("Done."),
        ],
        mid=[
            Reply(
                tool_calls=[
                    ToolCall(id="m1", name="Delegate", args={"agent": "leaf", "input": "go"})
                ],
                stop_reason="tool_use",
            ),
            text_reply("Mid done."),
        ],
        leaf=[text_reply("Leaf done.")],
    )
    collect = Collect()
    run = await Engine(root, providers=provided({"strong": llm, "judge": None})).run(
        "lead", "acme", "Go", on_event=collect
    )
    assert isinstance(run, Generation)
    [mid] = run.children
    assert isinstance(mid, Generation)
    [leaf] = mid.children
    runs = started(collect)
    mid_call, leaf_call = runs[mid.run_id].call, runs[leaf.run_id].call
    assert mid_call is not None and (mid_call.run_id, mid_call.call_id) == (run.run_id, "d1")
    assert leaf_call is not None and (leaf_call.run_id, leaf_call.call_id) == (mid.run_id, "m1")


async def test_a_run_routed_under_a_delegated_decision_names_the_delegate_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "agents").mkdir()
    root = write_folder(
        tmp_path / "agents",
        {
            "lead": {
                "kind": "generation",
                "model": "strong",
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["router"]},
            },
            "router": {
                "kind": "decision",
                "model": "judge",
                "description": "Routes",
                "instructions": "router.md",
                "questions": ["router/*.json"],
                "orchestration": {"route": {"by": "team", "to": {"billing": "chat"}}},
            },
            "chat": {
                "kind": "generation",
                "model": "strong",
                "description": "Talks",
                "instructions": "chat.md",
            },
        },
        files={"lead.md": "You are lead.\n", "chat.md": "You are chat.\n", **ROUTER_TEAM},
    )
    llm = ByAgent(
        [
            Reply(
                tool_calls=[
                    ToolCall(
                        id="d1", name="Delegate", args={"agent": "router", "input": "Charged twice"}
                    )
                ],
                stop_reason="tool_use",
            ),
            text_reply("Routed."),
        ],
        chat=[text_reply("Billing here.")],
    )
    decision = Fixed(
        {"team": ChoiceAnswer(choice="billing", probabilities={"billing": 0.9, "other": 0.1})}
    )
    collect = Collect()
    run = await Engine(root, providers=provided({"strong": llm, "judge": decision})).run(
        "lead", "acme", "Help", on_event=collect
    )
    assert isinstance(run, Generation)
    [router] = run.children
    assert isinstance(router, Decision) and isinstance(router.routed_to, Generation)
    call = started(collect)[router.routed_to.run_id].call
    assert call is not None and (call.run_id, call.call_id) == (run.run_id, "d1")
    assert [type(e) for e in collect.events if e.run_id == router.run_id] == [ReplyReady]


async def test_a_consult_made_inside_a_tool_names_that_tool_call(tmp_path: Path) -> None:
    llm = ByAgent(
        [
            Reply(
                tool_calls=[ToolCall(id="c1", name="ask_chat", args={"input": "Hi"})],
                stop_reason="tool_use",
            ),
            text_reply("Asked."),
        ],
        chat=[text_reply("Hello.")],
    )
    engine = Engine(
        make_tree(tmp_path, run_tools={"lead": ["ask_chat"]}),
        providers=provided({"strong": llm, "judge": None}),
    )
    collect = Collect()
    run = await engine.run(
        "lead",
        "acme",
        "Ask",
        tools=engine.tool("chat", scope="acme", name="ask_chat"),
        on_event=collect,
    )
    assert isinstance(run, Generation) and run.reply == "Asked."
    [consult] = [e for e in collect.events if isinstance(e, RunStarted) and e.agent == "chat"]
    assert consult.call is not None
    assert (consult.call.run_id, consult.call.tool, consult.call.call_id) == (
        run.run_id,
        "ask_chat",
        "c1",
    )


async def test_a_run_stopped_before_it_starts_still_says_it_started(tmp_path: Path) -> None:
    stop = Stop()
    stop.now()
    collect = Collect()
    run = await Engine(
        make_tree(tmp_path), providers=provided({"strong": Scripted(), "judge": None})
    ).run("chat", "acme", "Hi", on_event=collect, stop=stop)
    assert isinstance(run, Generation) and run.stopped == "now"
    assert collect.events == [
        RunStarted(run_id=run.run_id, agent="chat", call=None, input="Hi"),
        RunEnded(run_id=run.run_id, agent="chat", call=None, stopped="now", error=None),
    ]


class Kept:
    def __init__(self) -> None:
        self.runs: list[Any] = []

    async def record(self, run: Any, *, skip: Collection[str] = ()) -> None:
        self.runs.append(run)


async def test_a_routed_run_whose_start_handler_raises_sits_in_its_router(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    root = write_folder(
        tmp_path / "agents",
        {
            "router": {
                "kind": "decision",
                "model": "judge",
                "description": "Routes",
                "instructions": "router.md",
                "questions": ["router/*.json"],
                "orchestration": {"route": {"by": "team", "to": {"billing": "chat"}}},
            },
            "chat": {
                "kind": "generation",
                "model": "strong",
                "description": "Talks",
                "instructions": "chat.md",
            },
        },
        files={"chat.md": "You are chat.\n", **ROUTER_TEAM},
    )

    async def breaking(event: Event) -> None:
        if isinstance(event, RunStarted):
            raise RuntimeError("the socket is gone")

    kept = Kept()
    decision = Fixed(
        {"team": ChoiceAnswer(choice="billing", probabilities={"billing": 0.9, "other": 0.1})}
    )
    llm = Scripted()
    engine = Engine(root, refinery=kept, providers=provided({"strong": llm, "judge": decision}))
    with pytest.raises(RuntimeError, match="the socket is gone"):
        await engine.run("router", "acme", "Charged twice", on_event=breaking)
    [router] = kept.runs
    assert isinstance(router, Decision) and isinstance(router.routed_to, Generation)
    routed = router.routed_to
    assert routed.error == "RuntimeError: the socket is gone" and routed.calls == 0
    assert llm.calls == []


async def test_every_generation_run_ends_with_run_ended(tmp_path: Path) -> None:
    llm = ByAgent(
        [
            Reply(
                tool_calls=[
                    ToolCall(id="d1", name="Delegate", args={"agent": "chat", "input": "A"})
                ],
                stop_reason="tool_use",
            ),
            text_reply("Done."),
        ],
        chat=looking,
    )
    collect = Collect()
    run = await Engine(
        make_tree(tmp_path, run_tools={"lead": ["lookup"]}),
        providers=provided({"strong": llm, "judge": None}),
    ).run("lead", "acme", "Go", tools=TOOLS, on_event=collect)
    assert isinstance(run, Generation)
    [child] = run.children
    ended = [e for e in collect.events if isinstance(e, RunEnded)]
    assert len(ended) == 2
    for event in ended:
        own = [e for e in collect.events if e.run_id == event.run_id]
        assert own[-1] is event
        assert event.stopped is None and event.error is None
    by_run = {e.run_id: e for e in ended}
    assert by_run[run.run_id].call is None and by_run[run.run_id].agent == "lead"
    assert by_run[child.run_id].agent == "chat"
    assert by_run[child.run_id].call == started(collect)[child.run_id].call
    child_call = by_run[child.run_id].call
    assert child_call is not None and child_call.run_id == run.run_id


async def test_run_ended_carries_the_stop_and_the_error(tmp_path: Path) -> None:
    stop = Stop()
    stop.now()
    stopped = Collect()
    await Engine(
        make_tree(tmp_path / "a"), providers=provided({"strong": Scripted(), "judge": None})
    ).run("chat", "acme", "Hi", on_event=stopped, stop=stop)
    [event] = [e for e in stopped.events if isinstance(e, RunEnded)]
    assert event.stopped == "now" and event.error is None

    failed = Collect()
    call = ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})
    looped = Reply(tool_calls=[call], stop_reason="tool_use")
    with pytest.raises(RunawayError):
        await Engine(
            make_tree(tmp_path / "b", run_tools={"chat": ["lookup"]}),
            max_model_calls=2,
            providers=provided({"strong": Scripted(looped, looped), "judge": None}),
        ).run("chat", "acme", "Hi", tools=TOOLS, on_event=failed)
    [event] = [e for e in failed.events if isinstance(e, RunEnded)]
    assert event.stopped is None and event.error is not None
    assert event.error == "RunawayError: engine: run exceeded 2 model calls"
    assert failed.events[-1] is event


async def test_a_decision_run_sends_no_run_ended(tmp_path: Path) -> None:
    root = tmp_path / "agents"
    root.mkdir(parents=True)
    root = write_folder(
        root,
        {
            "router": {
                "kind": "decision",
                "model": "judge",
                "description": "Routes",
                "instructions": "router.md",
                "questions": ["router/*.json"],
            }
        },
        files=ROUTER_TEAM,
    )
    collect = Collect()
    run = await Engine(
        root,
        providers=provided(
            {
                "strong": None,
                "judge": Fixed(
                    {
                        "team": ChoiceAnswer(
                            choice="billing", probabilities={"billing": 0.9, "other": 0.1}
                        )
                    }
                ),
            }
        ),
    ).run("router", "acme", "Help", on_event=collect)
    assert isinstance(run, Decision)
    assert [type(e) for e in collect.events] == [ReplyReady]


class Ending(Collect):
    async def __call__(self, event: Event) -> None:
        await super().__call__(event)
        if isinstance(event, RunEnded):
            raise RuntimeError("the audit log is down")


async def test_a_handler_that_raises_on_run_ended_fails_the_run_without_a_second_one(
    tmp_path: Path,
) -> None:
    kept = Kept()
    collect = Ending()
    with pytest.raises(RuntimeError, match="the audit log is down"):
        await Engine(
            make_tree(tmp_path),
            refinery=kept,
            providers=provided({"strong": Scripted(text_reply("Hi.")), "judge": None}),
        ).run("chat", "acme", "Hi", on_event=collect)
    [event] = [e for e in collect.events if isinstance(e, RunEnded)]
    assert event.error is None and collect.events[-1] is event
    [record] = kept.runs
    assert isinstance(record, Generation)
    assert record.error == "RuntimeError: the audit log is down"


def ended_last(collect: Collect, run: Generation) -> RunEnded:
    own = [e for e in collect.events if e.run_id == run.run_id]
    [event] = [e for e in own if isinstance(e, RunEnded)]
    assert own[-1] is event
    return event


async def test_a_stop_after_the_step_ends_with_after_step(tmp_path: Path) -> None:
    stop = Stop()

    class After(Collect):
        async def __call__(self, event: Event) -> None:
            await super().__call__(event)
            if isinstance(event, ToolResultReady):
                stop.after_step()

    collect = After()
    call = ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})
    run = await Engine(
        make_tree(tmp_path, run_tools={"chat": ["lookup"]}),
        providers=provided(
            {
                "strong": Scripted(Reply(tool_calls=[call], stop_reason="tool_use")),
                "judge": None,
            }
        ),
    ).run("chat", "acme", "Hi", tools=TOOLS, stop=stop, on_event=collect)
    assert isinstance(run, Generation) and run.stopped == "after_step"
    event = ended_last(collect, run)
    assert event.stopped == "after_step" and event.error is None


async def test_an_accepted_output_ends_with_one_run_ended(tmp_path: Path) -> None:
    collect = Collect()
    reply = Reply(
        tool_calls=[ToolCall(id="o1", name="Output", args={"ok": True})], stop_reason="tool_use"
    )
    run = await Engine(
        make_tree(tmp_path),
        providers=provided({"strong": Scripted(reply), "judge": None}),
    ).run("chat", "acme", "Hi", expect=Verdict, on_event=collect)
    assert isinstance(run, Generation) and run.reply == '{"ok":true}'
    event = ended_last(collect, run)
    assert event.stopped is None and event.error is None


class Waiting:
    def __init__(self) -> None:
        self.asked = asyncio.Event()

    async def generate(self, system: str, messages: list[Message], tools: list[Any]) -> Reply:
        self.asked.set()
        await asyncio.sleep(30)
        return text_reply("never")


async def test_a_stop_while_the_model_is_called_ends_with_now(tmp_path: Path) -> None:
    stop = Stop()
    collect = Collect()
    waiting = Waiting()
    work = asyncio.create_task(
        Engine(
            make_tree(tmp_path),
            providers=provided({"strong": waiting, "judge": None}),
        ).run("chat", "acme", "Hi", stop=stop, on_event=collect)
    )
    await asyncio.wait_for(waiting.asked.wait(), 5)
    stop.now()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "now"
    event = ended_last(collect, run)
    assert event.stopped == "now" and event.error is None


def lookup_call() -> Reply:
    return Reply(
        tool_calls=[ToolCall(id="c1", name="lookup", args={"order_id": "A-1"})],
        stop_reason="tool_use",
        usage=Usage(input=10, output=2),
        model="m1",
        settings={"temperature": 0.5},
    )


async def two_replies(tmp_path: Path, collect: Collect) -> Generation:
    llm = Scripted(lookup_call(), text_reply("Done.", usage=Usage(input=20, output=4), model="m2"))
    run = await Engine(
        make_tree(tmp_path, run_tools={"lead": ["lookup"]}),
        providers=provided({"strong": llm, "judge": None}),
    ).run("lead", "acme", "Check A-1", tools=TOOLS, on_event=collect)
    assert isinstance(run, Generation)
    return run


async def test_each_model_reply_sends_reply_ready(tmp_path: Path) -> None:
    collect = Collect()
    run = await two_replies(tmp_path, collect)
    ready = [e for e in collect.events if isinstance(e, ReplyReady)]
    assert ready == [
        ReplyReady(
            run_id=run.run_id,
            agent="lead",
            turn=0,
            model="m1",
            usage=Usage(input=10, output=2),
            settings={"temperature": 0.5},
            stop_reason="tool_use",
        ),
        ReplyReady(
            run_id=run.run_id,
            agent="lead",
            turn=1,
            model="m2",
            usage=Usage(input=20, output=4),
            settings=ready[1].settings,
            stop_reason="end_turn",
        ),
    ]
    first_tool = next(i for i, e in enumerate(collect.events) if isinstance(e, ToolCallStarted))
    assert collect.events.index(ready[0]) < first_tool < collect.events.index(ready[1])


async def test_reply_ready_sums_to_the_run_s_usage(tmp_path: Path) -> None:
    collect = Collect()
    run = await two_replies(tmp_path, collect)
    total = Usage()
    for e in collect.events:
        if isinstance(e, ReplyReady):
            total = total + e.usage
    assert total == run.usage == Usage(input=30, output=6)


async def test_a_refused_output_reply_still_sends_reply_ready(tmp_path: Path) -> None:
    collect = Collect()
    bad = Reply(
        tool_calls=[ToolCall(id="o1", name="Output", args={"ok": "nope"})],
        stop_reason="tool_use",
        usage=Usage(input=3),
    )
    good = Reply(
        tool_calls=[ToolCall(id="o2", name="Output", args={"ok": True})],
        stop_reason="tool_use",
        usage=Usage(input=4),
    )
    await Engine(
        make_tree(tmp_path), providers=provided({"strong": Scripted(bad, good), "judge": None})
    ).run("chat", "acme", "Hi", expect=Verdict, on_event=collect)
    ready = [e for e in collect.events if isinstance(e, ReplyReady)]
    assert [(e.turn, e.usage) for e in ready] == [(0, Usage(input=3)), (1, Usage(input=4))]


class Raising:
    async def generate(self, system: str, messages: list[Message], tools: list[Any]) -> Reply:
        raise RuntimeError("the model is down")


async def test_a_stopped_or_failed_model_call_sends_no_reply_ready(tmp_path: Path) -> None:
    stop, collect, waiting = Stop(), Collect(), Waiting()
    work = asyncio.create_task(
        Engine(make_tree(tmp_path), providers=provided({"strong": waiting, "judge": None})).run(
            "chat", "acme", "Hi", stop=stop, on_event=collect
        )
    )
    await asyncio.wait_for(waiting.asked.wait(), 5)
    stop.now()
    await asyncio.wait_for(work, 5)
    failed = Collect()
    with pytest.raises(RuntimeError, match="the model is down"):
        await Engine(
            make_tree(tmp_path / "again"), providers=provided({"strong": Raising(), "judge": None})
        ).run("chat", "acme", "Hi", on_event=failed)
    assert not any(isinstance(e, ReplyReady) for e in [*collect.events, *failed.events])


async def test_a_raising_reply_ready_handler_fails_the_run_after_the_store_has_the_reply(
    tmp_path: Path,
) -> None:
    store = MemoryStore()

    async def handler(event: Event) -> None:
        if isinstance(event, ReplyReady):
            raise OSError("handler broke")

    with pytest.raises(OSError, match="handler broke"):
        await Engine(
            make_tree(tmp_path),
            providers=provided({"strong": Scripted(text_reply("Hello.")), "judge": None}),
        ).run("chat", "acme", "Hi", store=store, conversation="c1", on_event=handler)
    held = store.held[("acme", "c1")]
    assert [m.content for m in held if m.role == "assistant"] == ["Hello."]
