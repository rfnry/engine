import asyncio
import contextlib
import gc
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine import Event, Stop, TextDelta, ToolCallStarted, ToolResultReady
from rfnry_engine.engine import Engine
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Scripted, Streaming, text_reply

STOPPED = "stopped by the user"


class Verdict(BaseModel):
    ok: bool


def make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    for name, front in {
        "lead": "kind: generation\ndescription: Leads\nagents: [chat]",
        "chat": "kind: generation\ndescription: Talks",
    }.items():
        (root / name).mkdir(parents=True)
        (root / name / "AGENT.md").write_text(
            f"---\nname: {name}\n{front}\n---\nYou are {name}.\n", encoding="utf-8"
        )
    return root


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
    engine = Engine(make_tree(tmp_path), generation=llm, tool_timeout=0.05)
    run = await engine.run(
        "lead", "acme", "Handle A-1", tools=TOOLS, approve=approve, on_event=collect
    )
    assert isinstance(run, Generation) and run.reply == "Done."
    check(run, calls, collect.of(run.run_id))
    ok, failed, late, refused, delegated = run.steps
    assert ok.result == '{"order": "A-1", "status": "shipped"}' and not ok.is_error
    assert failed.result == "RuntimeError: the ledger is down" and failed.is_error
    assert late.result == "TimeoutError: slow timed out after 0.05s" and late.is_error
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
        make_tree(tmp_path), generation=Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
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
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=handler
        ),
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
        Engine(make_tree(tmp_path), generation=llm).run(
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
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=handler
        ),
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
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=handler
        ),
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
        Engine(make_tree(tmp_path), generation=llm).run(
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
        Engine(make_tree(tmp_path), generation=llm).run(
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
        Engine(make_tree(tmp_path), generation=llm).run(
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

    async def stuck(event: Event) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == "c2":
            stop.now()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                left.set()

    async def slow_to_die(args: dict[str, Any]) -> str:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            dying.set()
            await asyncio.shield(release.wait())
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
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=tools, stop=stop, on_event=stuck
        )
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
    assert died.is_set()
    assert asyncio.all_tasks() <= before


async def test_cancelling_a_run_cancels_the_handler_of_a_call_that_swallowed_the_stop(
    tmp_path: Path,
) -> None:
    stop = Stop()
    held = {"c1": asyncio.Event(), "c2": asyncio.Event()}
    left = {"c1": asyncio.Event(), "c2": asyncio.Event()}

    async def holding(event: Event) -> None:
        if isinstance(event, ToolResultReady):
            if event.call_id == "c2":
                stop.now()
            held[event.call_id].set()
            try:
                await asyncio.Event().wait()
            finally:
                left[event.call_id].set()

    async def swallows(args: dict[str, Any]) -> str:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(30)
        return "kept going"

    before = asyncio.all_tasks()
    calls = [
        ToolCall(id="c1", name="swallows", args={}),
        ToolCall(id="c2", name="lookup", args={"order_id": "A-1"}),
    ]
    tools = Tools([Tool("swallows", body=swallows), Tool("lookup", body=lookup)])
    llm = Scripted(Reply(tool_calls=calls, stop_reason="tool_use"))
    running = asyncio.create_task(
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=tools, stop=stop, on_event=holding
        )
    )
    async with asyncio.timeout(2):
        await held["c2"].wait()
        await held["c1"].wait()
    running.cancel()
    async with asyncio.timeout(2):
        await left["c1"].wait()
        await left["c2"].wait()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(2):
            await running
    assert asyncio.all_tasks() <= before


class Dying:
    def __init__(self, kept: str | None = None) -> None:
        self.entered, self.died = asyncio.Event(), asyncio.Event()
        self.kept = kept

    async def __call__(self, *args: Any) -> Any:
        self.entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.sleep(0.2))
            self.died.set()
            if self.kept is None:
                raise
            return self.kept
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
    assert dying.died.is_set()


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
        Engine(make_tree(tmp_path), generation=llm).run("lead", "acme", "Check A-1", tools=tools)
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
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=Tools([Tool("dying", body=dying)])
        )
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
        Engine(make_tree(tmp_path), generation=llm).run(
            "lead", "acme", "Check A-1", tools=TOOLS, approve=approve
        )
    )
    await cancel_while_dying(running, dying, 2)
    assert asyncio.all_tasks() <= before


@pytest.mark.parametrize("cancelled", [False, True])
async def test_a_run_with_a_stop_leaves_no_task_behind(tmp_path: Path, cancelled: bool) -> None:
    dying = Dying()
    reply = (
        Reply(tool_calls=[ToolCall(id="c1", name="dying", args={})], stop_reason="tool_use")
        if cancelled
        else text_reply("Done.")
    )
    engine = Engine(make_tree(tmp_path), generation=Scripted(reply))
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
            await Engine(make_tree(tmp_path), generation=llm).run(
                "lead", "acme", "Check A-1", tools=TOOLS, approve=approve
            )
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
        Engine(make_tree(tmp_path), generation=Scripted(reply)).run(
            "lead", "acme", "Finish", tools=TOOLS, stop=stop, on_event=kill
        ),
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
        Engine(make_tree(tmp_path), generation=llm, max_steps=1).run(
            "lead", "acme", "Check A-1", tools=TOOLS, stop=stop, on_event=kill
        ),
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
    run = await Engine(make_tree(tmp_path), generation=llm).run(
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
