import asyncio
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from rfnry_engine import MessagesTaken, RunEnded, RunStarted, Stop
from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError, NotFoundError
from rfnry_engine.models import (
    ChoiceAnswer,
    Message,
    QuestionLike,
    Reply,
    State,
    Text,
    ToolCall,
    ToolSpec,
)
from rfnry_engine.records import Decision, Generation, Run, Spoken, SpokenLine, VoiceStart
from rfnry_engine.stop import requested
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import text_reply
from .trees import provided, write_folder

STOPPED = "stopped by the user"
INTENT = '{"type": "choice", "instructions": "What is this?", "options": {"talk": "Just talk", "other": "Anything else"}}'


def make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    root.mkdir(parents=True)
    return write_folder(
        root,
        {
            "lead": {
                "kind": "generation",
                "model": "lead",
                "description": "Leads",
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["helper", "desk", "deputy", "switch", "router"]},
            },
            "front": {
                "kind": "generation",
                "model": "lead",
                "description": "Consults",
                "instructions": "lead.md",
                "tools": {"allow": ["consult"]},
                "orchestration": {"delegate_to": ["helper"]},
            },
            "helper": {
                "kind": "generation",
                "model": "helper",
                "description": "Looks things up",
                "instructions": "helper.md",
                "tools": {"allow": ["lookup"]},
            },
            "deputy": {
                "kind": "generation",
                "model": "deputy",
                "description": "Asks around",
                "instructions": "helper.md",
                "tools": {"allow": ["consult"]},
            },
            "desk": {
                "kind": "decision",
                "model": "judge",
                "description": "Sorts",
                "instructions": "desk.md",
                "questions": ["desk/*.json"],
            },
            "router": {
                "kind": "decision",
                "model": "judge",
                "description": "Routes",
                "instructions": "desk.md",
                "questions": ["router/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"talk": "helper"}}},
            },
            "switch": {
                "kind": "decision",
                "model": "switch",
                "description": "Routes to the desk",
                "instructions": "desk.md",
                "questions": ["switch/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"talk": "desk"}}},
            },
            "voice": {
                "kind": "voice",
                "model": "live",
                "description": "Talks",
                "instructions": "helper.md",
                "tools": {"allow": ["consult"]},
            },
        },
        runtime_tools=["consult", "lookup"],
        files={
            "lead.md": "You lead.\n",
            "helper.md": "You look things up.\n",
            "desk.md": "You sort.\n",
            "desk/intent.json": INTENT,
            "router/intent.json": INTENT,
            "switch/intent.json": INTENT,
        },
    )


class Model:
    def __init__(self, *replies: Reply | Exception, gate: asyncio.Event | None = None) -> None:
        self.replies = list(replies)
        self.gate = gate
        self.calls: list[list[Message]] = []
        self.asked = asyncio.Event()

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.calls.append(list(messages))
        self.asked.set()
        if self.gate is not None:
            await self.gate.wait()
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class Desk:
    def __init__(self, open: bool = False) -> None:
        self.gate = asyncio.Event()
        self.asked = asyncio.Event()
        self.cancelled = False
        self.cancelled_at: float | None = None
        if open:
            self.gate.set()

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
        self.asked.set()
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            self.cancelled_at = time.monotonic()
            raise
        return {"intent": ChoiceAnswer(choice="talk", probabilities={"talk": 0.9, "other": 0.1})}


class Stubborn:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.ended_at: float | None = None

    async def __call__(self, args: dict[str, Any]) -> str:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.sleep(0.3))
            self.ended_at = time.monotonic()
            raise
        return "never"


class Live:
    model = "live-1"

    async def open(self, *, tools: list[ToolSpec], instructions: str | None) -> object:
        return object()


class SlowRecord:
    def __init__(self) -> None:
        self.runs: list[Run] = []
        self.writing: list[Run] = []
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.cancelled = False

    async def record(self, run: Run) -> None:
        self.writing.append(run)
        self.started.set()
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.runs.append(run)


class Kept:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


class Lookup:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def __call__(self, args: dict[str, Any]) -> str:
        self.started.set()
        await self.gate.wait()
        return "found"


class Seen:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def __call__(self, event: Any) -> None:
        self.events.append(event)

    def started(self, below: bool) -> list[str]:
        return [
            e.run_id
            for e in self.events
            if isinstance(e, RunStarted) and (e.call is not None) is below
        ]


async def until(check: Callable[[], bool], what: str) -> None:
    try:
        async with asyncio.timeout(10):
            while not check():
                await asyncio.sleep(0.01)
    except TimeoutError:
        pytest.fail(f"{what} never happened")


def ended_once(seen: Seen, run_id: str) -> RunEnded:
    own: list[Any] = [e for e in seen.events if getattr(e, "run_id", None) == run_id]
    [event] = [e for e in own if isinstance(e, RunEnded)]
    assert own[-1] is event
    return cast(RunEnded, event)


async def child_of(seen: Seen) -> str:
    await until(lambda: bool(seen.started(below=True)), "the child's RunStarted")
    [child_id] = seen.started(below=True)
    return child_id


def use(name: str, call_id: str, **args: Any) -> Reply:
    return Reply(tool_calls=[ToolCall(id=call_id, name=name, args=args)], stop_reason="tool_use")


def delegating() -> Reply:
    return use("Delegate", "d1", agent="helper", input="find the invoice")


def engine_of(
    tmp_path: Path,
    lead: Any = None,
    helper: Any = None,
    desk: Any = None,
    deputy: Any = None,
    refinery: Kept | SlowRecord | None = None,
    switch: Any = None,
    live: Any = None,
) -> Engine:
    return Engine(
        make_tree(tmp_path),
        refinery=refinery,
        providers=provided(
            {
                "lead": lead,
                "helper": helper,
                "judge": desk,
                "deputy": deputy,
                "switch": switch,
                "live": live,
            }
        ),
    )


def decision_in_flight(engine: Engine) -> str:
    [run_id] = [run_id for (_, run_id), held in engine._running.items() if held.kind == "decision"]
    return run_id


def tools_of(lookup: Lookup | None = None) -> Tools:
    return Tools([Tool("lookup", "Look a thing up", body=lookup or Lookup())])


async def test_stopping_one_delegated_run_leaves_its_parent_running(tmp_path: Path) -> None:
    lead = Model(delegating(), text_reply("went on"))
    helper = Model(text_reply("never"), gate=asyncio.Event())
    engine = engine_of(tmp_path, lead, helper)
    seen = Seen()
    work = asyncio.create_task(engine.run("lead", "acme", "x", tools=tools_of(), on_event=seen))
    child_id = await child_of(seen)
    await helper.asked.wait()
    engine.stop(child_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation)
    [child] = run.children
    assert isinstance(child, Generation) and child.run_id == child_id
    assert child.stopped == "now"
    assert ended_once(seen, child_id).stopped == "now"
    assert ended_once(seen, run.run_id).stopped is None
    step = run.steps[0]
    assert (step.tool, step.result, step.is_error, step.stopped, step.child_run_id) == (
        "Delegate",
        STOPPED,
        True,
        True,
        child_id,
    )
    assert len(lead.calls) == 2 and run.calls == 2
    assert run.stopped is None and run.reply == "went on"


async def test_stopping_a_run_stops_the_runs_below_it(tmp_path: Path) -> None:
    lead = Model(delegating())
    helper = Model(text_reply("never"), gate=asyncio.Event())
    engine = engine_of(tmp_path, lead, helper)
    seen = Seen()
    work = asyncio.create_task(engine.run("lead", "acme", "x", tools=tools_of(), on_event=seen))
    child_id = await child_of(seen)
    await helper.asked.wait()
    [top_id] = seen.started(below=False)
    engine.stop(top_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.run_id == top_id
    assert run.stopped == "now" and len(lead.calls) == 1
    [child] = run.children
    assert isinstance(child, Generation) and child.run_id == child_id
    assert child.stopped == "now"


async def test_after_step_lets_the_stopped_run_finish_its_step(tmp_path: Path) -> None:
    lead = Model(delegating(), text_reply("went on"))
    helper = Model(use("lookup", "l1", q="invoice"), text_reply("never"))
    lookup = Lookup()
    engine = engine_of(tmp_path, lead, helper)
    seen = Seen()
    work = asyncio.create_task(
        engine.run("lead", "acme", "x", tools=tools_of(lookup), on_event=seen)
    )
    child_id = await child_of(seen)
    await lookup.started.wait()
    engine.stop(child_id, scope="acme", after_step=True)
    lookup.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation)
    [child] = run.children
    assert isinstance(child, Generation) and child.run_id == child_id
    assert [(s.tool, s.result, s.is_error, s.stopped) for s in child.steps] == [
        ("lookup", "found", False, False)
    ]
    assert child.stopped == "after_step" and len(helper.calls) == 1
    assert ended_once(seen, child_id).stopped == "after_step"
    assert ended_once(seen, run.run_id).stopped is None
    assert (run.steps[0].result, run.steps[0].stopped) == (STOPPED, True)
    assert run.stopped is None and run.reply == "went on"


async def test_a_message_sent_to_a_delegated_run_is_read_before_its_next_model_call(
    tmp_path: Path,
) -> None:
    lead = Model(delegating(), text_reply("went on"))
    helper = Model(use("lookup", "l1", q="invoice"), text_reply("found it, March too"))
    lookup = Lookup()
    engine = engine_of(tmp_path, lead, helper)
    seen = Seen()
    work = asyncio.create_task(
        engine.run("lead", "acme", "x", tools=tools_of(lookup), on_event=seen)
    )
    child_id = await child_of(seen)
    await lookup.started.wait()
    await engine.send(child_id, "also March", scope="acme")
    lookup.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "went on"
    assert helper.calls[1][-2:] == [
        Message(role="tool", tool_call_id="l1", content="found"),
        Message(role="user", content="also March"),
    ]
    taken = [e for e in seen.events if isinstance(e, MessagesTaken)]
    assert [(e.run_id, e.turn) for e in taken] == [(child_id, 1)]
    assert taken[0].messages[0].content == "also March"
    [child] = run.children
    assert isinstance(child, Generation) and child.reply == "found it, March too"
    assert run.steps[0].result == "found it, March too"


async def test_send_reaches_the_queue_a_top_level_run_was_given(tmp_path: Path) -> None:
    helper = Model(use("lookup", "l1", q="invoice"), text_reply("done"))
    lookup = Lookup()
    engine = engine_of(tmp_path, helper=helper)
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    seen = Seen()
    work = asyncio.create_task(
        engine.run("helper", "acme", "x", tools=tools_of(lookup), inbox=inbox, on_event=seen)
    )
    await lookup.started.wait()
    [top_id] = seen.started(below=False)
    await engine.send(top_id, "also March", scope="acme")
    assert inbox.qsize() == 1
    lookup.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.run_id == top_id
    assert helper.calls[1][-2:] == [
        Message(role="tool", tool_call_id="l1", content="found"),
        Message(role="user", content="also March"),
    ]
    assert [e.run_id for e in seen.events if isinstance(e, MessagesTaken)] == [top_id]
    assert inbox.empty()


async def test_send_to_a_decision_run_is_refused(tmp_path: Path) -> None:
    desk = Desk()
    engine = engine_of(tmp_path, desk=desk)
    work = asyncio.create_task(engine.run("desk", "acme", "x"))
    await desk.asked.wait()
    [(scope, desk_id)] = list(engine._running)
    with pytest.raises(InputError) as refused:
        await engine.send(desk_id, "also March", scope="acme")
    assert str(refused.value) == (
        f"engine: run {desk_id!r} is a decision run and reads no messages"
    )
    desk.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Decision) and (scope, run.run_id) == ("acme", desk_id)


async def not_running(engine: Engine, run_id: str, scope: str) -> None:
    text = f"engine: run {run_id!r} is not running"
    with pytest.raises(NotFoundError) as sent:
        await engine.send(run_id, "also March", scope=scope)
    with pytest.raises(NotFoundError) as stopped:
        engine.stop(run_id, scope=scope)
    with pytest.raises(NotFoundError) as stopped_after:
        engine.stop(run_id, scope=scope, after_step=True)
    assert [str(sent.value), str(stopped.value), str(stopped_after.value)] == [text] * 3


async def test_a_run_not_in_flight_or_of_another_scope_is_not_found(tmp_path: Path) -> None:
    helper = Model(text_reply("done"), gate=asyncio.Event())
    engine = engine_of(tmp_path, helper=helper)
    seen = Seen()
    work = asyncio.create_task(engine.run("helper", "acme", "x", tools=tools_of(), on_event=seen))
    await helper.asked.wait()
    [top_id] = seen.started(below=False)
    await not_running(engine, "ghost", "acme")
    await not_running(engine, top_id, "other")
    assert helper.gate is not None
    helper.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.run_id == top_id and run.stopped is None
    await not_running(engine, top_id, "acme")
    assert engine._running == {}


async def test_stopping_a_consult_leaves_the_run_that_called_it_running(tmp_path: Path) -> None:
    lead = Model(use("consult", "c1", input="find the invoice"), text_reply("went on"))
    helper = Model(text_reply("never"), gate=asyncio.Event())
    engine = engine_of(tmp_path, lead, helper)
    consult = engine.tool("helper", scope="acme", name="consult", tools=tools_of())
    seen = Seen()
    work = asyncio.create_task(engine.run("front", "acme", "x", tools=consult, on_event=seen))
    consult_id = await child_of(seen)
    await helper.asked.wait()
    engine.stop(consult_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation)
    assert [(s.tool, s.result, s.stopped) for s in run.steps] == [("consult", STOPPED, True)]
    assert run.stopped is None and run.reply == "went on" and len(lead.calls) == 2


async def test_a_run_s_stop_leaves_the_stop_above_it_when_the_run_ends(tmp_path: Path) -> None:
    lead = Model(delegating(), use("consult", "c1", input="again"), text_reply("done"))
    helper = Model(
        use("lookup", "l1", q="a"), text_reply("one"), use("lookup", "l2", q="b"), text_reply("two")
    )
    lookup = Lookup()
    lookup.gate.set()
    engine = engine_of(tmp_path, lead, helper)
    consult = engine.tool("helper", scope="acme", name="consult", tools=tools_of(lookup))
    stop = Stop()
    run = await engine.run("front", "acme", "x", tools=tools_of(lookup) + consult, stop=stop)
    assert isinstance(run, Generation) and run.reply == "done" and len(run.children) == 1
    assert [s.result for s in run.steps] == ["one", "two"]
    assert stop._children == set()


@pytest.mark.parametrize("mode", ["now", "after_step"])
def test_a_stop_made_below_a_pressed_one_starts_pressed(mode: str) -> None:
    above = Stop()
    getattr(above, mode)()
    assert requested(above._below()) == mode


def test_a_press_reaches_every_stop_below_it_and_none_above() -> None:
    top = Stop()
    middle = top._below()
    bottom = middle._below()
    middle.after_step()
    assert [requested(s) for s in (top, middle, bottom)] == [None, "after_step", "after_step"]
    top.now()
    assert [requested(s) for s in (top, middle, bottom)] == ["now", "now", "now"]


def test_a_stop_that_dropped_out_is_not_reached_by_its_parent() -> None:
    above = Stop()
    kept, gone = above._below(), above._below()
    gone._drop()
    above.now()
    assert (requested(kept), requested(gone)) == ("now", None)
    assert above._children == {kept}


async def test_stopping_a_delegated_decision_run_leaves_its_parent_running(tmp_path: Path) -> None:
    lead = Model(use("Delegate", "d1", agent="desk", input="sort this"), text_reply("went on"))
    desk = Desk()
    engine = engine_of(tmp_path, lead, desk=desk)
    work = asyncio.create_task(engine.run("lead", "acme", "x"))
    await desk.asked.wait()
    engine.stop(decision_in_flight(engine), scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and desk.cancelled
    step = run.steps[0]
    assert (step.tool, step.result, step.is_error, step.stopped, step.child_run_id) == (
        "Delegate",
        STOPPED,
        True,
        True,
        None,
    )
    assert run.children == [] and len(lead.calls) == 2
    assert run.stopped is None and run.reply == "went on"
    assert engine._running == {}


async def test_stopping_a_consulted_decision_run_leaves_the_run_that_called_it_running(
    tmp_path: Path,
) -> None:
    front = Model(use("consult", "c1", input="sort this"), text_reply("went on"))
    desk = Desk()
    kept = Kept()
    engine = engine_of(tmp_path, front, desk=desk, refinery=kept)
    consult = engine.tool("desk", scope="acme", name="consult")
    work = asyncio.create_task(engine.run("front", "acme", "x", tools=consult))
    await desk.asked.wait()
    engine.stop(decision_in_flight(engine), scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and desk.cancelled
    assert [(s.tool, s.result, s.is_error, s.stopped) for s in run.steps] == [
        ("consult", STOPPED, True, True)
    ]
    assert run.stopped is None and run.reply == "went on"
    assert [r.agent for r in kept.runs] == ["front"] and engine._running == {}


@pytest.mark.parametrize("pressed", ["by_id", "by_its_stop"])
async def test_a_top_level_decision_run_takes_a_stop_by_its_id_as_it_takes_its_stop(
    tmp_path: Path, pressed: str
) -> None:
    router = Desk()
    helper = Model(text_reply("never"))
    engine = engine_of(tmp_path, helper=helper, desk=router)
    stop = Stop()
    work = asyncio.create_task(engine.run("router", "acme", "x", tools=tools_of(), stop=stop))
    await router.asked.wait()
    router_id = decision_in_flight(engine)
    if pressed == "by_id":
        engine.stop(router_id, scope="acme")
    else:
        stop.now()
    await asyncio.sleep(0.05)
    router.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Decision) and run.run_id == router_id and not router.cancelled
    answer = run.answers["intent"]
    assert isinstance(answer, ChoiceAnswer) and answer.choice == "talk"
    routed = run.routed_to
    assert isinstance(routed, Generation) and routed.stopped == "now" and routed.calls == 0
    assert helper.calls == [] and requested(stop) == "now"


async def test_a_decision_run_a_top_level_router_stopped_by_its_id_then_routes_to_answers(
    tmp_path: Path,
) -> None:
    switch = Desk()
    desk = Desk(open=True)
    kept = Kept()
    engine = engine_of(tmp_path, desk=desk, switch=switch, refinery=kept)
    work = asyncio.create_task(engine.run("switch", "acme", "x"))
    await switch.asked.wait()
    engine.stop(decision_in_flight(engine), scope="acme")
    await asyncio.sleep(0.05)
    switch.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Decision) and not switch.cancelled and not desk.cancelled
    routed = run.routed_to
    assert isinstance(routed, Decision) and sorted(routed.answers) == ["intent"]
    assert kept.runs == [run] and engine._running == {}


async def test_send_reaches_a_routed_run(tmp_path: Path) -> None:
    helper = Model(use("lookup", "l1", q="invoice"), text_reply("done"))
    lookup = Lookup()
    engine = engine_of(tmp_path, helper=helper, desk=Desk(open=True))
    seen = Seen()
    work = asyncio.create_task(
        engine.run("router", "acme", "x", tools=tools_of(lookup), on_event=seen)
    )
    await lookup.started.wait()
    [routed_id] = seen.started(below=False)
    await engine.send(routed_id, "also March", scope="acme")
    lookup.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.run_id == routed_id and run.routed_to.reply == "done"
    assert helper.calls[1][-1] == Message(role="user", content="also March")
    assert [e.run_id for e in seen.events if isinstance(e, MessagesTaken)] == [routed_id]


async def test_stop_reaches_a_routed_run_and_not_the_run_that_routed_it(tmp_path: Path) -> None:
    helper = Model(text_reply("never"), gate=asyncio.Event())
    engine = engine_of(tmp_path, helper=helper, desk=Desk(open=True))
    stop = Stop()
    seen = Seen()
    work = asyncio.create_task(
        engine.run("router", "acme", "x", tools=tools_of(), stop=stop, on_event=seen)
    )
    await helper.asked.wait()
    [routed_id] = seen.started(below=False)
    engine.stop(routed_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.run_id == routed_id and run.routed_to.stopped == "now"
    assert requested(stop) is None


async def test_send_reaches_a_top_level_run_given_no_inbox(tmp_path: Path) -> None:
    helper = Model(use("lookup", "l1", q="invoice"), text_reply("done"))
    lookup = Lookup()
    engine = engine_of(tmp_path, helper=helper)
    seen = Seen()
    work = asyncio.create_task(
        engine.run("helper", "acme", "x", tools=tools_of(lookup), on_event=seen)
    )
    await lookup.started.wait()
    [top_id] = seen.started(below=False)
    await engine.send(top_id, "also March", scope="acme")
    lookup.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    assert helper.calls[1][-2:] == [
        Message(role="tool", tool_call_id="l1", content="found"),
        Message(role="user", content="also March"),
    ]
    assert [e.run_id for e in seen.events if isinstance(e, MessagesTaken)] == [top_id]


@pytest.mark.parametrize("end", ["stopped", "failed", "cancelled"])
async def test_a_run_that_ends_mid_flight_leaves_nothing_in_flight(
    tmp_path: Path, end: str
) -> None:
    lead = Model(delegating(), RuntimeError("the lead failed"))
    helper = Model(RuntimeError("the helper failed"), gate=asyncio.Event())
    engine = engine_of(tmp_path, lead, helper)
    stop = Stop()
    seen = Seen()
    work = asyncio.create_task(
        engine.run("lead", "acme", "x", tools=tools_of(), stop=stop, on_event=seen)
    )
    child_id = await child_of(seen)
    await helper.asked.wait()
    [top_id] = seen.started(below=False)
    assert set(engine._running) == {("acme", top_id), ("acme", child_id)}
    assert len(stop._children) == 1
    if end == "stopped":
        engine.stop(top_id, scope="acme")
        run = await asyncio.wait_for(work, 5)
        assert isinstance(run, Generation) and run.stopped == "now"
    elif end == "failed":
        assert helper.gate is not None
        helper.gate.set()
        with pytest.raises(RuntimeError, match="the lead failed"):
            await asyncio.wait_for(work, 5)
    else:
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
    assert engine._running == {} and stop._children == set()


async def test_a_message_json_cannot_hold_is_refused_where_it_is_sent(tmp_path: Path) -> None:
    helper = Model(text_reply("done"), gate=asyncio.Event())
    engine = engine_of(tmp_path, helper=helper)
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    seen = Seen()
    work = asyncio.create_task(
        engine.run("helper", "acme", "x", tools=tools_of(), inbox=inbox, on_event=seen)
    )
    await helper.asked.wait()
    [top_id] = seen.started(below=False)
    with pytest.raises(InputError) as refused:
        await engine.send(top_id, cast(Any, {"months": {"March"}}), scope="acme")
    assert str(refused.value) == (
        "engine: a message is a string, a list of parts or a value JSON can hold; "
        "Object of type set is not JSON serializable"
    )
    assert inbox.empty()
    assert helper.gate is not None
    helper.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"


@pytest.mark.parametrize("given", ["the_caller_s", "the_top_s"])
@pytest.mark.parametrize("consulted", ["helper", "desk"])
@pytest.mark.parametrize("when", ["running", "recording"])
async def test_a_press_of_the_top_stop_reaches_a_consult_in_a_delegated_run_as_in_0_1_13(
    tmp_path: Path, given: str, consulted: str, when: str
) -> None:
    running = when == "running"
    lead = Model(use("Delegate", "d1", agent="deputy", input="ask around"))
    deputy = Model(use("consult", "c1", input="find the invoice"))
    helper = Model(text_reply("found"), gate=asyncio.Event() if running else None)
    desk = Desk(open=not running)
    recorder = SlowRecord()
    engine = engine_of(tmp_path, lead, helper, desk, deputy, refinery=recorder)
    top = Stop()
    consult = engine.tool(
        consulted,
        scope="acme",
        name="consult",
        tools=tools_of(),
        stop=(lambda: top) if given == "the_top_s" else None,
    )
    work = asyncio.create_task(engine.run("lead", "acme", "x", tools=consult, stop=top))
    if running:
        recorder.gate.set()
        await (helper.asked if consulted == "helper" else desk.asked).wait()
    else:
        await recorder.started.wait()
    top.now()
    await asyncio.sleep(0.05)
    recorder.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "now"
    [asked] = run.children
    assert isinstance(asked, Generation) and asked.stopped == "now"
    assert [(s.tool, s.result, s.stopped) for s in asked.steps] == [("consult", STOPPED, True)]
    assert not recorder.cancelled and desk.cancelled is (consulted == "desk" and running)
    cut = consulted == "desk" and running
    kept = [(r.agent, getattr(r, "stopped", None)) for r in recorder.runs]
    assert kept == [
        *([] if cut else [(consulted, "now" if running else None)]),
        ("lead", "now"),
    ]
    assert engine._running == {}


async def test_stopping_a_run_by_its_id_stops_a_consult_given_the_stop_above_it(
    tmp_path: Path,
) -> None:
    lead = Model(use("Delegate", "d1", agent="deputy", input="ask around"), text_reply("went on"))
    deputy = Model(use("consult", "c1", input="find the invoice"))
    helper = Model(text_reply("never"), gate=asyncio.Event())
    kept = Kept()
    engine = engine_of(tmp_path, lead, helper, deputy=deputy, refinery=kept)
    top = Stop()
    consult = engine.tool(
        "helper", scope="acme", name="consult", tools=tools_of(), stop=lambda: top
    )
    seen = Seen()
    work = asyncio.create_task(
        engine.run("lead", "acme", "x", tools=consult, stop=top, on_event=seen)
    )
    await helper.asked.wait()
    deputy_id, consulted_id = seen.started(below=True)
    engine.stop(deputy_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped is None and run.reply == "went on"
    assert requested(top) is None
    [asked] = run.children
    assert isinstance(asked, Generation) and asked.stopped == "now"
    assert [(s.tool, s.result, s.stopped) for s in asked.steps] == [("consult", STOPPED, True)]
    assert [(r.run_id, getattr(r, "stopped", None)) for r in kept.runs] == [
        (consulted_id, "now"),
        (run.run_id, None),
    ]
    assert engine._running == {}


async def test_a_consult_inside_a_session_long_consult_stops_with_the_turn_as_in_0_1_13(
    tmp_path: Path,
) -> None:
    deputy = Model(use("consult", "c1", input="find the invoice"))
    helper = Model(text_reply("never"), gate=asyncio.Event())
    kept = Kept()
    engine = engine_of(tmp_path, helper=helper, deputy=deputy, refinery=kept)
    turn = Stop()
    inner = engine.tool("helper", scope="acme", name="consult", tools=tools_of(), stop=lambda: turn)
    outer = engine.tool("deputy", scope="acme", name="ask", tools=inner, stop=lambda: turn)
    asking = asyncio.create_task(outer.dispatch("ask", {"input": "x"}))
    await helper.asked.wait()
    turn.now()
    assert await asyncio.wait_for(asking, 5) == ""
    consulted, asked = kept.runs
    assert isinstance(consulted, Generation) and isinstance(asked, Generation)
    assert (consulted.agent, consulted.stopped) == ("helper", "now")
    assert (asked.agent, asked.stopped) == ("deputy", "now")
    assert [(s.tool, s.result, s.stopped) for s in asked.steps] == [("consult", STOPPED, True)]
    assert engine._running == {}


def gathering(engine: Engine, lookup: Any, got: list[Any]) -> Tools:
    generation = engine.tool(
        "helper", scope="acme", name="h", tools=Tools([Tool("lookup", "Look", body=lookup)])
    )
    decision = engine.tool("desk", scope="acme", name="d")

    async def both(args: dict[str, Any]) -> str:
        got.extend(
            await asyncio.gather(
                generation.dispatch("h", {"input": "find it"}),
                decision.dispatch("d", {"input": "sort it"}),
            )
        )
        return "both answered"

    return Tools([Tool("consult", "Consult both", body=both)])


async def test_a_press_from_above_reaches_a_gathered_decision_consult_as_in_0_1_13(
    tmp_path: Path,
) -> None:
    stubborn = Stubborn()
    helper = Model(use("lookup", "l1", q="x"), text_reply("never"))
    deputy = Model(use("consult", "c1", input="x"), text_reply("never"))
    desk = Desk()
    kept = Kept()
    engine = engine_of(tmp_path, helper=helper, desk=desk, deputy=deputy, refinery=kept)
    got: list[Any] = []
    stop = Stop()
    work = asyncio.create_task(
        engine.run("deputy", "acme", "x", tools=gathering(engine, stubborn, got), stop=stop)
    )
    await stubborn.started.wait()
    await desk.asked.wait()
    stop.now()
    run = await asyncio.wait_for(work, 5)
    returned_at = time.monotonic()
    assert engine._running == {}
    assert stubborn.ended_at is not None and stubborn.ended_at <= returned_at
    assert desk.cancelled_at is not None and stubborn.ended_at <= desk.cancelled_at
    assert isinstance(run, Generation) and run.stopped == "now" and desk.cancelled
    assert [(s.tool, s.result, s.stopped) for s in run.steps] == [("consult", STOPPED, True)]
    assert [r.agent for r in kept.runs] == ["helper", "deputy"] and got == []


async def test_a_gathered_decision_consult_stopped_by_its_id_is_cut_and_its_sibling_goes_on(
    tmp_path: Path,
) -> None:
    lookup = Lookup()
    helper = Model(use("lookup", "l1", q="x"), text_reply("found it"))
    deputy = Model(use("consult", "c1", input="x"), text_reply("went on"))
    desk = Desk()
    kept = Kept()
    engine = engine_of(tmp_path, helper=helper, desk=desk, deputy=deputy, refinery=kept)
    got: list[Any] = []
    work = asyncio.create_task(
        engine.run("deputy", "acme", "x", tools=gathering(engine, lookup, got))
    )
    await lookup.started.wait()
    await desk.asked.wait()
    engine.stop(decision_in_flight(engine), scope="acme")
    await until(lambda: desk.cancelled, "the decision consult's cut")
    assert sorted(held.kind for held in engine._running.values()) == ["generation", "generation"]
    lookup.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert got == ["found it", {"answers": {}}]
    assert isinstance(run, Generation) and run.stopped is None and run.reply == "went on"
    assert [(s.tool, s.result, s.stopped) for s in run.steps] == [("consult", STOPPED, True)]
    assert [r.agent for r in kept.runs] == ["helper", "deputy"] and engine._running == {}


@pytest.mark.parametrize("router", ["delegated", "top_level"])
async def test_a_routed_decision_run_stopped_by_its_id_leaves_its_router_answered(
    tmp_path: Path, router: str
) -> None:
    lead = Model(use("Delegate", "d1", agent="switch", input="sort this"), text_reply("went on"))
    desk = Desk()
    kept = Kept()
    engine = engine_of(tmp_path, lead, desk=desk, switch=Desk(open=True), refinery=kept)
    work = asyncio.create_task(
        engine.run("lead" if router == "delegated" else "switch", "acme", "x")
    )
    await desk.asked.wait()
    switch_id, desk_id = [
        run_id for (_, run_id), held in engine._running.items() if held.kind == "decision"
    ]
    engine.stop(desk_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    routed_by = run.children[0] if isinstance(run, Generation) else run
    assert isinstance(routed_by, Decision) and routed_by.run_id == switch_id
    assert routed_by.routed_to is None
    answer = routed_by.answers["intent"]
    assert isinstance(answer, ChoiceAnswer) and answer.choice == "talk"
    if isinstance(run, Generation):
        step = run.steps[0]
        assert (step.tool, step.result, step.is_error, step.stopped, step.child_run_id) == (
            "Delegate",
            STOPPED,
            True,
            True,
            switch_id,
        )
        assert run.children == [routed_by] and run.stopped is None and run.reply == "went on"
    assert [r.agent for r in kept.runs] == [run.agent]
    assert desk.cancelled and engine._running == {}


async def test_send_leaves_a_message_on_the_queue_as_it_was_given(tmp_path: Path) -> None:
    helper = Model(text_reply("done"), gate=asyncio.Event())
    engine = engine_of(tmp_path, helper=helper)
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    seen = Seen()
    work = asyncio.create_task(
        engine.run("helper", "acme", "x", tools=tools_of(), inbox=inbox, on_event=seen)
    )
    await helper.asked.wait()
    [top_id] = seen.started(below=False)
    message = {"month": "March"}
    await engine.send(top_id, cast(Any, message), scope="acme")
    assert helper.gate is not None
    helper.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    assert inbox.get_nowait() is message


async def test_a_message_that_mixes_parts_with_other_values_is_refused_where_it_is_sent(
    tmp_path: Path,
) -> None:
    helper = Model(text_reply("done"), gate=asyncio.Event())
    engine = engine_of(tmp_path, helper=helper)
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    seen = Seen()
    work = asyncio.create_task(
        engine.run("helper", "acme", "x", tools=tools_of(), inbox=inbox, on_event=seen)
    )
    await helper.asked.wait()
    [top_id] = seen.started(below=False)
    with pytest.raises(InputError) as refused:
        await engine.send(top_id, cast(Any, [Text(text="also March"), 3]), scope="acme")
    assert str(refused.value) == "engine: a message mixes parts with other values"
    assert inbox.empty()
    assert helper.gate is not None
    helper.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"


async def test_a_consult_a_voice_session_starts_has_a_stop_of_its_own(tmp_path: Path) -> None:
    helper = Model(text_reply("never"), gate=asyncio.Event())
    engine = engine_of(tmp_path, helper=helper, live=Live())
    consult = engine.tool("helper", scope="acme", name="consult", tools=tools_of())
    seen = Seen()
    heard: list[str] = []
    said = Spoken(transcript=[SpokenLine(speaker="person", text="Find it")])

    async def session(start: VoiceStart) -> Spoken:
        try:
            await start.tools.dispatch("consult", {"input": "find it"})
        except InputError as exc:
            heard.append(str(exc))
        return said

    talking = asyncio.create_task(
        engine.talk("voice", "acme", session, tools=consult, on_event=seen)
    )
    consult_id = await child_of(seen)
    await helper.asked.wait()
    assert engine._running[("acme", consult_id)].stop._parent is None
    engine.stop(consult_id, scope="acme")
    voice = await asyncio.wait_for(talking, 5)
    assert heard == [f"engine: {STOPPED}"]
    assert [(s.tool, s.result, s.stopped) for s in voice.steps] == [("consult", STOPPED, True)]
    assert engine._running == {}


async def test_a_press_of_the_stop_decision_consults_were_given_takes_the_0_1_13_path(
    tmp_path: Path,
) -> None:
    desk = Desk()
    kept = Kept()
    engine = engine_of(tmp_path, desk=desk, refinery=kept)
    turn = Stop()
    consult = engine.tool("desk", scope="acme", name="sort", stop=lambda: turn)
    both = asyncio.gather(
        consult.dispatch("sort", {"input": "a"}), consult.dispatch("sort", {"input": "b"})
    )
    await until(lambda: len(engine._running) == 2, "both consults asking")
    turn.now()
    await asyncio.sleep(0.05)
    assert not desk.cancelled and len(engine._running) == 2
    desk.gate.set()
    got = list(await asyncio.wait_for(both, 5))
    assert [sorted(cast(dict[str, Any], one)["answers"]) for one in got] == [["intent"], ["intent"]]
    assert [r.agent for r in kept.runs] == ["desk", "desk"] and engine._running == {}


async def test_engine_stop_cuts_one_of_two_decision_consults_given_the_same_stop(
    tmp_path: Path,
) -> None:
    desk = Desk()
    kept = Kept()
    engine = engine_of(tmp_path, desk=desk, refinery=kept)
    turn = Stop()
    consult = engine.tool("desk", scope="acme", name="sort", stop=lambda: turn)
    both = asyncio.gather(
        consult.dispatch("sort", {"input": "a"}), consult.dispatch("sort", {"input": "b"})
    )
    await until(lambda: len(engine._running) == 2, "both consults asking")
    first_id, second_id = [run_id for _, run_id in engine._running]
    engine.stop(first_id, scope="acme")
    await until(lambda: len(engine._running) == 1, "the first consult's cut")
    assert requested(turn) is None
    desk.gate.set()
    first, second = await asyncio.wait_for(both, 5)
    assert first == {"answers": {}}
    assert sorted(cast(dict[str, Any], second)["answers"]) == ["intent"]
    assert [r.run_id for r in kept.runs] == [second_id]


async def test_stopping_the_turn_of_a_consulted_router_takes_the_0_1_13_path(
    tmp_path: Path,
) -> None:
    helper = Model(text_reply("never"), gate=asyncio.Event())
    kept = Kept()
    engine = engine_of(tmp_path, helper=helper, desk=Desk(open=True), refinery=kept)
    turn = Stop()
    consult = engine.tool("router", scope="acme", name="ask", tools=tools_of(), stop=lambda: turn)
    asking = asyncio.create_task(consult.dispatch("ask", {"input": "talk to me"}))
    await helper.asked.wait()
    turn.now()
    got = cast(dict[str, Any], await asyncio.wait_for(asking, 5))
    assert sorted(got["answers"]) == ["intent"]
    assert got["routed_to"] == {"agent": "helper", "reply": ""}
    [router] = kept.runs
    assert isinstance(router, Decision) and isinstance(router.routed_to, Generation)
    assert router.routed_to.stopped == "now" and engine._running == {}


async def test_a_press_while_a_decision_consult_is_recorded_cancels_nothing(
    tmp_path: Path,
) -> None:
    desk = Desk()
    recorder = SlowRecord()
    engine = engine_of(tmp_path, desk=desk, refinery=recorder)
    turn = Stop()
    consult = engine.tool("desk", scope="acme", name="sort", stop=lambda: turn)
    asking = asyncio.create_task(consult.dispatch("sort", {"input": "a"}))
    await desk.asked.wait()
    desk.gate.set()
    await recorder.started.wait()
    [writing] = recorder.writing
    assert engine._running == {}
    turn.now()
    with pytest.raises(NotFoundError):
        engine.stop(writing.run_id, scope="acme")
    await asyncio.sleep(0.05)
    recorder.gate.set()
    got = cast(dict[str, Any], await asyncio.wait_for(asking, 5))
    assert sorted(got["answers"]) == ["intent"]
    assert recorder.runs == [writing] and not recorder.cancelled


@pytest.mark.parametrize("router", ["delegated", "top_level"])
async def test_a_router_stopped_by_its_id_after_it_routed_stops_its_routed_generation_run(
    tmp_path: Path, router: str
) -> None:
    lead = Model(use("Delegate", "d1", agent="router", input="talk"), text_reply("went on"))
    helper = Model(text_reply("never"), gate=asyncio.Event())
    kept = Kept()
    engine = engine_of(tmp_path, lead, helper, desk=Desk(open=True), refinery=kept)
    seen = Seen()
    work = asyncio.create_task(
        engine.run(
            "lead" if router == "delegated" else "router",
            "acme",
            "x",
            tools=tools_of(),
            on_event=seen,
        )
    )
    await helper.asked.wait()
    router_id = decision_in_flight(engine)
    engine.stop(router_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    routed_by = run.children[0] if isinstance(run, Generation) else run
    assert isinstance(routed_by, Decision) and routed_by.run_id == router_id
    routed = routed_by.routed_to
    assert isinstance(routed, Generation) and routed.stopped == "now"
    if isinstance(run, Generation):
        step = run.steps[0]
        assert (step.tool, step.result, step.stopped, step.child_run_id) == (
            "Delegate",
            STOPPED,
            True,
            router_id,
        )
        assert run.children == [routed_by] and run.stopped is None and run.reply == "went on"
    assert [r.agent for r in kept.runs] == [run.agent] and engine._running == {}


async def test_a_router_stopped_by_its_id_after_it_routed_has_its_route_cut_its_decision_run(
    tmp_path: Path,
) -> None:
    lead = Model(use("Delegate", "d1", agent="switch", input="sort this"), text_reply("went on"))
    desk = Desk()
    engine = engine_of(tmp_path, lead, desk=desk, switch=Desk(open=True))
    work = asyncio.create_task(engine.run("lead", "acme", "x"))
    await desk.asked.wait()
    switch_id, _ = [
        run_id for (_, run_id), held in engine._running.items() if held.kind == "decision"
    ]
    engine.stop(switch_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and desk.cancelled
    [switch] = run.children
    assert isinstance(switch, Decision) and switch.run_id == switch_id
    assert switch.routed_to is None and sorted(switch.answers) == ["intent"]
    step = run.steps[0]
    assert (step.result, step.stopped, step.child_run_id) == (STOPPED, True, switch_id)
    assert run.reply == "went on" and engine._running == {}


@pytest.mark.parametrize("agent", ["desk", "router"])
async def test_after_step_by_id_lets_a_decision_run_answer_and_rides_to_its_route(
    tmp_path: Path, agent: str
) -> None:
    lead = Model(use("Delegate", "d1", agent=agent, input="talk"), text_reply("went on"))
    helper = Model(text_reply("never"))
    desk = Desk()
    engine = engine_of(tmp_path, lead, helper, desk=desk)
    work = asyncio.create_task(engine.run("lead", "acme", "x", tools=tools_of()))
    await desk.asked.wait()
    engine.stop(decision_in_flight(engine), scope="acme", after_step=True)
    await asyncio.sleep(0.05)
    assert not desk.cancelled
    desk.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "went on"
    [child] = run.children
    assert isinstance(child, Decision) and sorted(child.answers) == ["intent"]
    step = run.steps[0]
    if agent == "desk":
        assert child.routed_to is None and (step.stopped, step.child_run_id) == (
            False,
            child.run_id,
        )
        assert json.loads(cast(str, step.result))["answers"]["intent"]["choice"] == "talk"
    else:
        routed = child.routed_to
        assert isinstance(routed, Generation) and routed.stopped == "after_step"
        assert routed.calls == 0 and helper.calls == []
        assert (step.result, step.stopped, step.child_run_id) == (STOPPED, True, child.run_id)
