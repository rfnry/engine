import asyncio
import json
import re
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine import Later, Stop
from rfnry_engine.conversation import live
from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import Message, Pause, Refused, Reply, ToolCall, ToolSpec
from rfnry_engine.records import (
    Decision,
    Generation,
    Run,
    RunEnded,
    ToolCallStarted,
    ToolResultReady,
)
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import (
    Crew,
    Fixed,
    MemoryStore,
    Seen,
    background,
    crew,
    delegating,
    team,
    text_reply,
    until,
)
from .test_approve import INTENT, Refunds
from .test_paused import (
    HELPS,
    LOOKUP,
    NO_WAITING,
    REFUND,
    Blocking,
    calling,
    ended_below,
    found,
    later_for,
    no_mark_in,
)
from .test_voice import Recording
from .trees import amend, provided, write_folder

KEY = ("acme", "c1")
OWN = ToolCall(id="c-own", name="refund", args={"amount": 3})
SECOND = ToolCall(id="c-refund-2", name="refund", args={"amount": 5})


class Reported(BaseModel):
    text: str


def tree(tmp_path: Path, background: bool = False) -> Path:
    root = tmp_path / "agents"
    root.mkdir(parents=True)
    return write_folder(
        root,
        {
            "lead": {
                "kind": "generation",
                "model": "crew",
                "description": "Leads the work",
                "instructions": "lead.md",
                "tools": {"allow": ["refund"]},
                "orchestration": {
                    "delegate_to": ["mid", "desk", "front"],
                    **({"background": True} if background else {}),
                },
            },
            "mid": {
                "kind": "generation",
                "model": "crew",
                "description": "Passes work on",
                "instructions": "mid.md",
                "orchestration": {"delegate_to": ["desk"]},
            },
            "desk": {
                "kind": "generation",
                "model": "crew",
                "description": "Refunds orders",
                "instructions": "desk.md",
                "tools": {"allow": ["refund"]},
            },
            "front": {
                "kind": "decision",
                "model": "judge",
                "description": "Sorts requests",
                "instructions": "front.md",
                "questions": ["front/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"help": "desk"}}},
            },
        },
        runtime_tools=["refund"],
        files={
            "lead.md": "Lead.\n",
            "mid.md": "Pass it on.\n",
            "desk.md": "Refund.\n",
            "front.md": "",
            "front/intent.json": INTENT,
        },
    )


def mark_of(messages: list[Message]) -> Pause:
    pause = messages[-1].pause
    assert pause is not None
    return pause


def engine_of(
    tmp_path: Path, model: Crew, judge: Any = None, background: bool = False, **engine: Any
) -> Engine:
    return Engine(
        tree(tmp_path, background), providers=provided({"crew": model, "judge": judge}), **engine
    )


async def lead_run(
    engine: Engine, store: MemoryStore, refunds: Refunds, input: str = "go", **options: Any
) -> Generation:
    run = await engine.run(
        "lead",
        "acme",
        input,
        store=store,
        conversation="c1",
        tools=refunds.tools(),
        **options,
    )
    assert isinstance(run, Generation)
    return run


def two_levels(*desk: Reply) -> Crew:
    return Crew(
        [delegating("mid", "pass it on"), text_reply("all done")],
        {
            "pass it on": [delegating("desk", "refund it"), text_reply("desk refunded")],
            "refund it": [calling(REFUND), *desk],
        },
    )


def both(child: str = "desk") -> Reply:
    calls = [
        ToolCall(id="d1", name="Delegate", args={"agent": child, "input": "refund it"}),
        OWN.model_copy(),
    ]
    return Reply(tool_calls=calls, stop_reason="tool_use")


async def test_a_paused_child_pauses_its_parent(tmp_path: Path) -> None:
    store, refunds, seen = MemoryStore(), Refunds(), Seen()
    model = Crew(
        [delegating("desk", "refund it"), text_reply("never")],
        {"refund it": [calling(REFUND), text_reply("never")]},
    )
    lead = await lead_run(
        engine_of(tmp_path, model), store, refunds, approve=later_for("refund"), on_event=seen
    )
    [desk] = lead.children
    assert isinstance(desk, Generation)
    desk_run = desk.run_id
    [w] = lead.paused
    assert (w.agent, w.conversation) == ("desk", f"c1/{desk_run}")
    assert (w.run_id, w.call) == (desk_run, REFUND) and desk.paused == [w]
    step = lead.steps[0]
    assert step.waiting and step.child_run_id == desk_run
    assert (step.tool, step.result, step.is_error, step.approved) == ("Delegate", "", False, True)
    mark = mark_of(store.held[KEY])
    assert mark.delegated == {"d1": f"c1/{desk_run}"}
    assert (mark.waiting, mark.below, mark.run_id) == ([], [w], lead.run_id)
    assert live(store.held[("acme", f"c1/{desk_run}")]) is not None
    assert len(model.calls["lead"]) == 1 and refunds.done == []
    assert lead.stopped is None and lead.error is None
    [ended] = [e for e in seen.events if isinstance(e, RunEnded) and e.call is None]
    assert ended.paused == [w]
    assert not [e for e in seen.events if isinstance(e, ToolResultReady) and e.call_id == "d1"]


async def test_a_resume_walks_down_two_levels(tmp_path: Path) -> None:
    store, refunds, seen = MemoryStore(), Refunds(), Seen()
    model = two_levels(text_reply("refunded"))
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [w] = first.paused
    [mid_before] = first.children
    assert isinstance(mid_before, Generation)
    [desk_before] = mid_before.children
    assert w.conversation == f"c1/{mid_before.run_id}/{desk_before.run_id}"
    first_mark, mid_mark = mark_of(first.conversation), mark_of(mid_before.conversation)
    assert first_mark.delegated == {"d1": f"c1/{mid_before.run_id}"}
    assert (mid_mark.delegated, mid_mark.below) == ({"d1": w.conversation}, [w])
    keys_before = {k for (_, k) in store.held}
    again = await engine.run(
        "lead",
        "acme",
        "go on",
        store=store,
        conversation="c1",
        tools=refunds.tools(),
        answers={w.id: True},
        on_event=seen,
    )
    assert isinstance(again, Generation)
    assert refunds.done == [{"amount": 10}]
    mid = again.children[0]
    assert isinstance(mid, Generation)
    desk = mid.children[0]
    assert isinstance(desk, Generation)
    assert mid.resumed and desk.resumed and desk.input == ""
    assert mid.input == "" and again.resumed and again.input == "go on"
    assert {k for (_, k) in store.held} == keys_before
    assert store.held[KEY][-1].role == "assistant"
    assert again.reply == "all done" and again.paused == [] and mid.paused == []
    [step] = again.steps
    assert (step.tool, step.result, step.child_run_id, step.approved) == (
        "Delegate",
        "desk refunded",
        mid.run_id,
        True,
    )
    assert [(s.tool, s.result) for s in mid.steps] == [("Delegate", "refunded")]
    desk_sent = model.calls["refund it"][-1]
    assert [(m.role, m.content) for m in desk_sent[-2:]] == [
        ("assistant", ""),
        ("tool", "refunded"),
    ]
    lead_sent = model.calls["lead"][-1]
    assert [m.role for m in lead_sent[-3:]] == ["assistant", "tool", "user"]
    assert lead_sent[-1].content == "go on" and no_mark_in(lead_sent)
    told = [
        (type(e).__name__, e.call_id)
        for e in seen.events
        if isinstance(e, ToolCallStarted | ToolResultReady) and e.run_id == again.run_id
    ]
    assert told == [("ToolCallStarted", "d1"), ("ToolResultReady", "d1")]


async def test_a_child_that_waits_again_pauses_its_parent_again_and_holds_the_input(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    model = two_levels(calling(SECOND), text_reply("refunded both"))
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [w] = first.paused
    again = await lead_run(
        engine, store, refunds, "go on", answers={w.id: True}, approve=later_for("refund")
    )
    assert again.paused and again.paused[0].id != w.id
    assert again.paused[0].call == SECOND and again.resumed and len(model.calls["lead"]) == 1
    mark = mark_of(store.held[KEY])
    assert mark.run_id == again.run_id
    assert mark.delegated == mark_of(first.conversation).delegated
    assert mark_of(store.held[KEY]).held[-1].content == "go on"
    assert [s.waiting for s in again.steps] == [True]
    third = await lead_run(engine, store, refunds, "and now", answers={again.paused[0].id: True})
    assert third.reply == "all done" and third.paused == []
    assert refunds.done == [{"amount": 10}, {"amount": 5}]
    sent = model.calls["lead"][-1]
    assert [m.content for m in sent if m.role == "user"][-2:] == ["go on", "and now"]


@pytest.mark.parametrize("given", [{}, None], ids=["empty", "none"])
async def test_a_resume_must_answer_the_calls_under_the_tops_own_delegate(
    tmp_path: Path, given: dict[str, Any] | None
) -> None:
    store, refunds = MemoryStore(), Refunds()
    model = two_levels(text_reply("never"))
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [w] = first.paused
    held = {key: list(messages) for key, messages in store.held.items()}
    match = re.escape(f"engine: refund of agent 'desk' waits for an answer under id {w.id!r}; ")
    with pytest.raises(InputError, match=match):
        await lead_run(engine, store, refunds, "go on", answers=given)
    assert store.held == held and len(model.calls["lead"]) == 1 and refunds.done == []


async def test_a_resumed_runs_added_starts_after_the_mark_another_run_wrote(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    model = Crew(
        [both(), text_reply("one refused, one done")],
        {"refund it": [calling(REFUND), text_reply("refunded")]},
    )
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    own, below = first.paused
    assert (own.agent, own.call.id, below.agent) == ("lead", "c-own", "desk")
    before = len(store.held[KEY])
    again = await lead_run(
        engine, store, refunds, "go on", answers={own.id: Refused("not yet"), below.id: True}
    )
    assert again.error is None and again.reply == "one refused, one done"
    assert again.added == store.held[KEY][before:]
    refused, fresh, answered = again.added[:3]
    assert (refused.tool_call_id, refused.content) == (
        "c-own",
        "NotApproved: refund was not approved: not yet",
    )
    assert fresh.pause is not None and fresh.pause.run_id == again.run_id
    assert (fresh.pause.waiting, fresh.pause.delegated, fresh.pause.below) == (
        [],
        mark_of(first.conversation).delegated,
        [],
    )
    assert (answered.tool_call_id, answered.content) == ("d1", "refunded")
    assert refunds.done == [{"amount": 10}]


async def test_a_resume_that_ends_a_cut_call_keeps_the_pause_after_the_ended_result(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Blocking()
    model = Crew(
        [both(), text_reply("it may have gone through")],
        {"refund it": [calling(REFUND), text_reply("it was cut off")]},
    )
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    own, below = first.paused
    answers = {own.id: True, below.id: True}
    cut = asyncio.create_task(lead_run(engine, store, refunds, "go on", answers=answers))
    await until(lambda: refunds.started == 2, "both approved refunds started")
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    cut_mark = mark_of(store.held[KEY])
    assert live(store.held[KEY]) == cut_mark
    assert (cut_mark.waiting, cut_mark.below, cut_mark.delegated) == (
        [],
        [],
        mark_of(first.conversation).delegated,
    )
    before, appends = len(store.held[KEY]), len(store.appended)
    with pytest.raises(InputError, match=re.escape(NO_WAITING.format(id=below.id))):
        await lead_run(engine, store, refunds, "go on", answers={below.id: True})
    third = await lead_run(engine, store, refunds, "go on", answers={})
    assert third.error is None and third.resumed and third.reply == "it may have gone through"
    held = store.held[KEY]
    ended, mark = held[before], held[before + 1]
    assert ended.tool_call_id == "c-own" and str(ended.content).startswith("Ended: refund")
    assert mark.pause is not None and mark.pause.run_id == third.run_id
    assert (mark.pause.delegated, mark.pause.below) == (cut_mark.delegated, [])
    assert live(held[: before + 2]) == mark.pause
    assert store.appended[appends] == ("c1", before, 2)
    assert held[before + 2].tool_call_id == "d1" and refunds.started == 2
    desk_sent = model.calls["refund it"][-1]
    assert [m.role for m in desk_sent] == ["user", "assistant", "tool"]
    assert str(desk_sent[-1].content).startswith("Ended: refund did not return")


async def test_an_answer_sent_again_after_a_cut_resume_is_refused_and_the_child_goes_on(
    tmp_path: Path,
) -> None:
    store, refunds, recording = MemoryStore(), Blocking(), Recording()
    model = Crew(
        [delegating("desk", "refund it"), text_reply("it may have gone through")],
        {"refund it": [calling(REFUND), text_reply("it was cut off")]},
    )
    engine = engine_of(tmp_path, model, refinery=recording)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [w] = first.paused
    [desk] = first.children
    desk_key = ("acme", f"c1/{desk.run_id}")
    cut = asyncio.create_task(lead_run(engine, store, refunds, "go on", answers={w.id: True}))
    await until(lambda: refunds.started == 1, "the approved refund started")
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    cut_mark = mark_of(store.held[KEY])
    assert live(store.held[KEY]) == cut_mark and cut_mark.below == []
    assert cut_mark.delegated == {"d1": desk_key[1]}
    assert live(store.held[desk_key]) is None
    held = {key: list(messages) for key, messages in store.held.items()}
    runs = len(recording.runs)
    with pytest.raises(InputError, match=re.escape(NO_WAITING.format(id=w.id))):
        await lead_run(engine, store, refunds, "go on", answers={w.id: True})
    assert store.held == held and len(recording.runs) == runs and refunds.started == 1
    third = await lead_run(engine, store, refunds, "go on", answers={})
    assert third.error is None and third.resumed and third.reply == "it may have gone through"
    [again] = third.children
    assert isinstance(again, Generation) and not again.resumed and again.input == ""
    ended = [m for m in store.held[desk_key] if str(m.content).startswith("Ended:")]
    assert [m.content for m in ended] == [
        "Ended: refund did not return; the run that made the call ended, and what it did is unknown"
    ]
    assert third.steps[0].result == "it was cut off" and store.held[KEY][-1].role == "assistant"
    assert refunds.started == 1 and len(model.calls["lead"]) == 2


async def test_a_paused_router_below_resumes_the_agent_its_mark_names(tmp_path: Path) -> None:
    store, refunds, judge = MemoryStore(), Refunds(), Fixed(HELPS)
    model = Crew(
        [delegating("front", "refund it"), text_reply("done")],
        {"refund it": [calling(REFUND), text_reply("refunded")]},
    )
    engine = engine_of(tmp_path, model, judge)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [front] = first.children
    assert isinstance(front, Decision) and isinstance(front.routed_to, Generation)
    [w] = first.paused
    assert front.paused == [w] and w.conversation == f"c1/{front.run_id}"
    again = await lead_run(engine, store, refunds, "go on", answers={w.id: True})
    assert again.reply == "done" and len(judge.calls) == 1
    [desk] = again.children
    assert isinstance(desk, Generation) and desk.agent == "desk" and desk.resumed
    assert again.steps[0].result == "refunded" and refunds.done == [{"amount": 10}]


async def test_output_is_refused_while_a_delegate_of_its_run_waits(tmp_path: Path) -> None:
    store, refunds = MemoryStore(), Refunds()
    output = ToolCall(id="c-out", name="Output", args={"text": "done"})
    lead = Reply(
        tool_calls=[*delegating("desk", "refund it").tool_calls, output], stop_reason="tool_use"
    )
    model = Crew([lead, text_reply("never")], {"refund it": [calling(REFUND), text_reply("never")]})
    run = await lead_run(
        engine_of(tmp_path, model), store, refunds, approve=later_for("refund"), expect=Reported
    )
    assert run.paused and [s.tool for s in run.steps] == ["Delegate", "Output"]
    assert run.steps[1].result == (
        "engine: waiting for an answer: Delegate; call Output again once they have one"
    )


class Stalling(Recording):
    def __init__(self) -> None:
        super().__init__()
        self.stalled = asyncio.Event()

    async def record(self, run: Run, *, skip: Collection[str] = ()) -> None:
        if run.agent == "desk" and isinstance(run, Generation) and run.resumed:
            self.stalled.set()
            await asyncio.Event().wait()
        await super().record(run, skip=skip)


async def test_a_child_that_replied_before_a_crash_answers_with_its_stored_reply(
    tmp_path: Path,
) -> None:
    store, refunds, recording = MemoryStore(), Refunds(), Stalling()
    model = Crew(
        [delegating("desk", "refund it"), text_reply("done")],
        {"refund it": [calling(REFUND), text_reply("refunded")]},
    )
    engine = engine_of(tmp_path, model, refinery=recording)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [w] = first.paused
    [desk] = first.children
    desk_key = ("acme", f"c1/{desk.run_id}")
    cut = asyncio.create_task(lead_run(engine, store, refunds, "go on", answers={w.id: True}))
    await asyncio.wait_for(recording.stalled.wait(), 5)
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    assert (store.held[desk_key][-1].role, store.held[desk_key][-1].content) == (
        "assistant",
        "refunded",
    )
    calls = len(model.calls["refund it"])
    third = await lead_run(engine, store, refunds, "go on", answers={})
    assert third.error is None and third.reply == "done"
    assert len(model.calls["refund it"]) == calls
    assert third.steps[0].result == "refunded" and refunds.done == [{"amount": 10}]
    [again] = third.children
    assert isinstance(again, Generation)
    assert (again.calls, again.reply, again.messages, again.added) == (0, "refunded", [], [])
    assert store.held[desk_key][-1].content == "refunded"


async def test_a_router_below_goes_on_as_the_agent_its_inert_mark_names(tmp_path: Path) -> None:
    store, refunds, judge = MemoryStore(), Blocking(), Fixed(HELPS)
    model = Crew(
        [delegating("front", "refund it"), text_reply("done")],
        {"refund it": [calling(REFUND), text_reply("it was cut off")]},
    )
    engine = engine_of(tmp_path, model, judge)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [w] = first.paused
    [front] = first.children
    cut = asyncio.create_task(lead_run(engine, store, refunds, "go on", answers={w.id: True}))
    await until(lambda: refunds.started == 1, "the approved refund started")
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    assert live(store.held[("acme", f"c1/{front.run_id}")]) is None
    third = await lead_run(engine, store, refunds, "go on", answers={})
    assert third.error is None and third.reply == "done" and len(judge.calls) == 1
    [desk] = third.children
    assert isinstance(desk, Generation) and desk.agent == "desk"
    assert third.steps[0].result == "it was cut off" and refunds.started == 1


async def test_a_held_outcome_is_told_once_across_two_resumes(tmp_path: Path) -> None:
    store, refunds = MemoryStore(), Refunds()
    calls = [
        ToolCall(
            id="b1",
            name="Delegate",
            args={"agent": "mid", "input": "slow part", "background": True},
        ),
        *delegating("desk", "refund it", call_id="s1").tool_calls,
    ]
    model = Crew(
        [Reply(tool_calls=calls, stop_reason="tool_use"), text_reply("all done")],
        {
            "slow part": [text_reply("slow done")],
            "refund it": [calling(REFUND), calling(SECOND), text_reply("refunded both")],
        },
    )
    engine = engine_of(tmp_path, model, background=True)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [w] = first.paused
    assert [
        (m.notice.job, m.notice.kind) for m in mark_of(first.conversation).held if m.notice
    ] == [("d1", "finished")]
    again = await lead_run(
        engine, store, refunds, "go on", answers={w.id: True}, approve=later_for("refund")
    )
    [w2] = again.paused
    held = mark_of(store.held[KEY]).held
    assert [(m.notice.job, m.notice.kind) for m in held if m.notice] == [("d1", "finished")]
    assert held[-1].content == "go on"
    third = await lead_run(engine, store, refunds, "and now", answers={w2.id: True})
    assert third.error is None and third.reply == "all done"
    sent = model.calls["lead"][-1]
    assert not [m for m in sent if "· lost]" in str(m.content)]
    told = [m for m in sent if str(m.content).startswith("[helper d1 · mid · finished]")]
    assert len(told) == 1
    stored = [m for m in store.held[KEY] if m.notice is not None]
    assert [(m.notice.job, m.notice.kind) for m in stored if m.notice] == [("d1", "finished")]
    assert [m.content for m in sent if m.role == "user"][-2:] == ["go on", "and now"]
    assert refunds.done == [{"amount": 10}, {"amount": 5}]


HELPER_KEY = ("acme", "c1/d1")
ASK = ToolCall(id="a1", name="Ask", args={"question": "which one?"})
JOB = ToolCall(id="c-job", name="Job", args={"id": "d1"})
JOBS = ToolCall(id="c-jobs", name="Jobs", args={})
PAUSED = "engine: waiting for an answer: {names}; call Output again once they have one"
E11 = (
    "InputError: tools/helpers: helper d1 is paused until its calls are answered; "
    "send reaches only a helper running or waiting on Ask"
)


def helped(tmp_path: Path, model: Crew, **engine: Any) -> Engine:
    folder = crew(tmp_path / "crew", worker={"tools": {"allow": ["lookup"]}}, background=True)
    amend(folder, "lead", tools={"allow": ["refund"]})
    return team(folder, model, **engine)


async def crew_run(
    engine: Engine, store: MemoryStore, approve: Any = None, **options: Any
) -> Generation:
    run = await engine.run(
        "lead",
        "acme",
        "go",
        store=store,
        conversation="c1",
        tools=Tools([Tool("lookup", "Look an order up", body=found), Refunds().tools()]),
        approve=later_for("lookup") if approve is None else approve,
        **options,
    )
    assert isinstance(run, Generation)
    return run


def beside_its_own(*inputs: str) -> Reply:
    return Reply(
        tool_calls=[*background(*inputs).tool_calls, OWN.model_copy()], stop_reason="tool_use"
    )


def result_of(run: Generation, call_id: str) -> Any:
    return json.loads(str(next(m.content for m in run.conversation if m.tool_call_id == call_id)))


def batch_stored(store: MemoryStore) -> bool:
    return any(m.tool_call_id == "b1" for m in store.held.get(KEY, []))


def kinds(messages: list[Message]) -> list[str]:
    return [m.notice.kind for m in messages if m.notice is not None]


async def test_a_paused_helper_ends_paused_and_its_lead_hears_it(tmp_path: Path) -> None:
    store, ended = MemoryStore(), asyncio.Event()
    model = Crew(
        [background("part one"), ended, calling(JOB, JOBS), text_reply("it waits")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    lead = await crew_run(helped(tmp_path, model), store, on_event=ended_below(ended))
    [worker] = lead.children
    assert isinstance(worker, Generation) and worker.stopped is None and worker.error is None
    assert [(w.agent, w.call, w.conversation) for w in worker.paused] == [
        ("worker", LOOKUP, "c1/d1")
    ]
    notice = next(m for m in lead.conversation if m.notice and m.notice.kind == "paused")
    assert notice.content == "[helper d1 · worker · paused]\nwaits for an answer to: lookup"
    assert notice.notice is not None and notice.notice.run_id == worker.run_id
    jobs_result, job_result = result_of(lead, "c-jobs"), result_of(lead, "c-job")
    assert jobs_result["jobs"][0]["ended"] == "paused"
    assert jobs_result["jobs"][0]["running"] is False
    assert job_result == {
        "running": False,
        "agent": "worker",
        "input": "part one",
        "asking": None,
        "paused": "waits for an answer to: lookup",
    }
    assert lead.paused == worker.paused and len(model.calls["lead"]) == 3


async def test_a_lead_with_only_paused_helpers_pauses(tmp_path: Path) -> None:
    store = MemoryStore()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("it waits")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    lead = await crew_run(helped(tmp_path, model), store)
    [worker] = lead.children
    assert isinstance(worker, Generation)
    assert lead.paused == worker.paused and lead.stopped is None and lead.error is None
    mark = mark_of(store.held[KEY])
    assert mark.below == worker.paused and (mark.waiting, mark.delegated) == ([], {})
    assert lead.reply == "it waits" and len(model.calls["lead"]) == 3
    assert live(store.held[HELPER_KEY]) is not None


async def test_a_lead_keeps_working_beside_a_paused_helper(tmp_path: Path) -> None:
    store, both_ended = MemoryStore(), asyncio.Event()
    model = Crew(
        [
            background("part one", "part two"),
            both_ended,
            text_reply("both heard"),
            text_reply("d1 waits"),
        ],
        {
            "part one": [calling(LOOKUP), text_reply("never")],
            "part two": [text_reply("part two done")],
        },
    )
    ends = 0

    async def watch(event: Any) -> None:
        nonlocal ends
        if isinstance(event, RunEnded) and event.call is not None:
            ends += 1
            if ends == 2:
                both_ended.set()

    lead = await crew_run(helped(tmp_path, model), store, on_event=watch)
    assert len(model.calls["lead"]) == 3 and lead.reply == "d1 waits"
    heard = {str(m.content).split("\n")[0] for m in model.calls["lead"][-1] if m.role == "user"}
    assert {"[helper d1 · worker · paused]", "[helper d2 · worker · finished]"} <= heard
    by_input = {str(c.input): c for c in lead.children if isinstance(c, Generation)}
    assert lead.paused == by_input["part one"].paused and by_input["part two"].paused == []
    assert mark_of(store.held[KEY]).held == []


async def test_output_is_refused_while_a_helper_is_paused(tmp_path: Path) -> None:
    store, ended = MemoryStore(), asyncio.Event()
    output = ToolCall(id="c-out", name="Output", args={"text": "done"})
    model = Crew(
        [background("part one"), ended, calling(output), text_reply("never")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    lead = await crew_run(
        helped(tmp_path, model), store, expect=Reported, on_event=ended_below(ended)
    )
    out = next(s for s in lead.steps if s.tool == "Output")
    assert out.result == PAUSED.format(names="d1") and out.is_error
    assert lead.paused == lead.children[0].paused and len(model.calls["lead"]) == 2
    assert mark_of(store.held[KEY]).below == lead.paused


async def test_a_lead_whose_call_waits_waits_for_its_running_helpers_first(tmp_path: Path) -> None:
    store, gate = MemoryStore(), asyncio.Event()
    model = Crew(
        [beside_its_own("part one"), text_reply("never")],
        {"part one": [gate, text_reply("part done")]},
    )
    running = asyncio.create_task(
        crew_run(helped(tmp_path, model), store, approve=later_for("refund"))
    )
    await until(lambda: batch_stored(store), "the lead's batch was stored")
    assert not running.done()
    gate.set()
    lead = await asyncio.wait_for(running, 5)
    assert len(model.calls["lead"]) == 1
    assert [w.call for w in lead.paused] == [OWN]
    mark = mark_of(store.held[KEY])
    assert kinds(mark.held) == ["finished"]
    assert mark.held[0].content == "[helper d1 · worker · finished]\npart done"
    assert not any(m.notice and m.notice.kind == "finished" for m in store.held[KEY])
    [worker] = lead.children
    assert isinstance(worker, Generation) and worker.reply == "part done"


async def test_a_helper_that_asks_a_lead_waiting_to_pause_pauses_on_its_ask(
    tmp_path: Path,
) -> None:
    store, gate = MemoryStore(), asyncio.Event()
    model = Crew(
        [beside_its_own("part one"), text_reply("never")],
        {"part one": [gate, calling(ASK), text_reply("never")]},
    )
    running = asyncio.create_task(
        crew_run(helped(tmp_path, model), store, approve=later_for("refund"))
    )
    await until(lambda: batch_stored(store), "the lead's batch was stored")
    gate.set()
    lead = await asyncio.wait_for(running, 5)
    d1 = mark_of(store.held[HELPER_KEY])
    assert d1.asking == "a1" and d1.waiting == [] and d1.below == []
    held = mark_of(store.held[KEY]).held
    assert kinds(held) == ["asks", "paused"]
    assert [m.content for m in held] == [
        "[helper d1 · worker · asks]\nwhich one?",
        "[helper d1 · worker · paused]\nwaits for your answer to its question",
    ]
    assert all(w.call.name != "Ask" for w in lead.paused)
    assert [w.call for w in lead.paused] == [OWN]
    [worker] = lead.children
    assert isinstance(worker, Generation) and worker.paused == []
    [step] = worker.steps
    assert (step.tool, step.result, step.waiting) == ("Ask", "", True)


async def test_a_question_waiting_when_its_lead_pauses_pauses_its_helper(tmp_path: Path) -> None:
    store = MemoryStore()
    model = Crew(
        [background("part one"), text_reply("waiting"), calling(OWN), text_reply("never")],
        {"part one": [calling(ASK), text_reply("never")]},
    )
    lead = await crew_run(helped(tmp_path, model), store, approve=later_for("refund"))
    assert len(model.calls["lead"]) == 3
    assert kinds(lead.conversation[:-1]) == ["asks"]
    assert kinds(mark_of(store.held[KEY]).held) == ["paused"]
    assert mark_of(store.held[HELPER_KEY]).asking == "a1"
    assert [w.call for w in lead.paused] == [OWN]


async def test_a_helper_that_asks_twice_while_its_lead_waits_to_pause_waits_on_the_first(
    tmp_path: Path,
) -> None:
    store, gate = MemoryStore(), asyncio.Event()
    second = ToolCall(id="a2", name="Ask", args={"question": "and which day?"})
    model = Crew(
        [beside_its_own("part one"), text_reply("never")],
        {"part one": [gate, calling(ASK, second), text_reply("never")]},
    )
    running = asyncio.create_task(
        crew_run(helped(tmp_path, model), store, approve=later_for("refund"))
    )
    await until(lambda: batch_stored(store), "the lead's batch was stored")
    gate.set()
    await asyncio.wait_for(running, 5)
    helper = store.held[HELPER_KEY]
    assert mark_of(helper).asking == "a1"
    [answer] = [m for m in helper if m.tool_call_id == "a2"]
    assert answer.is_error and answer.content == (
        "InputError: tools/helpers: your lead ended its turn without answering"
    )
    held = mark_of(store.held[KEY]).held
    assert [m.content for m in held if m.notice and m.notice.kind == "asks"] == [
        "[helper d1 · worker · asks]\nwhich one?"
    ]


async def test_a_lead_may_stop_a_paused_helper(tmp_path: Path) -> None:
    store, ended = MemoryStore(), asyncio.Event()
    stopping = ToolCall(id="c-job", name="Job", args={"id": "d1", "stop": True})
    model = Crew(
        [background("part one"), ended, calling(stopping), text_reply("stopped it")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    lead = await crew_run(helped(tmp_path, model), store, on_event=ended_below(ended))
    held = store.held[HELPER_KEY]
    assert any(m.content == "stopped by the user" and m.tool_call_id == "c-lookup" for m in held)
    assert live(held) is None and len(model.calls["part one"]) == 1
    job_result = result_of(lead, "c-job")
    assert job_result["stopped"] is not None and job_result["running"] is False
    assert lead.reply == "stopped it" and lead.paused == [] and live(store.held[KEY]) is None
    paused, stopped = lead.children
    assert isinstance(stopped, Generation) and isinstance(paused, Generation)
    assert stopped.stopped == "now" and stopped.resumed and stopped.input == ""
    assert stopped.run_id != paused.run_id
    assert [(s.tool, s.result, s.stopped) for s in stopped.steps] == [
        ("lookup", "stopped by the user", True)
    ]
    assert kinds(lead.conversation) == ["paused", "stopped"]


async def test_send_to_a_helper_paused_on_its_calls_is_refused(tmp_path: Path) -> None:
    store, ended = MemoryStore(), asyncio.Event()
    sending = ToolCall(id="c-job", name="Job", args={"id": "d1", "send": "hurry"})
    model = Crew(
        [background("part one"), ended, calling(sending), text_reply("it waits")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    lead = await crew_run(helped(tmp_path, model), store, on_event=ended_below(ended))
    [message] = [m for m in lead.conversation if m.tool_call_id == "c-job"]
    assert message.is_error and message.content == E11
    assert lead.paused == lead.children[0].paused


class Breaks(Crew):
    def __init__(self, at: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.at = at

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        if self.marker in system and len(self.calls["lead"]) == self.at:
            self.calls["lead"].append(list(messages))
            raise RuntimeError("the model broke")
        return await super().generate(system, messages, tools)


async def test_a_lead_that_fails_stops_its_paused_helpers(tmp_path: Path) -> None:
    store, ended, recording = MemoryStore(), asyncio.Event(), Recording()
    model = Breaks(
        2,
        [background("part one"), ended, calling(JOBS)],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    with pytest.raises(RuntimeError, match="the model broke"):
        await crew_run(
            helped(tmp_path, model, refinery=recording), store, on_event=ended_below(ended)
        )
    held = store.held[HELPER_KEY]
    assert held[-1].content == "stopped by the user" and held[-1].tool_call_id == "c-lookup"
    assert live(held) is None
    lead = recording.runs[-1]
    assert isinstance(lead, Generation) and lead.error == "RuntimeError: the model broke"
    paused, stopped = lead.children
    assert isinstance(paused, Generation) and isinstance(stopped, Generation)
    assert (paused.stopped, stopped.stopped) == (None, "now")
    assert stopped.call is not None and stopped.call == paused.call
    assert (stopped.call.tool, stopped.call.call_id) == ("Delegate", "b1")
    assert kinds(store.held[KEY]) == ["paused", "stopped"]


async def test_a_stop_pressed_while_the_lead_waits_to_pause_wins(tmp_path: Path) -> None:
    store, stop = MemoryStore(), Stop()
    model = Crew(
        [beside_its_own("part one"), text_reply("never")],
        {"part one": [asyncio.Event(), text_reply("never")]},
    )
    running = asyncio.create_task(
        crew_run(helped(tmp_path, model), store, approve=later_for("refund"), stop=stop)
    )
    await until(lambda: batch_stored(store), "the lead's batch was stored")
    stop.now()
    lead = await asyncio.wait_for(running, 5)
    assert lead.stopped == "now" and lead.paused == []
    own = next(s for s in lead.steps if s.tool == "refund")
    assert (own.result, own.stopped, own.waiting) == ("stopped by the user", True, False)
    [worker] = lead.children
    assert isinstance(worker, Generation) and worker.stopped == "now"
    assert live(store.held[KEY]) is None and kinds(store.held[KEY]) == ["stopped"]


async def test_a_lead_stopped_while_its_child_waits_ends_the_child_stopped(
    tmp_path: Path,
) -> None:
    store, refunds, stop = MemoryStore(), Refunds(), Stop()
    calls = [
        ToolCall(
            id="b1",
            name="Delegate",
            args={"agent": "mid", "input": "slow part", "background": True},
        ),
        *delegating("desk", "refund it", call_id="s1").tool_calls,
    ]
    model = Crew(
        [Reply(tool_calls=calls, stop_reason="tool_use"), text_reply("never")],
        {"slow part": [asyncio.Event()], "refund it": [calling(REFUND), text_reply("never")]},
    )
    engine = engine_of(tmp_path, model, background=True)
    running = asyncio.create_task(
        lead_run(engine, store, refunds, approve=later_for("refund"), stop=stop)
    )
    await until(lambda: batch_stored(store), "the lead's batch was stored")
    stop.now()
    lead = await asyncio.wait_for(running, 5)
    assert lead.stopped == "now" and lead.paused == []
    [_, s1] = [s for s in lead.steps if s.tool == "Delegate"]
    assert (s1.result, s1.stopped, s1.waiting) == ("stopped by the user", True, False)
    desk_runs = [c for c in lead.children if c.agent == "desk"]
    assert [isinstance(c, Generation) and c.stopped for c in desk_runs] == [None, "now"]
    desk_key = ("acme", f"c1/{desk_runs[0].run_id}")
    assert store.held[desk_key][-1].content == "stopped by the user"
    assert store.held[desk_key][-1].tool_call_id == "c-refund"
    assert live(store.held[desk_key]) is None and refunds.done == []


async def test_a_lead_that_fails_while_its_child_waits_ends_it_two_levels_down(
    tmp_path: Path,
) -> None:
    store, refunds, recording = MemoryStore(), Refunds(), Recording()
    first = Reply(
        tool_calls=[*delegating("mid", "pass it on").tool_calls, OWN.model_copy()],
        stop_reason="tool_use",
    )
    model = Crew(
        [first, text_reply("never")],
        {
            "pass it on": [delegating("desk", "refund it"), text_reply("never")],
            "refund it": [calling(REFUND), text_reply("never")],
        },
    )

    async def approve(agent: str, call: ToolCall) -> bool | Later:
        return Later() if agent == "desk" else True

    async def breaks(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == "c-own":
            raise RuntimeError("the handler broke")

    with pytest.raises(RuntimeError, match="the handler broke"):
        await lead_run(
            engine_of(tmp_path, model, refinery=recording),
            store,
            refunds,
            approve=approve,
            on_event=breaks,
        )
    lead = recording.runs[-1]
    assert isinstance(lead, Generation) and lead.agent == "lead"
    [delegated] = [s for s in lead.steps if s.tool == "Delegate"]
    assert delegated.result == "NotRun: Delegate was not run; the run failed"
    mid_runs = [c for c in lead.children if c.agent == "mid"]
    assert [isinstance(c, Generation) and c.stopped for c in mid_runs] == [None, "now"]
    for mid in mid_runs:
        assert mid.call is not None and (mid.call.run_id, mid.call.call_id) == (lead.run_id, "d1")
    [mid_key] = [k for k in store.held if k[1].count("/") == 1]
    [desk_key] = [k for k in store.held if k[1].count("/") == 2]
    for key, call_id in ((mid_key, "d1"), (desk_key, "c-refund")):
        assert live(store.held[key]) is None
        assert [m.content for m in store.held[key] if m.tool_call_id == call_id] == [
            "stopped by the user"
        ]
    assert refunds.done == [{"amount": 3}]
