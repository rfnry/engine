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
from rfnry_engine.models import (
    Message,
    Pause,
    Refused,
    Reply,
    State,
    Text,
    ToolCall,
    ToolSpec,
)
from rfnry_engine.records import (
    Decision,
    Generation,
    MessagesTaken,
    Run,
    RunEnded,
    RunStarted,
    ToolCallStarted,
    ToolResultReady,
)
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
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


def two_desks() -> Reply:
    calls = [
        ToolCall(id="d1", name="Delegate", args={"agent": "desk", "input": "refund A"}),
        ToolCall(id="d2", name="Delegate", args={"agent": "desk", "input": "refund B"}),
    ]
    return Reply(tool_calls=calls, stop_reason="tool_use")


@pytest.mark.parametrize("slots", [None, 1], ids=["no cap", "max_parallel 1"])
async def test_two_paused_delegates_of_one_reply_resume_one_approved_and_one_refused(
    tmp_path: Path, slots: int | None
) -> None:
    store, refunds = MemoryStore(), Refunds()
    second = REFUND.model_copy(update={"args": {"amount": 20}})
    model = Crew(
        [two_desks(), text_reply("both answered")],
        {
            "refund A": [calling(REFUND), text_reply("A refunded")],
            "refund B": [calling(second), text_reply("B not refunded")],
        },
    )
    folder = tree(tmp_path)
    if slots is not None:
        amend(folder, "lead", orchestration={"max_parallel": slots})
    engine = Engine(folder, providers=provided({"crew": model, "judge": None}))
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [a] = [w for w in first.paused if w.call.args == {"amount": 10}]
    [b] = [w for w in first.paused if w.call.args == {"amount": 20}]
    mark = mark_of(store.held[KEY])
    assert set(mark.delegated) == {"d1", "d2"} and mark.below == first.paused
    again = await lead_run(
        engine, store, refunds, "go on", answers={a.id: True, b.id: Refused("too much")}
    )
    assert again.error is None and again.paused == [] and again.reply == "both answered"
    assert refunds.done == [{"amount": 10}]
    assert sorted(str(s.result) for s in again.steps) == ["A refunded", "B not refunded"]
    for key in mark.delegated.values():
        assert live(store.held[("acme", key)]) is None
    assert live(store.held[KEY]) is None


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


async def test_a_stopped_paused_helper_reads_what_it_said_before_it_paused(
    tmp_path: Path,
) -> None:
    store, ended = MemoryStore(), asyncio.Event()
    stopping = ToolCall(id="c-job", name="Job", args={"id": "d1", "stop": True})
    looking = Reply(text="Looking it up.", tool_calls=[LOOKUP], stop_reason="tool_use")
    model = Crew(
        [background("part one"), ended, calling(stopping), text_reply("stopped it")],
        {"part one": [looking, text_reply("never")]},
    )
    lead = await crew_run(helped(tmp_path, model), store, on_event=ended_below(ended))
    assert result_of(lead, "c-job")["stopped"] == "Looking it up."
    [notice] = [m for m in lead.conversation if m.notice and m.notice.kind == "stopped"]
    assert notice.content == "[helper d1 · worker · stopped]\nLooking it up."
    paused, stopped = lead.children
    assert isinstance(stopped, Generation) and stopped.reply == "" and stopped.calls == 0


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


CAPPED = (
    "InputError: tools/delegate: max_background is {cap}, and {held}; "
    "wait for a helper to end or stop one"
)


class Overlapping:
    def __init__(self) -> None:
        self.now = 0
        self.most = 0

    def tools(self) -> Tools:
        async def lookup(args: dict[str, Any]) -> str:
            self.now += 1
            self.most = max(self.most, self.now)
            await asyncio.sleep(0.05)
            self.now -= 1
            return "found"

        return Tools([Tool("lookup", "Look an order up", body=lookup), Refunds().tools()])


def capped(tmp_path: Path, model: Crew, cap: int) -> Engine:
    folder = crew(
        tmp_path / "crew",
        worker={"tools": {"allow": ["lookup"]}},
        background=True,
        max_background=cap,
    )
    amend(folder, "lead", tools={"allow": ["refund"]})
    return team(folder, model)


def refusal_of(run: Generation, *call_ids: str) -> str:
    [refused] = [m for m in run.conversation if m.tool_call_id in call_ids and m.is_error]
    return str(refused.content)


async def test_a_paused_helper_keeps_its_max_background_place_through_its_resume(
    tmp_path: Path,
) -> None:
    store, overlapping, ended = MemoryStore(), Overlapping(), asyncio.Event()
    model = Crew(
        [
            background("part one"),
            ended,
            background("part two", prefix="e"),
            text_reply("one at a time"),
            text_reply("part one goes on"),
            text_reply("all done"),
        ],
        {
            "part one": [calling(LOOKUP), text_reply("part one done")],
            "part two": [calling(LOOKUP), text_reply("part two done")],
        },
    )
    engine = capped(tmp_path, model, 1)
    first = await engine.run(
        "lead",
        "acme",
        "go",
        store=store,
        conversation="c1",
        tools=overlapping.tools(),
        approve=later_for("lookup"),
        on_event=ended_below(ended),
    )
    assert isinstance(first, Generation)
    assert refusal_of(first, "e1") == CAPPED.format(cap=1, held="paused until answered: d1")
    [w] = first.paused
    assert w.conversation == "c1/d1" and model.calls["part two"] == []
    again = await engine.run(
        "lead",
        "acme",
        "go on",
        store=store,
        conversation="c1",
        tools=overlapping.tools(),
        approve=later_for("lookup"),
        answers={w.id: True for w in first.paused},
    )
    assert isinstance(again, Generation)
    assert again.error is None and again.paused == [] and again.reply == "all done"
    assert overlapping.most == 1 and notices_of(store, "d1") == ["paused", "finished"]


async def test_a_background_delegate_past_the_cap_names_the_helpers_running_and_paused(
    tmp_path: Path,
) -> None:
    store, ended, gate = MemoryStore(), asyncio.Event(), asyncio.Event()
    model = Crew(
        [
            background("part one"),
            ended,
            background("part two", "part three", prefix="e"),
            text_reply("waiting on part two"),
            text_reply("part one waits"),
        ],
        {
            "part one": [calling(LOOKUP), text_reply("never")],
            "part two": [gate, text_reply("part two done")],
            "part three": [text_reply("never")],
        },
    )
    watch = ended_below(ended)

    async def opened(event: Any) -> None:
        await watch(event)
        if isinstance(event, ToolResultReady) and event.step.is_error:
            gate.set()

    lead = await asyncio.wait_for(crew_run(capped(tmp_path, model, 2), store, on_event=opened), 10)
    held = "running now: d2; paused until answered: d1"
    assert refusal_of(lead, "e1", "e2") == CAPPED.format(cap=2, held=held)
    assert model.calls["part three"] == [] and len(model.calls["part two"]) == 1
    assert [w.conversation for w in lead.paused] == ["c1/d1"]
    assert notices_of(store, "d2") == ["finished"]


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


LEAD_STOPPED = (
    "InputError: tools/helpers: your lead was stopped or failed before it answered; "
    "you stop with it"
)


async def test_an_ask_made_once_a_lead_waiting_to_pause_is_stopped_reads_its_lead_stopped(
    tmp_path: Path,
) -> None:
    store, stop, gate, asking = MemoryStore(), Stop(), asyncio.Event(), asyncio.Event()
    model = Crew(
        [beside_its_own("part one"), text_reply("never")],
        {"part one": [gate, calling(ASK), text_reply("never")]},
    )

    async def approve(agent: str, call: ToolCall) -> bool | Later:
        if call.name == "Ask":
            asking.set()
        return Later() if call.name == "refund" else True

    async def watch(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == OWN.id and event.step.stopped:
            gate.set()
            await asking.wait()
            for _ in range(20):
                await asyncio.sleep(0)

    running = asyncio.create_task(
        crew_run(helped(tmp_path, model), store, approve=approve, stop=stop, on_event=watch)
    )
    await until(lambda: batch_stored(store), "the lead's batch was stored")
    stop.after_step()
    lead = await asyncio.wait_for(running, 5)
    assert lead.stopped == "after_step" and lead.paused == []
    answered = answer_in(store, HELPER_KEY, "a1")
    assert answered.is_error and answered.content == LEAD_STOPPED
    [worker] = lead.children
    assert isinstance(worker, Generation) and worker.stopped == "after_step"
    assert live(store.held[HELPER_KEY]) is None and "paused" not in kinds(store.held[KEY])


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


class Writing(Recording):
    def __init__(self) -> None:
        super().__init__()
        self.writing = asyncio.Event()
        self.gate = asyncio.Event()

    async def record(self, run: Run, *, skip: Collection[str] = ()) -> None:
        if isinstance(run, Generation) and run.agent == "worker" and run.paused:
            self.writing.set()
            await self.gate.wait()
        await super().record(run, skip=skip)


def told(messages: list[Message]) -> list[tuple[str, str | None]]:
    return [(m.notice.kind, m.notice.run_id) for m in messages if m.notice is not None]


def paused_then_stopped(lead: Generation) -> tuple[Generation, Generation]:
    paused, stopped = lead.children
    assert isinstance(paused, Generation) and isinstance(stopped, Generation)
    assert paused.paused and paused.stopped is None
    assert stopped.stopped == "now" and stopped.resumed and stopped.paused == []
    return paused, stopped


async def test_a_paused_helper_is_stopped_once_its_record_is_written(tmp_path: Path) -> None:
    store, recording, seen = MemoryStore(), Writing(), Seen()
    stopping = ToolCall(id="c-job", name="Job", args={"id": "d1", "stop": True})
    model = Crew(
        [background("part one"), recording.writing, calling(stopping), text_reply("stopped it")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    running = asyncio.create_task(
        crew_run(helped(tmp_path, model, refinery=recording), store, on_event=seen)
    )
    await until(
        lambda: any(isinstance(e, ToolCallStarted) and e.call_id == "c-job" for e in seen.events),
        "the lead's Job call started",
    )
    await asyncio.sleep(0.05)
    assert not running.done()
    recording.gate.set()
    lead = await asyncio.wait_for(running, 5)
    paused, stopped = paused_then_stopped(lead)
    assert told(store.held[KEY]) == [("paused", paused.run_id), ("stopped", stopped.run_id)]
    job_result = result_of(lead, "c-job")
    assert (job_result["running"], job_result["stopped"]) == (False, "")
    workers = [r.run_id for r in recording.runs if r.agent == "worker"]
    assert workers == [paused.run_id, stopped.run_id]
    assert lead.reply == "stopped it" and live(store.held[KEY]) is None


async def test_a_stopped_lead_stops_its_paused_helper_once_its_record_is_written(
    tmp_path: Path,
) -> None:
    store, recording, stop = MemoryStore(), Writing(), Stop()
    model = Crew(
        [background("part one"), asyncio.Event(), text_reply("never")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    running = asyncio.create_task(
        crew_run(helped(tmp_path, model, refinery=recording), store, stop=stop)
    )
    await asyncio.wait_for(recording.writing.wait(), 5)
    stop.now()
    await asyncio.sleep(0.05)
    assert not running.done()
    recording.gate.set()
    lead = await asyncio.wait_for(running, 5)
    assert lead.stopped == "now" and lead.paused == []
    paused, stopped = paused_then_stopped(lead)
    assert told(store.held[KEY]) == [("paused", paused.run_id), ("stopped", stopped.run_id)]
    held = store.held[HELPER_KEY]
    assert (held[-1].tool_call_id, held[-1].content) == ("c-lookup", "stopped by the user")
    assert live(held) is None


async def test_a_stopped_paused_helper_reads_back_beside_its_pause(
    tmp_path: Path, storage: Storage
) -> None:
    store, ended, refinery = MemoryStore(), asyncio.Event(), Refinery(storage)
    stopping = ToolCall(id="c-job", name="Job", args={"id": "d1", "stop": True})
    model = Crew(
        [
            background("part one"),
            ended,
            calling(stopping),
            text_reply("stopped it"),
            text_reply("heard it"),
        ],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    lead = await asyncio.wait_for(
        crew_run(helped(tmp_path, model, refinery=refinery), store, on_event=ended_below(ended)),
        10,
    )
    paused, stopped = paused_then_stopped(lead)
    assert told(store.held[KEY]) == [("paused", paused.run_id), ("stopped", stopped.run_id)]
    read = await refinery.run("acme", lead.run_id)
    assert isinstance(read, Generation)
    assert sorted(c.run_id for c in read.children) == sorted([paused.run_id, stopped.run_id])


async def test_a_resume_whose_stop_is_pressed_ends_the_child_it_would_go_on_with(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    model = Crew(
        [delegating("desk", "refund it"), text_reply("never")],
        {"refund it": [calling(REFUND), text_reply("never")]},
    )
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [w] = first.paused
    [desk] = first.children
    desk_key = ("acme", f"c1/{desk.run_id}")
    stop = Stop()
    stop.now()
    again = await lead_run(engine, store, refunds, "go on", answers={w.id: True}, stop=stop)
    assert again.stopped == "now" and again.paused == []
    assert [(s.tool, s.result) for s in again.steps] == [("Delegate", "stopped by the user")]
    assert live(store.held[KEY]) is None and live(store.held[desk_key]) is None
    assert [m.content for m in store.held[desk_key] if m.tool_call_id == "c-refund"] == [
        "stopped by the user"
    ]
    [ended] = again.children
    assert isinstance(ended, Generation) and ended.agent == "desk"
    assert again.steps[0].child_run_id == ended.run_id
    assert ended.stopped == "now" and ended.resumed and ended.input == ""
    assert ended.call is not None and (ended.call.run_id, ended.call.call_id) == (
        again.run_id,
        "d1",
    )
    assert refunds.done == [] and len(model.calls["refund it"]) == 1


async def test_a_resume_whose_stop_is_pressed_ends_its_children_two_levels_down(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    model = two_levels(text_reply("never"))
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [w] = first.paused
    stop = Stop()
    stop.now()
    again = await lead_run(engine, store, refunds, "go on", answers={w.id: True}, stop=stop)
    assert again.stopped == "now" and again.paused == []
    [mid_key] = [k for k in store.held if k[1].count("/") == 1]
    [desk_key] = [k for k in store.held if k[1].count("/") == 2]
    for key, call_id in ((KEY, "d1"), (mid_key, "d1"), (desk_key, "c-refund")):
        assert live(store.held[key]) is None
        assert [m.content for m in store.held[key] if m.tool_call_id == call_id] == [
            "stopped by the user"
        ]
    [mid] = again.children
    assert isinstance(mid, Generation) and mid.stopped == "now"
    [desk] = mid.children
    assert isinstance(desk, Generation) and desk.stopped == "now"
    assert refunds.done == [] and len(model.calls["refund it"]) == 1


class Looked:
    def __init__(self) -> None:
        self.times = 0
        self.refunds = Refunds()

    def tools(self) -> Tools:
        async def lookup(args: dict[str, Any]) -> str:
            self.times += 1
            return "found"

        return Tools([Tool("lookup", "Look an order up", body=lookup), self.refunds.tools()])


async def team_run(
    engine: Engine, store: MemoryStore, looked: Looked, input: State = "go", **options: Any
) -> Generation:
    options.setdefault("approve", later_for("lookup"))
    run = await engine.run(
        "lead",
        "acme",
        input,
        store=store,
        conversation="c1",
        tools=looked.tools(),
        **options,
    )
    assert isinstance(run, Generation)
    return run


class Ends:
    def __init__(self) -> None:
        self.seen: list[RunEnded] = []
        self.waiting: list[tuple[str, int, asyncio.Event]] = []

    def of(self, agent: str, count: int = 1) -> asyncio.Event:
        ended = asyncio.Event()
        self.waiting.append((agent, count, ended))
        return ended

    async def __call__(self, event: Any) -> None:
        if isinstance(event, RunEnded) and event.call is not None:
            self.seen.append(event)
            for agent, count, ended in self.waiting:
                if sum(seen.agent == agent for seen in self.seen) >= count:
                    ended.set()


def notices_of(store: MemoryStore, job: str, key: tuple[str, str] = KEY) -> list[str]:
    return [m.notice.kind for m in store.held[key] if m.notice is not None and m.notice.job == job]


def answer_in(store: MemoryStore, key: tuple[str, str], call_id: str) -> Message:
    [message] = [m for m in store.held[key] if m.tool_call_id == call_id]
    return message


async def test_a_resume_reattaches_a_paused_helper_under_its_job_id(tmp_path: Path) -> None:
    store, looked = MemoryStore(), Looked()
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            text_reply("it waits"),
            background("part two", prefix="e"),
            text_reply("both going"),
            text_reply("all done"),
        ],
        {
            "part one": [calling(LOOKUP), text_reply("part one done")],
            "part two": [text_reply("part two done")],
        },
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked)
    [w] = first.paused
    assert w.conversation == "c1/d1" and looked.times == 0
    again = await team_run(engine, store, looked, "go on", answers={w.id: True})
    assert looked.times == 1
    assert answer_in(store, HELPER_KEY, "c-lookup").content == "found"
    assert set(store.held) == {KEY, HELPER_KEY, ("acme", "c1/d2")}
    assert notices_of(store, "d1")[-1] == "finished"
    assert not any(m.notice and m.notice.kind == "lost" for m in store.held[KEY])
    assert result_of(again, "e1") == {"job": "d2"}
    assert again.error is None and again.paused == [] and again.reply == "all done"
    by_input = {str(c.input): c for c in again.children if isinstance(c, Generation)}
    resumed = by_input[""]
    assert resumed.agent == "worker" and resumed.resumed and resumed.reply == "part one done"
    assert resumed.run_id != first.children[0].run_id
    assert resumed.call is not None
    assert (resumed.call.run_id, resumed.call.call_id) == (again.run_id, "b1")
    assert live(store.held[HELPER_KEY]) is None


async def test_a_reattached_helpers_run_started_names_its_paused_run(tmp_path: Path) -> None:
    store, looked, seen = MemoryStore(), Looked(), Seen()
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            text_reply("it waits"),
            background("part two", prefix="e"),
            text_reply("both going"),
            text_reply("all done"),
        ],
        {
            "part one": [calling(LOOKUP), text_reply("part one done")],
            "part two": [text_reply("part two done")],
        },
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, on_event=seen)
    [w] = first.paused
    [paused] = first.children
    again = await team_run(engine, store, looked, "go on", answers={w.id: True}, on_event=seen)
    assert again.error is None and again.reply == "all done"
    by_input = {str(c.input): c for c in again.children if isinstance(c, Generation)}
    assert w.run_id == paused.run_id and by_input[""].resumed
    resumes = {e.run_id: e.resumes for e in seen.events if isinstance(e, RunStarted)}
    assert resumes == {
        first.run_id: None,
        paused.run_id: None,
        again.run_id: first.run_id,
        by_input[""].run_id: paused.run_id,
        by_input["part two"].run_id: None,
    }


async def test_a_helper_run_again_after_a_cut_resume_names_the_run_that_wrote_its_mark(
    tmp_path: Path,
) -> None:
    engine, store, _, first = await paused_alone(
        tmp_path, text_reply("heard"), text_reply("still waits")
    )
    [w] = first.paused
    cut = mark_of(store.held[KEY]).model_copy(update={"below": [], "run_id": "cut-off"})
    store.held[KEY].append(Message(role="assistant", content="[paused]", pause=cut))
    found = mark_of(store.held[HELPER_KEY]).run_id
    seen = Seen()
    again = await team_run(
        engine, store, Looked(), "the process restarted; carry on", on_event=seen
    )
    assert not again.resumed and again.paused == [w] and found == w.run_id
    started = [(e.agent, e.resumes) for e in seen.events if isinstance(e, RunStarted)]
    assert started == [("lead", None), ("worker", found)]


async def test_a_paused_helper_with_no_answer_stays_paused(tmp_path: Path) -> None:
    store, looked = MemoryStore(), Looked()
    ends = Ends()
    both_ended = ends.of("worker", count=2)
    second = ToolCall(id="c-lookup-2", name="lookup", args={})
    sending = ToolCall(id="c-job", name="Job", args={"id": "d2", "send": "hurry"})
    model = Crew(
        [
            background("Work on d1", "Work on d2"),
            both_ended,
            text_reply("both heard"),
            text_reply("both wait"),
            calling(JOBS, sending),
            text_reply("d1 goes on"),
            text_reply("d2 waits"),
        ],
        {
            "Work on d1": [calling(LOOKUP), text_reply("d1 done")],
            "Work on d2": [calling(second), text_reply("never")],
        },
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, on_event=ends)
    w1, w2 = first.paused
    assert (w1.conversation, w2.conversation) == ("c1/d1", "c1/d2")
    calls_before = len(model.calls["Work on d2"])
    again = await team_run(engine, store, looked, "go on", answers={w1.id: True})
    assert len(model.calls["Work on d2"]) == calls_before
    assert again.paused == [w2] and mark_of(store.held[KEY]).below == [w2]
    listed = {job["job"]: job for job in result_of(again, "c-jobs")["jobs"]}
    assert listed["d2"]["ended"] == "paused" and listed["d2"]["running"] is False
    assert listed["d2"]["input"] == "Work on d2"
    [refused] = [m for m in again.conversation if m.tool_call_id == "c-job"]
    assert refused.is_error and refused.content == E11.replace("d1", "d2")
    assert notices_of(store, "d1")[-1] == "finished" and notices_of(store, "d2") == ["paused"]
    assert live(store.held[("acme", "c1/d2")]) is not None and looked.times == 1


async def test_calls_with_the_same_model_id_in_two_helpers_wait_apart(tmp_path: Path) -> None:
    store, looked = MemoryStore(), Looked()
    ends = Ends()
    both_ended = ends.of("worker", count=2)
    model = Crew(
        [
            background("Work on d1", "Work on d2"),
            both_ended,
            text_reply("both heard"),
            text_reply("both wait"),
            text_reply("d2 goes on"),
            text_reply("d1 waits"),
        ],
        {
            "Work on d1": [calling(LOOKUP), text_reply("never")],
            "Work on d2": [calling(LOOKUP), text_reply("d2 done")],
        },
    )
    engine = helped(tmp_path, model)
    lead = await team_run(engine, store, looked, on_event=ends)
    assert [w.call.id for w in lead.paused] == ["c-lookup", "c-lookup"]
    assert len({w.id for w in lead.paused}) == 2
    w1, w2 = lead.paused
    again = await team_run(engine, store, looked, "go on", answers={w2.id: True})
    assert looked.times == 1 and again.paused == [w1]
    assert answer_in(store, ("acme", "c1/d2"), "c-lookup").content == "found"
    assert not [m for m in store.held[HELPER_KEY] if m.tool_call_id == "c-lookup"]
    assert len(model.calls["Work on d1"]) == 1


async def asked_and_paused(tmp_path: Path, *lead: Reply) -> tuple[Engine, MemoryStore, Crew]:
    store, gate = MemoryStore(), asyncio.Event()
    model = Crew(
        [beside_its_own("part one"), *lead],
        {"part one": [gate, calling(ASK), text_reply("part done")]},
    )
    engine = helped(tmp_path, model)
    running = asyncio.create_task(crew_run(engine, store, approve=later_for("refund")))
    await until(lambda: batch_stored(store), "the lead's batch was stored")
    gate.set()
    await asyncio.wait_for(running, 5)
    assert mark_of(store.held[HELPER_KEY]).asking == "a1"
    return engine, store, model


async def test_a_helper_paused_on_its_ask_runs_again_once_its_lead_answers(
    tmp_path: Path,
) -> None:
    answering = ToolCall(id="c-job", name="Job", args={"id": "d1", "send": "yes"})
    engine, store, model = await asked_and_paused(
        tmp_path, calling(answering), text_reply("waiting"), text_reply("all done")
    )
    [w] = mark_of(store.held[KEY]).waiting
    again = await team_run(
        engine, store, Looked(), "go on", answers={w.id: True}, approve=later_for("refund")
    )
    assert answer_in(store, HELPER_KEY, "a1").content == "yes"
    assert not answer_in(store, HELPER_KEY, "a1").is_error
    assert notices_of(store, "d1")[-1] == "finished"
    assert again.error is None and again.paused == [] and again.reply == "all done"
    job_result = result_of(again, "c-job")
    assert job_result["running"] is True and job_result["asking"] is None
    [worker] = [c for c in again.children if isinstance(c, Generation)]
    assert worker.reply == "part done" and worker.input == ""
    [ask] = [s for s in worker.steps if s.tool == "Ask"]
    assert (ask.result, ask.is_error, ask.waiting) == ("yes", False, False)
    assert live(store.held[HELPER_KEY]) is None


async def test_a_lead_that_answers_no_ask_lets_its_paused_helper_go_on(tmp_path: Path) -> None:
    engine, store, model = await asked_and_paused(
        tmp_path, text_reply("not now"), text_reply("all done")
    )
    [w] = mark_of(store.held[KEY]).waiting
    again = await team_run(
        engine, store, Looked(), "go on", answers={w.id: True}, approve=later_for("refund")
    )
    answered = answer_in(store, HELPER_KEY, "a1")
    assert answered.is_error and answered.content == (
        "InputError: tools/helpers: your lead ended its turn without answering"
    )
    assert notices_of(store, "d1")[-1] == "finished"
    assert again.error is None and again.paused == [] and again.reply == "all done"
    assert model.calls["part one"][-1][-1].content == answered.content


async def paused_alone(
    tmp_path: Path, *lead: Reply, **engine: Any
) -> tuple[Engine, MemoryStore, Crew, Generation]:
    store = MemoryStore()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("it waits"), *lead],
        {"part one": [calling(LOOKUP), text_reply("part one done")]},
    )
    built = helped(tmp_path, model, **engine)
    first = await team_run(built, store, Looked())
    assert len(first.paused) == 1 and len(model.calls["lead"]) == 3
    return built, store, model, first


async def test_a_resume_that_answers_no_helper_call_pauses_again_at_once(tmp_path: Path) -> None:
    engine, store, model, first = await paused_alone(tmp_path)
    again = await team_run(engine, store, Looked(), "any news?", answers={})
    assert again.resumed and again.paused == first.paused and again.calls == 0
    assert len(model.calls["lead"]) == 3 and len(model.calls["part one"]) == 1
    mark = mark_of(store.held[KEY])
    assert mark.below == first.paused and mark.held[-1].content == "any news?"
    assert notices_of(store, "d1") == ["paused"]


async def test_a_resume_that_pauses_again_at_once_reads_back_and_is_not_replayed(
    tmp_path: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    engine, store, model, first = await paused_alone(tmp_path, refinery=refinery)
    news = [Text(text="any news?")]
    again = await team_run(engine, store, Looked(), news, answers={})
    assert again.calls == 0 and again.paused == first.paused and again.input == news
    assert again.messages[-1].pause is not None
    assert mark_of(again.conversation).held[-1].content == news
    kept = await refinery.run("acme", again.run_id)
    assert isinstance(kept, Generation) and kept.input == news
    candidate = helped(tmp_path / "candidate", Crew([text_reply("x")]))
    pairs = [
        pair
        async for pair in refinery.replay("acme", candidate, agent="lead", tools=Looked().tools())
    ]
    [replayed] = [pair for pair in pairs if pair.recorded.run_id == again.run_id]
    assert replayed.candidate is None and replayed.error == (
        f"InputError: refinery: run {again.run_id!r} resumed a paused conversation and ended "
        "before its first model call; it cannot be replayed"
    )


async def test_an_output_refused_for_a_helper_paused_on_its_ask_lets_the_lead_answer_it(
    tmp_path: Path,
) -> None:
    store, gate = MemoryStore(), asyncio.Event()
    output = ToolCall(id="c-out", name="Output", args={"text": "done"})
    answering = ToolCall(id="c-job", name="Job", args={"id": "d1", "send": "the first"})
    model = Crew(
        [
            beside_its_own("part one"),
            calling(output),
            calling(answering),
            text_reply("waiting"),
            calling(output.model_copy(update={"id": "c-out-2"})),
        ],
        {"part one": [gate, calling(ASK), text_reply("part done")]},
    )
    engine = helped(tmp_path, model)
    running = asyncio.create_task(
        crew_run(engine, store, approve=later_for("refund"), expect=Reported)
    )
    await until(lambda: batch_stored(store), "the lead's batch was stored")
    gate.set()
    first = await asyncio.wait_for(running, 5)
    [w] = first.paused
    again = await team_run(
        engine,
        store,
        Looked(),
        "go on",
        answers={w.id: True},
        approve=later_for("refund"),
        expect=Reported,
    )
    refused = next(s for s in again.steps if s.tool == "Output")
    assert refused.result == PAUSED.format(names="d1") and refused.is_error
    assert again.paused == [] and again.error is None
    assert json.loads(again.reply) == {"text": "done"}
    assert answer_in(store, HELPER_KEY, "a1").content == "the first"
    fresh, after = [m.pause for m in store.held[KEY] if m.pause and m.pause.run_id == again.run_id]
    assert fresh.waiting == [] and live(store.held[KEY]) is None
    assert (after.waiting, after.below, after.delegated) == ([], [], {})
    assert after.held == fresh.held and kinds(after.held)[-1] == "paused"


async def test_an_output_refused_for_a_helper_its_reply_then_stops_lets_the_lead_go_on(
    tmp_path: Path,
) -> None:
    store, ended = MemoryStore(), asyncio.Event()
    output = ToolCall(id="c-out", name="Output", args={"text": "done"})
    stopping = ToolCall(id="c-job", name="Job", args={"id": "d1", "stop": True})
    model = Crew(
        [
            background("part one"),
            ended,
            calling(output, stopping),
            calling(output.model_copy(update={"id": "c-out-2"})),
        ],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    lead = await crew_run(
        helped(tmp_path, model), store, expect=Reported, on_event=ended_below(ended)
    )
    refused = next(s for s in lead.steps if s.tool == "Output")
    assert refused.result == PAUSED.format(names="d1") and refused.is_error
    assert lead.error is None and lead.paused == [] and json.loads(lead.reply) == {"text": "done"}
    assert all(m.pause is None for m in store.held[KEY]) and len(model.calls["lead"]) == 3
    assert notices_of(store, "d1") == ["paused", "stopped"]


async def test_a_send_answers_only_the_ask_its_helper_paused_on(tmp_path: Path) -> None:
    answering = ToolCall(id="c-job", name="Job", args={"id": "d1", "send": "yes"})
    waiting = ToolCall(id="c-wait", name="Job", args={"id": "d1", "wait": 5})
    again_asked = ToolCall(id="c-job-2", name="Job", args={"id": "d1", "send": "the second"})
    engine, store, model = await asked_and_paused(
        tmp_path,
        calling(answering),
        calling(waiting),
        calling(again_asked),
        text_reply("all done"),
        text_reply("all done"),
    )
    mark = mark_of(store.held[HELPER_KEY])
    cut = mark.model_copy(update={"asking": None, "run_id": "cut-off"})
    store.held[HELPER_KEY].append(Message(role="assistant", content="[paused]", pause=cut))
    second = ToolCall(id="a2", name="Ask", args={"question": "and which day?"})
    model.replies["part one"] = [calling(second), text_reply("part done")]
    [w] = mark_of(store.held[KEY]).waiting
    again = await team_run(
        engine, store, Looked(), "go on", answers={w.id: True}, approve=later_for("refund")
    )
    assert str(answer_in(store, HELPER_KEY, "a1").content).startswith("Ended: Ask did not return")
    assert answer_in(store, HELPER_KEY, "a2").content == "the second"
    assert again.error is None and again.paused == [] and again.reply == "all done"
    assert notices_of(store, "d1")[-2:] == ["asks", "finished"]


async def test_a_stop_pressed_after_the_resume_step_ends_its_restored_paused_helper(
    tmp_path: Path,
) -> None:
    store, looked, stop = MemoryStore(), Looked(), Stop()
    both = later_for("lookup", "refund")
    model = Crew(
        [beside_its_own("part one"), text_reply("never")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, approve=both)
    own, lookup = first.paused

    async def press(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == OWN.id:
            stop.now()

    again = await team_run(
        engine, store, looked, "go on", answers={own.id: True}, stop=stop, on_event=press
    )
    assert again.stopped == "now" and again.paused == [] and again.calls == 0
    assert looked.refunds.done == [{"amount": 3}] and answer_in(store, KEY, OWN.id).content == (
        "refunded"
    )
    assert answer_in(store, HELPER_KEY, LOOKUP.id).content == "stopped by the user"
    assert live(store.held[HELPER_KEY]) is None and live(store.held[KEY]) is None
    assert notices_of(store, "d1") == ["paused", "stopped"]
    [stopped] = again.children
    assert isinstance(stopped, Generation) and stopped.stopped == "now"
    assert len(model.calls["part one"]) == 1 and looked.times == 0


async def test_a_resume_whose_stop_is_pressed_ends_its_restored_paused_helper(
    tmp_path: Path,
) -> None:
    engine, store, model, first = await paused_alone(tmp_path)
    stop = Stop()
    stop.now()
    again = await team_run(engine, store, Looked(), "go on", answers={}, stop=stop)
    assert again.stopped == "now" and again.paused == []
    assert answer_in(store, HELPER_KEY, "c-lookup").content == "stopped by the user"
    assert live(store.held[HELPER_KEY]) is None and live(store.held[KEY]) is None
    assert notices_of(store, "d1") == ["paused", "stopped"]
    [stopped] = again.children
    assert isinstance(stopped, Generation) and stopped.stopped == "now"
    assert stopped.call is not None
    assert (stopped.call.run_id, stopped.call.call_id) == (again.run_id, "b1")
    assert len(model.calls["part one"]) == 1


async def test_a_stopped_helper_stops_its_own_paused_helpers(tmp_path: Path) -> None:
    store, refunds = MemoryStore(), Refunds()
    folder = tree(tmp_path, background=True)
    amend(folder, "mid", orchestration={"delegate_to": ["desk"], "background": True})
    ends = Ends()
    mid_ended, desk_ended = ends.of("mid"), ends.of("desk")

    model = Crew(
        [background("pass it on", agent="mid"), mid_ended, calling(JOBS), text_reply("waits")],
        {
            "pass it on": [
                background("refund it", agent="desk"),
                desk_ended,
                calling(JOBS),
                text_reply("mid waits"),
            ],
            "refund it": [calling(REFUND), text_reply("never")],
        },
    )
    engine = Engine(folder, providers=provided({"crew": model, "judge": None}))
    first = await lead_run(engine, store, refunds, approve=later_for("refund"), on_event=ends)
    [w] = first.paused
    assert w.conversation == "c1/d1/d1"
    stop = Stop()
    stop.now()
    again = await lead_run(engine, store, refunds, "go on", stop=stop)
    assert again.stopped == "now" and again.paused == []
    desk_key, mid_key = ("acme", "c1/d1/d1"), ("acme", "c1/d1")
    assert answer_in(store, desk_key, "c-refund").content == "stopped by the user"
    for key in (KEY, mid_key, desk_key):
        assert live(store.held[key]) is None
    assert notices_of(store, "d1", mid_key) == ["paused", "stopped"]
    assert notices_of(store, "d1") == ["paused", "stopped"]
    assert refunds.done == [] and len(model.calls["refund it"]) == 1


async def test_a_paused_helper_its_leads_mark_no_longer_lists_goes_on(tmp_path: Path) -> None:
    engine, store, model, first = await paused_alone(
        tmp_path, text_reply("heard"), text_reply("still waits")
    )
    [w] = first.paused
    cut = mark_of(store.held[KEY]).model_copy(update={"below": [], "run_id": "cut-off"})
    store.held[KEY].append(Message(role="assistant", content="[paused]", pause=cut))
    again = await team_run(engine, store, Looked(), "the process restarted; carry on")
    assert not again.resumed and again.paused == [w]
    assert len(model.calls["part one"]) == 1
    assert not [m for m in store.held[HELPER_KEY] if m.tool_call_id == "c-lookup"]
    assert mark_of(store.held[HELPER_KEY]).waiting == [w]
    assert notices_of(store, "d1") == ["paused", "paused"]
    assert not any(m.notice and m.notice.kind == "lost" for m in store.held[KEY])


async def test_a_paused_helper_whose_agent_is_gone_is_told_lost(tmp_path: Path) -> None:
    engine, store, model, first = await paused_alone(tmp_path)
    for message in store.held[KEY]:
        for call in message.tool_calls:
            if call.name == "Delegate":
                call.args["agent"] = "retired"
        if message.notice is not None:
            message.notice.agent = "retired"
    cut = mark_of(store.held[KEY]).model_copy(update={"below": [], "run_id": "cut-off"})
    store.held[KEY].append(Message(role="assistant", content="[paused]", pause=cut))
    model.replies["lead"].append(text_reply("noted"))
    again = await team_run(engine, store, Looked(), "carry on")
    assert again.paused == [] and again.reply == "noted"
    assert notices_of(store, "d1") == ["paused", "lost"]
    assert len(model.calls["part one"]) == 1


async def test_a_reattached_helper_holds_its_conversation_while_it_runs(tmp_path: Path) -> None:
    store, gate, looked = MemoryStore(), asyncio.Event(), Looked()
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            text_reply("it waits"),
            text_reply("going"),
            text_reply("all done"),
        ],
        {"part one": [calling(LOOKUP), gate, text_reply("part one done")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked)
    [w] = first.paused
    running = asyncio.create_task(team_run(engine, store, looked, "go on", answers={w.id: True}))
    await until(lambda: looked.times == 1, "the re-attached helper ran its call")
    with pytest.raises(
        InputError, match=re.escape("conversation 'c1/d1' of scope 'acme' is in use")
    ):
        await engine.run("worker", "acme", "x", store=store, conversation="c1/d1")
    gate.set()
    again = await asyncio.wait_for(running, 5)
    assert again.error is None and notices_of(store, "d1")[-1] == "finished"


def cut_after_its_fresh_mark(store: MemoryStore) -> Message:
    mark = mark_of(store.held[KEY])
    cut = mark.model_copy(update={"waiting": [], "below": [], "run_id": "cut-off"})
    store.held[KEY].append(Message(role="assistant", content="[paused]", pause=cut))
    [held] = [m for m in mark.held if m.notice is not None]
    return held


async def test_a_cut_resume_keeps_a_helper_whose_paused_notice_its_mark_held(
    tmp_path: Path,
) -> None:
    store, looked = MemoryStore(), Looked()
    both = later_for("lookup", "refund")
    model = Crew(
        [beside_its_own("part one"), text_reply("heard"), text_reply("still waits")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, approve=both)
    own, lookup = first.paused
    assert (own.call.name, lookup.call.name) == ("refund", "lookup")
    assert kinds(mark_of(store.held[KEY]).held) == ["paused"]
    held = cut_after_its_fresh_mark(store)
    again = await team_run(engine, store, looked, "the process restarted; carry on", approve=both)
    assert held.notice is not None
    told = [m for m in store.held[KEY] if m.notice and m.notice.run_id == held.notice.run_id]
    assert told == [held]
    assert notices_of(store, "d1") == ["paused", "paused"]
    assert not again.resumed and again.paused == [lookup]
    assert mark_of(store.held[HELPER_KEY]).waiting == [lookup]
    assert len(model.calls["part one"]) == 1 and looked.times == 0


async def test_an_inert_marks_held_outcome_is_appended_once_and_not_lost(tmp_path: Path) -> None:
    store, gate = MemoryStore(), asyncio.Event()
    model = Crew(
        [beside_its_own("part one"), text_reply("done")],
        {"part one": [gate, text_reply("part done")]},
    )
    engine = helped(tmp_path, model)
    running = asyncio.create_task(crew_run(engine, store, approve=later_for("refund")))
    await until(lambda: batch_stored(store), "the lead's batch was stored")
    gate.set()
    await asyncio.wait_for(running, 5)
    held = cut_after_its_fresh_mark(store)
    assert held.notice is not None and held.notice.kind == "finished"
    again = await team_run(engine, store, Looked(), "carry on")
    assert again.error is None and again.reply == "done" and again.paused == []
    assert notices_of(store, "d1") == ["finished"]
    assert [m for m in store.held[KEY] if m.notice is not None] == [held]
    assert held.content in [m.content for m in model.calls["lead"][-1]]


def split(tmp_path: Path, *lead: Reply) -> tuple[Engine, Crew]:
    calls = [
        ToolCall(
            id="b1",
            name="Delegate",
            args={"agent": "desk", "input": "refund later", "background": True},
        ),
        ToolCall(id="s1", name="Delegate", args={"agent": "desk", "input": "refund it"}),
    ]
    model = Crew(
        [Reply(tool_calls=calls, stop_reason="tool_use"), *lead],
        {
            "refund later": [calling(REFUND), text_reply("later done")],
            "refund it": [calling(SECOND), text_reply("it done")],
        },
    )
    return engine_of(tmp_path, model, background=True), model


async def paused_split(
    tmp_path: Path, store: MemoryStore, refunds: Refunds, *lead: Reply
) -> tuple[Engine, Generation, dict[tuple[str, str], list[Message]]]:
    engine, _ = split(tmp_path, *lead)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    assert [w.conversation for w in first.paused][1] == "c1/d1"
    assert live(store.held[KEY]) is not None
    return engine, first, {key: list(held) for key, held in store.held.items()}


def nothing_stopped(store: MemoryStore) -> bool:
    return not any(
        m.content == "stopped by the user" or (m.notice and m.notice.kind == "stopped")
        for held in store.held.values()
        for m in held
    )


async def retried(
    engine: Engine, store: MemoryStore, refunds: Refunds, first: Generation
) -> Generation:
    again = await lead_run(
        engine, store, refunds, "go on", answers={w.id: True for w in first.paused}
    )
    assert again.error is None and again.paused == [] and again.reply == "all done"
    assert sorted(done["amount"] for done in refunds.done) == [5, 10]
    assert notices_of(store, "d1")[-1] == "finished"
    return again


async def test_a_resume_whose_run_started_handler_raises_leaves_the_tree_paused(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    engine, first, before = await paused_split(
        tmp_path, store, refunds, text_reply("going"), text_reply("all done")
    )

    async def breaks(event: Any) -> None:
        if isinstance(event, RunStarted) and event.call is None:
            raise RuntimeError("the handler broke")

    with pytest.raises(RuntimeError, match="the handler broke"):
        await lead_run(
            engine,
            store,
            refunds,
            "go on",
            answers={w.id: True for w in first.paused},
            on_event=breaks,
        )
    assert store.held == before and nothing_stopped(store) and refunds.done == []
    await retried(engine, store, refunds, first)


async def test_a_resume_whose_store_fails_to_append_leaves_the_tree_paused(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    engine, first, before = await paused_split(
        tmp_path, store, refunds, text_reply("going"), text_reply("all done")
    )
    store.fail["append"] = OSError("the store is down")
    with pytest.raises(OSError, match="the store is down"):
        await lead_run(engine, store, refunds, "go on", answers={w.id: True for w in first.paused})
    assert store.held == before and nothing_stopped(store) and refunds.done == []
    await retried(engine, store, refunds, first)


async def test_a_resume_whose_window_fails_leaves_its_unreached_helper_paused(
    tmp_path: Path,
) -> None:
    broken = False

    def windows(conversation: str, length: int) -> int:
        return -1 if broken and conversation == "c1" else 0

    store, refunds = MemoryStore(windows), Refunds()
    engine, first, before = await paused_split(tmp_path, store, refunds, text_reply("d1 waits"))
    sync, background_call = first.paused
    broken = True
    with pytest.raises(InputError, match="the store's window for conversation 'c1'"):
        await lead_run(engine, store, refunds, "go on", answers={sync.id: True})
    assert store.held[HELPER_KEY] == before[HELPER_KEY] and nothing_stopped(store)
    assert refunds.done == [{"amount": 5}] and notices_of(store, "d1") == ["paused"]
    broken = False
    with pytest.raises(InputError, match=re.escape(NO_WAITING.format(id=sync.id))):
        await lead_run(engine, store, refunds, "go on", answers={sync.id: True})
    again = await lead_run(engine, store, refunds, "carry on")
    assert again.error is None and again.paused == [background_call]
    assert store.held[HELPER_KEY] == before[HELPER_KEY]


async def test_a_resume_step_that_fails_keeps_the_notices_its_mark_held(tmp_path: Path) -> None:
    store, looked = MemoryStore(), Looked()
    both = later_for("lookup", "refund")
    model = Crew(
        [beside_its_own("part one"), text_reply("going"), text_reply("all done")],
        {"part one": [calling(LOOKUP), text_reply("part one done")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, approve=both)
    own, lookup = first.paused
    [held] = mark_of(store.held[KEY]).held
    before = list(store.held[HELPER_KEY])

    async def breaks(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == "c-own":
            raise RuntimeError("the handler broke")

    with pytest.raises(RuntimeError, match="the handler broke"):
        await team_run(engine, store, looked, "go on", answers={own.id: True}, on_event=breaks)
    mark = live(store.held[KEY])
    assert mark is not None and mark.held == [held]
    assert (mark.waiting, mark.below) == ([], [lookup])
    assert answer_in(store, KEY, "c-own").content == "refunded"
    assert store.held[HELPER_KEY] == before
    again = await team_run(engine, store, looked, "carry on", answers={}, approve=both)
    assert again.resumed and again.paused == [lookup] and again.calls == 0
    assert not any(m.notice and m.notice.kind == "lost" for m in store.held[KEY])
    last = await team_run(engine, store, looked, "go on", answers={lookup.id: True})
    assert last.error is None and last.paused == [] and last.reply == "all done"
    assert notices_of(store, "d1") == ["paused", "finished"] and looked.times == 1
    assert len(looked.refunds.done) == 1


async def breaks_at_its_start(event: Any) -> None:
    if isinstance(event, RunStarted) and event.call is None:
        raise RuntimeError("the handler broke")


async def test_a_cut_resume_whose_next_resume_fails_at_its_start_stays_paused(
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
    cut = asyncio.create_task(
        lead_run(engine, store, refunds, "go on", answers={own.id: True, below.id: True})
    )
    await until(lambda: refunds.started == 2, "both approved refunds started")
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    cut_mark = mark_of(store.held[KEY])
    [desk_key] = [k for k in store.held if k != KEY]
    child, length = list(store.held[desk_key]), len(store.held[KEY])
    with pytest.raises(RuntimeError, match="the handler broke"):
        await lead_run(engine, store, refunds, "go on", answers={}, on_event=breaks_at_its_start)
    assert store.appended[-1] == ("c1", length, 2)
    mark = live(store.held[KEY])
    assert mark is not None and mark.delegated == cut_mark.delegated
    ended = [m for m in store.held[KEY] if str(m.content).startswith("Ended:")]
    assert [m.tool_call_id for m in ended] == ["c-own"]
    assert store.held[desk_key] == child and nothing_stopped(store)
    third = await lead_run(engine, store, refunds, "go on", answers={})
    assert third.error is None and third.resumed and third.reply == "it may have gone through"
    ended = [m for m in store.held[KEY] if str(m.content).startswith("Ended:")]
    assert [m.tool_call_id for m in ended] == ["c-own"]
    assert answer_in(store, KEY, "d1").content == "it was cut off"
    assert refunds.started == 2


async def test_a_paused_helper_stays_restorable_when_a_cut_resume_fails_at_its_start(
    tmp_path: Path,
) -> None:
    store, looked = MemoryStore(), Looked()
    both = later_for("lookup", "refund")
    model = Crew(
        [beside_its_own("part one"), text_reply("going"), text_reply("all done")],
        {"part one": [calling(LOOKUP), text_reply("part one done")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, approve=both)
    _, lookup = first.paused
    mark = mark_of(store.held[KEY])
    [held] = mark.held
    cut = mark.model_copy(update={"waiting": [], "run_id": "cut-off"})
    store.held[KEY].append(Message(role="assistant", content="[paused]", pause=cut))
    before = list(store.held[HELPER_KEY])
    with pytest.raises(RuntimeError, match="the handler broke"):
        await team_run(
            engine, store, looked, "go on", answers={}, on_event=breaks_at_its_start, approve=both
        )
    again = live(store.held[KEY])
    assert again is not None and again.held == [held] and again.below == [lookup]
    assert store.held[HELPER_KEY] == before and nothing_stopped(store)
    stored = list(store.held[KEY])
    with pytest.raises(RuntimeError, match="the handler broke"):
        await team_run(
            engine, store, looked, "go on", answers={}, on_event=breaks_at_its_start, approve=both
        )
    assert store.held[KEY] == stored
    retried = await team_run(engine, store, looked, "carry on", answers={}, approve=both)
    assert retried.resumed and retried.paused == [lookup]
    assert not any(m.notice and m.notice.kind == "lost" for m in store.held[KEY])
    last = await team_run(engine, store, looked, "go on", answers={lookup.id: True})
    assert last.error is None and last.reply == "all done"
    assert notices_of(store, "d1") == ["paused", "finished"] and looked.times == 1


def nested(
    tmp_path: Path, mid: dict[str, Any], *lead: Reply | asyncio.Event
) -> tuple[Engine, Crew]:
    folder = tree(tmp_path, background=True)
    amend(folder, "mid", orchestration=mid)
    model = Crew(
        list(lead),
        {
            "pass it on": [],
            "refund it": [calling(REFUND), text_reply("never")],
        },
    )
    return Engine(folder, providers=provided({"crew": model, "judge": None})), model


def fails_on_start(agent: str) -> Any:
    async def breaks(event: Any) -> None:
        if isinstance(event, RunStarted) and event.agent == agent:
            raise RuntimeError(f"{agent}'s handler broke")

    return breaks


def paused_mid(tmp_path: Path, *after: Reply) -> tuple[Engine, Crew, Ends]:
    ends = Ends()
    engine, model = nested(
        tmp_path,
        {"delegate_to": ["desk"], "background": True},
        background("pass it on", agent="mid"),
        ends.of("mid"),
        calling(JOBS),
        text_reply("waits"),
        text_reply("noted"),
        text_reply("done"),
    )
    model.replies["pass it on"] = [
        background("refund it", agent="desk"),
        ends.of("desk"),
        calling(JOBS),
        text_reply("mid waits"),
        *after,
    ]
    return engine, model, ends


async def test_a_resumed_lead_whose_reply_called_no_tool_waits_for_its_helpers_notice(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    engine, model, ends = paused_mid(tmp_path, text_reply("mid done"))
    first = await lead_run(engine, store, refunds, approve=later_for("refund"), on_event=ends)
    [w] = first.paused
    assert mark_of(store.held[("acme", "c1/d1")]).held == []
    again = await lead_run(engine, store, refunds, "go on", answers={w.id: True})
    assert again.error is None and again.paused == [] and refunds.done == [{"amount": 10}]
    sent = model.calls["pass it on"]
    assert all(call[-1].role != "assistant" for call in sent)
    assert len(sent) == 4 and [m.role for m in sent[-1][-2:]] == ["assistant", "user"]
    assert str(sent[-1][-1].content).startswith("[helper d1 · desk · finished]")
    assert notices_of(store, "d1", ("acme", "c1/d1")) == ["paused", "finished"]
    assert notices_of(store, "d1")[-1] == "finished"


@pytest.mark.parametrize("stopped", [False, True], ids=["going on", "stopped now"])
async def test_a_resumed_lead_whose_reply_called_no_tool_ends_with_it_when_nothing_of_its_runs(
    tmp_path: Path, stopped: bool
) -> None:
    store, refunds = MemoryStore(), Refunds()
    engine, model, ends = paused_mid(tmp_path)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"), on_event=ends)
    [w] = first.paused
    mid_key = ("acme", "c1/d1")
    held = store.held[mid_key]
    [n] = [n for n, m in enumerate(held) if m.notice is not None and m.notice.kind == "paused"]
    notice = held[n].notice
    assert notice is not None
    held[n] = held[n].model_copy(
        update={
            "content": "[helper d1 · desk · finished]\nrefunded",
            "notice": notice.model_copy(update={"kind": "finished"}),
        }
    )
    calls, stop = len(model.calls["pass it on"]), Stop()
    if stopped:
        stop.now()
    again = await lead_run(engine, store, refunds, "go on", answers={w.id: True}, stop=stop)
    assert len(model.calls["pass it on"]) == calls and refunds.done == []
    assert again.error is None and again.paused == []
    assert (again.stopped, again.reply) == (("now", "") if stopped else (None, "done"))
    [mid] = [c for c in again.children if isinstance(c, Generation) and c.agent == "mid"]
    ended = "now" if stopped else None
    assert (mid.reply, mid.calls, mid.paused, mid.stopped) == ("mid waits", 0, [], ended)
    assert notices_of(store, "d1")[-1] == ("stopped" if stopped else "finished")
    assert live(store.held[mid_key]) is None


async def test_a_reattached_helper_that_fails_before_its_model_call_stops_its_helpers(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    ends = Ends()
    mid_ended, desk_ended = ends.of("mid"), ends.of("desk")

    engine, model = nested(
        tmp_path,
        {"delegate_to": ["desk"], "background": True},
        background("pass it on", agent="mid"),
        mid_ended,
        calling(JOBS),
        text_reply("waits"),
        text_reply("noted"),
        text_reply("done"),
    )
    model.replies["pass it on"] = [
        background("refund it", agent="desk"),
        desk_ended,
        calling(JOBS),
        text_reply("mid waits"),
    ]
    first = await lead_run(engine, store, refunds, approve=later_for("refund"), on_event=ends)
    [w] = first.paused
    again = await lead_run(
        engine, store, refunds, "go on", answers={w.id: True}, on_event=fails_on_start("mid")
    )
    desk_key, mid_key = ("acme", "c1/d1/d1"), ("acme", "c1/d1")
    assert answer_in(store, desk_key, "c-refund").content == "stopped by the user"
    assert live(store.held[desk_key]) is None and live(store.held[mid_key]) is None
    assert notices_of(store, "d1", mid_key) == ["paused", "stopped"]
    assert notices_of(store, "d1") == ["paused", "failed"]
    assert again.error is None and again.reply == "done" and refunds.done == []


async def test_a_resumed_child_that_fails_before_its_model_call_stops_its_helpers(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    ends = Ends()
    desk_ended = ends.of("desk")
    engine, model = nested(
        tmp_path,
        {"delegate_to": ["desk"], "background": True},
        delegating("mid", "pass it on"),
        text_reply("noted"),
    )
    model.replies["pass it on"] = [
        background("refund it", agent="desk"),
        desk_ended,
        calling(JOBS),
        text_reply("mid waits"),
    ]
    first = await lead_run(engine, store, refunds, approve=later_for("refund"), on_event=ends)
    [w] = first.paused
    mid_key = ("acme", str(mark_of(store.held[KEY]).delegated["d1"]))
    desk_key = (mid_key[0], f"{mid_key[1]}/d1")
    again = await lead_run(
        engine, store, refunds, "go on", answers={w.id: True}, on_event=fails_on_start("mid")
    )
    assert answer_in(store, desk_key, "c-refund").content == "stopped by the user"
    assert live(store.held[desk_key]) is None and live(store.held[mid_key]) is None
    assert notices_of(store, "d1", mid_key) == ["paused", "stopped"]
    [delegated] = [s for s in again.steps if s.tool == "Delegate"]
    assert delegated.result == "RuntimeError: mid's handler broke" and delegated.is_error
    assert again.error is None and again.reply == "noted" and refunds.done == []


async def test_a_reattached_helper_that_fails_before_its_model_call_stops_its_children(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    ends = Ends()
    mid_ended = ends.of("mid")
    engine, model = nested(
        tmp_path,
        {"delegate_to": ["desk"]},
        background("pass it on", agent="mid"),
        mid_ended,
        calling(JOBS),
        text_reply("waits"),
        text_reply("noted"),
        text_reply("done"),
    )
    model.replies["pass it on"] = [delegating("desk", "refund it"), text_reply("never")]
    first = await lead_run(engine, store, refunds, approve=later_for("refund"), on_event=ends)
    [w] = first.paused
    [desk_key] = [k for k in store.held if k[1].count("/") == 2]
    again = await lead_run(
        engine, store, refunds, "go on", answers={w.id: True}, on_event=fails_on_start("mid")
    )
    assert answer_in(store, desk_key, "c-refund").content == "stopped by the user"
    assert live(store.held[desk_key]) is None and live(store.held[HELPER_KEY]) is None
    delegated = answer_in(store, HELPER_KEY, "d1")
    assert delegated.content == "NotRun: Delegate was not run; the run failed"
    assert notices_of(store, "d1") == ["paused", "failed"]
    assert again.error is None and again.reply == "done" and refunds.done == []


def breaks_on(call_id: str) -> Any:
    async def breaks(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == call_id:
            raise RuntimeError("the handler broke")

    return breaks


async def test_a_resume_step_that_fails_after_its_delegate_returned_answers_it_once(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    engine, first, _ = await paused_split(
        tmp_path, store, refunds, text_reply("going"), text_reply("all done")
    )
    sync, waiting = first.paused
    [held] = mark_of(store.held[KEY]).held
    with pytest.raises(RuntimeError, match="the handler broke"):
        await lead_run(
            engine, store, refunds, "go on", answers={sync.id: True}, on_event=breaks_on("s1")
        )
    mark = live(store.held[KEY])
    assert mark is not None and (mark.waiting, mark.delegated, mark.below) == ([], {}, [waiting])
    assert mark.held == [held] and answer_in(store, KEY, "s1").content == "it done"
    again = await lead_run(engine, store, refunds, "carry on", answers={})
    assert again.error is None and again.resumed and again.paused == [waiting]
    assert again.calls == 0 and answer_in(store, KEY, "s1").content == "it done"
    last = await lead_run(engine, store, refunds, "go on", answers={waiting.id: True})
    assert last.error is None and last.paused == [] and last.reply == "all done"
    assert answer_in(store, KEY, "s1").content == "it done"
    assert sorted(done["amount"] for done in refunds.done) == [5, 10]
    assert notices_of(store, "d1") == ["paused", "finished"]


async def test_a_resume_step_that_fails_once_every_call_returned_leaves_a_plain_conversation(
    tmp_path: Path,
) -> None:
    store, refunds = MemoryStore(), Refunds()
    model = Crew(
        [both(), text_reply("lead done")],
        {"refund it": [calling(REFUND), text_reply("desk done")]},
    )
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    own, below = first.paused
    with pytest.raises(RuntimeError, match="the handler broke"):
        await lead_run(
            engine,
            store,
            refunds,
            "go on",
            answers={own.id: True, below.id: True},
            on_event=breaks_on("d1"),
        )
    held = store.held[KEY]
    assert live(held) is None and held[-1].tool_call_id == "d1"
    assert answer_in(store, KEY, "d1").content == "desk done"
    again = await lead_run(engine, store, refunds, "carry on")
    assert again.error is None and not again.resumed and again.reply == "lead done"
    assert answer_in(store, KEY, "d1").content == "desk done"
    assert answer_in(store, KEY, "c-own").content == "refunded"
    assert len(refunds.done) == 2


async def test_a_resume_after_a_cut_one_that_fails_once_its_delegate_returned_answers_it_once(
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
    cut = asyncio.create_task(
        lead_run(engine, store, refunds, "go on", answers={own.id: True, below.id: True})
    )
    await until(lambda: refunds.started == 2, "both approved refunds started")
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    with pytest.raises(RuntimeError, match="the handler broke"):
        await lead_run(engine, store, refunds, "go on", answers={}, on_event=breaks_on("d1"))
    assert live(store.held[KEY]) is None
    assert answer_in(store, KEY, "d1").content == "it was cut off"
    again = await lead_run(engine, store, refunds, "carry on")
    assert again.error is None and not again.resumed
    assert again.reply == "it may have gone through"
    assert answer_in(store, KEY, "d1").content == "it was cut off"
    assert refunds.started == 2


async def breaks_once_the_lead_ends(event: Any) -> None:
    if isinstance(event, RunEnded) and event.call is None:
        raise RuntimeError("the handler broke")


@pytest.mark.parametrize(
    "breaks",
    [breaks_on("c-own"), breaks_once_the_lead_ends],
    ids=["on-a-result", "once-the-lead-ends"],
)
async def test_a_child_that_paused_again_stays_listed_when_its_resume_fails(
    breaks: Any, tmp_path: Path
) -> None:
    store, refunds = MemoryStore(), Refunds()
    model = Crew(
        [both(), text_reply("lead done")],
        {"refund it": [calling(REFUND), calling(SECOND), text_reply("desk done")]},
    )
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    own, below = first.paused
    with pytest.raises(RuntimeError, match="the handler broke"):
        await lead_run(
            engine,
            store,
            refunds,
            "go on",
            answers={own.id: True, below.id: True},
            on_event=breaks,
            approve=later_for("refund"),
        )
    [desk_key] = [key for key in store.held if key != KEY]
    desk, mark = live(store.held[desk_key]), live(store.held[KEY])
    assert desk is not None and mark is not None
    assert mark.delegated == {"d1": desk_key[1]} and mark.below == desk.waiting
    [again] = mark.below
    assert again.call.id == SECOND.id and again.id != below.id
    assert not any(m.tool_call_id == "d1" for m in store.held[KEY])
    assert answer_in(store, KEY, "c-own").content == "refunded"
    last = await lead_run(engine, store, refunds, "go on", answers={again.id: True})
    assert last.error is None and last.paused == [] and last.reply == "lead done"
    assert answer_in(store, KEY, "d1").content == "desk done"
    assert [done["amount"] for done in refunds.done] == [3, 10, 5]


class Stuck(Looked):
    def tools(self) -> Tools:
        async def lookup(args: dict[str, Any]) -> str:
            self.times += 1
            await asyncio.Event().wait()
            return "found"

        return Tools([Tool("lookup", "Look an order up", body=lookup), self.refunds.tools()])


async def test_a_failed_resume_step_lists_no_call_of_a_helper_it_stopped(tmp_path: Path) -> None:
    store, looked = MemoryStore(), Stuck()
    second = LOOKUP.model_copy(update={"id": "c-lookup-2"})
    both = later_for("lookup", "refund")
    model = Crew(
        [beside_its_own("part one"), text_reply("going")],
        {"part one": [calling(LOOKUP, second), text_reply("never")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, approve=both)
    own, lookup, _ = first.paused
    with pytest.raises(RuntimeError, match="the handler broke"):
        await team_run(
            engine,
            store,
            looked,
            "go on",
            answers={own.id: True, lookup.id: True},
            on_event=breaks_on("c-own"),
            approve=both,
        )
    assert live(store.held[KEY]) is None and live(store.held[HELPER_KEY]) is None
    assert notices_of(store, "d1") == ["paused", "stopped"]
    assert answer_in(store, HELPER_KEY, "c-lookup-2").content == "stopped by the user"
    again = await team_run(engine, store, looked, "carry on", approve=both)
    assert again.error is None and not again.resumed and again.reply == "going"
    assert not any(m.notice and m.notice.kind == "lost" for m in store.held[KEY])


async def test_a_resume_cut_before_it_goes_on_keeps_its_results_and_the_notices_its_mark_held(
    tmp_path: Path,
) -> None:
    store, looked = MemoryStore(), Looked()
    both = later_for("lookup", "refund")
    model = Crew(
        [beside_its_own("part one"), text_reply("going"), text_reply("all done")],
        {"part one": [calling(LOOKUP), text_reply("part one done")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, approve=both)
    own, lookup = first.paused
    [held] = mark_of(store.held[KEY]).held
    taking = asyncio.Event()

    async def hangs(event: Any) -> None:
        if isinstance(event, MessagesTaken) and event.agent == "lead":
            taking.set()
            await asyncio.Event().wait()

    cut = asyncio.create_task(
        team_run(engine, store, looked, "go on", answers={own.id: True}, on_event=hangs)
    )
    await asyncio.wait_for(taking.wait(), 5)
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    assert len(looked.refunds.done) == 1
    mark = live(store.held[KEY])
    assert mark is not None and mark.held == [held] and mark.below == [lookup]
    assert answer_in(store, KEY, "c-own").content == "refunded"
    again = await team_run(engine, store, looked, "carry on", answers={}, approve=both)
    assert again.resumed and again.paused == [lookup] and again.calls == 0
    assert answer_in(store, KEY, "c-own").content == "refunded"
    assert not any(m.notice and m.notice.kind == "lost" for m in store.held[KEY])
    last = await team_run(engine, store, looked, "go on", answers={lookup.id: True})
    assert last.error is None and last.reply == "all done"
    assert notices_of(store, "d1") == ["paused", "finished"] and len(looked.refunds.done) == 1


async def test_a_resume_cut_while_it_waits_to_pause_again_keeps_its_results(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    third = ToolCall(id="c-refund-3", name="refund", args={"amount": 7})
    calls = [
        ToolCall(
            id="b1",
            name="Delegate",
            args={"agent": "desk", "input": "refund later", "background": True},
        ),
        ToolCall(id="s1", name="Delegate", args={"agent": "desk", "input": "refund it"}),
        OWN.model_copy(),
    ]
    model = Crew(
        [
            Reply(tool_calls=calls, stop_reason="tool_use"),
            text_reply("all done"),
            text_reply("all done"),
        ],
        {
            "refund later": [calling(REFUND), gate, text_reply("later done")],
            "refund it": [calling(SECOND), calling(third), text_reply("it done")],
        },
    )
    engine = engine_of(tmp_path, model, background=True)
    store, refunds = MemoryStore(), Refunds()
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    [held] = mark_of(store.held[KEY]).held
    cut = asyncio.create_task(
        lead_run(
            engine,
            store,
            refunds,
            "go on",
            answers={w.id: True for w in first.paused},
            approve=later_for("refund"),
        )
    )
    await until(lambda: len(refunds.done) == 3, "the three approved refunds ran")
    await until(lambda: any(m.tool_call_id == "c-own" for m in store.held[KEY]), "results stored")
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    assert answer_in(store, KEY, "c-own").content == "refunded"
    mark = live(store.held[KEY])
    assert mark is not None and list(mark.delegated) == ["s1"] and mark.held == [held]
    [waits] = mark.below
    assert waits.call.id == third.id and waits.conversation == mark.delegated["s1"]
    gate.set()
    again = await lead_run(
        engine, store, refunds, "carry on", answers={waits.id: True}, approve=later_for("refund")
    )
    assert again.error is None and again.paused == [] and again.reply == "all done"
    assert answer_in(store, KEY, "c-own").content == "refunded"
    assert answer_in(store, KEY, "s1").content == "it done"
    assert sorted(done["amount"] for done in refunds.done) == [3, 5, 7, 10]
    assert notices_of(store, "d1") == ["paused", "finished"]


async def test_a_resume_step_keeps_listing_the_calls_of_a_helper_it_set_running(
    tmp_path: Path,
) -> None:
    store, looked, gate = MemoryStore(), Stuck(), asyncio.Event()
    second = LOOKUP.model_copy(update={"id": "c-lookup-2"})
    both = later_for("lookup", "refund")
    model = Crew(
        [beside_its_own("part one"), gate, text_reply("waiting")],
        {"part one": [calling(LOOKUP, second), text_reply("never")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, approve=both)
    own, lookup, still = first.paused
    cut = asyncio.create_task(
        team_run(
            engine, store, looked, "go on", answers={own.id: True, lookup.id: True}, approve=both
        )
    )
    await until(
        lambda: looked.times == 1 and store.held[KEY][-1].content == "go on", "the lead went on"
    )
    held = store.held[KEY]
    after = held[held.index(answer_in(store, KEY, "c-own")) + 1].pause
    assert after is not None and after.below == [still] and kinds(after.held) == ["paused"]
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    again = await team_run(engine, store, looked, "carry on", approve=both)
    assert again.error is None and not again.resumed and again.paused == [still]
    assert notices_of(store, "d1") == ["paused"] and looked.times == 1
