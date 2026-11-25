import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Later
from rfnry_engine.conversation import live, open_calls
from rfnry_engine.models import Message, ToolCall
from rfnry_engine.records import Generation, ToolCallStarted

from .fakes import Crew, MemoryStore, background, crew, delegating, team, text_reply
from .test_approve import Refunds
from .test_paused import KEY, LOOKUP, REFUND, calling, later_for, pausing, stored
from .test_paused_tree import (
    ASK,
    HELPER_KEY,
    OWN,
    SECOND,
    Looked,
    beside_its_own,
    both,
    breaks_on,
    breaks_once_the_lead_ends,
    crew_run,
    engine_of,
    helped,
    lead_run,
    mark_of,
    team_run,
)


def unanswered(conversation: Sequence[Message]) -> frozenset[str]:
    return frozenset(
        call.id
        for n, message in enumerate(conversation)
        for call in message.tool_calls
        if not any(
            later.role == "tool" and later.tool_call_id == call.id
            for later in conversation[n + 1 :]
        )
    )


def marked(conversation: Sequence[Message]) -> set[str]:
    left = unanswered(conversation)
    assert left == open_calls(live(conversation))
    return set(left)


def checked(held: Mapping[str, Sequence[Message]]) -> dict[str, set[str]]:
    found = {key: marked(conversation) for key, conversation in held.items()}
    marks = {key: live(conversation) for key, conversation in held.items()}
    for key, mark in marks.items():
        below = set() if mark is None else {(w.conversation, w.call.id, w.id) for w in mark.below}
        under = {
            (other, w.call.id, w.id)
            for other, theirs in marks.items()
            if theirs is not None and other.startswith(f"{key}/")
            for w in theirs.waiting
        }
        assert below == under, key
    return found


def every(store: MemoryStore) -> dict[str, set[str]]:
    return checked({key: held for (_, key), held in store.held.items()})


@pytest.mark.parametrize("with_store", [False, True], ids=["history", "store"])
async def test_a_call_answered_later_is_the_one_call_its_mark_names(
    tmp_path: Path, with_store: bool
) -> None:
    store = MemoryStore() if with_store else None
    _, run, _, _ = await pausing(
        tmp_path, calling(REFUND, LOOKUP), text_reply("never"), **stored(store)
    )
    assert isinstance(run, Generation) and [w.call.id for w in run.paused] == ["c-refund"]
    held = checked({"c1": run.conversation}) if store is None else every(store)
    assert held == {"c1": {"c-refund"}}


async def test_a_resume_that_answers_the_lead_s_call_leaves_a_mark_over_its_helper_s_call(
    tmp_path: Path,
) -> None:
    store, looked = MemoryStore(), Looked()
    model = Crew(
        [beside_its_own("part one"), text_reply("going")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, approve=later_for("lookup", "refund"))
    own, lookup = first.paused
    again = await team_run(engine, store, looked, "go on", answers={own.id: True})
    assert again.resumed and again.paused == [lookup]
    assert looked.refunds.done == [{"amount": 3}] and looked.times == 0
    assert mark_of(store.held[KEY]).below == [lookup]
    assert every(store) == {"c1": set(), "c1/d1": {"c-lookup"}}


async def test_a_child_that_pauses_leaves_its_parent_a_mark_naming_its_delegate(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    model = Crew(
        [delegating("desk", "refund it"), text_reply("never")],
        {"refund it": [calling(REFUND), text_reply("never")]},
    )
    lead = await lead_run(engine_of(tmp_path, model), store, Refunds(), approve=later_for("refund"))
    [child] = lead.children
    assert every(store) == {"c1": {"d1"}, f"c1/{child.run_id}": {"c-refund"}}


async def test_a_helper_paused_on_its_ask_names_the_ask_beside_its_lead_s_own_call(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    model = Crew(
        [background("part one"), text_reply("waiting"), calling(OWN), text_reply("never")],
        {"part one": [calling(ASK), text_reply("never")]},
    )
    lead = await crew_run(helped(tmp_path, model), store, approve=later_for("refund"))
    assert [w.call for w in lead.paused] == [OWN]
    assert mark_of(store.held[HELPER_KEY]).asking == "a1"
    assert every(store) == {"c1": {"c-own"}, "c1/d1": {"a1"}}


async def test_a_paused_helper_and_its_paused_lead_each_name_their_own_calls(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    model = Crew(
        [beside_its_own("part one"), text_reply("never")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    lead = await team_run(
        helped(tmp_path, model), store, Looked(), approve=later_for("lookup", "refund")
    )
    assert [w.call.id for w in lead.paused] == ["c-own", "c-lookup"]
    assert every(store) == {"c1": {"c-own"}, "c1/d1": {"c-lookup"}}


async def test_a_lead_paused_only_by_its_helper_names_no_call_of_its_own(tmp_path: Path) -> None:
    store = MemoryStore()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("it waits")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    lead = await crew_run(helped(tmp_path, model), store)
    assert [w.call.id for w in lead.paused] == ["c-lookup"]
    assert every(store) == {"c1": set(), "c1/d1": {"c-lookup"}}


@pytest.mark.parametrize(
    "breaks",
    [breaks_on("c-own"), breaks_once_the_lead_ends],
    ids=["in-its-answers", "once-paused-again"],
)
async def test_a_resume_that_fails_before_its_model_call_keeps_a_mark_naming_its_delegate(
    tmp_path: Path, breaks: Any
) -> None:
    store, refunds = MemoryStore(), Refunds()
    model = Crew(
        [both(), text_reply("lead done")],
        {"refund it": [calling(REFUND), calling(SECOND), text_reply("desk done")]},
    )
    engine = engine_of(tmp_path, model)
    first = await lead_run(engine, store, refunds, approve=later_for("refund"))
    own, below = first.paused
    desk = f"c1/{first.children[0].run_id}"
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
    assert len(model.calls["lead"]) == 1 and [done["amount"] for done in refunds.done] == [3, 10]
    assert every(store) == {"c1": {"d1"}, desk: {"c-refund-2"}}


async def test_a_resume_that_fails_beside_a_paused_helper_keeps_a_mark_naming_nothing_of_its_own(
    tmp_path: Path,
) -> None:
    store, looked = MemoryStore(), Looked()
    model = Crew(
        [beside_its_own("part one"), text_reply("never")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    engine = helped(tmp_path, model)
    first = await team_run(engine, store, looked, approve=later_for("lookup", "refund"))
    own, _ = first.paused
    with pytest.raises(RuntimeError, match="the handler broke"):
        await team_run(
            engine, store, looked, "go on", answers={own.id: True}, on_event=breaks_on("c-own")
        )
    assert looked.refunds.done == [{"amount": 3}]
    assert every(store) == {"c1": set(), "c1/d1": {"c-lookup"}}


async def test_a_run_that_fails_while_its_call_waits_ends_with_no_mark_and_nothing_open(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    with pytest.raises(RuntimeError, match="the handler broke"):
        await pausing(
            tmp_path,
            calling(REFUND, LOOKUP),
            text_reply("never"),
            on_event=breaks_on("c-lookup"),
            **stored(store),
        )
    assert live(store.held[KEY]) is None
    assert every(store) == {"c1": set()}


async def test_a_child_that_fails_while_its_call_waits_leaves_no_mark_and_nothing_open(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    model = Crew(
        [delegating("desk", "refund it"), text_reply("the desk failed")],
        {"refund it": [calling(REFUND, SECOND), text_reply("never")]},
    )

    async def approve(agent: str, call: ToolCall) -> bool | Later:
        return Later() if call.id == REFUND.id else True

    lead = await lead_run(
        engine_of(tmp_path, model),
        store,
        Refunds(),
        approve=approve,
        on_event=breaks_on(SECOND.id),
    )
    [child] = lead.children
    assert lead.error is None and child.error == "RuntimeError: the handler broke"
    desk = f"c1/{child.run_id}"
    assert store.held[("acme", desk)][-1].content == "NotRun: refund was not run; the run failed"
    assert all(live(held) is None for held in store.held.values())
    assert every(store) == {"c1": set(), desk: set()}


async def test_a_run_that_fails_during_a_batch_ends_with_no_mark_and_nothing_open(
    tmp_path: Path,
) -> None:
    store, gate = MemoryStore(), asyncio.Event()
    model = Crew(
        [background("look"), calling(ToolCall(id="j1", name="Jobs")), text_reply("done")],
        {"look": [gate, text_reply("never")]},
    )

    async def breaks(event: Any) -> None:
        if isinstance(event, ToolCallStarted) and event.tool == "Jobs":
            raise ValueError("the handler broke")

    with pytest.raises(ValueError, match="the handler broke"):
        await team(crew(tmp_path, background=True), model).run(
            "lead", "acme", "go", store=store, conversation="c1", on_event=breaks
        )
    assert all(live(held) is None for held in store.held.values())
    assert every(store) == {"c1": set(), "c1/d1": set()}
