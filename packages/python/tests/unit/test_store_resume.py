import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import MessagesTaken, Notice, Refinery, RunStarted, Storage
from rfnry_engine.engine import Engine
from rfnry_engine.helpers import lost_notices
from rfnry_engine.models import Message, ToolCall
from rfnry_engine.records import Generation

from .fakes import Crew, MemoryStore, Scripted, Seen, background, crew, released, team, text_reply
from .test_store import KEY, chat, engine_for, use

RESTART = "the process restarted; carry on"


def ended(call_id: str, name: str) -> Message:
    return Message(
        role="tool",
        tool_call_id=call_id,
        content=f"Ended: {name} did not return; the process that ran it ended, and what it did is unknown",
        is_error=True,
    )


def lost(job: str) -> Message:
    return Message(
        role="user",
        content=f"[helper {job} · worker · lost]\nthe process that ran it ended before it did; its conversation is c1/{job}",
        notice=Notice(job=job, agent="worker", run_id=None, kind="lost"),
    )


def sent(messages: list[Message]) -> list[Message]:
    return [message.model_copy(update={"notice": None}) for message in messages]


def started(call_id: str, text: str) -> ToolCall:
    args = {"agent": "worker", "input": text, "background": True}
    return ToolCall(id=call_id, name="Delegate", args=args)


def handed_out() -> list[Message]:
    return [
        Message(role="user", content="go"),
        Message(
            role="assistant", tool_calls=[started("x1", "part one"), started("x2", "part two")]
        ),
        Message(role="tool", tool_call_id="x1", content=json.dumps({"job": "d1"})),
        Message(role="tool", tool_call_id="x2", content=json.dumps({"job": "d2"})),
        Message(role="assistant", content="waiting"),
        Message(
            role="user",
            content="[helper d1 · worker · finished]\none",
            notice=Notice(job="d1", agent="worker", run_id="r1", kind="finished"),
        ),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="w1", name="Job", args={"id": "d2", "wait": 30})],
        ),
    ]


def told_lost(store: MemoryStore) -> list[Message]:
    return [
        message
        for message in store.held[KEY]
        if message.notice is not None and message.notice.kind == "lost"
    ]


async def resumed(
    engine: Engine, store: MemoryStore, input: str = RESTART, **options: Any
) -> Generation:
    run = await engine.run("lead", "acme", input, store=store, conversation="c1", **options)
    assert isinstance(run, Generation)
    return run


async def test_a_resumed_conversation_ends_the_calls_a_crash_cut_off(tmp_path: Path) -> None:
    store, llm = MemoryStore(), Scripted(text_reply("Carrying on."))
    calls = [ToolCall(id="a", name="slow"), ToolCall(id="b", name="lookup", args={"order": 7})]
    stored = [
        Message(role="user", content="work"),
        Message(role="assistant", tool_calls=calls),
        Message(role="tool", tool_call_id="a", content="done"),
    ]
    store.held[KEY] = list(stored)
    run = await chat(engine_for(tmp_path, llm), store, RESTART)
    asked = Message(role="user", content=RESTART)
    assert llm.calls[0][1][-3:] == [stored[2], ended("b", "lookup"), asked]
    assert store.held[KEY][:5] == [*stored, ended("b", "lookup"), asked]
    assert run.reply == "Carrying on." and store.held[KEY] == run.conversation


async def test_a_job_without_an_outcome_is_told_to_the_lead_as_lost(tmp_path: Path) -> None:
    store, seen = MemoryStore(), Seen()
    store.held[KEY] = handed_out()
    model = Crew([text_reply("done")])
    run = await resumed(team(crew(tmp_path, background=True), model), store, on_event=seen)
    asked = Message(role="user", content=RESTART)
    assert told_lost(store) == [lost("d2")]
    assert store.held[KEY][7:10] == [ended("w1", "Job"), lost("d2"), asked]
    assert model.calls["lead"][0][-3:] == sent([ended("w1", "Job"), lost("d2"), asked])
    assert [type(event) for event in seen.events[:2]] == [RunStarted, MessagesTaken]
    [event] = [e for e in seen.events if isinstance(e, MessagesTaken)]
    assert (event.run_id, event.turn, event.messages, event.notices) == (
        run.run_id,
        0,
        [],
        [lost("d2")],
    )
    assert run.messages[-3:] == [ended("w1", "Job"), lost("d2"), asked]


async def test_a_job_already_told_lost_is_not_told_again(tmp_path: Path) -> None:
    store, seen = MemoryStore(), Seen()
    store.held[KEY] = handed_out()
    model = Crew([text_reply("done"), text_reply("still done")])
    engine = team(crew(tmp_path, background=True), model)
    await resumed(engine, store)
    again = await resumed(engine, store, "and now?", on_event=seen)
    assert told_lost(store) == [lost("d2")]
    assert [e for e in seen.events if isinstance(e, MessagesTaken)] == []
    last, asked = model.calls["lead"][1][-2:]
    assert (last.role, last.content) == ("assistant", "done")
    assert asked == Message(role="user", content="and now?")
    assert again.messages[-1] == asked and again.reply == "still done"


async def test_a_resumed_lead_never_writes_into_a_cut_off_helper_s_conversation(
    tmp_path: Path,
) -> None:
    store, seen = MemoryStore(), Seen()
    store.held[KEY] = [
        Message(role="user", content="go"),
        Message(role="assistant", tool_calls=[started("x1", "part one")]),
    ]
    helper = [Message(role="user", content="part one"), Message(role="assistant", content="half")]
    store.held[("acme", "c1/d1")] = list(helper)
    gate = asyncio.Event()
    model = Crew(
        [background("part two", prefix="n"), text_reply("waiting"), text_reply("final")],
        {"part two": [gate, text_reply("the answer")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(resumed(engine, store, on_event=seen))
    run = await released(work, model, gate)
    asked = Message(role="user", content=RESTART)
    assert model.calls["lead"][0][-2:] == [ended("x1", "Delegate"), asked]
    assert told_lost(store) == []
    assert [
        e.turn for e in seen.events if isinstance(e, MessagesTaken) and e.run_id == run.run_id
    ] == [2]
    assert run.steps[0].result == '{"job": "d2"}'
    assert store.held[("acme", "c1/d1")] == helper


async def test_a_run_cancelled_during_its_tools_resumes_with_their_calls_ended(
    tmp_path: Path,
) -> None:
    store, entered, gate = MemoryStore(), asyncio.Event(), asyncio.Event()

    async def waiting(args: dict[str, Any]) -> str:
        entered.set()
        await gate.wait()
        return "done"

    llm = Scripted(use(), text_reply("Carrying on."))
    engine = engine_for(tmp_path, llm)
    work = asyncio.create_task(chat(engine, store, body=waiting))
    await asyncio.wait_for(entered.wait(), 5)
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    reply = store.held[KEY][-1]
    assert [message.role for message in store.held[KEY]] == ["user", "assistant"]
    assert reply.tool_calls == [ToolCall(id="c1", name="slow")]
    assert engine._conversations == set()
    run = await chat(engine, store, RESTART)
    asked = Message(role="user", content=RESTART)
    assert llm.calls[1][1][-2:] == [ended("c1", "slow"), asked]
    assert store.held[KEY][2:4] == [ended("c1", "slow"), asked]
    assert run.reply == "Carrying on."


async def test_a_lost_notice_reads_back_from_the_record(tmp_path: Path, storage: Storage) -> None:
    store = MemoryStore()
    store.held[KEY] = handed_out()
    refinery = Refinery(storage)
    model = Crew([text_reply("done")])
    run = await resumed(team(crew(tmp_path, background=True), model, refinery=refinery), store)
    assert lost("d2") in run.conversation
    read = await refinery.run("acme", run.run_id)
    assert isinstance(read, Generation) and read.conversation == run.conversation


def test_only_a_background_delegate_s_job_without_an_outcome_is_lost() -> None:
    synchronous = ToolCall(id="s1", name="Delegate", args={"agent": "worker", "input": "a"})
    history = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            tool_calls=[synchronous, started("x1", "part one"), started("x2", "part two")],
        ),
        Message(role="tool", tool_call_id="s1", content=json.dumps({"job": "d1"})),
        Message(role="tool", tool_call_id="x1", content=json.dumps({"job": "d2"}), is_error=True),
        Message(role="tool", tool_call_id="x2", content=json.dumps({"job": "d3"})),
        Message(
            role="user",
            content="[helper d3 · worker · asks]\nwhich one?",
            notice=Notice(job="d3", agent="worker", run_id="r3", kind="asks"),
        ),
    ]
    assert lost_notices(history, "c1") == [lost("d3")]
