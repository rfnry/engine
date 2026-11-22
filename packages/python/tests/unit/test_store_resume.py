import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import MessagesTaken, Notice, Refinery, RunStarted, Storage, ToolCallStarted
from rfnry_engine.engine import Engine
from rfnry_engine.helpers import lost_notices
from rfnry_engine.models import Message, Reply, ToolCall
from rfnry_engine.records import Generation

from .fakes import (
    Answering,
    Crew,
    MemoryStore,
    Scripted,
    Seen,
    background,
    crew,
    released,
    team,
    text_reply,
)
from .test_background import Talking
from .test_store import KEY, chat, engine_for, use
from .test_store_tree import switching
from .trees import provided

RESTART = "the process restarted; carry on"


def ended(call_id: str, name: str) -> Message:
    return Message(
        role="tool",
        tool_call_id=call_id,
        content=f"Ended: {name} did not return; the run that made the call ended, and what it did is unknown",
        is_error=True,
    )


def lost(job: str, agent: str = "worker", kept: bool = True) -> Message:
    body = (
        "the run that started it ended before its outcome reached you, and what it did is unknown"
    )
    return Message(
        role="user",
        content=f"[helper {job} · {agent} · lost]\n{body}"
        + (f"; its conversation is c1/{job}" if kept else ""),
        notice=Notice(job=job, agent=agent, run_id=None, kind="lost"),
    )


def outcome(job: str, kind: Any, body: str) -> Message:
    return Message(
        role="user",
        content=f"[helper {job} · worker · {kind}]\n{body}",
        notice=Notice(job=job, agent="worker", run_id=f"r{job}", kind=kind),
    )


def in_place(messages: list[Message]) -> bool:
    return all(
        [m.tool_call_id for m in messages[n + 1 : n + 1 + len(message.tool_calls)]]
        == [call.id for call in message.tool_calls]
        for n, message in enumerate(messages)
        if message.tool_calls
    )


def sent(messages: list[Message]) -> list[Message]:
    return [message.model_copy(update={"notice": None}) for message in messages]


def started(call_id: str, text: str, agent: str = "worker") -> ToolCall:
    args = {"agent": agent, "input": text, "background": True}
    return ToolCall(id=call_id, name="Delegate", args=args)


def job(call_id: str, held: str) -> Message:
    return Message(role="tool", tool_call_id=call_id, content=json.dumps({"job": held}))


def handed_out() -> list[Message]:
    return [
        Message(role="user", content="go"),
        Message(
            role="assistant", tool_calls=[started("x1", "part one"), started("x2", "part two")]
        ),
        Message(role="tool", tool_call_id="x1", content=json.dumps({"job": "d1"})),
        Message(role="tool", tool_call_id="x2", content=json.dumps({"job": "d2"})),
        Message(role="assistant", content="waiting"),
        outcome("d1", "finished", "one"),
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
        outcome("d3", "asks", "which one?"),
    ]
    assert lost_notices(history, "c1", frozenset()) == [lost("d3")]


async def test_a_job_whose_stopped_or_failed_outcome_is_stored_is_not_lost(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    store.held[KEY] = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            tool_calls=[started(f"x{n}", f"part {n}") for n in (1, 2, 3)],
        ),
        *[job(f"x{n}", f"d{n}") for n in (1, 2, 3)],
        Message(role="assistant", content="waiting"),
        outcome("d1", "stopped", "half way"),
        outcome("d2", "failed", "OSError: disk full"),
        Message(role="assistant", content="noted"),
    ]
    await resumed(team(crew(tmp_path, background=True), Crew([text_reply("done")])), store)
    assert told_lost(store) == [lost("d3")]


async def test_jobs_before_the_window_without_an_outcome_are_told_lost_in_job_order(
    tmp_path: Path,
) -> None:
    store = MemoryStore(lambda conversation, length: 7 if conversation == "c1" else 0)
    store.held[KEY] = [
        Message(role="user", content="earlier"),
        Message(role="assistant", tool_calls=[started("x1", "a"), started("x2", "b")]),
        job("x1", "d2"),
        job("x2", "d1"),
        Message(role="assistant", tool_calls=[started("x3", "c")]),
        job("x3", "d3"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="later"),
        Message(role="assistant", content="noted"),
    ]
    model = Crew([text_reply("done")])
    await resumed(team(crew(tmp_path, background=True), model), store)
    told = [lost("d1"), lost("d2"), lost("d3")]
    assert told_lost(store) == told
    assert [message.content for message in model.calls["lead"][0]] == [
        "later",
        "noted",
        *[message.content for message in told],
        RESTART,
    ]


async def test_a_lost_helper_that_keeps_no_conversation_is_told_without_a_name(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    store.held[KEY] = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            tool_calls=[
                started("x1", "sort", agent="desk"),
                started("x2", "route", agent="switch"),
                started("x3", "old", agent="gone"),
            ],
        ),
        *[job(f"x{n}", f"d{n}") for n in (1, 2, 3)],
        Message(role="assistant", content="waiting"),
    ]
    model = Crew([text_reply("done")])
    providers = provided({"crew": model, "judge": Answering(), "switch": Talking()})
    await resumed(Engine(switching(tmp_path), providers=providers), store)
    assert told_lost(store) == [
        lost("d1", "desk", kept=False),
        lost("d2", "switch"),
        lost("d3", "gone"),
    ]


async def test_a_run_that_fails_during_a_batch_ends_its_calls_before_its_helpers_notices(
    tmp_path: Path,
) -> None:
    store, gate = MemoryStore(), asyncio.Event()
    listing = Reply(tool_calls=[ToolCall(id="j1", name="Jobs")], stop_reason="tool_use")
    model = Crew(
        [background("look"), listing, text_reply("done")],
        {"look": [gate, text_reply("never")]},
    )
    engine = team(crew(tmp_path, background=True), model)

    async def breaks(event: object) -> None:
        if isinstance(event, ToolCallStarted) and event.tool == "Jobs":
            raise ValueError("handler broke")

    with pytest.raises(ValueError, match="^handler broke$"):
        await engine.run("lead", "acme", "go", store=store, conversation="c1", on_event=breaks)
    reply, result, notice = store.held[KEY][-3:]
    assert [call.id for call in reply.tool_calls] == ["j1"] and result == ended("j1", "Jobs")
    assert notice.notice is not None and (notice.notice.job, notice.notice.kind) == (
        "d1",
        "stopped",
    )
    assert in_place(store.held[KEY])
    await resumed(engine, store)
    first = model.calls["lead"][2]
    assert in_place(first) and told_lost(store) == []
    assert first[-2:] == sent([notice, Message(role="user", content=RESTART)])
