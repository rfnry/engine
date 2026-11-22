import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

import pytest

from rfnry_engine import Stop, ToolCallStarted
from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import Message, Reply, ToolCall
from rfnry_engine.records import Decision, Generation

from .fakes import (
    URGENT,
    Answering,
    Crew,
    MemoryStore,
    Scripted,
    background,
    crew,
    delegating,
    said,
    team,
    text_reply,
    until,
)
from .test_background import INTENT, Talking, pending
from .test_failed_runs import Kept
from .trees import provided, write_folder

KEY = ("acme", "c1")


def at(conversation: str) -> tuple[str, str]:
    return ("acme", conversation)


def keys(store: MemoryStore) -> set[str]:
    return {conversation for _, _, conversation in store.log}


def jobs(run: Generation) -> dict[str, Generation | Decision]:
    children = {child.run_id: child for child in run.children}
    return {
        json.loads(str(step.result))["job"]: children[str(step.child_run_id)]
        for step in run.steps
        if step.tool == "Delegate"
    }


def kept(store: MemoryStore, conversation: str, child: Generation | Decision) -> bool:
    return isinstance(child, Generation) and store.held[at(conversation)] == child.conversation


class Below(MemoryStore):
    def __init__(self, method: str, error: BaseException) -> None:
        super().__init__()
        self.method, self.error = method, error

    def called(self, method: str, scope: str, conversation: str) -> None:
        super().called(method, scope, conversation)
        if method == self.method and conversation != "c1":
            raise self.error


async def lead(engine: Engine, store: MemoryStore, **options: Any) -> Generation:
    run = await engine.run("lead", "acme", "go", store=store, conversation="c1", **options)
    assert isinstance(run, Generation)
    return run


async def test_a_lead_s_helpers_keep_their_conversations_under_its_key(tmp_path: Path) -> None:
    store = MemoryStore()
    model = Crew(
        [background("part one", "part two"), *[text_reply("waiting")] * 3],
        {"part one": [text_reply("one")], "part two": [text_reply("two")]},
    )
    run = await lead(team(crew(tmp_path, background=True), model), store)
    helped = jobs(run)
    assert [
        (job, child.agent, child.conversation[0].content)
        for job, child in helped.items()
        if isinstance(child, Generation)
    ] == [
        ("d1", "worker", "part one"),
        ("d2", "worker", "part two"),
    ]
    assert kept(store, "c1/d1", helped["d1"]) and kept(store, "c1/d2", helped["d2"])
    assert store.held[KEY] == run.conversation
    assert keys(store) == {"c1", "c1/d1", "c1/d2"}


async def test_a_helper_s_helpers_keep_theirs_under_its_key(tmp_path: Path) -> None:
    store = MemoryStore()
    model = Crew(
        [background("part one"), *[text_reply("waiting")] * 3],
        {
            "part one": [background("deep"), *[text_reply("waiting below")] * 3],
            "deep": [text_reply("found")],
        },
    )
    worker = {"orchestration": {"delegate_to": ["worker"], "background": True}}
    run = await lead(team(crew(tmp_path, worker=worker, background=True), model), store)
    captain = jobs(run)["d1"]
    assert isinstance(captain, Generation)
    deep = jobs(captain)["d1"]
    assert kept(store, "c1/d1", captain) and kept(store, "c1/d1/d1", deep)
    assert keys(store) == {"c1", "c1/d1", "c1/d1/d1"}


async def test_a_synchronous_child_keeps_its_conversation_under_its_run_id(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    model = Crew(
        [delegating("worker", "part one"), text_reply("done")], {"part one": [text_reply("one")]}
    )
    run = await lead(team(crew(tmp_path), model), store)
    [child] = run.children
    assert run.steps[0].child_run_id == child.run_id
    assert kept(store, f"c1/{child.run_id}", child)
    assert keys(store) == {"c1", f"c1/{child.run_id}"}


def routing(root: Path) -> Path:
    return write_folder(
        root,
        {
            "front": {
                "kind": "decision",
                "model": "judge",
                "description": "Routes",
                "instructions": "front.md",
                "questions": ["front/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"talk": "chat"}}},
            },
            "chat": {
                "kind": "generation",
                "model": "strong",
                "description": "Talks",
                "instructions": "chat.md",
            },
        },
        files={"front.md": "", "front/intent.json": INTENT, "chat.md": "Talk.\n"},
    )


async def test_a_routed_run_keeps_its_router_s_conversation(tmp_path: Path) -> None:
    store, llm = MemoryStore(), Scripted(text_reply("First."), text_reply("Then this."))
    engine = Engine(routing(tmp_path), providers=provided({"strong": llm, "judge": Talking()}))
    first = await engine.run("front", "acme", "hello", store=store, conversation="c1")
    second = await engine.run("front", "acme", "and then?", store=store, conversation="c1")
    assert isinstance(first, Decision) and isinstance(second, Decision)
    talked, again = first.routed_to, second.routed_to
    assert isinstance(talked, Generation) and isinstance(again, Generation)
    asked = Message(role="user", content="and then?")
    assert llm.calls[1][1] == said([*talked.conversation, asked])
    assert store.held[KEY] == [*talked.conversation, *again.added] == again.conversation
    assert keys(store) == {"c1"}


def switching(root: Path) -> Path:
    return write_folder(
        root,
        {
            "lead": {
                "kind": "generation",
                "model": "crew",
                "description": "Leads the crew",
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["desk", "switch"], "background": True},
            },
            "desk": {
                "kind": "decision",
                "model": "judge",
                "description": "Sorts",
                "instructions": "desk.md",
                "questions": ["desk/*.json"],
            },
            "switch": {
                "kind": "decision",
                "model": "switch",
                "description": "Routes to the worker",
                "instructions": "desk.md",
                "questions": ["switch/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"talk": "worker"}}},
            },
            "worker": {
                "kind": "generation",
                "model": "crew",
                "description": "Works on one part",
                "instructions": "worker.md",
            },
        },
        files={
            "lead.md": "Lead.\n",
            "worker.md": "Work.\n",
            "desk.md": "",
            "desk/urgent.json": URGENT,
            "switch/intent.json": INTENT,
        },
    )


@pytest.mark.parametrize("started", ["background", "synchronous"])
async def test_a_decision_helper_keeps_no_conversation(started: str, tmp_path: Path) -> None:
    store = MemoryStore()
    extra = {"background": True} if started == "background" else {}
    asked = Reply(
        tool_calls=[
            ToolCall(id="s1", name="Delegate", args={"agent": "desk", "input": "x", **extra}),
            ToolCall(
                id="s2", name="Delegate", args={"agent": "switch", "input": "routed", **extra}
            ),
        ],
        stop_reason="tool_use",
    )
    model = Crew([asked, *[text_reply("waiting")] * 3], {"routed": [text_reply("talked")]})
    providers = provided({"crew": model, "judge": Answering(), "switch": Talking()})
    run = await lead(Engine(switching(tmp_path), providers=providers), store)
    desk, switch = sorted(run.children, key=lambda child: child.agent)
    assert isinstance(desk, Decision) and isinstance(switch, Decision)
    assert isinstance(switch.routed_to, Generation) and switch.routed_to.reply == "talked"
    assert keys(store) == {"c1"} and set(store.held) == {KEY}


@pytest.mark.parametrize("taken", ["kept", "in-flight"])
async def test_a_helper_moves_past_a_key_already_in_use(taken: str, tmp_path: Path) -> None:
    store, hold = MemoryStore(), asyncio.Event()
    model = Crew(
        [
            background("part one") if taken == "kept" else background("part one", "part two"),
            *[text_reply("waiting")] * 3,
        ],
        {
            "part one": [text_reply("one")],
            "part two": [text_reply("two")],
            "elsewhere": [hold, text_reply("done there")],
        },
    )
    engine = team(crew(tmp_path, background=True), model)
    if taken == "kept":
        old = [Message(role="user", content="old"), Message(role="assistant", content="old")]
        store.held[at("c1/d1")] = list(old)
        run = await lead(engine, store)
        assert [step.result for step in run.steps] == ['{"job": "d2"}']
        assert store.held[at("c1/d1")] == old and kept(store, "c1/d2", jobs(run)["d2"])
        return
    there = asyncio.create_task(
        engine.run("worker", "acme", "elsewhere", store=store, conversation="c1/d2")
    )
    await until(lambda: len(model.calls["elsewhere"]) == 1, "the run on c1/d2")
    run = await lead(engine, store)
    hold.set()
    elsewhere = await asyncio.wait_for(there, 5)
    assert [step.result for step in run.steps] == ['{"job": "d1"}', '{"job": "d3"}']
    helped = jobs(run)
    assert kept(store, "c1/d1", helped["d1"]) and kept(store, "c1/d3", helped["d3"])
    assert kept(store, "c1/d2", elsewhere)
    assert store.log.count(("length", "acme", "c1/d2")) == 1


def consulting(root: Path) -> Path:
    return write_folder(
        root,
        {
            "lead": {
                "kind": "generation",
                "model": "crew",
                "description": "Leads the crew",
                "instructions": "lead.md",
                "tools": {"allow": ["consult"]},
                "orchestration": {"delegate_to": ["worker"], "background": True},
            },
            "worker": {
                "kind": "generation",
                "model": "crew",
                "description": "Works on one part",
                "instructions": "worker.md",
                "tools": {"allow": ["consult"]},
            },
            "advisor": {
                "kind": "generation",
                "model": "crew",
                "description": "Advises",
                "instructions": "advisor.md",
            },
        },
        runtime_tools=["consult"],
        files={"lead.md": "Lead.\n", "worker.md": "Work.\n", "advisor.md": "Advise.\n"},
    )


def consult(text: str, call_id: str) -> Reply:
    call = ToolCall(id=call_id, name="consult", args={"input": text})
    return Reply(tool_calls=[call], stop_reason="tool_use")


async def test_a_consult_inside_a_store_run_keeps_no_conversation(tmp_path: Path) -> None:
    store = MemoryStore()
    model = Crew(
        [consult("ask", "k1"), background("part one"), *[text_reply("waiting")] * 3],
        {
            "part one": [consult("ask below", "k2"), text_reply("one")],
            "ask": [text_reply("advice")],
            "ask below": [text_reply("advice below")],
        },
    )
    engine = Engine(consulting(tmp_path), providers=provided({"crew": model}))
    run = await lead(engine, store, tools=engine.tool("advisor", scope="acme", name="consult"))
    helper = jobs(run)["d1"]
    assert isinstance(helper, Generation)
    assert run.steps[0].result == "advice" and helper.steps[0].result == "advice below"
    assert all(key == "c1" or key.startswith("c1/") for key in keys(store))
    assert keys(store) == {"c1", "c1/d1"}


async def test_a_lead_whose_store_raises_stops_its_helpers_and_fails(tmp_path: Path) -> None:
    store, kept_runs, error = MemoryStore(), Kept(), OSError("disk full")
    lead_gate, gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("part one"), lead_gate, text_reply("waiting")],
        {"part one": [gate, text_reply("never")]},
    )
    engine = team(crew(tmp_path, background=True), model, refinery=kept_runs)
    before = asyncio.all_tasks()
    work = asyncio.create_task(lead(engine, store))
    await until(
        lambda: len(model.calls["part one"]) == 1 and len(model.calls["lead"]) == 2,
        "the helper's and the lead's model calls",
    )
    stored, held = list(store.held[at("c1/d1")]), list(store.held[KEY])
    store.fail["append"] = error
    lead_gate.set()
    with pytest.raises(OSError) as caught:
        await asyncio.wait_for(work, 5)
    assert caught.value is error
    assert pending(before) == [] and not gate.is_set()
    [failure] = kept_runs.runs
    assert isinstance(failure, Generation) and failure.error == "OSError: disk full"
    [helper] = failure.children
    assert isinstance(helper, Generation) and helper.stopped == "now"
    assert store.held[at("c1/d1")] == stored == helper.conversation
    assert store.held[KEY] == held == failure.conversation[: len(held)]
    assert store.log[-1] == ("append", "acme", "c1")


@pytest.mark.parametrize("started", ["background", "synchronous"])
async def test_a_helper_whose_store_raises_fails_alone(started: str, tmp_path: Path) -> None:
    store = Below("append", OSError("disk full"))
    first = background("part one") if started == "background" else delegating("worker", "part one")
    model = Crew([first, *[text_reply("waiting")] * 3], {"part one": [text_reply("never")]})
    run = await lead(team(crew(tmp_path, background=True), model), store)
    assert run.error is None and run.reply == "waiting"
    [child] = run.children
    assert isinstance(child, Generation) and child.error == "OSError: disk full"
    assert model.calls["part one"] == [] and set(store.held) == {KEY}
    told = [message.content for message in run.conversation if message.notice is not None]
    if started == "background":
        assert told == ["[helper d1 · worker · failed]\nOSError: disk full"]
    else:
        assert told == [] and run.steps[0].result == "OSError: disk full"
        assert run.steps[0].is_error and run.steps[0].child_run_id == child.run_id
    assert store.held[KEY] == run.conversation


async def test_a_store_error_while_a_helper_starts_fails_the_lead(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    error, kept_runs = OSError("disk full"), Kept()
    store = Below("length", error)
    model = Crew(
        [background("part one", "part two"), text_reply("never")],
        {"part one": [text_reply("never")], "part two": [text_reply("never")]},
    )
    engine = team(crew(tmp_path, background=True), model, refinery=kept_runs)
    before = asyncio.all_tasks()
    with (
        caplog.at_level(logging.ERROR, logger="rfnry_engine.engine"),
        pytest.raises(OSError) as caught,
    ):
        await lead(engine, store)
    assert caught.value is error and pending(before) == []
    assert model.calls["part one"] == model.calls["part two"] == []
    [failure] = kept_runs.runs
    assert isinstance(failure, Generation) and failure.error == "OSError: disk full"
    assert failure.steps == [] and failure.children == []
    assert all(message.role != "tool" for message in failure.conversation)
    assert store.held[KEY] == failure.conversation
    assert store.log[-1] == ("length", "acme", "c1/d1") and keys(store) == {"c1", "c1/d1"}
    assert store.log.count(("length", "acme", "c1/d1")) == 1 and caplog.records == []


class Miscounted(MemoryStore):
    def __init__(self, length: object) -> None:
        super().__init__()
        self.length_of_d1 = length

    async def length(self, scope: str, conversation: str) -> int:
        if conversation == "c1/d1":
            self.called("length", scope, conversation)
            return cast(int, self.length_of_d1)
        return await super().length(scope, conversation)


@pytest.mark.parametrize("length", [-1, True, "3", 2.0])
async def test_a_helper_key_s_length_that_is_not_a_whole_number_fails_the_lead(
    length: object, tmp_path: Path
) -> None:
    store, kept_runs = Miscounted(length), Kept()
    model = Crew([background("part one"), text_reply("never")], {"part one": [text_reply("never")]})
    engine = team(crew(tmp_path, background=True), model, refinery=kept_runs)
    wrong = f"conversation: the store's length of conversation 'c1/d1' must be a whole number from 0; got {length!r}"
    with pytest.raises(InputError) as caught:
        await lead(engine, store)
    assert str(caught.value) == wrong and model.calls["part one"] == []
    [failure] = kept_runs.runs
    assert isinstance(failure, Generation) and failure.steps == [] and failure.children == []
    assert store.held[KEY] == failure.conversation


async def test_a_store_error_is_what_its_lead_raises_when_another_call_of_the_batch_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    error, broken = OSError("disk full"), ValueError("the handler broke")
    store = Below("length", error)

    async def on_event(event: object) -> None:
        if isinstance(event, ToolCallStarted) and event.call_id == "b1":
            raise broken

    model = Crew([background("part one", "part two")], {"part two": [text_reply("never")]})
    engine = team(crew(tmp_path, background=True), model)
    with (
        caplog.at_level(logging.ERROR, logger="rfnry_engine.engine"),
        pytest.raises(OSError) as caught,
    ):
        await lead(engine, store, on_event=on_event)
    assert caught.value is error and model.calls["part two"] == []
    [line] = caplog.records
    assert line.getMessage() == "engine: a further call of the same batch failed: the handler broke"
    assert line.exc_info is not None and line.exc_info[1] is broken


class Slow(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.asked, self.gate = asyncio.Event(), asyncio.Event()

    async def length(self, scope: str, conversation: str) -> int:
        if conversation == "c1/d1":
            self.asked.set()
            await self.gate.wait()
        return await super().length(scope, conversation)


async def test_a_stop_while_a_helper_s_key_is_checked_starts_no_helper(tmp_path: Path) -> None:
    store, stop = Slow(), Stop()
    model = Crew([background("part one"), text_reply("never")], {"part one": [text_reply("never")]})
    engine = team(crew(tmp_path, background=True), model)
    before = asyncio.all_tasks()
    work = asyncio.create_task(lead(engine, store, stop=stop))
    await asyncio.wait_for(store.asked.wait(), 5)
    stop.now()
    run = await asyncio.wait_for(work, 5)
    assert run.stopped == "now" and run.steps[0].result == "stopped by the user"
    assert run.children == [] and model.calls["part one"] == []
    assert store.held[KEY] == run.conversation and run.conversation[-1].role == "tool"
    assert pending(before) == [] and not store.gate.is_set()


class Yielding(MemoryStore):
    async def length(self, scope: str, conversation: str) -> int:
        await asyncio.sleep(0)
        return await super().length(scope, conversation)


@pytest.mark.parametrize("cap", [None, 1])
async def test_helpers_of_one_batch_are_checked_and_started_one_at_a_time(
    cap: int | None, tmp_path: Path
) -> None:
    store = Yielding()
    store.held[at("c1/d2")] = [Message(role="user", content="old")]
    model = Crew(
        [background("part one", "part two"), *[text_reply("waiting")] * 3],
        {"part one": [text_reply("one")], "part two": [text_reply("two")]},
    )
    capped: dict[str, Any] = {} if cap is None else {"max_background": cap}
    run = await lead(team(crew(tmp_path, background=True, **capped), model), store)
    first, second = (step.result for step in run.steps)
    assert first == '{"job": "d1"}'
    if cap is None:
        assert second == '{"job": "d3"}' and kept(store, "c1/d3", jobs(run)["d3"])
    else:
        assert second == (
            "InputError: tools/delegate: max_background is 1, and running now: d1; "
            "wait for a helper to end or stop one"
        )
        assert model.calls["part two"] == []
    assert store.held[at("c1/d2")] == [Message(role="user", content="old")]
