import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import pytest

from rfnry_engine import Stop, ToolCallStarted
from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError, ModelError
from rfnry_engine.models import (
    ChoiceAnswer,
    Message,
    QuestionLike,
    Reply,
    State,
    ToolCall,
    ToolSpec,
)
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
    jobs,
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


class Gated:
    def __init__(self) -> None:
        self.gate, self.entered = asyncio.Event(), asyncio.Event()

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.entered.set()
        await self.gate.wait()
        return text_reply("ok")


class Counted(Talking):
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
        self.calls += 1
        return await super().decide(state, questions)


async def test_a_routing_run_reads_its_conversation_once_before_its_decision_call(
    tmp_path: Path,
) -> None:
    store, judge, kept = MemoryStore(), Counted(), Kept()
    llm = Scripted(text_reply("First."), text_reply("Then this."))
    engine = Engine(
        routing(tmp_path), providers=provided({"strong": llm, "judge": judge}), refinery=kept
    )
    await engine.run("front", "acme", "hello", store=store, conversation="c1")
    store.log.clear()
    second = await engine.run("front", "acme", "and then?", store=store, conversation="c1")
    assert isinstance(second, Decision) and isinstance(second.routed_to, Generation)
    assert [method for method, _, _ in store.log] == [
        "length",
        "read",
        "append",
        "window",
        "append",
    ]
    store.fail["read"] = OSError("the store is down")
    with pytest.raises(OSError, match="the store is down"):
        await engine.run("front", "acme", "once more", store=store, conversation="c1")
    assert judge.calls == 2 and len(kept.runs) == 2 and len(llm.calls) == 2


async def test_a_routing_top_run_is_refused_as_in_use_before_its_decision_call(
    tmp_path: Path,
) -> None:
    llm, judge, kept, store = Gated(), Counted(), Kept(), MemoryStore()
    providers = provided({"strong": llm, "judge": judge})
    engine = Engine(routing(tmp_path), providers=providers, refinery=kept)
    first = asyncio.create_task(
        engine.run("front", "acme", "hello", store=store, conversation="c1")
    )
    await asyncio.wait_for(llm.entered.wait(), 5)
    in_use = "engine: conversation 'c1' of scope 'acme' is in use by a run in flight"
    for agent in ("front", "chat", "front"):
        with pytest.raises(InputError, match=f"^{re.escape(in_use)}$"):
            await engine.run(agent, "acme", "again", store=store, conversation="c1")
    assert judge.calls == 1 and kept.runs == []
    assert [method for method, _, _ in store.log] == ["length", "append", "window"]
    llm.gate.set()
    run = await asyncio.wait_for(first, 5)
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert [each.run_id for each in kept.runs] == [run.run_id] and engine._conversations == set()


class Elsewhere:
    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
        return {"intent": ChoiceAnswer(choice="other", probabilities={"talk": 0.1, "other": 0.9})}


class Down:
    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
        raise ModelError("models: the judge is down")


class Broken:
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        raise ModelError("models: the vendor is down")


@pytest.mark.parametrize(
    ("judge", "llm"),
    [
        (Talking(), Scripted(text_reply("ok"))),
        (Elsewhere(), Scripted()),
        (Down(), Scripted()),
        (Talking(), Broken()),
    ],
    ids=["routed", "not-routed", "decision-failed", "routed-run-failed"],
)
async def test_a_routing_top_run_gives_its_name_back_however_it_ends(
    judge: Any, llm: Any, tmp_path: Path
) -> None:
    engine = Engine(routing(tmp_path), providers=provided({"strong": llm, "judge": judge}))
    with suppress(ModelError):
        await engine.run("front", "acme", "hello", store=MemoryStore(), conversation="c1")
    assert engine._conversations == set()


class Deciding(Talking):
    def __init__(self) -> None:
        self.entered, self.gate = asyncio.Event(), asyncio.Event()

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
        self.entered.set()
        await self.gate.wait()
        return await super().decide(state, questions)


async def test_a_routing_top_run_stopped_during_its_decision_gives_its_name_back(
    tmp_path: Path,
) -> None:
    judge, llm, store, stop = Deciding(), Scripted(text_reply("Again.")), MemoryStore(), Stop()
    engine = Engine(routing(tmp_path), providers=provided({"strong": llm, "judge": judge}))
    work = asyncio.create_task(
        engine.run("front", "acme", "hello", store=store, conversation="c1", stop=stop)
    )
    await asyncio.wait_for(judge.entered.wait(), 5)
    stop.now()
    judge.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.stopped == "now" and llm.calls == []
    assert engine._conversations == set()
    again = await engine.run("front", "acme", "and now?", store=store, conversation="c1")
    assert isinstance(again, Decision) and isinstance(again.routed_to, Generation)
    assert again.routed_to.reply == "Again." and engine._conversations == set()
    assert [message.content for message in store.held[KEY]] == ["hello", "and now?", "Again."]


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
    routed = switch.routed_to
    assert isinstance(routed, Generation) and routed.reply == "talked"
    below = "c1/d2" if started == "background" else f"c1/{switch.run_id}"
    assert kept(store, below, routed)
    assert keys(store) == {"c1", below} and set(store.held) == {KEY, at(below)}


def routed_by(started: str, text: str) -> Reply:
    extra = {"background": True} if started == "background" else {}
    call = ToolCall(id="s1", name="Delegate", args={"agent": "switch", "input": text, **extra})
    return Reply(tool_calls=[call], stop_reason="tool_use")


@pytest.mark.parametrize("started", ["background", "synchronous"])
async def test_a_delegated_router_s_routed_run_keeps_its_conversation_under_the_router_s_key(
    started: str, tmp_path: Path
) -> None:
    store = MemoryStore()
    model = Crew(
        [routed_by(started, "routed"), *[text_reply("waiting")] * 3],
        {"routed": [text_reply("talked")]},
    )
    providers = provided({"crew": model, "judge": Answering(), "switch": Talking()})
    engine = Engine(switching(tmp_path), providers=providers)
    run = await lead(engine, store)
    [switch] = run.children
    assert isinstance(switch, Decision)
    routed = switch.routed_to
    assert isinstance(routed, Generation) and routed.parent_run_id == switch.run_id
    below = "c1/d1" if started == "background" else f"c1/{switch.run_id}"
    assert kept(store, below, routed) and keys(store) == {"c1", below}
    assert engine._conversations == set()


async def test_a_routing_decision_helper_moves_past_a_key_already_in_use(tmp_path: Path) -> None:
    store = MemoryStore()
    old = [Message(role="user", content="old")]
    store.held[at("c1/d1")] = list(old)
    model = Crew(
        [routed_by("background", "routed"), *[text_reply("waiting")] * 3],
        {"routed": [text_reply("talked")]},
    )
    providers = provided({"crew": model, "judge": Answering(), "switch": Talking()})
    run = await lead(Engine(switching(tmp_path), providers=providers), store)
    assert [step.result for step in run.steps] == ['{"job": "d2"}']
    switch = jobs(run)["d2"]
    assert isinstance(switch, Decision) and isinstance(switch.routed_to, Generation)
    assert store.held[at("c1/d1")] == old and kept(store, "c1/d2", switch.routed_to)


async def test_a_helper_moves_past_a_key_the_store_holds(tmp_path: Path) -> None:
    store = MemoryStore()
    old = [Message(role="user", content="old"), Message(role="assistant", content="old")]
    store.held[at("c1/d1")] = list(old)
    model = Crew(
        [background("part one"), *[text_reply("waiting")] * 3], {"part one": [text_reply("one")]}
    )
    run = await lead(team(crew(tmp_path, background=True), model), store)
    assert [step.result for step in run.steps] == ['{"job": "d2"}']
    assert store.held[at("c1/d1")] == old and kept(store, "c1/d2", jobs(run)["d2"])


async def test_a_helper_moves_past_a_key_a_run_in_flight_holds(tmp_path: Path) -> None:
    store, hold = MemoryStore(), asyncio.Event()
    model = Crew(
        [background("part one", "part two"), *[text_reply("waiting")] * 3],
        {
            "part one": [text_reply("one")],
            "part two": [text_reply("two")],
            "elsewhere": [hold, text_reply("done there")],
        },
    )
    engine = team(crew(tmp_path, background=True), model)
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
    [helper, failure] = kept_runs.runs
    assert isinstance(failure, Generation) and failure.error == "OSError: disk full"
    assert failure.children == [helper]
    assert isinstance(helper, Generation) and helper.stopped == "now"
    assert store.held[at("c1/d1")] == stored == helper.conversation
    assert store.held[KEY] == held == failure.conversation[: len(held)]
    assert store.log[-1] == ("append", "acme", "c1") and engine._conversations == set()


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
    assert engine._conversations == set()


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
    length: object, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store, kept_runs = Miscounted(length), Kept()
    model = Crew(
        [background("part one", "part two"), text_reply("never")],
        {"part one": [text_reply("never")], "part two": [text_reply("never")]},
    )
    engine = team(crew(tmp_path, background=True), model, refinery=kept_runs)
    wrong = f"conversation: the store's length of conversation 'c1/d1' must be a whole number from 0; got {length!r}"
    with (
        caplog.at_level(logging.ERROR, logger="rfnry_engine.engine"),
        pytest.raises(InputError) as caught,
    ):
        await lead(engine, store)
    assert str(caught.value) == wrong
    assert model.calls["part one"] == model.calls["part two"] == []
    [failure] = kept_runs.runs
    assert isinstance(failure, Generation) and failure.steps == [] and failure.children == []
    assert store.held[KEY] == failure.conversation and keys(store) == {"c1", "c1/d1"}
    assert store.log.count(("length", "acme", "c1/d1")) == 1 and caplog.records == []
    assert engine._conversations == set()


@pytest.mark.parametrize("failing", ["raises", "not-a-number"])
async def test_a_store_error_is_what_its_lead_raises_when_another_call_of_the_batch_fails(
    failing: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    broken, error = ValueError("the handler broke"), OSError("disk full")
    store: MemoryStore = Below("length", error) if failing == "raises" else Miscounted(-1)

    async def on_event(event: object) -> None:
        if isinstance(event, ToolCallStarted) and event.call_id == "b1":
            raise broken

    model = Crew([background("part one", "part two")], {"part two": [text_reply("never")]})
    engine = team(crew(tmp_path, background=True), model)
    with (
        caplog.at_level(logging.ERROR, logger="rfnry_engine.engine"),
        pytest.raises((OSError, InputError, ValueError)) as caught,
    ):
        await lead(engine, store, on_event=on_event)
    if failing == "raises":
        assert caught.value is error
    else:
        assert type(caught.value) is InputError and "'c1/d1'" in str(caught.value)
    assert model.calls["part two"] == []
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


class Crowded(MemoryStore):
    def __init__(self, stop: Stop) -> None:
        super().__init__()
        self.stop, self.asked = stop, 0

    async def length(self, scope: str, conversation: str) -> int:
        if conversation == "c1":
            return await super().length(scope, conversation)
        self.asked += 1
        if self.asked == 50:
            self.stop.now()
        return 1 if self.asked < 100_000 else 0


async def test_a_stop_lands_while_its_lead_passes_over_keys_in_use(tmp_path: Path) -> None:
    stop = Stop()
    store = Crowded(stop)
    model = Crew([background("part one"), text_reply("never")], {"part one": [text_reply("never")]})
    engine = team(crew(tmp_path, background=True), model)
    before = asyncio.all_tasks()
    run = await asyncio.wait_for(lead(engine, store, stop=stop), 30)
    assert run.stopped == "now" and run.steps[0].result == "stopped by the user"
    assert run.children == [] and model.calls["part one"] == [] and store.asked < 100
    assert store.held[KEY] == run.conversation and engine._conversations == set()
    assert pending(before) == []


class Racing(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.race: Callable[[], Awaitable[Any]] | None = None
        self.raced: asyncio.Future[Any] | None = None

    async def length(self, scope: str, conversation: str) -> int:
        if conversation == "c1/d1" and self.race is not None and self.raced is None:
            self.raced = asyncio.ensure_future(self.race())
        return await super().length(scope, conversation)


async def test_no_run_takes_a_helper_s_key_between_its_check_and_its_start(
    tmp_path: Path,
) -> None:
    store = Racing()
    model = Crew(
        [background("part one"), *[text_reply("waiting")] * 3],
        {"part one": [text_reply("one")], "elsewhere": [text_reply("there")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    store.race = lambda: engine.run(
        "worker", "acme", "elsewhere", store=store, conversation="c1/d1"
    )
    run = await lead(engine, store)
    assert store.raced is not None
    in_use = "engine: conversation 'c1/d1' of scope 'acme' is in use by a run in flight"
    with pytest.raises(InputError, match=f"^{in_use}$"):
        await store.raced
    assert kept(store, "c1/d1", jobs(run)["d1"]) and model.calls["elsewhere"] == []
    assert engine._conversations == set()


class Cancelling(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.work: asyncio.Task[Any] | None = None

    async def length(self, scope: str, conversation: str) -> int:
        if conversation == "c1/d1" and self.work is not None:
            self.work.cancel()
        return await super().length(scope, conversation)


async def test_a_helper_that_never_starts_gives_its_key_back(tmp_path: Path) -> None:
    store = Cancelling()
    model = Crew([background("part one"), text_reply("never")], {"part one": [text_reply("never")]})
    engine = team(crew(tmp_path, background=True), model)
    before = asyncio.all_tasks()
    store.work = asyncio.create_task(lead(engine, store))
    with pytest.raises(asyncio.CancelledError):
        await store.work
    assert model.calls["part one"] == [] and at("c1/d1") not in store.held
    assert engine._conversations == set() and pending(before) == []


class Holding(MemoryStore):
    def __init__(self, method: str, below: bool) -> None:
        super().__init__()
        self.method: str | None = method
        self.below = below
        self.entered, self.gate = asyncio.Event(), asyncio.Event()
        self.cut = 0

    async def held_up(self, method: str, conversation: str) -> None:
        if method == self.method and (conversation != "c1") is self.below:
            self.method = None
            self.entered.set()
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cut += 1
                raise

    async def append(
        self, scope: str, conversation: str, start: int, messages: Sequence[Message]
    ) -> None:
        await self.held_up("append", conversation)
        await super().append(scope, conversation, start, messages)

    async def window(self, scope: str, conversation: str, length: int) -> int:
        await self.held_up("window", conversation)
        return await super().window(scope, conversation, length)


@pytest.mark.parametrize("method", ["append", "window"])
@pytest.mark.parametrize("who", ["lead", "synchronous child", "helper"])
async def test_a_stop_pressed_while_a_store_call_waits_stores_every_message_once(
    who: str, method: str, tmp_path: Path
) -> None:
    stop, store = Stop(), Holding(method, below=who != "lead")
    if who == "lead":
        model = Crew([text_reply("never")])
    elif who == "synchronous child":
        model = Crew(
            [delegating("worker", "part one"), text_reply("never")],
            {"part one": [text_reply("never")]},
        )
    else:
        model = Crew(
            [background("part one"), *[text_reply("waiting")] * 3],
            {"part one": [text_reply("never")]},
        )
    engine = team(crew(tmp_path, background=who == "helper"), model)
    before = asyncio.all_tasks()
    work = asyncio.create_task(lead(engine, store, stop=stop))
    await asyncio.wait_for(store.entered.wait(), 5)
    stop.now()
    for _ in range(20):
        await asyncio.sleep(0)
    assert not work.done()
    store.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert run.stopped == "now" and store.cut == 0
    runs = [run, *run.children]
    named = {"c1": run} if who == "lead" else {"c1": run, f"c1/{jobs_or_id(run)}": runs[1]}
    assert set(store.held) == {at(key) for key in named}
    for key, kept_run in named.items():
        assert isinstance(kept_run, Generation) and store.held[at(key)] == kept_run.conversation
    assert engine._conversations == set() and pending(before) == []


def jobs_or_id(run: Generation) -> str:
    [step] = run.steps
    if str(step.result).startswith("{"):
        return str(json.loads(str(step.result))["job"])
    return str(step.child_run_id)
