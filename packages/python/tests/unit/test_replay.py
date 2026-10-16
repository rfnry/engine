import asyncio
import json
import shutil
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import aclosing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError, NotFoundError
from rfnry_engine.models import (
    Answer,
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    HistoryLike,
    Image,
    Message,
    QuestionLike,
    Reply,
    ScoreAnswer,
    State,
    Text,
    ToolCall,
    ToolSpec,
)
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.refinery.holdout import is_holdout
from rfnry_engine.refinery.labels import Label, Outcome, Replay, Review
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import LabelRow, Storage
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import (
    PICK,
    TOOLS,
    Answering,
    Desk,
    Fixed,
    Scripted,
    delegate_triage,
    delegating,
    generation_record,
    lead,
    said,
    text_reply,
)

BASE = datetime(2026, 9, 1, tzinfo=UTC)


def ids(held: bool, n: int, percent: int = 20) -> list[str]:
    found = (f"run-{i}" for i in range(10_000))
    return [i for i in found if is_holdout(i, holdout_percent=percent) is held][:n]


async def put(refinery: Refinery, run: Run, run_id: str, minute: int) -> Run:
    moved = run.model_copy(update={"run_id": run_id, "at": BASE + timedelta(minutes=minute)})
    await refinery.record(moved)
    return moved


class Echo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Message], list[ToolSpec]]] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.calls.append((system, list(messages), list(tools)))
        return text_reply(f"candidate {len(self.calls)}")


async def collect(pairs: Any) -> list[Replay]:
    return [pair async for pair in pairs]


def candidate(fixtures: Path, llm: Any = None, **kwargs: Any) -> Engine:
    return Engine(fixtures, generation=llm or Echo(), decision=Desk(), tools=TOOLS, **kwargs)


async def test_it_replays_the_top_level_runs_of_the_agent_oldest_first(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures,
        generation=Scripted(delegate_triage(), text_reply("Handled."), text_reply("Hi.")),
        decision=Desk(),
        tools=TOOLS,
        refinery=refinery,
    )
    first = await recorder.run("support", "acme", "It is down")
    second = await recorder.run("support", "acme", "Hello")
    await recorder.run("triage", "acme", "It is down")
    pairs = await collect(refinery.replay("acme", candidate(fixtures), agent="support"))
    assert [pair.recorded.run_id for pair in pairs] == [first.run_id, second.run_id]
    assert pairs[0].recorded == first and pairs[0].labels == []
    triage = await collect(refinery.replay("acme", candidate(fixtures), agent="triage"))
    assert len(triage) == 1 and isinstance(triage[0].recorded, Decision)
    assert triage[0].recorded.parent_run_id is None
    limited = await collect(refinery.replay("acme", candidate(fixtures), agent="support", limit=1))
    assert [pair.recorded.run_id for pair in limited] == [first.run_id]
    assert await collect(refinery.replay("other", candidate(fixtures), agent="support")) == []


async def test_holdout_selects_by_the_same_hash_and_percent_as_export(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    held, kept = ids(True, 2), ids(False, 3)
    for minute, run_id in enumerate([kept[0], held[0], kept[1], held[1], kept[2]]):
        await put(refinery, generation_record(), run_id, minute)
    engine = candidate(fixtures)
    pairs = await collect(refinery.replay("acme", engine, agent="support", holdout=True))
    assert [pair.recorded.run_id for pair in pairs] == held
    everything = await collect(
        refinery.replay("acme", engine, agent="support", holdout=True, holdout_percent=100)
    )
    assert len(everything) == 5
    assert (
        await collect(
            refinery.replay("acme", engine, agent="support", holdout=True, holdout_percent=0)
        )
        == []
    )


async def test_labeled_keeps_runs_a_person_or_the_world_labeled(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    for i in range(4):
        await put(refinery, generation_record(), f"r{i}", i)
    await refinery.review("acme", "r0", Review(stance="approval"))
    await refinery.outcome("acme", "r1", Outcome(status="failed"))
    judged = Label(label_id="j", run_id="r2", scope="acme", at=BASE, source="judge", answers={})
    await refinery.storage.put_label(
        "acme",
        LabelRow(
            scope="acme",
            label_id="j",
            run_id="r2",
            source="judge",
            at=BASE,
            body=judged.model_dump(mode="json"),
        ),
    )
    pairs = await collect(
        refinery.replay("acme", candidate(fixtures), agent="support", labeled=True)
    )
    assert [pair.recorded.run_id for pair in pairs] == ["r0", "r1"]
    assert [label.source for label in pairs[0].labels] == ["human"]
    assert [label.source for label in pairs[1].labels] == ["world"]
    every = await collect(refinery.replay("acme", candidate(fixtures), agent="support"))
    assert [[label.source for label in pair.labels] for pair in every] == [
        ["human"],
        ["world"],
        ["judge"],
        [],
    ]


async def test_tags_select_by_containment(fixtures: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    arms = [{"arm": "a", "customer": "c-1"}, {"arm": "b"}, {"arm": "a"}]
    for i, tags in enumerate(arms):
        await put(refinery, generation_record().model_copy(update={"tags": tags}), f"r{i}", i)
    pairs = await collect(
        refinery.replay("acme", candidate(fixtures), agent="support", tags={"arm": "a"})
    )
    assert [pair.recorded.run_id for pair in pairs] == ["r0", "r2"]
    only = await collect(
        refinery.replay(
            "acme", candidate(fixtures), agent="support", tags={"arm": "a", "customer": "c-1"}
        )
    )
    assert [pair.recorded.run_id for pair in only] == ["r0"]


async def test_a_fingerprint_selects_one_recorded_version(
    fixtures: Path, tmp_path: Path, storage: Storage
) -> None:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    refinery = Refinery(storage)
    old = Engine(root, generation=Scripted(text_reply("a")), tools=TOOLS, refinery=refinery)
    before = await old.run("support", "acme", "hi")
    (root / "support/rules/refunds.md").write_text("Refund only the second charge.\n")
    new = Engine(root, generation=Scripted(text_reply("b")), tools=TOOLS, refinery=refinery)
    after = await new.run("support", "acme", "hi")
    await put(refinery, generation_record(), "fixture", 0)
    assert before.fingerprint != after.fingerprint
    engine = candidate(fixtures)
    for version, run in ((before.fingerprint, before), (after.fingerprint, after)):
        pairs = await collect(refinery.replay("acme", engine, agent="support", fingerprint=version))
        assert [pair.recorded.run_id for pair in pairs] == [run.run_id]
    assert len(await collect(refinery.replay("acme", engine, agent="support"))) == 3


async def test_the_candidate_is_sent_the_input_history_task_and_context_as_recorded(
    fixtures: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures,
        generation=Scripted(text_reply("one"), text_reply("two"), text_reply("three")),
        tools=TOOLS,
        refinery=refinery,
    )
    parts = [Text(text="Charged twice?"), Image(media_type="image/png", data=b"\x89PNG\r\n\x1a\n")]
    history = [
        Message(
            role="user", content=[Text(text="Earlier"), Image(media_type="image/png", url="u")]
        ),
        Message(role="assistant", content="Noted."),
    ]
    with_parts = await recorder.run(
        "support",
        "acme",
        parts,
        history=history,
        task="triage",
        context="Acme is on the enterprise plan.",
    )
    with_object = await recorder.run("support", "acme", {"order": "A-1", "items": [1, 2]})
    with_text = await recorder.run("support", "acme", "Plain text.", history=history[1:])
    llm = Echo()
    pairs = await collect(refinery.replay("acme", candidate(fixtures, llm), agent="support"))
    assert [pair.error for pair in pairs] == [None, None, None]
    for (system, messages, _), pair, recorded in zip(
        llm.calls, pairs, (with_parts, with_object, with_text), strict=True
    ):
        assert isinstance(recorded, Generation) and isinstance(pair.candidate, Generation)
        assert messages == recorded.messages
        assert system == recorded.system
        assert pair.candidate.messages == recorded.messages
        assert pair.candidate.task == recorded.task
        assert pair.candidate.context == recorded.context
        assert pair.candidate.fingerprint == recorded.fingerprint
    first = pairs[0].candidate
    assert isinstance(first, Generation) and first.input == parts
    assert isinstance(first.messages[-1].content, list)
    assert first.messages[-1].content[1] == parts[1]
    second = pairs[1].candidate
    assert isinstance(second, Generation) and second.input == {"order": "A-1", "items": [1, 2]}
    assert second.messages[-1].content == json.dumps({"order": "A-1", "items": [1, 2]})
    assert "Decide whether this message needs a person." in llm.calls[0][0]
    assert "Acme is on the enterprise plan." in llm.calls[0][0]


class Verdict(BaseModel):
    refund: bool


async def test_expect_and_approve_are_the_caller_s(fixtures: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures, generation=Scripted(text_reply("ok")), tools=TOOLS, refinery=refinery
    )
    await recorder.run("support", "acme", "Refund?")
    call = ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"})
    llm = Scripted(
        Reply(tool_calls=[call], stop_reason="tool_use"),
        Reply(
            tool_calls=[ToolCall(id="o1", name="Output", args={"refund": True})],
            stop_reason="tool_use",
        ),
    )
    asked: list[tuple[str, str]] = []

    async def refuse(agent: str, call: ToolCall) -> bool:
        asked.append((agent, call.name))
        return False

    [pair] = await collect(
        refinery.replay(
            "acme", candidate(fixtures, llm), agent="support", expect=Verdict, approve=refuse
        )
    )
    assert isinstance(pair.candidate, Generation)
    assert "Output" in {spec.name for spec in pair.candidate.tools}
    assert Verdict.model_validate_json(pair.candidate.reply) == Verdict(refund=True)
    assert asked == [("support", "lookup_order")]
    assert pair.candidate.steps[0].approved is False
    assert pair.candidate.steps[0].result == "NotApproved: lookup_order was not approved"


class Flaky:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("the vendor is down")
        return text_reply(f"reply {self.calls}")


async def test_a_candidate_that_fails_yields_its_error_and_the_replay_goes_on(
    fixtures: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    for i in range(3):
        await put(refinery, generation_record(), f"r{i}", i)
    pairs = await collect(refinery.replay("acme", candidate(fixtures, Flaky()), agent="support"))
    assert [pair.recorded.run_id for pair in pairs] == ["r0", "r1", "r2"]
    assert [pair.error for pair in pairs] == [None, "RuntimeError: the vendor is down", None]
    assert pairs[1].candidate is None
    assert [pair.candidate.reply for pair in pairs if isinstance(pair.candidate, Generation)] == [
        "reply 1",
        "reply 3",
    ]
    lost = Engine(fixtures, decision=Desk(), tools=TOOLS)
    [missing] = await collect(refinery.replay("acme", lost, agent="support", limit=1))
    assert missing.candidate is None and missing.error is not None
    assert missing.error.startswith("ConfigurationError: engine: agent 'support'")


async def test_repeat_replays_each_recorded_run_that_many_times_one_after_another(
    fixtures: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    for i in range(2):
        await put(refinery, generation_record(), f"r{i}", i)
    llm = Scripted(*(text_reply(word) for word in ("one", "two", "three", "four", "five", "six")))
    pairs = await collect(
        refinery.replay("acme", candidate(fixtures, llm), agent="support", repeat=3)
    )
    assert [(pair.recorded.run_id, pair.attempt) for pair in pairs] == [
        ("r0", 0),
        ("r0", 1),
        ("r0", 2),
        ("r1", 0),
        ("r1", 1),
        ("r1", 2),
    ]
    assert pairs[0].recorded == pairs[1].recorded == pairs[2].recorded
    assert pairs[0].labels == pairs[1].labels == pairs[2].labels
    runs = [pair.candidate for pair in pairs]
    assert all(isinstance(run, Generation) for run in runs)
    assert [run.reply for run in runs if isinstance(run, Generation)] == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
    ]
    assert len({run.run_id for run in runs if run is not None}) == 6
    assert len(llm.calls) == 6
    once = await collect(refinery.replay("acme", candidate(fixtures), agent="support"))
    assert [(pair.recorded.run_id, pair.attempt) for pair in once] == [("r0", 0), ("r1", 0)]


async def test_limit_counts_recorded_runs_not_attempts(fixtures: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    for i in range(3):
        await put(refinery, generation_record(), f"r{i}", i)
    pairs = await collect(
        refinery.replay("acme", candidate(fixtures), agent="support", limit=2, repeat=2)
    )
    assert [(pair.recorded.run_id, pair.attempt) for pair in pairs] == [
        ("r0", 0),
        ("r0", 1),
        ("r1", 0),
        ("r1", 1),
    ]


async def test_an_attempt_that_fails_yields_its_error_and_the_other_attempts_go_on(
    fixtures: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    await put(refinery, generation_record(), "r0", 0)
    pairs = await collect(
        refinery.replay("acme", candidate(fixtures, Flaky()), agent="support", repeat=3)
    )
    assert [pair.attempt for pair in pairs] == [0, 1, 2]
    assert [pair.error for pair in pairs] == [None, "RuntimeError: the vendor is down", None]
    assert pairs[1].candidate is None
    assert [pair.candidate.reply for pair in pairs if isinstance(pair.candidate, Generation)] == [
        "reply 1",
        "reply 3",
    ]


async def test_a_decision_run_replays_its_state_and_what_its_route_was_given(
    fixtures: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures,
        generation=Scripted(text_reply("Routed.")),
        decision=Desk(),
        tools=TOOLS,
        refinery=refinery,
    )
    triaged = await recorder.run("triage", "acme", {"message": "It is down", "plan": "pro"})
    assert isinstance(triaged, Decision)
    history = [Message(role="user", content="Earlier"), Message(role="assistant", content="Ok.")]
    routed = await recorder.run(
        "frontdesk",
        "acme",
        [Text(text="It is"), Text(text="down")],
        history=history,
        context="On call: Ana.",
    )
    assert isinstance(routed, Decision) and isinstance(routed.routed_to, Generation)
    dm = Fixed(
        {
            "urgent": CheckAnswer(probability=0.2),
            "severity": ScoreAnswer(score=0, probabilities=[1, 0, 0, 0]),
        }
    )
    [pair] = await collect(refinery.replay("acme", Engine(fixtures, decision=dm), agent="triage"))
    assert isinstance(pair.recorded, Decision) and isinstance(pair.candidate, Decision)
    assert dm.calls[0][0] == {"message": "It is down", "plan": "pro"} == triaged.state
    assert pair.candidate.answers["urgent"].confidence == pytest.approx(0.6)
    llm = Echo()
    [front] = await collect(refinery.replay("acme", candidate(fixtures, llm), agent="frontdesk"))
    assert isinstance(front.candidate, Decision) and front.candidate.state == "It is\ndown"
    routed_again = front.candidate.routed_to
    assert isinstance(routed_again, Generation)
    assert routed_again.input == [Text(text="It is"), Text(text="down")]
    assert routed_again.messages == routed.routed_to.messages
    assert routed_again.context == "On call: Ana."
    [(_, messages, _)] = llm.calls
    assert messages[:-1] == history


async def test_a_run_with_children_replays_as_one_run_the_candidate_delegates_as_it_does(
    fixtures: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures,
        generation=Scripted(delegate_triage(), text_reply("Escalated.")),
        decision=Desk(),
        tools=TOOLS,
        refinery=refinery,
    )
    run = await recorder.run("support", "acme", "It is down")
    assert isinstance(run, Generation) and len(run.children) == 1
    [pair] = await collect(refinery.replay("acme", candidate(fixtures), agent="support"))
    assert pair.recorded == run
    assert isinstance(pair.candidate, Generation) and pair.candidate.children == []
    [again] = await collect(
        refinery.replay(
            "acme",
            candidate(fixtures, Scripted(delegate_triage(), text_reply("Again."))),
            agent="support",
        )
    )
    assert isinstance(again.candidate, Generation) and len(again.candidate.children) == 1


async def test_a_replay_records_nothing_and_labels_nothing(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures,
        generation=Scripted(delegate_triage(), text_reply("Handled.")),
        decision=Desk(),
        tools=TOOLS,
        refinery=refinery,
    )
    run = await recorder.run("support", "acme", "It is down")
    await refinery.review("acme", run.run_id, Review(stance="approval"))
    counted = await storage.count_scope("acme")
    llm = Scripted(delegate_triage(), text_reply("Handled again."))
    engine = Engine(fixtures, generation=llm, decision=Desk(), tools=TOOLS, refinery=refinery)
    [pair] = await collect(refinery.replay("acme", engine, agent="support", labeled=True))
    assert isinstance(pair.candidate, Generation) and len(pair.candidate.children) == 1
    assert await storage.count_scope("acme") == counted == 3
    assert len(await storage.labels("acme")) == 1


class Gate:
    def __init__(self) -> None:
        self.running = 0
        self.most = 0
        self.started = 0
        self.finished = 0
        self.release: dict[str, asyncio.Event] = {}

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        text = str(messages[-1].content)
        self.started += 1
        self.running += 1
        self.most = max(self.most, self.running)
        try:
            await self.release.setdefault(text, asyncio.Event()).wait()
        finally:
            self.running -= 1
        self.finished += 1
        return text_reply(f"re: {text}")


async def recorded_inputs(refinery: Refinery, texts: list[str]) -> None:
    for minute, text in enumerate(texts):
        run = generation_record().model_copy(
            update={"input": text, "messages": [Message(role="user", content=text)]}
        )
        await put(refinery, run, f"r{minute}", minute)


async def until(holds: Any) -> None:
    for _ in range(1000):
        if holds():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the condition never held")


async def test_concurrency_bounds_the_runs_in_flight_and_order_holds(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    await recorded_inputs(refinery, ["a", "b", "c", "d"])
    gate = Gate()
    pairs = refinery.replay("acme", candidate(fixtures, gate), agent="support", concurrency=2)
    got: list[str] = []

    async def consume() -> None:
        async for pair in pairs:
            assert isinstance(pair.candidate, Generation)
            got.append(pair.candidate.reply)

    def release(text: str) -> None:
        gate.release.setdefault(text, asyncio.Event()).set()

    task = asyncio.create_task(consume())
    await until(lambda: gate.running == 2)
    release("b")
    await until(lambda: gate.finished == 1)
    await asyncio.sleep(0.01)
    assert got == [] and gate.started == 2
    release("a")
    await until(lambda: gate.running == 2 and gate.started == 4)
    assert got == ["re: a", "re: b"]
    release("d")
    release("c")
    await asyncio.wait_for(task, 5)
    assert got == ["re: a", "re: b", "re: c", "re: d"]
    assert gate.most == 2


async def test_concurrency_bounds_every_attempt_together_and_order_holds(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    await recorded_inputs(refinery, ["a", "b"])
    gate = Gate()
    pairs = refinery.replay(
        "acme", candidate(fixtures, gate), agent="support", concurrency=3, repeat=2
    )
    got: list[tuple[str, int, str]] = []

    async def consume() -> None:
        async for pair in pairs:
            assert isinstance(pair.candidate, Generation)
            got.append((pair.recorded.run_id, pair.attempt, pair.candidate.reply))

    task = asyncio.create_task(consume())
    await until(lambda: gate.running == 3)
    gate.release.setdefault("b", asyncio.Event()).set()
    await until(lambda: gate.finished == 1)
    await asyncio.sleep(0.01)
    assert got == [] and gate.started == 3
    gate.release.setdefault("a", asyncio.Event()).set()
    await asyncio.wait_for(task, 5)
    assert got == [("r0", 0, "re: a"), ("r0", 1, "re: a"), ("r1", 0, "re: b"), ("r1", 1, "re: b")]
    assert gate.most == 3 and gate.started == 4


async def test_one_at_a_time_runs_the_next_only_when_the_last_is_taken(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    await recorded_inputs(refinery, ["a", "b", "c"])
    llm = Echo()
    async with aclosing(
        refinery.replay("acme", candidate(fixtures, llm), agent="support")
    ) as pairs:
        async for pair in pairs:
            assert isinstance(pair.candidate, Generation)
            await asyncio.sleep(0.01)
            assert len(llm.calls) == int(pair.recorded.run_id[1:]) + 1
            if pair.recorded.run_id == "r1":
                break
    assert len(llm.calls) == 2


async def test_closing_a_replay_cancels_the_runs_in_flight(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    await recorded_inputs(refinery, ["a", "b", "c"])
    gate = Gate()
    gate.release["a"] = asyncio.Event()
    gate.release["a"].set()
    async with aclosing(
        refinery.replay("acme", candidate(fixtures, gate), agent="support", concurrency=3)
    ) as pairs:
        async for pair in pairs:
            assert pair.recorded.run_id == "r0"
            break
    await asyncio.sleep(0.05)
    assert gate.finished == 1 and gate.running == 0
    for event in gate.release.values():
        event.set()
    await asyncio.sleep(0.05)
    assert gate.finished == 1


async def test_a_replay_closed_under_a_cancel_returns_once_its_runs_have_ended(
    tmp_path: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(graded(tmp_path), decision=Answering(), refinery=refinery)
    for text in ["a", "b"]:
        await recorder.run("grader", "acme", text)
    entered, release, died = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class SlowToDie(Answering):
        async def decide(
            self, state: State, questions: Mapping[str, QuestionLike]
        ) -> dict[str, Answer]:
            if state == "a":
                return await super().decide(state, questions)
            entered.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.shield(release.wait())
                died.set()
                raise
            raise AssertionError("a cancelled replay must not end on its own")

    async def read_one() -> None:
        engine = Engine(tmp_path, decision=SlowToDie())
        async with aclosing(
            refinery.replay("acme", engine, agent="grader", concurrency=2)
        ) as pairs:
            async for _ in pairs:
                await entered.wait()
                break

    reading = asyncio.create_task(read_one())
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        await asyncio.sleep(0.05)
        reading.cancel()
        await asyncio.sleep(0.05)
        assert not reading.done() and not died.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(2):
            await reading
    assert died.is_set()


async def test_a_reader_cancelled_twice_cancels_its_replay_once(
    tmp_path: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(graded(tmp_path), decision=Answering(), refinery=refinery)
    await recorder.run("grader", "acme", "a")
    entered, release, died = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class SlowToDie(Answering):
        async def decide(
            self, state: State, questions: Mapping[str, QuestionLike]
        ) -> dict[str, Answer]:
            entered.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await release.wait()
                died.set()
                raise
            raise AssertionError("a cancelled replay must not end on its own")

    async def read() -> None:
        engine = Engine(tmp_path, decision=SlowToDie())
        async for _ in refinery.replay("acme", engine, agent="grader"):
            raise AssertionError("a cancelled replay must yield nothing")

    reading = asyncio.create_task(read())
    try:
        async with asyncio.timeout(2):
            await entered.wait()
        for _ in range(2):
            reading.cancel()
            await asyncio.sleep(0.05)
            assert not reading.done() and not died.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(2):
            await reading
    assert died.is_set()


async def recall(args: dict[str, Any]) -> str:
    return "nothing remembered"


async def test_the_tools_and_workspace_a_run_had_are_the_caller_s(
    tmp_path: Path, storage: Storage
) -> None:
    (tmp_path / "coder").mkdir()
    (tmp_path / "coder/AGENT.md").write_text(
        "---\nname: coder\nkind: generation\nworkspace: true\n---\nYou code.\n"
    )
    (tmp_path / "work").mkdir()
    work = Workspace(Local(tmp_path / "work"), scope="acme")
    memory = Tools([Tool("recall", "Recall a fact", body=recall)])
    refinery = Refinery(storage)
    recorder = Engine(tmp_path, generation=Scripted(text_reply("done")), refinery=refinery)
    run = await recorder.run("coder", "acme", "fix it", tools=memory, workspace=work)
    engine = Engine(tmp_path, generation=Echo())
    [bare] = await collect(refinery.replay("acme", engine, agent="coder"))
    assert bare.candidate is None and bare.error is not None
    assert bare.error.startswith("ConfigurationError: engine_tools: agent 'coder'")
    [pair] = await collect(
        refinery.replay("acme", engine, agent="coder", tools=memory, workspace=work)
    )
    assert isinstance(pair.candidate, Generation)
    assert pair.candidate.fingerprint == run.fingerprint
    assert {"recall", "Shell", "Write", "Edit"} <= {spec.name for spec in pair.candidate.tools}


async def test_a_bad_call_is_refused_before_anything_is_read(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    engine = candidate(fixtures)
    with pytest.raises(InputError, match="^refinery: scope is required$"):
        refinery.replay("", engine, agent="support")
    with pytest.raises(InputError, match="^refinery: holdout_percent must be 0-100, got 101$"):
        refinery.replay("acme", engine, agent="support", holdout_percent=101)
    with pytest.raises(InputError, match="^refinery: limit must be at least 1, got 0$"):
        refinery.replay("acme", engine, agent="support", limit=0)
    with pytest.raises(InputError, match="^refinery: concurrency must be at least 1, got 0$"):
        refinery.replay("acme", engine, agent="support", concurrency=0)
    with pytest.raises(InputError, match="^refinery: repeat must be at least 1, got 0$"):
        refinery.replay("acme", engine, agent="support", repeat=0)
    with pytest.raises(InputError, match="^refinery: tags must hold at least one tag$"):
        refinery.replay("acme", engine, agent="support", tags={})
    with pytest.raises(NotFoundError, match="^refinery: the candidate has no agent 'ghost'$"):
        refinery.replay("acme", engine, agent="ghost")


def graded(root: Path, urgent: str = "Is it urgent?") -> Path:
    for path, text in {
        "grader/AGENT.md": "---\nname: grader\nkind: decision\n---\n",
        "grader/questions/urgent.md": f"---\ntype: check\n---\n{urgent}\n",
        "grader/tasks/tone/polite.md": f"---\ntype: check\n---\n{urgent} Polite?\n",
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return root


AREA = Choice(
    instructions={"question": "Which area?", "within": "orders"},
    options={"delivery": {"what": "Parcels", "examples": ["Late"]}, "returns": None},
)


async def test_a_decision_replay_re_asks_runtime_questions_as_recorded_and_the_folder_from_the_candidate(
    tmp_path: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(graded(tmp_path / "old"), decision=Answering(), refinery=refinery)
    asked = await recorder.run("grader", "acme", "x", questions={"area": AREA})
    tasked = await recorder.run("grader", "acme", "y", task="tone", questions={"area": AREA})
    plain = await recorder.run("grader", "acme", "z")
    dm = Answering()
    candidate_ = Engine(graded(tmp_path / "new", "Must it be done today?"), decision=dm)
    pairs = await collect(refinery.replay("acme", candidate_, agent="grader"))
    assert [pair.error for pair in pairs] == [None, None, None]
    first, second, third = (pair.candidate for pair in pairs)
    assert isinstance(first, Decision) and isinstance(second, Decision)
    assert isinstance(third, Decision)
    assert first.questions == {
        "urgent": Check(instructions="Must it be done today?"),
        "area": AREA,
    }
    assert first.runtime == ["area"] and first.state == "x"
    assert second.task == "tone" and second.runtime == ["area"]
    assert second.questions == {
        "urgent": Check(instructions="Must it be done today?"),
        "polite": Check(instructions="Must it be done today? Polite?"),
        "area": AREA,
    }
    assert third.runtime == [] and set(third.questions) == {"urgent"}
    assert [call[1] for call in dm.calls] == [first.questions, second.questions, third.questions]
    for recorded, pair in zip((asked, tasked, plain), pairs, strict=True):
        assert isinstance(recorded, Decision) and isinstance(pair.candidate, Decision)
        assert recorded.fingerprint.agent != pair.candidate.fingerprint.agent
        assert recorded.fingerprint.questions != pair.candidate.fingerprint.questions


async def test_a_routed_decision_replays_with_the_router_s_task_not_the_routed_run_s(
    tmp_path: Path,
    storage: Storage,
) -> None:
    for path, text in {
        "desk/AGENT.md": "---\nname: desk\nkind: decision\nroute: {by: team, to: {b: grader}}\n---\n",
        "desk/questions/team.md": "---\ntype: choice\noptions: {a: A, b: B}\n---\nWhich?\n",
        "desk/tasks/audit/angry.md": "---\ntype: check\n---\nAngry?\n",
    }.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    graded(tmp_path)
    refinery = Refinery(storage)
    recorded = await Engine(tmp_path, decision=Answering(), refinery=refinery).run(
        "desk", "acme", "x", task="audit"
    )
    assert isinstance(recorded, Decision) and isinstance(recorded.routed_to, Decision)
    assert recorded.routed_to.task is None
    dm = Answering()
    [pair] = await collect(refinery.replay("acme", Engine(tmp_path, decision=dm), agent="desk"))
    assert isinstance(pair.candidate, Decision) and pair.candidate.task == "audit"
    assert [list(asked) for _, asked in dm.calls] == [["team", "angry"], ["urgent"]]


async def test_a_runtime_question_the_candidate_s_folder_now_holds_fails_its_pair(
    tmp_path: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(graded(tmp_path / "old"), decision=Answering(), refinery=refinery)
    await recorder.run("grader", "acme", "x", questions={"area": AREA})
    root = graded(tmp_path / "new")
    (root / "grader/questions/area.md").write_text("---\ntype: check\n---\nOrders?\n")
    [pair] = await collect(
        refinery.replay("acme", Engine(root, decision=Answering()), agent="grader")
    )
    assert pair.candidate is None
    assert pair.error == (
        "InputError: engine: decision agent 'grader' already asks 'area' from its folder"
    )


async def test_a_fingerprint_selects_decision_runs_by_all_three_parts(
    tmp_path: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(graded(tmp_path), decision=Answering(), refinery=refinery)
    folder = await recorder.run("grader", "acme", "x")
    runtime = await recorder.run("grader", "acme", "x", questions={"area": AREA})
    again = await recorder.run("grader", "acme", "y", questions={"area": AREA})
    tasked = await recorder.run("grader", "acme", "x", task="tone")
    prints = [run.fingerprint for run in (folder, runtime, tasked)]
    assert len({p.agent for p in prints}) == 1
    assert len({p.questions for p in prints}) == 3
    engine = Engine(graded(tmp_path), decision=Answering())
    for run, expected in ((folder, [folder]), (runtime, [runtime, again]), (tasked, [tasked])):
        pairs = await collect(
            refinery.replay("acme", engine, agent="grader", fingerprint=run.fingerprint)
        )
        assert [pair.recorded.run_id for pair in pairs] == [r.run_id for r in expected]
    unasked = runtime.fingerprint.model_copy(update={"questions": None})
    assert await collect(refinery.replay("acme", engine, agent="grader", fingerprint=unasked)) == []


async def test_a_decision_run_with_media_replays_its_parts(
    tmp_path: Path, storage: Storage
) -> None:
    class Seeing(Answering):
        media_types = {"image/*"}

    refinery = Refinery(storage)
    recorder = Engine(graded(tmp_path), decision=Seeing(), refinery=refinery)
    parts = [Text(text="Urgent?"), Image(media_type="image/png", data=b"\x89PNG\r\n\x1a\n")]
    await recorder.run("grader", "acme", parts)
    seeing = Seeing()
    [pair] = await collect(
        refinery.replay("acme", Engine(graded(tmp_path), decision=seeing), agent="grader")
    )
    assert isinstance(pair.candidate, Decision) and pair.candidate.state == parts
    assert seeing.calls[0][0] == parts
    [refused] = await collect(
        refinery.replay("acme", Engine(graded(tmp_path), decision=Answering()), agent="grader")
    )
    assert refused.error == (
        "InputError: engine: decision agent 'grader' takes text only; got image/png"
    )


async def test_a_parent_whose_child_was_given_a_task_and_questions_replays_as_one_run(
    tmp_path: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    asked = delegating("grader", "Sorry!", task="tone", questions={"area": PICK})
    recorder = Engine(
        lead(tmp_path),
        generation=Scripted(asked, text_reply("Graded.")),
        decision=Answering(),
        refinery=refinery,
    )
    recorded = await recorder.run("lead", "acme", "Grade this")
    read = await refinery.run("acme", recorded.run_id)
    assert read == recorded and isinstance(read, Generation)
    [child] = read.children
    assert isinstance(child, Decision) and child.task == "tone" and child.runtime == ["area"]
    assert read.steps[0].args == {
        "agent": "grader",
        "input": "Sorry!",
        "task": "tone",
        "questions": {"area": PICK},
    }
    llm = Scripted(delegating("grader", "Sorry!", questions={"area": PICK}), text_reply("Again."))
    dm = Answering()
    [pair] = await collect(
        refinery.replay("acme", Engine(tmp_path, generation=llm, decision=dm), agent="lead")
    )
    assert pair.error is None and isinstance(pair.candidate, Generation)
    assert pair.candidate.reply == "Again." and pair.candidate.task is None
    [again] = pair.candidate.children
    assert isinstance(again, Decision) and again.task is None and again.runtime == ["area"]
    assert [list(q) for _, q in dm.calls] == [["urgent", "area"]]
    assert pair.candidate.fingerprint == recorded.fingerprint


async def test_a_failed_run_is_replayed_and_its_pair_says_it_failed(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    failed = generation_record(agent="support", reply="").model_copy(
        update={"error": "ModelError: anthropic: overloaded"}
    )
    await refinery.record(failed)
    candidate = Engine(fixtures, generation=Scripted(text_reply("Refunded.")), tools=TOOLS)
    async with aclosing(refinery.replay("acme", candidate, agent="support")) as pairs:
        [pair] = [p async for p in pairs]
    assert pair.recorded.error == "ModelError: anthropic: overloaded"
    assert isinstance(pair.candidate, Generation) and pair.candidate.reply == "Refunded."


EARLIER = [
    Message(role="user", content="Where is A-1?"),
    Message(role="assistant", content="On its way."),
    Message(role="user", content="And A-2?"),
    Message(role="assistant", content="Delivered."),
    Message(role="user", content="Charged twice for A-1."),
    Message(role="assistant", content="Refunded."),
]


class Held:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages
        self.reads: list[tuple[int, int | None]] = []

    async def read(self, start: int, stop: int | None = None) -> Sequence[object]:
        self.reads.append((start, stop))
        return self.messages[start:stop]


async def replayed(
    refinery: Refinery,
    candidate: Engine,
    agent: str,
    conversation: Callable[[Run], Sequence[object] | HistoryLike],
) -> Replay:
    async with aclosing(
        refinery.replay("acme", candidate, agent=agent, conversation=conversation)
    ) as pairs:
        [pair] = [p async for p in pairs]
    return pair


async def test_a_windowed_run_replays_with_its_conversation_and_history(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures, generation=Scripted(text_reply("Still refunded.")), tools=TOOLS, refinery=refinery
    )
    recorded = await recorder.run("support", "acme", "Is it refunded?", history=EARLIER, start=4)
    assert isinstance(recorded, Generation)
    later = [Message(role="user", content="Later."), Message(role="assistant", content="Later.")]
    whole = [*EARLIER, *recorded.added, *later]
    window = said([*EARLIER[4:], Message(role="user", content="Is it refunded?")])
    held = Held(whole)
    for source in (whole, deque(whole), held):
        llm = Scripted(text_reply("Yes."))
        candidate = Engine(fixtures, generation=llm, tools=TOOLS)
        pair = await replayed(refinery, candidate, "support", lambda run, kept=source: kept)
        assert pair.error is None and isinstance(pair.candidate, Generation)
        assert pair.candidate.start == 4
        assert said(llm.calls[0][1]) == window
        assert "History" in [spec.name for spec in llm.calls[0][2]]
    assert held.reads == [(4, 6)]


async def test_a_routed_run_replays_its_generation_run_with_the_conversation(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures,
        generation=Scripted(text_reply("Rebooted.")),
        decision=Desk(),
        tools=TOOLS,
        refinery=refinery,
    )
    await recorder.run("frontdesk", "acme", "It is down again.", history=EARLIER, start=4)
    llm = Scripted(text_reply("Try again."))
    candidate = Engine(fixtures, generation=llm, decision=Desk(), tools=TOOLS)
    seen: list[Run] = []

    def conversation(run: Run) -> list[Message]:
        seen.append(run)
        return EARLIER

    pair = await replayed(refinery, candidate, "frontdesk", conversation)
    assert pair.error is None
    assert "History" in [spec.name for spec in llm.calls[0][2]]
    assert isinstance(pair.candidate, Decision)
    assert isinstance(pair.candidate.routed_to, Generation) and pair.candidate.routed_to.start == 4
    [called] = seen
    assert isinstance(called, Decision) and called.agent == "frontdesk"
    assert said(llm.calls[0][1]) == said(
        [*EARLIER[4:], Message(role="user", content="It is down again.")]
    )


async def test_a_conversation_that_raises_gives_the_pair_an_error(
    fixtures: Path, storage: Storage
) -> None:
    def missing(run: Run) -> list[Message]:
        raise LookupError("no conversation for this run")

    refinery = Refinery(storage)
    recorder = Engine(
        fixtures, generation=Scripted(text_reply("Ok.")), tools=TOOLS, refinery=refinery
    )
    await recorder.run("support", "acme", "Hi.", history=EARLIER, start=4)
    candidate = Engine(fixtures, generation=Scripted(), tools=TOOLS)
    pair = await replayed(refinery, candidate, "support", missing)
    assert pair.error == "LookupError: no conversation for this run"


async def test_a_conversation_shorter_than_the_recorded_start_gives_the_pair_an_error(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures, generation=Scripted(text_reply("Ok.")), tools=TOOLS, refinery=refinery
    )
    await recorder.run("support", "acme", "Hi.", history=EARLIER, start=4)
    candidate = Engine(fixtures, generation=Scripted(), tools=TOOLS)
    for short, end in (
        (EARLIER[:2], 2),
        (Held(EARLIER[:2]), 4),
        (EARLIER[:5], 5),
        (Held(EARLIER[:5]), 5),
    ):
        pair = await replayed(refinery, candidate, "support", lambda run, kept=short: kept)
        assert pair.error == (
            f"InputError: refinery: the conversation has no message at position {end}, "
            "before the recorded input at position 6"
        )


async def test_history_in_a_replay_reads_before_the_input_and_never_past_it(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures, generation=Scripted(text_reply("Still refunded.")), tools=TOOLS, refinery=refinery
    )
    recorded = await recorder.run("support", "acme", "Is it refunded?", history=EARLIER, start=4)
    assert isinstance(recorded, Generation)
    later = [Message(role="user", content="Later."), Message(role="assistant", content="Later.")]
    whole = [*EARLIER, *recorded.added, *later]
    for source in (whole, Held(whole)):
        llm = Scripted(
            Reply(tool_calls=[ToolCall(id="h1", name="History", args={})], stop_reason="tool_use"),
            text_reply("Yes."),
        )
        candidate = Engine(fixtures, generation=llm, tools=TOOLS)
        pair = await replayed(refinery, candidate, "support", lambda run, kept=source: kept)
        assert pair.error is None
        [result] = [m for m in llm.calls[1][1] if m.role == "tool"][-1:]
        seen = json.dumps(result.model_dump(mode="json"))
        assert "Where is A-1?" in seen
        assert "Is it refunded?" not in seen and "Later." not in seen


async def test_a_source_that_ignores_stop_never_passes_the_cut(
    fixtures: Path, storage: Storage
) -> None:
    class Sloppy:
        def __init__(self, messages: list[Message]) -> None:
            self.messages = messages

        async def read(self, start: int, stop: int | None = None) -> Sequence[object]:
            return self.messages[start:]

    refinery = Refinery(storage)
    recorder = Engine(
        fixtures, generation=Scripted(text_reply("Still refunded.")), tools=TOOLS, refinery=refinery
    )
    recorded = await recorder.run("support", "acme", "Is it refunded?", history=EARLIER, start=4)
    assert isinstance(recorded, Generation)
    later = [Message(role="user", content="Later."), Message(role="assistant", content="Later.")]
    whole = [*EARLIER, *recorded.added, *later]
    llm = Scripted(text_reply("Yes."))
    candidate = Engine(fixtures, generation=llm, tools=TOOLS)
    pair = await replayed(refinery, candidate, "support", lambda run: Sloppy(whole))
    assert pair.error is None
    assert said(llm.calls[0][1]) == said(
        [*EARLIER[4:], Message(role="user", content="Is it refunded?")]
    )


async def test_a_decision_run_that_routes_nowhere_never_calls_conversation(
    fixtures: Path, storage: Storage
) -> None:
    other = Fixed(
        {
            "intent": ChoiceAnswer(
                choice="other", probabilities={"billing": 0, "technical": 0, "other": 1}
            )
        }
    )
    refinery = Refinery(storage)
    recorder = Engine(
        fixtures, generation=Scripted(), decision=other, tools=TOOLS, refinery=refinery
    )
    recorded = await recorder.run("frontdesk", "acme", "Hello.")
    assert isinstance(recorded, Decision) and recorded.routed_to is None
    called: list[Run] = []

    def conversation(run: Run) -> list[Message]:
        called.append(run)
        return EARLIER

    candidate = Engine(fixtures, generation=Scripted(), decision=other, tools=TOOLS)
    pair = await replayed(refinery, candidate, "frontdesk", conversation)
    assert pair.error is None and called == []


async def test_a_workspace_of_another_scope_is_refused_by_replay(
    fixtures: Path, storage: Storage, tmp_path: Path
) -> None:
    work = Workspace(Local(tmp_path), scope="acme")
    with pytest.raises(
        InputError, match="^refinery: workspace belongs to scope 'acme', not 'globex'$"
    ):
        Refinery(storage).replay("globex", candidate(fixtures), agent="support", workspace=work)
