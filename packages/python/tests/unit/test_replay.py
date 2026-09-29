import asyncio
import json
import shutil
from contextlib import aclosing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError, NotFoundError
from rfnry_engine.models import (
    CheckAnswer,
    Image,
    Message,
    Reply,
    ScoreAnswer,
    Text,
    ToolCall,
    ToolSpec,
)
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.refinery.holdout import is_holdout
from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.labels import Label, Outcome, Replay, Review
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import LabelRow
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Workspace

from .fakes import TOOLS, Desk, Fixed, Scripted, delegate_triage, generation_record, text_reply

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


async def test_it_replays_the_top_level_runs_of_the_agent_oldest_first(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def test_holdout_selects_by_the_same_hash_and_percent_as_export(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def test_labeled_keeps_runs_a_person_or_the_world_labeled(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def test_tags_select_by_containment(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def test_a_fingerprint_selects_one_recorded_version(fixtures: Path, tmp_path: Path) -> None:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    refinery = Refinery(InMemoryStorage())
    old = Engine(root, generation=Scripted(text_reply("a")), tools=TOOLS, refinery=refinery)
    before = await old.run("support", "acme", "hi")
    (root / "support/rules/refunds.md").write_text("Refund only the second charge.\n")
    new = Engine(root, generation=Scripted(text_reply("b")), tools=TOOLS, refinery=refinery)
    after = await new.run("support", "acme", "hi")
    await put(refinery, generation_record(), "unprinted", 0)
    assert before.fingerprint is not None and after.fingerprint is not None
    assert before.fingerprint != after.fingerprint
    engine = candidate(fixtures)
    for version, run in ((before.fingerprint, before), (after.fingerprint, after)):
        pairs = await collect(refinery.replay("acme", engine, agent="support", fingerprint=version))
        assert [pair.recorded.run_id for pair in pairs] == [run.run_id]
    assert len(await collect(refinery.replay("acme", engine, agent="support"))) == 3


async def test_the_candidate_is_sent_the_input_history_task_and_context_as_recorded(
    fixtures: Path,
) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def test_expect_and_approve_are_the_caller_s(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
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
) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def test_a_decision_run_replays_its_state_and_what_its_route_was_given(
    fixtures: Path,
) -> None:
    refinery = Refinery(InMemoryStorage())
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
) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def test_a_replay_records_nothing_and_labels_nothing(fixtures: Path) -> None:
    storage = InMemoryStorage()
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


async def test_concurrency_bounds_the_runs_in_flight_and_order_holds(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def test_one_at_a_time_runs_the_next_only_when_the_last_is_taken(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def test_closing_a_replay_cancels_the_runs_in_flight(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
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


async def recall(args: dict[str, Any]) -> str:
    return "nothing remembered"


async def test_the_tools_and_workspace_a_run_had_are_the_caller_s(tmp_path: Path) -> None:
    (tmp_path / "coder").mkdir()
    (tmp_path / "coder/AGENT.md").write_text(
        "---\nname: coder\nkind: generation\nworkspace: true\n---\nYou code.\n"
    )
    (tmp_path / "work").mkdir()
    work = Workspace(tmp_path / "work", unsandboxed=True)
    memory = Tools([Tool("recall", "Recall a fact", body=recall)])
    refinery = Refinery(InMemoryStorage())
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


async def test_a_bad_call_is_refused_before_anything_is_read(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
    engine = candidate(fixtures)
    with pytest.raises(InputError, match="^refinery: scope is required$"):
        refinery.replay("", engine, agent="support")
    with pytest.raises(InputError, match="^refinery: holdout_percent must be 0-100, got 101$"):
        refinery.replay("acme", engine, agent="support", holdout_percent=101)
    with pytest.raises(InputError, match="^refinery: limit must be at least 1, got 0$"):
        refinery.replay("acme", engine, agent="support", limit=0)
    with pytest.raises(InputError, match="^refinery: concurrency must be at least 1, got 0$"):
        refinery.replay("acme", engine, agent="support", concurrency=0)
    with pytest.raises(InputError, match="^refinery: tags must hold at least one tag$"):
        refinery.replay("acme", engine, agent="support", tags={})
    with pytest.raises(NotFoundError, match="^refinery: the candidate has no agent 'ghost'$"):
        refinery.replay("acme", engine, agent="ghost")
