import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import Answer, CheckAnswer, ChoiceAnswer, Reply, ScoreAnswer, ToolCall
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.refinery.labels import Review
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import (
    TOOLS,
    Fixed,
    MemoryBucket,
    Scripted,
    decision_record,
    generation_record,
    text_reply,
)

TRIAGE: dict[str, Answer] = {
    "urgent": CheckAnswer(probability=0.9),
    "severity": ScoreAnswer(score=3, probabilities=[0, 0, 0, 1]),
}
TAGS = {"commit": "4f2c9e1-sentinel", "arm-sentinel": "b"}


class Recorded:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


def _delegate(agent: str, text: str) -> Reply:
    return Reply(
        tool_calls=[ToolCall(id="d1", name="Delegate", args={"agent": agent, "input": text})],
        stop_reason="tool_use",
    )


def _technical(state: object) -> dict[str, Answer]:
    return {
        "intent": ChoiceAnswer(
            choice="technical", probabilities={"technical": 0.9, "billing": 0.05, "other": 0.05}
        )
    }


async def test_a_generation_run_carries_its_tags(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(text_reply("ok"), text_reply("ok")), tools=TOOLS)
    run = await engine.run("billing", "acme", "x", tags=TAGS)
    assert isinstance(run, Generation) and run.tags == TAGS
    untagged = await engine.run("billing", "acme", "x")
    assert untagged.tags == {}


async def test_a_decision_run_carries_its_tags(fixtures: Path) -> None:
    engine = Engine(fixtures, decision=Fixed(TRIAGE))
    run = await engine.run("triage", "acme", "It is down", tags=TAGS)
    assert isinstance(run, Decision) and run.tags == TAGS
    assert (await engine.run("triage", "acme", "It is down")).tags == {}


async def test_tags_never_reach_a_model(fixtures: Path) -> None:
    llm = Scripted(_delegate("triage", "It is down"), text_reply("Escalating."))
    dm = Fixed(TRIAGE)
    engine = Engine(fixtures, generation=llm, decision=dm, tools=TOOLS)
    await engine.run("support", "acme", "It is down", tags=TAGS)
    sent = json.dumps(
        [
            [
                system,
                [m.model_dump(mode="json") for m in messages],
                [t.model_dump(mode="json") for t in tools],
            ]
            for system, messages, tools in llm.calls
        ]
        + [[state, {k: repr(q) for k, q in questions.items()}] for state, questions in dm.calls]
    )
    assert len(llm.calls) == 2 and len(dm.calls) == 1
    for word in ("sentinel", "tags"):
        assert word not in sent


async def test_delegated_children_inherit_the_tags(fixtures: Path) -> None:
    engine = Engine(
        fixtures,
        generation=Scripted(_delegate("triage", "It is down"), text_reply("ok")),
        decision=Fixed(TRIAGE),
        tools=TOOLS,
    )
    run = await engine.run("support", "acme", "It is down", tags=TAGS)
    assert isinstance(run, Generation)
    [child] = run.children
    assert child.tags == TAGS


async def test_routed_runs_and_their_children_inherit_the_tags(fixtures: Path) -> None:
    answers: list[Any] = [_technical, lambda state: TRIAGE]
    dm = Fixed(lambda state: answers.pop(0)(state))
    llm = Scripted(_delegate("triage", "It is down"), text_reply("Handled."))
    engine = Engine(fixtures, generation=llm, decision=dm, tools=TOOLS)
    run = await engine.run("frontdesk", "acme", "It is down", tags=TAGS)
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.tags == run.routed_to.tags == run.routed_to.children[0].tags == TAGS


async def test_an_agent_as_a_tool_tags_the_runs_it_starts(fixtures: Path) -> None:
    recorder = Recorded()
    engine = Engine(
        fixtures, generation=Scripted(text_reply("Refunded.")), tools=TOOLS, refinery=recorder
    )
    consult = engine.tool("billing", scope="acme", tags=TAGS)
    await consult.dispatch("billing", {"input": "Charged twice"})
    [run] = recorder.runs
    assert run.tags == TAGS
    with pytest.raises(InputError, match="^engine: "):
        engine.tool("billing", scope="acme", tags={"": "x"})


async def test_engine_tools_tag_a_dispatched_delegate(fixtures: Path) -> None:
    recorder = Recorded()
    engine = Engine(
        fixtures,
        generation=Scripted(text_reply("ok")),
        decision=Fixed(TRIAGE),
        tools=TOOLS,
        refinery=recorder,
    )
    surface = EngineTools(engine, agent="support", scope="acme", tags=TAGS)
    await surface.dispatch("Delegate", {"agent": "triage", "input": "It is down"})
    [run] = recorder.runs
    assert run.agent == "triage" and run.tags == TAGS


@pytest.mark.parametrize(
    "tags",
    [
        {"": "x"},
        {"arm": 1},
        {1: "x"},
        {"k" * 65: "x"},
        {"arm": "v" * 257},
        {f"k{i}": "x" for i in range(33)},
        ["arm", "b"],
    ],
    ids=[
        "empty-key",
        "int-value",
        "int-key",
        "long-key",
        "long-value",
        "too-many",
        "not-a-mapping",
    ],
)
async def test_invalid_tags_are_refused_before_any_model_call(fixtures: Path, tags: Any) -> None:
    llm = Scripted()
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
    with pytest.raises(InputError, match="^engine: tags"):
        await engine.run("billing", "acme", "x", tags=tags)
    assert llm.calls == []


async def test_tags_at_the_limits_are_accepted(fixtures: Path) -> None:
    tags = {f"{i:02d}" + "k" * 62: "v" * 256 for i in range(32)}
    engine = Engine(fixtures, generation=Scripted(text_reply("ok")), tools=TOOLS)
    assert (await engine.run("billing", "acme", "x", tags=tags)).tags == tags


def _lines(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode().splitlines()]


async def test_every_dataset_row_carries_the_tags(storage: Storage) -> None:
    refinery = Refinery(storage)
    generation = generation_record(reply="Sorry, no.").model_copy(update={"tags": TAGS})
    decision = decision_record(agent="frontdesk").model_copy(update={"tags": {"arm": "a"}})
    await refinery.record(generation)
    await refinery.record(decision)
    await refinery.review("acme", generation.run_id, Review(correction="Refund in full."))
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    rows = {key.split("/")[2]: _lines(data) for key, data in bucket.objects.items()}
    assert [row["tags"] for row in rows["generation"]] == [TAGS]
    assert [row["tags"] for row in rows["preference"]] == [TAGS]
    assert [row["tags"] for row in rows["decision"]] == [{"arm": "a"}]
