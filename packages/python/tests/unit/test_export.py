import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import Document, Image, Message, Part, Reply, Text, ToolCall
from rfnry_engine.records import Generation, Step
from rfnry_engine.refinery.holdout import is_holdout
from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.labels import Outcome, Review
from rfnry_engine.refinery.refinery import Refinery, flatten
from rfnry_engine.refinery.storage import RunRow
from rfnry_engine.tools.tools import Tool

from .fakes import (
    TOOLS,
    MemoryBucket,
    Scripted,
    decision_record,
    generation_record,
    lookup_order,
    text_reply,
)


def test_holdout_is_derived_deterministic_and_roughly_twenty_percent() -> None:
    ids = [f"run-{i}" for i in range(2000)]
    assert all(is_holdout(i) is is_holdout(i) for i in ids)
    holdout = sum(is_holdout(i) for i in ids)
    assert 340 <= holdout <= 460
    assert (
        is_holdout("x", holdout_percent=0) is False and is_holdout("x", holdout_percent=100) is True
    )


def _lines(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode().splitlines()]


async def test_export_writes_three_datasets_with_labels_and_holdout() -> None:
    refinery = Refinery(InMemoryStorage())
    generation = generation_record(reply="Sorry, no.")
    decision = decision_record(agent="frontdesk", probability=0.8)
    await refinery.record(generation)
    await refinery.record(decision)
    await refinery.review(
        "acme", generation.run_id, Review(stance="objection", correction="Refund in full.")
    )
    await refinery.outcome(
        "acme", generation.run_id, Outcome(status="failed", detail="Customer escalated.")
    )
    await refinery.review("acme", decision.run_id, Review(correction={"urgent": False}))
    bucket = MemoryBucket()
    exported = await refinery.export("acme", bucket)
    assert exported.rows == {"generation": 1, "preference": 1, "decision": 1}
    labels = await refinery.storage.labels("acme")
    runs = await refinery.storage.runs("acme")
    assert exported.watermark == max(
        [*(run.recorded for run in runs), *(label.at for label in labels)]
    )
    keys = sorted(bucket.objects)
    assert [k.split("/")[:3] for k in keys] == [
        ["acme", "frontdesk", "decision"],
        ["acme", "support", "generation"],
        ["acme", "support", "preference"],
    ]
    assert all(len(k.split("/")) == 4 and k.endswith(".jsonl") for k in keys)
    [decision_row] = _lines(bucket.objects[keys[0]])
    assert decision_row["holdout"] is is_holdout(decision.run_id)
    assert decision_row["question_id"] == "urgent" and decision_row["answer"] == 0.8
    assert (
        decision_row["truth"] is False
        and decision_row["truth_source"] == "human"
        and decision_row["outcome"] is None
    )
    [generation_row] = _lines(bucket.objects[keys[1]])
    assert (
        generation_row["holdout"] is is_holdout(generation.run_id)
        and generation_row["stopped"] is None
    )
    assert generation_row["reply"] == "Sorry, no." and sorted(
        label["source"] for label in generation_row["labels"]
    ) == ["human", "world"]
    human, world = sorted(generation_row["labels"], key=lambda label: label["source"])
    assert human["correction"] == "Refund in full." and human["detail"] is None
    assert (
        world["outcome"] == "failed"
        and world["detail"] == "Customer escalated."
        and world["correction"] is None
    )
    [preference_row] = _lines(bucket.objects[keys[2]])
    assert preference_row == {
        "run_id": generation.run_id,
        "holdout": is_holdout(generation.run_id),
        "tags": {},
        "fingerprint": None,
        "prompt": {"system": generation.system, "messages": generation_row["messages"]},
        "chosen": "Refund in full.",
        "rejected": "Sorry, no.",
        "label_id": preference_row["label_id"],
        "at": preference_row["at"],
    }


async def test_export_since_the_watermark_is_incremental() -> None:
    refinery = Refinery(InMemoryStorage())
    old = generation_record()
    await refinery.record(old)
    first = await refinery.export("acme", MemoryBucket())
    later = generation_record().model_copy(update={"at": datetime.now(UTC) + timedelta(seconds=1)})
    await refinery.record(later)
    bucket = MemoryBucket()
    second = await refinery.export("acme", bucket, since=first.watermark)
    assert (
        second.rows["generation"] == 1
        and _lines(next(iter(bucket.objects.values())))[0]["run_id"] == later.run_id
    )
    assert (await refinery.export("acme", MemoryBucket(), since=second.watermark)).rows == {}


async def test_a_run_still_working_at_an_export_is_in_the_next_one(fixtures: Path) -> None:
    started, gate = asyncio.Event(), asyncio.Event()

    async def slow(args: dict[str, Any]) -> str:
        started.set()
        await gate.wait()
        return "2026-09-22T00:00:00Z"

    class Model:
        async def generate(self, system: str, messages: list[Message], tools: list[Any]) -> Reply:
            if messages[-1].content == "long":
                return Reply(tool_calls=[ToolCall(id="c1", name="now")], stop_reason="tool_use")
            return text_reply("answer")

    refinery = Refinery(InMemoryStorage())
    engine = Engine(
        fixtures,
        generation=Model(),
        tools=[Tool("lookup_order", body=lookup_order), Tool("now", "The time", body=slow)],
        refinery=refinery,
    )
    long = asyncio.create_task(engine.run("support", "acme", "long"))
    await started.wait()
    await engine.run("support", "acme", "quick")
    first = await refinery.export("acme", MemoryBucket())
    gate.set()
    late = await long
    assert first.watermark is not None and late.at < first.watermark
    bucket = MemoryBucket()
    second = await refinery.export("acme", bucket, since=first.watermark)
    assert second.rows == {"generation": 1}
    assert [row["run_id"] for row in _lines(next(iter(bucket.objects.values())))] == [late.run_id]
    assert (await refinery.export("acme", MemoryBucket(), since=second.watermark)).rows == {}


async def test_a_label_added_after_export_re_exports_its_run() -> None:
    refinery = Refinery(InMemoryStorage())
    reviewed = generation_record(reply="Sorry, no.")
    untouched = generation_record()
    await refinery.record(reviewed)
    await refinery.record(untouched)
    first = await refinery.export("acme", MemoryBucket())
    label = await refinery.review("acme", reviewed.run_id, Review(correction="Refund in full."))
    bucket = MemoryBucket()
    second = await refinery.export("acme", bucket, since=first.watermark)
    assert second.rows == {"generation": 1, "preference": 1} and second.watermark == label.at
    rows = {key.split("/")[2]: _lines(data) for key, data in bucket.objects.items()}
    assert [row["run_id"] for row in rows["generation"]] == [reviewed.run_id]
    assert rows["generation"][0]["labels"][0]["source"] == "human"
    assert rows["preference"][0]["chosen"] == "Refund in full."
    assert all(
        row["holdout"] is is_holdout(reviewed.run_id) for part in rows.values() for row in part
    )
    assert (await refinery.export("acme", MemoryBucket(), since=second.watermark)).rows == {}


async def test_conflicts_are_two_distinct_corrections_on_one_run() -> None:
    refinery = Refinery(InMemoryStorage())
    run = generation_record()
    await refinery.record(run)
    await refinery.review("acme", run.run_id, Review(correction="A"))
    await refinery.review("acme", run.run_id, Review(correction="A"))
    assert await refinery.conflicts("acme") == []
    await refinery.review("acme", run.run_id, Review(correction="B"))
    [conflict] = await refinery.conflicts("acme")
    assert conflict.run_id == run.run_id and sorted(c[2] for c in conflict.corrections) == [
        "A",
        "A",
        "B",
    ]


async def test_holdout_percent_marks_every_row_and_never_moves_between_exports() -> None:
    refinery = Refinery(InMemoryStorage())
    run = generation_record(reply="Sorry, no.")
    await refinery.record(run)
    await refinery.review("acme", run.run_id, Review(correction="Refund in full."))
    everything = MemoryBucket()
    await refinery.export("acme", everything, holdout_percent=100)
    assert all(
        row["holdout"] is True for data in everything.objects.values() for row in _lines(data)
    )
    nothing = MemoryBucket()
    await refinery.export("acme", nothing, holdout_percent=0)
    assert all(row["holdout"] is False for data in nothing.objects.values() for row in _lines(data))
    first, second = MemoryBucket(), MemoryBucket()
    await refinery.export("acme", first)
    await refinery.export("acme", second)
    flags = [
        [row["holdout"] for data in bucket.objects.values() for row in _lines(data)]
        for bucket in (first, second)
    ]
    assert flags[0] == flags[1] == [is_holdout(run.run_id)] * 2


async def test_a_holdout_percent_outside_0_to_100_is_refused_even_with_nothing_to_export() -> None:
    empty, recorded = Refinery(InMemoryStorage()), Refinery(InMemoryStorage())
    await recorded.record(generation_record(reply="Sorry, no."))
    for refinery in (empty, recorded):
        for bad in (-1, 101):
            bucket = MemoryBucket()
            expected = f"^refinery: holdout_percent must be 0-100, got {bad}$"
            with pytest.raises(InputError, match=expected):
                await refinery.export("acme", bucket, holdout_percent=bad)
            assert bucket.objects == {}


class Capped(InMemoryStorage):
    async def runs(self, scope: str, **kwargs: Any) -> list[RunRow]:
        rows = await super().runs(scope, **{**kwargs, "limit": None})
        return rows if kwargs.get("limit") is None else rows[:2]


async def test_export_reads_every_run_without_a_cap() -> None:
    storage = Capped()
    refinery = Refinery(storage)
    for _ in range(3):
        await refinery.record(generation_record())
    runs = await storage.runs("acme", limit=None)
    assert len(runs) == 3 and len(await storage.runs("acme", limit=100_000)) == 2
    exported = await refinery.export("acme", MemoryBucket())
    assert exported.rows == {"generation": 3} and exported.watermark == max(
        r.recorded for r in runs
    )


async def test_a_run_with_parts_is_recorded_and_exported_with_its_bytes_as_base64() -> None:
    png = b"\x89PNG\r\n\x1a\n\xff\xfe"
    parts: list[Part] = [Text(text="What is this?"), Image(media_type="image/png", data=png)]
    step = Step(
        tool="photo",
        args={},
        result=[Document(media_type="application/pdf", data=b"%PDF")],
        is_error=False,
        at=datetime.now(UTC),
        duration_ms=1,
    )
    run = generation_record().model_copy(
        update={"input": parts, "messages": [Message(role="user", content=parts)], "steps": [step]}
    )
    refinery = Refinery(InMemoryStorage())
    await refinery.record(run)
    stored = await refinery.storage.get_run("acme", run.run_id)
    assert stored is not None
    assert Generation.model_validate(stored.body).messages == run.messages
    assert Generation.model_validate(stored.body).steps[0].result == step.result
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    [row] = _lines(next(data for key, data in bucket.objects.items() if "/generation/" in key))
    assert row["messages"][0]["content"][1] == {
        "type": "image",
        "media_type": "image/png",
        "data": base64.b64encode(png).decode(),
        "url": None,
    }
    assert row["steps"][0]["result"][0]["data"] == base64.b64encode(b"%PDF").decode()


async def test_a_generation_row_carries_the_tools_offered_and_the_conversation_in_order(
    fixtures: Path,
) -> None:
    llm = Scripted(
        Reply(
            text="Checking.",
            tool_calls=[
                ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"}),
                ToolCall(id="c2", name="Ghost"),
            ],
            stop_reason="tool_use",
        ),
        text_reply("Refunded."),
    )
    refinery = Refinery(InMemoryStorage())
    engine = Engine(fixtures, generation=llm, tools=TOOLS, refinery=refinery)
    run = await engine.run(
        "support",
        "acme",
        "Charged twice.",
        history=[Message(role="user", content="hi"), Message(role="assistant", content="hello")],
    )
    assert isinstance(run, Generation)
    offered = llm.calls[0][2]
    assert run.tools == offered and {"lookup_order", "now", "Read", "Delegate"} <= {
        spec.name for spec in offered
    }
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    [row] = _lines(next(data for key, data in bucket.objects.items() if "/generation/" in key))
    assert row["tools"] == [spec.model_dump(mode="json") for spec in offered]
    assert set(row["tools"][0]) == {"name", "description", "input_schema"}
    assert [(m["role"], m["content"]) for m in row["conversation"]] == [
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "Charged twice."),
        ("assistant", "Checking."),
        ("tool", '{"order": "A-1"}'),
        ("tool", run.steps[1].result),
        ("assistant", "Refunded."),
    ]
    assert row["conversation"][3]["tool_calls"] == [
        {"id": "c1", "name": "lookup_order", "args": {"order_id": "A-1"}},
        {"id": "c2", "name": "Ghost", "args": {}},
    ]
    assert [(m["tool_call_id"], m["is_error"]) for m in row["conversation"][4:6]] == [
        ("c1", False),
        ("c2", True),
    ]
    assert row["conversation"] == [m.model_dump(mode="json") for m in run.conversation]
    assert (
        row["messages"] == [m.model_dump(mode="json") for m in run.messages]
        and row["reply"] == "Refunded."
    )


async def test_a_run_offered_no_tools_exports_an_empty_list() -> None:
    refinery = Refinery(InMemoryStorage())
    run = generation_record(reply="Sure.")
    assert run.tools == []
    await refinery.record(run)
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    [row] = _lines(next(iter(bucket.objects.values())))
    assert row["tools"] == []
    assert row["conversation"] == [
        {
            "role": "user",
            "content": "hi",
            "tool_calls": [],
            "tool_call_id": None,
            "reasoning": None,
            "is_error": False,
        },
        {
            "role": "assistant",
            "content": "Sure.",
            "tool_calls": [],
            "tool_call_id": None,
            "reasoning": None,
            "is_error": False,
        },
    ]


async def test_a_run_stored_before_tools_were_recorded_still_loads_and_exports() -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    run = generation_record()
    [stored] = flatten(run, datetime.now(UTC))
    old = {key: value for key, value in stored.body.items() if key != "tools"}
    await storage.put_run("acme", stored.model_copy(update={"body": old}))
    assert Generation.model_validate(old).tools == []
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    [row] = _lines(next(iter(bucket.objects.values())))
    assert row["tools"] == [] and row["conversation"] == old["conversation"]
    bare = {key: value for key, value in old.items() if key != "conversation"}
    await storage.put_run("acme", stored.model_copy(update={"body": bare}))
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    [row] = _lines(next(iter(bucket.objects.values())))
    assert row["tools"] == [] and row["conversation"] == []


async def test_a_generation_row_carries_the_context_its_run_was_given() -> None:
    refinery = Refinery(InMemoryStorage())
    given = generation_record().model_copy(update={"context": "Acme is on the enterprise plan."})
    bare = generation_record()
    await refinery.record(given)
    await refinery.record(bare)
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    [key] = [k for k in bucket.objects if "/generation/" in k]
    rows = {row["run_id"]: row for row in _lines(bucket.objects[key])}
    assert rows[given.run_id]["context"] == "Acme is on the enterprise plan."
    assert rows[bare.run_id]["context"] is None
