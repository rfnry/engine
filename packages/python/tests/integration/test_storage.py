import asyncio
import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import asyncpg
import pytest
from unit.fakes import (
    PICK,
    Answering,
    Desk,
    MemoryBucket,
    Scripted,
    ask,
    decision_record,
    delegate_triage,
    delegating,
    generation_record,
    lead,
    sorter,
    text_reply,
)

from rfnry_engine.engine import Engine
from rfnry_engine.errors import NotFoundError, StorageError
from rfnry_engine.models import (
    Check,
    CheckAnswer,
    Choice,
    Document,
    Image,
    Message,
    Part,
    Reply,
    Text,
    ToolCall,
    ToolSpec,
    Usage,
)
from rfnry_engine.records import Decision, Fingerprint, Generation
from rfnry_engine.refinery.labels import Outcome, Review
from rfnry_engine.refinery.postgres import PostgresStorage
from rfnry_engine.refinery.refinery import Refinery, label_of
from rfnry_engine.refinery.storage import LabelRow, RunRow, Storage
from rfnry_engine.stop import Stop
from rfnry_engine.tools.tools import Tool, Tools

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


def _run(
    scope: str,
    run_id: str,
    *,
    agent: str = "support",
    kind: str = "generation",
    at: datetime | None = None,
    recorded: datetime | None = None,
    parent: str | None = None,
) -> RunRow:
    at = at or datetime.now(UTC)
    return RunRow(
        scope=scope,
        run_id=run_id,
        parent_run_id=parent,
        agent=agent,
        kind=kind,
        task=None,
        at=at,
        recorded=recorded or at,
        duration_ms=1,
        model="m",
        body={"reply": run_id, "nested": {"n": [1, 2]}},
    )


def _label(
    scope: str, label_id: str, run_id: str, *, source: str = "human", at: datetime | None = None
) -> LabelRow:
    return LabelRow(
        scope=scope,
        label_id=label_id,
        run_id=run_id,
        source=source,
        at=at or datetime.now(UTC),
        body={"source": source},
    )


async def test_put_get_upsert_roundtrip(storage: Storage) -> None:
    await storage.put_runs("a", [_run("a", "r1")])
    got = await storage.get_run("a", "r1")
    assert got is not None and got.body == {"reply": "r1", "nested": {"n": [1, 2]}}
    await storage.put_runs("a", [_run("a", "r1", agent="billing")])
    got = await storage.get_run("a", "r1")
    assert got is not None and got.agent == "billing"
    assert await storage.get_run("a", "ghost") is None


async def test_runs_filters_order_and_limit(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_runs("a", [_run("a", "old", at=t0 - timedelta(minutes=2))])
    await storage.put_runs(
        "a", [_run("a", "mid", agent="billing", kind="decision", at=t0 - timedelta(minutes=1))]
    )
    await storage.put_runs("a", [_run("a", "new", at=t0)])
    assert [r.run_id for r in await storage.runs("a")] == ["new", "mid", "old"]
    assert [r.run_id for r in await storage.runs("a", kind="generation")] == ["new", "old"]
    assert [r.run_id for r in await storage.runs("a", since=t0 - timedelta(minutes=1))] == ["new"]
    assert [r.run_id for r in await storage.runs("a", limit=2)] == ["new", "mid"]
    assert [r.run_id for r in await storage.runs("a", limit=None)] == ["new", "mid", "old"]
    await storage.put_label("a", _label("a", "l1", "new", source="judge"))
    assert [r.run_id for r in await storage.runs("a", unlabeled_by="judge")] == ["mid", "old"]
    assert [r.run_id for r in await storage.runs("a", unlabeled_by="teacher")] == [
        "new",
        "mid",
        "old",
    ]


async def test_runs_since_selects_what_was_recorded_after_it(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_runs(
        "a", [_run("a", "long", at=t0 - timedelta(minutes=5), recorded=t0 + timedelta(seconds=1))]
    )
    await storage.put_runs("a", [_run("a", "quick", at=t0, recorded=t0)])
    assert [r.run_id for r in await storage.runs("a", since=t0)] == ["long"]
    got = await storage.get_run("a", "long")
    assert got is not None and got.recorded == t0 + timedelta(seconds=1)
    assert got.at == t0 - timedelta(minutes=5)


async def test_an_export_finds_a_run_recorded_after_its_watermark(storage: Storage) -> None:
    refinery = Refinery(storage)
    long, quick = generation_record(scope="a"), generation_record(scope="a")
    await refinery.record(quick)
    first = await refinery.export("a", MemoryBucket())
    await refinery.record(long)
    assert first.watermark is not None and long.at < first.watermark
    bucket = MemoryBucket()
    second = await refinery.export("a", bucket, since=first.watermark)
    assert second.rows == {"generation": 1}
    [data] = bucket.objects.values()
    assert [json.loads(line)["run_id"] for line in data.decode().splitlines()] == [long.run_id]
    assert (await refinery.export("a", MemoryBucket(), since=second.watermark)).rows == {}


async def test_labels_filters_and_order(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_runs("a", [_run("a", "r1")])
    await storage.put_label("a", _label("a", "l2", "r1", source="world", at=t0))
    await storage.put_label("a", _label("a", "l1", "r1", at=t0 - timedelta(seconds=1)))
    await storage.put_label("a", _label("a", "l3", "r2", at=t0 + timedelta(seconds=1)))
    assert [label.label_id for label in await storage.labels("a")] == ["l1", "l2", "l3"]
    assert [label.label_id for label in await storage.labels("a", run_ids=["r1"])] == ["l1", "l2"]
    assert [label.label_id for label in await storage.labels("a", source="world")] == ["l2"]
    assert [label.label_id for label in await storage.labels("a", since=t0)] == ["l3"]


async def test_runs_by_id_and_labels_since_find_late_labels(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_runs("a", [_run("a", "r1", at=t0 - timedelta(minutes=2))])
    await storage.put_runs("a", [_run("a", "r2", at=t0 - timedelta(minutes=1))])
    await storage.put_runs("b", [_run("b", "r1", at=t0)])
    await storage.put_label("a", _label("a", "l1", "r1", at=t0 + timedelta(seconds=1)))
    late = {label.run_id for label in await storage.labels("a", since=t0)}
    assert late == {"r1"}
    assert [r.run_id for r in await storage.runs("a", run_ids=sorted(late))] == ["r1"]
    assert [r.run_id for r in await storage.runs("a", run_ids=["r1", "r2", "ghost"])] == [
        "r2",
        "r1",
    ]
    assert await storage.runs("a", run_ids=[]) == []
    assert [r.scope for r in await storage.runs("b", run_ids=["r1", "r2"])] == ["b"]


async def test_put_label_keeps_the_first_row_under_an_id_and_returns_it(
    storage: Storage,
) -> None:
    t0 = datetime.now(UTC)
    first = _label("a", "m1", "r1", at=t0)
    assert await storage.put_label("a", first) == first
    again = _label("a", "m1", "r2", source="world", at=t0 + timedelta(seconds=1))
    assert await storage.put_label("a", again.model_copy(update={"body": {"n": 2}})) == first
    assert await storage.labels("a") == [first]
    other = _label("b", "m1", "r2", source="world")
    assert await storage.put_label("b", other) == other
    assert await storage.labels("a") == [first] and await storage.labels("b") == [other]


async def test_puts_racing_on_one_label_id_store_one_row(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    rows = [
        _label("a", "m1", f"r{i}", at=t0 + timedelta(seconds=i)).model_copy(
            update={"body": {"i": i}}
        )
        for i in range(20)
    ]
    stored = await asyncio.gather(*(storage.put_label("a", row) for row in rows))
    assert len({row.label_id for row in stored}) == 1 and all(row == stored[0] for row in stored)
    assert stored[0] in rows and await storage.labels("a") == [stored[0]]


async def test_reviews_racing_on_one_label_id_store_one_label(storage: Storage) -> None:
    run = generation_record(scope="a")
    await Refinery(storage).record(run)
    reviews = [Review(stance="objection", correction=f"Refund {i}.") for i in range(10)]
    labels = await asyncio.gather(
        *(
            Refinery(storage).review("a", run.run_id, review, label_id="message-81")
            for review in reviews
        ),
        Refinery(storage).outcome("a", run.run_id, Outcome(status="failed"), label_id="message-82"),
        Refinery(storage).outcome("a", run.run_id, Outcome(status="failed"), label_id="message-82"),
    )
    reviewed, outcomes = labels[:10], labels[10:]
    assert all(label == reviewed[0] for label in reviewed) and reviewed[0].review in reviews
    assert outcomes[0] == outcomes[1] and outcomes[0].label_id == "message-82"
    stored = sorted(await storage.labels("a"), key=lambda row: row.label_id)
    assert [label_of(row) for row in stored] == [reviewed[0], outcomes[0]]


async def test_scope_mismatch_is_refused(storage: Storage) -> None:
    with pytest.raises(StorageError):
        await storage.put_runs("a", [_run("b", "r1")])
    with pytest.raises(StorageError):
        await storage.put_label("a", _label("b", "l1", "r1"))


async def test_delete_and_count(storage: Storage) -> None:
    await storage.put_runs("a", [_run("a", "r1")])
    await storage.put_label("a", _label("a", "l1", "r1"))
    await storage.put_runs("b", [_run("b", "r2")])
    assert await storage.count_scope("a") == 2
    assert await storage.delete_scope("a") == 2
    assert await storage.count_scope("a") == 0 and await storage.count_scope("b") == 1


async def test_bodies_round_trip_exactly(storage: Storage) -> None:
    body = {
        "zeta": 1,
        "a": {"refund": "r", "no": "n", "billing": "b"},
        "text": "a\x00b",
        "big": 1e16,
        "tiny": 1.5e-7,
    }
    await storage.put_runs("a", [_run("a", "r1").model_copy(update={"body": body})])
    await storage.put_label("a", _label("a", "l1", "r1").model_copy(update={"body": body}))
    got = await storage.get_run("a", "r1")
    assert got is not None
    for stored in (got.body, (await storage.labels("a"))[0].body):
        assert stored == body
        assert list(stored) == list(body) and list(stored["a"]) == ["refund", "no", "billing"]
        assert type(stored["big"]) is float


async def test_ties_order_by_code_point_whatever_the_collation(storage: Storage) -> None:
    at = datetime.now(UTC)
    ids = ["ab", "B", "a-b", "_z", "a"]
    for run_id in ids:
        await storage.put_runs("a", [_run("a", run_id, at=at)])
        await storage.put_label("a", _label("a", run_id, run_id, at=at))
    assert [r.run_id for r in await storage.runs("a")] == sorted(ids, reverse=True)
    assert [label.label_id for label in await storage.labels("a")] == sorted(ids)


async def test_a_body_with_parts_round_trips(storage: Storage) -> None:
    raw = bytes(range(256))
    conversation = [
        Message(role="user", content=[Text(text="look"), Image(media_type="image/png", data=raw)]),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="read_pdf", args={"path": "a.pdf"})],
        ),
        Message(
            role="tool",
            tool_call_id="c1",
            content=[Document(media_type="application/pdf", name="a.pdf", url="https://x/a.pdf")],
        ),
    ]
    body = {"conversation": [m.model_dump(mode="json") for m in conversation]}
    await storage.put_runs(
        "a",
        [
            RunRow(
                scope="a",
                run_id="p",
                parent_run_id=None,
                agent="support",
                kind="generation",
                task=None,
                at=datetime.now(UTC),
                recorded=datetime.now(UTC),
                duration_ms=1,
                model="m",
                body=body,
            )
        ],
    )
    got = await storage.get_run("a", "p")
    assert got is not None and got.body == body
    assert [Message.model_validate(m) for m in got.body["conversation"]] == conversation


async def test_a_generation_with_its_tools_round_trips(storage: Storage) -> None:
    at = datetime.now(UTC)
    run = Generation(
        run_id="g",
        agent="support",
        scope="a",
        at=at,
        duration_ms=1,
        model="m",
        fingerprint=Fingerprint(agent="a" * 16, tools="b" * 16, questions=None),
        input="hi",
        system="s",
        messages=[Message(role="user", content="hi")],
        conversation=[
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"})],
            ),
            Message(role="tool", tool_call_id="c1", content='{"order": "A-1"}'),
            Message(role="assistant", content="Found it."),
        ],
        tools=[
            ToolSpec(
                name="lookup_order",
                description="Fetch an order",
                input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
            )
        ],
        steps=[],
        calls=2,
        reply="Found it.",
        usage=Usage(),
    )
    refinery = Refinery(storage)
    await refinery.record(run)
    assert await refinery.run("a", "g") == run


async def test_tags_round_trip(storage: Storage) -> None:
    at = datetime.now(UTC)
    decision = Decision(
        run_id="d",
        agent="triage",
        scope="a",
        at=at,
        duration_ms=1,
        model="d",
        fingerprint=Fingerprint(agent="c" * 16, tools="d" * 16, questions="e" * 16),
        state="hi",
        questions={"urgent": Check(instructions="Urgent?")},
        answers={"urgent": CheckAnswer(probability=0.9)},
        usage=Usage(),
        tags={"arm": "b", "commit": "a81d07b"},
    )
    run = Generation(
        run_id="g",
        agent="support",
        scope="a",
        at=at,
        duration_ms=1,
        model="m",
        fingerprint=Fingerprint(agent="a" * 16, tools="b" * 16, questions=None),
        input="hi",
        system="s",
        messages=[Message(role="user", content="hi")],
        conversation=[Message(role="user", content="hi")],
        steps=[],
        calls=1,
        reply="ok",
        usage=Usage(),
        children=[decision],
        tags={"arm": "b", "commit": "a81d07b"},
    )
    refinery = Refinery(storage)
    await refinery.record(run)
    for run_id in ("g", "d"):
        got = await storage.get_run("a", run_id)
        assert got is not None and got.body["tags"] == {"arm": "b", "commit": "a81d07b"}
        assert (await refinery.run("a", run_id)).tags == {"arm": "b", "commit": "a81d07b"}


def _tagged(
    scope: str,
    run_id: str,
    tags: dict[str, str],
    *,
    parent: str | None = None,
    at: datetime | None = None,
    extra: dict[str, Any] | None = None,
) -> RunRow:
    row = _run(scope, run_id, parent=parent, at=at)
    return row.model_copy(update={"body": {**row.body, **(extra or {}), "tags": tags}})


async def test_run_tree_follows_parent_run_id_down_every_level(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    rows = [
        _run("a", "root", at=t0),
        _run("a", "child", parent="root", at=t0 + timedelta(seconds=1)),
        _run("a", "grandchild", parent="child", at=t0 + timedelta(seconds=2)),
        _run("a", "second", parent="root", at=t0 + timedelta(seconds=3)),
        _run("a", "other", at=t0 + timedelta(seconds=4)),
        _run("b", "root", at=t0),
        _run("b", "stranger", parent="root", at=t0),
    ]
    for row in rows:
        await storage.put_runs(row.scope, [row])
    tree = await storage.run_tree("a", run_ids=["root"])
    assert [r.run_id for r in tree] == ["root", "child", "grandchild", "second"]
    assert all(r.scope == "a" for r in tree) and tree[1].body == rows[1].body
    assert [r.run_id for r in await storage.run_tree("a", run_ids=["child"])] == [
        "child",
        "grandchild",
    ]
    assert [r.run_id for r in await storage.run_tree("a", run_ids=["child", "other"])] == [
        "child",
        "grandchild",
        "other",
    ]
    assert await storage.run_tree("a", run_ids=["ghost"]) == []
    assert await storage.run_tree("a", run_ids=[]) == []
    assert [r.run_id for r in await storage.run_tree("b", run_ids=["root"])] == [
        "root",
        "stranger",
    ]


async def test_run_tree_finds_orphans_and_survives_a_cycle(storage: Storage) -> None:
    await storage.put_runs("a", [_run("a", "orphan", parent="gone")])
    await storage.put_runs("a", [_run("a", "x", parent="y")])
    await storage.put_runs("a", [_run("a", "y", parent="x")])
    assert [r.run_id for r in await storage.run_tree("a", run_ids=["gone"])] == ["orphan"]
    assert {r.run_id for r in await storage.run_tree("a", run_ids=["x"])} == {"x", "y"}


async def test_run_tree_by_tags_is_containment_and_takes_descendants(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_runs("a", [_tagged("a", "r1", {"customer": "c-81", "arm": "b"}, at=t0)])
    await storage.put_runs("a", [_run("a", "r1-child", parent="r1", at=t0 + timedelta(seconds=1))])
    await storage.put_runs("a", [_tagged("a", "r2", {"customer": "c-82"}, at=t0)])
    await storage.put_runs("a", [_run("a", "untagged", at=t0)])
    await storage.put_runs("b", [_tagged("b", "r3", {"customer": "c-81"}, at=t0)])

    async def found(tags: dict[str, str]) -> list[str]:
        return [r.run_id for r in await storage.run_tree("a", tags=tags)]

    assert await found({"customer": "c-81"}) == ["r1", "r1-child"]
    assert await found({"arm": "b", "customer": "c-81"}) == ["r1", "r1-child"]
    assert await found({"customer": "c-81", "arm": "a"}) == []
    assert await found({"customer": "c-8"}) == []
    assert await found({"customer": "c-82"}) == ["r2"]
    assert [r.run_id for r in await storage.run_tree("a", run_ids=["r1"], tags={"arm": "b"})] == [
        "r1",
        "r1-child",
    ]


async def test_tags_match_beside_bodies_json_cannot_hold(storage: Storage) -> None:
    hostile = {"reply": "a\x00b \ud800 \udc00 퀀 \\u0000 \\\x00 \\ud800", "k\x00": "\x00"}
    await storage.put_runs("a", [_tagged("a", "nul", {"customer": "c-81"}, extra=hostile)])
    await storage.put_runs("a", [_tagged("a", "emoji", {"customer": "😀", "hangul": "퀀"})])
    await storage.put_runs("a", [_tagged("a", "other", {"customer": "😁"})])
    assert [r.run_id for r in await storage.run_tree("a", tags={"customer": "c-81"})] == ["nul"]
    assert [r.run_id for r in await storage.run_tree("a", tags={"customer": "😀"})] == ["emoji"]
    assert [r.run_id for r in await storage.run_tree("a", tags={"hangul": "퀀"})] == ["emoji"]
    got = await storage.get_run("a", "nul")
    assert got is not None and got.body["reply"] == hostile["reply"]
    assert await storage.delete_runs("a", tags={"customer": "c-81"}) == 1
    assert {r.run_id for r in await storage.runs("a")} == {"emoji", "other"}


async def test_delete_runs_takes_descendants_and_labels_in_its_scope(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_runs("a", [_tagged("a", "r1", {"customer": "c-81"}, at=t0)])
    await storage.put_runs("a", [_run("a", "r1-child", parent="r1", at=t0)])
    await storage.put_runs("a", [_run("a", "r1-grandchild", parent="r1-child", at=t0)])
    await storage.put_runs("a", [_tagged("a", "r2", {"customer": "c-82"}, at=t0)])
    await storage.put_runs("b", [_tagged("b", "r1", {"customer": "c-81"}, at=t0)])
    await storage.put_runs("b", [_run("b", "r1-child", parent="r1", at=t0)])
    for scope, label_id, run_id in (
        ("a", "l1", "r1"),
        ("a", "l2", "r1-grandchild"),
        ("a", "l3", "r2"),
        ("a", "l4", "ghost"),
        ("b", "l1", "r1"),
    ):
        await storage.put_label(scope, _label(scope, label_id, run_id))
    assert await storage.delete_runs("a", tags={"customer": "c-81"}) == 5
    assert [r.run_id for r in await storage.runs("a")] == ["r2"]
    assert [label.label_id for label in await storage.labels("a")] == ["l3", "l4"]
    assert await storage.delete_runs("a", run_ids=["ghost", "r2"]) == 3
    assert await storage.count_scope("a") == 0
    assert await storage.count_scope("b") == 3
    assert await storage.delete_runs("b", run_ids=["r1-child"]) == 1
    assert [r.run_id for r in await storage.runs("b")] == ["r1"]


async def test_a_recorded_tree_reads_back_as_it_ran(storage: Storage) -> None:
    refinery = Refinery(storage)
    stop = Stop()

    async def lookup(args: dict[str, Any]) -> list[Part]:
        stop.after_step()
        return [Text(text="a\x00b"), Image(media_type="image/png", data=bytes(range(256)))]

    both = delegate_triage("d1", "one").tool_calls + delegate_triage("d2", "two").tool_calls
    looking = Tools([Tool("lookup_order", "Fetch an order", body=lookup)])
    engine = Engine(
        FIXTURES,
        refinery=refinery,
        models={
            "strong": Scripted(
                Reply(tool_calls=both, stop_reason="tool_use"),
                Reply(
                    tool_calls=[ToolCall(id="c1", name="lookup_order", args={"order_id": "😀"})],
                    stop_reason="tool_use",
                ),
                text_reply("Continued."),
            ),
            "judge": Desk(),
        },
    )
    run = await engine.run(
        "frontdesk", "a", "It is down", tags={"customer": "c-81"}, stop=stop, tools=looking
    )
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    support = run.routed_to
    assert support.stopped == "after_step" and len(support.children) == 2
    got = await refinery.run("a", run.run_id)
    assert got.model_dump() == run.model_dump() and got == run
    assert await refinery.run("a", support.run_id) == support
    assert await refinery.run("a", support.children[1].run_id) == support.children[1]
    with pytest.raises(NotFoundError, match=f"^refinery: run '{run.run_id}' was not found$"):
        await refinery.run("b", run.run_id)
    last = await refinery.run("a", support.run_id)
    assert isinstance(last, Generation)
    after = await engine.run("support", "a", "Go on.", history=last.conversation)
    assert isinstance(after, Generation) and after.reply == "Continued."
    assert after.messages == [*support.conversation, Message(role="user", content="Go on.")]


async def test_a_tree_reads_back_with_its_one_record_stamp(storage: Storage) -> None:
    refinery = Refinery(storage)
    engine = Engine(
        FIXTURES,
        refinery=refinery,
        models={"strong": Scripted(delegate_triage(), text_reply("Handled.")), "judge": Desk()},
    )
    run = await engine.run("frontdesk", "a", "It is down")
    rows = await storage.run_tree("a", run_ids=[run.run_id])
    [stamp] = {row.recorded for row in rows}
    assert len(rows) == 3 and stamp >= run.at
    for row in rows:
        stored = await storage.get_run("a", row.run_id)
        assert stored is not None and stored.recorded == stamp
    assert await refinery.run("a", run.run_id) == run
    assert {row.recorded for row in await storage.run_tree("a", run_ids=[run.run_id])} == {stamp}
    assert (await refinery.export("a", MemoryBucket())).watermark == stamp


async def test_fingerprints_and_context_round_trip(
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    engine = Engine(
        FIXTURES,
        refinery=refinery,
        models={"strong": Scripted(delegate_triage(), text_reply("Handled.")), "judge": Desk()},
    )
    run = await engine.run("frontdesk", "a", "It is down", context="Acme is enterprise.")
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.context == "Acme is enterprise."
    assert await refinery.run("a", run.run_id) == run
    bucket = MemoryBucket()
    await refinery.export("a", bucket)
    rows = [
        json.loads(line) for data in bucket.objects.values() for line in data.decode().splitlines()
    ]
    runs = {r.run_id: r for r in (run, run.routed_to, run.routed_to.children[0])}
    assert len(rows) >= 3 and all(
        row["fingerprint"] == runs[row["run_id"]].model_dump(mode="json")["fingerprint"]
        for row in rows
    )


async def test_structured_questions_round_trip_through_the_record_and_the_export(
    storage: Storage, tmp_path: Path
) -> None:
    refinery = Refinery(storage)
    dm = sorter(tmp_path)
    run = await Engine(tmp_path, refinery=refinery, models={"judge": dm}).run("sorter", "a", "Hi")
    assert isinstance(run, Decision)
    [(_, asked)] = dm.calls
    again = await refinery.run("a", run.run_id)
    assert again == run and isinstance(again, Decision) and again.questions == asked
    assert list(again.questions["team"].model_dump()["options"]) == ["billing", "orders", "other"]
    bucket = MemoryBucket()
    await refinery.export("a", bucket)
    rows = [
        json.loads(line)
        for key, data in bucket.objects.items()
        if "/decision/" in key
        for line in data.decode().splitlines()
    ]
    assert {row["question_id"]: row["question"] for row in rows} == {
        qid: question.model_dump(mode="json") for qid, question in run.questions.items()
    }


class Seeing:
    def __init__(self) -> None:
        self.seen: list[list[Message]] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.seen.append(list(messages))
        return text_reply("Candidate.")


async def test_replay_selects_reads_back_and_resends_what_each_run_was_given(
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        FIXTURES,
        refinery=refinery,
        models={
            "strong": Scripted(
                delegate_triage(), text_reply("Handled."), text_reply("Hi."), text_reply("Bye.")
            ),
            "judge": Desk(),
        },
    )
    parts: list[Part] = [
        Text(text="Charged twice?"),
        Image(media_type="image/png", data=bytes(range(256))),
    ]
    history = [Message(role="user", content="Earlier"), Message(role="assistant", content="Ok.")]
    first = await recorder.run(
        "support",
        "a",
        list(parts),
        history=history,
        task="triage",
        context="Acme is enterprise.",
        tags={"arm": "a", "customer": "c-1"},
    )
    second = await recorder.run("support", "a", "Hello", tags={"arm": "b"})
    third = await recorder.run("support", "a", {"order": "A-1"}, tags={"arm": "a"})
    await refinery.review("a", second.run_id, Review(stance="approval"))
    await refinery.outcome("a", third.run_id, Outcome(status="failed"))
    counted = await storage.count_scope("a")
    model = Seeing()
    candidate = Engine(FIXTURES, refinery=refinery, models={"strong": model, "judge": Desk()})

    async def replayed(**selectors: Any) -> list[str]:
        pairs = [p async for p in refinery.replay("a", candidate, agent="support", **selectors)]
        return [pair.recorded.run_id for pair in pairs]

    pairs = [p async for p in refinery.replay("a", candidate, agent="support")]
    assert [pair.recorded.model_dump(mode="json") for pair in pairs] == [
        run.model_dump(mode="json") for run in (first, second, third)
    ]
    assert [len(pair.labels) for pair in pairs] == [0, 1, 1]
    assert all(isinstance(run, Generation) for run in (first, second, third))
    assert model.seen == [
        run.messages for run in (first, second, third) if isinstance(run, Generation)
    ]
    assert isinstance(model.seen[0][-1].content, list) and model.seen[0][-1].content == parts
    candidate_first = pairs[0].candidate
    assert isinstance(candidate_first, Generation) and isinstance(first, Generation)
    assert candidate_first.task == "triage" and candidate_first.context == "Acme is enterprise."
    assert candidate_first.fingerprint == first.fingerprint
    assert await replayed(tags={"arm": "a"}) == [first.run_id, third.run_id]
    assert await replayed(tags={"arm": "a", "customer": "c-1"}) == [first.run_id]
    assert await replayed(labeled=True) == [second.run_id, third.run_id]
    assert await replayed(fingerprint=first.fingerprint) == [
        r.run_id for r in (first, second, third)
    ]
    assert (
        await replayed(fingerprint=Fingerprint(agent="0" * 16, tools="0" * 16, questions=None))
        == []
    )
    assert len(await replayed(holdout=True, holdout_percent=100)) == 3
    assert await replayed(holdout=True, holdout_percent=0) == []
    assert await replayed(limit=2) == [first.run_id, second.run_id]
    assert await storage.count_scope("a") == counted
    assert [p async for p in refinery.replay("b", candidate, agent="support")] == []


AREA = Choice(
    instructions={"question": "Which area?", "within": "orders"},
    options={"returns": None, "delivery": {"what": "Parcels", "examples": ["Late"]}},
)


def grader(root: Path, urgent: str = "Is it urgent?") -> Path:
    for path, text in {
        "grader/AGENT.md": "---\nname: grader\nkind: decision\nmodel: judge\n---\n",
        "grader/questions/urgent.md": f"---\ntype: check\n---\n{urgent}\n",
        "grader/tasks/tone/polite.md": "---\ntype: check\n---\nPolite?\n",
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return root


async def test_runtime_questions_a_task_and_the_questions_part_round_trip_and_replay(
    storage: Storage, tmp_path: Path
) -> None:
    refinery = Refinery(storage)
    engine = Engine(grader(tmp_path / "old"), refinery=refinery, models={"judge": Answering()})
    asked = await ask(engine, "grader", "a", "x", task="tone", questions={"area": AREA})
    plain = await engine.run("grader", "a", "y")
    assert isinstance(asked, Decision)
    again = await refinery.run("a", asked.run_id)
    assert again == asked and isinstance(again, Decision)
    assert again.task == "tone" and again.runtime == ["area"]
    assert list(again.questions) == ["urgent", "polite", "area"]
    assert list(again.questions["area"].model_dump()["options"]) == ["returns", "delivery"]
    [row] = await storage.runs("a", run_ids=[asked.run_id])
    assert row.task == "tone"
    bucket = MemoryBucket()
    await refinery.export("a", bucket)
    rows = [
        json.loads(line)
        for key, data in bucket.objects.items()
        if "/decision/" in key
        for line in data.decode().splitlines()
    ]
    mine = [r for r in rows if r["run_id"] == asked.run_id]
    assert {r["question_id"] for r in mine} == {"urgent", "polite", "area"}
    assert all(r["task"] == "tone" for r in mine)
    assert all(r["fingerprint"] == asked.fingerprint.model_dump() for r in mine)
    dm = Answering()
    candidate = Engine(grader(tmp_path / "new", "Due today?"), models={"judge": dm})
    pairs = [
        p
        async for p in refinery.replay(
            "a", candidate, agent="grader", fingerprint=asked.fingerprint
        )
    ]
    assert [pair.recorded.run_id for pair in pairs] == [asked.run_id]
    [(_, sent)] = dm.calls
    expected: dict[str, Any] = {
        "urgent": Check(instructions="Due today?"),
        "polite": Check(instructions="Polite?"),
        "area": AREA,
    }
    assert sent == expected
    assert plain.fingerprint.questions != asked.fingerprint.questions


async def test_a_delegated_child_keeps_its_task_and_runtime_questions(
    storage: Storage, tmp_path: Path
) -> None:
    refinery = Refinery(storage)
    asked = delegating("grader", "Sorry!", task="tone", questions={"area": PICK})
    engine = Engine(
        lead(tmp_path),
        refinery=refinery,
        models={"strong": Scripted(asked, text_reply("Graded.")), "judge": Answering()},
    )
    run = await engine.run("lead", "a", "Grade this")
    assert isinstance(run, Generation)
    [child] = run.children
    assert await refinery.run("a", run.run_id) == run
    again = await refinery.run("a", child.run_id)
    assert again == child and isinstance(again, Decision)
    assert again.task == "tone" and again.runtime == ["area"]
    assert list(again.questions) == ["urgent", "polite", "area"]
    [row] = await storage.runs("a", run_ids=[child.run_id])
    assert row.task == "tone" and row.parent_run_id == run.run_id


class Looking(Answering):
    media_types = {"image/*"}


async def test_a_state_of_parts_round_trips_as_parts_and_replays(
    storage: Storage, tmp_path: Path
) -> None:
    refinery = Refinery(storage)
    root = grader(tmp_path)
    parts: list[Part] = [
        Text(text="Urgent?"),
        Image(media_type="image/png", data=bytes(range(256))),
    ]
    run = await Engine(root, refinery=refinery, models={"judge": Looking()}).run(
        "grader", "a", parts
    )
    again = await refinery.run("a", run.run_id)
    assert again == run and isinstance(again, Decision) and again.state == parts
    looking = Looking()
    [pair] = [
        p
        async for p in refinery.replay("a", Engine(root, models={"judge": looking}), agent="grader")
    ]
    assert pair.error is None and looking.calls[0][0] == parts


async def test_start_prefix_and_usage_round_trip(
    storage: Storage,
) -> None:
    written = Message(
        role="assistant",
        content="Found it.",
        prefix="0123456789abcdef",
        usage=Usage(input=7, output=2, cache_read=5),
    )
    run = Generation(
        run_id="w",
        agent="support",
        scope="a",
        at=datetime.now(UTC),
        duration_ms=1,
        model="m",
        fingerprint=Fingerprint(agent="a" * 16, tools="b" * 16, questions=None),
        input="hi",
        system="s",
        start=4,
        messages=[Message(role="user", content="hi")],
        conversation=[Message(role="user", content="hi"), written],
        steps=[],
        calls=1,
        reply="Found it.",
        usage=Usage(),
    )
    refinery = Refinery(storage)
    await refinery.record(run)
    assert await refinery.run("a", "w") == run
    bucket = MemoryBucket()
    exported = await refinery.export("a", bucket)
    [key] = [k for k in exported.keys if k.startswith("a/support/generation/")]
    data = bucket.objects[key]
    [row] = [json.loads(line) for line in data.splitlines()]
    assert row["start"] == 4 and row["conversation"][-1]["prefix"] == "0123456789abcdef"


async def test_a_failed_run_and_its_failed_child_read_back_as_recorded(storage: Storage) -> None:
    refinery = Refinery(storage)
    child = decision_record(run_id="d1").model_copy(
        update={"parent_run_id": "g1", "answers": {}, "error": "TimeoutError: decide timed out"}
    )
    failed = generation_record(run_id="g1", reply="", children=[child]).model_copy(
        update={"error": "ModelError: anthropic: overloaded", "stop_reason": "tool_use"}
    )
    await refinery.record(failed)
    back = await refinery.run("acme", "g1")
    assert isinstance(back, Generation) and back.error == "ModelError: anthropic: overloaded"
    assert back.stop_reason == "tool_use" and back.reply == ""
    [kid] = back.children
    assert isinstance(kid, Decision) and kid.answers == {}
    assert kid.error == "TimeoutError: decide timed out"


async def test_put_runs_stores_a_tree_whole(storage: Storage) -> None:
    await storage.put_runs("a", [_run("a", "root"), _run("a", "child", parent="root")])
    assert [r.run_id for r in await storage.run_tree("a", run_ids=["root"])] == ["root", "child"]


async def test_put_runs_with_a_row_of_another_scope_stores_nothing(storage: Storage) -> None:
    with pytest.raises(StorageError, match="^postgres: row scope does not match the call scope$"):
        await storage.put_runs("a", [_run("a", "r1"), _run("b", "r2")])
    assert await storage.count_scope("a") == 0 and await storage.count_scope("b") == 0


async def test_put_runs_stores_nothing_when_the_database_refuses_a_row(storage: Storage) -> None:
    bad = _run("a", "r2", agent="a\x00b")
    with pytest.raises(StorageError, match="^postgres: storage rejected put_runs: "):
        await storage.put_runs("a", [_run("a", "r1"), bad])
    assert await storage.get_run("a", "r1") is None


async def test_put_runs_with_no_rows_does_nothing(storage: Storage) -> None:
    await storage.put_runs("a", [])
    assert await storage.count_scope("a") == 0


async def test_a_row_without_tags_matches_no_tag_selector(storage: Storage) -> None:
    await storage.put_runs("a", [_run("a", "plain")])
    assert await storage.run_tree("a", tags={"customer": "c-81"}) == []
    stored = await storage.get_run("a", "plain")
    assert stored is not None and stored.body == _run("a", "plain").body


async def test_initialize_makes_the_indexes_and_runs_again(pool: Any) -> None:
    await PostgresStorage(pool).initialize()
    records = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'engine_runs' AND schemaname = current_schema()"
    )
    names = {r["indexname"] for r in records}
    assert {"ix_engine_runs_scope_parent", "ix_engine_runs_tags"} <= names


async def test_initialize_on_a_table_an_earlier_version_made_says_what_to_do(pool: Any) -> None:
    schema = f"engine_old_{uuid4().hex}"
    await pool.execute(f"CREATE SCHEMA {schema}")
    old = await asyncpg.create_pool(
        os.environ["RFNRY_TEST_POSTGRES_URL"], min_size=1, server_settings={"search_path": schema}
    )
    try:
        await old.execute(
            "CREATE TABLE engine_runs (scope TEXT NOT NULL, run_id TEXT NOT NULL, "
            "parent_run_id TEXT NULL, agent TEXT NOT NULL, kind TEXT NOT NULL, task TEXT NULL, "
            "at TIMESTAMPTZ NOT NULL, recorded TIMESTAMPTZ NOT NULL, duration_ms INTEGER NOT NULL, "
            "model TEXT NULL, body JSON NOT NULL, PRIMARY KEY (scope, run_id))"
        )
        with pytest.raises(StorageError) as refused:
            await PostgresStorage(old).initialize()
        assert str(refused.value) == (
            "postgres: engine_runs was made by an earlier version and lacks tags; "
            "drop engine_runs and engine_labels, then run initialize() again"
        )
    finally:
        await old.close()
        await pool.execute(f"DROP SCHEMA {schema} CASCADE")


async def test_a_recorded_call_reads_back(storage: Storage, tmp_path: Path) -> None:
    for name, front in (
        ("lead", "name: lead\nkind: generation\nmodel: strong\nagents: [child, triage]"),
        ("child", "name: child\nkind: generation\nmodel: strong"),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "AGENT.md").write_text(f"---\n{front}\n---\nYou are {name}.\n")
    shutil.copytree(FIXTURES / "triage", tmp_path / "triage")
    refinery = Refinery(storage)
    engine = Engine(
        tmp_path,
        refinery=refinery,
        models={
            "strong": Scripted(
                Reply(
                    tool_calls=[
                        ToolCall(id="g1", name="Delegate", args={"agent": "child", "input": "x"}),
                        ToolCall(id="d1", name="Delegate", args={"agent": "triage", "input": "y"}),
                    ],
                    stop_reason="tool_use",
                ),
                text_reply("Billed."),
                text_reply("Done."),
            ),
            "judge": Desk(),
        },
    )
    run = await engine.run("lead", "a", "go")
    assert isinstance(run, Generation)
    assert {c.call.call_id for c in run.children if c.call} == {"g1", "d1"}
    again = await Refinery(storage).run("a", run.run_id)
    assert isinstance(again, Generation)
    assert [c.call for c in again.children] == [c.call for c in run.children]
