from datetime import UTC, datetime, timedelta

import pytest

from rfnry_engine.errors import StorageError
from rfnry_engine.models import Document, Image, Message, Text, ToolCall
from rfnry_engine.refinery.storage import LabelRow, RunRow, Storage


def _run(scope: str, run_id: str, *, agent: str = "support", kind: str = "generation", at: datetime | None = None, parent: str | None = None) -> RunRow:
    return RunRow(scope=scope, run_id=run_id, parent_run_id=parent, agent=agent, kind=kind, task=None,
                  at=at or datetime.now(UTC), duration_ms=1, model="m", body={"reply": run_id, "nested": {"n": [1, 2]}})


def _label(scope: str, label_id: str, run_id: str, *, source: str = "human", at: datetime | None = None) -> LabelRow:
    return LabelRow(scope=scope, label_id=label_id, run_id=run_id, source=source, at=at or datetime.now(UTC), body={"source": source})


async def test_put_get_upsert_roundtrip(storage: Storage) -> None:
    await storage.put_run("a", _run("a", "r1"))
    got = await storage.get_run("a", "r1")
    assert got is not None and got.body == {"reply": "r1", "nested": {"n": [1, 2]}}
    await storage.put_run("a", _run("a", "r1", agent="billing"))
    got = await storage.get_run("a", "r1")
    assert got is not None and got.agent == "billing"
    assert await storage.get_run("a", "ghost") is None


async def test_runs_filters_order_and_limit(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_run("a", _run("a", "old", at=t0 - timedelta(minutes=2)))
    await storage.put_run("a", _run("a", "mid", agent="billing", kind="decision", at=t0 - timedelta(minutes=1)))
    await storage.put_run("a", _run("a", "new", at=t0))
    assert [r.run_id for r in await storage.runs("a")] == ["new", "mid", "old"]
    assert [r.run_id for r in await storage.runs("a", kind="generation")] == ["new", "old"]
    assert [r.run_id for r in await storage.runs("a", since=t0 - timedelta(minutes=1))] == ["new"]
    assert [r.run_id for r in await storage.runs("a", limit=2)] == ["new", "mid"]
    assert [r.run_id for r in await storage.runs("a", limit=None)] == ["new", "mid", "old"]
    await storage.put_label("a", _label("a", "l1", "new", source="judge"))
    assert [r.run_id for r in await storage.runs("a", unlabeled_by="judge")] == ["mid", "old"]
    assert [r.run_id for r in await storage.runs("a", unlabeled_by="teacher")] == ["new", "mid", "old"]


async def test_labels_filters_and_order(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_run("a", _run("a", "r1"))
    await storage.put_label("a", _label("a", "l2", "r1", source="world", at=t0))
    await storage.put_label("a", _label("a", "l1", "r1", at=t0 - timedelta(seconds=1)))
    await storage.put_label("a", _label("a", "l3", "r2", at=t0 + timedelta(seconds=1)))
    assert [label.label_id for label in await storage.labels("a")] == ["l1", "l2", "l3"]
    assert [label.label_id for label in await storage.labels("a", run_ids=["r1"])] == ["l1", "l2"]
    assert [label.label_id for label in await storage.labels("a", source="world")] == ["l2"]
    assert [label.label_id for label in await storage.labels("a", since=t0)] == ["l3"]


async def test_runs_by_id_and_labels_since_find_late_labels(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_run("a", _run("a", "r1", at=t0 - timedelta(minutes=2)))
    await storage.put_run("a", _run("a", "r2", at=t0 - timedelta(minutes=1)))
    await storage.put_run("b", _run("b", "r1", at=t0))
    await storage.put_label("a", _label("a", "l1", "r1", at=t0 + timedelta(seconds=1)))
    late = {label.run_id for label in await storage.labels("a", since=t0)}
    assert late == {"r1"}
    assert [r.run_id for r in await storage.runs("a", run_ids=sorted(late))] == ["r1"]
    assert [r.run_id for r in await storage.runs("a", run_ids=["r1", "r2", "ghost"])] == ["r2", "r1"]
    assert await storage.runs("a", run_ids=[]) == []
    assert [r.scope for r in await storage.runs("b", run_ids=["r1", "r2"])] == ["b"]


async def test_scope_mismatch_is_refused(storage: Storage) -> None:
    with pytest.raises(StorageError):
        await storage.put_run("a", _run("b", "r1"))
    with pytest.raises(StorageError):
        await storage.put_label("a", _label("b", "l1", "r1"))


async def test_delete_and_count(storage: Storage) -> None:
    await storage.put_run("a", _run("a", "r1"))
    await storage.put_label("a", _label("a", "l1", "r1"))
    await storage.put_run("b", _run("b", "r2"))
    assert await storage.count_scope("a") == 2
    assert await storage.delete_scope("a") == 2
    assert await storage.count_scope("a") == 0 and await storage.count_scope("b") == 1


async def test_bodies_round_trip_exactly(storage: Storage) -> None:
    body = {"zeta": 1, "a": {"refund": "r", "no": "n", "billing": "b"}, "text": "a\x00b", "big": 1e16, "tiny": 1.5e-7}
    await storage.put_run("a", _run("a", "r1").model_copy(update={"body": body}))
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
        await storage.put_run("a", _run("a", run_id, at=at))
        await storage.put_label("a", _label("a", run_id, run_id, at=at))
    assert [r.run_id for r in await storage.runs("a")] == sorted(ids, reverse=True)
    assert [label.label_id for label in await storage.labels("a")] == sorted(ids)


async def test_a_body_with_parts_round_trips(storage: Storage) -> None:
    raw = bytes(range(256))
    conversation = [
        Message(role="user", content=[Text(text="look"), Image(media_type="image/png", data=raw)]),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="read_pdf", args={"path": "a.pdf"})]),
        Message(role="tool", tool_call_id="c1", content=[Document(media_type="application/pdf", name="a.pdf", url="https://x/a.pdf")]),
    ]
    body = {"conversation": [m.model_dump(mode="json") for m in conversation]}
    await storage.put_run("a", RunRow(scope="a", run_id="p", parent_run_id=None, agent="support", kind="generation", task=None, at=datetime.now(UTC), duration_ms=1, model="m", body=body))
    got = await storage.get_run("a", "p")
    assert got is not None and got.body == body
    assert [Message.model_validate(m) for m in got.body["conversation"]] == conversation
