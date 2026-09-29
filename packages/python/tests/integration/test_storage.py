import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from unit.fakes import (
    TOOLS,
    Desk,
    MemoryBucket,
    Scripted,
    delegate_triage,
    generation_record,
    text_reply,
)

from rfnry_engine.engine import Engine
from rfnry_engine.errors import NotFoundError, StorageError
from rfnry_engine.models import (
    Check,
    CheckAnswer,
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
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import LabelRow, RunRow, Storage
from rfnry_engine.stop import Stop
from rfnry_engine.tools.tools import Tool

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
    await storage.put_run(
        "a", _run("a", "mid", agent="billing", kind="decision", at=t0 - timedelta(minutes=1))
    )
    await storage.put_run("a", _run("a", "new", at=t0))
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
    await storage.put_run(
        "a", _run("a", "long", at=t0 - timedelta(minutes=5), recorded=t0 + timedelta(seconds=1))
    )
    await storage.put_run("a", _run("a", "quick", at=t0, recorded=t0))
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
    assert [r.run_id for r in await storage.runs("a", run_ids=["r1", "r2", "ghost"])] == [
        "r2",
        "r1",
    ]
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
    body = {
        "zeta": 1,
        "a": {"refund": "r", "no": "n", "billing": "b"},
        "text": "a\x00b",
        "big": 1e16,
        "tiny": 1.5e-7,
    }
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
    await storage.put_run(
        "a",
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
        ),
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
    await Refinery(storage).record(run)
    got = await storage.get_run("a", "g")
    assert got is not None and Generation.model_validate(got.body) == run
    old = {key: value for key, value in got.body.items() if key != "tools"}
    await storage.put_run("a", got.model_copy(update={"body": old}))
    again = await storage.get_run("a", "g")
    assert again is not None and Generation.model_validate(again.body).tools == []


async def test_tags_round_trip_and_old_bodies_read_as_no_tags(storage: Storage) -> None:
    at = datetime.now(UTC)
    decision = Decision(
        run_id="d",
        agent="triage",
        scope="a",
        at=at,
        duration_ms=1,
        model="d",
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
    await Refinery(storage).record(run)
    for run_id, kind in (("g", Generation), ("d", Decision)):
        got = await storage.get_run("a", run_id)
        assert got is not None and got.body["tags"] == {"arm": "b", "commit": "a81d07b"}
        assert kind.model_validate(got.body).tags == {"arm": "b", "commit": "a81d07b"}
        old = {key: value for key, value in got.body.items() if key != "tags"}
        await storage.put_run("a", got.model_copy(update={"body": old}))
        again = await storage.get_run("a", run_id)
        assert again is not None and kind.model_validate(again.body).tags == {}


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
        await storage.put_run(row.scope, row)
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
    await storage.put_run("a", _run("a", "orphan", parent="gone"))
    await storage.put_run("a", _run("a", "x", parent="y"))
    await storage.put_run("a", _run("a", "y", parent="x"))
    assert [r.run_id for r in await storage.run_tree("a", run_ids=["gone"])] == ["orphan"]
    assert {r.run_id for r in await storage.run_tree("a", run_ids=["x"])} == {"x", "y"}


async def test_run_tree_by_tags_is_containment_and_takes_descendants(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_run("a", _tagged("a", "r1", {"customer": "c-81", "arm": "b"}, at=t0))
    await storage.put_run("a", _run("a", "r1-child", parent="r1", at=t0 + timedelta(seconds=1)))
    await storage.put_run("a", _tagged("a", "r2", {"customer": "c-82"}, at=t0))
    await storage.put_run("a", _run("a", "untagged", at=t0))
    await storage.put_run("b", _tagged("b", "r3", {"customer": "c-81"}, at=t0))

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
    await storage.put_run("a", _tagged("a", "nul", {"customer": "c-81"}, extra=hostile))
    await storage.put_run("a", _tagged("a", "emoji", {"customer": "😀", "hangul": "퀀"}))
    await storage.put_run("a", _tagged("a", "other", {"customer": "😁"}))
    assert [r.run_id for r in await storage.run_tree("a", tags={"customer": "c-81"})] == ["nul"]
    assert [r.run_id for r in await storage.run_tree("a", tags={"customer": "😀"})] == ["emoji"]
    assert [r.run_id for r in await storage.run_tree("a", tags={"hangul": "퀀"})] == ["emoji"]
    got = await storage.get_run("a", "nul")
    assert got is not None and got.body["reply"] == hostile["reply"]
    assert await storage.delete_runs("a", tags={"customer": "c-81"}) == 1
    assert {r.run_id for r in await storage.runs("a")} == {"emoji", "other"}


async def test_delete_runs_takes_descendants_and_labels_in_its_scope(storage: Storage) -> None:
    t0 = datetime.now(UTC)
    await storage.put_run("a", _tagged("a", "r1", {"customer": "c-81"}, at=t0))
    await storage.put_run("a", _run("a", "r1-child", parent="r1", at=t0))
    await storage.put_run("a", _run("a", "r1-grandchild", parent="r1-child", at=t0))
    await storage.put_run("a", _tagged("a", "r2", {"customer": "c-82"}, at=t0))
    await storage.put_run("b", _tagged("b", "r1", {"customer": "c-81"}, at=t0))
    await storage.put_run("b", _run("b", "r1-child", parent="r1", at=t0))
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
    engine = Engine(
        FIXTURES,
        generation=Scripted(
            Reply(tool_calls=both, stop_reason="tool_use"),
            Reply(
                tool_calls=[ToolCall(id="c1", name="lookup_order", args={"order_id": "😀"})],
                stop_reason="tool_use",
            ),
            text_reply("Continued."),
        ),
        decision=Desk(),
        tools=[Tool("lookup_order", body=lookup), TOOLS[1]],
        refinery=refinery,
    )
    run = await engine.run("frontdesk", "a", "It is down", tags={"customer": "c-81"}, stop=stop)
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
        generation=Scripted(delegate_triage(), text_reply("Handled.")),
        decision=Desk(),
        tools=TOOLS,
        refinery=refinery,
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


async def test_fingerprints_and_context_round_trip_and_old_bodies_read_as_none(
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    engine = Engine(
        FIXTURES,
        generation=Scripted(delegate_triage(), text_reply("Handled.")),
        decision=Desk(),
        tools=TOOLS,
        refinery=refinery,
    )
    run = await engine.run("frontdesk", "a", "It is down", context="Acme is enterprise.")
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.fingerprint is not None and run.routed_to.fingerprint is not None
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
    for stored in await storage.run_tree("a", run_ids=[run.run_id]):
        old = {k: v for k, v in stored.body.items() if k not in ("fingerprint", "context")}
        await storage.put_run("a", stored.model_copy(update={"body": old}))
    again = await refinery.run("a", run.run_id)
    assert isinstance(again, Decision) and isinstance(again.routed_to, Generation)
    assert again.fingerprint is None and again.routed_to.fingerprint is None
    assert again.routed_to.context is None
    bucket = MemoryBucket()
    await refinery.export("a", bucket)
    assert all(
        json.loads(line)["fingerprint"] is None
        for data in bucket.objects.values()
        for line in data.decode().splitlines()
    )


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
        generation=Scripted(
            delegate_triage(), text_reply("Handled."), text_reply("Hi."), text_reply("Bye.")
        ),
        decision=Desk(),
        tools=TOOLS,
        refinery=refinery,
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
    candidate = Engine(FIXTURES, generation=model, decision=Desk(), tools=TOOLS, refinery=refinery)

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
    assert first.fingerprint is not None
    assert await replayed(fingerprint=first.fingerprint) == [
        r.run_id for r in (first, second, third)
    ]
    assert await replayed(fingerprint=Fingerprint(agent="0" * 16, tools="0" * 16)) == []
    assert len(await replayed(holdout=True, holdout_percent=100)) == 3
    assert await replayed(holdout=True, holdout_percent=0) == []
    assert await replayed(limit=2) == [first.run_id, second.run_id]
    assert await storage.count_scope("a") == counted
    assert [p async for p in refinery.replay("b", candidate, agent="support")] == []
