from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from unit.fakes import MemoryBucket, Scripted, generation_record, text_reply
from unit.trees import provided, write_folder

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError, NotFoundError
from rfnry_engine.models import ToolSpec, Usage
from rfnry_engine.records import Spoken, SpokenLine, Voice, VoiceStart
from rfnry_engine.refinery.labels import Exported, Outcome, Review
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.tools import Result, Tool, Tools

ENTRY: dict[str, Any] = {
    "kind": "voice",
    "model": "live",
    "instructions": "voice.md",
    "tools": {"include": ["lookup"]},
    "modes": {"duplex": {"instructions": "duplex.md"}},
}
FILES = {"voice.md": "Speak plainly.\n", "duplex.md": "Talk both ways.\n"}
HEARD_AT = datetime(2026, 11, 10, 12, 30, tzinfo=UTC)
NO_CORRECTION = r"^refinery: a voice run takes a stance and a comment, not a correction$"
Session = Callable[[VoiceStart], Awaitable[Spoken]]


class Live:
    model = "declared-1"

    async def open(self, *, tools: list[ToolSpec], instructions: str | None) -> object:
        raise AssertionError("the engine hands the model to the session and never calls it")


async def looked_up(args: dict[str, Any]) -> Result:
    return f"found {args['q']}"


LOOKUP = Tools([Tool("lookup", "Look something up", body=looked_up)])


def saying(text: str, usage: Usage | None = None) -> Session:
    async def session(start: VoiceStart) -> Spoken:
        await start.tools.dispatch("lookup", {"q": text})
        return Spoken(
            transcript=[
                SpokenLine(speaker="person", text=text, at=HEARD_AT),
                SpokenLine(speaker="assistant", text="On it."),
            ],
            model="live-1",
            usage=usage,
        )

    return session


async def dropping(start: VoiceStart) -> Spoken:
    await start.tools.dispatch("lookup", {"q": "before the drop"})
    raise RuntimeError("the line dropped")


def voice_engine(root: Path, refinery: Refinery) -> Engine:
    write_folder(root, {"voice": ENTRY}, files=FILES)
    return Engine(root, providers=provided({"live": Live()}), refinery=refinery)


async def test_a_voice_run_reads_back_equal(tmp_path: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    engine = voice_engine(tmp_path, refinery)
    run = await engine.talk(
        "voice",
        "acme",
        saying("Where is my order?"),
        mode="duplex",
        context="The caller is Dana.",
        tools=LOOKUP,
        tags={"call": "c-1"},
    )
    read = await refinery.run("acme", run.run_id)
    assert isinstance(read, Voice) and read == run
    assert [(line.speaker, line.text, line.at) for line in read.transcript] == [
        ("person", "Where is my order?", HEARD_AT),
        ("assistant", "On it.", None),
    ]
    assert [(step.tool, step.args, step.result, step.is_error) for step in read.steps] == [
        ("lookup", {"q": "Where is my order?"}, "found Where is my order?", False)
    ]
    assert read.mode == "duplex" and read.usage is None
    row = await storage.get_run("acme", run.run_id)
    assert row is not None and row.mode == "duplex" and row.body["usage"] is None

    counted = await engine.talk(
        "voice", "acme", saying("Hello", Usage(input=3, output=4)), tools=LOOKUP
    )
    assert await refinery.run("acme", counted.run_id) == counted
    assert counted.usage == Usage(input=3, output=4)

    with pytest.raises(RuntimeError, match="the line dropped"):
        await engine.talk("voice", "acme", dropping, tools=LOOKUP)
    [failed] = [
        row for row in await storage.runs("acme") if row.run_id not in {run.run_id, counted.run_id}
    ]
    dropped = await refinery.run("acme", failed.run_id)
    assert isinstance(dropped, Voice) and dropped.error == "RuntimeError: the line dropped"
    assert dropped.transcript == [] and [step.tool for step in dropped.steps] == ["lookup"]


async def test_a_voice_run_stays_in_its_scope(tmp_path: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    engine = voice_engine(tmp_path, refinery)
    a = await engine.talk("voice", "tenant-a", saying("MARKER-A"), tools=LOOKUP)
    b = await engine.talk("voice", "tenant-b", saying("plain-b"), tools=LOOKUP)
    await refinery.review("tenant-a", a.run_id, Review(comment="MARKER-A"))
    await refinery.outcome("tenant-a", a.run_id, Outcome(status="failed", detail="MARKER-A"))

    assert await storage.get_run("tenant-b", a.run_id) is None
    with pytest.raises(NotFoundError, match=r"^refinery: run '.+' was not found$"):
        await refinery.run("tenant-b", a.run_id)
    with pytest.raises(NotFoundError, match=r"^refinery: run '.+' was not found$"):
        await refinery.review("tenant-b", a.run_id, Review(stance="approval"))
    with pytest.raises(NotFoundError, match=r"^refinery: run '.+' was not found$"):
        await refinery.outcome("tenant-b", a.run_id, Outcome(status="succeeded"))

    assert [row.run_id for row in await storage.runs("tenant-b")] == [b.run_id]
    assert await storage.labels("tenant-b") == []
    assert await refinery.run("tenant-b", b.run_id) == b
    pairs = [pair async for pair in refinery.replay("tenant-b", engine, agent="voice")]
    assert [pair.recorded.run_id for pair in pairs] == [b.run_id]
    assert "MARKER-A" not in str(pairs)

    assert await refinery.delete_scope("tenant-a") == 3
    assert await refinery.verify_absent("tenant-a")
    assert await refinery.run("tenant-b", b.run_id) == b


async def test_a_voice_run_takes_a_stance_and_a_comment_but_no_correction(
    tmp_path: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    engine = voice_engine(tmp_path, refinery)
    run = await engine.talk("voice", "acme", saying("Hello"), tools=LOOKUP)

    stance = await refinery.review("acme", run.run_id, Review(stance="objection"))
    comment = await refinery.review("acme", run.run_id, Review(comment="It spoke over the caller."))
    assert stance.source == "human" and stance.review == Review(stance="objection")
    assert comment.review == Review(comment="It spoke over the caller.")

    corrections: list[str | dict[str, Any]] = ["Say it differently.", "", {"urgent": False}]
    for correction in corrections:
        with pytest.raises(InputError, match=NO_CORRECTION):
            await refinery.review(
                "acme", run.run_id, Review(stance="approval", correction=correction)
            )
    held = await storage.labels("acme", run_ids=[run.run_id])
    assert {row.label_id for row in held} == {stance.label_id, comment.label_id}


async def test_export_leaves_voice_runs_out(tmp_path: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    engine = voice_engine(tmp_path, refinery)
    spoken = await engine.talk(
        "voice", "acme", saying("MARKER-SPOKEN"), mode="duplex", tools=LOOKUP
    )
    with pytest.raises(RuntimeError, match="the line dropped"):
        await engine.talk("voice", "acme", dropping, tools=LOOKUP)
    await refinery.review("acme", spoken.run_id, Review(comment="MARKER-SPOKEN"))

    bucket = MemoryBucket()
    assert await refinery.export("acme", bucket) == Exported(keys=[], rows={}, watermark=None)
    assert await refinery.export("acme", bucket, mode="duplex") == Exported(
        keys=[], rows={}, watermark=None
    )
    since = datetime.now(UTC)
    await refinery.review("acme", spoken.run_id, Review(comment="MARKER-SPOKEN again"))
    assert await refinery.export("acme", bucket, since=since) == Exported(
        keys=[], rows={}, watermark=since
    )
    assert bucket.objects == {}

    written = generation_record(reply="Refunded.")
    await refinery.record(written)
    row = await storage.get_run("acme", written.run_id)
    assert row is not None
    exported = await refinery.export("acme", bucket)
    assert exported.rows == {"generation": 1} and exported.watermark == row.recorded
    assert all(key.startswith("acme/support/generation/") for key in bucket.objects)
    assert len(bucket.objects) == 1
    assert all(b"MARKER-SPOKEN" not in data for data in bucket.objects.values())


async def test_replay_refuses_a_voice_run(tmp_path: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    spoken_in, other = tmp_path / "spoken_in", tmp_path / "other"
    spoken_in.mkdir()
    other.mkdir()
    engine = voice_engine(spoken_in, refinery)
    run = await engine.talk("voice", "acme", saying("Hello"), mode="duplex", tools=LOOKUP)
    label = await refinery.review("acme", run.run_id, Review(stance="approval"))
    refused = f"InputError: refinery: run {run.run_id!r} is a voice run; a spoken conversation cannot be replayed"

    [pair] = [pair async for pair in refinery.replay("acme", engine, agent="voice")]
    assert pair.candidate is None and pair.error == refused
    assert pair.recorded == run and [held.label_id for held in pair.labels] == [label.label_id]

    write_folder(
        other,
        {"voice": {"kind": "generation", "model": "strong", "instructions": "voice.md"}},
        files=FILES,
    )
    writer = Engine(other, providers=provided({"strong": Scripted(text_reply("A reply."))}))
    [again] = [pair async for pair in refinery.replay("acme", writer, agent="voice")]
    assert again.candidate is None and again.error == refused
