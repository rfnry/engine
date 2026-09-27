from pathlib import Path

import pytest
from unit.fakes import TOOLS, Desk, MemoryBucket, Scripted, delegate_triage, text_reply

from rfnry_engine.engine import Engine
from rfnry_engine.errors import NotFoundError
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.labels import Outcome, Review
from rfnry_engine.refinery.refinery import Refinery, flatten
from rfnry_engine.refinery.storage import RunRow, Storage
from rfnry_engine.tools.engine_tools import EngineTools

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


async def test_nothing_of_a_is_reachable_from_b(storage: Storage) -> None:
    refinery = Refinery(storage)
    engine = Engine(
        FIXTURES,
        generation=Scripted(text_reply("MARKER-A"), text_reply("plain-b")),
        refinery=refinery,
        tools=TOOLS,
    )
    a = await engine.run("billing", "tenant-a", "hello")
    b = await engine.run("billing", "tenant-b", "hello")
    await refinery.review("tenant-a", a.run_id, Review(correction="MARKER-A-CORRECTION"))
    await refinery.outcome("tenant-a", a.run_id, Outcome(status="failed", detail="MARKER-A"))
    assert await storage.get_run("tenant-b", a.run_id) is None
    assert [r.run_id for r in await storage.runs("tenant-b")] == [b.run_id]
    assert await storage.labels("tenant-b") == []
    assert await refinery.conflicts("tenant-b") == []
    bucket = MemoryBucket()
    await refinery.export("tenant-b", bucket)
    assert all(b"MARKER-A" not in data for data in bucket.objects.values())
    assert all(key.startswith("tenant-b/") for key in bucket.objects)
    labeled = await refinery.label("tenant-b", engine, judge="judge")
    assert set(labeled.errors) <= {b.run_id} and a.run_id not in labeled.errors
    read = await EngineTools(engine, agent="support", scope="tenant-b").dispatch(
        "Read", {"path": "rules/refunds.md"}
    )
    assert "MARKER-A" not in str(read)


async def test_erasure_is_provable_and_falsifiable(storage: Storage) -> None:
    refinery = Refinery(storage)
    engine = Engine(FIXTURES, generation=Scripted(text_reply("x")), refinery=refinery, tools=TOOLS)
    run = await engine.run("billing", "tenant-a", "hello")
    await refinery.review("tenant-a", run.run_id, Review(stance="approval"))
    assert not await refinery.verify_absent("tenant-a")
    assert await refinery.delete_scope("tenant-a") == 2
    assert await refinery.verify_absent("tenant-a")
    await storage.put_run(
        "tenant-a",
        RunRow(
            scope="tenant-a",
            run_id="orphan",
            parent_run_id=None,
            agent="x",
            kind="generation",
            task=None,
            at=run.at,
            duration_ms=0,
            model=None,
            body={},
        ),
    )
    assert not await refinery.verify_absent("tenant-a")


async def test_nothing_of_one_customer_survives_its_erasure(storage: Storage) -> None:
    refinery = Refinery(storage)
    engine = Engine(
        FIXTURES,
        generation=Scripted(
            delegate_triage(text="MARKER-81"),
            text_reply("MARKER-81 handled."),
            delegate_triage(),
            text_reply("plain-82"),
        ),
        decision=Desk(),
        refinery=refinery,
        tools=TOOLS,
    )
    gone = await engine.run("frontdesk", "desk", "MARKER-81 is down", tags={"customer": "c-81"})
    kept = await engine.run("frontdesk", "desk", "it is down", tags={"customer": "c-82"})
    assert isinstance(gone, Decision) and isinstance(gone.routed_to, Generation)
    [triage] = gone.routed_to.children
    await refinery.review("desk", gone.routed_to.run_id, Review(correction="MARKER-81-FIX"))
    await refinery.review("desk", triage.run_id, Review(correction={"urgent": False}))
    await refinery.outcome("desk", gone.run_id, Outcome(status="failed", detail="MARKER-81"))
    await refinery.review("desk", kept.run_id, Review(stance="approval"))
    assert await refinery.delete_runs("desk", tags={"customer": "c-81"}) == 6
    erased = [gone.run_id, gone.routed_to.run_id, triage.run_id]
    assert all([await storage.get_run("desk", run_id) is None for run_id in erased])
    assert await storage.labels("desk", run_ids=erased) == []
    assert await refinery.verify_absent("desk", tags={"customer": "c-81"})
    assert await refinery.verify_absent("desk", run_ids=erased)
    assert not await refinery.verify_absent("desk", tags={"customer": "c-82"})
    assert [label.run_id for label in await storage.labels("desk")] == [kept.run_id]
    bucket = MemoryBucket()
    exported = await refinery.export("desk", bucket)
    assert exported.rows == {"generation": 1, "decision": 3}
    assert all(b"MARKER-81" not in data for data in bucket.objects.values())
    assert any(b"plain-82" in data for data in bucket.objects.values())
    with pytest.raises(NotFoundError):
        await refinery.review("desk", gone.run_id, Review(stance="approval"))
    stray = next(row for row in flatten(gone) if row.run_id == gone.routed_to.run_id)
    await storage.put_run("desk", stray)
    assert not await refinery.verify_absent("desk", run_ids=[gone.run_id])
    assert not await refinery.verify_absent("desk", tags={"customer": "c-81"})
