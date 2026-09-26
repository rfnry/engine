from pathlib import Path

from unit.fakes import TOOLS, MemoryBucket, Scripted, text_reply

from rfnry_engine.engine import Engine
from rfnry_engine.refinery.labels import Outcome, Review
from rfnry_engine.refinery.refinery import Refinery
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
