import pytest

from rfnry_engine.errors import InputError, NotFoundError
from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.labels import Outcome, Review
from rfnry_engine.refinery.refinery import Refinery

from .fakes import decision_record, generation_record


async def test_record_flattens_children_into_rows() -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    child = decision_record()
    parent = generation_record(children=[child])
    child = child.model_copy(update={"parent_run_id": parent.run_id})
    parent = parent.model_copy(update={"children": [child]})
    await refinery.record(parent)
    rows = await storage.runs("acme")
    assert {r.run_id: r.parent_run_id for r in rows} == {
        parent.run_id: None,
        child.run_id: parent.run_id,
    }
    stored = await storage.get_run("acme", parent.run_id)
    assert stored is not None and stored.body["children"] == [] and stored.kind == "generation"


async def test_review_and_outcome_become_labels_with_their_source() -> None:
    refinery = Refinery(InMemoryStorage())
    run = generation_record()
    await refinery.record(run)
    review = await refinery.review(
        "acme", run.run_id, Review(stance="objection", correction="Refund in full.")
    )
    outcome = await refinery.outcome(
        "acme", run.run_id, Outcome(status="failed", detail="escalated")
    )
    labels = await refinery.storage.labels("acme", run_ids=[run.run_id])
    assert sorted(label.source for label in labels) == ["human", "world"]
    assert {labels[0].label_id, labels[1].label_id} == {review.label_id, outcome.label_id}
    world = next(label for label in labels if label.source == "world")
    assert world.body["outcome"]["status"] == "failed"


async def test_a_correction_must_match_the_run_kind() -> None:
    refinery = Refinery(InMemoryStorage())
    generation, decision = generation_record(), decision_record()
    await refinery.record(generation)
    await refinery.record(decision)
    with pytest.raises(InputError):
        await refinery.review("acme", generation.run_id, Review(correction={"urgent": False}))
    with pytest.raises(InputError):
        await refinery.review("acme", decision.run_id, Review(correction="text"))
    label = await refinery.review("acme", decision.run_id, Review(correction={"urgent": False}))
    assert label.review is not None and label.review.correction == {"urgent": False}


async def test_review_needs_something_and_a_known_run() -> None:
    refinery = Refinery(InMemoryStorage())
    with pytest.raises(ValueError):
        Review()
    with pytest.raises(NotFoundError, match="^refinery: run 'ghost' was not found$"):
        await refinery.review("acme", "ghost", Review(stance="approval"))
    with pytest.raises(NotFoundError, match="^refinery: run 'ghost' was not found$"):
        await refinery.outcome("acme", "ghost", Outcome(status="failed"))


async def test_scopes_never_mix_and_deletion_is_provable() -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    a, b = generation_record(scope="a"), generation_record(scope="b")
    await refinery.record(a)
    await refinery.record(b)
    await refinery.review("a", a.run_id, Review(stance="approval"))
    assert await storage.get_run("b", a.run_id) is None
    assert [r.run_id for r in await storage.runs("b")] == [b.run_id]
    assert await storage.labels("b") == []
    with pytest.raises(NotFoundError):
        await refinery.review("b", a.run_id, Review(stance="approval"))
    assert await refinery.delete_scope("a") == 2
    assert await refinery.verify_absent("a") and not await refinery.verify_absent("b")
