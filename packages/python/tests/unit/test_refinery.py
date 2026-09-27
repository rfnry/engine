from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError, NotFoundError
from rfnry_engine.models import Image, Message, Part, Reply, Text, ToolCall
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.labels import Outcome, Review
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.stop import Stop
from rfnry_engine.tools.tools import Tool

from .fakes import (
    TOOLS,
    Desk,
    MemoryBucket,
    Scripted,
    decision_record,
    delegate_triage,
    generation_record,
    text_reply,
)


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
    assert len({r.recorded for r in rows}) == 1 and rows[0].recorded >= parent.at
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
    with pytest.raises(
        InputError, match=f"^refinery: run '{decision.run_id}' asked no question 'urgnet'$"
    ):
        await refinery.review(
            "acme", decision.run_id, Review(correction={"urgent": True, "urgnet": False})
        )
    assert len(await refinery.storage.labels("acme")) == 1


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


def desk(fixtures: Path, refinery: Refinery, customers: int) -> Engine:
    replies = [delegate_triage(), text_reply("Handled.")] * customers
    return Engine(
        fixtures, generation=Scripted(*replies), decision=Desk(), tools=TOOLS, refinery=refinery
    )


def tree_ids(run: Run) -> set[str]:
    children = run.children if isinstance(run, Generation) else [run.routed_to]
    return {run.run_id}.union(*(tree_ids(child) for child in children if child is not None))


async def test_delete_runs_by_tag_takes_the_routed_and_delegated_runs_and_their_labels(
    fixtures: Path,
) -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    engine = desk(fixtures, refinery, customers=2)
    gone = await engine.run(
        "frontdesk", "acme", "It is down", tags={"customer": "c-81", "arm": "b"}
    )
    kept = await engine.run("frontdesk", "acme", "It is down", tags={"customer": "c-82"})
    assert isinstance(gone, Decision) and isinstance(gone.routed_to, Generation)
    assert len(tree_ids(gone)) == len(tree_ids(kept)) == 3
    await refinery.review("acme", gone.routed_to.run_id, Review(stance="objection"))
    await refinery.outcome("acme", gone.run_id, Outcome(status="failed"))
    await refinery.review("acme", kept.run_id, Review(stance="approval"))
    assert not await refinery.verify_absent("acme", tags={"customer": "c-81"})
    assert await refinery.delete_runs("acme", tags={"customer": "c-81"}) == 5
    assert {row.run_id for row in await storage.runs("acme")} == tree_ids(kept)
    assert [label.run_id for label in await storage.labels("acme")] == [kept.run_id]
    assert await refinery.verify_absent("acme", tags={"customer": "c-81"})
    assert await refinery.verify_absent("acme", run_ids=sorted(tree_ids(gone)))
    assert not await refinery.verify_absent("acme", tags={"customer": "c-82"})
    assert not await refinery.verify_absent("acme", run_ids=[kept.run_id])
    assert not await refinery.verify_absent("acme")
    assert await refinery.delete_runs("acme", tags={"customer": "c-81"}) == 0


async def test_a_tag_matches_by_containment(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
    engine = desk(fixtures, refinery, customers=1)
    await engine.run("frontdesk", "acme", "It is down", tags={"customer": "c-81", "arm": "b"})
    for absent in ({"customer": "c-81", "arm": "a"}, {"customer": "c-8"}, {"region": "eu"}):
        assert await refinery.verify_absent("acme", tags=absent)
    for present in ({"customer": "c-81"}, {"arm": "b"}, {"arm": "b", "customer": "c-81"}):
        assert not await refinery.verify_absent("acme", tags=present)


async def test_delete_runs_by_id_takes_that_run_and_its_descendants_only(fixtures: Path) -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    engine = desk(fixtures, refinery, customers=2)
    first = await engine.run("frontdesk", "acme", "It is down")
    second = await engine.run("frontdesk", "acme", "It is down")
    assert isinstance(first, Decision) and isinstance(first.routed_to, Generation)
    support = first.routed_to
    assert await refinery.delete_runs("acme", run_ids=[support.run_id]) == 2
    assert await refinery.verify_absent("acme", run_ids=[support.run_id])
    assert await storage.get_run("acme", first.run_id) is not None
    assert await refinery.delete_runs("acme", run_ids=[first.run_id, second.run_id]) == 4
    assert await refinery.verify_absent("acme")


async def test_delete_runs_never_reaches_another_scope(fixtures: Path) -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    engine = desk(fixtures, refinery, customers=2)
    mine = await engine.run("frontdesk", "acme", "It is down", tags={"customer": "c-81"})
    theirs = await engine.run("frontdesk", "globex", "It is down", tags={"customer": "c-81"})
    await refinery.review("globex", theirs.run_id, Review(stance="approval"))
    assert await refinery.delete_runs("acme", run_ids=[theirs.run_id]) == 0
    assert await refinery.delete_runs("acme", tags={"customer": "c-81"}) == 3
    assert await refinery.verify_absent("acme") and await refinery.verify_absent(
        "acme", run_ids=[mine.run_id]
    )
    assert {row.run_id for row in await storage.runs("globex")} == tree_ids(theirs)
    assert len(await storage.labels("globex")) == 1


async def test_delete_runs_and_verify_absent_take_exactly_one_selector() -> None:
    refinery = Refinery(InMemoryStorage())
    with pytest.raises(InputError, match="^refinery: delete_runs takes run_ids or tags$"):
        await refinery.delete_runs("acme")
    for verb in (refinery.delete_runs, refinery.verify_absent):
        with pytest.raises(InputError, match="takes run_ids or tags, not both$"):
            await verb("acme", run_ids=["r1"], tags={"customer": "c-81"})
        with pytest.raises(InputError, match="^refinery: run_ids must name at least one run$"):
            await verb("acme", run_ids=[])
        with pytest.raises(InputError, match="^refinery: tags must hold at least one tag$"):
            await verb("acme", tags={})
    assert await refinery.verify_absent("acme")


async def test_a_recorded_run_reads_back_as_engine_run_returned_it(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
    engine = desk(fixtures, refinery, customers=1)
    run = await engine.run("frontdesk", "acme", "It is down", tags={"customer": "c-81"})
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    support = run.routed_to
    [triage] = support.children
    got = await refinery.run("acme", run.run_id)
    assert got.model_dump() == run.model_dump() and got == run
    assert await refinery.run("acme", support.run_id) == support
    assert await refinery.run("acme", triage.run_id) == triage


async def test_delegated_children_read_back_in_the_order_of_their_steps(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
    calls = [delegate_triage(f"d{n}", f"ticket {n}").tool_calls[0] for n in range(3)]
    llm = Scripted(
        Reply(tool_calls=calls[:2], stop_reason="tool_use"),
        Reply(tool_calls=calls[2:], stop_reason="tool_use"),
        text_reply("All triaged."),
    )
    engine = Engine(fixtures, generation=llm, decision=Desk(), tools=TOOLS, refinery=refinery)
    run = await engine.run("support", "acme", "Three tickets")
    assert isinstance(run, Generation) and len(run.children) == 3
    got = await refinery.run("acme", run.run_id)
    assert got.model_dump() == run.model_dump()


async def test_a_stopped_run_read_back_continues_the_next_run(fixtures: Path) -> None:
    stop = Stop()

    async def lookup(args: dict[str, Any]) -> list[Part]:
        stop.after_step()
        return [Text(text=args["order_id"]), Image(media_type="image/png", data=bytes(range(256)))]

    lookup_call = ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"})
    llm = Scripted(
        Reply(tool_calls=[lookup_call], stop_reason="tool_use"),
        text_reply("The second charge is refunded."),
    )
    refinery = Refinery(InMemoryStorage())
    tools = [Tool("lookup_order", body=lookup), TOOLS[1]]
    engine = Engine(fixtures, generation=llm, tools=tools, refinery=refinery)
    run = await engine.run("support", "acme", "Charged twice.", stop=stop)
    assert isinstance(run, Generation) and run.stopped == "after_step"
    last = await refinery.run("acme", run.run_id)
    assert isinstance(last, Generation) and last.model_dump() == run.model_dump()
    after = await engine.run("support", "acme", "And the second charge?", history=last.conversation)
    question = Message(role="user", content="And the second charge?")
    assert llm.calls[1][1] == [*run.conversation, question]
    assert isinstance(after, Generation) and after.reply == "The second charge is refunded."


async def test_a_run_reads_back_only_in_its_own_scope() -> None:
    refinery = Refinery(InMemoryStorage())
    run = generation_record(scope="a")
    await refinery.record(run)
    assert await refinery.run("a", run.run_id) == run
    with pytest.raises(NotFoundError, match=f"^refinery: run '{run.run_id}' was not found$"):
        await refinery.run("b", run.run_id)
    with pytest.raises(NotFoundError, match="^refinery: run 'ghost' was not found$"):
        await refinery.run("a", "ghost")


async def test_every_refinery_verb_needs_a_scope(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
    run = generation_record()
    await refinery.record(run)
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS, refinery=refinery)
    verbs = [
        refinery.record(generation_record(scope="")),
        refinery.review("", run.run_id, Review(stance="approval")),
        refinery.outcome("", run.run_id, Outcome(status="failed")),
        refinery.conflicts(""),
        refinery.export("", MemoryBucket()),
        refinery.label("", engine, judge="judge"),
        refinery.delete_scope(""),
        refinery.verify_absent(""),
    ]
    for verb in verbs:
        with pytest.raises(InputError, match="^refinery: scope is required$"):
            await verb
    assert await refinery.storage.count_scope("") == 0
