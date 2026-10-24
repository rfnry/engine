import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from rfnry_engine.engine import Engine
from rfnry_engine.models import Check, Choice, Message, Question, Reply, Score, ToolCall
from rfnry_engine.records import Decision, Fingerprint, Generation, Run
from rfnry_engine.refinery.labels import Review
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import (
    Answering,
    Desk,
    MemoryBucket,
    Scripted,
    ask,
    by_hand,
    decision_record,
    delegate_triage,
    generation_record,
    now,
    text_reply,
)

NO_TOOLS = hashlib.sha256(b"[]").hexdigest()[:16]


def asked(questions: dict[str, Question]) -> str:
    ordered = {qid: questions[qid].model_dump(mode="json") for qid in sorted(questions)}
    canonical = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def tree(fixtures: Path, tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    return root


def engine(root: Path, *replies: Any, **kwargs: Any) -> Engine:
    llm = Scripted(*(replies or [text_reply("ok")]))
    return Engine(root, **kwargs, models={"strong": llm, "judge": Desk()})


async def generation(root: Path, **kwargs: Any) -> Generation:
    run = await engine(root).run("support", "acme", "hi", **kwargs)
    assert isinstance(run, Generation)
    return run


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


async def test_the_agent_fingerprint_is_every_file_of_its_folder_by_path_and_content(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    run = await generation(root)
    assert run.fingerprint.agent == by_hand(root / "support")
    assert len(run.fingerprint.agent) == 16 and len(run.fingerprint.tools) == 16
    specs = sorted((spec.model_dump(mode="json") for spec in run.tools), key=lambda t: t["name"])
    canonical = json.dumps(specs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert run.fingerprint.tools == hashlib.sha256(canonical.encode()).hexdigest()[:16]


async def test_an_unchanged_tree_gives_the_same_fingerprint_on_every_run_and_every_load(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    loaded = engine(root, text_reply("a"), text_reply("b"))
    first = await loaded.run("support", "acme", "hi")
    second = await loaded.run("support", "acme", "hi")
    again = await generation(root)
    elsewhere = await generation(fixtures)
    assert first.fingerprint == second.fingerprint == again.fingerprint == elsewhere.fingerprint


async def test_what_varies_per_run_leaves_the_fingerprint_alone(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    plain = await generation(root)
    varied = await generation(
        root,
        context="Acme is on the enterprise plan.",
        tags={"commit": "4f2c9e1"},
        history=[Message(role="user", content="earlier"), Message(role="assistant", content="ok")],
        task="triage",
    )
    other = await engine(root).run("support", "acme", {"message": "a different input"})
    assert plain.fingerprint == varied.fingerprint == other.fingerprint


async def test_an_edit_to_a_rule_changes_the_agent_fingerprint_and_not_the_tools(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    before = await generation(root)
    (root / "support/rules/refunds.md").write_text("Refund only the second charge.\n")
    after = await generation(root)
    assert before.fingerprint.agent != after.fingerprint.agent
    assert before.fingerprint.tools == after.fingerprint.tools


async def test_the_folder_is_read_when_the_engine_loads_it(fixtures: Path, tmp_path: Path) -> None:
    root = tree(fixtures, tmp_path)
    loaded = engine(root, text_reply("a"), text_reply("b"))
    first = await loaded.run("support", "acme", "hi")
    (root / "support/rules/refunds.md").write_text("Refund only the second charge.\n")
    second = await loaded.run("support", "acme", "hi")
    assert first.fingerprint == second.fingerprint
    assert (await generation(root)).fingerprint != first.fingerprint


async def test_a_new_file_or_a_renamed_one_changes_it_and_a_hidden_one_does_not(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    before = await generation(root)
    (root / "support/.notes").write_text("scratch")
    (root / "support/.cache").mkdir()
    (root / "support/.cache/x.md").write_text("scratch")
    (root / "support/knowledge/.draft.md").write_text("scratch")
    assert (await generation(root)).fingerprint == before.fingerprint
    (root / "support/knowledge/plans.md").rename(root / "support/knowledge/tiers.md")
    renamed = await generation(root)
    assert renamed.fingerprint != before.fingerprint
    (root / "support/knowledge/rates.pdf").write_bytes(b"%PDF-1.4")
    assert (await generation(root)).fingerprint != renamed.fingerprint


async def test_a_link_out_of_the_folder_is_not_part_of_it(fixtures: Path, tmp_path: Path) -> None:
    root = tree(fixtures, tmp_path)
    outside = tmp_path / "secret.md"
    outside.write_text("one")
    (root / "support/knowledge/secret.md").symlink_to(outside)
    before = await generation(root)
    outside.write_text("two")
    assert (await generation(root)).fingerprint == before.fingerprint


async def test_a_folder_tool_s_schema_changes_the_tools_fingerprint(
    fixtures: Path, tmp_path: Path
) -> None:
    plain = await generation(fixtures)
    zoned = tree(fixtures, tmp_path / "zoned")
    edit(
        zoned / "support/tools/now.json",
        '"properties": {}',
        '"properties": {"zone": {"type": "string"}}',
    )
    defaulted = tree(fixtures, tmp_path / "defaulted")
    edit(
        defaulted / "support/tools/now.json",
        '"properties": {}',
        '"properties": {"zone": {"type": "string", "default": "UTC"}}',
    )
    prints = [plain.fingerprint, (await generation(zoned)).fingerprint]
    prints.append((await generation(defaulted)).fingerprint)
    assert len({p.tools for p in prints}) == 3 and len({p.agent for p in prints}) == 3


async def test_the_order_an_agent_keeps_tools_in_changes_its_fingerprint_and_not_the_tools(
    fixtures: Path, tmp_path: Path
) -> None:
    listed = tree(fixtures, tmp_path / "listed")
    flipped = tree(fixtures, tmp_path / "flipped")
    for root, kept in ((listed, "[lookup_order, now]"), (flipped, "[now, lookup_order]")):
        edit(root / "support/AGENT.md", "agents: [triage]\n", f"agents: [triage]\ntools: {kept}\n")
    llm = Scripted(text_reply("ok"))
    run = await Engine(flipped, models={"strong": llm, "judge": Desk()}).run(
        "support", "acme", "hi"
    )
    assert isinstance(run, Generation)
    assert [t.name for t in llm.calls[0][2]] == ["lookup_order", "now", "Delegate"]
    original = await generation(listed)
    assert run.fingerprint.tools == original.fingerprint.tools
    assert run.fingerprint.agent != original.fingerprint.agent


async def test_a_folder_tool_s_description_changes_both_fingerprints_its_script_the_agent_s(
    fixtures: Path, tmp_path: Path
) -> None:
    plain = await generation(fixtures)
    reworded = tree(fixtures, tmp_path / "reworded")
    edit(reworded / "support/tools/lookup_order.json", "Fetch an order by id.", "Fetch an order.")
    rewritten = tree(fixtures, tmp_path / "rewritten")
    edit(rewritten / "support/tools/lookup_order.py", "return {", "return {'by': 'id', ")
    described, scripted = await generation(reworded), await generation(rewritten)
    assert plain.fingerprint.tools != described.fingerprint.tools
    assert plain.fingerprint.agent != described.fingerprint.agent
    assert plain.fingerprint.tools == scripted.fingerprint.tools
    assert plain.fingerprint.agent != scripted.fingerprint.agent


class Verdict(BaseModel):
    refund: bool


async def test_the_tools_a_run_is_offered_are_the_tools_it_is_fingerprinted_by(
    fixtures: Path,
) -> None:
    plain = await generation(fixtures)
    expecting = await generation(fixtures, expect=Verdict)
    recall = Tools([Tool("recall", "Recall a fact", body=now)])
    mounted = await engine(fixtures).run("support", "acme", "hi", tools=recall)
    assert isinstance(mounted, Generation) and "recall" in {t.name for t in mounted.tools}
    prints = [run.fingerprint for run in (plain, expecting, mounted)]
    assert len({p.tools for p in prints}) == 3
    assert len({p.agent for p in prints}) == 1


async def test_a_decision_run_and_every_nested_run_carry_their_own(fixtures: Path) -> None:
    engine_ = engine(fixtures, delegate_triage(), text_reply("Handled."))
    parent = await engine_.run("support", "acme", "It is down")
    assert isinstance(parent, Generation)
    [child] = parent.children
    assert isinstance(child, Decision)
    assert child.fingerprint == Fingerprint(
        agent=by_hand(fixtures / "triage"), tools=NO_TOOLS, questions=asked(child.questions)
    )
    assert parent.fingerprint.agent == by_hand(fixtures / "support")
    routed = await engine(fixtures).run("frontdesk", "acme", "It is down")
    assert isinstance(routed, Decision)
    assert routed.fingerprint.agent == by_hand(fixtures / "frontdesk")
    assert routed.routed_to is not None
    assert routed.routed_to.fingerprint.agent == parent.fingerprint.agent


async def test_the_context_a_run_was_given_is_recorded_and_reaches_its_children(
    fixtures: Path,
) -> None:
    engine_ = engine(fixtures, delegate_triage(), text_reply("Handled."))
    run = await engine_.run("support", "acme", "It is down", context="Acme is enterprise.")
    assert isinstance(run, Generation) and run.context == "Acme is enterprise."
    assert "Acme is enterprise." in run.system
    plain = await generation(fixtures)
    assert plain.context is None
    routed = await engine(fixtures).run("frontdesk", "acme", "It is down", context="On call: Ana.")
    assert isinstance(routed, Decision) and isinstance(routed.routed_to, Generation)
    assert routed.routed_to.context == "On call: Ana."


async def test_the_record_and_every_exported_row_carry_the_fingerprint(
    fixtures: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    engine_ = engine(fixtures, delegate_triage(), text_reply("Sorry, no."), refinery=refinery)
    run = await engine_.run("support", "acme", "It is down", context="Acme is enterprise.")
    assert isinstance(run, Generation)
    read = await refinery.run("acme", run.run_id)
    assert isinstance(read, Generation)
    assert read.fingerprint == run.fingerprint and read.context == "Acme is enterprise."
    assert read.children[0].fingerprint == run.children[0].fingerprint
    await refinery.review("acme", run.run_id, Review(correction="Refund it."))
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    rows = {
        key.split("/")[2]: [json.loads(line) for line in data.decode().splitlines()]
        for key, data in bucket.objects.items()
    }
    child = run.children[0].fingerprint
    assert child is not None
    assert [row["fingerprint"] for row in rows["generation"]] == [run.fingerprint.model_dump()]
    assert [row["fingerprint"] for row in rows["preference"]] == [run.fingerprint.model_dump()]
    assert rows["decision"] and all(
        row["fingerprint"] == child.model_dump() for row in rows["decision"]
    )


async def test_a_tool_call_does_not_move_the_fingerprint(fixtures: Path) -> None:
    call = ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"})
    async with engine(
        fixtures, Reply(tool_calls=[call], stop_reason="tool_use"), text_reply("Found it.")
    ) as engine_:
        called = await engine_.run("support", "acme", "Where is A-1?")
    assert called.fingerprint == (await generation(fixtures)).fingerprint


AREA = Choice(
    instructions={"question": "Which area?", "within": "orders"},
    options={"delivery": {"what": "Parcels", "examples": ["Where is it?"]}, "returns": None},
)


def triage(fixtures: Path, root: Path | None = None) -> Engine:
    return Engine(root or fixtures, models={"judge": Answering()})


async def test_a_decision_run_s_questions_part_is_the_questions_it_asked(fixtures: Path) -> None:
    folder = await triage(fixtures).run("triage", "acme", "It is down")
    both = await ask(triage(fixtures), "triage", "acme", "It is down", questions={"area": AREA})
    assert isinstance(folder, Decision) and isinstance(both, Decision)
    assert folder.fingerprint.questions == asked(folder.questions)
    assert both.fingerprint.questions == asked(both.questions)
    assert len(folder.fingerprint.questions or "") == 16
    assert folder.fingerprint.questions != both.fingerprint.questions
    assert folder.fingerprint.agent == both.fingerprint.agent
    assert folder.fingerprint.tools == both.fingerprint.tools == NO_TOOLS


async def test_the_questions_part_is_stable_across_runs_loads_and_the_order_questions_come_in(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    other = Check(instructions="Is the customer angry?")
    loaded = triage(fixtures)
    first = await ask(loaded, "triage", "acme", "a", questions={"area": AREA, "angry": other})
    again = await ask(loaded, "triage", "acme", "b", questions={"angry": other, "area": AREA})
    elsewhere = await ask(
        triage(fixtures, root),
        "triage",
        "acme",
        "c",
        questions={"area": AREA.model_copy(), "angry": other},
    )
    assert first.fingerprint.questions is not None
    assert first.fingerprint == again.fingerprint == elsewhere.fingerprint


def _print(run: object) -> str | None:
    assert isinstance(run, Decision)
    return run.fingerprint.questions


async def test_the_questions_part_changes_with_a_question_its_options_or_their_order(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    engine_ = triage(fixtures, root)
    base = _print(await ask(engine_, "triage", "acme", "x", questions={"area": AREA}))
    variants = [
        AREA.model_copy(update={"options": dict(reversed(list(AREA.options.items())))}),
        AREA.model_copy(update={"options": {**AREA.options, "returns": "Sending it back"}}),
        AREA.model_copy(update={"instructions": {"within": "orders", "question": "Which area?"}}),
        AREA.model_copy(update={"instructions": "Which area?"}),
    ]
    prints = [
        _print(await ask(engine_, "triage", "acme", "x", questions={"area": variant}))
        for variant in variants
    ]
    renamed = _print(await ask(engine_, "triage", "acme", "x", questions={"zone": AREA}))
    assert len({base, *prints, renamed}) == 6
    (root / "triage/questions/urgent.json").write_text(
        '{"type": "check", "instructions": "Is it urgent?"}'
    )
    edited = triage(fixtures, root)
    assert _print(await ask(edited, "triage", "acme", "x", questions={"area": AREA})) != base
    levels = Score(instructions="How bad?", levels=["Mild", "Severe"])
    reordered = Score(instructions="How bad?", levels=["Severe", "Mild"])
    assert _print(
        await ask(engine_, "triage", "acme", "x", questions={"area": AREA, "bad": levels})
    ) != _print(
        await ask(engine_, "triage", "acme", "x", questions={"area": AREA, "bad": reordered})
    )


async def test_a_generation_run_has_no_questions_part(fixtures: Path) -> None:
    run = await generation(fixtures)
    assert run.fingerprint.questions is None


async def test_the_questions_part_is_recorded_exported(
    fixtures: Path,
    storage: Storage,
) -> None:
    refinery = Refinery(storage)
    run = await Engine(fixtures, refinery=refinery, models={"judge": Desk()}).run(
        "triage", "acme", "x"
    )
    assert isinstance(run, Decision)
    assert (await refinery.run("acme", run.run_id)).fingerprint == run.fingerprint
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    rows = {
        row["run_id"]: row
        for data in bucket.objects.values()
        for row in map(json.loads, data.decode().splitlines())
    }
    assert rows[run.run_id]["fingerprint"] == {
        "agent": run.fingerprint.agent,
        "tools": NO_TOOLS,
        "questions": run.fingerprint.questions,
    }


async def test_an_edit_to_a_read_folder_changes_the_fingerprint_of_every_agent_reading_it(
    tmp_path: Path,
) -> None:
    for name in ("coding", "writing", "other"):
        (tmp_path / name).mkdir()
        front = "reads: [books]\n" if name != "other" else ""
        (tmp_path / name / "AGENT.md").write_text(
            f"---\nname: {name}\nkind: generation\nmodel: strong\n{front}---\nHi\n"
        )
    (tmp_path / "books").mkdir()
    (tmp_path / "books" / "a.md").write_text("one\n")
    before = Engine(tmp_path, models={"strong": Scripted()})
    (tmp_path / "books" / "a.md").write_text("two\n")
    after = Engine(tmp_path, models={"strong": Scripted()})
    for name in ("coding", "writing"):
        assert before._spec(name).fingerprint != after._spec(name).fingerprint
    assert before._spec("other").fingerprint == after._spec("other").fingerprint


@pytest.mark.parametrize("record", [generation_record(), decision_record()])
def test_a_record_must_carry_a_fingerprint(record: Run) -> None:
    body = record.model_dump(mode="json")
    del body["fingerprint"]
    for missing in ({}, {"fingerprint": None}):
        with pytest.raises(ValidationError, match="fingerprint"):
            type(record).model_validate({**body, **missing})
