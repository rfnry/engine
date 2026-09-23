from pathlib import Path
from typing import Any

import pytest
from unit.fakes import TOOLS, MemoryBucket

from rfnry_engine import ChoiceAnswer, Decision, Engine, Generation, InMemoryStorage, Refinery

pytestmark = pytest.mark.live
FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


async def test_a_generation_agent_reads_its_rules(adapters: Any) -> None:
    engine = Engine(FIXTURES, generation=adapters.generation(), tools=TOOLS)
    run = await engine.run("support", "live", "I was charged twice for order A-1. Is a refund owed?")
    assert isinstance(run, Generation) and run.reply
    assert any(step.tool in ("Read", "Grep", "lookup_order") for step in run.steps)


async def test_a_decision_agent_routes(adapters: Any) -> None:
    engine = Engine(FIXTURES, generation=adapters.generation(), decision=adapters.decision(), tools=TOOLS)
    run = await engine.run("frontdesk", "live", "My invoice shows two charges for one order.")
    assert isinstance(run, Decision)
    intent = run.answers["intent"]
    assert isinstance(intent, ChoiceAnswer) and intent.choice == "billing"
    assert run.routed_to is not None and run.routed_to.agent == "billing"


async def test_label_and_export_round_trip(adapters: Any) -> None:
    refinery = Refinery(InMemoryStorage())
    engine = Engine(FIXTURES, generation=adapters.generation(), decision=adapters.decision(), refinery=refinery, tools=TOOLS)
    await engine.run("billing", "live", "Refund my duplicate charge.")
    await engine.run("triage", "live", {"message": "Everything is down for all users."})
    labeled = await refinery.label("live", engine, judge="judge", teacher="teacher")
    assert labeled.judged == 1 and labeled.taught == 1 and labeled.errors == {}
    exported = await refinery.export("live", MemoryBucket())
    assert exported.rows["generation"] == 1 and exported.rows["decision"] == 2


async def test_a_coder_writes_and_runs_code(adapters: Any, tmp_path: Path) -> None:
    from rfnry_engine import Workspace

    agents = tmp_path / "agents" / "coder"
    agents.mkdir(parents=True)
    (agents / "AGENT.md").write_text(
        "---\nname: coder\nkind: generation\ndescription: Writes and runs code\nworkspace: true\n---\n"
        "You write Python files with Write and run them with Shell. Reply with the program's output only.\n",
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    engine = Engine(tmp_path / "agents", generation=adapters.generation())
    run = await engine.run("coder", "live", "Write sum.py that prints the sum of 1..100, run it.", workspace=Workspace(work))
    assert isinstance(run, Generation)
    assert {"Write", "Shell"} <= {s.tool for s in run.steps}
    assert "5050" in run.reply
