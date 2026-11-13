from pathlib import Path
from typing import Any

import pytest
from unit.fakes import MemoryBucket
from unit.trees import provided, write_folder

from rfnry_engine import ChoiceAnswer, Decision, Engine, Generation, Refinery, Storage

pytestmark = pytest.mark.live
FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"
AGAIN = (
    "It was charged three times, not twice. Check the order and the refund rule again and tell "
    "me exactly what is owed now."
)


async def test_a_generation_agent_reads_its_rules(adapters: Any) -> None:
    async with Engine(
        FIXTURES, providers=provided({"strong": adapters.generation(), "judge": None})
    ) as engine:
        run = await engine.run(
            "support", "live", "I was charged twice for order A-1. Is a refund owed?"
        )
    assert isinstance(run, Generation) and run.reply
    assert any(step.tool in ("Read", "Grep", "lookup_order") for step in run.steps)


async def test_a_decision_agent_routes(adapters: Any) -> None:
    async with Engine(
        FIXTURES,
        providers=provided({"strong": adapters.generation(), "judge": adapters.decision()}),
    ) as engine:
        run = await engine.run("frontdesk", "live", "My invoice shows two charges for one order.")
    assert isinstance(run, Decision)
    intent = run.answers["intent"]
    assert isinstance(intent, ChoiceAnswer) and intent.choice == "billing"
    assert run.routed_to is not None and run.routed_to.agent == "billing"


async def test_label_and_export_round_trip(adapters: Any, storage: Storage) -> None:
    refinery = Refinery(storage)
    async with Engine(
        FIXTURES,
        refinery=refinery,
        providers=provided({"strong": adapters.generation(), "judge": adapters.decision()}),
    ) as engine:
        await engine.run("billing", "live", "Refund my duplicate charge.")
        await engine.run("triage", "live", {"message": "Everything is down for all users."})
        labeled = await refinery.label("live", engine, judge="judge", teacher="teacher")
    assert labeled.judged == 1 and labeled.taught == 1 and labeled.errors == {}
    exported = await refinery.export("live", MemoryBucket())
    assert exported.rows["generation"] == 1 and exported.rows["decision"] == 2


async def test_a_coder_writes_and_runs_code(adapters: Any, tmp_path: Path) -> None:
    from rfnry_engine import Local, Workspace

    agents = tmp_path / "agents"
    agents.mkdir()
    write_folder(
        agents,
        {
            "coder": {
                "kind": "generation",
                "model": "strong",
                "description": "Writes and runs code",
                "instructions": "coder.md",
                "workspace": True,
            }
        },
        files={
            "coder.md": "You write Python files with Write and run them with Shell. Reply with the program's output only.\n"
        },
    )
    work = tmp_path / "work"
    work.mkdir()
    async with Engine(
        tmp_path / "agents", providers=provided({"strong": adapters.generation()})
    ) as engine:
        run = await engine.run(
            "coder",
            "live",
            "Write sum.py that prints the sum of 1..100, run it.",
            workspace=Workspace(Local(work), scope="live"),
        )
    assert isinstance(run, Generation)
    assert {"Write", "Shell"} <= {s.tool for s in run.steps}
    assert "5050" in run.reply


async def test_a_coder_calls_a_tool_from_its_script(adapters: Any, tmp_path: Path) -> None:
    import random

    from rfnry_engine import Local, Tool, Tools, Workspace

    values = {name: random.randint(100_000, 999_999) for name in "abc"}

    async def constant(args: dict[str, Any]) -> dict[str, Any]:
        return {"value": values[args["name"]]}

    agents = tmp_path / "agents"
    agents.mkdir()
    write_folder(
        agents,
        {
            "coder": {
                "kind": "generation",
                "model": "strong",
                "description": "Writes and runs code",
                "instructions": "coder.md",
                "workspace": True,
                "tools": {"allow": ["constant"]},
            }
        },
        runtime_tools=["constant"],
        files={
            "coder.md": "You write Python and run it with Shell. Reply with the program's output only.\n"
        },
    )
    work = tmp_path / "work"
    work.mkdir()
    tool = Tool(
        "constant",
        "Returns the constant of a name: a, b or c",
        body=constant,
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    async with Engine(
        tmp_path / "agents", providers=provided({"strong": adapters.generation()})
    ) as engine:
        run = await engine.run(
            "coder",
            "live",
            "Write one Python script that calls the tool `constant` for the names a, b and c through the tool "
            "bridge your Shell description explains, and prints their sum. Run it. Reply with the sum only.",
            workspace=Workspace(Local(work), scope="live"),
            tools=Tools([tool]),
        )
    assert isinstance(run, Generation)
    total = sum(values.values())
    assert str(total) in run.reply.replace(",", "").replace(" ", "")
    calls = [s for s in run.steps if s.tool == "constant"]
    assert calls
    assert all(s.by for s in calls)


async def test_reasoning_goes_back_only_where_it_was_given(adapters: Any) -> None:
    async with Engine(
        FIXTURES, providers=provided({"strong": adapters.generation(), "judge": None})
    ) as engine:
        today, tomorrow = "Today is 2026-10-11.", "Today is 2026-10-12."

        async def reasoned(text: str, **given: Any) -> Generation:
            runs: list[Generation] = []
            while len(runs) < 3 and not any(
                m.reasoning for m in (runs[-1].added if runs else []) if m.role == "assistant"
            ):
                run = await engine.run("support", "live", text, **given)
                assert isinstance(run, Generation)
                runs.append(run)
            return runs[-1]

        first = await reasoned(
            "I was charged twice for order A-1. Read the refund rule and tell me what is owed.",
            context=today,
        )
        assert any(m.reasoning for m in first.conversation if m.role == "assistant"), (
            "the live model returned no reasoning in three first runs; the check would prove nothing"
        )
        same = await reasoned(AGAIN, history=first.conversation, context=today)
        moved = await engine.run(
            "support", "live", AGAIN, history=first.conversation, context=tomorrow
        )
        assert isinstance(same, Generation) and isinstance(moved, Generation)
        assert any(m.reasoning for m in same.added if m.role == "assistant"), (
            "the live model returned no reasoning in three second runs; the cut check would prove nothing"
        )
        conversation = [*first.conversation, *same.added]
        cut = await engine.run(
            "support",
            "live",
            "Thanks, that is all.",
            history=conversation,
            start=len(first.conversation),
            context=today,
        )
    assert isinstance(cut, Generation) and cut.reply
    assert all(m.reasoning is None for m in moved.messages)
    assert all(m.reasoning is None for m in cut.messages)
    kept = [m.reasoning for m in first.conversation if m.role == "assistant"]
    assert [m.reasoning for m in same.messages if m.role == "assistant"] == kept, (
        "reasoning given under the same history must go back unchanged"
    )
