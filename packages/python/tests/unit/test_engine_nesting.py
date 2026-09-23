import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import Answer, CheckAnswer, ChoiceAnswer, Reply, ScoreAnswer, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import TOOLS, Fixed, Scripted, text_reply

TRIAGE: dict[str, Answer] = {"urgent": CheckAnswer(probability=0.9), "severity": ScoreAnswer(score=3, probabilities=[0, 0, 0, 1])}


def _delegate(agent: str, text: str, call_id: str = "d1") -> Reply:
    return Reply(tool_calls=[ToolCall(id=call_id, name="Delegate", args={"agent": agent, "input": text})], stop_reason="tool_use")


async def test_delegate_tool_exists_only_when_agents_are_named(fixtures: Path) -> None:
    llm = Scripted(text_reply("a"), text_reply("b"))
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
    await engine.run("support", "acme", "x")
    await engine.run("billing", "acme", "x")
    assert "Delegate" in [t.name for t in llm.calls[0][2]]
    assert "Delegate" not in [t.name for t in llm.calls[1][2]]
    spec = next(t for t in llm.calls[0][2] if t.name == "Delegate")
    assert spec.input_schema["properties"]["agent"]["enum"] == ["triage"]
    assert "triage: Judges urgency" in spec.description
    assert "Delegate" in EngineTools(engine, agent="support", scope="acme")
    assert "Delegate" not in EngineTools(engine, agent="billing", scope="acme")


async def test_generation_agent_consults_a_decision_agent(fixtures: Path) -> None:
    llm = Scripted(_delegate("triage", "It is down"), text_reply("Escalating."))
    engine = Engine(fixtures, generation=llm, decision=Fixed(TRIAGE), tools=TOOLS)
    run = await engine.run("support", "acme", "It is down")
    assert isinstance(run, Generation) and len(run.children) == 1
    child = run.children[0]
    assert isinstance(child, Decision) and child.agent == "triage" and child.parent_run_id == run.run_id
    assert run.steps[0].child_run_id == child.run_id
    assert json.loads(str(run.steps[0].result))["urgent"]["probability"] == 0.9


async def test_the_whole_chain_frontdesk_support_triage(fixtures: Path) -> None:
    intent = {"intent": ChoiceAnswer(choice="technical", probabilities={"technical": 0.9, "billing": 0.05, "other": 0.05})}
    answers: list[Any] = [intent, TRIAGE]
    dm = Fixed(lambda state: answers.pop(0))
    llm = Scripted(_delegate("triage", "It is down"), text_reply("Handled."))
    engine = Engine(fixtures, generation=llm, decision=dm, tools=TOOLS)
    run = await engine.run("frontdesk", "acme", "It is down")
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.agent == "support" and run.routed_to.children[0].agent == "triage"


async def test_a_loop_is_an_error_result_not_a_crash(tmp_path: Path) -> None:
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        other = "b" if name == "a" else "a"
        (tmp_path / name / "AGENT.md").write_text(f"---\nname: {name}\nkind: generation\nagents: [{other}]\n---\nhi\n")
    llm = Scripted(_delegate("b", "go"), _delegate("a", "back"), text_reply("b done"), text_reply("a done"))
    run = await Engine(tmp_path, generation=llm).run("a", "acme", "start")
    assert isinstance(run, Generation)
    child = run.children[0]
    assert isinstance(child, Generation) and child.steps[0].is_error
    assert child.steps[0].result == "DelegationLoopError: delegate: loop at a -> b -> a"


async def test_depth_is_capped(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text("---\nname: a\nkind: generation\nagents: [b]\n---\nhi\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "AGENT.md").write_text("---\nname: b\nkind: generation\n---\nhi\n")
    llm = Scripted(_delegate("b", "go"), text_reply("a done"))
    run = await Engine(tmp_path, generation=llm, max_depth=1).run("a", "acme", "start")
    assert isinstance(run, Generation) and run.steps[0].is_error
    assert "DelegationDepthError" in run.steps[0].result and run.children == []


async def test_the_child_serves_the_same_tenant_with_its_context(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text("---\nname: a\nkind: generation\nagents: [b]\n---\nhi\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "AGENT.md").write_text("---\nname: b\nkind: generation\n---\nhi\n")
    llm = Scripted(_delegate("b", "go"), text_reply("b done"), text_reply("a done"))
    run = await Engine(tmp_path, generation=llm).run("a", "acme", "start", context="Acme is enterprise.")
    assert isinstance(run, Generation)
    child = run.children[0]
    assert isinstance(child, Generation) and child.scope == "acme"
    assert "Acme is enterprise." in child.system
    assert run.steps[0].result == "b done"


async def test_dispatch_delegate_outside_a_run(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(), decision=Fixed(TRIAGE), tools=TOOLS)
    tools = EngineTools(engine, agent="support", scope="acme")
    result = await tools.dispatch("Delegate", {"agent": "triage", "input": "It is down"})
    assert isinstance(result, str) and json.loads(result)["urgent"]["probability"] == 0.9


async def test_dispatch_delegate_gives_the_child_the_tenant_tools(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text("---\nname: a\nkind: generation\nagents: [b]\n---\nhi\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "AGENT.md").write_text("---\nname: b\nkind: generation\n---\nhi\n")

    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    pinned = Tools([Tool("ping", "Ping", body=ping)])
    llm = Scripted(text_reply("b done"))
    tools = EngineTools(Engine(tmp_path, generation=llm), agent="a", scope="acme", tools=pinned)
    assert await tools.dispatch("Delegate", {"agent": "b", "input": "go"}) == "b done"
    assert "ping" in [t.name for t in llm.calls[0][2]]


async def test_delegate_is_not_bound_by_the_tool_deadline(fixtures: Path) -> None:
    class Slow(Fixed):
        async def decide(self, state: Any, questions: Any) -> Any:
            await asyncio.sleep(0.2)
            return TRIAGE

    llm = Scripted(_delegate("triage", "It is down"), text_reply("Escalating."))
    engine = Engine(fixtures, generation=llm, decision=Slow(TRIAGE), tools=TOOLS, tool_timeout=0.05)
    run = await engine.run("support", "acme", "It is down")
    assert isinstance(run, Generation) and not run.steps[0].is_error
    assert run.children[0].agent == "triage"


BAD_DELEGATIONS = [
    ({"agent": "billing", "input": "x"}, "delegate: agent 'support' may not delegate to 'billing'"),
    ({"agent": "triage"}, "delegate: invalid arguments: "),
]


async def test_a_refused_delegation_reads_the_same_in_a_run_and_from_code(fixtures: Path) -> None:
    for args, message in BAD_DELEGATIONS:
        call = Reply(tool_calls=[ToolCall(id="d", name="Delegate", args=args)], stop_reason="tool_use")
        llm = Scripted(call, text_reply("done"))
        engine = Engine(fixtures, generation=llm, decision=Fixed(TRIAGE), tools=TOOLS)
        run = await engine.run("support", "acme", "x")
        assert isinstance(run, Generation) and run.steps[0].is_error and run.children == []
        assert str(run.steps[0].result).startswith(f"InputError: {message}")
        with pytest.raises(InputError) as raised:
            await EngineTools(engine, agent="support", scope="acme").dispatch("Delegate", args)
        assert run.steps[0].result == f"InputError: {raised.value}"
