import re
import shutil
from pathlib import Path

import pytest

from rfnry_engine import Engine
from rfnry_engine.bundle import build_bundle
from rfnry_engine.models import ToolSpec
from rfnry_engine.records import Generation
from rfnry_engine.tree import load_tree

from .fakes import Scripted, text_reply

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


def _labels(bundle: str) -> list[str]:
    return re.findall(r"^<<<([A-Z]+)>>>$", bundle, flags=re.MULTILINE)


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=f"{name} it", input_schema={})


def test_sections_in_order_with_missing_ones_skipped() -> None:
    agents = load_tree(FIXTURES)
    tools = [ToolSpec(name="Read", description="Read a file", input_schema={})]
    bundle = build_bundle(
        agents["support"],
        tools=tools,
        agents=[agents["triage"]],
        context="Plan: Team",
    )
    assert _labels(bundle) == ["AGENT", "FILES", "SKILLS", "TOOLS", "AGENTS", "CONTEXT"]
    assert (
        "<<<FILES>>>\nknowledge/\n  plans.md — The three plans\nrules/\n"
        "  refunds.md — When a refund is owed\n\n<<<SKILLS>>>\n"
        "skills/escalate.md — Hand a conversation to a person\n\n<<<TOOLS>>>"
    ) in bundle
    assert "- Read: Read a file" in bundle
    assert "- triage: Judges urgency and severity of a message" in bundle
    assert bundle.endswith("<<<CONTEXT>>>\nPlan: Team")


def test_an_old_index_file_is_an_ordinary_file(tmp_path: Path) -> None:
    (tmp_path / "a" / "rules").mkdir(parents=True)
    (tmp_path / "a" / "AGENT.md").write_text(
        "---\nname: a\nkind: generation\nmodel: strong\n---\nHi\n"
    )
    (tmp_path / "a" / "INDEX.md").write_text("Read rules/x.md first.\n")
    (tmp_path / "a" / "rules" / "_index.md").write_text("- x.md — the x rule\n")
    bundle = build_bundle(load_tree(tmp_path)["a"], tools=[_tool("Read")], agents=[], context=None)
    assert _labels(bundle) == ["AGENT", "FILES", "TOOLS"]
    assert "Read rules/x.md first." not in bundle
    assert "<<<FILES>>>\nINDEX.md\nrules/\n  _index.md\n\n<<<TOOLS>>>" in bundle


def test_bare_agent_has_only_agent_and_tools() -> None:
    agents = load_tree(FIXTURES)
    bundle = build_bundle(agents["billing"], tools=[], agents=[], context=None)
    assert _labels(bundle) == ["AGENT"]
    assert bundle == "<<<AGENT>>>\nYou are billing."


@pytest.mark.parametrize("tools", [[], [_tool("lookup")]])
def test_an_agent_that_cannot_open_files_has_no_files_or_skills_sections(
    tools: list[ToolSpec],
) -> None:
    agents = load_tree(FIXTURES)
    bundle = build_bundle(agents["support"], tools=tools, agents=[], context=None)
    assert "FILES" not in _labels(bundle) and "SKILLS" not in _labels(bundle)
    assert "knowledge/" not in bundle and "skills/escalate.md" not in bundle


@pytest.mark.parametrize("name", ["Read", "Load"])
def test_read_or_load_alone_shows_the_sections(name: str) -> None:
    agents = load_tree(FIXTURES)
    bundle = build_bundle(agents["support"], tools=[_tool(name)], agents=[], context=None)
    assert _labels(bundle)[:3] == ["AGENT", "FILES", "SKILLS"]


@pytest.mark.parametrize("tools", ["none", "[lookup_order]"])
async def test_an_agent_without_read_or_load_still_gets_a_skill_body_in_the_input(
    tmp_path: Path, tools: str
) -> None:
    (tmp_path / "a" / "skills").mkdir(parents=True)
    (tmp_path / "a" / "tools").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text(
        f"---\nname: a\nkind: generation\nmodel: strong\ntools: {tools}\nskills: all\n---\nHi\n"
    )
    (tmp_path / "a" / "skills" / "x.md").write_text("---\ndescription: do x\n---\nBODY-OF-X\n")
    for suffix in ("json", "py"):
        shutil.copy(FIXTURES / "support/tools" / f"lookup_order.{suffix}", tmp_path / "a/tools")
    llm = Scripted(text_reply("a"))
    async with Engine(tmp_path, models={"strong": llm}) as engine:
        await engine.run("a", "acme", "Go.", skill="x")
    system, messages, _ = llm.calls[0]
    assert "<<<SKILLS>>>" not in system and "<<<FILES>>>" not in system
    assert "BODY-OF-X" in str(messages[-1])


async def test_the_system_prompt_of_an_agent_varies_only_by_context() -> None:
    llm = Scripted(text_reply("a"), text_reply("b"), text_reply("c"))
    async with Engine(FIXTURES, models={"strong": llm}) as engine:
        plain = await engine.run("support", "acme", "Charged twice.")
        skilled = await engine.run("support", "acme", "Hand me over.", skill="escalate")
        given = await engine.run("support", "acme", "Charged twice.", context="x")
    assert isinstance(plain, Generation) and isinstance(skilled, Generation)
    assert isinstance(given, Generation)
    assert plain.system == skilled.system
    assert given.system == f"{plain.system}\n\n<<<CONTEXT>>>\nx"
