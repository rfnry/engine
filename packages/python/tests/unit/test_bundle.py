import re
import shutil
from pathlib import Path

import pytest

from rfnry_engine import Engine
from rfnry_engine.bundle import build_bundle
from rfnry_engine.compiled import load_folder
from rfnry_engine.models import ToolSpec
from rfnry_engine.records import Generation

from .fakes import Scripted, text_reply
from .trees import provided, solo, write_folder

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


def _labels(bundle: str) -> list[str]:
    return re.findall(r"^<<<([A-Z]+)>>>$", bundle, flags=re.MULTILINE)


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=f"{name} it", input_schema={})


def test_sections_in_order_with_missing_ones_skipped() -> None:
    agents = load_folder(FIXTURES)
    tools = [ToolSpec(name="Read", description="Read a file", input_schema={})]
    bundle = build_bundle(
        agents["support"],
        tools=tools,
        agents=[agents["triage"]],
        context="Plan: Team",
    )
    assert _labels(bundle) == ["AGENT", "FILES", "SKILLS", "TOOLS", "AGENTS", "CONTEXT"]
    assert (
        "<<<FILES>>>\nsupport/\n  knowledge/\n    plans.md — The three plans\n  rules/\n"
        "    refunds.md — When a refund is owed\n\n<<<SKILLS>>>\n"
        "support/skills/escalate/SKILL.md — Hand a conversation to a person\n\n<<<TOOLS>>>"
    ) in bundle
    assert "- Read: Read a file" in bundle
    assert "- triage: Judges urgency and severity of a message" in bundle
    assert bundle.endswith("<<<CONTEXT>>>\nPlan: Team")


def test_an_old_index_file_is_an_ordinary_file(tmp_path: Path) -> None:
    solo(
        tmp_path,
        files={"include": ["INDEX.md", "rules"]},
        written={
            "INDEX.md": "Read rules/x.md first.\n",
            "rules/_index.md": "- x.md — the x rule\n",
        },
    )
    bundle = build_bundle(
        load_folder(tmp_path)["a"], tools=[_tool("Read")], agents=[], context=None
    )
    assert _labels(bundle) == ["AGENT", "FILES", "TOOLS"]
    assert "Read rules/x.md first." not in bundle
    assert "<<<FILES>>>\nINDEX.md\nrules/\n  _index.md\n\n<<<TOOLS>>>" in bundle


def test_bare_agent_has_only_agent_and_tools() -> None:
    agents = load_folder(FIXTURES)
    bundle = build_bundle(agents["billing"], tools=[], agents=[], context=None)
    assert _labels(bundle) == ["AGENT"]
    assert bundle == "<<<AGENT>>>\nYou are billing."


@pytest.mark.parametrize("tools", [[], [_tool("lookup")]])
def test_an_agent_that_cannot_open_files_has_no_files_or_skills_sections(
    tools: list[ToolSpec],
) -> None:
    agents = load_folder(FIXTURES)
    bundle = build_bundle(agents["support"], tools=tools, agents=[], context=None)
    assert "FILES" not in _labels(bundle) and "SKILLS" not in _labels(bundle)
    assert "knowledge/" not in bundle and "skills/escalate" not in bundle


@pytest.mark.parametrize("name", ["Read", "Load"])
def test_read_or_load_alone_shows_the_sections(name: str) -> None:
    agents = load_folder(FIXTURES)
    bundle = build_bundle(agents["support"], tools=[_tool(name)], agents=[], context=None)
    assert _labels(bundle)[:3] == ["AGENT", "FILES", "SKILLS"]


@pytest.mark.parametrize("tools", [[], ["lookup_order"]])
async def test_an_agent_without_read_or_load_still_gets_a_skill_body_in_the_input(
    tmp_path: Path, tools: list[str]
) -> None:
    for suffix in ("json", "py"):
        (tmp_path / "t").mkdir(exist_ok=True)
        shutil.copy(FIXTURES / "support/tools" / f"lookup_order.{suffix}", tmp_path / "t")
    write_folder(
        tmp_path,
        {
            "a": {
                "kind": "generation",
                "model": "strong",
                "instructions": "a.md",
                "skills": {"preload": ["skills/x"]},
                "tools": {
                    "exclude": ["Read", "Load", "Grep", "Glob", "LS"],
                    **({"include": tools} if tools else {}),
                },
            }
        },
        tools={"lookup_order": {"module": "t/lookup_order.py", "schema": "t/lookup_order.json"}}
        if tools
        else None,
        files={"a.md": "Hi\n", "skills/x/SKILL.md": "---\ndescription: do x\n---\nBODY-OF-X\n"},
    )
    llm = Scripted(text_reply("a"))
    async with Engine(tmp_path, providers=provided({"strong": llm})) as engine:
        await engine.run("a", "acme", "Go.", skill="x")
    system, messages, _ = llm.calls[0]
    assert "<<<SKILLS>>>" not in system and "<<<FILES>>>" not in system
    assert "BODY-OF-X" in str(messages[-1])


async def test_the_system_prompt_of_an_agent_varies_only_by_context() -> None:
    llm = Scripted(text_reply("a"), text_reply("b"), text_reply("c"))
    async with Engine(FIXTURES, providers=provided({"strong": llm, "judge": None})) as engine:
        plain = await engine.run("support", "acme", "Charged twice.")
        skilled = await engine.run("support", "acme", "Hand me over.", skill="escalate")
        given = await engine.run("support", "acme", "Charged twice.", context="x")
    assert isinstance(plain, Generation) and isinstance(skilled, Generation)
    assert isinstance(given, Generation)
    assert plain.system == skilled.system
    assert given.system == f"{plain.system}\n\n<<<CONTEXT>>>\nx"
