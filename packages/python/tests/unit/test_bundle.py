import re
from pathlib import Path

import pytest

from rfnry_engine.bundle import build_bundle
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import ToolSpec
from rfnry_engine.tree import load_tree

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


def _labels(bundle: str) -> list[str]:
    return re.findall(r"^<<<([A-Z]+)>>>$", bundle, flags=re.MULTILINE)


def test_sections_in_order_with_missing_ones_skipped() -> None:
    agents = load_tree(FIXTURES)
    tools = [ToolSpec(name="Read", description="Read a file", input_schema={})]
    bundle = build_bundle(
        agents["support"], task="triage", tools=tools, agents=[agents["triage"]], context="Plan: Team"
    )
    assert _labels(bundle) == [
        "AGENT", "INDEX", "TASK", "RULES", "SKILLS", "TOOLS", "KNOWLEDGE", "AGENTS", "CONTEXT",
    ]
    assert "Decide whether this message needs a person." in bundle
    assert "- Read: Read a file" in bundle
    assert "- triage: Judges urgency and severity of a message" in bundle
    assert bundle.endswith("<<<CONTEXT>>>\nPlan: Team")


def test_bare_agent_has_only_agent_and_tools() -> None:
    agents = load_tree(FIXTURES)
    bundle = build_bundle(agents["billing"], task=None, tools=[], agents=[], context=None)
    assert _labels(bundle) == ["AGENT"]
    assert bundle == "<<<AGENT>>>\nYou are billing."


def test_unknown_task_is_refused() -> None:
    agents = load_tree(FIXTURES)
    with pytest.raises(ConfigurationError, match="ghost"):
        build_bundle(agents["support"], task="ghost", tools=[], agents=[], context=None)
