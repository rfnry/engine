import re
from pathlib import Path

from rfnry_engine.bundle import build_bundle
from rfnry_engine.models import ToolSpec
from rfnry_engine.tree import load_tree

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


def _labels(bundle: str) -> list[str]:
    return re.findall(r"^<<<([A-Z]+)>>>$", bundle, flags=re.MULTILINE)


def test_sections_in_order_with_missing_ones_skipped() -> None:
    agents = load_tree(FIXTURES)
    tools = [ToolSpec(name="Read", description="Read a file", input_schema={})]
    bundle = build_bundle(
        agents["support"],
        task="triage",
        tools=tools,
        agents=[agents["triage"]],
        context="Plan: Team",
    )
    assert _labels(bundle) == ["AGENT", "TASK", "FILES", "TOOLS", "AGENTS", "CONTEXT"]
    assert "Decide whether this message needs a person." in bundle
    assert (
        "<<<FILES>>>\nknowledge/\n  plans.md — The three plans\nrules/\n"
        "  refunds.md — When a refund is owed\nskills/\n"
        "  escalate.md — Hand a conversation to a person\n\n<<<TOOLS>>>"
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
    bundle = build_bundle(load_tree(tmp_path)["a"], task=None, tools=[], agents=[], context=None)
    assert _labels(bundle) == ["AGENT", "FILES"]
    assert "Read rules/x.md first." not in bundle
    assert bundle.endswith("<<<FILES>>>\nINDEX.md\nrules/\n  _index.md")


def test_bare_agent_has_only_agent_and_tools() -> None:
    agents = load_tree(FIXTURES)
    bundle = build_bundle(agents["billing"], task=None, tools=[], agents=[], context=None)
    assert _labels(bundle) == ["AGENT"]
    assert bundle == "<<<AGENT>>>\nYou are billing."
