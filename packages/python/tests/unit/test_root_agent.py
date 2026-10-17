import ast
from pathlib import Path

import pytest

import rfnry_engine
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Reply, Text, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tree import load_tree

from .fakes import Scripted, text_reply


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def orchestra(root: Path, front: str = "agents: [billing, triage]\n") -> Path:
    write(root / "AGENT.md", f"---\nkind: generation\ndescription: Leads\n{front}---\nYou lead.\n")
    write(root / "playbook.md", "---\ndescription: How we work\n---\nBe kind.\n")
    write(root / "policies/returns.md", "---\ndescription: Returns\n---\n30 days.\n")
    write(root / "examples/old/AGENT.md", "---\nkind: generation\n---\nA sample.\n")
    write(
        root / "billing/AGENT.md",
        "---\nkind: generation\ndescription: Bills\nreads: [policies]\n---\nYou bill.\n",
    )
    write(root / "billing/rates.md", "---\ndescription: Rates\n---\n$1.\n")
    write(root / "triage/AGENT.md", "---\nkind: decision\ndescription: Sorts\n---\nSort.\n")
    write(root / "triage/questions/urgent.md", "---\ntype: check\n---\nUrgent?\n")
    return root


def test_a_root_agent_has_its_sub_agents_beside_it(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk")
    tree = load_tree(root)
    assert list(tree) == ["desk", "billing", "triage"]
    lead = tree["desk"]
    assert lead.listing.splitlines() == [
        "examples/",
        "  old/",
        "    AGENT.md",
        "playbook.md — How we work",
        "policies/",
        "  returns.md — Returns",
    ]
    assert "../policies/\n  returns.md — Returns" in tree["billing"].listing
    assert tree["triage"].kind == "decision"


def test_a_root_agent_preloads_its_folder_without_its_sub_agents(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk", "agents: [billing]\npreload: true\n")
    sent = "".join(p.text for p in load_tree(root)["desk"].preload if isinstance(p, Text))
    assert "<<<playbook.md>>>" in sent and "<<<policies/returns.md>>>" in sent
    assert "rates.md" not in sent and "You bill." not in sent


async def test_a_root_agent_delegates_to_its_sub_agents(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk")
    llm = Scripted(
        Reply(
            tool_calls=[
                ToolCall(id="d1", name="Delegate", args={"agent": "billing", "input": "x"})
            ],
            stop_reason="tool_use",
        ),
        text_reply("billed"),
        text_reply("done"),
    )
    run = await Engine(root, generation=llm).run("desk", "acme", "Charge them.")
    assert isinstance(run, Generation) and run.reply == "done"
    assert [child.agent for child in run.children] == ["billing"]


def test_name_renames_the_root_agent_only(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk")
    assert list(Engine(root, generation=Scripted(), name="lead").agents) == [
        "lead",
        "billing",
        "triage",
    ]


def test_a_sub_agent_named_like_the_root_agent_is_refused(tmp_path: Path) -> None:
    root = orchestra(tmp_path / "desk")
    write(root / "desk/AGENT.md", "---\nkind: generation\n---\nMe again.\n")
    with pytest.raises(ConfigurationError, match="both named 'desk'; rename the folder"):
        load_tree(root)


TREE_MODULES = ("tree.py", "frontmatter.py", "folders.py", "tools/declared.py")


def test_every_tree_error_says_its_fix() -> None:
    source = Path(rfnry_engine.__file__).parent
    silent = []
    for module in TREE_MODULES:
        path = source / module
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", None) == "ConfigurationError"
            ):
                continue
            said = ast.unparse(node.exc.args[0])
            if "; " not in said and "{exc}" not in said:
                silent.append(f"{module}:{node.lineno}: {said}")
    assert silent == []
