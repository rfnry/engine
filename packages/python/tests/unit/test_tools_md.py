from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.declared import declared_tools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tree import load_agent, load_tree

TOOLS = """---
lookup_order:
reworded:
  description: Find a customer's order
contract:
  description: |-
    Fetch an order by id

    Returns the order as JSON.
  input:
    order_id: {type: string, description: The order id, required: true}
---
"""


class LookupArgs(BaseModel):
    order_id: str
    deep: bool = False


async def echo(args: dict[str, Any]) -> Any:
    return {"given": sorted(args)}


def code() -> dict[str, Tool]:
    return {
        name: Tool(name, "Look up an order", body=echo, args=LookupArgs)
        for name in ("lookup_order", "reworded", "contract")
    }


def agent(root: Path, name: str = "a", front: str = "kind: generation\n") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "AGENT.md").write_text(f"---\nname: {name}\n{front}---\nYou are {name}.\n")
    return folder


def test_tools_md_declares_each_tool_by_name(tmp_path: Path) -> None:
    (agent(tmp_path) / "TOOLS.md").write_text(TOOLS)
    tools = load_agent(tmp_path / "a").tools
    assert list(tools) == ["lookup_order", "reworded", "contract"]
    assert (tools["lookup_order"].description, tools["lookup_order"].input) == (None, None)
    assert (tools["reworded"].description, tools["reworded"].input) == (
        "Find a customer's order",
        None,
    )
    assert tools["contract"].description == "Fetch an order by id\n\nReturns the order as JSON."
    assert tools["contract"].input is not None and list(tools["contract"].input) == ["order_id"]


async def test_each_part_of_an_entry_replaces_the_code_s_alone(tmp_path: Path) -> None:
    (agent(tmp_path) / "TOOLS.md").write_text(TOOLS)
    tools = Tools(declared_tools(load_agent(tmp_path / "a"), code()))
    shown = {d["name"]: d for d in tools.definitions()}
    assert shown["lookup_order"]["description"] == "Look up an order"
    assert set(shown["lookup_order"]["input_schema"]["properties"]) == {"order_id", "deep"}
    assert shown["reworded"]["description"] == "Find a customer's order"
    assert set(shown["reworded"]["input_schema"]["properties"]) == {"order_id", "deep"}
    assert shown["contract"]["description"].startswith("Fetch an order by id")
    assert set(shown["contract"]["input_schema"]["properties"]) == {"order_id"}
    given = await tools.dispatch("reworded", {"order_id": "A-1", "deep": True})
    assert given == {"given": ["deep", "order_id"]}
    with pytest.raises(InputError):
        await tools.dispatch("reworded", {})
    with pytest.raises(InputError, match="extra"):
        await tools.dispatch("contract", {"order_id": "A-1", "deep": True})


def test_a_tool_with_no_description_anywhere_is_refused(tmp_path: Path) -> None:
    (agent(tmp_path) / "TOOLS.md").write_text("---\nlookup_order:\n---\n")
    spec = load_agent(tmp_path / "a")
    with pytest.raises(ConfigurationError, match="no description"):
        declared_tools(spec, {"lookup_order": Tool("lookup_order", body=echo)})


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("---\nlookup_order:\n---\nUse it often.\n", "holds its tools in its frontmatter"),
        ("---\nlookup_order: {descripton: Typo}\n---\n", "unknown key 'lookup_order.descripton'"),
        ("---\nlookup_order: Find it\n---\n", "invalid tools"),
        ("lookup_order:\n", "frontmatter: missing"),
    ],
)
def test_a_bad_tools_md_is_refused_by_name(tmp_path: Path, text: str, said: str) -> None:
    (agent(tmp_path) / "TOOLS.md").write_text(text)
    with pytest.raises(ConfigurationError, match=said) as refused:
        load_agent(tmp_path / "a")
    assert "TOOLS.md" in str(refused.value)


def test_a_decision_agent_declares_no_tools(tmp_path: Path) -> None:
    folder = agent(tmp_path, front="kind: decision\n")
    (folder / "questions").mkdir()
    (folder / "questions" / "q.md").write_text("---\ntype: check\n---\nIs it?\n")
    (folder / "TOOLS.md").write_text("---\nlookup_order:\n---\n")
    with pytest.raises(ConfigurationError, match="decision agent 'a' cannot declare TOOLS.md"):
        load_agent(folder)


def test_tools_md_is_out_of_the_map_and_the_preload_but_in_the_fingerprint(
    tmp_path: Path,
) -> None:
    folder = agent(tmp_path, front="kind: generation\npreload: true\n")
    (folder / "TOOLS.md").write_text("---\nlookup_order:\n---\n")
    (folder / "tools").mkdir()
    (folder / "tools" / "hammer.md").write_text("---\ndescription: How to hammer\n---\nHit it.\n")
    spec = load_agent(folder)
    assert "TOOLS.md" not in spec.listing
    assert "tools/\n  hammer.md — How to hammer" in spec.listing
    sent = "".join(getattr(part, "text", "") for part in spec.preload)
    assert "<<<tools/hammer.md>>>" in sent and "TOOLS.md" not in sent
    before = spec.fingerprint
    (folder / "TOOLS.md").write_text("---\nlookup_order:\n  description: Find it\n---\n")
    assert load_agent(folder).fingerprint != before


def test_a_reader_does_not_see_another_agent_s_tools_md(tmp_path: Path) -> None:
    agent(tmp_path, "reader", "kind: generation\nreads: [owner]\n")
    owner = agent(tmp_path, "owner")
    (owner / "TOOLS.md").write_text("---\nlookup_order:\n---\n")
    (owner / "notes.md").write_text("---\ndescription: Notes\n---\n")
    tree = load_tree(tmp_path)
    assert "../owner/\n  notes.md — Notes" in tree["reader"].listing
    assert "TOOLS.md" not in tree["reader"].listing
