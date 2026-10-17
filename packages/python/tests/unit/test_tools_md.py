from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.frontmatter import described
from rfnry_engine.tools.declared import declared_tools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tree import load_agent, load_tree

from .fakes import Scripted

LISTED = """---
lookup_order:
reworded: Find a customer's order
long: >-
  Find a customer's order
  by its number
---
"""

CONTRACT = """---
name: contract
description: Fetch an order by id
input:
  order_id: {type: string, description: The order id, required: true}
---
Returns the order as JSON.
"""

WORDED = """---
name: worded
description: Look an order up by its number
---
Say which order you looked up.
"""


class LookupArgs(BaseModel):
    order_id: str
    deep: bool = False


async def echo(args: dict[str, Any]) -> Any:
    return {"given": sorted(args)}


def code(*names: str) -> dict[str, Tool]:
    return {name: Tool(name, "Look up an order", body=echo, args=LookupArgs) for name in names}


def agent(root: Path, name: str = "a", front: str = "kind: generation\n") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "AGENT.md").write_text(f"---\nname: {name}\n{front}---\nYou are {name}.\n")
    return folder


def both(folder: Path) -> None:
    (folder / "TOOLS.md").write_text(LISTED)
    (folder / "tools").mkdir()
    (folder / "tools" / "contract.md").write_text(CONTRACT)
    (folder / "tools" / "worded.md").write_text(WORDED)


def test_tools_md_lists_names_and_tools_files_carry_contracts(tmp_path: Path) -> None:
    both(agent(tmp_path))
    tools = load_agent(tmp_path / "a").tools
    assert sorted(tools) == ["contract", "long", "lookup_order", "reworded", "worded"]
    assert (tools["lookup_order"].description, tools["lookup_order"].input) == (None, None)
    assert (tools["reworded"].description, tools["reworded"].input) == (
        "Find a customer's order",
        None,
    )
    assert tools["long"].description == "Find a customer's order by its number"
    assert tools["contract"].description == "Fetch an order by id\n\nReturns the order as JSON."
    assert tools["contract"].input is not None and list(tools["contract"].input) == ["order_id"]
    assert tools["worded"].input is None


async def test_each_part_given_replaces_the_code_s_alone(tmp_path: Path) -> None:
    both(agent(tmp_path))
    names = ("lookup_order", "reworded", "long", "contract", "worded")
    tools = Tools(declared_tools(load_agent(tmp_path / "a"), code(*names)))
    shown = {d["name"]: d for d in tools.definitions()}
    assert shown["lookup_order"]["description"] == "Look up an order"
    assert shown["reworded"]["description"] == "Find a customer's order"
    assert shown["worded"]["description"].endswith("Say which order you looked up.")
    for kept in ("lookup_order", "reworded", "worded"):
        assert set(shown[kept]["input_schema"]["properties"]) == {"order_id", "deep"}
    assert set(shown["contract"]["input_schema"]["properties"]) == {"order_id"}
    given = await tools.dispatch("worded", {"order_id": "A-1", "deep": True})
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
    ("where", "text", "said"),
    [
        (
            "TOOLS.md",
            "---\nlookup_order:\n---\nUse it often.\n",
            "lists its tools in its frontmatter",
        ),
        ("TOOLS.md", "---\nlookup_order: {description: Find}\n---\n", "invalid tools"),
        ("TOOLS.md", "---\nlookup_order: ''\n---\n", "blank description"),
        ("TOOLS.md", "---\nlookup_order: '  '\n---\n", "blank description"),
        (
            "TOOLS.md",
            "---\nlookup_order:\nlookup_order: Find\n---\n",
            "duplicate key 'lookup_order'",
        ),
        ("TOOLS.md", "lookup_order:\n", "frontmatter: missing"),
        ("tools/lookup_order.md", "---\nname: lookup_order\n---\n", "listed in TOOLS.md"),
        (
            "tools/lookup_order.md",
            "---\nname: lookup_order\ndescription: ' '\n---\n",
            "no description",
        ),
        (
            "tools/lookup_order.md",
            "---\nname: other\ndescription: Find\n---\n",
            "must live in other.md",
        ),
    ],
)
def test_a_bad_declaration_is_refused_by_name(
    tmp_path: Path, where: str, text: str, said: str
) -> None:
    path = agent(tmp_path) / where
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    with pytest.raises(ConfigurationError, match=said) as refused:
        load_agent(tmp_path / "a")
    assert where.split("/")[-1] in str(refused.value)


def test_a_tool_declared_in_both_places_is_refused(tmp_path: Path) -> None:
    folder = agent(tmp_path)
    (folder / "TOOLS.md").write_text("---\ncontract:\n---\n")
    (folder / "tools").mkdir()
    (folder / "tools" / "contract.md").write_text(CONTRACT)
    with pytest.raises(ConfigurationError, match="declares 'contract' in both TOOLS.md and tools/"):
        load_agent(folder)


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("---\nlookup_order:\n---", ["lookup_order"]),
        ("---\nlookup_order:\n---   \n", ["lookup_order"]),
        ("---\n---\n", []),
        ("﻿---\nlookup_order:\n---\n", ["lookup_order"]),
    ],
)
def test_a_frontmatter_only_file_reads_however_it_ends(
    tmp_path: Path, text: str, names: list[str]
) -> None:
    (agent(tmp_path) / "TOOLS.md").write_text(text, encoding="utf-8")
    assert list(load_agent(tmp_path / "a").tools) == names


def test_a_decision_agent_declares_no_tools(tmp_path: Path) -> None:
    folder = agent(tmp_path, front="kind: decision\n")
    (folder / "questions").mkdir()
    (folder / "questions" / "q.md").write_text("---\ntype: check\n---\nIs it?\n")
    (folder / "TOOLS.md").write_text("---\nlookup_order:\n---\n")
    with pytest.raises(ConfigurationError, match="decision agent 'a' cannot declare tools"):
        load_agent(folder)


def test_declarations_are_out_of_the_map_and_the_preload_but_in_the_fingerprint(
    tmp_path: Path,
) -> None:
    folder = agent(tmp_path, front="kind: generation\npreload: true\n")
    both(folder)
    (folder / "notes.md").write_text("---\ndescription: Notes\n---\nRead me.\n")
    spec = load_agent(folder)
    assert "TOOLS.md" not in spec.listing and "tools/" not in spec.listing
    sent = "".join(getattr(part, "text", "") for part in spec.preload)
    assert "<<<notes.md>>>" in sent and "TOOLS.md" not in sent and "contract.md" not in sent
    before = spec.fingerprint
    (folder / "TOOLS.md").write_text("---\nlookup_order: Find it\n---\n")
    after = load_agent(folder).fingerprint
    (folder / "tools" / "worded.md").write_text(WORDED.replace("Say", "Tell"))
    assert len({before, after, load_agent(folder).fingerprint}) == 3


def test_a_reader_does_not_see_another_agent_s_declarations(tmp_path: Path) -> None:
    agent(tmp_path, "reader", "kind: generation\nreads: [owner]\n")
    owner = agent(tmp_path, "owner")
    both(owner)
    (owner / "notes.md").write_text("---\ndescription: Notes\n---\n")
    tree = load_tree(tmp_path)
    assert "../owner/\n  notes.md — Notes" in tree["reader"].listing
    assert "TOOLS.md" not in tree["reader"].listing and "contract.md" not in tree["reader"].listing


def test_a_tree_with_bodies_nobody_declares_is_told_where_to_declare(tmp_path: Path) -> None:
    agent(tmp_path)
    with pytest.raises(ConfigurationError, match="TOOLS.md or tools/"):
        Engine(tmp_path, generation=Scripted(), tools=[Tool("lookup_order", "Look", body=echo)])


@pytest.mark.parametrize(
    ("front", "name"),
    [
        ("kind: generation\n", "Read"),
        ("kind: generation\n", "History"),
        ("kind: generation\nagents: [a]\n", "Delegate"),
        ("kind: generation\nworkspace: true\n", "Shell"),
    ],
)
def test_a_tool_named_like_a_builtin_is_refused_at_build(
    tmp_path: Path, front: str, name: str
) -> None:
    (agent(tmp_path, front=front) / "TOOLS.md").write_text(f"---\n{name}:\n---\n")
    with pytest.raises(ConfigurationError, match=f"'{name}'"):
        Engine(tmp_path, generation=Scripted(), tools=[Tool(name, "Mine", body=echo)])


def test_yaml_merge_keys_pass_the_duplicate_check(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(
        "---\nbase: &base {a: 1}\nmore: {<<: *base, b: 2}\ndescription: Notes\n---\nBody\n"
    )
    assert described(path) == "Notes"


def test_a_described_file_after_a_byte_order_mark_is_in_the_map(tmp_path: Path) -> None:
    folder = agent(tmp_path)
    (folder / "notes.md").write_text("﻿---\ndescription: Notes\n---\nBody\n", encoding="utf-8")
    assert "notes.md — Notes" in load_agent(folder).listing
