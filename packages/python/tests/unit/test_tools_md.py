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
description: Fetch an order by id
input:
  order_id: {type: string, description: The order id, required: true}
---
Returns the order as JSON.
"""

WORDED = """---
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
    (folder / "AGENT.md").write_text(f"---\n{front}---\nYou are {name}.\n")
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
        ("tools/lookup_order.md", "---\ninput: {q: {type: string}}\n---\n", "listed in TOOLS.md"),
        (
            "tools/lookup_order.md",
            "---\ndescription: ' '\n---\n",
            "no description",
        ),
        (
            "tools/lookup_order.md",
            "---\nname: other\ndescription: Find\n---\n",
            "remove the name: line",
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


async def test_an_input_may_use_any_property_name(tmp_path: Path) -> None:
    folder = agent(tmp_path)
    (folder / "tools").mkdir()
    (folder / "tools" / "odd.md").write_text(
        "---\ndescription: Odd names\ninput:\n"
        "  _x: {type: string, required: true}\n"
        "  model_config: {type: string}\n"
        "  schema: {type: integer}\n"
        "---\n"
    )
    tools = Tools(declared_tools(load_agent(folder), {"odd": Tool("odd", "Odd", body=echo)}))
    [shown] = tools.definitions()
    assert list(shown["input_schema"]["properties"]) == ["_x", "model_config", "schema"]
    assert shown["input_schema"]["required"] == ["_x"]
    given = await tools.dispatch("odd", {"_x": "a", "model_config": "b", "schema": 3})
    assert given == {"given": ["_x", "model_config", "schema"]}
    with pytest.raises(InputError, match="extra"):
        await tools.dispatch("odd", {"_x": "a", "other": 1})
    with pytest.raises(InputError):
        await tools.dispatch("odd", {"model_config": "b"})


@pytest.mark.parametrize(
    ("front", "line", "fix"),
    [
        (
            "kind: generation\ndescription: Refunds: what we owe\n",
            3,
            "quote a value that holds ': '",
        ),
        ("kind: generation\n\tagents: []\n", 3, "indent with spaces, not tabs"),
        ("kind: generation\ndescription: 'open\n", 3, "close the quote"),
        ("kind: generation\nkind: decision\n", 3, "remove one of them"),
    ],
)
def test_a_yaml_mistake_is_said_on_one_line_with_its_place_and_fix(
    tmp_path: Path, front: str, line: int, fix: str
) -> None:
    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "AGENT.md").write_text(f"---\n{front}---\nHi.\n")
    with pytest.raises(ConfigurationError) as refused:
        load_agent(folder)
    said = str(refused.value)
    assert "\n" not in said and f"at line {line}" in said and fix in said


def test_a_yaml_mistake_in_a_map_file_is_said_the_same_way(tmp_path: Path) -> None:
    folder = agent(tmp_path)
    (folder / "notes.md").write_text("---\ndescription: Notes: ours\n---\nBody\n")
    with pytest.raises(ConfigurationError) as refused:
        load_agent(folder)
    said = str(refused.value)
    assert "\n" not in said and "notes.md at line 2" in said and "quote" in said


@pytest.mark.parametrize(
    ("front", "line", "fix"),
    [
        ("kind: generation\ndescription: 'open\nagents: []\n", 3, "close the quote"),
        ("kind: generation\ndescription\nagents: []\n", 3, "key: value"),
        ("kind: generation\nagents: [a, b\n", 3, "close the ["),
        ("kind: generation\ndescription: bad\x1bhere\n", 3, "remove the control character"),
        ("kind: generation\nagents:\n\t- a\n", 4, "indent with spaces, not tabs"),
    ],
)
def test_a_yaml_mistake_is_placed_where_it_starts(
    tmp_path: Path, front: str, line: int, fix: str
) -> None:
    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "AGENT.md").write_text(f"---\n{front}---\nHi.\n")
    with pytest.raises(ConfigurationError) as refused:
        load_agent(folder)
    said = str(refused.value)
    assert "\n" not in said and f"at line {line}," in said and fix in said


def test_a_control_character_is_placed_past_wide_characters(tmp_path: Path) -> None:
    folder = tmp_path / "a"
    folder.mkdir()
    front = f"kind: generation\ndescription: {'ñ' * 40}\nx: abcdefg\x07\n"
    (folder / "AGENT.md").write_text(f"---\n{front}---\nHi.\n")
    with pytest.raises(ConfigurationError) as refused:
        load_agent(folder)
    assert "at line 4, column 11:" in str(refused.value)


@pytest.mark.parametrize(
    ("text", "fix"),
    [
        ("kind: generation\n\tagents: []\n", "indent with spaces, not tabs"),
        ("kind: generation\nagents:\n\t- a\n", "indent with spaces, not tabs"),
        ("x: abc\x07\n", "remove the control character"),
        ("kind: generation\n  agents: []\nx: 1\n", "line the key up"),
    ],
)
def test_pure_pyyaml_mistakes_get_the_same_fixes(text: str, fix: str) -> None:
    import yaml

    from rfnry_engine.frontmatter import _unreadable

    with pytest.raises(yaml.YAMLError) as raised:
        yaml.load(text, Loader=yaml.SafeLoader)
    assert fix in _unreadable(raised.value, text)[2]
