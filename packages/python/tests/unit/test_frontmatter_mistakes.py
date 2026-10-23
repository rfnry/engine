from pathlib import Path

import pytest

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.frontmatter import described
from rfnry_engine.tree import load_agent


def agent(root: Path, name: str = "a", front: str = "kind: generation\nmodel: strong\n") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "AGENT.md").write_text(f"---\nname: {name}\n{front}---\nYou are {name}.\n")
    return folder


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


@pytest.mark.parametrize(
    ("front", "line", "fix"),
    [
        (
            "kind: generation\ndescription: Refunds: what we owe\n",
            3,
            "quote a value that holds ': '",
        ),
        (
            "kind: generation\n\tagents: []\n",
            3,
            "indent with spaces, not tabs",
        ),
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
        (
            "kind: generation\ndescription: 'open\nagents: []\n",
            3,
            "close the quote",
        ),
        ("kind: generation\ndescription\nagents: []\n", 3, "key: value"),
        ("kind: generation\nagents: [a, b\n", 3, "close the ["),
        (
            "kind: generation\ndescription: bad\x1bhere\n",
            3,
            "remove the control character",
        ),
        (
            "kind: generation\nagents:\n\t- a\n",
            4,
            "indent with spaces, not tabs",
        ),
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
        (
            "kind: generation\nmodel: strong\n\tagents: []\n",
            "indent with spaces, not tabs",
        ),
        (
            "kind: generation\nmodel: strong\nagents:\n\t- a\n",
            "indent with spaces, not tabs",
        ),
        ("x: abc\x07\n", "remove the control character"),
        ("kind: generation\nmodel: strong\n  agents: []\nx: 1\n", "line the key up"),
    ],
)
def test_pure_pyyaml_mistakes_get_the_same_fixes(text: str, fix: str) -> None:
    import yaml

    from rfnry_engine.frontmatter import _unreadable

    with pytest.raises(yaml.YAMLError) as raised:
        yaml.load(text, Loader=yaml.SafeLoader)
    assert fix in _unreadable(raised.value, text)[2]
