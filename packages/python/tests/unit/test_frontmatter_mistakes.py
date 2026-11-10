from pathlib import Path

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.frontmatter import leniently_described
from rfnry_engine.manifest import read_manifest

from .trees import write_folder


def refusal(tmp_path: Path, text: str) -> str:
    (tmp_path / "rfnry.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError) as refused:
        read_manifest(tmp_path)
    return str(refused.value)


def test_yaml_merge_keys_pass_the_duplicate_check(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(
        "---\nbase: &base {a: 1}\nmore: {<<: *base, b: 2}\ndescription: Notes\n---\nBody\n"
    )
    assert leniently_described(path) == "Notes"


def test_a_described_file_after_a_byte_order_mark_is_in_the_map(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                "kind": "generation",
                "model": "strong",
                "instructions": "a.md",
                "files": {"include": ["notes.md"]},
            }
        },
        files={"a.md": "You are a.\n", "notes.md": "﻿---\ndescription: Notes\n---\nBody\n"},
    )
    assert "notes.md — Notes" in load_folder(folder)["a"].listing


@pytest.mark.parametrize(
    ("text", "line", "fix"),
    [
        (
            "kind: generation\ndescription: Refunds: what we owe\n",
            2,
            "quote a value that holds ': '",
        ),
        ("kind: generation\n\tagents: []\n", 2, "indent with spaces, not tabs"),
        ("kind: generation\ndescription: 'open\n", 2, "close the quote"),
        ("kind: generation\nkind: decision\n", 2, "remove one of them"),
    ],
)
def test_a_yaml_mistake_is_said_on_one_line_with_its_place_and_fix(
    tmp_path: Path, text: str, line: int, fix: str
) -> None:
    said = refusal(tmp_path, text)
    assert "\n" not in said and f"at line {line}" in said and fix in said


@pytest.mark.parametrize(
    ("text", "line", "fix"),
    [
        ("kind: generation\ndescription: 'open\nagents: []\n", 2, "close the quote"),
        ("kind: generation\ndescription\nagents: []\n", 2, "key: value"),
        ("kind: generation\nagents: [a, b\n", 2, "close the ["),
        ("kind: generation\ndescription: bad\x1bhere\n", 2, "remove the control character"),
        ("kind: generation\nagents:\n\t- a\n", 3, "indent with spaces, not tabs"),
    ],
)
def test_a_yaml_mistake_is_placed_where_it_starts(
    tmp_path: Path, text: str, line: int, fix: str
) -> None:
    said = refusal(tmp_path, text)
    assert "\n" not in said and f"at line {line}," in said and fix in said


def test_a_control_character_is_placed_past_wide_characters(tmp_path: Path) -> None:
    said = refusal(tmp_path, f"kind: generation\ndescription: {'ñ' * 40}\nx: abcdefg\x07\n")
    assert "at line 3, column 11:" in said


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

    from rfnry_engine.frontmatter import unreadable

    with pytest.raises(yaml.YAMLError) as raised:
        yaml.load(text, Loader=yaml.SafeLoader)
    assert fix in unreadable(raised.value, text)[2]
