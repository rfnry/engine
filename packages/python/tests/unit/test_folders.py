import logging
import os
from pathlib import Path, PurePosixPath

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.folders import pattern_reaches
from rfnry_engine.frontmatter import leniently_described

from .trees import write_folder


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_description_is_the_frontmatter_s_description_with_whitespace_collapsed(
    tmp_path: Path,
) -> None:
    write(
        tmp_path / "a.md",
        "---\nname: a\ndescription: |\n  Refunds,  returns\n  and credits\nother: 1\n---\nBody\n",
    )
    assert leniently_described(tmp_path / "a.md") == "Refunds, returns and credits"


@pytest.mark.parametrize(
    "text",
    [
        "No frontmatter\n",
        "---\nname: a\n---\nBody\n",
        "---\ndescription: '   '\n---\n",
        "---\n- a list\n- not a mapping\n---\n",
        "---\n---\nEmpty frontmatter\n",
    ],
)
def test_a_file_without_a_description_has_none(tmp_path: Path, text: str) -> None:
    write(tmp_path / "a.md", text)
    assert leniently_described(tmp_path / "a.md") is None


def test_crlf_and_a_missing_last_newline_still_describe(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"---\r\ndescription: Windows\r\n---\r\nBody\r\n")
    (tmp_path / "b.md").write_bytes(b"---\ndescription: Bare\n---")
    assert leniently_described(tmp_path / "a.md") == "Windows"
    assert leniently_described(tmp_path / "b.md") == "Bare"


def test_a_md_file_that_is_not_utf8_has_no_description(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"---\ndescription: caf\xe9\n---\n\x00\x01")
    assert leniently_described(tmp_path / "a.md") is None


@pytest.mark.parametrize(
    "text",
    [
        "---\ndescription: {a: 1}\n---\n",
        "---\ndescription: 12\n---\n",
        "---\ndescription: [a]\n---\n",
    ],
)
def test_a_description_that_is_not_text_is_no_description(tmp_path: Path, text: str) -> None:
    write(tmp_path / "a.md", text)
    assert leniently_described(tmp_path / "a.md") is None


def test_an_unreadable_folder_is_named_in_a_warning_and_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = write_folder(
        tmp_path,
        {
            "a": {
                "kind": "generation",
                "model": "strong",
                "instructions": "a.md",
                "files": {"include": ["."]},
            }
        },
        files={"a.md": "Work.\n", "notes/a.md": "A.\n", "lost/b.md": "B.\n"},
    )
    blocked = (root / "lost").resolve()
    real = os.listdir

    def listdir(path: object = ".") -> object:
        if Path(str(path)) == blocked:
            raise PermissionError(13, "denied", str(path))
        return real(path)  # type: ignore[call-overload]

    monkeypatch.setattr(os, "listdir", listdir)
    with caplog.at_level(logging.WARNING, logger="rfnry_engine.folders"):
        spec = load_folder(root)["a"]
    assert spec.listing == "notes/\n  a.md"
    assert f"folders: cannot read {blocked}: [Errno 13] denied: '{blocked}'; it is left out" in (
        caplog.messages
    )


@pytest.mark.parametrize(
    ("pattern", "path", "reached"),
    [
        ("q/*.json", "q/a.json", True),
        ("q/*.json", "r/a.json", False),
        ("q/*.json", "q/.a.json", False),
        ("q/.a.json", "q/.a.json", True),
        (".x/*.json", ".x/a.json", True),
        (".x/.x/*.json", ".x/.x/a.json", True),
        (".x/**/*.json", ".x/a.json", True),
        (".x/**/*.json", ".x/b/c/a.json", True),
        (".x/**/*.json", ".x/.x/secret.json", False),
        (".x/**/*.json", ".x/b/.x/secret.json", False),
        ("**/.x/*.json", "b/c/.x/a.json", True),
        ("**/.x/*.json", ".y/.x/a.json", False),
        ("**/*.json", ".x/a.json", False),
        ("a/**/**/b.json", "a/b.json", True),
        ("a/**", "a/b/c", True),
        ("**", ".x", False),
        (".*/a.json", ".x/a.json", False),
        ("?x/a.json", ".x/a.json", False),
        ("[.]x/a.json", ".x/a.json", False),
    ],
)
def test_a_pattern_reaches_a_hidden_name_only_where_it_spells_it(
    pattern: str, path: str, reached: bool
) -> None:
    assert pattern_reaches(pattern, PurePosixPath(path)) is reached
