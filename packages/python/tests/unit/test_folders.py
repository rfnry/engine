from pathlib import Path

import pytest

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.folders import content, entries, joined
from rfnry_engine.frontmatter import described


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def coding(root: Path) -> Path:
    write(
        root / "AGENT.md",
        "---\nname: coding\nkind: generation\nmodel: strong\ndescription: Codes\n---\nHi\n",
    )
    write(root / "tasks/review.md", "---\ndescription: Reviews\n---\nReview it.\n")
    write(root / "questions/q.json", '{"type": "check", "instructions": "Is it?"}')
    write(root / "notes/style.md", "---\ndescription: The house style\n---\nTabs never.\n")
    write(root / "notes/draft.md", "No frontmatter, so no description.\n")
    write(root / "notes/.secret.md", "---\ndescription: Hidden\n---\n")
    write(root / ".git/config", "[core]\n")
    write(
        root / "skills/pdf/SKILL.md",
        "---\nname: pdf\ndescription: >\n  Fill and\n  read PDFs\n---\n",
    )
    write(root / "skills/pdf/reference.md", "---\ndescription: Field names\n---\n")
    write(root / "README.md", "---\ndescription: Start here\n---\n")
    write(root / "docs/tools/hammer.md", "---\ndescription: A deep tools folder is content\n---\n")
    (root / "empty").mkdir()
    return root


def test_a_description_is_the_frontmatter_s_description_with_whitespace_collapsed(
    tmp_path: Path,
) -> None:
    write(
        tmp_path / "a.md",
        "---\nname: a\ndescription: |\n  Refunds,  returns\n  and credits\nother: 1\n---\nBody\n",
    )
    assert described(tmp_path / "a.md") == "Refunds, returns and credits"


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
    assert described(tmp_path / "a.md") is None


def test_crlf_and_a_missing_last_newline_still_describe(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"---\r\ndescription: Windows\r\n---\r\nBody\r\n")
    (tmp_path / "b.md").write_bytes(b"---\ndescription: Bare\n---")
    assert described(tmp_path / "a.md") == "Windows"
    assert described(tmp_path / "b.md") == "Bare"


def test_a_md_file_that_is_not_utf8_has_no_description(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"---\ndescription: caf\xe9\n---\n\x00\x01")
    assert described(tmp_path / "a.md") is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("---\ndescription: [unclosed\n---\n", "invalid YAML"),
        ("---\ndescription: {a: 1}\n---\n", "description must be a string"),
        ("---\ndescription: 12\n---\n", "description must be a string"),
    ],
)
def test_bad_frontmatter_is_refused_naming_the_file(
    tmp_path: Path, text: str, message: str
) -> None:
    write(tmp_path / "bad.md", text)
    with pytest.raises(ConfigurationError, match=rf"^frontmatter: {message} in .*bad\.md"):
        described(tmp_path / "bad.md")


def test_the_map_shows_every_folder_and_file_in_name_order_with_its_description(
    tmp_path: Path,
) -> None:
    root = coding(tmp_path / "coding")
    assert "\n".join(entries(root, True)) == (
        "README.md — Start here\n"
        "docs/\n"
        "  tools/\n"
        "    hammer.md — A deep tools folder is content\n"
        "empty/\n"
        "notes/\n"
        "  draft.md\n"
        "  style.md — The house style\n"
        "skills/\n"
        "  pdf/\n"
        "    SKILL.md — Fill and read PDFs\n"
        "    reference.md — Field names"
    )


def test_a_folder_without_agent_md_shows_its_declaration_folders_as_content(
    tmp_path: Path,
) -> None:
    root = coding(tmp_path / "coding")
    shown = entries(root, False)
    assert "tasks/" in shown and "  review.md — Reviews" in shown
    assert "questions/" in shown


def test_read_folders_follow_under_their_path_without_an_agent_s_declarations(
    tmp_path: Path,
) -> None:
    own = tmp_path / "coding"
    write(own / "AGENT.md", "---\nname: own\nkind: generation\nmodel: strong\n---\n")
    write(own / "a.md", "---\ndescription: A\n---\n")
    books = tmp_path / "books"
    write(books / "python/async.md", "---\ndescription: Async\n---\n")
    writing = tmp_path / "writing"
    write(
        writing / "AGENT.md",
        "---\nname: writing\nkind: generation\nmodel: strong\ndescription: Writes\n---\n",
    )
    write(writing / "tasks/x.md", "---\ndescription: A task\n---\n")
    write(writing / "voice.md", "---\ndescription: Voice\n---\n")
    shown = {"../books": entries(books, False), "../writing": entries(writing, True)}
    assert joined(entries(own, True), shown) == (
        "a.md — A\n../books/\n  python/\n    async.md — Async\n../writing/\n  voice.md — Voice"
    )


def test_content_leaves_out_what_the_prompt_already_holds(tmp_path: Path) -> None:
    root = coding(tmp_path / "coding")
    assert [rel for rel, _ in content(root, True)] == [
        "README.md",
        "docs/tools/hammer.md",
        "notes/draft.md",
        "notes/style.md",
        "skills/pdf/SKILL.md",
        "skills/pdf/reference.md",
    ]
    plain = [rel for rel, _ in content(root, False)]
    assert "AGENT.md" in plain
    assert "tasks/review.md" in plain


def test_a_file_named_like_a_declaration_folder_is_content(tmp_path: Path) -> None:
    write(tmp_path / "AGENT.md", "---\nname: root\nkind: generation\nmodel: strong\n---\n")
    write(tmp_path / "tasks", "a file, not a folder\n")
    assert [rel for rel, _ in content(tmp_path, True)] == ["tasks"]


def test_a_link_out_of_the_folder_is_neither_mapped_nor_content(tmp_path: Path) -> None:
    root = tmp_path / "agent"
    outside = tmp_path / "outside"
    write(outside / "secret.md", "---\ndescription: Secret\n---\n")
    write(root / "AGENT.md", "---\nname: root\nkind: generation\nmodel: strong\n---\n")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (root / "secret.md").symlink_to(outside / "secret.md")
    assert entries(root, True) == [] and content(root, True) == []


def test_a_bad_description_stops_the_map_naming_the_file(tmp_path: Path) -> None:
    root = coding(tmp_path / "coding")
    write(root / "notes/broken.md", "---\ndescription: [x\n---\n")
    with pytest.raises(ConfigurationError, match=r"^frontmatter: invalid YAML in .*broken\.md"):
        entries(root, True)


def test_a_folder_holding_an_agent_md_below_the_top_is_left_out_unless_inner(
    tmp_path: Path,
) -> None:
    root = coding(tmp_path / "coding")
    write(root / "notes/inner/AGENT.md", "---\nname: inner\nkind: generation\nmodel: strong\n---\n")
    write(root / "notes/inner/voice.md", "Warm.\n")
    assert "notes/inner/voice.md" not in [rel for rel, _ in content(root, False)]
    assert "notes/\n  draft.md\n  style.md — The house style\n" in "\n".join(entries(root, True))
    inner = [rel for rel, _ in content(root, False, inner=True)]
    assert "notes/inner/AGENT.md" in inner and "notes/inner/voice.md" in inner
