from pathlib import Path

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tree import _tree_folders, from_agent, load_tree


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


def books(root: Path, reads: str = "[shared/books]") -> Path:
    write(root / "shared/books/python/async.md", "---\ndescription: Async\n---\nAwait.\n")
    write(root / "coding/AGENT.md", agent_md("coding", extra=f"reads: {reads}\n"))
    return root


def test_reads_names_a_folder_by_its_path_from_the_root(tmp_path: Path) -> None:
    spec = load_tree(books(tmp_path))["coding"]
    assert spec.reads == {"shared/books": (tmp_path / "shared/books").resolve()}
    assert "../shared/books/\n  python/\n    async.md — Async" in spec.listing
    assert Engine(tmp_path).agents["coding"].reads == ["shared/books"]


def test_the_fingerprint_follows_the_files_of_a_read_folder(tmp_path: Path) -> None:
    root = books(tmp_path)
    before = load_tree(root)["coding"].fingerprint
    write(root / "shared/books/python/async.md", "---\ndescription: Async\n---\nAwait more.\n")
    assert load_tree(root)["coding"].fingerprint != before


async def test_the_tools_take_a_read_folder_by_the_path_the_map_shows(tmp_path: Path) -> None:
    surface = EngineTools(Engine(books(tmp_path)), agent="coding", scope="acme")
    assert "Await." in str(
        await surface.dispatch("Read", {"path": "../shared/books/python/async.md"})
    )
    assert (
        await surface.dispatch("Glob", {"pattern": "../shared/books/**/*.md"})
        == "../shared/books/python/async.md"
    )
    found = await surface.dispatch(
        "Grep", {"pattern": "Await", "path": "../shared/books", "mode": "content"}
    )
    assert str(found).startswith("../shared/books/python/async.md:")


async def test_a_glob_cannot_climb_out_of_a_read_folder(tmp_path: Path) -> None:
    surface = EngineTools(Engine(books(tmp_path)), agent="coding", scope="acme")
    with pytest.raises(Exception, match="stay inside"):
        await surface.dispatch("Glob", {"pattern": "../shared/books/../*.md"})
    with pytest.raises(Exception, match="stay inside"):
        await surface.dispatch("Glob", {"pattern": "../shared/*.md"})


def test_from_agent_spells_a_folder_from_the_agents_own(tmp_path: Path) -> None:
    place = tmp_path / "x"
    assert from_agent("coding", {"shared/books": place}) == {"../shared/books": place}
    assert from_agent("team/coding", {"shared/books": place}) == {"../../shared/books": place}


def test_a_trailing_slash_names_the_same_folder(tmp_path: Path) -> None:
    assert list(load_tree(books(tmp_path, "[shared/books/]"))["coding"].reads) == ["shared/books"]


@pytest.mark.parametrize(
    "written",
    [
        "./shared/books",
        "../shared",
        "/shared/books",
        "shared\\books",
        "shared//books",
        "shared/../shared/books",
        ".hidden",
    ],
)
def test_reads_written_other_than_a_path_from_the_root_is_refused(
    tmp_path: Path, written: str
) -> None:
    with pytest.raises(ConfigurationError, match="path from the root"):
        load_tree(books(tmp_path, f"['{written}']"))


def test_reads_of_a_folder_the_tree_does_not_have_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="reads unknown folder 'shared/films'"):
        load_tree(books(tmp_path, "[shared/films]"))


def test_reads_of_its_own_folder_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="'coding', its own folder"):
        load_tree(books(tmp_path, "[coding]"))


def test_reads_of_a_folder_inside_its_own_is_refused(tmp_path: Path) -> None:
    books(tmp_path)
    write(tmp_path / "AGENT.md", agent_md("desk", extra="reads: [shared/books]\n"))
    with pytest.raises(ConfigurationError, match="inside its own folder"):
        load_tree(tmp_path)


def test_reads_of_two_folders_one_inside_the_other_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="keep one"):
        load_tree(books(tmp_path, "[shared, shared/books]"))


def test_reads_of_a_link_that_leaves_the_root_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "elsewhere/notes.md", "Notes.\n")
    root = books(tmp_path / "root", "[shared/out]")
    (root / "shared/out").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ConfigurationError, match="inside the root"):
        load_tree(root)


def test_the_tree_folders_leave_out_hidden_declared_and_inner_links(tmp_path: Path) -> None:
    root = books(tmp_path / "root")
    write(root / ".git/x", "x")
    write(root / "coding/tools/a.md", "---\ndescription: A\n---\n")
    write(root / "coding/notes/n.md", "N\n")
    write(tmp_path / "elsewhere/sub/n.md", "N\n")
    (root / "zagain").symlink_to(root / "shared", target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    (root / "out").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    found = _tree_folders(root)
    assert sorted(found) == [
        "coding",
        "coding/notes",
        "out",
        "out/sub",
        "shared",
        "shared/books",
        "shared/books/python",
    ]
    assert found["out"] == (tmp_path / "elsewhere").resolve()


def test_a_link_that_sorts_before_its_target_does_not_take_its_folders(tmp_path: Path) -> None:
    root = books(tmp_path)
    (root / "alink").symlink_to(root / "shared", target_is_directory=True)
    spec = load_tree(root)["coding"]
    assert spec.reads == {"shared/books": (root / "shared/books").resolve()}
