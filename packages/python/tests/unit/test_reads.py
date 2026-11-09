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


def books(root: Path, reads: str = "['../shared/books']") -> Path:
    write(root / "shared/books/python/async.md", "---\ndescription: Async\n---\nAwait.\n")
    write(root / "coding/AGENT.md", agent_md("coding", extra=f"reads: {reads}\n"))
    return root


def test_reads_names_a_folder_by_its_path_from_the_agent_s_folder(tmp_path: Path) -> None:
    spec = load_tree(books(tmp_path))["coding"]
    assert spec.reads == {"shared/books": (tmp_path / "shared/books").resolve()}
    assert "../shared/books/\n  python/\n    async.md — Async" in spec.listing
    assert Engine(tmp_path, models={}).agents["coding"].reads == ["shared/books"]


def test_a_bytecode_cache_is_no_folder_of_the_tree(tmp_path: Path) -> None:
    write(tmp_path / "shared/books/__pycache__/async.cpython-314.pyc", "x")
    books(tmp_path, "['../shared/books/__pycache__']")
    assert "shared/books/__pycache__" not in _tree_folders(tmp_path)[0]
    with pytest.raises(ConfigurationError, match="shared/books/__pycache__"):
        load_tree(tmp_path)


def test_the_fingerprint_follows_the_files_of_a_read_folder(tmp_path: Path) -> None:
    root = books(tmp_path)
    before = load_tree(root)["coding"].fingerprint
    write(root / "shared/books/python/async.md", "---\ndescription: Async\n---\nAwait more.\n")
    assert load_tree(root)["coding"].fingerprint != before


async def test_the_tools_take_a_read_folder_by_the_path_the_map_shows(tmp_path: Path) -> None:
    surface = EngineTools(Engine(books(tmp_path), models={}), agent="coding", scope="acme")
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
    surface = EngineTools(Engine(books(tmp_path), models={}), agent="coding", scope="acme")
    with pytest.raises(Exception, match="stay inside"):
        await surface.dispatch("Glob", {"pattern": "../shared/books/../*.md"})
    with pytest.raises(Exception, match="stay inside"):
        await surface.dispatch("Glob", {"pattern": "../shared/*.md"})


def test_from_agent_spells_a_folder_from_the_agents_own(tmp_path: Path) -> None:
    place = tmp_path / "x"
    assert from_agent("coding", {"shared/books": place}) == {"../shared/books": place}
    assert from_agent("team/coding", {"shared/books": place}) == {"../../shared/books": place}


def test_a_trailing_slash_names_the_same_folder(tmp_path: Path) -> None:
    assert list(load_tree(books(tmp_path, "['../shared/books/']"))["coding"].reads) == [
        "shared/books"
    ]


NOT_FROM_THE_FOLDER = (
    "tree: agent 'coding' reads {written!r}, which is not a path from its folder; write it as './knowledge' or '../shared', "
    "with no leading /, no .. after a name, no hidden folder, no empty segment and no backslash"
)


def test_a_read_folder_is_written_from_the_agent_s_folder(tmp_path: Path) -> None:
    write(tmp_path / "shared/books/a.md", "---\ndescription: A\n---\nAwait.\n")
    write(
        tmp_path / "desks/support/AGENT.md",
        agent_md("support", extra="reads: ['../../shared/books']\n"),
    )
    spec = load_tree(tmp_path)["support"]
    assert spec.reads == {"shared/books": (tmp_path / "shared/books").resolve()}
    assert "../../shared/books/\n  a.md — A" in spec.listing


async def test_a_read_folder_written_from_the_agent_s_folder_is_read_by_that_path(
    tmp_path: Path,
) -> None:
    write(tmp_path / "shared/books/a.md", "---\ndescription: A\n---\nAwait.\n")
    write(
        tmp_path / "desks/support/AGENT.md",
        agent_md("support", extra="reads: ['../../shared/books']\n"),
    )
    surface = EngineTools(Engine(tmp_path, models={}), agent="support", scope="acme")
    assert "Await." in str(await surface.dispatch("Read", {"path": "../../shared/books/a.md"}))


@pytest.mark.parametrize(
    "written",
    ["books", "/books", "./../x", "./a/../b", "./.hidden", "./a//b", "../a\\b", "../.hidden", ""],
    ids=[
        "bare",
        "absolute",
        "dot-dotdot",
        "dotdot-after-name",
        "hidden",
        "empty-segment",
        "backslash",
        "hidden-up",
        "blank",
    ],
)
def test_a_path_not_written_from_the_agent_s_folder_is_refused(
    tmp_path: Path, written: str
) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_tree(books(tmp_path, f"['{written}']"))
    assert str(caught.value) == NOT_FROM_THE_FOLDER.format(written=written)


@pytest.mark.parametrize("written", ["../../x", "../../../shared"])
def test_a_read_above_the_root_of_the_tree_is_refused(tmp_path: Path, written: str) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_tree(books(tmp_path, f"['{written}']"))
    assert str(caught.value) == (
        f"tree: agent 'coding' reads {written!r}, above the root of the tree; a path in AGENT.md stays inside the tree"
    )


def test_a_deep_agent_reading_exactly_the_root_is_refused_as_holding_it(tmp_path: Path) -> None:
    write(tmp_path / "a/b/AGENT.md", agent_md("deep", extra="reads: ['../..']\n"))
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        "tree: agent 'deep' reads '../..', its own folder or one that holds it; reads: names other folders, so drop it"
    )


def test_a_root_agent_reading_up_is_above_the_root(tmp_path: Path) -> None:
    write(tmp_path / "AGENT.md", agent_md("top", extra="reads: ['..']\n"))
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        "tree: agent 'top' reads '..', above the root of the tree; a path in AGENT.md stays inside the tree"
    )


def test_two_spellings_of_one_folder_are_named_as_written(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_tree(books(tmp_path, "['../shared', '../shared/']"))
    assert str(caught.value) == (
        "tree: agent 'coding' reads '../shared' and '../shared/', one inside the other; keep one"
    )


def test_reads_takes_folders_only(tmp_path: Path) -> None:
    write(tmp_path / "policy.md", "Policy.\n")
    with pytest.raises(ConfigurationError) as caught:
        load_tree(books(tmp_path, "['../policy.md']"))
    assert str(caught.value) == (
        "tree: agent 'coding' reads unknown folder '../policy.md'; reads: names a folder by its path from the agent's folder, one of: ../shared, ../shared/books, ../shared/books/python"
    )


def test_a_folder_that_holds_the_agent_is_still_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_tree(books(tmp_path, "['..']"))
    assert str(caught.value) == (
        "tree: agent 'coding' reads '..', its own folder or one that holds it; reads: names other folders, so drop it"
    )


def test_a_read_unknown_folder_is_named_as_written_and_offered_from_the_agent_s_folder(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ConfigurationError, match=r"unknown folder '../shared/films'.*one of: \.\./shared"
    ):
        load_tree(books(tmp_path, "['../shared/films']"))


def test_reads_of_a_folder_the_tree_does_not_have_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="reads unknown folder '../shared/films'"):
        load_tree(books(tmp_path, "['../shared/films']"))


def test_reads_of_its_own_folder_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="'\\.', its own folder"):
        load_tree(books(tmp_path, "['.']"))


def test_reads_of_a_folder_inside_its_own_is_refused(tmp_path: Path) -> None:
    books(tmp_path)
    write(tmp_path / "AGENT.md", agent_md("desk", extra="reads: ['./shared/books']\n"))
    with pytest.raises(ConfigurationError, match="inside its own folder"):
        load_tree(tmp_path)


def test_reads_of_two_folders_one_inside_the_other_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="keep one"):
        load_tree(books(tmp_path, "['../shared', '../shared/books']"))


def test_reads_of_a_link_that_leaves_the_root_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "elsewhere/notes.md", "Notes.\n")
    root = books(tmp_path / "root", "['../shared/out']")
    (root / "shared/out").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ConfigurationError, match="inside the root"):
        load_tree(root)


def test_the_tree_folders_leave_out_hidden_declared_and_inner_links(tmp_path: Path) -> None:
    root = books(tmp_path / "root")
    write(root / ".git/x", "x")
    write(root / "coding/questions/a.json", '{"type": "check", "instructions": "A?"}')
    write(root / "coding/notes/n.md", "N\n")
    write(tmp_path / "elsewhere/sub/n.md", "N\n")
    (root / "zagain").symlink_to(root / "shared", target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    (root / "out").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    found, _ = _tree_folders(root)
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
