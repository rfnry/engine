import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Engine
from rfnry_engine.compiled import load_folder
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import Text
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tools

from .fakes import Scripted
from .trees import provided, write_folder

SHOWN = "data/\n  clean.csv\nkeep.log\nnotes.md"
SEEN = "a.md\ndata/clean.csv\nkeep.log\nnotes.md"
TOOL_JSON = '{"description": "Look up", "input_schema": {"type": "object"}}'


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def folder(
    root: Path,
    files: dict[str, str],
    *,
    gitignore: bool = True,
    include: list[str] | None = None,
    preload: list[str] | None = None,
    tools: dict[str, Any] | None = None,
    picks: list[str] | None = None,
) -> Path:
    section: dict[str, Any] = {"gitignore": gitignore}
    if include is not None or preload is None:
        section["include"] = include or ["."]
    if preload is not None:
        section["preload"] = preload
    entry: dict[str, Any] = {
        "kind": "generation",
        "model": "strong",
        "instructions": "a.md",
        "files": section,
    }
    if picks:
        entry["tools"] = {"include": picks}
    root.mkdir(parents=True, exist_ok=True)
    return write_folder(root, {"a": entry}, tools=tools, files={"a.md": "Work.\n", **files})


FURNISHED = {
    ".gitignore": "*.log\n!keep.log\nbuild/\n",
    "data/.gitignore": "raw.csv\n",
    "notes.md": "Notes.\n",
    "debug.log": "secret\n",
    "keep.log": "kept\n",
    "build/out.txt": "built\n",
    "build/keep.log": "built log\n",
    "data/raw.csv": "raw\n",
    "data/clean.csv": "clean\n",
}


def furnished(root: Path, **options: Any) -> Path:
    return folder(root, FURNISHED, **options)


def tools_of(root: Path, agent: str = "a") -> Tools:
    return EngineTools(
        Engine(root, providers=provided({"strong": Scripted()})), agent=agent, scope="acme"
    )


def missing(path: str) -> str:
    return f"tools/builtins: not a file: {path}"


async def refused(tools: Tools, tool: str, args: dict[str, Any]) -> str:
    with pytest.raises(InputError) as caught:
        await tools.dispatch(tool, args)
    return str(caught.value)


def test_ignored_files_are_left_out_of_the_map(tmp_path: Path) -> None:
    assert load_folder(furnished(tmp_path))["a"].listing == SHOWN


def test_ignored_files_are_left_out_of_a_preloaded_folder(tmp_path: Path) -> None:
    spec = load_folder(furnished(tmp_path, preload=["."]))["a"]
    assert [part.text.split("\n", 1)[0] for part in spec.preload if isinstance(part, Text)] == [
        "<<<a.md>>>",
        "<<<data/clean.csv>>>",
        "<<<keep.log>>>",
        "<<<notes.md>>>",
    ]


@pytest.mark.parametrize("entry", ["debug.log", "build", "build/out.txt", "data/raw.csv"])
def test_naming_an_ignored_file_in_preload_reaches_nothing(tmp_path: Path, entry: str) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_folder(furnished(tmp_path, preload=[entry], include=["notes.md"]))
    assert str(caught.value) == (
        f"compiled: agent 'a' lists {entry!r} in files.preload, which reaches nothing; write a folder, a file or a glob from the folder root that reaches a file, or remove it"
    )


@pytest.mark.parametrize(
    ("tool", "args", "seen"),
    [
        ("Read", {"path": "debug.log"}, None),
        ("Read", {"path": "build/keep.log"}, None),
        ("Read", {"path": "data/raw.csv"}, None),
        ("Read", {"path": "keep.log"}, "kept\n"),
        ("Load", {"path": "debug.log"}, None),
        ("Load", {"path": "build/out.txt"}, None),
        ("Grep", {"pattern": "."}, SEEN),
        ("Grep", {"pattern": ".", "path": "debug.log"}, "no matches found"),
        ("Grep", {"pattern": ".", "path": "build"}, "no matches found"),
        ("Glob", {"pattern": "**/*"}, SEEN),
        ("Glob", {"pattern": "build/*"}, ""),
        ("Glob", {"pattern": "*.log"}, "keep.log"),
        ("LS", {}, "a.md\ndata/\nkeep.log\nnotes.md"),
        ("LS", {"path": "data"}, "clean.csv"),
        ("LS", {"path": "build"}, None),
    ],
)
async def test_the_file_tools_do_not_see_an_ignored_file(
    tmp_path: Path, tool: str, args: dict[str, Any], seen: str | None
) -> None:
    tools = tools_of(furnished(tmp_path))
    if seen is None and tool == "LS":
        assert await refused(tools, tool, args) == f"tools/builtins: not a folder: {args['path']}"
    elif seen is None:
        assert await refused(tools, tool, args) == missing(args["path"])
    else:
        assert await tools.dispatch(tool, args) == seen


async def test_a_link_is_judged_by_its_own_name_and_by_where_it_leads(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    (root / "pointer.txt").symlink_to("debug.log")
    (root / "latest.log").symlink_to("notes.md")
    (root / "out").symlink_to("build")
    assert load_folder(root)["a"].listing == SHOWN
    tools = tools_of(root)
    for path in ("pointer.txt", "latest.log", "out/out.txt"):
        assert await refused(tools, "Read", {"path": path}) == missing(path)
    assert await tools.dispatch("Read", {"path": "notes.md"}) == "Notes.\n"


async def test_a_path_with_dots_is_judged_where_it_leads(tmp_path: Path) -> None:
    tools = tools_of(furnished(tmp_path))
    path = "data/../debug.log"
    assert await refused(tools, "Read", {"path": path}) == missing(path)
    assert await tools.dispatch("Read", {"path": "build/../notes.md"}) == "Notes.\n"


@pytest.mark.parametrize(
    ("tool", "args"),
    [("Read", {}), ("Load", {})],
    ids=["Read", "Load"],
)
async def test_a_link_followed_by_dots_is_judged_where_it_leads(
    tmp_path: Path, tool: str, args: dict[str, Any]
) -> None:
    folder(
        tmp_path,
        {
            "sub/.gitignore": "hidden.txt\n",
            "sub/hidden.txt": "hidden\n",
            "sub/deep/seen.txt": "seen\n",
            "x/y.txt": "y\n",
        },
    )
    (tmp_path / "x/hop").symlink_to("../sub/deep")
    path = "x/hop/../hidden.txt"
    assert await refused(tools_of(tmp_path), tool, {**args, "path": path}) == missing(path)


def crowded(root: Path) -> Path:
    furnished(root)
    write(root / ".gitignore", "*.log\n!keep.log\nbuild/\nnode_modules/\n")
    for n in range(60):
        write(root / f"node_modules/pkg/f{n}.js", "needle\n")
        write(root / f"trace{n}.log", "needle\n")
    return root


@pytest.mark.parametrize(
    ("tool", "args"),
    [("Grep", {"pattern": "."}), ("Glob", {"pattern": "**/*"})],
    ids=["Grep", "Glob"],
)
async def test_a_walk_does_not_count_what_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, args: dict[str, Any]
) -> None:
    monkeypatch.setattr("rfnry_engine.tools.builtins._MAX_WALK", 20)
    assert await tools_of(crowded(tmp_path)).dispatch(tool, args) == SEEN


async def test_a_path_holding_a_nul_reads_as_missing(tmp_path: Path) -> None:
    tools = tools_of(furnished(tmp_path))
    path = "de\x00bug.log"
    assert await refused(tools, "Read", {"path": path}) == missing(path)


@pytest.mark.parametrize("line", ["a.md", "*.md"], ids=str)
async def test_an_edit_to_the_instructions_moves_the_fingerprint_whatever_a_gitignore_says(
    tmp_path: Path, line: str
) -> None:
    root = folder(tmp_path, {".gitignore": f"{line}\n", "keep.txt": "Kept.\n"})
    before = load_folder(root)["a"].fingerprint
    write(root / "a.md", "Edited.\n")
    spec = load_folder(root)["a"]
    assert spec.body == "Edited." and spec.fingerprint != before
    assert await refused(tools_of(root), "Read", {"path": "a.md"}) == missing("a.md")


@pytest.mark.parametrize("line", ["t/", "*.py"], ids=str)
def test_an_edit_to_a_tool_moves_the_fingerprint_whatever_a_gitignore_says(
    tmp_path: Path, line: str
) -> None:
    root = folder(
        tmp_path,
        {
            ".gitignore": f"{line}\n",
            "keep.txt": "Kept.\n",
            "t/lookup.json": TOOL_JSON,
            "t/lookup.py": "def run(args, call):\n    return {}\n",
        },
        tools={"lookup": {"module": "t/lookup.py", "schema": "t/lookup.json"}},
        picks=["lookup"],
    )
    before = load_folder(root)["a"].fingerprint
    write(root / "t/lookup.py", "def run(args, call):\n    return {'new': 1}\n")
    spec = load_folder(root)["a"]
    assert "lookup" in spec.folder_tools and spec.fingerprint != before


def test_an_edit_to_an_ignored_file_leaves_the_fingerprint_alone(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    before = load_folder(root)["a"].fingerprint
    for path in ("debug.log", "build/out.txt", "data/raw.csv"):
        write(root / path, "edited\n")
    assert load_folder(root)["a"].fingerprint == before
    write(root / "keep.log", "edited\n")
    assert load_folder(root)["a"].fingerprint != before


async def test_a_nested_gitignore_and_a_negation_follow_git(tmp_path: Path) -> None:
    files = {
        ".gitignore": "*.log\n!keep.log\nbuild/\n/anchored.txt\n",
        "logs/.gitignore": "!trace.log\n",
        "data/.gitignore": "raw.csv\n/only-here.csv\n",
        "build/.gitignore": "!*\n",
        **dict.fromkeys(
            (
                "anchored.txt",
                "debug.log",
                "keep.log",
                "raw.csv",
                "build/keep.log",
                "data/anchored.txt",
                "data/raw.csv",
                "data/only-here.csv",
                "data/deep/raw.csv",
                "data/deep/only-here.csv",
                "logs/app.log",
                "logs/keep.log",
                "logs/trace.log",
            ),
            "x\n",
        ),
    }
    root = folder(tmp_path, files)
    assert load_folder(root)["a"].listing == (
        "data/\n  anchored.txt\n  deep/\n    only-here.csv\n"
        "keep.log\nlogs/\n  keep.log\n  trace.log\nraw.csv"
    )
    assert await refused(tools_of(root), "Read", {"path": "build/keep.log"}) == missing(
        "build/keep.log"
    )


def test_an_allowlist_gitignore_follows_git(tmp_path: Path) -> None:
    root = folder(
        tmp_path,
        {
            ".gitignore": "*\n!*/\n!*.md\n",
            "notes.md": "x\n",
            "data.csv": "x\n",
            "deep/more/guide.md": "x\n",
            "deep/more/data.csv": "x\n",
        },
    )
    assert load_folder(root)["a"].listing == "deep/\n  more/\n    guide.md\nnotes.md"


def test_a_negated_folder_says_nothing_about_the_files_inside_it(tmp_path: Path) -> None:
    root = folder(
        tmp_path,
        {
            ".gitignore": "*.log\n",
            "group/.gitignore": "!*/\n!logs/\n",
            "group/logs/app.log": "x\n",
            "group/logs/app.txt": "x\n",
        },
    )
    assert load_folder(root)["a"].listing == "group/\n  logs/\n    app.txt"


@pytest.mark.parametrize("line", ["!", "\\", "foo\\", "[z-a]"], ids=str)
def test_a_line_git_skips_is_skipped(tmp_path: Path, line: str) -> None:
    root = furnished(tmp_path)
    write(root / ".gitignore", f"{line}\n*.log\n!keep.log\nbuild/\n")
    assert load_folder(root)["a"].listing == SHOWN


@pytest.mark.parametrize("separator", ["\r", "\x0c", "\x85", " "], ids=["cr", "ff", "nel", "ls"])
def test_a_line_of_a_gitignore_ends_only_at_a_newline(tmp_path: Path, separator: str) -> None:
    root = furnished(tmp_path)
    write(root / ".gitignore", f"nothing{separator}debug.log\n")
    assert "debug.log" in load_folder(root)["a"].listing


def test_a_gitignore_saved_with_crlf_still_ignores(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    (root / ".gitignore").write_bytes(b"*.log\r\n!keep.log\r\nbuild/\r\n")
    assert load_folder(root)["a"].listing == SHOWN


@pytest.mark.parametrize("line", ["foo/**/", "foo/**/  ", "/foo/**/"], ids=str)
def test_a_pattern_ending_in_double_star_slash_follows_git(tmp_path: Path, line: str) -> None:
    root = folder(tmp_path, {".gitignore": f"{line}\n", "foo/x.txt": "x\n", "foo/sub/y.txt": "x\n"})
    assert load_folder(root)["a"].listing == "foo/\n  x.txt"


def a_link(path: Path) -> None:
    (path.parent / "rules.txt").write_text("*.log\n", encoding="utf-8")
    path.unlink()
    path.symlink_to("rules.txt")


def dangling(path: Path) -> None:
    path.unlink()
    path.symlink_to("missing")


def looping(path: Path) -> None:
    path.unlink()
    path.symlink_to(path.name)


def a_fifo(path: Path) -> None:
    path.unlink()
    os.mkfifo(path)


@pytest.mark.parametrize(
    "make", [a_link, dangling, looping, a_fifo], ids=lambda make: make.__name__
)
def test_a_gitignore_that_is_not_a_plain_file_is_not_read(
    tmp_path: Path, make: Callable[[Path], None]
) -> None:
    root = furnished(tmp_path)
    make(root / ".gitignore")
    listing = load_folder(root)["a"].listing
    assert "debug.log" in listing and "build/" in listing


def test_a_gitignore_inside_an_ignored_folder_is_not_read(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    (root / "build/.gitignore").write_bytes(b"\xff\xfe")
    assert load_folder(root)["a"].listing == SHOWN


def test_a_byte_order_mark_does_not_change_a_gitignore(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    (root / ".gitignore").write_bytes(b"\xef\xbb\xbf*.log\n!keep.log\nbuild/\n")
    assert load_folder(root)["a"].listing == SHOWN


async def test_without_gitignore_true_every_file_is_there(tmp_path: Path) -> None:
    root = furnished(tmp_path, gitignore=False)
    (root / ".gitignore").chmod(0)
    try:
        assert load_folder(root)["a"].listing == (
            "build/\n  keep.log\n  out.txt\ndata/\n  clean.csv\n  raw.csv\n"
            "debug.log\nkeep.log\nnotes.md"
        )
        tools = tools_of(root)
        assert await tools.dispatch("Read", {"path": "debug.log"}) == "secret\n"
        assert await tools.dispatch("LS", {"path": "build"}) == "keep.log\nout.txt"
    finally:
        (root / ".gitignore").chmod(0o644)


def test_a_gitignore_above_the_folder_root_is_not_read(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.md\nsecret.txt\n", encoding="utf-8")
    root = folder(tmp_path / "agents", {"secret.txt": "mine\n", "guide.md": "Guide.\n"})
    listing = load_folder(root)["a"].listing
    assert "secret.txt" in listing and "guide.md" in listing


@pytest.mark.parametrize("gitignored", ["first", "second"])
async def test_one_agent_s_gitignore_leaves_another_agent_s_view_alone(
    tmp_path: Path, gitignored: str
) -> None:
    entries: dict[str, Any] = {}
    for name in ("first", "second"):
        entries[name] = {
            "kind": "generation",
            "model": "strong",
            "instructions": f"{name}.md",
            "files": {"include": ["shared"], "gitignore": name == gitignored},
        }
    root = write_folder(
        tmp_path,
        entries,
        files={
            "first.md": "Work.\n",
            "second.md": "Work.\n",
            "shared/.gitignore": "*.tmp\n",
            "shared/guide.md": "Guide.\n",
            "shared/scratch.tmp": "scratch\n",
        },
    )
    other = "second" if gitignored == "first" else "first"
    agents = load_folder(root)
    assert "scratch.tmp" not in agents[gitignored].listing
    assert "scratch.tmp" in agents[other].listing
    path = "shared/scratch.tmp"
    assert await refused(tools_of(root, gitignored), "Read", {"path": path}) == missing(path)
    assert await tools_of(root, other).dispatch("Read", {"path": path}) == "scratch\n"


def unreadable(path: Path) -> None:
    path.chmod(0)


def undecodable(path: Path) -> None:
    path.write_bytes(b"\xff\xfe*.log\n")


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (unreadable, "Permission denied"),
        (undecodable, "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"),
    ],
    ids=["unreadable", "undecodable"],
)
def test_an_unreadable_gitignore_is_refused(
    tmp_path: Path, damage: Callable[[Path], None], reason: str
) -> None:
    root = furnished(tmp_path)
    broken = root / "data/.gitignore"
    damage(broken)
    try:
        with pytest.raises(ConfigurationError) as caught:
            load_folder(root)
    finally:
        broken.chmod(0o644)
    assert str(caught.value) == (
        f"tree: cannot read {broken.resolve()}: {reason}; make it readable to the engine"
    )
