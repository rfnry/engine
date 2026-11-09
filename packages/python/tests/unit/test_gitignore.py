import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pathspec import GitIgnoreSpec

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.folders import Gitignored
from rfnry_engine.models import Text
from rfnry_engine.tools.builtins import _ignored
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tools
from rfnry_engine.tree import _gitignore, load_tree

from .fakes import Scripted

ON = "gitignore: true\n"
SHOWN = "data/\n  clean.csv\nkeep.log\nnotes.md"
SEEN = "AGENT.md\ndata/clean.csv\nkeep.log\nnotes.md"
BAD_FRONTMATTER = "---\n: [not yaml\n---\nBody.\n"
TOOL_JSON = '{"description": "Look up", "input_schema": {"type": "object"}}'


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, extra: str = "", kind: str = "generation") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: strong\n{extra}---\nWork.\n"


def furnished(root: Path, extra: str = ON) -> Path:
    write(root / "a/AGENT.md", agent_md("a", extra))
    write(root / "a/.gitignore", "*.log\n!keep.log\nbuild/\n")
    write(root / "a/data/.gitignore", "raw.csv\n")
    write(root / "a/notes.md", "Notes.\n")
    write(root / "a/debug.log", "secret\n")
    write(root / "a/keep.log", "kept\n")
    write(root / "a/build/out.txt", "built\n")
    write(root / "a/build/keep.log", "built log\n")
    write(root / "a/data/raw.csv", "raw\n")
    write(root / "a/data/clean.csv", "clean\n")
    return root


def tools_of(root: Path, agent: str = "a") -> Tools:
    return EngineTools(Engine(root, models={"strong": Scripted()}), agent=agent, scope="acme")


def ignored(path: str) -> str:
    return (
        f"tools/builtins: {path} is ignored by a .gitignore; it is not part of the agent's folder"
    )


async def refused(tools: Tools, tool: str, args: dict[str, Any]) -> str:
    with pytest.raises(InputError) as caught:
        await tools.dispatch(tool, args)
    return str(caught.value)


def test_ignored_files_are_left_out_of_the_map(tmp_path: Path) -> None:
    assert load_tree(furnished(tmp_path))["a"].listing == SHOWN


def test_ignored_files_are_left_out_of_a_preloaded_folder(tmp_path: Path) -> None:
    spec = load_tree(furnished(tmp_path, ON + "preload: ['.']\n"))["a"]
    assert [part.text.split("\n", 1)[0] for part in spec.preload if isinstance(part, Text)] == [
        "<<<data/clean.csv>>>",
        "<<<keep.log>>>",
        "<<<notes.md>>>",
    ]


@pytest.mark.parametrize(
    ("entry", "shown"),
    [
        ("./debug.log", "debug.log"),
        ("./build", "build"),
        ("./build/out.txt", "build/out.txt"),
        ("./data/raw.csv", "data/raw.csv"),
    ],
)
def test_naming_an_ignored_file_in_preload_is_refused(
    tmp_path: Path, entry: str, shown: str
) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_tree(furnished(tmp_path, ON + f"preload: ['{entry}']\n"))
    assert str(caught.value) == (
        f"tree: agent 'a' cannot preload: {ignored(shown)}; fix that file, or drop it from preload:"
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
        ("Grep", {"pattern": ".", "path": "debug.log"}, None),
        ("Grep", {"pattern": ".", "path": "build"}, None),
        ("Glob", {"pattern": "**/*"}, SEEN),
        ("Glob", {"pattern": "build/*"}, ""),
        ("Glob", {"pattern": "*.log"}, "keep.log"),
        ("LS", {}, "AGENT.md\ndata/\nkeep.log\nnotes.md"),
        ("LS", {"path": "data"}, "clean.csv"),
        ("LS", {"path": "build"}, None),
    ],
)
async def test_the_file_tools_do_not_see_an_ignored_file(
    tmp_path: Path, tool: str, args: dict[str, Any], seen: str | None
) -> None:
    tools = tools_of(furnished(tmp_path))
    if seen is None:
        assert await refused(tools, tool, args) == ignored(args["path"])
    else:
        assert await tools.dispatch(tool, args) == seen


async def test_a_link_is_judged_by_its_own_name_and_by_where_it_leads(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    (root / "a/pointer.txt").symlink_to("debug.log")
    (root / "a/latest.log").symlink_to("notes.md")
    (root / "a/out").symlink_to("build")
    assert load_tree(root)["a"].listing == SHOWN
    tools = tools_of(root)
    for path in ("pointer.txt", "latest.log", "out/out.txt"):
        assert await refused(tools, "Read", {"path": path}) == ignored(path)
    assert await tools.dispatch("Read", {"path": "notes.md"}) == "Notes.\n"


async def test_a_path_with_dots_is_judged_where_it_leads(tmp_path: Path) -> None:
    tools = tools_of(furnished(tmp_path))
    path = "data/../debug.log"
    assert await refused(tools, "Read", {"path": path}) == ignored(path)
    assert await tools.dispatch("Read", {"path": "build/../notes.md"}) == "Notes.\n"


@pytest.mark.parametrize(
    ("tool", "args"),
    [("Read", {}), ("Load", {}), ("Grep", {"pattern": "."})],
    ids=["Read", "Load", "Grep"],
)
async def test_a_link_followed_by_dots_is_judged_where_it_leads(
    tmp_path: Path, tool: str, args: dict[str, Any]
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", ON))
    write(tmp_path / "a/sub/.gitignore", "hidden.txt\n")
    write(tmp_path / "a/sub/hidden.txt", "hidden\n")
    write(tmp_path / "a/sub/deep/seen.txt", "seen\n")
    write(tmp_path / "a/x/y.txt", "y\n")
    (tmp_path / "a/x/hop").symlink_to("../sub/deep")
    path = "x/hop/../hidden.txt"
    assert await refused(tools_of(tmp_path), tool, {**args, "path": path}) == ignored(path)


def crowded(root: Path) -> Path:
    furnished(root)
    write(root / "a/.gitignore", "*.log\n!keep.log\nbuild/\nnode_modules/\n")
    for n in range(60):
        write(root / f"a/node_modules/pkg/f{n}.js", "needle\n")
        write(root / f"a/trace{n}.log", "needle\n")
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


async def test_grep_does_not_walk_into_an_ignored_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = tools_of(crowded(tmp_path))
    asked: list[str] = []

    def counted(jail: Any, path: str | Path) -> bool:
        asked.append(str(path))
        return _ignored(jail, path)

    monkeypatch.setattr("rfnry_engine.tools.builtins._ignored", counted)
    assert await tools.dispatch("Grep", {"pattern": "."}) == SEEN
    assert asked and not any("node_modules/" in path for path in asked)


async def test_a_path_the_jail_refuses_is_still_the_jail_s_to_refuse(tmp_path: Path) -> None:
    tools = tools_of(furnished(tmp_path))
    path = "de\x00bug.log"
    assert await refused(tools, "Read", {"path": path}) == (
        f"jail: path outside the agent directory: {path}"
    )


@pytest.mark.parametrize("line", ["AGENT.md", "*.md", "*"], ids=str)
async def test_an_edit_to_agent_md_moves_the_fingerprint_whatever_a_gitignore_says(
    tmp_path: Path, line: str
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", ON))
    write(tmp_path / "a/.gitignore", f"{line}\n")
    before = load_tree(tmp_path)["a"].fingerprint
    write(tmp_path / "a/AGENT.md", agent_md("a", ON).replace("Work.", "Edited."))
    spec = load_tree(tmp_path)["a"]
    assert spec.body == "Edited." and spec.fingerprint != before
    assert await refused(tools_of(tmp_path), "Read", {"path": "AGENT.md"}) == ignored("AGENT.md")


@pytest.mark.parametrize("line", ["tools/", "*.py", "*"], ids=str)
def test_an_edit_to_a_tool_moves_the_fingerprint_whatever_a_gitignore_says(
    tmp_path: Path, line: str
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", ON))
    write(tmp_path / "a/.gitignore", f"{line}\n")
    write(tmp_path / "a/tools/lookup.json", TOOL_JSON)
    write(tmp_path / "a/tools/lookup.py", "def run(args, call):\n    return {}\n")
    before = load_tree(tmp_path)["a"].fingerprint
    write(tmp_path / "a/tools/lookup.py", "def run(args, call):\n    return {'new': 1}\n")
    spec = load_tree(tmp_path)["a"]
    assert "lookup" in spec.folder_tools and spec.fingerprint != before


def test_an_edit_to_an_ignored_file_leaves_the_fingerprint_alone(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    before = load_tree(root)["a"].fingerprint
    for path in ("debug.log", "build/out.txt", "data/raw.csv"):
        write(root / "a" / path, "edited\n")
    assert load_tree(root)["a"].fingerprint == before
    write(root / "a/keep.log", "edited\n")
    assert load_tree(root)["a"].fingerprint != before


async def test_a_nested_gitignore_and_a_negation_follow_git(tmp_path: Path) -> None:
    root = tmp_path
    write(root / "a/AGENT.md", agent_md("a", ON))
    write(root / "a/.gitignore", "*.log\n!keep.log\nbuild/\n/anchored.txt\n")
    write(root / "a/logs/.gitignore", "!trace.log\n")
    write(root / "a/data/.gitignore", "raw.csv\n/only-here.csv\n")
    write(root / "a/build/.gitignore", "!*\n")
    for path in (
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
    ):
        write(root / "a" / path, "x\n")
    assert load_tree(root)["a"].listing == (
        "data/\n  anchored.txt\n  deep/\n    only-here.csv\n"
        "keep.log\nlogs/\n  keep.log\n  trace.log\nraw.csv"
    )
    assert await refused(tools_of(root), "Read", {"path": "build/keep.log"}) == ignored(
        "build/keep.log"
    )


def test_an_allowlist_gitignore_follows_git(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", ON))
    write(tmp_path / "a/.gitignore", "*\n!*/\n!*.md\n")
    write(tmp_path / "a/notes.md", "x\n")
    write(tmp_path / "a/data.csv", "x\n")
    write(tmp_path / "a/deep/more/guide.md", "x\n")
    write(tmp_path / "a/deep/more/data.csv", "x\n")
    assert load_tree(tmp_path)["a"].listing == "deep/\n  more/\n    guide.md\nnotes.md"


def test_a_negated_folder_says_nothing_about_the_files_inside_it(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", ON))
    write(tmp_path / "a/.gitignore", "*.log\n")
    write(tmp_path / "a/group/.gitignore", "!*/\n!logs/\n")
    write(tmp_path / "a/group/logs/app.log", "x\n")
    write(tmp_path / "a/group/logs/app.txt", "x\n")
    assert load_tree(tmp_path)["a"].listing == "group/\n  logs/\n    app.txt"


@pytest.mark.parametrize("line", ["!", "\\", "foo\\", "[z-a]"], ids=str)
def test_a_line_git_skips_is_skipped(tmp_path: Path, line: str) -> None:
    root = furnished(tmp_path)
    write(root / "a/.gitignore", f"{line}\n*.log\n!keep.log\nbuild/\n")
    assert load_tree(root)["a"].listing == SHOWN


@pytest.mark.parametrize(
    "separator", ["\r", "\x0c", "\x85", "\u2028"], ids=["cr", "ff", "nel", "ls"]
)
def test_a_line_of_a_gitignore_ends_only_at_a_newline(tmp_path: Path, separator: str) -> None:
    root = furnished(tmp_path)
    write(root / "a/.gitignore", f"nothing{separator}debug.log\n")
    assert "debug.log" in load_tree(root)["a"].listing


def test_a_gitignore_saved_with_crlf_still_ignores(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    (root / "a/.gitignore").write_bytes(b"*.log\r\n!keep.log\r\nbuild/\r\n")
    assert load_tree(root)["a"].listing == SHOWN


@pytest.mark.parametrize("line", ["foo/**/", "foo/**/  ", "/foo/**/"], ids=str)
def test_a_pattern_ending_in_double_star_slash_follows_git(tmp_path: Path, line: str) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", ON))
    write(tmp_path / "a/.gitignore", f"{line}\n")
    write(tmp_path / "a/foo/x.txt", "x\n")
    write(tmp_path / "a/foo/sub/y.txt", "x\n")
    assert load_tree(tmp_path)["a"].listing == "foo/\n  x.txt"


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
    make(root / "a/.gitignore")
    listing = load_tree(root)["a"].listing
    assert "debug.log" in listing and "build/" in listing


def test_a_gitignore_inside_an_ignored_folder_is_not_read(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    (root / "a/build/.gitignore").write_bytes(b"\xff\xfe")
    assert load_tree(root)["a"].listing == SHOWN


def test_a_byte_order_mark_does_not_change_a_gitignore(tmp_path: Path) -> None:
    root = furnished(tmp_path)
    (root / "a/.gitignore").write_bytes(b"\xef\xbb\xbf*.log\n!keep.log\nbuild/\n")
    assert load_tree(root)["a"].listing == SHOWN


@pytest.mark.parametrize("extra", ["", "gitignore: false\n"], ids=["absent", "false"])
async def test_without_gitignore_true_every_file_is_there(tmp_path: Path, extra: str) -> None:
    root = furnished(tmp_path, extra)
    (root / "a/.gitignore").chmod(0)
    try:
        assert load_tree(root)["a"].listing == (
            "build/\n  keep.log\n  out.txt\ndata/\n  clean.csv\n  raw.csv\n"
            "debug.log\nkeep.log\nnotes.md"
        )
        tools = tools_of(root)
        assert await tools.dispatch("Read", {"path": "debug.log"}) == "secret\n"
        assert await tools.dispatch("LS", {"path": "build"}) == "keep.log\nout.txt"
    finally:
        (root / "a/.gitignore").chmod(0o644)


async def test_gitignore_applies_in_read_folders_and_inherited_skills(tmp_path: Path) -> None:
    root = tmp_path
    write(root / "skills/.gitignore", "draft.md\nwip/\n")
    write(root / "skills/close.md", "---\ndescription: Close it\n---\nClose.\n")
    write(root / "skills/draft.md", "---\ndescription: Draft it\n---\nDraft.\n")
    write(root / "skills/wip/SKILL.md", "---\ndescription: Not yet\n---\nLater.\n")
    write(root / "shared/.gitignore", "*.tmp\n")
    write(root / "shared/guide.md", "Guide.\n")
    write(root / "shared/scratch.tmp", "scratch\n")
    write(root / "desks/a/AGENT.md", agent_md("a", ON + "reads: ['../../shared']\n"))
    write(root / "desks/a/.gitignore", "mine.md\nbroken.md\n")
    write(root / "desks/a/skills/mine.md", "---\ndescription: Mine\n---\nMine.\n")
    write(root / "desks/a/skills/broken.md", BAD_FRONTMATTER)
    write(root / "desks/a/skills/own.md", "---\ndescription: Own\n---\nOwn.\n")
    spec = load_tree(root)["a"]
    assert sorted(spec.skills) == ["close", "own"]
    assert "guide.md" in spec.listing and "scratch.tmp" not in spec.listing
    tools = tools_of(root)
    for path in ("../../shared/scratch.tmp", "../../skills/draft.md", "skills/mine.md"):
        assert await refused(tools, "Read", {"path": path}) == ignored(path)
    assert "Close." in await tools.dispatch("Read", {"path": "../../skills/close.md"})
    before = spec.fingerprint
    write(root / "shared/scratch.tmp", "edited\n")
    write(root / "skills/draft.md", "---\ndescription: Draft it\n---\nEdited.\n")
    assert load_tree(root)["a"].fingerprint == before
    write(root / "skills/close.md", "---\ndescription: Close it\n---\nEdited.\n")
    assert load_tree(root)["a"].fingerprint != before


def test_a_gitignore_above_the_agent_s_folder_is_not_read(tmp_path: Path) -> None:
    root = tmp_path
    write(root / ".gitignore", "*.md\nsecret.txt\n")
    write(root / "desks/.gitignore", "*.txt\n*.md\n")
    write(root / "desks/a/AGENT.md", agent_md("a", ON + "reads: ['../../shared']\n"))
    write(root / "desks/a/secret.txt", "mine\n")
    write(root / "shared/guide.md", "Guide.\n")
    write(root / "desks/skills/close.md", "---\ndescription: Close it\n---\nClose.\n")
    spec = load_tree(root)["a"]
    assert "secret.txt" in spec.listing and "guide.md" in spec.listing
    assert sorted(spec.skills) == ["close"]


@pytest.mark.parametrize("gitignored", ["first", "second"])
async def test_one_agent_s_gitignore_leaves_another_agent_s_view_alone(
    tmp_path: Path, gitignored: str
) -> None:
    root = tmp_path
    write(root / "shared/.gitignore", "*.tmp\n")
    write(root / "shared/guide.md", "Guide.\n")
    write(root / "shared/scratch.tmp", "scratch\n")
    for name in ("first", "second"):
        extra = (ON if name == gitignored else "") + "reads: ['../shared']\n"
        write(root / name / "AGENT.md", agent_md(name, extra))
    other = "second" if gitignored == "first" else "first"
    agents = load_tree(root)
    assert "scratch.tmp" not in agents[gitignored].listing
    assert "scratch.tmp" in agents[other].listing
    path = "../shared/scratch.tmp"
    assert await refused(tools_of(root, gitignored), "Read", {"path": path}) == ignored(path)
    assert await tools_of(root, other).dispatch("Read", {"path": path}) == "scratch\n"


async def test_each_gitignore_is_read_once_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    write(root / "skills/.gitignore", "draft.md\n")
    write(root / "skills/close.md", "---\ndescription: Close it\n---\nClose.\n")
    write(root / "shared/.gitignore", "*.tmp\n")
    write(root / "shared/scratch.tmp", "scratch\n")
    for name in ("first", "second"):
        write(root / name / "AGENT.md", agent_md(name, ON + "reads: ['../shared']\n"))
    write(root / "reader/AGENT.md", agent_md("reader", ON + "reads: ['../group']\n"))
    write(root / "group/y/AGENT.md", agent_md("y", ON))
    write(root / "group/y/.gitignore", "*.tmp\n")
    read: list[Path] = []
    checks: list[Path] = []

    def counted(path: Path) -> GitIgnoreSpec:
        read.append(path)
        return _gitignore(path)

    def tallied(place: Path) -> Gitignored:
        checks.append(place)
        return Gitignored(place)

    monkeypatch.setattr("rfnry_engine.tree._gitignore", counted)
    monkeypatch.setattr("rfnry_engine.tree.Gitignored", tallied)
    tools = tools_of(root, "first")
    assert sorted(read) == sorted(
        (root / name / ".gitignore").resolve() for name in ("group/y", "shared", "skills")
    )
    assert sorted(checks) == sorted(
        (root / name).resolve()
        for name in ("first", "group", "group/y", "reader", "second", "shared", "skills")
    )
    write(root / "shared/.gitignore", "")
    path = "../shared/scratch.tmp"
    assert await refused(tools, "Read", {"path": path}) == ignored(path)
    assert len(read) == 3


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
    broken = root / "a/data/.gitignore"
    damage(broken)
    try:
        with pytest.raises(ConfigurationError) as caught:
            load_tree(root)
    finally:
        broken.chmod(0o644)
    assert str(caught.value) == (
        f"tree: cannot read {broken.resolve()}: {reason}; make it readable to the engine"
    )


@pytest.mark.parametrize("value", ["true", "false"])
def test_a_decision_agent_cannot_declare_gitignore(tmp_path: Path, value: str) -> None:
    write(tmp_path / "d/AGENT.md", agent_md("d", f"gitignore: {value}\n", kind="decision"))
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        "tree: decision agent 'd' cannot declare gitignore:; a decision agent reads no files, so give it what it needs in its input or its questions"
    )


@pytest.mark.parametrize("key", ["skills", "public_skills"])
def test_an_ignored_skill_is_refused_as_hidden_not_as_missing(tmp_path: Path, key: str) -> None:
    extra = f"{ON}skills: [local]\n" if key == "skills" else f"{ON}public_skills: [local]\n"
    write(tmp_path / "a/AGENT.md", agent_md("a", extra))
    write(tmp_path / "a/.gitignore", "skills/local.md\n")
    write(tmp_path / "a/skills/local.md", "---\ndescription: Local\n---\nBody.\n")
    verb = "keeps" if key == "skills" else "names"
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: {tmp_path / 'a'}/AGENT.md {verb} 'local' in {key}:, a skill a .gitignore hides "
        f"from agent 'a'; stop ignoring it, or drop it from {key}:"
    )
