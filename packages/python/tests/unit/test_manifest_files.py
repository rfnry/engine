import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import Message, Part, Reply, Text, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Scripted, text_reply
from .trees import provided, write_folder

FILE_TOOLS = {"Read", "Grep", "Glob", "LS", "Load"}
SECRET = "root:x:0:0\n"
FILES = {
    "a.md": "Work.\n",
    "knowledge/refunds.md": "A duplicate is refunded.\n",
    "knowledge/rates.md": "Rates go up.\n",
    "knowledge/drafts/next.md": "A draft refund.\n",
    "private/notes.md": "A private refund.\n",
}


def _agent(files: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"kind": "generation", "model": "m", "instructions": "a.md", "files": files, **extra}


def _folder(
    root: Path, files: dict[str, Any], *, written: dict[str, str] | None = None, **extra: Any
) -> Path:
    return write_folder(root, {"a": _agent(files, **extra)}, files={**FILES, **(written or {})})


def _surface(folder: Path) -> EngineTools:
    return EngineTools(
        Engine(folder, providers=provided({"m": Scripted()})), agent="a", scope="acme"
    )


def _map(folder: Path) -> list[str]:
    return load_folder(folder)["a"].listing.splitlines()


async def _answer(surface: EngineTools, tool: str, args: dict[str, Any]) -> str:
    try:
        return str(await surface.dispatch(tool, args))
    except InputError as exc:
        return str(exc)


def _link(place: Path, target: Path | str) -> None:
    place.parent.mkdir(parents=True, exist_ok=True)
    place.symlink_to(target)


def _reads_as_missing(answer: str, path: str) -> None:
    assert answer == f"tools/builtins: not a file: {path}"


def _refused(folder: Path, pattern: str) -> None:
    with pytest.raises(ConfigurationError, match=pattern):
        load_folder(folder)


def test_the_map_is_what_include_reaches_less_exclude(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge"], "exclude": ["knowledge/drafts/**"]})
    assert _map(folder) == ["knowledge/", "  rates.md", "  refunds.md"]


async def test_the_prompt_shows_the_map(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge/refunds.md"]})
    model = Scripted(text_reply("done"))
    await Engine(folder, providers=provided({"m": model})).run("a", "acme", "hi")
    assert "<<<FILES>>>\nknowledge/\n  refunds.md" in model.calls[0][0]


def test_a_glob_reaches_files_and_the_folders_it_matches(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge/*"], "exclude": ["knowledge/r*.md"]})
    assert _map(folder) == ["knowledge/", "  drafts/", "    next.md"]


async def test_read_takes_root_relative_paths(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"include": ["knowledge"]}))
    assert await surface.dispatch("Read", {"path": "knowledge/refunds.md"}) == (
        "A duplicate is refunded.\n"
    )


async def test_an_excluded_path_reads_as_missing(tmp_path: Path) -> None:
    surface = _surface(
        _folder(tmp_path, {"include": ["knowledge"], "exclude": ["knowledge/drafts/**"]})
    )
    path = "knowledge/drafts/next.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)


@pytest.mark.parametrize(
    ("tool", "args", "absent"),
    [
        ("Read", {"path": "private/notes.md"}, {"path": "absent/notes.md"}),
        ("Load", {"path": "private/notes.md"}, {"path": "absent/notes.md"}),
        ("LS", {"path": "private"}, {"path": "absent"}),
        ("Grep", {"pattern": "refund", "path": "private"}, {"pattern": "refund", "path": "absent"}),
        (
            "Grep",
            {"pattern": "refund", "path": "private/notes.md"},
            {"pattern": "refund", "path": "absent/notes.md"},
        ),
        ("Glob", {"pattern": "private/*"}, {"pattern": "absent/*"}),
    ],
)
async def test_a_path_outside_include_reads_as_missing(
    tmp_path: Path, tool: str, args: dict[str, Any], absent: dict[str, Any]
) -> None:
    surface = _surface(_folder(tmp_path, {"include": ["knowledge"]}))
    answer = await _answer(surface, tool, args)
    assert "A private refund" not in answer
    assert answer == (await _answer(surface, tool, absent)).replace("absent", "private")


async def test_the_searching_tools_stay_inside_include(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"include": ["knowledge"]}))
    expected = ["knowledge/drafts/next.md", "knowledge/rates.md", "knowledge/refunds.md"]
    assert str(await surface.dispatch("Glob", {"pattern": "**/*.md"})).splitlines() == expected
    assert str(await surface.dispatch("LS", {})).splitlines() == ["knowledge/"]
    assert str(await surface.dispatch("Grep", {"pattern": "refund"})).splitlines() == [
        "knowledge/drafts/next.md",
        "knowledge/refunds.md",
    ]


async def test_a_path_that_tidies_into_reach_but_leads_elsewhere_reads_as_missing(
    tmp_path: Path,
) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["knowledge"]},
        written={"deep/place/here.md": "x\n", "deep/knowledge/refunds.md": "Elsewhere.\n"},
    )
    _link(folder / "tunnel", folder / "deep/place")
    surface = _surface(folder)
    path = "tunnel/../knowledge/refunds.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)
    path = "knowledge/../private/notes.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)


async def test_the_agents_own_instructions_tools_and_skills_stay_out_of_its_map(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, {"include": ["."]})
    assert _map(folder) == [
        "knowledge/",
        "  drafts/",
        "    next.md",
        "  rates.md",
        "  refunds.md",
        "private/",
        "  notes.md",
    ]


async def test_instructions_spelled_with_a_dot_stay_out_of_the_map(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"a": {**_agent({"include": ["."]}), "instructions": "./a.md"}},
        files={"a.md": "Work.\n", "b.md": "Other.\n"},
    )
    assert _map(folder) == ["b.md"]


@pytest.mark.parametrize("key", ["include", "preload"])
def test_naming_rfnry_yaml_is_refused(tmp_path: Path, key: str) -> None:
    _refused(_folder(tmp_path, {key: ["rfnry.yaml"]}), r"a model never reads that file")


async def test_rfnry_yaml_is_never_readable(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["."]})
    _link(folder / "knowledge/copy.yaml", "../rfnry.yaml")
    assert not any("yaml" in line for line in _map(folder))
    surface = _surface(folder)
    for tool in ("Read", "Load"):
        for path in ("rfnry.yaml", "knowledge/copy.yaml"):
            _reads_as_missing(await _answer(surface, tool, {"path": path}), path)
    assert "rfnry.yaml" not in str(await surface.dispatch("LS", {}))
    assert "yaml" not in str(await surface.dispatch("Glob", {"pattern": "**/*"}))
    assert await surface.dispatch("Grep", {"pattern": "version: 1"}) == "no matches found"
    assert await surface.dispatch("Grep", {"pattern": "version", "path": "rfnry.yaml"}) == (
        "no matches found"
    )


async def test_preload_is_sent_every_call_and_is_readable(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"preload": ["rules", "more/one.md"]},
        written={"rules/b.md": "Be brief.\n", "rules/a.md": "Be kind.\n", "more/one.md": "One.\n"},
    )
    call = ToolCall(id="r1", name="Read", args={"path": "rules/a.md"})
    model = Scripted(Reply(tool_calls=[call], stop_reason="tool_use"), text_reply("done"))
    run = await Engine(folder, providers=provided({"m": model})).run("a", "acme", "hi")
    assert isinstance(run, Generation)
    sent: list[Part] = [
        Text(text="<<<rules/a.md>>>\nBe kind.\n"),
        Text(text="<<<rules/b.md>>>\nBe brief.\n"),
        Text(text="<<<more/one.md>>>\nOne.\n"),
    ]
    assert run.preload == sent
    assert len(model.calls) == 2
    for _, messages, _ in model.calls:
        assert messages[0] == Message(role="user", content=sent)
    assert run.steps[0].result == "Be kind.\n"


def test_preload_needs_no_include_and_stays_off_the_map(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"preload": ["knowledge/refunds.md"]})
    assert _map(folder) == []


def test_an_included_preload_is_on_the_map_and_sent_once(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["knowledge/refunds.md"], "preload": ["knowledge", "knowledge/refunds.md"]},
    )
    spec = load_folder(folder)["a"]
    assert spec.listing.splitlines() == ["knowledge/", "  refunds.md"]
    names = [str(getattr(part, "text", "")).splitlines()[0] for part in spec.preload]
    assert names == [
        "<<<knowledge/drafts/next.md>>>",
        "<<<knowledge/rates.md>>>",
        "<<<knowledge/refunds.md>>>",
    ]


def _preloaded_names(folder: Path) -> list[str]:
    return [
        str(getattr(part, "text", "")).splitlines()[0] for part in load_folder(folder)["a"].preload
    ]


async def test_exclude_narrows_a_preload_folder(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"exclude": ["knowledge/drafts/**"], "preload": ["knowledge"]})
    assert _preloaded_names(folder) == ["<<<knowledge/rates.md>>>", "<<<knowledge/refunds.md>>>"]
    surface = _surface(folder)
    path = "knowledge/drafts/next.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)
    assert await surface.dispatch("Read", {"path": "knowledge/rates.md"}) == "Rates go up.\n"


def test_exclude_narrows_a_preload_glob(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"exclude": ["knowledge/rates.md"], "preload": ["knowledge/r*.md"]})
    assert _preloaded_names(folder) == ["<<<knowledge/refunds.md>>>"]


@pytest.mark.parametrize("include", [{}, {"include": ["knowledge"]}], ids=["alone", "included"])
def test_a_preload_exclude_leaves_with_nothing_is_refused(
    tmp_path: Path, include: dict[str, Any]
) -> None:
    folder = _folder(
        tmp_path,
        {**include, "exclude": ["knowledge/rates.md"], "preload": ["knowledge/rates.md"]},
    )
    _refused(
        folder,
        r"^compiled: agent 'a' lists 'knowledge/rates\.md' in files\.preload, which reaches nothing",
    )


def test_a_preload_changes_the_fingerprint_with_its_text(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"preload": ["knowledge/refunds.md"]})
    before = load_folder(folder)["a"].fingerprint
    (folder / "knowledge/refunds.md").write_text("Changed.\n", encoding="utf-8")
    assert load_folder(folder)["a"].fingerprint != before


def test_a_reached_file_changes_the_fingerprint_and_one_outside_does_not(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge"]})
    before = load_folder(folder)["a"].fingerprint
    (folder / "private/notes.md").write_text("Changed.\n", encoding="utf-8")
    assert load_folder(folder)["a"].fingerprint == before
    (folder / "knowledge/rates.md").write_text("Changed.\n", encoding="utf-8")
    assert load_folder(folder)["a"].fingerprint != before


async def test_a_named_hidden_folder_is_reachable_but_hidden_names_below_stay_hidden(
    tmp_path: Path,
) -> None:
    folder = _folder(
        tmp_path,
        {"include": [".claude/agents"]},
        written={
            ".claude/agents/reviewer.md": "Review.\n",
            ".claude/agents/.draft.md": "Draft.\n",
            ".claude/agents/.cache/old.md": "Old.\n",
            ".claude/other.md": "Other.\n",
            ".git/config": "Remote.\n",
        },
    )
    assert _map(folder) == [".claude/", "  agents/", "    reviewer.md"]
    surface = _surface(folder)
    assert await surface.dispatch("Read", {"path": ".claude/agents/reviewer.md"}) == "Review.\n"
    assert await surface.dispatch("LS", {}) == ".claude/"
    assert await surface.dispatch("LS", {"path": ".claude"}) == "agents/"
    assert await surface.dispatch("LS", {"path": ".claude/agents"}) == "reviewer.md"
    assert await surface.dispatch("Glob", {"pattern": "**/*"}) == ".claude/agents/reviewer.md"
    assert await surface.dispatch("Grep", {"pattern": "."}) == ".claude/agents/reviewer.md"
    for path in (".claude/agents/.draft.md", ".claude/agents/.cache/old.md"):
        assert "is hidden" in await _answer(surface, "Read", {"path": path})
    _reads_as_missing(
        await _answer(surface, "Read", {"path": ".claude/other.md"}), ".claude/other.md"
    )
    assert "is hidden" in await _answer(surface, "Read", {"path": ".git/config"})


def test_a_hidden_name_is_reached_only_by_a_literal_segment(tmp_path: Path) -> None:
    written = {".claude/x.md": "X.\n", "knowledge/.env": "KEY=1\n"}
    _refused(_folder(tmp_path, {"include": ["*/x.md"]}, written=written), r"reaches nothing")
    assert _map(_folder(tmp_path, {"include": ["knowledge/*"]}, written=written)) == [
        "knowledge/",
        "  drafts/",
        "    next.md",
        "  rates.md",
        "  refunds.md",
    ]
    assert _map(_folder(tmp_path, {"include": ["knowledge/.env"]}, written=written)) == [
        "knowledge/",
        "  .env",
    ]


async def _dotdot_is_refused(root: Path) -> None:
    for pattern in ("../x", "knowledge/../../x", "knowledge/*/../../x", "**/../x"):
        for key in ("include", "exclude", "preload"):
            _refused(_folder(root, {key: [pattern]}), r"not written from the folder root")


async def _a_link_out_is_left_out(root: Path) -> None:
    folder = root / "agents"
    folder.mkdir()
    outside = root / "outside"
    outside.mkdir()
    (outside / "passwd").write_text(SECRET, encoding="utf-8")
    _folder(folder, {"include": ["knowledge"]})
    _link(folder / "knowledge/link", outside)
    _link(folder / "knowledge/passwd", outside / "passwd")
    _link(folder / "knowledge/etc", "/etc")
    assert not any(word in line for word in ("link", "passwd", "etc") for line in _map(folder))
    surface = _surface(folder)
    for tool in ("Read", "Load"):
        for path in (
            "knowledge/link",
            "knowledge/link/passwd",
            "knowledge/passwd",
            "knowledge/etc/passwd",
        ):
            _reads_as_missing(await _answer(surface, tool, {"path": path}), path)
    assert await _answer(surface, "LS", {"path": "knowledge/link"}) == (
        "tools/builtins: not a folder: knowledge/link"
    )
    for path in ("knowledge/link", "knowledge/link/passwd", "knowledge/passwd"):
        assert await surface.dispatch("Grep", {"pattern": "root", "path": path}) == (
            "no matches found"
        )
    assert await surface.dispatch("Grep", {"pattern": "root"}) == "no matches found"
    assert "link" not in str(await surface.dispatch("LS", {"path": "knowledge"}))
    assert await surface.dispatch("Glob", {"pattern": "knowledge/link/*"}) == ""
    assert "passwd" not in str(await surface.dispatch("Glob", {"pattern": "**/*"}))


async def _a_hidden_env_reads_as_hidden(root: Path) -> None:
    folder = _folder(root, {"include": ["knowledge"]}, written={"knowledge/.env": "KEY=1\n"})
    assert not any(".env" in line for line in _map(folder))
    surface = _surface(folder)
    for tool in ("Read", "Load"):
        assert "is hidden" in await _answer(surface, tool, {"path": "knowledge/.env"})
    assert "is hidden" in await _answer(
        surface, "Grep", {"pattern": "KEY", "path": "knowledge/.env"}
    )
    assert await surface.dispatch("Grep", {"pattern": "KEY"}) == "no matches found"
    assert await surface.dispatch("Glob", {"pattern": "knowledge/.*"}) == ""
    assert ".env" not in str(await surface.dispatch("LS", {"path": "knowledge"}))


@pytest.mark.parametrize(
    "check", [_dotdot_is_refused, _a_link_out_is_left_out, _a_hidden_env_reads_as_hidden]
)
async def test_nothing_escapes(tmp_path: Path, check: Callable[[Path], Awaitable[None]]) -> None:
    await check(tmp_path)


async def test_a_dotdot_glob_given_to_the_tool_is_refused(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"include": ["knowledge"]}))
    assert "must be relative" in await _answer(surface, "Glob", {"pattern": "knowledge/../*"})


async def test_a_link_inside_the_root_reads_by_its_own_name(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge"]})
    _link(folder / "knowledge/note.md", "../private/notes.md")
    _link(folder / "knowledge/shared", "../private")
    assert _map(folder) == [
        "knowledge/",
        "  drafts/",
        "    next.md",
        "  note.md",
        "  rates.md",
        "  refunds.md",
        "  shared/",
        "    notes.md",
    ]
    surface = _surface(folder)
    for path in ("knowledge/note.md", "knowledge/shared/notes.md"):
        assert await surface.dispatch("Read", {"path": path}) == "A private refund.\n"
    assert await surface.dispatch("LS", {"path": "knowledge/shared"}) == "notes.md"
    assert await surface.dispatch("Grep", {"pattern": "private", "path": "knowledge/shared"}) == (
        "knowledge/shared/notes.md"
    )
    assert await surface.dispatch("Glob", {"pattern": "knowledge/shared/*"}) == (
        "knowledge/shared/notes.md"
    )
    _reads_as_missing(
        await _answer(surface, "Read", {"path": "private/notes.md"}), "private/notes.md"
    )


def test_a_link_cycle_ends(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge"]})
    _link(folder / "knowledge/up", "..")
    _link(folder / "knowledge/drafts/loop", "../drafts")
    assert "  up/" not in _map(folder)
    assert "    loop/" not in _map(folder)


def test_a_folder_link_to_a_hidden_folder_is_left_out(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge"]}, written={".keys/k.md": "K.\n"})
    _link(folder / "knowledge/keys", "../.keys")
    assert not any("keys" in line for line in _map(folder))


async def test_gitignore_true_leaves_out_what_a_gitignore_ignores(tmp_path: Path) -> None:
    written = {
        ".gitignore": "drafts/\n",
        "knowledge/.gitignore": "rates.md\n",
    }
    folder = _folder(tmp_path, {"include": ["knowledge"], "gitignore": True}, written=written)
    assert _map(folder) == ["knowledge/", "  refunds.md"]
    surface = _surface(folder)
    for path in ("knowledge/rates.md", "knowledge/drafts/next.md"):
        _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)
    assert _map(_folder(tmp_path, {"include": ["knowledge"]}, written=written)) == [
        "knowledge/",
        "  drafts/",
        "    next.md",
        "  rates.md",
        "  refunds.md",
    ]


def test_a_gitignore_inside_a_named_hidden_folder_counts(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": [".claude/rules"], "gitignore": True},
        written={
            ".claude/rules/kept.md": "K.\n",
            ".claude/rules/gone.md": "G.\n",
            ".claude/.gitignore": "gone.md\n",
        },
    )
    assert _map(folder) == [".claude/", "  rules/", "    kept.md"]


def test_a_gitignored_preload_reaches_nothing(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"preload": ["knowledge/rates.md"], "gitignore": True},
        written={".gitignore": "rates.md\n"},
    )
    _refused(folder, r"files\.preload.*reaches nothing")


async def test_file_tools_come_with_files_and_tools_exclude_drops_one(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"include": ["knowledge"]}))
    assert set(surface.names) >= FILE_TOOLS
    surface = _surface(_folder(tmp_path, {"preload": ["knowledge/rates.md"]}))
    assert set(surface.names) >= FILE_TOOLS
    surface = _surface(
        _folder(tmp_path, {"include": ["knowledge"]}, tools={"exclude": ["Grep", "Load"]})
    )
    assert FILE_TOOLS & set(surface.names) == {"Read", "Glob", "LS"}
    bare = write_folder(
        tmp_path, {"a": {"kind": "generation", "model": "m", "instructions": "a.md"}}, files=FILES
    )
    assert not FILE_TOOLS & set(_surface(bare).names)


def test_a_preload_is_offered_to_load_only_when_load_is_kept(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge/map.png").write_bytes(b"\x89PNG\r\n\x1a\n\0")
    folder = _folder(tmp_path, {"preload": ["knowledge/map.png"]})
    assert "Load opens it" in str(load_folder(folder)["a"].preload)
    folder = _folder(tmp_path, {"preload": ["knowledge/map.png"]}, tools={"exclude": ["Load"]})
    assert "Load opens it" not in str(load_folder(folder)["a"].preload)


@pytest.mark.parametrize(
    ("files", "key", "pattern"),
    [
        ({"include": ["nowhere"]}, "files.include", "nowhere"),
        ({"include": ["knowledge/*.pdf"]}, "files.include", "knowledge/*.pdf"),
        ({"preload": ["nowhere/x.md"]}, "files.preload", "nowhere/x.md"),
        (
            {"include": ["knowledge"], "exclude": ["knowledge/none/**"]},
            "files.exclude",
            "knowledge/none/**",
        ),
        ({"include": ["knowledge"], "exclude": ["private"]}, "files.exclude", "private"),
    ],
)
def test_a_pattern_reaching_nothing_is_refused(
    tmp_path: Path, files: dict[str, Any], key: str, pattern: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        load_folder(_folder(tmp_path, files))
    said = str(raised.value)
    assert said.startswith("compiled: agent 'a' ")
    assert key in said and repr(pattern) in said


def test_a_pattern_reaching_only_a_link_out_is_refused(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge/link"]})
    _link(folder / "knowledge/link", tmp_path.parent)
    _refused(folder, r"files\.include.*reaches nothing")


def test_a_folder_of_hidden_files_only_reaches_nothing(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["vault"]}, written={"vault/.env": "KEY=1\n"})
    _refused(folder, r"files\.include.*reaches nothing")


def test_a_mode_is_named_when_its_pattern_reaches_nothing(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **_agent({"include": ["knowledge"]}),
                "modes": {"chat": {"files": {"include": ["x"]}}},
            }
        },
        files=FILES,
    )
    with pytest.raises(ConfigurationError) as raised:
        load_folder(folder)
    assert str(raised.value).startswith("compiled: mode 'chat' of agent 'a' ")


async def test_a_mode_adds_to_what_its_entry_reaches(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **_agent({"include": ["knowledge/rates.md"]}),
                "modes": {"chat": {"files": {"include": ["private"]}}},
            }
        },
        files=FILES,
    )
    agent = load_folder(folder)["a"]
    assert agent.listing.splitlines() == ["knowledge/", "  rates.md"]
    assert agent.modes["chat"].listing.splitlines() == [
        "knowledge/",
        "  rates.md",
        "private/",
        "  notes.md",
    ]


def test_an_entry_every_mode_completes_is_checked_on_its_modes(tmp_path: Path) -> None:
    entry = {
        "kind": "generation",
        "model": "m",
        "files": {"exclude": ["knowledge/drafts/**"]},
        "modes": {"chat": {"instructions": "a.md", "files": {"include": ["knowledge"]}}},
    }
    agent = load_folder(write_folder(tmp_path, {"a": entry}, files=FILES))["a"]
    assert agent.needs_mode
    assert agent.modes["chat"].listing.splitlines() == ["knowledge/", "  rates.md", "  refunds.md"]


async def _nothing_is_open(surface: EngineTools) -> None:
    for tool in ("Read", "Load"):
        for path in ("rfnry.yaml", "private/notes.md", "knowledge/refunds.md"):
            _reads_as_missing(await _answer(surface, tool, {"path": path}), path)
    assert await surface.dispatch("LS", {}) == "(empty)"
    assert await surface.dispatch("Grep", {"pattern": "."}) == "no matches found"
    assert await surface.dispatch("Glob", {"pattern": "**/*"}) == ""


@pytest.mark.parametrize(
    "files",
    [
        {"include": ["knowledge"], "exclude": ["knowledge"]},
        {"include": ["knowledge"], "exclude": ["knowledge/**"]},
        {"include": ["knowledge/refunds.md"], "exclude": ["knowledge/refunds.md"]},
    ],
    ids=["folder", "glob", "file"],
)
async def test_excluding_everything_leaves_nothing_open(
    tmp_path: Path, files: dict[str, Any]
) -> None:
    surface = _surface(_folder(tmp_path, files))
    assert set(surface.names) >= FILE_TOOLS
    await _nothing_is_open(surface)


def test_an_entry_without_instructions_opens_nothing_itself(tmp_path: Path) -> None:
    entry = {
        "kind": "generation",
        "model": "m",
        "files": {"include": ["knowledge"]},
        "modes": {"chat": {"instructions": "a.md"}},
    }
    folder = write_folder(tmp_path, {"a": entry}, files=FILES)
    spec = load_folder(folder)["a"]
    assert spec.needs_mode and spec.listing == "" and spec.preload == []
    assert spec.withheld is not None and spec.withheld(spec.root / "knowledge/refunds.md")
    with pytest.raises(InputError, match=r"^engine: agent 'a' needs a mode; its modes are chat$"):
        _surface(folder)


async def test_reach_is_fixed_at_load(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge"]})
    surface = _surface(folder)
    (folder / "knowledge/later.md").write_text("A later refund.\n", encoding="utf-8")
    path = "knowledge/later.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)
    assert "later" not in str(await surface.dispatch("LS", {"path": "knowledge"}))
    assert "later" not in str(await surface.dispatch("Glob", {"pattern": "**/*"}))
    assert "later" not in str(await surface.dispatch("Grep", {"pattern": "refund"}))
    assert await _surface(folder).dispatch("Read", {"path": path}) == "A later refund.\n"


async def test_a_file_link_to_a_hidden_file_reads_under_the_links_name(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge"]}, written={".env": "KEY=1\n"})
    _link(folder / "knowledge/env", "../.env")
    assert "  env" in _map(folder)
    surface = _surface(folder)
    assert await surface.dispatch("Read", {"path": "knowledge/env"}) == "KEY=1\n"
    assert "KEY" not in await _answer(surface, "Read", {"path": ".env"})
    assert await surface.dispatch("Glob", {"pattern": "*"}) == ""


async def test_a_named_hidden_file_through_a_folder_link_is_as_the_map_shows(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, {"include": ["alias/.h.md"]}, written={"real/.h.md": "Hello.\n"})
    _link(folder / "alias", "real")
    assert _map(folder) == ["alias/", "  .h.md"]
    surface = _surface(folder)
    assert await surface.dispatch("Read", {"path": "alias/.h.md"}) == "Hello.\n"
    assert await surface.dispatch("LS", {"path": "alias"}) == ".h.md"
    assert await surface.dispatch("Glob", {"pattern": "alias/*"}) == "alias/.h.md"
    assert await surface.dispatch("Grep", {"pattern": "Hello", "path": "alias"}) == "alias/.h.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": "real/.h.md"}), "real/.h.md")


def test_a_gitignore_beside_a_links_target_counts(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["knowledge"], "gitignore": True},
        written={"private/.gitignore": "notes.md\n", "private/deep/kept.md": "K.\n"},
    )
    _link(folder / "knowledge/note.md", "../private/notes.md")
    _link(folder / "knowledge/kept.md", "../private/deep/kept.md")
    shown = _map(folder)
    assert "  note.md" not in shown
    assert "  kept.md" in shown


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("Read", {"path": "{}"}),
        ("Load", {"path": "{}"}),
        ("LS", {"path": "{}"}),
        ("Grep", {"pattern": "K", "path": "{}"}),
    ],
    ids=["Read", "Load", "LS", "Grep"],
)
@pytest.mark.parametrize("place", ["private", "knowledge"])
async def test_a_withheld_link_to_a_hidden_name_answers_as_an_absent_name(
    tmp_path: Path, place: str, tool: str, args: dict[str, Any]
) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge"]}, written={".keys/k.md": "K.\n"})
    _link(folder / place / "hl", "../.keys")
    surface = _surface(folder)

    def at(name: str) -> dict[str, Any]:
        return {key: value.format(f"{place}/{name}") for key, value in args.items()}

    assert await _answer(surface, tool, at("hl")) == (
        await _answer(surface, tool, at("absent"))
    ).replace("absent", "hl")


async def test_the_shell_is_told_where_the_included_folders_are(tmp_path: Path) -> None:
    folder = tmp_path / "agents"
    folder.mkdir()
    _folder(
        folder,
        {"include": ["knowledge", "private/notes.md", "*.md"]},
        workspace=True,
    )
    room = tmp_path / "room"
    room.mkdir()
    model = Scripted(text_reply("done"))
    await Engine(folder, providers=provided({"m": model})).run(
        "a",
        "acme",
        "hi",
        workspace=Workspace(Local(room), scope="acme", tree=str(folder)),
    )
    [shell] = [tool for tool in model.calls[0][2] if tool.name == "Shell"]
    assert "The folders it reads are at $RFNRY_TREE/knowledge." in (shell.description or "")


async def test_the_shell_is_told_the_root_itself_when_it_is_included(tmp_path: Path) -> None:
    folder = tmp_path / "agents"
    folder.mkdir()
    _folder(folder, {"include": ["."]}, workspace=True)
    room = tmp_path / "room"
    room.mkdir()
    model = Scripted(text_reply("done"))
    await Engine(folder, providers=provided({"m": model})).run(
        "a",
        "acme",
        "hi",
        workspace=Workspace(Local(room), scope="acme", tree=str(folder)),
    )
    [shell] = [tool for tool in model.calls[0][2] if tool.name == "Shell"]
    assert "The folders it reads are at $RFNRY_TREE." in (shell.description or "")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_an_unreadable_file_is_left_out_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = _folder(tmp_path, {"include": ["knowledge"]})
    locked = folder / "knowledge/rates.md"
    locked.chmod(0)
    try:
        assert "  rates.md" not in _map(folder)
    finally:
        locked.chmod(0o644)
    assert f"cannot read {locked}" in caplog.text
