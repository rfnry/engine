import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import ChoiceAnswer, Message, Part, Reply, Text, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Fixed, Scripted, text_reply
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


def test_the_map_is_what_allow_reaches_less_deny(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge"], "deny": ["knowledge/drafts/**"]})
    assert _map(folder) == ["knowledge/", "  rates.md", "  refunds.md"]


async def test_the_prompt_shows_the_map(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge/refunds.md"]})
    model = Scripted(text_reply("done"))
    await Engine(folder, providers=provided({"m": model})).run("a", "acme", "hi")
    assert "<<<FILES>>>\nknowledge/\n  refunds.md" in model.calls[0][0]


def test_a_glob_reaches_files_and_the_folders_it_matches(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge/*"], "deny": ["knowledge/r*.md"]})
    assert _map(folder) == ["knowledge/", "  drafts/", "    next.md"]


async def test_read_takes_root_relative_paths(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"allow": ["knowledge"]}))
    assert await surface.dispatch("Read", {"path": "knowledge/refunds.md"}) == (
        "A duplicate is refunded.\n"
    )


async def test_a_denied_path_reads_as_missing(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"allow": ["knowledge"], "deny": ["knowledge/drafts/**"]}))
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
async def test_a_path_outside_allow_reads_as_missing(
    tmp_path: Path, tool: str, args: dict[str, Any], absent: dict[str, Any]
) -> None:
    surface = _surface(_folder(tmp_path, {"allow": ["knowledge"]}))
    answer = await _answer(surface, tool, args)
    assert "A private refund" not in answer
    assert answer == (await _answer(surface, tool, absent)).replace("absent", "private")


async def test_the_searching_tools_stay_inside_allow(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"allow": ["knowledge"]}))
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
        {"allow": ["knowledge"]},
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
    folder = _folder(tmp_path, {"allow": ["."]})
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
        {"a": {**_agent({"allow": ["."]}), "instructions": "./a.md"}},
        files={"a.md": "Work.\n", "b.md": "Other.\n"},
    )
    assert _map(folder) == ["b.md"]


@pytest.mark.parametrize("key", ["allow", "preload"])
def test_naming_rfnry_yaml_is_refused(tmp_path: Path, key: str) -> None:
    _refused(_folder(tmp_path, {key: ["rfnry.yaml"]}), r"a model never reads that file")


async def test_rfnry_yaml_is_never_readable(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["."]})
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


def test_preload_needs_no_allow_and_stays_off_the_map(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"preload": ["knowledge/refunds.md"]})
    assert _map(folder) == []


def test_an_allowed_preload_is_on_the_map_and_sent_once(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"allow": ["knowledge/refunds.md"], "preload": ["knowledge", "knowledge/refunds.md"]},
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


async def test_deny_narrows_a_preload_folder(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"deny": ["knowledge/drafts/**"], "preload": ["knowledge"]})
    assert _preloaded_names(folder) == ["<<<knowledge/rates.md>>>", "<<<knowledge/refunds.md>>>"]
    surface = _surface(folder)
    path = "knowledge/drafts/next.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)
    assert await surface.dispatch("Read", {"path": "knowledge/rates.md"}) == "Rates go up.\n"


def test_deny_narrows_a_preload_glob(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"deny": ["knowledge/rates.md"], "preload": ["knowledge/r*.md"]})
    assert _preloaded_names(folder) == ["<<<knowledge/refunds.md>>>"]


@pytest.mark.parametrize("allow", [{}, {"allow": ["knowledge"]}], ids=["alone", "allowed"])
def test_a_preload_deny_leaves_with_nothing_is_refused(
    tmp_path: Path, allow: dict[str, Any]
) -> None:
    folder = _folder(
        tmp_path,
        {**allow, "deny": ["knowledge/rates.md"], "preload": ["knowledge/rates.md"]},
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
    folder = _folder(tmp_path, {"allow": ["knowledge"]})
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
        {"allow": [".claude/agents"]},
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
    _refused(_folder(tmp_path, {"allow": ["*/x.md"]}, written=written), r"reaches nothing")
    assert _map(_folder(tmp_path, {"allow": ["knowledge/*"]}, written=written)) == [
        "knowledge/",
        "  drafts/",
        "    next.md",
        "  rates.md",
        "  refunds.md",
    ]
    assert _map(_folder(tmp_path, {"allow": ["knowledge/.env"]}, written=written)) == [
        "knowledge/",
        "  .env",
    ]


async def _dotdot_is_refused(root: Path) -> None:
    for pattern in ("../x", "knowledge/../../x", "knowledge/*/../../x", "**/../x"):
        for key in ("allow", "deny", "preload"):
            _refused(_folder(root, {key: [pattern]}), r"not written from the folder root")


async def _a_link_out_is_left_out(root: Path) -> None:
    folder = root / "agents"
    folder.mkdir()
    outside = root / "outside"
    outside.mkdir()
    (outside / "passwd").write_text(SECRET, encoding="utf-8")
    _folder(folder, {"allow": ["knowledge"], "preload": ["knowledge"]})
    _link(folder / "knowledge/link", outside)
    _link(folder / "knowledge/passwd", outside / "passwd")
    _link(folder / "knowledge/etc", "/etc")
    assert not any(word in line for word in ("link", "passwd", "etc") for line in _map(folder))
    preloaded = str(load_folder(folder)["a"].preload)
    assert "root:" not in preloaded and "<<<knowledge/refunds.md>>>" in preloaded
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
    folder = _folder(root, {"allow": ["knowledge"]}, written={"knowledge/.env": "KEY=1\n"})
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
    surface = _surface(_folder(tmp_path, {"allow": ["knowledge"]}))
    assert "must be relative" in await _answer(surface, "Glob", {"pattern": "knowledge/../*"})


async def test_a_link_inside_the_root_reads_by_its_own_name(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge"]})
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


async def test_a_folder_holding_only_links_is_listed_searched_and_globbed(tmp_path: Path) -> None:
    shared = {"shared/policy.md": "A shared policy.\n", "shared/rates.md": "Shared rates.\n"}
    folder = _folder(tmp_path, {"allow": ["linked"]}, written=shared)
    _link(folder / "linked/policy.md", "../shared/policy.md")
    _link(folder / "linked/rates.md", "../shared/rates.md")
    assert _map(folder) == ["linked/", "  policy.md", "  rates.md"]
    surface = _surface(folder)
    both = "linked/policy.md\nlinked/rates.md"
    for pattern, found in (
        ("**/*.md", both),
        ("linked/*", both),
        ("linked/rates.md", "linked/rates.md"),
    ):
        assert await surface.dispatch("Glob", {"pattern": pattern}) == found
    assert await surface.dispatch("Grep", {"pattern": "hared"}) == both
    assert await surface.dispatch("LS", {}) == "linked/"
    assert await surface.dispatch("LS", {"path": "linked"}) == "policy.md\nrates.md"
    assert await surface.dispatch("Read", {"path": "linked/policy.md"}) == "A shared policy.\n"
    _reads_as_missing(
        await _answer(surface, "Read", {"path": "shared/policy.md"}), "shared/policy.md"
    )
    assert "shared/" not in str(await surface.dispatch("Glob", {"pattern": "*/*"}))


def test_a_link_cycle_ends(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge"]})
    _link(folder / "knowledge/up", "..")
    _link(folder / "knowledge/drafts/loop", "../drafts")
    assert "  up/" not in _map(folder)
    assert "    loop/" not in _map(folder)


def test_a_folder_link_to_a_hidden_folder_is_left_out(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge"]}, written={".keys/k.md": "K.\n"})
    _link(folder / "knowledge/keys", "../.keys")
    assert not any("keys" in line for line in _map(folder))


async def test_gitignore_true_leaves_out_what_a_gitignore_ignores(tmp_path: Path) -> None:
    written = {
        ".gitignore": "drafts/\n",
        "knowledge/.gitignore": "rates.md\n",
    }
    folder = _folder(tmp_path, {"allow": ["knowledge"], "gitignore": True}, written=written)
    assert _map(folder) == ["knowledge/", "  refunds.md"]
    surface = _surface(folder)
    for path in ("knowledge/rates.md", "knowledge/drafts/next.md"):
        _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)
    assert _map(_folder(tmp_path, {"allow": ["knowledge"]}, written=written)) == [
        "knowledge/",
        "  drafts/",
        "    next.md",
        "  rates.md",
        "  refunds.md",
    ]


def test_a_gitignore_inside_a_named_hidden_folder_counts(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"allow": [".claude/rules"], "gitignore": True},
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


async def test_file_tools_come_with_files_and_tools_deny_drops_one(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"allow": ["knowledge"]}))
    assert set(surface.names) >= FILE_TOOLS
    surface = _surface(_folder(tmp_path, {"preload": ["knowledge/rates.md"]}))
    assert set(surface.names) >= FILE_TOOLS
    surface = _surface(
        _folder(tmp_path, {"allow": ["knowledge"]}, tools={"deny": ["Grep", "Load"]})
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
    folder = _folder(tmp_path, {"preload": ["knowledge/map.png"]}, tools={"deny": ["Load"]})
    assert "Load opens it" not in str(load_folder(folder)["a"].preload)


@pytest.mark.parametrize(
    ("files", "key", "pattern"),
    [
        ({"allow": ["nowhere"]}, "files.allow", "nowhere"),
        ({"allow": ["knowledge/*.pdf"]}, "files.allow", "knowledge/*.pdf"),
        ({"preload": ["nowhere/x.md"]}, "files.preload", "nowhere/x.md"),
        (
            {"allow": ["knowledge"], "deny": ["knowledge/none/**"]},
            "files.deny",
            "knowledge/none/**",
        ),
        ({"allow": ["knowledge"], "deny": ["private"]}, "files.deny", "private"),
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


@pytest.mark.parametrize("key", ["allow", "preload"])
def test_a_pattern_reaching_only_a_link_out_is_refused(tmp_path: Path, key: str) -> None:
    folder = _folder(tmp_path, {key: ["knowledge/link"]})
    _link(folder / "knowledge/link", tmp_path.parent)
    _refused(folder, rf"files\.{key}.*reaches nothing")


def test_a_preload_of_a_link_to_a_hidden_folder_is_refused(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"preload": ["knowledge/hl"]}, written={".keys/k.md": "K.\n"})
    _link(folder / "knowledge/hl", "../.keys")
    _refused(folder, r"lists 'knowledge/hl' in files\.preload, which reaches nothing")


def test_a_preload_of_a_bytecode_cache_is_refused(tmp_path: Path) -> None:
    named = "knowledge/__pycache__/x.pyc"
    folder = _folder(tmp_path, {"preload": [named]}, written={named: "x"})
    _refused(
        folder, r"lists 'knowledge/__pycache__/x\.pyc' in files\.preload, which reaches nothing"
    )


def test_a_folder_of_hidden_files_only_reaches_nothing(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["vault"]}, written={"vault/.env": "KEY=1\n"})
    _refused(folder, r"files\.allow.*reaches nothing")


def test_a_mode_is_named_when_its_pattern_reaches_nothing(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **_agent({"allow": ["knowledge"]}),
                "modes": {"chat": {"files": {"allow": ["x"]}}},
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
                **_agent({"allow": ["knowledge/rates.md"]}),
                "modes": {"chat": {"files": {"allow": ["private"]}}},
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
        "files": {"deny": ["knowledge/drafts/**"]},
        "modes": {"chat": {"instructions": "a.md", "files": {"allow": ["knowledge"]}}},
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
        {"allow": ["knowledge"], "deny": ["knowledge"]},
        {"allow": ["knowledge"], "deny": ["knowledge/**"]},
        {"allow": ["knowledge/refunds.md"], "deny": ["knowledge/refunds.md"]},
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
        "files": {"allow": ["knowledge"]},
        "modes": {"chat": {"instructions": "a.md"}},
    }
    folder = write_folder(tmp_path, {"a": entry}, files=FILES)
    spec = load_folder(folder)["a"]
    assert spec.needs_mode and spec.listing == "" and spec.preload == []
    assert spec.withheld is not None and spec.withheld(spec.root / "knowledge/refunds.md")
    with pytest.raises(InputError, match=r"^engine: agent 'a' needs a mode; its modes are chat$"):
        _surface(folder)


async def test_reach_is_fixed_at_load(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge"]})
    surface = _surface(folder)
    (folder / "knowledge/later.md").write_text("A later refund.\n", encoding="utf-8")
    path = "knowledge/later.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)
    assert "later" not in str(await surface.dispatch("LS", {"path": "knowledge"}))
    assert "later" not in str(await surface.dispatch("Glob", {"pattern": "**/*"}))
    assert "later" not in str(await surface.dispatch("Grep", {"pattern": "refund"}))
    assert await _surface(folder).dispatch("Read", {"path": path}) == "A later refund.\n"


async def test_a_file_link_to_a_hidden_file_reads_under_the_links_name(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge"]}, written={".env": "KEY=1\n"})
    _link(folder / "knowledge/env", "../.env")
    assert "  env" in _map(folder)
    surface = _surface(folder)
    assert await surface.dispatch("Read", {"path": "knowledge/env"}) == "KEY=1\n"
    assert "KEY" not in await _answer(surface, "Read", {"path": ".env"})
    assert await surface.dispatch("Glob", {"pattern": "*"}) == ""


async def test_a_named_hidden_file_through_a_folder_link_is_as_the_map_shows(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, {"allow": ["alias/.h.md"]}, written={"real/.h.md": "Hello.\n"})
    _link(folder / "alias", "real")
    assert _map(folder) == ["alias/", "  .h.md"]
    surface = _surface(folder)
    assert await surface.dispatch("Read", {"path": "alias/.h.md"}) == "Hello.\n"
    assert await surface.dispatch("LS", {"path": "alias"}) == ".h.md"
    assert await surface.dispatch("Glob", {"pattern": "alias/*"}) == "alias/.h.md"
    assert await surface.dispatch("Grep", {"pattern": "Hello", "path": "alias"}) == "alias/.h.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": "real/.h.md"}), "real/.h.md")


@pytest.mark.parametrize(
    ("deny", "alias", "target"),
    [
        ("knowledge/drafts/**", "knowledge/alias.md", "drafts/next.md"),
        ("private/**", "knowledge/note.md", "../private/notes.md"),
    ],
    ids=["a-denied-file", "a-denied-file-only-the-link-reaches"],
)
async def test_a_link_to_a_denied_file_reads_as_missing_and_is_not_preloaded(
    tmp_path: Path, deny: str, alias: str, target: str
) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge"], "preload": ["knowledge"], "deny": [deny]})
    _link(folder / alias, target)
    spec = load_folder(folder)["a"]
    assert f"  {alias.split('/')[-1]}" not in spec.listing.splitlines()
    assert f"<<<{alias}>>>" not in str(spec.preload)
    assert "<<<knowledge/refunds.md>>>" in str(spec.preload)
    surface = _surface(folder)
    for tool in ("Read", "Load"):
        _reads_as_missing(await _answer(surface, tool, {"path": alias}), alias)
    found = str(await surface.dispatch("Grep", {"pattern": "refund", "path": "knowledge"}))
    assert "knowledge/refunds.md" in found.splitlines() and alias not in found


SKILLS = {"skills/close/SKILL.md": "Close it.\n", "skills/tone/SKILL.md": "Be warm.\n"}


@pytest.mark.parametrize(
    ("skills", "modes"),
    [
        ({"allow": ["skills/*"], "deny": ["close"]}, {}),
        (
            {"preload": ["skills/tone"]},
            {"modes": {"m": {"skills": {"preload": ["skills/close"]}}}},
        ),
    ],
    ids=["denied", "only-a-mode-names-it"],
)
async def test_a_link_into_a_skill_the_entry_does_not_open_reads_as_missing_and_is_not_preloaded(
    tmp_path: Path, skills: dict[str, Any], modes: dict[str, Any]
) -> None:
    folder = _folder(
        tmp_path,
        {"allow": ["knowledge"], "preload": ["knowledge"]},
        written=SKILLS,
        skills=skills,
        **modes,
    )
    _link(folder / "knowledge/sk.md", "../skills/close/SKILL.md")
    spec = load_folder(folder)["a"]
    assert "  sk.md" not in spec.listing.splitlines()
    assert "<<<knowledge/sk.md>>>" not in str(spec.preload)
    surface = _surface(folder)
    for tool in ("Read", "Load"):
        _reads_as_missing(
            await _answer(surface, tool, {"path": "knowledge/sk.md"}), "knowledge/sk.md"
        )


async def test_a_link_into_a_preloaded_skill_reads_under_its_own_name(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path, {"allow": ["knowledge"]}, written=SKILLS, skills={"preload": ["skills/close"]}
    )
    _link(folder / "knowledge/sk.md", "../skills/close/SKILL.md")
    assert "  sk.md" in _map(folder)
    assert await _surface(folder).dispatch("Read", {"path": "knowledge/sk.md"}) == "Close it.\n"


async def test_a_link_inside_an_offered_skill_reaches_only_what_the_entry_opens(
    tmp_path: Path,
) -> None:
    folder = _folder(
        tmp_path,
        {"allow": ["knowledge"]},
        written=SKILLS,
        skills={"allow": ["skills/*"], "deny": ["close"]},
    )
    _link(folder / "skills/tone/close.md", "../close/SKILL.md")
    _link(folder / "skills/tone/warm.md", "SKILL.md")
    _link(folder / "knowledge/tone.md", "../skills/tone/SKILL.md")
    surface = _surface(folder)
    _reads_as_missing(
        await _answer(surface, "Read", {"path": "skills/tone/close.md"}), "skills/tone/close.md"
    )
    for path in ("skills/tone/warm.md", "knowledge/tone.md"):
        assert await surface.dispatch("Read", {"path": path}) == "Be warm.\n"


async def test_a_link_into_a_closed_skill_reads_where_files_allow_names_the_target(
    tmp_path: Path,
) -> None:
    folder = _folder(
        tmp_path,
        {"allow": ["knowledge", "skills/close"]},
        written=SKILLS,
        skills={"preload": ["skills/close"]},
    )
    _link(folder / "knowledge/sk.md", "../skills/close/SKILL.md")
    assert await _surface(folder).dispatch("Read", {"path": "knowledge/sk.md"}) == "Close it.\n"


def test_a_gitignore_beside_a_links_target_counts(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"allow": ["knowledge"], "gitignore": True},
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
    folder = _folder(tmp_path, {"allow": ["knowledge"]}, written={".keys/k.md": "K.\n"})
    _link(folder / place / "hl", "../.keys")
    surface = _surface(folder)

    def at(name: str) -> dict[str, Any]:
        return {key: value.format(f"{place}/{name}") for key, value in args.items()}

    assert await _answer(surface, tool, at("hl")) == (
        await _answer(surface, tool, at("absent"))
    ).replace("absent", "hl")


WHERE = "from pathlib import Path\ndef run(args, call):\n    return Path(__file__).name\n"


async def _seen_from(folder: Path) -> tuple[object, ...]:
    spec = load_folder(folder)["a"]
    async with Engine(folder, providers=provided({"m": Scripted()})) as engine:
        surface = EngineTools(engine, agent="a", scope="acme")
        return (
            spec.listing,
            spec.fingerprint,
            await _answer(surface, "Read", {"path": "knowledge/refunds.md"}),
            await _answer(surface, "Read", {"path": "knowledge/drafts/next.md"}),
            await surface.dispatch("where", {}),
        )


def _refusal(folder: Path) -> str:
    with pytest.raises(ConfigurationError) as raised:
        load_folder(folder)
    return str(raised.value)


async def test_a_folder_reached_through_a_link_answers_as_the_folder_itself(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    write_folder(
        real,
        {
            "a": _agent(
                {"allow": ["knowledge"], "deny": ["knowledge/drafts/**"]},
                tools={"allow": ["where"]},
            )
        },
        tools={"where": {"module": "tools/where.py", "description": "Says where"}},
        files={**FILES, "tools/where.py": WHERE},
    )
    linked = tmp_path / "linked"
    linked.symlink_to(real)
    seen = await _seen_from(real)
    assert seen[2:] == (
        "A duplicate is refunded.\n",
        "tools/builtins: not a file: knowledge/drafts/next.md",
        "where.py",
    )
    assert await _seen_from(linked) == seen
    (real / "tools/where.py").unlink()
    (tmp_path / "where.py").write_text(WHERE, encoding="utf-8")
    (real / "tools/where.py").symlink_to(tmp_path / "where.py")
    assert _refusal(linked) == _refusal(real)
    assert "links to" in _refusal(real)


SPELLINGS = [
    "./knowledge/{}/next.md",
    "knowledge//{}/next.md",
    "knowledge/./{}/next.md",
    "knowledge/{}/next.md/",
    "{root}/knowledge/{}/next.md",
]
SPELLED = [
    ("Read", {"path": "{}"}),
    ("Load", {"path": "{}"}),
    ("LS", {"path": "{}"}),
    ("Grep", {"pattern": "refund", "path": "{}"}),
]


SPELLED_BESIDE = [("Read", {"path": "{}", "where": "folder"}), *SPELLED[1:]]


def _spelled(spelling: str, place: str, folder: Path, args: dict[str, str]) -> dict[str, str]:
    path = spelling.format(place, root=folder)
    return {key: value.format(path) for key, value in args.items()}


async def _withheld_as_absent(
    folder: Path, surface: EngineTools, asked: list[tuple[str, dict[str, str]]]
) -> None:
    for spelling in SPELLINGS:
        for tool, args in asked:
            withheld = await _answer(surface, tool, _spelled(spelling, "drafts", folder, args))
            absent = await _answer(surface, tool, _spelled(spelling, "ghost", folder, args))
            assert withheld == absent.replace("ghost", "drafts"), (tool, spelling)
            assert "draft refund" not in withheld


async def test_every_spelling_of_a_denied_path_answers_as_an_absent_one(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge"], "deny": ["knowledge/drafts/**"]})
    await _withheld_as_absent(folder, _surface(folder), SPELLED)


async def test_every_spelling_of_a_denied_path_answers_as_an_absent_one_beside_a_workspace(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "agents"
    folder.mkdir()
    _folder(
        folder,
        {"allow": ["knowledge"], "deny": ["knowledge/drafts/**"]},
        workspace=True,
    )
    room = tmp_path / "room"
    (room / "knowledge/drafts").mkdir(parents=True)
    (room / "knowledge/drafts/next.md").write_text(
        "A draft refund in the room.\n", encoding="utf-8"
    )
    surface = EngineTools(
        Engine(folder, providers=provided({"m": Scripted()})),
        agent="a",
        scope="acme",
        workspace=Workspace(Local(room), scope="acme", tree=str(folder)),
    )
    await _withheld_as_absent(folder, surface, SPELLED_BESIDE)


EXCLUDED = "InputError: tools/builtins: not a file: knowledge/drafts/next.md"
CHILD = {
    "kind": "generation",
    "model": "m",
    "instructions": "a.md",
    "description": "Reads",
    "files": {"allow": ["knowledge"], "deny": ["knowledge/drafts/**"]},
}
INTENT = '{"type": "choice", "instructions": "Which?", "options": {"child": null, "other": null}}'


def _family(root: Path) -> Path:
    return write_folder(
        root,
        {
            "lead": {
                **_agent({"allow": ["knowledge"]}),
                "orchestration": {"delegate_to": ["child"]},
            },
            "child": CHILD,
            "desk": {
                "kind": "decision",
                "model": "judge",
                "instructions": "desk.md",
                "questions": ["q/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"child": "child"}}},
            },
        },
        files={**FILES, "desk.md": "", "q/intent.json": INTENT},
    )


def _reading() -> Reply:
    calls = [
        ToolCall(id=f"r{n}", name="Read", args={"path": path})
        for n, path in enumerate(["knowledge/drafts/next.md", "knowledge/refunds.md"])
    ]
    return Reply(tool_calls=calls, stop_reason="tool_use")


def _read(run: object) -> list[object]:
    assert isinstance(run, Generation)
    return [step.result for step in run.steps if step.tool == "Read"]


class _Kept:
    def __init__(self) -> None:
        self.runs: list[object] = []

    async def record(self, run: object) -> None:
        self.runs.append(run)


ROUTE = Fixed({"intent": ChoiceAnswer(choice="child", probabilities={"child": 0.9, "other": 0.1})})


async def test_a_delegated_a_routed_and_an_engine_tool_run_open_only_what_their_entry_opens(
    tmp_path: Path,
) -> None:
    folder = _family(tmp_path)
    seen = [EXCLUDED, "A duplicate is refunded.\n"]
    model = Scripted(
        Reply(
            tool_calls=[
                ToolCall(id="own", name="Read", args={"path": "knowledge/drafts/next.md"}),
                ToolCall(id="d1", name="Delegate", args={"agent": "child", "input": "x"}),
            ],
            stop_reason="tool_use",
        ),
        _reading(),
        text_reply("c"),
        text_reply("done"),
    )
    lead = await Engine(folder, providers=provided({"m": model, "judge": ROUTE})).run(
        "lead", "acme", "hi"
    )
    assert _read(lead) == ["A draft refund.\n"]
    assert isinstance(lead, Generation) and _read(lead.children[0]) == seen
    model = Scripted(_reading(), text_reply("c"))
    desk = await Engine(folder, providers=provided({"m": model, "judge": ROUTE})).run(
        "desk", "acme", "hi"
    )
    assert isinstance(desk, Decision) and _read(desk.routed_to) == seen
    kept = _Kept()
    model = Scripted(_reading(), text_reply("c"))
    engine = Engine(folder, refinery=kept, providers=provided({"m": model, "judge": ROUTE}))
    assert await engine.tool("child", scope="acme").dispatch("child", {"input": "x"}) == "c"
    assert _read(kept.runs[-1]) == seen


async def test_the_shell_is_told_where_the_allowed_folders_are(tmp_path: Path) -> None:
    folder = tmp_path / "agents"
    folder.mkdir()
    _folder(
        folder,
        {"allow": ["knowledge", "private/notes.md", "*.md"]},
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
    assert "The folders files.allow opens are at $RFNRY_TREE/knowledge." in (
        shell.description or ""
    )


async def test_the_shell_is_told_the_root_itself_when_it_is_allowed(tmp_path: Path) -> None:
    folder = tmp_path / "agents"
    folder.mkdir()
    _folder(folder, {"allow": ["."]}, workspace=True)
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
    assert "The folders files.allow opens are at $RFNRY_TREE." in (shell.description or "")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_an_unreadable_file_is_left_out_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = _folder(tmp_path, {"allow": ["knowledge"]})
    locked = folder / "knowledge/rates.md"
    locked.chmod(0)
    try:
        assert "  rates.md" not in _map(folder)
    finally:
        locked.chmod(0o644)
    assert f"cannot read {locked}" in caplog.text
