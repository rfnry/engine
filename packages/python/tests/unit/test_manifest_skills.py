from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, NotFoundError
from rfnry_engine.models import Content, Reply, State, Text, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tree import AgentSpec

from .fakes import Scripted, text_reply
from .trees import provided, write_folder

TONE = "---\nname: tone\ndescription: Keep it warm\n---\nBe warm.\n"
ESCALATE = "---\ndescription: Hand over\n---\nHand it over.\n"
INTERNAL = "Internal notes.\n"
FILES = {
    "a.md": "Work.\n",
    "skills/tone/SKILL.md": TONE,
    "skills/tone/refs/examples.md": "Warm examples.\n",
    "skills/escalate/SKILL.md": ESCALATE,
    "skills/internal/SKILL.md": INTERNAL,
    "skills/loose.md": "Not a skill.\n",
    "knowledge/refunds.md": "A duplicate is refunded.\n",
}
SKILL_TOOLS = ["Glob", "Grep", "LS", "Load", "Read"]


def _agent(skills: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    agent = {"kind": "generation", "model": "m", "instructions": "a.md", **extra}
    return agent if skills is None else {**agent, "skills": skills}


def _folder(
    root: Path,
    skills: dict[str, Any] | None,
    *,
    written: dict[str, str] | None = None,
    **extra: Any,
) -> Path:
    return write_folder(root, {"a": _agent(skills, **extra)}, files={**FILES, **(written or {})})


def _spec(folder: Path) -> AgentSpec:
    return load_folder(folder)["a"]


def _refused(folder: Path, pattern: str) -> None:
    with pytest.raises(ConfigurationError, match=pattern):
        load_folder(folder)


def _surface(folder: Path, invoked: str | None = None) -> EngineTools:
    engine = Engine(folder, providers=provided({"m": Scripted()}))
    return EngineTools(engine, agent="a", scope="acme", invoked=invoked)


async def _answer(surface: EngineTools, tool: str, args: dict[str, Any]) -> str:
    try:
        return str(await surface.dispatch(tool, args))
    except InputError as exc:
        return str(exc)


async def _ran(folder: Path, input: State, **given: Any) -> tuple[Generation, Scripted]:
    model = Scripted(text_reply("Done."))
    run = await Engine(folder, providers=provided({"m": model})).run("a", "acme", input, **given)
    assert isinstance(run, Generation)
    return run, model


def _sent(model: Scripted) -> Content:
    return model.calls[0][1][-1].content


def _invoked(name: str, body: str, said: str) -> list[Text]:
    return [Text(text=f"<<<SKILL skills/{name}/SKILL.md>>>\n{body}"), Text(text=said)]


def _reads_as_missing(answer: str, path: str) -> None:
    assert answer == f"tools/builtins: not a file: {path}"


def test_a_skill_is_a_folder_with_skill_md(tmp_path: Path) -> None:
    spec = _spec(_folder(tmp_path, {"include": ["skills/*"]}))
    assert list(spec.skills) == ["escalate", "internal", "tone"]
    assert spec.skills["tone"].path == "skills/tone/SKILL.md"
    assert spec.skills["tone"].place == tmp_path / "skills/tone"
    assert spec.skills["tone"].description == "Keep it warm"
    assert spec.skills["tone"].body == "Be warm."
    assert spec.skills["internal"].description == ""
    assert spec.offered == {"escalate", "internal", "tone"}


@pytest.mark.parametrize("key", ["include", "preload"])
@pytest.mark.parametrize(
    "path",
    ["skills/loose.md", "skills", "skills/tone/SKILL.md", "skills/ghost", "skills/g*"],
    ids=["a-flat-file", "a-folder-of-skills", "the-skill-file", "missing", "a-glob-of-nothing"],
)
def test_a_path_that_reaches_no_skill_folder_is_refused(
    tmp_path: Path, key: str, path: str
) -> None:
    _refused(
        _folder(tmp_path, {key: [path]}),
        rf"^compiled: agent 'a' lists '{path.replace('*', '.')}' in skills\.{key}, which reaches no skill; a skill is a folder holding SKILL\.md",
    )


def test_a_skill_file_linking_out_of_the_folder_is_no_skill(tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("Elsewhere.\n", encoding="utf-8")
    (tmp_path / "root").mkdir()
    folder = _folder(tmp_path / "root", {"include": ["skills/*"]})
    (folder / "skills/linked").mkdir()
    (folder / "skills/linked/SKILL.md").symlink_to(outside)
    assert "linked" not in _spec(folder).skills


def test_skill_md_at_the_root_is_no_skill(tmp_path: Path) -> None:
    _refused(
        _folder(tmp_path, {"include": ["."]}, written={"SKILL.md": "Root.\n"}), r"reaches no skill"
    )


def test_a_glob_can_reach_skills_at_any_depth(tmp_path: Path) -> None:
    written = {"desk/deep/skills/brief/SKILL.md": "Brief.\n"}
    spec = _spec(_folder(tmp_path, {"include": ["desk/**"]}, written=written))
    assert list(spec.skills) == ["brief"]
    assert spec.skills["brief"].path == "desk/deep/skills/brief/SKILL.md"


@pytest.mark.parametrize(
    "exclude",
    ["internal", "skills/internal", "skills/int*"],
    ids=["a-bare-name", "a-path", "a-glob"],
)
async def test_the_list_is_include_less_exclude(tmp_path: Path, exclude: str) -> None:
    folder = _folder(tmp_path, {"include": ["skills/*"], "exclude": [exclude]})
    assert _spec(folder).offered == {"escalate", "tone"}
    _, model = await _ran(folder, "hi")
    assert model.calls[0][0].split("<<<SKILLS>>>\n")[1].split("\n\n")[0].splitlines() == [
        "skills/escalate/SKILL.md — Hand over",
        "skills/tone/SKILL.md — Keep it warm",
    ]


def test_an_exclude_reaching_no_skill_is_refused(tmp_path: Path) -> None:
    _refused(
        _folder(tmp_path, {"include": ["skills/*"], "exclude": ["ghost"]}),
        r"^compiled: agent 'a' lists 'ghost' in skills\.exclude, which reaches no skill it names",
    )


def test_an_entrys_exclude_may_name_a_skill_only_a_mode_includes(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/tone"], "exclude": ["internal"]},
        modes={"m": {"skills": {"include": ["skills/internal"]}}},
    )
    spec = _spec(folder)
    assert spec.offered == {"tone"}
    assert spec.modes["m"].offered == {"tone"}
    assert list(spec.modes["m"].skills) == ["internal", "tone"]


def test_a_mode_adds_to_the_list(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/tone"]},
        modes={"m": {"skills": {"include": ["skills/escalate"]}}},
    )
    spec = _spec(folder)
    assert spec.offered == {"tone"}
    assert spec.modes["m"].offered == {"escalate", "tone"}


async def test_preload_sends_a_skills_text_that_the_list_does_not_offer(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["skills/tone"], "preload": ["skills/escalate"]})
    spec = _spec(folder)
    assert spec.offered == {"tone"}
    assert list(spec.skills) == ["escalate", "tone"]
    run, model = await _ran(folder, "hi")
    sent = [Text(text="<<<SKILL skills/escalate/SKILL.md>>>\nHand it over.")]
    assert run.preload == sent
    assert model.calls[0][1][0].content == sent
    system = model.calls[0][0]
    assert "skills/tone/SKILL.md" in system
    assert "skills/escalate/SKILL.md" not in system


async def test_a_preloaded_skill_not_offered_stays_closed_to_the_file_tools(
    tmp_path: Path,
) -> None:
    surface = _surface(_folder(tmp_path, {"preload": ["skills/tone"]}))
    path = "skills/tone/SKILL.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)


async def test_preload_and_files_preload_are_both_sent(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"preload": ["skills/tone"]},
        files={"preload": ["knowledge/refunds.md"]},
    )
    assert _spec(folder).preload == [
        Text(text="<<<knowledge/refunds.md>>>\nA duplicate is refunded.\n"),
        Text(text="<<<SKILL skills/tone/SKILL.md>>>\nBe warm."),
    ]


def test_a_preloaded_skill_sends_no_frontmatter(tmp_path: Path) -> None:
    written = {
        "skills/tone/SKILL.md": (
            "---\nname: tone\ndescription: Keep it warm\nallowed-tools: Read, Grep\n"
            "model: opus\n---\nBe warm.\n"
        )
    }
    spec = _spec(_folder(tmp_path, {"preload": ["skills/tone"]}, written=written))
    assert spec.preload == [Text(text="<<<SKILL skills/tone/SKILL.md>>>\nBe warm.")]


def test_files_preload_of_a_skill_file_stays_raw_and_is_sent_once(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"preload": ["skills/tone", "skills/escalate"]},
        files={"preload": ["skills/tone/SKILL.md"]},
    )
    assert _spec(folder).preload == [
        Text(text=f"<<<skills/tone/SKILL.md>>>\n{TONE}"),
        Text(text="<<<SKILL skills/escalate/SKILL.md>>>\nHand it over."),
    ]


def test_exclude_narrows_preload(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"preload": ["skills/*"], "exclude": ["internal"]})
    assert [str(getattr(part, "text", "")).splitlines()[0] for part in _spec(folder).preload] == [
        "<<<SKILL skills/escalate/SKILL.md>>>",
        "<<<SKILL skills/tone/SKILL.md>>>",
    ]


def test_a_preload_exclude_leaves_with_nothing_is_refused(tmp_path: Path) -> None:
    _refused(
        _folder(tmp_path, {"preload": ["skills/tone"], "exclude": ["tone"]}),
        r"^compiled: agent 'a' lists 'skills/tone' in skills\.preload, which skills\.exclude takes back out",
    )


async def test_skill_reaches_any_skill_the_entry_or_a_mode_names(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/tone"]},
        modes={"m": {"skills": {"include": ["skills/escalate"]}}},
    )
    for name, body in (("tone", "Be warm."), ("escalate", "Hand it over.")):
        run, model = await _ran(folder, "hi", skill=name)
        assert run.skill == name
        assert _sent(model) == _invoked(name, body, "hi")
    engine = Engine(folder, providers=provided({"m": Scripted()}))
    with pytest.raises(NotFoundError) as caught:
        await engine.run("a", "acme", "hi", skill="internal")
    assert str(caught.value) == (
        "engine: agent 'a' has no skill 'internal'; its skills are escalate, tone"
    )


async def test_skill_reaches_a_skill_the_list_excludes(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["skills/*"], "exclude": ["internal"]})
    run, model = await _ran(folder, "hi", skill="internal")
    assert run.skill == "internal"
    assert _sent(model) == _invoked("internal", "Internal notes.", "hi")


async def test_a_run_of_an_agent_with_no_skills_says_how_to_name_one(tmp_path: Path) -> None:
    engine = Engine(_folder(tmp_path, None), providers=provided({"m": Scripted()}))
    with pytest.raises(NotFoundError) as caught:
        await engine.run("a", "acme", "hi", skill="tone")
    assert str(caught.value) == (
        "engine: agent 'a' has no skill 'tone'; it has no skills; name a folder holding SKILL.md "
        "in skills.include or skills.preload"
    )


async def test_slash_reaches_only_public(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["skills/*"], "public": ["escalate"]})
    run, model = await _ran(folder, "/escalate x")
    assert run.skill == "escalate" and run.input == "x"
    assert _sent(model) == _invoked("escalate", "Hand it over.", "x")
    run, model = await _ran(folder, "/tone x")
    assert run.skill is None and run.input == "/tone x" and _sent(model) == "/tone x"


async def test_public_all_is_every_offered_skill(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {
            "include": ["skills/tone", "skills/escalate"],
            "preload": ["skills/internal"],
            "public": "all",
        },
    )
    assert _spec(folder).public_skills == ("escalate", "tone")
    assert Engine(folder, providers=provided({"m": Scripted()})).agents["a"].public_skills == {
        "escalate": "Hand over",
        "tone": "Keep it warm",
    }
    run, _ = await _ran(folder, "/tone x")
    assert run.skill == "tone"
    run, _ = await _ran(folder, "/internal x")
    assert run.skill is None and run.input == "/internal x"


def test_public_all_with_nothing_offered_is_no_error(tmp_path: Path) -> None:
    assert _spec(_folder(tmp_path, {"public": "all"})).public_skills == ()


@pytest.mark.parametrize(
    ("public", "pattern"),
    [
        (["my skill"], r"'my skill'"),
        ("all", r"'my skill'"),
        (["other skill"], r"'other skill'"),
    ],
    ids=["named", "through-all", "not-offered"],
)
def test_public_names_with_whitespace_are_refused(
    tmp_path: Path, public: list[str] | str, pattern: str
) -> None:
    written = {"skills/my skill/SKILL.md": "Body.\n"}
    _refused(
        _folder(tmp_path, {"include": ["skills/*"], "public": public}, written=written),
        rf"^compiled: agent 'a' makes {pattern} public, and no one can type it with a space in it; rename",
    )


@pytest.mark.parametrize(
    "skills",
    [
        {"include": ["skills/tone"], "public": ["internal"]},
        {"include": ["skills/*"], "exclude": ["internal"], "public": ["internal"]},
        {"include": ["skills/tone"], "preload": ["skills/internal"], "public": ["internal"]},
        {"public": ["internal"]},
    ],
    ids=["named-nowhere", "excluded", "only-preloaded", "offers-nothing"],
)
def test_a_public_name_the_entry_does_not_offer_is_refused(
    tmp_path: Path, skills: dict[str, Any]
) -> None:
    _refused(
        _folder(tmp_path, skills),
        r"^compiled: agent 'a' lists 'internal' in skills\.public, which it does not offer",
    )


def test_a_public_name_a_mode_excludes_is_refused_on_that_mode(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/tone"], "public": ["tone"]},
        modes={"m": {"skills": {"exclude": ["tone"]}}},
    )
    _refused(
        folder,
        r"^compiled: mode 'm' of agent 'a' lists 'tone' in skills\.public, which it does not offer",
    )


def test_public_on_an_entry_that_needs_a_mode_is_checked_on_its_modes(tmp_path: Path) -> None:
    entry = {
        "kind": "generation",
        "model": "m",
        "skills": {"public": ["tone"]},
        "modes": {
            "m": {"instructions": "a.md", "skills": {"include": ["skills/tone"]}},
        },
    }
    spec = load_folder(write_folder(tmp_path, {"a": entry}, files=FILES))["a"]
    assert spec.needs_mode and spec.public_skills == () and spec.offered == frozenset()
    assert list(spec.skills) == ["tone"]
    assert spec.modes["m"].public_skills == ("tone",)


async def test_an_invoked_skills_other_files_are_reachable_on_that_run(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"preload": ["skills/tone"]},
        modes={"m": {"skills": {"preload": ["skills/escalate"]}}},
        files={"include": ["knowledge"]},
    )
    path = "skills/tone/refs/examples.md"
    call = ToolCall(id="r1", name="Read", args={"path": path})
    model = Scripted(Reply(tool_calls=[call], stop_reason="tool_use"), text_reply("done"))
    run = await Engine(folder, providers=provided({"m": model})).run(
        "a", "acme", "hi", skill="tone"
    )
    assert isinstance(run, Generation)
    assert run.steps[0].result == "Warm examples.\n"
    surface = _surface(folder, invoked="tone")
    assert await _answer(surface, "Read", {"path": "knowledge/refunds.md"}) == (
        "A duplicate is refunded.\n"
    )
    assert await _answer(surface, "Read", {"path": path}) == "Warm examples.\n"
    assert "Warm examples." in await _answer(surface, "Load", {"path": path})
    other = _surface(folder, invoked="escalate")
    for closed in (_surface(folder), other):
        for tool in ("Read", "Load"):
            _reads_as_missing(await _answer(closed, tool, {"path": path}), path)


async def test_an_offered_skills_files_are_open_on_every_run(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"include": ["skills/tone"]}))
    assert await _answer(surface, "Read", {"path": "skills/tone/SKILL.md"}) == TONE
    assert await _answer(surface, "Read", {"path": "skills/tone/refs/examples.md"}) == (
        "Warm examples.\n"
    )


async def test_a_skill_excluded_from_the_list_reads_as_missing(tmp_path: Path) -> None:
    surface = _surface(_folder(tmp_path, {"include": ["skills/*"], "exclude": ["internal"]}))
    path = "skills/internal/SKILL.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)


CLAUDE_SKILL = (
    "---\nname: tone\ndescription: Keep it warm\nallowed-tools: Read, Grep\n"
    "argument-hint: [file]\n---\nBe warm.\n"
)


async def test_claude_skills_load_as_they_are(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": [".claude/skills/*"]},
        written={
            ".claude/skills/tone/SKILL.md": CLAUDE_SKILL,
            ".claude/skills/tone/refs/examples.md": "Examples.\n",
            ".claude/skills/tone/.secret": "Hidden.\n",
            ".claude/skills/.drafts/SKILL.md": "Hidden.\n",
            ".claude/settings.json": "{}\n",
        },
    )
    spec = _spec(folder)
    assert list(spec.skills) == ["tone"]
    assert spec.skills["tone"].description == "Keep it warm"
    assert spec.skills["tone"].body == "Be warm."
    _, model = await _ran(folder, "hi")
    assert "<<<SKILLS>>>\n.claude/skills/tone/SKILL.md — Keep it warm" in model.calls[0][0]
    surface = _surface(folder)
    assert await _answer(surface, "Read", {"path": ".claude/skills/tone/SKILL.md"}) == CLAUDE_SKILL
    path = ".claude/skills/tone/refs/examples.md"
    assert await _answer(surface, "Read", {"path": path}) == "Examples.\n"
    assert "is hidden" in await _answer(surface, "Read", {"path": ".claude/skills/tone/.secret"})
    path = ".claude/settings.json"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)


async def test_a_hidden_skill_not_offered_opens_when_invoked(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/tone"]},
        modes={"m": {"skills": {"include": [".claude/skills/*"]}}},
        written={
            ".claude/skills/close/SKILL.md": "Close.\n",
            ".claude/skills/close/notes.md": "Notes.\n",
        },
    )
    path = ".claude/skills/close/notes.md"
    assert await _answer(_surface(folder, invoked="close"), "Read", {"path": path}) == "Notes.\n"
    _reads_as_missing(await _answer(_surface(folder), "Read", {"path": path}), path)


@pytest.mark.parametrize(
    "agent",
    [
        {"skills": {"include": ["one/tone", "two/tone"]}},
        {"skills": {"include": ["one/tone"], "preload": ["two/tone"]}},
        {
            "skills": {"include": ["one/tone"]},
            "modes": {"m": {"skills": {"include": ["two/tone"]}}},
        },
        {
            "modes": {
                "m": {"instructions": "a.md", "skills": {"include": ["one/tone"]}},
                "n": {"instructions": "a.md", "skills": {"include": ["two/tone"]}},
            },
        },
    ],
    ids=["two-includes", "include-and-preload", "entry-and-mode", "two-modes"],
)
def test_two_folders_with_one_name_are_refused(tmp_path: Path, agent: dict[str, Any]) -> None:
    entry: dict[str, Any] = {"kind": "generation", "model": "m", **agent}
    if "modes" not in agent or "skills" in agent:
        entry["instructions"] = "a.md"
    folder = write_folder(
        tmp_path,
        {"a": entry},
        files={"a.md": "Work.\n", "one/tone/SKILL.md": "One.\n", "two/tone/SKILL.md": "Two.\n"},
    )
    _refused(
        folder,
        r"^compiled: .*agent 'a' reaches two skills named 'tone', one/tone and two/tone; a skill is named by its folder, so rename one",
    )


def test_one_folder_named_twice_is_one_skill(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/tone", "./skills/t*"], "preload": ["skills/tone/"]},
    )
    assert list(_spec(folder).skills) == ["tone"]


@pytest.mark.parametrize("key", ["include", "preload"])
@pytest.mark.parametrize("tool", SKILL_TOOLS)
def test_a_skills_section_grants_the_file_tools(tmp_path: Path, key: str, tool: str) -> None:
    folder = _folder(tmp_path, {key: ["skills/tone"]}, tools={"include": [tool]})
    assert _spec(folder).keep == SKILL_TOOLS


@pytest.mark.parametrize("skills", [None, {"public": "all"}], ids=["no-section", "only-public"])
def test_a_skills_section_that_opens_nothing_grants_no_file_tools(
    tmp_path: Path, skills: dict[str, Any] | None
) -> None:
    folder = _folder(tmp_path, skills, tools={"include": ["Read"]})
    _refused(folder, r"which comes with files or skills, and it has none")


def test_the_file_tools_come_with_skills_without_being_listed(tmp_path: Path) -> None:
    spec = _spec(_folder(tmp_path, {"include": ["skills/tone"]}))
    assert spec.keep == SKILL_TOOLS


async def test_an_offered_skill_leaves_the_map(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/tone", "skills/internal"], "exclude": ["internal"]},
        files={"include": ["skills", "knowledge"]},
    )
    assert _spec(folder).listing.splitlines() == [
        "knowledge/",
        "  refunds.md",
        "skills/",
        "  escalate/",
        "    SKILL.md — Hand over",
        "  internal/",
        "    SKILL.md",
        "  loose.md",
    ]


async def test_a_skill_not_offered_is_reachable_where_files_include_covers_it(
    tmp_path: Path,
) -> None:
    folder = _folder(tmp_path, {"preload": ["skills/tone"]}, files={"include": ["skills/tone"]})
    assert _spec(folder).listing.splitlines() == [
        "skills/",
        "  tone/",
        "    SKILL.md — Keep it warm",
        "    refs/",
        "      examples.md",
    ]
    assert await _answer(_surface(folder), "Read", {"path": "skills/tone/SKILL.md"}) == TONE


async def test_gitignore_true_leaves_out_what_a_gitignore_ignores_in_a_skill(
    tmp_path: Path,
) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/*"]},
        files={"gitignore": True},
        written={".gitignore": "skills/internal/\nskills/tone/refs/\n"},
    )
    assert list(_spec(folder).skills) == ["escalate", "tone"]
    path = "skills/tone/refs/examples.md"
    _reads_as_missing(await _answer(_surface(folder), "Read", {"path": path}), path)


def test_skill_text_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["skills/tone"]})
    (folder / "skills/tone/SKILL.md").write_bytes(b"\xff\xfe")
    _refused(folder, r"^compiled: agent 'a' reads 'skills/tone/SKILL\.md', not UTF-8 text")


OTHER_HARNESS = {
    ".claude/skills/review/SKILL.md": (
        "---\nname: review\ndescription: Reviews a pull request\n"
        "argument-hint: [pr-number] [priority] [assignee]\n---\nReview it.\n"
    ),
    ".claude/skills/colon/SKILL.md": "---\ndescription: Review code: find bugs\n---\nFind them.\n",
    ".claude/skills/double/SKILL.md": (
        '---\nargument-hint: [a] [b]\ndescription: "Quoted: yes"\n---\nDouble.\n'
    ),
    ".claude/skills/single/SKILL.md": (
        "---\nargument-hint: [a] [b]\ndescription: 'Single'\n---\nSingle.\n"
    ),
    ".claude/skills/mismatched/SKILL.md": (
        "---\nargument-hint: [a] [b]\ndescription: \"Half'\n---\nHalf.\n"
    ),
    ".claude/skills/nested/SKILL.md": (
        "---\nmetadata:\n  description: Not this\nargument-hint: [a] [b]\n---\nNested.\n"
    ),
    ".claude/skills/block/SKILL.md": (
        "---\nargument-hint: [a] [b]\ndescription: >-\n  Folded\n---\nBlock.\n"
    ),
    ".claude/skills/absent/SKILL.md": "---\nargument-hint: [a] [b]\n---\nAbsent.\n",
}


@pytest.mark.parametrize(
    ("name", "said", "body"),
    [
        ("review", "Reviews a pull request", "Review it."),
        ("colon", "Review code: find bugs", "Find them."),
        ("double", "Quoted: yes", "Double."),
        ("single", "Single", "Single."),
        ("mismatched", "\"Half'", "Half."),
        ("nested", "", "Nested."),
        ("block", "", "Block."),
        ("absent", "", "Absent."),
    ],
)
def test_frontmatter_that_does_not_parse_is_read_for_its_description_line(
    tmp_path: Path, name: str, said: str, body: str
) -> None:
    folder = _folder(tmp_path, {"include": [".claude/skills/*"]}, written=OTHER_HARNESS)
    skill = _spec(folder).skills[name]
    assert skill.description == said
    assert skill.body == body


def test_a_description_that_parses_and_is_no_string_is_still_refused(tmp_path: Path) -> None:
    written = {"skills/bad/SKILL.md": "---\ndescription: [one, two]\n---\nBody.\n"}
    _refused(
        _folder(tmp_path, {"include": ["skills/*"]}, written=written),
        r"^frontmatter: description must be a string in .*skills/bad/SKILL\.md",
    )


def test_a_file_on_the_map_with_frontmatter_that_does_not_parse_loads(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path, None, files={"include": [".claude/skills/colon"]}, written=OTHER_HARNESS
    )
    assert _spec(folder).listing.splitlines() == [
        ".claude/",
        "  skills/",
        "    colon/",
        "      SKILL.md — Review code: find bugs",
    ]


async def test_files_exclude_holds_inside_an_offered_skill(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/tone"]},
        files={"include": ["skills/tone"], "exclude": ["skills/tone/refs/**"]},
    )
    surface = _surface(folder)
    path = "skills/tone/refs/examples.md"
    for tool in ("Read", "Load"):
        _reads_as_missing(await _answer(surface, tool, {"path": path}), path)
    assert await _answer(surface, "Read", {"path": "skills/tone/SKILL.md"}) == TONE


async def test_files_exclude_holds_on_a_run_that_invokes_the_skill(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"preload": ["skills/tone"]},
        files={"include": ["skills/tone"], "exclude": ["skills/tone/refs/**"]},
    )
    path = "skills/tone/refs/examples.md"
    _reads_as_missing(await _answer(_surface(folder, invoked="tone"), "Read", {"path": path}), path)


async def test_files_exclude_may_reach_only_into_a_skill_folder(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path, {"include": ["skills/tone"]}, files={"exclude": ["skills/tone/refs/**"]}
    )
    path = "skills/tone/refs/examples.md"
    _reads_as_missing(await _answer(_surface(folder), "Read", {"path": path}), path)


async def test_files_exclude_leaves_a_listed_skills_own_file_open(tmp_path: Path) -> None:
    folder = _folder(
        tmp_path,
        {"include": ["skills/tone"]},
        files={"include": ["skills"], "exclude": ["skills/**"]},
    )
    surface = _surface(folder)
    assert await _answer(surface, "Read", {"path": "skills/tone/SKILL.md"}) == TONE
    path = "skills/tone/refs/examples.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)


NESTED = {"skills/tone/child/SKILL.md": "Child.\n", "skills/tone/child/notes.md": "Notes.\n"}


async def test_an_excluded_nested_skill_stays_closed_through_its_parent(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["skills/**"], "exclude": ["child"]}, written=NESTED)
    assert "child" not in _spec(folder).offered
    closed = _surface(folder)
    for path in ("skills/tone/child/SKILL.md", "skills/tone/child/notes.md"):
        _reads_as_missing(await _answer(closed, "Read", {"path": path}), path)
    assert await _answer(closed, "Read", {"path": "skills/tone/SKILL.md"}) == TONE
    invoked = _surface(folder, invoked="tone")
    path = "skills/tone/child/SKILL.md"
    _reads_as_missing(await _answer(invoked, "Read", {"path": path}), path)


async def test_an_excluded_nested_skill_opens_when_it_is_the_one_invoked(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["skills/**"], "exclude": ["child"]}, written=NESTED)
    invoked = _surface(folder, invoked="child")
    assert await _answer(invoked, "Read", {"path": "skills/tone/child/notes.md"}) == "Notes.\n"


async def test_a_skill_offered_inside_an_excluded_one_stays_open(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["skills/**"], "exclude": ["tone"]}, written=NESTED)
    surface = _surface(folder)
    assert await _answer(surface, "Read", {"path": "skills/tone/child/notes.md"}) == "Notes.\n"
    path = "skills/tone/refs/examples.md"
    _reads_as_missing(await _answer(surface, "Read", {"path": path}), path)


def test_a_skill_named_in_another_mode_is_in_the_fingerprint(tmp_path: Path) -> None:
    def fingerprints(body: str, examples: str) -> str:
        written = {"skills/escalate/SKILL.md": body, "skills/escalate/notes.md": examples}
        folder = _folder(
            tmp_path,
            {"include": ["skills/tone"]},
            modes={"m": {"skills": {"include": ["skills/escalate"]}}},
            written=written,
        )
        return _spec(folder).fingerprint

    before = fingerprints("One.\n", "Notes.\n")
    assert fingerprints("Two.\n", "Notes.\n") != before
    assert fingerprints("One.\n", "Other notes.\n") != before
    assert fingerprints("One.\n", "Notes.\n") == before


def test_skills_the_entry_does_not_name_stay_out_of_the_fingerprint(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["skills/tone"]})
    before = _spec(folder).fingerprint
    (folder / "skills/escalate/SKILL.md").write_text("Changed.\n", encoding="utf-8")
    assert _spec(folder).fingerprint == before


async def test_a_link_to_rfnry_yaml_inside_a_skill_folder_reads_as_missing(tmp_path: Path) -> None:
    folder = _folder(tmp_path, {"include": ["skills/tone"]})
    (folder / "skills/tone/rules.yaml").symlink_to(folder / "rfnry.yaml")
    path = "skills/tone/rules.yaml"
    _reads_as_missing(await _answer(_surface(folder), "Read", {"path": path}), path)


async def test_a_delegated_run_takes_no_skill(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "lead": _agent({"include": ["skills/tone"]}, orchestration={"delegate_to": ["child"]}),
            "child": _agent({"include": ["skills/escalate"]}),
        },
        files=FILES,
    )
    call = ToolCall(id="d1", name="Delegate", args={"agent": "child", "input": "x"})
    model = Scripted(
        Reply(tool_calls=[call], stop_reason="tool_use"), text_reply("c"), text_reply("done")
    )
    run = await Engine(folder, providers=provided({"m": model})).run(
        "lead", "acme", "hi", skill="tone"
    )
    assert isinstance(run, Generation) and run.skill == "tone"
    child = run.children[0]
    assert isinstance(child, Generation) and child.skill is None
    assert model.calls[1][1][-1].content == "x"
