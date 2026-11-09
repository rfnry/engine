import logging
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, NotFoundError
from rfnry_engine.models import ChoiceAnswer, Document, Reply, Text, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tools
from rfnry_engine.tools.workspace import Local, Workspace
from rfnry_engine.tree import load_tree

from .fakes import Fixed, Scripted, delegating, text_reply

TONE = "---\ndescription: Root tone\n---\nKeep it warm.\n"
DESK_BRIEF = "---\ndescription: Desk brief\n---\nBrief the desk.\n"
RUN = 'print("plotted")\n'
CLOSE = "Close the ticket.\n"
PLOT = "Plot it with run.py.\n"
REFUSED = "a skills/ folder or a folder inside one; a skills/ folder gives its skills to the agents below it and is not read as content, so drop it from reads:"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", extra: str = "") -> str:
    model = "judge" if kind == "decision" else "strong"
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


def shared(root: Path, **more: str) -> Path:
    for rel, text in {
        "skills/tone.md": TONE,
        "skills/brief.md": "---\ndescription: Root brief\n---\nBe brief.\n",
        "desks/skills/brief.md": DESK_BRIEF,
        "desks/skills/plot/SKILL.md": "Plot it with run.py.\n",
        "desks/skills/plot/run.py": RUN,
        "desks/support/AGENT.md": agent_md("support"),
        "desks/support/skills/close.md": "Close the ticket.\n",
        "desks/triage/AGENT.md": agent_md("triage", "decision"),
        "billing/AGENT.md": agent_md("billing"),
        **more,
    }.items():
        write(root / rel, text)
    return root


async def test_an_agent_has_the_skills_of_every_folder_above_it_a_nearer_one_replacing(
    tmp_path: Path,
) -> None:
    engine = Engine(shared(tmp_path), models={"strong": Scripted(text_reply("Done."))})
    assert engine.agents["support"].skills == {
        "brief": "Desk brief",
        "close": "",
        "plot": "",
        "tone": "Root tone",
    }
    run = await engine.run("support", "acme", "Hello.")
    assert isinstance(run, Generation)
    assert (
        "<<<SKILLS>>>\n../skills/brief.md — Desk brief\nskills/close.md\n../skills/plot/SKILL.md\n../../skills/tone.md — Root tone\n\n<<<TOOLS>>>"
        in run.system
    )
    assert engine.agents["billing"].skills == {"brief": "Root brief", "tone": "Root tone"}


async def test_an_inherited_skill_is_invoked_by_the_path_read_takes(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Done."))
    await Engine(shared(tmp_path), models={"strong": llm}).run(
        "support", "acme", "Write back.", skill="tone"
    )
    first = llm.calls[0][1][0].content[0]
    assert isinstance(first, Text)
    assert first.text.startswith("<<<SKILL ../../skills/tone.md>>>\n")
    assert first.text == "<<<SKILL ../../skills/tone.md>>>\nKeep it warm."


async def test_read_and_load_reach_a_skill_above_by_its_listed_path(tmp_path: Path) -> None:
    root = shared(tmp_path, **{"outside.md": "Not support's.\n"})
    tools = EngineTools(Engine(root, models={}), agent="support", scope="acme")
    assert await tools.dispatch("Read", {"path": "../../skills/tone.md"}) == TONE
    assert await tools.dispatch("Read", {"path": "../skills/plot/run.py"}) == RUN
    loaded = await tools.dispatch("Load", {"path": "../../skills/tone.md"})
    assert isinstance(loaded, list) and len(loaded) == 1
    assert isinstance(loaded[0], Document) and loaded[0].data == TONE.encode()
    assert (
        await tools.dispatch("Glob", {"pattern": "../skills/**/*.md"})
        == "../skills/brief.md\n../skills/plot/SKILL.md"
    )
    found = await tools.dispatch("Grep", {"pattern": "warm", "path": "../../skills"})
    assert found == "../../skills/tone.md"
    assert await tools.dispatch("LS", {"path": "../skills"}) == "brief.md\nplot/"
    for path in ("../../outside.md", "../skills/../../outside.md"):
        with pytest.raises(InputError) as caught:
            await tools.dispatch("Read", {"path": path})
        assert str(caught.value) == f"jail: path outside the agent directory: {path}"


@pytest.mark.parametrize("linked", ["desks/support", "desks"], ids=["agent", "group"])
async def test_a_skill_above_is_reached_by_its_listed_path_through_a_linked_folder(
    tmp_path: Path, linked: str
) -> None:
    root = shared(
        tmp_path / "root",
        **{
            "outside.md": "Not support's.\n",
            "desks/skills/.secret.md": "Hidden.\n",
            "desks/skills-old/x.md": "Old.\n",
        },
    )
    write(tmp_path / "outside.md", "Not support's.\n")
    write(tmp_path / "elsewhere/outside.md", "Not support's.\n")
    away = tmp_path / "elsewhere" / Path(linked).name
    (root / linked).rename(away)
    (root / linked).symlink_to(away, target_is_directory=True)
    engine = Engine(root, models={})
    assert {name: skill.path for name, skill in engine._spec("support").skills.items()} == {
        "brief": "../skills/brief.md",
        "close": "skills/close.md",
        "plot": "../skills/plot/SKILL.md",
        "tone": "../../skills/tone.md",
    }
    tools = EngineTools(engine, agent="support", scope="acme")
    assert await tools.dispatch("Read", {"path": "../../skills/tone.md"}) == TONE
    assert await tools.dispatch("Read", {"path": "../skills/brief.md"}) == DESK_BRIEF
    assert await tools.dispatch("Read", {"path": "../skills//brief.md"}) == DESK_BRIEF
    assert await tools.dispatch("Read", {"path": "../skills/plot/run.py"}) == RUN
    assert await tools.dispatch("Read", {"path": "skills/close.md"}) == "Close the ticket.\n"
    for path, text in (("../../skills/tone.md", TONE), ("../skills/brief.md", DESK_BRIEF)):
        loaded = await tools.dispatch("Load", {"path": path})
        assert isinstance(loaded, list) and len(loaded) == 1
        assert isinstance(loaded[0], Document) and loaded[0].data == text.encode()
    found = await tools.dispatch("Grep", {"pattern": "warm", "path": "../../skills"})
    assert found == "../../skills/tone.md"
    found = await tools.dispatch("Grep", {"pattern": "the desk", "path": "../skills/brief.md"})
    assert found == "../skills/brief.md"
    assert await tools.dispatch("LS", {"path": "../skills"}) == "brief.md\nplot/"
    assert await tools.dispatch("LS", {"path": "../../skills"}) == "brief.md\ntone.md"
    for tool, args in (
        ("Read", {"path": "../../outside.md"}),
        ("Read", {"path": "../skills/../../outside.md"}),
        ("Load", {"path": "../skills/../../outside.md"}),
        ("Grep", {"pattern": "Not", "path": "../skills/../../outside.md"}),
        ("LS", {"path": "../skills/../.."}),
        ("Read", {"path": "../skills-old/x.md"}),
    ):
        with pytest.raises(InputError) as caught:
            await tools.dispatch(tool, args)
        assert str(caught.value) == f"jail: path outside the agent directory: {args['path']}"
    with pytest.raises(InputError) as caught:
        await tools.dispatch("Read", {"path": "../skills/.secret.md"})
    assert str(caught.value) == (
        "tools/builtins: ../skills/.secret.md is hidden; names starting with . and __pycache__ are not part of the agent's folder"
    )
    with pytest.raises(InputError) as caught:
        await tools.dispatch("Read", {"path": "../skills/brief.md/x"})
    assert str(caught.value) == "tools/builtins: not a folder: ../skills/brief.md/x"


def test_an_edit_to_a_skill_above_changes_the_fingerprint_of_exactly_the_agents_below_it(
    tmp_path: Path,
) -> None:
    root = shared(tmp_path)
    before = {name: spec.fingerprint for name, spec in load_tree(root).items()}
    write(root / "desks/skills/brief.md", "---\ndescription: Desk brief\n---\nBrief them all.\n")
    after = {name: spec.fingerprint for name, spec in load_tree(root).items()}
    assert after["support"] != before["support"]
    assert after["billing"] == before["billing"] and after["triage"] == before["triage"]
    write(root / "desks/skills/plot/run.py", 'print("plotted twice")\n')
    again = {name: spec.fingerprint for name, spec in load_tree(root).items()}
    assert again["support"] != after["support"]
    assert again["billing"] == before["billing"] and again["triage"] == before["triage"]


def test_no_skills_folder_appears_in_any_map_or_preload(tmp_path: Path) -> None:
    root = shared(
        tmp_path,
        **{
            "AGENT.md": agent_md("lead", extra="preload: ['.']\n"),
            "desks/notes.md": "Desk notes.\n",
            "desks/support/guide.md": "The guide.\n",
            "docs/skills/plain.md": "Plain content.\n",
            "librarian/AGENT.md": agent_md(
                "librarian", extra="reads: ['../docs']\npreload: ['../docs']\n"
            ),
            "reader/AGENT.md": agent_md(
                "reader", extra="reads: ['../desks']\npreload: ['../desks']\n"
            ),
            "peer/AGENT.md": agent_md(
                "peer", extra="reads: ['../desks/support']\npreload: ['../desks/support']\n"
            ),
        },
    )
    agents = load_tree(root)

    def preloaded(name: str) -> list[str]:
        return [part.text.split("\n")[0] for part in agents[name].preload if isinstance(part, Text)]

    assert agents["reader"].listing == "../desks/\n  notes.md"
    assert preloaded("reader") == ["<<<../desks/notes.md>>>"]
    assert agents["peer"].listing == "../desks/support/\n  guide.md"
    assert preloaded("peer") == ["<<<../desks/support/guide.md>>>"]
    assert agents["librarian"].listing == "../docs/\n  skills/\n    plain.md"
    assert preloaded("librarian") == ["<<<../docs/skills/plain.md>>>"]
    assert agents["lead"].listing == "desks/\n  notes.md\ndocs/\n  skills/\n    plain.md"
    assert preloaded("lead") == ["<<<desks/notes.md>>>", "<<<docs/skills/plain.md>>>"]
    assert sorted(agents["lead"].skills) == ["brief", "tone"]


@pytest.mark.parametrize("named", ["desks/skills", "desks/skills/plot", "skills"])
def test_reads_of_a_skills_folder_is_refused(tmp_path: Path, named: str) -> None:
    root = shared(tmp_path, **{"b/AGENT.md": agent_md("b", extra=f"reads: ['../{named}']\n")})
    with pytest.raises(ConfigurationError) as caught:
        load_tree(root)
    assert str(caught.value) == f"tree: agent 'b' reads '../{named}', {REFUSED}"


def test_reads_of_an_unknown_folder_offers_no_skills_folder(tmp_path: Path) -> None:
    root = shared(tmp_path, **{"b/AGENT.md": agent_md("b", extra="reads: ['../nowhere']\n")})
    with pytest.raises(ConfigurationError) as caught:
        load_tree(root)
    assert str(caught.value) == (
        "tree: agent 'b' reads unknown folder '../nowhere'; reads: names a folder by its path from the agent's folder, one of: ../billing, ../desks, ../desks/support, ../desks/triage"
    )


def test_an_agent_md_under_any_skills_folder_is_never_an_agent(tmp_path: Path) -> None:
    root = shared(
        tmp_path,
        **{
            "desks/skills/x/AGENT.md": agent_md("x"),
            "desks/skills/x/SKILL.md": "Not a skill: its folder holds an AGENT.md.\n",
            "lib/skills/y/AGENT.md": agent_md("y"),
        },
    )
    (root / "alias").symlink_to(root / "lib", target_is_directory=True)
    agents = load_tree(root)
    assert sorted(agents) == ["billing", "support", "triage"]
    assert sorted(agents["support"].skills) == ["brief", "close", "plot", "tone"]


def test_a_decision_agent_takes_no_skills_from_above(tmp_path: Path) -> None:
    engine = Engine(shared(tmp_path), models={})
    assert engine.agents["triage"].skills == {}
    assert engine._spec("triage").skill_folders == {}


def test_a_skills_folder_above_that_is_a_link_gives_no_skills(tmp_path: Path) -> None:
    root = shared(tmp_path / "root")
    elsewhere = tmp_path / "elsewhere"
    (root / "desks/skills").rename(elsewhere)
    (root / "desks/skills").symlink_to(elsewhere, target_is_directory=True)
    spec = load_tree(root)["support"]
    assert {name: skill.description for name, skill in spec.skills.items()} == {
        "brief": "Root brief",
        "close": "",
        "tone": "Root tone",
    }
    assert spec.skill_folders == {"skills": (root / "skills").resolve()}
    write(elsewhere / "brief.md", "Changed behind the link.\n")
    assert load_tree(root)["support"].fingerprint == spec.fingerprint


def test_a_skill_above_that_links_out_of_its_skills_folder_is_never_read(tmp_path: Path) -> None:
    root = shared(tmp_path, **{"desks/notes.md": "Desk notes.\n"})
    (root / "desks/skills/notes.md").symlink_to(root / "desks/notes.md")
    (root / "desks/skills/again.md").symlink_to(root / "desks/skills/brief.md")
    skills = load_tree(root)["support"].skills
    assert sorted(skills) == ["again", "brief", "close", "plot", "tone"]
    assert skills["again"].path == "../skills/again.md"


@pytest.mark.parametrize("locked", ["desks/skills", "desks/skills/plot"])
def test_an_unreadable_skills_folder_above_is_refused_naming_it(
    tmp_path: Path, locked: str, caplog: pytest.LogCaptureFixture
) -> None:
    root = shared(tmp_path)
    place = root / locked
    place.chmod(0)
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(ConfigurationError) as caught:
            load_tree(root)
    finally:
        place.chmod(0o755)
    assert str(caught.value) == (
        f"tree: cannot read {root}/{locked}: Permission denied; make it readable to the engine"
    )
    assert [m for m in caplog.messages if "cannot read" in m] == [
        f"tree: cannot read {place}: [Errno 13] Permission denied: '{place}'; it is left out"
    ]


def with_skills(root: Path, line: str) -> Path:
    return shared(root, **{"desks/support/AGENT.md": agent_md("support", extra=line)})


@pytest.mark.parametrize("line", ["", "skills: all\n"], ids=["absent", "all"])
def test_skills_absent_or_all_keeps_every_skill(tmp_path: Path, line: str) -> None:
    engine = Engine(with_skills(tmp_path, line), models={"strong": Scripted(text_reply("x"))})
    assert set(engine.agents["support"].skills) == {"brief", "close", "plot", "tone"}


async def test_skills_is_what_the_model_is_offered(tmp_path: Path) -> None:
    engine = Engine(
        with_skills(tmp_path, "skills: [brief, tone]\n"),
        models={"strong": Scripted(text_reply("Done."))},
    )
    run = await engine.run("support", "acme", "Hello.")
    assert isinstance(run, Generation)
    assert (
        "<<<SKILLS>>>\n../skills/brief.md — Desk brief\n../../skills/tone.md — Root tone\n\n<<<TOOLS>>>"
        in run.system
    )
    assert engine.agents["support"].skills == {
        "brief": "Desk brief",
        "close": "",
        "plot": "",
        "tone": "Root tone",
    }


async def test_the_code_reaches_a_skill_the_model_is_not_offered(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Done."))
    await Engine(with_skills(tmp_path, "skills: [brief, tone]\n"), models={"strong": llm}).run(
        "support", "acme", "Close it.", skill="close"
    )
    first = llm.calls[0][1][0].content[0]
    assert isinstance(first, Text)
    assert first.text == "<<<SKILL skills/close.md>>>\nClose the ticket."


async def test_skills_none_lists_nothing_and_the_code_still_invokes(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Done."), text_reply("Done."))
    engine = Engine(with_skills(tmp_path, "skills: none\n"), models={"strong": llm})
    run = await engine.run("support", "acme", "Hello.")
    assert isinstance(run, Generation)
    assert "<<<SKILLS>>>" not in run.system
    assert engine._spec("support").offered == frozenset()
    assert set(engine.agents["support"].skills) == {"brief", "close", "plot", "tone"}
    invoked = await engine.run("support", "acme", "Hello.", skill="brief")
    assert isinstance(invoked, Generation) and invoked.skill == "brief"
    first = llm.calls[1][1][0].content[0]
    assert isinstance(first, Text)
    assert first.text == "<<<SKILL ../skills/brief.md>>>\nBrief the desk."


async def test_an_unknown_skill_is_refused_listing_what_the_agent_holds(tmp_path: Path) -> None:
    engine = Engine(with_skills(tmp_path, "skills: [brief]\n"), models={"strong": Scripted()})
    with pytest.raises(NotFoundError) as caught:
        await engine.run("support", "acme", "Hello.", skill="ghost")
    assert str(caught.value) == (
        "engine: agent 'support' has no skill 'ghost'; its skills are brief, close, plot, tone"
    )


def test_skills_naming_a_skill_the_agent_does_not_have_is_refused_with_the_ones_it_has(
    tmp_path: Path,
) -> None:
    with_skills(tmp_path, "skills: [brief, ghost]\n")
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: {tmp_path / 'desks/support'}/AGENT.md keeps 'ghost' in skills:, a skill agent "
        "'support' does not have; its skills are brief, close, plot, tone"
    )


@pytest.mark.parametrize(
    ("line", "said"),
    [
        (
            "skills: [' x']\n",
            "skills: an entry must not be blank or padded, got ' x'; write the name of a skill to keep, such as skills: [escalate]",
        ),
        ("skills: [brief, brief]\n", "skills: 'brief' is listed twice; list each skill once"),
    ],
)
def test_skills_entries_are_plain_names_each_once(tmp_path: Path, line: str, said: str) -> None:
    with_skills(tmp_path, line)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert f"frontmatter: invalid agent {tmp_path / 'desks/support/AGENT.md'}; {said}" in str(
        caught.value
    )


async def refused(tools: Tools, tool: str, args: dict[str, Any]) -> str:
    with pytest.raises(InputError) as caught:
        await tools.dispatch(tool, args)
    return str(caught.value)


def tools_of(root: Path, invoked: str | None = None) -> Tools:
    engine = Engine(root, models={})
    return EngineTools(engine, agent="support", scope="acme", invoked=invoked)


def not_a_file(path: str) -> str:
    return f"tools/builtins: not a file: {path}"


def read(path: str, call_id: str = "r1") -> Reply:
    call = ToolCall(id=call_id, name="Read", args={"path": path})
    return Reply(tool_calls=[call], stop_reason="tool_use")


@pytest.mark.parametrize("tool", ["Read", "Load", "Grep", "Glob", "LS"])
@pytest.mark.parametrize(
    ("line", "hidden", "folder", "missing", "pattern", "listed", "globbed"),
    [
        (
            "skills: [tone, brief]\n",
            "skills/close.md",
            "skills",
            "skills/gone.md",
            "Close the",
            "(empty)",
            "",
        ),
        (
            "skills: [tone, plot]\n",
            "../skills/brief.md",
            "../skills",
            "../skills/gone.md",
            "Brief the",
            "plot/",
            "../skills/plot/SKILL.md\n../skills/plot/run.py",
        ),
        (
            "skills: [tone, brief]\n",
            "../skills/plot/run.py",
            "../skills",
            "../skills/gone/run.py",
            "plotted",
            "brief.md",
            "../skills/brief.md",
        ),
    ],
    ids=["own", "inherited", "inherited-with-its-folder"],
)
async def test_a_skill_not_offered_is_not_there_for_the_file_tools(
    tmp_path: Path,
    tool: str,
    line: str,
    hidden: str,
    folder: str,
    missing: str,
    pattern: str,
    listed: str,
    globbed: str,
) -> None:
    tools = tools_of(with_skills(tmp_path, line))
    if tool in ("Read", "Load"):
        said = await refused(tools, tool, {"path": hidden})
        assert said == not_a_file(hidden)
        assert said == (await refused(tools, tool, {"path": missing})).replace(missing, hidden)
    elif tool == "LS":
        assert await tools.dispatch("LS", {"path": folder}) == listed
        assert await refused(tools, "LS", {"path": hidden}) == (
            f"tools/builtins: not a folder: {hidden}"
        )
    elif tool == "Glob":
        assert await tools.dispatch("Glob", {"pattern": f"{folder}/**/*"}) == globbed
    else:
        for path in (folder, hidden):
            found = await tools.dispatch("Grep", {"pattern": pattern, "path": path})
            assert found == "no matches found"


@pytest.mark.parametrize(
    "path",
    [
        "skills/../skills/close.md",
        "skills/./close.md",
        "skills//close.md",
        "skills/close.md/x",
        "skills/close.md/",
        "link.md",
        "../skills/../skills/brief.md",
        "../skills//brief.md",
        "../skills/plot",
    ],
)
async def test_a_skill_not_offered_is_not_there_however_it_is_spelled(
    tmp_path: Path, path: str
) -> None:
    root = with_skills(tmp_path, "skills: [tone]\n")
    (root / "desks/support/link.md").symlink_to("skills/close.md")
    tools = tools_of(root)
    assert await refused(tools, "Read", {"path": path}) == not_a_file(path)
    assert "link.md" not in await tools.dispatch("LS", {})


async def test_a_path_out_past_a_skill_not_offered_reads_as_past_a_missing_one(
    tmp_path: Path,
) -> None:
    tools = tools_of(with_skills(tmp_path, "skills: [tone]\n"))
    said = await refused(tools, "Read", {"path": "skills/close.md/.."})
    assert said == (await refused(tools, "Read", {"path": "skills/gone.md/.."})).replace(
        "gone.md", "close.md"
    )


@pytest.mark.parametrize("linked", ["desks/support", "desks"], ids=["agent", "group"])
async def test_a_skill_not_offered_is_not_there_through_a_linked_folder(
    tmp_path: Path, linked: str
) -> None:
    root = with_skills(tmp_path / "root", "skills: [tone]\n")
    away = tmp_path / "elsewhere" / Path(linked).name
    away.parent.mkdir(parents=True)
    (root / linked).rename(away)
    (root / linked).symlink_to(away, target_is_directory=True)
    tools = tools_of(root)
    for path in ("skills/close.md", "../skills/brief.md", "../skills/plot/run.py"):
        assert await refused(tools, "Read", {"path": path}) == not_a_file(path)
    assert await tools.dispatch("LS", {"path": "../skills"}) == "(empty)"
    assert await tools.dispatch("Read", {"path": "../../skills/tone.md"}) == TONE


@pytest.mark.parametrize(
    ("offered", "shown", "hidden"),
    [("again", "again", "brief"), ("brief", "brief", "again")],
    ids=["the-link-offered", "the-file-offered"],
)
async def test_a_skill_that_links_to_another_is_judged_by_its_own_name(
    tmp_path: Path, offered: str, shown: str, hidden: str
) -> None:
    root = with_skills(tmp_path, f"skills: [tone, {offered}]\n")
    (root / "desks/skills/again.md").symlink_to(root / "desks/skills/brief.md")
    tools = tools_of(root)
    assert await tools.dispatch("Read", {"path": f"../skills/{shown}.md"}) == DESK_BRIEF
    missing = f"../skills/{hidden}.md"
    assert await refused(tools, "Read", {"path": missing}) == not_a_file(missing)


async def test_a_skill_not_offered_reads_as_missing_beside_a_workspace(tmp_path: Path) -> None:
    root = with_skills(tmp_path / "tree", "skills: [tone]\nworkspace: true\n")
    (tmp_path / "work").mkdir()
    tools = EngineTools(
        Engine(root, models={}),
        agent="support",
        scope="acme",
        workspace=Workspace(Local(tmp_path / "work"), scope="acme"),
    )
    for tool, verb in (("Read", "reads"), ("Load", "loads")):
        for path, missing in (
            ("skills/close.md", "skills/gone.md"),
            ("../skills/plot", "../skills/gone"),
            ("skills/close.md/x", "skills/gone.md/x"),
        ):
            said = await refused(tools, tool, {"path": path})
            assert said == (
                f"tools/builtins: not a file of your folder: {path}; "
                f'{tool} with where="workspace" {verb} from your workspace'
            )
            assert said == (await refused(tools, tool, {"path": missing})).replace(missing, path)
    assert await refused(tools, "LS", {"path": "../skills/plot"}) == (
        "tools/builtins: not a directory of your folder: ../skills/plot; Shell lists your workspace"
    )


async def test_a_gitignored_skill_keeps_its_reading_beside_one_not_offered(
    tmp_path: Path,
) -> None:
    root = with_skills(tmp_path, "skills: [tone]\ngitignore: true\n")
    write(root / "desks/support/.gitignore", "close.md\n")
    tools = tools_of(root)
    assert await refused(tools, "Read", {"path": "skills/close.md"}) == (
        "tools/builtins: skills/close.md is ignored by a .gitignore; it is not part of the agent's folder"
    )
    assert await refused(tools, "Read", {"path": "../skills/brief.md"}) == (
        not_a_file("../skills/brief.md")
    )


async def test_a_skill_in_a_folder_the_agent_reads_is_any_file_of_that_folder(
    tmp_path: Path,
) -> None:
    root = with_skills(tmp_path, "skills: [tone]\nreads: ['../../other']\n")
    write(root / "other/AGENT.md", agent_md("other"))
    write(root / "other/skills/close.md", "Another agent's close.\n")
    tools = tools_of(root)
    assert await tools.dispatch("Read", {"path": "../../other/skills/close.md"}) == (
        "Another agent's close.\n"
    )
    assert await refused(tools, "Read", {"path": "skills/close.md"}) == not_a_file(
        "skills/close.md"
    )


async def test_the_invoked_skill_is_there_on_its_run(tmp_path: Path) -> None:
    root = with_skills(tmp_path, "skills: [tone]\npublic_skills: [plot]\n")
    invoked = tools_of(root, invoked="plot")
    assert await invoked.dispatch("Read", {"path": "../skills/plot/SKILL.md"}) == PLOT
    assert await invoked.dispatch("Read", {"path": "../skills/plot/run.py"}) == RUN
    assert await invoked.dispatch("LS", {"path": "../skills/plot"}) == "SKILL.md\nrun.py"
    for path in ("skills/close.md", "../skills/brief.md"):
        assert await refused(invoked, "Read", {"path": path}) == not_a_file(path)
    nothing = tools_of(root)
    for path in ("../skills/plot/run.py", "skills/close.md"):
        assert await refused(nothing, "Read", {"path": path}) == not_a_file(path)
    for said, skill in (("Draw it.", "plot"), ("/plot Draw it.", None)):
        llm = Scripted(read("../skills/plot/run.py"), text_reply("Done."))
        run = await Engine(root, models={"strong": llm}).run("support", "acme", said, skill=skill)
        assert isinstance(run, Generation)
        [step] = run.steps
        assert step.result == RUN and not step.is_error
    llm = Scripted(read("../skills/plot/run.py"), text_reply("Done."))
    unasked = await Engine(root, models={"strong": llm}).run("support", "acme", "Draw it.")
    assert isinstance(unasked, Generation)
    [step] = unasked.steps
    assert step.is_error and step.result == f"InputError: {not_a_file('../skills/plot/run.py')}"


async def test_a_delegated_run_sees_only_what_is_offered(tmp_path: Path) -> None:
    root = with_skills(tmp_path, "skills: [tone]\n")
    write(root / "lead/AGENT.md", agent_md("lead", extra="agents: [support]\n"))
    llm = Scripted(
        delegating("support", "Close it."),
        read("skills/close.md"),
        text_reply("Done."),
        text_reply("Done."),
    )
    run = await Engine(root, models={"strong": llm}).run("lead", "acme", "Go.")
    assert isinstance(run, Generation)
    [child] = run.children
    assert isinstance(child, Generation) and child.skill is None
    [step] = child.steps
    assert step.is_error and step.result == f"InputError: {not_a_file('skills/close.md')}"


async def test_a_routed_run_sees_only_what_is_offered(tmp_path: Path) -> None:
    root = with_skills(tmp_path, "skills: [tone]\n")
    write(
        root / "frontdesk/AGENT.md",
        "---\nname: frontdesk\nkind: decision\nmodel: judge\nroute:\n  by: intent\n  to:\n    help: support\n---\n",
    )
    write(
        root / "frontdesk/questions/intent.json",
        '{"type": "choice", "instructions": "Which?", "options": {"help": null, "other": null}}',
    )
    llm = Scripted(read("../skills/brief.md"), text_reply("Done."))
    judge = Fixed(
        {"intent": ChoiceAnswer(choice="help", probabilities={"help": 0.9, "other": 0.1})}
    )
    run = await Engine(root, models={"strong": llm, "judge": judge}).run(
        "frontdesk", "acme", "Help."
    )
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    [step] = run.routed_to.steps
    assert step.is_error and step.result == f"InputError: {not_a_file('../skills/brief.md')}"


def test_a_skill_named_in_preload_is_sent_though_the_model_is_not_offered_it(
    tmp_path: Path,
) -> None:
    spec = load_tree(with_skills(tmp_path, "skills: [tone]\npreload: ['./skills/close.md']\n"))[
        "support"
    ]
    assert "close" not in spec.offered
    assert spec.preload == [Text(text=f"<<<skills/close.md>>>\n{CLOSE}")]


def test_a_skill_skills_leaves_out_stays_in_the_fingerprint(tmp_path: Path) -> None:
    root = with_skills(tmp_path, "skills: [brief]\n")
    before = load_tree(root)["support"].fingerprint
    write(root / "desks/support/skills/close.md", "Close it twice.\n")
    own = load_tree(root)["support"].fingerprint
    assert own != before
    write(root / "skills/tone.md", TONE + "More.\n")
    assert load_tree(root)["support"].fingerprint not in (before, own)


EMPTY = {
    "tools": "it is empty; write tools: all to keep every tool, tools: none to keep none, or the tools to keep, such as tools: [lookup_order]",
    "skills": "it is empty; write skills: all to keep every skill, skills: none to keep none, or the skills to keep, such as skills: [escalate]",
    "public_skills": "it is empty; write public_skills: none so no one can call a skill, public_skills: all so people may call every skill the agent keeps, or the skills people may call, such as public_skills: [escalate]",
}
KEYS = ["tools", "skills", "public_skills"]


def seen(tmp_path: Path, line: str) -> str:
    with_skills(tmp_path, line)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    return str(caught.value)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", ["[]", ""], ids=["empty-list", "no-value"])
def test_an_empty_list_or_no_value_is_refused(tmp_path: Path, key: str, value: str) -> None:
    line = f"{key}: {value}".rstrip() + "\n"
    assert seen(tmp_path, line) == (
        f"frontmatter: invalid agent {tmp_path / 'desks/support/AGENT.md'}; {key}: {EMPTY[key]}"
    )


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize(
    ("value", "got"),
    [
        ("brief", "'brief'"),
        ("true", "True"),
        ("3", "3"),
        ("1.5", "1.5"),
        ("[1]", "[1]"),
        ("[1.5]", "[1.5]"),
        ("[a, 2]", "['a', 2]"),
    ],
)
def test_any_other_word_is_refused(tmp_path: Path, key: str, value: str, got: str) -> None:
    assert seen(tmp_path, f"{key}: {value}\n") == (
        f"frontmatter: invalid agent {tmp_path / 'desks/support/AGENT.md'}; "
        f"{key}: write all, none or a list of names, got {got}"
    )


@pytest.mark.parametrize("key", ["tools", "skills"])
def test_all_equals_leaving_the_line_out(tmp_path: Path, key: str) -> None:
    left_out = tmp_path / "out"
    written = tmp_path / "all"
    before = load_tree(with_skills(left_out, ""))["support"]
    after = load_tree(with_skills(written, f"{key}: all\n"))["support"]
    assert after.keep == before.keep and after.offered == before.offered
    assert after.public_skills == before.public_skills
    a = Engine(left_out, models={}).agents["support"]
    b = Engine(written, models={}).agents["support"]
    assert a.model_dump() == b.model_dump()


def test_none_keeps_nothing(tmp_path: Path) -> None:
    spec = load_tree(with_skills(tmp_path, "tools: none\nskills: none\npublic_skills: none\n"))
    assert spec["support"].keep == [] and spec["support"].offered == frozenset()
    assert spec["support"].public_skills == ()


def test_a_word_inside_a_list_is_a_name(tmp_path: Path) -> None:
    root = with_skills(tmp_path, "skills: [all]\n")
    write(root / "desks/support/skills/all.md", "Everything.\n")
    assert load_tree(root)["support"].offered == {"all"}
    assert load_tree(root)["support"].keep is None
    write(root / "desks/support/AGENT.md", agent_md("support", extra="tools: [none]\n"))
    assert load_tree(root)["support"].keep == ["none"]


def test_a_none_in_a_list_with_no_such_skill_is_the_unknown_skill_refusal(
    tmp_path: Path,
) -> None:
    assert seen(tmp_path, "skills: [none]\n") == (
        f"tree: {tmp_path / 'desks/support'}/AGENT.md keeps 'none' in skills:, a skill agent "
        "'support' does not have; its skills are brief, close, plot, tone"
    )


def test_public_skills_all_is_what_the_model_is_offered(tmp_path: Path) -> None:
    every = load_tree(with_skills(tmp_path / "every", "public_skills: all\n"))["support"]
    assert every.public_skills == ("brief", "close", "plot", "tone")
    root = with_skills(tmp_path / "kept", "skills: [brief, tone]\npublic_skills: all\n")
    write(root / "desks/support/skills/my skill.md", "Do it.\n")
    kept = load_tree(root)["support"]
    assert kept.public_skills == ("brief", "tone")
    assert set(kept.skills) == {"brief", "close", "my skill", "plot", "tone"}


def test_public_skills_all_refuses_a_kept_skill_that_cannot_be_typed(tmp_path: Path) -> None:
    root = with_skills(tmp_path, "public_skills: all\n")
    write(root / "desks/support/skills/my skill.md", "Do it.\n")
    with pytest.raises(ConfigurationError) as caught:
        load_tree(root)
    assert str(caught.value) == (
        f"tree: {root / 'desks/support'}/AGENT.md makes 'my skill' public through "
        "public_skills: all, and no one can type /my skill; rename the skill to one word, "
        "or list the skills people may call"
    )


@pytest.mark.parametrize(
    ("line", "said"),
    [
        ("skills: all\n", "cannot declare skills:; skills are for generation agents"),
        ("skills: none\n", "cannot declare skills:; skills are for generation agents"),
        (
            "public_skills: none\n",
            "cannot declare public_skills:; skills are for generation agents",
        ),
        (
            "tools: all\n",
            "cannot list tools:; tools are for generation agents, so give them to one it routes to",
        ),
    ],
)
def test_a_word_on_a_decision_agent_is_refused(tmp_path: Path, line: str, said: str) -> None:
    shared(tmp_path, **{"desks/triage/AGENT.md": agent_md("triage", "decision", line)})
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == f"tree: decision agent 'triage' {said}"
