import logging
from pathlib import Path

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, NotFoundError
from rfnry_engine.models import Document, Text
from rfnry_engine.records import Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tree import load_tree

from .fakes import Scripted, text_reply

TONE = "---\ndescription: Root tone\n---\nKeep it warm.\n"
DESK_BRIEF = "---\ndescription: Desk brief\n---\nBrief the desk.\n"
RUN = 'print("plotted")\n'
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


async def test_skills_keeps_the_ones_it_lists(tmp_path: Path) -> None:
    engine = Engine(
        with_skills(tmp_path, "skills: [brief, tone]\n"),
        models={"strong": Scripted(text_reply("Done."))},
    )
    assert engine.agents["support"].skills == {"brief": "Desk brief", "tone": "Root tone"}
    run = await engine.run("support", "acme", "Hello.")
    assert isinstance(run, Generation)
    assert (
        "<<<SKILLS>>>\n../skills/brief.md — Desk brief\n../../skills/tone.md — Root tone\n\n<<<TOOLS>>>"
        in run.system
    )
    with pytest.raises(NotFoundError) as caught:
        await engine.run("support", "acme", "Hello.", skill="close")
    assert (
        str(caught.value)
        == "engine: agent 'support' has no skill 'close'; its skills are brief, tone"
    )


async def test_skills_none_keeps_none(tmp_path: Path) -> None:
    engine = Engine(
        with_skills(tmp_path, "skills: none\n"), models={"strong": Scripted(text_reply("Done."))}
    )
    run = await engine.run("support", "acme", "Hello.")
    assert isinstance(run, Generation)
    assert "<<<SKILLS>>>" not in run.system
    assert engine.agents["support"].skills == {}
    with pytest.raises(NotFoundError) as caught:
        await engine.run("support", "acme", "Hello.", skill="brief")
    assert str(caught.value) == (
        "engine: agent 'support' has no skill 'brief'; it has no skills; a skill is "
        "skills/<name>.md or skills/<name>/SKILL.md in its folder or a folder above it, and "
        "skills: keeps the ones it lists"
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


async def test_a_skill_skills_leaves_out_is_still_a_file_read_reaches(tmp_path: Path) -> None:
    root = with_skills(tmp_path, "skills: [brief]\n")
    engine = Engine(root, models={})
    assert list(engine.agents["support"].skills) == ["brief"]
    tools = EngineTools(engine, agent="support", scope="acme")
    assert await tools.dispatch("Read", {"path": "skills/close.md"}) == "Close the ticket.\n"
    assert await tools.dispatch("Read", {"path": "../../skills/tone.md"}) == TONE


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
    assert after.keep == before.keep and after.skills == before.skills
    assert after.public_skills == before.public_skills
    a = Engine(left_out, models={}).agents["support"]
    b = Engine(written, models={}).agents["support"]
    assert a.model_dump() == b.model_dump()


def test_none_keeps_nothing(tmp_path: Path) -> None:
    spec = load_tree(with_skills(tmp_path, "tools: none\nskills: none\npublic_skills: none\n"))
    assert spec["support"].keep == [] and spec["support"].skills == {}
    assert spec["support"].public_skills == ()


def test_a_word_inside_a_list_is_a_name(tmp_path: Path) -> None:
    root = with_skills(tmp_path, "skills: [all]\n")
    write(root / "desks/support/skills/all.md", "Everything.\n")
    assert list(load_tree(root)["support"].skills) == ["all"]
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


def test_public_skills_all_makes_every_kept_skill_public(tmp_path: Path) -> None:
    every = load_tree(with_skills(tmp_path / "every", "public_skills: all\n"))["support"]
    assert every.public_skills == ("brief", "close", "plot", "tone")
    kept = load_tree(with_skills(tmp_path / "kept", "skills: [brief, tone]\npublic_skills: all\n"))
    assert kept["support"].public_skills == ("brief", "tone")


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
