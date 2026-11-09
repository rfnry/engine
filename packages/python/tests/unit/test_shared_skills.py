import logging
from pathlib import Path

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import Document, Text
from rfnry_engine.records import Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tree import load_tree

from .fakes import Scripted, text_reply

TONE = "---\ndescription: Root tone\n---\nKeep it warm.\n"
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
        "desks/skills/brief.md": "---\ndescription: Desk brief\n---\nBrief the desk.\n",
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
    with pytest.raises(InputError) as caught:
        await tools.dispatch("Read", {"path": "../../outside.md"})
    assert str(caught.value) == "jail: path outside the agent directory: ../../outside.md"


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
            "AGENT.md": agent_md("lead", extra="preload: true\n"),
            "desks/notes.md": "Desk notes.\n",
            "desks/support/guide.md": "The guide.\n",
            "reader/AGENT.md": agent_md("reader", extra="reads: [desks]\npreload: true\n"),
            "peer/AGENT.md": agent_md("peer", extra="reads: [desks/support]\npreload: true\n"),
        },
    )
    agents = load_tree(root)

    def preloaded(name: str) -> list[str]:
        return [part.text.split("\n")[0] for part in agents[name].preload if isinstance(part, Text)]

    assert agents["reader"].listing == "../desks/\n  notes.md"
    assert preloaded("reader") == ["<<<../desks/notes.md>>>"]
    assert agents["peer"].listing == "../desks/support/\n  guide.md"
    assert preloaded("peer") == ["<<<../desks/support/guide.md>>>"]
    assert agents["lead"].listing == "desks/\n  notes.md"
    assert preloaded("lead") == ["<<<desks/notes.md>>>"]
    assert sorted(agents["lead"].skills) == ["brief", "tone"]


@pytest.mark.parametrize("named", ["desks/skills", "desks/skills/plot", "skills"])
def test_reads_of_a_skills_folder_is_refused(tmp_path: Path, named: str) -> None:
    root = shared(tmp_path, **{"b/AGENT.md": agent_md("b", extra=f"reads: [{named}]\n")})
    with pytest.raises(ConfigurationError) as caught:
        load_tree(root)
    assert str(caught.value) == f"tree: agent 'b' reads {named!r}, {REFUSED}"


def test_reads_of_an_unknown_folder_offers_no_skills_folder(tmp_path: Path) -> None:
    root = shared(tmp_path, **{"b/AGENT.md": agent_md("b", extra="reads: [nowhere]\n")})
    with pytest.raises(ConfigurationError) as caught:
        load_tree(root)
    assert str(caught.value) == (
        "tree: agent 'b' reads unknown folder 'nowhere'; reads: names a folder by its path from the root, one of: b, billing, desks, desks/support, desks/triage"
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
