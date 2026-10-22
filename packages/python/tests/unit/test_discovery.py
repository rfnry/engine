import logging
import os
from pathlib import Path

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, NotFoundError
from rfnry_engine.models import Text
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tree import _tree_folders, load_tree


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


def team(root: Path) -> Path:
    write(root / "lead/AGENT.md", agent_md("lead", extra="agents: [writer, triage]\n"))
    write(root / "lead/playbook.md", "Be kind.\n")
    write(root / "lead/helpers/notes.md", "Notes.\n")
    write(root / "lead/helpers/writer/AGENT.md", agent_md("writer"))
    write(root / "lead/helpers/writer/voice.md", "Warm.\n")
    write(
        root / "desks/support/triage/AGENT.md", agent_md("triage", kind="decision", model="judge")
    )
    write(root / "desks/support/triage/questions/urgent.md", "---\ntype: check\n---\nUrgent?\n")
    return root


def test_agents_are_found_at_any_depth_in_folders_of_any_name(tmp_path: Path) -> None:
    tree = load_tree(team(tmp_path))
    assert list(tree) == ["lead", "triage", "writer"]
    assert [tree[n].path for n in tree] == ["lead", "desks/support/triage", "lead/helpers/writer"]


async def test_an_agent_inside_another_s_folder_leaves_its_map_and_stays_readable(
    tmp_path: Path,
) -> None:
    lead = load_tree(team(tmp_path))["lead"]
    assert lead.listing.splitlines() == ["helpers/", "  notes.md", "playbook.md"]
    surface = EngineTools(Engine(tmp_path, models={}), agent="lead", scope="acme")
    assert "Warm." in str(await surface.dispatch("Read", {"path": "helpers/writer/voice.md"}))


def test_an_outer_agent_preloads_its_folder_without_its_inner_agents(tmp_path: Path) -> None:
    team(tmp_path)
    write(
        tmp_path / "lead/AGENT.md",
        agent_md("lead", extra="agents: [writer, triage]\npreload: true\n"),
    )
    sent = "".join(p.text for p in load_tree(tmp_path)["lead"].preload if isinstance(p, Text))
    assert "<<<playbook.md>>>" in sent and "<<<helpers/notes.md>>>" in sent
    assert "helpers/writer" not in sent and "Warm." not in sent


def test_an_outer_agent_s_fingerprint_counts_its_inner_agents(tmp_path: Path) -> None:
    before = load_tree(team(tmp_path))["lead"].fingerprint
    write(tmp_path / "lead/helpers/writer/voice.md", "Cold.\n")
    assert load_tree(tmp_path)["lead"].fingerprint != before


def test_hidden_and_declaration_folders_are_never_searched_for_agents(tmp_path: Path) -> None:
    team(tmp_path)
    for place in (".draft/x", "lead/tasks/x", "lead/tools/x", "desks/support/triage/questions/x"):
        write(tmp_path / place / "AGENT.md", agent_md("ghost"))
    assert list(load_tree(tmp_path)) == ["lead", "triage", "writer"]


async def test_a_folder_without_agent_md_is_never_run(tmp_path: Path) -> None:
    engine = Engine(team(tmp_path), models={})
    assert "desks" not in engine.agents
    with pytest.raises(NotFoundError):
        await engine.run("desks", "acme", "x")


def test_a_tree_without_any_agent_md_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "notes.md", "Notes.\n")
    with pytest.raises(ConfigurationError, match="holds no AGENT.md"):
        load_tree(tmp_path)


def test_a_tree_without_any_agent_md_is_told_what_to_write(tmp_path: Path) -> None:
    write(tmp_path / "notes.md", "Notes.\n")
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: {tmp_path} holds no AGENT.md; give each agent a folder with an AGENT.md that"
        " names its name:, kind: and model:"
    )


def test_two_agents_with_one_name_at_different_depths_are_refused_with_both_paths(
    tmp_path: Path,
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("same"))
    write(tmp_path / "b/c/d/AGENT.md", agent_md("same"))
    with pytest.raises(ConfigurationError, match="both name the agent 'same'") as caught:
        load_tree(tmp_path)
    assert str(tmp_path / "a/AGENT.md") in str(caught.value)
    assert str(tmp_path / "b/c/d/AGENT.md") in str(caught.value)


def test_a_linked_agent_folder_at_depth_is_found(tmp_path: Path) -> None:
    write(tmp_path / "elsewhere/payments/AGENT.md", agent_md("billing"))
    (tmp_path / "root/desks").mkdir(parents=True)
    (tmp_path / "root/desks/billing").symlink_to(
        tmp_path / "elsewhere/payments", target_is_directory=True
    )
    assert load_tree(tmp_path / "root")["billing"].path == "desks/billing"


@pytest.mark.parametrize(
    ("link", "target"), [("desks/again", "lead"), ("lead/helpers/loop", "lead")]
)
def test_a_link_back_into_the_tree_is_refused_with_both_paths(
    tmp_path: Path, link: str, target: str
) -> None:
    team(tmp_path)
    (tmp_path / link).symlink_to(tmp_path / target, target_is_directory=True)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(tmp_path / link) in str(caught.value) and str(tmp_path / target) in str(caught.value)


def test_a_link_inside_the_tree_to_a_group_of_agents_is_refused_with_both_paths(
    tmp_path: Path,
) -> None:
    team(tmp_path)
    (tmp_path / "desks/helpers").symlink_to(tmp_path / "lead/helpers", target_is_directory=True)
    with pytest.raises(ConfigurationError, match="holds an AGENT.md") as caught:
        load_tree(tmp_path)
    said = str(caught.value)
    assert str(tmp_path / "desks/helpers") in said and str(tmp_path / "lead/helpers") in said


def test_a_link_inside_the_tree_to_a_folder_that_holds_it_is_refused(tmp_path: Path) -> None:
    team(tmp_path)
    write(tmp_path / "shared/books/a.md", "A.\n")
    (tmp_path / "shared/books/loop").symlink_to(tmp_path / "shared", target_is_directory=True)
    with pytest.raises(ConfigurationError, match="holds the link") as caught:
        load_tree(tmp_path)
    said = str(caught.value)
    assert str(tmp_path / "shared/books/loop") in said and str(tmp_path / "shared") in said


async def test_a_link_inside_the_tree_to_a_folder_without_agents_is_ignored(
    tmp_path: Path,
) -> None:
    team(tmp_path)
    write(tmp_path / "shared/books/a.md", "A.\n")
    write(tmp_path / "coder/AGENT.md", agent_md("coder", extra="reads: [shared/books]\n"))
    (tmp_path / "desks/books").symlink_to(tmp_path / "shared/books", target_is_directory=True)
    folders, _ = _tree_folders(tmp_path)
    assert "desks/books" not in folders
    assert folders["shared/books"] == (tmp_path / "shared/books").resolve()
    assert list(load_tree(tmp_path)) == ["coder", "lead", "triage", "writer"]
    surface = EngineTools(Engine(tmp_path, models={}), agent="coder", scope="acme")
    assert "A." in str(await surface.dispatch("Read", {"path": "../shared/books/a.md"}))


@pytest.mark.parametrize("second", ["elsewhere/payments", "elsewhere"])
def test_an_agent_reached_by_two_links_out_of_the_tree_is_refused_with_both_paths(
    tmp_path: Path, second: str
) -> None:
    write(tmp_path / "elsewhere/payments/AGENT.md", agent_md("billing"))
    root = tmp_path / "root"
    (root / "desks").mkdir(parents=True)
    (root / "desks/billing").symlink_to(tmp_path / "elsewhere/payments", target_is_directory=True)
    (root / "again").symlink_to(tmp_path / second, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="both reach") as caught:
        load_tree(root)
    said = str(caught.value)
    assert str(root / "desks/billing") in said and str(root / "again") in said
    assert str(tmp_path / "elsewhere/payments") in said


def test_a_link_out_of_the_tree_to_a_folder_that_holds_the_tree_is_refused(
    tmp_path: Path,
) -> None:
    root = team(tmp_path / "root")
    (root / "desks/up").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="holds the link") as caught:
        load_tree(root)
    assert str(root / "desks/up") in str(caught.value)


def test_a_folder_reached_by_two_links_out_of_the_tree_without_agents_is_read_twice(
    tmp_path: Path,
) -> None:
    write(tmp_path / "elsewhere/notes/a.md", "A.\n")
    root = team(tmp_path / "root")
    (root / "one").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    (root / "two").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    folders, _ = _tree_folders(root)
    assert folders["one/notes"] == folders["two/notes"] == (tmp_path / "elsewhere/notes").resolve()
    assert list(load_tree(root)) == ["lead", "triage", "writer"]


def test_a_read_folder_two_levels_away_is_spelled_from_the_agent(tmp_path: Path) -> None:
    write(tmp_path / "shared/books/a.md", "A.\n")
    write(tmp_path / "team/coder/AGENT.md", agent_md("coder", extra="reads: [shared/books]\n"))
    assert "../../shared/books/\n  a.md" in load_tree(tmp_path)["coder"].listing


async def test_a_glob_of_a_read_folder_two_levels_away_is_spelled_from_the_agent(
    tmp_path: Path,
) -> None:
    write(tmp_path / "shared/books/a.md", "A.\n")
    write(tmp_path / "team/coder/AGENT.md", agent_md("coder", extra="reads: [shared/books]\n"))
    surface = EngineTools(Engine(tmp_path, models={}), agent="coder", scope="acme")
    found = await surface.dispatch("Glob", {"pattern": "../../shared/books/*.md"})
    assert found == "../../shared/books/a.md"


def test_a_group_of_agents_named_in_reads_shows_without_its_agents(tmp_path: Path) -> None:
    write(tmp_path / "team/notes.md", "Notes.\n")
    write(tmp_path / "team/writer/AGENT.md", agent_md("writer"))
    write(tmp_path / "lead/AGENT.md", agent_md("lead", extra="reads: [team]\n"))
    assert load_tree(tmp_path)["lead"].listing.splitlines() == ["../team/", "  notes.md"]


def test_the_engine_logs_the_names_of_the_agents_it_loaded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="rfnry_engine"):
        Engine(team(tmp_path), models={})
    assert [r.getMessage() for r in caplog.records] == [
        "engine: agents loaded [lead, triage, writer]"
    ]


def test_a_link_out_of_the_tree_to_a_folder_holding_a_link_to_itself_is_refused(
    tmp_path: Path,
) -> None:
    root = team(tmp_path / "root")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside/self").symlink_to(tmp_path / "outside", target_is_directory=True)
    (root / "out").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ConfigurationError, match="holds the link") as caught:
        load_tree(root)
    said = str(caught.value)
    assert str(root / "out/self") in said and str(tmp_path / "outside") in said


def test_two_outside_folders_linking_each_other_are_refused_with_both_paths(
    tmp_path: Path,
) -> None:
    root = team(tmp_path / "root")
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "one/next").symlink_to(tmp_path / "two", target_is_directory=True)
    (tmp_path / "two/next").symlink_to(tmp_path / "one", target_is_directory=True)
    (root / "out").symlink_to(tmp_path / "one", target_is_directory=True)
    with pytest.raises(ConfigurationError, match="holds the link") as caught:
        load_tree(root)
    said = str(caught.value)
    assert str(root / "out/next/next") in said and str(tmp_path / "one") in said


def test_an_unreadable_folder_beside_an_agent_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    team(tmp_path)
    (tmp_path / "lost+found").mkdir()
    blocked = tmp_path / "lost+found"
    real = os.scandir

    def scandir(path: object = ".") -> object:
        if Path(str(path)) == blocked:
            raise PermissionError(13, "denied", str(path))
        return real(path)  # type: ignore[call-overload]

    monkeypatch.setattr(os, "scandir", scandir)
    with pytest.raises(PermissionError):
        os.scandir(blocked)
    assert list(load_tree(tmp_path)) == ["lead", "triage", "writer"]
