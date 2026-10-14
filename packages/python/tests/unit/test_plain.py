from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import AgentInfo, Engine, Tool, Tools, Workspace
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.records import Generation

from .fakes import Fixed, Scripted, text_reply


def notes(root: Path) -> Path:
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "hammer.md").write_text("---\ndescription: How to hammer\n---\nHit it.\n")
    (root / "launch.md").write_text("---\ndescription: The launch plan\n---\nShip Monday.\n")
    return root


async def test_a_root_without_agent_md_is_one_agent_named_for_its_folder(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Drafted."))
    engine = Engine(notes(tmp_path / "notes"), generation=llm)
    assert engine.agents == {
        "notes": AgentInfo(name="notes", kind="generation", description="", agents=[], reads=[])
    }
    run = await engine.run("notes", "acme", "Draft the note.", context="You write release notes.")
    assert isinstance(run, Generation) and run.reply == "Drafted."
    assert run.system == (
        "<<<FILES>>>\nlaunch.md — The launch plan\ntools/\n  hammer.md — How to hammer\n\n"
        "<<<TOOLS>>>\n"
        + "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in llm.calls[0][2])
        + "\n\n<<<CONTEXT>>>\nYou write release notes."
    )
    assert [t.name for t in llm.calls[0][2]] == ["Read", "Grep", "Glob", "LS", "Load"]


async def test_name_names_the_plain_agent(tmp_path: Path) -> None:
    engine = Engine(notes(tmp_path / "notes"), generation=Scripted(), name="writer")
    assert list(engine.agents) == ["writer"]


async def test_its_tools_folder_is_content_and_the_run_s_tools_and_workspace_ride_along(
    tmp_path: Path,
) -> None:
    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    (tmp_path / "work").mkdir()
    llm = Scripted(text_reply("a"), text_reply("b"))
    engine = Engine(notes(tmp_path / "notes"), generation=llm)
    await engine.run("notes", "acme", "x", tools=Tools([Tool("ping", "Ping", body=ping)]))
    await engine.run("notes", "acme", "x", workspace=Workspace(tmp_path / "work", unsandboxed=True))
    first, second = ([t.name for t in call[2]] for call in llm.calls)
    assert "ping" in first and "Shell" not in first
    assert {"Shell", "Write", "Edit"} <= set(second)


def test_a_tool_body_has_nothing_to_serve(tmp_path: Path) -> None:
    async def hammer(args: dict[str, Any]) -> str:
        return "bang"

    with pytest.raises(ConfigurationError, match="declared by no agent"):
        Engine(
            notes(tmp_path / "notes"), generation=Scripted(), tools=[Tool("hammer", body=hammer)]
        )


async def test_a_decision_only_engine_refuses_it(tmp_path: Path) -> None:
    engine = Engine(notes(tmp_path / "notes"), decision=Fixed({}))
    with pytest.raises(ConfigurationError, match="no generation model was configured"):
        await engine.run("notes", "acme", "x")


@pytest.mark.parametrize("single", [True, False])
def test_name_on_a_root_with_agents_is_refused(tmp_path: Path, single: bool) -> None:
    agent = tmp_path if single else tmp_path / "a"
    agent.mkdir(exist_ok=True)
    (agent / "AGENT.md").write_text(f"---\nname: {agent.name}\nkind: generation\n---\nHi\n")
    with pytest.raises(
        ConfigurationError,
        match=r"^tree: name= names a root with no AGENT\.md; .* has agents$",
    ):
        Engine(tmp_path, generation=Scripted(), name="writer")


def test_an_empty_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="^engine: name must not be empty$"):
        Engine(notes(tmp_path / "notes"), generation=Scripted(), name="")
