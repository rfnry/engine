import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import AgentInfo, Engine, Local, Tool, Tools, Workspace
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Reply, ToolCall
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
        "notes": AgentInfo(
            name="notes",
            kind="generation",
            description="",
            agents=[],
            reads=[],
            tasks={},
            skills={},
        )
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
    await engine.run(
        "notes", "acme", "x", workspace=Workspace(Local(tmp_path / "work"), scope="acme")
    )
    first, second = ([t.name for t in call[2]] for call in llm.calls)
    assert "ping" in first and "Shell" not in first
    assert {"Shell", "Write", "Edit"} <= set(second)


async def test_its_workspace_tools_keep_the_workspace_timeout_over_the_engine_s(
    tmp_path: Path,
) -> None:
    (tmp_path / "work").mkdir()
    shell = ToolCall(id="s", name="Shell", args={"command": "sleep 0.5; echo done"})
    llm = Scripted(Reply(tool_calls=[shell], stop_reason="tool_use"), text_reply("ok"))
    engine = Engine(notes(tmp_path / "notes"), generation=llm, tool_timeout=0.1)
    work = Workspace(Local(tmp_path / "work"), scope="acme")
    run = await engine.run("notes", "acme", "x", workspace=work)
    assert isinstance(run, Generation) and not run.steps[0].is_error
    assert json.loads(str(run.steps[0].result))["stdout"] == "done\n"


async def test_it_runs_as_a_tool_and_reads_its_tools_folder_as_content(tmp_path: Path) -> None:
    read = ToolCall(id="r1", name="Read", args={"path": "tools/hammer.md"})
    llm = Scripted(Reply(tool_calls=[read], stop_reason="tool_use"), text_reply("Hit it."))
    engine = Engine(notes(tmp_path / "notes"), generation=llm)
    tool = engine.tool("notes", scope="acme", context="You answer from the notes.")
    assert await tool.dispatch("notes", {"input": "How do I hammer?"}) == "Hit it."
    assert llm.calls[0][0].endswith("<<<CONTEXT>>>\nYou answer from the notes.")
    result = llm.calls[1][1][-1]
    assert result.role == "tool" and not result.is_error
    assert result.content == "---\ndescription: How to hammer\n---\nHit it.\n"


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


def test_name_names_a_root_agent(tmp_path: Path) -> None:
    (tmp_path / "AGENT.md").write_text("---\nkind: generation\n---\nHi\n")
    assert list(Engine(tmp_path, generation=Scripted(), name="writer").agents) == ["writer"]
    assert list(Engine(tmp_path, generation=Scripted()).agents) == [tmp_path.name]


def test_name_on_a_root_that_is_only_a_tree_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text("---\nkind: generation\n---\nHi\n")
    with pytest.raises(
        ConfigurationError,
        match=r"^tree: name= names the root's own agent, and .* has no AGENT\.md, only agents in its folders; drop name=",
    ):
        Engine(tmp_path, generation=Scripted(), name="writer")


def test_an_empty_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="^engine: name must name one folder, not blank, hidden, padded or with slashes; got ''$",
    ):
        Engine(notes(tmp_path / "notes"), generation=Scripted(), name="")
