import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import ChoiceAnswer, Reply, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Workspace

from .fakes import Fixed, Scripted, text_reply

AGENTS = {
    "coder": "kind: generation\ndescription: Writes and runs code\nworkspace: true",
    "chat": "kind: generation\ndescription: Talks",
    "lead": "kind: generation\ndescription: Leads\nagents: [coder, chat]",
    "desk": "kind: decision\ndescription: Sorts\nroute:\n  by: intent\n  to: {code: coder, talk: chat}",
}
INTENT = "---\ntype: choice\noptions:\n  code: Write code\n  talk: Just talk\n---\nWhat is this?\n"


def make_tree(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    root = tmp_path / "agents"
    for name, front in {**AGENTS, **(extra or {})}.items():
        (root / name).mkdir(parents=True)
        (root / name / "AGENT.md").write_text(f"---\nname: {name}\n{front}\n---\nYou are {name}.\n", encoding="utf-8")
        if "kind: decision" in front:
            (root / name / "questions").mkdir()
            (root / name / "questions" / "intent.md").write_text(INTENT, encoding="utf-8")
    return root


def use(*calls: ToolCall) -> Reply:
    return Reply(text="", tool_calls=list(calls), stop_reason="tool_use")


def names(llm: Scripted, index: int) -> list[str]:
    return [t.name for t in llm.calls[index][2]]


@pytest.fixture
def work(tmp_path: Path) -> Workspace:
    root = tmp_path / "work"
    root.mkdir()
    return Workspace(root)


async def test_an_agent_without_the_flag_never_sees_the_workspace(tmp_path: Path, work: Workspace) -> None:
    llm = Scripted(text_reply("hi"), text_reply("hi"))
    engine = Engine(make_tree(tmp_path), generation=llm)
    await engine.run("chat", "acme", "x", workspace=work)
    await engine.run("chat", "acme", "x")
    for index in (0, 1):
        assert not {"Shell", "Write", "Edit"} & set(names(llm, index))


async def test_an_agent_with_the_flag_sees_the_three_tools(tmp_path: Path, work: Workspace) -> None:
    llm = Scripted(text_reply("ok"))
    engine = Engine(make_tree(tmp_path), generation=llm)
    await engine.run("coder", "acme", "x", workspace=work)
    assert {"Shell", "Write", "Edit"} <= set(names(llm, 0))


async def test_a_flagged_agent_without_a_workspace_is_refused_before_the_model(tmp_path: Path) -> None:
    llm = Scripted(text_reply("never"))
    engine = Engine(make_tree(tmp_path), generation=llm)
    with pytest.raises(ConfigurationError, match="workspace"):
        await engine.run("coder", "acme", "x")
    assert llm.calls == []


def test_a_decision_agent_cannot_declare_a_workspace(tmp_path: Path) -> None:
    extra = {"judge2": "kind: decision\ndescription: Judges\nworkspace: true"}
    with pytest.raises(ConfigurationError, match="workspace"):
        Engine(make_tree(tmp_path, extra))


async def test_end_to_end_write_run_edit_run(tmp_path: Path, work: Workspace) -> None:
    python = sys.executable
    llm = Scripted(
        use(ToolCall(id="1", name="Write", args={"path": "main.py", "content": "print(1)\n"})),
        use(ToolCall(id="2", name="Shell", args={"command": f"{python} main.py"})),
        use(ToolCall(id="3", name="Edit", args={"path": "main.py", "old": "1", "new": "2"})),
        use(ToolCall(id="4", name="Shell", args={"command": f"{python} main.py"})),
        text_reply("done"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run("coder", "acme", "print two", workspace=work)
    assert isinstance(run, Generation) and run.reply == "done"
    assert [s.tool for s in run.steps] == ["Write", "Shell", "Edit", "Shell"]
    assert not any(s.is_error for s in run.steps)
    assert json.loads(str(run.steps[1].result))["stdout"] == "1\n"
    assert json.loads(str(run.steps[3].result))["stdout"] == "2\n"
    assert run.steps[0].args == {"path": "main.py", "content": "print(1)\n"}


async def test_the_run_deadline_kills_a_shell(tmp_path: Path, work: Workspace) -> None:
    llm = Scripted(
        use(ToolCall(id="1", name="Shell", args={"command": "sleep 30 & echo $! > child.pid; wait"})),
        text_reply("gave up"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm, tool_timeout=0.5)
    run = await engine.run("coder", "acme", "x", workspace=work)
    assert isinstance(run, Generation) and run.reply == "gave up"
    assert run.steps[0].is_error and "timed out after 0.5s" in run.steps[0].result
    pid = int((work.root / "child.pid").read_text())
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("the shell's child survived the deadline")


async def test_delegation_passes_the_workspace_to_a_child_that_declares_it(tmp_path: Path, work: Workspace) -> None:
    llm = Scripted(
        use(ToolCall(id="d", name="Delegate", args={"agent": "coder", "input": "make it"})),
        use(ToolCall(id="s", name="Shell", args={"command": "echo hi > made.txt"})),
        text_reply("made"),
        text_reply("ok"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run("lead", "acme", "x", workspace=work)
    assert isinstance(run, Generation) and run.reply == "ok"
    assert (work.root / "made.txt").read_text() == "hi\n"
    assert "Shell" not in names(llm, 0) and "Shell" in names(llm, 1)


async def test_delegating_to_a_flagged_child_without_a_workspace_is_a_tool_error(tmp_path: Path) -> None:
    llm = Scripted(
        use(ToolCall(id="d", name="Delegate", args={"agent": "coder", "input": "make it"})),
        text_reply("could not"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run("lead", "acme", "x")
    assert isinstance(run, Generation)
    assert run.steps[0].is_error and "ConfigurationError" in run.steps[0].result


async def test_a_routed_target_receives_the_workspace(tmp_path: Path, work: Workspace) -> None:
    llm = Scripted(use(ToolCall(id="s", name="Shell", args={"command": "echo routed > r.txt"})), text_reply("done"))
    decision = Fixed({"intent": ChoiceAnswer(choice="code", probabilities={"code": 0.9, "talk": 0.1})})
    engine = Engine(make_tree(tmp_path), generation=llm, decision=decision)
    run = await engine.run("desk", "acme", "build it", workspace=work)
    assert isinstance(run, Decision) and run.routed_to is not None and run.routed_to.agent == "coder"
    assert (work.root / "r.txt").read_text() == "routed\n"


async def test_a_duplicate_shell_is_refused(tmp_path: Path, work: Workspace) -> None:
    async def body(args: dict[str, Any]) -> str:
        return "mine"

    engine = Engine(make_tree(tmp_path), generation=Scripted(text_reply("never")))
    with pytest.raises(ConfigurationError, match="Shell"):
        await engine.run("coder", "acme", "x", tools=Tools([Tool("Shell", body=body)]), workspace=work)


async def test_engine_tools_carry_the_workspace(tmp_path: Path, work: Workspace) -> None:
    engine = Engine(make_tree(tmp_path), generation=Scripted())
    assert "Shell" in EngineTools(engine, agent="coder", scope="acme", workspace=work)
    assert "Shell" not in EngineTools(engine, agent="chat", scope="acme", workspace=work)
    with pytest.raises(ConfigurationError, match="workspace"):
        EngineTools(engine, agent="coder", scope="acme")
    tools = EngineTools(engine, agent="coder", scope="acme", workspace=work)
    result = await tools.dispatch("Shell", {"command": "echo via-dispatch"})
    assert isinstance(result, dict) and result["stdout"] == "via-dispatch\n"
