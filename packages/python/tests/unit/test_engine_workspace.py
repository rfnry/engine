import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import ChoiceAnswer, Image, Reply, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Local, Workspace

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
        (root / name / "AGENT.md").write_text(
            f"---\nname: {name}\n{front}\n---\nYou are {name}.\n", encoding="utf-8"
        )
        if "kind: decision" in front:
            (root / name / "questions").mkdir()
            (root / name / "questions" / "intent.md").write_text(INTENT, encoding="utf-8")
    return root


def use(*calls: ToolCall) -> Reply:
    return Reply(text="", tool_calls=list(calls), stop_reason="tool_use")


def names(llm: Scripted, index: int) -> list[str]:
    return [t.name for t in llm.calls[index][2]]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def work(root: Path) -> Workspace:
    return Workspace(Local(root), scope="acme")


async def test_an_agent_without_the_flag_never_sees_the_workspace(
    tmp_path: Path, work: Workspace
) -> None:
    llm = Scripted(text_reply("hi"), text_reply("hi"))
    engine = Engine(make_tree(tmp_path), generation=llm)
    await engine.run("chat", "acme", "x", workspace=work)
    await engine.run("chat", "acme", "x")
    for index in (0, 1):
        assert not {"Shell", "Job", "Jobs", "Write", "Edit"} & set(names(llm, index))


async def test_an_agent_with_the_flag_sees_the_five_tools(tmp_path: Path, work: Workspace) -> None:
    llm = Scripted(text_reply("ok"))
    engine = Engine(make_tree(tmp_path), generation=llm)
    await engine.run("coder", "acme", "x", workspace=work)
    assert {"Shell", "Job", "Jobs", "Write", "Edit"} <= set(names(llm, 0))


async def test_a_flagged_agent_without_a_workspace_is_refused_before_the_model(
    tmp_path: Path,
) -> None:
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


async def test_the_workspace_timeout_stops_a_shell_and_the_run_goes_on(tmp_path: Path) -> None:
    root = tmp_path / "quick"
    root.mkdir()
    llm = Scripted(
        use(
            ToolCall(
                id="1",
                name="Shell",
                args={"command": "echo started; sleep 30 & echo $! > child.pid; wait"},
            )
        ),
        text_reply("gave up"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm, tool_timeout=None)
    run = await engine.run(
        "coder", "acme", "x", workspace=Workspace(Local(root), scope="acme", timeout=0.5)
    )
    assert isinstance(run, Generation) and run.reply == "gave up"
    result = json.loads(str(run.steps[0].result))
    assert (
        not run.steps[0].is_error
        and result["timed_out"] is True
        and result["stdout"] == "started\n"
    )
    pid = int((root / "child.pid").read_text())
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


async def test_the_engine_tool_timeout_leaves_workspace_tools_to_their_own(
    tmp_path: Path, work: Workspace
) -> None:
    llm = Scripted(
        use(ToolCall(id="1", name="Shell", args={"command": "sleep 0.5; echo done"})),
        text_reply("ok"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm, tool_timeout=0.1)
    run = await engine.run("coder", "acme", "x", workspace=work)
    assert isinstance(run, Generation) and not run.steps[0].is_error
    assert json.loads(str(run.steps[0].result))["stdout"] == "done\n"


async def test_delegation_passes_the_workspace_to_a_child_that_declares_it(
    tmp_path: Path, root: Path, work: Workspace
) -> None:
    llm = Scripted(
        use(ToolCall(id="d", name="Delegate", args={"agent": "coder", "input": "make it"})),
        use(ToolCall(id="s", name="Shell", args={"command": "echo hi > made.txt"})),
        text_reply("made"),
        text_reply("ok"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run("lead", "acme", "x", workspace=work)
    assert isinstance(run, Generation) and run.reply == "ok"
    assert (root / "made.txt").read_text() == "hi\n"
    assert "Shell" not in names(llm, 0) and "Shell" in names(llm, 1)


async def test_delegating_to_a_flagged_child_without_a_workspace_is_a_tool_error(
    tmp_path: Path,
) -> None:
    llm = Scripted(
        use(ToolCall(id="d", name="Delegate", args={"agent": "coder", "input": "make it"})),
        text_reply("could not"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run("lead", "acme", "x")
    assert isinstance(run, Generation)
    assert run.steps[0].is_error and "ConfigurationError" in run.steps[0].result


async def test_a_routed_target_receives_the_workspace(
    tmp_path: Path, root: Path, work: Workspace
) -> None:
    llm = Scripted(
        use(ToolCall(id="s", name="Shell", args={"command": "echo routed > r.txt"})),
        text_reply("done"),
    )
    decision = Fixed(
        {"intent": ChoiceAnswer(choice="code", probabilities={"code": 0.9, "talk": 0.1})}
    )
    engine = Engine(make_tree(tmp_path), generation=llm, decision=decision)
    run = await engine.run("desk", "acme", "build it", workspace=work)
    assert (
        isinstance(run, Decision) and run.routed_to is not None and run.routed_to.agent == "coder"
    )
    assert (root / "r.txt").read_text() == "routed\n"


async def test_a_duplicate_shell_is_refused(tmp_path: Path, work: Workspace) -> None:
    async def body(args: dict[str, Any]) -> str:
        return "mine"

    engine = Engine(make_tree(tmp_path), generation=Scripted(text_reply("never")))
    with pytest.raises(ConfigurationError, match="Shell"):
        await engine.run(
            "coder", "acme", "x", tools=Tools([Tool("Shell", body=body)]), workspace=work
        )


async def test_engine_tools_carry_the_workspace(tmp_path: Path, work: Workspace) -> None:
    engine = Engine(make_tree(tmp_path), generation=Scripted())
    assert "Shell" in EngineTools(engine, agent="coder", scope="acme", workspace=work)
    assert "Shell" not in EngineTools(engine, agent="chat", scope="acme", workspace=work)
    with pytest.raises(ConfigurationError, match="workspace"):
        EngineTools(engine, agent="coder", scope="acme")
    tools = EngineTools(engine, agent="coder", scope="acme", workspace=work)
    result = await tools.dispatch("Shell", {"command": "echo via-dispatch"})
    assert isinstance(result, dict) and result["stdout"] == "via-dispatch\n"


async def test_an_agent_loads_the_chart_it_made_in_its_workspace(
    tmp_path: Path, root: Path, work: Workspace
) -> None:
    png = b"\x89PNG\r\n\x1a\nchart"
    (root / "chart.png").write_bytes(png)
    llm = Scripted(
        use(ToolCall(id="1", name="Load", args={"path": "chart.png", "where": "workspace"})),
        text_reply("the line goes up"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run("coder", "acme", "x", workspace=work)
    assert isinstance(run, Generation) and run.steps[0].result == [
        Image(media_type="image/png", data=png)
    ]
    [load] = [t for t in llm.calls[0][2] if t.name == "Load"]
    assert load.input_schema["properties"]["where"]["enum"] == ["folder", "workspace"]


async def appears(path: Path) -> None:
    for _ in range(100):
        if path.exists():
            return
        await asyncio.sleep(0.02)
    pytest.fail(f"{path.name} never appeared")


async def test_a_job_started_in_one_run_is_read_and_stopped_in_the_next(
    tmp_path: Path, root: Path, work: Workspace
) -> None:
    start = {"command": "echo started; touch ready; sleep 30", "background": True}
    llm = Scripted(
        use(ToolCall(id="1", name="Shell", args=start)),
        text_reply("started it"),
        use(ToolCall(id="2", name="Job", args={"id": "j1", "stop": True})),
        text_reply("stopped it"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    first = await engine.run("coder", "acme", "start it", workspace=work)
    assert isinstance(first, Generation)
    assert json.loads(str(first.steps[0].result)) == {"job": "j1"}
    await appears(root / "ready")
    second = await engine.run("coder", "acme", "stop it", workspace=work)
    assert isinstance(second, Generation)
    assert json.loads(str(second.steps[0].result)) == {
        "output": "started\n",
        "running": False,
        "exit_code": -15,
    }


async def test_approve_sees_a_background_command(tmp_path: Path, work: Workspace) -> None:
    seen: list[ToolCall] = []

    async def approve(agent: str, call: ToolCall) -> bool:
        seen.append(call)
        return False

    llm = Scripted(
        use(ToolCall(id="1", name="Shell", args={"command": "serve", "background": True})),
        text_reply("refused"),
    )
    engine = Engine(make_tree(tmp_path), generation=llm)
    run = await engine.run("coder", "acme", "x", workspace=work, approve=approve)
    assert seen[0].args == {"command": "serve", "background": True}
    assert isinstance(run, Generation) and run.steps[0].approved is False
    jobs = next(t for t in work.tools() if t.name == "Jobs")
    assert await jobs.call({}) == {"jobs": []}
