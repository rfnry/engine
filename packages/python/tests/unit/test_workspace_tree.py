import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Scripted, text_reply
from .test_engine_workspace import AGENTS, appears, make_tree, use
from .test_workspace import Command, Machine, prints
from .trees import provided


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def shell(command: str, **more: object) -> ToolCall:
    return ToolCall(id="1", name="Shell", args={"command": command, **more})


async def stdout_of(tmp_path: Path, work: Workspace, command: str, tree: Path | None = None) -> str:
    llm = Scripted(use(shell(command)), text_reply("ok"))
    engine = Engine(tree or make_tree(tmp_path), providers=provided({"strong": llm, "judge": None}))
    run = await engine.run("coder", "acme", "x", workspace=work)
    assert isinstance(run, Generation)
    assert not run.steps[0].is_error
    return str(json.loads(str(run.steps[0].result))["stdout"])


async def test_local_adds_the_variables_over_its_env(root: Path) -> None:
    local = Local(root, env={"PATH": os.environ["PATH"], "A": "1"})
    process = await local.spawn("echo $A $B", {"B": "2"})
    data = b""
    while chunk := await process.stdout.read(1024):
        data += chunk
    await process.wait()
    assert data == b"1 2\n"


async def test_a_command_of_the_run_sees_the_folder_at_rfnry_tree(
    tmp_path: Path, root: Path
) -> None:
    tree = make_tree(tmp_path)
    work = Workspace(Local(root), scope="acme", tree=str(tree))
    out = await stdout_of(tmp_path, work, 'cat "$RFNRY_TREE/coder.md"; echo "$RFNRY_TREE"', tree)
    assert "You are coder." in out
    assert out.endswith(f"{tree}\n")


async def test_a_command_runs_a_skill_s_script_from_the_folder(tmp_path: Path, root: Path) -> None:
    tree = make_tree(tmp_path)
    skill = tree / "coder" / "skills" / "plot"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: plot\ndescription: Plots\n---\nPlot it.\n", encoding="utf-8"
    )
    (skill / "run.py").write_text('print("plotted")\n', encoding="utf-8")
    work = Workspace(Local(root), scope="acme", tree=str(tree))
    out = await stdout_of(
        tmp_path, work, f'{sys.executable} "$RFNRY_TREE/coder/skills/plot/run.py"', tree
    )
    assert out == "plotted\n"


async def test_a_background_job_of_the_run_sees_the_folder_too(tmp_path: Path, root: Path) -> None:
    tree = make_tree(tmp_path)
    work = Workspace(Local(root), scope="acme", tree=str(tree))
    llm = Scripted(use(shell('echo "$RFNRY_TREE" > where.txt', background=True)), text_reply("ok"))
    engine = Engine(tree, providers=provided({"strong": llm, "judge": None}))
    try:
        await engine.run("coder", "acme", "x", workspace=work)
        await appears(root / "where.txt")
        assert (root / "where.txt").read_text(encoding="utf-8") == f"{tree}\n"
    finally:
        await work.close()


async def test_the_shell_description_names_the_folder_and_what_the_agent_reads(
    tmp_path: Path, root: Path
) -> None:
    tree = make_tree(
        tmp_path,
        {"coder": {**AGENTS["coder"], "files": {"include": ["books"]}}},
        written={"books/a.md": "a"},
    )
    work = Workspace(Local(root), scope="acme", tree=str(tree))
    llm = Scripted(text_reply("ok"))
    engine = Engine(tree, providers=provided({"strong": llm, "judge": None}))
    await engine.run("coder", "acme", "x", workspace=work)
    described = next(t for t in llm.calls[0][2] if t.name == "Shell").description
    assert "$RFNRY_TREE/books" in described
    assert str(tree) not in described


async def test_without_tree_nothing_changes(tmp_path: Path, root: Path) -> None:
    work = Workspace(Local(root), scope="acme")
    llm = Scripted(use(shell("env | grep -c '^RFNRY_TREE' || true")), text_reply("ok"))
    engine = Engine(make_tree(tmp_path), providers=provided({"strong": llm, "judge": None}))
    run = await engine.run("coder", "acme", "x", workspace=work)
    assert isinstance(run, Generation)
    assert "0\\n" in str(run.steps[0].result)
    described = next(t for t in llm.calls[0][2] if t.name == "Shell").description
    assert "RFNRY_TREE" not in described


async def test_a_shell_from_code_outside_a_run_gets_no_variables(
    tmp_path: Path, root: Path
) -> None:
    tree = make_tree(tmp_path)
    work = Workspace(Local(root), scope="acme", tree=str(tree))
    engine = Engine(tree, providers=provided({"strong": Scripted(text_reply("ok")), "judge": None}))
    surface = EngineTools(engine, agent="coder", scope="acme", workspace=work)
    result = await surface.dispatch("Shell", {"command": "echo ${RFNRY_TREE:-none}"})
    assert isinstance(result, dict) and result["stdout"] == "none\n"


def test_tree_must_be_absolute(root: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="tools/workspace: tree must be an absolute path, as commands see it, such as /srv/agents; got 'agents'",
    ):
        Workspace(Local(root), scope="acme", tree="agents")


class Seen(Machine):
    def __init__(self) -> None:
        super().__init__(prints())
        self.envs: list[dict[str, str]] = []

    async def spawn(self, command: str, env: Mapping[str, str]) -> Command:
        self.envs.append(dict(env))
        return await super().spawn(command, env)


async def test_a_tree_with_a_trailing_slash_gives_clean_variables(tmp_path: Path) -> None:
    machine = Seen()
    work = Workspace(machine, scope="acme", tree="/srv/agents/")
    llm = Scripted(use(shell("x")), text_reply("ok"))
    await Engine(make_tree(tmp_path), providers=provided({"strong": llm, "judge": None})).run(
        "coder", "acme", "x", workspace=work
    )
    assert work.tree == "/srv/agents"
    [env] = machine.envs
    assert env["RFNRY_TREE"] == "/srv/agents"


async def test_the_shell_and_background_paths_hand_lent_env_to_spawn(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    machine = Seen()
    work = Workspace(machine, scope="acme", tree=str(tree))
    job = ToolCall(id="2", name="Shell", args={"command": "b", "background": True})
    llm = Scripted(use(shell("a")), use(job), text_reply("ok"))
    try:
        await Engine(tree, providers=provided({"strong": llm, "judge": None})).run(
            "coder", "acme", "x", workspace=work
        )
    finally:
        await work.close()
    assert [ran.line for ran in machine.commands] == ["a", "b"]
    assert machine.envs == [{"RFNRY_TREE": str(tree)}] * 2
