import asyncio
import json
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Approve, Engine
from rfnry_engine.foreground import current_call
from rfnry_engine.models import ToolCall
from rfnry_engine.records import (
    CallInfo,
    Event,
    Generation,
    Step,
    ToolCallStarted,
    ToolResultReady,
)
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Backend, Local, Process, Workspace

from .fakes import Scripted, text_reply
from .test_engine_workspace import make_tree, use

CLIENT = """
import json, os, socket
def call(tool, **args):
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(os.environ["RFNRY_TOOLS"])
        s.sendall(json.dumps({"key": os.environ["RFNRY_TOOLS_KEY"], "tool": tool, "args": args}).encode() + b"\\n")
        return json.loads(s.makefile(encoding="utf-8").readline())
"""
PY = sys.executable
PREFIX = "tools/workspace: no tool bridge for this command: "


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def script(root: Path, body: str) -> None:
    (root / "client.py").write_text(CLIENT, encoding="utf-8")
    (root / "script.py").write_text(f"from client import call\n{body}", encoding="utf-8")


def shell(command: str = f"{PY} script.py", id: str = "1") -> ToolCall:
    return ToolCall(id=id, name="Shell", args={"command": command})


def result(step: Step) -> dict[str, Any]:
    assert step.tool == "Shell" and isinstance(step.result, str), step
    found = json.loads(step.result)
    assert isinstance(found, dict)
    return found


def described(llm: Scripted) -> str:
    return next(t for t in llm.calls[0][2] if t.name == "Shell").description


async def lookup(args: dict[str, Any]) -> dict[str, Any]:
    return {"value": 2}


async def found(args: dict[str, Any]) -> str:
    return "found"


async def run_coder(
    tmp_path: Path,
    work: Workspace,
    *,
    tools: Sequence[Tool] = (),
    tree: Path | None = None,
    command: str = f"{PY} script.py",
    tool_timeout: float = 30.0,
    approve: Approve | None = None,
) -> tuple[Generation, list[Event], Scripted]:
    events: list[Event] = []

    async def handler(event: Event) -> None:
        events.append(event)

    llm = Scripted(use(shell(command)), text_reply("ok"))
    async with Engine(
        tree or make_tree(tmp_path), models={"strong": llm}, tool_timeout=tool_timeout
    ) as engine:
        run = await engine.run(
            "coder",
            "acme",
            "x",
            workspace=work,
            tools=Tools(tools),
            on_event=handler,
            approve=approve,
        )
    assert isinstance(run, Generation)
    return run, events, llm


async def test_one_shell_call_reaches_a_run_tool_and_a_folder_tool(
    tmp_path: Path, root: Path
) -> None:
    tree = make_tree(tmp_path)
    (tree / "coder" / "tools").mkdir()
    (tree / "coder" / "tools" / "double.json").write_text(
        json.dumps(
            {
                "description": "Double a number",
                "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}},
            }
        ),
        encoding="utf-8",
    )
    (tree / "coder" / "tools" / "double.py").write_text(
        "def run(args, call):\n    return args['n'] * 2\n", encoding="utf-8"
    )
    seen: list[CallInfo | None] = []

    async def looked(args: dict[str, Any]) -> dict[str, Any]:
        seen.append(current_call())
        return {"value": 2}

    script(
        root,
        "import json\n"
        'print(call("double", n=json.loads(call("lookup", topic="x")["result"])["value"])["result"])\n',
    )
    async with Workspace(Local(root), scope="acme") as work:
        run, events, _ = await run_coder(
            tmp_path, work, tools=[Tool("lookup", "Look a topic up", body=looked)], tree=tree
        )
    assert result(run.steps[-1])["stdout"] == "4\n"
    assert [s.tool for s in run.steps] == ["lookup", "double", "Shell"]
    assert [s.by for s in run.steps] == ["1", "1", None]
    started = [e for e in events if isinstance(e, ToolCallStarted)]
    assert [e.call_id for e in started] == ["1", "1.1", "1.2"]
    assert [e.by for e in started] == [None, "1", "1"]
    ready = [e for e in events if isinstance(e, ToolResultReady)]
    assert sorted(e.call_id for e in ready) == ["1", "1.1", "1.2"]
    for event in ready:
        assert event.by == event.step.by
        assert event.step in run.steps
    assert [m.tool_call_id for m in run.conversation if m.role == "tool"] == ["1"]
    [info] = seen
    assert info is not None and info.by == "1" and info.call_id == "1.1"


async def test_three_calls_at_once_run_at_once(tmp_path: Path, root: Path) -> None:
    async def nap(args: dict[str, Any]) -> str:
        await asyncio.sleep(0.5)
        return "rested"

    script(
        root,
        "from concurrent.futures import ThreadPoolExecutor\n"
        "with ThreadPoolExecutor(3) as pool:\n"
        "    print(list(pool.map(lambda _: call('nap'), range(3))))\n",
    )
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(tmp_path, work, tools=[Tool("nap", "Rest", body=nap)])
    naps = [s for s in run.steps if s.tool == "nap"]
    assert len(naps) == 3
    assert all(s.duration_ms >= 500 for s in naps)
    assert run.steps[-1].tool == "Shell" and run.steps[-1].duration_ms < 1200


async def test_approve_sees_a_script_s_call_and_can_refuse_it(tmp_path: Path, root: Path) -> None:
    asked: list[tuple[str, str | None]] = []

    async def approve(agent: str, call: ToolCall) -> bool:
        info = current_call()
        assert info is not None
        asked.append((call.id, info.by))
        return call.name != "lookup"

    script(root, 'import json\nprint(json.dumps(call("lookup", topic="x")))\n')
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(
            tmp_path, work, tools=[Tool("lookup", "Look", body=lookup)], approve=approve
        )
    assert json.loads(result(run.steps[-1])["stdout"]) == {
        "error": "NotApproved: lookup was not approved"
    }
    assert asked == [("1", None), ("1.1", "1")]
    [looked] = [s for s in run.steps if s.tool == "lookup"]
    assert looked.approved is False


async def test_a_call_past_tool_timeout_reads_the_model_s_timeout_text(
    tmp_path: Path, root: Path
) -> None:
    async def slow(args: dict[str, Any]) -> str:
        await asyncio.sleep(1)
        return "late"

    script(root, 'import json\nprint(json.dumps(call("slow")))\n')
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(
            tmp_path, work, tools=[Tool("slow", "Slow", body=slow)], tool_timeout=0.2
        )
    assert json.loads(result(run.steps[-1])["stdout"]) == {
        "error": "TimeoutError: engine: slow timed out after 0.2s"
    }


async def test_a_script_whose_output_is_redirected_or_piped_still_calls(
    tmp_path: Path, root: Path
) -> None:
    script(root, 'print(call("lookup", topic="x")["result"])\n')
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(
            tmp_path,
            work,
            tools=[Tool("lookup", "Look", body=found)],
            command=f"{PY} script.py > out.txt; {PY} script.py | cat",
        )
    assert (root / "out.txt").read_text(encoding="utf-8") == "found\n"
    assert result(run.steps[-1])["stdout"] == "found\n"
    assert [s.tool for s in run.steps] == ["lookup", "lookup", "Shell"]


async def test_the_workspace_s_own_tools_and_delegate_are_not_offered(
    tmp_path: Path, root: Path
) -> None:
    tree = make_tree(
        tmp_path,
        {
            "coder": "name: coder\nkind: generation\nmodel: strong\ndescription: Writes and runs code\nworkspace: true\nagents: [chat]"
        },
    )
    names = ["Shell", "Write", "Delegate", "Output", "Nope"]
    script(root, f"import json\nfor name in {names!r}:\n    print(json.dumps(call(name)))\n")
    async with Workspace(Local(root), scope="acme") as work:
        run, events, _ = await run_coder(tmp_path, work, tree=tree)
    printed = [json.loads(line) for line in result(run.steps[-1])["stdout"].splitlines()]
    assert printed == [{"error": f"InputError: tools: unknown tool {name!r}"} for name in names]
    assert [s.tool for s in run.steps] == ["Shell"]
    assert all(e.by is None for e in events if isinstance(e, ToolCallStarted | ToolResultReady))


class Rootless:
    def __init__(self, local: Local) -> None:
        self._local = local

    async def spawn(self, command: str, env: Any) -> Process:
        return await self._local.spawn(command, env)

    async def read(self, path: str, limit: int) -> bytes:
        return await self._local.read(path, limit)

    async def write(self, path: str, data: bytes) -> None:
        await self._local.write(path, data)


SEEN = "import glob, os\nprint(os.environ.get('RFNRY_TOOLS'), sorted(glob.glob('.rfnry-*')))\n"


@pytest.mark.parametrize("without", ["switch", "root", "run"])
async def test_no_bridge_without_the_switch_a_root_or_a_run(
    tmp_path: Path, root: Path, without: str
) -> None:
    script(root, SEEN)
    tools = [Tool("lookup", "Look", body=lookup)]
    backend: Backend = Rootless(Local(root)) if without == "root" else Local(root)
    async with Workspace(backend, scope="acme", bridge=without != "switch") as work:
        if without == "run":
            async with Engine(make_tree(tmp_path), models={"strong": Scripted()}) as engine:
                surface = EngineTools(
                    engine, agent="coder", scope="acme", workspace=work, tools=Tools(tools)
                )
                shown = await surface.dispatch("Shell", {"command": f"{PY} script.py"})
                description = next(
                    d["description"] for d in surface.definitions() if d["name"] == "Shell"
                )
            assert isinstance(shown, dict)
            out = shown["stdout"]
        else:
            run, _, llm = await run_coder(tmp_path, work, tools=tools)
            out = result(run.steps[-1])["stdout"]
            description = described(llm)
    assert out == "None []\n"
    assert "RFNRY_TOOLS" not in description
    assert list(root.glob(".rfnry-*")) == []


async def test_the_socket_is_gone_after_the_command(tmp_path: Path, root: Path) -> None:
    script(root, "import os\nprint(os.environ['RFNRY_TOOLS'])\n")
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(tmp_path, work)
    path = Path(result(run.steps[-1])["stdout"].strip())
    assert path.parent == root and path.name.startswith(".rfnry-")
    assert not path.exists()
    with socket.socket(socket.AF_UNIX) as s, pytest.raises(OSError):
        s.connect(str(path))
    assert list(root.glob(".rfnry-*")) == []


async def test_the_description_tells_the_bridge_without_a_path(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    prints: list[str] = []
    descriptions: list[str] = []
    for name in ("one", "two"):
        place = tmp_path / name
        place.mkdir()
        llm = Scripted(text_reply("ok"))
        async with (
            Workspace(Local(place), scope="acme") as work,
            Engine(tree, models={"strong": llm}) as engine,
        ):
            run = await engine.run("coder", "acme", "x", workspace=work)
        description = described(llm)
        assert "RFNRY_TOOLS" in description and "RFNRY_TOOLS_KEY" in description
        assert str(place) not in description
        descriptions.append(description)
        prints.append(run.fingerprint.tools)
    assert descriptions[0] == descriptions[1]
    assert prints[0] == prints[1]


async def test_a_root_too_long_for_a_socket_runs_the_command_without_the_bridge(
    tmp_path: Path,
) -> None:
    root = tmp_path / ("d" * 60) / ("e" * 60)
    root.mkdir(parents=True)
    assert len(str(root.resolve()).encode()) > 110
    script(root, "import os\nprint(os.environ.get('RFNRY_TOOLS'))\n")
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(tmp_path, work, tools=[Tool("lookup", "Look", body=lookup)])
    shown = result(run.steps[-1])
    assert shown["exit_code"] == 0
    assert shown["stdout"] == "None\n"
    assert shown["stderr"].endswith("\n")
    assert shown["stderr"].splitlines()[-1].startswith(PREFIX)


async def test_a_large_and_non_ascii_result_reaches_the_script_whole(
    tmp_path: Path, root: Path
) -> None:
    async def big(args: dict[str, Any]) -> str:
        return "é" * 1_500_000

    script(root, 'got = call("big")["result"]\nprint(len(got), set(got))\n')
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(tmp_path, work, tools=[Tool("big", "Big", body=big)])
    assert result(run.steps[-1])["stdout"] == "1500000 {'é'}\n"


async def test_a_silent_connection_does_not_hold_the_shell(tmp_path: Path, root: Path) -> None:
    script(
        root,
        "import os, socket\n"
        "s = socket.socket(socket.AF_UNIX)\n"
        "s.connect(os.environ['RFNRY_TOOLS'])\n",
    )
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(tmp_path, work, tools=[Tool("lookup", "Look", body=lookup)])
    assert result(run.steps[-1])["exit_code"] == 0
    assert [s.tool for s in run.steps] == ["Shell"]
    assert run.steps[-1].duration_ms < 2000


async def test_a_delegated_child_s_command_calls_the_child_s_tools(
    tmp_path: Path, root: Path
) -> None:
    script(root, 'print(call("lookup", topic="x")["result"])\n')
    events: list[Event] = []

    async def handler(event: Event) -> None:
        events.append(event)

    llm = Scripted(
        use(ToolCall(id="d", name="Delegate", args={"agent": "coder", "input": "look"})),
        use(shell(id="s")),
        text_reply("made"),
        text_reply("ok"),
    )
    async with (
        Workspace(Local(root), scope="acme") as work,
        Engine(make_tree(tmp_path), models={"strong": llm}) as engine,
    ):
        run = await engine.run(
            "lead",
            "acme",
            "x",
            workspace=work,
            tools=Tools([Tool("lookup", "Look", body=found)]),
            on_event=handler,
        )
    assert isinstance(run, Generation)
    assert [s.tool for s in run.steps] == ["Delegate"]
    [child] = run.children
    assert isinstance(child, Generation)
    assert [(s.tool, s.by) for s in child.steps] == [("lookup", "s"), ("Shell", None)]
    assert result(child.steps[-1])["stdout"] == "found\n"
    [bridged] = [e for e in events if isinstance(e, ToolCallStarted) and e.tool == "lookup"]
    assert bridged.run_id == child.run_id and bridged.call_id == "s.1" and bridged.by == "s"
