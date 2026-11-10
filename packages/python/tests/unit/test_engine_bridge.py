import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Approve, Engine
from rfnry_engine.foreground import current_call
from rfnry_engine.models import Message, Reply, ToolCall, ToolSpec
from rfnry_engine.records import (
    CallInfo,
    CommandOutput,
    Event,
    Generation,
    Step,
    ToolCallStarted,
    ToolResultReady,
)
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.stop import Stop
from rfnry_engine.tools import bridge as bridging
from rfnry_engine.tools.bridge import CALLS, Bridge
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Backend, Local, Process, Workspace

from .fakes import Ledger, Scripted, text_reply
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
UNTIL = """
import os, threading, time
def until(name):
    deadline = time.monotonic() + 10
    while not os.path.exists(name) and time.monotonic() < deadline:
        time.sleep(0.01)
"""


@pytest.fixture
def root(short_root: Path) -> Path:
    return short_root


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
    stop: Stop | None = None,
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
            stop=stop,
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
    starts: list[float] = []
    ends: list[float] = []

    async def nap(args: dict[str, Any]) -> str:
        starts.append(time.monotonic())
        await asyncio.sleep(0.5)
        ends.append(time.monotonic())
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
    assert len(starts) == 3 and max(starts) < min(ends)


PAIR = """
import json, os, socket, sys
mine, other = sys.argv[1], sys.argv[2]
with open(mine + ".tmp", "w") as f:
    f.write(os.environ["RFNRY_TOOLS_KEY"])
os.replace(mine + ".tmp", mine + ".key")
until(other + ".key")
with socket.socket(socket.AF_UNIX) as s, open(other + ".key") as f:
    s.connect(os.environ["RFNRY_TOOLS"])
    s.sendall(json.dumps({"key": f.read(), "tool": "lookup", "args": {}}).encode() + b"\\n")
    refused = json.loads(s.makefile(encoding="utf-8").readline())
print(json.dumps([refused, call("lookup")]))
open(mine + ".done", "w").close()
until(other + ".done")
"""


async def test_two_commands_at_once_refuse_each_other_s_key(tmp_path: Path, root: Path) -> None:
    script(root, UNTIL + PAIR)
    llm = Scripted(
        use(shell(f"{PY} script.py a b", "1"), shell(f"{PY} script.py b a", "2")),
        text_reply("ok"),
    )
    async with (
        asyncio.timeout(30),
        Workspace(Local(root), scope="acme") as work,
        Engine(make_tree(tmp_path), models={"strong": llm}) as engine,
    ):
        run = await engine.run(
            "coder", "acme", "x", workspace=work, tools=Tools([Tool("lookup", "Look", body=found)])
        )
    assert isinstance(run, Generation)
    shells = [s for s in run.steps if s.tool == "Shell"]
    assert [json.loads(result(s)["stdout"]) for s in shells] == [
        [{"error": "tools/bridge: the key is not this command's"}, {"result": "found"}]
    ] * 2
    assert sorted((s.tool, s.by) for s in run.steps if s.tool == "lookup") == [
        ("lookup", "1"),
        ("lookup", "2"),
    ]


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


async def test_a_run_tool_named_output_is_the_model_s_but_not_a_script_s(
    tmp_path: Path, root: Path
) -> None:
    names = ["Output", "Delegate"]
    script(root, f"import json\nprint(json.dumps([call(name) for name in {names!r}]))\n")
    llm = Scripted(use(ToolCall(id="o", name="Output", args={})), use(shell()), text_reply("ok"))
    async with (
        Workspace(Local(root), scope="acme") as work,
        Engine(make_tree(tmp_path), models={"strong": llm}) as engine,
    ):
        run = await engine.run(
            "coder",
            "acme",
            "x",
            workspace=work,
            tools=Tools([Tool(name, "Emit", body=found) for name in names]),
        )
    assert isinstance(run, Generation)
    called, shelled = run.steps
    assert (called.tool, called.result, called.is_error) == ("Output", "found", False)
    assert json.loads(result(shelled)["stdout"]) == [
        {"error": f"InputError: tools: unknown tool {name!r}"} for name in names
    ]


class Rootless:
    def __init__(self, local: Local) -> None:
        self._local = local

    async def spawn(self, command: str, env: Any) -> Process:
        return await self._local.spawn(command, env)

    async def read(self, path: str, limit: int) -> bytes:
        return await self._local.read(path, limit)

    async def write(self, path: str, data: bytes) -> None:
        await self._local.write(path, data)


class Relative(Rootless):
    root = Path("work")


async def test_a_backend_with_a_relative_root_gives_the_command_an_absolute_socket(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(root.parent)
    script(root, 'import os\nprint(os.environ["RFNRY_TOOLS"], call("lookup")["result"])\n')
    async with Workspace(Relative(Local(root)), scope="acme") as work:
        run, _, _ = await run_coder(tmp_path, work, tools=[Tool("lookup", "Look", body=found)])
    path, said = result(run.steps[-1])["stdout"].split()
    assert Path(path).is_absolute() and Path(path).parent == root.resolve()
    assert said == "found"


class Linked(Rootless):
    def __init__(self, local: Local, root: Path) -> None:
        super().__init__(local)
        self.root = root


async def test_a_backend_s_symlinked_root_keeps_the_socket_on_the_path_it_gave(
    tmp_path: Path, root: Path
) -> None:
    link = root.parent / "link"
    link.symlink_to(root, target_is_directory=True)
    script(root, 'import os\nprint(os.environ["RFNRY_TOOLS"], call("lookup")["result"])\n')
    async with Workspace(Linked(Local(root), link), scope="acme") as work:
        run, _, _ = await run_coder(tmp_path, work, tools=[Tool("lookup", "Look", body=found)])
    path, said = result(run.steps[-1])["stdout"].split()
    assert Path(path).parent == link
    assert said == "found"


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


async def test_a_shell_inside_another_tool_call_gets_no_variables_and_no_bridge(
    tmp_path: Path, root: Path
) -> None:
    (root / "script.py").write_text(
        "import glob, os\n"
        "print(sorted(k for k in os.environ if k.startswith('RFNRY_')), len(glob.glob('.rfnry-*')))\n",
        encoding="utf-8",
    )
    tree = make_tree(tmp_path)
    llm = Scripted(use(ToolCall(id="n", name="nested", args={})), use(shell()), text_reply("ok"))
    async with (
        Workspace(Local(root), scope="acme", tree="/srv/agents") as work,
        Engine(tree, models={"strong": llm}) as engine,
    ):

        async def nested(args: dict[str, Any]) -> Any:
            surface = EngineTools(engine, agent="coder", scope="acme", workspace=work)
            return await surface.dispatch("Shell", {"command": f"{PY} script.py"})

        run = await engine.run(
            "coder",
            "acme",
            "x",
            workspace=work,
            tools=Tools([Tool("nested", "Run a command from code", body=nested)]),
        )
    assert isinstance(run, Generation)
    inner, outer = run.steps
    assert inner.tool == "nested" and isinstance(inner.result, str)
    assert json.loads(inner.result)["stdout"] == "[] 0\n"
    assert result(outer)["stdout"] == (
        "['RFNRY_AGENT', 'RFNRY_TOOLS', 'RFNRY_TOOLS_KEY', 'RFNRY_TREE'] 1\n"
    )


async def test_a_run_with_no_tool_to_offer_gets_no_bridge(tmp_path: Path, root: Path) -> None:
    tree = make_tree(
        tmp_path,
        {
            "coder": "name: coder\nkind: generation\nmodel: strong\ndescription: Writes and runs code\nworkspace: true\ntools: none"
        },
    )
    (tree / "coder" / "tools").mkdir()
    (tree / "coder" / "tools" / "double.json").write_text(
        json.dumps({"description": "Double a number", "input_schema": {"type": "object"}}),
        encoding="utf-8",
    )
    (tree / "coder" / "tools" / "double.py").write_text(
        "def run(args, call):\n    return 2\n", encoding="utf-8"
    )
    script(root, SEEN)
    async with Workspace(Local(root), scope="acme") as work:
        run, _, llm = await run_coder(tmp_path, work, tree=tree)
    assert [t.name for t in llm.calls[0][2]] == ["Shell", "Job", "Jobs", "Write", "Edit"]
    assert result(run.steps[-1])["stdout"] == "None []\n"
    assert "RFNRY_TOOLS" not in described(llm)


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


async def test_a_file_the_command_put_at_the_socket_s_path_is_kept(
    tmp_path: Path, root: Path
) -> None:
    script(
        root,
        "import os\np = os.environ['RFNRY_TOOLS']\nos.unlink(p)\n"
        "with open(p, 'w') as f:\n    f.write('mine')\nprint(p)\n",
    )
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(tmp_path, work)
    path = Path(result(run.steps[-1])["stdout"].strip())
    assert path.parent == root and path.name.startswith(".rfnry-")
    assert path.read_text(encoding="utf-8") == "mine"


async def test_a_command_that_makes_the_root_read_only_reads_its_output(
    tmp_path: Path, root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    try:
        with caplog.at_level(logging.DEBUG):
            async with Workspace(Local(root), scope="acme") as work:
                run, _, llm = await run_coder(
                    tmp_path, work, command='chmod a-w . && echo "protected $RFNRY_TOOLS"'
                )
    finally:
        root.chmod(0o755)
    assert "RFNRY_TOOLS" in described(llm)
    assert not run.steps[-1].is_error
    shown = result(run.steps[-1])
    said, path = shown.pop("stdout").split()
    assert said == "protected" and Path(path).parent == root
    assert shown == {"exit_code": 0, "stderr": "", "timed_out": False}
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


@pytest.mark.parametrize(
    "swap", ["os.unlink(p)\nos.mkdir(p)\n", "os.chmod('.', 0o555)\n"], ids=["dir", "read-only"]
)
async def test_a_call_in_flight_is_recorded_when_its_socket_cannot_be_removed(
    tmp_path: Path, root: Path, swap: str
) -> None:
    ended: list[float] = []

    async def nap(args: dict[str, Any]) -> str:
        (root / "napping").touch()
        try:
            await asyncio.sleep(10)
            return "rested"
        finally:
            ended.append(time.monotonic())

    script(
        root,
        UNTIL
        + 'threading.Thread(target=call, args=("nap",), daemon=True).start()\nuntil("napping")\n'
        + f"p = os.environ['RFNRY_TOOLS']\n{swap}",
    )
    try:
        async with Workspace(Local(root), scope="acme") as work:
            run, events, _ = await run_coder(tmp_path, work, tools=[Tool("nap", "Rest", body=nap)])
            returned, seen = time.monotonic(), len(events)
            await asyncio.sleep(0.5)
    finally:
        root.chmod(0o755)
    assert [(s.tool, s.stopped, s.by) for s in run.steps] == [
        ("nap", True, "1"),
        ("Shell", False, None),
    ]
    assert not run.steps[-1].is_error and result(run.steps[-1])["exit_code"] == 0
    assert len(ended) == 1 and ended[0] <= returned
    assert len(events) == seen


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


async def test_a_bridged_call_cut_when_the_command_exits_keeps_what_approve_answered(
    tmp_path: Path, root: Path
) -> None:
    async def nap(args: dict[str, Any]) -> str:
        (root / "napping").touch()
        await asyncio.sleep(10)
        return "rested"

    async def allow(agent: str, call: ToolCall) -> bool:
        return True

    script(
        root,
        UNTIL
        + 'threading.Thread(target=call, args=("nap",), daemon=True).start()\nuntil("napping")\n',
    )
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(
            tmp_path, work, tools=[Tool("nap", "Rest", body=nap)], approve=allow
        )
    [napped] = [s for s in run.steps if s.tool == "nap"]
    assert napped.stopped is True and napped.approved is True


async def test_a_call_still_running_when_the_command_exits_is_cut(
    tmp_path: Path, root: Path
) -> None:
    async def nap(args: dict[str, Any]) -> str:
        (root / "napping").touch()
        await asyncio.sleep(10)
        return "rested"

    script(
        root,
        UNTIL
        + 'threading.Thread(target=call, args=("nap",), daemon=True).start()\nuntil("napping")\n',
    )
    async with Workspace(Local(root), scope="acme") as work:
        run, events, _ = await run_coder(tmp_path, work, tools=[Tool("nap", "Rest", body=nap)])
    [napped] = [s for s in run.steps if s.tool == "nap"]
    assert napped.stopped is True
    assert napped.result == "stopped by the user"
    assert napped.by == "1"
    assert [e.step for e in events if isinstance(e, ToolResultReady) and e.call_id == "1.1"] == [
        napped
    ]
    assert not run.steps[-1].is_error and result(run.steps[-1])["exit_code"] == 0
    assert run.duration_ms < 5000


async def test_a_stop_ends_the_command_and_cuts_its_calls(tmp_path: Path, root: Path) -> None:
    stop = Stop()

    async def halt(args: dict[str, Any]) -> str:
        stop.now()
        await asyncio.sleep(10)
        return "late"

    script(root, 'call("halt")\n')
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(
            tmp_path, work, tools=[Tool("halt", "Halt", body=halt)], stop=stop
        )
    assert run.stopped == "now"
    assert [(s.tool, s.stopped, s.by) for s in run.steps] == [
        ("halt", True, "1"),
        ("Shell", True, None),
    ]
    assert run.duration_ms < 5000


async def test_the_workspace_timeout_cuts_the_calls_of_its_command(
    tmp_path: Path, root: Path
) -> None:
    async def nap(args: dict[str, Any]) -> str:
        await asyncio.sleep(10)
        return "rested"

    script(root, 'call("nap")\n')
    async with Workspace(Local(root), scope="acme", timeout=1.0) as work:
        run, _, _ = await run_coder(tmp_path, work, tools=[Tool("nap", "Rest", body=nap)])
    assert result(run.steps[-1])["timed_out"] is True
    [napped] = [s for s in run.steps if s.tool == "nap"]
    assert napped.stopped is True and napped.by == "1"


async def test_a_handler_that_raises_on_a_script_s_call_fails_the_run(
    tmp_path: Path, root: Path
) -> None:
    async def handler(event: Event) -> None:
        if isinstance(event, ToolCallStarted) and event.by is not None:
            raise RuntimeError("boom")

    script(
        root, 'import json\nwith open("reply.json", "w") as f:\n    json.dump(call("lookup"), f)\n'
    )
    llm = Scripted(use(shell()), text_reply("ok"))
    async with (
        Workspace(Local(root), scope="acme") as work,
        Engine(make_tree(tmp_path), models={"strong": llm}) as engine,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await engine.run(
                "coder",
                "acme",
                "x",
                workspace=work,
                tools=Tools([Tool("lookup", "Look", body=lookup)]),
                on_event=handler,
            )
    assert json.loads((root / "reply.json").read_text(encoding="utf-8")) == {
        "error": "RuntimeError: boom"
    }


async def test_a_handler_that_raises_on_a_cut_call_s_result_fails_the_run(
    tmp_path: Path, root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def nap(args: dict[str, Any]) -> str:
        (root / "napping").touch()
        await asyncio.sleep(10)
        return "rested"

    async def handler(event: Event) -> None:
        if isinstance(event, ToolResultReady) and event.by is not None:
            raise RuntimeError("boom")

    script(
        root,
        UNTIL
        + 'threading.Thread(target=call, args=("nap",), daemon=True).start()\nuntil("napping")\n',
    )
    llm = Scripted(use(shell()), text_reply("ok"))
    with caplog.at_level(logging.DEBUG):
        async with (
            Workspace(Local(root), scope="acme") as work,
            Engine(make_tree(tmp_path), models={"strong": llm}) as engine,
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await engine.run(
                    "coder",
                    "acme",
                    "x",
                    workspace=work,
                    tools=Tools([Tool("nap", "Rest", body=nap)]),
                    on_event=handler,
                )
    assert len(llm.calls) == 1
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


class Waiting:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.never = asyncio.Event()
        self.cancelled = False

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        (self.root / "asked").touch()
        try:
            await self.never.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("Waiting was answered")


async def test_a_consult_started_by_a_script_s_call_ends_with_the_command(
    tmp_path: Path, root: Path, storage: Storage
) -> None:
    tree = make_tree(
        tmp_path, {"chat": "name: chat\nkind: generation\nmodel: slow\ndescription: Talks"}
    )
    script(
        root,
        UNTIL
        + 'threading.Thread(target=call, args=("chat",), kwargs={"input": "hi"}, daemon=True).start()\n'
        'until("asked")\n',
    )
    llm = Scripted(use(shell()), text_reply("ok"))
    waiting = Waiting(root)
    async with (
        Workspace(Local(root), scope="acme") as work,
        Engine(tree, models={"strong": llm, "slow": waiting}, refinery=Refinery(storage)) as engine,
    ):
        run = await engine.run(
            "coder", "acme", "x", workspace=work, tools=engine.tool("chat", scope="acme")
        )
    assert isinstance(run, Generation)
    assert waiting.cancelled is True
    [chatted] = [s for s in run.steps if s.tool == "chat"]
    assert chatted.stopped is True and chatted.by == "1"
    assert run.duration_ms < 5000
    assert [row.agent for row in await storage.runs("acme")] == ["coder"]


TALK = """
def talk():
    while True:
        print("tick", flush=True)
        time.sleep(0.01)
threading.Thread(target=talk, daemon=True).start()
"""


def billing(tmp_path: Path, root: Path, talk: bool = False) -> Path:
    script(
        root,
        UNTIL
        + (TALK if talk else "")
        + 'threading.Thread(target=call, args=("billing",), kwargs={"input": "hi"}, daemon=True).start()\n'
        'until("asked")\ncall("halt")\n',
    )
    return make_tree(
        tmp_path, {"billing": "name: billing\nkind: generation\nmodel: slow\ndescription: Bills"}
    )


def consults(engine: Engine, stop: Stop) -> Tools:
    async def halt(args: dict[str, Any]) -> str:
        stop.now()
        await asyncio.sleep(10)
        return "late"

    return Tools([engine.tool("billing", scope="acme"), Tool("halt", "Halt", body=halt)])


async def test_a_consult_a_script_started_keeps_its_record_under_a_stop_with_slow_storage(
    tmp_path: Path, root: Path, pool: Any, storage: Storage
) -> None:
    stop = Stop()
    tree = billing(tmp_path, root)
    llm = Scripted(use(shell()), text_reply("ok"))
    waiting = Waiting(root)
    refinery = Refinery(Ledger(pool, "billing", hold=0.3))
    async with (
        asyncio.timeout(10),
        Workspace(Local(root), scope="acme") as work,
        Engine(tree, models={"strong": llm, "slow": waiting}, refinery=refinery) as engine,
    ):
        run = await engine.run(
            "coder", "acme", "x", workspace=work, tools=consults(engine, stop), stop=stop
        )
    assert isinstance(run, Generation) and run.stopped == "now"
    assert waiting.cancelled is True
    [billed] = [s for s in run.steps if s.tool == "billing"]
    assert billed.stopped is True and billed.by == "1"
    [row] = [row for row in await storage.runs("acme") if row.agent == "billing"]
    kept = await refinery.run("acme", row.run_id)
    assert isinstance(kept, Generation) and kept.stopped == "now"


async def test_a_consult_a_script_started_keeps_its_record_through_a_second_stop(
    tmp_path: Path, root: Path, pool: Any, storage: Storage
) -> None:
    stop = Stop()
    ledger = Ledger(pool, "billing", hold=0.3)
    pressed: list[bool] = []

    async def handler(event: Event) -> None:
        if isinstance(event, ToolResultReady) and event.step.tool == "halt":
            await ledger.writing.wait()
            pressed.append(not ledger.written.is_set())
            stop.now()

    tree = billing(tmp_path, root)
    llm = Scripted(use(shell()), text_reply("ok"))
    refinery = Refinery(ledger)
    async with (
        asyncio.timeout(10),
        Workspace(Local(root), scope="acme") as work,
        Engine(tree, models={"strong": llm, "slow": Waiting(root)}, refinery=refinery) as engine,
    ):
        run = await engine.run(
            "coder",
            "acme",
            "x",
            workspace=work,
            tools=consults(engine, stop),
            on_event=handler,
            stop=stop,
        )
    assert pressed == [True]
    assert isinstance(run, Generation) and run.stopped == "now"
    [billed] = [s for s in run.steps if s.tool == "billing"]
    assert billed.stopped is True and billed.by == "1"
    [row] = [row for row in await storage.runs("acme") if row.agent == "billing"]
    kept = await refinery.run("acme", row.run_id)
    assert isinstance(kept, Generation) and kept.stopped == "now"


@pytest.mark.parametrize("stops", [1, 2], ids=["one-stop", "two-stops"])
async def test_a_cancel_of_the_run_cuts_the_record_write_of_a_consult_a_script_started(
    tmp_path: Path, root: Path, pool: Any, storage: Storage, stops: int
) -> None:
    stop = Stop()
    ledger = Ledger(pool, "billing")
    halted = asyncio.Event()

    async def handler(event: Event) -> None:
        if isinstance(event, ToolResultReady) and event.step.tool == "halt":
            await ledger.writing.wait()
            if stops == 2:
                stop.now()
            halted.set()

    tree = billing(tmp_path, root)
    llm = Scripted(use(shell()), text_reply("ok"))
    async with (
        asyncio.timeout(10),
        Workspace(Local(root), scope="acme") as work,
        Engine(
            tree, models={"strong": llm, "slow": Waiting(root)}, refinery=Refinery(ledger)
        ) as engine,
    ):
        task = asyncio.ensure_future(
            engine.run(
                "coder",
                "acme",
                "x",
                workspace=work,
                tools=consults(engine, stop),
                on_event=handler,
                stop=stop,
            )
        )
        try:
            await halted.wait()
            held, _ = await asyncio.wait([task], timeout=0.2)
            task.cancel()
            ended, _ = await asyncio.wait([task], timeout=1)
        finally:
            ledger.released.set()
            await asyncio.wait([task])
    assert held == set() and ended == {task}
    assert task.cancelled() and len(ledger.cut) == 1
    assert await storage.runs("acme") == []


async def test_a_stop_during_a_bridged_consult_with_a_slow_output_handler_keeps_the_record(
    tmp_path: Path, root: Path, pool: Any, storage: Storage
) -> None:
    stop = Stop()
    events: list[Event] = []
    halted: list[float] = []

    async def handler(event: Event) -> None:
        events.append(event)
        if isinstance(event, ToolCallStarted) and event.tool == "halt":
            halted.append(time.monotonic())
        elif isinstance(event, CommandOutput):
            await asyncio.sleep(0.3)

    tree = billing(tmp_path, root, talk=True)
    llm = Scripted(use(shell()), text_reply("ok"))
    refinery = Refinery(Ledger(pool, "billing", hold=0.3))
    async with (
        asyncio.timeout(10),
        Workspace(Local(root), scope="acme") as work,
        Engine(tree, models={"strong": llm, "slow": Waiting(root)}, refinery=refinery) as engine,
    ):
        run = await engine.run(
            "coder",
            "acme",
            "x",
            workspace=work,
            tools=consults(engine, stop),
            on_event=handler,
            stop=stop,
        )
        returned = time.monotonic()
    assert isinstance(run, Generation) and run.stopped == "now"
    assert returned - halted[0] < 3
    [billed] = [s for s in run.steps if s.tool == "billing"]
    assert billed.stopped is True and billed.by == "1"
    [row] = [row for row in await storage.runs("acme") if row.agent == "billing"]
    kept = await refinery.run("acme", row.run_id)
    assert isinstance(kept, Generation) and kept.stopped == "now"
    shown = [i for i, e in enumerate(events) if isinstance(e, CommandOutput) and e.call_id == "1"]
    [ready] = [
        i for i, e in enumerate(events) if isinstance(e, ToolResultReady) and e.call_id == "1"
    ]
    assert shown and max(shown) < ready


async def test_a_cancel_of_the_run_during_a_bridged_consult_with_a_slow_output_handler_stores_nothing(
    tmp_path: Path, root: Path, pool: Any, storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = Stop()
    ledger = Ledger(pool, "billing")
    talking = asyncio.Event()
    entered = asyncio.Event()
    halted = asyncio.Event()
    close = Bridge.close

    async def closing(self: Bridge) -> None:
        entered.set()
        await close(self)

    monkeypatch.setattr(Bridge, "close", closing)

    async def handler(event: Event) -> None:
        if isinstance(event, ToolResultReady) and event.step.tool == "halt":
            halted.set()
        elif isinstance(event, CommandOutput):
            talking.set()
            try:
                await asyncio.sleep(0.3)
            finally:
                talking.clear()

    tree = billing(tmp_path, root, talk=True)
    llm = Scripted(use(shell()), text_reply("ok"))
    async with (
        asyncio.timeout(10),
        Workspace(Local(root), scope="acme") as work,
        Engine(
            tree, models={"strong": llm, "slow": Waiting(root)}, refinery=Refinery(ledger)
        ) as engine,
    ):
        task = asyncio.ensure_future(
            engine.run(
                "coder",
                "acme",
                "x",
                workspace=work,
                tools=consults(engine, stop),
                on_event=handler,
                stop=stop,
            )
        )
        try:
            await ledger.writing.wait()
            await entered.wait()
            await halted.wait()
            await talking.wait()
            task.cancel()
            ended, _ = await asyncio.wait([task], timeout=1)
        finally:
            ledger.released.set()
            await asyncio.wait([task])
    assert ended == {task} and task.cancelled() and len(ledger.cut) == 1
    assert await storage.runs("acme") == []


async def test_a_tool_deadline_cuts_a_consult_a_script_started_whose_record_write_hangs_under_a_stop(
    tmp_path: Path, root: Path, pool: Any, storage: Storage
) -> None:
    stop = Stop()
    ledger = Ledger(pool, "billing")
    halted: list[float] = []

    async def handler(event: Event) -> None:
        if isinstance(event, ToolResultReady) and event.step.tool == "halt":
            halted.append(time.monotonic())

    tree = billing(tmp_path, root)
    llm = Scripted(use(shell()), text_reply("ok"))
    try:
        async with (
            asyncio.timeout(10),
            Workspace(Local(root), scope="acme") as work,
            Engine(
                tree,
                models={"strong": llm, "slow": Waiting(root)},
                refinery=Refinery(ledger),
                tool_timeout=0.2,
            ) as engine,
        ):
            run = await engine.run(
                "coder",
                "acme",
                "x",
                workspace=work,
                tools=consults(engine, stop),
                on_event=handler,
                stop=stop,
            )
            returned = time.monotonic()
    finally:
        ledger.released.set()
    [cut] = ledger.cut
    assert halted[0] < cut < returned < cut + 1
    assert isinstance(run, Generation) and run.stopped == "now"
    [billed] = [s for s in run.steps if s.tool == "billing"]
    assert (billed.result, billed.is_error, billed.stopped, billed.by) == (
        "stopped by the user",
        True,
        True,
        "1",
    )
    assert [row.agent for row in await storage.runs("acme")] == ["coder"]


async def test_a_cancel_of_the_run_reaches_a_bridged_call_waiting_on_its_handler_after_its_command_ended(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    entered = asyncio.Event()
    blocked = asyncio.Event()
    released = asyncio.Event()
    cut: list[bool] = []
    close = Bridge.close

    async def closing(self: Bridge) -> None:
        order.append("close")
        entered.set()
        await close(self)

    monkeypatch.setattr(Bridge, "close", closing)

    async def look(args: dict[str, Any]) -> str:
        (root / "looked").touch()
        return "found"

    async def handler(event: Event) -> None:
        if isinstance(event, ToolResultReady) and event.by == "1":
            order.append("blocked")
            blocked.set()
            try:
                await released.wait()
            except asyncio.CancelledError:
                cut.append(True)
                raise

    script(
        root,
        UNTIL
        + 'threading.Thread(target=call, args=("look",), daemon=True).start()\nuntil("looked")\n',
    )
    llm = Scripted(use(shell()), text_reply("ok"))
    async with (
        asyncio.timeout(10),
        Workspace(Local(root), scope="acme") as work,
        Engine(make_tree(tmp_path), models={"strong": llm}) as engine,
    ):
        task = asyncio.ensure_future(
            engine.run(
                "coder",
                "acme",
                "x",
                workspace=work,
                tools=Tools([Tool("look", "Look", body=look)]),
                on_event=handler,
            )
        )
        try:
            await blocked.wait()
            await entered.wait()
            held, _ = await asyncio.wait([task], timeout=0.2)
            task.cancel()
            ended, _ = await asyncio.wait([task], timeout=1)
        finally:
            released.set()
            await asyncio.wait([task])
    assert order == ["blocked", "close"] and held == set()
    assert ended == {task} and task.cancelled() and cut == [True]


ESCAPE = (
    UNTIL
    + """
import subprocess, sys
def escape(name):
    held = subprocess.Popen([sys.executable, name], start_new_session=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(name + ".pid", "w") as f:
        f.write(str(held.pid))
"""
)
HOLDER = """
import json, os, socket, time
s = socket.socket(socket.AF_UNIX)
s.connect(os.environ["RFNRY_TOOLS"])
s.sendall(json.dumps({"key": os.environ["RFNRY_TOOLS_KEY"], "tool": "stubborn", "args": {}}).encode() + b"\\n")
time.sleep(60)
"""
READER = """
import json, os
from client import call
reply = call("nap")
with open("read.tmp", "w") as f:
    json.dump(reply, f)
os.replace("read.tmp", "read.json")
"""


def end_escaped(root: Path, name: str) -> None:
    with contextlib.suppress(FileNotFoundError, ProcessLookupError):
        os.kill(int((root / f"{name}.pid").read_text(encoding="utf-8")), signal.SIGKILL)


async def test_a_process_that_escaped_the_command_reads_its_cut_call_s_error(
    tmp_path: Path, root: Path
) -> None:
    async def nap(args: dict[str, Any]) -> str:
        (root / "started").touch()
        await asyncio.sleep(10)
        return "rested"

    (root / "reader.py").write_text(READER, encoding="utf-8")
    script(root, ESCAPE + 'escape("reader.py")\nuntil("started")\n')
    try:
        async with Workspace(Local(root), scope="acme") as work:
            run, _, _ = await run_coder(tmp_path, work, tools=[Tool("nap", "Rest", body=nap)])
        async with asyncio.timeout(10):
            while not (root / "read.json").exists():
                await asyncio.sleep(0.01)
    finally:
        end_escaped(root, "reader.py")
    [napped] = [s for s in run.steps if s.tool == "nap"]
    assert napped.stopped is True
    assert json.loads((root / "read.json").read_text(encoding="utf-8")) == {
        "error": "stopped by the user"
    }


async def test_a_reply_a_process_that_escaped_the_command_never_reads_does_not_hold_the_shell(
    tmp_path: Path, root: Path
) -> None:
    async def stubborn(args: dict[str, Any]) -> str:
        (root / "started").touch()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(10)
        return "x" * 5_000_000

    (root / "holder.py").write_text(HOLDER, encoding="utf-8")
    script(root, ESCAPE + 'escape("holder.py")\nuntil("started")\n')
    async with Workspace(Local(root), scope="acme") as work:
        running = asyncio.ensure_future(
            run_coder(tmp_path, work, tools=[Tool("stubborn", "Stubborn", body=stubborn)])
        )
        try:
            await asyncio.wait([running], timeout=10)
        finally:
            end_escaped(root, "holder.py")
        run, _, _ = await running
    assert [s.tool for s in run.steps] == ["stubborn", "Shell"]
    assert result(run.steps[-1])["exit_code"] == 0
    assert run.duration_ms < 5000


STUCK = """
import os, socket, time
s = socket.socket(socket.AF_UNIX)
s.connect(os.environ["RFNRY_TOOLS"])
open("connected", "w").close()
time.sleep(30)
"""


async def test_a_stuck_connection_is_cut_when_its_command_ends(tmp_path: Path, root: Path) -> None:
    (root / "stuck.py").write_text(STUCK, encoding="utf-8")
    script(root, ESCAPE + 'escape("stuck.py")\nuntil("connected")\n')
    async with Workspace(Local(root), scope="acme") as work:
        running = asyncio.ensure_future(
            run_coder(tmp_path, work, tools=[Tool("lookup", "Look", body=lookup)])
        )
        try:
            ended, _ = await asyncio.wait([running], timeout=10)
        finally:
            end_escaped(root, "stuck.py")
        run, _, _ = await running
    assert ended == {running}
    assert (root / "connected").exists()
    assert [s.tool for s in run.steps] == ["Shell"]
    assert result(run.steps[-1])["exit_code"] == 0
    assert run.steps[-1].duration_ms < 3000


async def test_a_call_past_the_cap_is_refused_and_leaves_no_step(
    tmp_path: Path, root: Path
) -> None:
    entered = 0

    async def wait(args: dict[str, Any]) -> str:
        nonlocal entered
        entered += 1
        if entered == CALLS:
            (root / "full").write_text("x", encoding="utf-8")
        async with asyncio.timeout(30):
            while not (root / "go").exists():
                await asyncio.sleep(0.01)
        return "done"

    script(
        root,
        f"import json\n{UNTIL}"
        "from concurrent.futures import ThreadPoolExecutor\n"
        f"with ThreadPoolExecutor({CALLS}) as pool:\n"
        f"    held = [pool.submit(call, 'wait') for _ in range({CALLS})]\n"
        "    until('full')\n"
        "    extra = call('wait')\n"
        "    open('go', 'w').write('x')\n"
        "    replies = [f.result() for f in held]\n"
        "print(json.dumps([extra, replies]))\n",
    )
    async with Workspace(Local(root), scope="acme") as work:
        run, _, _ = await run_coder(tmp_path, work, tools=[Tool("wait", "Wait", body=wait)])
    [shell_step] = [s for s in run.steps if s.tool == "Shell"]
    extra, replies = json.loads(result(shell_step)["stdout"])
    assert extra == {
        "error": f"tools/bridge: at most {CALLS} calls at once; wait for one to finish"
    }
    assert replies == [{"result": "done"}] * CALLS
    bridged = [s for s in run.steps if s.tool == "wait"]
    assert len(bridged) == CALLS
    assert all(s.by == "1" for s in bridged)


def bridged(run: Generation, events: list[Event]) -> tuple[list[Step], list[str]]:
    started = [e.call_id for e in events if isinstance(e, ToolCallStarted | ToolResultReady)]
    return [s for s in run.steps if s.by is not None], started


async def test_a_command_past_its_call_budget_is_refused_and_leaves_no_step(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridging, "COMMAND_CALLS", 3)
    script(root, "import json\nprint(json.dumps([call('lookup') for _ in range(5)]))\n")
    async with asyncio.timeout(30), Workspace(Local(root), scope="acme", timeout=20) as work:
        run, events, _ = await run_coder(
            tmp_path, work, tools=[Tool("lookup", "Look", body=lookup)]
        )
    replies = json.loads(result(run.steps[-1])["stdout"])
    assert [json.loads(r["result"]) for r in replies[:3]] == [{"value": 2}] * 3
    refused = {"error": "tools/bridge: this command made its 3 calls; no more are answered"}
    assert replies[3:] == [refused] * 2
    steps, seen = bridged(run, events)
    assert len(steps) == bridging.COMMAND_CALLS
    assert [s.tool for s in run.steps] == ["lookup"] * 3 + ["Shell"]
    assert sorted(seen) == sorted(["1", "1", "1.1", "1.1", "1.2", "1.2", "1.3", "1.3"])


async def test_a_command_past_its_byte_budget_is_refused_and_leaves_no_step(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridging, "COMMAND_BYTES", 2_500)
    script(
        root,
        "import json\n"
        "print(json.dumps([call('lookup', pad='x' * 1000) for _ in range(3)] + [call('lookup')]))\n",
    )
    async with asyncio.timeout(30), Workspace(Local(root), scope="acme", timeout=20) as work:
        run, events, _ = await run_coder(
            tmp_path, work, tools=[Tool("lookup", "Look", body=lookup)]
        )
    replies = json.loads(result(run.steps[-1])["stdout"])
    assert [json.loads(r["result"]) for r in replies[:2]] == [{"value": 2}] * 2
    refused = "tools/bridge: this command sent its 2500 bytes of calls; no more are answered"
    assert replies[2:] == [{"error": refused}] * 2
    steps, seen = bridged(run, events)
    assert [s.args for s in steps] == [{"pad": "x" * 1000}] * 2
    assert sorted(seen) == sorted(["1", "1", "1.1", "1.1", "1.2", "1.2"])


NEST = """
def nest(levels):
    value = []
    for _ in range(levels - 1):
        value = [value]
    return value
"""


def nest(levels: int) -> list[Any]:
    value: list[Any] = []
    for _ in range(levels - 1):
        value = [value]
    return value


async def test_a_call_whose_args_nest_past_the_depth_is_refused_and_leaves_no_step(
    tmp_path: Path, root: Path, storage: Storage
) -> None:
    script(
        root,
        f"import json\n{NEST}"
        "print(json.dumps([call('lookup', x=nest(127)), call('lookup', x=nest(128))]))\n",
    )
    events: list[Event] = []

    async def handler(event: Event) -> None:
        events.append(event)

    refinery = Refinery(storage)
    llm = Scripted(use(shell()), text_reply("ok"))
    async with (
        asyncio.timeout(30),
        Workspace(Local(root), scope="acme", timeout=20) as work,
        Engine(make_tree(tmp_path), models={"strong": llm}, refinery=refinery) as engine,
    ):
        run = await engine.run(
            "coder",
            "acme",
            "x",
            workspace=work,
            tools=Tools([Tool("lookup", "Look", body=lookup)]),
            on_event=handler,
        )
    assert isinstance(run, Generation)
    kept, refused = json.loads(result(run.steps[-1])["stdout"])
    assert json.loads(kept["result"]) == {"value": 2}
    assert refused == {"error": "tools/bridge: a call's args nest deeper than 128 levels"}
    steps, seen = bridged(run, events)
    assert [s.args for s in steps] == [{"x": nest(127)}]
    assert sorted(seen) == sorted(["1", "1", "1.1", "1.1"])
    read = await refinery.run("acme", run.run_id)
    assert isinstance(read, Generation) and read.steps == run.steps


async def test_the_description_states_a_command_s_budgets(tmp_path: Path, root: Path) -> None:
    llm = Scripted(text_reply("ok"))
    async with (
        Workspace(Local(root), scope="acme") as work,
        Engine(make_tree(tmp_path), models={"strong": llm}) as engine,
    ):
        await engine.run("coder", "acme", "x", workspace=work)
    assert (
        " A command makes at most 10000 calls in all, sending at most 2 MiB of request lines;"
        " past either, each further call is answered with an error, so spread larger work over"
        " several commands." in described(llm)
    )
