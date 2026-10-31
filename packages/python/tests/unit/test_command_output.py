import asyncio
import json
import sys
from pathlib import Path

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.models import ToolCall
from rfnry_engine.records import CommandOutput, Event, EventHandler, Generation, ToolResultReady
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Scripted, text_reply
from .test_engine_workspace import make_tree, use


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def python(root: Path, source: str) -> ToolCall:
    (root / "script.py").write_text(source, encoding="utf-8")
    return ToolCall(id="1", name="Shell", args={"command": f"{sys.executable} script.py"})


async def go(
    tmp_path: Path, root: Path, call: ToolCall, handler: EventHandler, work: Workspace | None = None
) -> Generation:
    llm = Scripted(use(call), text_reply("ok"))
    engine = Engine(make_tree(tmp_path), models={"strong": llm})
    run = await engine.run(
        "coder",
        "acme",
        "x",
        workspace=work or Workspace(Local(root), scope="acme"),
        on_event=handler,
    )
    assert isinstance(run, Generation)
    return run


def texts(events: list[Event], stream: str) -> str:
    return "".join(e.text for e in events if isinstance(e, CommandOutput) and e.stream == stream)


async def test_a_command_s_output_arrives_while_it_runs(tmp_path: Path, root: Path) -> None:
    source = (
        "import os, time\n"
        "print('ready', flush=True)\n"
        "for _ in range(250):\n"
        "    if os.path.exists('go'):\n"
        "        print('done')\n"
        "        break\n"
        "    time.sleep(0.02)\n"
        "else:\n"
        "    print('late')\n"
    )

    async def handler(event: Event) -> None:
        if isinstance(event, CommandOutput) and "ready" in event.text:
            (root / "go").write_text("", encoding="utf-8")

    run = await go(tmp_path, root, python(root, source), handler)
    assert json.loads(str(run.steps[0].result))["stdout"] == "ready\ndone\n"


async def test_every_byte_arrives_once_in_order_before_the_shell_s_result(
    tmp_path: Path, root: Path
) -> None:
    source = (
        "import sys, time\n"
        "print('a', flush=True)\n"
        "time.sleep(0.05)\n"
        "sys.stderr.write('b\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(0.05)\n"
        "print('c', flush=True)\n"
    )
    events: list[Event] = []

    async def handler(event: Event) -> None:
        events.append(event)

    run = await go(tmp_path, root, python(root, source), handler)
    step = json.loads(str(run.steps[0].result))
    assert texts(events, "stdout") == step["stdout"]
    assert texts(events, "stderr") == step["stderr"]
    shown = [e for e in events if isinstance(e, CommandOutput)]
    assert [e.stream for e in shown] == ["stdout", "stderr", "stdout"]
    done = next(
        i for i, e in enumerate(events) if isinstance(e, ToolResultReady) and e.call_id == "1"
    )
    for e in shown:
        assert (e.call_id, e.run_id, e.agent) == ("1", run.run_id, "coder")
        assert events.index(e) < done


async def test_a_character_split_across_writes_arrives_whole(tmp_path: Path, root: Path) -> None:
    source = (
        "import sys, time\n"
        "sys.stdout.buffer.write(b'\\xc3')\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(0.05)\n"
        "sys.stdout.buffer.write(b'\\xa9\\n')\n"
        "sys.stdout.buffer.flush()\n"
    )
    events: list[Event] = []

    async def handler(event: Event) -> None:
        events.append(event)

    await go(tmp_path, root, python(root, source), handler)
    assert texts(events, "stdout") == "é\n"
    assert not any("\ufffd" in e.text for e in events if isinstance(e, CommandOutput))


async def test_a_slow_handler_neither_slows_the_command_nor_changes_its_result(
    tmp_path: Path, root: Path
) -> None:
    source = "import sys\nsys.stdout.write('x' * 3_000_000)\n"
    events: list[CommandOutput] = []
    slow = True

    async def handler(event: Event) -> None:
        nonlocal slow
        if isinstance(event, CommandOutput):
            events.append(event)
            if slow:
                slow = False
                await asyncio.sleep(1.0)

    work = Workspace(Local(root), scope="acme", timeout=0.5)
    run = await go(tmp_path, root, python(root, source), handler, work)
    step = json.loads(str(run.steps[0].result))
    assert step["timed_out"] is False
    assert "[... " in step["stdout"] and "bytes omitted" in step["stdout"]
    shown = [e.text for e in events]
    notes = [t for t in shown if t.startswith("[... ") and "bytes not shown ...]" in t]
    assert notes
    missed = sum(int(t.split()[1]) for t in notes)
    assert sum(t.count("x") for t in shown) + missed == 3_000_000


async def test_a_handler_that_raises_on_command_output_fails_the_run(
    tmp_path: Path, root: Path
) -> None:
    source = (
        "import time\nprint('hi', flush=True)\ntime.sleep(0.3)\nopen('after.txt', 'w').close()\n"
    )

    async def handler(event: Event) -> None:
        if isinstance(event, CommandOutput):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await go(tmp_path, root, python(root, source), handler)
    assert (root / "after.txt").exists()


async def test_a_background_job_sends_no_command_output(tmp_path: Path, root: Path) -> None:
    call = ToolCall(
        id="1", name="Shell", args={"command": "echo hi; sleep 0.3", "background": True}
    )
    events: list[Event] = []

    async def handler(event: Event) -> None:
        events.append(event)

    await go(tmp_path, root, call, handler)
    assert not any(isinstance(e, CommandOutput) for e in events)
