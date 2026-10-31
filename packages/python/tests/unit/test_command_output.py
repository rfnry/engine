import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.models import ToolCall
from rfnry_engine.records import CommandOutput, Event, EventHandler, Generation, ToolResultReady
from rfnry_engine.stop import Stop
from rfnry_engine.tools.workspace import BACKLOG, Local, Workspace, _Told

from .fakes import Scripted, text_reply
from .test_engine_workspace import make_tree, use


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


class Kept:
    def __init__(self) -> None:
        self.runs: list[Any] = []

    async def record(self, run: Any) -> None:
        self.runs.append(run)


def python(root: Path, source: str) -> ToolCall:
    (root / "script.py").write_text(source, encoding="utf-8")
    return ToolCall(id="1", name="Shell", args={"command": f"{sys.executable} script.py"})


async def go(
    tmp_path: Path,
    root: Path,
    call: ToolCall,
    handler: EventHandler,
    work: Workspace | None = None,
    kept: Kept | None = None,
    stop: Stop | None = None,
) -> Generation:
    llm = Scripted(use(call), text_reply("ok"))
    engine = Engine(make_tree(tmp_path), models={"strong": llm}, refinery=kept)
    run = await engine.run(
        "coder",
        "acme",
        "x",
        workspace=work or Workspace(Local(root), scope="acme"),
        on_event=handler,
        stop=stop,
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


async def test_every_byte_arrives_once_in_order_per_stream_before_the_shell_s_result(
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
    assert step["stdout"] == "a\nc\n" and step["stderr"] == "b\n"
    shown = [e for e in events if isinstance(e, CommandOutput)]
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

    kept = Kept()
    with pytest.raises(RuntimeError, match="boom"):
        await go(tmp_path, root, python(root, source), handler, kept=kept)
    assert (root / "after.txt").exists()
    [recorded] = kept.runs
    assert [step.tool for step in recorded.steps] == ["Shell"]
    assert [m.role for m in recorded.conversation] == ["user", "assistant", "tool"]


async def test_a_handler_error_fails_a_run_whose_call_was_stopped(
    tmp_path: Path, root: Path
) -> None:
    source = "import time\nprint('hi', flush=True)\ntime.sleep(5)\n"
    stop = Stop()

    async def handler(event: Event) -> None:
        if isinstance(event, CommandOutput):
            asyncio.get_running_loop().call_later(0.3, stop.now)
            raise RuntimeError("boom")

    kept = Kept()
    with pytest.raises(RuntimeError, match="boom"):
        await go(tmp_path, root, python(root, source), handler, kept=kept, stop=stop)
    [recorded] = kept.runs
    assert [step.stopped for step in recorded.steps] == [True]


async def test_chunks_that_queued_while_the_handler_was_busy_arrive_joined(
    tmp_path: Path, root: Path
) -> None:
    source = (
        "import os, time\n"
        "for i in range(200):\n"
        "    os.write(1, str(i % 10).encode())\n"
        "    time.sleep(0.002)\n"
    )
    events: list[CommandOutput] = []

    async def handler(event: Event) -> None:
        if isinstance(event, CommandOutput):
            events.append(event)
            await asyncio.sleep(0.05)

    await go(tmp_path, root, python(root, source), handler)
    assert "".join(e.text for e in events) == "".join(str(i % 10) for i in range(200))
    assert len(events) < 100


async def test_alternating_streams_with_a_slow_handler_do_not_hold_the_shell(
    tmp_path: Path, root: Path
) -> None:
    source = (
        "import os, time\n"
        "for i in range(300):\n"
        "    os.write(1, f'o{i}\\n'.encode())\n"
        "    os.write(2, f'e{i}\\n'.encode())\n"
        "    time.sleep(0.001)\n"
        "open('ended', 'w').write(repr(time.time()))\n"
    )
    events: list[Event] = []
    ready: list[float] = []

    async def handler(event: Event) -> None:
        if isinstance(event, CommandOutput):
            events.append(event)
            await asyncio.sleep(0.02)
        elif isinstance(event, ToolResultReady):
            ready.append(time.time())

    await go(tmp_path, root, python(root, source), handler)
    assert texts(events, "stdout") == "".join(f"o{i}\n" for i in range(300))
    assert texts(events, "stderr") == "".join(f"e{i}\n" for i in range(300))
    [shown] = ready
    assert shown - float((root / "ended").read_text(encoding="utf-8")) < 1.0


async def test_a_chunk_after_a_gap_starts_its_own_event_and_a_batch_joins_per_stream() -> None:
    said: list[tuple[str, str]] = []
    busy = asyncio.Event()
    free = asyncio.Event()

    async def say(stream: Literal["stdout", "stderr"], text: str) -> None:
        said.append((stream, text))
        busy.set()
        await free.wait()

    told = _Told(say)
    told.add("stdout", b"a")
    await busy.wait()
    told.add("stdout", b"b")
    told.add("stderr", b"1")
    told.add("stdout", b"x" * BACKLOG)
    told.add("stdout", b"c")
    told.add("stderr", b"2")
    free.set()
    told.close()
    await told.task
    assert said == [
        ("stdout", "a"),
        ("stdout", "b"),
        ("stderr", "12"),
        ("stdout", f"[... {BACKLOG} bytes not shown ...]\nc"),
    ]


async def test_a_background_job_sends_no_command_output(tmp_path: Path, root: Path) -> None:
    call = ToolCall(
        id="1", name="Shell", args={"command": "echo hi; sleep 0.3", "background": True}
    )
    events: list[Event] = []

    async def handler(event: Event) -> None:
        events.append(event)

    await go(tmp_path, root, call, handler)
    assert not any(isinstance(e, CommandOutput) for e in events)
