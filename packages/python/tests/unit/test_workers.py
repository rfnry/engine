import asyncio
import contextlib
import datetime
import gc
import logging
import os
import signal
import sys
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Coroutine, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, WorkerError
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.workers.pool import CAP, CLOSED, Helper, Script, WorkerPool, Workers, _Worker

from .fakes import Scripted, delegating, text_reply
from .trees import write_folder

CALL = {"scope": "acme"}
PID = "import os, time\ndef run(args, call):\n    time.sleep(args.get('sleep', 0))\n    return os.getpid()\n"
WAIT = "import asyncio, os\nasync def run(args, call):\n    await asyncio.sleep(args.get('sleep', 0))\n    return os.getpid()\n"
NOISE = "import os, sys\ndef run(args, call):\n    print('x' * 200 * 1024, end='', file=sys.stderr, flush=True)\n    return os.getpid()\n"


def script(tmp_path: Path, name: str, body: str) -> Script:
    path = tmp_path / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    helpers = {p.stem: Helper(p, p.read_text(encoding="utf-8")) for p in path.parent.glob("_*.py")}
    return Script(name, path, body, helpers)


def pid(value: JsonValue) -> int:
    assert isinstance(value, int)
    return value


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state != "Z"


async def gone(pid: int) -> bool:
    for _ in range(40):
        if not alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


def ended(pid: int) -> bool:
    for _ in range(40):
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


def on_a_loop_of_its_own(work: Awaitable[JsonValue]) -> JsonValue:
    async def bounded() -> JsonValue:
        async with asyncio.timeout(10):
            return await work

    return asyncio.run(bounded())


def reaped(pid: int) -> bool:
    return not Path(f"/proc/{pid}").exists()


def held() -> tuple[int, list[str]]:
    states = []
    for task in Path("/proc/self/task").glob("*/children"):
        try:
            children = task.read_text().split()
        except OSError:
            continue
        for child in children:
            with contextlib.suppress(OSError):
                states.append(Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[0])
    return len(os.listdir("/proc/self/fd")), states


def loop_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "asyncio" and record.levelno >= logging.ERROR
    ]


@contextlib.contextmanager
def no_collector() -> Iterator[None]:
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


async def test_a_call_runs_the_script_and_returns_its_result(tmp_path: Path) -> None:
    pool = WorkerPool(
        [
            script(
                tmp_path,
                "echo",
                "def run(args, call):\n    return {'got': args, 'scope': call['scope']}\n",
            )
        ]
    )
    try:
        assert await pool.call("echo", {"x": 1}, {"scope": "acme"}, 5) == {
            "got": {"x": 1},
            "scope": "acme",
        }
    finally:
        await pool.close()


BIG = "def run(args, call):\n    return 'x' * args['size']\n"
LOUD = "def run(args, call):\n    raise ValueError('x' * args['size'])\n"
OVER = "^workers/process: big sent back {} bytes, over the limit of 100 bytes$"


async def test_a_result_over_the_limit_fails_its_call_and_the_worker_serves_on(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "big", BIG)], limit=100)
    try:
        assert await pool.call("big", {"size": 50}, CALL, 5) == "x" * 50
        with pytest.raises(WorkerError, match=OVER.format(300_023)):
            await pool.call("big", {"size": 300_000}, CALL, 5)
        assert await pool.call("big", {"size": 10}, CALL, 5) == "x" * 10
    finally:
        await pool.close()


async def test_an_error_over_the_limit_is_cut_the_same_way(tmp_path: Path) -> None:
    pool = WorkerPool([script(tmp_path, "big", LOUD)], limit=100)
    try:
        with pytest.raises(WorkerError, match="^ValueError: x{10}$"):
            await pool.call("big", {"size": 10}, CALL, 5)
        with pytest.raises(WorkerError, match=OVER.format(300_034)):
            await pool.call("big", {"size": 300_000}, CALL, 5)
    finally:
        await pool.close()


async def test_async_scripts_run_together_on_one_process(tmp_path: Path) -> None:
    body = "import asyncio\nasync def run(args, call):\n    await asyncio.sleep(0.5)\n    return args['i']\n"
    pool = WorkerPool([script(tmp_path, "slow", body)])
    try:
        started = time.monotonic()
        results = await asyncio.gather(
            *(pool.call("slow", {"i": i}, {"scope": "a"}, 5) for i in range(20))
        )
        assert results == list(range(20)) and time.monotonic() - started < 2
    finally:
        await pool.close()


async def test_a_script_that_raises_reads_as_its_error(tmp_path: Path) -> None:
    pool = WorkerPool(
        [script(tmp_path, "boom", "def run(args, call):\n    raise ValueError('no order A-1')\n")]
    )
    try:
        with pytest.raises(WorkerError, match="ValueError: no order A-1"):
            await pool.call("boom", {}, {"scope": "a"}, 5)
    finally:
        await pool.close()


async def test_a_call_for_a_tool_the_worker_does_not_have_reads_as_an_error_and_holds_nothing(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)], threads=1)
    try:
        first = await pool.call("nap", {}, CALL, 5)
        with pytest.raises(WorkerError, match="workers/process: the worker has no tool ghost"):
            await pool.call("support/tools/ghost", {}, CALL, 1)
        assert await pool.call("nap", {}, CALL, 5) == first
    finally:
        await pool.close()


async def test_an_error_whose_message_cannot_be_read_still_reads_as_its_type(
    tmp_path: Path,
) -> None:
    body = (
        "class Broken(Exception):\n"
        "    def __str__(self):\n"
        "        raise RuntimeError('no message')\n"
        "def run(args, call):\n"
        "    if args.get('break'):\n"
        "        raise Broken()\n"
        "    return 'fine'\n"
    )
    pool = WorkerPool([script(tmp_path, "odd", body)], threads=1)
    try:
        with pytest.raises(WorkerError, match=r"^Broken: <exception str\(\) failed>$"):
            await pool.call("odd", {"break": True}, CALL, 1)
        assert await pool.call("odd", {}, CALL, 5) == "fine"
    finally:
        await pool.close()


async def test_a_script_that_exits_reads_as_its_error_and_its_worker_goes_on(
    tmp_path: Path,
) -> None:
    pool = WorkerPool(
        [
            script(tmp_path, "quit", "import sys\ndef run(args, call):\n    sys.exit(3)\n"),
            script(tmp_path, "nap", PID),
        ]
    )
    try:
        first = await pool.call("nap", {}, CALL, 5)
        with pytest.raises(WorkerError, match="SystemExit: 3"):
            await pool.call("quit", {}, CALL, 5)
        assert await pool.call("nap", {}, CALL, 5) == first
    finally:
        await pool.close()


async def test_a_script_that_reads_stdin_reads_nothing_and_the_calls_go_on(
    tmp_path: Path,
) -> None:
    body = "import sys\ndef run(args, call):\n    return [sys.stdin.read(), args]\n"
    pool = WorkerPool([script(tmp_path, "reader", body)])
    try:
        calls = [pool.call("reader", {"i": i}, CALL, 5) for i in range(3)]
        assert await asyncio.gather(*calls) == [["", {"i": i}] for i in range(3)]
    finally:
        await pool.close()


async def test_what_a_script_prints_goes_to_the_log_and_leaves_its_result_intact(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="rfnry_engine.workers.pool")
    body = (
        "import os, sys\n"
        "print('while loading')\n"
        "def run(args, call):\n"
        "    print('to stdout')\n"
        "    print('x' * 200_000)\n"
        "    print('to stderr', file=sys.stderr)\n"
        "    os.write(1, b'to fd 1\\n')\n"
        "    return {'ok': True}\n"
    )
    pool = WorkerPool([script(tmp_path, "noisy", body)])
    try:
        assert await pool.call("noisy", {}, CALL, 5) == {"ok": True}
        assert await pool.call("noisy", {}, CALL, 5) == {"ok": True}
        printed = ("while loading", "to stdout", "to stderr", "to fd 1")
        for _ in range(100):
            if all(text in caplog.text for text in printed):
                break
            await asyncio.sleep(0.02)
        assert all(text in caplog.text for text in printed)
    finally:
        await pool.close()


async def test_a_line_that_never_ends_is_logged_in_pieces_no_longer_than_the_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="rfnry_engine.workers.pool")
    pool = WorkerPool([script(tmp_path, "noisy", NOISE)])
    try:
        worker = pid(await pool.call("noisy", {}, CALL, 5))
    finally:
        await pool.close()
    pieces = [r.getMessage() for r in caplog.records if r.name == "rfnry_engine.workers.pool"]
    prefix = f"workers/pool: worker {worker}: "
    assert len(pieces) > 1 and sum(piece.count("x") for piece in pieces) == 200 * 1024
    assert all(piece.startswith(prefix) and len(piece) <= len(prefix) + CAP for piece in pieces)


async def test_with_debug_off_what_a_script_prints_is_dropped_as_it_comes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="rfnry_engine.workers.pool")
    pool = WorkerPool([script(tmp_path, "noisy", NOISE)])
    try:
        await pool.call("noisy", {}, CALL, 5)
        assert [bytes(w._err) for w in pool._workers if w is not None] == [b""]
    finally:
        await pool.close()


async def test_a_result_that_is_not_json_is_an_error_naming_its_type(tmp_path: Path) -> None:
    body = (
        "import datetime\n"
        "class Order:\n"
        "    pass\n"
        "def run(args, call):\n"
        "    kind = args['kind']\n"
        "    if kind == 'datetime':\n"
        "        return datetime.datetime(2026, 10, 24)\n"
        "    if kind == 'set':\n"
        "        return {1, 2}\n"
        "    if kind == 'object':\n"
        "        return Order()\n"
        "    if kind == 'nan':\n"
        "        return float('nan')\n"
        "    return 'fine'\n"
    )
    pool = WorkerPool([script(tmp_path, "odd", body)])
    try:
        for kind, error in (
            ("datetime", "TypeError: Object of type datetime is not JSON serializable"),
            ("set", "TypeError: Object of type set is not JSON serializable"),
            ("object", "TypeError: Object of type Order is not JSON serializable"),
            ("nan", "ValueError: Out of range float values are not JSON compliant"),
        ):
            with pytest.raises(WorkerError, match=error):
                await pool.call("odd", {"kind": kind}, CALL, 5)
        assert await pool.call("odd", {"kind": "text"}, CALL, 5) == "fine"
    finally:
        await pool.close()


LATER = (
    "import asyncio, os\n"
    "def later(f):\n"
    "    def wrapper(args, call):\n"
    "        async def go():\n"
    "            await asyncio.sleep(args.get('sleep', 0))\n"
    "            return f(args, call)\n"
    "        return go()\n"
    "    return wrapper\n"
    "@later\n"
    "def run(args, call):\n"
    "    return [os.getpid(), 'later']\n"
)
LOGGED = (
    "import asyncio, os\n"
    "def logged(f):\n"
    "    def wrapper(args, call):\n"
    "        return f(args, call)\n"
    "    return wrapper\n"
    "@logged\n"
    "async def run(args, call):\n"
    "    await asyncio.sleep(args.get('sleep', 0))\n"
    "    return [os.getpid(), 'logged']\n"
)
BLOCKING = (
    "import asyncio, os\n"
    "def blocking(f):\n"
    "    def wrapper(args, call):\n"
    "        return asyncio.run(f(args, call))\n"
    "    return wrapper\n"
    "@blocking\n"
    "async def run(args, call):\n"
    "    await asyncio.sleep(args.get('sleep', 0))\n"
    "    return [os.getpid(), 'blocking']\n"
)
EAGER = (
    "import asyncio, os\n"
    "def eager(f):\n"
    "    async def wrapper(args, call):\n"
    "        await asyncio.sleep(args.get('sleep', 0))\n"
    "        return f(args, call)\n"
    "    return wrapper\n"
    "@eager\n"
    "def run(args, call):\n"
    "    return [os.getpid(), 'eager']\n"
)


DECORATED = {"later": LATER, "logged": LOGGED, "blocking": BLOCKING, "eager": EAGER}


@pytest.mark.parametrize("said", list(DECORATED))
async def test_a_decorated_run_is_called_as_it_really_is(tmp_path: Path, said: str) -> None:
    pool = WorkerPool([script(tmp_path, "decorated", DECORATED[said])])
    try:
        result = await pool.call("decorated", {}, CALL, 5)
        assert isinstance(result, list) and result[1] == said
    finally:
        await pool.close()


@pytest.mark.parametrize(("said", "kept"), [("blocking", False), ("eager", True)])
async def test_a_decorated_run_holds_a_thread_only_when_it_takes_one(
    tmp_path: Path, said: str, kept: bool
) -> None:
    pool = WorkerPool([script(tmp_path, "decorated", DECORATED[said])], threads=1)
    try:
        first = cast(list[int], await pool.call("decorated", {}, CALL, 5))[0]
        with pytest.raises(TimeoutError):
            await pool.call("decorated", {"sleep": 30}, CALL, 0.2)
        after = cast(list[int], await pool.call("decorated", {}, CALL, 2))[0]
        assert (after == first) is kept
    finally:
        await pool.close()


async def test_two_folders_with_a_helper_of_one_name_each_import_their_own(
    tmp_path: Path,
) -> None:
    scripts = []
    for folder in ("customer_support", "ops"):
        tools = tmp_path / folder / "tools"
        tools.mkdir(parents=True)
        (tools / "_client.py").write_text(f"NAME = {folder!r}\n", encoding="utf-8")
        body = "import _client\ndef run(args, call):\n    return _client.NAME\n"
        scripts.append(script(tmp_path, f"{folder}/tools/lookup", body))
    pool = WorkerPool(scripts)
    try:
        assert await pool.call("customer_support/tools/lookup", {}, CALL, 5) == "customer_support"
        assert await pool.call("ops/tools/lookup", {}, CALL, 5) == "ops"
        assert list(tmp_path.rglob("__pycache__")) == []
    finally:
        await pool.close()


async def test_a_worker_runs_the_source_it_was_given_under_the_script_s_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tools/given.py"
    path.parent.mkdir(parents=True)
    path.write_text("def run(args, call):\n    return 'on disk'\n", encoding="utf-8")
    (tmp_path / "tools/_client.py").write_text("NAME = 'on disk'\n", encoding="utf-8")
    body = (
        "import _client\n"
        "def run(args, call):\n"
        "    return [_client.NAME, __file__, run.__code__.co_filename, _client.__file__]\n"
    )
    given = Helper(tmp_path / "tools/_client.py", "NAME = 'given'\n")
    pool = WorkerPool([Script("tools/given", path, body, {"_client": given})])
    try:
        assert await pool.call("tools/given", {}, CALL, 5) == [
            "given",
            str(path),
            str(path),
            str(tmp_path / "tools/_client.py"),
        ]
        assert list(tmp_path.rglob("__pycache__")) == []
    finally:
        await pool.close()


async def test_a_large_result_round_trips_intact(tmp_path: Path) -> None:
    body = "def run(args, call):\n    return '0123456789' * 500_000 + 'fim é'\n"
    pool = WorkerPool([script(tmp_path, "big", body)])
    try:
        assert await pool.call("big", {}, CALL, 10) == "0123456789" * 500_000 + "fim é"
    finally:
        await pool.close()


async def test_a_sync_call_past_its_timeout_raises_while_other_calls_proceed(
    tmp_path: Path,
) -> None:
    body = "import time\ndef run(args, call):\n    time.sleep(args['sleep'])\n    return args['sleep']\n"
    pool = WorkerPool([script(tmp_path, "nap", body)])
    try:
        stuck, quick = await asyncio.gather(
            pool.call("nap", {"sleep": 30}, CALL, 0.2),
            pool.call("nap", {"sleep": 0.3}, CALL, 5),
            return_exceptions=True,
        )
        assert isinstance(stuck, TimeoutError) and quick == 0.3
    finally:
        await pool.close()


async def test_once_every_thread_is_held_past_its_deadline_the_next_call_gets_a_new_process(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)], threads=2)
    try:
        first = pid(await pool.call("nap", {}, CALL, 5))
        stuck = await asyncio.gather(
            *(pool.call("nap", {"sleep": 30}, CALL, 0.2) for _ in range(2)),
            return_exceptions=True,
        )
        assert all(isinstance(result, TimeoutError) for result in stuck)
        await asyncio.sleep(0.2)
        assert pid(await pool.call("nap", {}, CALL, 5)) != first
        assert await gone(first)
    finally:
        await pool.close()


async def test_a_late_reply_frees_the_thread_its_call_held(tmp_path: Path) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)], threads=2)
    try:
        first = await pool.call("nap", {}, CALL, 5)
        with pytest.raises(TimeoutError):
            await pool.call("nap", {"sleep": 0.4}, CALL, 0.1)
        assert await pool.call("nap", {}, CALL, 5) == first
        await asyncio.sleep(0.5)
        with pytest.raises(TimeoutError):
            await pool.call("nap", {"sleep": 30}, CALL, 0.1)
        assert await pool.call("nap", {}, CALL, 5) == first
    finally:
        await pool.close()


async def test_a_call_cancelled_from_outside_holds_its_thread_like_a_timeout(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)], threads=1)
    try:
        first = await pool.call("nap", {}, CALL, 5)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(pool.call("nap", {"sleep": 30}, CALL, None), 0.2)
        await asyncio.sleep(0.2)
        assert await pool.call("nap", {}, CALL, 5) != first
    finally:
        await pool.close()


async def test_an_async_call_past_its_timeout_is_cancelled_in_the_worker(tmp_path: Path) -> None:
    body = (
        "import asyncio, os\n"
        "from pathlib import Path\n"
        "async def run(args, call):\n"
        "    if 'file' in args:\n"
        "        await asyncio.sleep(0.3)\n"
        "        Path(args['file']).write_text('late')\n"
        "    return os.getpid()\n"
    )
    pool = WorkerPool([script(tmp_path, "wait", body)], threads=1)
    late = tmp_path / "late.txt"
    try:
        first = await pool.call("wait", {}, CALL, 5)
        with pytest.raises(TimeoutError):
            await pool.call("wait", {"file": str(late)}, CALL, 0.1)
        await asyncio.sleep(0.5)
        assert not late.exists()
        assert await pool.call("wait", {}, CALL, 5) == first
    finally:
        await pool.close()


async def test_an_async_call_past_its_deadline_holds_no_thread(tmp_path: Path) -> None:
    pool = WorkerPool([script(tmp_path, "wait", WAIT)], threads=1)
    try:
        first = await pool.call("wait", {}, CALL, 5)
        with pytest.raises(TimeoutError):
            await pool.call("wait", {"sleep": 30}, CALL, 0.1)
        assert await pool.call("wait", {}, CALL, 5) == first
    finally:
        await pool.close()


async def test_an_async_call_that_blocks_its_worker_s_loop_gets_the_worker_retired(
    tmp_path: Path,
) -> None:
    body = "import os, time\nasync def run(args, call):\n    time.sleep(args.get('sleep', 0))\n    return os.getpid()\n"
    pool = WorkerPool([script(tmp_path, "block", body)], threads=2)
    try:
        first = pid(await pool.call("block", {}, CALL, 5))
        for _ in range(2):
            with pytest.raises(TimeoutError):
                await pool.call("block", {"sleep": 30}, CALL, 0.2)
        assert pid(await pool.call("block", {}, CALL, 5)) != first
        assert await gone(first)
    finally:
        await pool.close()


async def test_a_call_cancelled_from_outside_after_an_unanswered_cancel_holds_no_thread(
    tmp_path: Path,
) -> None:
    body = "import os, time\nasync def run(args, call):\n    time.sleep(args.get('sleep', 0))\n    return os.getpid()\n"
    pool = WorkerPool([script(tmp_path, "block", body)], threads=2)
    try:
        with pytest.raises(TimeoutError):
            await pool.call("block", {"sleep": 30}, CALL, 0.2)
        second = asyncio.create_task(pool.call("block", {}, CALL, 5))
        await asyncio.sleep(0.2)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        worker = pool._workers[0]
        assert worker is not None and worker.held == set() and len(worker.cancelling) == 2
    finally:
        await pool.close()


@pytest.mark.parametrize(
    ("ending", "reason"),
    [("os._exit(3)", "exited with code 3"), ("os.kill(os.getpid(), 9)", "killed by signal 9")],
)
async def test_a_worker_that_exits_fails_every_call_in_flight_and_the_next_call_starts_anew(
    tmp_path: Path, ending: str, reason: str
) -> None:
    crash = f"import os, time\ndef run(args, call):\n    time.sleep(0.2)\n    {ending}\n"
    pool = WorkerPool(
        [
            script(tmp_path, "support/tools/crash", crash),
            script(tmp_path, "support/tools/wait", WAIT),
        ]
    )
    try:
        first = await pool.call("support/tools/wait", {}, CALL, 5)
        results = await asyncio.gather(
            pool.call("support/tools/wait", {"sleep": 5}, CALL, 10),
            pool.call("support/tools/wait", {"sleep": 5}, CALL, 10),
            pool.call("support/tools/crash", {}, CALL, 10),
            return_exceptions=True,
        )
        assert all(isinstance(result, WorkerError) for result in results)
        assert [str(result) for result in results] == [
            f"workers/pool: the worker for wait stopped: {reason}",
            f"workers/pool: the worker for wait stopped: {reason}",
            f"workers/pool: the worker for crash stopped: {reason}",
        ]
        assert await pool.call("support/tools/wait", {}, CALL, 5) != first
    finally:
        await pool.close()


@pytest.mark.parametrize("watch", ["waitid", "wait"])
@pytest.mark.parametrize(
    "fork", ["os.fork()", "ctypes.CDLL(None).fork()"], ids=["python", "c-pipe-held"]
)
async def test_a_worker_that_exits_past_a_child_it_forked_fails_its_call_with_its_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fork: str, watch: str
) -> None:
    if watch == "wait":
        monkeypatch.delattr(os, "waitid")
    body = (
        "import ctypes, os, time\n"
        "from pathlib import Path\n"
        "def run(args, call):\n"
        f"    child = {fork}\n"
        "    if child == 0:\n"
        "        time.sleep(30)\n"
        "        os._exit(0)\n"
        "    Path(args['file']).write_text(str(child))\n"
        "    os._exit(3)\n"
    )
    pool = WorkerPool([script(tmp_path, "forks", body)])
    held = tmp_path / "child.txt"
    try:
        started = time.monotonic()
        with pytest.raises(WorkerError) as raised:
            await pool.call("forks", {"file": str(held)}, CALL, 10)
        assert time.monotonic() - started < 5
        assert str(raised.value) == "workers/pool: the worker for forks stopped: exited with code 3"
        assert await gone(int(held.read_text()))
    finally:
        await pool.close()
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(held.read_text()), signal.SIGKILL)


async def test_a_worker_that_dies_before_its_start_up_line_fails_its_calls(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pool = WorkerPool([script(tmp_path, "early", "import os\nos._exit(3)\n" + PID)])
    try:
        async with asyncio.timeout(5):
            calls = (pool.call("early", {}, CALL, 10) for _ in range(2))
            results = await asyncio.gather(*calls, return_exceptions=True)
        assert [str(result) for result in results] == [
            "workers/pool: the worker for early stopped: exited with code 3"
        ] * 2
        assert all(isinstance(result, WorkerError) for result in results)
        assert loop_errors(caplog) == []
    finally:
        await pool.close()


GARBLE = (
    "import fcntl, os, time\n"
    "def run(args, call):\n"
    "    err = os.readlink('/proc/self/fd/2')\n"
    "    for name in os.listdir('/proc/self/fd'):\n"
    "        fd = int(name)\n"
    "        try:\n"
    "            link = os.readlink(f'/proc/self/fd/{fd}')\n"
    "            mode = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE\n"
    "        except OSError:\n"
    "            continue\n"
    "        if fd > 2 and link.startswith('pipe:') and link != err and mode == os.O_WRONLY:\n"
    "            os.write(fd, args['line'].encode() + b'\\n')\n"
    "    time.sleep(30)\n"
)


@pytest.mark.parametrize(
    "line", ["not a reply", "[1]", '{"id": 0}', '{"id": 0, "error": 1}', '{"threaded": [{}]}']
)
async def test_a_line_that_is_not_a_reply_stops_the_worker_and_fails_its_calls(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, line: str
) -> None:
    pool = WorkerPool([script(tmp_path, "garble", GARBLE), script(tmp_path, "wait", WAIT)])
    try:
        async with asyncio.timeout(5):
            results = await asyncio.gather(
                pool.call("garble", {"line": line}, CALL, 10),
                pool.call("wait", {"sleep": 30}, CALL, 10),
                return_exceptions=True,
            )
        assert [str(result) for result in results] == [
            "workers/pool: the worker for garble stopped: it sent a line that is not a reply",
            "workers/pool: the worker for wait stopped: it sent a line that is not a reply",
        ]
        assert loop_errors(caplog) == []
    finally:
        await pool.close()


async def test_stopping_a_worker_whose_process_group_is_gone_raises_nothing(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    try:
        first = pid(await pool.call("nap", {}, CALL, 5))
        flight = asyncio.create_task(pool.call("nap", {"sleep": 30}, CALL, None))
        await asyncio.sleep(0.1)
        [worker] = pool._workers
        assert worker is not None
        os.kill(first, signal.SIGKILL)
        worker.process.wait()
        with pytest.raises(ProcessLookupError):
            os.killpg(first, 0)
        worker.stop()
        with pytest.raises(WorkerError) as raised:
            async with asyncio.timeout(5):
                await flight
        assert str(raised.value) == "workers/pool: the worker for nap stopped: killed by signal 9"
    finally:
        await pool.close()


async def test_a_reply_sent_just_before_the_worker_exits_still_answers_its_call(
    tmp_path: Path,
) -> None:
    body = (
        "import os, threading\n"
        "def run(args, call):\n"
        "    if args.get('exit'):\n"
        "        threading.Timer(0.1, os._exit, (3,)).start()\n"
        "    return 'done'\n"
    )
    pool = WorkerPool([script(tmp_path, "last", body)])
    try:
        assert await pool.call("last", {}, CALL, 5) == "done"
        last = asyncio.create_task(pool.call("last", {"exit": True}, CALL, 5))
        await asyncio.sleep(0)
        time.sleep(0.5)
        assert await last == "done"
    finally:
        await pool.close()


async def test_many_processes_share_the_calls(tmp_path: Path) -> None:
    body = "import asyncio, os\nasync def run(args, call):\n    await asyncio.sleep(0.3)\n    return os.getpid()\n"
    pool = WorkerPool([script(tmp_path, "wait", body)], processes=2)
    try:
        pids = await asyncio.gather(*(pool.call("wait", {}, CALL, 5) for _ in range(10)))
        assert sorted(Counter(pids).values()) == [5, 5]
    finally:
        await pool.close()


async def test_a_script_that_cannot_import_fails_only_its_own_calls(tmp_path: Path) -> None:
    pool = WorkerPool(
        [
            script(
                tmp_path, "broken", "import not_installed\ndef run(args, call):\n    return 1\n"
            ),
            script(tmp_path, "echo", "def run(args, call):\n    return args\n"),
        ]
    )
    try:
        with pytest.raises(
            WorkerError, match="ModuleNotFoundError: No module named 'not_installed'"
        ):
            await pool.call("broken", {}, CALL, 5)
        assert await pool.call("echo", {"x": 1}, CALL, 5) == {"x": 1}
    finally:
        await pool.close()


async def test_close_stops_the_process_group_and_leaves_no_process_behind(
    tmp_path: Path,
) -> None:
    body = (
        "import os, subprocess\n"
        "def run(args, call):\n"
        "    child = subprocess.Popen(['sleep', '60'])\n"
        "    return [os.getpid(), child.pid]\n"
    )
    pool = WorkerPool([script(tmp_path, "spawn", body)])
    try:
        started = await pool.call("spawn", {}, CALL, 5)
    finally:
        await pool.close()
    assert isinstance(started, list)
    worker, child = (pid(value) for value in started)
    assert await gone(worker) and await gone(child)


async def test_close_fails_the_calls_in_flight_and_refuses_new_ones(tmp_path: Path) -> None:
    pool = WorkerPool([script(tmp_path, "wait", WAIT)])
    try:
        await pool.call("wait", {}, CALL, 5)
        flight = asyncio.create_task(pool.call("wait", {"sleep": 30}, CALL, None))
        await asyncio.sleep(0.1)
    finally:
        await pool.close()
    closed = "workers/pool: the worker for wait stopped: it was closed"
    with pytest.raises(WorkerError, match=closed):
        async with asyncio.timeout(5):
            await flight
    with pytest.raises(WorkerError, match=closed):
        await pool.call("wait", {}, CALL, 5)


async def test_retiring_a_worker_fails_its_other_calls_in_flight_saying_why(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID), script(tmp_path, "wait", WAIT)], threads=1)
    try:
        first = pid(await pool.call("nap", {}, CALL, 5))
        flight = asyncio.create_task(pool.call("wait", {"sleep": 30}, CALL, None))
        await asyncio.sleep(0.1)
        with pytest.raises(TimeoutError):
            await pool.call("nap", {"sleep": 30}, CALL, 0.2)
        assert pid(await pool.call("nap", {}, CALL, 5)) != first
        with pytest.raises(WorkerError) as raised:
            async with asyncio.timeout(5):
                await flight
        assert str(raised.value) == (
            "workers/pool: the worker for wait stopped: every thread was held by a call past its deadline"
        )
        assert await gone(first)
    finally:
        await pool.close()


async def test_a_call_that_waits_past_its_deadline_for_a_worker_raises_the_pool_s_timeout(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    try:
        first = pid(await pool.call("nap", {}, CALL, 5))
        async with pool._lock:
            with pytest.raises(TimeoutError) as raised:
                await pool.call("nap", {}, CALL, 0.2)
        assert str(raised.value) == "workers/pool: nap timed out after 0.2s"
        assert pid(await pool.call("nap", {}, CALL, 5)) == first
    finally:
        await pool.close()


@pytest.mark.parametrize(
    ("args", "kind"),
    [({"when": datetime.datetime(2026, 1, 1)}, "datetime"), ({"ids": {1, 2}}, "set")],
)
async def test_arguments_json_cannot_hold_are_refused_before_any_worker_sees_them(
    tmp_path: Path, args: dict[str, object], kind: str
) -> None:
    pool = WorkerPool([script(tmp_path, "echo", "def run(args, call):\n    return args\n")])
    try:
        with pytest.raises(InputError) as raised:
            await pool.call("echo", args, CALL, 5)  # type: ignore[arg-type]
        assert str(raised.value) == (
            f"workers/pool: the arguments for echo are not JSON: Object of type {kind} is not JSON serializable"
        )
        assert await pool.call("echo", {"x": 1}, CALL, 5) == {"x": 1}
        assert [(len(w.calls), len(w.held)) for w in pool._workers if w is not None] == [(0, 0)]
    finally:
        await pool.close()


def test_a_pool_called_from_another_event_loop_moves_its_worker_there(tmp_path: Path) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)], threads=2)
    first = pid(on_a_loop_of_its_own(pool.call("nap", {}, CALL, 5)))
    assert alive(first)
    second = pid(on_a_loop_of_its_own(pool.call("nap", {}, CALL, 5)))
    assert second != first and ended(first)
    third = [pid(on_a_loop_of_its_own(pool.call("nap", {}, CALL, 5))) for _ in range(3)]
    assert len({second, *third}) == 4 and all(ended(old) for old in [second, *third[:-1]])
    on_a_loop_of_its_own(pool.close())
    assert ended(third[-1])


def test_close_from_another_event_loop_leaves_no_worker_behind(tmp_path: Path) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)], processes=2)

    async def both() -> JsonValue:
        return await asyncio.gather(*(pool.call("nap", {"sleep": 0.2}, CALL, 5) for _ in range(2)))

    first = on_a_loop_of_its_own(both())
    assert isinstance(first, list) and len({pid(value) for value in first}) == 2
    on_a_loop_of_its_own(pool.close())
    assert all(ended(pid(value)) for value in first)
    with pytest.raises(WorkerError, match="the worker for nap stopped: it was closed"):
        on_a_loop_of_its_own(pool.call("nap", {}, CALL, 5))
    on_a_loop_of_its_own(pool.close())


async def test_close_from_another_thread_stops_the_workers_on_the_loop_they_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stops: list[int] = []
    stop = _Worker.stop

    def stopping(worker: _Worker, reason: str | None = None) -> None:
        stops.append(threading.get_ident())
        stop(worker, reason)

    monkeypatch.setattr(_Worker, "stop", stopping)
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        first = asyncio.run_coroutine_threadsafe(pool.call("nap", {}, CALL, 5), loop)
        worker = pid(await asyncio.wrap_future(first))
        await pool.close()
        assert reaped(worker) and set(stops) == {thread.ident}
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


async def test_close_from_another_thread_as_the_workers_loop_stops_returns_and_stops_them(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        first = asyncio.run_coroutine_threadsafe(pool.call("nap", {}, CALL, 5), loop)
        worker = pid(await asyncio.wrap_future(first))
        loop.call_soon_threadsafe(loop.stop)
        async with asyncio.timeout(5):
            await pool.close()
        assert reaped(worker)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def plain_task(
    loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, Any]
) -> asyncio.Task[Any]:
    return asyncio.Task(coro, loop=loop)


async def test_close_from_another_thread_returns_when_the_workers_loop_starts_no_task_eagerly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stops: list[int] = []
    stop = _Worker.stop

    def stopping(worker: _Worker, reason: str | None = None) -> None:
        stops.append(threading.get_ident())
        stop(worker, reason)

    monkeypatch.setattr(_Worker, "stop", stopping)
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    loop = asyncio.new_event_loop()
    loop.set_task_factory(plain_task)
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        first = asyncio.run_coroutine_threadsafe(pool.call("nap", {}, CALL, 5), loop)
        worker = pid(await asyncio.wrap_future(first))
        async with asyncio.timeout(5):
            await pool.close()
        assert reaped(worker) and set(stops) == {thread.ident}
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        pool._drop(CLOSED)
        loop.close()


async def test_close_from_another_thread_raises_what_stopping_a_worker_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = _Worker.stop

    def failing(worker: _Worker, reason: str | None = None) -> None:
        stop(worker, reason)
        raise OSError("the pipe would not close")

    pool = WorkerPool([script(tmp_path, "nap", PID)])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        first = asyncio.run_coroutine_threadsafe(pool.call("nap", {}, CALL, 5), loop)
        worker = pid(await asyncio.wrap_future(first))
        monkeypatch.setattr(_Worker, "stop", failing)
        with pytest.raises(OSError, match="the pipe would not close"):
            async with asyncio.timeout(5):
                await pool.close()
        assert reaped(worker)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


async def test_close_stops_every_worker_before_it_raises_what_the_first_stop_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="rfnry_engine.workers.pool")
    stop = _Worker.stop
    stopped: list[_Worker] = []

    def failing(worker: _Worker, reason: str | None = None) -> None:
        stop(worker, reason)
        stopped.append(worker)
        raise OSError(f"stop {len(stopped)} would not close its pipe")

    pool = WorkerPool([script(tmp_path, "wait", WAIT)], processes=2)
    pids = await asyncio.gather(*(pool.call("wait", {"sleep": 0.2}, CALL, 5) for _ in range(2)))
    assert len(set(pids)) == 2
    monkeypatch.setattr(_Worker, "stop", failing)
    with pytest.raises(OSError, match="^stop 1 would not close its pipe$"):
        await pool.close()
    assert len(stopped) == 2 and all(reaped(pid(value)) for value in pids)
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "rfnry_engine.workers.pool"
    ] == ["workers/pool: could not stop a worker: stop 2 would not close its pipe"]


async def test_a_stop_that_raises_still_stops_the_other_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = _Worker.stop
    refused: list[_Worker] = []

    def refusing(worker: _Worker, reason: str | None = None) -> None:
        if not refused:
            refused.append(worker)
            raise OSError("the first worker would not stop")
        stop(worker, reason)

    pool = WorkerPool([script(tmp_path, "wait", WAIT)], processes=2)
    calls = (pool.call("wait", {"sleep": 0.2}, CALL, 5) for _ in range(2))
    pids = {pid(value) for value in await asyncio.gather(*calls)}
    started = [worker for worker in pool._workers if worker is not None]
    assert len(pids) == len(started) == 2
    monkeypatch.setattr(_Worker, "stop", refusing)
    try:
        with pytest.raises(OSError, match="^the first worker would not stop$"):
            await pool.close()
        [kept] = refused
        assert alive(kept.pid) and all(reaped(value) for value in pids - {kept.pid})
    finally:
        monkeypatch.undo()
        for worker in started:
            stop(worker, CLOSED)


async def test_closing_the_workers_closes_every_pool_before_it_raises_what_the_first_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="rfnry_engine.workers.pool")
    first = WorkerPool([script(tmp_path / "one", "nap", PID)])
    second = WorkerPool([script(tmp_path / "two", "nap", PID)])
    third = WorkerPool([script(tmp_path / "three", "nap", PID)])
    pids = [pid(await pool.call("nap", {}, CALL, 5)) for pool in (first, second, third)]
    close = WorkerPool.close

    async def failing(pool: WorkerPool, said: str) -> None:
        await close(pool)
        raise OSError(f"the {said} pool would not close")

    async def slow() -> None:
        await asyncio.sleep(0.2)
        await close(second)

    workers = Workers({}, {}, None)
    workers._pools = {"default": first, "billing": second, "support": third}
    monkeypatch.setattr(first, "close", lambda: failing(first, "first"))
    monkeypatch.setattr(second, "close", slow)
    monkeypatch.setattr(third, "close", lambda: failing(third, "third"))
    with pytest.raises(OSError, match="^the first pool would not close$"):
        await workers.close()
    assert all(reaped(value) for value in pids)
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "rfnry_engine.workers.pool"
    ] == ["workers/pool: could not stop a worker: the third pool would not close"]


async def test_a_cancel_of_closing_the_workers_from_another_thread_still_closes_every_pool(
    tmp_path: Path,
) -> None:
    first = WorkerPool([script(tmp_path / "one", "nap", PID)])
    second = WorkerPool([script(tmp_path / "two", "nap", PID)])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        pids = [
            pid(await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(call, loop)))
            for call in (first.call("nap", {}, CALL, 5), second.call("nap", {}, CALL, 5))
        ]
        workers = Workers({}, {}, None)
        workers._pools = {"default": first, "billing": second}
        busy = threading.Event()

        def block() -> None:
            busy.set()
            time.sleep(0.3)

        loop.call_soon_threadsafe(block)
        assert busy.wait(5)
        closing = asyncio.ensure_future(workers.close())
        await asyncio.sleep(0.05)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(5):
                await closing
        assert first._closed and second._closed
        assert all(ended(value) for value in pids)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        first._drop(CLOSED)
        second._drop(CLOSED)
        loop.close()


def blocking(loop: asyncio.AbstractEventLoop, seconds: float) -> None:
    busy = threading.Event()

    def block() -> None:
        busy.set()
        time.sleep(seconds)

    loop.call_soon_threadsafe(block)
    assert busy.wait(5)


async def test_a_cancel_of_closing_the_workers_does_not_wait_for_a_blocked_loop(
    tmp_path: Path,
) -> None:
    first = WorkerPool([script(tmp_path / "one", "nap", PID)])
    second = WorkerPool([script(tmp_path / "two", "nap", PID)])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        pids = [
            pid(await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(call, loop)))
            for call in (first.call("nap", {}, CALL, 5), second.call("nap", {}, CALL, 5))
        ]
        workers = Workers({}, {}, None)
        workers._pools = {"default": first, "billing": second}
        blocking(loop, 2.0)
        began = time.monotonic()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.3):
                await workers.close()
        assert time.monotonic() - began < 1.5
        assert first._closed and second._closed
        assert all(ended(value) for value in pids)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        first._drop(CLOSED)
        second._drop(CLOSED)
        loop.close()


async def test_a_cancelled_close_from_another_thread_still_logs_a_stop_that_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="rfnry_engine.workers.pool")
    stop = _Worker.stop

    def failing(worker: _Worker, reason: str | None = None) -> None:
        stop(worker, reason)
        raise OSError("the pipe would not close")

    pool = WorkerPool([script(tmp_path, "nap", PID)])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        first = asyncio.run_coroutine_threadsafe(pool.call("nap", {}, CALL, 5), loop)
        worker = pid(await asyncio.wrap_future(first))
        monkeypatch.setattr(_Worker, "stop", failing)
        blocking(loop, 0.3)
        closing = asyncio.ensure_future(pool.close())
        await asyncio.sleep(0.05)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(5):
                await closing

        def logged() -> list[str]:
            return [r.getMessage() for r in caplog.records if r.name == "rfnry_engine.workers.pool"]

        for _ in range(100):
            if logged():
                break
            await asyncio.sleep(0.02)
        assert logged() == ["workers/pool: could not stop a worker: the pipe would not close"]
        assert reaped(worker)
    finally:
        monkeypatch.undo()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        pool._drop(CLOSED)
        loop.close()


async def test_close_returns_once_a_worker_a_call_was_still_starting_has_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect = _Worker.connect
    started: list[int] = []
    connected = asyncio.Event()

    async def slow(worker: _Worker) -> None:
        started.append(worker.pid)
        await connect(worker)
        connected.set()
        await asyncio.sleep(0.5)

    monkeypatch.setattr(_Worker, "connect", slow)
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    calling = asyncio.ensure_future(pool.call("nap", {}, CALL, 5))
    try:
        async with asyncio.timeout(5):
            await connected.wait()
            await pool.close()
        [spawned] = started
        assert reaped(spawned)
        with pytest.raises(WorkerError, match="the worker for nap stopped: it was closed"):
            await calling
    finally:
        if not calling.done():
            calling.cancel()
        await asyncio.wait([calling])
        pool._drop(CLOSED)


async def test_a_stop_error_of_close_is_raised_only_once_a_worker_still_starting_has_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)], processes=2)
    first = pid(await pool.call("nap", {}, CALL, 5))
    busy = asyncio.ensure_future(pool.call("nap", {"sleep": 2}, CALL, 5))
    await asyncio.sleep(0.2)
    [target] = [w for w in pool._workers if w is not None and w.pid == first]
    connect, stop = _Worker.connect, _Worker.stop
    started: list[int] = []
    connected = asyncio.Event()

    async def slow(worker: _Worker) -> None:
        started.append(worker.pid)
        await connect(worker)
        connected.set()
        await asyncio.sleep(0.5)

    def failing(worker: _Worker, reason: str | None = None) -> None:
        stop(worker, reason)
        if worker is target:
            raise OSError("the first worker would not stop")

    monkeypatch.setattr(_Worker, "connect", slow)
    monkeypatch.setattr(_Worker, "stop", failing)
    starting = asyncio.ensure_future(pool.call("nap", {}, CALL, 5))
    try:
        async with asyncio.timeout(5):
            await connected.wait()
            with pytest.raises(OSError, match="^the first worker would not stop$"):
                await pool.close()
        [spawned] = started
        assert reaped(spawned)
    finally:
        await asyncio.wait([busy, starting])
        monkeypatch.undo()
        pool._drop(CLOSED)


async def test_close_from_another_loop_that_stops_mid_start_still_stops_that_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect = _Worker.connect
    started: list[int] = []
    connected = threading.Event()

    async def slow(worker: _Worker) -> None:
        started.append(worker.pid)
        await connect(worker)
        connected.set()
        await asyncio.sleep(30)

    unraised: list[BaseException | None] = []
    monkeypatch.setattr(sys, "unraisablehook", lambda hook: unraised.append(hook.exc_value))
    monkeypatch.setattr(_Worker, "connect", slow)
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    calling: Any = None
    try:
        calling = asyncio.run_coroutine_threadsafe(pool.call("nap", {}, CALL, None), loop)
        assert connected.wait(5)
        closing = asyncio.ensure_future(pool.close())
        await asyncio.sleep(0.3)
        loop.call_soon_threadsafe(loop.stop)
        async with asyncio.timeout(5):
            await closing
        [spawned] = started
        assert reaped(spawned)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()
        del calling
        gc.collect()
        monkeypatch.undo()
        pool._drop(CLOSED)
    assert [error for error in unraised if isinstance(error, RuntimeError)] == []


@pytest.mark.parametrize("how", ["closed", "cancelled", "released"])
async def test_a_stop_that_fails_on_a_loop_that_stops_mid_start_reaches_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, how: str
) -> None:
    caplog.set_level(logging.ERROR, logger="rfnry_engine.workers.pool")
    connect, stop = _Worker.connect, _Worker.stop
    started: list[int] = []
    connected, refused = threading.Event(), threading.Event()

    async def slow(worker: _Worker) -> None:
        started.append(worker.pid)
        await connect(worker)
        connected.set()
        await asyncio.sleep(30)

    pool = WorkerPool([script(tmp_path, "nap", PID)], processes=2)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    calls: list[Any] = []
    try:
        first = asyncio.run_coroutine_threadsafe(pool.call("nap", {}, CALL, 5), loop)
        worker = pid(await asyncio.wrap_future(first))
        [target] = [held for held in pool._workers if held is not None]

        def failing(held: _Worker, reason: str | None = None) -> None:
            stop(held, reason)
            if held is target:
                refused.set()
                raise OSError("the first worker would not stop")

        monkeypatch.setattr(_Worker, "connect", slow)
        monkeypatch.setattr(_Worker, "stop", failing)
        sleeps: list[JsonValue] = [{"sleep": 5}, {}]
        for args in sleeps:
            calls.append(asyncio.run_coroutine_threadsafe(pool.call("nap", args, CALL, None), loop))
        assert connected.wait(5)
        closing = asyncio.ensure_future(pool.close()) if how != "released" else None
        if closing is None:
            pool.release()
        assert await asyncio.to_thread(refused.wait, 5)
        if closing is not None and how == "cancelled":
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        if closing is not None and how == "closed":
            with pytest.raises(OSError, match="^the first worker would not stop$"):
                async with asyncio.timeout(5):
                    await closing
        logged = [r.getMessage() for r in caplog.records if r.name == "rfnry_engine.workers.pool"]
        assert logged == (
            []
            if how == "closed"
            else ["workers/pool: could not stop a worker: the first worker would not stop"]
        )
        [spawned] = started
        assert reaped(worker) and (how != "closed" or reaped(spawned))
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        monkeypatch.undo()
        pool._drop(CLOSED, starting=True)
        loop.close()
        del calls
        gc.collect()


async def test_a_worker_close_reaped_while_its_loop_was_stopped_is_never_signalled_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect, killpg = _Worker.connect, os.killpg
    started: list[int] = []
    signalled: list[int] = []
    connected = threading.Event()

    async def slow(worker: _Worker) -> None:
        started.append(worker.pid)
        await connect(worker)
        connected.set()
        await asyncio.sleep(0.3)

    def watching(group: int, sent: int) -> None:
        signalled.append(group)
        killpg(group, sent)

    monkeypatch.setattr(_Worker, "connect", slow)
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        calling = asyncio.run_coroutine_threadsafe(pool.call("nap", {}, CALL, 10), loop)
        assert connected.wait(5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        async with asyncio.timeout(5):
            await pool.close()
        [spawned] = started
        assert reaped(spawned)
        monkeypatch.setattr(os, "killpg", watching)
        thread = threading.Thread(target=loop.run_forever)
        thread.start()
        with pytest.raises(WorkerError, match="the worker for nap stopped: it was closed"):
            async with asyncio.timeout(5):
                await asyncio.wrap_future(calling)
        assert spawned not in signalled
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        monkeypatch.undo()
        pool._drop(CLOSED)
        loop.close()


def test_a_worker_started_while_its_pool_closes_is_stopped(tmp_path: Path) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    _, children = held()
    loop = asyncio.new_event_loop()
    try:
        call = loop.create_task(pool.call("nap", {}, CALL, 5))
        loop.run_until_complete(asyncio.sleep(0))
        assert len(held()[1]) == len(children) + 1
        on_a_loop_of_its_own(pool.close())
        with pytest.raises(WorkerError, match="the worker for nap stopped: it was closed"):
            loop.run_until_complete(call)
    finally:
        on_a_loop_of_its_own(pool.close())
        loop.close()
    assert held()[1] == children


def test_a_pool_that_follows_one_event_loop_after_another_holds_no_more_files_or_processes(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)], threads=2)
    try:
        on_a_loop_of_its_own(pool.call("nap", {}, CALL, 5))
        fds, children = held()
        with no_collector():
            pids = [pid(on_a_loop_of_its_own(pool.call("nap", {}, CALL, 5))) for _ in range(50)]
            after, now = held()
    finally:
        on_a_loop_of_its_own(pool.close())
    assert after <= fds + 3 and now == children == ["S"]
    assert len(set(pids)) == 50 and all(reaped(old) for old in pids)


def test_closing_pools_from_another_event_loop_holds_no_more_files_or_processes(
    tmp_path: Path,
) -> None:
    nap = script(tmp_path, "nap", PID)
    fds, children = held()
    with no_collector():
        for _ in range(50):
            pool = WorkerPool([nap])
            last = pid(on_a_loop_of_its_own(pool.call("nap", {}, CALL, 5)))
            on_a_loop_of_its_own(pool.close())
            assert reaped(last)
        after, now = held()
    assert after <= fds + 3 and now == children == []


def test_a_call_left_on_an_event_loop_closed_under_it_does_not_hold_the_next_loop(
    tmp_path: Path,
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    first = asyncio.new_event_loop()
    first.set_exception_handler(lambda loop, context: None)
    left = first.create_task(pool.call("nap", {"sleep": 30}, CALL, None))
    first.run_until_complete(asyncio.sleep(0.5))
    first.close()
    try:
        assert not left.done()
        second = pid(on_a_loop_of_its_own(pool.call("nap", {}, CALL, 5)))
        on_a_loop_of_its_own(pool.close())
        assert reaped(second)
    finally:
        on_a_loop_of_its_own(pool.close())


@pytest.mark.parametrize("ending", ["os._exit(3)", "os.kill(os.getpid(), 9)"])
async def test_a_worker_that_dies_is_reaped_before_its_calls_fail_and_its_last_words_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, ending: str
) -> None:
    caplog.set_level(logging.DEBUG, logger="rfnry_engine.workers.pool")
    body = (
        "import os, sys\n"
        "def run(args, call):\n"
        "    if args.get('die'):\n"
        "        print('last words', file=sys.stderr, flush=True)\n"
        f"        {ending}\n"
        "    return os.getpid()\n"
    )
    pool = WorkerPool([script(tmp_path, "dying", body)])
    try:
        first = pid(await pool.call("dying", {}, CALL, 5))
        with pytest.raises(WorkerError, match="the worker for dying stopped"):
            await pool.call("dying", {"die": True}, CALL, 5)
        assert reaped(first) and "last words" in caplog.text
        assert loop_errors(caplog) == []
    finally:
        await pool.close()


async def test_a_call_to_a_worker_killed_from_outside_fails_with_its_signal_and_reaps_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)])
    try:
        first = pid(await pool.call("nap", {}, CALL, 5))
        os.kill(first, signal.SIGKILL)
        time.sleep(0.1)
        with pytest.raises(WorkerError) as raised:
            await pool.call("nap", {}, CALL, 2)
        assert str(raised.value) == "workers/pool: the worker for nap stopped: killed by signal 9"
        assert reaped(first) and loop_errors(caplog) == []
        assert pid(await pool.call("nap", {}, CALL, 5)) != first
    finally:
        await pool.close()


async def test_a_retired_worker_is_reaped_before_the_next_call_returns(tmp_path: Path) -> None:
    pool = WorkerPool([script(tmp_path, "nap", PID)], threads=1)
    try:
        first = pid(await pool.call("nap", {}, CALL, 5))
        with pytest.raises(TimeoutError):
            await pool.call("nap", {"sleep": 30}, CALL, 0.2)
        assert pid(await pool.call("nap", {}, CALL, 5)) != first and reaped(first)
    finally:
        await pool.close()


async def test_close_reaps_every_worker_before_it_returns(tmp_path: Path) -> None:
    pool = WorkerPool([script(tmp_path, "wait", WAIT)], processes=2)
    try:
        pids = await asyncio.gather(*(pool.call("wait", {"sleep": 0.2}, CALL, 5) for _ in range(2)))
    finally:
        await pool.close()
    assert len(set(pids)) == 2 and all(reaped(pid(value)) for value in pids)


GENERATION: dict[str, Any] = {"kind": "generation", "model": "m", "instructions": "a.md"}
PICKS_PID: dict[str, Any] = {**GENERATION, "tools": {"include": ["pid"]}}
PID_TOOL = {"pid": {"module": "tools/pid.py", "description": "Its process"}}
PID_FILES = {"a.md": "Work.\n", "tools/pid.py": PID}
HOLD = (
    "import time\n"
    "from pathlib import Path\n"
    "def run(args, call):\n"
    "    Path(args['started']).write_text('')\n"
    "    while not Path(args['go']).exists():\n"
    "        time.sleep(0.01)\n"
    "    return 'answered'\n"
)
MEET = (
    "import threading\n"
    "both = threading.Barrier(2)\n"
    "def run(args, call):\n"
    "    both.wait(args['seconds'])\n"
    "    return 'met'\n"
)
LONG = "def run(args, call):\n    return 'x' * args['size']\n"


def _engine(folder: Path, **limits: Any) -> Engine:
    return Engine(folder, models={"m": Scripted()}, **limits)


def _surface(engine: Engine, agent: str = "a", mode: str | None = None) -> EngineTools:
    return EngineTools(engine, agent=agent, scope="acme", mode=mode)


async def test_no_workers_section_is_one_default_worker(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path, {"a": PICKS_PID, "b": PICKS_PID}, tools=PID_TOOL, files=PID_FILES
    )
    async with _engine(folder) as engine:
        answers = [await _surface(engine, name).dispatch("pid", {}) for name in ("a", "b")]
        assert list(engine._workers._pools) == ["default"]
        assert len(engine._workers.pool(engine._spec("a"))._workers) == 1
    assert answers[0] == answers[1]


async def test_a_mode_picks_its_own_worker(tmp_path: Path) -> None:
    agent = {
        **PICKS_PID,
        "worker": "copilot",
        "modes": {"voice": {"worker": "voice"}, "email": {"description": "Email"}},
    }
    folder = write_folder(
        tmp_path,
        {"a": agent},
        workers={"copilot": {"processes": 1}, "voice": {"processes": 1}},
        tools=PID_TOOL,
        files=PID_FILES,
    )
    async with _engine(folder) as engine:
        entry, email, voice = [
            await _surface(engine, mode=mode).dispatch("pid", {})
            for mode in (None, "email", "voice")
        ]
    assert entry == email
    assert voice != entry


async def test_a_worker_holds_only_the_tools_that_pick_it(tmp_path: Path) -> None:
    tools = {
        name: {"module": f"tools/{name}.py", "description": name}
        for name in ("front", "back", "both", "extra")
    }
    agents = {
        "a": {**GENERATION, "worker": "one", "tools": {"include": ["front", "both"]}},
        "b": {
            **GENERATION,
            "worker": "two",
            "tools": {"include": ["back", "both"]},
            "modes": {"chat": {"worker": "three", "tools": {"include": ["extra"]}}},
        },
        "t": {
            "kind": "decision",
            "model": "judge",
            "instructions": "a.md",
            "worker": "one",
            "questions": ["q/plain.json"],
            "modes": {"chat": {"worker": "two", "questions": ["q/order.json"]}},
        },
    }
    folder = write_folder(
        tmp_path,
        agents,
        workers={name: {"processes": 1} for name in ("one", "two", "three")},
        tools=tools,
        files={
            "a.md": "Work.\n",
            **{f"tools/{name}.py": PID for name in tools},
            "q/plain.json": '{"type": "check", "instructions": "?", "yes": "y", "no": "n"}',
            "q/order.json": '{"type": "choice", "instructions": "?"}',
            "q/order.py": "def run(input, call):\n    return {'options': {'a': 'A', 'b': 'B'}}\n",
        },
    )
    async with _engine(folder) as engine:
        held = {
            name: [item["key"] for item in cast(Any, pool._scripts)["scripts"]]
            for name, pool in engine._workers._pools.items()
        }
    assert held == {
        "one": ["tools/both", "tools/front"],
        "two": ["q/order", "tools/back", "tools/both"],
        "three": ["tools/back", "tools/both", "tools/extra"],
    }


async def test_a_killed_process_leaves_another_worker_alone(tmp_path: Path) -> None:
    agent = {
        **GENERATION,
        "worker": "copilot",
        "tools": {"include": ["hold"]},
        "modes": {"voice": {"worker": "voice"}},
    }
    folder = write_folder(
        tmp_path,
        {"a": agent},
        workers={"copilot": {"processes": 1}, "voice": {"processes": 1}},
        tools={"hold": {"module": "tools/hold.py", "description": "Holds"}},
        files={"a.md": "Work.\n", "tools/hold.py": HOLD},
    )
    go = tmp_path / "go"

    def held(started: Path) -> dict[str, JsonValue]:
        return {"started": str(started), "go": str(go)}

    async def begun(*marks: Path) -> None:
        async with asyncio.timeout(10):
            while not all(mark.exists() for mark in marks):
                await asyncio.sleep(0.01)

    async with _engine(folder) as engine:
        copilot_began, voice_began = tmp_path / "copilot", tmp_path / "voice"
        copilot = _surface(engine)
        copilot_call = asyncio.ensure_future(copilot.dispatch("hold", held(copilot_began)))
        voice_call = asyncio.ensure_future(
            _surface(engine, mode="voice").dispatch("hold", held(voice_began))
        )
        await begun(copilot_began, voice_began)
        [process] = engine._workers.pool(engine._spec("a"))._workers
        assert process is not None
        os.kill(process.pid, signal.SIGKILL)
        with pytest.raises(WorkerError, match="killed by signal 9"):
            await copilot_call
        assert not voice_call.done()
        go.touch()
        assert await voice_call == "answered"
        assert await copilot.dispatch("hold", held(copilot_began)) == "answered"


async def test_a_process_starts_on_the_first_tool_call(tmp_path: Path) -> None:
    agents = {"a": {**PICKS_PID, "worker": "used"}, "b": {**PICKS_PID, "worker": "idle"}}
    folder = write_folder(
        tmp_path,
        agents,
        workers={"used": {"processes": 1}, "idle": {"processes": 2}},
        tools=PID_TOOL,
        files=PID_FILES,
    )
    async with _engine(folder) as engine:
        used = engine._workers.pool(engine._spec("a"))
        idle = engine._workers.pool(engine._spec("b"))
        surface = _surface(engine)
        assert used._workers == [None] and idle._workers == [None, None]
        await surface.dispatch("pid", {})
        assert used._workers[0] is not None and idle._workers == [None, None]


def test_more_processes_than_the_ceiling_are_refused(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"a": {**GENERATION, "worker": "x"}, "b": GENERATION},
        workers={"x": {"processes": 4}},
        files={"a.md": "Work.\n"},
    )
    _engine(folder, max_worker_processes=5)
    with pytest.raises(ConfigurationError) as caught:
        _engine(folder, max_worker_processes=4)
    assert str(caught.value) == (
        "engine: the workers of this folder run 5 processes in all, over max_worker_processes=4; "
        "lower processes: under workers:, or raise Engine(max_worker_processes=)"
    )


def test_the_default_ceiling_is_sixteen_processes(tmp_path: Path) -> None:
    fits = write_folder(
        tmp_path,
        {"a": {**GENERATION, "worker": "x"}},
        workers={"x": {"processes": 16}},
        files={"a.md": "Work.\n"},
    )
    _engine(fits)
    over = write_folder(
        tmp_path,
        {"a": {**GENERATION, "worker": "x"}},
        workers={"x": {"processes": 17}},
        files={"a.md": "Work.\n"},
    )
    with pytest.raises(ConfigurationError, match="17 processes in all, over .*=16;"):
        _engine(over)


def test_an_entry_that_only_holds_modes_does_not_use_the_default_worker(tmp_path: Path) -> None:
    agent = {
        "kind": "generation",
        "model": "m",
        "modes": {"chat": {"instructions": "a.md", "worker": "x"}},
    }
    folder = write_folder(
        tmp_path,
        {"a": agent},
        workers={"x": {"processes": 1}},
        files={"a.md": "Work.\n"},
    )
    engine = _engine(folder, max_worker_processes=1)
    assert list(engine._workers._pools) == ["x"]


async def test_threads_and_max_result_bytes_reach_the_worker(tmp_path: Path) -> None:
    tools = {
        "meet": {"module": "tools/meet.py", "description": "Meets"},
        "long": {"module": "tools/long.py", "description": "Long"},
    }
    picks = {**GENERATION, "tools": {"include": ["meet", "long"]}}
    folder = write_folder(
        tmp_path,
        {"pair": {**picks, "worker": "pair"}, "solo": {**picks, "worker": "solo"}},
        workers={"pair": {"threads": 2}, "solo": {"threads": 1, "max_result_bytes": 100}},
        tools=tools,
        files={"a.md": "Work.\n", "tools/meet.py": MEET, "tools/long.py": LONG},
    )
    async with _engine(folder) as engine:
        pair, solo = _surface(engine, "pair"), _surface(engine, "solo")
        met = await asyncio.gather(*(pair.dispatch("meet", {"seconds": 30}) for _ in range(2)))
        apart = await asyncio.gather(
            *(solo.dispatch("meet", {"seconds": 0.2}) for _ in range(2)), return_exceptions=True
        )
        assert await pair.dispatch("long", {"size": 200}) == "x" * 200
        assert await solo.dispatch("long", {"size": 10}) == "x" * 10
        with pytest.raises(WorkerError, match="over the limit of 100 bytes"):
            await solo.dispatch("long", {"size": 200})
    assert met == ["met", "met"]
    assert all(isinstance(result, WorkerError) for result in apart)


@pytest.mark.parametrize(
    ("engine_limit", "worker_limit", "applied"),
    [(50, 100, 50), (100, 50, 50), (None, 50, 50), (60, None, 60), (None, None, None)],
)
def test_max_result_bytes_lowers_the_engines_limit_and_never_raises(
    tmp_path: Path, engine_limit: int | None, worker_limit: int | None, applied: int | None
) -> None:
    declared = {} if worker_limit is None else {"max_result_bytes": worker_limit}
    folder = write_folder(
        tmp_path,
        {"a": {**GENERATION, "worker": "x"}},
        workers={"x": {"processes": 1, **declared}},
        files={"a.md": "Work.\n"},
    )
    engine = _engine(folder, worker_result_bytes=engine_limit)
    pool = engine._workers.pool(engine._spec("a"))
    assert cast(Any, pool._scripts)["limit"] == applied


async def test_a_delegated_run_uses_its_own_agents_worker(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "lead": {**GENERATION, "worker": "front", "orchestration": {"delegate_to": ["helper"]}},
            "helper": {**PICKS_PID, "model": "h", "worker": "back"},
        },
        workers={"front": {"processes": 1}, "back": {"processes": 1}},
        tools=PID_TOOL,
        files=PID_FILES,
    )
    lead = Scripted(delegating("helper"), text_reply("done"))
    helper = Scripted(
        Reply(tool_calls=[ToolCall(id="c1", name="pid", args={})], stop_reason="tool_use"),
        text_reply("done"),
    )
    async with Engine(folder, models={"m": lead, "h": helper}) as engine:
        await engine.run("lead", "acme", "x")
        front = engine._workers.pool(engine._spec("lead"))
        [process] = engine._workers.pool(engine._spec("helper"))._workers
        assert front._workers == [None]
        assert process is not None and str(process.pid) in str(helper.calls[1][1])
