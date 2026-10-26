import asyncio
import contextlib
import datetime
import gc
import logging
import os
import signal
import time
from collections import Counter
from collections.abc import Awaitable, Iterator
from pathlib import Path

import pytest
from pydantic import JsonValue

from rfnry_engine.errors import InputError, WorkerError
from rfnry_engine.workers.pool import CAP, Script, WorkerPool

CALL = {"scope": "acme"}
PID = "import os, time\ndef run(args, call):\n    time.sleep(args.get('sleep', 0))\n    return os.getpid()\n"
WAIT = "import asyncio, os\nasync def run(args, call):\n    await asyncio.sleep(args.get('sleep', 0))\n    return os.getpid()\n"
NOISE = "import os, sys\ndef run(args, call):\n    print('x' * 200 * 1024, end='', file=sys.stderr, flush=True)\n    return os.getpid()\n"


def script(tmp_path: Path, name: str, body: str) -> Script:
    path = tmp_path / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    helpers = {p.stem: p.read_text(encoding="utf-8") for p in path.parent.glob("_*.py")}
    return Script(name, path, body, helpers, "async def run" in body)


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
        for child in task.read_text().split():
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
    assert [r for r in caplog.records if r.name == "rfnry_engine.workers.pool"] == []


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
    pool = WorkerPool([Script("tools/given", path, body, {"_client": "NAME = 'given'\n"}, False)])
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
