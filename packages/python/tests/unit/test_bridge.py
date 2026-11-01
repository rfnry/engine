import asyncio
import contextlib
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rfnry_engine.tools import bridge as bridging
from rfnry_engine.tools.bridge import CALLS, Bridge

BAD = "tools/bridge: a request is one JSON line: " + '{"key": …, "tool": …, "args": {…}}'
WRONG = "tools/bridge: the key is not this command's"


class Seen:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, args))
        return {"result": args.get("x", args.get("q"))}


@pytest.fixture
def root(short_root: Path) -> Path:
    return short_root


@pytest.fixture
def seen() -> Seen:
    return Seen()


@pytest.fixture
async def bridge(root: Path, seen: Seen) -> AsyncIterator[Bridge]:
    b = await Bridge.open(root, seen)
    yield b
    await b.close()


async def ask(b: Bridge, raw: bytes, eof: bool = False) -> dict[str, Any]:
    reader, writer = await asyncio.open_unix_connection(b.path, limit=1 << 26)
    writer.write(raw)
    if eof:
        writer.write_eof()
    async with asyncio.timeout(5):
        line = await reader.readline()
    writer.close()
    reply: dict[str, Any] = json.loads(line)
    return reply


def request(b: Bridge, **fields: Any) -> bytes:
    body = {"key": b.key, "tool": "echo", "args": {"x": 1}} | fields
    return json.dumps({k: v for k, v in body.items() if v is not ...}).encode() + b"\n"


async def test_env_names_the_socket_and_the_key(bridge: Bridge, root: Path) -> None:
    assert bridge.env == {"RFNRY_TOOLS": str(bridge.path), "RFNRY_TOOLS_KEY": bridge.key}
    assert bridge.path.parent == root
    assert re.match(r"^\.rfnry-[0-9a-f]{8}\.sock$", bridge.path.name)
    assert re.match(r"^[0-9a-f]{32}$", bridge.key)


async def test_a_request_gets_its_answer(bridge: Bridge, seen: Seen) -> None:
    assert await ask(bridge, request(bridge)) == {"result": 1}
    assert seen.calls == [("echo", {"x": 1})]


async def test_a_request_without_a_newline_is_read_at_end_of_input(bridge: Bridge) -> None:
    assert await ask(bridge, request(bridge).rstrip(b"\n"), eof=True) == {"result": 1}


@pytest.mark.parametrize("key", ["wrong", ..., 1])
async def test_a_wrong_or_missing_key_runs_nothing(bridge: Bridge, seen: Seen, key: Any) -> None:
    assert await ask(bridge, request(bridge, key=key)) == {"error": WRONG}
    assert seen.calls == []


@pytest.mark.parametrize(
    "raw",
    [b"not json\n", b"[1]\n", b"\xff\n", b"", "tool", "args", "no args"],
)
async def test_a_malformed_request_is_refused(bridge: Bridge, seen: Seen, raw: Any) -> None:
    if raw == "tool":
        raw = request(bridge, tool=1)
    elif raw == "args":
        raw = request(bridge, args=[])
    elif raw == "no args":
        raw = request(bridge, args=...)
    assert await ask(bridge, raw, eof=True) == {"error": BAD}
    assert seen.calls == []


async def test_a_line_over_the_cap_is_refused(bridge: Bridge, seen: Seen) -> None:
    reader, writer = await asyncio.open_unix_connection(bridge.path, limit=1 << 26)
    writer.write(b"x" * (2 << 20) + b"\n")
    async with asyncio.timeout(5):
        line = await reader.readline()
    writer.close()
    assert json.loads(line) == {"error": "tools/bridge: a request is at most 1048576 bytes"}
    assert seen.calls == []


async def test_two_bridges_in_one_root_refuse_each_other_s_key(root: Path, seen: Seen) -> None:
    a = await Bridge.open(root, seen)
    b = await Bridge.open(root, seen)
    try:
        assert a.path != b.path
        assert await ask(a, request(b)) == {"error": WRONG}
        assert await ask(b, request(b)) == {"result": 1}
    finally:
        await a.close()
        await b.close()


async def test_close_removes_the_socket_and_refuses_new_connections(root: Path, seen: Seen) -> None:
    b = await Bridge.open(root, seen)
    await b.close()
    assert not b.path.exists()
    with pytest.raises(OSError):
        await asyncio.open_unix_connection(b.path)


async def test_close_keeps_a_file_put_at_the_socket_s_path(root: Path, seen: Seen) -> None:
    b = await Bridge.open(root, seen)
    b.path.unlink()
    b.path.write_text("mine", encoding="utf-8")
    await b.close()
    assert b.path.read_text(encoding="utf-8") == "mine"


async def test_open_closes_its_server_when_the_socket_is_gone_at_once(
    root: Path, seen: Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gone(path: Path, *, follow_symlinks: bool) -> None:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(bridging, "os", SimpleNamespace(stat=gone))
    with pytest.raises(FileNotFoundError):
        await Bridge.open(root, seen)
    [path] = root.glob(".rfnry-*.sock")
    with pytest.raises(ConnectionRefusedError):
        await asyncio.open_unix_connection(path)


async def test_close_cancels_a_request_in_flight(root: Path) -> None:
    started = asyncio.Event()
    record: list[str] = []

    async def answer(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            record.append("cancelled")
            raise
        return {}

    b = await Bridge.open(root, answer)
    _, writer = await asyncio.open_unix_connection(b.path)
    writer.write(request(b))
    async with asyncio.timeout(1):
        await started.wait()
        await b.close()
    assert record == ["cancelled"]
    writer.close()


@pytest.mark.parametrize(
    "cancelled", [False, True], ids=["entered-uncancelled", "entered-cancelled"]
)
async def test_close_passes_each_further_cancel_on(root: Path, cancelled: bool) -> None:
    started = asyncio.Event()
    released = asyncio.Event()
    cancels: list[int] = []

    async def answer(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        started.set()
        while not released.is_set():
            try:
                await released.wait()
            except asyncio.CancelledError:
                cancels.append(len(cancels) + 1)
        return {}

    async def seen(count: int) -> None:
        while len(cancels) < count:
            await asyncio.sleep(0)

    b = await Bridge.open(root, answer)
    raised: list[bool] = []

    async def close() -> None:
        try:
            await b.close()
        except asyncio.CancelledError:
            raised.append(True)
            raise

    async def closer() -> None:
        if cancelled:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await close()
                raise
        await close()

    _, writer = await asyncio.open_unix_connection(b.path)
    writer.write(request(b))
    async with asyncio.timeout(2):
        await started.wait()
        closing = asyncio.create_task(closer())
        try:
            if cancelled:
                await asyncio.sleep(0)
                closing.cancel()
            await seen(1)
            await asyncio.sleep(0.05)
            first = len(cancels)
            closing.cancel()
            await seen(2)
            closing.cancel()
            await seen(3)
        finally:
            released.set()
            await asyncio.wait([closing])
    writer.close()
    assert first == 1 and cancels == [1, 2, 3] and closing.cancelled() and raised == [True]


async def test_a_silent_client_does_not_hold_close(
    root: Path, seen: Seen, caplog: pytest.LogCaptureFixture
) -> None:
    b = await Bridge.open(root, seen)
    _, writer = await asyncio.open_unix_connection(b.path)
    async with asyncio.timeout(1):
        while b._open == 0:
            await asyncio.sleep(0)
    with caplog.at_level(logging.DEBUG):
        async with asyncio.timeout(1):
            await b.close()
    writer.close()
    assert [r for r in caplog.records if r.levelno > logging.DEBUG] == []


async def test_a_client_gone_before_the_reply_is_harmless(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    entered = asyncio.Event()
    gone = asyncio.Event()

    async def answer(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        await gone.wait()
        return {"result": "x" * 1_000_000}

    b = await Bridge.open(root, answer)
    with caplog.at_level(logging.DEBUG):
        _, writer = await asyncio.open_unix_connection(b.path)
        writer.write(request(b))
        await writer.drain()
        async with asyncio.timeout(5):
            await entered.wait()
            writer.close()
            await writer.wait_closed()
            gone.set()
            while b._handlers:
                await asyncio.sleep(0.01)
        await b.close()
    assert [r for r in caplog.records if r.levelno > logging.DEBUG] == []


async def test_a_large_reply_arrives_whole(root: Path) -> None:
    async def answer(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"result": "x" * 3_000_000}

    b = await Bridge.open(root, answer)
    try:
        reply = await ask(b, request(b))
    finally:
        await b.close()
    assert len(reply["result"]) == 3_000_000


async def test_non_ascii_round_trips(root: Path) -> None:
    got: list[dict[str, Any]] = []

    async def answer(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        got.append(args)
        return {"result": "naïve"}

    b = await Bridge.open(root, answer)
    try:
        reader, writer = await asyncio.open_unix_connection(b.path)
        writer.write(request(b, args={"q": "café"}))
        async with asyncio.timeout(5):
            line = await reader.readline()
        writer.close()
    finally:
        await b.close()
    assert got == [{"q": "café"}]
    assert "naïve" in line.decode()
    assert json.loads(line) == {"result": "naïve"}


async def test_open_raises_when_the_path_is_too_long(tmp_path: Path, seen: Seen) -> None:
    root = tmp_path / ("d" * 120)
    root.mkdir()
    with pytest.raises(OSError):
        await Bridge.open(root, seen)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("surrogate", ["key", "nested", "nested_key"])
async def test_hostile_requests_get_a_reply_and_log_nothing(
    bridge: Bridge, seen: Seen, caplog: pytest.LogCaptureFixture, surrogate: str
) -> None:
    raw = {
        "key": b'{"key": "\\ud800", "tool": "echo", "args": {}}\n',
        "nested": b"[" * 200_000 + b"\n",
        "nested_key": b'{"key": "'
        + bridge.key.encode()
        + b'", "tool": "echo", "args": '
        + b"[" * 200_000
        + b"\n",
    }[surrogate]
    with caplog.at_level(logging.DEBUG):
        reply = await ask(bridge, raw)
        await asyncio.sleep(0.05)
    assert reply == ({"error": WRONG} if surrogate == "key" else {"error": DEEP})
    assert seen.calls == []
    assert [r for r in caplog.records if r.levelno > logging.DEBUG] == []


async def test_a_lone_surrogate_in_a_reply_is_escaped(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def answer(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"error": f"unknown tool {tool}"}

    b = await Bridge.open(root, answer)
    try:
        with caplog.at_level(logging.DEBUG):
            reply = await ask(b, request(b, tool="\ud800"))
            await asyncio.sleep(0.05)
    finally:
        await b.close()
    assert reply == {"error": "unknown tool \ud800"}
    assert [r for r in caplog.records if r.levelno > logging.DEBUG] == []


async def test_a_client_that_never_reads_a_large_reply_does_not_hold_close(root: Path) -> None:
    async def answer(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"result": "x" * 8_000_000}

    b = await Bridge.open(root, answer)
    _, writer = await asyncio.open_unix_connection(b.path)
    writer.write(request(b))
    await writer.drain()
    await asyncio.sleep(0.1)
    async with asyncio.timeout(1):
        await b.close()
    writer.close()


async def test_a_reply_made_after_close_began_does_not_wait_for_a_client_that_never_reads(
    root: Path,
) -> None:
    started = asyncio.Event()

    async def answer(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        started.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()
        return {"result": "x" * 8_000_000}

    b = await Bridge.open(root, answer)
    _, writer = await asyncio.open_unix_connection(b.path)
    writer.write(request(b))
    async with asyncio.timeout(5):
        await started.wait()
    closing = asyncio.ensure_future(b.close())
    done, _ = await asyncio.wait([closing], timeout=1)
    writer.close()
    await closing
    assert done == {closing}


class Held:
    def __init__(self) -> None:
        self.calls = 0
        self.gates = [asyncio.Event() for _ in range(CALLS)]
        self.entered = asyncio.Event()

    def release(self, *which: int) -> None:
        for i in which or range(CALLS):
            self.gates[i].set()

    async def __call__(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.calls == CALLS:
            self.entered.set()
        if args["x"] < CALLS:
            await self.gates[args["x"]].wait()
        return {"result": args["x"]}


async def hold(b: Bridge, n: int) -> list[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    held = []
    for i in range(n):
        reader, writer = await asyncio.open_unix_connection(b.path, limit=1 << 26)
        writer.write(request(b, args={"x": i}))
        held.append((reader, writer))
    return held


async def test_a_connection_past_the_cap_is_refused_at_once(root: Path) -> None:
    held = Held()
    b = await Bridge.open(root, held)
    try:
        connections = await hold(b, CALLS)
        async with asyncio.timeout(10):
            await held.entered.wait()
        reader, writer = await asyncio.open_unix_connection(b.path, limit=1 << 26)
        async with asyncio.timeout(1):
            line = await reader.readline()
        assert json.loads(line) == {
            "error": f"tools/bridge: at most {CALLS} calls at once; wait for one to finish"
        }
        writer.close()
        assert held.calls == CALLS
        held.release()
        async with asyncio.timeout(10):
            replies = [json.loads(await reader.readline()) for reader, _ in connections]
        assert replies == [{"result": i} for i in range(CALLS)]
        for _, writer in connections:
            writer.close()
    finally:
        held.release()
        await b.close()


async def test_a_slot_frees_when_a_call_ends(root: Path) -> None:
    held = Held()
    b = await Bridge.open(root, held)
    try:
        connections = await hold(b, CALLS)
        async with asyncio.timeout(10):
            await held.entered.wait()
        held.release(0)
        async with asyncio.timeout(10):
            assert json.loads(await connections[0][0].readline()) == {"result": 0}
        async with asyncio.timeout(1):
            while b._open >= CALLS:
                await asyncio.sleep(0.01)
        assert await ask(b, request(b, args={"x": CALLS})) == {"result": CALLS}
        assert held.calls == CALLS + 1
        assert all(not reader.at_eof() for reader, _ in connections[1:])
        held.release()
        async with asyncio.timeout(10):
            replies = [json.loads(await reader.readline()) for reader, _ in connections[1:]]
        assert replies == [{"result": i} for i in range(1, CALLS)]
        for _, writer in connections:
            writer.close()
    finally:
        held.release()
        await b.close()


CALLED = "tools/bridge: this command made its {} calls; no more are answered"
SENT = "tools/bridge: this command sent its {} bytes of calls; no more are answered"
DEEP = "tools/bridge: a call's args nest deeper than 128 levels"
CAPPED = "tools/bridge: at most 1 calls at once; wait for one to finish"


def deep(b: Bridge, depth: int) -> bytes:
    head = json.dumps({"key": b.key, "tool": "echo"})[:-1].encode()
    return head + b', "args": {"x": ' + b"[" * (depth - 1) + b"]" * (depth - 1) + b"}}\n"


def nest(levels: int) -> list[Any]:
    value: list[Any] = []
    for _ in range(levels - 1):
        value = [value]
    return value


async def test_the_budgets_are_the_bench_s(bridge: Bridge) -> None:
    assert (bridging.COMMAND_CALLS, bridging.COMMAND_BYTES, bridging.DEPTH) == (
        10_000,
        2_097_152,
        128,
    )


async def test_a_command_past_its_call_budget_is_refused(
    bridge: Bridge, seen: Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridging, "COMMAND_CALLS", 3)
    assert await ask(bridge, b"not json\n") == {"error": BAD}
    replies = [await ask(bridge, request(bridge, args={"x": i})) for i in range(5)]
    assert replies == [{"result": i} for i in range(3)] + [{"error": CALLED.format(3)}] * 2
    assert seen.calls == [("echo", {"x": i}) for i in range(3)]


async def test_a_command_past_its_byte_budget_is_refused(
    bridge: Bridge, seen: Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    line, bad = request(bridge), b"not json\n"
    budget = len(bad) + 2 * len(line)
    monkeypatch.setattr(bridging, "COMMAND_BYTES", budget)
    assert await ask(bridge, bad) == {"error": BAD}
    assert [await ask(bridge, line) for _ in range(2)] == [{"result": 1}] * 2
    assert await ask(bridge, line) == {"error": SENT.format(budget)}
    assert await ask(bridge, b"[\n") == {"error": SENT.format(budget)}
    assert seen.calls == [("echo", {"x": 1})] * 2


async def test_a_connection_refused_at_the_cap_counts_toward_no_budget(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    got: list[int] = []

    async def answer(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        got.append(args["x"])
        if args["x"] == 0:
            await gate.wait()
        return {"result": args["x"]}

    b = await Bridge.open(root, answer)
    try:
        first = request(b, args={"x": 0})
        monkeypatch.setattr(bridging, "CALLS", 1)
        monkeypatch.setattr(bridging, "COMMAND_CALLS", 2)
        monkeypatch.setattr(bridging, "COMMAND_BYTES", 2 * len(first))
        reader, writer = await asyncio.open_unix_connection(b.path)
        writer.write(first)
        async with asyncio.timeout(5):
            while not got:
                await asyncio.sleep(0.01)
        for _ in range(3):
            over, gone = await asyncio.open_unix_connection(b.path)
            with contextlib.suppress(ConnectionError):
                gone.write(request(b, args={"x": 9}))
                await gone.drain()
            line = b""
            async with asyncio.timeout(5):
                with contextlib.suppress(ConnectionError):
                    line = await over.readline()
            gone.close()
            assert line in (b"", b'{"error": "' + CAPPED.encode() + b'"}\n')
        gate.set()
        async with asyncio.timeout(5):
            assert json.loads(await reader.readline()) == {"result": 0}
            writer.close()
            while b._open:
                await asyncio.sleep(0.01)
        assert await ask(b, request(b, args={"x": 1})) == {"result": 1}
        assert await ask(b, request(b, args={"x": 2})) == {"error": CALLED.format(2)}
        assert got == [0, 1]
    finally:
        gate.set()
        await b.close()


async def test_a_call_whose_args_nest_past_the_depth_is_refused(
    bridge: Bridge, seen: Seen, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        assert await ask(bridge, deep(bridge, 128)) == {"result": nest(127)}
        assert await ask(bridge, deep(bridge, 129)) == {"error": DEEP}
        assert await ask(bridge, deep(bridge, 60_000)) == {"error": DEEP}
        await asyncio.sleep(0.05)
    assert seen.calls == [("echo", {"x": nest(127)})]
    assert [r for r in caplog.records if r.levelno > logging.DEBUG] == []


async def test_the_depth_is_of_lists_and_objects_outside_strings(
    bridge: Bridge, seen: Seen
) -> None:
    text = '[[[[{{{{"\\' * 50 + '\\"]]'
    args = {"[[[[": text, "s": text + "\\", "x": nest(127)}
    assert await ask(bridge, request(bridge, args=args)) == {"result": nest(127)}
    deeper = {"[[[[": text, "s": text + "\\", "x": nest(128)}
    assert await ask(bridge, request(bridge, args=deeper)) == {"error": DEEP}
    assert await ask(bridge, request(bridge, extra=nest(128))) == {"result": 1}
    assert await ask(bridge, request(bridge, extra=nest(129))) == {"error": DEEP}
    assert seen.calls == [("echo", args), ("echo", {"x": 1})]
