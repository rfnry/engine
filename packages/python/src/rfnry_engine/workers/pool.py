import asyncio
import contextlib
import itertools
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, NamedTuple, cast

from pydantic import JsonValue

from rfnry_engine.errors import InputError, WorkerError

if TYPE_CHECKING:
    from rfnry_engine.tree import AgentSpec

logger = logging.getLogger(__name__)

PROGRAM = Path(__file__).with_name("process.py")
CLOSED = "it was closed"
RETIRED = "every thread was held by a call past its deadline"
MOVED = "a call came from another event loop"
GARBLED = "it sent a line that is not a reply"
CAP = 64 * 1024
DRAIN = 1024 * 1024
POLL = 0.01


class Script(NamedTuple):
    key: str
    path: Path
    source: str
    helpers: Mapping[str, str]


def _lines(pending: bytearray, data: bytes) -> list[bytes]:
    lines = []
    start = 0
    while (end := data.find(b"\n", start)) >= 0:
        lines.append(bytes(pending) + data[start:end])
        pending.clear()
        start = end + 1
    pending += data[start:]
    return lines


def _reply(line: bytes) -> dict[str, Any] | None:
    try:
        reply = json.loads(line)
    except ValueError:
        return None
    if not isinstance(reply, dict):
        return None
    match reply:
        case {"threaded": [*keys]} if len(reply) == 1 and all(isinstance(k, str) for k in keys):
            return reply
        case {"id": int(), "result": _} | {"id": int(), "error": str()} if len(reply) == 2:
            return reply
    return None


def _reason(code: int) -> str:
    return f"killed by signal {-code}" if code < 0 else f"exited with code {code}"


def _kill(group: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group, signal.SIGKILL)


class _Pipe(asyncio.Protocol):
    def __init__(self, worker: _Worker, fd: int) -> None:
        self.worker = worker
        self.fd = fd

    def data_received(self, data: bytes) -> None:
        self.worker.received(self.fd, data)

    def connection_lost(self, exc: Exception | None) -> None:
        if self.fd == 1:
            self.worker.stop()


class _Worker:
    def __init__(self, process: subprocess.Popen[bytes], loop: asyncio.AbstractEventLoop) -> None:
        self.process = process
        self.pid = process.pid
        self.calls: dict[int, tuple[str, asyncio.Future[JsonValue], int]] = {}
        self.held: set[int] = set()
        self.cancelling: dict[int, int] = {}
        self.threaded: frozenset[str] = frozenset()
        self.sent = 0
        self.ended = False
        self.killed = False
        self._loop = loop
        self._stdin: asyncio.WriteTransport | None = None
        self._readers: list[asyncio.ReadTransport] = []
        self._out = bytearray()
        self._err = bytearray()

    async def connect(self) -> None:
        self._stdin, _ = await self._loop.connect_write_pipe(
            partial(_Pipe, self, 0), self.process.stdin
        )
        for fd, pipe in ((2, self.process.stderr), (1, self.process.stdout)):
            reader, _ = await self._loop.connect_read_pipe(partial(_Pipe, self, fd), pipe)
            self._readers.append(reader)
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self) -> None:
        if hasattr(os, "waitid"):
            with contextlib.suppress(ChildProcessError):
                os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOWAIT)
        else:
            self.process.wait()
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._exited)

    def _exited(self) -> None:
        if self.ended:
            return
        try:
            left = os.read(cast(IO[bytes], self.process.stdout).fileno(), DRAIN)
        except BlockingIOError:
            left = b""
        self.received(1, left)
        self.stop()

    def received(self, fd: int, data: bytes) -> None:
        if fd == 1:
            for line in _lines(self._out, data):
                reply = _reply(line)
                if reply is None:
                    self.stop(GARBLED)
                    return
                self._answer(reply)
        elif logger.isEnabledFor(logging.DEBUG):
            for line in _lines(self._err, data):
                self._log(line)
            while len(self._err) >= CAP:
                self._log(bytes(self._err[:CAP]))
                del self._err[:CAP]

    def write(self, text: str) -> None:
        cast(asyncio.WriteTransport, self._stdin).write(f"{text}\n".encode())

    def send(self, id: int, name: str, text: str) -> asyncio.Future[JsonValue]:
        reply: asyncio.Future[JsonValue] = asyncio.get_running_loop().create_future()
        self.sent += 1
        self.calls[id] = (name, reply, self.sent)
        self.write(text)
        return reply

    def abandon(self, id: int, held: bool, expired: bool) -> None:
        entry = self.calls.pop(id, None)
        if entry is None or self.ended:
            return
        _, _, number = entry
        wedged = expired and any(number > mark for mark in self.cancelling.values())
        if wedged:
            self.held.update(self.cancelling)
            self.cancelling.clear()
        if held or wedged:
            self.held.add(id)
        else:
            self.cancelling[id] = self.sent
        self.write(json.dumps({"id": id, "cancel": True}))

    def kill(self) -> None:
        if not self.killed:
            self.killed = True
            _kill(self.pid)

    def stop(self, reason: str | None = None) -> None:
        if self.ended:
            return
        self.ended = True
        self.kill()
        if not self._loop.is_closed():
            for reader in self._readers:
                if not reader.is_closing():
                    reader.close()
            if self._stdin is not None and not self._stdin.is_closing():
                self._stdin.abort()
        code = self.process.wait()
        stderr = cast(IO[bytes], self.process.stderr)
        if not stderr.closed:
            self.received(2, stderr.read() or b"")
        if self._err:
            self._log(bytes(self._err))
        for pipe in (self.process.stdin, self.process.stdout, stderr):
            cast(IO[bytes], pipe).close()
        stopped = f"stopped: {reason or _reason(code)}"
        for name, future, _ in self.calls.values():
            if not future.done() and not future.get_loop().is_closed():
                future.set_exception(WorkerError(f"workers/pool: the worker for {name} {stopped}"))
        self.calls.clear()
        self.held.clear()
        self.cancelling.clear()

    def _answer(self, reply: dict[str, Any]) -> None:
        if "threaded" in reply:
            self.threaded = frozenset(reply["threaded"])
            return
        self.held.discard(reply["id"])
        self.cancelling.pop(reply["id"], None)
        entry = self.calls.pop(reply["id"], None)
        if entry is None or entry[1].done():
            return
        _, future, _ = entry
        if "error" in reply:
            future.set_exception(WorkerError(reply["error"]))
        else:
            future.set_result(reply["result"])

    def _log(self, line: bytes) -> None:
        logger.debug("workers/pool: worker %d: %s", self.pid, line.decode(errors="replace"))


def _log(*errors: BaseException) -> None:
    for error in errors:
        logger.error("workers/pool: could not stop a worker: %s", error, exc_info=error)


def _logged(handed: Future[None]) -> None:
    if not handed.cancelled() and (error := handed.exception()) is not None:
        _log(error)


def _load(worker: _Worker | None) -> int:
    return 0 if worker is None else len(worker.calls) + len(worker.held) + len(worker.cancelling)


class WorkerPool:
    def __init__(
        self,
        scripts: Sequence[Script],
        *,
        processes: int = 1,
        threads: int = 32,
        limit: int | None = None,
    ) -> None:
        self._scripts = {
            "scripts": [
                {"key": key, "path": str(path), "source": source}
                for key, path, source, _ in scripts
            ],
            "helpers": {str(path.parent): dict(helpers) for _, path, _, helpers in scripts},
            "threads": threads,
            "limit": limit,
        }
        self._threads = threads
        self._workers: list[_Worker | None] = [None] * processes
        self._ids = itertools.count()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._spawning: asyncio.Future[None] | None = None
        self._starting: _Worker | None = None

    async def call(
        self, key: str, args: JsonValue, call: Mapping[str, str], timeout: float | None
    ) -> JsonValue:
        name = key.rsplit("/", 1)[-1]
        id = next(self._ids)
        try:
            text = json.dumps({"id": id, "key": key, "args": args, "call": dict(call)})
        except (TypeError, ValueError) as exc:
            raise InputError(
                f"workers/pool: the arguments for {name} are not JSON: {exc}"
            ) from None
        self._follow()
        deadline = asyncio.timeout(timeout)
        worker: _Worker | None = None
        try:
            async with deadline:
                worker = await self._pick(name)
                return await worker.send(id, name, text)
        except TimeoutError:
            if not deadline.expired():
                raise
            raise TimeoutError(f"workers/pool: {name} timed out after {timeout:g}s") from None
        finally:
            if worker is not None:
                worker.abandon(id, held=key in worker.threaded, expired=deadline.expired())

    async def close(self) -> None:
        try:
            handing = self._shut()
        except BaseException as exc:
            try:
                await self._started()
            except asyncio.CancelledError:
                _log(exc)
                raise
            raise
        if handing is None:
            await self._started()
            return
        loop, handed, dropped, soon = handing
        try:
            while not handed.done() and loop.is_running():
                await asyncio.sleep(POLL)
        except asyncio.CancelledError:
            dropped.add_done_callback(_logged)
            raise
        if not handed.done():
            soon.cancel()
            self._drop(CLOSED, starting=True)
        if dropped.done():
            dropped.result()

    async def _started(self) -> None:
        spawning = self._spawning
        if spawning is not None and spawning.get_loop() is asyncio.get_running_loop():
            await asyncio.wait([spawning])

    def release(self) -> None:
        handing = self._shut()
        if handing is not None:
            handing[2].add_done_callback(_logged)

    def _shut(
        self,
    ) -> tuple[asyncio.AbstractEventLoop, Future[None], Future[None], asyncio.Handle] | None:
        self._closed = True
        loop = self._loop
        here = asyncio.get_running_loop()
        if loop is not None and loop is not here and loop.is_running():
            handed: Future[None] = Future()
            dropped: Future[None] = Future()
            return loop, handed, dropped, loop.call_soon_threadsafe(self._hand, handed, dropped)
        self._drop(CLOSED, starting=loop is not None and loop is not here)
        return None

    def _hand(self, handed: Future[None], dropped: Future[None]) -> None:
        try:
            self._drop(CLOSED)
        except BaseException as exc:
            dropped.set_exception(exc)
        else:
            dropped.set_result(None)
        spawning = self._spawning
        if spawning is None:
            handed.set_result(None)
        else:
            spawning.add_done_callback(lambda _: handed.set_result(None))

    def _drop(self, reason: str, starting: bool = False) -> None:
        workers = [worker for worker in self._workers if worker is not None]
        held = self._starting
        if starting and held is not None:
            held.kill()
            held.process.wait()
        self._workers = [None] * len(self._workers)
        failed: list[BaseException] = []
        for worker in workers:
            try:
                worker.stop(reason)
            except BaseException as exc:
                failed.append(exc)
        _log(*failed[1:])
        if failed:
            raise failed[0]

    def _follow(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._drop(MOVED)
            self._lock = asyncio.Lock()
            self._loop = loop

    async def _pick(self, name: str) -> _Worker:
        async with self._lock:
            if self._closed:
                raise WorkerError(f"workers/pool: the worker for {name} stopped: {CLOSED}")
            for index, worker in enumerate(self._workers):
                if worker is not None and len(worker.held) >= self._threads:
                    worker.stop(RETIRED)
                if worker is not None and worker.ended:
                    self._workers[index] = None
            index = min(range(len(self._workers)), key=lambda i: _load(self._workers[i]))
            chosen = self._workers[index]
            if chosen is None:
                spawning = asyncio.get_running_loop().create_future()
                self._spawning = spawning
                try:
                    chosen = await self._spawn()
                    if self._closed:
                        chosen.stop(CLOSED)
                        raise WorkerError(f"workers/pool: the worker for {name} stopped: {CLOSED}")
                finally:
                    self._spawning = self._starting = None
                    if not spawning.get_loop().is_closed():
                        spawning.set_result(None)
                self._workers[index] = chosen
            return chosen

    async def _spawn(self) -> _Worker:
        process = subprocess.Popen(
            [sys.executable, "-P", str(PROGRAM)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        worker = self._starting = _Worker(process, asyncio.get_running_loop())
        try:
            await worker.connect()
        except BaseException:
            worker.stop()
            raise
        worker.write(json.dumps(self._scripts))
        return worker


def _group(path: str, groups: Collection[str]) -> str | None:
    return next((name for name in groups if path == name or path.startswith(f"{name}/")), None)


class Workers:
    def __init__(
        self,
        agents: Mapping[str, AgentSpec],
        groups: Mapping[str, int] | None,
        limit: int | None = None,
    ) -> None:
        named = dict(groups or {})
        scripts: dict[str | None, dict[str, Script]] = {None: {}, **{name: {} for name in named}}
        self._groups = {name: _group(spec.path, named) for name, spec in agents.items()}
        for name, spec in agents.items():
            own = scripts[self._groups[name]]
            for tool in spec.folder_tools.values():
                own[tool.key] = Script(tool.key, tool.script, tool.source, tool.helpers)
            for _, script in spec.scripted.values():
                own[script.key] = script
        self._pools = {
            group: WorkerPool(
                [held[key] for key in sorted(held)],
                processes=1 if group is None else named[group],
                limit=limit,
            )
            for group, held in scripts.items()
        }

    def pool(self, agent: str) -> WorkerPool:
        return self._pools[self._groups[agent]]

    async def close(self) -> None:
        failed: list[BaseException] = []
        cancelled: asyncio.CancelledError | None = None
        for pool in self._pools.values():
            try:
                if cancelled is None:
                    await pool.close()
                else:
                    pool.release()
            except asyncio.CancelledError as exc:
                cancelled = exc
            except Exception as exc:
                failed.append(exc)
        if cancelled is not None:
            _log(*failed)
            raise cancelled
        _log(*failed[1:])
        if failed:
            raise failed[0]
