import asyncio
import contextlib
import itertools
import json
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Collection, Mapping, Sequence
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
CAP = 64 * 1024


class Script(NamedTuple):
    key: str
    path: Path
    source: str
    helpers: Mapping[str, str]
    awaits: bool


def _lines(pending: bytearray, data: bytes) -> list[bytes]:
    lines = []
    start = 0
    while (end := data.find(b"\n", start)) >= 0:
        lines.append(bytes(pending) + data[start:end])
        pending.clear()
        start = end + 1
    pending += data[start:]
    return lines


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
        self.calls: dict[int, tuple[str, asyncio.Future[JsonValue]]] = {}
        self.held: set[int] = set()
        self.ended = False
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

    def received(self, fd: int, data: bytes) -> None:
        if fd == 1:
            for line in _lines(self._out, data):
                self._answer(json.loads(line))
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
        self.calls[id] = (name, reply)
        self.write(text)
        return reply

    def abandon(self, id: int, held: bool) -> None:
        if self.calls.pop(id, None) is not None and not self.ended:
            if held:
                self.held.add(id)
            self.write(json.dumps({"id": id, "cancel": True}))

    def stop(self, reason: str | None = None) -> None:
        if self.ended:
            return
        self.ended = True
        _kill(self.pid)
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
        for name, future in self.calls.values():
            if not future.done() and not future.get_loop().is_closed():
                future.set_exception(WorkerError(f"workers/pool: the worker for {name} {stopped}"))
        self.calls.clear()
        self.held.clear()

    def _answer(self, reply: dict[str, Any]) -> None:
        self.held.discard(reply["id"])
        entry = self.calls.pop(reply["id"], None)
        if entry is None or entry[1].done():
            return
        _, future = entry
        if "error" in reply:
            future.set_exception(WorkerError(reply["error"]))
        else:
            future.set_result(reply["result"])

    def _log(self, line: bytes) -> None:
        logger.debug("workers/pool: worker %d: %s", self.pid, line.decode(errors="replace"))


def _load(worker: _Worker | None) -> int:
    return 0 if worker is None else len(worker.calls) + len(worker.held)


class WorkerPool:
    def __init__(self, scripts: Sequence[Script], *, processes: int = 1, threads: int = 32) -> None:
        self._scripts = {
            "scripts": [
                {"key": key, "path": str(path), "source": source}
                for key, path, source, _, _ in scripts
            ],
            "helpers": {str(path.parent): dict(helpers) for _, path, _, helpers, _ in scripts},
            "threads": threads,
        }
        self._awaits = {script.key for script in scripts if script.awaits}
        self._threads = threads
        self._workers: list[_Worker | None] = [None] * processes
        self._ids = itertools.count()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        self._closed = False

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
                worker.abandon(id, held=key not in self._awaits)

    async def close(self) -> None:
        self._closed = True
        loop = self._loop
        if loop is asyncio.get_running_loop():
            async with self._lock:
                self._drop(CLOSED)
        elif loop is not None and loop.is_running():
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self.close(), loop))
        else:
            self._drop(CLOSED)

    def _drop(self, reason: str) -> None:
        workers = [worker for worker in self._workers if worker is not None]
        self._workers = [None] * len(self._workers)
        for worker in workers:
            worker.stop(reason)

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
                chosen = self._workers[index] = await self._spawn()
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
        worker = _Worker(process, asyncio.get_running_loop())
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
    def __init__(self, agents: Mapping[str, AgentSpec], groups: Mapping[str, int] | None) -> None:
        named = dict(groups or {})
        scripts: dict[str | None, dict[str, Script]] = {None: {}, **{name: {} for name in named}}
        self._groups = {name: _group(spec.path, named) for name, spec in agents.items()}
        for name, spec in agents.items():
            own = scripts[self._groups[name]]
            for tool in spec.folder_tools.values():
                own[tool.key] = Script(
                    tool.key, tool.script, tool.source, tool.helpers, tool.awaits
                )
            for _, script in spec.scripted.values():
                own[script.key] = script
        self._pools = {
            group: WorkerPool(
                [held[key] for key in sorted(held)],
                processes=1 if group is None else named[group],
            )
            for group, held in scripts.items()
        }

    def pool(self, agent: str) -> WorkerPool:
        return self._pools[self._groups[agent]]

    async def close(self) -> None:
        await asyncio.gather(*(pool.close() for pool in self._pools.values()))
