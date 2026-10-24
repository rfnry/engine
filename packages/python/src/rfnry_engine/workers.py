import asyncio
import contextlib
import itertools
import json
import logging
import os
import signal
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

from pydantic import JsonValue

from rfnry_engine.errors import WorkerError
from rfnry_engine.tree import AgentSpec

logger = logging.getLogger(__name__)

PROGRAM = Path(__file__).with_name("_worker.py")
CLOSED = "it was closed"
RETIRED = "every thread was held by a call past its deadline"


class Script(NamedTuple):
    key: str
    path: Path


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


class _Worker(asyncio.SubprocessProtocol):
    transport: asyncio.SubprocessTransport
    pid: int

    def __init__(self) -> None:
        self.calls: dict[int, tuple[str, asyncio.Future[JsonValue]]] = {}
        self.held: set[int] = set()
        self.ended = asyncio.Event()
        self._stopped_by: str | None = None
        self._out = bytearray()
        self._err = bytearray()
        self._open = {1, 2}
        self._exited = False

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.SubprocessTransport, transport)
        self.pid = self.transport.get_pid()

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        if fd == 1:
            for line in _lines(self._out, data):
                self._answer(json.loads(line))
        else:
            for line in _lines(self._err, data):
                self._log(line)

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        if fd == 2 and self._err:
            self._log(bytes(self._err))
            self._err.clear()
        self._open.discard(fd)
        self._finish()

    def process_exited(self) -> None:
        self._exited = True
        self._finish()

    def write(self, message: dict[str, Any]) -> None:
        pipe = cast(asyncio.WriteTransport, self.transport.get_pipe_transport(0))
        pipe.write(f"{json.dumps(message)}\n".encode())

    def send(self, id: int, name: str, message: dict[str, Any]) -> asyncio.Future[JsonValue]:
        reply: asyncio.Future[JsonValue] = asyncio.get_running_loop().create_future()
        self.calls[id] = (name, reply)
        self.write(message)
        return reply

    def abandon(self, id: int) -> None:
        if self.calls.pop(id, None) is not None and not self.ended.is_set():
            self.held.add(id)
            self.write({"id": id, "cancel": True})

    def kill(self, reason: str) -> None:
        if not self.ended.is_set():
            self._stopped_by = self._stopped_by or reason
            _kill(self.pid)

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
        logger.debug("workers: worker %d: %s", self.pid, line.decode(errors="replace"))

    def _finish(self) -> None:
        if not self._exited or 1 in self._open:
            return
        if not self.ended.is_set():
            _kill(self.pid)
            reason = self._stopped_by or _reason(cast(int, self.transport.get_returncode()))
            for name, future in self.calls.values():
                if not future.done():
                    future.set_exception(
                        WorkerError(f"workers: the worker for {name} stopped: {reason}")
                    )
            self.calls.clear()
            self.held.clear()
            self.ended.set()
        if 2 not in self._open:
            self.transport.close()


def _load(worker: _Worker | None) -> int:
    return 0 if worker is None else len(worker.calls) + len(worker.held)


class WorkerPool:
    def __init__(self, scripts: Sequence[Script], *, processes: int = 1, threads: int = 32) -> None:
        self._scripts = {
            "scripts": [{"key": key, "path": str(path)} for key, path in scripts],
            "threads": threads,
        }
        self._threads = threads
        self._workers: list[_Worker | None] = [None] * processes
        self._ids = itertools.count()
        self._lock = asyncio.Lock()
        self._closed = False

    async def call(
        self, key: str, args: JsonValue, call: Mapping[str, str], timeout: float | None
    ) -> JsonValue:
        name = key.rsplit("/", 1)[-1]
        worker = await self._pick(name)
        id = next(self._ids)
        reply = worker.send(id, name, {"id": id, "key": key, "args": args, "call": dict(call)})
        try:
            async with asyncio.timeout(timeout):
                return await reply
        except TimeoutError:
            raise TimeoutError(f"workers: {name} timed out after {timeout:g}s") from None
        finally:
            worker.abandon(id)

    async def close(self) -> None:
        self._closed = True
        async with self._lock:
            workers = [worker for worker in self._workers if worker is not None]
            self._workers = [None] * len(self._workers)
        for worker in workers:
            worker.kill(CLOSED)
        for worker in workers:
            await worker.ended.wait()

    async def _pick(self, name: str) -> _Worker:
        async with self._lock:
            if self._closed:
                raise WorkerError(f"workers: the worker for {name} stopped: {CLOSED}")
            for index, worker in enumerate(self._workers):
                if worker is not None and len(worker.held) >= self._threads:
                    worker.kill(RETIRED)
                    await worker.ended.wait()
                if worker is not None and worker.ended.is_set():
                    self._workers[index] = None
            index = min(range(len(self._workers)), key=lambda i: _load(self._workers[i]))
            chosen = self._workers[index]
            if chosen is None:
                chosen = self._workers[index] = await self._spawn()
            return chosen

    async def _spawn(self) -> _Worker:
        _, worker = await asyncio.get_running_loop().subprocess_exec(
            _Worker,
            sys.executable,
            "-P",
            str(PROGRAM),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        worker.write(self._scripts)
        return worker


def _group(path: str, groups: Collection[str]) -> str | None:
    return next((name for name in groups if path == name or path.startswith(f"{name}/")), None)


class Workers:
    def __init__(self, agents: Mapping[str, AgentSpec], groups: Mapping[str, int] | None) -> None:
        named = dict(groups or {})
        scripts: dict[str | None, dict[str, Path]] = {None: {}, **{name: {} for name in named}}
        self._groups = {name: _group(spec.path, named) for name, spec in agents.items()}
        for name, spec in agents.items():
            for definition in spec.folder_tools.values():
                scripts[self._groups[name]][definition.key] = definition.script
        self._pools = {
            group: WorkerPool(
                [Script(key, path) for key, path in sorted(held.items())],
                processes=1 if group is None else named[group],
            )
            for group, held in scripts.items()
        }

    def pool(self, agent: str) -> WorkerPool:
        return self._pools[self._groups[agent]]

    async def close(self) -> None:
        await asyncio.gather(*(pool.close() for pool in self._pools.values()))
