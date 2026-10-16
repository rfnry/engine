from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import os
import shutil
import signal
import stat
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.jail import Jail
from rfnry_engine.tools.tools import Tool

logger = logging.getLogger(__name__)

WORKSPACE_TOOLS = frozenset({"Shell", "Job", "Write", "Edit"})
GRACE = 1.0
CHUNK = 65_536
TEXT_BYTES = 5 * 1024 * 1024


class Stream(Protocol):
    async def read(self, n: int) -> bytes: ...


class Process(Protocol):
    @property
    def stdout(self) -> Stream: ...

    @property
    def stderr(self) -> Stream: ...

    async def wait(self) -> int: ...

    async def kill(self) -> None: ...


@runtime_checkable
class Backend(Protocol):
    async def spawn(self, command: str) -> Process: ...

    async def read(self, path: str, limit: int) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...


class ShellArgs(BaseModel):
    command: str
    background: bool = False


class JobArgs(BaseModel):
    id: str
    wait: float = 0
    stop: bool = False


class WriteArgs(BaseModel):
    path: str
    content: str


class EditArgs(BaseModel):
    path: str
    old: str
    new: str
    all: bool = False


class _Pipes(asyncio.SubprocessProtocol):
    def __init__(self) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.exited: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.open = {1, 2}
        self.transport: asyncio.SubprocessTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.SubprocessTransport, transport)
        for fd, stream in ((1, self.stdout), (2, self.stderr)):
            pipe = self.transport.get_pipe_transport(fd)
            if pipe is not None:
                stream.set_transport(pipe)

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        (self.stdout if fd == 1 else self.stderr).feed_data(data)

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        (self.stdout if fd == 1 else self.stderr).feed_eof()
        self.open.discard(fd)
        self._finish()

    def process_exited(self) -> None:
        if not self.exited.done():
            self.exited.set_result(None)
        self._finish()

    def _finish(self) -> None:
        if self.exited.done() and not self.open and self.transport is not None:
            self.transport.close()


def _signal(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def _alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    return True


class _Running:
    def __init__(self, transport: asyncio.SubprocessTransport, pipes: _Pipes) -> None:
        self.stdout = pipes.stdout
        self.stderr = pipes.stderr
        self._transport = transport
        self._exited = pipes.exited
        self._pid = transport.get_pid()

    async def wait(self) -> int:
        await asyncio.shield(self._exited)
        return cast(int, self._transport.get_returncode())

    async def kill(self) -> None:
        _signal(self._pid, signal.SIGTERM)
        try:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(GRACE):
                    while _alive(self._pid):
                        await asyncio.sleep(0.02)
        finally:
            _signal(self._pid, signal.SIGKILL)


def _words(value: Sequence[str], name: str, example: str) -> list[str]:
    if (
        isinstance(value, str)
        or not value
        or not all(isinstance(part, str) and part for part in value)
    ):
        raise ConfigurationError(
            f"workspace: {name} must be a non-empty list of strings, such as {example}"
        )
    return list(value)


class Local:
    def __init__(
        self,
        root: Path | str,
        *,
        shell: Sequence[str] = ("bash", "-c"),
        env: dict[str, str] | None = None,
        description: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ConfigurationError(f"workspace: root is not a directory: {root}")
        self.env = (
            dict(env)
            if env is not None
            else {
                "PATH": os.environ.get("PATH", os.defpath),
                "HOME": str(self.root),
                "LANG": "C.UTF-8",
            }
        )
        self.shell = _words(shell, "shell", "['bash', '-c']")
        if shutil.which(self.shell[0], path=self.env.get("PATH")) is None:
            raise ConfigurationError(
                f"workspace: {self.shell[0]} was not found; pass shell=, such as ['sh', '-c']"
            )
        if description is not None and not isinstance(description, str):
            raise ConfigurationError("workspace: description must be a string")
        self.description = description
        self._jail = Jail(self.root)

    async def spawn(self, command: str) -> Process:
        loop = asyncio.get_running_loop()
        try:
            transport, pipes = await loop.subprocess_exec(
                _Pipes,
                *self.shell,
                command,
                cwd=self.root,
                env=self.env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise InputError(f"workspace: cannot start the command: {exc.strerror}") from None
        return _Running(transport, pipes)

    async def read(self, path: str, limit: int) -> bytes:
        return await asyncio.to_thread(self._read, path, limit)

    async def write(self, path: str, data: bytes) -> None:
        await asyncio.to_thread(self._write, path, data)

    def _open(self, path: str, flags: int) -> int:
        try:
            return self._jail.open(path, flags)
        except InputError:
            raise InputError(f"workspace: path outside the workspace: {path}") from None

    def _read(self, path: str, limit: int) -> bytes:
        try:
            fd = self._open(path, os.O_RDONLY)
            try:
                found = os.fstat(fd)
                if not stat.S_ISREG(found.st_mode):
                    raise InputError(f"workspace: not a file: {path}")
                if found.st_size > limit:
                    raise InputError(f"workspace: file exceeds {limit} bytes: {path}")
                data = bytearray()
                while len(data) <= limit and (
                    chunk := os.read(fd, min(CHUNK, limit + 1 - len(data)))
                ):
                    data += chunk
            finally:
                os.close(fd)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENXIO, errno.ENOTDIR):
                raise InputError(f"workspace: not a file: {path}") from None
            raise InputError(f"workspace: cannot read {path}: {exc.strerror}") from None
        if len(data) > limit:
            raise InputError(f"workspace: file exceeds {limit} bytes: {path}")
        return bytes(data)

    def _write(self, path: str, data: bytes) -> None:
        try:
            fd = self._open(path, os.O_WRONLY | os.O_CREAT)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise InputError(f"workspace: not a file: {path}")
                os.ftruncate(fd, 0)
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view) :]
            finally:
                os.close(fd)
        except OSError as exc:
            if exc.errno == errno.EISDIR:
                raise InputError(f"workspace: is a directory: {path}") from None
            if exc.errno == errno.ENXIO:
                raise InputError(f"workspace: not a file: {path}") from None
            raise InputError(f"workspace: cannot write {path}: {exc.strerror}") from None


def _complete(data: bytearray) -> int:
    for back in range(1, min(3, len(data)) + 1):
        byte = data[-back]
        if 0x80 <= byte <= 0xBF:
            continue
        needs = (
            2
            if 0xC2 <= byte <= 0xDF
            else 3
            if 0xE0 <= byte <= 0xEF
            else 4
            if 0xF0 <= byte <= 0xF4
            else 0
        )
        return len(data) - back if needs > back else len(data)
    return len(data)


class _Kept:
    def __init__(self, cap: int) -> None:
        self.half = cap // 2
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def add(self, data: bytes) -> None:
        self.total += len(data)
        room = self.half - len(self.head)
        if room > 0:
            self.head += data[:room]
            data = data[room:]
        self.tail += data
        if len(self.tail) > self.half:
            del self.tail[: len(self.tail) - self.half]

    def take(self, final: bool = True) -> str:
        held = b""
        if not final:
            if self.total == len(self.head) + len(self.tail):
                whole = self.head + self.tail
                held = bytes(whole[_complete(whole) :])
                cut = len(held)
                from_tail = min(cut, len(self.tail))
                if from_tail:
                    del self.tail[len(self.tail) - from_tail :]
                if cut > from_tail:
                    del self.head[len(self.head) - (cut - from_tail) :]
            else:
                held = bytes(self.tail[_complete(self.tail) :])
                if held:
                    del self.tail[len(self.tail) - len(held) :]
        head, tail = bytes(self.head), bytes(self.tail)
        omitted = self.total - len(held) - len(head) - len(tail)
        self.head, self.tail, self.total = bytearray(held), bytearray(), len(held)
        if omitted <= 0:
            return (head + tail).decode("utf-8", errors="replace")
        return (
            head.decode("utf-8", errors="replace")
            + f"\n[... {omitted} bytes omitted ...]\n"
            + tail.decode("utf-8", errors="replace")
        )


async def _drain(stream: Stream, kept: _Kept) -> None:
    while chunk := await stream.read(CHUNK):
        kept.add(chunk)


def _reading(process: Process, out: _Kept, err: _Kept) -> list[asyncio.Task[None]]:
    return [
        asyncio.create_task(_drain(process.stdout, out)),
        asyncio.create_task(_drain(process.stderr, err)),
    ]


async def _settle(reading: list[asyncio.Task[None]]) -> None:
    await asyncio.wait(reading, timeout=GRACE)
    _discard(reading)
    await asyncio.wait(reading)
    for task in reading:
        if not task.cancelled():
            task.result()


def _discard(reading: list[asyncio.Task[None]]) -> None:
    for task in reading:
        if not task.cancel() and not task.cancelled():
            task.exception()


async def _reap(spawning: asyncio.Task[Process]) -> None:
    try:
        process = await spawning
    except Exception:
        return
    await process.kill()


def _log(*errors: BaseException | None) -> None:
    for error in errors:
        if error is not None:
            logger.error("workspace: could not stop the command: %s", error, exc_info=error)


async def _finish(*kills: Awaitable[None]) -> list[BaseException | None]:
    killing = [asyncio.ensure_future(kill) for kill in kills]
    cancelled: asyncio.CancelledError | None = None
    while pending := [task for task in killing if not task.done()]:
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
    errors = [task.exception() for task in killing]
    if cancelled is not None:
        _log(*errors)
        raise cancelled
    return errors


async def _exit_code(process: Process) -> int | None:
    try:
        async with asyncio.timeout(GRACE):
            return await process.wait()
    except TimeoutError:
        return None


@dataclass
class _Job:
    command: str
    process: Process
    kept: _Kept
    started: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    code: int | None = None
    task: asyncio.Task[None] | None = None


def _note(kept: _Kept, error: BaseException | None) -> None:
    if error is not None:
        kept.add(f"\n{type(error).__name__}: {error}\n".encode())


async def _follow(job: _Job) -> None:
    reading = _reading(job.process, job.kept, job.kept)
    try:
        try:
            job.code = await job.process.wait()
        except Exception as exc:
            _note(job.kept, exc)
        finally:
            [error] = await _finish(job.process.kill())
            _note(job.kept, error)
        await _settle(reading)
    except Exception as exc:
        _note(job.kept, exc)
    finally:
        _discard(reading)
        job.ended.set()


class Workspace:
    def __init__(
        self,
        backend: Backend,
        *,
        scope: str,
        timeout: float = 120.0,
        output_bytes: int = 30_000,
    ) -> None:
        if not isinstance(backend, Backend):
            raise ConfigurationError("workspace: backend must have spawn, read and write")
        description = getattr(backend, "description", None)
        if description is not None and not isinstance(description, str):
            raise ConfigurationError("workspace: a backend's description must be a string")
        if not scope:
            raise ConfigurationError("workspace: scope is required")
        if timeout <= 0:
            raise ConfigurationError("workspace: timeout must be positive")
        if output_bytes <= 0:
            raise ConfigurationError("workspace: output_bytes must be positive")
        self.backend = backend
        self.scope = scope
        self.timeout = timeout
        self.output_bytes = output_bytes
        self._description = description
        self._jobs: dict[str, _Job] = {}
        self._count = 0

    async def __aenter__(self) -> Workspace:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def tools(self) -> list[Tool]:
        half = self.output_bytes // 2
        shell = (
            "Run a shell command. It starts in the workspace directory, so relative paths work and "
            "there is no need to search the filesystem for it. Each command starts fresh: a cd or a "
            "variable does not carry over to the next one. A command is stopped after "
            f"{self.timeout:g} seconds, and anything it leaves running is stopped when it ends. To "
            "keep something running, such as a server, a watcher or a long build, pass "
            "background=true: it starts a job and returns its id at once, and the job keeps running "
            "after your reply until it is stopped. Job reads a job's new output, waits for it to end and stops it; Jobs "
            "lists them. Returns exit_code, stdout, stderr and timed_out; stdout and stderr keep "
            f"their first and last {half} bytes, so send a long output to a file and read it from "
            "there."
        )
        if self._description:
            shell = f"{shell}\n\n{self._description}"
        return [
            Tool("Shell", shell, body=self._shell, args=ShellArgs),
            Tool(
                "Job",
                "Read what a background job printed since the last read, stdout and stderr in the "
                "order they arrived, whether it is running, and its exit_code once it has ended. "
                f"wait is seconds, up to {self.timeout:g}, to wait for it to end first; stop=true "
                f"stops it, after the wait. The output keeps its first and last {half} bytes.",
                body=self._job,
                args=JobArgs,
            ),
            Tool(
                "Jobs",
                "List the workspace's background jobs: id, command, when it started, whether it is "
                "running and its exit_code. A job that ended is listed until Job reads its last output.",
                body=self._list,
            ),
            Tool(
                "Write",
                "Write a file in the workspace with exactly this content, creating parent directories.",
                body=self._write,
                args=WriteArgs,
            ),
            Tool(
                "Edit",
                "Replace one exact occurrence of old with new in a workspace file, or every occurrence with all=true.",
                body=self._edit,
                args=EditArgs,
            ),
        ]

    async def close(self) -> None:
        jobs = list(self._jobs.items())
        running = [job for _, job in jobs if not job.ended.is_set()]
        results = await _finish(*(job.process.kill() for job in running))
        errors = dict(zip(map(id, running), results, strict=True))
        gone = [(name, job) for name, job in jobs if errors.get(id(job)) is None]
        for name, job in gone:
            if self._jobs.get(name) is job:
                del self._jobs[name]
        await asyncio.gather(
            *(job.task for _, job in gone if job.task is not None), return_exceptions=True
        )
        failed = [result for result in results if result is not None]
        if failed:
            raise failed[0]

    async def _shell(self, args: dict[str, Any]) -> dict[str, Any]:
        a = ShellArgs.model_validate(args)
        if a.background:
            return {"job": await asyncio.shield(self._start(a.command))}
        spawning = asyncio.create_task(self.backend.spawn(a.command))
        try:
            process = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            _log(*await _finish(_reap(spawning)))
            raise
        out, err = _Kept(self.output_bytes), _Kept(self.output_bytes)
        reading = _reading(process, out, err)
        code: int | None = None
        try:
            async with asyncio.timeout(self.timeout):
                code = await process.wait()
            timed_out = False
        except TimeoutError:
            timed_out = True
        except BaseException:
            _discard(reading)
            _log(*await _finish(process.kill()))
            raise
        try:
            [error] = await _finish(process.kill())
            if error is not None:
                raise error
            if timed_out:
                code = await _exit_code(process)
            await _settle(reading)
        finally:
            _discard(reading)
        return {
            "exit_code": code,
            "stdout": out.take(),
            "stderr": err.take(),
            "timed_out": timed_out,
        }

    async def _start(self, command: str) -> str:
        process = await self.backend.spawn(command)
        self._count += 1
        name = f"j{self._count}"
        job = _Job(command=command, process=process, kept=_Kept(self.output_bytes))
        job.task = asyncio.create_task(_follow(job))
        self._jobs[name] = job
        return name

    async def _job(self, args: dict[str, Any]) -> dict[str, Any]:
        a = JobArgs.model_validate(args)
        if not 0 <= a.wait <= self.timeout:
            raise InputError(
                f"workspace: wait must be between 0 and {self.timeout:g} seconds; got {a.wait:g}"
            )
        deadline = asyncio.get_running_loop().time() + self.timeout
        job = self._jobs.get(a.id)
        if job is None:
            raise InputError(f"workspace: there is no job {a.id}")
        if a.wait:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(a.wait):
                    await job.ended.wait()
        if a.stop and not job.ended.is_set():
            [error] = await _finish(job.process.kill())
            if error is not None:
                raise error
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout_at(deadline):
                    await job.ended.wait()
        running = not job.ended.is_set()
        output = job.kept.take(final=not running)
        if not running and self._jobs.get(a.id) is job:
            del self._jobs[a.id]
        return {"output": output, "running": running, "exit_code": None if running else job.code}

    async def _list(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "jobs": [
                {
                    "job": name,
                    "command": job.command,
                    "started": job.started.isoformat(timespec="seconds"),
                    "running": not job.ended.is_set(),
                    "exit_code": job.code if job.ended.is_set() else None,
                }
                for name, job in self._jobs.items()
            ]
        }

    async def _bounded[T](self, name: str, work: Awaitable[T]) -> T:
        try:
            async with asyncio.timeout(self.timeout):
                return await work
        except TimeoutError:
            raise TimeoutError(
                f"workspace: {name} did not finish within {self.timeout:g}s"
            ) from None

    async def _write(self, args: dict[str, Any]) -> str:
        a = WriteArgs.model_validate(args)
        data = a.content.encode("utf-8")
        await self._bounded("Write", self.backend.write(a.path, data))
        return f"wrote {len(data)} bytes to {a.path}"

    async def _edit(self, args: dict[str, Any]) -> str:
        a = EditArgs.model_validate(args)
        if not a.old:
            raise InputError("workspace: old must not be empty")
        count = await self._bounded("Edit", self._replace(a))
        return f"edited {a.path} ({count} occurrence{'' if count == 1 else 's'})"

    async def _replace(self, a: EditArgs) -> int:
        try:
            text = (await self.backend.read(a.path, TEXT_BYTES)).decode("utf-8")
        except UnicodeDecodeError:
            raise InputError(f"workspace: not a UTF-8 text file: {a.path}") from None
        count = text.count(a.old)
        if count == 0 or (count > 1 and not a.all):
            need = "at least once" if a.all else "exactly once"
            raise InputError(
                f"workspace: old appears {count} times in {a.path}; it must appear {need}"
            )
        await self.backend.write(
            a.path, text.replace(a.old, a.new, -1 if a.all else 1).encode("utf-8")
        )
        return count
