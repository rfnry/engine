from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
from collections.abc import Awaitable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.jail import Jail
from rfnry_engine.tools.tools import Tool

WORKSPACE_TOOLS = frozenset({"Shell", "Write", "Edit"})
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


class WriteArgs(BaseModel):
    path: str
    content: str


class EditArgs(BaseModel):
    path: str
    old: str
    new: str


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
        return _Running(transport, pipes)

    async def read(self, path: str, limit: int) -> bytes:
        return await asyncio.to_thread(self._read, path, limit)

    async def write(self, path: str, data: bytes) -> None:
        await asyncio.to_thread(self._write, path, data)

    def _target(self, path: str) -> Path:
        try:
            return self._jail.resolve(path)
        except InputError:
            raise InputError(f"workspace: path outside the workspace: {path}") from None

    def _read(self, path: str, limit: int) -> bytes:
        target = self._target(path)
        if not target.is_file():
            raise InputError(f"workspace: not a file: {path}")
        if target.stat().st_size > limit:
            raise InputError(f"workspace: file exceeds {limit} bytes: {path}")
        return target.read_bytes()

    def _write(self, path: str, data: bytes) -> None:
        target = self._target(path)
        if target.is_dir():
            raise InputError(f"workspace: is a directory: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


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

    def take(self) -> str:
        head, tail = bytes(self.head), bytes(self.tail)
        omitted = self.total - len(head) - len(tail)
        self.head, self.tail, self.total = bytearray(), bytearray(), 0
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


async def _exit_code(process: Process) -> int | None:
    try:
        async with asyncio.timeout(GRACE):
            return await process.wait()
    except TimeoutError:
        return None


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

    def tools(self) -> list[Tool]:
        half = self.output_bytes // 2
        shell = (
            "Run a shell command. It starts in the workspace directory, so relative paths work and "
            "there is no need to search the filesystem for it. Each command starts fresh: a cd or a "
            "variable does not carry over to the next one. A command is stopped after "
            f"{self.timeout:g} seconds, and anything it leaves running is stopped when it ends. "
            "Returns exit_code, stdout, stderr and timed_out; stdout and stderr keep their first and "
            f"last {half} bytes, so send a long output to a file and read it from there."
        )
        if self._description:
            shell = f"{shell}\n\n{self._description}"
        return [
            Tool("Shell", shell, body=self._shell, args=ShellArgs),
            Tool(
                "Write",
                "Write a file in the workspace with exactly this content, creating parent directories.",
                body=self._write,
                args=WriteArgs,
            ),
            Tool(
                "Edit",
                "Replace one exact occurrence of old with new in a workspace file.",
                body=self._edit,
                args=EditArgs,
            ),
        ]

    async def _shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = ShellArgs.model_validate(args).command
        process = await self.backend.spawn(command)
        out, err = _Kept(self.output_bytes), _Kept(self.output_bytes)
        reading = _reading(process, out, err)
        try:
            code: int | None
            try:
                async with asyncio.timeout(self.timeout):
                    code = await process.wait()
                timed_out = False
            except TimeoutError:
                timed_out = True
            await process.kill()
            if timed_out:
                code = await _exit_code(process)
            await _settle(reading)
        except BaseException:
            _discard(reading)
            await process.kill()
            raise
        return {
            "exit_code": code,
            "stdout": out.take(),
            "stderr": err.take(),
            "timed_out": timed_out,
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
        await self._bounded("Edit", self._replace(a))
        return f"edited {a.path}"

    async def _replace(self, a: EditArgs) -> None:
        try:
            text = (await self.backend.read(a.path, TEXT_BYTES)).decode("utf-8")
        except UnicodeDecodeError:
            raise InputError(f"workspace: not a UTF-8 text file: {a.path}") from None
        count = text.count(a.old)
        if count != 1:
            raise InputError(
                f"workspace: old appears {count} times in {a.path}; it must appear exactly once"
            )
        await self.backend.write(a.path, text.replace(a.old, a.new, 1).encode("utf-8"))
