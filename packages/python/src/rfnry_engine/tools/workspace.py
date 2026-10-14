from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.jail import Jail
from rfnry_engine.tools.tools import Tool

WORKSPACE_TOOLS = frozenset({"Shell", "Write", "Edit"})
GRACE = 1.0
CHUNK = 65_536


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


class _Capture(asyncio.SubprocessProtocol):
    def __init__(self, loop: asyncio.AbstractEventLoop, cap: int) -> None:
        self.half = cap // 2
        self.heads = {1: bytearray(), 2: bytearray()}
        self.tails = {1: bytearray(), 2: bytearray()}
        self.totals = {1: 0, 2: 0}
        self.open = {1, 2}
        self.exited: asyncio.Future[None] = loop.create_future()
        self.closed: asyncio.Future[None] = loop.create_future()

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        half = self.half
        head, tail = self.heads[fd], self.tails[fd]
        self.totals[fd] += len(data)
        room = half - len(head)
        if room > 0:
            head += data[:room]
            data = data[room:]
        tail += data
        if len(tail) > half:
            del tail[: len(tail) - half]

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        self.open.discard(fd)
        if not self.open and not self.closed.done():
            self.closed.set_result(None)

    def process_exited(self) -> None:
        if not self.exited.done():
            self.exited.set_result(None)

    def text(self, fd: int) -> str:
        head, tail = bytes(self.heads[fd]), bytes(self.tails[fd])
        omitted = self.totals[fd] - len(head) - len(tail)
        if omitted <= 0:
            return (head + tail).decode("utf-8", errors="replace")
        return (
            head.decode("utf-8", errors="replace")
            + f"\n[... {omitted} bytes omitted ...]\n"
            + tail.decode("utf-8", errors="replace")
        )


def _signal(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


async def _stop(pid: int, exited: asyncio.Future[None]) -> None:
    _signal(pid, signal.SIGTERM)
    try:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(GRACE):
                await asyncio.shield(exited)
    finally:
        _signal(pid, signal.SIGKILL)


def _bounded(
    handler: Callable[[dict[str, Any]], str], name: str, timeout: float
) -> Callable[[dict[str, Any]], Awaitable[str]]:
    async def body(args: dict[str, Any]) -> str:
        try:
            async with asyncio.timeout(timeout):
                return await asyncio.to_thread(handler, args)
        except TimeoutError:
            raise TimeoutError(f"workspace: {name} did not finish within {timeout:g}s") from None

    return body


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
                    await asyncio.shield(self._exited)
        finally:
            _signal(self._pid, signal.SIGKILL)


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


class Workspace:
    def __init__(
        self,
        root: Path | str,
        *,
        sandbox: Sequence[str] | None = None,
        unsandboxed: bool = False,
        shell: Sequence[str] = ("bash", "-c"),
        timeout: float = 120.0,
        output_bytes: int = 30_000,
        env: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ConfigurationError(f"workspace: root is not a directory: {root}")
        if sandbox is None and not unsandboxed:
            raise ConfigurationError(
                "workspace: the shell is not sandboxed; pass sandbox= (the engine documentation has "
                "recipes), or unsandboxed=True when this machine holds nothing the model must not reach"
            )
        if sandbox is not None and unsandboxed:
            raise ConfigurationError("workspace: pass sandbox= or unsandboxed=True, not both")
        if timeout <= 0:
            raise ConfigurationError("workspace: timeout must be positive")
        if output_bytes <= 0:
            raise ConfigurationError("workspace: output_bytes must be positive")
        self.env = (
            dict(env)
            if env is not None
            else {
                "PATH": os.environ.get("PATH", os.defpath),
                "HOME": str(self.root),
                "LANG": "C.UTF-8",
            }
        )
        self.sandbox = [] if sandbox is None else _words(sandbox, "sandbox", "['bwrap', ...]")
        self.shell = _words(shell, "shell", "['bash', '-c']")
        if not self.sandbox and shutil.which(self.shell[0], path=self.env.get("PATH")) is None:
            raise ConfigurationError(
                f"workspace: {self.shell[0]} was not found; pass shell=, such as ['sh', '-c']"
            )
        self.timeout = timeout
        self.output_bytes = output_bytes
        self.jail = Jail(self.root)

    def tools(self) -> list[Tool]:
        shell = (
            f"Run a {Path(self.shell[0]).name} command. It starts in the workspace directory, so relative paths work and "
            "there is no need to search the filesystem for it. Each command starts fresh: a cd or a variable does not "
            f"carry over to the next one. A command is stopped after {self.timeout:g} seconds. Returns exit_code, "
            f"stdout, stderr and timed_out; stdout and stderr keep their first and last {self.output_bytes // 2} bytes."
        )
        return [
            Tool("Shell", shell, body=self._shell, args=ShellArgs),
            Tool(
                "Write",
                "Write a file in the workspace with exactly this content, creating parent directories.",
                body=_bounded(self._write, "Write", self.timeout),
                args=WriteArgs,
            ),
            Tool(
                "Edit",
                "Replace one exact occurrence of old with new in a workspace file.",
                body=_bounded(self._edit, "Edit", self.timeout),
                args=EditArgs,
            ),
        ]

    async def _shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = ShellArgs.model_validate(args).command
        loop = asyncio.get_running_loop()
        transport, capture = await loop.subprocess_exec(
            lambda: _Capture(loop, self.output_bytes),
            *self.sandbox,
            *self.shell,
            command,
            cwd=self.root,
            env=self.env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pid = transport.get_pid()
        timed_out = False
        try:
            async with asyncio.timeout(self.timeout):
                await asyncio.shield(capture.exited)
                _signal(pid, signal.SIGKILL)
                await asyncio.shield(capture.closed)
        except TimeoutError:
            timed_out = not capture.exited.done()
            await _stop(pid, capture.exited)
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(1.0):
                    await asyncio.shield(capture.exited)
        except BaseException:
            await _stop(pid, capture.exited)
            raise
        finally:
            transport.close()
        return {
            "exit_code": transport.get_returncode(),
            "stdout": capture.text(1),
            "stderr": capture.text(2),
            "timed_out": timed_out,
        }

    def _target(self, path: str) -> Path:
        try:
            return self.jail.resolve(path)
        except InputError:
            raise InputError(f"workspace: path outside the workspace: {path}") from None

    def _write(self, args: dict[str, Any]) -> str:
        a = WriteArgs.model_validate(args)
        target = self._target(a.path)
        if target.is_dir():
            raise InputError(f"workspace: is a directory: {a.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = a.content.encode("utf-8")
        target.write_bytes(data)
        return f"wrote {len(data)} bytes to {target.relative_to(self.root)}"

    def _edit(self, args: dict[str, Any]) -> str:
        a = EditArgs.model_validate(args)
        if not a.old:
            raise InputError("workspace: old must not be empty")
        target = self._target(a.path)
        if not target.is_file():
            raise InputError(f"workspace: not a file: {a.path}")
        try:
            text = target.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raise InputError(f"workspace: not a UTF-8 text file: {a.path}") from None
        count = text.count(a.old)
        if count != 1:
            raise InputError(
                f"workspace: old appears {count} times in {a.path}; it must appear exactly once"
            )
        target.write_bytes(text.replace(a.old, a.new, 1).encode("utf-8"))
        return f"edited {target.relative_to(self.root)}"
