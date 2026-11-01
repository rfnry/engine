import asyncio
import codecs
import contextlib
import errno
import logging
import os
import posixpath
import shutil
import signal
import stat
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.foreground import current
from rfnry_engine.jail import Jail
from rfnry_engine.records import Pipe
from rfnry_engine.stop import join, wait
from rfnry_engine.tools.bridge import CALLS, COMMAND_BYTES, COMMAND_CALLS, Answer, Bridge
from rfnry_engine.tools.tools import Tool

logger = logging.getLogger(__name__)

WORKSPACE_TOOLS = frozenset({"Shell", "Job", "Write", "Edit"})
WORKSPACE_NAMES = frozenset({*WORKSPACE_TOOLS, "Jobs"})
GRACE = 1.0
CHUNK = 65_536
TEXT_BYTES = 5 * 1024 * 1024
BACKLOG = 1_048_576


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
    async def spawn(self, command: str, env: Mapping[str, str]) -> Process: ...

    async def read(self, path: str, limit: int) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...


class JobInfo(BaseModel):
    job: str
    command: str
    started: datetime
    running: bool
    exit_code: int | None


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
        self.abandoned = False
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

    def abandon(self) -> None:
        self.abandoned = True
        self._finish()

    def _finish(self) -> None:
        done = self.abandoned or not self.open
        if self.exited.done() and done and self.transport is not None:
            self.transport.close()


class _Read:
    def __init__(self, stream: asyncio.StreamReader, pipes: _Pipes) -> None:
        self._stream = stream
        self._pipes = pipes

    async def read(self, n: int) -> bytes:
        try:
            return await self._stream.read(n)
        except asyncio.CancelledError:
            self._pipes.abandon()
            raise


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
        self.stdout = _Read(pipes.stdout, pipes)
        self.stderr = _Read(pipes.stderr, pipes)
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
            f"tools/workspace: {name} must be a non-empty list of strings, such as {example}"
        )
    return list(value)


async def _threaded[T](work: Callable[..., T], *args: Any) -> T:
    running = asyncio.get_running_loop().run_in_executor(None, work, *args)
    await join([running])
    return running.result()


class Local:
    def __init__(
        self,
        root: Path | str,
        *,
        shell: Sequence[str] = ("bash", "-c"),
        env: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ConfigurationError(f"tools/workspace: root is not a directory: {root}")
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
                f"tools/workspace: {self.shell[0]} was not found; pass shell=, such as ['sh', '-c']"
            )
        self._jail = Jail(self.root)

    async def spawn(self, command: str, env: Mapping[str, str]) -> Process:
        loop = asyncio.get_running_loop()
        try:
            transport, pipes = await loop.subprocess_exec(
                _Pipes,
                *self.shell,
                command,
                cwd=self.root,
                env={**self.env, **env},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise InputError(f"tools/workspace: cannot start the command: {exc.strerror}") from None
        return _Running(transport, pipes)

    async def read(self, path: str, limit: int) -> bytes:
        return await _threaded(self._read, path, limit)

    async def write(self, path: str, data: bytes) -> None:
        await _threaded(self._write, path, data)

    def _open(self, path: str, flags: int) -> int:
        try:
            return self._jail.open(path, flags)
        except InputError:
            raise InputError(f"tools/workspace: path outside the workspace: {path}") from None

    def _read(self, path: str, limit: int) -> bytes:
        try:
            fd = self._open(path, os.O_RDONLY)
            try:
                found = os.fstat(fd)
                if not stat.S_ISREG(found.st_mode):
                    raise InputError(f"tools/workspace: not a file: {path}")
                if found.st_size > limit:
                    raise InputError(f"tools/workspace: file exceeds {limit} bytes: {path}")
                data = bytearray()
                while len(data) <= limit and (
                    chunk := os.read(fd, min(CHUNK, limit + 1 - len(data)))
                ):
                    data += chunk
            finally:
                os.close(fd)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENXIO, errno.ENOTDIR):
                raise InputError(f"tools/workspace: not a file: {path}") from None
            raise InputError(f"tools/workspace: cannot read {path}: {exc.strerror}") from None
        if len(data) > limit:
            raise InputError(f"tools/workspace: file exceeds {limit} bytes: {path}")
        return bytes(data)

    def _write(self, path: str, data: bytes) -> None:
        try:
            fd = self._open(path, os.O_WRONLY | os.O_CREAT)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise InputError(f"tools/workspace: not a file: {path}")
                os.ftruncate(fd, 0)
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view) :]
            finally:
                os.close(fd)
        except OSError as exc:
            if exc.errno == errno.EISDIR:
                raise InputError(f"tools/workspace: is a directory: {path}") from None
            if exc.errno == errno.ENXIO:
                raise InputError(f"tools/workspace: not a file: {path}") from None
            raise InputError(f"tools/workspace: cannot write {path}: {exc.strerror}") from None


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

    def _split(self, final: bool) -> tuple[bytes, bytes, bytes]:
        head, tail, held = bytearray(self.head), bytearray(self.tail), b""
        if not final:
            if self.total == len(head) + len(tail):
                whole = head + tail
                held = bytes(whole[_complete(whole) :])
                cut = len(held)
                from_tail = min(cut, len(tail))
                if from_tail:
                    del tail[len(tail) - from_tail :]
                if cut > from_tail:
                    del head[len(head) - (cut - from_tail) :]
            else:
                held = bytes(tail[_complete(tail) :])
                if held:
                    del tail[len(tail) - len(held) :]
        return bytes(head), bytes(tail), held

    def show(self, final: bool) -> str:
        head, tail, held = self._split(final)
        return self._text(head, tail, self.total - len(held) - len(head) - len(tail))

    def take(self, final: bool = True) -> str:
        head, tail, held = self._split(final)
        omitted = self.total - len(held) - len(head) - len(tail)
        self.head, self.tail, self.total = bytearray(held), bytearray(), len(held)
        return self._text(head, tail, omitted)

    @staticmethod
    def _text(head: bytes, tail: bytes, omitted: int) -> str:
        if omitted <= 0:
            return (head + tail).decode("utf-8", errors="replace")
        return (
            head.decode("utf-8", errors="replace")
            + f"\n[... {omitted} bytes omitted ...]\n"
            + tail.decode("utf-8", errors="replace")
        )


_Sink = Callable[[bytes], None]
_Said = Callable[[Pipe, str], Awaitable[None]]


class _Told:
    def __init__(self, said: _Said) -> None:
        self.said = said
        self.queue: asyncio.Queue[tuple[Pipe, int, bytes] | None] = asyncio.Queue()
        self.queued = 0
        self.missed: dict[Pipe, int] = {"stdout": 0, "stderr": 0}
        self.dropping: set[Pipe] = set()
        self.decoders: dict[Pipe, codecs.IncrementalDecoder] = {
            "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        }
        self.task = asyncio.create_task(self._send())

    def add(self, stream: Pipe, data: bytes) -> None:
        if stream in self.dropping or self.queued + len(data) > BACKLOG:
            self.dropping.add(stream)
            self.missed[stream] += len(data)
            if self.queue.empty():
                self.queue.put_nowait((stream, 0, b""))
            return
        self.queued += len(data)
        self.queue.put_nowait((stream, self.missed[stream], data))
        self.missed[stream] = 0

    def close(self) -> None:
        self.queue.put_nowait(None)

    async def _send(self) -> None:
        closing = False
        while not closing:
            taken = [await self.queue.get()]
            while not self.queue.empty():
                taken.append(self.queue.get_nowait())
            self.dropping.clear()
            if taken[-1] is None:
                closing = True
                taken.pop()
            events: list[tuple[Pipe, int, bytearray]] = []
            last: dict[Pipe, bytearray] = {}
            for item in taken:
                assert item is not None
                stream, gap, data = item
                self.queued -= len(data)
                if stream in last and not gap:
                    last[stream].extend(data)
                else:
                    last[stream] = bytearray(data)
                    events.append((stream, gap, last[stream]))
            for stream, gap, joined in events:
                await self._say(stream, gap, bytes(joined))
                if self.task.cancelling():
                    return
        for stream, gap in self.missed.items():
            await self._say(stream, gap, b"", final=True)
            if self.task.cancelling():
                return

    async def _say(self, stream: Pipe, gap: int, data: bytes, final: bool = False) -> None:
        decoder = self.decoders[stream]
        if gap:
            gap += len(decoder.getstate()[0])
            decoder.reset()
        text = (f"[... {gap} bytes not shown ...]\n" if gap else "") + decoder.decode(data, final)
        if text:
            await self.said(stream, text)


async def _drain(stream: Stream, *sinks: _Sink) -> None:
    while chunk := await stream.read(CHUNK):
        for sink in sinks:
            sink(chunk)


def _reading(
    process: Process, out: Sequence[_Sink], err: Sequence[_Sink]
) -> list[asyncio.Task[None]]:
    return [
        asyncio.create_task(_drain(process.stdout, *out)),
        asyncio.create_task(_drain(process.stderr, *err)),
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


def _log(*errors: BaseException | None) -> None:
    for error in errors:
        if error is not None:
            logger.error("tools/workspace: could not stop the command: %s", error, exc_info=error)


async def _kill_all(
    *kills: Coroutine[Any, Any, None],
) -> tuple[list[BaseException | None], asyncio.CancelledError | None]:
    killing = [asyncio.ensure_future(kill) for kill in kills]
    caller = asyncio.current_task()
    cancelled: asyncio.CancelledError | None = None
    if killing:
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            cancelled = exc
    hurried = False
    while True:
        if not hurried and caller is not None and caller.cancelling() > 1:
            hurried = True
            for task in killing:
                task.cancel()
        if not (pending := [task for task in killing if not task.done()]):
            break
        try:
            await asyncio.wait(pending)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
    return [None if task.cancelled() else task.exception() for task in killing], cancelled


async def _see_through(*kills: Coroutine[Any, Any, None]) -> list[BaseException | None]:
    errors, cancelled = await _kill_all(*kills)
    if cancelled is not None:
        _log(*errors)
        raise cancelled
    return errors


async def _exit_code(process: Process) -> int | None:
    deadline = asyncio.timeout(GRACE)
    try:
        async with deadline:
            return await process.wait()
    except TimeoutError:
        if not deadline.expired():
            raise
        return None


async def _bridge(root: Path, answer: Answer) -> tuple[Bridge | None, bytes]:
    opening = asyncio.ensure_future(Bridge.open(root, answer))
    try:
        await join([opening])
    except asyncio.CancelledError:
        if not opening.cancelled() and opening.exception() is None:
            await opening.result().close()
        raise
    failed = opening.exception()
    if isinstance(failed, OSError):
        why = failed.strerror or failed
        return None, f"tools/workspace: no tool bridge for this command: {why}\n".encode()
    return opening.result(), b""


@dataclass
class _Job:
    command: str
    process: Process
    kept: _Kept
    shown: _Kept
    started: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    code: int | None = None
    task: asyncio.Task[None] | None = None


def _note(job: _Job, error: BaseException | None, said: str = "") -> str:
    line = "" if error is None else f"\n{type(error).__name__}: {error}\n"
    if line and line != said:
        job.kept.add(line.encode())
        job.shown.add(line.encode())
    return line


async def _follow(job: _Job) -> None:
    reading = _reading(job.process, [job.kept.add, job.shown.add], [job.kept.add, job.shown.add])
    said = ""
    try:
        try:
            job.code = await job.process.wait()
        except Exception as exc:
            said = _note(job, exc)
        finally:
            [error] = await _see_through(job.process.kill())
            _note(job, error, said)
        await _settle(reading)
    except Exception as exc:
        _note(job, exc)
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
        bridge: bool = True,
        tree: str | None = None,
    ) -> None:
        if not isinstance(backend, Backend):
            raise ConfigurationError("tools/workspace: backend must have spawn, read and write")
        if not scope:
            raise ConfigurationError("tools/workspace: scope is required")
        if timeout <= 0:
            raise ConfigurationError("tools/workspace: timeout must be positive")
        if output_bytes <= 0:
            raise ConfigurationError("tools/workspace: output_bytes must be positive")
        if tree is not None and not PurePosixPath(tree).is_absolute():
            raise ConfigurationError(
                f"tools/workspace: tree must be an absolute path, as commands see it, such as /srv/agents; got {tree!r}"
            )
        self.backend = backend
        self.scope = scope
        self.bridge = bridge
        self.tree = None if tree is None else posixpath.normpath(tree)
        self.timeout = timeout
        self.output_bytes = output_bytes
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
        results, cancelled = await _kill_all(*(job.process.kill() for job in running))
        errors = dict(zip(map(id, running), results, strict=True))
        gone = [(name, job) for name, job in jobs if errors.get(id(job)) is None]
        for name, job in gone:
            if self._jobs.get(name) is job:
                del self._jobs[name]
        failed = [result for result in results if result is not None]
        readers = [job.task for _, job in gone if job.task is not None]
        caller = asyncio.current_task()
        passed = caller is not None and caller.cancelling() > 1
        while pending := [task for task in readers if not task.done()]:
            if passed and caller is not None:
                for task in pending:
                    while task.cancelling() < caller.cancelling():
                        task.cancel()
            try:
                await asyncio.wait(pending)
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
                passed = True
        for task in readers:
            if not task.cancelled():
                task.exception()
        if cancelled is not None:
            _log(*failed)
            raise cancelled
        _log(*failed[1:])
        if failed:
            raise failed[0]

    async def _shell(self, args: dict[str, Any]) -> dict[str, Any]:
        a = ShellArgs.model_validate(args)
        here = current()
        lent = here.lent if here is not None else None
        env = lent.env if lent is not None else {}
        if a.background:
            starting = asyncio.ensure_future(self._start(a.command, env))
            await join([starting])
            return {"job": starting.result()}
        root = getattr(self.backend, "root", None)
        bridge: Bridge | None = None
        missing = b""
        if lent is not None and lent.call is not None and root is not None:
            bridge, missing = await _bridge(Path(root).absolute(), lent.call)
        if bridge is not None:
            env = {**env, **bridge.env}
        told = None if lent is None or lent.said is None else _Told(lent.said)
        try:
            try:
                code, timed_out, out, err = await self._foreground(a.command, env, told)
            finally:
                if bridge is not None:
                    await bridge.close()
            if missing:
                err.add(missing)
                if told is not None:
                    told.add("stderr", missing)
            if told is not None:
                told.close()
                await wait([told.task], None)
        finally:
            if told is not None and not told.task.done():
                told.task.cancel()
                await wait([told.task], None)
        return {
            "exit_code": code,
            "stdout": out.take(),
            "stderr": err.take(),
            "timed_out": timed_out,
        }

    async def _foreground(
        self, command: str, env: Mapping[str, str], told: _Told | None
    ) -> tuple[int | None, bool, _Kept, _Kept]:
        spawning = asyncio.create_task(self.backend.spawn(command, env))
        try:
            process = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            try:
                await join([spawning])
            finally:
                if not spawning.cancelled() and spawning.exception() is None:
                    _log(*await _see_through(spawning.result().kill()))
            raise
        out, err = _Kept(self.output_bytes), _Kept(self.output_bytes)
        reading = _reading(
            process,
            [out.add] if told is None else [out.add, partial(told.add, "stdout")],
            [err.add] if told is None else [err.add, partial(told.add, "stderr")],
        )
        code: int | None = None
        deadline = asyncio.timeout(self.timeout)
        try:
            async with deadline:
                code = await process.wait()
            timed_out = False
        except BaseException as exc:
            if not isinstance(exc, TimeoutError) or not deadline.expired():
                _discard(reading)
                _log(*await _see_through(process.kill()))
                raise
            timed_out = True
        try:
            [error] = await _see_through(process.kill())
            if error is not None:
                raise error
            if timed_out:
                code = await _exit_code(process)
            await _settle(reading)
        finally:
            _discard(reading)
        return code, timed_out, out, err

    async def _start(self, command: str, env: Mapping[str, str]) -> str:
        process = await self.backend.spawn(command, env)
        self._count += 1
        name = f"j{self._count}"
        job = _Job(
            command=command,
            process=process,
            kept=_Kept(self.output_bytes),
            shown=_Kept(self.output_bytes),
        )
        job.task = asyncio.create_task(_follow(job))
        self._jobs[name] = job
        return name

    @property
    def _bridging(self) -> bool:
        return self.bridge and getattr(self.backend, "root", None) is not None

    def _told(self, reads: Sequence[str], bridged: bool) -> str:
        told = ""
        if self.tree is not None:
            told += (
                " The agent's own folder is at $RFNRY_AGENT, read-only: its files, skills and"
                " scripts can be read and run from there."
            )
            if reads:
                places = ", ".join(f"$RFNRY_TREE/{name}" for name in reads)
                told += f" The folders it reads are at {places}."
        if bridged:
            told += (
                " While a command runs, it can call your tools: $RFNRY_TOOLS is a Unix socket and"
                " $RFNRY_TOOLS_KEY its key, for that command only. Connect, send one JSON line,"
                ' {"key": <the key>, "tool": <name>, "args": {...}}, and read one line back:'
                ' {"result": <what you would read for that call>} or {"error": <its error>}.'
                f" Each connection makes one call; open several to make calls at once, up to {CALLS},"
                " and a connection past that is closed at once, with the error line or a broken pipe or"
                " a reset, so retry it later. A command makes at most"
                f" {COMMAND_CALLS} calls in all, sending at most {COMMAND_BYTES // 1_048_576} MiB"
                " of request lines; past either, each further call is answered with an error, so"
                " spread larger work over several commands. Every tool you have is there except"
                " the workspace's own (Shell, Job, Jobs, Write, Edit), Delegate and Output."
            )
        return told

    async def _job(self, args: dict[str, Any]) -> dict[str, Any]:
        a = JobArgs.model_validate(args)
        if not 0 <= a.wait <= self.timeout:
            raise InputError(
                f"tools/workspace: wait must be between 0 and {self.timeout:g} seconds; got {a.wait:g}"
            )
        deadline = asyncio.get_running_loop().time() + self.timeout
        job = self._jobs.get(a.id)
        if job is None:
            raise InputError(f"tools/workspace: there is no job {a.id}")
        if a.wait:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(a.wait):
                    await job.ended.wait()
        if a.stop and not job.ended.is_set():
            [error] = await _see_through(job.process.kill())
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
                {**info.model_dump(), "started": info.started.isoformat(timespec="seconds")}
                for info in self.jobs()
            ]
        }

    def jobs(self) -> list[JobInfo]:
        return [
            JobInfo(
                job=name,
                command=job.command,
                started=job.started,
                running=not job.ended.is_set(),
                exit_code=job.code if job.ended.is_set() else None,
            )
            for name, job in self._jobs.items()
        ]

    def output(self, job: str) -> str:
        found = self._jobs.get(job)
        if found is None:
            raise InputError(f"tools/workspace: there is no job {job}")
        return found.shown.show(final=found.ended.is_set())

    async def _bounded[T](self, name: str, work: Awaitable[T]) -> T:
        deadline = asyncio.timeout(self.timeout)
        try:
            async with deadline:
                return await work
        except TimeoutError:
            if not deadline.expired():
                raise
            raise TimeoutError(
                f"tools/workspace: {name} did not finish within {self.timeout:g}s"
            ) from None

    async def _write(self, args: dict[str, Any]) -> str:
        a = WriteArgs.model_validate(args)
        data = a.content.encode("utf-8")
        await self._bounded("Write", self.backend.write(a.path, data))
        return f"wrote {len(data)} bytes to {a.path}"

    async def _edit(self, args: dict[str, Any]) -> str:
        a = EditArgs.model_validate(args)
        if not a.old:
            raise InputError("tools/workspace: old must not be empty")
        count = await self._bounded("Edit", self._replace(a))
        return f"edited {a.path} ({count} occurrence{'' if count == 1 else 's'})"

    async def _replace(self, a: EditArgs) -> int:
        try:
            text = (await self.backend.read(a.path, TEXT_BYTES)).decode("utf-8")
        except UnicodeDecodeError:
            raise InputError(f"tools/workspace: not a UTF-8 text file: {a.path}") from None
        count = text.count(a.old)
        if count == 0 or (count > 1 and not a.all):
            need = "at least once" if a.all else "exactly once"
            raise InputError(
                f"tools/workspace: old appears {count} times in {a.path}; it must appear {need}"
            )
        await self.backend.write(
            a.path, text.replace(a.old, a.new, -1 if a.all else 1).encode("utf-8")
        )
        return count
