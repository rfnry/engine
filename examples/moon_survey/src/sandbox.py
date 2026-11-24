import asyncio
import contextlib
import os
import shutil
import signal
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import Any

from rfnry_engine import Local, current_call

import moon

EXAMPLE = Path(__file__).resolve().parent.parent
VENV = EXAMPLE / ".venv"
SRC = EXAMPLE / "src"
KILLED = -signal.SIGKILL
SYSTEM = (
    "--ro-bind",
    "/usr",
    "/usr",
    "--symlink",
    "usr/bin",
    "/bin",
    "--symlink",
    "usr/sbin",
    "/sbin",
    "--symlink",
    "usr/lib",
    "/lib",
    "--symlink",
    "usr/lib64",
    "/lib64",
    "--proc",
    "/proc",
    "--dev",
    "/dev",
    "--tmpfs",
    "/tmp",
)
CONFINED = ("--unshare-all", "--die-with-parent", "--new-session", "bash", "-c")


class SandboxError(Exception):
    pass


def read_only(path: Path) -> list[str]:
    return ["--ro-bind", str(path), str(path)]


def runtime(venv: Path) -> list[str]:
    python = venv / "bin" / "python"
    if not python.exists():
        raise SandboxError(f"sandbox: {python} is missing; run uv sync in {venv.parent}")
    real = python.resolve().parent.parent
    linked = real
    if python.is_symlink():
        linked = Path(os.path.normpath(python.parent / os.readlink(python))).parent.parent
    words = read_only(real)
    if linked != real:
        words += ["--symlink", str(real), str(linked)]
    return words + read_only(venv)


def folder(path: Path, name: str) -> Path:
    found = Path(path).resolve()
    if not found.is_dir():
        raise SandboxError(f"sandbox: the {name} folder {path} is not a directory")
    return found


async def through(work: Coroutine[Any, Any, None]) -> None:
    task = asyncio.ensure_future(work)
    while not task.done():
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait([task])
    task.result()


class Later:
    def __init__(self, slotted: Slotted, name: str) -> None:
        self.slotted = slotted
        self.name = name
        self.last = b"\n"
        self.told = False

    async def read(self, n: int) -> bytes:
        await self.slotted.ready.wait()
        if self.slotted.process is None:
            return b""
        chunk = await getattr(self.slotted.process, self.name).read(n)
        if chunk:
            self.last = chunk[-1:]
            return chunk
        if self.name == "stderr" and self.slotted.stopped and not self.told:
            self.told = True
            notice = (
                f"sandbox: stopped: a command may run for at most {self.slotted.limit:g} seconds\n"
            )
            return (b"" if self.last == b"\n" else b"\n") + notice.encode()
        return b""


class Slotted:
    def __init__(
        self,
        local: Local,
        command: str,
        env: Mapping[str, str],
        slots: asyncio.Semaphore,
        limit: float,
    ) -> None:
        self.stdout = Later(self, "stdout")
        self.stderr = Later(self, "stderr")
        self.limit = limit
        self.process: Any = None
        self.killed = False
        self.holding = False
        self.stopped = False
        self.ready = asyncio.Event()
        self.code: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self.task = asyncio.create_task(self.run(local, command, dict(env), slots))
        self.task.add_done_callback(self.ended)

    async def run(
        self, local: Local, command: str, env: dict[str, str], slots: asyncio.Semaphore
    ) -> int:
        await slots.acquire()
        self.holding = True
        try:
            if self.killed:
                return KILLED
            self.process = await local.spawn(command, env)
            self.ready.set()
            if self.killed:
                await self.process.kill()
            deadline = asyncio.timeout(self.limit)
            try:
                async with deadline:
                    return await self.process.wait()
            except TimeoutError:
                if not deadline.expired():
                    raise
            self.stopped = True
            await self.process.kill()
            return await self.process.wait()
        finally:
            self.ready.set()
            slots.release()

    def ended(self, task: asyncio.Task[int]) -> None:
        self.ready.set()
        if task.cancelled():
            self.code.set_result(KILLED)
        elif (error := task.exception()) is not None:
            self.code.set_exception(error)
        else:
            self.code.set_result(task.result())

    async def wait(self) -> int:
        return await asyncio.shield(self.code)

    async def kill(self) -> None:
        self.killed = True
        if not self.holding:
            self.task.cancel()
            return
        try:
            await self.ready.wait()
        except asyncio.CancelledError:
            await through(self.end_now())
            raise
        if self.process is not None and not self.task.done():
            await self.process.kill()

    async def end_now(self) -> None:
        await self.ready.wait()
        if self.process is None or self.task.done():
            return
        killing = asyncio.ensure_future(self.process.kill())
        await asyncio.sleep(0)
        killing.cancel()
        await asyncio.wait([killing])


class Sandboxes:
    def __init__(
        self,
        base: Path,
        *,
        agents: Path,
        data: Path,
        slots: int = 16,
        run_timeout: float = 120.0,
    ) -> None:
        if slots < 1:
            raise SandboxError(f"sandbox: slots must be at least 1; got {slots}")
        if run_timeout <= 0:
            raise SandboxError(f"sandbox: run_timeout must be positive; got {run_timeout:g}")
        self.base = Path(base).resolve()
        self.agents = folder(agents, "agents")
        self.data = folder(data, "data")
        self.run_timeout = run_timeout
        self.path = f"{VENV / 'bin'}:/usr/bin:/bin"
        self.bwrap = shutil.which("bwrap", path=self.path)
        self.mounts = [
            *SYSTEM,
            *runtime(VENV),
            *read_only(SRC),
            *read_only(self.agents),
            *read_only(self.data),
            *read_only(moon.SOURCES),
        ]
        self.slots = asyncio.Semaphore(slots)

    def place(self, run_id: str) -> Path:
        if run_id in ("", ".", "..") or "/" in run_id or "\0" in run_id:
            raise SandboxError(f"sandbox: run id {run_id!r} cannot name a directory")
        return self.base / run_id

    def moved(self, run_id: str, to: str) -> None:
        held = self.place(run_id)
        if held.is_dir():
            held.rename(self.place(to))

    def shell(self, place: Path) -> list[str]:
        if self.bwrap is None:
            raise SandboxError("sandbox: bwrap was not found; install bubblewrap")
        return [
            self.bwrap,
            *self.mounts,
            "--bind",
            str(place),
            str(place),
            "--remount-ro",
            "/",
            "--chdir",
            str(place),
            *CONFINED,
        ]

    def here(self) -> Path:
        call = current_call()
        if call is None:
            raise SandboxError(
                "sandbox: no run is calling; a sandbox is reached only in a tool call"
            )
        place = self.place(call.run_id)
        place.mkdir(parents=True, exist_ok=True)
        return place

    def env(self, place: Path) -> dict[str, str]:
        return {
            "PATH": self.path,
            "HOME": str(place),
            "LANG": "C.UTF-8",
            "MOON_SURVEY_DATA": str(self.data),
            "MOON_SURVEY_SRC": str(SRC),
        }

    def files(self) -> Local:
        place = self.here()
        return Local(place, env=self.env(place))

    async def spawn(self, command: str, env: Mapping[str, str]) -> Slotted:
        place = self.here()
        local = Local(place, shell=self.shell(place), env=self.env(place))
        return Slotted(local, command, env, self.slots, self.run_timeout)

    async def read(self, path: str, limit: int) -> bytes:
        return await self.files().read(path, limit)

    async def write(self, path: str, data: bytes) -> None:
        await self.files().write(path, data)

    async def read_at(self, run_id: str, path: str, limit: int) -> bytes:
        place = self.place(run_id)
        target = (place / path).resolve()
        if target.is_relative_to(place.resolve()) and not target.is_file():
            raise FileNotFoundError(f"sandbox: '{path}' is not a file in the place of '{run_id}'")
        return await Local(place, env=self.env(place)).read(path, limit)
