import asyncio
import os
import shutil
import signal
from collections.abc import Mapping
from pathlib import Path

from rfnry_engine import Local, current_call

EXAMPLE = Path(__file__).resolve().parent.parent
VENV = EXAMPLE / ".venv"
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


class Later:
    def __init__(self, slotted: Slotted, name: str) -> None:
        self.slotted = slotted
        self.name = name

    async def read(self, n: int) -> bytes:
        await self.slotted.ready.wait()
        if self.slotted.process is None:
            return b""
        return await getattr(self.slotted.process, self.name).read(n)


class Slotted:
    def __init__(
        self, local: Local, command: str, env: Mapping[str, str], slots: asyncio.Semaphore
    ) -> None:
        self.stdout = Later(self, "stdout")
        self.stderr = Later(self, "stderr")
        self.process = None
        self.killed = False
        self.holding = False
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
        await self.ready.wait()
        if self.process is not None:
            await self.process.kill()


class Sandboxes:
    def __init__(self, base: Path, *, agents: Path, data: Path, slots: int = 16) -> None:
        if slots < 1:
            raise SandboxError(f"sandbox: slots must be at least 1; got {slots}")
        self.base = Path(base).resolve()
        self.agents = folder(agents, "agents")
        self.data = folder(data, "data")
        self.path = f"{VENV / 'bin'}:/usr/bin:/bin"
        if shutil.which("bwrap", path=self.path) is None:
            raise SandboxError("sandbox: bwrap was not found; install bubblewrap")
        self.mounts = [
            *SYSTEM,
            *runtime(VENV),
            *read_only(EXAMPLE / "src"),
            *read_only(self.agents),
            *read_only(self.data),
        ]
        self.slots = asyncio.Semaphore(slots)

    def place(self, run_id: str) -> Path:
        if run_id in ("", ".", "..") or "/" in run_id or "\0" in run_id:
            raise SandboxError(f"sandbox: run id {run_id!r} cannot name a directory")
        return self.base / run_id

    def shell(self, place: Path) -> list[str]:
        return [
            "bwrap",
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

    def here(self) -> Local:
        call = current_call()
        if call is None:
            raise SandboxError(
                "sandbox: no run is calling; a sandbox is reached only in a tool call"
            )
        place = self.place(call.run_id)
        place.mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": self.path,
            "HOME": str(place),
            "LANG": "C.UTF-8",
            "MOON_SURVEY_DATA": str(self.data),
        }
        return Local(place, shell=self.shell(place), env=env)

    async def spawn(self, command: str, env: Mapping[str, str]) -> Slotted:
        return Slotted(self.here(), command, env, self.slots)

    async def read(self, path: str, limit: int) -> bytes:
        return await self.here().read(path, limit)

    async def write(self, path: str, data: bytes) -> None:
        await self.here().write(path, data)
