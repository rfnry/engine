import asyncio
import contextlib
import os
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from rfnry_engine import InputError, Local, Process

TAGGED = (
    'for p in /proc/[0-9]*; do tr "\\0" "\\n" < "$p/environ" 2>/dev/null '
    '| grep -qx "RFNRY_JOB=$1" && kill -"$2" "${p#/proc/}" 2>/dev/null; done; true'
)
READ = 'test -f "$1" || exit 3; [ "$(wc -c < "$1")" -le "$2" ] || exit 4; cat -- "$1"'
WRITE = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'
GRACE = 1.0


CAUSES = {
    "no such file or directory": "no such file",
    "permission denied": "permission denied",
    "is a directory": "is a directory",
    "not a directory": "not a directory",
    "read-only file system": "read-only file system",
    "no space left": "no space left",
    "no such container": "no such container",
    "cannot connect to the docker daemon": "docker daemon unreachable",
}


def cause(stderr: bytes) -> str:
    text = stderr.decode(errors="replace").lower()
    for known, phrase in CAUSES.items():
        if known in text:
            return f": {phrase}"
    return ""


class Docker:
    def __init__(
        self,
        container: str,
        *,
        workdir: str = "/work",
        shell: Sequence[str] = ("bash", "-c"),
    ) -> None:
        self.container = container
        self.workdir = workdir
        self.shell = list(shell)
        self.env = {"PATH": os.environ["PATH"], "HOME": str(Path.home())}

    async def spawn(self, command: str) -> Process:
        tag = uuid4().hex
        client = Local(
            Path.cwd(),
            shell=[
                *("docker", "exec", "-i", "-w", self.workdir),
                *("-e", f"RFNRY_JOB={tag}", self.container, *self.shell),
            ],
            env=self.env,
        )
        return Inside(self, await client.spawn(command), tag)

    async def read(self, path: str, limit: int) -> bytes:
        code, out, err = await self.script(READ, path, str(limit))
        if code == 3:
            raise InputError(f"docker: not a file: {path}")
        if code == 4:
            raise InputError(f"docker: file exceeds {limit} bytes: {path}")
        if code:
            raise InputError(f"docker: could not read {path}{cause(err)}")
        return out

    async def write(self, path: str, data: bytes) -> None:
        code, _, err = await self.script(WRITE, path, data=data)
        if code:
            raise InputError(f"docker: could not write {path}{cause(err)}")

    async def signal(self, tag: str, name: str) -> None:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(GRACE):
                await self.script(TAGGED, tag, name)

    async def script(
        self, text: str, *args: str, data: bytes | None = None
    ) -> tuple[int, bytes, bytes]:
        process = await asyncio.create_subprocess_exec(
            *("docker", "exec", "-i", "-w", self.workdir, self.container),
            *("sh", "-c", text, "sh", *args),
            stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        try:
            out, err = await process.communicate(data)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise
        return process.returncode or 0, out, err


class Inside:
    def __init__(self, docker: Docker, client: Process, tag: str) -> None:
        self.docker = docker
        self.client = client
        self.tag = tag
        self.stdout = client.stdout
        self.stderr = client.stderr

    async def wait(self) -> int:
        return await self.client.wait()

    async def kill(self) -> None:
        try:
            await self.docker.signal(self.tag, "TERM")
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(GRACE):
                    await self.client.wait()
        finally:
            await self.docker.signal(self.tag, "KILL")
            await self.client.kill()
