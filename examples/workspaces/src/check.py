import asyncio
import subprocess
from typing import Any

from rfnry_engine import InputError, Workspace

from docker import Docker

IMAGE = "python:3.14-slim"
NAME = "rfnry-workspace-check"
FETCH = (
    'python3 -c "import time, urllib.request\n'
    "for _ in range(50):\n"
    "    try:\n"
    "        print(urllib.request.urlopen('http://127.0.0.1:8000/').read().decode(), end='')\n"
    "        break\n"
    "    except OSError:\n"
    '        time.sleep(0.1)"'
)


def tool(workspace: Workspace, name: str) -> Any:
    return next(t for t in workspace.tools() if t.name == name)


async def refused(call: Any, message: str) -> None:
    try:
        await call
    except InputError as exc:
        assert message in str(exc), exc
    else:
        raise AssertionError(f"expected: {message}")


async def main() -> None:
    subprocess.run(["docker", "rm", "-f", NAME], check=False, capture_output=True)
    subprocess.run(
        [
            *("docker", "run", "-d", "--rm", "--init", "--name", NAME),
            *("-w", "/work", IMAGE, "sleep", "infinity"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    try:
        async with Workspace(Docker(NAME), scope="check", timeout=60) as workspace:
            shell, job = tool(workspace, "Shell"), tool(workspace, "Job")
            await tool(workspace, "Write").call({"path": "site/index.html", "content": "hello\n"})
            box = Docker(NAME)
            assert await box.read("site/index.html", 100) == b"hello\n"
            await refused(box.read("site/index.html", 2), "file exceeds 2 bytes")
            await refused(box.read("site", 100), "not a file")
            await refused(box.read("missing.txt", 100), "not a file")
            serve = "python3 -m http.server 8000 --bind 127.0.0.1 --directory site"
            assert await shell.call({"command": serve, "background": True}) == {"job": "j1"}
            fetched = await shell.call({"command": FETCH})
            assert fetched["stdout"] == "hello\n", fetched
            stopped = await job.call({"id": "j1", "stop": True})
            assert not stopped["running"] and "GET / HTTP" in stopped["output"], stopped
            left = await shell.call({"command": "sleep 300 & echo started"})
            assert left["stdout"] == "started\n", left
            await shell.call({"command": "sleep 300", "background": True})
        listed = subprocess.run(
            ["docker", "exec", NAME, "sh", "-c", "cat /proc/[0-9]*/cmdline | tr '\\0' ' '"],
            capture_output=True,
            text=True,
        ).stdout
        assert "sleep 300" not in listed, listed
    finally:
        subprocess.run(["docker", "rm", "-f", NAME], check=False, stdout=subprocess.DEVNULL)
    print("ok: a job's server answered the next command, stop ended it, closing ended the rest")


if __name__ == "__main__":
    asyncio.run(main())
