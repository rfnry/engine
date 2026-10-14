import asyncio
import os
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.workspace import Backend, Local, Process, Stream, Workspace


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state != "Z"


async def gone(pid: int) -> bool:
    for _ in range(40):
        if not alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


async def appears(path: Path) -> None:
    for _ in range(100):
        if path.exists():
            return
        await asyncio.sleep(0.02)
    pytest.fail(f"{path.name} never appeared")


async def everything(stream: Stream) -> bytes:
    chunks = []
    while chunk := await stream.read(65_536):
        chunks.append(chunk)
    return b"".join(chunks)


async def finished(process: Process) -> tuple[int, bytes, bytes]:
    out, err, code = await asyncio.gather(
        everything(process.stdout), everything(process.stderr), process.wait()
    )
    return code, out, err


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def local(root: Path) -> Local:
    return Local(root)


def test_a_local_is_a_backend(local: Local) -> None:
    assert isinstance(local, Backend)


async def test_a_command_runs_in_the_root_with_both_streams(local: Local, root: Path) -> None:
    process = await local.spawn("pwd; echo err >&2; echo hi > made.txt; exit 3")
    assert await finished(process) == (3, f"{root}\n".encode(), b"err\n")
    assert (root / "made.txt").read_text() == "hi\n"


async def test_the_default_environment_does_not_leak_the_service(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SERVICE_SECRET", "s3cret")
    process = await Local(root).spawn('echo "[$SERVICE_SECRET]"; echo "$HOME"')
    assert await finished(process) == (0, f"[]\n{root}\n".encode(), b"")


async def test_env_passes_through(root: Path) -> None:
    local = Local(root, env={"PATH": os.environ.get("PATH", os.defpath), "GREETING": "hi"})
    assert (await finished(await local.spawn('echo "$GREETING"')))[1] == b"hi\n"


async def test_the_default_shell_is_bash(local: Local) -> None:
    assert Path(local.shell[0]).name == "bash" and local.shell[1:] == ["-c"]
    process = await local.spawn(
        'set -o pipefail; for i in {1..3}; do [[ $i -gt 1 ]] && echo "$i"; done'
    )
    assert await finished(process) == (0, b"2\n3\n", b"")


async def test_a_sandbox_goes_in_front_of_the_shell(root: Path) -> None:
    wrapped = Local(root, shell=["env", "WRAPPED=yes", "bash", "-c"])
    assert (await finished(await wrapped.spawn('echo "$WRAPPED"')))[1] == b"yes\n"
    posix = Local(root, shell=["sh", "-c"])
    assert (await finished(await posix.spawn("echo plain")))[1] == b"plain\n"


async def test_wait_returns_on_exit_while_a_child_holds_the_output(local: Local) -> None:
    process = await local.spawn("sleep 30 & echo $!")
    began = time.monotonic()
    assert await asyncio.wait_for(process.wait(), 5) == 0
    assert time.monotonic() - began < 2
    pid = int((await process.stdout.read(100)).decode())
    assert alive(pid)
    await process.kill()
    assert await gone(pid)


async def test_kill_sends_sigterm_first_so_a_command_can_clean_up(local: Local, root: Path) -> None:
    process = await local.spawn(
        "trap 'echo stopped > stopped.txt; exit 0' TERM; touch ready; sleep 30 & wait"
    )
    await appears(root / "ready")
    began = time.monotonic()
    await process.kill()
    assert time.monotonic() - began < 0.9
    assert await process.wait() == 0
    assert (root / "stopped.txt").read_text() == "stopped\n"


async def test_kill_ends_a_command_that_ignores_sigterm_a_second_later(
    local: Local, root: Path
) -> None:
    process = await local.spawn("trap '' TERM; sleep 30 & echo $! > child.pid; touch ready; wait")
    await appears(root / "ready")
    began = time.monotonic()
    await process.kill()
    assert 0.9 < time.monotonic() - began < 3
    assert await asyncio.wait_for(process.wait(), 2) == -9
    assert await gone(int((root / "child.pid").read_text()))


async def test_kill_after_the_exit_ends_what_the_command_left_running(
    local: Local, root: Path
) -> None:
    process = await local.spawn("sleep 30 & echo $! > child.pid")
    assert await asyncio.wait_for(process.wait(), 5) == 0
    pid = int((root / "child.pid").read_text())
    assert alive(pid)
    await process.kill()
    await process.kill()
    assert await gone(pid)


async def test_write_creates_parents_and_read_returns_the_bytes(local: Local, root: Path) -> None:
    await local.write("src/app/main.py", b"a\r\nb\n")
    assert (root / "src/app/main.py").read_bytes() == b"a\r\nb\n"
    assert await local.read("src/app/main.py", 100) == b"a\r\nb\n"


async def test_read_and_write_stay_inside_the_root(
    local: Local, root: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.txt").write_text("s")
    (root / "out").symlink_to(elsewhere)
    for path in ("../x.txt", "/etc/x.txt", "out/x.txt"):
        with pytest.raises(InputError, match=f"^workspace: path outside the workspace: {path}$"):
            await local.write(path, b"x")
    with pytest.raises(InputError, match="^workspace: path outside the workspace: out/secret.txt$"):
        await local.read("out/secret.txt", 100)
    assert not (elsewhere / "x.txt").exists()


async def test_read_refuses_a_missing_file_a_directory_and_a_file_over_the_limit(
    local: Local, root: Path
) -> None:
    (root / "dir").mkdir()
    (root / "big.txt").write_bytes(b"x" * 11)
    with pytest.raises(InputError, match="^workspace: not a file: missing.txt$"):
        await local.read("missing.txt", 100)
    with pytest.raises(InputError, match="^workspace: not a file: dir$"):
        await local.read("dir", 100)
    with pytest.raises(InputError, match="^workspace: file exceeds 10 bytes: big.txt$"):
        await local.read("big.txt", 10)
    assert await local.read("big.txt", 11) == b"x" * 11


async def test_write_refuses_a_directory(local: Local, root: Path) -> None:
    (root / "dir").mkdir()
    with pytest.raises(InputError, match="^workspace: is a directory: dir$"):
        await local.write("dir", b"x")
    assert (root / "dir").is_dir()


async def test_read_reads_up_to_the_limit_across_chunks(local: Local, root: Path) -> None:
    (root / "big.bin").write_bytes(b"y" * 200_001)
    with pytest.raises(InputError, match="^workspace: file exceeds 200000 bytes: big.bin$"):
        await local.read("big.bin", 200_000)
    assert await local.read("big.bin", 200_001) == b"y" * 200_001


async def test_a_pipe_is_not_a_file_and_never_blocks(local: Local, root: Path) -> None:
    os.mkfifo(root / "pipe")
    with pytest.raises(InputError, match="^workspace: not a file: pipe$"):
        await asyncio.wait_for(local.read("pipe", 100), 5)
    with pytest.raises(InputError, match="^workspace: not a file: pipe$"):
        await asyncio.wait_for(local.write("pipe", b"x"), 5)


async def test_a_link_inside_the_workspace_still_reads_and_writes(local: Local, root: Path) -> None:
    (root / "real").mkdir()
    (root / "real/a.txt").write_bytes(b"a")
    (root / "alias").symlink_to("real")
    (root / "link.txt").symlink_to("real/a.txt")
    assert await local.read("alias/a.txt", 10) == b"a"
    assert await local.read("link.txt", 10) == b"a"
    await local.write("alias/deep/b.txt", b"b")
    assert (root / "real/deep/b.txt").read_bytes() == b"b"
    await local.write("link.txt", b"c")
    assert (root / "real/a.txt").read_bytes() == b"c" and (root / "link.txt").is_symlink()


def swap_after_the_check(monkeypatch: pytest.MonkeyPatch, swap: Callable[[], None]) -> None:
    checked = Local._target

    def swapped(self: Local, path: str) -> Path:
        target = checked(self, path)
        swap()
        return target

    monkeypatch.setattr(Local, "_target", swapped)


async def call(local: Local, verb: str, path: str) -> None:
    if verb == "read":
        await local.read(path, 100)
    else:
        await local.write(path, b"x")


@pytest.mark.parametrize(
    ("verb", "path"),
    [("read", "sub/secret.txt"), ("write", "sub/secret.txt"), ("write", "sub/new/new.txt")],
)
async def test_a_folder_swapped_for_a_link_after_the_check_is_refused(
    local: Local,
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verb: str,
    path: str,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.txt").write_text("secret")
    (root / "sub").mkdir()

    def swap() -> None:
        (root / "sub").rmdir()
        (root / "sub").symlink_to(elsewhere)

    swap_after_the_check(monkeypatch, swap)
    with pytest.raises(InputError, match=f"^workspace: path outside the workspace: {path}$"):
        await call(local, verb, path)
    assert [p.name for p in elsewhere.iterdir()] == ["secret.txt"]
    assert (elsewhere / "secret.txt").read_text() == "secret"


@pytest.mark.parametrize("verb", ["read", "write"])
async def test_a_file_swapped_for_a_link_after_the_check_is_refused(
    local: Local, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.txt").write_text("secret")
    (root / "notes.txt").write_text("notes")

    def swap() -> None:
        (root / "notes.txt").unlink()
        (root / "notes.txt").symlink_to(elsewhere / "secret.txt")

    swap_after_the_check(monkeypatch, swap)
    with pytest.raises(InputError, match="^workspace: path outside the workspace: notes.txt$"):
        await call(local, verb, "notes.txt")
    assert (elsewhere / "secret.txt").read_text() == "secret"


def test_root_must_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="^workspace: root is not a directory: "):
        Local(tmp_path / "missing")


def test_a_missing_shell_is_refused_with_the_way_out(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match=r"^workspace: bash was not found; pass shell=, such as \['sh', '-c'\]$",
    ):
        Local(tmp_path, env={"PATH": str(tmp_path)})


@pytest.mark.parametrize("shell", ["bash -c", [], ["bash", ""]])
def test_a_shell_that_is_not_a_list_of_words_is_refused(tmp_path: Path, shell: Any) -> None:
    with pytest.raises(
        ConfigurationError, match="^workspace: shell must be a non-empty list of strings"
    ):
        Local(tmp_path, shell=shell)


def test_a_description_must_be_a_string(tmp_path: Path) -> None:
    assert Local(tmp_path, description="bash on a laptop").description == "bash on a laptop"
    with pytest.raises(ConfigurationError, match="^workspace: description must be a string$"):
        Local(tmp_path, description=3)  # type: ignore[arg-type]


CHILD_TRAP = "(trap 'sleep 0.3; echo cleaned > cleaned.txt; exit 0' TERM; sleep 30 & wait) & "


async def test_kill_gives_a_child_the_grace_to_clean_up_while_the_leader_runs(
    local: Local, root: Path
) -> None:
    process = await local.spawn(CHILD_TRAP + "sleep 30")
    await asyncio.sleep(0.3)
    began = time.monotonic()
    await process.kill()
    assert time.monotonic() - began < 2
    assert (root / "cleaned.txt").read_text() == "cleaned\n"


async def test_kill_gives_a_child_the_grace_to_clean_up_after_the_leader_exited(
    local: Local, root: Path
) -> None:
    process = await local.spawn(CHILD_TRAP + "echo started")
    assert await asyncio.wait_for(process.wait(), 5) == 0
    await asyncio.sleep(0.3)
    began = time.monotonic()
    await process.kill()
    assert time.monotonic() - began < 2
    assert (root / "cleaned.txt").read_text() == "cleaned\n"


def workspace(root: Path, **kwargs: Any) -> Workspace:
    return Workspace(Local(root), scope="acme", **kwargs)


def tool(ws: Workspace, name: str) -> Any:
    return next(t for t in ws.tools() if t.name == name)


async def shell(ws: Workspace, command: str) -> dict[str, Any]:
    result = await tool(ws, "Shell").call({"command": command})
    assert isinstance(result, dict)
    return result


async def test_shell_runs_a_real_command(root: Path) -> None:
    result = await shell(workspace(root), "echo out; echo err >&2; exit 3")
    assert result == {"exit_code": 3, "stdout": "out\n", "stderr": "err\n", "timed_out": False}


async def test_real_output_is_capped_head_and_tail(root: Path) -> None:
    ws = workspace(root)
    out = (await shell(ws, "head -c 200000 /dev/zero | tr '\\0' a"))["stdout"]
    assert len(out) < ws.output_bytes + 100
    assert "bytes omitted" in out and out.startswith("a") and out.endswith("a")


async def test_the_deadline_kills_the_process_tree(root: Path) -> None:
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.5):
            await shell(workspace(root), "sleep 30 & echo $! > child.pid; wait")
    assert await gone(int((root / "child.pid").read_text()))


async def test_endless_output_is_bounded_and_killed(root: Path) -> None:
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(1.0):
            await shell(workspace(root), "yes & echo $! > yes.pid; wait")
    assert await gone(int((root / "yes.pid").read_text()))


async def test_a_background_process_does_not_hold_the_call(root: Path) -> None:
    began = time.monotonic()
    result = await shell(workspace(root), "sleep 30 & echo $!; echo started")
    assert time.monotonic() - began < 5
    pid, started = result["stdout"].split()
    assert started == "started"
    assert await gone(int(pid))


async def test_parallel_shell_calls_run_independently(root: Path) -> None:
    ws = workspace(root)
    began = time.monotonic()
    a, b = await asyncio.gather(shell(ws, "sleep 0.3; echo a"), shell(ws, "sleep 0.3; echo b"))
    assert (a["stdout"], b["stdout"]) == ("a\n", "b\n")
    assert time.monotonic() - began < 2


async def test_the_timeout_stops_the_command_and_keeps_what_it_printed(root: Path) -> None:
    began = time.monotonic()
    result = await shell(
        workspace(root, timeout=0.5), "echo before; sleep 30 & echo $! > child.pid; wait"
    )
    assert time.monotonic() - began < 5
    assert result["timed_out"] is True and result["stdout"] == "before\n"
    assert await gone(int((root / "child.pid").read_text()))


async def test_the_timeout_sends_sigterm_first_so_a_command_can_clean_up(root: Path) -> None:
    began = time.monotonic()
    result = await shell(
        workspace(root, timeout=0.5),
        "trap 'echo stopping; exit 0' TERM; echo started; sleep 30 & wait",
    )
    assert time.monotonic() - began < 1.4
    assert result == {
        "exit_code": 0,
        "stdout": "started\nstopping\n",
        "stderr": "",
        "timed_out": True,
    }


async def test_a_command_that_ignores_sigterm_is_killed_a_second_later(root: Path) -> None:
    began = time.monotonic()
    result = await shell(
        workspace(root, timeout=0.5), "trap '' TERM; sleep 30 & echo $! > child.pid; wait"
    )
    assert 1.4 < time.monotonic() - began < 4
    assert result["timed_out"] is True and result["exit_code"] == -9
    assert await gone(int((root / "child.pid").read_text()))


async def test_a_cancelled_command_gets_sigterm_first(root: Path) -> None:
    task = asyncio.create_task(
        shell(
            workspace(root),
            "trap 'echo stopped > stopped.txt; exit 0' TERM; touch ready; sleep 30 & wait",
        )
    )
    await appears(root / "ready")
    began = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - began < 0.9
    assert (root / "stopped.txt").read_text() == "stopped\n"


async def test_a_cancelled_command_that_ignores_sigterm_is_killed_a_second_later(
    root: Path,
) -> None:
    task = asyncio.create_task(
        shell(workspace(root), "trap '' TERM; sleep 30 & echo $! > child.pid; touch ready; wait")
    )
    await appears(root / "ready")
    began = time.monotonic()
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done and task.cancelled()
    assert 0.9 < time.monotonic() - began < 3
    assert await gone(int((root / "child.pid").read_text()))


async def test_a_second_cancel_kills_at_once(root: Path) -> None:
    task = asyncio.create_task(
        shell(workspace(root), "trap '' TERM; sleep 30 & echo $! > child.pid; touch ready; wait")
    )
    await appears(root / "ready")
    began = time.monotonic()
    task.cancel()
    await asyncio.sleep(0.2)
    assert not task.done()
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done and task.cancelled()
    assert time.monotonic() - began < 0.8
    assert await gone(int((root / "child.pid").read_text()))


async def test_write_and_edit_reach_the_directory(root: Path, tmp_path: Path) -> None:
    ws = workspace(root)
    written = await tool(ws, "Write").call({"path": "src/main.py", "content": "x = 1\r\ny = 2\n"})
    assert written == "wrote 13 bytes to src/main.py"
    edited = await tool(ws, "Edit").call({"path": "src/main.py", "old": "x = 1", "new": "x = 3"})
    assert edited == "edited src/main.py (1 occurrence)"
    assert (root / "src/main.py").read_bytes() == b"x = 3\r\ny = 2\n"
    with pytest.raises(InputError, match="^workspace: path outside the workspace: ../x.txt$"):
        await tool(ws, "Write").call({"path": "../x.txt", "content": "x"})
    assert not (tmp_path / "x.txt").exists()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def written_pid(path: Path) -> int:
    for _ in range(100):
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        await asyncio.sleep(0.02)
    pytest.fail(f"{path.name} never held a pid")


async def test_a_server_started_as_a_job_answers_the_next_command(root: Path) -> None:
    port = free_port()
    (root / "fetch.py").write_text(
        "import time, urllib.request\n"
        "for _ in range(100):\n"
        "    try:\n"
        f"        print(urllib.request.urlopen('http://127.0.0.1:{port}/').status)\n"
        "        break\n"
        "    except OSError:\n"
        "        time.sleep(0.05)\n"
    )
    ws = workspace(root)
    serve = f"{sys.executable} -m http.server {port} --bind 127.0.0.1"
    assert await tool(ws, "Shell").call({"command": serve, "background": True}) == {"job": "j1"}
    assert (await shell(ws, f"{sys.executable} fetch.py"))["stdout"] == "200\n"
    stopped = await tool(ws, "Job").call({"id": "j1", "stop": True})
    assert stopped["running"] is False and '"GET / HTTP/1.1" 200' in stopped["output"]


async def test_a_job_that_cannot_run_ends_with_its_error(root: Path) -> None:
    ws = workspace(root)
    await tool(ws, "Shell").call({"command": "no-such-command-anywhere", "background": True})
    for _ in range(100):
        [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
        if not listed["running"]:
            break
        await asyncio.sleep(0.02)
    result = await tool(ws, "Job").call({"id": "j1"})
    assert result["running"] is False and result["exit_code"] == 127
    assert "no-such-command-anywhere: command not found" in result["output"]


async def test_closing_the_workspace_stops_a_real_job_and_what_it_started(root: Path) -> None:
    ws = workspace(root)
    command = "sleep 30 & echo $! > child.pid; wait"
    await tool(ws, "Shell").call({"command": command, "background": True})
    pid = await written_pid(root / "child.pid")
    await ws.close()
    assert await gone(pid)
