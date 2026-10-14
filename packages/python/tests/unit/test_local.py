import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.workspace import Backend, Local, Process, Stream


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
