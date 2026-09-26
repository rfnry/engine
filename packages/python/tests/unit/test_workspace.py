import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.workspace import Workspace


def tool(ws: Workspace, name: str) -> Any:
    return next(t for t in ws.tools() if t.name == name)


async def shell(ws: Workspace, command: str) -> dict[str, Any]:
    result = await tool(ws, "Shell").call({"command": command})
    assert isinstance(result, dict)
    return result


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


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "work"
    root.mkdir()
    return Workspace(root)


def test_root_must_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not a directory"):
        Workspace(tmp_path / "missing")


def test_three_tools_in_order(ws: Workspace) -> None:
    assert [t.name for t in ws.tools()] == ["Shell", "Write", "Edit"]


async def test_shell_returns_exit_code_and_both_streams(ws: Workspace) -> None:
    result = await shell(ws, "echo out; echo err >&2; exit 3")
    assert result == {"exit_code": 3, "stdout": "out\n", "stderr": "err\n", "timed_out": False}


async def test_shell_runs_in_the_root(ws: Workspace) -> None:
    result = await shell(ws, "pwd; echo hi > made.txt")
    assert result["stdout"] == f"{ws.root}\n"
    assert (ws.root / "made.txt").read_text() == "hi\n"


async def test_the_default_environment_does_not_leak_the_service(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVICE_SECRET", "s3cret")
    result = await shell(Workspace(ws.root), 'echo "[$SERVICE_SECRET]"; echo "$HOME"')
    assert result["stdout"] == f"[]\n{ws.root}\n"


async def test_env_passes_through(ws: Workspace) -> None:
    custom = Workspace(ws.root, env={"PATH": os.environ.get("PATH", os.defpath), "GREETING": "hi"})
    assert (await shell(custom, 'echo "$GREETING"'))["stdout"] == "hi\n"


async def test_output_is_capped_head_and_tail(ws: Workspace) -> None:
    result = await shell(ws, "head -c 200000 /dev/zero | tr '\\0' a")
    out = result["stdout"]
    assert len(out) < ws.output_bytes + 100
    assert "bytes omitted" in out and out.startswith("a") and out.endswith("a")


async def test_binary_output_is_replaced(ws: Workspace) -> None:
    result = await shell(ws, "printf '\\377\\376ok'")
    assert "ok" in result["stdout"] and "�" in result["stdout"]


async def test_the_deadline_kills_the_process_tree(ws: Workspace) -> None:
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.5):
            await shell(ws, "sleep 30 & echo $! > child.pid; wait")
    assert await gone(int((ws.root / "child.pid").read_text()))


async def test_endless_output_is_bounded_and_killed(ws: Workspace) -> None:
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(1.0):
            await shell(ws, "yes & echo $! > yes.pid; wait")
    assert await gone(int((ws.root / "yes.pid").read_text()))


async def test_a_background_process_does_not_hold_the_call(ws: Workspace) -> None:
    began = time.monotonic()
    result = await shell(ws, "sleep 30 & echo $!; echo started")
    assert time.monotonic() - began < 5
    pid, started = result["stdout"].split()
    assert started == "started"
    assert await gone(int(pid))


async def test_parallel_shell_calls_run_independently(ws: Workspace) -> None:
    began = time.monotonic()
    a, b = await asyncio.gather(shell(ws, "sleep 0.3; echo a"), shell(ws, "sleep 0.3; echo b"))
    assert (a["stdout"], b["stdout"]) == ("a\n", "b\n")
    assert time.monotonic() - began < 2


async def test_write_creates_parents_and_is_exact(ws: Workspace) -> None:
    message = await tool(ws, "Write").call({"path": "src/app/main.py", "content": "a\r\nb\n"})
    assert (ws.root / "src/app/main.py").read_bytes() == b"a\r\nb\n"
    assert message == "wrote 5 bytes to src/app/main.py"


async def test_write_refuses_paths_outside_the_root(ws: Workspace, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (ws.root / "out").symlink_to(elsewhere)
    for path in ("../x.txt", "/etc/x.txt", "out/x.txt"):
        with pytest.raises(InputError, match="outside the workspace"):
            await tool(ws, "Write").call({"path": path, "content": "x"})
    assert not (elsewhere / "x.txt").exists()


async def test_write_refuses_a_directory(ws: Workspace) -> None:
    (ws.root / "dir").mkdir()
    with pytest.raises(InputError, match="is a directory"):
        await tool(ws, "Write").call({"path": "dir", "content": "x"})
    assert (ws.root / "dir").is_dir()


async def test_edit_replaces_exactly_one_occurrence(ws: Workspace) -> None:
    (ws.root / "f.txt").write_bytes(b"x = 1\r\ny = 2\n")
    assert await tool(ws, "Edit").call({"path": "f.txt", "old": "x = 1", "new": "x = 3"}) == "edited f.txt"
    assert (ws.root / "f.txt").read_bytes() == b"x = 3\r\ny = 2\n"


@pytest.mark.parametrize(("content", "count"), [("a = 1\n", 0), ("x\nx\n", 2)])
async def test_edit_refuses_zero_or_several_matches(ws: Workspace, content: str, count: int) -> None:
    (ws.root / "f.txt").write_text(content)
    with pytest.raises(InputError, match=f"appears {count} times"):
        await tool(ws, "Edit").call({"path": "f.txt", "old": "x", "new": "y"})
    assert (ws.root / "f.txt").read_text() == content


async def test_edit_refuses_empty_old_missing_file_and_binary(ws: Workspace) -> None:
    (ws.root / "bin").write_bytes(b"\xff\xfe")
    (ws.root / "f.txt").write_text("x")
    with pytest.raises(InputError, match="must not be empty"):
        await tool(ws, "Edit").call({"path": "f.txt", "old": "", "new": "y"})
    with pytest.raises(InputError, match="not a file"):
        await tool(ws, "Edit").call({"path": "missing.txt", "old": "x", "new": "y"})
    with pytest.raises(InputError, match="not a UTF-8 text file"):
        await tool(ws, "Edit").call({"path": "bin", "old": "x", "new": "y"})


def test_shell_tells_the_model_its_terms(tmp_path: Path) -> None:
    description = next(
        t.description
        for t in Workspace(tmp_path, timeout=45, output_bytes=8000).tools()
        if t.name == "Shell"
    )
    assert description.startswith("Run a bash command. It starts in the workspace directory")
    assert "relative paths work" in description and "does not carry over" in description
    assert "stopped after 45 seconds" in description and "first and last 4000 bytes" in description


async def test_the_default_shell_is_bash(ws: Workspace) -> None:
    assert Path(ws.shell[0]).name == "bash" and ws.shell[1:] == ["-c"]
    result = await shell(
        ws, 'set -o pipefail; for i in {1..3}; do [[ $i -gt 1 ]] && echo "$i"; done'
    )
    assert result["stdout"] == "2\n3\n" and result["exit_code"] == 0


def test_a_missing_bash_is_refused_with_the_way_out(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match=r"^workspace: bash was not found; pass shell=, such as \['sh', '-c'\]$",
    ):
        Workspace(tmp_path, env={"PATH": str(tmp_path)})


async def test_the_shell_can_be_chosen_and_wrapped(ws: Workspace) -> None:
    posix = Workspace(ws.root, shell=["sh", "-c"])
    assert (await shell(posix, "echo plain"))["stdout"] == "plain\n"
    assert next(t.description for t in posix.tools() if t.name == "Shell").startswith(
        "Run a sh command."
    )
    wrapped = Workspace(ws.root, shell=["env", "WRAPPED=yes", "bash", "-c"])
    assert (await shell(wrapped, 'echo "$WRAPPED"'))["stdout"] == "yes\n"
    assert next(t.description for t in wrapped.tools() if t.name == "Shell").startswith(
        "Run a bash command."
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shell": "bash -c"}, "shell must be a non-empty list of strings"),
        ({"shell": []}, "shell must be a non-empty list of strings"),
        ({"shell": ["bash", ""]}, "shell must be a non-empty list of strings"),
        ({"timeout": 0}, "timeout must be positive"),
        ({"output_bytes": 0}, "output_bytes must be positive"),
    ],
)
def test_a_workspace_that_cannot_work_is_refused(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=f"^workspace: {message}"):
        Workspace(tmp_path, **kwargs)


async def test_the_timeout_stops_the_command_and_keeps_what_it_printed(tmp_path: Path) -> None:
    quick = Workspace(tmp_path, timeout=0.5)
    began = time.monotonic()
    result = await shell(quick, "echo before; sleep 30 & echo $! > child.pid; wait")
    assert time.monotonic() - began < 5
    assert result["timed_out"] is True and result["stdout"] == "before\n"
    assert await gone(int((tmp_path / "child.pid").read_text()))


async def test_output_bytes_sets_the_cap(tmp_path: Path) -> None:
    small = Workspace(tmp_path, output_bytes=1000)
    out = (await shell(small, "head -c 50000 /dev/zero | tr '\\0' a"))["stdout"]
    assert len(out) < 1100 and "49000 bytes omitted" in out


async def test_write_and_edit_are_bounded_by_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow = Workspace(tmp_path, timeout=0.2)
    def late(args: dict[str, Any]) -> str:
        time.sleep(1)
        return "late"

    monkeypatch.setattr(slow, "_write", late)
    with pytest.raises(TimeoutError, match=r"^workspace: Write did not finish within 0\.2s$"):
        await tool(slow, "Write").call({"path": "a.txt", "content": "x"})
