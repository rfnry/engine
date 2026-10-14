import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.workspace import TEXT_BYTES, Workspace


class Pipe:
    def __init__(self) -> None:
        self.chunks: asyncio.Queue[bytes] = asyncio.Queue()

    def say(self, data: bytes) -> None:
        self.chunks.put_nowait(data)

    def close(self) -> None:
        self.chunks.put_nowait(b"")

    async def read(self, n: int) -> bytes:
        return await self.chunks.get()


class Command:
    def __init__(self, line: str) -> None:
        self.line = line
        self.stdout = Pipe()
        self.stderr = Pipe()
        self.code: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self.kills = 0
        self.stubborn = False
        self.closed = False

    def exit(self, code: int, *, hold: bool = False) -> None:
        if not self.code.done():
            self.code.set_result(code)
        if not hold and not self.closed:
            self.closed = True
            self.stdout.close()
            self.stderr.close()

    async def wait(self) -> int:
        return await asyncio.shield(self.code)

    async def kill(self) -> None:
        self.kills += 1
        if not self.stubborn:
            self.exit(-15)


class Machine:
    def __init__(self, act: Callable[[Command], None] | None = None) -> None:
        self.act = act
        self.commands: list[Command] = []
        self.files: dict[str, bytes] = {}
        self.limits: list[int] = []

    async def spawn(self, command: str) -> Command:
        ran = Command(command)
        self.commands.append(ran)
        if self.act is not None:
            self.act(ran)
        return ran

    async def read(self, path: str, limit: int) -> bytes:
        self.limits.append(limit)
        if path not in self.files:
            raise InputError(f"machine: not a file: {path}")
        return self.files[path]

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data


def prints(out: bytes = b"", err: bytes = b"", code: int = 0) -> Callable[[Command], None]:
    def act(ran: Command) -> None:
        if out:
            ran.stdout.say(out)
        if err:
            ran.stderr.say(err)
        ran.exit(code)

    return act


def tool(ws: Workspace, name: str) -> Any:
    return next(t for t in ws.tools() if t.name == name)


async def shell(ws: Workspace, command: str) -> dict[str, Any]:
    result = await tool(ws, "Shell").call({"command": command})
    assert isinstance(result, dict)
    return result


async def started(machine: Machine, count: int = 1) -> Command:
    for _ in range(100):
        if len(machine.commands) >= count:
            return machine.commands[count - 1]
        await asyncio.sleep(0.01)
    pytest.fail("the command never started")


def test_the_tools_in_order() -> None:
    ws = Workspace(Machine(), scope="acme")
    assert [t.name for t in ws.tools()] == ["Shell", "Write", "Edit"]


async def test_shell_returns_exit_code_and_both_streams() -> None:
    machine = Machine(prints(b"out\n", b"err\n", 3))
    result = await shell(Workspace(machine, scope="acme"), "anything")
    assert result == {"exit_code": 3, "stdout": "out\n", "stderr": "err\n", "timed_out": False}
    assert machine.commands[0].line == "anything"


async def test_what_a_command_left_running_is_stopped_when_it_ends() -> None:
    machine = Machine(prints(b"ok\n"))
    await shell(Workspace(machine, scope="acme"), "serve &")
    assert machine.commands[0].kills == 1


async def test_output_is_capped_head_and_tail() -> None:
    machine = Machine(prints(b"a" * 50_000))
    out = (await shell(Workspace(machine, scope="acme", output_bytes=1000), "x"))["stdout"]
    assert out == "a" * 500 + "\n[... 49000 bytes omitted ...]\n" + "a" * 500


async def test_binary_output_is_replaced() -> None:
    machine = Machine(prints(b"\xff\xfeok"))
    assert (await shell(Workspace(machine, scope="acme"), "x"))["stdout"] == "��ok"


async def test_a_character_split_across_reads_is_kept_whole() -> None:
    def act(ran: Command) -> None:
        ran.stdout.say("é".encode()[:1])
        ran.stdout.say("é".encode()[1:])
        ran.exit(0)

    assert (await shell(Workspace(Machine(act), scope="acme"), "x"))["stdout"] == "é"


async def test_the_timeout_stops_the_command_and_keeps_what_it_printed() -> None:
    machine = Machine(lambda ran: ran.stdout.say(b"before\n"))
    began = time.monotonic()
    result = await shell(Workspace(machine, scope="acme", timeout=0.2), "x")
    assert time.monotonic() - began < 1
    assert result == {"exit_code": -15, "stdout": "before\n", "stderr": "", "timed_out": True}
    assert machine.commands[0].kills == 1


async def test_a_cancelled_command_is_stopped() -> None:
    machine = Machine()
    task = asyncio.create_task(shell(Workspace(machine, scope="acme"), "x"))
    ran = await started(machine)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ran.kills == 1


async def test_output_still_open_after_the_exit_is_read_for_at_most_a_second() -> None:
    def act(ran: Command) -> None:
        ran.stubborn = True
        ran.stdout.say(b"partial\n")
        ran.exit(0, hold=True)

    began = time.monotonic()
    result = await shell(Workspace(Machine(act), scope="acme"), "x")
    assert 0.9 < time.monotonic() - began < 2
    assert result == {"exit_code": 0, "stdout": "partial\n", "stderr": "", "timed_out": False}


async def test_write_goes_through_the_backend_exactly() -> None:
    machine = Machine()
    write = tool(Workspace(machine, scope="acme"), "Write")
    message = await write.call({"path": "src/main.py", "content": "a\r\nb\n"})
    assert message == "wrote 5 bytes to src/main.py"
    assert machine.files == {"src/main.py": b"a\r\nb\n"}


async def test_edit_replaces_exactly_one_occurrence_through_the_backend() -> None:
    machine = Machine()
    machine.files["f.txt"] = b"x = 1\r\ny = 2\n"
    edit = tool(Workspace(machine, scope="acme"), "Edit")
    assert await edit.call({"path": "f.txt", "old": "x = 1", "new": "x = 3"}) == "edited f.txt"
    assert machine.files["f.txt"] == b"x = 3\r\ny = 2\n"
    assert machine.limits == [TEXT_BYTES]


@pytest.mark.parametrize(("content", "count"), [("a = 1\n", 0), ("x\nx\n", 2)])
async def test_edit_refuses_zero_or_several_matches(content: str, count: int) -> None:
    machine = Machine()
    machine.files["f.txt"] = content.encode()
    edit = tool(Workspace(machine, scope="acme"), "Edit")
    with pytest.raises(
        InputError,
        match=f"^workspace: old appears {count} times in f.txt; it must appear exactly once$",
    ):
        await edit.call({"path": "f.txt", "old": "x", "new": "y"})
    assert machine.files["f.txt"] == content.encode()


async def test_edit_refuses_empty_old_and_binary_and_passes_the_backends_errors_on() -> None:
    machine = Machine()
    machine.files.update({"bin": b"\xff\xfe", "f.txt": b"x"})
    edit = tool(Workspace(machine, scope="acme"), "Edit")
    with pytest.raises(InputError, match="^workspace: old must not be empty$"):
        await edit.call({"path": "f.txt", "old": "", "new": "y"})
    with pytest.raises(InputError, match="^machine: not a file: missing.txt$"):
        await edit.call({"path": "missing.txt", "old": "x", "new": "y"})
    with pytest.raises(InputError, match="^workspace: not a UTF-8 text file: bin$"):
        await edit.call({"path": "bin", "old": "x", "new": "y"})


async def test_write_and_edit_are_bounded_by_the_timeout() -> None:
    class Slow(Machine):
        async def write(self, path: str, data: bytes) -> None:
            await asyncio.sleep(1)

    write = tool(Workspace(Slow(), scope="acme", timeout=0.2), "Write")
    with pytest.raises(TimeoutError, match=r"^workspace: Write did not finish within 0\.2s$"):
        await write.call({"path": "a.txt", "content": "x"})


def test_shell_tells_the_model_its_terms() -> None:
    ws = Workspace(Machine(), scope="acme", timeout=45, output_bytes=8000)
    description = tool(ws, "Shell").description
    assert description.startswith("Run a shell command. It starts in the workspace directory")
    assert "relative paths work" in description and "does not carry over" in description
    assert "stopped after 45 seconds" in description and "first and last 4000 bytes" in description
    assert "send a long output to a file" in description


def test_a_backends_description_ends_the_shells() -> None:
    class Described(Machine):
        description = "bash in Debian 13, no network"

    description = tool(Workspace(Described(), scope="acme"), "Shell").description
    assert description.endswith("\n\nbash in Debian 13, no network")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scope": ""}, "scope is required"),
        ({"timeout": 0}, "timeout must be positive"),
        ({"output_bytes": 0}, "output_bytes must be positive"),
    ],
)
def test_a_workspace_that_cannot_work_is_refused(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=f"^workspace: {message}$"):
        Workspace(Machine(), **{"scope": "acme", **kwargs})


def test_a_backend_needs_its_three_methods_and_a_string_description() -> None:
    class Numbered(Machine):
        description = 3

    with pytest.raises(
        ConfigurationError, match="^workspace: backend must have spawn, read and write$"
    ):
        Workspace(object(), scope="acme")  # type: ignore[arg-type]
    with pytest.raises(
        ConfigurationError, match="^workspace: a backend's description must be a string$"
    ):
        Workspace(Numbered(), scope="acme")
