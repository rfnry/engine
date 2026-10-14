import asyncio
import time
from collections.abc import Callable
from datetime import datetime
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
    assert [t.name for t in ws.tools()] == ["Shell", "Job", "Jobs", "Write", "Edit"]


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


async def background(ws: Workspace, command: str) -> Any:
    return await tool(ws, "Shell").call({"command": command, "background": True})


async def test_a_background_command_returns_its_job_at_once() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    assert await background(ws, "serve") == {"job": "j1"}
    assert await background(ws, "watch") == {"job": "j2"}
    assert [c.line for c in machine.commands] == ["serve", "watch"]
    await ws.close()


async def test_job_returns_new_output_in_the_order_it_arrived_then_nothing_new() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    ran = machine.commands[0]
    for pipe, data in (
        (ran.stdout, b"listening\n"),
        (ran.stderr, b"warning\n"),
        (ran.stdout, b"GET /\n"),
    ):
        pipe.say(data)
        await asyncio.sleep(0.01)
    job = tool(ws, "Job")
    assert await job.call({"id": "j1"}) == {
        "output": "listening\nwarning\nGET /\n",
        "running": True,
        "exit_code": None,
    }
    assert await job.call({"id": "j1"}) == {"output": "", "running": True, "exit_code": None}
    await ws.close()


async def test_a_jobs_output_is_capped_head_and_tail_between_reads() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme", output_bytes=1000)
    await background(ws, "serve")
    machine.commands[0].stdout.say(b"a" * 50_000)
    await asyncio.sleep(0.05)
    output = (await tool(ws, "Job").call({"id": "j1"}))["output"]
    assert output == "a" * 500 + "\n[... 49000 bytes omitted ...]\n" + "a" * 500
    await ws.close()


async def test_stop_ends_a_job_returns_its_last_output_and_then_it_is_gone() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    ran = machine.commands[0]
    ran.stdout.say(b"bye\n")
    await asyncio.sleep(0.01)
    assert await tool(ws, "Job").call({"id": "j1", "stop": True}) == {
        "output": "bye\n",
        "running": False,
        "exit_code": -15,
    }
    assert ran.kills >= 1
    with pytest.raises(InputError, match="^workspace: there is no job j1$"):
        await tool(ws, "Job").call({"id": "j1"})
    assert await tool(ws, "Jobs").call({}) == {"jobs": []}


async def test_a_job_that_ended_is_listed_until_its_last_output_is_read() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "build")
    [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert (listed["job"], listed["command"], listed["running"], listed["exit_code"]) == (
        "j1",
        "build",
        True,
        None,
    )
    assert datetime.fromisoformat(listed["started"]).tzinfo is not None
    ran = machine.commands[0]
    ran.stdout.say(b"built\n")
    ran.exit(0)
    await asyncio.sleep(0.05)
    [ended] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert (ended["running"], ended["exit_code"]) == (False, 0)
    assert await tool(ws, "Job").call({"id": "j1"}) == {
        "output": "built\n",
        "running": False,
        "exit_code": 0,
    }
    assert await tool(ws, "Jobs").call({}) == {"jobs": []}


async def test_output_that_arrives_after_the_exit_is_kept() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "build")
    ran = machine.commands[0]
    ran.stubborn = True
    ran.exit(0, hold=True)
    await asyncio.sleep(0.05)
    ran.stdout.say(b"late\n")
    ran.stdout.close()
    ran.stderr.close()
    await asyncio.sleep(0.05)
    assert await tool(ws, "Job").call({"id": "j1"}) == {
        "output": "late\n",
        "running": False,
        "exit_code": 0,
    }


async def test_stopping_a_job_that_ended_returns_its_output_without_a_kill() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "build")
    ran = machine.commands[0]
    ran.stdout.say(b"done\n")
    ran.exit(0)
    await asyncio.sleep(0.05)
    kills = ran.kills
    assert await tool(ws, "Job").call({"id": "j1", "stop": True}) == {
        "output": "done\n",
        "running": False,
        "exit_code": 0,
    }
    assert ran.kills == kills


async def test_a_command_that_cannot_start_is_the_models_error_and_takes_no_id() -> None:
    class Broken(Machine):
        def __init__(self) -> None:
            super().__init__()
            self.broken = True

        async def spawn(self, command: str) -> Command:
            if self.broken:
                self.broken = False
                raise RuntimeError("docker: the daemon is not running")
            return await super().spawn(command)

    ws = Workspace(Broken(), scope="acme")
    with pytest.raises(RuntimeError, match="the daemon is not running"):
        await background(ws, "serve")
    assert await tool(ws, "Jobs").call({}) == {"jobs": []}
    assert await background(ws, "serve") == {"job": "j1"}
    await ws.close()


async def test_close_stops_every_job_and_ids_keep_counting() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    await background(ws, "watch")
    async with ws:
        pass
    assert all(c.kills >= 1 for c in machine.commands)
    assert await tool(ws, "Jobs").call({}) == {"jobs": []}
    assert await background(ws, "again") == {"job": "j3"}
    await ws.close()


async def test_a_background_command_cut_by_a_stop_still_registers_its_job() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    task = asyncio.create_task(background(ws, "serve"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.01)
    [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert (listed["job"], listed["running"]) == ("j1", True)
    await ws.close()


def test_shell_tells_the_model_about_jobs() -> None:
    description = tool(Workspace(Machine(), scope="acme"), "Shell").description
    assert "background=true" in description and "keeps running after your reply" in description
    assert "Job reads" in description and "Jobs lists" in description


async def test_a_character_split_across_job_reads_is_kept_whole() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    ran = machine.commands[0]
    ran.stdout.say(b"a\xc3")
    await asyncio.sleep(0.01)
    job = tool(ws, "Job")
    assert await job.call({"id": "j1"}) == {"output": "a", "running": True, "exit_code": None}
    ran.stdout.say(b"\xa9b")
    ran.exit(0)
    await asyncio.sleep(0.05)
    assert await job.call({"id": "j1"}) == {"output": "éb", "running": False, "exit_code": 0}


async def test_a_lone_lead_byte_is_flushed_when_the_job_ends() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    ran = machine.commands[0]
    ran.stdout.say(b"\xc3")
    await asyncio.sleep(0.01)
    job = tool(ws, "Job")
    assert await job.call({"id": "j1"}) == {"output": "", "running": True, "exit_code": None}
    ran.exit(0)
    await asyncio.sleep(0.05)
    assert await job.call({"id": "j1"}) == {"output": "\ufffd", "running": False, "exit_code": 0}
