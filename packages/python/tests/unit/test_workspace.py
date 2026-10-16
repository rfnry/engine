import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.workspace import TEXT_BYTES, Workspace, _Kept


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


async def until(check: Callable[[], bool], what: str, deadline: float = 5.0) -> None:
    try:
        async with asyncio.timeout(deadline):
            while not check():
                await asyncio.sleep(0.005)
    except TimeoutError:
        pytest.fail(f"{what} never happened")


async def delivered(ws: Workspace, total: int, job: str = "j1") -> None:
    await until(lambda: ws._jobs[job].kept.total >= total, f"{total} bytes of output arriving")


async def finished(ws: Workspace, job: str = "j1") -> None:
    async with asyncio.timeout(5):
        await ws._jobs[job].ended.wait()


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


class Starting(Machine):
    def __init__(self, *, fails: bool = False) -> None:
        super().__init__()
        self.fails = fails
        self.release = asyncio.Event()

    async def spawn(self, command: str) -> Command:
        ran = await super().spawn(command)
        await self.release.wait()
        if self.fails:
            raise RuntimeError("docker: the daemon went away")
        return ran


async def test_a_command_cancelled_while_it_starts_is_stopped_once_it_has_started() -> None:
    machine = Starting()
    task = asyncio.create_task(shell(Workspace(machine, scope="acme"), "x"))
    ran = await started(machine)
    task.cancel()
    machine.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ran.kills == 1


async def test_a_second_cancel_while_it_starts_still_waits_to_stop_the_command() -> None:
    machine = Starting()
    task = asyncio.create_task(shell(Workspace(machine, scope="acme"), "x"))
    ran = await started(machine)
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done() and ran.kills == 0
    finally:
        machine.release.set()
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done and task.cancelled()
    assert ran.kills == 1


async def test_a_start_that_fails_after_a_cancel_is_still_a_cancel() -> None:
    machine = Starting(fails=True)
    task = asyncio.create_task(shell(Workspace(machine, scope="acme"), "x"))
    ran = await started(machine)
    task.cancel()
    machine.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ran.kills == 0


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
    assert (
        await edit.call({"path": "f.txt", "old": "x = 1", "new": "x = 3"})
        == "edited f.txt (1 occurrence)"
    )
    assert machine.files["f.txt"] == b"x = 3\r\ny = 2\n"
    assert machine.limits == [TEXT_BYTES]


async def test_edit_with_all_replaces_every_occurrence() -> None:
    machine = Machine()
    machine.files["f.txt"] = b"x = x + x\n"
    edit = tool(Workspace(machine, scope="acme"), "Edit")
    result = await edit.call({"path": "f.txt", "old": "x", "new": "y", "all": True})
    assert result == "edited f.txt (3 occurrences)"
    assert machine.files["f.txt"] == b"y = y + y\n"


async def test_edit_with_all_still_needs_one_occurrence() -> None:
    machine = Machine()
    machine.files["f.txt"] = b"a\n"
    edit = tool(Workspace(machine, scope="acme"), "Edit")
    with pytest.raises(
        InputError, match="^workspace: old appears 0 times in f.txt; it must appear at least once$"
    ):
        await edit.call({"path": "f.txt", "old": "x", "new": "y", "all": True})


async def test_a_replacement_that_holds_the_old_text_is_made_once_per_occurrence() -> None:
    machine = Machine()
    machine.files["f.txt"] = b"ab ab"
    edit = tool(Workspace(machine, scope="acme"), "Edit")
    result = await edit.call({"path": "f.txt", "old": "ab", "new": "abab", "all": True})
    assert result == "edited f.txt (2 occurrences)"
    assert machine.files["f.txt"] == b"abab abab"


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
    total = 0
    for pipe, data in (
        (ran.stdout, b"listening\n"),
        (ran.stderr, b"warning\n"),
        (ran.stdout, b"GET /\n"),
    ):
        pipe.say(data)
        total += len(data)
        await delivered(ws, total)
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
    await delivered(ws, 50_000)
    output = (await tool(ws, "Job").call({"id": "j1"}))["output"]
    assert output == "a" * 500 + "\n[... 49000 bytes omitted ...]\n" + "a" * 500
    await ws.close()


async def test_stop_ends_a_job_returns_its_last_output_and_then_it_is_gone() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    ran = machine.commands[0]
    ran.stdout.say(b"bye\n")
    await delivered(ws, 4)
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
    await finished(ws)
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
    await until(lambda: ran.code.done(), "the command exiting")
    ran.stdout.say(b"late\n")
    ran.stdout.close()
    ran.stderr.close()
    await finished(ws)
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
    await finished(ws)
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
    await until(lambda: "j1" in ws._jobs, "the job registering")
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
    await delivered(ws, 2)
    job = tool(ws, "Job")
    assert await job.call({"id": "j1"}) == {"output": "a", "running": True, "exit_code": None}
    ran.stdout.say(b"\xa9b")
    ran.exit(0)
    await finished(ws)
    assert await job.call({"id": "j1"}) == {"output": "éb", "running": False, "exit_code": 0}


async def test_a_lone_lead_byte_is_flushed_when_the_job_ends() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    ran = machine.commands[0]
    ran.stdout.say(b"\xc3")
    await delivered(ws, 1)
    job = tool(ws, "Job")
    assert await job.call({"id": "j1"}) == {"output": "", "running": True, "exit_code": None}
    ran.exit(0)
    await finished(ws)
    assert await job.call({"id": "j1"}) == {"output": "\ufffd", "running": False, "exit_code": 0}


def test_a_character_split_between_head_and_tail_is_held_whole() -> None:
    kept = _Kept(10)
    kept.add(b"abcd\xf0\x9f")
    assert kept.take(final=False) == "abcd"
    kept.add(b"\x98\x80")
    assert kept.take() == "\U0001f600"


def test_held_bytes_do_not_disturb_the_omitted_count() -> None:
    kept = _Kept(10)
    kept.add(b"a" * 100 + b"\xf0\x9f")
    first = kept.take(final=False)
    assert first == "a" * 5 + "\n[... 92 bytes omitted ...]\n" + "a" * 3
    kept.add(b"\x98\x80z")
    assert kept.take() == "\U0001f600z"


class Dying(Machine):
    def __init__(self, *, kill_fails: int = 0, wait_fails: bool = False) -> None:
        super().__init__()
        self.kill_fails = kill_fails
        self.wait_fails = wait_fails

    async def spawn(self, command: str) -> Command:
        ran = await super().spawn(command)
        machine = self

        async def kill() -> None:
            ran.kills += 1
            if machine.kill_fails > 0:
                machine.kill_fails -= 1
                raise RuntimeError("docker: kill failed")
            ran.exit(-15)

        async def wait() -> int:
            if machine.wait_fails:
                raise RuntimeError("docker: wait failed")
            return await asyncio.shield(ran.code)

        ran.kill = kill  # type: ignore[method-assign]
        ran.wait = wait  # type: ignore[method-assign]
        return ran


async def test_a_failed_kill_leaves_the_job_reachable_by_a_second_close() -> None:
    machine = Dying(kill_fails=1)
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    following = ws._jobs["j1"].task
    with pytest.raises(RuntimeError, match="kill failed"):
        await ws.close()
    [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert (listed["job"], listed["running"]) == ("j1", True)
    await ws.close()
    assert await tool(ws, "Jobs").call({}) == {"jobs": []}
    assert machine.commands[0].kills >= 2
    await asyncio.sleep(0)
    assert following is not None and following.done()


async def test_a_wait_that_raises_is_in_the_output_and_the_process_is_killed() -> None:
    machine = Dying(wait_fails=True)
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    await finished(ws)
    [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert listed["running"] is False
    assert machine.commands[0].kills == 1
    result = await tool(ws, "Job").call({"id": "j1"})
    assert "RuntimeError: docker: wait failed" in result["output"]
    assert result["running"] is False


async def test_a_kill_that_raises_after_a_wait_that_raised_is_in_the_output_too() -> None:
    machine = Dying(wait_fails=True, kill_fails=1)
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    await finished(ws)
    result = await tool(ws, "Job").call({"id": "j1"})
    assert "RuntimeError: docker: wait failed" in result["output"]
    assert "RuntimeError: docker: kill failed" in result["output"]
    assert result["running"] is False


async def test_a_wait_and_a_kill_that_fail_the_same_way_are_in_the_output_once() -> None:
    class Gone(Machine):
        async def spawn(self, command: str) -> Command:
            ran = await super().spawn(command)

            async def wait() -> int:
                raise RuntimeError("docker: the daemon went away")

            async def kill() -> None:
                ran.kills += 1
                ran.stdout.close()
                ran.stderr.close()
                raise RuntimeError("docker: the daemon went away")

            ran.wait = wait  # type: ignore[method-assign]
            ran.kill = kill  # type: ignore[method-assign]
            return ran

    machine = Gone()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    await finished(ws)
    assert machine.commands[0].kills == 1
    assert await tool(ws, "Job").call({"id": "j1"}) == {
        "output": "\nRuntimeError: docker: the daemon went away\n",
        "running": False,
        "exit_code": None,
    }


@pytest.mark.parametrize("kill_fails", [0, 1])
async def test_a_job_whose_reader_is_cancelled_is_killed_and_ends(kill_fails: int) -> None:
    machine = Dying(kill_fails=kill_fails)
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    task = ws._jobs["j1"].task
    assert task is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert machine.commands[0].kills == 1
    [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert (listed["running"], listed["exit_code"]) == (False, None)
    output = (await tool(ws, "Job").call({"id": "j1"}))["output"]
    assert ("RuntimeError: docker: kill failed" in output) is bool(kill_fails)


async def test_a_failed_exit_kill_is_in_the_output_and_not_tried_again() -> None:
    machine = Dying(kill_fails=2)
    ws = Workspace(machine, scope="acme")
    await background(ws, "build")
    machine.commands[0].exit(0)
    await finished(ws)
    assert machine.commands[0].kills == 1
    assert await tool(ws, "Job").call({"id": "j1"}) == {
        "output": "\nRuntimeError: docker: kill failed\n",
        "running": False,
        "exit_code": 0,
    }


@pytest.mark.parametrize("awaited", [False, True])
async def test_close_returns_after_a_job_whose_reader_was_cancelled(awaited: bool) -> None:
    ws = Workspace(Machine(), scope="acme")
    await background(ws, "serve")
    task = ws._jobs["j1"].task
    assert task is not None
    task.cancel()
    if awaited:
        with pytest.raises(asyncio.CancelledError):
            await task
    await ws.close()
    assert await tool(ws, "Jobs").call({}) == {"jobs": []}


async def test_a_close_that_is_cancelled_is_cancelled() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    machine.commands[0].stubborn = True
    closing = asyncio.create_task(ws.close())
    await until(lambda: machine.commands[0].kills >= 1, "close stopping the job")
    assert not closing.done()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing


async def test_job_waits_for_the_job_to_end_and_returns_as_soon_as_it_does() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "build")
    ran = machine.commands[0]

    def finish() -> None:
        ran.stdout.say(b"built\n")
        ran.exit(0)

    asyncio.get_running_loop().call_later(0.2, finish)
    began = time.monotonic()
    result = await tool(ws, "Job").call({"id": "j1", "wait": 5})
    assert time.monotonic() - began < 1
    assert result == {"output": "built\n", "running": False, "exit_code": 0}


async def test_job_returns_at_the_wait_with_the_job_still_running() -> None:
    ws = Workspace(Machine(), scope="acme")
    await background(ws, "serve")
    began = time.monotonic()
    result = await tool(ws, "Job").call({"id": "j1", "wait": 0.2})
    assert 0.2 <= time.monotonic() - began < 1
    assert result == {"output": "", "running": True, "exit_code": None}
    await ws.close()


async def test_wait_with_stop_gives_the_job_that_long_then_stops_it() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "build")
    began = time.monotonic()
    result = await tool(ws, "Job").call({"id": "j1", "wait": 0.2, "stop": True})
    assert time.monotonic() - began >= 0.2
    assert result == {"output": "", "running": False, "exit_code": -15}
    assert machine.commands[0].kills >= 1


@pytest.mark.parametrize("wait", [-1, 121])
async def test_a_wait_outside_the_timeout_is_refused(wait: int) -> None:
    ws = Workspace(Machine(), scope="acme", timeout=120)
    await background(ws, "serve")
    with pytest.raises(
        InputError, match=f"^workspace: wait must be between 0 and 120 seconds; got {wait}$"
    ):
        await tool(ws, "Job").call({"id": "j1", "wait": wait})
    await ws.close()


async def test_a_stop_ends_the_wait_and_the_job_runs_on() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    task = asyncio.create_task(tool(ws, "Job").call({"id": "j1", "wait": 10}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert listed["running"] is True and machine.commands[0].kills == 0
    await ws.close()


def test_job_tells_the_model_how_long_it_may_wait() -> None:
    description = tool(Workspace(Machine(), scope="acme", timeout=300), "Job").description
    assert "wait is seconds, up to 300, to wait for it to end first" in description


async def test_a_job_call_is_bounded_by_the_timeout_as_a_whole() -> None:
    machine = Machine()
    ws = Workspace(machine, scope="acme", timeout=0.3)
    await background(ws, "serve")
    ran = machine.commands[0]
    ran.stubborn = True
    began = time.monotonic()
    result = await tool(ws, "Job").call({"id": "j1", "wait": 0.3, "stop": True})
    assert time.monotonic() - began < 0.5
    assert result["running"] is True and ran.kills == 1
    ran.stubborn = False
    await ws.close()


class Slow(Machine):
    def __init__(self, *, fails: bool = False) -> None:
        super().__init__()
        self.fails = fails
        self.release = asyncio.Event()
        self.finished = 0
        self.cut = 0

    async def spawn(self, command: str) -> Command:
        ran = await super().spawn(command)
        machine = self

        async def kill() -> None:
            ran.kills += 1
            try:
                await machine.release.wait()
            except asyncio.CancelledError:
                machine.cut += 1
                raise
            finally:
                await machine.release.wait()
                machine.finished += 1
                if machine.fails:
                    raise RuntimeError("docker: kill failed")
                ran.exit(-15)

        ran.kill = kill  # type: ignore[method-assign]
        return ran


class Late(Slow):
    def __init__(self, *, fails: bool = False) -> None:
        super().__init__(fails=fails)
        self.begin = asyncio.Event()

    async def spawn(self, command: str) -> Command:
        ran = await super().spawn(command)
        await self.begin.wait()
        return ran


LOST = "workspace: could not stop the command: docker: kill failed"


def kills(machine: Machine) -> int:
    return sum(ran.kills for ran in machine.commands)


def pending(before: set[asyncio.Task[Any]]) -> list[asyncio.Task[Any]]:
    return [
        t for t in asyncio.all_tasks() - before if not t.done() and t is not asyncio.current_task()
    ]


def logged(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "rfnry_engine.tools.workspace" and record.levelname == "ERROR"
    ]


async def cancel_during_the_kill(
    task: asyncio.Task[Any], machine: Slow, count: int = 1, *, own: bool = False
) -> None:
    try:
        await until(lambda: kills(machine) == count, "the kill starting")
        if own:
            task.cancel()
            await asyncio.sleep(0.05)
            assert not task.done() and machine.cut == 0
        task.cancel()
        await until(lambda: machine.cut == count, "the cancel cutting the kill")
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done() and kills(machine) == machine.cut == count
        assert machine.finished == 0
    finally:
        machine.release.set()
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done and task.cancelled()
    assert machine.cut == count


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("path", ["cancel", "exit", "timeout"])
async def test_a_cancelled_shell_returns_once_its_one_kill_has_finished(
    path: str, fails: bool, caplog: pytest.LogCaptureFixture
) -> None:
    machine = Slow(fails=fails)
    if path == "exit":
        machine.act = prints()
    before = asyncio.all_tasks()
    ws = Workspace(machine, scope="acme", timeout=0.05 if path == "timeout" else 120)
    task = asyncio.create_task(shell(ws, "x"))
    await started(machine)
    if path == "cancel":
        await asyncio.sleep(0.01)
        task.cancel()
    await cancel_during_the_kill(task, machine, own=path != "cancel")
    await asyncio.sleep(0)
    assert kills(machine) == machine.finished == 1
    assert not pending(before)
    assert logged(caplog) == ([LOST] if fails else [])


@pytest.mark.parametrize("fails", [False, True])
async def test_a_command_cancelled_while_it_starts_returns_once_it_is_stopped(
    fails: bool, caplog: pytest.LogCaptureFixture
) -> None:
    machine = Late(fails=fails)
    before = asyncio.all_tasks()
    task = asyncio.create_task(shell(Workspace(machine, scope="acme"), "x"))
    try:
        await started(machine)
        task.cancel()
        await asyncio.sleep(0.01)
    finally:
        machine.begin.set()
    await cancel_during_the_kill(task, machine)
    await asyncio.sleep(0)
    assert kills(machine) == machine.finished == 1
    assert not pending(before)
    assert logged(caplog) == ([LOST] if fails else [])


async def test_a_second_cancel_while_it_starts_cuts_the_kill_of_what_it_started() -> None:
    machine = Late()
    before = asyncio.all_tasks()
    task = asyncio.create_task(shell(Workspace(machine, scope="acme"), "x"))
    try:
        await started(machine)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
        assert not task.done() and kills(machine) == 0
        machine.begin.set()
        await until(lambda: machine.cut == 1, "the cancel cutting the kill")
        await asyncio.sleep(0.05)
        assert not task.done() and kills(machine) == 1 and machine.finished == 0
    finally:
        machine.begin.set()
        machine.release.set()
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done and task.cancelled()
    assert kills(machine) == machine.cut == machine.finished == 1
    await asyncio.sleep(0)
    assert not pending(before)


@pytest.mark.parametrize("starting", [False, True])
async def test_two_cancels_at_the_same_moment_cut_the_kill_they_start(starting: bool) -> None:
    machine = Late() if starting else Slow()
    before = asyncio.all_tasks()
    task = asyncio.create_task(shell(Workspace(machine, scope="acme"), "x"))
    try:
        await started(machine)
        await asyncio.sleep(0.01)
        task.cancel()
        task.cancel()
        await asyncio.sleep(0.01)
        if isinstance(machine, Late):
            assert not task.done() and kills(machine) == 0
            machine.begin.set()
        await until(lambda: machine.cut == 1, "the second cancel cutting the kill")
        assert not task.done() and kills(machine) == 1 and machine.finished == 0
    finally:
        if isinstance(machine, Late):
            machine.begin.set()
        machine.release.set()
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done and task.cancelled()
    assert kills(machine) == machine.cut == machine.finished == 1
    await asyncio.sleep(0)
    assert not pending(before)


async def test_a_cancelled_job_stop_returns_once_its_kill_has_finished() -> None:
    machine = Slow()
    before = asyncio.all_tasks()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    task = asyncio.create_task(tool(ws, "Job").call({"id": "j1", "stop": True}))
    await cancel_during_the_kill(task, machine, own=True)
    await finished(ws)
    assert kills(machine) == machine.finished == 2
    assert not pending(before)


async def test_a_cancelled_close_returns_once_every_kill_has_finished() -> None:
    machine = Slow()
    before = asyncio.all_tasks()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    await background(ws, "watch")
    closing = asyncio.create_task(ws.close())
    await cancel_during_the_kill(closing, machine, count=2, own=True)
    assert ws._jobs == {}
    assert kills(machine) == machine.finished == 4
    await asyncio.sleep(0)
    assert not pending(before)


class Held(Machine):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.failing: set[str] = set()

    async def spawn(self, command: str) -> Command:
        ran = await super().spawn(command)
        kill = ran.kill
        machine = self

        async def held() -> None:
            await machine.release.wait()
            if command in machine.failing:
                ran.kills += 1
                raise RuntimeError("docker: kill failed")
            await kill()

        ran.kill = held  # type: ignore[method-assign]
        return ran


async def cancel_while_close_kills(ws: Workspace, machine: Held) -> asyncio.Task[None]:
    closing = asyncio.create_task(ws.close())
    try:
        await asyncio.sleep(0.01)
        closing.cancel()
        await asyncio.sleep(0.01)
        assert not closing.done()
    finally:
        machine.release.set()
    return closing


async def test_a_close_cancelled_while_it_kills_still_forgets_the_jobs_it_stopped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    machine = Held()
    machine.failing.add("build")
    before = asyncio.all_tasks()
    ws = Workspace(machine, scope="acme")
    await background(ws, "build")
    await background(ws, "serve")
    serving = ws._jobs["j2"].task
    closing = await cancel_while_close_kills(ws, machine)
    done, _ = await asyncio.wait({closing}, timeout=5)
    assert closing in done and closing.cancelled()
    assert serving is not None and serving.done()
    assert logged(caplog) == [LOST]
    [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert (listed["job"], listed["running"]) == ("j1", True)
    machine.failing.clear()
    async with asyncio.timeout(5):
        await ws.close()
    assert await tool(ws, "Jobs").call({}) == {"jobs": []}
    await asyncio.sleep(0)
    assert not pending(before)


async def test_a_further_cancel_of_a_cancelled_close_ends_the_jobs_it_waits_for() -> None:
    machine = Held()
    before = asyncio.all_tasks()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    ran = machine.commands[0]
    ran.stubborn = True
    reading = ws._jobs["j1"].task
    closing = await cancel_while_close_kills(ws, machine)
    await until(lambda: ran.kills == 1, "close stopping the job")
    await asyncio.sleep(0.05)
    assert not closing.done() and ws._jobs == {}
    closing.cancel()
    done, _ = await asyncio.wait({closing}, timeout=5)
    assert closing in done and closing.cancelled()
    assert reading is not None and reading.done() and ran.kills == 2
    await asyncio.sleep(0)
    assert not pending(before)


async def test_a_kill_that_fails_while_close_is_cancelled_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    machine = Slow(fails=True)
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    closing = asyncio.create_task(ws.close())
    await cancel_during_the_kill(closing, machine, own=True)
    assert logged(caplog) == [LOST]
    machine.fails = False
    async with asyncio.timeout(5):
        await ws.close()
    assert await tool(ws, "Jobs").call({}) == {"jobs": []}


@pytest.mark.parametrize("exits", [False, True])
async def test_a_cancelled_job_reader_ends_once_its_one_kill_has_finished(exits: bool) -> None:
    machine = Slow()
    before = asyncio.all_tasks()
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    ran = machine.commands[0]
    task = ws._jobs["j1"].task
    assert task is not None
    if exits:
        ran.exit(0)
    else:
        task.cancel()
    await cancel_during_the_kill(task, machine, own=exits)
    assert ran.kills == machine.finished == 1
    assert ws._jobs["j1"].ended.is_set()
    await asyncio.sleep(0)
    assert not pending(before)


async def test_a_kill_that_raises_after_a_cancel_is_logged_and_the_call_is_still_cancelled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    machine = Dying(kill_fails=1)
    task = asyncio.create_task(shell(Workspace(machine, scope="acme"), "x"))
    ran = await started(machine)
    await asyncio.sleep(0.01)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done and task.cancelled()
    assert ran.kills == 1
    assert logged(caplog) == [LOST]


@pytest.mark.parametrize("exits", [True, False])
async def test_a_kill_that_raises_after_the_command_exits_or_times_out_is_raised(
    exits: bool, caplog: pytest.LogCaptureFixture
) -> None:
    machine = Dying(kill_fails=1)
    if exits:
        machine.act = prints()
    ws = Workspace(machine, scope="acme", timeout=120 if exits else 0.05)
    with pytest.raises(RuntimeError, match="^docker: kill failed$"):
        await shell(ws, "x")
    assert machine.commands[0].kills == 1
    assert logged(caplog) == []


async def test_a_kill_that_raises_on_job_stop_is_raised(caplog: pytest.LogCaptureFixture) -> None:
    machine = Dying(kill_fails=1)
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    with pytest.raises(RuntimeError, match="^docker: kill failed$"):
        await tool(ws, "Job").call({"id": "j1", "stop": True})
    [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert listed["running"] is True
    assert logged(caplog) == []
    await ws.close()


async def test_a_shell_whose_wait_raises_raises_that_and_logs_its_kills_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    machine = Dying(wait_fails=True, kill_fails=1)
    with pytest.raises(RuntimeError, match="^docker: wait failed$"):
        await shell(Workspace(machine, scope="acme"), "x")
    assert machine.commands[0].kills == 1
    assert logged(caplog) == [LOST]


async def test_close_raises_the_first_failed_kill_and_logs_the_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    machine = Dying(kill_fails=2)
    ws = Workspace(machine, scope="acme")
    await background(ws, "serve")
    await background(ws, "watch")
    with pytest.raises(RuntimeError, match="^docker: kill failed$"):
        await ws.close()
    assert logged(caplog) == [LOST]
    assert [listed["job"] for listed in (await tool(ws, "Jobs").call({}))["jobs"]] == ["j1", "j2"]
    await ws.close()
    assert await tool(ws, "Jobs").call({}) == {"jobs": []}


async def test_a_close_cancelled_while_it_waits_for_its_jobs_logs_the_kill_that_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    machine = Dying(kill_fails=1)
    before = asyncio.all_tasks()
    ws = Workspace(machine, scope="acme")
    await background(ws, "build")
    await background(ws, "serve")
    serve = machine.commands[1]

    async def stubborn() -> None:
        serve.kills += 1

    serve.kill = stubborn  # type: ignore[method-assign]
    closing = asyncio.create_task(ws.close())
    await until(lambda: "j2" not in ws._jobs, "close forgetting the job it stopped")
    await asyncio.sleep(0.05)
    assert not closing.done()
    closing.cancel()
    done, _ = await asyncio.wait({closing}, timeout=5)
    assert closing in done and closing.cancelled()
    assert logged(caplog) == [LOST]
    [listed] = (await tool(ws, "Jobs").call({}))["jobs"]
    assert (listed["job"], listed["running"]) == ("j1", True)
    async with asyncio.timeout(5):
        await ws.close()
    await asyncio.sleep(0)
    assert not pending(before)
