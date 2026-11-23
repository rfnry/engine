import asyncio
import json
import secrets
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy
import pytest
from rfnry_engine import CallInfo, InputError, Workspace

import moon
import sandbox as sandbox_module
from sandbox import SandboxError, Sandboxes, Slotted

EXAMPLE = Path(__file__).resolve().parent.parent
AGENTS = EXAMPLE / "agents"
CUT = EXAMPLE / "tests" / "data" / "lp_cut"
SKILL = "$RFNRY_TREE/surveyor/skills/dossier"


@pytest.fixture
def base():
    if shutil.which("bwrap") is None:
        pytest.skip("bwrap is not installed")
    made = EXAMPLE / ".work" / f"test-{secrets.token_hex(4)}"
    made.mkdir(parents=True)
    yield made
    shutil.rmtree(made, ignore_errors=True)


@pytest.fixture
def sandboxes(base):
    return Sandboxes(base, agents=AGENTS, data=CUT)


@pytest.fixture
def run_as(monkeypatch):
    def become(run_id):
        call = CallInfo(run_id=run_id, agent="surveyor", scope="moon", tool="Shell", call_id="c")
        monkeypatch.setattr(sandbox_module, "current_call", lambda: call)

    return become


async def drain(stream):
    found = bytearray()
    while chunk := await stream.read(65_536):
        found += chunk
    return found.decode()


async def run(sandboxes, command, env=None):
    process = await sandboxes.spawn(command, env or {})
    out, err = await asyncio.gather(drain(process.stdout), drain(process.stderr))
    code = await process.wait()
    await process.kill()
    return code, out, err


async def until(check, seconds=20):
    deadline = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < deadline, "the commands did not start in time"
        await asyncio.sleep(0.05)


async def sleepers(sandboxes, place, count=16):
    started = [
        await sandboxes.spawn('touch "started-$N"; exec sleep 1000', {"N": str(n)})
        for n in range(count)
    ]
    await until(lambda: len(list(place.glob("started-*"))) == count)
    return started


async def test_a_command_runs_in_its_own_place(sandboxes, base, run_as):
    run_as("a")
    assert sandboxes.place("a") == base.resolve() / "a"
    assert sandboxes.shell(sandboxes.place("a"))[0] == shutil.which("bwrap")
    assert await run(sandboxes, "pwd; echo $HOME") == (0, f"{base.resolve() / 'a'}\n" * 2, "")
    run_as("b")
    assert await run(sandboxes, "pwd") == (0, f"{base.resolve() / 'b'}\n", "")


async def test_a_command_outside_a_call_is_refused(sandboxes, monkeypatch):
    monkeypatch.setattr(sandbox_module, "current_call", lambda: None)
    with pytest.raises(SandboxError, match="^sandbox: no run is calling"):
        await sandboxes.spawn("pwd", {})
    with pytest.raises(SandboxError, match="^sandbox: "):
        await sandboxes.read("x", 10)
    with pytest.raises(SandboxError, match="^sandbox: "):
        await sandboxes.write("x", b"")


async def test_a_run_id_that_is_no_directory_name_is_refused(sandboxes):
    for run_id in ("", ".", "..", "a/b"):
        with pytest.raises(SandboxError, match="^sandbox: run id .* cannot name a directory"):
            sandboxes.place(run_id)


CONNECT = """
import socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=5).close()
    print("connected")
except OSError as error:
    print("refused", error.errno)
"""


async def test_a_sandbox_has_no_network(sandboxes, run_as):
    run_as("a")
    await sandboxes.write("connect.py", CONNECT.encode())
    code, out, err = await run(sandboxes, "python connect.py")
    assert (code, err) == (0, "")
    assert out.startswith("refused ")


async def test_a_sandbox_writes_only_its_place(sandboxes, run_as):
    run_as("b")
    assert await run(sandboxes, "echo b > mine") == (0, "", "")
    other = sandboxes.place("b")
    run_as("a")
    for target in (CUT / "x", AGENTS / "x", other / "x", other / "mine", EXAMPLE / "x"):
        code, _, err = await run(sandboxes, f'echo a > "{target}"')
        assert code != 0, target
        assert err, target
    with pytest.raises(InputError, match="outside the workspace"):
        await sandboxes.write("../b/x", b"a")
    assert (other / "mine").read_text() == "b\n"
    for target in (CUT / "x", AGENTS / "x", other / "x", EXAMPLE / "x"):
        assert not target.exists(), target
    assert await run(sandboxes, "echo a > mine") == (0, "", "")
    assert (sandboxes.place("a") / "mine").read_text() == "a\n"
    assert await sandboxes.read("mine", 10) == b"a\n"


COUNT = """
import os
import numpy
rows = numpy.loadtxt(os.path.join(os.environ["MOON_SURVEY_DATA"], "hydrogenlow.txt"), delimiter=",")
print(rows.shape[0], repr(float(rows[:, 4].mean())))
"""


async def test_a_sandbox_reads_the_data_and_imports_numpy(sandboxes, run_as):
    run_as("a")
    await sandboxes.write("count.py", COUNT.encode())
    code, out, err = await run(sandboxes, "python count.py")
    assert (code, err) == (0, "")
    rows = numpy.loadtxt(CUT / "hydrogenlow.txt", delimiter=",")
    assert out.split() == [str(rows.shape[0]), repr(float(rows[:, 4].mean()))]


async def test_at_most_sixteen_commands_run_at_once(base, run_as):
    sandboxes = Sandboxes(base, agents=AGENTS, data=CUT)
    run_as("a")
    command = 'date +%s%N > "start-$N"; sleep 2; date +%s%N > "end-$N"'
    began = time.monotonic()
    processes = [await sandboxes.spawn(command, {"N": str(n)}) for n in range(20)]
    spawned = time.monotonic() - began
    codes = await asyncio.gather(*(process.wait() for process in processes))
    for process in processes:
        await process.kill()
    place = sandboxes.place("a")
    steps = sorted(
        (int((place / f"{kind}-{n}").read_text()), step)
        for n in range(20)
        for kind, step in (("start", 1), ("end", -1))
    )
    running = peak = 0
    for _, step in steps:
        running += step
        peak = max(peak, running)
    assert spawned < 0.5
    assert codes == [0] * 20
    assert peak == 16


async def test_a_killed_command_frees_its_slot(sandboxes, run_as):
    run_as("a")
    place = sandboxes.place("a")
    started = await sleepers(sandboxes, place)
    try:
        late = await sandboxes.spawn("touch late", {})
        await asyncio.sleep(0.5)
        assert not (place / "late").exists()
        await started[0].kill()
        assert await started[0].wait() < 0
        assert await asyncio.wait_for(late.wait(), 10) == 0
        assert (place / "late").exists()
    finally:
        await asyncio.gather(*(process.kill() for process in started))


async def test_a_kill_before_start_cancels_the_wait(sandboxes, run_as):
    run_as("a")
    place = sandboxes.place("a")
    started = await sleepers(sandboxes, place)
    try:
        early = await sandboxes.spawn("touch never", {})
        await asyncio.sleep(0.5)
        assert not (place / "never").exists()
        await early.kill()
        assert await asyncio.wait_for(early.wait(), 1) == -signal.SIGKILL
        assert await early.stdout.read(100) == b""
        assert await early.stderr.read(100) == b""
        await started[0].kill()
        after = await sandboxes.spawn("touch after", {})
        assert await asyncio.wait_for(after.wait(), 10) == 0
        assert (place / "after").exists()
        assert not (place / "never").exists()
    finally:
        await asyncio.gather(*(process.kill() for process in started))


def shell_of(workspace):
    return next(tool for tool in workspace.tools() if tool.name == "Shell")


async def test_a_command_that_never_ends_is_stopped_and_frees_its_slot(sandboxes, run_as):
    run_as("a")
    place = sandboxes.place("a")
    short = shell_of(Workspace(sandboxes, scope="moon", timeout=1))
    long = shell_of(Workspace(sandboxes, scope="moon", timeout=30))
    began = time.time_ns()
    stuck = [
        asyncio.create_task(
            short.body({"command": f"touch started-{n}; while true; do sleep 1; done"})
        )
        for n in range(16)
    ]
    await until(lambda: len(list(place.glob("started-*"))) == 16)
    assert await long.body({"command": "date +%s%N > begun; echo done"}) == {
        "exit_code": 0,
        "stdout": "done\n",
        "stderr": "",
        "timed_out": False,
    }
    assert int((place / "begun").read_text()) >= began + 1_000_000_000
    assert [result["timed_out"] for result in await asyncio.gather(*stuck)] == [True] * 16


async def test_the_skill_runs_its_scripts_in_a_sandbox(sandboxes, run_as, monkeypatch, dossier):
    run_as("a")
    env = {"RFNRY_TREE": str(AGENTS)}
    code, out, err = await run(sandboxes, f'python "{SKILL}/measure.py" N75-90/0', env)
    assert (code, err) == (0, "")
    monkeypatch.setenv("MOON_SURVEY_DATA", str(CUT))
    expected = moon.measured("N75-90/0")
    printed = json.loads(out)
    assert printed["region"] == "N75-90/0"
    assert printed["values"] == {
        name: moon.value_of(name, value) for name, value in expected.items()
    }
    assert {name: value["unit"] for name, value in printed["values"].items()} == {
        "FeO": "wt%",
        "TiO2": "wt%",
        "Al2O3": "wt%",
        "Th": "ppm",
        "H": "ppm",
    }
    assert printed["source"] == moon.sources()
    assert printed["source"]["H"].endswith("/hydrogenlow.txt")
    assert printed["source"]["FeO"].endswith("/lpgrs_high1_elem_abundance_2deg.tab")
    await sandboxes.write("dossier.json", json.dumps(dossier).encode())
    assert await run(sandboxes, f'python "{SKILL}/check.py" dossier.json', env) == (0, "", "")
    dossier["region"] = "N00-15/40"
    await sandboxes.write("dossier.json", json.dumps(dossier).encode())
    assert await run(sandboxes, f'python "{SKILL}/check.py" dossier.json', env) == (
        1,
        'region: "N00-15/40" is no region of the grid\n',
        "",
    )


async def test_measure_names_a_region_that_does_not_exist(sandboxes, run_as):
    run_as("a")
    env = {"RFNRY_TREE": str(AGENTS)}
    assert await run(sandboxes, f'python "{SKILL}/measure.py" N00-15/40', env) == (
        1,
        "",
        "moon: region 'N00-15/40' does not exist\n",
    )


async def test_a_sandbox_sees_none_of_the_rest_of_the_machine(sandboxes, run_as):
    run_as("a")
    hidden = [EXAMPLE / "data" / "truth.json", EXAMPLE / "pyproject.toml", Path.home() / ".bashrc"]
    hidden = [path for path in hidden if path.exists()]
    assert len(hidden) >= 2
    tests = " ".join(
        f"test -e {shlex.quote(str(path))} && echo {shlex.quote(str(path))};" for path in hidden
    )
    assert await run(sandboxes, f"{tests} true") == (0, "", "")
    home = Path.home()
    words = sandboxes.shell(sandboxes.place("a"))
    reachable = {Path(w).relative_to(home).parts[0] for w in words if w.startswith(f"{home}/")}
    code, out, err = await run(sandboxes, f"ls -A {shlex.quote(str(home))}")
    assert (code, err) == (0, "")
    assert set(out.split()) == reachable


def lingering(marker):
    inner = f': {marker}; trap "" TERM; while true; do date +%s%N > beat; sleep 0.1; done'
    return f"setsid bash -c {shlex.quote(inner)} &"


async def gone(place, marker):
    await asyncio.sleep(0.3)
    first = (place / "beat").read_text()
    await asyncio.sleep(0.4)
    found = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    return (place / "beat").read_text() == first and found.stdout == ""


async def test_a_child_that_leaves_its_session_ends_when_its_command_exits(sandboxes, run_as):
    run_as("a")
    marker = f"linger-{secrets.token_hex(4)}"
    assert (await run(sandboxes, f"{lingering(marker)} sleep 0.5"))[0] == 0
    assert await gone(sandboxes.place("a"), marker)


async def test_a_child_that_leaves_its_session_ends_when_its_command_is_killed(sandboxes, run_as):
    run_as("a")
    place = sandboxes.place("a")
    marker = f"linger-{secrets.token_hex(4)}"
    process = await sandboxes.spawn(f"{lingering(marker)} exec sleep 1000", {})
    await until(lambda: (place / "beat").exists())
    await process.kill()
    assert await process.wait() < 0
    assert await gone(place, marker)


async def test_a_queued_command_is_not_killed_by_waiting(base, run_as):
    sandboxes = Sandboxes(base, agents=AGENTS, data=CUT, run_timeout=4)
    run_as("a")
    began = time.monotonic()
    busy = [await sandboxes.spawn("sleep 1", {}) for _ in range(80)]
    late = await sandboxes.spawn("echo done", {})
    out, err = await asyncio.gather(drain(late.stdout), drain(late.stderr))
    assert (await late.wait(), out, err) == (0, "done\n", "")
    assert time.monotonic() - began > 4
    assert await asyncio.gather(*(process.wait() for process in busy)) == [0] * 80


async def test_a_command_past_the_run_limit_is_stopped_and_frees_its_slot(base, run_as):
    sandboxes = Sandboxes(base, agents=AGENTS, data=CUT, run_timeout=2)
    run_as("a")
    place = sandboxes.place("a")
    shell = shell_of(Workspace(sandboxes, scope="moon", timeout=30))
    began = time.time_ns()
    stuck = [
        asyncio.create_task(
            shell.body(
                {"command": f"touch started-{n}; echo working; while true; do sleep 1; done"}
            )
        )
        for n in range(16)
    ]
    await until(lambda: len(list(place.glob("started-*"))) == 16)
    assert await shell.body({"command": "date +%s%N > begun; echo done"}) == {
        "exit_code": 0,
        "stdout": "done\n",
        "stderr": "",
        "timed_out": False,
    }
    assert int((place / "begun").read_text()) >= began + 2_000_000_000
    for result in await asyncio.gather(*stuck):
        assert result["timed_out"] is False
        assert result["exit_code"] < 0
        assert result["stdout"] == "working\n"
        assert result["stderr"] == "sandbox: stopped: a command may run for at most 2 seconds\n"


async def test_a_background_job_is_stopped_at_the_run_limit(base, run_as):
    sandboxes = Sandboxes(base, agents=AGENTS, data=CUT, run_timeout=1.5)
    run_as("a")
    async with Workspace(sandboxes, scope="moon", timeout=30) as workspace:
        tools = {tool.name: tool for tool in workspace.tools()}
        started = await tools["Shell"].body(
            {"command": "echo working; sleep 1000", "background": True}
        )
        assert started == {"job": "j1"}
        result = await tools["Job"].body({"id": "j1", "wait": 10})
    assert result["running"] is False
    assert result["exit_code"] < 0
    assert result["output"] == (
        "working\nsandbox: stopped: a command may run for at most 1.5 seconds\n"
    )


class Starting:
    def __init__(self):
        self.ended = asyncio.Event()

    async def kill(self):
        try:
            await asyncio.wait_for(self.ended.wait(), 10)
        finally:
            self.ended.set()

    async def wait(self):
        await self.ended.wait()
        return -signal.SIGKILL


class Spawning:
    def __init__(self, process):
        self.process = process
        self.go = asyncio.Event()

    async def spawn(self, command, env):
        await self.go.wait()
        return self.process


async def test_a_cancelled_kill_still_ends_a_command_that_was_starting():
    process = Starting()
    local = Spawning(process)
    slotted = Slotted(local, "x", {}, asyncio.Semaphore(1), 120)
    await asyncio.sleep(0)
    assert slotted.holding
    killing = asyncio.create_task(slotted.kill())
    await asyncio.sleep(0)
    killing.cancel()
    await asyncio.sleep(0)
    local.go.set()
    with pytest.raises(asyncio.CancelledError):
        await killing
    assert process.ended.is_set()
    assert await slotted.wait() == -signal.SIGKILL


class Ended:
    def __init__(self):
        self.kills = 0

    async def kill(self):
        self.kills += 1

    async def wait(self):
        return 0


class Started:
    def __init__(self, process):
        self.process = process

    async def spawn(self, command, env):
        return self.process


async def test_a_kill_after_the_command_ended_signals_nothing():
    process = Ended()
    slotted = Slotted(Started(process), "x", {}, asyncio.Semaphore(1), 120)
    assert await slotted.wait() == 0
    await slotted.kill()
    assert process.kills == 0


async def test_a_file_is_read_in_a_named_run_place_and_jailed(sandboxes, run_as):
    run_as("a")
    await sandboxes.write("dossier.json", b"{}")
    run_as("b")
    assert await sandboxes.read_at("a", "dossier.json", 10) == b"{}"
    with pytest.raises(InputError):
        await sandboxes.read_at("a", "../b/x", 10)
    with pytest.raises(InputError):
        await sandboxes.read_at("a", "dossier.json", 1)
    with pytest.raises(FileNotFoundError):
        await sandboxes.read_at("a", "missing.json", 10)
    with pytest.raises(SandboxError, match="^sandbox: run id"):
        await sandboxes.read_at("..", "x", 10)
