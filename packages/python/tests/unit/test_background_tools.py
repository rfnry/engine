import asyncio
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Stop
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, RunawayError
from rfnry_engine.models import Message, Reply, ToolCall
from rfnry_engine.records import (
    Decision,
    Generation,
    RunEnded,
    Step,
    ToolCallStarted,
    ToolResultReady,
)
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Crew, Seen, background, crew, released, team, text_reply, until
from .test_background import (
    Gated,
    Watched,
    ended,
    ends,
    halfway,
    notice,
    taken,
)
from .test_engine_bridge import PY, result, run_coder, script
from .test_engine_workspace import AGENTS, make_tree
from .test_failed_runs import Kept
from .test_reach import child_of, use
from .trees import provided, write_folder

HELPER = (
    "whether it is running, its agent, its input, under asking the question it waits on you to "
    "answer through Ask (null when none), and once it has ended its reply (a decision agent's "
    "answers), "
    "its error, what it had said when it was stopped, or under paused what it waits for. wait is "
    "seconds, up to 30, to wait "
    "first; it returns as soon as the helper ends or any notice from a helper reaches you, and at "
    "once while the helper waits on your answer. "
    "stop=true stops the helper and every helper it started, after the wait, and waits for it to "
    "end, for what is left of the call's 30 seconds. send answers the question under asking; "
    "when there is none, it gives a running generation helper a message, which it reads before "
    "its next model call; a decision helper takes none. Its end reaches you as a notice whether "
    "or not you read it here."
)
JOB_ALONE = (
    f"Read a helper you started with Delegate in the background, by its id (d1, d2, …): {HELPER}"
)
JOBS_ALONE = (
    "List the helpers you started in the background: id, agent, input, when it started, and "
    "whether it is running or, once it has ended, how: finished, failed, stopped or paused."
)
JOB_BESIDE = (
    "Read a background job of the workspace (j1, j2, …) or a helper you started with Delegate in "
    "the background (d1, d2, …). A job: what it printed since the last read, stdout and stderr in "
    "the order they arrived, whether it is running, and its exit_code once it has ended; wait is "
    "seconds, up to 120, to wait for it to end first; stop=true stops it, after the wait; its "
    f"output keeps its first and last 15000 bytes; send is for a helper only. A helper: {HELPER}"
)
JOBS_BESIDE = (
    "List the workspace's background jobs and the helpers you started in the background. A job: "
    "id, command, when it started, whether it is running and its exit_code; a job that ended is "
    "listed until Job reads its last output. A helper: id, agent, input, when it started, "
    "and whether it is running or, once it has ended, how: finished, failed, stopped or paused."
)


def folder(root: Path, lead: Mapping[str, Any] | None = None, captain: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    agents: dict[str, Any] = {
        "lead": {
            "kind": "generation",
            "model": "crew",
            "description": "Leads the crew",
            "instructions": "lead.md",
            "orchestration": {
                "delegate_to": ["worker", "desk", *(["captain"] if captain else [])],
                "background": True,
            },
            **(lead or {}),
        },
        "worker": {
            "kind": "generation",
            "model": "crew",
            "description": "Works on one part",
            "instructions": "worker.md",
        },
        "desk": {
            "kind": "decision",
            "model": "judge",
            "description": "Sorts",
            "instructions": "desk.md",
            "questions": ["desk/*.json"],
        },
    }
    if captain:
        agents["captain"] = {
            "kind": "generation",
            "model": "crew",
            "description": "Runs a crew of its own",
            "instructions": "captain.md",
            "orchestration": {"delegate_to": ["worker"], "background": True},
        }
    return write_folder(
        root,
        agents,
        files={
            "lead.md": "Lead.\n",
            "worker.md": "Work.\n",
            "captain.md": "Captain.\n",
            "desk.md": "",
            "desk/urgent.json": '{"type": "check", "instructions": "Urgent?"}',
        },
    )


def started(*calls: tuple[str, str, str]) -> Reply:
    return Reply(
        tool_calls=[
            ToolCall(
                id=call_id,
                name="Delegate",
                args={"agent": agent, "input": text, "background": True},
            )
            for call_id, agent, text in calls
        ],
        stop_reason="tool_use",
    )


def calling(*calls: ToolCall) -> Reply:
    return Reply(tool_calls=list(calls), stop_reason="tool_use")


def read(step: Step) -> Any:
    assert isinstance(step.result, str) and not step.is_error, step
    return json.loads(step.result)


def stamped(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for entry in entries:
        assert datetime.fromisoformat(entry.pop("started")).microsecond == 0
    return entries


def jobbing(seen: Seen) -> bool:
    return any(isinstance(e, ToolCallStarted) and e.tool == "Job" for e in seen.events)


async def test_without_a_workspace_job_and_jobs_serve_helpers_only(tmp_path: Path) -> None:
    plain = Crew([text_reply("done")])
    run = await team(crew(tmp_path / "plain"), plain).run("lead", "acme", "go")
    assert isinstance(run, Generation) and [tool.name for tool in run.tools] == ["Delegate"]
    model = Crew(
        [
            calling(
                ToolCall(id="r1", name="Job", args={"id": "j1"}), ToolCall(id="l1", name="Jobs")
            ),
            text_reply("done"),
        ]
    )
    run = await team(crew(tmp_path / "opted", background=True), model).run("lead", "acme", "go")
    assert isinstance(run, Generation) and run.reply == "done"
    assert [tool.name for tool in run.tools] == ["Delegate", "Job", "Jobs"]
    _, job, jobs = run.tools
    assert (job.description, jobs.description) == (JOB_ALONE, JOBS_ALONE)
    assert list(job.input_schema["properties"]) == ["id", "wait", "stop", "send"]
    assert job.input_schema["required"] == ["id"]
    assert jobs.input_schema == {"type": "object", "properties": {}}
    assert [(step.result, step.is_error) for step in run.steps] == [
        ("InputError: tools/helpers: there is no helper j1", True),
        ('{"jobs": []}', False),
    ]


async def test_jobs_lists_helpers_beside_workspace_jobs(tmp_path: Path) -> None:
    gate = asyncio.Event()
    start = calling(
        ToolCall(id="s1", name="Shell", args={"command": "sleep 60", "background": True}),
        ToolCall(
            id="b1", name="Delegate", args={"agent": "worker", "input": "one", "background": True}
        ),
    )
    model = Crew(
        [start, use("Jobs", "l1"), text_reply("waiting"), text_reply("done")],
        {"one": [gate, text_reply("first")]},
    )
    room = tmp_path / "room"
    room.mkdir()
    engine = team(folder(tmp_path / "agents", lead={"workspace": True}), model)
    async with Workspace(Local(room), scope="acme") as work:
        running = asyncio.create_task(engine.run("lead", "acme", "go", workspace=work))
        await until(lambda: len(model.calls["lead"]) == 3, "the lead's third call")
        gate.set()
        run = await asyncio.wait_for(running, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    assert [tool.name for tool in run.tools] == [
        "Shell",
        "Write",
        "Edit",
        "Delegate",
        "Job",
        "Jobs",
    ]
    job, jobs = run.tools[4:]
    assert (job.description, jobs.description) == (JOB_BESIDE, JOBS_BESIDE)
    assert stamped(read(run.steps[2])["jobs"]) == [
        {"job": "j1", "command": "sleep 60", "running": True, "exit_code": None},
        {"job": "d1", "agent": "worker", "input": "one", "running": True},
    ]


async def test_job_reads_a_helper_running_then_ended(tmp_path: Path) -> None:
    gate, lead_gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [
            background("one"),
            use("Job", "r1", id="d1"),
            lead_gate,
            use("Job", "r2", id="d1"),
            use("Jobs", "l1"),
            text_reply("done"),
        ],
        {"one": [gate, text_reply("the answer")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 3, "the lead's third call")
    gate.set()
    await until(ends(seen, 1), "d1's end")
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    helper_id = str(run.steps[0].child_run_id)
    assert read(run.steps[1]) == {
        "running": True,
        "agent": "worker",
        "input": "one",
        "asking": None,
    }
    assert read(run.steps[2]) == {
        "running": False,
        "agent": "worker",
        "input": "one",
        "asking": None,
        "reply": "the answer",
    }
    assert stamped(read(run.steps[3])["jobs"]) == [
        {"job": "d1", "agent": "worker", "input": "one", "running": False, "ended": "finished"}
    ]
    told = notice("d1", "worker", helper_id, "the answer")
    assert model.calls["lead"][3][-1].content == told.content
    assert [m for m in run.conversation if m.notice is not None] == [told]


async def test_job_returns_the_helper_s_input(tmp_path: Path) -> None:
    gate, lead_gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [
            background("one"),
            use("Job", "r1", id="d1"),
            lead_gate,
            use("Job", "r2", id="d1"),
            text_reply("done"),
        ],
        {"one": [gate, text_reply("the answer")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 3, "the lead's third call")
    gate.set()
    await until(ends(seen, 1), "d1's end")
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation)
    assert read(run.steps[1])["input"] == "one"
    assert read(run.steps[2])["input"] == "one"


async def test_jobs_lists_an_input_whole(tmp_path: Path) -> None:
    whole = "".join(chr(97 + i % 26) for i in range(20_000))
    model = Crew(
        [background(whole), use("Jobs", "l1"), use("Job", "r1", id="d1"), text_reply("done")],
        {whole: [text_reply("first")]},
    )
    run = await team(crew(tmp_path, background=True), model).run("lead", "acme", "go")
    assert isinstance(run, Generation)
    assert stamped(read(run.steps[1])["jobs"])[0]["input"] == whole
    assert read(run.steps[2])["input"] == whole


@pytest.mark.parametrize(
    ("agent", "worker", "replies", "outcome"),
    [
        pytest.param(
            "desk",
            {},
            [],
            {"answers": {"urgent": {"type": "check", "probability": 0.9, "confidence": 0.8}}},
            id="answers",
        ),
        pytest.param(
            "worker",
            {"max_model_calls": 1},
            [halfway()],
            {"error": "RunawayError: engine: run exceeded 1 model calls"},
            id="error",
        ),
    ],
)
async def test_job_reads_how_a_helper_ended(
    tmp_path: Path,
    agent: str,
    worker: dict[str, Any],
    replies: list[Reply | asyncio.Event],
    outcome: dict[str, Any],
) -> None:
    model = Crew(
        [
            background("part one", agent=agent),
            use("Job", "r1", id="d1", wait=30),
            text_reply("done"),
        ],
        {"part one": replies},
    )
    engine = team(crew(tmp_path, worker=worker, background=True), model)
    run = await asyncio.wait_for(engine.run("lead", "acme", "go"), 5)
    assert isinstance(run, Generation) and run.reply == "done"
    assert read(run.steps[1]) == {
        "running": False,
        "agent": agent,
        "input": "part one",
        "asking": None,
        **outcome,
    }


async def test_job_wait_returns_when_any_notice_arrives(tmp_path: Path) -> None:
    one, two = asyncio.Event(), asyncio.Event()
    model = Crew(
        [
            background("one", "two"),
            use("Job", "r1", id="d1", wait=30),
            text_reply("waiting"),
            text_reply("done"),
        ],
        {"one": [one, text_reply("first")], "two": [two, text_reply("second")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: jobbing(seen), "the lead's Job call")
    two.set()
    await until(lambda: len(model.calls["lead"]) == 3, "the lead's third call")
    assert not one.is_set()
    one.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    assert read(run.steps[2]) == {
        "running": True,
        "agent": "worker",
        "input": "one",
        "asking": None,
    }
    second = str(run.steps[1].child_run_id)
    assert model.calls["lead"][2][-1].content == notice("d2", "worker", second, "second").content


async def test_job_stop_stops_one_helper_and_its_crew(tmp_path: Path) -> None:
    lead_gate, side = asyncio.Event(), asyncio.Event()
    model = Crew(
        [
            started(("b1", "captain", "crew"), ("b2", "worker", "side")),
            lead_gate,
            use("Job", "r1", id="d1", stop=True),
            use("Jobs", "l1"),
            text_reply("waiting"),
            text_reply("done"),
        ],
        {
            "crew": [background("deep"), text_reply("waiting for deep"), text_reply("never")],
            "deep": [asyncio.Event(), text_reply("never")],
            "side": [side, text_reply("side done")],
        },
    )
    seen = Seen()
    engine = team(folder(tmp_path, captain=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(
        lambda: len(model.calls["crew"]) == 2 and len(model.calls["deep"]) == 1,
        "the captain's wait on its own helper",
    )
    lead_gate.set()
    await until(lambda: len(model.calls["lead"]) == 4, "the lead's fourth call")
    assert not side.is_set() and len(model.calls["side"]) == 1
    side.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done" and run.stopped is None
    captain, sibling = run.children
    assert isinstance(captain, Generation) and captain.agent == "captain"
    assert captain.stopped == "now"
    [deep] = captain.children
    assert isinstance(deep, Generation) and deep.stopped == "now"
    assert isinstance(sibling, Generation) and (sibling.reply, sibling.stopped) == (
        "side done",
        None,
    )
    assert read(run.steps[2]) == {
        "running": False,
        "agent": "captain",
        "input": "crew",
        "asking": None,
        "stopped": captain.reply,
    }
    assert stamped(read(run.steps[3])["jobs"]) == [
        {"job": "d1", "agent": "captain", "input": "crew", "running": False, "ended": "stopped"},
        {"job": "d2", "agent": "worker", "input": "side", "running": True},
    ]
    notices = [m.notice for m in run.conversation if m.notice is not None]
    assert [(n.job, n.kind) for n in notices] == [("d1", "stopped"), ("d2", "finished")]
    assert ended(seen)[-1] == run.run_id


async def test_job_send_reaches_a_running_helper_s_inbox(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            background("part one"),
            use("Job", "r1", id="d1", send="use the cache"),
            text_reply("waiting"),
            text_reply("done"),
        ],
        {"part one": [gate, halfway(), text_reply("cached")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 3, "the lead's third call")
    gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    assert read(run.steps[1]) == {
        "running": True,
        "agent": "worker",
        "input": "part one",
        "asking": None,
    }
    [child] = run.children
    assert isinstance(child, Generation) and child.reply == "cached"
    sent = Message(role="user", content="use the cache")
    assert model.calls["part one"][1][-1] == sent
    [event] = taken(seen, child.run_id)
    assert event.messages == [sent] and event.notices == []


async def test_send_reaches_only_a_generation_helper_still_running(tmp_path: Path) -> None:
    desk = Gated()
    model = Crew(
        [
            started(("b1", "desk", "is it urgent?"), ("b2", "worker", "one")),
            use("Job", "r1", id="d2", wait=30),
            calling(
                ToolCall(id="r2", name="Job", args={"id": "d1", "send": "hurry"}),
                ToolCall(id="r3", name="Job", args={"id": "d2", "send": "more"}),
            ),
            text_reply("waiting"),
            text_reply("done"),
        ],
        {"one": [text_reply("first")]},
    )
    engine = team(crew(tmp_path, background=True), model, desk=desk)
    work = asyncio.create_task(engine.run("lead", "acme", "go"))
    await until(lambda: len(model.calls["lead"]) == 4, "the lead's fourth call")
    desk.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    assert read(run.steps[2])["reply"] == "first"
    assert [(step.result, step.is_error) for step in run.steps[3:]] == [
        ("InputError: tools/helpers: helper d1 is a decision run and reads no messages", True),
        (
            "InputError: tools/helpers: helper d2 has ended; send reaches only a helper still running",
            True,
        ),
    ]
    assert isinstance(run.children[1], Decision)


async def test_send_on_a_command_job_is_a_tool_error(tmp_path: Path) -> None:
    model = Crew(
        [
            use("Shell", "s1", command="sleep 60", background=True),
            calling(
                ToolCall(id="r1", name="Job", args={"id": "j1", "send": "x"}),
                ToolCall(id="r2", name="Job", args={"id": "d9"}),
                ToolCall(id="r3", name="Job", args={"id": "j9", "send": "x"}),
            ),
            use("Job", "r4", id="j1", stop=True),
            use("Job", "r5", id="j1"),
            text_reply("done"),
        ]
    )
    room = tmp_path / "room"
    room.mkdir()
    engine = team(folder(tmp_path / "agents", lead={"workspace": True}), model)
    async with Workspace(Local(room), scope="acme") as work:
        run = await asyncio.wait_for(engine.run("lead", "acme", "go", workspace=work), 10)
    assert isinstance(run, Generation) and run.reply == "done"
    assert [(step.result, step.is_error) for step in run.steps[1:4]] == [
        ("InputError: tools/helpers: job j1 is a command; send reaches only a helper", True),
        ("InputError: tools/helpers: there is no helper d9", True),
        ("InputError: tools/helpers: there is no helper j9", True),
    ]
    stopped = read(run.steps[4])
    assert (stopped["output"], stopped["running"]) == ("", False)
    assert (run.steps[5].result, run.steps[5].is_error) == (
        "InputError: tools/workspace: there is no job j1",
        True,
    )


@pytest.mark.parametrize(
    ("deadline", "wait", "refused"),
    [
        pytest.param(5.0, 6, "wait must be between 0 and 5 seconds; got 6", id="past-the-deadline"),
        pytest.param(None, -1, "wait must not be negative; got -1", id="negative"),
    ],
)
async def test_wait_on_a_helper_is_bounded_by_the_tool_deadline(
    tmp_path: Path, deadline: float | None, wait: float, refused: str
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            background("one"),
            use("Job", "r1", id="d1", wait=wait),
            text_reply("waiting"),
            text_reply("done"),
        ],
        {"one": [gate, text_reply("first")]},
    )
    engine = team(crew(tmp_path, background=True), model, tool_timeout=deadline)
    run = await released(asyncio.create_task(engine.run("lead", "acme", "go")), model, gate, 3)
    assert run.reply == "done"
    assert (run.steps[1].result, run.steps[1].is_error) == (
        f"InputError: tools/helpers: {refused}",
        True,
    )
    job = next(tool for tool in run.tools if tool.name == "Job")
    bound = "" if deadline is None else ", up to 5,"
    assert f"wait is seconds{bound} to wait first;" in job.description


async def test_the_bridge_offers_none_of_the_helper_tools(tmp_path: Path, short_root: Path) -> None:
    coder = {
        **AGENTS["coder"],
        "files": {"allow": ["coder.md"]},
        "orchestration": {"delegate_to": ["chat"], "background": True},
    }
    tree = make_tree(tmp_path, {"coder": coder})
    names = ["Job", "Jobs", "Delegate"]
    script(short_root, f"import json\nfor name in {names!r}:\n    print(json.dumps(call(name)))\n")
    async with Workspace(Local(short_root), scope="acme") as work:
        run, _, llm = await run_coder(tmp_path, work, tree=tree)
    assert {"Job", "Jobs", "Delegate"} <= {tool.name for tool in llm.calls[0][2]}
    printed = [json.loads(line) for line in result(run.steps[-1])["stdout"].splitlines()]
    assert printed == [{"error": f"InputError: tools: unknown tool {name!r}"} for name in names]
    assert [step.tool for step in run.steps] == ["Shell"]


async def mine(args: dict[str, Any]) -> str:
    return "mine"


@pytest.mark.parametrize("name", ["Job", "Jobs"])
async def test_a_lead_that_opts_in_reserves_job_and_jobs(tmp_path: Path, name: str) -> None:
    given = Tools([Tool(name, "Mine", body=mine)])
    plain = Crew([text_reply("done")])
    run = await team(crew(tmp_path / "plain"), plain).run("lead", "acme", "go", tools=given)
    assert isinstance(run, Generation) and [tool.name for tool in run.tools] == ["Delegate"]
    refused = rf"^tools/engine_tools: duplicate tool name '{name}'$"
    opted = team(crew(tmp_path / "opted", background=True), Crew([text_reply("never")]))
    with pytest.raises(ConfigurationError, match=refused):
        EngineTools(opted, agent="lead", scope="acme", tools=given)
    with pytest.raises(ConfigurationError, match=refused):
        await opted.run("lead", "acme", "go", tools=given)
    room = tmp_path / "room"
    room.mkdir()
    beside = team(folder(tmp_path / "beside", lead={"workspace": True}), Crew([]))
    async with Workspace(Local(room), scope="acme") as work:
        with pytest.raises(ConfigurationError, match=refused):
            EngineTools(beside, agent="lead", scope="acme", tools=given, workspace=work)
        with pytest.raises(ConfigurationError, match=refused):
            await beside.run("lead", "acme", "go", tools=given, workspace=work)


async def test_a_lead_names_job_and_jobs_or_denies_them(tmp_path: Path) -> None:
    allowed = folder(tmp_path / "allowed", lead={"tools": {"allow": ["Job", "Jobs"]}})
    run = await team(allowed, Crew([text_reply("done")])).run("lead", "acme", "go")
    assert isinstance(run, Generation)
    assert [tool.name for tool in run.tools] == ["Delegate", "Job", "Jobs"]
    for n, deny in enumerate([["Job", "Jobs"], ["Shell", "Job", "Jobs"]]):
        denied = folder(tmp_path / f"denied{n}", lead={"tools": {"deny": deny}})
        given = Tools([Tool("Job", "Mine", body=mine)])
        model = Crew([use("Job", "r1", id="d1"), text_reply("done")])
        run = await team(denied, model).run("lead", "acme", "go", tools=given)
        assert isinstance(run, Generation) and [tool.name for tool in run.tools] == ["Delegate"]
        assert run.steps[0].result == "InputError: tools: unknown tool 'Job'"


@pytest.mark.parametrize(
    ("lead", "deny", "refused"),
    [
        pytest.param(
            {},
            ["Jobs"],
            "lists only Jobs of Job and Jobs in tools.deny; Jobs lists the helpers Job reads, "
            "waits on, messages and stops, so deny the two together or neither",
            id="helpers-only",
        ),
        pytest.param(
            {"workspace": True},
            ["Job", "Jobs"],
            "lists only Job, Jobs of Shell, Job and Jobs in tools.deny; Shell can start a job only "
            "Job and Jobs can read or stop, so deny the three together or none of them",
            id="with-a-workspace",
        ),
    ],
)
def test_job_and_jobs_are_denied_together_and_with_shell_only_where_it_exists(
    tmp_path: Path, lead: dict[str, Any], deny: list[str], refused: str
) -> None:
    root = folder(tmp_path, lead={**lead, "tools": {"deny": deny}})
    with pytest.raises(ConfigurationError, match=rf"^compiled: agent 'lead' {re.escape(refused)}$"):
        team(root, Crew([]))


@pytest.mark.parametrize("workspace", [False, True], ids=["alone", "beside-a-workspace"])
async def test_a_d_id_no_helper_holds_reads_there_is_no_helper(
    tmp_path: Path, workspace: bool
) -> None:
    model = Crew(
        [
            calling(
                ToolCall(id="r1", name="Job", args={"id": "d0"}),
                ToolCall(id="r2", name="Job", args={"id": "d01"}),
                ToolCall(id="r3", name="Job", args={"id": "d1"}),
            ),
            text_reply("done"),
        ]
    )
    room = tmp_path / "room"
    room.mkdir()
    engine = team(folder(tmp_path / "agents", lead={"workspace": workspace}), model)
    async with Workspace(Local(room), scope="acme") as work:
        run = await engine.run("lead", "acme", "go", workspace=work if workspace else None)
    assert isinstance(run, Generation) and run.reply == "done"
    assert [step.result for step in run.steps] == [
        f"InputError: tools/helpers: there is no helper {job}" for job in ("d0", "d01", "d1")
    ]


@pytest.mark.parametrize(
    ("deadline", "wait"),
    [
        pytest.param(30.0, 0.05, id="within-the-deadline"),
        pytest.param(0.2, 0.2, id="at-the-deadline"),
    ],
)
async def test_a_wait_that_runs_out_reads_the_helper_still_running(
    tmp_path: Path, deadline: float, wait: float
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            background("one"),
            use("Job", "r1", id="d1", wait=wait),
            text_reply("waiting"),
            text_reply("done"),
        ],
        {"one": [gate, text_reply("first")]},
    )
    engine = team(crew(tmp_path, background=True), model, tool_timeout=deadline)
    run = await released(asyncio.create_task(engine.run("lead", "acme", "go")), model, gate, 3)
    assert run.reply == "done"
    assert read(run.steps[1]) == {
        "running": True,
        "agent": "worker",
        "input": "one",
        "asking": None,
    }


@pytest.mark.parametrize("deadline", [30.0, None], ids=["bounded", "no-deadline"])
async def test_job_stop_returns_once_its_helper_ended_while_another_notice_waits(
    tmp_path: Path, deadline: float | None
) -> None:
    lead_gate, quick = asyncio.Event(), asyncio.Event()
    model = Crew(
        [
            started(("b1", "worker", "slow"), ("b2", "worker", "quick")),
            lead_gate,
            use("Job", "r1", id="d1", stop=True),
            text_reply("done"),
        ],
        {
            "slow": [asyncio.Event(), text_reply("never")],
            "quick": [quick, text_reply("quick done")],
        },
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model, tool_timeout=deadline)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    quick.set()
    await until(
        lambda: any(isinstance(e, RunEnded) for e in seen.events), "d2's end during that call"
    )
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    assert read(run.steps[2]) == {
        "running": False,
        "agent": "worker",
        "input": "slow",
        "asking": None,
        "stopped": "",
    }
    notices = [m.notice for m in run.conversation if m.notice is not None]
    assert [(n.job, n.kind) for n in notices] == [("d2", "finished"), ("d1", "stopped")]


async def test_job_stop_on_a_helper_that_ended_reads_how_it_ended(tmp_path: Path) -> None:
    model = Crew(
        [
            background("one"),
            use("Job", "r1", id="d1", wait=30),
            use("Job", "r2", id="d1", stop=True),
            text_reply("done"),
        ],
        {"one": [text_reply("first")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    run = await asyncio.wait_for(engine.run("lead", "acme", "go"), 5)
    assert isinstance(run, Generation) and run.reply == "done"
    ended_read = {
        "running": False,
        "agent": "worker",
        "input": "one",
        "asking": None,
        "reply": "first",
    }
    assert read(run.steps[1]) == read(run.steps[2]) == ended_read
    [child] = run.children
    assert isinstance(child, Generation) and child.stopped is None
    assert [m.notice.kind for m in run.conversation if m.notice is not None] == ["finished"]


@pytest.mark.parametrize(
    ("tool", "result_text"),
    [
        pytest.param(
            ToolCall(id="r1", name="Job", args={"id": "d3"}),
            '{"running": false, "agent": "worker", "asking": null, "reply": "old"}',
            id="job",
        ),
        pytest.param(
            ToolCall(id="l1", name="Jobs"),
            '{"jobs": [{"job": "d3", "agent": "worker", "started": "2026-11-21T10:00:00+00:00", "running": true}]}',
            id="jobs",
        ),
    ],
)
async def test_job_ids_continue_after_a_job_or_jobs_call_in_the_window(
    tmp_path: Path, tool: ToolCall, result_text: str
) -> None:
    history = [
        Message(role="user", content="earlier"),
        Message(role="assistant", tool_calls=[tool]),
        Message(role="tool", tool_call_id=tool.id, content=result_text),
        Message(role="assistant", content="ok"),
    ]
    gate = asyncio.Event()
    model = Crew(
        [background("one"), text_reply("waiting"), text_reply("done")],
        {"one": [gate, text_reply("first")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", history=history))
    run = await released(work, model, gate)
    assert run.steps[0].result == '{"job": "d4"}'


ASK = (
    "Ask lead, the agent that started you in the background, a question, and wait for its "
    "answer: its answer is this call's result. If lead ends its turn without answering, the call "
    "ends with an error, and you go on with what you have."
)
UNANSWERED = "InputError: tools/helpers: your lead ended its turn without answering"
LEFT = (
    "InputError: tools/helpers: your lead was stopped or failed before it answered; "
    "you stop with it"
)
ROUTED = '{"type": "choice", "instructions": "Which?", "options": {"talk": null, "work": null}}'


def asking(question: str = "which lemma?", call_id: str = "a1") -> Reply:
    return use("Ask", call_id, question=question)


def asked(seen: Seen) -> bool:
    return any(isinstance(e, ToolCallStarted) and e.tool == "Ask" for e in seen.events)


def kinds(run: Generation) -> list[str]:
    return [m.notice.kind for m in run.conversation if m.notice is not None]


def outcome(step: Step) -> tuple[str, Any, bool]:
    return step.tool, step.result, step.is_error


async def test_ask_waits_for_the_lead_s_answer(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            use("Job", "r1", id="d1"),
            use("Job", "r2", id="d1", send="lemma 4", wait=30),
            text_reply("done"),
        ],
        {"part one": [gate, asking(), text_reply("proved with lemma 4")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    run = await released(asyncio.create_task(engine.run("lead", "acme", "go")), model, gate)
    assert run.reply == "done"
    [child] = run.children
    assert isinstance(child, Generation) and child.reply == "proved with lemma 4"
    [tool] = child.tools
    assert (tool.name, tool.description) == ("Ask", ASK)
    assert list(tool.input_schema["properties"]) == ["question"]
    assert tool.input_schema["required"] == ["question"]
    [step] = child.steps
    assert outcome(step) == ("Ask", "lemma 4", False)
    assert step.args == {"question": "which lemma?"}
    answered = Message(role="tool", tool_call_id="a1", content="lemma 4")
    assert model.calls["part one"][1][-1] == answered
    told = notice("d1", "worker", child.run_id, "which lemma?", kind="asks")
    assert told.content == "[helper d1 · worker · asks]\nwhich lemma?"
    assert model.calls["lead"][2][-1] == Message(role="user", content=told.content)
    assert [m for m in run.conversation if m.notice is not None] == [
        told,
        notice("d1", "worker", child.run_id, "proved with lemma 4"),
    ]
    assert read(run.steps[1]) == {
        "running": True,
        "agent": "worker",
        "input": "part one",
        "asking": "which lemma?",
    }
    assert read(run.steps[2]) == {
        "running": False,
        "agent": "worker",
        "input": "part one",
        "asking": None,
        "reply": "proved with lemma 4",
    }


async def test_only_a_background_helper_is_offered_ask(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            use("Delegate", "s1", agent="worker", input="in the turn"),
            background("beside"),
            text_reply("waiting"),
            text_reply("done"),
        ],
        {"in the turn": [text_reply("synchronous")], "beside": [gate, text_reply("helped")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go"))
    run = await released(work, model, gate, 3)
    assert run.reply == "done"
    assert [tool.name for tool in run.tools] == ["Delegate", "Job", "Jobs"]
    synchronous, helper = run.children
    assert isinstance(synchronous, Generation) and synchronous.reply == "synchronous"
    assert isinstance(helper, Generation) and helper.reply == "helped"
    assert synchronous.tools == []
    assert [tool.name for tool in helper.tools] == ["Ask"]


async def test_a_run_a_decision_helper_routes_to_is_offered_no_ask(tmp_path: Path) -> None:
    root = write_folder(
        tmp_path,
        {
            "lead": {
                "kind": "generation",
                "model": "crew",
                "description": "Leads the crew",
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["router"], "background": True},
            },
            "router": {
                "kind": "decision",
                "model": "judge",
                "description": "Routes",
                "instructions": "router.md",
                "questions": ["router/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"work": "worker"}}},
            },
            "worker": {
                "kind": "generation",
                "model": "crew",
                "description": "Works on one part",
                "instructions": "worker.md",
            },
        },
        files={
            "lead.md": "Lead.\n",
            "router.md": "",
            "worker.md": "Work.\n",
            "router/intent.json": ROUTED,
        },
    )
    gate = asyncio.Event()
    model = Crew(
        [background("part one", agent="router"), text_reply("waiting"), text_reply("done")],
        {"part one": [gate, text_reply("routed reply")]},
    )
    run = await released(
        asyncio.create_task(team(root, model).run("lead", "acme", "go")), model, gate
    )
    assert run.reply == "done"
    [router] = run.children
    assert isinstance(router, Decision)
    routed = router.routed_to
    assert isinstance(routed, Generation) and routed.reply == "routed reply"
    assert routed.tools == []


async def test_a_lead_that_ends_its_turn_without_answering_fails_the_ask(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            text_reply("no answer"),
            text_reply("done"),
        ],
        {"part one": [gate, asking(), text_reply("went on without it")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    run = await released(asyncio.create_task(engine.run("lead", "acme", "go")), model, gate)
    assert (run.reply, run.calls) == ("done", 4)
    [child] = run.children
    assert isinstance(child, Generation)
    assert (child.reply, child.stopped) == ("went on without it", None)
    assert [outcome(step) for step in child.steps] == [("Ask", UNANSWERED, True)]
    failed = Message(role="tool", tool_call_id="a1", content=UNANSWERED, is_error=True)
    assert model.calls["part one"][1][-1] == failed
    assert model.calls["lead"][2][-1].content == "[helper d1 · worker · asks]\nwhich lemma?"
    assert kinds(run) == ["asks", "finished"]


async def test_a_question_that_arrives_during_the_lead_s_model_call_is_given_next(
    tmp_path: Path,
) -> None:
    gate, lead_gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [
            background("part one"),
            lead_gate,
            text_reply("waiting"),
            use("Job", "r1", id="d1", send="lemma 4", wait=30),
            text_reply("done"),
        ],
        {"part one": [gate, asking(), text_reply("proved")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gate.set()
    await until(lambda: asked(seen), "the helper's question")
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and (run.reply, run.calls) == ("done", 4)
    [child] = run.children
    assert isinstance(child, Generation) and child.reply == "proved"
    assert [outcome(step) for step in child.steps] == [("Ask", "lemma 4", False)]
    assert all(
        m.content != "[helper d1 · worker · asks]\nwhich lemma?" for m in model.calls["lead"][1]
    )
    assert model.calls["lead"][2][-1].content == "[helper d1 · worker · asks]\nwhich lemma?"
    assert kinds(run) == ["asks", "finished"]


@pytest.mark.parametrize("how", ["now", "after_step", "failed"])
async def test_a_lead_that_stops_fails_the_ask_and_stops_the_helper(
    tmp_path: Path, how: str
) -> None:
    gate, lead_gate = asyncio.Event(), asyncio.Event()
    ghost = Reply(tool_calls=[ToolCall(id="g1", name="ghost", args={})], stop_reason="tool_use")
    model = Crew(
        [background("part one"), text_reply("waiting"), lead_gate, ghost],
        {"part one": [gate, asking(), text_reply("never")]},
    )
    seen, stop, kept = Seen(), Stop(), Kept()
    engine = team(crew(tmp_path, background=True), model, max_model_calls=3, refinery=kept)
    work = asyncio.create_task(engine.run("lead", "acme", "go", stop=stop, on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gate.set()
    await until(lambda: len(model.calls["lead"]) == 3, "the lead's third call")
    if how == "now":
        stop.now()
    if how == "after_step":
        stop.after_step()
    lead_gate.set()
    if how == "failed":
        with pytest.raises(RunawayError, match=r"^engine: run exceeded 3 model calls$"):
            await asyncio.wait_for(work, 5)
        [run] = kept.runs
    else:
        run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation)
    assert run.stopped == (None if how == "failed" else how)
    [child] = run.children
    assert isinstance(child, Generation)
    assert child.stopped == ("after_step" if how == "after_step" else "now")
    [step] = child.steps
    cut = how == "now"
    assert outcome(step) == ("Ask", "stopped by the user" if cut else LEFT, True)
    assert step.stopped is cut
    assert model.calls["lead"][2][-1].content == "[helper d1 · worker · asks]\nwhich lemma?"
    assert kinds(run) == ["asks", "stopped"]
    assert ended(seen) == [child.run_id, run.run_id]


async def test_an_ask_made_once_the_lead_is_stopping_fails_at_once(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [gate, asking(), text_reply("never")]},
    )
    stop, inbox = Stop(), Watched()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", stop=stop, inbox=inbox))
    await until(lambda: inbox.waiting == 1 and len(model.calls["part one"]) == 1, "the lead's hold")
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the lead's settle")
    gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "after_step"
    [child] = run.children
    assert isinstance(child, Generation) and child.stopped == "after_step"
    assert [outcome(step) for step in child.steps] == [("Ask", LEFT, True)]
    assert kinds(run) == ["stopped"]


async def test_an_app_send_never_answers_an_ask(tmp_path: Path) -> None:
    gate, lead_gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            lead_gate,
            use("Job", "r1", id="d1", send="lemma 4", wait=30),
            text_reply("done"),
        ],
        {"part one": [gate, asking(), text_reply("proved")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    helper_id = await child_of(seen)
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gate.set()
    await until(lambda: len(model.calls["lead"]) == 3, "the lead's third call")
    await engine.send(helper_id, "from the app", scope="acme")
    assert not any(isinstance(e, ToolResultReady) and e.run_id == helper_id for e in seen.events)
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    [child] = run.children
    assert isinstance(child, Generation) and child.reply == "proved"
    assert [outcome(step) for step in child.steps] == [("Ask", "lemma 4", False)]
    sent = Message(role="user", content="from the app")
    assert model.calls["part one"][1][-2:] == [
        Message(role="tool", tool_call_id="a1", content="lemma 4"),
        sent,
    ]
    [event] = taken(seen, helper_id)
    assert event.messages == [sent] and event.notices == []
    assert read(run.steps[1])["reply"] == "proved"


async def test_the_bridge_offers_no_ask(tmp_path: Path, short_root: Path) -> None:
    lead = {**AGENTS["lead"], "orchestration": {"delegate_to": ["coder"], "background": True}}
    coder = {**AGENTS["coder"], "files": {"allow": ["coder.md"]}}
    tree = make_tree(tmp_path, {"lead": lead, "coder": coder})
    script(short_root, "import json\nprint(json.dumps(call('Ask', question='which lemma?')))\n")
    gate = asyncio.Event()
    model = Crew(
        [background("part one", agent="coder"), text_reply("waiting"), text_reply("done")],
        {"part one": [gate, use("Shell", "s1", command=f"{PY} script.py"), text_reply("ran it")]},
        marker="You are lead.",
    )
    engine = Engine(tree, providers=provided({"strong": model, "judge": None}))
    async with Workspace(Local(short_root), scope="acme") as work:
        running = asyncio.create_task(engine.run("lead", "acme", "go", workspace=work))
        run = await released(running, model, gate)
    assert run.reply == "done"
    [child] = run.children
    assert isinstance(child, Generation) and child.reply == "ran it"
    assert "Ask" in [tool.name for tool in child.tools]
    printed = [json.loads(line) for line in result(child.steps[-1])["stdout"].splitlines()]
    assert printed == [{"error": "InputError: tools: unknown tool 'Ask'"}]
    assert [step.tool for step in child.steps] == ["Shell"]


async def hang(args: dict[str, Any]) -> str:
    await asyncio.Event().wait()
    return "never"


def timed_out(seen: Seen) -> bool:
    return any(isinstance(e, ToolResultReady) and e.step.tool == "lookup" for e in seen.events)


async def test_ask_holds_no_slot_and_no_deadline(tmp_path: Path) -> None:
    gate, lead_gate = asyncio.Event(), asyncio.Event()
    both = calling(
        ToolCall(id="a1", name="Ask", args={"question": "which lemma?"}),
        ToolCall(id="t1", name="lookup", args={}),
    )
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            lead_gate,
            use("Job", "r1", id="d1", send="lemma 4", wait=30),
            text_reply("done"),
        ],
        {"part one": [gate, both, text_reply("proved")]},
    )
    seen = Seen()
    worker = {"tools": {"allow": ["lookup"], "timeout": 0.1, "max_parallel": 1}}
    engine = team(crew(tmp_path, worker=worker, background=True), model)
    given = Tools([Tool("lookup", "Hangs", body=hang)])
    work = asyncio.create_task(engine.run("lead", "acme", "go", tools=given, on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gate.set()
    await until(lambda: timed_out(seen), "lookup's deadline")
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    [child] = run.children
    assert isinstance(child, Generation) and child.reply == "proved"
    ask, lookup = child.steps
    assert outcome(ask) == ("Ask", "lemma 4", False) and ask.duration_ms >= 100
    assert outcome(lookup) == ("lookup", "TimeoutError: engine: lookup timed out after 0.1s", True)


async def test_a_runtime_tool_named_ask_fails_the_background_start(tmp_path: Path) -> None:
    desk = Gated()
    model = Crew(
        [
            started(("b1", "worker", "part one"), ("b2", "desk", "is it urgent?")),
            text_reply("waiting"),
            text_reply("done"),
        ],
        {"part one": [text_reply("never")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model, desk=desk)
    given = Tools([Tool("Ask", "Mine", body=mine)])
    work = asyncio.create_task(engine.run("lead", "acme", "go", tools=given, on_event=seen))
    run = await released(work, model, desk.gate)
    assert run.reply == "done"
    refused, begun = run.steps
    assert (refused.result, refused.is_error, refused.child_run_id) == (
        "ConfigurationError: tools/engine_tools: a tool named 'Ask' collides with the builtin",
        True,
        None,
    )
    assert read(begun) == {"job": "d1"}
    [decided] = run.children
    assert isinstance(decided, Decision) and decided.run_id == begun.child_run_id
    assert model.calls["part one"] == []
    assert seen.started(below=True) == []


async def test_each_send_answers_the_oldest_question_still_waiting(tmp_path: Path) -> None:
    gate, lead_gate = asyncio.Event(), asyncio.Event()
    both = calling(
        ToolCall(id="a1", name="Ask", args={"question": "first?"}),
        ToolCall(id="a2", name="Ask", args={"question": "second?"}),
    )
    model = Crew(
        [
            background("part one"),
            lead_gate,
            text_reply("waiting"),
            use("Job", "r1", id="d1"),
            use("Job", "r2", id="d1", send="first answer"),
            use("Job", "r3", id="d1"),
            use("Job", "r4", id="d1", send="second answer", wait=30),
            text_reply("done"),
        ],
        {"part one": [gate, both, text_reply("both answered")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gate.set()
    await until(
        lambda: sum(isinstance(e, ToolCallStarted) and e.tool == "Ask" for e in seen.events) == 2,
        "both questions",
    )
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "done"
    [child] = run.children
    assert isinstance(child, Generation) and child.reply == "both answered"
    assert [outcome(step) for step in child.steps] == [
        ("Ask", "first answer", False),
        ("Ask", "second answer", False),
    ]
    first, _, second, last = (read(step) for step in run.steps[1:])
    assert (first["asking"], second["asking"]) == ("first?", "second?")
    assert last == {
        "running": False,
        "agent": "worker",
        "input": "part one",
        "asking": None,
        "reply": "both answered",
    }
    assert model.calls["lead"][2][-2:] == [
        Message(role="user", content="[helper d1 · worker · asks]\nfirst?"),
        Message(role="user", content="[helper d1 · worker · asks]\nsecond?"),
    ]


async def test_ask_goes_through_approve(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("done")],
        {"part one": [gate, asking(), text_reply("went on")]},
    )
    named: list[str] = []

    async def approve(agent: str, call: ToolCall) -> bool:
        named.append(call.name)
        return call.name != "Ask"

    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", approve=approve))
    run = await released(work, model, gate)
    assert run.reply == "done" and named == ["Delegate", "Ask"]
    [child] = run.children
    assert isinstance(child, Generation) and child.reply == "went on"
    [step] = child.steps
    assert outcome(step) == ("Ask", "NotApproved: Ask was not approved", True)
    assert step.approved is False
    assert kinds(run) == ["finished"]


async def test_job_wait_on_a_helper_waiting_in_ask_returns_at_once(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            use("Job", "r1", id="d1", wait=30),
            use("Job", "r2", id="d1", send="lemma 4", wait=30),
            text_reply("done"),
        ],
        {"part one": [gate, asking(), text_reply("proved")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    run = await released(asyncio.create_task(engine.run("lead", "acme", "go")), model, gate)
    assert run.reply == "done"
    assert model.calls["lead"][2][-1].content == "[helper d1 · worker · asks]\nwhich lemma?"
    assert read(run.steps[1]) == {
        "running": True,
        "agent": "worker",
        "input": "part one",
        "asking": "which lemma?",
    }
    assert run.steps[1].duration_ms < 5000
    assert read(run.steps[2])["reply"] == "proved"
