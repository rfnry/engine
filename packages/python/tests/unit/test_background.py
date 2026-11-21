import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import rfnry_engine.helpers
from rfnry_engine import (
    CheckAnswer,
    ChoiceAnswer,
    EngineTools,
    Fingerprint,
    MessagesTaken,
    Notice,
    Refinery,
    RunEnded,
    RunStarted,
    Stop,
    Storage,
    ToolResultReady,
    current_call,
)
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ModelError, RunawayError
from rfnry_engine.models import Message, QuestionLike, Reply, State, ToolCall, ToolSpec
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.delegate import child_result
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import (
    URGENT,
    Answering,
    Crew,
    Seen,
    background,
    crew,
    released,
    team,
    text_reply,
    until,
)
from .test_failed_runs import Kept
from .trees import provided, write_folder

AGENT_0_1_13 = "34102ac5a160ec02"
TOOLS_0_1_13 = "1df013bbe33bc7a6"
PREFIX_0_1_13 = "cf5a7c0b8c982083"
DESCRIPTION_0_1_13 = (
    "Hand an input to another agent and get its reply.\n"
    "- worker: Works on one part\n"
    "- desk (decision agent): Sorts"
)
URGENT_QUESTION = {"type": "check", "instructions": "Urgent?"}
PAST_CAP = "InputError: tools/delegate: max_background is 1, and running now: d1; wait for a helper to end or stop one"
STILL_RUNNING = "engine: still running: d1; stop them or wait for them, then call Output again"


def notice(job: str, agent: str, run_id: str, body: str, kind: Any = "finished") -> Message:
    return Message(
        role="user",
        content=f"[helper {job} · {agent} · {kind}]\n{body}",
        notice=Notice(job=job, agent=agent, run_id=run_id, kind=kind),
    )


def halfway() -> Reply:
    ghost = ToolCall(id="h1", name="ghost", args={})
    return Reply(text="half way", tool_calls=[ghost], stop_reason="tool_use")


def output(text: str, call_id: str = "o1") -> Reply:
    call = ToolCall(id=call_id, name="Output", args={"text": text})
    return Reply(tool_calls=[call], stop_reason="tool_use")


def pending(before: set[asyncio.Task[Any]]) -> list[asyncio.Task[Any]]:
    return [task for task in asyncio.all_tasks() if task not in before and not task.done()]


class Answer(BaseModel):
    text: str


class Watched(asyncio.Queue[Any]):
    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self.waiting = 0

    async def get(self) -> Any:
        self.waiting += 1
        try:
            return await super().get()
        finally:
            self.waiting -= 1


class WatchedLifo(Watched, asyncio.LifoQueue[Any]):
    pass


class WatchedPriority(Watched, asyncio.PriorityQueue[Any]):
    pass


class Stubborn:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancels = 0

    async def body(self, args: dict[str, Any]) -> str:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancels += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancels += 1
            raise
        return "never"

    def team(self, root: Path, lead: list[Reply | asyncio.Event]) -> tuple[Engine, Crew, Tools]:
        use = Reply(tool_calls=[ToolCall(id="t1", name="lookup", args={})], stop_reason="tool_use")
        model = Crew(lead, {"part one": [use, text_reply("never")]})
        folder = crew(root, worker={"tools": {"allow": ["lookup"]}}, background=True)
        return (
            team(folder, model),
            model,
            Tools([Tool("lookup", "Ignores one cancel", body=self.body)]),
        )


class Failing:
    def __init__(self, gate: asyncio.Event) -> None:
        self.gate = gate

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        await self.gate.wait()
        raise ModelError("worker: the model is down")


def said(messages: list[Message]) -> list[str]:
    return [str(m.content) for m in messages if m.role == "assistant" and not m.tool_calls]


def ended(seen: Seen) -> list[str]:
    return [e.run_id for e in seen.events if isinstance(e, RunEnded)]


def ends(seen: Seen, count: int) -> Callable[[], bool]:
    return lambda: len(ended(seen)) == count


def taken(seen: Seen, run_id: str) -> list[MessagesTaken]:
    return [e for e in seen.events if isinstance(e, MessagesTaken) and e.run_id == run_id]


class Gated:
    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.asked = asyncio.Event()

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
        self.asked.set()
        await self.gate.wait()
        return {"urgent": CheckAnswer(probability=0.9)}


async def test_background_is_offered_only_to_an_entry_that_opts_in(tmp_path: Path) -> None:
    plain = Crew([text_reply("done", reasoning="thought")])
    run = await team(crew(tmp_path / "plain"), plain).run("lead", "acme", "go")
    assert isinstance(run, Generation)
    [tool] = run.tools
    assert tool.name == "Delegate" and tool.description == DESCRIPTION_0_1_13
    assert list(tool.input_schema["properties"]) == ["agent", "input", "questions"]
    assert run.fingerprint == Fingerprint(agent=AGENT_0_1_13, tools=TOOLS_0_1_13, questions=None)
    assert run.conversation[-1].prefix == PREFIX_0_1_13
    opted = Crew([text_reply("done")])
    run = await team(crew(tmp_path / "opted", background=True), opted).run("lead", "acme", "go")
    assert isinstance(run, Generation)
    assert [tool.name for tool in run.tools] == ["Delegate", "Job", "Jobs"]
    tool = run.tools[0]
    assert list(tool.input_schema["properties"]) == ["agent", "input", "questions", "background"]
    assert tool.input_schema["properties"]["background"]["type"] == "boolean"
    assert tool.input_schema["required"] == ["agent", "input"]
    assert tool.description.startswith("Hand an input to another agent and get its reply. ")
    assert tool.description.endswith(
        "\n- worker: Works on one part\n- desk (decision agent): Sorts"
    )
    assert run.fingerprint.tools != TOOLS_0_1_13


async def test_a_lead_that_does_not_opt_in_runs_a_background_delegate_as_a_synchronous_one(
    tmp_path: Path,
) -> None:
    unoffered = ToolCall(
        id="b2", name="Delegate", args={"agent": "worker", "input": "two", "background": "soon"}
    )
    model = Crew(
        [
            background("one"),
            Reply(tool_calls=[unoffered], stop_reason="tool_use"),
            text_reply("done"),
        ],
        {"one": [text_reply("one done")], "two": [text_reply("two done")]},
    )
    run = await team(crew(tmp_path), model).run("lead", "acme", "go")
    assert isinstance(run, Generation) and run.reply == "done"
    one, two = run.children
    assert [(s.result, s.is_error) for s in run.steps] == [("one done", False), ("two done", False)]
    assert [s.child_run_id for s in run.steps] == [one.run_id, two.run_id]
    assert run.steps[0].args == {"agent": "worker", "input": "one", "background": True}
    assert [m.notice for m in run.conversation] == [None] * len(run.conversation)


async def test_a_delegate_dispatched_from_code_runs_a_background_one_synchronously(
    tmp_path: Path,
) -> None:
    model = Crew([], {"one": [text_reply("one done")]})
    engine = team(crew(tmp_path, background=True), model)
    tools = EngineTools(engine, agent="lead", scope="acme")
    [delegate] = [d for d in tools.definitions() if d["name"] == "Delegate"]
    assert list(delegate["input_schema"]["properties"]) == ["agent", "input", "questions"]
    assert delegate["description"] == DESCRIPTION_0_1_13
    args = {"agent": "worker", "input": "one", "background": True}
    assert await tools.dispatch("Delegate", args) == "one done"
    assert len(model.calls["one"]) == 1


async def test_a_background_delegate_returns_a_job_at_once_and_the_lead_goes_on(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("started"), text_reply("all done")],
        {"part one": [gate, text_reply("part one done")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    assert len(model.calls["part one"]) == 1 and not gate.is_set()
    [helper_id] = seen.started(below=True)
    run = await released(work, model, gate)
    step = run.steps[0]
    assert (step.tool, step.result, step.is_error) == ("Delegate", '{"job": "d1"}', False)
    assert step.child_run_id == helper_id
    assert model.calls["lead"][1][-1] == Message(
        role="tool", tool_call_id="b1", content='{"job": "d1"}'
    )
    [child] = run.children
    assert isinstance(child, Generation) and child.run_id == helper_id
    assert child.parent_run_id == run.run_id and child.reply == "part one done"
    assert run.reply == "all done"


async def test_a_helper_s_end_reaches_the_lead_as_a_notice_before_its_next_model_call(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("done")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    run = await released(work, model, gate)
    helper_id = str(run.steps[0].child_run_id)
    expected = notice("d1", "worker", helper_id, "the answer")
    assert run.conversation[-2] == expected
    sent = model.calls["lead"][2][-1]
    assert (sent.role, sent.content, sent.notice) == ("user", expected.content, None)
    [event] = taken(seen, run.run_id)
    assert event.notices == [expected] and event.messages == [] and event.turn == 2


async def test_a_reply_while_helpers_run_is_held_until_they_end(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("final")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    run = await released(work, model, gate)
    helper_id = str(run.steps[0].child_run_id)
    assert model.calls["lead"][2][-1].content == "[helper d1 · worker · finished]\nthe answer"
    assert run.reply == "final" and run.calls == 3
    assert said(run.conversation) == ["waiting", "final"]
    assert ended(seen) == [helper_id, run.run_id]


async def test_a_notice_that_arrived_during_the_model_call_keeps_the_run_going(
    tmp_path: Path,
) -> None:
    lead_gate, gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("part one"), lead_gate, text_reply("waiting"), text_reply("final")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gate.set()
    await until(ends(seen, 1), "the helper's end")
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "final"
    assert len(model.calls["lead"]) == 3
    assert all(m.notice is None for m in run.conversation[:4])
    assert model.calls["lead"][2][-1].content == "[helper d1 · worker · finished]\nthe answer"


async def test_helpers_join_children_in_the_order_they_end(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("one", "two"), *[text_reply("waiting")] * 4],
        {"one": [gate, text_reply("first")], "two": [text_reply("second")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(ends(seen, 1), "d2's end")
    gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation)
    one, two = (step.child_run_id for step in run.steps)
    assert [json.loads(str(step.result))["job"] for step in run.steps] == ["d1", "d2"]
    assert [child.run_id for child in run.children] == [two, one]
    notices = [m.notice for m in run.conversation if m.notice is not None]
    assert [(n.job, n.run_id) for n in notices] == [("d2", two), ("d1", one)]


async def test_many_helpers_ending_between_two_model_calls_are_one_wake_up(
    tmp_path: Path,
) -> None:
    lead_gate = asyncio.Event()
    gates = {str(n): asyncio.Event() for n in range(1, 6)}
    model = Crew(
        [background(*gates), lead_gate, text_reply("waiting"), text_reply("final")],
        {name: [gate, text_reply(f"reply {name}")] for name, gate in gates.items()},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    order = ["3", "1", "5", "2", "4"]
    for n, name in enumerate(order, 1):
        gates[name].set()
        await until(ends(seen, n), f"helper {name}'s end")
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "final"
    assert len(model.calls["lead"]) == 3 and run.calls == 3
    [event] = taken(seen, run.run_id)
    assert [m.notice.job for m in event.notices if m.notice] == [f"d{n}" for n in order]
    assert [m.content for m in event.notices] == [
        f"[helper d{n} · worker · finished]\nreply {n}" for n in order
    ]
    assert event.messages == [] and event.turn == 2
    assert model.calls["lead"][2][-5:] == [
        m.model_copy(update={"notice": None}) for m in event.notices
    ]


async def test_an_inbox_message_wakes_a_held_lead_but_keeps_no_run_going(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            text_reply("still waiting"),
            text_reply("final"),
        ],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen, inbox = Seen(), asyncio.Queue[Any]()
    engine = team(crew(tmp_path / "held", background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen, inbox=inbox))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    inbox.put_nowait("6 hours left")
    inbox.put_nowait("hurry")
    await until(lambda: len(model.calls["lead"]) == 3, "the lead's call on the message")
    assert len(model.calls["part one"]) == 1 and ended(seen) == []
    assert [m.content for m in model.calls["lead"][2][-2:]] == ["6 hours left", "hurry"]
    run = await released(work, model, gate, calls=3)
    first, second = taken(seen, run.run_id)
    assert first.messages == [
        Message(role="user", content="6 hours left"),
        Message(role="user", content="hurry"),
    ]
    assert first.notices == [] and first.turn == 2
    assert second.messages == [] and [m.notice for m in second.notices] == [
        Notice(job="d1", agent="worker", run_id=str(run.steps[0].child_run_id), kind="finished")
    ]
    assert said(run.conversation) == ["waiting", "still waiting", "final"]
    assert inbox.empty()

    lone_gate = asyncio.Event()
    lone = Crew([lone_gate, text_reply("done")])
    left = asyncio.Queue[Any]()
    engine = team(crew(tmp_path / "lone", background=True), lone)
    work = asyncio.create_task(engine.run("lead", "acme", "go", inbox=left))
    await until(lambda: len(lone.calls["lead"]) == 1, "the lone lead's call")
    left.put_nowait("one more thing")
    lone_gate.set()
    alone = await asyncio.wait_for(work, 5)
    assert isinstance(alone, Generation) and alone.reply == "done" and alone.calls == 1
    assert left.get_nowait() == "one more thing"


async def test_notices_come_before_the_inbox_s_messages(tmp_path: Path) -> None:
    lead_gate, gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("part one"), lead_gate, text_reply("waiting"), text_reply("final")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen, inbox = Seen(), asyncio.Queue[Any]()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen, inbox=inbox))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gate.set()
    await until(ends(seen, 1), "the helper's end")
    inbox.put_nowait("a person's words")
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "final"
    helper_id = str(run.steps[0].child_run_id)
    person = Message(role="user", content="a person's words")
    expected = notice("d1", "worker", helper_id, "the answer")
    assert run.conversation[-3:-1] == [expected, person]
    assert [m.content for m in model.calls["lead"][2][-2:]] == [expected.content, person.content]
    [event] = taken(seen, run.run_id)
    assert event.notices == [expected] and event.messages == [person]


async def test_job_ids_continue_after_the_history(tmp_path: Path) -> None:
    history = [
        Message(role="user", content="earlier"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="x1", name="Delegate", args={"agent": "worker", "input": "a"}),
                ToolCall(id="x2", name="lookup", args={}),
            ],
        ),
        Message(role="tool", tool_call_id="x1", content='{"job": "d2"}'),
        Message(role="tool", tool_call_id="x2", content='{"job": "d7"}'),
        Message(role="assistant", content="ok"),
    ]
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("final")],
        {"part one": [gate, text_reply("the answer")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", history=history))
    run = await released(work, model, gate)
    assert run.steps[0].result == '{"job": "d3"}'
    [last] = [m.notice for m in run.conversation if m.notice is not None]
    assert last.job == "d3"


async def test_past_max_background_a_background_delegate_is_refused_at_once(
    tmp_path: Path,
) -> None:
    first, second = asyncio.Event(), asyncio.Event()
    model = Crew(
        [
            background("one", "two"),
            text_reply("waiting"),
            background("three", prefix="c"),
            text_reply("waiting"),
            text_reply("final"),
        ],
        {
            "one": [first, text_reply("first")],
            "two": [text_reply("never")],
            "three": [second, text_reply("third")],
        },
    )
    engine = team(crew(tmp_path, background=True, max_background=1), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go"))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    first.set()
    run = await released(work, model, second, calls=4)
    refused = run.steps[1]
    assert (refused.result, refused.is_error, refused.child_run_id) == (PAST_CAP, True, None)
    assert model.calls["two"] == [] and len(run.children) == 2
    assert [step.result for step in (run.steps[0], run.steps[2])] == [
        '{"job": "d1"}',
        '{"job": "d2"}',
    ]
    assert run.reply == "final"


async def test_questions_refused_before_the_start_spend_no_id(tmp_path: Path) -> None:
    gate = asyncio.Event()
    asking = ToolCall(
        id="b1",
        name="Delegate",
        args={
            "agent": "worker",
            "input": "x",
            "questions": {"q": URGENT_QUESTION},
            "background": True,
        },
    )
    model = Crew(
        [
            Reply(tool_calls=[asking], stop_reason="tool_use"),
            background("part one", prefix="c"),
            text_reply("waiting"),
            text_reply("final"),
        ],
        {"x": [text_reply("never")], "part one": [gate, text_reply("the answer")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go"))
    run = await released(work, model, gate, calls=3)
    refused, started = run.steps
    assert (refused.result, refused.child_run_id) == (
        "InputError: engine: generation agent 'worker' takes no questions",
        None,
    )
    assert started.result == '{"job": "d1"}' and model.calls["x"] == []


async def test_a_decision_helper_s_answers_arrive_as_its_notice(tmp_path: Path) -> None:
    desk = Gated()
    model = Crew(
        [background("is it urgent?", agent="desk"), text_reply("waiting"), text_reply("final")]
    )
    engine = team(crew(tmp_path, background=True), model, desk=desk)
    work = asyncio.create_task(engine.run("lead", "acme", "go"))
    run = await released(work, model, desk.gate)
    [child] = run.children
    assert isinstance(child, Decision) and child.run_id == run.steps[0].child_run_id
    body = child_result(child)
    assert json.loads(body)["answers"]["urgent"]["probability"] == 0.9
    assert run.conversation[-2] == notice("d1", "desk", child.run_id, body)
    assert run.reply == "final"


async def test_approve_refusing_a_background_delegate_starts_no_helper(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            background("one"),
            background("two", prefix="c"),
            text_reply("waiting"),
            text_reply("final"),
        ],
        {"one": [text_reply("never")], "two": [gate, text_reply("second")]},
    )

    async def approve(agent: str, call: ToolCall) -> bool:
        return str(call.args["input"]) != "one"

    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", approve=approve))
    run = await released(work, model, gate, calls=3)
    refused, started = run.steps
    assert (refused.result, refused.approved, refused.child_run_id) == (
        "NotApproved: Delegate was not approved",
        False,
        None,
    )
    assert model.calls["one"] == []
    assert started.result == '{"job": "d1"}' and started.approved is True
    [child] = run.children
    assert child.run_id == started.child_run_id


async def test_a_background_delegate_takes_no_delegate_slot(tmp_path: Path) -> None:
    slow = asyncio.Event()
    sync = ToolCall(id="s1", name="Delegate", args={"agent": "worker", "input": "slow"})
    fast = ToolCall(
        id="b1", name="Delegate", args={"agent": "worker", "input": "fast", "background": True}
    )
    model = Crew(
        [Reply(tool_calls=[sync, fast], stop_reason="tool_use"), text_reply("final")],
        {"slow": [slow, text_reply("slow done")], "fast": [text_reply("fast done")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True, max_parallel=1), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(ends(seen, 1), "the helper's end")
    assert len(model.calls["slow"]) == 1 and not work.done()
    assert len(model.calls["fast"]) == 1
    slow.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "final"
    assert [step.result for step in run.steps] == ["slow done", '{"job": "d1"}']


async def test_a_helper_runs_with_the_run_s_tools_tags_and_handlers(tmp_path: Path) -> None:
    gate = asyncio.Event()
    calls: list[Any] = []

    async def lookup(args: dict[str, Any]) -> str:
        calls.append(current_call())
        return "found"

    asked: list[tuple[str, str]] = []

    async def approve(agent: str, call: ToolCall) -> bool:
        asked.append((agent, call.name))
        return True

    use = Reply(tool_calls=[ToolCall(id="l1", name="lookup", args={})], stop_reason="tool_use")
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("final")],
        {"part one": [gate, use, text_reply("the answer")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, worker={"tools": {"allow": ["lookup"]}}, background=True), model)
    work = asyncio.create_task(
        engine.run(
            "lead",
            "acme",
            "go",
            tools=Tools([Tool("lookup", "Look a thing up", body=lookup)]),
            tags={"ticket": "t-1"},
            approve=approve,
            on_event=seen,
        )
    )
    run = await released(work, model, gate)
    [child] = run.children
    assert isinstance(child, Generation) and child.tags == {"ticket": "t-1"}
    assert child.scope == "acme" and child.steps[0].result == "found"
    [call] = calls
    assert (call.run_id, call.scope, call.tags) == (child.run_id, "acme", {"ticket": "t-1"})
    assert asked == [("lead", "Delegate"), ("worker", "lookup")]
    [started] = [e for e in seen.events if isinstance(e, RunStarted) and e.run_id == child.run_id]
    assert started.call is not None and started.call.run_id == run.run_id
    assert (started.call.tool, started.call.call_id) == ("Delegate", "b1")
    assert child.call == started.call


@pytest.mark.parametrize("press", ["now", "after_step"])
async def test_a_message_a_held_lead_took_stays_in_its_conversation_when_a_stop_ends_it(
    tmp_path: Path, press: str
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("never")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen, inbox, stop = Seen(), asyncio.Queue[Any](), Stop()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", on_event=seen, inbox=inbox, stop=stop)
    )
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    inbox.put_nowait("wrap up")
    inbox.put_nowait("and this")
    getattr(stop, press)()
    if press == "after_step":
        gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == press and run.calls == 2
    helper_id = str(run.steps[0].child_run_id)
    told = (
        notice("d1", "worker", helper_id, "", kind="stopped")
        if press == "now"
        else notice("d1", "worker", helper_id, "the answer")
    )
    wrap = Message(role="user", content="wrap up")
    assert run.conversation[-3:] == [Message(role="assistant", content="waiting"), told, wrap]
    [event] = taken(seen, run.run_id)
    assert event.notices == [told] and event.messages == [wrap] and event.turn == 2
    assert inbox.get_nowait() == "and this" and inbox.empty()
    assert ended(seen) == [helper_id, run.run_id]


async def test_a_held_reply_on_the_last_allowed_call_raises_at_once(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [halfway(), gate, text_reply("the answer")]},
    )
    seen, kept = Seen(), Kept()
    engine = team(crew(tmp_path, background=True), model, max_model_calls=2, refinery=kept)
    with pytest.raises(RunawayError, match=r"^engine: run exceeded 2 model calls$"):
        await asyncio.wait_for(engine.run("lead", "acme", "go", on_event=seen), 2)
    assert len(model.calls["lead"]) == 2 and not gate.is_set()
    [failure] = kept.runs
    assert isinstance(failure, Generation)
    [child] = failure.children
    assert isinstance(child, Generation) and child.stopped == "now"
    assert failure.conversation[-1] == notice(
        "d1", "worker", child.run_id, child.reply, kind="stopped"
    )
    assert ended(seen) == [child.run_id, failure.run_id]


async def test_a_stored_message_holds_a_notice_key_only_when_it_has_one(
    tmp_path: Path, storage: Storage
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("final")],
        {"part one": [gate, text_reply("the answer")]},
    )
    refinery = Refinery(storage)
    engine = team(crew(tmp_path, background=True), model, refinery=refinery)
    work = asyncio.create_task(engine.run("lead", "acme", "go"))
    run = await released(work, model, gate)
    row = await storage.get_run("acme", run.run_id)
    assert row is not None
    stored = row.body["conversation"]
    assert stored[0] == {
        "role": "user",
        "content": "go",
        "tool_calls": [],
        "tool_call_id": None,
        "reasoning": None,
        "is_error": False,
        "prefix": None,
        "usage": None,
        "model": None,
        "settings": None,
    }
    assert [m["notice"] for m in stored if "notice" in m] == [
        {
            "job": "d1",
            "agent": "worker",
            "run_id": run.steps[0].child_run_id,
            "kind": "finished",
        }
    ]
    read = await refinery.run("acme", run.run_id)
    assert isinstance(read, Generation) and read.conversation == run.conversation


@pytest.mark.parametrize(("start", "job"), [(0, "d5"), (5, "d3")])
async def test_job_ids_continue_after_the_window_the_run_reads(
    tmp_path: Path, start: int, job: str
) -> None:
    def delegated(call_id: str, held: str) -> list[Message]:
        call = ToolCall(id=call_id, name="Delegate", args={"agent": "worker", "input": "a"})
        return [
            Message(role="assistant", tool_calls=[call]),
            Message(role="tool", tool_call_id=call_id, content=json.dumps({"job": held})),
            Message(role="assistant", content="ok"),
        ]

    history = [
        Message(role="user", content="earlier"),
        *delegated("x1", "d4"),
        Message(role="assistant", content="more"),
        Message(role="user", content="later"),
        *delegated("x2", "d2"),
    ]
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("final")],
        {"part one": [gate, text_reply("the answer")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", history=history, start=start))
    run = await released(work, model, gate)
    assert run.steps[0].result == json.dumps({"job": job})


async def test_a_decision_helper_cut_by_its_id_sends_a_stopped_notice_and_leaves_no_record(
    tmp_path: Path,
) -> None:
    desk = Gated()
    model = Crew(
        [background("is it urgent?", agent="desk"), text_reply("waiting"), text_reply("final")]
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model, desk=desk)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    [helper_id] = [e.step.child_run_id for e in seen.events if isinstance(e, ToolResultReady)]
    assert helper_id is not None
    engine.stop(helper_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == "final" and run.children == []
    assert run.conversation[-2] == Message(
        role="user",
        content="[helper d1 · desk · stopped]\n",
        notice=Notice(job="d1", agent="desk", run_id=helper_id, kind="stopped"),
    )


async def test_a_decision_helper_stopped_before_its_call_leaves_nothing_on_the_given_stop(
    tmp_path: Path,
) -> None:
    desk = Gated()
    model = Crew(
        [background("is it urgent?", agent="desk"), text_reply("final"), text_reply("final")]
    )
    engine = team(crew(tmp_path, background=True), model, desk=desk)
    given = Stop()

    async def on_event(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.step.child_run_id is not None:
            engine.stop(event.step.child_run_id, scope="acme")

    run = await asyncio.wait_for(engine.run("lead", "acme", "go", on_event=on_event, stop=given), 5)
    assert isinstance(run, Generation) and run.children == [] and not desk.asked.is_set()
    assert given._children == set()


def broken(model: str | None, effort: str | None, /) -> Any:
    raise RuntimeError("no key for solo")


@pytest.mark.parametrize(
    ("worker", "engine", "error"),
    [
        pytest.param(
            {},
            {"max_depth": 1},
            "DelegationDepthError: tools/delegate: depth 1 exceeded at lead -> worker",
            id="too-deep",
        ),
        pytest.param({"model": "solo"}, {}, "RuntimeError: no key for solo", id="provider-raises"),
    ],
)
async def test_a_helper_that_fails_before_any_record_sends_a_failed_notice(
    tmp_path: Path, worker: dict[str, Any], engine: dict[str, Any], error: str
) -> None:
    model = Crew([background("part one"), text_reply("noted")])
    root = crew(tmp_path, worker=worker, background=True)
    providers = {**provided({"crew": model, "judge": Answering()}), "solo": broken}
    seen = Seen()
    run = await asyncio.wait_for(
        Engine(root, providers=providers, **engine).run("lead", "acme", "go", on_event=seen), 5
    )
    assert isinstance(run, Generation) and run.children == [] and run.reply == "noted"
    helper_id = run.steps[0].child_run_id
    assert helper_id is not None and ended(seen) == [run.run_id]
    assert model.calls["lead"][1][-1].content == f"[helper d1 · worker · failed]\n{error}"
    assert run.conversation[-2].notice == Notice(
        job="d1", agent="worker", run_id=helper_id, kind="failed"
    )


async def test_a_delegate_that_wins_its_slot_after_a_stop_starts_no_child(
    tmp_path: Path,
) -> None:
    calls = [
        ToolCall(id=f"s{n}", name="Delegate", args={"agent": "worker", "input": text})
        for n, text in enumerate(["a", "b"], 1)
    ]
    model = Crew(
        [Reply(tool_calls=calls, stop_reason="tool_use")],
        {"a": [text_reply("a done")], "b": [text_reply("never")]},
    )
    stop = Stop()

    async def on_event(event: Any) -> None:
        if isinstance(event, RunEnded) and event.agent == "worker":
            stop.now()

    engine = team(crew(tmp_path, max_parallel=1), model)
    run = await engine.run("lead", "acme", "go", stop=stop, on_event=on_event)
    assert isinstance(run, Generation) and run.stopped == "now"
    assert [(s.result, s.stopped, s.child_run_id is None) for s in run.steps] == [
        ("a done", False, False),
        ("stopped by the user", True, True),
    ]
    assert len(run.children) == 1 and model.calls["b"] == []


async def test_job_ids_continue_after_a_notice_in_the_window(tmp_path: Path) -> None:
    call = ToolCall(
        id="x1", name="Delegate", args={"agent": "worker", "input": "a", "background": True}
    )
    old = Notice(job="d1", agent="worker", run_id="r-old", kind="finished")
    history = [
        Message(role="user", content="earlier"),
        Message(role="assistant", tool_calls=[call]),
        Message(role="tool", tool_call_id="x1", content='{"job": "d1"}'),
        Message(role="assistant", content="waiting"),
        Message(role="user", content="[helper d1 · worker · finished]\nold", notice=old),
        Message(role="assistant", content="ok"),
    ]
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("final")],
        {"part one": [gate, text_reply("the answer")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", history=history, start=4))
    run = await released(work, model, gate)
    assert run.steps[0].result == '{"job": "d2"}'
    assert model.calls["lead"][2][-1].content == "[helper d2 · worker · finished]\nthe answer"


async def test_a_held_reply_on_the_last_allowed_call_ends_stopped_when_its_stop_is_pending(
    tmp_path: Path,
) -> None:
    gate, lead_gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("part one"), lead_gate, text_reply("waiting")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen, stop = Seen(), Stop()
    engine = team(crew(tmp_path, background=True), model, max_model_calls=2)
    work = asyncio.create_task(engine.run("lead", "acme", "go", stop=stop, on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    stop.after_step()
    lead_gate.set()
    gate.set()
    run = await asyncio.wait_for(work, 2)
    assert isinstance(run, Generation)
    assert (run.stopped, run.calls, run.reply) == ("after_step", 2, "waiting")
    helper_id = str(run.steps[0].child_run_id)
    assert run.conversation[-1] == notice("d1", "worker", helper_id, "the answer")
    assert ended(seen) == [helper_id, run.run_id]


@pytest.mark.parametrize("where", ["held", "in-a-model-call"])
async def test_the_lead_s_stop_stops_its_helpers_and_keeps_their_outcomes(
    tmp_path: Path, where: str
) -> None:
    lead_gate, gate = asyncio.Event(), asyncio.Event()
    second = text_reply("waiting") if where == "held" else lead_gate
    model = Crew(
        [background("part one"), second, text_reply("never")],
        {"part one": [halfway(), gate, text_reply("never")]},
    )
    seen, stop, inbox = Seen(), Stop(), Watched()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", stop=stop, on_event=seen, inbox=inbox)
    )
    await until(
        lambda: (
            len(model.calls["part one"]) == 2
            and (inbox.waiting == 1 if where == "held" else len(model.calls["lead"]) == 2)
        ),
        "the lead's wait",
    )
    stop.now()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "now"
    assert len(model.calls["lead"]) == 2 and not gate.is_set()
    [child] = run.children
    assert isinstance(child, Generation) and child.run_id == run.steps[0].child_run_id
    assert (child.stopped, child.reply) == ("now", "half way")
    told = notice("d1", "worker", child.run_id, "half way", kind="stopped")
    assert run.conversation[-1] == told
    [event] = taken(seen, run.run_id)
    assert event.notices == [told] and event.messages == [] and event.turn == 2
    assert ended(seen) == [child.run_id, run.run_id]


async def test_the_lead_s_stop_cuts_a_decision_helper_s_call(tmp_path: Path) -> None:
    desk = Gated()
    model = Crew([background("is it urgent?", agent="desk"), text_reply("waiting")])
    seen, stop, inbox = Seen(), Stop(), Watched()
    engine = team(crew(tmp_path, background=True), model, desk=desk)
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", stop=stop, on_event=seen, inbox=inbox)
    )
    await until(lambda: inbox.waiting == 1, "the lead's hold")
    stop.now()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "now" and run.calls == 2
    assert run.children == [] and not desk.gate.is_set()
    helper_id = str(run.steps[0].child_run_id)
    assert run.conversation[-1] == notice("d1", "desk", helper_id, "", kind="stopped")
    assert ended(seen) == [run.run_id]


async def test_after_step_lets_each_helper_finish_its_step(tmp_path: Path) -> None:
    looking, found = asyncio.Event(), asyncio.Event()

    async def lookup(args: dict[str, Any]) -> str:
        looking.set()
        await found.wait()
        return "found"

    use = Reply(
        text="looking it up",
        tool_calls=[ToolCall(id="l1", name="lookup", args={})],
        stop_reason="tool_use",
    )
    both = Reply(
        tool_calls=[
            *background("part one").tool_calls,
            *background("is it urgent?", agent="desk", prefix="c").tool_calls,
        ],
        stop_reason="tool_use",
    )
    desk = Gated()
    model = Crew([both, text_reply("waiting")], {"part one": [use]})
    seen, stop, inbox = Seen(), Stop(), Watched()
    root = crew(tmp_path, worker={"tools": {"allow": ["lookup"]}}, background=True)
    engine = team(root, model, desk=desk)
    tools = Tools([Tool("lookup", "Look a thing up", body=lookup)])
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", tools=tools, stop=stop, on_event=seen, inbox=inbox)
    )
    await until(lambda: inbox.waiting == 1 and looking.is_set(), "the lead's hold")
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the held lead's wake-up")
    found.set()
    await until(lambda: bool(ended(seen)), "the worker's end")
    desk.gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "after_step" and run.calls == 2
    worker, decided = run.children
    assert isinstance(worker, Generation) and isinstance(decided, Decision)
    assert (worker.stopped, worker.calls) == ("after_step", 1)
    assert [(step.result, step.stopped) for step in worker.steps] == [("found", False)]
    assert json.loads(child_result(decided))["answers"]["urgent"]["probability"] == 0.9
    assert run.conversation[-2:] == [
        notice("d1", "worker", worker.run_id, "looking it up", kind="stopped"),
        notice("d2", "desk", decided.run_id, child_result(decided)),
    ]
    [event] = taken(seen, run.run_id)
    assert event.notices == run.conversation[-2:] and event.messages == []
    assert ended(seen) == [worker.run_id, run.run_id]


async def test_after_step_wakes_a_held_lead_which_reads_no_more_of_its_inbox(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen, stop, inbox = Seen(), Stop(), Watched()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", stop=stop, on_event=seen, inbox=inbox)
    )
    await until(lambda: inbox.waiting == 1, "the lead's hold")
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the held lead's wake-up")
    inbox.put_nowait("late")
    gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and (run.stopped, run.calls) == ("after_step", 2)
    helper_id = str(run.steps[0].child_run_id)
    assert run.conversation[-1] == notice("d1", "worker", helper_id, "the answer")
    [event] = taken(seen, run.run_id)
    assert event.messages == [] and inbox.get_nowait() == "late"


async def test_a_failing_lead_stops_its_helpers_and_records_them(tmp_path: Path) -> None:
    lead_gate, gate = asyncio.Event(), asyncio.Event()
    ghost = Reply(tool_calls=[ToolCall(id="g1", name="ghost", args={})], stop_reason="tool_use")
    model = Crew(
        [background("part one"), lead_gate, ghost],
        {"part one": [halfway(), gate, text_reply("never")]},
    )
    seen, kept = Seen(), Kept()
    engine = team(crew(tmp_path, background=True), model, max_model_calls=2, refinery=kept)
    before = asyncio.all_tasks()
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen))
    await until(lambda: len(model.calls["part one"]) == 2, "the helper's second call")
    lead_gate.set()
    with pytest.raises(RunawayError, match=r"^engine: run exceeded 2 model calls$"):
        await asyncio.wait_for(work, 5)
    assert pending(before) == [] and not gate.is_set()
    [failure] = kept.runs
    assert isinstance(failure, Generation)
    assert failure.error == "RunawayError: engine: run exceeded 2 model calls"
    [child] = failure.children
    assert isinstance(child, Generation) and (child.stopped, child.reply) == ("now", "half way")
    assert failure.conversation[-1] == notice(
        "d1", "worker", child.run_id, "half way", kind="stopped"
    )
    assert ended(seen) == [child.run_id, failure.run_id]


async def test_cancelling_the_lead_leaves_no_helper_running(tmp_path: Path) -> None:
    deep, other = asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("part one", "part two"), text_reply("waiting")],
        {
            "part one": [background("deep"), text_reply("waiting below")],
            "part two": [halfway(), other, text_reply("never")],
            "deep": [deep, text_reply("never")],
        },
    )
    seen, inbox = Seen(), Watched()
    worker = {"orchestration": {"delegate_to": ["worker"], "background": True}}
    engine = team(crew(tmp_path, worker=worker, background=True), model)
    before = asyncio.all_tasks()
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen, inbox=inbox))
    await until(
        lambda: (
            inbox.waiting == 1
            and len(model.calls["part one"]) == 2
            and len(model.calls["part two"]) == 2
            and len(model.calls["deep"]) == 1
        ),
        "every run's wait",
    )
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(work, 5)
    assert pending(before) == []
    assert ended(seen) == [] and not deep.is_set() and not other.is_set()


async def test_a_message_that_wakes_a_lead_as_it_is_cancelled_stays_on_its_inbox(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [gate, text_reply("never")]},
    )
    inbox = Watched()
    engine = team(crew(tmp_path, background=True), model)
    before = asyncio.all_tasks()
    work = asyncio.create_task(engine.run("lead", "acme", "go", inbox=inbox))
    await until(lambda: inbox.waiting == 1, "the lead's hold")
    inbox.put_nowait("wrap up")
    inbox.put_nowait("and this")
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(work, 5)
    assert pending(before) == []
    assert [inbox.get_nowait() for _ in range(inbox.qsize())] == ["wrap up", "and this"]
    inbox.task_done()
    inbox.task_done()
    await asyncio.wait_for(inbox.join(), 1)


@pytest.mark.parametrize(
    ("worker", "error"),
    [
        pytest.param(
            {"max_model_calls": 1},
            "RunawayError: engine: run exceeded 1 model calls",
            id="runs-away",
        ),
        pytest.param({"model": "down"}, "ModelError: worker: the model is down", id="raises"),
    ],
)
async def test_a_failing_helper_reaches_the_lead_as_a_failed_notice(
    tmp_path: Path, worker: dict[str, Any], error: str
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("noted")],
        {"part one": [gate, halfway()]},
    )
    root = crew(tmp_path, worker=worker, background=True)
    providers = provided({"crew": model, "judge": Answering(), "down": Failing(gate)})
    seen = Seen()
    work = asyncio.create_task(
        Engine(root, providers=providers).run("lead", "acme", "go", on_event=seen)
    )
    run = await released(work, model, gate)
    assert run.reply == "noted" and run.error is None and run.calls == 3
    [child] = run.children
    assert isinstance(child, Generation) and child.error == error
    expected = notice("d1", "worker", child.run_id, error, kind="failed")
    assert run.conversation[-2] == expected
    assert model.calls["lead"][2][-1].content == expected.content
    assert ended(seen) == [child.run_id, run.run_id]


async def test_output_while_helpers_run_is_refused(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), output("early"), text_reply("waiting"), output("final", "o2")],
        {"part one": [gate, text_reply("the answer")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", expect=Answer))
    run = await released(work, model, gate, calls=3)
    assert isinstance(run, Generation) and run.reply == '{"text":"final"}' and run.calls == 4
    early = run.steps[1]
    assert (early.tool, early.args, early.result, early.is_error) == (
        "Output",
        {"text": "early"},
        STILL_RUNNING,
        True,
    )
    assert model.calls["lead"][2][-1] == Message(
        role="tool", tool_call_id="o1", content=STILL_RUNNING, is_error=True
    )
    helper_id = str(run.steps[0].child_run_id)
    assert run.conversation[-3] == notice("d1", "worker", helper_id, "the answer")
    assert run.conversation[-1] == Message(role="tool", tool_call_id="o2", content="accepted")


async def test_an_accepted_output_appends_the_notices_that_wait(tmp_path: Path) -> None:
    lead_gate, gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("part one"), lead_gate, output("final")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", expect=Answer, on_event=seen))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gate.set()
    await until(ends(seen, 1), "the helper's end")
    lead_gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.reply == '{"text":"final"}' and run.calls == 2
    helper_id = str(run.steps[0].child_run_id)
    expected = notice("d1", "worker", helper_id, "the answer")
    assert run.conversation[-2:] == [
        Message(role="tool", tool_call_id="o1", content="accepted"),
        expected,
    ]
    [event] = taken(seen, run.run_id)
    assert event.notices == [expected] and event.messages == [] and event.turn == 2
    assert ended(seen) == [helper_id, run.run_id]


async def test_a_stop_that_cuts_an_output_stops_the_helpers_too(tmp_path: Path) -> None:
    gate = asyncio.Event()
    both = Reply(
        tool_calls=[
            *background("part one").tool_calls,
            ToolCall(id="o1", name="Output", args={"text": "x"}),
        ],
        stop_reason="tool_use",
    )
    model = Crew([both], {"part one": [gate, text_reply("never")]})
    seen, stop = Seen(), Stop()

    async def on_event(event: Any) -> None:
        await seen(event)
        if isinstance(event, ToolResultReady) and event.call_id == "b1":
            stop.now()

    engine = team(crew(tmp_path, background=True), model)
    run = await asyncio.wait_for(
        engine.run("lead", "acme", "go", expect=Answer, stop=stop, on_event=on_event), 5
    )
    assert isinstance(run, Generation) and run.stopped == "now"
    cut = run.steps[1]
    assert (cut.tool, cut.result, cut.stopped) == ("Output", "stopped by the user", True)
    [child] = run.children
    assert isinstance(child, Generation) and child.stopped == "now"
    assert run.conversation[-1] == notice("d1", "worker", child.run_id, "", kind="stopped")
    assert ended(seen) == [child.run_id, run.run_id]


async def test_a_stop_pressed_on_taken_notices_still_keeps_every_helper_s_outcome(
    tmp_path: Path,
) -> None:
    first, second = asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("one", "two"), text_reply("waiting"), text_reply("never")],
        {"one": [first, text_reply("first")], "two": [second, text_reply("never")]},
    )
    seen, stop = Seen(), Stop()

    async def on_event(event: Any) -> None:
        await seen(event)
        if isinstance(event, MessagesTaken) and event.agent == "lead":
            stop.now()

    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", stop=stop, on_event=on_event))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    first.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and (run.stopped, run.calls) == ("now", 2)
    one, two = (str(step.child_run_id) for step in run.steps)
    assert run.conversation[-2:] == [
        notice("d1", "worker", one, "first"),
        notice("d2", "worker", two, "", kind="stopped"),
    ]
    assert [[m.notice.job for m in e.notices if m.notice] for e in taken(seen, run.run_id)] == [
        ["d1"],
        ["d2"],
    ]
    assert ended(seen) == [one, two, run.run_id] and not second.is_set()


INTENT = '{"type": "choice", "instructions": "What is this?", "options": {"talk": "Just talk", "other": "Anything else"}}'


class Talking:
    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
        return {"intent": ChoiceAnswer(choice="talk", probabilities={"talk": 0.9, "other": 0.1})}


@pytest.mark.parametrize("press", ["the-lead-s-stop", "its-id"])
async def test_a_decision_helper_whose_route_is_cut_ends_stopped(
    tmp_path: Path, press: str
) -> None:
    root = write_folder(
        tmp_path,
        {
            "lead": {
                "kind": "generation",
                "model": "crew",
                "description": "Leads the crew",
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["switch"], "background": True},
            },
            "switch": {
                "kind": "decision",
                "model": "switch",
                "description": "Routes to the desk",
                "instructions": "desk.md",
                "questions": ["switch/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"talk": "desk"}}},
            },
            "desk": {
                "kind": "decision",
                "model": "judge",
                "description": "Sorts",
                "instructions": "desk.md",
                "questions": ["desk/*.json"],
            },
        },
        files={
            "lead.md": "Lead.\n",
            "desk.md": "",
            "switch/intent.json": INTENT,
            "desk/urgent.json": URGENT,
        },
    )
    desk = Gated()
    model = Crew([background("hello", agent="switch"), text_reply("waiting"), text_reply("final")])
    providers = provided({"crew": model, "switch": Talking(), "judge": desk})
    seen, stop, inbox = Seen(), Stop(), Watched()
    engine = Engine(root, providers=providers)
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", stop=stop, on_event=seen, inbox=inbox)
    )
    await until(lambda: inbox.waiting == 1 and desk.asked.is_set(), "the routed call")
    if press == "its-id":
        [helper_id] = [e.step.child_run_id for e in seen.events if isinstance(e, ToolResultReady)]
        assert helper_id is not None
        engine.stop(helper_id, scope="acme")
    else:
        stop.now()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and not desk.gate.is_set()
    [router] = run.children
    assert isinstance(router, Decision) and router.routed_to is None
    told = [m for m in run.conversation if m.notice is not None]
    assert told == [notice("d1", "switch", router.run_id, child_result(router), kind="stopped")]
    assert run.reply == ("final" if press == "its-id" else "waiting")


@pytest.mark.parametrize("kind", ["decision", "generation"])
@pytest.mark.parametrize("press", ["stop-now", "engine-stop"])
async def test_now_after_after_step_cuts_every_helper_still_running(
    tmp_path: Path, kind: str, press: str
) -> None:
    desk, gate = Gated(), asyncio.Event()
    if kind == "decision":
        model = Crew([background("is it urgent?", agent="desk"), text_reply("waiting")])
    else:
        model = Crew(
            [background("part one"), text_reply("waiting")],
            {"part one": [halfway(), gate, text_reply("never")]},
        )

    def working() -> bool:
        return desk.asked.is_set() if kind == "decision" else len(model.calls["part one"]) == 2

    seen, stop, inbox = Seen(), Stop(), Watched()
    engine = team(crew(tmp_path, background=True), model, desk=desk)
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", stop=stop, on_event=seen, inbox=inbox)
    )
    await until(lambda: inbox.waiting == 1 and working(), "the lead's hold")
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the held lead's wake-up")
    if press == "stop-now":
        stop.now()
    else:
        [lead_id] = seen.started(below=False)
        engine.stop(lead_id, scope="acme")
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "now"
    assert not desk.gate.is_set() and not gate.is_set()
    helper_id = str(run.steps[0].child_run_id)
    if kind == "decision":
        assert run.children == []
        assert run.conversation[-1] == notice("d1", "desk", helper_id, "", kind="stopped")
    else:
        [child] = run.children
        assert isinstance(child, Generation) and (child.stopped, child.reply) == ("now", "half way")
        assert run.conversation[-1] == notice("d1", "worker", helper_id, "half way", kind="stopped")
    [end] = [e for e in seen.events if isinstance(e, RunEnded) and e.run_id == run.run_id]
    assert end.stopped == "now"


@pytest.mark.parametrize("holding", [False, True], ids=["empty", "holding-a-message"])
async def test_an_inbox_shut_down_while_a_lead_waits_never_fails_it(
    tmp_path: Path, holding: bool
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [
            background("part one"),
            text_reply("waiting"),
            *([text_reply("still waiting")] if holding else []),
            text_reply("final"),
        ],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen, inbox = Seen(), Watched()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen, inbox=inbox))
    await until(lambda: inbox.waiting == 1, "the lead's hold")
    if holding:
        inbox.put_nowait("last word")
    inbox.shutdown()
    calls = 3 if holding else 2
    run = await released(work, model, gate, calls=calls)
    assert run.reply == "final" and run.error is None and inbox.empty()
    if holding:
        last = Message(role="user", content="last word")
        assert model.calls["lead"][2][-1] == last
        [first, _] = taken(seen, run.run_id)
        assert first.messages == [last] and first.notices == []
    await asyncio.wait_for(inbox.join(), 1)


async def test_a_lead_s_first_now_leaves_its_helpers_their_grace_and_a_second_ends_them(
    tmp_path: Path,
) -> None:
    tool = Stubborn()
    engine, model, tools = tool.team(tmp_path, [background("part one"), text_reply("waiting")])
    stop, inbox = Stop(), Watched()
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", tools=tools, stop=stop, inbox=inbox)
    )
    await until(lambda: inbox.waiting == 1 and tool.started.is_set(), "the lead's hold")
    stop.now()
    await until(lambda: inbox.waiting == 0 and tool.cancels >= 1, "the first press")
    assert tool.cancels == 1 and not work.done()
    stop.now()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "now" and tool.cancels == 2
    [child] = run.children
    assert isinstance(child, Generation) and child.stopped == "now"
    assert [step.result for step in child.steps] == ["stopped by the user"]


async def test_a_consumer_s_timeout_during_the_settle_surfaces_as_a_timeout_error(
    tmp_path: Path,
) -> None:
    tool = Stubborn()
    engine, model, tools = tool.team(tmp_path, [background("part one"), text_reply("waiting")])
    stop, inbox = Stop(), Watched()
    deadline: list[asyncio.Timeout] = []
    before = asyncio.all_tasks()

    async def consumer() -> tuple[str, int]:
        raised = "nothing"
        try:
            async with asyncio.timeout(None) as held:
                deadline.append(held)
                await engine.run("lead", "acme", "go", tools=tools, stop=stop, inbox=inbox)
        except TimeoutError:
            raised = "TimeoutError"
        except asyncio.CancelledError:
            raised = "CancelledError"
        task = asyncio.current_task()
        return raised, -1 if task is None else task.cancelling()

    work = asyncio.create_task(consumer())
    await until(lambda: inbox.waiting == 1 and tool.started.is_set(), "the lead's hold")
    stop.now()
    await until(lambda: inbox.waiting == 0 and tool.cancels == 1, "the lead's settle")
    deadline[0].reschedule(asyncio.get_running_loop().time())
    assert await asyncio.wait_for(work, 5) == ("TimeoutError", 0)
    assert tool.cancels == 2 and pending(before) == []


class Recorded:
    def __init__(self) -> None:
        self.waits: list[list[bool]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def wait(self, futures: Any, *args: Any, **kwargs: Any) -> Any:
        self.waits.append([future.done() for future in futures])
        return await asyncio.wait(futures, *args, **kwargs)


def left(inbox: asyncio.Queue[Any]) -> list[Any]:
    return [inbox.get_nowait() for _ in range(inbox.qsize())]


def users(run: Generation) -> list[Any]:
    return [m.content for m in run.conversation[1:] if m.role == "user" and m.notice is None]


async def test_an_after_step_settle_waits_without_spinning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = Recorded()
    monkeypatch.setattr(rfnry_engine.helpers, "asyncio", recorded)
    one, two = asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("part one", "part two"), text_reply("waiting")],
        {"part one": [one, text_reply("one")], "part two": [two, text_reply("two")]},
    )
    seen, stop, inbox = Seen(), Stop(), Watched()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", stop=stop, on_event=seen, inbox=inbox)
    )
    await until(lambda: inbox.waiting == 1 and len(model.calls["part two"]) == 1, "the lead's hold")
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the held lead's wake-up")
    one.set()
    await until(ends(seen, 1), "d1's end")
    two.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "after_step"
    assert recorded.waits == [[False, False, False], [False, False]]


@pytest.mark.parametrize(
    ("queue", "waker", "later"),
    [
        pytest.param(Watched, "a", "z", id="fifo"),
        pytest.param(WatchedLifo, "a", "z", id="lifo"),
        pytest.param(WatchedPriority, "b", "a", id="priority"),
    ],
)
async def test_the_message_that_wakes_a_lead_is_the_run_s_and_the_rest_stays_queued(
    tmp_path: Path, queue: type[Watched], waker: str, later: str
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen, stop, inbox = Seen(), Stop(), queue()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(
        engine.run("lead", "acme", "go", stop=stop, on_event=seen, inbox=inbox)
    )
    await until(lambda: inbox.waiting == 1, "the lead's hold")
    inbox.put_nowait(waker)
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the held lead's wake-up")
    inbox.put_nowait(later)
    gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "after_step"
    assert users(run) == [waker] and left(inbox) == [later]
    [event] = taken(seen, run.run_id)
    assert event.messages == [Message(role="user", content=waker)]


@pytest.mark.parametrize(
    ("queue", "joined", "rest"),
    [
        pytest.param(Watched, "a", "b", id="fifo"),
        pytest.param(WatchedLifo, "b", "a", id="lifo"),
    ],
)
async def test_a_stop_exit_on_a_shut_down_inbox_takes_only_the_message_that_woke_the_lead(
    tmp_path: Path, queue: type[Watched], joined: str, rest: str
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [gate, text_reply("the answer")]},
    )
    stop, inbox = Stop(), queue()
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", stop=stop, inbox=inbox))
    await until(lambda: inbox.waiting == 1, "the lead's hold")
    inbox.put_nowait("a")
    inbox.put_nowait("b")
    inbox.shutdown()
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the held lead's wake-up")
    gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == "after_step"
    assert users(run) == [joined] and left(inbox) == [rest]


@pytest.mark.parametrize(
    ("queue", "back"),
    [
        pytest.param(Watched, ["z", "a"], id="fifo"),
        pytest.param(WatchedLifo, ["a", "z"], id="lifo"),
        pytest.param(WatchedPriority, ["a", "z"], id="priority"),
    ],
)
async def test_a_cancelled_lead_puts_the_message_that_woke_it_back_as_a_new_one(
    tmp_path: Path, queue: type[Watched], back: list[str]
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [gate, text_reply("never")]},
    )
    stop, inbox = Stop(), queue()
    engine = team(crew(tmp_path, background=True), model)
    before = asyncio.all_tasks()
    work = asyncio.create_task(engine.run("lead", "acme", "go", stop=stop, inbox=inbox))
    await until(lambda: inbox.waiting == 1, "the lead's hold")
    inbox.put_nowait("a")
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the held lead's wake-up")
    inbox.put_nowait("z")
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(work, 5)
    assert pending(before) == [] and left(inbox) == back
    for _ in back:
        inbox.task_done()
    await asyncio.wait_for(inbox.join(), 1)


@pytest.mark.parametrize("end", ["shut-down", "full"])
async def test_a_cancelled_lead_s_message_goes_with_it_when_its_inbox_takes_no_more(
    tmp_path: Path, end: str
) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [gate, text_reply("never")]},
    )
    stop, inbox = Stop(), Watched(maxsize=1)
    engine = team(crew(tmp_path, background=True), model)
    before = asyncio.all_tasks()
    work = asyncio.create_task(engine.run("lead", "acme", "go", stop=stop, inbox=inbox))
    await until(lambda: inbox.waiting == 1, "the lead's hold")
    inbox.put_nowait("a")
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the held lead's wake-up")
    if end == "shut-down":
        inbox.shutdown()
    else:
        inbox.put_nowait("z")
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(work, 5)
    assert pending(before) == [] and left(inbox) == ([] if end == "shut-down" else ["z"])
    if end == "full":
        inbox.task_done()
    await asyncio.wait_for(inbox.join(), 1)


async def test_a_put_back_that_raises_still_cancels_the_helpers(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [gate, text_reply("never")]},
    )
    stop, inbox = Stop(), WatchedPriority()
    engine = team(crew(tmp_path, background=True), model)
    before = asyncio.all_tasks()
    work = asyncio.create_task(engine.run("lead", "acme", "go", stop=stop, inbox=inbox))
    await until(lambda: inbox.waiting == 1, "the lead's hold")
    inbox.put_nowait({"text": "first"})
    stop.after_step()
    await until(lambda: inbox.waiting == 0, "the held lead's wake-up")
    inbox.put_nowait({"text": "second"})
    work.cancel()
    with pytest.raises(TypeError, match="not supported between instances of 'dict' and 'dict'"):
        await asyncio.wait_for(work, 5)
    assert pending(before) == [] and not gate.is_set()


async def test_an_inbox_item_the_run_cannot_read_keeps_the_notices_taken_with_it(
    tmp_path: Path,
) -> None:
    lead_gate, gate = asyncio.Event(), asyncio.Event()
    model = Crew(
        [background("part one"), lead_gate, text_reply("waiting"), text_reply("never")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen, kept = Seen(), Kept()
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    engine = team(crew(tmp_path, background=True), model, refinery=kept)
    work = asyncio.create_task(engine.run("lead", "acme", "go", on_event=seen, inbox=inbox))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gate.set()
    await until(ends(seen, 1), "the helper's end")
    inbox.put_nowait({1, 2})
    lead_gate.set()
    with pytest.raises(TypeError, match="^Object of type set is not JSON serializable$"):
        await asyncio.wait_for(work, 5)
    [failure] = kept.runs
    assert isinstance(failure, Generation) and failure.error is not None
    [child] = failure.children
    assert failure.conversation[-1] == notice("d1", "worker", child.run_id, "the answer")
    assert len(model.calls["lead"]) == 2 and taken(seen, failure.run_id) == []


async def test_a_helper_is_reached_by_its_id_from_the_step_that_starts_it(tmp_path: Path) -> None:
    model = Crew(
        [background("part one", "part two"), *[text_reply("waiting")] * 3],
        {"part one": [halfway(), text_reply("one done")], "part two": [text_reply("never")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    reached: list[str] = []

    async def on_event(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.step.child_run_id is not None:
            job = json.loads(str(event.step.result))["job"]
            if job == "d1":
                await engine.send(event.step.child_run_id, "hello", scope="acme")
            else:
                engine.stop(event.step.child_run_id, scope="acme")
            reached.append(job)

    run = await asyncio.wait_for(engine.run("lead", "acme", "go", on_event=on_event), 5)
    assert isinstance(run, Generation) and reached == ["d1", "d2"]
    by_id = {child.run_id: child for child in run.children}
    one, two = (by_id[str(step.child_run_id)] for step in run.steps)
    assert isinstance(one, Generation) and one.reply == "one done"
    assert Message(role="user", content="hello") in model.calls["part one"][1]
    assert isinstance(two, Generation) and (two.stopped, two.calls) == ("now", 0)
    assert model.calls["part two"] == []
    assert engine._running == {}


async def test_a_decision_helper_stopped_from_the_step_that_starts_it_never_asks(
    tmp_path: Path,
) -> None:
    desk = Gated()
    model = Crew(
        [background("is it urgent?", agent="desk"), text_reply("final"), text_reply("final")]
    )
    engine = team(crew(tmp_path, background=True), model, desk=desk)

    async def on_event(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.step.child_run_id is not None:
            engine.stop(event.step.child_run_id, scope="acme")

    run = await asyncio.wait_for(engine.run("lead", "acme", "go", on_event=on_event), 5)
    assert isinstance(run, Generation) and run.reply == "final" and run.children == []
    helper_id = str(run.steps[0].child_run_id)
    assert run.conversation[-2] == Message(
        role="user",
        content="[helper d1 · desk · stopped]\n",
        notice=Notice(job="d1", agent="desk", run_id=helper_id, kind="stopped"),
    )
    assert not desk.asked.is_set() and engine._running == {}
