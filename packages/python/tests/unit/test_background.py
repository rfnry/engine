import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import (
    CheckAnswer,
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
from rfnry_engine.errors import InputError, RunawayError
from rfnry_engine.models import Message, QuestionLike, Reply, State, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.tools.delegate import child_result
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Answering, Crew, text_reply
from .test_reach import Seen, until
from .trees import provided, write_folder

URGENT = '{"type": "check", "instructions": "Urgent?"}'
AGENT_0_1_13 = "34102ac5a160ec02"
TOOLS_0_1_13 = "1df013bbe33bc7a6"
PREFIX_0_1_13 = "cf5a7c0b8c982083"
DESCRIPTION_0_1_13 = (
    "Hand an input to another agent and get its reply.\n"
    "- worker: Works on one part\n"
    "- desk (decision agent): Sorts"
)
URGENT_QUESTION = {"type": "check", "instructions": "Urgent?"}
REFUSED = "InputError: tools/delegate: agent 'lead' may not delegate in the background"
PAST_CAP = "InputError: tools/delegate: max_background is 1, and running now: d1; wait for a helper to end or stop one"


def crew(root: Path, worker: Mapping[str, Any] | None = None, **orchestration: Any) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return write_folder(
        root,
        {
            "lead": {
                "kind": "generation",
                "model": "crew",
                "description": "Leads the crew",
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["worker", "desk"], **orchestration},
            },
            "worker": {
                "kind": "generation",
                "model": "crew",
                "description": "Works on one part",
                "instructions": "worker.md",
                **(worker or {}),
            },
            "desk": {
                "kind": "decision",
                "model": "judge",
                "description": "Sorts",
                "instructions": "desk.md",
                "questions": ["desk/*.json"],
            },
        },
        runtime_tools=["lookup"] if worker and "tools" in worker else (),
        files={
            "lead.md": "Lead.\n",
            "worker.md": "Work.\n",
            "desk.md": "",
            "desk/urgent.json": URGENT,
        },
    )


def team(root: Path, model: Crew, desk: Any = None, **engine: Any) -> Engine:
    return Engine(root, providers=provided({"crew": model, "judge": desk or Answering()}), **engine)


def background(*inputs: str, agent: str = "worker", prefix: str = "b") -> Reply:
    calls = [
        ToolCall(
            id=f"{prefix}{n}",
            name="Delegate",
            args={"agent": agent, "input": text, "background": True},
        )
        for n, text in enumerate(inputs, 1)
    ]
    return Reply(tool_calls=calls, stop_reason="tool_use")


def notice(job: str, agent: str, run_id: str, body: str) -> Message:
    return Message(
        role="user",
        content=f"[helper {job} · {agent} · finished]\n{body}",
        notice=Notice(job=job, agent=agent, run_id=run_id, kind="finished"),
    )


def said(messages: list[Message]) -> list[str]:
    return [str(m.content) for m in messages if m.role == "assistant" and not m.tool_calls]


def ended(seen: Seen) -> list[str]:
    return [e.run_id for e in seen.events if isinstance(e, RunEnded)]


def ends(seen: Seen, count: int) -> Callable[[], bool]:
    return lambda: len(ended(seen)) == count


def taken(seen: Seen, run_id: str) -> list[MessagesTaken]:
    return [e for e in seen.events if isinstance(e, MessagesTaken) and e.run_id == run_id]


async def released(
    work: asyncio.Task[Any], model: Crew, gate: asyncio.Event, calls: int = 2
) -> Generation:
    await until(lambda: len(model.calls["lead"]) == calls, f"the lead's call {calls}")
    assert not work.done()
    gate.set()
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation)
    return run


class Gated:
    def __init__(self) -> None:
        self.gate = asyncio.Event()

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> dict[str, Any]:
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
    [tool] = run.tools
    assert list(tool.input_schema["properties"]) == ["agent", "input", "questions", "background"]
    assert tool.input_schema["properties"]["background"]["type"] == "boolean"
    assert tool.input_schema["required"] == ["agent", "input"]
    assert tool.description.startswith("Hand an input to another agent and get its reply. ")
    assert tool.description.endswith(
        "\n- worker: Works on one part\n- desk (decision agent): Sorts"
    )
    assert run.fingerprint.tools != TOOLS_0_1_13


async def test_a_lead_that_does_not_opt_in_is_refused_a_background_delegate(
    tmp_path: Path,
) -> None:
    model = Crew([background("one"), text_reply("done")], {"one": [text_reply("never")]})
    run = await team(crew(tmp_path), model).run("lead", "acme", "go")
    assert isinstance(run, Generation) and run.reply == "done" and run.children == []
    step = run.steps[0]
    assert (step.result, step.is_error, step.child_run_id) == (REFUSED, True, None)
    assert model.calls["one"] == []


async def test_a_delegate_dispatched_from_code_has_no_background(tmp_path: Path) -> None:
    engine = team(crew(tmp_path, background=True), Crew([]))
    tools = EngineTools(engine, agent="lead", scope="acme")
    [delegate] = [d for d in tools.definitions() if d["name"] == "Delegate"]
    assert list(delegate["input_schema"]["properties"]) == ["agent", "input", "questions"]
    assert delegate["description"] == DESCRIPTION_0_1_13
    args = {"agent": "worker", "input": "one", "background": True}
    with pytest.raises(InputError) as raised:
        await tools.dispatch("Delegate", args)
    assert f"InputError: {raised.value}" == REFUSED


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
    run = await asyncio.wait_for(work, 5)
    assert isinstance(run, Generation) and run.stopped == press and run.calls == 2
    wrap = Message(role="user", content="wrap up")
    assert run.conversation[-2:] == [Message(role="assistant", content="waiting"), wrap]
    [event] = taken(seen, run.run_id)
    assert event.messages == [wrap] and event.notices == []
    assert inbox.get_nowait() == "and this" and inbox.empty()
    gate.set()
    await until(ends(seen, 2), "the helper's end")


async def test_a_held_reply_on_the_last_allowed_call_raises_at_once(tmp_path: Path) -> None:
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting")],
        {"part one": [gate, text_reply("the answer")]},
    )
    seen = Seen()
    engine = team(crew(tmp_path, background=True), model, max_model_calls=2)
    with pytest.raises(RunawayError, match=r"^engine: run exceeded 2 model calls$"):
        await asyncio.wait_for(engine.run("lead", "acme", "go", on_event=seen), 2)
    assert len(model.calls["lead"]) == 2
    gate.set()
    await until(ends(seen, 2), "the helper's end")


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
    run = await asyncio.wait_for(work, 2)
    assert isinstance(run, Generation)
    assert (run.stopped, run.calls, run.reply) == ("after_step", 2, "waiting")
    gate.set()
    await until(ends(seen, 2), "the helper's end")
