import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine import Later, Pause, Review, Waiting, chain
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Answer, ChoiceAnswer, Message, Reply, ToolCall, ToolSpec
from rfnry_engine.records import (
    Decision,
    Generation,
    RunEnded,
    Spoken,
    ToolResultReady,
    VoiceStart,
)
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.stop import Stop
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import (
    Crew,
    Fixed,
    MemoryBucket,
    MemoryStore,
    Scripted,
    Seen,
    background as backgrounded,
    crew,
    delegating,
    team,
    text_reply,
)
from .test_approve import Refunds, make_tree, use
from .test_engine_bridge import result, run_coder, script
from .test_voice import SAID, Recording, engine_of, with_tools
from .trees import provided

CANNOT = "engine: approve answered Later() for {name} of agent {agent!r}, which cannot wait: {why}"
VOICE = "a voice session cannot wait for an answer"
TOOL = "a run engine.tool or EngineTools starts keeps no conversation to wait in"
BELOW = "a run below the top waits only in a store; pass store= and conversation= to engine.run"
REPLAY = "a replay cannot wait for an answer"
REFUND = ToolCall(id="c-refund", name="refund", args={"amount": 10})
LOOKUP = ToolCall(id="c-lookup", name="lookup", args={})


class Reported(BaseModel):
    text: str


def later_for(*names: str) -> Callable[[str, ToolCall], Awaitable[bool | Later]]:
    async def approve(agent: str, call: ToolCall) -> bool | Later:
        return Later() if call.name in names else True

    return approve


def cannot(name: str, agent: str, why: str) -> str:
    return re.escape(CANNOT.format(name=name, agent=agent, why=why))


def calling(*calls: ToolCall) -> Reply:
    return Reply(tool_calls=[call.model_copy() for call in calls], stop_reason="tool_use")


async def found(args: dict[str, Any]) -> str:
    return "found"


def desk_tools(
    refunds: Refunds, lookup: Callable[[dict[str, Any]], Awaitable[str]] = found
) -> Tools:
    return Tools([refunds.tools(), Tool("lookup", "Look an order up", body=lookup)])


def desk(
    tmp_path: Path, model: Scripted, judge: Any = None, consult: bool = False, **engine: Any
) -> Engine:
    run_tools = {"desk": ["refund", "lookup"], **({"lead": ["consult"]} if consult else {})}
    return Engine(
        make_tree(tmp_path, run_tools=run_tools),
        providers=provided({"strong": model, "judge": judge}),
        **engine,
    )


async def paused_desk(
    tmp_path: Path, *replies: Reply, **options: Any
) -> tuple[Generation, Refunds, Scripted]:
    refunds = Refunds()
    model = Scripted(*replies, text_reply("never"))
    run = await desk(tmp_path, model).run(
        "desk",
        "acme",
        "refund it",
        tools=desk_tools(refunds),
        approve=later_for("refund"),
        **options,
    )
    assert isinstance(run, Generation)
    return run, refunds, model


async def test_a_call_answered_later_pauses_the_run_after_its_batch(tmp_path: Path) -> None:
    run, refunds, model = await paused_desk(tmp_path, calling(REFUND, LOOKUP))
    [waiting] = run.paused
    assert (waiting.run_id, waiting.agent, waiting.conversation) == (run.run_id, "desk", None)
    assert waiting.call == ToolCall(id="c-refund", name="refund", args={"amount": 10})
    assert re.fullmatch(r"[0-9a-f]{32}", waiting.id)
    assert refunds.done == [] and len(model.calls) == 1
    mark = run.conversation[-1]
    assert mark.pause is not None
    assert (mark.role, mark.content, mark.pause.waiting) == ("assistant", "[paused]", run.paused)
    assert (mark.pause.agent, mark.pause.run_id, mark.pause.below) == ("desk", run.run_id, [])
    assert [m.tool_call_id for m in run.conversation if m.role == "tool"] == ["c-lookup"]
    step = next(s for s in run.steps if s.tool == "refund")
    assert (step.waiting, step.result, step.is_error, step.approved) == (True, "", False, None)
    assert [s.tool for s in run.steps] == ["refund", "lookup"]
    assert run.stopped is None and run.error is None


async def test_paused_lists_the_waiting_calls_in_call_order_whatever_order_approve_answers(
    tmp_path: Path,
) -> None:
    async def slow_first(agent: str, call: ToolCall) -> Later:
        if call.name == "refund":
            await asyncio.sleep(0.05)
        return Later()

    output = ToolCall(id="c-out", name="Output", args={"text": "done"})
    model = Scripted(calling(REFUND, LOOKUP, output), text_reply("never"))
    run = await desk(tmp_path, model).run(
        "desk",
        "acme",
        "refund it",
        tools=desk_tools(Refunds()),
        approve=slow_first,
        expect=Reported,
    )
    assert isinstance(run, Generation)
    assert [waiting.call.id for waiting in run.paused] == ["c-refund", "c-lookup"]
    mark = run.conversation[-1].pause
    assert mark is not None and mark.waiting == run.paused
    out = next(s for s in run.steps if s.tool == "Output")
    assert out.result == (
        "engine: waiting for an answer: refund, lookup; call Output again once they have one"
    )


async def test_a_paused_run_says_so_in_run_ended(tmp_path: Path) -> None:
    seen = Seen()
    run, _, _ = await paused_desk(tmp_path, calling(REFUND, LOOKUP), on_event=seen)
    ended = [e for e in seen.events if isinstance(e, RunEnded)]
    assert ended[-1].paused == run.paused and ended[-1].stopped is None
    assert not [
        e for e in seen.events if isinstance(e, ToolResultReady) and e.call_id == "c-refund"
    ]
    assert [e.call_id for e in seen.events if isinstance(e, ToolResultReady)] == ["c-lookup"]


@pytest.mark.parametrize("mode", ["now", "after_step"])
async def test_a_stop_pressed_during_the_batch_answers_the_waiting_call_as_stopped(
    tmp_path: Path, mode: str
) -> None:
    stop, seen = Stop(), Seen()

    async def pressing(args: dict[str, Any]) -> str:
        if mode == "now":
            stop.now()
        else:
            stop.after_step()
        return "found"

    model = Scripted(calling(REFUND, LOOKUP), text_reply("never"))
    run = await desk(tmp_path, model).run(
        "desk",
        "acme",
        "refund it",
        tools=desk_tools(Refunds(), pressing),
        approve=later_for("refund"),
        stop=stop,
        on_event=seen,
    )
    assert isinstance(run, Generation)
    assert run.paused == [] and run.stopped == mode and len(model.calls) == 1
    step = next(s for s in run.steps if s.tool == "refund")
    assert (step.result, step.stopped, step.waiting) == ("stopped by the user", True, False)
    assert step.is_error and step.approved is None
    [answered] = [m for m in run.conversation if m.tool_call_id == "c-refund"]
    assert (answered.content, answered.is_error) == ("stopped by the user", True)
    assert all(m.pause is None for m in run.conversation)
    [ready] = [e for e in seen.events if isinstance(e, ToolResultReady) and e.call_id == "c-refund"]
    assert ready.step.stopped and not ready.step.waiting


async def test_a_paused_run_with_a_store_appends_its_mark(tmp_path: Path) -> None:
    store = MemoryStore()
    run, _, _ = await paused_desk(tmp_path, calling(REFUND, LOOKUP), store=store, conversation="c1")
    held = store.held[("acme", "c1")]
    assert held[-1].pause is not None
    assert held[-1].pause.waiting == run.paused and run.paused[0].conversation == "c1"
    assert held[-2].tool_call_id == "c-lookup"
    assert held == run.conversation


async def test_output_is_refused_while_a_call_of_its_run_waits(tmp_path: Path) -> None:
    output = ToolCall(id="c-out", name="Output", args={"text": "done"})
    run, _, model = await paused_desk(tmp_path, calling(REFUND, output), expect=Reported)
    out = next(s for s in run.steps if s.tool == "Output")
    assert (
        out.result == "engine: waiting for an answer: refund; call Output again once they have one"
    )
    assert out.is_error and run.paused and run.reply == "" and len(model.calls) == 1
    [answered] = [m for m in run.conversation if m.tool_call_id == "c-out"]
    assert answered.is_error and answered.content == out.result


async def test_a_routed_run_that_pauses_shows_its_calls_on_the_decision(tmp_path: Path) -> None:
    helps: dict[str, Answer] = {
        "intent": ChoiceAnswer(choice="help", probabilities={"help": 0.9, "other": 0.1})
    }
    model = Scripted(calling(REFUND), text_reply("never"))
    decision = await desk(tmp_path, model, Fixed(helps)).run(
        "router", "acme", "refund it", tools=desk_tools(Refunds()), approve=later_for("refund")
    )
    assert isinstance(decision, Decision) and isinstance(decision.routed_to, Generation)
    assert decision.paused == decision.routed_to.paused != []
    assert decision.paused[0].agent == "desk"


async def test_later_from_a_commands_bridged_call_is_a_refusal(
    tmp_path: Path, short_root: Path
) -> None:
    script(short_root, 'import json\nprint(json.dumps(call("lookup", topic="x")))\n')
    async with Workspace(Local(short_root), scope="acme") as work:
        run, _, _ = await run_coder(
            tmp_path,
            work,
            tools=[Tool("lookup", "Look", body=found)],
            approve=later_for("lookup"),
        )
    step = next(s for s in run.steps if s.by is not None)
    assert (step.result, step.approved) == (
        "NotApproved: lookup waits for an answer a command cannot wait for",
        False,
    )
    assert step.is_error and not step.waiting
    assert json.loads(result(run.steps[-1])["stdout"]) == {"error": step.result}
    assert run.paused == [] and run.reply == "ok"


async def test_later_in_a_voice_session_is_a_configuration_error(tmp_path: Path) -> None:
    with_tools(tmp_path, "refund")
    refunds = Refunds()

    async def session(start: VoiceStart) -> Spoken:
        await start.tools.dispatch("refund", {"amount": 10})
        return SAID

    with pytest.raises(ConfigurationError, match=cannot("refund", "voice", VOICE)):
        await engine_of(tmp_path).talk(
            "voice", "acme", session, tools=refunds.tools(), approve=later_for("refund")
        )
    assert refunds.done == []


@pytest.mark.parametrize("via", ["tool", "engine_tools", "a_run_s_tool"])
async def test_later_in_a_run_engine_tool_starts_is_a_configuration_error(
    tmp_path: Path, via: str
) -> None:
    refunds = Refunds()
    model = Scripted(
        *([use("consult", input="refund it")] if via == "a_run_s_tool" else []),
        calling(REFUND),
        text_reply("never"),
    )
    engine = desk(tmp_path, model, consult=via == "a_run_s_tool")
    approve = later_for("refund")
    with pytest.raises(ConfigurationError, match=cannot("refund", "desk", TOOL)):
        if via == "tool":
            consult = engine.tool("desk", scope="acme", tools=desk_tools(refunds), approve=approve)
            await consult.dispatch("desk", {"input": "refund it"})
        elif via == "engine_tools":
            surface = EngineTools(
                engine, agent="lead", scope="acme", tools=desk_tools(refunds), approve=approve
            )
            await surface.dispatch("Delegate", {"agent": "desk", "input": "refund it"})
        else:
            consult = engine.tool(
                "desk", scope="acme", name="consult", tools=desk_tools(refunds), approve=approve
            )
            await engine.run("lead", "acme", "go", tools=consult)
    assert refunds.done == []
    assert all(
        "ConfigurationError" not in str(m.content) for _, sent, _ in model.calls for m in sent
    )


@pytest.mark.parametrize("background", [False, True])
async def test_later_below_the_top_without_a_store_fails_the_top_level_run(
    tmp_path: Path, background: bool
) -> None:
    first = backgrounded("part one") if background else delegating("worker", "part one")
    failed = asyncio.Event()
    model = Crew(
        [first, failed, text_reply("waiting"), text_reply("never")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    folder = crew(
        tmp_path,
        worker={"tools": {"allow": ["lookup"]}},
        **({"background": True} if background else {}),
    )
    seen = Seen()

    async def watch(event: Any) -> None:
        await seen(event)
        if isinstance(event, RunEnded) and event.call is not None:
            failed.set()

    recording = Recording()
    with pytest.raises(ConfigurationError, match=cannot("lookup", "worker", BELOW)):
        await team(folder, model, refinery=recording).run(
            "lead",
            "acme",
            "go",
            tools=Tools([Tool("lookup", "Look", body=found)]),
            approve=later_for("lookup"),
            on_event=watch,
        )
    assert all(
        "ConfigurationError" not in str(m.content) for call in model.calls["lead"] for m in call
    )
    [ended] = [e for e in seen.events if isinstance(e, RunEnded) and e.call is None]
    error = "ConfigurationError: " + CANNOT.format(name="lookup", agent="worker", why=BELOW)
    assert ended.error == error
    [lead] = recording.runs
    assert isinstance(lead, Generation) and lead.error == error
    [worker] = lead.children
    assert worker.agent == "worker" and worker.error == error


async def test_a_helper_s_later_wins_over_the_lead_s_last_model_call(tmp_path: Path) -> None:
    failed = asyncio.Event()
    model = Crew(
        [backgrounded("part one"), failed, text_reply("waiting")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    folder = crew(tmp_path, worker={"tools": {"allow": ["lookup"]}}, background=True)

    async def watch(event: Any) -> None:
        if isinstance(event, RunEnded) and event.call is not None:
            failed.set()

    with pytest.raises(ConfigurationError, match=cannot("lookup", "worker", BELOW)):
        await team(folder, model, max_model_calls=2).run(
            "lead",
            "acme",
            "go",
            tools=Tools([Tool("lookup", "Look", body=found)]),
            approve=later_for("lookup"),
            on_event=watch,
        )
    assert len(model.calls["lead"]) == 2


class Overlapping(Crew):
    def __init__(self, began: asyncio.Event, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.began = began

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        if self.marker in system and len(self.calls["lead"]) == 1:
            self.began.set()
        return await super().generate(system, messages, tools)


def ended_below(ended: asyncio.Event) -> Callable[[Any], Awaitable[None]]:
    async def watch(event: Any) -> None:
        if isinstance(event, RunEnded) and event.call is not None:
            ended.set()

    return watch


async def test_a_helper_s_later_during_the_lead_s_last_turn_fails_the_lead_that_pauses(
    tmp_path: Path,
) -> None:
    failed, recording = asyncio.Event(), Recording()
    second = calling(
        ToolCall(id="c-job", name="Job", args={"id": "d1", "wait": 5}),
        ToolCall(id="c-jobs", name="Jobs", args={}),
    )
    model = Crew(
        [backgrounded("part one"), failed, second, text_reply("never")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    folder = crew(tmp_path, worker={"tools": {"allow": ["lookup"]}}, background=True)
    with pytest.raises(ConfigurationError, match=cannot("lookup", "worker", BELOW)):
        await team(folder, model, refinery=recording).run(
            "lead",
            "acme",
            "go",
            tools=Tools([Tool("lookup", "Look", body=found)]),
            approve=later_for("lookup", "Jobs"),
            on_event=ended_below(failed),
        )
    assert len(model.calls["lead"]) == 2
    [lead] = recording.runs
    assert isinstance(lead, Generation) and lead.paused == []
    jobs = next(s for s in lead.steps if s.tool == "Jobs")
    assert (jobs.result, jobs.waiting) == ("NotRun: Jobs was not run; the run failed", False)
    assert all(m.pause is None for m in lead.conversation)


async def test_a_helper_s_later_beside_a_stop_answers_the_lead_s_waiting_call_not_run(
    tmp_path: Path,
) -> None:
    failed, stop, recording = asyncio.Event(), Stop(), Recording()

    async def approve(agent: str, call: ToolCall) -> bool | Later:
        if call.name == "Job":
            stop.after_step()
        return Later() if call.name in ("lookup", "Jobs") else True

    second = calling(
        ToolCall(id="c-job", name="Job", args={"id": "d1"}),
        ToolCall(id="c-jobs", name="Jobs", args={}),
    )
    model = Crew(
        [backgrounded("part one"), failed, second, text_reply("never")],
        {"part one": [calling(LOOKUP), text_reply("never")]},
    )
    folder = crew(tmp_path, worker={"tools": {"allow": ["lookup"]}}, background=True)
    with pytest.raises(ConfigurationError, match=cannot("lookup", "worker", BELOW)):
        await team(folder, model, refinery=recording).run(
            "lead",
            "acme",
            "go",
            tools=Tools([Tool("lookup", "Look", body=found)]),
            approve=approve,
            stop=stop,
            on_event=ended_below(failed),
        )
    [lead] = recording.runs
    assert isinstance(lead, Generation) and lead.paused == []
    jobs = next(s for s in lead.steps if s.tool == "Jobs")
    assert (jobs.result, jobs.stopped) == ("NotRun: Jobs was not run; the run failed", False)


async def test_a_helper_that_ends_during_the_lead_s_last_turn_is_heard_in_the_mark(
    tmp_path: Path,
) -> None:
    began, ended = asyncio.Event(), asyncio.Event()
    model = Overlapping(
        began,
        [backgrounded("part one"), ended, calling(ToolCall(id="c-jobs", name="Jobs"))],
        {"part one": [began, text_reply("part done")]},
    )
    folder = crew(tmp_path, background=True)
    run = await team(folder, model).run(
        "lead", "acme", "go", approve=later_for("Jobs"), on_event=ended_below(ended)
    )
    assert isinstance(run, Generation) and [w.call.id for w in run.paused] == ["c-jobs"]
    mark = run.conversation[-1].pause
    assert mark is not None
    assert [(m.notice.job, m.notice.kind) for m in mark.held if m.notice] == [("d1", "finished")]
    assert mark.held[0].content == "[helper d1 · worker · finished]\npart done"
    assert not [m for m in run.conversation[:-1] if m.notice is not None]
    assert [child.agent for child in run.children] == ["worker"]
    assert not any(m.notice for call in model.calls["lead"] for m in call)


async def test_a_waiting_call_of_a_run_that_fails_reads_not_run(tmp_path: Path) -> None:
    store, recording = MemoryStore(), Recording()

    async def breaks(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == "c-lookup":
            raise RuntimeError("the handler broke")

    model = Scripted(calling(REFUND, LOOKUP), text_reply("never"))
    with pytest.raises(RuntimeError, match="the handler broke"):
        await desk(tmp_path, model, refinery=recording).run(
            "desk",
            "acme",
            "refund it",
            tools=desk_tools(Refunds()),
            approve=later_for("refund"),
            on_event=breaks,
            store=store,
            conversation="c1",
        )
    [run] = recording.runs
    assert isinstance(run, Generation)
    assert run.paused == [] and run.error == "RuntimeError: the handler broke"
    not_run = "NotRun: refund was not run; the run failed"
    step = next(s for s in run.steps if s.tool == "refund")
    assert (step.result, step.is_error, step.waiting) == (not_run, True, False)
    held = store.held[("acme", "c1")]
    for kept in (held, run.conversation):
        assert [m.content for m in kept if m.tool_call_id == "c-refund"] == [not_run]
        assert all(m.pause is None for m in kept)


async def test_a_replay_whose_approve_answers_later_fails_with_its_own_reason(
    tmp_path: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    await desk(tmp_path / "recorded", Scripted(text_reply("done")), refinery=refinery).run(
        "desk", "acme", "refund it", tools=desk_tools(Refunds())
    )
    refunds = Refunds()
    candidate = desk(tmp_path / "candidate", Scripted(calling(REFUND), text_reply("never")))
    pairs = [
        pair
        async for pair in refinery.replay(
            "acme",
            candidate,
            agent="desk",
            tools=desk_tools(refunds),
            approve=later_for("refund"),
        )
    ]
    [pair] = pairs
    assert pair.candidate is None and refunds.done == []
    assert pair.error == "ConfigurationError: " + CANNOT.format(
        name="refund", agent="desk", why=REPLAY
    )


async def test_a_paused_record_reads_back(tmp_path: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    model = Scripted(calling(REFUND), text_reply("never"))
    run = await desk(tmp_path, model, refinery=refinery).run(
        "desk", "acme", "refund it", tools=desk_tools(Refunds()), approve=later_for("refund")
    )
    assert isinstance(run, Generation) and run.paused
    again = await refinery.run("acme", run.run_id)
    assert isinstance(again, Generation)
    assert again.paused == run.paused and again.steps[0].waiting
    assert again.conversation[-1].pause == run.conversation[-1].pause


async def test_an_export_leaves_marks_out(tmp_path: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    model = Scripted(calling(REFUND, LOOKUP), text_reply("never"))
    run = await desk(tmp_path, model, refinery=refinery).run(
        "desk", "acme", "refund it", tools=desk_tools(Refunds()), approve=later_for("refund")
    )
    assert isinstance(run, Generation) and run.conversation[-1].pause is not None
    await refinery.review("acme", run.run_id, Review(correction="I refunded it."))
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    rows = [
        json.loads(line)
        for key, data in bucket.objects.items()
        if "/generation/" in key
        for line in data.decode().splitlines()
    ]
    preferences = [
        json.loads(line)
        for key, data in bucket.objects.items()
        if "/preference/" in key
        for line in data.decode().splitlines()
    ]
    assert rows and preferences
    assert not any("pause" in m for row in rows for m in [*row["messages"], *row["conversation"]])
    assert not any("pause" in m for row in preferences for m in row["prompt"]["messages"])
    assert rows[0]["conversation"][-1]["tool_call_id"] == "c-lookup"
    assert [s["waiting"] for s in rows[0]["steps"]] == [True, False]


def test_a_mark_is_a_message_that_carries_a_pause() -> None:
    waiting = Waiting(id="w1", run_id="r1", agent="desk", call=REFUND)
    mark = Message(
        role="assistant",
        content="[paused]",
        pause=Pause(agent="desk", run_id="r1", waiting=[waiting]),
    )
    dumped = mark.model_dump(mode="json")
    assert dumped["pause"]["waiting"][0]["call"] == REFUND.model_dump(mode="json")
    assert Message.model_validate(dumped) == mark
    assert "pause" not in Message(role="assistant", content="x").model_dump(mode="json")
    assert Message.model_validate({**dumped, "pause": {"agent": 1}}).pause is None
    assert Later() == Later()


def test_a_mark_leaves_the_reasoning_chain_where_it_was() -> None:
    head = chain.opening("m", "system", [], "")
    mark = Message(role="assistant", content="[paused]", pause=Pause(agent="desk", run_id="r1"))
    assert chain.extend(head, mark) == head
    assert chain.extend(head, Message(role="assistant", content="[paused]")) != head
