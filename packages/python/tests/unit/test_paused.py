import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine import Later, Pause, Refused, Review, Waiting, chain
from rfnry_engine.conversation import ended_calls, live
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import Answer, ChoiceAnswer, Message, Reply, ToolCall, ToolSpec
from rfnry_engine.records import (
    Decision,
    Generation,
    MessagesTaken,
    RunEnded,
    RunStarted,
    Spoken,
    ToolCallStarted,
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
from .test_approve import Approver, Refunds, make_tree, use
from .test_engine_bridge import result, run_coder, script
from .test_voice import SAID, Recording, engine_of, with_tools
from .trees import amend, provided

CANNOT = "engine: approve answered Later() for {name} of agent {agent!r}, which cannot wait: {why}"
VOICE = "a voice session cannot wait for an answer"
TOOL = "a run engine.tool or EngineTools starts keeps no conversation to wait in"
BELOW = "a run below the top waits only in a store; pass store= and conversation= to engine.run"
REPLAY = "a replay cannot wait for an answer"
REFUND = ToolCall(id="c-refund", name="refund", args={"amount": 10})
LOOKUP = ToolCall(id="c-lookup", name="lookup", args={})
ESCALATE = "---\ndescription: Hand it to a person\n---\nSay so, then stop.\n"


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
    tmp_path: Path,
    model: Scripted,
    judge: Any = None,
    consult: bool = False,
    skilled: bool = False,
    **engine: Any,
) -> Engine:
    run_tools = {"desk": ["refund", "lookup"], **({"lead": ["consult"]} if consult else {})}
    root = make_tree(tmp_path, run_tools=run_tools)
    if skilled:
        amend(root, "desk", skills={"allow": ["skills/*"], "public": ["escalate"]})
        (root / "skills/escalate").mkdir(parents=True)
        (root / "skills/escalate/SKILL.md").write_text(ESCALATE, encoding="utf-8")
    return Engine(root, providers=provided({"strong": model, "judge": judge}), **engine)


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


HELPS: dict[str, Answer] = {
    "intent": ChoiceAnswer(choice="help", probabilities={"help": 0.9, "other": 0.1})
}
NO_WAITING = "engine: no call waits for an answer under id {id!r}"
UNANSWERED = (
    "engine: {name} of agent {agent!r} waits for an answer under id {id!r}; "
    "give it one in answers= before the run goes on"
)
ANSWERS = "engine: answers maps each id to True, False or Refused(reason); got {got}"
OTHER_AGENT = (
    "engine: the conversation waits on calls of agent {agent!r}; resume it with that agent"
)
NO_SKILL = "engine: a run that resumes a paused conversation invokes no skill; got {got}"
KEY = ("acme", "c1")


def stored(store: MemoryStore | None) -> dict[str, Any]:
    return {} if store is None else {"store": store, "conversation": "c1"}


async def pausing(
    tmp_path: Path,
    *replies: Reply,
    judge: Any = None,
    agent: str = "desk",
    refunds: Refunds | None = None,
    approve: Any = None,
    skilled: bool = False,
    **options: Any,
) -> tuple[Engine, Generation | Decision, Refunds, Scripted]:
    refunds = refunds or Refunds()
    model = Scripted(*replies)
    engine = desk(tmp_path, model, judge, skilled=skilled)
    run = await engine.run(
        agent,
        "acme",
        "refund it",
        tools=desk_tools(refunds),
        approve=approve or later_for("refund"),
        **options,
    )
    return engine, run, refunds, model


async def going_on(
    engine: Engine,
    run: Generation | Decision,
    refunds: Refunds,
    answers: Any,
    store: MemoryStore | None = None,
    agent: str = "desk",
    input: str = "go on",
    **options: Any,
) -> Generation | Decision:
    held = run if isinstance(run, Generation) else run.routed_to
    assert isinstance(held, Generation)
    history: dict[str, Any] = {} if store is not None else {"history": held.conversation}
    return await engine.run(
        agent,
        "acme",
        input,
        tools=desk_tools(refunds),
        answers=answers,
        **history,
        **stored(store),
        **options,
    )


def sent_last(model: Scripted) -> list[Message]:
    return model.calls[-1][1]


def no_mark_in(sent: list[Message]) -> bool:
    return not any(
        m.pause is not None
        or (m.role == "assistant" and not m.tool_calls and m.content == "[paused]")
        for m in sent
    )


async def test_an_approved_call_runs_on_resume_without_asking_again(tmp_path: Path) -> None:
    engine, run, refunds, model = await pausing(
        tmp_path, calling(REFUND, LOOKUP), text_reply("refunded it")
    )
    assert isinstance(run, Generation)
    counting, seen = Approver(), Seen()
    again = await going_on(
        engine, run, refunds, {run.paused[0].id: True}, approve=counting, on_event=seen
    )
    assert isinstance(again, Generation)
    assert refunds.done == [{"amount": 10}] and counting.asked == []
    assert again.resumed and not run.resumed
    [step] = again.steps
    assert (step.tool, step.result, step.approved, step.is_error) == (
        "refund",
        "refunded",
        True,
        False,
    )
    sent = sent_last(model)
    assert [m.role for m in sent[-4:]] == ["assistant", "tool", "tool", "user"]
    assert [m.tool_call_id for m in sent[-3:-1]] == ["c-lookup", "c-refund"]
    assert sent[-1].content == "go on" and no_mark_in(sent)
    assert again.messages[-1].content == "go on" and again.reply == "refunded it"
    assert again.paused == [] and again.input == "go on"
    events = [type(e).__name__ for e in seen.events if e.run_id == again.run_id]
    assert events[:3] == ["RunStarted", "ToolCallStarted", "ToolResultReady"]
    [ready] = [e for e in seen.events if isinstance(e, ToolResultReady)]
    assert ready.call_id == "c-refund" and ready.step == step


@pytest.mark.parametrize(
    ("answer", "result"),
    [
        (Refused("not today"), "NotApproved: refund was not approved: not today"),
        (Refused("  "), "NotApproved: refund was not approved"),
        (False, "NotApproved: refund was not approved"),
    ],
)
async def test_a_refusal_on_resume_reads_its_reason(
    tmp_path: Path, answer: bool | Refused, result: str
) -> None:
    engine, run, refunds, model = await pausing(
        tmp_path, calling(REFUND, LOOKUP), text_reply("I could not")
    )
    seen = Seen()
    again = await going_on(engine, run, refunds, {run.paused[0].id: answer}, on_event=seen)
    assert isinstance(again, Generation)
    [step] = again.steps
    assert (step.result, step.approved, step.is_error) == (result, False, True)
    assert refunds.done == []
    [answered] = [m for m in sent_last(model) if m.tool_call_id == "c-refund"]
    assert (answered.content, answered.is_error) == (result, True)
    told = [e for e in seen.events if isinstance(e, ToolCallStarted | ToolResultReady)]
    assert [(type(e), e.run_id, e.call_id) for e in told] == [
        (ToolCallStarted, again.run_id, "c-refund"),
        (ToolResultReady, again.run_id, "c-refund"),
    ]
    started, ready = told
    assert isinstance(started, ToolCallStarted) and isinstance(ready, ToolResultReady)
    assert (started.tool, started.args, started.by) == ("refund", {"amount": 10}, None)
    assert ready.step == step


async def test_a_resume_with_a_store_continues_the_stored_conversation(tmp_path: Path) -> None:
    store = MemoryStore()
    engine, run, refunds, model = await pausing(
        tmp_path,
        calling(REFUND, LOOKUP),
        text_reply("refunded, not looked up"),
        approve=later_for("refund", "lookup"),
        **stored(store),
    )
    assert isinstance(run, Generation)
    refund, lookup = run.paused
    before = len(store.held[KEY])
    appends = len(store.appended)
    seen = Seen()
    again = await going_on(
        engine,
        run,
        refunds,
        {refund.id: True, lookup.id: Refused("no need")},
        store,
        on_event=seen,
    )
    assert isinstance(again, Generation) and again.resumed and again.error is None
    held = store.held[KEY]
    marks = [m for m in held if m.pause is not None]
    assert len(marks) == 2 and marks[1].pause is not None
    assert marks[1].pause.waiting == [] and marks[1].pause.run_id == run.run_id
    assert not any(str(m.content).startswith("Ended:") for m in held)
    assert [(m.role, m.tool_call_id) for m in held[before - 1 :]] == [
        ("assistant", None),
        ("tool", "c-lookup"),
        ("assistant", None),
        ("tool", "c-refund"),
        ("user", None),
        ("assistant", None),
    ]
    assert store.appended[appends] == ("c1", before, 2)
    assert held[-1].role == "assistant" and held[-1].content == "refunded, not looked up"
    assert refunds.done == [{"amount": 10}] and no_mark_in(sent_last(model))
    assert [s.tool for s in again.steps] == ["lookup", "refund"]
    assert again.added[0].tool_call_id == "c-lookup" and again.added == held[before:]
    assert [
        (type(e).__name__, e.call_id)
        for e in seen.events
        if isinstance(e, ToolCallStarted | ToolResultReady)
    ] == [
        ("ToolCallStarted", "c-lookup"),
        ("ToolResultReady", "c-lookup"),
        ("ToolCallStarted", "c-refund"),
        ("ToolResultReady", "c-refund"),
    ]


class Blocking(Refunds):
    def __init__(self) -> None:
        super().__init__()
        self.started = 0
        self.entered = asyncio.Event()

    def tools(self) -> Tools:
        async def refund(args: dict[str, Any]) -> str:
            self.started += 1
            self.entered.set()
            await asyncio.Event().wait()
            return "refunded"

        return Tools([Tool("refund", "Refund an order", body=refund)])


async def test_an_approved_call_runs_at_most_once(tmp_path: Path) -> None:
    store, refunds = MemoryStore(), Blocking()
    engine, run, _, model = await pausing(
        tmp_path,
        calling(REFUND, LOOKUP),
        text_reply("it may have gone through"),
        refunds=refunds,
        approve=later_for("refund", "lookup"),
        skilled=True,
        **stored(store),
    )
    refund, lookup = run.paused
    answers = {refund.id: True, lookup.id: Refused("no need")}
    resuming = asyncio.create_task(going_on(engine, run, refunds, answers, store))
    await asyncio.wait_for(refunds.entered.wait(), 5)
    resuming.cancel()
    with pytest.raises(asyncio.CancelledError):
        await resuming
    fresh = store.held[KEY][-1].pause
    assert fresh is not None and fresh.waiting == [] and live(store.held[KEY]) is None
    for again in ({refund.id: True}, {lookup.id: True}):
        [wid] = again
        with pytest.raises(InputError, match=re.escape(NO_WAITING.format(id=wid))):
            await going_on(engine, run, refunds, again, store)
    before = len(store.held[KEY])
    third = await going_on(
        engine, run, refunds, None, store, input="did it go through?", skill="escalate"
    )
    assert isinstance(third, Generation) and third.error is None
    assert third.resumed is False and third.skill == "escalate"
    held = store.held[KEY]
    ended = held[before]
    assert ended.tool_call_id == "c-refund"
    assert str(ended.content).startswith("Ended: refund did not return")
    assert third.added == held[before + 1 :] and third.added[0].role == "user"
    assert third.input == "did it go through?"
    assert refunds.started == 1 and len(model.calls) == 2


@pytest.mark.parametrize("given", [{}, None], ids=["empty", "none"])
@pytest.mark.parametrize("with_store", [False, True], ids=["history", "store"])
async def test_a_resume_that_leaves_the_tops_call_unanswered_is_refused(
    tmp_path: Path, given: dict[str, Any] | None, with_store: bool
) -> None:
    store, recording, seen = MemoryStore() if with_store else None, Recording(), Seen()
    model = Scripted(calling(REFUND, LOOKUP), text_reply("never"))
    engine = desk(tmp_path, model, refinery=recording)
    refunds = Refunds()
    run = await engine.run(
        "desk",
        "acme",
        "refund it",
        tools=desk_tools(refunds),
        approve=later_for("refund"),
        **stored(store),
    )
    assert isinstance(run, Generation)
    stored_before = None if store is None else list(store.held[KEY])
    [waiting] = run.paused
    match = re.escape(UNANSWERED.format(name="refund", agent="desk", id=waiting.id))
    with pytest.raises(InputError, match=match):
        await going_on(engine, run, refunds, given, store, on_event=seen)
    assert len(model.calls) == 1 and seen.events == [] and recording.runs == [run]
    if store is not None:
        assert store.held[KEY] == stored_before


@pytest.mark.parametrize("paused", [True, False], ids=["paused", "not_paused"])
@pytest.mark.parametrize("with_store", [False, True], ids=["history", "store"])
async def test_an_answer_for_no_waiting_call_is_refused(
    tmp_path: Path, paused: bool, with_store: bool
) -> None:
    store, recording, seen = MemoryStore() if with_store else None, Recording(), Seen()
    first = calling(REFUND, LOOKUP) if paused else text_reply("done")
    model = Scripted(first, text_reply("never"))
    engine = desk(tmp_path, model, refinery=recording)
    refunds = Refunds()
    run = await engine.run(
        "desk",
        "acme",
        "refund it",
        tools=desk_tools(refunds),
        approve=later_for("refund"),
        **stored(store),
    )
    assert isinstance(run, Generation) and bool(run.paused) is paused
    stored_before = None if store is None else list(store.held[KEY])
    answers = {**{w.id: True for w in run.paused}, "ghost": True}
    with pytest.raises(InputError, match=re.escape(NO_WAITING.format(id="ghost"))):
        await going_on(engine, run, refunds, answers, store, on_event=seen)
    assert len(model.calls) == 1 and seen.events == [] and recording.runs == [run]
    assert refunds.done == []
    if store is not None:
        assert store.held[KEY] == stored_before


@pytest.mark.parametrize("value", [Later(), "yes", None, 1], ids=repr)
async def test_an_answer_that_is_not_true_false_or_refused_is_refused(
    tmp_path: Path, value: object
) -> None:
    engine, run, refunds, model = await pausing(tmp_path, calling(REFUND), text_reply("never"))
    [waiting] = run.paused
    got = f"{value!r} for {waiting.id!r}"
    with pytest.raises(InputError, match=re.escape(ANSWERS.format(got=got))):
        await going_on(engine, run, refunds, {waiting.id: value})
    with pytest.raises(InputError, match=re.escape(ANSWERS.format(got="list"))):
        await going_on(engine, run, refunds, [(waiting.id, True)])
    assert len(model.calls) == 1 and refunds.done == []


async def test_a_resume_invokes_no_skill(tmp_path: Path) -> None:
    engine, run, refunds, model = await pausing(
        tmp_path, calling(REFUND), text_reply("never"), skilled=True
    )
    assert isinstance(run, Generation)
    answers = {run.paused[0].id: True}
    cases: list[tuple[dict[str, Any], str]] = [
        ({"skill": "escalate"}, "skill='escalate'"),
        ({"input": "/escalate now"}, "/escalate"),
        ({"skill": "escalate", "answers": {}}, "skill='escalate'"),
    ]
    for options, got in cases:
        given = {"answers": answers, **options}
        with pytest.raises(InputError, match=re.escape(NO_SKILL.format(got=got))):
            await going_on(engine, run, refunds, **given)
    ghost = re.escape(NO_WAITING.format(id="ghost"))
    with pytest.raises(InputError, match=ghost):
        await going_on(engine, run, refunds, {"ghost": True}, skill="escalate")
    assert len(model.calls) == 1 and refunds.done == []


@pytest.mark.parametrize(
    ("pauser", "other"),
    [("desk", "lead"), ("lead", "desk"), ("lead", "router")],
)
@pytest.mark.parametrize("with_store", [False, True], ids=["history", "store"])
async def test_a_resume_with_another_agent_is_refused(
    tmp_path: Path, pauser: str, other: str, with_store: bool
) -> None:
    store, judge = MemoryStore() if with_store else None, Fixed(HELPS)
    first = calling(REFUND) if pauser == "desk" else use("Delegate", agent="desk", input="x")
    approve = later_for("refund" if pauser == "desk" else "Delegate")
    engine, run, refunds, model = await pausing(
        tmp_path,
        first,
        text_reply("never"),
        judge=judge,
        agent=pauser,
        approve=approve,
        **stored(store),
    )
    assert isinstance(run, Generation) and run.paused
    answers = {run.paused[0].id: True}
    with pytest.raises(InputError, match=re.escape(OTHER_AGENT.format(agent=pauser))):
        await going_on(engine, run, refunds, answers, store, agent=other)
    with pytest.raises(InputError, match=re.escape(NO_WAITING.format(id="ghost"))):
        await going_on(engine, run, refunds, {"ghost": True}, store, agent=other)
    assert len(model.calls) == 1 and judge.calls == [] and refunds.done == []


async def test_a_resumed_runs_added_starts_at_its_answers(tmp_path: Path) -> None:
    engine, run, refunds, _ = await pausing(
        tmp_path, calling(REFUND), text_reply("done"), text_reply("you are welcome")
    )
    assert isinstance(run, Generation)
    again = await going_on(engine, run, refunds, {run.paused[0].id: True})
    assert isinstance(again, Generation)
    assert again.added[0].tool_call_id == "c-refund"
    assert [m.role for m in again.added] == ["tool", "user", "assistant"]
    kept = [*run.conversation, *again.added]
    third = await engine.run("desk", "acme", "thanks", history=kept, tools=desk_tools(refunds))
    assert isinstance(third, Generation) and third.error is None and not third.resumed
    assert third.added == third.conversation[len(kept) :]
    assert live(kept) is None and run.conversation[-1].pause == live(run.conversation)


async def test_a_resume_of_a_resume_adds_from_the_mark_it_resumed(tmp_path: Path) -> None:
    second = ToolCall(id="c-refund-2", name="refund", args={"amount": 5})
    engine, run, refunds, _ = await pausing(
        tmp_path, calling(REFUND, LOOKUP), calling(second), text_reply("both done")
    )
    assert isinstance(run, Generation)
    (refund,) = run.paused
    paused_again = await going_on(
        engine, run, refunds, {refund.id: Refused("not yet")}, approve=later_for("refund")
    )
    assert isinstance(paused_again, Generation) and paused_again.paused
    assert [(m.role, m.tool_call_id) for m in paused_again.added] == [
        ("tool", "c-refund"),
        ("user", None),
        ("assistant", None),
        ("assistant", None),
    ]
    kept = [*run.conversation, *paused_again.added]
    assert live(kept) == paused_again.conversation[-1].pause
    last = await engine.run(
        "desk",
        "acme",
        "go on",
        history=kept,
        tools=desk_tools(refunds),
        answers={paused_again.paused[0].id: True},
    )
    assert isinstance(last, Generation) and last.resumed
    assert sum(m.pause is not None for m in last.messages) == 2
    assert [(m.role, m.tool_call_id) for m in last.added] == [
        ("tool", "c-refund-2"),
        ("user", None),
        ("assistant", None),
    ]
    assert refunds.done == [{"amount": 5}]


@pytest.mark.parametrize("with_store", [False, True], ids=["history", "store"])
async def test_a_router_resumes_the_agent_its_mark_names(tmp_path: Path, with_store: bool) -> None:
    store, judge = MemoryStore() if with_store else None, Fixed(HELPS)
    engine, decision, refunds, model = await pausing(
        tmp_path,
        calling(REFUND),
        text_reply("refunded"),
        judge=judge,
        agent="router",
        **stored(store),
    )
    assert isinstance(decision, Decision) and isinstance(decision.routed_to, Generation)
    answers = {decision.paused[0].id: True}
    again = await going_on(engine, decision, refunds, answers, store, agent="router")
    assert isinstance(again, Generation) and again.agent == "desk" and len(judge.calls) == 1
    assert again.resumed and again.parent_run_id is None and again.reply == "refunded"
    assert refunds.done == [{"amount": 10}] and no_mark_in(sent_last(model))


@pytest.mark.parametrize("with_store", [False, True], ids=["history", "store"])
async def test_a_router_given_answers_on_a_conversation_that_is_not_paused_decides_nothing(
    tmp_path: Path, with_store: bool
) -> None:
    store, judge = MemoryStore() if with_store else None, Fixed(HELPS)
    engine, decision, refunds, model = await pausing(
        tmp_path, text_reply("done"), judge=judge, agent="router", **stored(store)
    )
    with pytest.raises(InputError, match=re.escape(NO_WAITING.format(id="ghost"))):
        await going_on(engine, decision, refunds, {"ghost": True}, store, agent="router")
    assert len(judge.calls) == 1 and len(model.calls) == 1


async def test_a_call_after_the_resume_is_put_to_approve(tmp_path: Path) -> None:
    again_refund = ToolCall(id="c-refund-2", name="refund", args={"amount": 5})
    engine, run, refunds, _ = await pausing(
        tmp_path, calling(REFUND), calling(again_refund), text_reply("one refunded")
    )
    counting = Approver(allow=False)
    again = await going_on(engine, run, refunds, {run.paused[0].id: True}, approve=counting)
    assert isinstance(again, Generation)
    assert counting.asked == [("desk", "refund", {"amount": 5})]
    assert [(s.result, s.approved) for s in again.steps] == [
        ("refunded", True),
        ("NotApproved: refund was not approved", False),
    ]
    assert refunds.done == [{"amount": 10}]


@pytest.mark.parametrize("at", [2, 3], ids=["the_mark", "the_fresh_mark"])
async def test_a_window_that_starts_at_a_mark_is_refused(tmp_path: Path, at: int) -> None:
    store = MemoryStore(lambda conversation, length: at if length == 6 else 0)
    engine, run, refunds, model = await pausing(
        tmp_path, calling(REFUND), text_reply("never"), **stored(store)
    )
    assert len(store.held[KEY]) == 3
    match = "engine: the window of conversation 'c1' breaks tool call 'c-refund'"
    with pytest.raises(InputError, match=re.escape(match)):
        await going_on(engine, run, refunds, {run.paused[0].id: True}, store)
    assert len(model.calls) == 1 and len(store.held[KEY]) == 6


@pytest.mark.parametrize("with_store", [False, True], ids=["history", "store"])
async def test_reasoning_survives_a_pause_and_its_resume(tmp_path: Path, with_store: bool) -> None:
    store = MemoryStore() if with_store else None
    reasoning = {"thought": "refund the double charge"}
    thinking = Reply(tool_calls=[REFUND], stop_reason="tool_use", reasoning=reasoning)
    engine, run, refunds, model = await pausing(
        tmp_path, thinking, text_reply("refunded"), **stored(store)
    )
    again = await going_on(engine, run, refunds, {run.paused[0].id: True}, store)
    assert isinstance(again, Generation) and again.error is None
    sent = sent_last(model)
    assert [m.role for m in sent] == ["user", "assistant", "tool", "user"]
    assert sent[1].reasoning == reasoning and no_mark_in(sent)


async def test_a_resume_appends_the_held_notices_after_its_answers_and_tells_them(
    tmp_path: Path,
) -> None:
    began, ended = asyncio.Event(), asyncio.Event()
    model = Overlapping(
        began,
        [
            backgrounded("part one"),
            ended,
            calling(ToolCall(id="c-jobs", name="Jobs")),
            text_reply("all done"),
        ],
        {"part one": [began, text_reply("part done")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    run = await engine.run(
        "lead", "acme", "go", approve=later_for("Jobs"), on_event=ended_below(ended)
    )
    assert isinstance(run, Generation)
    mark = run.conversation[-1].pause
    assert mark is not None and len(mark.held) == 1
    seen = Seen()
    again = await engine.run(
        "lead",
        "acme",
        "go on",
        history=run.conversation,
        answers={run.paused[0].id: True},
        on_event=seen,
    )
    assert isinstance(again, Generation) and again.reply == "all done"
    sent = model.calls["lead"][-1]
    assert [(m.role, m.tool_call_id) for m in sent[-3:]] == [
        ("tool", "c-jobs"),
        ("user", None),
        ("user", None),
    ]
    assert sent[-2].content == mark.held[0].content and sent[-1].content == "go on"
    assert [m for m in again.messages if m.notice is not None] == mark.held
    assert no_mark_in(sent)
    [taken] = [e for e in seen.events if isinstance(e, MessagesTaken)]
    assert (taken.turn, taken.messages, taken.notices) == (0, [], mark.held)
    order = [type(e) for e in seen.events if e.run_id == again.run_id]
    assert order.index(ToolResultReady) < order.index(MessagesTaken)
    assert order[0] is RunStarted


async def test_a_mark_that_is_not_last_is_no_pause(tmp_path: Path) -> None:
    store = MemoryStore()

    async def breaks(event: Any) -> None:
        if isinstance(event, RunEnded):
            raise RuntimeError("the handler broke")

    with pytest.raises(RuntimeError, match="the handler broke"):
        await pausing(
            tmp_path, calling(REFUND), text_reply("done"), on_event=breaks, **stored(store)
        )
    held = store.held[KEY]
    assert held[-2].pause is not None and live(held) is None
    assert held[-1].content == "NotRun: refund was not run; the run failed"
    [waiting] = held[-2].pause.waiting
    engine = Engine(tmp_path / "agents", providers=provided({"strong": Scripted(), "judge": None}))
    with pytest.raises(InputError, match=re.escape(NO_WAITING.format(id=waiting.id))):
        await engine.run("desk", "acme", "go on", answers={waiting.id: True}, **stored(store))


def test_ended_calls_leaves_the_named_calls_open() -> None:
    reply = Message(role="assistant", tool_calls=[REFUND, LOOKUP])
    [ended] = ended_calls([Message(role="user", content="x"), reply], named={"c-refund"})
    assert ended.tool_call_id == "c-lookup" and str(ended.content).startswith("Ended: lookup")
    assert live([]) is None and live([reply]) is None


@pytest.mark.parametrize("with_store", [False, True], ids=["history", "store"])
async def test_a_stop_pressed_before_a_resume_cuts_its_approved_calls(
    tmp_path: Path, with_store: bool
) -> None:
    store, stop = MemoryStore() if with_store else None, Stop()
    engine, run, refunds, model = await pausing(
        tmp_path,
        calling(REFUND, LOOKUP),
        text_reply("never"),
        approve=later_for("refund", "lookup"),
        **stored(store),
    )
    assert isinstance(run, Generation)
    refund, lookup = run.paused
    stop.now()
    again = await going_on(
        engine, run, refunds, {refund.id: True, lookup.id: Refused("no")}, store, stop=stop
    )
    assert isinstance(again, Generation) and again.stopped == "now" and again.calls == 0
    assert [(s.tool, s.result, s.approved, s.stopped) for s in again.steps] == [
        ("lookup", "NotApproved: lookup was not approved: no", False, False),
        ("refund", "stopped by the user", True, True),
    ]
    assert refunds.done == [] and len(model.calls) == 1
    kept = again.conversation if store is None else store.held[KEY]
    assert kept[-1].content == "go on" and live(kept) is None
    assert [m.tool_call_id for m in kept if m.role == "tool"] == ["c-lookup", "c-refund"]


@pytest.mark.parametrize(
    ("with_store", "breaks"),
    [(False, "handler"), (True, "handler"), (True, "append")],
    ids=["history-handler", "store-handler", "store-append"],
)
async def test_a_resume_that_fails_in_its_answers_records_from_the_reply_with_the_calls(
    tmp_path: Path, with_store: bool, breaks: str
) -> None:
    store, recording, refunds = MemoryStore() if with_store else None, Recording(), Refunds()
    engine = desk(
        tmp_path, Scripted(calling(REFUND, LOOKUP), text_reply("never")), refinery=recording
    )
    run = await engine.run(
        "desk",
        "acme",
        "refund it",
        tools=desk_tools(refunds),
        approve=later_for("refund", "lookup"),
        **stored(store),
    )
    assert isinstance(run, Generation)
    refund, lookup = run.paused

    async def breaking(event: Any) -> None:
        if isinstance(event, ToolResultReady) and event.call_id == "c-refund":
            raise RuntimeError("the handler broke")

    if store is not None and breaks == "append":
        store.fail["append"] = OSError("the store is down")
    raised = "the handler broke" if breaks == "handler" else "the store is down"
    with pytest.raises((RuntimeError, OSError), match=raised):
        await going_on(
            engine,
            run,
            refunds,
            {refund.id: True, lookup.id: Refused("no")},
            store,
            on_event=breaking,
        )
    failed = recording.runs[-1]
    assert isinstance(failed, Generation) and failed.resumed and failed.error is not None
    assert failed.messages[-1].pause is not None and failed.messages[-1].pause.waiting
    if with_store:
        reply = failed.conversation[0]
        assert reply.role == "assistant" and [c.id for c in reply.tool_calls] == [
            "c-refund",
            "c-lookup",
        ]
        assert failed.start == 1 and failed.messages[0] == reply and len(failed.messages) == 2
    else:
        assert failed.start == 0 and failed.messages == run.conversation
    made: set[str] = set()
    for message in failed.conversation:
        made |= {call.id for call in message.tool_calls}
        assert message.role != "tool" or message.tool_call_id in made
    assert failed.added == failed.conversation[len(failed.messages) :]
    assert failed.added[0].tool_call_id == "c-lookup"
    assert refunds.done == ([] if breaks == "append" else [{"amount": 10}])


class Gated(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def read(
        self, scope: str, conversation: str, start: int, stop: int | None = None
    ) -> Sequence[object]:
        if self.gate is not None:
            self.entered.set()
            await self.gate.wait()
        return await super().read(scope, conversation, start, stop)


async def test_two_router_resumes_of_one_conversation_run_its_approved_call_once(
    tmp_path: Path,
) -> None:
    store, judge = Gated(), Fixed(HELPS)
    engine, decision, refunds, _ = await pausing(
        tmp_path,
        calling(REFUND),
        text_reply("refunded"),
        judge=judge,
        agent="router",
        **stored(store),
    )
    answers = {decision.paused[0].id: True}
    store.gate = asyncio.Event()
    first = asyncio.create_task(going_on(engine, decision, refunds, answers, store, agent="router"))
    await asyncio.wait_for(store.entered.wait(), 5)
    in_use = "engine: conversation 'c1' of scope 'acme' is in use by a run in flight"
    with pytest.raises(InputError, match=re.escape(in_use)):
        await asyncio.wait_for(
            going_on(engine, decision, refunds, answers, store, agent="router"), 5
        )
    store.gate.set()
    again = await asyncio.wait_for(first, 5)
    assert isinstance(again, Generation) and again.agent == "desk" and again.resumed
    assert refunds.done == [{"amount": 10}] and len(judge.calls) == 1
