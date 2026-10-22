import asyncio
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from itertools import groupby
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.errors import (
    ConfigurationError,
    InputError,
    ModelError,
    NotFoundError,
    RunawayError,
)
from rfnry_engine.models import (
    Answer,
    CheckAnswer,
    Document,
    Image,
    Message,
    Part,
    Reply,
    ScoreAnswer,
    Text,
    ToolCall,
    ToolSpec,
    Usage,
)
from rfnry_engine.records import Event, Generation, Run, TextDelta
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.stop import Stop
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import TOOLS, Fixed, Scripted, Streaming, said, text_reply


def _tool_use(*calls: ToolCall) -> Reply:
    return Reply(
        text="", tool_calls=list(calls), stop_reason="tool_use", usage=Usage(input=10, output=1)
    )


def engine_for(fixtures: Path, llm: Any, **kwargs: Any) -> Engine:
    return Engine(fixtures, generation=llm, tools=TOOLS, **kwargs)


async def test_plain_reply_is_the_record(fixtures: Path) -> None:
    llm = Scripted(text_reply("Refunded.", usage=Usage(input=100, output=5), model="m1"))
    engine = engine_for(fixtures, llm)
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    run = await engine.run("billing", "acme", "Charged twice.", history=history)
    assert isinstance(run, Generation)
    assert run.reply == "Refunded." and run.calls == 1 and run.model == "m1"
    assert run.usage == Usage(input=100, output=5)
    assert run.messages == [*history, Message(role="user", content="Charged twice.")]
    assert run.system.startswith("<<<AGENT>>>\nYou are billing.")
    assert run.scope == "acme" and run.agent == "billing" and run.steps == []
    system, messages, tools = llm.calls[0]
    assert messages == run.messages and [t.name for t in tools] == [
        "Read",
        "Grep",
        "Glob",
        "LS",
        "Load",
    ]


class Declaring(Scripted):
    model = "m"


async def test_a_reply_s_model_wins_over_the_one_its_adapter_declares(fixtures: Path) -> None:
    llm = Declaring(text_reply("a", model="m-2026-09-01"), text_reply("b"))
    reported = await engine_for(fixtures, llm).run("billing", "acme", "x")
    unreported = await engine_for(fixtures, llm).run("billing", "acme", "y")
    assert reported.model == "m-2026-09-01" and unreported.model == "m"


async def test_the_tree_decides_what_the_model_sees(fixtures: Path) -> None:
    llm = Scripted(text_reply("ok"), text_reply("ok"))
    engine = engine_for(fixtures, llm)
    await engine.run("support", "acme", "x")
    await engine.run("billing", "acme", "x")
    assert {t.name for t in llm.calls[0][2]} >= {"lookup_order", "now"}
    assert "now" not in [t.name for t in llm.calls[1][2]]
    assert "- now: The current UTC time" in llm.calls[0][0]


async def test_tool_loop_runs_tools_and_feeds_results_back(fixtures: Path) -> None:
    llm = Scripted(
        _tool_use(ToolCall(id="c1", name="Read", args={"path": "rules/refunds.md"})),
        text_reply("Refund in full."),
    )
    run = await engine_for(fixtures, llm).run("support", "acme", "Charged twice.")
    assert isinstance(run, Generation)
    assert run.calls == 2 and len(run.steps) == 1
    assert run.steps[0].tool == "Read" and "refunded in full" in run.steps[0].result
    second_messages = llm.calls[1][1]
    assert second_messages[-1].role == "tool" and second_messages[-1].tool_call_id == "c1"
    assert said(second_messages)[-2] == Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="Read", args={"path": "rules/refunds.md"})],
    )


async def test_unknown_tool_is_an_error_result_and_the_loop_continues(fixtures: Path) -> None:
    llm = Scripted(_tool_use(ToolCall(id="c1", name="Ghost", args={})), text_reply("ok"))
    run = await engine_for(fixtures, llm).run("support", "acme", "x")
    assert isinstance(run, Generation)
    assert (
        run.steps[0].is_error and run.steps[0].result == "InputError: tools: unknown tool 'Ghost'"
    )
    assert llm.calls[1][1][-1].is_error


async def test_tool_messages_say_whether_the_step_failed(fixtures: Path) -> None:
    llm = Scripted(
        _tool_use(
            ToolCall(id="g", name="Ghost", args={}),
            ToolCall(id="r", name="Read", args={"path": "rules/refunds.md"}),
        ),
        _tool_use(ToolCall(id="o", name="Output", args={})),
        _tool_use(ToolCall(id="o2", name="Output", args={"refund": True, "reason": "dup"})),
    )
    await engine_for(fixtures, llm).run("support", "acme", "x", expect=Verdict)
    tool_messages = [m for m in llm.calls[2][1] if m.role == "tool"]
    assert [(m.tool_call_id, m.is_error) for m in tool_messages] == [
        ("g", True),
        ("r", False),
        ("o", True),
    ]
    assert Message(role="user", content="hi").is_error is False


async def test_call_tools_reach_the_model_for_that_run_only(fixtures: Path) -> None:
    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    llm = Scripted(
        _tool_use(ToolCall(id="p", name="ping", args={})), text_reply("ok"), text_reply("ok")
    )
    engine = engine_for(fixtures, llm)
    run = await engine.run("billing", "acme", "x", tools=Tools([Tool("ping", "Ping", body=ping)]))
    assert isinstance(run, Generation) and run.steps[0].result == "pong"
    await engine.run("billing", "acme", "x")
    assert "ping" not in [t.name for t in llm.calls[2][2]]


class Verdict(BaseModel):
    refund: bool
    reason: str


async def test_expect_forces_a_typed_reply_and_retries_on_invalid(fixtures: Path) -> None:
    llm = Scripted(
        _tool_use(ToolCall(id="o1", name="Output", args={"refund": "maybe"})),
        _tool_use(ToolCall(id="o2", name="Output", args={"refund": True, "reason": "dup"})),
    )
    run = await engine_for(fixtures, llm).run("support", "acme", "x", expect=Verdict)
    assert isinstance(run, Generation)
    assert Verdict.model_validate_json(run.reply) == Verdict(refund=True, reason="dup")
    assert run.calls == 2 and run.steps[0].tool == "Output" and run.steps[0].is_error
    retry_messages = llm.calls[1][1]
    assert retry_messages[-1].role == "tool" and "refund" in str(retry_messages[-1].content)


async def test_calls_after_an_accepted_output_are_answered_but_never_run(
    fixtures: Path,
) -> None:
    looked: list[dict[str, Any]] = []

    async def lookup(args: dict[str, Any]) -> str:
        looked.append(args)
        return "found"

    llm = Scripted(
        _tool_use(
            ToolCall(id="o", name="Output", args={"refund": True, "reason": "dup"}),
            ToolCall(id="l", name="lookup_order", args={"order_id": "A-1"}),
        )
    )
    engine = Engine(fixtures, generation=llm, tools=[Tool("lookup_order", body=lookup), TOOLS[1]])
    run = await engine.run("support", "acme", "x", expect=Verdict)
    assert isinstance(run, Generation) and looked == []
    assert [(m.tool_call_id, m.is_error, m.content) for m in run.conversation[-2:]] == [
        ("o", False, "accepted"),
        ("l", True, "NotRun: lookup_order was not run; Output ended the run"),
    ]
    assert [(s.tool, s.is_error) for s in run.steps] == [("Output", False), ("lookup_order", True)]


async def test_calls_in_a_reply_that_ran_out_of_room_are_answered_but_never_run(
    fixtures: Path,
) -> None:
    looked: list[dict[str, Any]] = []

    async def lookup(args: dict[str, Any]) -> str:
        looked.append(args)
        return "found"

    cut = Reply(
        text="Let me look",
        tool_calls=[ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"})],
        stop_reason="max_tokens",
    )
    llm = Scripted(cut, text_reply("It is on its way."))
    engine = Engine(fixtures, generation=llm, tools=[Tool("lookup_order", body=lookup), TOOLS[1]])
    first = await engine.run("support", "acme", "Where is A-1?")
    assert isinstance(first, Generation) and first.reply == "Let me look" and looked == []
    assert first.conversation[-1] == Message(
        role="tool",
        tool_call_id="c1",
        content="NotRun: lookup_order was not run; the reply ended with max_tokens",
        is_error=True,
    )
    assert [(s.tool, s.is_error, s.duration_ms) for s in first.steps] == [("lookup_order", True, 0)]
    second = await engine.run("support", "acme", "And now?", history=first.added, start=0)
    assert isinstance(second, Generation) and second.reply == "It is on its way."
    assert looked == []


async def test_streaming_emits_deltas(fixtures: Path) -> None:
    llm = Streaming(text_reply("one two"))
    seen: list[Event] = []

    async def collect(event: Event) -> None:
        seen.append(event)

    run = await engine_for(fixtures, llm).run("support", "acme", "x", on_event=collect)
    assert [e.text for e in seen if isinstance(e, TextDelta)] == ["one ", "two "]
    assert all(e.run_id == run.run_id for e in seen)


class Faithful(Scripted):
    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], Awaitable[None]],
    ) -> Reply:
        reply = await self.generate(system, messages, tools)
        for piece in re.findall(r"\S+\s*", reply.text):
            await on_delta(piece)
        return reply


@pytest.mark.parametrize("model", [Scripted, Faithful])
async def test_each_delta_names_its_turn_and_the_reply_is_the_last_turns_text(
    fixtures: Path, model: type[Scripted]
) -> None:
    lookup = ToolCall(id="c", name="lookup_order", args={"order_id": "A-1"})
    llm = model(
        Reply(text="Let me look that up.", tool_calls=[lookup], stop_reason="tool_use"),
        text_reply("A-1 shipped today."),
    )
    seen: list[Event] = []

    async def collect(event: Event) -> None:
        seen.append(event)

    run = await engine_for(fixtures, llm).run("support", "acme", "Where is A-1?", on_event=collect)
    assert isinstance(run, Generation)
    order = [e.turn if isinstance(e, TextDelta) else type(e).__name__ for e in seen]
    assert [kind for kind, _ in groupby(order)] == [
        "RunStarted",
        0,
        "ToolCallStarted",
        "ToolResultReady",
        1,
    ]
    deltas = [e for e in seen if isinstance(e, TextDelta)]
    assert "".join(e.text for e in deltas if e.turn == 0) == "Let me look that up."
    assert "".join(e.text for e in deltas if e.turn == 1) == run.reply == "A-1 shipped today."


async def test_runaway_loop_raises(fixtures: Path) -> None:
    call = _tool_use(ToolCall(id="c", name="LS", args={}))
    engine = engine_for(fixtures, Scripted(call, call, call), max_steps=2)
    with pytest.raises(RunawayError):
        await engine.run("support", "acme", "x")


async def test_a_run_that_raises_is_recorded_with_its_children(
    fixtures: Path, storage: Storage
) -> None:
    triage: dict[str, Answer] = {
        "urgent": CheckAnswer(probability=0.9),
        "severity": ScoreAnswer(score=0, probabilities=[0.7, 0.1, 0.1, 0.1]),
    }
    delegate = ToolCall(id="d", name="Delegate", args={"agent": "triage", "input": "It is down"})
    llm = Scripted(_tool_use(delegate), _tool_use(ToolCall(id="c", name="LS", args={})))
    engine = engine_for(
        fixtures, llm, decision=Fixed(triage), refinery=Refinery(storage), max_steps=2
    )
    with pytest.raises(RunawayError):
        await engine.run("support", "acme", "x")
    assert await storage.count_scope("acme") == 2


async def test_refusals(fixtures: Path) -> None:
    engine = engine_for(fixtures, Scripted(text_reply("x")))
    with pytest.raises(NotFoundError, match="^engine: agent 'ghost' was not found$"):
        await engine.run("ghost", "acme", "x")
    with pytest.raises(InputError):
        await engine.run("billing", "", "x")
    with pytest.raises(ConfigurationError):
        await engine.run("triage", "acme", "x")
    with pytest.raises(ConfigurationError):
        await Engine(fixtures, tools=TOOLS).run("billing", "acme", "x")


@pytest.mark.parametrize("task", ["ghost", "../AGENT", "../../billing/AGENT", "../rules/refunds"])
async def test_a_task_is_looked_up_like_an_agent_and_never_a_path_out_of_tasks(
    fixtures: Path, task: str, storage: Storage
) -> None:
    llm = Scripted(text_reply("x"))
    refinery = Refinery(storage)
    engine = engine_for(fixtures, llm, refinery=refinery)
    message = f"^engine: agent 'support' has no task {re.escape(repr(task))}; its tasks are triage$"
    with pytest.raises(NotFoundError, match=message):
        await engine.run("support", "acme", "x", task=task)
    assert llm.calls == [] and await refinery.storage.runs("acme") == []
    run = await engine.run("support", "acme", "x", task="triage")
    assert isinstance(run, Generation) and run.task == "triage"
    assert "<<<TASK>>>\nDecide whether this message needs a person." in run.system


async def test_recording_is_opt_in_per_run(fixtures: Path) -> None:
    class Recorder:
        def __init__(self) -> None:
            self.runs: list[Run] = []

        async def record(self, run: Run) -> None:
            self.runs.append(run)

    recorder = Recorder()
    engine = engine_for(fixtures, Scripted(text_reply("a"), text_reply("b")), refinery=recorder)
    first = await engine.run("billing", "acme", "x")
    await engine.run("billing", "acme", "y", record=False)
    assert [r.run_id for r in recorder.runs] == [first.run_id]


async def _slow(args: dict[str, Any]) -> str:
    await asyncio.sleep(10)
    return "never"


async def test_a_slow_call_tool_times_out_and_the_run_continues(fixtures: Path) -> None:
    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    llm = Scripted(
        _tool_use(ToolCall(id="s", name="slow", args={}), ToolCall(id="p", name="ping", args={})),
        text_reply("carried on"),
    )
    engine = engine_for(fixtures, llm, tool_timeout=0.05)
    started = time.monotonic()
    run = await engine.run(
        "billing",
        "acme",
        "x",
        tools=Tools([Tool("slow", "Slow", body=_slow), Tool("ping", "Ping", body=ping)]),
    )
    assert time.monotonic() - started < 2
    assert isinstance(run, Generation) and run.reply == "carried on"
    assert (
        run.steps[0].is_error and run.steps[0].result == "TimeoutError: slow timed out after 0.05s"
    )
    assert not run.steps[1].is_error and run.steps[1].result == "pong"
    assert llm.calls[1][1][-2].content == "TimeoutError: slow timed out after 0.05s"


async def test_a_slow_declared_tool_times_out(fixtures: Path) -> None:
    llm = Scripted(
        _tool_use(ToolCall(id="l", name="lookup_order", args={"order_id": "1"})), text_reply("ok")
    )
    engine = Engine(
        fixtures,
        generation=llm,
        tools=[Tool("lookup_order", body=_slow), TOOLS[1]],
        tool_timeout=0.05,
    )
    run = await engine.run("support", "acme", "x")
    assert isinstance(run, Generation)
    assert run.steps[0].result == "TimeoutError: lookup_order timed out after 0.05s"


async def test_a_tool_raising_its_own_timeout_is_not_the_deadline(fixtures: Path) -> None:
    async def gives_up(args: dict[str, Any]) -> str:
        raise TimeoutError("upstream gave up")

    llm = Scripted(_tool_use(ToolCall(id="g", name="gives_up", args={})), text_reply("ok"))
    run = await engine_for(fixtures, llm).run(
        "billing", "acme", "x", tools=Tools([Tool("gives_up", "Gives up", body=gives_up)])
    )
    assert isinstance(run, Generation) and run.steps[0].result == "TimeoutError: upstream gave up"


async def test_the_deadline_defaults_to_thirty_seconds_and_none_disables_it(fixtures: Path) -> None:
    async def nap(args: dict[str, Any]) -> str:
        await asyncio.sleep(0.1)
        return "rested"

    assert engine_for(fixtures, Scripted()).tool_timeout == 30.0
    llm = Scripted(_tool_use(ToolCall(id="n", name="nap", args={})), text_reply("ok"))
    run = await engine_for(fixtures, llm, tool_timeout=None).run(
        "billing", "acme", "x", tools=Tools([Tool("nap", "Nap", body=nap)])
    )
    assert isinstance(run, Generation) and run.steps[0].result == "rested"
    for bad in (0, -1.0):
        with pytest.raises(ConfigurationError, match="tool_timeout"):
            engine_for(fixtures, Scripted(), tool_timeout=bad)


def test_step_and_depth_caps_must_be_positive(fixtures: Path) -> None:
    for name in ("max_steps", "max_depth"):
        for bad in (0, -1):
            with pytest.raises(ConfigurationError, match=f"^engine: {name} must be positive$"):
                engine_for(fixtures, Scripted(), **{name: bad})


def test_a_root_that_is_not_a_directory_is_refused_by_its_path(tmp_path: Path) -> None:
    a_file = tmp_path / "AGENT.md"
    a_file.write_text("---\nname: a\nkind: generation\nmodel: strong\n---\n", encoding="utf-8")
    for root in (tmp_path / "missing", a_file, str(tmp_path / "missing")):
        expected = f"^engine: root is not a directory: {re.escape(str(root))}$"
        with pytest.raises(ConfigurationError, match=expected):
            Engine(root, generation=Scripted())


class ForeignCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = {}


class ForeignUsage(BaseModel):
    input: int = 0
    output: int = 0


class ForeignReply(BaseModel):
    text: str = ""
    tool_calls: list[ForeignCall] = []
    usage: ForeignUsage | None = None
    stop_reason: str = "end_turn"
    model: str | None = None


async def test_a_foreign_reply_is_accepted_by_its_fields(fixtures: Path) -> None:
    llm = Scripted(
        ForeignReply(
            tool_calls=[ForeignCall(id="c1", name="Read", args={"path": "rules/refunds.md"})],
            stop_reason="tool_use",
        ),  # type: ignore[arg-type]
        ForeignReply(text="done", usage=ForeignUsage(input=3, output=1), model="m9"),  # type: ignore[arg-type]
    )
    run = await engine_for(fixtures, llm).run("support", "acme", "x")
    assert isinstance(run, Generation) and run.reply == "done" and run.model == "m9"
    assert run.steps[0].tool == "Read" and run.usage == Usage(input=3, output=1)
    with pytest.raises(ModelError):
        await engine_for(fixtures, Scripted(ForeignReply(stop_reason="later"))).run(  # type: ignore[arg-type]
            "support", "acme", "x"
        )


async def test_a_run_returns_the_conversation_to_continue_from(fixtures: Path) -> None:
    llm = Scripted(
        Reply(
            text="Checking.",
            tool_calls=[
                ToolCall(id="c1", name="Read", args={"path": "rules/refunds.md"}),
                ToolCall(id="c2", name="Ghost", args={}),
            ],
            stop_reason="tool_use",
            reasoning={"thought": "read the rule"},
        ),
        text_reply("Refund in full."),
        text_reply("Done."),
    )
    engine = engine_for(fixtures, llm)
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    first = await engine.run("support", "acme", "Charged twice.", history=history)
    assert isinstance(first, Generation)
    assert first.messages == [*history, Message(role="user", content="Charged twice.")]
    assert said(first.conversation) == [
        *first.messages,
        Message(
            role="assistant",
            content="Checking.",
            tool_calls=[
                ToolCall(id="c1", name="Read", args={"path": "rules/refunds.md"}),
                ToolCall(id="c2", name="Ghost"),
            ],
            reasoning={"thought": "read the rule"},
        ),
        Message(role="tool", tool_call_id="c1", content=first.steps[0].result),
        Message(role="tool", tool_call_id="c2", content=first.steps[1].result, is_error=True),
        Message(role="assistant", content="Refund in full."),
    ]

    second = await engine.run(
        "support", "acme", "And the second charge?", history=first.conversation
    )
    assert isinstance(second, Generation)
    assert llm.calls[2][1] == said(
        [*first.conversation, Message(role="user", content="And the second charge?")]
    )
    assert said(second.conversation)[-1] == Message(role="assistant", content="Done.")


async def test_a_typed_run_closes_its_output_call_in_the_conversation(fixtures: Path) -> None:
    llm = Scripted(
        _tool_use(ToolCall(id="o1", name="Output", args={"refund": True, "reason": "dup"}))
    )
    run = await engine_for(fixtures, llm).run("support", "acme", "x", expect=Verdict)
    assert isinstance(run, Generation)
    assert said(run.conversation)[-2] == Message(
        role="assistant",
        tool_calls=[ToolCall(id="o1", name="Output", args={"refund": True, "reason": "dup"})],
    )
    assert run.conversation[-1] == Message(role="tool", tool_call_id="o1", content="accepted")


async def test_a_tool_named_output_collides_with_expect(fixtures: Path) -> None:
    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    class Verdict(BaseModel):
        refund: bool

    engine = engine_for(fixtures, Scripted(text_reply("x")))
    with pytest.raises(
        ConfigurationError, match="^engine: a tool named 'Output' collides with expect=$"
    ):
        await engine.run(
            "billing",
            "acme",
            "x",
            expect=Verdict,
            tools=Tools([Tool("Output", "Clash", body=ping)]),
        )


async def test_a_failed_recording_is_logged_and_the_run_returned(
    fixtures: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class Broken:
        async def record(self, run: Run) -> None:
            raise RuntimeError("database down")

    engine = engine_for(fixtures, Scripted(text_reply("fine")), refinery=Broken())
    with caplog.at_level("WARNING"):
        run = await engine.run("billing", "acme", "x")
    assert isinstance(run, Generation) and run.reply == "fine"
    assert f"refinery.record failed for run {run.run_id}" in caplog.text


PNG = b"\x89PNG\r\n\x1a\n"


async def test_parts_as_input_are_the_first_user_message(fixtures: Path) -> None:
    llm = Scripted(text_reply("A cat."), text_reply("Grey."))
    engine = engine_for(fixtures, llm)
    parts: list[Part] = [Text(text="What is this?"), Image(media_type="image/png", data=PNG)]
    first = await engine.run("billing", "acme", parts)
    assert isinstance(first, Generation)
    assert llm.calls[0][1][-1] == Message(role="user", content=parts)
    assert first.input == parts and first.messages == [Message(role="user", content=parts)]
    second = await engine.run("billing", "acme", "Which colour?", history=first.conversation)
    assert isinstance(second, Generation)
    assert said(llm.calls[1][1]) == [
        Message(role="user", content=parts),
        Message(role="assistant", content="A cat."),
        Message(role="user", content="Which colour?"),
    ]
    assert llm.calls[1][1][0].content[1] == Image(media_type="image/png", data=PNG)


async def test_parts_in_history_reach_the_model_as_they_are(fixtures: Path) -> None:
    llm = Scripted(text_reply("ok"))
    history = [
        Message(
            role="user",
            content=[
                Text(text="the invoice"),
                Document(media_type="application/pdf", url="https://x/i.pdf"),
            ],
        )
    ]
    await engine_for(fixtures, llm).run("billing", "acme", "and?", history=history)
    assert llm.calls[0][1][0] == history[0]


async def test_a_tool_that_returns_parts_hands_them_to_the_model(fixtures: Path) -> None:
    async def photo(args: dict[str, Any]) -> list[Part]:
        return [Text(text="the photo"), Image(media_type="image/png", data=PNG)]

    llm = Scripted(_tool_use(ToolCall(id="p", name="photo", args={})), text_reply("A cat."))
    run = await engine_for(fixtures, llm).run(
        "billing", "acme", "x", tools=Tools([Tool("photo", body=photo)])
    )
    assert isinstance(run, Generation)
    returned: list[Part] = [Text(text="the photo"), Image(media_type="image/png", data=PNG)]
    assert llm.calls[1][1][-1] == Message(role="tool", tool_call_id="p", content=returned)
    assert run.steps[0].result == returned and not run.steps[0].is_error


async def test_a_list_that_mixes_parts_with_other_values_is_refused(fixtures: Path) -> None:
    llm = Scripted()
    with pytest.raises(InputError, match="^engine: input mixes parts with other values$"):
        await engine_for(fixtures, llm).run("billing", "acme", [Text(text="a"), {"b": 1}])
    assert llm.calls == []


async def test_a_list_of_values_is_still_json_state(fixtures: Path) -> None:
    llm = Scripted(text_reply("ok"))
    await engine_for(fixtures, llm).run("billing", "acme", [1, "two"])
    assert llm.calls[0][1][-1] == Message(role="user", content='[1, "two"]')


async def test_load_hands_the_model_an_image_and_a_pdf_from_its_folder(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    png, pdf = b"\x89PNG\r\n\x1a\nbody", b"%PDF-1.7 rates"
    (root / "billing" / "knowledge").mkdir()
    (root / "billing" / "knowledge" / "diagram.png").write_bytes(png)
    (root / "billing" / "knowledge" / "rates.pdf").write_bytes(pdf)
    llm = Scripted(
        _tool_use(
            ToolCall(id="i", name="Load", args={"path": "knowledge/diagram.png"}),
            ToolCall(id="d", name="Load", args={"path": "knowledge/rates.pdf"}),
        ),
        text_reply("Express is $25."),
    )
    run = await engine_for(root, llm).run("billing", "acme", "What does express cost?")
    assert isinstance(run, Generation)
    image: list[Part] = [Image(media_type="image/png", data=png)]
    document: list[Part] = [Document(media_type="application/pdf", name="rates.pdf", data=pdf)]
    assert llm.calls[1][1][-2:] == [
        Message(role="tool", tool_call_id="i", content=image),
        Message(role="tool", tool_call_id="d", content=document),
    ]
    assert [s.result for s in run.steps] == [image, document] and not any(
        s.is_error for s in run.steps
    )


class PicturesOnly(Scripted):
    media_types = {"image/*"}


async def test_the_model_adapter_says_which_files_load_may_hand_it(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    (root / "billing" / "knowledge").mkdir()
    (root / "billing" / "knowledge" / "rates.pdf").write_bytes(b"%PDF-1.7 rates")
    llm = PicturesOnly(
        _tool_use(ToolCall(id="d", name="Load", args={"path": "knowledge/rates.pdf"})),
        text_reply("I cannot open that file."),
    )
    run = await engine_for(root, llm).run("billing", "acme", "What does express cost?")
    assert isinstance(run, Generation) and run.steps[0].is_error
    assert run.steps[0].result == (
        "InputError: builtins: the model does not take application/pdf files: knowledge/rates.pdf"
    )


async def test_a_run_keeps_the_stop_reason_of_its_last_reply(fixtures: Path) -> None:
    ended = await engine_for(fixtures, Scripted(text_reply("Done."))).run("billing", "acme", "x")
    cut = await engine_for(
        fixtures, Scripted(Reply(text="Refunds are", stop_reason="max_tokens"))
    ).run("billing", "acme", "x")
    assert isinstance(ended, Generation) and ended.stop_reason == "end_turn"
    assert isinstance(cut, Generation) and cut.stop_reason == "max_tokens"
    assert cut.reply == "Refunds are"


class Refund(BaseModel):
    refund: bool


async def test_an_accepted_output_ends_on_tool_use(fixtures: Path) -> None:
    call = ToolCall(id="o1", name="Output", args={"refund": True})
    llm = Scripted(Reply(tool_calls=[call], stop_reason="tool_use"))
    run = await engine_for(fixtures, llm).run("billing", "acme", "x", expect=Refund)
    assert isinstance(run, Generation) and run.stop_reason == "tool_use"


async def test_a_run_stopped_before_any_reply_has_no_stop_reason(fixtures: Path) -> None:
    stop = Stop()
    stop.now()
    run = await engine_for(fixtures, Scripted()).run("billing", "acme", "x", stop=stop)
    assert isinstance(run, Generation) and run.stopped == "now" and run.stop_reason is None


async def test_a_run_that_calls_no_file_tool_never_uses_the_thread_pool(
    fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def pooled(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("engine: the pool was used")

    monkeypatch.setattr(asyncio, "to_thread", pooled)
    thought = {"thought": "check the rule"}
    history: list[Message] = []
    for i in range(5):
        history.append(Message(role="user", content=f"Question {i}."))
        history.append(Message(role="assistant", content=f"Answer {i}.", reasoning=thought))
    read = ToolCall(id="r", name="Read", args={"path": "rules/refunds.md"})
    llm = Scripted(
        text_reply("Refund in full.", reasoning=thought), _tool_use(read), text_reply("Done.")
    )
    engine = engine_for(fixtures, llm)
    run = await engine.run("support", "acme", "Charged twice.", history=history)
    assert isinstance(run, Generation) and run.reply == "Refund in full."
    reading = await engine.run("support", "acme", "Read the refund rule.")
    assert isinstance(reading, Generation) and reading.reply == "Done."
    step = reading.steps[0]
    assert step.is_error and isinstance(step.result, str) and "the pool was used" in step.result
