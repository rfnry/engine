import asyncio
import base64
import json
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine import OnDelta, ReasoningDelta, RunStarted, Stop, TextDelta, chain
from rfnry_engine.engine import Engine
from rfnry_engine.models import (
    Document,
    Image,
    Message,
    Part,
    Reply,
    Text,
    ToolCall,
    ToolSpec,
    Usage,
)
from rfnry_engine.records import Event, Generation
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Scripted, now, text_reply

THOUGHT = {"thought": "read the rule"}
CONTEXT = "Acme is on the enterprise plan."


def reading() -> Reply:
    return Reply(
        text="Checking.",
        tool_calls=[ToolCall(id="c1", name="Read", args={"path": "rules/refunds.md"})],
        stop_reason="tool_use",
        reasoning=THOUGHT,
        usage=Usage(input=40, output=3, cache_read=30),
    )


async def first_run(engine: Engine) -> Generation:
    run = await engine.run("support", "acme", "Charged twice.", context=CONTEXT)
    assert isinstance(run, Generation)
    return run


class Mutating(Scripted):
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        tools[0].input_schema.setdefault("properties", {})["sneaked"] = {"type": "string"}
        return await super().generate(system, messages, tools)


async def test_the_opening_digest_is_taken_before_the_first_call(fixtures: Path) -> None:
    plain = await first_run(
        Engine(fixtures, models={"strong": Scripted(reading(), text_reply("Refund in full."))})
    )
    mutated = await first_run(
        Engine(fixtures, models={"strong": Mutating(reading(), text_reply("Refund in full."))})
    )
    [one, *_] = [m.prefix for m in plain.conversation if m.role == "assistant"]
    [two, *_] = [m.prefix for m in mutated.conversation if m.role == "assistant"]
    assert one is not None and one == two


async def test_an_assistant_message_keeps_the_usage_of_the_call_that_wrote_it(
    fixtures: Path,
) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."))
    run = await first_run(Engine(fixtures, models={"strong": llm}))
    written = [m.usage for m in run.conversation if m.role == "assistant"]
    assert written == [Usage(input=40, output=3, cache_read=30), None]
    assert all(m.usage is None for m in run.conversation if m.role != "assistant")


async def test_an_assistant_message_keeps_the_model_that_wrote_it(fixtures: Path) -> None:
    strong = reading().model_copy(update={"model": "model-strong"})
    fast = text_reply("Refund in full.", model="model-fast")
    llm = Scripted(strong, fast)
    run = await first_run(Engine(fixtures, models={"strong": llm}))
    written = [m.model for m in run.conversation if m.role == "assistant"]
    assert written == ["model-strong", "model-fast"] and run.model == "model-fast"
    assert all(m.model is None for m in run.conversation if m.role != "assistant")
    assert all(m.model is None for call in llm.calls for m in call[1])


async def test_an_assistant_message_keeps_the_adapters_model_when_the_reply_names_none(
    fixtures: Path,
) -> None:
    llm = Named("model-declared", reading(), text_reply("Refund in full."))
    run = await first_run(Engine(fixtures, models={"strong": llm}))
    written = [m.model for m in run.conversation if m.role == "assistant"]
    assert written == ["model-declared", "model-declared"]


def test_a_model_from_elsewhere_that_is_not_a_string_reads_as_none() -> None:
    assert Message.model_validate({"role": "assistant", "model": {"name": "x"}}).model is None
    assert Message.model_validate({"role": "assistant", "model": "model-a"}).model == "model-a"


async def test_the_model_on_a_message_changes_nothing_about_what_reasoning_goes_back(
    fixtures: Path,
) -> None:
    later = {"thought": "the later charge"}
    thinking = Reply(text="The second one.", reasoning=later)
    llm = Scripted(reading(), text_reply("Refund in full."), thinking, text_reply("ok"))
    engine = Engine(fixtures, models={"strong": llm})
    first = await first_run(engine)
    second = await engine.run(
        "support", "acme", "Which charge?", history=first.conversation, context=CONTEXT
    )
    assert isinstance(second, Generation)
    history = [
        m.model_copy(update={"model": "renamed"}) if m.role == "assistant" else m
        for m in second.conversation
    ]
    await engine.run("support", "acme", "Thanks.", history=history, context=CONTEXT)
    sent = [m.reasoning for m in llm.calls[3][1] if m.role == "assistant"]
    assert sent == [THOUGHT, None, later]


async def test_only_an_assistant_message_with_reasoning_carries_a_prefix(
    fixtures: Path,
) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."))
    run = await first_run(Engine(fixtures, models={"strong": llm}))
    prefixes = [m.prefix for m in run.conversation if m.role == "assistant"]
    assert prefixes[0] is not None and re.fullmatch(r"[0-9a-f]{16}", prefixes[0])
    assert prefixes[1] is None
    assert all(m.prefix is None for m in run.conversation if m.role != "assistant")


async def test_a_window_without_reasoning_is_never_chained(
    fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashed: list[str] = []
    extend, opening = chain.extend, chain.opening

    def extending(digest: bytes, message: Message) -> bytes:
        hashed.append("extend")
        return extend(digest, message)

    def opened(*args: Any) -> bytes:
        hashed.append("opening")
        return opening(*args)

    monkeypatch.setattr(chain, "extend", extending)
    monkeypatch.setattr(chain, "opening", opened)
    history = [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}.")
        for i in range(40)
    ]
    llm = Scripted(text_reply("Refund in full."))
    run = await Engine(fixtures, models={"strong": llm}).run(
        "support", "acme", "Charged twice.", history=history, context=CONTEXT
    )
    assert isinstance(run, Generation) and run.reply == "Refund in full."
    assert hashed == ["opening"]


PNG = b"\x89PNG\r\n\x1a\n"


async def sent_again_after(fixtures: Path, shown: Part, swapped: Part) -> list[Any]:
    llm = Scripted(reading(), text_reply("Refund in full."), text_reply("ok"), text_reply("ok"))
    engine = Engine(fixtures, models={"strong": llm})
    first = await engine.run(
        "support", "acme", [Text(text="Charged twice."), shown], context=CONTEXT
    )
    assert isinstance(first, Generation)
    same = await engine.run(
        "support", "acme", "Which?", history=first.conversation, context=CONTEXT
    )
    opening = first.conversation[0].model_copy(
        update={"content": [Text(text="Charged twice."), swapped]}
    )
    moved = await engine.run(
        "support",
        "acme",
        "Which?",
        history=[opening, *first.conversation[1:]],
        context=CONTEXT,
    )
    assert isinstance(same, Generation) and isinstance(moved, Generation)
    assert same.messages[1].reasoning == llm.calls[2][1][1].reasoning
    assert moved.messages[1].reasoning == llm.calls[3][1][1].reasoning
    return [same.messages[1].reasoning, moved.messages[1].reasoning]


async def test_reasoning_after_an_image_goes_back_while_its_bytes_are_the_same(
    fixtures: Path,
) -> None:
    sent = await sent_again_after(
        fixtures,
        Image(media_type="image/png", data=PNG),
        Image(media_type="image/png", data=PNG + b"\0"),
    )
    assert sent == [THOUGHT, None]


async def test_reasoning_after_an_image_by_url_goes_back(fixtures: Path) -> None:
    sent = await sent_again_after(
        fixtures,
        Image(media_type="image/png", url="https://files.acme.com/a.png"),
        Image(media_type="image/png", url="https://files.acme.com/b.png"),
    )
    assert sent == [THOUGHT, None]


async def test_hashing_never_base64_encodes_media(
    fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("chain: media was base64-encoded")

    monkeypatch.setattr(base64, "b64encode", refused)
    history = [
        Message(
            role="user",
            content=[
                Text(text="Charged twice."),
                Image(media_type="image/png", data=PNG),
                Document(media_type="application/pdf", data=b"%PDF-1.7", name="invoice.pdf"),
            ],
        ),
        Message(role="assistant", content="Checking.", reasoning=THOUGHT),
    ]
    llm = Scripted(text_reply("Refund in full.", reasoning=THOUGHT))
    run = await Engine(fixtures, models={"strong": llm}).run(
        "support", "acme", "Which?", history=history, context=CONTEXT
    )
    assert isinstance(run, Generation) and run.conversation[-1].prefix is not None


def long_window() -> list[Message]:
    window: list[Message] = []
    for i in range(500):
        window.append(Message(role="user", content=f"Question {i}."))
        window.append(Message(role="assistant", content=f"Answer {i}.", reasoning=THOUGHT))
    return window


def slowed(monkeypatch: pytest.MonkeyPatch) -> None:
    extend = chain.extend

    def slow(digest: bytes, message: Message) -> bytes:
        time.sleep(0.00005)
        return extend(digest, message)

    monkeypatch.setattr(chain, "extend", slow)


async def test_a_long_window_with_reasoning_gives_the_loop_turns(
    fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slowed(monkeypatch)
    ticks = 0
    seen: list[int] = []

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    class Counting(Scripted):
        async def generate(
            self, system: str, messages: list[Message], tools: list[ToolSpec]
        ) -> Reply:
            seen.append(ticks)
            return await super().generate(system, messages, tools)

    engine = Engine(fixtures, models={"strong": Counting(text_reply("ok"))})
    ticking = asyncio.create_task(ticker())
    try:
        run = await engine.run("support", "acme", "Which?", history=long_window())
    finally:
        ticking.cancel()
    assert isinstance(run, Generation)
    ticks_at_first_call = seen[0]
    assert ticks_at_first_call >= 5


async def test_a_window_of_large_images_gives_the_loop_turns_while_it_is_hashed(
    fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = 0
    marks: list[int] = []
    opening = chain.opening

    def opened(*args: Any) -> bytes:
        marks.append(ticks)
        return opening(*args)

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    class Counting(Scripted):
        async def generate(
            self, system: str, messages: list[Message], tools: list[ToolSpec]
        ) -> Reply:
            marks.append(ticks)
            return await super().generate(system, messages, tools)

    monkeypatch.setattr(chain, "opening", opened)
    photo = Image(media_type="image/png", data=PNG + bytes(2 * 1024 * 1024))
    history: list[Message] = []
    for i in range(20):
        history.append(Message(role="user", content=[Text(text=f"Photo {i}."), photo]))
        history.append(Message(role="assistant", content=f"Seen {i}.", reasoning=THOUGHT))
    engine = Engine(fixtures, models={"strong": Counting(text_reply("ok"))})
    ticking = asyncio.create_task(ticker())
    try:
        run = await engine.run("support", "acme", "Which?", history=history)
    finally:
        ticking.cancel()
    assert isinstance(run, Generation)
    hashing_started, model_called = marks
    assert model_called - hashing_started >= 3


async def test_cancelling_a_run_while_its_window_is_hashed_raises_at_once(
    fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slowed(monkeypatch)
    llm = Scripted(text_reply("ok"))
    engine = Engine(fixtures, models={"strong": llm})
    task = asyncio.create_task(engine.run("support", "acme", "Which?", history=long_window()))
    for _ in range(3):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.calls == []


async def test_reasoning_goes_back_within_a_run_and_to_a_run_with_the_same_history(
    fixtures: Path,
) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."), text_reply("The second one."))
    engine = Engine(fixtures, models={"strong": llm})
    first = await first_run(engine)
    assert llm.calls[1][1][1].reasoning == THOUGHT
    second = await engine.run(
        "support", "acme", "Which charge?", history=first.conversation, context=CONTEXT
    )
    assert isinstance(second, Generation)
    assert llm.calls[2][1][1].reasoning == THOUGHT and second.messages[1].reasoning == THOUGHT


async def test_a_run_stamps_on_from_where_settle_stopped_and_both_reasonings_go_back(
    fixtures: Path,
) -> None:
    llm = Scripted(
        text_reply("One.", reasoning={"r": 1}),
        text_reply("Two."),
        text_reply("Three.", reasoning={"r": 3}),
        text_reply("Four."),
        text_reply("Five."),
    )
    engine = Engine(fixtures, models={"strong": llm})
    conversation: list[Message] = []
    for said in ["u1", "u2", "u3"]:
        run = await engine.run("support", "acme", said, history=conversation)
        assert isinstance(run, Generation)
        conversation = run.conversation
    assert [m.prefix is not None for m in conversation if m.role == "assistant"] == [
        True,
        False,
        True,
    ]
    await engine.run("support", "acme", "u4", history=conversation)
    sent = [m.reasoning for m in llm.calls[3][1] if m.role == "assistant"]
    assert sent == [{"r": 1}, None, {"r": 3}]
    edited = list(conversation)
    edited[2] = edited[2].model_copy(update={"content": "u2 edited"})
    await engine.run("support", "acme", "u5", history=edited)
    sent = [m.reasoning for m in llm.calls[4][1] if m.role == "assistant"]
    assert sent == [{"r": 1}, None, None]


async def test_reasoning_is_left_out_once_what_came_before_it_changed(fixtures: Path) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."), *[text_reply("ok") for _ in range(4)])
    engine = Engine(fixtures, models={"strong": llm})
    history = (await first_run(engine)).conversation
    extra = Tools([Tool("extra", "An extra tool", body=now)])
    runs = [
        await engine.run("support", "acme", "Which?", history=history, context="Acme is free."),
        await engine.run(
            "support", "acme", "Which?", history=history, context=CONTEXT, task="triage"
        ),
        await engine.run(
            "support", "acme", "Which?", history=history, context=CONTEXT, tools=extra
        ),
        await engine.run("billing", "acme", "Which?", history=history, context=CONTEXT),
    ]
    for run in runs:
        assert isinstance(run, Generation)
        assert run.messages[1].reasoning is None
        assert run.messages[1].content == "Checking."
        assert run.messages[1].tool_calls == history[1].tool_calls
    assert all(call[1][1].reasoning is None for call in llm.calls[2:])


async def test_reasoning_is_left_out_of_a_history_cut_short(fixtures: Path) -> None:
    thinking = Reply(text="The second one.", reasoning={"thought": "the later charge"})
    llm = Scripted(reading(), text_reply("Refund in full."), thinking, text_reply("ok"))
    engine = Engine(fixtures, models={"strong": llm})
    first = await first_run(engine)
    second = await engine.run(
        "support", "acme", "Which charge?", history=first.conversation, context=CONTEXT
    )
    assert isinstance(second, Generation)
    assert second.conversation[-1].reasoning == {"thought": "the later charge"}
    third = await engine.run(
        "support", "acme", "Thanks.", history=second.conversation[4:], context=CONTEXT
    )
    assert isinstance(third, Generation)
    assert third.messages[1].reasoning is None and llm.calls[3][1][1].reasoning is None


async def test_reasoning_without_a_prefix_of_its_own_is_left_out(fixtures: Path) -> None:
    llm = Scripted(text_reply("ok"), text_reply("ok"))
    engine = Engine(fixtures, models={"strong": llm})
    for prefix in (None, "0000000000000000"):
        history = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello", reasoning=THOUGHT, prefix=prefix),
        ]
        run = await engine.run("support", "acme", "Charged twice.", history=history)
        assert isinstance(run, Generation) and run.messages[1].reasoning is None
    assert all(call[1][1].reasoning is None for call in llm.calls)


class Foreign(BaseModel):
    role: str
    content: str
    reasoning: Any = None


async def test_a_message_from_another_library_goes_without_its_reasoning(fixtures: Path) -> None:
    llm = Scripted(text_reply("ok"))
    history = [
        Foreign(role="user", content="hi"),
        Foreign(role="assistant", content="hello", reasoning=THOUGHT),
    ]
    run = await Engine(fixtures, models={"strong": llm}).run(
        "support", "acme", "Charged twice.", history=history
    )
    assert isinstance(run, Generation)
    assert run.messages[1] == Message(role="assistant", content="hello")


async def test_the_model_receives_messages_without_the_engines_record(fixtures: Path) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."))
    run = await first_run(Engine(fixtures, models={"strong": llm}))
    assert all(m.prefix is None and m.usage is None for call in llm.calls for m in call[1])
    assert run.conversation[1].prefix is not None and run.conversation[1].usage is not None


class Named(Scripted):
    def __init__(self, name: str, *replies: Reply) -> None:
        super().__init__(*replies)
        self.model = name


async def test_reasoning_is_left_out_when_the_model_is_another(fixtures: Path) -> None:
    old = Named("model-a", reading(), text_reply("Refund in full."))
    history = (await first_run(Engine(fixtures, models={"strong": old}))).conversation
    new = Named("model-b", text_reply("ok"))
    run = await Engine(fixtures, models={"strong": new}).run(
        "support", "acme", "Which?", history=history, context=CONTEXT
    )
    assert isinstance(run, Generation) and run.messages[1].reasoning is None
    same = Named("model-a", text_reply("ok"))
    run = await Engine(fixtures, models={"strong": same}).run(
        "support", "acme", "Which?", history=history, context=CONTEXT
    )
    assert isinstance(run, Generation) and run.messages[1].reasoning == THOUGHT


async def test_reasoning_goes_back_when_a_stored_history_came_back_with_its_keys_reordered(
    fixtures: Path,
) -> None:
    unsorted = Reply(
        text="Checking.",
        tool_calls=[ToolCall(id="c1", name="Read", args={"path": "rules/refunds.md", "a": None})],
        stop_reason="tool_use",
        reasoning={"z": 1, "a": 2},
    )
    llm = Scripted(unsorted, text_reply("Refund in full."), text_reply("ok"))
    engine = Engine(fixtures, models={"strong": llm})
    first = await first_run(engine)
    stored = [
        Message.model_validate(json.loads(json.dumps(m.model_dump(mode="json"), sort_keys=True)))
        for m in first.conversation
    ]
    run = await engine.run("support", "acme", "Which?", history=stored, context=CONTEXT)
    assert isinstance(run, Generation)
    assert run.messages[1].reasoning == {"z": 1, "a": 2}
    assert llm.calls[2][1][1].reasoning == {"z": 1, "a": 2}


async def test_a_usage_or_prefix_from_elsewhere_reads_as_none(fixtures: Path) -> None:
    history = [
        {"role": "user", "content": "Where is A-1?"},
        {
            "role": "assistant",
            "content": "On its way.",
            "prefix": "not-a-mark",
            "usage": {"input": 3, "output": 1, "total": 4},
            "reasoning": THOUGHT,
        },
    ]
    engine = Engine(fixtures, models={"strong": Scripted(text_reply("Still."))})
    run = await engine.run("support", "acme", "And now?", history=history)
    assert isinstance(run, Generation)
    answered = run.messages[1]
    assert answered.prefix is None and answered.usage is None and answered.reasoning is None
    kept = Message.model_validate(
        {"role": "assistant", "prefix": "0123456789abcdef", "usage": {"input": 3}}
    )
    assert kept.prefix == "0123456789abcdef" and kept.usage == Usage(input=3)


async def test_reasoning_is_left_out_once_a_preloaded_file_changed(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    agent = root / "support" / "AGENT.md"
    agent.write_text(
        agent.read_text(encoding="utf-8").replace(
            "agents: [triage]\n", "agents: [triage]\npreload: true\n"
        ),
        encoding="utf-8",
    )
    llm = Scripted(reading(), text_reply("Refund in full."), text_reply("ok"), text_reply("ok"))
    first = await first_run(Engine(root, models={"strong": llm}))
    same = await Engine(root, models={"strong": llm}).run(
        "support", "acme", "Which?", history=first.conversation, context=CONTEXT
    )
    (root / "support" / "rules" / "refunds.md").write_text(
        "A duplicate charge is refunded within 30 days.\n", encoding="utf-8"
    )
    moved = await Engine(root, models={"strong": llm}).run(
        "support", "acme", "Which?", history=first.conversation, context=CONTEXT
    )
    assert isinstance(same, Generation) and isinstance(moved, Generation)
    assert same.messages[1].reasoning == THOUGHT and moved.messages[1].reasoning is None


async def test_a_message_that_arrives_mid_run_is_under_the_next_prefix(fixtures: Path) -> None:
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    inbox.put_nowait("It was order A-2, not A-1.")
    noted = Reply(text="Noted.", reasoning=THOUGHT, stop_reason="end_turn")
    llm = Scripted(reading(), noted, text_reply("ok"))
    engine = Engine(fixtures, models={"strong": llm})
    first = await engine.run("support", "acme", "Charged twice.", context=CONTEXT, inbox=inbox)
    assert isinstance(first, Generation)
    assert [m.role for m in first.conversation] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    second = await engine.run(
        "support", "acme", "Thanks.", history=first.conversation, context=CONTEXT
    )
    assert isinstance(second, Generation) and second.messages[4].reasoning == THOUGHT


class Thinking(Scripted):
    def __init__(self, *replies: Reply, thoughts: list[str], pressed: Stop | None = None) -> None:
        super().__init__(*replies)
        self.thoughts = thoughts
        self.pressed = pressed

    async def generate_stream(
        self, system: str, messages: list[Message], tools: list[ToolSpec], on_delta: OnDelta
    ) -> Reply:
        reply = await self.generate(system, messages, tools)
        for thought in self.thoughts:
            await on_delta(thought, reasoning=True)
            await asyncio.sleep(0)
        if self.pressed is not None:
            await asyncio.sleep(30)
        await on_delta(reply.text)
        return reply


class Plain(Scripted):
    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], Awaitable[None]],
    ) -> Reply:
        reply = await self.generate(system, messages, tools)
        await on_delta(reply.text)
        return reply


class Seen:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)


async def test_reasoning_streams_as_its_own_events_before_the_text(fixtures: Path) -> None:
    llm = Thinking(text_reply("Refund in full."), thoughts=["The rule says ", "refund."])
    seen = Seen()
    run = await Engine(fixtures, models={"strong": llm}).run(
        "support", "acme", "Charged twice.", on_event=seen
    )
    assert isinstance(run, Generation) and run.reply == "Refund in full."
    assert seen.events[1:] == [
        ReasoningDelta(run_id=run.run_id, agent="support", turn=0, text="The rule says "),
        ReasoningDelta(run_id=run.run_id, agent="support", turn=0, text="refund."),
        TextDelta(run_id=run.run_id, agent="support", turn=0, text="Refund in full."),
    ]
    assert all(m.content != "The rule says " for m in run.conversation)


async def test_an_adapter_that_streams_no_reasoning_works_as_before(fixtures: Path) -> None:
    seen = Seen()
    run = await Engine(fixtures, models={"strong": Plain(text_reply("Hi."))}).run(
        "support", "acme", "Hi", on_event=seen
    )
    assert isinstance(run, Generation) and run.reply == "Hi."
    assert [type(e) for e in seen.events] == [RunStarted, TextDelta]


async def test_a_stop_during_reasoning_keeps_no_reasoning_and_no_text(fixtures: Path) -> None:
    stop = Stop()

    class Pressing(Seen):
        async def __call__(self, event: Event) -> None:
            await super().__call__(event)
            if isinstance(event, ReasoningDelta):
                stop.now()

    llm = Thinking(text_reply("never"), thoughts=["Thinking"], pressed=stop)
    run = await asyncio.wait_for(
        Engine(fixtures, models={"strong": llm}).run(
            "support", "acme", "Charged twice.", on_event=Pressing(), stop=stop
        ),
        5,
    )
    assert isinstance(run, Generation) and run.stopped == "now" and run.reply == ""
    assert [m.role for m in run.conversation] == ["user"]


async def test_a_reply_that_is_not_streamed_sends_no_reasoning(fixtures: Path) -> None:
    seen = Seen()
    llm = Scripted(text_reply("Refund in full.", reasoning=THOUGHT))
    await Engine(fixtures, models={"strong": llm}).run(
        "support", "acme", "Charged twice.", on_event=seen
    )
    assert not any(isinstance(e, ReasoningDelta) for e in seen.events)


class Tuned(Scripted):
    def __init__(self, settings: dict[str, Any], *replies: Reply) -> None:
        super().__init__(*replies)
        self.settings = settings


async def test_changed_settings_keep_reasoning_going_back(fixtures: Path) -> None:
    low = Tuned({"effort": "low"}, reading(), text_reply("Refund in full."))
    first = await first_run(Engine(fixtures, models={"strong": low}))
    assert [m.settings for m in first.conversation if m.role == "assistant"] == [
        {"effort": "low"},
        {"effort": "low"},
    ]
    high = Tuned({"effort": "high"}, text_reply("ok"))
    await Engine(fixtures, models={"strong": high}).run(
        "support", "acme", "Thanks.", history=first.conversation, context=CONTEXT
    )
    sent = [m.reasoning for m in high.calls[0][1] if m.role == "assistant"]
    assert sent == [THOUGHT, None]
    assert all(m.settings is None for m in high.calls[0][1])
