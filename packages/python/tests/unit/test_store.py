import asyncio
import copy
import json
import logging
import re
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

import rfnry_engine
from rfnry_engine import RunEnded, Stop
from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError, ModelError, RunawayError
from rfnry_engine.models import Message, Reply, Text, ToolCall, ToolSpec, Usage
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import (
    URGENT,
    Answering,
    Crew,
    MemoryStore,
    Scripted,
    Seen,
    background,
    crew,
    released,
    said,
    team,
    text_reply,
)
from .test_failed_runs import Breaking, Kept
from .trees import provided, write_folder

KEY = ("acme", "c1")
Body = Callable[[dict[str, Any]], Awaitable[Any]]


def exactly(text: str) -> str:
    return f"^{re.escape(text)}$"


def folder(root: Path) -> Path:
    return write_folder(
        root,
        {
            "chat": {
                "kind": "generation",
                "model": "strong",
                "description": "Talks",
                "instructions": "chat.md",
                "tools": {"allow": ["slow"]},
            },
            "desk": {
                "kind": "decision",
                "model": "judge",
                "description": "Sorts",
                "instructions": "desk.md",
                "questions": ["desk/*.json"],
            },
        },
        runtime_tools=["slow"],
        files={"chat.md": "Talk.\n", "desk.md": "", "desk/urgent.json": URGENT},
    )


def engine_for(tmp_path: Path, llm: Any, **options: Any) -> Engine:
    return Engine(
        folder(tmp_path), providers=provided({"strong": llm, "judge": Answering()}), **options
    )


async def done(args: dict[str, Any]) -> str:
    return "done"


def slow(body: Body = done) -> Tools:
    return Tools([Tool("slow", "Works slowly", body=body)])


def use(call_id: str = "c1") -> Reply:
    return Reply(
        tool_calls=[ToolCall(id=call_id, name="slow", args={})],
        stop_reason="tool_use",
        usage=Usage(input=3, output=1),
    )


async def chat(
    engine: Engine, store: MemoryStore, input: str = "work", *, body: Body = done, **options: Any
) -> Generation:
    run = await engine.run(
        "chat", "acme", input, tools=slow(body), store=store, conversation="c1", **options
    )
    assert isinstance(run, Generation)
    return run


def shape(messages: list[Message]) -> list[tuple[str, object, list[str], str | None]]:
    return [
        (
            message.role,
            message.content,
            [call.id for call in message.tool_calls],
            message.tool_call_id,
        )
        for message in messages
    ]


async def test_store_and_conversation_go_together(tmp_path: Path) -> None:
    llm = Scripted()
    engine = engine_for(tmp_path, llm)
    alone = "engine: store= needs conversation=, the name the store keeps the conversation under"
    with pytest.raises(InputError, match=exactly(alone)):
        await engine.run("chat", "acme", "hi", tools=slow(), store=MemoryStore())
    with pytest.raises(
        InputError, match=exactly("engine: conversation= needs store=, the store that keeps it")
    ):
        await engine.run("chat", "acme", "hi", tools=slow(), conversation="c1")
    assert llm.calls == []


@pytest.mark.parametrize("given", [{"history": []}, {"start": 1}], ids=["history", "start"])
async def test_a_store_run_refuses_history_and_start(given: dict[str, Any], tmp_path: Path) -> None:
    store = MemoryStore()
    refused = "engine: a run with a store reads its history from it; leave out history= and start="
    with pytest.raises(InputError, match=exactly(refused)):
        await chat(engine_for(tmp_path, Scripted()), store, **given)
    assert store.log == []


@pytest.mark.parametrize("name", ["", "  ", 3])
async def test_a_conversation_name_is_non_blank_text(name: object, tmp_path: Path) -> None:
    store = MemoryStore()
    with pytest.raises(
        InputError, match=exactly(f"engine: conversation must be non-blank text; got {name!r}")
    ):
        await engine_for(tmp_path, Scripted()).run(
            "chat", "acme", "hi", tools=slow(), store=store, conversation=cast(str, name)
        )
    assert store.log == []


async def test_a_store_must_be_a_conversation_store(tmp_path: Path) -> None:
    assert isinstance(MemoryStore(), rfnry_engine.ConversationStore)
    wanted = "engine: store must have length(), read(), append() and window(); got object"
    with pytest.raises(InputError, match=exactly(wanted)):
        await engine_for(tmp_path, Scripted()).run(
            "chat", "acme", "hi", tools=slow(), store=cast(Any, object()), conversation="c1"
        )


async def test_a_new_conversation_is_stored_as_the_run_adds_it(tmp_path: Path) -> None:
    store = MemoryStore()
    run = await chat(engine_for(tmp_path, Scripted(use(), text_reply("Done."))), store)
    assert [message.role for message in run.conversation] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert store.held[KEY] == run.conversation
    assert all(
        kept is not given for kept, given in zip(store.held[KEY], run.conversation, strict=True)
    )
    assert store.held[KEY][1].usage == Usage(input=3, output=1)
    assert store.log[0] == ("length", "acme", "c1")
    assert all(scope == "acme" for _, scope, _ in store.log)
    assert ("read", "acme", "c1") not in store.log


class Opaque:
    def __init__(self) -> None:
        self.lock = threading.Lock()


async def test_a_store_run_copies_an_opaque_reasoning_as_a_run_without_a_store_does(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    opaque = Opaque()
    with pytest.raises(TypeError):
        copy.deepcopy(opaque)
    thought = Reply(
        tool_calls=[ToolCall(id="c1", name="slow", args={})],
        stop_reason="tool_use",
        reasoning=opaque,
    )
    llm = Scripted(thought, text_reply("Done."), thought, text_reply("Done."))
    engine, store = engine_for(tmp_path, llm), MemoryStore()
    alone = await engine.run("chat", "acme", "work", tools=slow())
    assert isinstance(alone, Generation)
    with caplog.at_level(logging.ERROR, logger="rfnry_engine.engine"):
        kept = await chat(engine, store)
    assert caplog.records == [] and kept.reply == alone.reply == "Done."
    assert said(kept.conversation) == said(alone.conversation)
    assert kept.conversation[1].reasoning is alone.conversation[1].reasoning is opaque
    assert store.held[KEY] == kept.conversation and store.held[KEY][1].reasoning is opaque
    assert [m.reasoning for m in llm.calls[3][1]] == [m.reasoning for m in llm.calls[1][1]]
    assert llm.calls[3][1][1].reasoning is opaque


class Watching(Scripted):
    def __init__(self, store: MemoryStore, *replies: Reply) -> None:
        super().__init__(*replies)
        self.store = store
        self.stored: list[list[Message]] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.stored.append(list(self.store.held.get(KEY, [])))
        return await super().generate(system, messages, tools)


async def test_every_message_a_model_is_sent_is_stored_before_the_call(tmp_path: Path) -> None:
    store, inbox = MemoryStore(), asyncio.Queue[Any]()
    llm = Watching(store, use("c1"), use("c2"), text_reply("Done."))

    async def sending(args: dict[str, Any]) -> str:
        inbox.put_nowait("also this")
        return "done"

    await chat(engine_for(tmp_path, llm), store, body=sending, inbox=inbox)
    assert len(llm.calls) == 3
    assert [shape(messages) for _, messages, _ in llm.calls] == [shape(held) for held in llm.stored]
    assert Message(role="user", content="also this") in llm.calls[1][1]


async def test_a_reply_is_stored_before_its_tools_run(tmp_path: Path) -> None:
    store = MemoryStore()
    seen: list[list[str]] = []

    async def looking(args: dict[str, Any]) -> str:
        seen.append([call.name for call in store.held[KEY][-1].tool_calls])
        return "done"

    await chat(engine_for(tmp_path, Scripted(use(), text_reply("Done."))), store, body=looking)
    assert seen == [["slow"]]


class Streamed(Scripted):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], Awaitable[None]],
    ) -> Reply:
        await on_delta("Let me ")
        await on_delta("look")
        self.entered.set()
        await asyncio.sleep(30)
        return text_reply("never")


class Verdict(BaseModel):
    refund: bool


def output(call_id: str, **args: Any) -> Reply:
    return Reply(
        tool_calls=[ToolCall(id=call_id, name="Output", args=args)], stop_reason="tool_use"
    )


Ending = tuple[Generation, MemoryStore]


async def ends_its_turn(tmp_path: Path) -> Ending:
    store = MemoryStore()
    run = await chat(engine_for(tmp_path, Scripted(text_reply("Hi."))), store, "hi")
    assert run.stop_reason == "end_turn"
    return run, store


async def stopped_while_streaming(tmp_path: Path) -> Ending:
    store, stop, llm = MemoryStore(), Stop(), Streamed()
    task = asyncio.create_task(chat(engine_for(tmp_path, llm), store, stop=stop, on_event=Seen()))
    await asyncio.wait_for(llm.entered.wait(), 5)
    stop.now()
    run = await asyncio.wait_for(task, 5)
    assert run.stopped == "now"
    assert run.conversation[-1] == Message(role="assistant", content="Let me look")
    return run, store


async def stopped_after_a_step(tmp_path: Path) -> Ending:
    store, stop = MemoryStore(), Stop()

    async def stopping(args: dict[str, Any]) -> str:
        stop.after_step()
        return "done"

    llm = Scripted(use(), text_reply("never"))
    run = await chat(engine_for(tmp_path, llm), store, body=stopping, stop=stop)
    assert run.stopped == "after_step" and len(llm.calls) == 1
    return run, store


async def output_refused_then_accepted(tmp_path: Path) -> Ending:
    store = MemoryStore()
    llm = Scripted(output("o1", refund="maybe"), output("o2", refund=True))
    run = await chat(engine_for(tmp_path, llm), store, expect=Verdict)
    assert [step.result == "accepted" for step in run.steps] == [False, True]
    return run, store


async def runs_away(tmp_path: Path) -> Ending:
    store, kept = MemoryStore(), Kept()
    engine = engine_for(tmp_path, Scripted(use("c1"), use("c2")), max_model_calls=2, refinery=kept)
    with pytest.raises(RunawayError):
        await chat(engine, store)
    [run] = kept.runs
    assert isinstance(run, Generation)
    return run, store


async def reads_its_inbox(tmp_path: Path) -> Ending:
    store, inbox = MemoryStore(), asyncio.Queue[Any]()

    async def sending(args: dict[str, Any]) -> str:
        inbox.put_nowait("also this")
        return "done"

    llm = Scripted(use(), text_reply("Both."))
    run = await chat(engine_for(tmp_path, llm), store, body=sending, inbox=inbox)
    assert run.conversation[-2] == Message(role="user", content="also this")
    return run, store


async def model_fails_after_a_tool_round(tmp_path: Path) -> Ending:
    store, kept = MemoryStore(), Kept()
    llm = Breaking(use(), error=ModelError("models: the vendor is down"))
    with pytest.raises(ModelError):
        await chat(engine_for(tmp_path, llm, refinery=kept), store)
    [run] = kept.runs
    assert isinstance(run, Generation) and run.error is not None
    return run, store


ENDINGS = [
    ends_its_turn,
    stopped_while_streaming,
    stopped_after_a_step,
    output_refused_then_accepted,
    runs_away,
    reads_its_inbox,
    model_fails_after_a_tool_round,
]


@pytest.mark.parametrize("ending", ENDINGS, ids=[ending.__name__ for ending in ENDINGS])
async def test_the_store_holds_the_conversation_however_the_run_ends(
    ending: Callable[[Path], Awaitable[Ending]], tmp_path: Path
) -> None:
    run, store = await ending(tmp_path)
    assert store.held[KEY] == run.conversation


async def test_the_store_holds_everything_before_run_ended(tmp_path: Path) -> None:
    store = MemoryStore()
    at_end: list[int] = []

    async def on_event(event: Any) -> None:
        if isinstance(event, RunEnded):
            at_end.append(len(store.held[KEY]))

    run = await chat(
        engine_for(tmp_path, Scripted(use(), text_reply("Done."))), store, on_event=on_event
    )
    assert at_end == [len(run.conversation)] and len(run.conversation) == 4


async def test_calling_again_with_the_same_name_continues_the_conversation(tmp_path: Path) -> None:
    store = MemoryStore()
    llm = Scripted(use(), text_reply("First."), text_reply("Then this."))
    engine = engine_for(tmp_path, llm)
    first = await chat(engine, store)
    second = await chat(engine, store, "and then?")
    asked = Message(role="user", content="and then?")
    assert llm.calls[2][1] == said([*first.conversation, asked])
    assert second.run_id != first.run_id and second.start == 0
    assert store.held[KEY] == [*first.conversation, *second.added] == second.conversation
    assert all(
        kept is not given for kept, given in zip(store.held[KEY], second.conversation, strict=True)
    )
    assert store.log.count(("read", "acme", "c1")) == 1


@pytest.mark.parametrize("method", ["append", "read"])
async def test_a_store_that_raises_fails_the_run_with_its_error(
    method: str, tmp_path: Path
) -> None:
    store, kept, seen, error = MemoryStore(), Kept(), Seen(), OSError("disk full")

    async def failing(args: dict[str, Any]) -> str:
        store.fail["append"] = error
        return "done"

    engine = engine_for(tmp_path, Scripted(use(), text_reply("Done.")), refinery=kept)
    if method == "read":
        await chat(engine, store)
        store.fail["read"] = error
    with pytest.raises(OSError) as caught:
        await chat(engine, store, body=failing, on_event=seen)
    assert caught.value is error
    assert store.log[-1] == (method, "acme", "c1")
    if method == "read":
        assert seen.events == [] and len(kept.runs) == 1
        return
    [failure] = kept.runs
    ended = [event.error for event in seen.events if isinstance(event, RunEnded)]
    assert failure.error == "OSError: disk full" and ended == [failure.error]


class Failing(Crew):
    def __init__(
        self,
        lead: list[Reply | asyncio.Event],
        helpers: dict[str, list[Reply | asyncio.Event]],
        failing: Callable[[], None] = lambda: None,
    ) -> None:
        super().__init__(lead, helpers)
        self.failing = failing

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        if self.marker in system and not self.replies["lead"]:
            self.calls["lead"].append(list(messages))
            self.failing()
            raise ValueError("the lead's model broke")
        return await super().generate(system, messages, tools)


async def test_a_failed_run_s_last_messages_are_stored(tmp_path: Path) -> None:
    gate, store, kept = asyncio.Event(), MemoryStore(), Kept()
    model = Failing([background("look")], {"look": [gate, text_reply("never")]})
    engine = team(crew(tmp_path, background=True), model, refinery=kept)
    with pytest.raises(ValueError, match=exactly("the lead's model broke")):
        await engine.run("lead", "acme", "go", store=store, conversation="c1")
    [failure] = kept.runs
    assert isinstance(failure, Generation)
    notice = store.held[KEY][-1].notice
    assert notice is not None and (notice.job, notice.kind) == ("d1", "stopped")
    assert store.held[KEY] == failure.conversation and not gate.is_set()


async def test_a_failed_run_s_messages_are_stored_before_its_run_ended(tmp_path: Path) -> None:
    gate, store, kept = asyncio.Event(), MemoryStore(), Kept()
    at_end: dict[str, int] = {}

    async def on_event(event: Any) -> None:
        if isinstance(event, RunEnded):
            at_end[event.run_id] = len(store.held.get(KEY, []))

    model = Failing([background("look")], {"look": [gate, text_reply("never")]})
    engine = team(crew(tmp_path, background=True), model, refinery=kept)
    with pytest.raises(ValueError, match=exactly("the lead's model broke")):
        await engine.run("lead", "acme", "go", store=store, conversation="c1", on_event=on_event)
    [failure] = kept.runs
    assert isinstance(failure, Generation) and failure.conversation[-1].notice is not None
    assert at_end[failure.run_id] == len(failure.conversation)


async def test_a_store_error_while_a_failed_run_ends_is_logged_and_the_run_s_error_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gate, store, kept, error = asyncio.Event(), MemoryStore(), Kept(), OSError("disk full")
    model = Failing(
        [background("look")],
        {"look": [gate, text_reply("never")]},
        failing=lambda: store.fail.update(append=error),
    )
    engine = team(crew(tmp_path, background=True), model, refinery=kept)
    with (
        caplog.at_level(logging.ERROR, logger="rfnry_engine.engine"),
        pytest.raises(ValueError, match=exactly("the lead's model broke")),
    ):
        await engine.run("lead", "acme", "go", store=store, conversation="c1")
    [failure] = kept.runs
    assert isinstance(failure, Generation)
    [line] = [record for record in caplog.records if "store" in record.getMessage()]
    assert line.levelno == logging.ERROR and line.name == "rfnry_engine.engine"
    assert line.getMessage() == f"engine: the store failed while run {failure.run_id} ended"
    assert line.exc_info is not None and line.exc_info[1] is error
    assert store.log[-1] == ("append", "acme", "c1")
    assert store.held[KEY] == failure.conversation[:-1]
    assert failure.conversation[-1].notice is not None


class Miscounted(MemoryStore):
    def __init__(self, by: int) -> None:
        super().__init__()
        self.by = by
        self.held[KEY] = [
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        ]

    async def length(self, scope: str, conversation: str) -> int:
        return await super().length(scope, conversation) + self.by


async def test_a_store_that_reads_back_less_than_its_length_is_refused(tmp_path: Path) -> None:
    store, kept, seen, llm = Miscounted(2), Kept(), Seen(), Scripted(text_reply("never"))
    short = "conversation: the store holds 5 messages of conversation 'c1' but read returned 3"
    with pytest.raises(InputError, match=exactly(short)):
        await chat(engine_for(tmp_path, llm, refinery=kept), store, on_event=seen)
    assert seen.events == [] and kept.runs == [] and llm.calls == []


async def test_a_store_that_reads_back_more_than_its_length_is_refused(tmp_path: Path) -> None:
    store, kept, seen, llm = Miscounted(-1), Kept(), Seen(), Scripted(text_reply("never"))
    more = "conversation: the store holds 2 messages of conversation 'c1' but read returned 3"
    with pytest.raises(InputError, match=exactly(more)):
        await chat(engine_for(tmp_path, llm, refinery=kept), store, on_event=seen)
    assert seen.events == [] and kept.runs == [] and llm.calls == []


async def test_stored_messages_that_do_not_fit_are_refused(tmp_path: Path) -> None:
    store, kept, seen, llm = MemoryStore(), Kept(), Seen(), Scripted(text_reply("never"))
    store.held[KEY] = cast(list[Message], [{"role": "nobody"}])
    with pytest.raises(ValidationError) as invalid:
        TypeAdapter(list[Message]).validate_python([{"role": "nobody"}])
    errors = invalid.value.errors(include_url=False, include_input=False)
    unfit = f"conversation: the store's conversation 'c1' does not fit the contract: {errors}"
    with pytest.raises(InputError, match=exactly(unfit)):
        await chat(engine_for(tmp_path, llm, refinery=kept), store, on_event=seen)
    assert seen.events == [] and kept.runs == [] and llm.calls == []


class Measured(MemoryStore):
    def __init__(self, measured: object) -> None:
        super().__init__()
        self.measured = measured

    async def length(self, scope: str, conversation: str) -> int:
        self.called("length", scope, conversation)
        return cast(int, self.measured)


@pytest.mark.parametrize("length", [-1, True, "3", 2.0])
async def test_a_store_length_that_is_not_a_whole_number_is_refused(
    length: object, tmp_path: Path
) -> None:
    store, seen, llm = Measured(length), Seen(), Scripted(text_reply("never"))
    wrong = f"conversation: the store's length of conversation 'c1' must be a whole number from 0; got {length!r}"
    with pytest.raises(InputError, match=exactly(wrong)):
        await chat(engine_for(tmp_path, llm), store, on_event=seen)
    assert seen.events == [] and llm.calls == [] and store.log == [("length", "acme", "c1")]


class Held:
    def __init__(self) -> None:
        self.gate, self.entered = asyncio.Event(), asyncio.Event()

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        heard = messages[-1].content
        if heard == "one":
            self.entered.set()
            await self.gate.wait()
        return text_reply(f"Heard {heard}.")


async def test_one_conversation_runs_once_at_a_time(tmp_path: Path) -> None:
    llm, store, seen = Held(), MemoryStore(), Seen()
    engine = engine_for(tmp_path, llm)
    first = asyncio.create_task(chat(engine, store, "one"))
    await asyncio.wait_for(llm.entered.wait(), 5)
    in_use = "engine: conversation 'c1' of scope 'acme' is in use by a run in flight"
    for _ in range(2):
        with pytest.raises(InputError, match=exactly(in_use)):
            await chat(engine, store, "two", on_event=seen)
    assert seen.events == [] and store.log.count(("length", "acme", "c1")) == 1
    elsewhere = await engine.run(
        "chat", "other", "three", tools=slow(), store=store, conversation="c1"
    )
    assert isinstance(elsewhere, Generation) and elsewhere.reply == "Heard three."
    llm.gate.set()
    await asyncio.wait_for(first, 5)
    again = await chat(engine, store, "four")
    assert [message.content for message in store.held[KEY]] == [
        "one",
        "Heard one.",
        "four",
        "Heard four.",
    ]
    assert store.held[KEY] == again.conversation
    assert store.held[("other", "c1")] == elsewhere.conversation


async def test_a_decision_run_makes_no_store_call(tmp_path: Path) -> None:
    store = MemoryStore()
    run = await engine_for(tmp_path, Scripted()).run(
        "desk", "acme", "It is down", store=store, conversation="c1"
    )
    assert isinstance(run, Decision) and store.log == [] and store.held == {}


def counting(*windows: object) -> tuple[MemoryStore, list[int]]:
    asked: list[int] = []

    def window(conversation: str, length: int) -> int:
        asked.append(length)
        return cast(int, windows[min(len(asked), len(windows)) - 1])

    return MemoryStore(window), asked


def earlier() -> list[Message]:
    return [
        Message(role="user", content="u0"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="u2"),
        Message(role="assistant", content="a3"),
    ]


class Ordered(Scripted):
    def __init__(self, order: list[object], *replies: Reply) -> None:
        super().__init__(*replies)
        self.order = order

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.order.append("call")
        return await super().generate(system, messages, tools)


async def test_the_store_is_asked_for_the_window_before_every_model_call(tmp_path: Path) -> None:
    order: list[object] = []

    def window(conversation: str, length: int) -> int:
        order.append((length, len(store.held[KEY]), store.log[-2][0]))
        return 0

    store = MemoryStore(window)
    llm = Ordered(order, use("c1"), use("c2"), text_reply("Done."))
    await chat(engine_for(tmp_path, llm), store)
    assert order == [(1, 1, "append"), "call", (3, 3, "append"), "call", (5, 5, "append"), "call"]
    assert store.log.count(("window", "acme", "c1")) == 3


async def test_a_later_window_sends_only_the_messages_from_its_start(tmp_path: Path) -> None:
    store, asked = counting(0, 0, 4)
    store.held[KEY] = earlier()
    llm = Watching(store, use("c1"), use("c2"), text_reply("Done."))
    run = await chat(engine_for(tmp_path, llm), store)
    assert asked == [5, 7, 9]
    assert [messages for _, messages, _ in llm.calls[:2]] == [said(held) for held in llm.stored[:2]]
    assert llm.calls[2][1] == said(llm.stored[2][4:])
    assert llm.calls[2][1][0] == Message(role="user", content="work")
    assert run.start == 0 and store.held[KEY] == run.conversation


@pytest.mark.parametrize(
    "window", [-1, "length", True, "3", 2.0], ids=["negative", "length", "bool", "text", "float"]
)
async def test_a_window_that_is_not_a_position_fails_the_run(
    window: object, tmp_path: Path
) -> None:
    store, asked = counting(0, 3 if window == "length" else window)
    kept, llm = Kept(), Scripted(use(), text_reply("never"))
    shown = 3 if window == "length" else window
    wrong = f"conversation: the store's window for conversation 'c1' must be a position from 0 to 2; got {shown!r}"
    with pytest.raises(InputError, match=exactly(wrong)):
        await chat(engine_for(tmp_path, llm, refinery=kept), store)
    assert asked == [1, 3] and len(llm.calls) == 1
    [failure] = kept.runs
    assert failure.error == f"InputError: {wrong}"


async def test_a_window_that_breaks_a_tool_call_from_its_result_fails_the_run(
    tmp_path: Path,
) -> None:
    store, asked = counting(0, 3)
    store.held[KEY] = [Message(role="user", content="earlier")]
    llm = Scripted(use("call_1"), text_reply("never"))
    broken = (
        "engine: the window of conversation 'c1' breaks tool call 'call_1' from its result "
        "at position 3; start at a message the person sent"
    )
    with pytest.raises(InputError, match=exactly(broken)):
        await chat(engine_for(tmp_path, llm), store)
    assert asked == [2, 4] and len(llm.calls) == 1


def names(tools: list[ToolSpec]) -> list[str]:
    return [tool.name for tool in tools]


async def test_a_store_run_is_offered_history_from_its_first_call(tmp_path: Path) -> None:
    replies = [use("c1"), use("c2"), text_reply("Done."), text_reply("Alone."), text_reply("Past.")]
    llm = Scripted(*replies)
    engine = engine_for(tmp_path, llm)
    stored = await chat(engine, MemoryStore())
    alone = await engine.run("chat", "acme", "work", tools=slow())
    past = await engine.run(
        "chat",
        "acme",
        "work",
        tools=slow(),
        history=[Message(role="user", content="earlier")],
        start=1,
    )
    first = llm.calls[0][2]
    assert "History" in names(first)
    assert llm.calls[1][2] == first and llm.calls[2][2] == first
    assert "History" not in names(llm.calls[3][2])
    assert llm.calls[4][2] == first
    assert isinstance(alone, Generation) and isinstance(past, Generation)
    assert stored.fingerprint.tools == past.fingerprint.tools != alone.fingerprint.tools


def reading(*calls: tuple[str, dict[str, Any]]) -> Reply:
    return Reply(
        tool_calls=[ToolCall(id=call_id, name="History", args=args) for call_id, args in calls],
        stop_reason="tool_use",
    )


def positions(result: object) -> list[int]:
    assert isinstance(result, list)
    headers = [
        part.text
        for part in result
        if isinstance(part, Text) and part.text.startswith('{"position"')
    ]
    return [json.loads(header)["position"] for header in headers]


async def test_history_reads_what_the_window_left_behind(tmp_path: Path) -> None:
    store, asked = counting(4, 5)
    store.held[KEY] = earlier()
    llm = Scripted(
        reading(("h1", {}), ("h2", {"before": 10})), reading(("h3", {})), text_reply("Done.")
    )
    run = await chat(engine_for(tmp_path, llm), store)
    assert asked == [5, 8, 10]
    results = [message.content for message in store.held[KEY] if message.role == "tool"]
    assert [positions(result) for result in results] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
    ]
    assert run.start == 4


THOUGHT = {"thought": "look it up"}


def thinking(call_id: str = "c1") -> Reply:
    return Reply(
        tool_calls=[ToolCall(id=call_id, name="slow", args={})],
        stop_reason="tool_use",
        reasoning=THOUGHT,
    )


async def test_reasoning_goes_back_while_the_window_holds_still(tmp_path: Path) -> None:
    store = MemoryStore()
    llm = Scripted(thinking(), text_reply("Done."))
    run = await chat(engine_for(tmp_path, llm), store)
    assert run.conversation[1].prefix is not None
    assert llm.calls[1][1][1].reasoning == THOUGHT


async def test_reasoning_is_not_sent_once_the_window_moves_under_it(tmp_path: Path) -> None:
    store, asked = counting(0, 2)
    store.held[KEY] = [Message(role="user", content="u0"), Message(role="assistant", content="a1")]
    llm = Scripted(thinking("c1"), thinking("c2"), text_reply("Done."), text_reply("Again."))
    engine = engine_for(tmp_path, llm)
    run = await chat(engine, store, "u2")
    held = store.held[KEY]
    moved = llm.calls[1][1]
    assert said(moved) == said([held[2], held[3].model_copy(update={"reasoning": None}), held[4]])
    assert moved[1].reasoning is None and held[3].reasoning == THOUGHT
    still = llm.calls[2][1]
    assert [message.reasoning for message in still] == [None, None, None, THOUGHT, None]
    assert run.conversation[3].reasoning == THOUGHT and run.messages == held[:3]
    await chat(engine, store, "u8")
    resumed = [message.reasoning for message in llm.calls[3][1]]
    assert resumed == [None, None, None, THOUGHT, None, None, None]


async def test_a_resumed_conversation_sends_back_the_reasoning_its_prefix_still_holds(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    llm = Scripted(thinking(), text_reply("Done."), text_reply("Again."), text_reply("Changed."))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    engine = engine_for(tmp_path / "a", llm)
    await chat(engine, store)
    await chat(engine, store, "and then?")
    assert llm.calls[2][1][1].reasoning == THOUGHT
    changed = folder(tmp_path / "b")
    (changed / "chat.md").write_text("Talk briefly.\n")
    other = Engine(changed, providers=provided({"strong": llm, "judge": Answering()}))
    await chat(other, store, "and now?")
    assert llm.calls[3][1][1].reasoning is None and store.held[KEY][1].reasoning == THOUGHT


async def test_a_store_run_records_where_its_first_window_began(
    tmp_path: Path, storage: Storage
) -> None:
    store = MemoryStore(lambda conversation, length: 2)
    store.held[KEY] = [
        *earlier(),
        Message(role="user", content="u4"),
        Message(role="assistant", content="a5"),
    ]
    refinery = Refinery(storage)
    engine = engine_for(tmp_path, Scripted(use(), text_reply("Done.")), refinery=refinery)
    run = await chat(engine, store)
    held = store.held[KEY]
    assert run.start == 2 and run.messages == held[2:7]
    assert run.conversation == held[2:] and run.added == held[6:]
    back = await refinery.run("acme", run.run_id)
    assert isinstance(back, Generation)
    assert (back.start, back.messages, back.conversation) == (2, held[2:7], held[2:])


async def test_job_ids_count_the_whole_stored_conversation(tmp_path: Path) -> None:
    def delegated(call_id: str, job: str) -> list[Message]:
        args = {"agent": "worker", "input": "a", "background": True}
        return [
            Message(
                role="assistant", tool_calls=[ToolCall(id=call_id, name="Delegate", args=args)]
            ),
            Message(role="tool", tool_call_id=call_id, content=json.dumps({"job": job})),
            Message(role="assistant", content="ok"),
        ]

    store = MemoryStore(lambda conversation, length: 10)
    store.held[KEY] = [
        Message(role="user", content="earlier"),
        *delegated("x1", "d1"),
        *delegated("x2", "d2"),
        *delegated("x3", "d3"),
        Message(role="user", content="later"),
        Message(role="assistant", content="noted"),
    ]
    gate = asyncio.Event()
    model = Crew(
        [background("part one"), text_reply("waiting"), text_reply("final")],
        {"part one": [gate, text_reply("the answer")]},
    )
    engine = team(crew(tmp_path, background=True), model)
    work = asyncio.create_task(engine.run("lead", "acme", "go", store=store, conversation="c1"))
    run = await released(work, model, gate)
    assert [message.content for message in model.calls["lead"][0]] == ["later", "noted", "go"]
    assert run.steps[0].result == '{"job": "d4"}'
