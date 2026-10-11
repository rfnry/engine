import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.models import Message, Reply, ToolCall, Usage
from rfnry_engine.records import Generation
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import TOOLS, Scripted, now, text_reply

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


async def test_an_assistant_message_keeps_the_usage_of_the_call_that_wrote_it(
    fixtures: Path,
) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."))
    run = await first_run(Engine(fixtures, generation=llm, tools=TOOLS))
    written = [m.usage for m in run.conversation if m.role == "assistant"]
    assert written == [Usage(input=40, output=3, cache_read=30), None]
    assert all(m.usage is None for m in run.conversation if m.role != "assistant")


async def test_an_assistant_message_carries_the_prefix_it_was_written_after(
    fixtures: Path,
) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."))
    run = await first_run(Engine(fixtures, generation=llm, tools=TOOLS))
    prefixes = [m.prefix for m in run.conversation if m.role == "assistant"]
    assert all(p is not None and re.fullmatch(r"[0-9a-f]{16}", p) for p in prefixes)
    assert len(set(prefixes)) == 2
    assert all(m.prefix is None for m in run.conversation if m.role != "assistant")


async def test_reasoning_goes_back_within_a_run_and_to_a_run_with_the_same_history(
    fixtures: Path,
) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."), text_reply("The second one."))
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
    first = await first_run(engine)
    assert llm.calls[1][1][1].reasoning == THOUGHT
    second = await engine.run(
        "support", "acme", "Which charge?", history=first.conversation, context=CONTEXT
    )
    assert isinstance(second, Generation)
    assert llm.calls[2][1][1].reasoning == THOUGHT and second.messages[1].reasoning == THOUGHT


async def test_reasoning_is_left_out_once_what_came_before_it_changed(fixtures: Path) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."), *[text_reply("ok") for _ in range(4)])
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
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
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
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
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
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
    run = await Engine(fixtures, generation=llm, tools=TOOLS).run(
        "support", "acme", "Charged twice.", history=history
    )
    assert isinstance(run, Generation)
    assert run.messages[1] == Message(role="assistant", content="hello")


async def test_the_model_receives_messages_without_the_engines_record(fixtures: Path) -> None:
    llm = Scripted(reading(), text_reply("Refund in full."))
    run = await first_run(Engine(fixtures, generation=llm, tools=TOOLS))
    assert all(m.prefix is None and m.usage is None for call in llm.calls for m in call[1])
    assert run.conversation[1].prefix is not None and run.conversation[1].usage is not None


class Named(Scripted):
    def __init__(self, name: str, *replies: Reply) -> None:
        super().__init__(*replies)
        self.model = name


async def test_reasoning_is_left_out_when_the_model_is_another(fixtures: Path) -> None:
    old = Named("model-a", reading(), text_reply("Refund in full."))
    history = (await first_run(Engine(fixtures, generation=old, tools=TOOLS))).conversation
    new = Named("model-b", text_reply("ok"))
    run = await Engine(fixtures, generation=new, tools=TOOLS).run(
        "support", "acme", "Which?", history=history, context=CONTEXT
    )
    assert isinstance(run, Generation) and run.messages[1].reasoning is None
    same = Named("model-a", text_reply("ok"))
    run = await Engine(fixtures, generation=same, tools=TOOLS).run(
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
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
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
    engine = Engine(fixtures, generation=Scripted(text_reply("Still.")), tools=TOOLS)
    run = await engine.run("support", "acme", "And now?", history=history)
    assert isinstance(run, Generation)
    answered = run.messages[1]
    assert answered.prefix is None and answered.usage is None and answered.reasoning is None
    kept = Message.model_validate(
        {"role": "assistant", "prefix": "0123456789abcdef", "usage": {"input": 3}}
    )
    assert kept.prefix == "0123456789abcdef" and kept.usage == Usage(input=3)


async def test_reasoning_is_left_out_once_a_booted_file_changed(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    agent = root / "support" / "AGENT.md"
    agent.write_text(
        agent.read_text(encoding="utf-8").replace(
            "agents: [triage]\n", "agents: [triage]\nboot: all\n"
        ),
        encoding="utf-8",
    )
    llm = Scripted(reading(), text_reply("Refund in full."), text_reply("ok"), text_reply("ok"))
    first = await first_run(Engine(root, generation=llm, tools=TOOLS))
    same = await Engine(root, generation=llm, tools=TOOLS).run(
        "support", "acme", "Which?", history=first.conversation, context=CONTEXT
    )
    (root / "support" / "rules" / "refunds.md").write_text(
        "A duplicate charge is refunded within 30 days.\n", encoding="utf-8"
    )
    moved = await Engine(root, generation=llm, tools=TOOLS).run(
        "support", "acme", "Which?", history=first.conversation, context=CONTEXT
    )
    assert isinstance(same, Generation) and isinstance(moved, Generation)
    assert same.messages[1].reasoning == THOUGHT and moved.messages[1].reasoning is None


async def test_a_message_that_arrives_mid_run_is_under_the_next_prefix(fixtures: Path) -> None:
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    inbox.put_nowait("It was order A-2, not A-1.")
    noted = Reply(text="Noted.", reasoning=THOUGHT, stop_reason="end_turn")
    llm = Scripted(reading(), noted, text_reply("ok"))
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
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
