from collections.abc import Sequence
from pathlib import Path

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import Message, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.refinery import Refinery

from .fakes import TOOLS, Answering, Desk, Scripted, delegating, lead, said, text_reply

BROKEN = (
    r"^engine: history breaks tool call 'c7' from its result at position {}; "
    r"start at a message the person sent$"
)


def looked_up(call_id: str = "c7") -> list[Message]:
    return [
        Message(role="user", content="Where is A-1?"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id=call_id, name="lookup_order", args={"order_id": "A-1"})],
        ),
        Message(role="tool", tool_call_id=call_id, content='{"order": "A-1"}'),
        Message(role="assistant", content="It is on its way."),
    ]


async def test_a_history_with_a_result_whose_call_is_cut_off_is_refused(fixtures: Path) -> None:
    llm = Scripted(text_reply("x"))
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
    with pytest.raises(InputError, match=BROKEN.format(0)):
        await engine.run("support", "acme", "And now?", history=looked_up()[2:])
    assert llm.calls == []


async def test_a_history_with_a_call_whose_result_is_cut_off_is_refused(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(text_reply("x")), tools=TOOLS)
    with pytest.raises(InputError, match=BROKEN.format(1)):
        await engine.run("support", "acme", "And now?", history=looked_up()[:2])


async def test_a_history_cut_at_a_message_the_person_sent_runs(fixtures: Path) -> None:
    llm = Scripted(text_reply("Still on its way."))
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
    history = [*looked_up("c1"), *looked_up("c7")]
    run = await engine.run("support", "acme", "And now?", history=history[4:])
    assert isinstance(run, Generation) and run.reply == "Still on its way."


class Kept:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages
        self.reads: list[tuple[int, int | None]] = []

    async def read(self, start: int, stop: int | None = None) -> Sequence[object]:
        self.reads.append((start, stop))
        return self.messages[start:stop]


class Rows(Kept):
    async def read(self, start: int, stop: int | None = None) -> Sequence[object]:
        found = await super().read(start, stop)
        return [m.model_dump(mode="json") for m in found if isinstance(m, Message)]


class Down:
    async def read(self, start: int, stop: int | None = None) -> Sequence[object]:
        raise ConnectionError("database is down")


def two() -> list[Message]:
    return [*looked_up("c1"), *looked_up("c7")]


async def test_a_run_sends_its_history_from_start_and_says_what_it_added(fixtures: Path) -> None:
    llm = Scripted(text_reply("Still on its way."))
    conversation = two()
    run = await Engine(fixtures, generation=llm, tools=TOOLS).run(
        "support", "acme", "And now?", history=conversation, start=4
    )
    asked = Message(role="user", content="And now?")
    assert isinstance(run, Generation)
    assert llm.calls[0][1] == [*conversation[4:], asked]
    assert run.start == 4 and run.messages == [*conversation[4:], asked]
    assert said(run.added) == [asked, Message(role="assistant", content="Still on its way.")]
    assert run.conversation == [*conversation[4:], *run.added]


async def test_a_kept_conversation_is_read_from_start_only(fixtures: Path) -> None:
    kept = Kept(two())
    run = await Engine(fixtures, generation=Scripted(text_reply("ok")), tools=TOOLS).run(
        "support", "acme", "And now?", history=kept, start=4
    )
    assert isinstance(run, Generation)
    assert kept.reads == [(4, None)] and run.messages[:-1] == kept.messages[4:]


async def test_a_kept_conversation_may_hand_back_rows(fixtures: Path) -> None:
    rows = Rows(two())
    run = await Engine(fixtures, generation=Scripted(text_reply("ok")), tools=TOOLS).run(
        "support", "acme", "And now?", history=rows, start=4
    )
    assert isinstance(run, Generation) and run.messages[:-1] == rows.messages[4:]


async def test_a_start_at_the_end_sends_only_the_input(fixtures: Path) -> None:
    llm = Scripted(text_reply("Hello again."))
    conversation = looked_up()
    run = await Engine(fixtures, generation=llm, tools=TOOLS).run(
        "support", "acme", "Hi again.", history=conversation, start=len(conversation)
    )
    assert isinstance(run, Generation) and run.start == 4
    assert llm.calls[0][1] == [Message(role="user", content="Hi again.")]


@pytest.mark.parametrize("start", [9, -1])
async def test_a_start_outside_a_list_is_refused(fixtures: Path, start: int) -> None:
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS)
    with pytest.raises(InputError, match=rf"^engine: start must be between 0 and 8; got {start}$"):
        await engine.run("support", "acme", "x", history=two(), start=start)


async def test_a_start_needs_a_history_and_never_goes_below_zero(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS)
    with pytest.raises(InputError, match=r"^engine: start needs a history$"):
        await engine.run("support", "acme", "x", start=2)
    with pytest.raises(InputError, match=r"^engine: start must not be negative; got -1$"):
        await engine.run("support", "acme", "x", history=Kept([]), start=-1)


async def test_a_cut_at_start_names_its_position_in_the_conversation(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS)
    with pytest.raises(InputError, match=BROKEN.format(6)):
        await engine.run("support", "acme", "x", history=two(), start=6)


async def test_a_kept_conversation_that_cannot_be_read_raises_as_it_is(fixtures: Path) -> None:
    llm = Scripted()
    with pytest.raises(ConnectionError, match="database is down"):
        await Engine(fixtures, generation=llm, tools=TOOLS).run(
            "support", "acme", "x", history=Down(), start=2
        )
    assert llm.calls == []


async def test_a_routed_run_is_given_the_history_and_start(fixtures: Path) -> None:
    llm = Scripted(text_reply("Restarted it."))
    engine = Engine(fixtures, generation=llm, decision=Desk(), tools=TOOLS)
    conversation = two()
    run = await engine.run("frontdesk", "acme", "And now?", history=conversation, start=4)
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.start == 4 and run.routed_to.messages[:-1] == conversation[4:]
    assert run.added == run.routed_to.added
    assert said(run.added)[0] == Message(role="user", content="And now?")


async def test_a_decision_that_routed_nowhere_added_nothing(fixtures: Path) -> None:
    run = await Engine(fixtures, decision=Desk()).run("triage", "acme", "It is down")
    assert isinstance(run, Decision) and run.added == []


async def test_a_delegated_run_starts_with_no_history(tmp_path: Path) -> None:
    llm = Scripted(delegating("writer", "Draft a reply"), text_reply("Draft."), text_reply("Sent."))
    engine = Engine(lead(tmp_path), generation=llm, decision=Answering())
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    run = await engine.run("lead", "acme", "Answer the customer.", history=history, start=1)
    assert isinstance(run, Generation) and run.start == 1
    [child] = run.children
    assert isinstance(child, Generation) and child.start == 0
    assert child.messages == [Message(role="user", content="Draft a reply")]


async def test_a_run_read_back_keeps_its_start_and_what_it_added(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
    engine = Engine(fixtures, generation=Scripted(text_reply("ok")), tools=TOOLS, refinery=refinery)
    run = await engine.run("support", "acme", "And now?", history=two(), start=4)
    back = await refinery.run("acme", run.run_id)
    assert isinstance(run, Generation) and isinstance(back, Generation)
    assert back.start == 4 and back.added == run.added
