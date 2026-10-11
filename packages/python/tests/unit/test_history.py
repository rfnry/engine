import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import HistoryLike, Image, Message, Reply, Text, ToolCall
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

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


async def test_a_reply_that_repeats_a_call_id_is_closed_by_a_result_per_call(
    fixtures: Path,
) -> None:
    calls = [
        ToolCall(id="c1", name="lookup_order", args={"order_id": "A-1"}),
        ToolCall(id="c1", name="lookup_order", args={"order_id": "A-2"}),
    ]
    twice = [
        Message(role="user", content="Where are A-1 and A-2?"),
        Message(role="assistant", tool_calls=calls),
        Message(role="tool", tool_call_id="c1", content='{"order": "A-1"}'),
        Message(role="tool", tool_call_id="c1", content='{"order": "A-2"}'),
        Message(role="assistant", content="Both are on their way."),
    ]
    engine = Engine(fixtures, generation=Scripted(text_reply("Still.")), tools=TOOLS)
    run = await engine.run("support", "acme", "And now?", history=twice)
    assert isinstance(run, Generation) and run.reply == "Still."
    with pytest.raises(InputError, match=BROKEN.replace("c7", "c1").format(1)):
        await engine.run("support", "acme", "And now?", history=twice[:3])
    both = [ToolCall(id="c7", name="now"), ToolCall(id="c1", name="now")]
    open_two = [twice[0], Message(role="assistant", tool_calls=both)]
    with pytest.raises(InputError, match=BROKEN.format(1)):
        await engine.run("support", "acme", "And now?", history=open_two)


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


PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 8


def surface(
    fixtures: Path, history: Sequence[object] | HistoryLike | None, start: int, **kwargs: Any
) -> EngineTools:
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS, **kwargs)
    return EngineTools(engine, agent="support", scope="acme", history=history, start=start)


def texts(result: object) -> list[str]:
    assert isinstance(result, list)
    return [part.text for part in result if isinstance(part, Text)]


def positions(result: object) -> list[int]:
    return [json.loads(t)["position"] for t in texts(result) if t.startswith('{"position"')]


def calling_history(**args: Any) -> Reply:
    return Reply(tool_calls=[ToolCall(id="h1", name="History", args=args)], stop_reason="tool_use")


async def test_history_is_offered_only_when_a_run_starts_past_the_beginning(fixtures: Path) -> None:
    llm = Scripted(text_reply("a"), text_reply("b"), text_reply("c"))
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
    whole = await engine.run("support", "acme", "x", history=two())
    cut = await engine.run("support", "acme", "x", history=two(), start=4)
    alone = await engine.run("support", "acme", "x")
    offered = ["History" in [spec.name for spec in call[2]] for call in llm.calls]
    assert offered == [False, True, False]
    assert whole.fingerprint and cut.fingerprint and alone.fingerprint
    assert whole.fingerprint.tools == alone.fingerprint.tools != cut.fingerprint.tools


async def test_a_routed_run_is_offered_history_and_a_delegated_one_is_not(
    fixtures: Path, tmp_path: Path
) -> None:
    llm = Scripted(text_reply("Restarted it."))
    await Engine(fixtures, generation=llm, decision=Desk(), tools=TOOLS).run(
        "frontdesk", "acme", "And now?", history=two(), start=4
    )
    assert "History" in [spec.name for spec in llm.calls[0][2]]
    led = Scripted(delegating("writer", "Draft a reply"), text_reply("Draft."), text_reply("Sent."))
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    await Engine(lead(tmp_path), generation=led, decision=Answering()).run(
        "lead", "acme", "Answer the customer.", history=history, start=1
    )
    assert "History" in [spec.name for spec in led.calls[0][2]]
    assert "History" not in [spec.name for spec in led.calls[1][2]]


async def test_history_reads_the_page_before_where_the_model_starts(fixtures: Path) -> None:
    tools = surface(fixtures, two(), start=4)
    first = await tools.dispatch("History", {"limit": 3})
    assert positions(first) == [1, 2, 3]
    assert json.loads(texts(first)[0]) == {
        "position": 1,
        "role": "assistant",
        "tool_calls": [{"id": "c1", "name": "lookup_order", "args": {"order_id": "A-1"}}],
    }
    assert json.loads(texts(first)[1]) == {
        "position": 2,
        "role": "tool",
        "tool_call_id": "c1",
        "is_error": False,
    }
    assert texts(first)[2] == '{"order": "A-1"}'
    assert texts(first)[-1] == "[more before position 1: pass before=1 to read them]"
    rest = await tools.dispatch("History", {"before": 1})
    assert positions(rest) == [0] and texts(rest)[-1] == "Where is A-1?"
    assert await tools.dispatch("History", {"before": 0}) == "no earlier messages"


async def test_history_reads_rows_as_messages(fixtures: Path) -> None:
    result = await surface(fixtures, Rows(two()), start=4).dispatch("History", {})
    assert positions(result) == [0, 1, 2, 3]


async def test_history_never_reads_what_the_model_was_given_or_after(fixtures: Path) -> None:
    kept = Kept(two())
    tools = surface(fixtures, kept, start=4)
    kept.messages.append(Message(role="user", content="Written while the run worked."))
    result = await tools.dispatch("History", {"before": 99, "limit": 100})
    assert positions(result) == [0, 1, 2, 3] and kept.reads == [(0, 4)]


async def test_history_reaches_everything_when_the_run_starts_at_the_end(fixtures: Path) -> None:
    conversation = two()
    tools = surface(fixtures, conversation, start=len(conversation))
    result = await tools.dispatch("History", {"limit": 100})
    assert positions(result) == list(range(len(conversation)))


async def test_history_searches_back_for_what_matches(fixtures: Path) -> None:
    tools = surface(fixtures, [*two(), *looked_up("c9")], start=8)
    found = await tools.dispatch("History", {"pattern": "a-1", "case_insensitive": True})
    assert positions(found) == [0, 1, 2, 4, 5, 6]
    some = await tools.dispatch("History", {"pattern": "A-1", "limit": 2})
    assert positions(some) == [5, 6]
    assert texts(some)[-1] == "[searched back to position 5: pass before=5 to search further]"
    assert await tools.dispatch("History", {"pattern": "refund"}) == "no matches before position 8"


async def test_a_search_stops_after_history_walk_messages_and_says_where(fixtures: Path) -> None:
    tools = surface(fixtures, [*two(), *looked_up("c9")], start=8, history_walk=3)
    result = await tools.dispatch("History", {"pattern": "A-1"})
    assert positions(result) == [5, 6]
    assert texts(result)[-1] == "[searched back to position 5: pass before=5 to search further]"
    nothing = await tools.dispatch("History", {"pattern": "refund"})
    assert nothing == "no matches between positions 5 and 8: pass before=5 to search further"


async def test_a_result_holds_at_most_history_chars_of_text_and_says_where_to_go_on(
    fixtures: Path,
) -> None:
    conversation = [
        Message(role="user", content="x" * 50),
        Message(role="assistant", content="y" * 50),
        Message(role="user", content="z"),
    ]
    page = await surface(fixtures, conversation, start=2, history_chars=120).dispatch("History", {})
    assert positions(page) == [1]
    assert texts(page)[-1] == "[more before position 1: pass before=1 to read them]"
    big = [Message(role="user", content="w" * 500), Message(role="user", content="now")]
    cut = await surface(fixtures, big, start=1, history_chars=100).dispatch("History", {})
    assert sum(len(t) for t in texts(cut)[:-1]) == 100
    assert texts(cut)[-1] == "[position 0 cut at 100 characters]"


async def test_history_hands_over_media_as_the_parts_they_are(fixtures: Path) -> None:
    image = Image(media_type="image/png", data=PNG)
    conversation = [
        Message(role="user", content=[Text(text="This one?"), image]),
        Message(role="assistant", content="A receipt."),
        Message(role="user", content="And now?"),
    ]
    result = await surface(fixtures, conversation, start=2).dispatch("History", {})
    assert isinstance(result, list) and image in result


async def test_an_invalid_pattern_is_an_error_the_model_reads(fixtures: Path) -> None:
    with pytest.raises(InputError, match=r"^history: invalid pattern"):
        await surface(fixtures, looked_up(), start=4).dispatch("History", {"pattern": "("})


async def test_a_source_that_fails_inside_history_is_a_tool_error(fixtures: Path) -> None:
    class Flaky(Kept):
        async def read(self, start: int, stop: int | None = None) -> Sequence[object]:
            if stop is not None:
                raise ConnectionError("database is down")
            return await super().read(start, stop)

    llm = Scripted(calling_history(), text_reply("I cannot see that far back."))
    run = await Engine(fixtures, generation=llm, tools=TOOLS).run(
        "support", "acme", "x", history=Flaky(two()), start=4
    )
    assert isinstance(run, Generation) and run.steps[0].tool == "History"
    assert run.steps[0].is_error and run.steps[0].result == "ConnectionError: database is down"


async def test_a_slow_source_meets_the_tool_deadline(fixtures: Path) -> None:
    class Slow(Kept):
        async def read(self, start: int, stop: int | None = None) -> Sequence[object]:
            if stop is not None:
                await asyncio.sleep(5)
            return await super().read(start, stop)

    llm = Scripted(calling_history(), text_reply("ok"))
    engine = Engine(fixtures, generation=llm, tools=TOOLS, tool_timeout=0.05)
    run = await engine.run("support", "acme", "x", history=Slow(looked_up()), start=4)
    assert isinstance(run, Generation)
    assert run.steps[0].result == "TimeoutError: History timed out after 0.05s"


def test_history_is_described_the_same_whatever_the_start(fixtures: Path) -> None:
    def definition(start: int) -> dict[str, Any]:
        found = surface(fixtures, two(), start).definitions()
        return next(d for d in found if d["name"] == "History")

    assert definition(2) == definition(6)


def test_engine_tools_refuse_a_start_they_cannot_use(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS)
    with pytest.raises(InputError, match=r"^engine_tools: start must not be negative; got -1$"):
        EngineTools(engine, agent="support", scope="acme", history=[], start=-1)
    with pytest.raises(InputError, match=r"^engine_tools: start needs a history$"):
        EngineTools(engine, agent="support", scope="acme", start=3)
    assert "History" not in EngineTools(engine, agent="support", scope="acme", history=two()).names


@pytest.mark.parametrize("name", ["history_walk", "history_chars"])
def test_history_bounds_must_be_positive(fixtures: Path, name: str) -> None:
    settings: dict[str, Any] = {name: 0}
    with pytest.raises(ConfigurationError, match=rf"^engine: {name} must be positive$"):
        Engine(fixtures, generation=Scripted(), tools=TOOLS, **settings)


COLLIDES = r"^engine_tools: a tool named 'History' collides with the builtin$"


async def mine(args: dict[str, Any]) -> str:
    return "mine"


async def test_a_tool_named_history_is_refused_on_every_run(fixtures: Path, tmp_path: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(text_reply("x")), tools=TOOLS)
    with pytest.raises(ConfigurationError, match=COLLIDES):
        await engine.run("support", "acme", "x", tools=Tools([Tool("History", "Mine", body=mine)]))
    (tmp_path / "desk" / "tools").mkdir(parents=True)
    (tmp_path / "desk" / "AGENT.md").write_text("---\nname: desk\nkind: generation\n---\nHelp.\n")
    (tmp_path / "desk" / "tools" / "History.md").write_text(
        "---\nname: History\ndescription: Mine\n---\n"
    )
    declared = Engine(tmp_path, generation=Scripted(), tools=[Tool("History", body=mine)])
    with pytest.raises(ConfigurationError, match=COLLIDES):
        await declared.run("desk", "acme", "x")


async def test_history_never_shows_what_a_source_hands_back_past_stop(fixtures: Path) -> None:
    class Unbounded(Kept):
        async def read(self, start: int, stop: int | None = None) -> Sequence[object]:
            return await super().read(start, None)

    tools = surface(fixtures, Unbounded(two()), start=4)
    assert positions(await tools.dispatch("History", {})) == [0, 1, 2, 3]
    assert positions(await tools.dispatch("History", {"pattern": "A-1"})) == [0, 1, 2]


async def test_a_page_past_the_end_of_a_source_says_where_to_read_further_back(
    fixtures: Path,
) -> None:
    tools = surface(fixtures, Kept(looked_up()), start=50)
    assert await tools.dispatch("History", {}) == (
        "no messages between positions 30 and 50: pass before=30 to read further back"
    )
    assert await tools.dispatch("History", {"before": 30}) == (
        "no messages between positions 10 and 30: pass before=10 to read further back"
    )
    assert positions(await tools.dispatch("History", {"before": 10})) == [0, 1, 2, 3]


async def test_a_row_that_does_not_fit_is_refused_without_its_content(fixtures: Path) -> None:
    class Odd(Kept):
        async def read(self, start: int, stop: int | None = None) -> Sequence[object]:
            return [{"content": "card 4242"}]

    with pytest.raises(InputError, match=r"^history: the conversation does not fit") as caught:
        await surface(fixtures, Odd([]), start=2).dispatch("History", {})
    assert "4242" not in str(caught.value)


async def test_a_cut_message_hands_over_no_empty_text(fixtures: Path) -> None:
    long = Message(role="user", content=[Text(text="w" * 500), Text(text="v" * 10)])
    conversation = [long, Message(role="user", content="now")]
    cut = await surface(fixtures, conversation, start=1, history_chars=100).dispatch("History", {})
    assert all(texts(cut)) and sum(len(t) for t in texts(cut)[:-1]) == 100
    assert texts(cut)[-1] == "[position 0 cut at 100 characters]"


async def test_a_search_reads_back_a_hundred_messages_at_a_time(fixtures: Path) -> None:
    kept = Kept(
        [Message(role="user", content="hit" if i % 10 == 0 else "miss") for i in range(250)]
    )
    tools = surface(fixtures, kept, start=250, history_walk=250)
    result = await tools.dispatch("History", {"pattern": "hit", "limit": 20})
    assert positions(result) == list(range(50, 250, 10))
    assert texts(result)[-1] == "[searched back to position 50: pass before=50 to search further]"
    assert kept.reads == [(150, 250), (50, 150)]


async def test_a_search_cut_by_history_chars_resumes_at_the_first_position_shown(
    fixtures: Path,
) -> None:
    conversation = [Message(role="user", content=f"hit {'x' * 100}") for _ in range(10)]
    tools = surface(fixtures, conversation, start=10, history_chars=300)
    result = await tools.dispatch("History", {"pattern": "hit"})
    assert positions(result) == [8, 9]
    assert texts(result)[-1] == "[searched back to position 8: pass before=8 to search further]"


async def test_a_history_given_as_a_generator_is_read_once(fixtures: Path) -> None:
    chat = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
        Message(role="user", content="Where is A-1?"),
        Message(role="assistant", content="On its way."),
    ]
    llm = Scripted(calling_history(), text_reply("You said hi."))
    once = (message for message in chat)
    run = await Engine(fixtures, generation=llm, tools=TOOLS).run(
        "support",
        "acme",
        "What did I say first?",
        history=once,  # type: ignore[arg-type]
        start=2,
    )
    assert isinstance(run, Generation) and run.messages[:-1] == chat[2:]
    assert positions(run.steps[0].result) == [0, 1]
