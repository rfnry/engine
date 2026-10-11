from pathlib import Path

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import Message, ToolCall
from rfnry_engine.records import Generation

from .fakes import TOOLS, Scripted, text_reply

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
