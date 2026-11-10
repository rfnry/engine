from pathlib import Path
from typing import Any

from rfnry_engine.engine import Engine
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Scripted, text_reply
from .trees import provided, solo


async def venue(args: dict[str, Any]) -> dict[str, Any]:
    return {"venue": "Palácio da Bolsa", "city": "Porto"}


async def test_a_dict_result_reaches_the_model_unescaped(tmp_path: Path) -> None:
    llm = Scripted(
        Reply(
            text="", tool_calls=[ToolCall(id="1", name="venue", args={})], stop_reason="tool_use"
        ),
        text_reply("ok"),
    )
    folder = solo(tmp_path, name="billing", tools=["venue"])
    run = await Engine(folder, providers=provided({"strong": llm})).run(
        "billing", "acme", "x", tools=Tools([Tool("venue", body=venue)])
    )
    assert isinstance(run, Generation)
    tool_message = llm.calls[1][1][-1]
    assert "Palácio da Bolsa" in str(tool_message.content)
    assert "\\u00e1" not in str(tool_message.content)
