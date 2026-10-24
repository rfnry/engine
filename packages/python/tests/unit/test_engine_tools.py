from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import Document
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Scripted, text_reply


async def test_engine_tools_is_the_agents_surface(fixtures: Path) -> None:
    async with Engine(fixtures, models={"strong": Scripted()}) as engine:
        tools = EngineTools(engine, agent="support", scope="acme")
        assert tools.names[:5] == ["Read", "Grep", "Glob", "LS", "Load"] and set(
            tools.names[5:]
        ) >= {"lookup_order", "now"}
        assert await tools.dispatch("lookup_order", {"order_id": "A-1"}) == {"order": "A-1"}
        assert "refunded in full" in await tools.dispatch("Read", {"path": "rules/refunds.md"})
        assert EngineTools(engine, agent="billing", scope="acme").names == [
            "Read",
            "Grep",
            "Glob",
            "LS",
            "Load",
        ]


async def test_call_tools_ride_along_and_collisions_are_refused(fixtures: Path) -> None:
    engine = Engine(fixtures, models={"strong": Scripted()})

    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    extra = Tools([Tool("ping", "Ping", body=ping)])
    tools = EngineTools(engine, agent="billing", scope="acme", tools=extra)
    assert "ping" in tools and await tools.dispatch("ping", {}) == "pong"
    with pytest.raises(ConfigurationError, match="Read"):
        EngineTools(
            engine, agent="billing", scope="acme", tools=Tools([Tool("Read", "Clash", body=ping)])
        )


def test_engine_tools_need_a_scope(fixtures: Path) -> None:
    engine = Engine(fixtures, models={"strong": Scripted()})
    with pytest.raises(InputError, match="^engine_tools: scope is required$"):
        EngineTools(engine, agent="support", scope="")


async def test_an_agent_reads_another_agent_s_folder_in_full(tmp_path: Path) -> None:
    for name, front in [("coding", "reads: [writing]\n"), ("writing", "")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "AGENT.md").write_text(
            f"---\nname: {name}\nkind: generation\nmodel: strong\n{front}---\nHi\n"
        )
    (tmp_path / "writing" / "tasks").mkdir()
    (tmp_path / "writing" / "tasks" / "draft.md").write_text("Draft it.\n")
    engine = Engine(tmp_path, models={"strong": Scripted()})
    assert engine.agents["coding"].reads == ["writing"] and engine.agents["writing"].reads == []
    tools = EngineTools(engine, agent="coding", scope="acme")
    assert "Hi" in await tools.dispatch("Read", {"path": "../writing/AGENT.md"})
    assert await tools.dispatch("Read", {"path": "../writing/tasks/draft.md"}) == "Draft it.\n"
    loaded = await tools.dispatch("Load", {"path": "../writing/tasks/draft.md"})
    assert isinstance(loaded, list) and len(loaded) == 1
    assert isinstance(loaded[0], Document) and loaded[0].data == b"Draft it.\n"
    assert engine._spec("coding").listing == "../writing/"


async def test_an_adapter_editing_a_builtin_schema_changes_no_later_run(fixtures: Path) -> None:
    seen_types: list[str] = []

    class Editing(Scripted):
        async def generate(self, system: Any, messages: Any, tools: Any) -> Any:
            [read] = [t for t in tools if t.name == "Read"]
            seen_types.append(read.input_schema["properties"]["path"]["type"])
            read.input_schema["properties"]["path"]["type"] = "integer"
            return await super().generate(system, messages, tools)

    model = Editing(text_reply("one"), text_reply("two"))
    engine = Engine(fixtures, models={"strong": model})
    await engine.run("billing", "acme", "hi")
    await engine.run("billing", "acme", "hi")
    assert seen_types == ["string", "string"]
