from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import TOOLS, Scripted


async def test_engine_tools_is_the_agents_surface(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS)
    tools = EngineTools(engine, agent="support", scope="acme")
    assert tools.names[:6] == ["Read", "Grep", "Glob", "LS", "Skill", "Load"] and set(
        tools.names[6:]
    ) >= {
        "lookup_order",
        "now",
    }
    assert await tools.dispatch("lookup_order", {"order_id": "A-1"}) == {"order": "A-1"}
    assert "refunded in full" in await tools.dispatch("Read", {"path": "rules/refunds.md"})
    assert EngineTools(engine, agent="billing", scope="acme").names == [
        "Read",
        "Grep",
        "Glob",
        "LS",
        "Skill",
        "Load",
    ]


async def test_call_tools_ride_along_and_collisions_are_refused(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS)

    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    extra = Tools([Tool("ping", "Ping", body=ping)])
    tools = EngineTools(engine, agent="billing", scope="acme", tools=extra)
    assert "ping" in tools and await tools.dispatch("ping", {}) == "pong"
    with pytest.raises(ConfigurationError, match="Read"):
        EngineTools(
            engine, agent="billing", scope="acme", tools=Tools([Tool("Read", "Clash", body=ping)])
        )


def test_the_engine_refuses_undeclared_and_bodiless_tools(fixtures: Path) -> None:
    with pytest.raises(ConfigurationError, match="lookup_order"):
        Engine(fixtures, generation=Scripted())
    with pytest.raises(ConfigurationError, match="weather"):
        Engine(
            fixtures,
            generation=Scripted(),
            tools=[*TOOLS, Tool("weather", "Forecast", body=TOOLS[1].body)],
        )
    with pytest.raises(ConfigurationError, match="lookup_order"):
        Engine(fixtures, generation=Scripted(), tools=[Tool("lookup_order"), TOOLS[1]])


def test_engine_tools_need_a_scope(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS)
    with pytest.raises(InputError, match="^engine_tools: scope is required$"):
        EngineTools(engine, agent="support", scope="")
