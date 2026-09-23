from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.declared import check_declarations, declared_tools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tree import load_tree

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


async def lookup(args: dict[str, Any]) -> Any:
    return {"order": args["order_id"], "given": sorted(args)}


async def now(args: dict[str, Any]) -> str:
    return "2026-09-22T00:00:00Z"


class NowArgs(BaseModel):
    zone: str = "UTC"


def _code() -> dict[str, Tool]:
    return {"lookup_order": Tool("lookup_order", body=lookup), "now": Tool("now", "The current UTC time", body=now, args=NowArgs)}


def test_contract_takes_the_tree_shape_and_adoption_takes_the_code_shape() -> None:
    support = load_tree(FIXTURES)["support"]
    tools = Tools(declared_tools(support, _code()))
    contract = next(d for d in tools.definitions() if d["name"] == "lookup_order")
    assert contract["description"].startswith("Fetch an order by id") and contract["input_schema"]["required"] == ["order_id"]
    adopted = next(d for d in tools.definitions() if d["name"] == "now")
    assert adopted["description"] == "The current UTC time" and "zone" in adopted["input_schema"]["properties"]


async def test_contract_validates_against_the_declared_fields() -> None:
    tools = Tools(declared_tools(load_tree(FIXTURES)["support"], _code()))
    assert await tools.dispatch("lookup_order", {"order_id": "A-1"}) == {"order": "A-1", "given": ["order_id"]}
    with pytest.raises(InputError, match="order_id"):
        await tools.dispatch("lookup_order", {})
    with pytest.raises(InputError, match="extra"):
        await tools.dispatch("lookup_order", {"order_id": "A-1", "extra": 1})
    assert await tools.dispatch("now", {}) == "2026-09-22T00:00:00Z"


def test_check_declarations_refuses_missing_bodies_and_undeclared_tools() -> None:
    agents = load_tree(FIXTURES)
    check_declarations(agents, _code(), bodies=True)
    with pytest.raises(ConfigurationError, match="lookup_order"):
        check_declarations(agents, {"now": _code()["now"]}, bodies=True)
    with pytest.raises(ConfigurationError, match="weather"):
        check_declarations(agents, {**_code(), "weather": Tool("weather", "Forecast", body=now)}, bodies=True)
    with pytest.raises(ConfigurationError, match="description"):
        check_declarations(agents, {**_code(), "now": Tool("now", body=now)}, bodies=True)


def test_without_bodies_only_undeclared_code_tools_are_refused() -> None:
    agents = load_tree(FIXTURES)
    check_declarations(agents, {}, bodies=False)
    with pytest.raises(ConfigurationError, match="weather"):
        check_declarations(agents, {"weather": Tool("weather", "Forecast", body=now)}, bodies=False)
