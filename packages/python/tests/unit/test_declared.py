from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.declared import check_declarations, declared_tools
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tree import load_tree

from .fakes import Scripted, text_reply

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


async def lookup(args: dict[str, Any]) -> Any:
    return {"order": args["order_id"], "given": sorted(args)}


async def now(args: dict[str, Any]) -> str:
    return "2026-09-22T00:00:00Z"


class LookupArgs(BaseModel):
    order_id: str = Field(description="The order id")


class NowArgs(BaseModel):
    zone: str = "UTC"


class Ticket(BaseModel):
    title: str
    description: str


class Pattern(BaseModel):
    order_id: str = Field(pattern="^A")


def _code() -> dict[str, Tool]:
    return {
        "lookup_order": Tool("lookup_order", "Fetch an order by id", body=lookup, args=LookupArgs),
        "now": Tool("now", "The current UTC time", body=now, args=NowArgs),
    }


def _listing(root: Path, tools: str = "[lookup]") -> Path:
    (root / "a").mkdir(parents=True)
    (root / "a" / "AGENT.md").write_text(
        f"---\nname: a\nkind: generation\nmodel: strong\ntools: {tools}\n---\nHi\n"
    )
    return root


def _definition(engine: Engine, name: str = "lookup") -> dict[str, Any]:
    [found] = [
        d for d in EngineTools(engine, agent="a", scope="acme").definitions() if d["name"] == name
    ]
    return found


def test_the_tools_an_agent_lists_come_back_sorted_with_their_code_contracts() -> None:
    support = load_tree(FIXTURES)["support"]
    tools = Tools(declared_tools(support, _code()))
    assert tools.names == ["lookup_order", "now"]
    shown = {d["name"]: d for d in tools.definitions()}
    assert shown["lookup_order"]["description"] == "Fetch an order by id"
    assert shown["lookup_order"]["input_schema"]["required"] == ["order_id"]
    assert "zone" in shown["now"]["input_schema"]["properties"]


async def test_a_listed_tool_validates_its_arguments() -> None:
    tools = Tools(declared_tools(load_tree(FIXTURES)["support"], _code()))
    assert await tools.dispatch("lookup_order", {"order_id": "A-1"}) == {
        "order": "A-1",
        "given": ["order_id"],
    }
    with pytest.raises(InputError, match="order_id"):
        await tools.dispatch("lookup_order", {})
    assert await tools.dispatch("now", {}) == "2026-09-22T00:00:00Z"


async def test_arguments_are_validated_by_args_before_the_body_runs(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    async def body(args: dict[str, Any]) -> Any:
        seen.append(args)
        return {"order": args["order_id"]}

    engine = Engine(
        _listing(tmp_path, "[fetch]"),
        models={},
        tools=[Tool("fetch", "Fetch", body=body, args=Pattern)],
    )
    surface = EngineTools(engine, agent="a", scope="acme")
    with pytest.raises(InputError, match="order_id"):
        await surface.dispatch("fetch", {"order_id": 5})
    with pytest.raises(InputError, match="order_id"):
        await surface.dispatch("fetch", {"order_id": "B1"})
    assert seen == []
    assert await surface.dispatch("fetch", {"order_id": "A1"}) == {"order": "A1"}
    assert seen == [{"order_id": "A1"}]


def test_check_declarations_refuses_missing_tools_bodiless_tools_and_unlisted_tools() -> None:
    agents = load_tree(FIXTURES)
    check_declarations(agents, _code(), given={"strong", "judge"})
    with pytest.raises(ConfigurationError, match="lookup_order"):
        check_declarations(agents, {"now": _code()["now"]}, given={"strong", "judge"})
    with pytest.raises(ConfigurationError, match="weather"):
        check_declarations(
            agents,
            {**_code(), "weather": Tool("weather", "Forecast", body=now)},
            given={"strong", "judge"},
        )
    with pytest.raises(ConfigurationError, match="has no body"):
        check_declarations(agents, {**_code(), "now": Tool("now", "Time")}, given={"strong"})


def test_without_a_given_model_only_unlisted_code_tools_are_refused() -> None:
    agents = load_tree(FIXTURES)
    check_declarations(agents, {}, given=set())
    with pytest.raises(ConfigurationError, match="weather"):
        check_declarations(agents, {"weather": Tool("weather", "Forecast", body=now)}, given=set())


def test_tools_without_a_generation_model_still_carry_their_contracts(tmp_path: Path) -> None:
    engine = Engine(
        _listing(tmp_path),
        tools=[Tool("lookup", "Look up", body=lookup, args=LookupArgs)],
        models={},
    )
    assert _definition(engine)["input_schema"]["required"] == ["order_id"]


async def test_tools_a_run_passes_keep_their_descriptions(tmp_path: Path) -> None:
    model = Scripted(text_reply("ok"))
    engine = Engine(
        _listing(tmp_path),
        models={"strong": model},
        tools=[Tool("lookup", "Look up an order", body=lookup, args=LookupArgs)],
    )
    extra = Tools([Tool("ping", "Ping the desk", body=lookup, args=LookupArgs)])
    surface = EngineTools(engine, agent="a", scope="acme", tools=extra)
    [ping] = [d for d in surface.definitions() if d["name"] == "ping"]
    assert ping["description"] == "Ping the desk"
    assert ping["input_schema"]["properties"]["order_id"]["description"] == "The order id"
    await engine.run("a", "acme", "hi", tools=extra)
    [(_, _, listed)] = model.calls
    sent = {t.name: t for t in listed}
    assert sent["ping"].description == "Ping the desk"
    assert sent["lookup"].description == "Look up an order"


async def test_an_adapter_editing_a_listed_schema_changes_no_later_run_in_any_scope(
    tmp_path: Path,
) -> None:
    seen: list[str] = []
    schema = {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}

    class Editing(Scripted):
        async def generate(self, system: Any, messages: Any, tools: Any) -> Any:
            [found] = [t for t in tools if t.name == "lookup"]
            seen.append(found.input_schema["properties"]["id"]["type"])
            found.input_schema["properties"]["id"]["type"] = "integer"
            return await super().generate(system, messages, tools)

    model = Editing(text_reply("one"), text_reply("two"))
    engine = Engine(
        _listing(tmp_path),
        tools=[Tool("lookup", "Look up", body=lookup, input_schema=schema)],
        models={"strong": model},
    )
    first = await engine.run("a", "acme", "hi")
    second = await engine.run("a", "other", "hi")
    assert seen == ["string", "string"]
    assert first.fingerprint.tools == second.fingerprint.tools
