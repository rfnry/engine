from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools import declared
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


class NowArgs(BaseModel):
    zone: str = "UTC"


def _code() -> dict[str, Tool]:
    return {
        "lookup_order": Tool("lookup_order", body=lookup),
        "now": Tool("now", body=now, args=NowArgs),
    }


def test_a_contract_takes_the_tree_shape_and_a_listed_tool_the_code_s_types() -> None:
    support = load_tree(FIXTURES)["support"]
    tools = Tools(declared_tools(support, _code()))
    contract = next(d for d in tools.definitions() if d["name"] == "lookup_order")
    assert contract["description"].startswith("Fetch an order by id") and contract["input_schema"][
        "required"
    ] == ["order_id"]
    listed = next(d for d in tools.definitions() if d["name"] == "now")
    assert (
        listed["description"] == "The current UTC time"
        and "zone" in listed["input_schema"]["properties"]
    )


async def test_contract_validates_against_the_declared_fields() -> None:
    tools = Tools(declared_tools(load_tree(FIXTURES)["support"], _code()))
    assert await tools.dispatch("lookup_order", {"order_id": "A-1"}) == {
        "order": "A-1",
        "given": ["order_id"],
    }
    with pytest.raises(InputError, match="order_id"):
        await tools.dispatch("lookup_order", {})
    with pytest.raises(InputError, match="extra"):
        await tools.dispatch("lookup_order", {"order_id": "A-1", "extra": 1})
    assert await tools.dispatch("now", {}) == "2026-09-22T00:00:00Z"


def test_check_declarations_refuses_missing_bodies_and_undeclared_tools() -> None:
    agents = load_tree(FIXTURES)
    check_declarations(agents, _code(), given={"strong", "judge"})
    with pytest.raises(ConfigurationError, match="lookup_order"):
        check_declarations(agents, {"now": _code()["now"]}, given={"strong", "judge"})
    with pytest.raises(ConfigurationError, match="weather"):
        check_declarations(
            agents,
            {**_code(), "weather": Tool("weather", body=now)},
            given={"strong", "judge"},
        )
    with pytest.raises(ConfigurationError, match="carries a description in code"):
        check_declarations(
            agents, {**_code(), "now": Tool("now", "Time", body=now)}, given={"strong", "judge"}
        )


def test_without_a_given_model_only_undeclared_code_tools_are_refused() -> None:
    agents = load_tree(FIXTURES)
    check_declarations(agents, {}, given=set())
    with pytest.raises(ConfigurationError, match="weather"):
        check_declarations(agents, {"weather": Tool("weather", body=now)}, given=set())


def _lookup_tree(root: Path) -> None:
    (root / "a" / "tools").mkdir(parents=True)
    (root / "a" / "AGENT.md").write_text("---\nname: a\nkind: generation\nmodel: strong\n---\nHi\n")
    (root / "a" / "tools" / "lookup.md").write_text(
        "---\ndescription: Look up\ninput:\n  id: {type: string, required: true}\n---\nBody\n"
    )


def test_declared_argument_classes_are_made_when_the_tree_loads_not_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lookup_tree(tmp_path)
    made: list[str] = []
    real = declared._args_model

    def counted(decl: Any) -> Any:
        made.append(decl.name)
        return real(decl)

    monkeypatch.setattr(declared, "_args_model", counted)
    engine = Engine(tmp_path, tools=[Tool("lookup", body=lookup)], models={"strong": Scripted()})
    loaded = len(made)
    EngineTools(engine, agent="a", scope="acme")
    EngineTools(engine, agent="a", scope="acme")
    assert loaded == 1 and len(made) == 1


def test_tools_without_a_generation_model_still_carry_their_contracts(tmp_path: Path) -> None:
    _lookup_tree(tmp_path)
    engine = Engine(tmp_path, tools=[Tool("lookup", body=lookup)], models={})
    surface = EngineTools(engine, agent="a", scope="acme")
    [found] = [d for d in surface.definitions() if d["name"] == "lookup"]
    assert found["input_schema"]["required"] == ["id"]


async def test_an_adapter_editing_a_declared_schema_changes_no_later_run_in_any_scope(
    tmp_path: Path,
) -> None:
    _lookup_tree(tmp_path)
    seen: list[str] = []

    class Editing(Scripted):
        async def generate(self, system: Any, messages: Any, tools: Any) -> Any:
            [found] = [t for t in tools if t.name == "lookup"]
            seen.append(found.input_schema["properties"]["id"]["type"])
            found.input_schema["properties"]["id"]["type"] = "integer"
            return await super().generate(system, messages, tools)

    model = Editing(text_reply("one"), text_reply("two"))
    engine = Engine(tmp_path, tools=[Tool("lookup", body=lookup)], models={"strong": model})
    first = await engine.run("a", "acme", "hi")
    second = await engine.run("a", "other", "hi")
    assert seen == ["string", "string"]
    assert first.fingerprint.tools == second.fingerprint.tools
