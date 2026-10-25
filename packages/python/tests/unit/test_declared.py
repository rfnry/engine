import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import Scripted, text_reply

SCHEMA = {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}


async def lookup(args: dict[str, Any]) -> Any:
    return {"order": args["order_id"], "given": sorted(args)}


class LookupArgs(BaseModel):
    order_id: str = Field(description="The order id")


class Pattern(BaseModel):
    order_id: str = Field(pattern="^A")


def _listing(root: Path, description: str = "Look up an order") -> Path:
    (root / "a/tools").mkdir(parents=True)
    (root / "a/AGENT.md").write_text("---\nname: a\nkind: generation\nmodel: strong\n---\nHi\n")
    (root / "a/tools/lookup.json").write_text(
        json.dumps({"description": description, "input_schema": SCHEMA})
    )
    (root / "a/tools/lookup.py").write_text("def run(args, call):\n    return args\n")
    return root


async def test_a_run_tool_s_arguments_are_validated_by_args_before_its_body_runs(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, Any]] = []

    async def body(args: dict[str, Any]) -> Any:
        seen.append(args)
        return {"order": args["order_id"]}

    fetch = Tools([Tool("fetch", "Fetch", body=body, args=Pattern)])
    async with Engine(_listing(tmp_path), models={}) as engine:
        surface = EngineTools(engine, agent="a", scope="acme", tools=fetch)
        with pytest.raises(InputError, match="order_id"):
            await surface.dispatch("fetch", {"order_id": 5})
        with pytest.raises(InputError, match="order_id"):
            await surface.dispatch("fetch", {"order_id": "B1"})
        assert seen == []
        assert await surface.dispatch("fetch", {"order_id": "A1"}) == {"order": "A1"}
    assert seen == [{"order_id": "A1"}]


async def test_tools_without_a_generation_model_still_carry_their_contracts(
    tmp_path: Path,
) -> None:
    extra = Tools([Tool("ping", "Ping", body=lookup, args=LookupArgs)])
    async with Engine(_listing(tmp_path), models={}) as engine:
        shown = {
            d["name"]: d
            for d in EngineTools(engine, agent="a", scope="acme", tools=extra).definitions()
        }
    assert shown["lookup"]["input_schema"] == SCHEMA
    assert shown["ping"]["input_schema"]["required"] == ["order_id"]


async def test_tools_a_run_passes_keep_their_descriptions(tmp_path: Path) -> None:
    model = Scripted(text_reply("ok"))
    extra = Tools([Tool("ping", "Ping the desk", body=lookup, args=LookupArgs)])
    async with Engine(_listing(tmp_path), models={"strong": model}) as engine:
        surface = EngineTools(engine, agent="a", scope="acme", tools=extra)
        [ping] = [d for d in surface.definitions() if d["name"] == "ping"]
        await engine.run("a", "acme", "hi", tools=extra)
    assert ping["description"] == "Ping the desk"
    assert ping["input_schema"]["properties"]["order_id"]["description"] == "The order id"
    [(_, _, listed)] = model.calls
    sent = {t.name: t for t in listed}
    assert sent["ping"].description == "Ping the desk"
    assert sent["lookup"].description == "Look up an order"


async def test_an_adapter_editing_a_folder_tool_s_schema_changes_no_later_run_in_any_scope(
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    class Editing(Scripted):
        async def generate(self, system: Any, messages: Any, tools: Any) -> Any:
            [found] = [t for t in tools if t.name == "lookup"]
            seen.append(found.input_schema["properties"]["id"]["type"])
            found.input_schema["properties"]["id"]["type"] = "integer"
            return await super().generate(system, messages, tools)

    model = Editing(text_reply("one"), text_reply("two"))
    async with Engine(_listing(tmp_path), models={"strong": model}) as engine:
        first = await engine.run("a", "acme", "hi")
        second = await engine.run("a", "other", "hi")
        assert await EngineTools(engine, agent="a", scope="acme").dispatch(
            "lookup", {"id": "A-1"}
        ) == {"id": "A-1"}
    assert seen == ["string", "string"]
    assert first.fingerprint.tools == second.fingerprint.tools
