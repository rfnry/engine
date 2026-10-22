import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.declared import prose
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tree import load_tree

from .fakes import Scripted, lookup_order, text_reply


class Ticket(BaseModel):
    title: str
    description: str


class Described(BaseModel):
    order_id: str = Field(description="The order id")


class Item(BaseModel):
    """A line of the order."""

    sku: str


class Nested(BaseModel):
    items: list[Item]


class Pattern(BaseModel):
    order_id: str = Field(pattern="^A")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


def shop(root: Path, tools_md: str = "fetch: Fetch an order\n") -> Path:
    write(root / "shop/AGENT.md", agent_md("shop"))
    write(root / "shop/TOOLS.md", f"---\n{tools_md}---\n")
    return root


def filed(root: Path, input_: str = "  order_id: {type: string, required: true}\n") -> Path:
    write(root / "shop/AGENT.md", agent_md("shop"))
    write(root / "shop/tools/fetch.md", f"---\ndescription: Fetch an order\ninput:\n{input_}---\n")
    return root


def definition(engine: Engine, name: str = "fetch") -> dict[str, Any]:
    [found] = [
        d
        for d in EngineTools(engine, agent="shop", scope="acme").definitions()
        if d["name"] == name
    ]
    return found


def test_a_tools_md_entry_without_a_description_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="lists 'fetch' without a description; write it as fetch: what it does",
    ):
        load_tree(shop(tmp_path, "fetch:\n"))


@pytest.mark.parametrize("models", [{}, {"strong": Scripted()}])
def test_a_code_tool_with_a_description_is_refused_with_or_without_models(
    tmp_path: Path, models: Any
) -> None:
    with pytest.raises(
        ConfigurationError,
        match="tool 'fetch' carries a description in code; describe it in its agent's TOOLS.md or tools/fetch.md",
    ):
        Engine(shop(tmp_path), models=models, tools=[Tool("fetch", "Fetch", body=lookup_order)])


@pytest.mark.parametrize(
    ("tool", "where"),
    [
        (Tool("fetch", body=lookup_order, args=Described), "properties.order_id.description"),
        (Tool("fetch", body=lookup_order, args=Nested), "$defs.Item.description"),
        (
            Tool(
                "fetch",
                body=lookup_order,
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string", "description": "Q"}},
                },
            ),
            "properties.q.description",
        ),
    ],
)
def test_a_description_in_a_code_tool_s_schema_is_refused(
    tmp_path: Path, tool: Tool, where: str
) -> None:
    with pytest.raises(
        ConfigurationError,
        match=re.escape(
            f"carries a description in code at {where}; describe its arguments under input: in tools/fetch.md"
        ),
    ):
        Engine(shop(tmp_path), models={}, tools=[tool])


def test_an_argument_named_description_is_an_argument(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"description": {"type": "string", "default": {"description": "x"}}},
    }
    root = shop(tmp_path)
    ticket = Engine(root, models={}, tools=[Tool("fetch", body=lookup_order, args=Ticket)])
    assert "description" in definition(ticket)["input_schema"]["properties"]
    loose = Engine(root, models={}, tools=[Tool("fetch", body=lookup_order, input_schema=schema)])
    assert "description" in definition(loose)["input_schema"]["properties"]


@pytest.mark.parametrize(
    ("schema", "where"),
    [
        ({"description": "Top"}, "description"),
        ({"properties": {"a": {"description": "A"}}}, "properties.a.description"),
        ({"anyOf": [{"type": "null"}, {"description": "B"}]}, "anyOf.1.description"),
        ({"properties": {"a": {"default": {"description": "x"}}}}, None),
        ({"properties": {"a": {"examples": [{"description": "x"}]}}}, None),
        ({"properties": {"a": {"const": {"description": "x"}}}}, None),
        ({"properties": {"a": {"enum": [{"description": "x"}]}}}, None),
        ({"properties": {"description": {"type": "string"}}}, None),
    ],
)
def test_prose_finds_the_first_description_and_skips_data(
    schema: dict[str, Any], where: str | None
) -> None:
    assert prose(schema) == where


async def test_an_input_and_args_both_check_a_call_declared_contract_first(
    tmp_path: Path,
) -> None:
    seen: list[dict[str, Any]] = []

    async def body(args: dict[str, Any]) -> Any:
        seen.append(args)
        return {"order": args["order_id"]}

    engine = Engine(filed(tmp_path), models={}, tools=[Tool("fetch", body=body, args=Pattern)])
    surface = EngineTools(engine, agent="shop", scope="acme")
    with pytest.raises(InputError, match="order_id"):
        await surface.dispatch("fetch", {"order_id": 5})
    with pytest.raises(InputError, match="order_id"):
        await surface.dispatch("fetch", {"order_id": "B1"})
    assert seen == []
    assert await surface.dispatch("fetch", {"order_id": "A1"}) == {"order": "A1"}
    assert seen == [{"order_id": "A1"}]


def test_an_input_and_args_naming_different_arguments_are_refused_with_both_sets(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError) as refused:
        Engine(
            filed(tmp_path, "  id: {type: string, required: true}\n"),
            models={},
            tools=[Tool("fetch", body=lookup_order, args=Pattern)],
        )
    said = str(refused.value)
    assert "declared:" in said and "'fetch'" in said and "['id']" in said and "['order_id']" in said


def test_an_input_beside_a_code_input_schema_is_refused(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"order_id": {"type": "string"}}}
    with pytest.raises(ConfigurationError, match="input_schema"):
        Engine(
            filed(tmp_path),
            models={},
            tools=[Tool("fetch", body=lookup_order, input_schema=schema)],
        )


async def test_tools_a_run_passes_keep_their_descriptions(tmp_path: Path) -> None:
    model = Scripted(text_reply("ok"))
    engine = Engine(
        shop(tmp_path), models={"strong": model}, tools=[Tool("fetch", body=lookup_order)]
    )
    extra = Tools([Tool("ping", "Ping the desk", body=lookup_order, args=Described)])
    surface = EngineTools(engine, agent="shop", scope="acme", tools=extra)
    [ping] = [d for d in surface.definitions() if d["name"] == "ping"]
    assert ping["description"] == "Ping the desk"
    assert ping["input_schema"]["properties"]["order_id"]["description"] == "The order id"
    assert definition(engine)["description"] == "Fetch an order"
    await engine.run("shop", "acme", "hi", tools=extra)
    [(_, _, listed)] = model.calls
    sent = {t.name: t for t in listed}
    assert sent["ping"].description == "Ping the desk"
    assert sent["fetch"].description == "Fetch an order"
