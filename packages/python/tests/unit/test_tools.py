from typing import Any

import pytest
from pydantic import BaseModel

from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.tools import Tool, Tools


class EchoArgs(BaseModel):
    text: str
    times: int = 1


async def echo(args: dict[str, Any]) -> str:
    return str(args["text"] * args.get("times", 1))


async def boom(args: dict[str, Any]) -> str:
    raise RuntimeError("it broke")


class Foreign:
    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "far",
                "description": "From another package",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    async def dispatch(self, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
        return {"seen": name}


def test_a_tool_defines_itself_from_its_args_model() -> None:
    tool = Tool("echo", "Echo text", body=echo, args=EchoArgs)
    definition = tool.definition()
    assert definition["name"] == "echo" and definition["input_schema"]["required"] == ["text"]
    bare = Tool("ping", "Ping", body=echo)
    assert bare.definition()["input_schema"] == {"type": "object", "properties": {}}


async def test_tools_dispatch_validates_and_routes() -> None:
    tools = Tools(
        [Tool("echo", "Echo text", body=echo, args=EchoArgs), Tool("boom", "Boom", body=boom)]
    )
    assert [d["name"] for d in tools.definitions()] == ["echo", "boom"]
    assert "echo" in tools and "ghost" not in tools and tools.names == ["echo", "boom"]
    assert await tools.dispatch("echo", {"text": "ab", "times": 2}) == "abab"
    with pytest.raises(InputError, match="ghost"):
        await tools.dispatch("ghost", {})
    with pytest.raises(InputError, match="text"):
        await tools.dispatch("echo", {})
    with pytest.raises(RuntimeError):
        await tools.dispatch("boom", {})


async def test_union_only_without_and_foreign_sets() -> None:
    mine = Tools([Tool("echo", "Echo", body=echo, args=EchoArgs)])
    combined = mine + Foreign()
    assert combined.names == ["echo", "far"]
    assert await combined.dispatch("far", {}) == {"seen": "far"}
    assert (combined.only("far")).names == ["far"] and (combined.without("far")).names == ["echo"]
    with pytest.raises(ConfigurationError, match="echo"):
        mine + Tools([Tool("echo", "Again", body=echo)])
    with pytest.raises(InputError):
        mine.only("ghost")


async def test_a_tool_without_a_body_cannot_be_dispatched() -> None:
    with pytest.raises(ConfigurationError, match="nobody"):
        await Tools([Tool("nobody", "No body")]).dispatch("nobody", {})


def test_a_schema_from_args_is_generated_once_and_shared_with_no_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Args(BaseModel):
        x: int

    made: list[int] = []
    real = Args.model_json_schema

    def counted(*a: Any, **k: Any) -> dict[str, Any]:
        made.append(1)
        return real(*a, **k)

    monkeypatch.setattr(Args, "model_json_schema", counted)
    first = Tool("t", args=Args).schema()
    first["properties"]["x"]["title"] = "changed"
    again = Tool("t", args=Args).schema()
    assert len(made) == 1 and again["properties"]["x"]["title"] == "X"
