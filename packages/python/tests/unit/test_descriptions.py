import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.declared import described
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Scripted, text_reply
from .trees import provided, write_folder


class Verdict(BaseModel):
    """The verdict on the refund."""

    reason: str = Field(description="Why")


def agent(root: Path, name: str, description: str | None = None) -> Path:
    entry: dict[str, Any] = {"kind": "generation", "model": "strong", "instructions": f"{name}.md"}
    if description is not None:
        entry["description"] = description
    return write_folder(root, {name: entry}, files={f"{name}.md": "Work.\n"})


async def test_an_expect_model_s_descriptions_reach_the_output_tool(tmp_path: Path) -> None:
    agent(tmp_path, "a")
    llm = Scripted(
        Reply(
            tool_calls=[ToolCall(id="o", name="Output", args={"reason": "double charge"})],
            stop_reason="tool_use",
        )
    )
    run = await Engine(tmp_path, providers=provided({"strong": llm})).run(
        "a", "acme", "x", expect=Verdict
    )
    schema = next(t for t in llm.calls[0][2] if t.name == "Output").input_schema
    assert schema["description"] == "The verdict on the refund."
    assert schema["properties"]["reason"]["description"] == "Why"
    assert isinstance(run, Generation)
    assert json.loads(run.reply) == {"reason": "double charge"}


async def test_a_run_with_a_fresh_expect_model_does_not_grow_the_described_cache(
    tmp_path: Path,
) -> None:
    agent(tmp_path, "a")

    def fresh() -> type[BaseModel]:
        return type("Fresh", (BaseModel,), {"__annotations__": {"reason": str}})

    async def ask(model: type[BaseModel]) -> None:
        engine = Engine(tmp_path, providers=provided({"strong": Scripted(text_reply("done"))}))
        await engine.run("a", "acme", "x", expect=model)

    await ask(fresh())
    before = described.cache_info().currsize
    await ask(fresh())
    assert described.cache_info().currsize == before


def test_run_takes_no_questions() -> None:
    assert "questions" not in inspect.signature(Engine.run).parameters


def test_run_takes_no_task() -> None:
    assert "task" not in inspect.signature(Engine.run).parameters
    assert "task" not in inspect.signature(Engine.tool).parameters


def test_engine_tool_is_described_by_its_entry(tmp_path: Path) -> None:
    agent(tmp_path, "billing", "Answers billing questions")
    tool = Engine(tmp_path, providers=provided({"strong": None})).tool("billing", scope="acme")
    assert tool.definitions()[0]["description"] == "Answers billing questions"
    assert "description" not in inspect.signature(Engine.tool).parameters


def test_engine_tool_on_an_agent_without_a_description_is_refused(tmp_path: Path) -> None:
    agent(tmp_path, "billing")
    with pytest.raises(ConfigurationError, match="^engine: agent 'billing' has no description;"):
        Engine(tmp_path, providers=provided({"strong": None})).tool("billing", scope="acme")


def test_shell_says_nothing_a_backend_describes(tmp_path: Path) -> None:
    assert "description" not in inspect.signature(Local).parameters
    backend = Local(tmp_path)
    backend.description = "A Debian box."  # type: ignore[attr-defined]
    shell = next(t for t in Workspace(backend, scope="acme").tools() if t.name == "Shell")
    assert "Debian" not in shell.description
