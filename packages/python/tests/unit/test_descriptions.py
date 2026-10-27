import inspect
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.declared import described
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Scripted, text_reply


class Verdict(BaseModel):
    """The verdict on the refund."""

    reason: str = Field(description="Why")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


async def test_an_expect_model_s_descriptions_reach_the_output_tool(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    llm = Scripted(
        Reply(
            tool_calls=[ToolCall(id="o", name="Output", args={"reason": "double charge"})],
            stop_reason="tool_use",
        )
    )
    run = await Engine(tmp_path, models={"strong": llm}).run("a", "acme", "x", expect=Verdict)
    schema = next(t for t in llm.calls[0][2] if t.name == "Output").input_schema
    assert schema["description"] == "The verdict on the refund."
    assert schema["properties"]["reason"]["description"] == "Why"
    assert isinstance(run, Generation)
    assert json.loads(run.reply) == {"reason": "double charge"}


async def test_a_run_with_a_fresh_expect_model_does_not_grow_the_described_cache(
    tmp_path: Path,
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))

    def fresh() -> type[BaseModel]:
        return type("Fresh", (BaseModel,), {"__annotations__": {"reason": str}})

    async def ask(model: type[BaseModel]) -> None:
        engine = Engine(tmp_path, models={"strong": Scripted(text_reply("done"))})
        await engine.run("a", "acme", "x", expect=model)

    await ask(fresh())
    before = described.cache_info().currsize
    await ask(fresh())
    assert described.cache_info().currsize == before


def test_run_takes_no_questions() -> None:
    assert "questions" not in inspect.signature(Engine.run).parameters


def test_engine_tool_is_described_by_the_agent_md(tmp_path: Path) -> None:
    write(
        tmp_path / "billing/AGENT.md",
        agent_md("billing", extra="description: Answers billing questions\n"),
    )
    tool = Engine(tmp_path, models={}).tool("billing", scope="acme")
    assert tool.definitions()[0]["description"] == "Answers billing questions"
    assert "description" not in inspect.signature(Engine.tool).parameters


def test_engine_tool_on_an_agent_without_a_description_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "billing/AGENT.md", agent_md("billing"))
    with pytest.raises(ConfigurationError, match="no description: in its AGENT.md"):
        Engine(tmp_path, models={}).tool("billing", scope="acme")


def test_engine_tool_on_an_agent_whose_description_is_only_whitespace_is_refused(
    tmp_path: Path,
) -> None:
    write(tmp_path / "billing/AGENT.md", agent_md("billing", extra='description: "   "\n'))
    with pytest.raises(ConfigurationError, match="no description: in its AGENT.md"):
        Engine(tmp_path, models={}).tool("billing", scope="acme")


def test_shell_says_nothing_a_backend_describes(tmp_path: Path) -> None:
    assert "description" not in inspect.signature(Local).parameters
    backend = Local(tmp_path)
    backend.description = "A Debian box."  # type: ignore[attr-defined]
    shell = next(t for t in Workspace(backend, scope="acme").tools() if t.name == "Shell")
    assert "Debian" not in shell.description
