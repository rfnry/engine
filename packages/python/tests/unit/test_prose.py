import inspect
import re
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.workspace import Local, Workspace

from .fakes import Scripted


class Verdict(BaseModel):
    reason: str = Field(description="Why")


class Plain(BaseModel):
    """What the output is."""

    reason: str


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


@pytest.mark.parametrize(
    ("model", "where"), [(Verdict, "properties.reason.description"), (Plain, "description")]
)
async def test_an_expect_model_with_a_description_is_refused_before_any_model_call(
    tmp_path: Path, model: type[BaseModel], where: str
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    llm = Scripted()
    message = f"carries a description at {where}; say what the output means in AGENT.md or a task"
    with pytest.raises(InputError, match=re.escape(message)):
        await Engine(tmp_path, models={"strong": llm}).run("a", "acme", "x", expect=model)
    assert llm.calls == []


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


def test_shell_says_nothing_a_backend_describes(tmp_path: Path) -> None:
    assert "description" not in inspect.signature(Local).parameters
    backend = Local(tmp_path)
    backend.description = "A Debian box."  # type: ignore[attr-defined]
    shell = next(t for t in Workspace(backend, scope="acme").tools() if t.name == "Shell")
    assert "Debian" not in shell.description
