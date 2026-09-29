from __future__ import annotations

import contextvars
import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from rfnry_engine.errors import DelegationDepthError, DelegationLoopError, InputError
from rfnry_engine.models import ToolCall
from rfnry_engine.records import Generation
from rfnry_engine.tools.tools import Tool, ToolsLike
from rfnry_engine.tree import AgentSpec

if TYPE_CHECKING:
    from rfnry_engine.engine import Engine
    from rfnry_engine.tools.workspace import Workspace

DELEGATE_TOOL = "Delegate"
QUESTION: dict[str, Any] = {
    "type": "object",
    "description": "A question the agent answers with a probability for each answer. choice: pick one of options. score: place on levels, in order. check: yes or no.",
    "properties": {
        "type": {"type": "string", "enum": ["choice", "score", "check"]},
        "instructions": {
            "type": ["string", "object", "array"],
            "description": "The question: a string, or an object or list with parts of its own",
        },
        "options": {
            "type": "object",
            "additionalProperties": {"type": ["string", "object", "array", "null"]},
            "description": "choice: two or more, each key to what it means",
        },
        "levels": {
            "type": "array",
            "items": {"type": ["string", "object", "array"]},
            "description": "score: two or more, in order",
        },
        "yes": {
            "type": ["string", "object", "array", "null"],
            "description": "check: what yes means",
        },
        "no": {
            "type": ["string", "object", "array", "null"],
            "description": "check: what no means",
        },
    },
    "required": ["type", "instructions"],
}

_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "delegation_stack", default=()
)


class AgentArgs(BaseModel):
    input: str
    task: str | None = None
    questions: dict[str, Any] | None = None


class DelegateArgs(AgentArgs):
    agent: str


def run_arguments(agents: list[AgentSpec]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    tasks = list(dict.fromkeys(task for agent in agents for task in agent.tasks))
    if tasks:
        arguments["task"] = {
            "type": "string",
            "enum": tasks,
            "description": "A task of the agent, to run it with",
        }
    if any(agent.kind == "decision" for agent in agents):
        arguments["questions"] = {
            "type": "object",
            "additionalProperties": copy.deepcopy(QUESTION),
            "description": "For a decision agent: questions of your own, by id, asked with its own",
        }
    return arguments


def permit(spec: AgentSpec, args: dict[str, Any]) -> DelegateArgs:
    try:
        parsed = DelegateArgs.model_validate(args)
    except ValidationError as exc:
        raise InputError(f"delegate: invalid arguments: {exc.errors(include_url=False)}") from exc
    if parsed.agent not in spec.agents:
        raise InputError(f"delegate: agent {spec.name!r} may not delegate to {parsed.agent!r}")
    return parsed


@asynccontextmanager
async def enter(name: str, max_depth: int) -> AsyncIterator[None]:
    stack = _stack.get()
    if name in stack:
        raise DelegationLoopError(f"delegate: loop at {' -> '.join([*stack, name])}")
    if len(stack) >= max_depth:
        raise DelegationDepthError(
            f"delegate: depth {max_depth} exceeded at {' -> '.join([*stack, name])}"
        )
    token = _stack.set((*stack, name))
    try:
        yield
    finally:
        _stack.reset(token)


def delegate_schema(engine: Engine, spec: AgentSpec) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "enum": list(spec.agents)},
            "input": {"type": "string"},
            **run_arguments([engine._spec(name) for name in spec.agents]),
        },
        "required": ["agent", "input"],
    }


def _listed(agent: AgentSpec) -> str:
    notes = ["decision agent"] if agent.kind == "decision" else []
    if agent.tasks:
        notes.append(f"tasks: {', '.join(agent.tasks)}")
    shown = f" ({'; '.join(notes)})" if notes else ""
    return f"- {agent.name}{shown}: {agent.description}"


def delegate_description(engine: Engine, spec: AgentSpec) -> str:
    lines = "\n".join(_listed(engine._spec(name)) for name in spec.agents)
    return f"Hand an input to another agent and get its reply.\n{lines}"


def child_result(child: Any) -> str:
    if isinstance(child, Generation):
        return child.reply
    return json.dumps({k: v.model_dump() for k, v in child.answers.items()}, ensure_ascii=False)


def delegate_tool(
    engine: Engine,
    spec: AgentSpec,
    scope: str,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    tags: Mapping[str, str] | None = None,
    approve: Callable[[str, ToolCall], Awaitable[bool]] | None = None,
) -> Tool:
    async def body(args: dict[str, Any]) -> str:
        parsed = permit(spec, args)
        return child_result(
            await engine.run(
                parsed.agent,
                scope,
                parsed.input,
                task=parsed.task,
                questions=parsed.questions,
                tools=tools,
                workspace=workspace,
                tags=tags,
                approve=approve,
            )
        )

    return Tool(
        DELEGATE_TOOL,
        delegate_description(engine, spec),
        body=body,
        input_schema=delegate_schema(engine, spec),
    )
