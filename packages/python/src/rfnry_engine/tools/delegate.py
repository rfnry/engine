from __future__ import annotations

import contextvars
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

_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("delegation_stack", default=())


class DelegateArgs(BaseModel):
    agent: str
    input: str


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
        raise DelegationDepthError(f"delegate: depth {max_depth} exceeded at {' -> '.join([*stack, name])}")
    token = _stack.set((*stack, name))
    try:
        yield
    finally:
        _stack.reset(token)


def delegate_schema(spec: AgentSpec) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"agent": {"type": "string", "enum": list(spec.agents)}, "input": {"type": "string"}},
        "required": ["agent", "input"],
    }


def delegate_description(engine: Engine, spec: AgentSpec) -> str:
    lines = "\n".join(f"- {name}: {engine._spec(name).description}" for name in spec.agents)
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
            await engine.run(parsed.agent, scope, parsed.input, tools=tools, workspace=workspace, tags=tags, approve=approve)
        )

    return Tool(DELEGATE_TOOL, delegate_description(engine, spec), body=body, input_schema=delegate_schema(spec))
