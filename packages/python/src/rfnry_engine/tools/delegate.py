import contextvars
import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError, field_validator

from rfnry_engine.errors import DelegationDepthError, DelegationLoopError, InputError
from rfnry_engine.foreground import RECORDING, hooks, returned
from rfnry_engine.models import DecisionModel, GenerationModel, ModelChoice, ToolCall
from rfnry_engine.records import EventHandler, Generation
from rfnry_engine.stop import Stop
from rfnry_engine.tools.tools import Tool, ToolsLike

if TYPE_CHECKING:
    from rfnry_engine.engine import Engine
    from rfnry_engine.tools.workspace import Workspace
    from rfnry_engine.tree import AgentSpec

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
_routes: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "delegation_routes", default=()
)
_reach: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "delegation_reach", default=None
)


class AgentArgs(BaseModel):
    input: str
    questions: dict[str, Any] | None = None

    @field_validator("questions")
    @classmethod
    def _whole(cls, questions: dict[str, Any] | None) -> dict[str, Any] | None:
        if questions is None:
            return None
        return {
            qid: question.model_dump() if isinstance(question, BaseModel) else question
            for qid, question in questions.items()
        }


class DelegateArgs(AgentArgs):
    agent: str


def run_arguments(agents: list[AgentSpec]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
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
        raise InputError(
            f"tools/delegate: invalid arguments: {exc.errors(include_url=False)}"
        ) from exc
    if parsed.agent not in spec.agents:
        raise InputError(
            f"tools/delegate: agent {spec.name!r} may not delegate to {parsed.agent!r}"
        )
    return parsed


def folder_reach(levels_below: int | None) -> int | None:
    inherited = _reach.get()
    if levels_below is None:
        return inherited
    own = len(_stack.get()) + levels_below + 1
    return own if inherited is None else min(own, inherited)


@asynccontextmanager
async def enter(
    name: str, limit: int, reach: int | None, *, routed: bool = False
) -> AsyncIterator[None]:
    stack = _stack.get()
    routes = _routes.get()
    if routed and name in routes:
        raise DelegationLoopError(f"tools/delegate: loop at {' -> '.join([*stack, name])}")
    if len(stack) >= limit:
        raise DelegationDepthError(
            f"tools/delegate: depth {limit} exceeded at {' -> '.join([*stack, name])}"
        )
    stack_token = _stack.set((*stack, name))
    routes_token = _routes.set((*routes, name) if routed else (name,))
    reach_token = _reach.set(reach)
    try:
        yield
    finally:
        _reach.reset(reach_token)
        _routes.reset(routes_token)
        _stack.reset(stack_token)


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
    shown = " (decision agent)" if agent.kind == "decision" else ""
    return f"- {agent.name}{shown}: {agent.description}"


def delegate_description(engine: Engine, spec: AgentSpec) -> str:
    lines = "\n".join(_listed(engine._spec(name)) for name in spec.agents)
    return f"Hand an input to another agent and get its reply.\n{lines}"


def run_result(run: Any) -> str | dict[str, Any]:
    if isinstance(run, Generation):
        return run.reply
    result: dict[str, Any] = {"answers": {k: v.model_dump() for k, v in run.answers.items()}}
    if run.routed_to is not None:
        routed = run_result(run.routed_to)
        result["routed_to"] = {
            "agent": run.routed_to.agent,
            **({"reply": routed} if isinstance(routed, str) else routed),
        }
    return result


def child_result(child: Any) -> str:
    result = run_result(child)
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def delegate_tool(
    engine: Engine,
    spec: AgentSpec,
    scope: str,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    tags: Mapping[str, str] | None = None,
    approve: Callable[[str, ToolCall], Awaitable[bool]] | None = None,
    on_event: EventHandler | None = None,
    stop: Callable[[], Stop | None] | None = None,
    models: Mapping[str, GenerationModel | DecisionModel | ModelChoice] | None = None,
) -> Tool:
    async def body(args: dict[str, Any]) -> str:
        parsed = permit(spec, args)
        watching, held = hooks(on_event, stop)
        return child_result(
            returned(
                await engine._start(
                    parsed.agent,
                    scope,
                    parsed.input,
                    questions=parsed.questions,
                    tools=tools,
                    workspace=workspace,
                    tags=tags,
                    approve=approve,
                    on_event=watching,
                    stop=held,
                    record=RECORDING.get(),
                    models=models,
                )
            )
        )

    return Tool(
        DELEGATE_TOOL,
        delegate_description(engine, spec),
        body=body,
        input_schema=delegate_schema(engine, spec),
    )
