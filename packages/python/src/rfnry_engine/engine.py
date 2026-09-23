from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter, ValidationError

from rfnry_engine.bundle import build_bundle
from rfnry_engine.errors import (
    ConfigurationError,
    InputError,
    NotFoundError,
    RunawayError,
)
from rfnry_engine.models import (
    ChoiceAnswer,
    Content,
    DecisionModel,
    EventHandler,
    GenerationModel,
    Message,
    Part,
    Reply,
    State,
    StreamingGenerationModel,
    Text,
    TextDelta,
    ToolCall,
    ToolCallStarted,
    ToolResultReady,
    ToolSpec,
    Usage,
    as_decided,
    as_reply,
    settle,
)
from rfnry_engine.records import Decision, Generation, Run, Step
from rfnry_engine.tools.declared import check_declarations
from rfnry_engine.tools.delegate import DELEGATE_TOOL, child_result, enter, permit
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, ToolsLike
from rfnry_engine.tools.workspace import Workspace
from rfnry_engine.tree import AgentSpec, Kind, load_tree, validate_tree

OUTPUT_TOOL = "Output"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Options:
    task: str | None
    history: list[Message]
    context: str | None
    expect: type[BaseModel] | None
    on_event: EventHandler | None
    tools: ToolsLike | None
    workspace: Workspace | None


class AgentInfo(BaseModel):
    name: str
    kind: Kind
    description: str
    agents: list[str]


class Recorder(Protocol):
    async def record(self, run: Run) -> None: ...


PART = TypeAdapter[Part](Part)
PART_TYPES = ("text", "image", "audio", "document")


def _part(item: object) -> Part | None:
    if isinstance(item, BaseModel) and getattr(item, "type", None) in PART_TYPES:
        return PART.validate_python(item, from_attributes=True)
    return None


def _parts(value: object) -> list[Part] | None:
    if not isinstance(value, list):
        return None
    parts = [_part(item) for item in value]
    found = [part for part in parts if part is not None]
    if not found:
        return None
    if len(found) != len(parts):
        raise InputError("engine: input mixes parts with other values")
    return found


def _content(value: object) -> Content:
    if isinstance(value, str):
        return value
    parts = _parts(value)
    return parts if parts is not None else json.dumps(value)


def _state(spec: AgentSpec, value: State) -> State:
    parts = _parts(value)
    if parts is None:
        return value
    media = next((part for part in parts if not isinstance(part, Text)), None)
    if media is not None:
        raise InputError(f"engine: decision agent {spec.name!r} takes text only; got {media.type}")
    return "\n".join(part.text for part in parts if isinstance(part, Text))


def _assistant(reply: Reply) -> Message:
    return Message(role="assistant", content=reply.text, tool_calls=reply.tool_calls, reasoning=reply.reasoning)


def _now() -> datetime:
    return datetime.now(UTC)


class Engine:
    def __init__(
        self,
        root: Path | str,
        *,
        generation: GenerationModel | None = None,
        decision: DecisionModel | None = None,
        tools: list[Tool] | None = None,
        refinery: Recorder | None = None,
        max_steps: int = 200,
        max_depth: int = 5,
        tool_timeout: float | None = 30.0,
    ) -> None:
        if tool_timeout is not None and tool_timeout <= 0:
            raise ConfigurationError("engine: tool_timeout must be positive, or None for no deadline")
        if max_steps <= 0:
            raise ConfigurationError("engine: max_steps must be positive")
        if max_depth <= 0:
            raise ConfigurationError("engine: max_depth must be positive")
        self.root = Path(root).resolve()
        self.generation = generation
        self.decision = decision
        self.tools: dict[str, Tool] = {}
        for tool in tools or []:
            if tool.name in self.tools:
                raise ConfigurationError(f"engine: duplicate tool {tool.name!r}")
            self.tools[tool.name] = tool
        self.refinery = refinery
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.tool_timeout = tool_timeout
        self._agents = load_tree(self.root)
        validate_tree(self._agents)
        check_declarations(self._agents, self.tools, bodies=generation is not None)

    @property
    def agents(self) -> dict[str, AgentInfo]:
        return {
            name: AgentInfo(name=name, kind=spec.kind, description=spec.description, agents=spec.agents)
            for name, spec in self._agents.items()
        }

    def _spec(self, agent: str) -> AgentSpec:
        spec = self._agents.get(agent)
        if spec is None:
            raise NotFoundError(f"engine: agent {agent!r} was not found")
        return spec

    async def run(
        self,
        agent: str,
        scope: str,
        input: State,
        *,
        task: str | None = None,
        history: list[Message] | None = None,
        context: str | None = None,
        expect: type[BaseModel] | None = None,
        on_event: EventHandler | None = None,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        record: bool = True,
    ) -> Run:
        if not scope:
            raise InputError("engine: scope is required")
        spec = self._spec(agent)
        if task is not None and spec.kind == "decision":
            raise InputError(f"engine: decision agent {agent!r} takes no task")
        options = _Options(
            task=task, history=list(history or []), context=context, expect=expect,
            on_event=on_event, tools=tools, workspace=workspace,
        )
        run = await self._run(spec, scope, input, options, None)
        if record and self.refinery is not None:
            try:
                await self.refinery.record(run)
            except Exception:
                logger.warning("refinery.record failed for run %s", run.run_id, exc_info=True)
        return run

    async def _run(
        self,
        spec: AgentSpec,
        scope: str,
        input: State,
        options: _Options,
        parent_run_id: str | None,
    ) -> Run:
        async with enter(spec.name, self.max_depth):
            if spec.kind == "generation":
                return await self._generation(spec, scope, input, options, parent_run_id)
            return await self._decision(spec, scope, input, options, parent_run_id)

    async def _decision(
        self,
        spec: AgentSpec,
        scope: str,
        input: State,
        options: _Options,
        parent_run_id: str | None,
    ) -> Decision:
        model_ = self.decision
        if model_ is None:
            raise ConfigurationError(f"engine: agent {spec.name!r} is a decision agent; no decision model was configured")
        self._check_routes(spec, set())
        state = _state(spec, input)
        run_id = uuid4().hex
        started = time.monotonic()
        at = _now()
        questions = dict(spec.questions)
        decided = as_decided(await model_.decide(state, questions))
        answers = settle(questions, decided.answers)
        routed: Run | None = None
        route = spec.route
        if route is not None:
            answer = answers[route.by]
            target = route.to.get(answer.choice) if isinstance(answer, ChoiceAnswer) else None
            floor = route.min_confidence
            if target is not None and (floor is None or answer.confidence >= floor):
                routed = await self._run(self._agents[target], scope, input, options, run_id)
        return Decision(
            run_id=run_id, parent_run_id=parent_run_id, agent=spec.name, scope=scope, at=at,
            duration_ms=int((time.monotonic() - started) * 1000), model=decided.model,
            state=state, questions=questions, answers=answers,
            usage=decided.usage or Usage(), routed_to=routed,
        )

    def _check_routes(self, spec: AgentSpec, seen: set[str]) -> None:
        seen.add(spec.name)
        for target in spec.route.to.values() if spec.route is not None else []:
            routed = self._agents[target]
            if routed.kind == "generation" and self.generation is None:
                raise ConfigurationError(f"engine: agent {spec.name!r} routes to generation agent {target!r}; no generation model was configured")
            if routed.kind == "decision" and target not in seen:
                self._check_routes(routed, seen)

    async def _generation(
        self,
        spec: AgentSpec,
        scope: str,
        input: State,
        options: _Options,
        parent_run_id: str | None,
    ) -> Generation:
        model_ = self.generation
        if model_ is None:
            raise ConfigurationError(f"engine: agent {spec.name!r} is a generation agent; no generation model was configured")
        run_id = uuid4().hex
        started = time.monotonic()
        at = _now()
        expect, on_event = options.expect, options.on_event
        surface = EngineTools(self, agent=spec.name, scope=scope, tools=options.tools, workspace=options.workspace)
        specs = [ToolSpec.model_validate(d) for d in surface.definitions()]
        if expect is not None:
            if OUTPUT_TOOL in surface:
                raise ConfigurationError(f"engine: a tool named {OUTPUT_TOOL!r} collides with expect=")
            specs.append(
                ToolSpec(
                    name=OUTPUT_TOOL,
                    description="Emit the final reply. Call exactly once with the complete result.",
                    input_schema=expect.model_json_schema(),
                )
            )
        targets = [self._agents[name] for name in spec.agents]
        system = await asyncio.to_thread(
            build_bundle, spec, task=options.task, tools=specs, agents=targets, context=options.context
        )
        messages = [*options.history, Message(role="user", content=_content(input))]
        sent = list(messages)
        steps: list[Step] = []
        children: list[Run] = []
        usage = Usage()
        calls = 0
        model: str | None = None
        streaming = on_event is not None and hasattr(model_, "generate_stream")

        async def emit(event: Any) -> None:
            if on_event is not None:
                await on_event(event)

        async def delta(text: str) -> None:
            await emit(TextDelta(run_id=run_id, agent=spec.name, text=text))

        async def delegate(call: ToolCall) -> tuple[Content, bool, str | None]:
            try:
                args = permit(spec, call.args)
                forwarded = replace(options, task=None, history=[], expect=None)
                child = await self._run(self._agents[args.agent], scope, args.input, forwarded, run_id)
            except Exception as exc:
                return f"{type(exc).__name__}: {exc}", True, None
            children.append(child)
            return child_result(child), False, child.run_id

        async def call_tool(call: ToolCall) -> Step:
            await emit(ToolCallStarted(run_id=run_id, agent=spec.name, call_id=call.id, tool=call.name))
            began = time.monotonic()
            child_run_id: str | None = None
            content: Content
            if spec.agents and call.name == DELEGATE_TOOL:
                content, is_error, child_run_id = await delegate(call)
            else:
                deadline = asyncio.timeout(self.tool_timeout)
                try:
                    async with deadline:
                        result = await surface.dispatch(call.name, call.args)
                    content, is_error = _content(result), False
                except Exception as exc:
                    if deadline.expired():
                        content = f"TimeoutError: {call.name} timed out after {self.tool_timeout:g}s"
                    else:
                        content = f"{type(exc).__name__}: {exc}"
                    is_error = True
            step = Step(
                tool=call.name, args=call.args, result=content, is_error=is_error, at=_now(),
                duration_ms=int((time.monotonic() - began) * 1000), child_run_id=child_run_id,
            )
            await emit(ToolResultReady(run_id=run_id, agent=spec.name, call_id=call.id, tool=call.name, is_error=is_error))
            return step

        async def run_batch(batch: list[ToolCall]) -> None:
            for call, step in zip(batch, await asyncio.gather(*(call_tool(c) for c in batch)), strict=True):
                messages.append(Message(role="tool", tool_call_id=call.id, content=step.result, is_error=step.is_error))
                steps.append(step)

        def finish(reply_text: str) -> Generation:
            return Generation(
                run_id=run_id, parent_run_id=parent_run_id, agent=spec.name, scope=scope, task=options.task,
                at=at, duration_ms=int((time.monotonic() - started) * 1000), model=model,
                input=input, system=system, messages=sent, conversation=list(messages), steps=steps, calls=calls,
                reply=reply_text, usage=usage, children=children,
            )

        for _ in range(self.max_steps):
            if streaming:
                reply = as_reply(await cast(StreamingGenerationModel, model_).generate_stream(system, messages, specs, delta))
            else:
                reply = as_reply(await model_.generate(system, messages, specs))
                if reply.text:
                    await delta(reply.text)
            calls += 1
            if reply.usage is not None:
                usage = usage + reply.usage
            if reply.model:
                model = reply.model
            messages.append(_assistant(reply))
            if reply.stop_reason != "tool_use" or not reply.tool_calls:
                return finish(reply.text)
            output = next((c for c in reply.tool_calls if expect is not None and c.name == OUTPUT_TOOL), None)
            before = reply.tool_calls if output is None else reply.tool_calls[: reply.tool_calls.index(output)]
            await run_batch(before)
            if output is None or expect is None:
                continue
            try:
                parsed = expect.model_validate(output.args)
            except ValidationError as exc:
                errors = f"invalid; fix and call {OUTPUT_TOOL} again: {exc.errors(include_url=False)}"
                messages.append(Message(role="tool", tool_call_id=output.id, content=errors, is_error=True))
                steps.append(Step(tool=OUTPUT_TOOL, args=output.args, result=errors, is_error=True, at=_now(), duration_ms=0))
                await run_batch(reply.tool_calls[reply.tool_calls.index(output) + 1 :])
                continue
            messages.append(Message(role="tool", tool_call_id=output.id, content="accepted"))
            steps.append(Step(tool=OUTPUT_TOOL, args=output.args, result="accepted", is_error=False, at=_now(), duration_ms=0))
            return finish(parsed.model_dump_json())
        raise RunawayError(f"engine: run exceeded {self.max_steps} steps")
