import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated, Any, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from rfnry_engine import chain
from rfnry_engine.bundle import build_bundle
from rfnry_engine.errors import (
    ConfigurationError,
    InputError,
    NotFoundError,
    RunawayError,
)
from rfnry_engine.fingerprint import questions_fingerprint, tools_fingerprint
from rfnry_engine.foreground import (
    CALL,
    FOREGROUND,
    CallInfo,
    Foreground,
    halted,
    hooks,
    returned,
    taking,
)
from rfnry_engine.models import (
    Answer,
    ChoiceAnswer,
    Content,
    Decided,
    DecisionModel,
    GenerationModel,
    HistoryLike,
    Message,
    Part,
    Question,
    QuestionLike,
    Reply,
    State,
    StopReason,
    StreamingGenerationModel,
    Text,
    ToolCall,
    ToolSpec,
    Usage,
    as_decided,
    as_reply,
    settle,
)
from rfnry_engine.records import (
    Decision,
    EventHandler,
    Fingerprint,
    Generation,
    Run,
    Step,
    TextDelta,
    ToolCallStarted,
    ToolResultReady,
)
from rfnry_engine.stop import Stop, Stopped, requested, wait
from rfnry_engine.tools.declared import check_declarations
from rfnry_engine.tools.delegate import (
    DELEGATE_TOOL,
    AgentArgs,
    child_result,
    enter,
    permit,
    run_arguments,
    run_result,
)
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Result, Tool, Tools, ToolsLike
from rfnry_engine.tools.workspace import WORKSPACE_TOOLS, Workspace
from rfnry_engine.tree import AgentSpec, Kind, load_tree, validate_tree

OUTPUT_TOOL = "Output"
STOPPED = "stopped by the user"
NO_TOOLS = tools_fingerprint([])

Approve = Callable[[str, ToolCall], Awaitable[bool]]

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
    stop: Stop | None = None
    inbox: asyncio.Queue[Any] | None = None
    tags: dict[str, str] = field(default_factory=dict)
    approve: Approve | None = None
    questions: Mapping[str, Question] = field(default_factory=dict)
    start: int = 0
    source: Sequence[object] | HistoryLike | None = None
    failed: list[Run] = field(default_factory=list)


class AgentInfo(BaseModel):
    name: str
    kind: Kind
    description: str
    agents: list[str]
    reads: list[str]


class Recorder(Protocol):
    async def record(self, run: Run) -> None: ...


PART = TypeAdapter[Part](Part)
HISTORY = TypeAdapter(list[Message])
QUESTIONS = TypeAdapter(dict[str, Annotated[Question, Field(discriminator="type")]])
PART_TYPES = ("text", "image", "audio", "document")
MAX_TAGS = 32
MAX_TAG_KEY = 64
MAX_TAG_VALUE = 256


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


def _history(value: Sequence[object] | None) -> list[Message]:
    try:
        return HISTORY.validate_python(list(value or []), from_attributes=True)
    except ValidationError as exc:
        raise InputError(
            "engine: history does not fit the contract: "
            f"{exc.errors(include_url=False, include_input=False)}"
        ) from exc


def _paired(history: list[Message], start: int) -> None:
    calls: dict[str, list[tuple[int, int]]] = {}
    broken: tuple[int, int, str | None] | None = None
    for position, message in enumerate(history, start):
        if message.role == "tool":
            waiting = calls.get(message.tool_call_id or "")
            if not waiting:
                broken = (position, 0, message.tool_call_id)
                break
            waiting.pop(0)
        for order, call in enumerate(message.tool_calls):
            calls.setdefault(call.id, []).append((position, order))
    if broken is None:
        broken = min(
            ((at, order, call_id) for call_id, held in calls.items() for at, order in held),
            default=None,
        )
    if broken is not None:
        position, _, call_id = broken
        raise InputError(
            f"engine: history breaks tool call {call_id!r} from its result "
            f"at position {position}; start at a message the person sent"
        )


async def _window(history: Sequence[object] | HistoryLike | None, start: int) -> list[Message]:
    if isinstance(history, HistoryLike):
        if start < 0:
            raise InputError(f"engine: start must not be negative; got {start}")
        return _history(await history.read(start))
    if history is None and start:
        raise InputError("engine: start needs a history")
    held = history or []
    if not 0 <= start <= len(held):
        raise InputError(f"engine: start must be between 0 and {len(held)}; got {start}")
    return _history(held[start:])


def _questions(value: Mapping[str, QuestionLike] | None) -> dict[str, Question]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InputError("engine: questions must map question ids to questions")
    try:
        asked = QUESTIONS.validate_python(dict(value), from_attributes=True)
    except ValidationError as exc:
        raise InputError(
            f"engine: questions do not fit the contract: {exc.errors(include_url=False)}"
        ) from exc
    if not all(asked):
        raise InputError("engine: a question id must be a non-empty string")
    return asked


def _asked(spec: AgentSpec, questions: Mapping[str, QuestionLike] | None) -> dict[str, Question]:
    if questions is not None and spec.kind == "generation":
        raise InputError(f"engine: generation agent {spec.name!r} takes no questions")
    return _questions(questions)


def _tags(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InputError("engine: tags must be a mapping of strings to strings")
    if len(value) > MAX_TAGS:
        raise InputError(f"engine: tags hold at most {MAX_TAGS} entries; got {len(value)}")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise InputError(
                f"engine: tags must be a mapping of strings to strings; got {key!r}: {item!r}"
            )
        if not key or len(key) > MAX_TAG_KEY:
            raise InputError(
                f"engine: tags need a key of 1 to {MAX_TAG_KEY} characters; got {key!r}"
            )
        if len(item) > MAX_TAG_VALUE:
            raise InputError(
                f"engine: tags need a value of at most {MAX_TAG_VALUE} characters; {key!r} has {len(item)}"
            )
    return dict(value)


def _content(value: object) -> Content:
    if isinstance(value, str):
        return value
    parts = _parts(value)
    return parts if parts is not None else json.dumps(value, ensure_ascii=False)


def _state(spec: AgentSpec, value: State, media_types: Collection[str] | None) -> State:
    parts = _parts(value)
    if parts is None:
        return value
    media = [part for part in parts if not isinstance(part, Text)]
    if not media:
        return "\n".join(part.text for part in parts if isinstance(part, Text))
    takes = sorted(media_types or [])
    for part in media:
        if not any(fnmatch(part.media_type, pattern) for pattern in takes):
            accepted = f"text and {', '.join(takes)}" if takes else "text only"
            raise InputError(
                f"engine: decision agent {spec.name!r} takes {accepted}; got {part.media_type}"
            )
    return parts


def _assistant(reply: Reply, prefix: str) -> Message:
    return Message(
        role="assistant",
        content=reply.text,
        tool_calls=reply.tool_calls,
        reasoning=reply.reasoning,
        prefix=prefix,
        usage=reply.usage,
    )


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _now() -> datetime:
    return datetime.now(UTC)


def _declared(model: object) -> str | None:
    named = getattr(model, "model", None)
    return named if isinstance(named, str) and named else None


class Engine:
    def __init__(
        self,
        root: Path | str,
        *,
        name: str | None = None,
        generation: GenerationModel | None = None,
        decision: DecisionModel | None = None,
        tools: list[Tool] | None = None,
        refinery: Recorder | None = None,
        max_steps: int = 200,
        max_depth: int = 5,
        tool_timeout: float | None = 30.0,
        history_walk: int = 10_000,
        history_chars: int = 100_000,
    ) -> None:
        if tool_timeout is not None and tool_timeout <= 0:
            raise ConfigurationError(
                "engine: tool_timeout must be positive, or None for no deadline"
            )
        if max_steps <= 0:
            raise ConfigurationError("engine: max_steps must be positive")
        if max_depth <= 0:
            raise ConfigurationError("engine: max_depth must be positive")
        if history_walk <= 0:
            raise ConfigurationError("engine: history_walk must be positive")
        if history_chars <= 0:
            raise ConfigurationError("engine: history_chars must be positive")
        if name is not None and (
            not name.strip()
            or name != name.strip()
            or name.startswith(".")
            or "/" in name
            or "\\" in name
        ):
            raise ConfigurationError(
                f"engine: name must name one folder, not blank, hidden, padded or with slashes; got {name!r}"
            )
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ConfigurationError(f"engine: root is not a directory: {root}")
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
        self.history_walk = history_walk
        self.history_chars = history_chars
        self._agents = load_tree(self.root, getattr(generation, "media_types", None), name)
        validate_tree(self._agents)
        check_declarations(self._agents, self.tools, bodies=generation is not None)

    @property
    def agents(self) -> dict[str, AgentInfo]:
        return {
            name: AgentInfo(
                name=name,
                kind=spec.kind,
                description=spec.description,
                agents=spec.agents,
                reads=list(spec.reads),
            )
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
        questions: Mapping[str, QuestionLike] | None = None,
        history: Sequence[object] | HistoryLike | None = None,
        start: int = 0,
        context: str | None = None,
        expect: type[BaseModel] | None = None,
        on_event: EventHandler | None = None,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        record: bool = True,
        stop: Stop | None = None,
        inbox: asyncio.Queue[Any] | None = None,
        tags: Mapping[str, str] | None = None,
        approve: Approve | None = None,
    ) -> Run:
        if not scope:
            raise InputError("engine: scope is required")
        if workspace is not None and workspace.scope != scope:
            raise InputError(
                f"engine: workspace belongs to scope {workspace.scope!r}, not {scope!r}"
            )
        spec = self._spec(agent)
        asked = _asked(spec, questions)
        tagged = _tags(tags)
        held = history if history is None or isinstance(history, HistoryLike) else list(history)
        given = await _window(held, start)
        _paired(given, start)
        options = _Options(
            task=task,
            history=given,
            context=context,
            expect=expect,
            on_event=on_event,
            tools=tools,
            workspace=workspace,
            stop=stop,
            inbox=inbox,
            tags=tagged,
            approve=approve,
            questions=asked,
            start=start,
            source=held,
        )
        try:
            run = await self._run(spec, scope, input, options, None)
        except Exception:
            if record and options.failed:
                with taking(stop):
                    await self._record(options.failed[-1])
            raise
        if record:
            with taking(stop):
                await self._record(run)
        return run

    async def _record(self, run: Run) -> None:
        if self.refinery is None:
            return
        try:
            await self.refinery.record(run)
        except Exception:
            logger.warning("refinery.record failed for run %s", run.run_id, exc_info=True)

    def tool(
        self,
        agent: str,
        *,
        scope: str,
        name: str | None = None,
        description: str | None = None,
        context: str | None = None,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        tags: Mapping[str, str] | None = None,
        approve: Approve | None = None,
        on_event: EventHandler | None = None,
        stop: Callable[[], Stop | None] | None = None,
    ) -> Tools:
        if not scope:
            raise InputError("engine: scope is required")
        if workspace is not None and workspace.scope != scope:
            raise InputError(
                f"engine: workspace belongs to scope {workspace.scope!r}, not {scope!r}"
            )
        spec = self._spec(agent)
        tagged = _tags(tags)

        async def body(args: dict[str, Any]) -> Result:
            watching, held = hooks(on_event, stop)
            run = await self.run(
                agent,
                scope,
                args["input"],
                task=args.get("task"),
                questions=args.get("questions"),
                context=context,
                tools=tools,
                workspace=workspace,
                tags=tagged,
                approve=approve,
                on_event=watching,
                stop=held,
            )
            return run_result(returned(run))

        described = spec.description if description is None else description
        schema = {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "What to hand the agent"},
                **run_arguments([spec]),
            },
            "required": ["input"],
        }
        return Tools(
            [Tool(name or spec.name, described, body=body, args=AgentArgs, input_schema=schema)]
        )

    async def _run(
        self,
        spec: AgentSpec,
        scope: str,
        input: State,
        options: _Options,
        parent_run_id: str | None,
        routed: bool = False,
    ) -> Run:
        if options.task is not None and options.task not in spec.tasks:
            raise NotFoundError(f"engine: agent {spec.name!r} has no task {options.task!r}")
        async with enter(spec.name, self.max_depth, routed=routed):
            if spec.kind == "generation":
                with taking(options.stop):
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
            raise ConfigurationError(
                f"engine: agent {spec.name!r} is a decision agent; no decision model was configured"
            )
        self._check_routes(spec, set())
        folder = dict(spec.questions)
        if options.task is not None:
            folder.update(spec.task_questions[options.task])
        clash = [qid for qid in options.questions if qid in folder]
        if clash:
            raise InputError(
                f"engine: decision agent {spec.name!r} already asks {', '.join(map(repr, clash))} from its folder"
            )
        questions = {**folder, **options.questions}
        if not questions:
            hint = "name a task or pass questions=" if spec.tasks else "pass questions="
            raise InputError(
                f"engine: decision agent {spec.name!r} has no questions to ask; {hint}"
            )
        state = _state(spec, input, getattr(model_, "media_types", None))
        run_id = uuid4().hex
        started = time.monotonic()
        at = _now()
        decided: Decided | None = None
        answers: dict[str, Answer] = {}
        routed: Run | None = None
        routing: list[Run] = []

        def decision(error: Exception | None = None) -> Decision:
            return Decision(
                run_id=run_id,
                parent_run_id=parent_run_id,
                agent=spec.name,
                scope=scope,
                tags=dict(options.tags),
                task=options.task,
                at=at,
                duration_ms=int((time.monotonic() - started) * 1000),
                model=(decided.model if decided is not None else None) or _declared(model_),
                fingerprint=Fingerprint(
                    agent=spec.fingerprint,
                    tools=NO_TOOLS,
                    questions=questions_fingerprint(questions),
                ),
                state=state,
                questions=questions,
                runtime=list(options.questions),
                answers=answers,
                usage=(decided.usage if decided is not None else None) or Usage(),
                routed_to=routing[-1] if routing else routed,
                error=None if error is None else _failure(error),
            )

        try:
            decided = as_decided(await model_.decide(state, questions))
            answers = settle(questions, decided.answers)
            route = spec.route
            if route is not None:
                answer = answers[route.by]
                target = route.to.get(answer.choice) if isinstance(answer, ChoiceAnswer) else None
                floor = route.min_confidence
                if target is not None and (floor is None or answer.confidence >= floor):
                    forwarded = replace(options, task=target.task, questions={}, failed=routing)
                    routed = await self._run(
                        self._agents[target.agent], scope, input, forwarded, run_id, routed=True
                    )
        except Exception as exc:
            options.failed.append(decision(exc))
            raise
        return decision()

    def _check_routes(self, spec: AgentSpec, seen: set[str]) -> None:
        seen.add(spec.name)
        for target in spec.route.to.values() if spec.route is not None else []:
            routed = self._agents[target.agent]
            if routed.kind == "generation" and self.generation is None:
                raise ConfigurationError(
                    f"engine: agent {spec.name!r} routes to generation agent {target.agent!r}; no generation model was configured"
                )
            if routed.kind == "decision" and target.agent not in seen:
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
            raise ConfigurationError(
                f"engine: agent {spec.name!r} is a generation agent; no generation model was configured"
            )
        run_id = uuid4().hex
        started = time.monotonic()
        at = _now()
        expect, on_event, stop, inbox = (
            options.expect,
            options.on_event,
            options.stop,
            options.inbox,
        )
        surface = EngineTools(
            self,
            agent=spec.name,
            scope=scope,
            tools=options.tools,
            workspace=options.workspace,
            tags=options.tags,
            approve=options.approve,
            history=options.source,
            start=options.start,
        )
        bounded_by_workspace = (
            WORKSPACE_TOOLS
            if (spec.workspace or spec.plain) and options.workspace is not None
            else frozenset()
        )
        specs = [ToolSpec.model_validate(d) for d in surface.definitions()]
        if expect is not None:
            if OUTPUT_TOOL in surface:
                raise ConfigurationError(
                    f"engine: a tool named {OUTPUT_TOOL!r} collides with expect="
                )
            specs.append(
                ToolSpec(
                    name=OUTPUT_TOOL,
                    description="Emit the final reply. Call exactly once with the complete result.",
                    input_schema=expect.model_json_schema(),
                )
            )
        fingerprint = Fingerprint(
            agent=spec.fingerprint, tools=tools_fingerprint(specs), questions=None
        )
        targets = [self._agents[name] for name in spec.agents]
        system = await asyncio.to_thread(
            build_bundle,
            spec,
            task=options.task,
            tools=specs,
            agents=targets,
            context=options.context,
        )
        preload = [Message(role="user", content=list(spec.preload))] if spec.preload else []
        history, digest = await asyncio.to_thread(
            lambda: chain.settle(
                options.history, chain.opening(_declared(model_), system, specs, list(spec.preload))
            )
        )
        hashed = len(history)
        messages = [*history, Message(role="user", content=_content(input))]
        sent = list(messages)
        steps: list[Step] = []
        children: list[Run] = []
        delegated: dict[str, Run] = {}
        ran: list[tuple[ToolCall, Step]] = []
        approving: set[int] = set()
        foregrounds: dict[int, Foreground] = {}
        asked: list[ToolCall] = []
        usage = Usage()
        calls = 0
        model = _declared(model_)
        streaming = on_event is not None and hasattr(model_, "generate_stream")
        streamed: list[str] = []
        said = ""
        ended: StopReason | None = None
        runner = asyncio.current_task()
        calm = 0 if runner is None else runner.cancelling()

        async def emit(event: Any) -> None:
            if on_event is not None:
                await on_event(event)

        def called(call: ToolCall) -> CallInfo:
            return CallInfo(
                run_id=run_id, agent=spec.name, scope=scope, tool=call.name, call_id=call.id
            )

        async def hand_over(call: ToolCall, step: Step) -> Step:
            if runner is not None and runner.cancelling() > calm:
                raise asyncio.CancelledError("engine: the run was cancelled")
            ran.append((call, step))
            token = CALL.set(called(call))
            try:
                await emit(
                    ToolResultReady(run_id=run_id, agent=spec.name, call_id=call.id, step=step)
                )
            finally:
                CALL.reset(token)
            return step

        async def delta(text: str) -> None:
            streamed.append(text)
            await emit(TextDelta(run_id=run_id, agent=spec.name, turn=turn, text=text))

        async def delegate(call: ToolCall) -> tuple[Content, bool, str | None, bool]:
            if requested(stop) == "now":
                return STOPPED, True, None, True
            failed: list[Run] = []
            try:
                args = permit(spec, call.args)
                target = self._agents[args.agent]
                forwarded = replace(
                    options,
                    task=args.task,
                    questions=_asked(target, args.questions),
                    history=[],
                    expect=None,
                    inbox=None,
                    start=0,
                    source=None,
                    failed=failed,
                )
                child = await self._run(target, scope, args.input, forwarded, run_id)
            except Exception as exc:
                kept = failed[-1] if failed else None
                if kept is not None:
                    delegated[kept.run_id] = kept
                return _failure(exc), True, None if kept is None else kept.run_id, False
            delegated[child.run_id] = child
            if halted(child):
                return STOPPED, True, child.run_id, True
            return child_result(child), False, child.run_id, False

        async def call_tool(call: ToolCall) -> Step:
            CALL.set(called(call))
            await emit(
                ToolCallStarted(
                    run_id=run_id,
                    agent=spec.name,
                    call_id=call.id,
                    tool=call.name,
                    args=call.args,
                )
            )
            began = time.monotonic()
            if requested(stop) == "now":
                return await cut(call, began)
            child_run_id: str | None = None
            stopped = False
            content: Content
            approved = None
            if options.approve is not None:
                asking = asyncio.ensure_future(options.approve(spec.name, call))
                approving.add(id(call))
                try:
                    halted = await wait([asking], stop)
                finally:
                    approving.discard(id(call))
                if halted:
                    return await cut(call, began)
                approved = asking.result()
            if approved is not None and not isinstance(approved, bool):
                raise ConfigurationError(
                    f"engine: approve must return True or False; got {approved!r}"
                )
            here = Foreground(stop, on_event, asyncio.current_task())
            foregrounds[id(call)] = here
            FOREGROUND.set(here)
            try:
                if approved is False:
                    content, is_error = f"NotApproved: {call.name} was not approved", True
                elif spec.agents and call.name == DELEGATE_TOOL:
                    content, is_error, child_run_id, stopped = await delegate(call)
                else:
                    limit = None if call.name in bounded_by_workspace else self.tool_timeout
                    deadline = asyncio.timeout(limit)
                    try:
                        async with deadline:
                            result = await surface.dispatch(call.name, call.args)
                        content, is_error = _content(result), False
                    except Exception as exc:
                        if deadline.expired():
                            content = f"TimeoutError: {call.name} timed out after {limit:g}s"
                        else:
                            content = f"{type(exc).__name__}: {exc}"
                        is_error = True
                    if here.stopped:
                        content, is_error, stopped = STOPPED, True, True
            finally:
                here.over = True
            step = Step(
                tool=call.name,
                args=call.args,
                result=content,
                is_error=is_error,
                at=_now(),
                duration_ms=int((time.monotonic() - began) * 1000),
                child_run_id=child_run_id,
                stopped=stopped,
                approved=approved,
            )
            return await hand_over(call, step)

        def answer(call: ToolCall, step: Step) -> None:
            messages.append(
                Message(
                    role="tool", tool_call_id=call.id, content=step.result, is_error=step.is_error
                )
            )
            steps.append(step)
            if step.child_run_id is not None:
                children.append(delegated.pop(step.child_run_id))

        def not_run(call: ToolCall, why: str) -> None:
            answer(
                call,
                Step(
                    tool=call.name,
                    args=call.args,
                    result=f"NotRun: {call.name} was not run; {why}",
                    is_error=True,
                    at=_now(),
                    duration_ms=0,
                ),
            )

        async def cut(call: ToolCall, began: float) -> Step:
            step = Step(
                tool=call.name,
                args=call.args,
                result=STOPPED,
                is_error=True,
                at=_now(),
                duration_ms=int((time.monotonic() - began) * 1000),
                stopped=True,
            )
            return await hand_over(call, step)

        async def run_batch(batch: list[ToolCall]) -> bool:
            began = time.monotonic()
            if requested(stop) == "now":
                for call in batch:
                    answer(call, await cut(call, began))
                return True
            tasks = [asyncio.ensure_future(call_tool(c)) for c in batch]

            def finishing() -> list[asyncio.Task[Step]]:
                kept = []
                for call, task in zip(batch, tasks, strict=True):
                    here = foregrounds.get(id(call))
                    if here is not None and here.live > 0:
                        here.pressed = True
                        kept.append(task)
                    elif id(call) in approving or any(call is done for done, _ in ran):
                        kept.append(task)
                return kept

            stopped = await wait(tasks, stop, finishing)
            for call, task in zip(batch, tasks, strict=True):
                answer(call, await cut(call, began) if task.cancelled() else task.result())
                foregrounds.pop(id(call), None)
            return stopped

        def drain() -> None:
            while inbox is not None and not inbox.empty():
                messages.append(Message(role="user", content=_content(inbox.get_nowait())))
                inbox.task_done()

        def marked() -> str:
            nonlocal digest, hashed
            for message in messages[hashed:]:
                digest = chain.extend(digest, message)
            hashed = len(messages)
            return chain.mark(digest)

        async def ask() -> Reply:
            sending = [
                *preload,
                *(m.model_copy(update={"prefix": None, "usage": None}) for m in messages),
            ]
            if streaming:
                return as_reply(
                    await cast(StreamingGenerationModel, model_).generate_stream(
                        system, sending, specs, delta
                    )
                )
            return as_reply(await model_.generate(system, sending, specs))

        def finish(
            reply_text: str, stopped: Stopped | None = None, error: Exception | None = None
        ) -> Generation:
            return Generation(
                run_id=run_id,
                parent_run_id=parent_run_id,
                agent=spec.name,
                scope=scope,
                tags=dict(options.tags),
                task=options.task,
                context=options.context,
                at=at,
                duration_ms=int((time.monotonic() - started) * 1000),
                model=model,
                fingerprint=fingerprint,
                input=input,
                system=system,
                preload=list(spec.preload),
                start=options.start,
                messages=sent,
                conversation=list(messages),
                tools=specs,
                steps=steps,
                calls=calls,
                reply=reply_text,
                usage=usage,
                children=children,
                stopped=stopped,
                stop_reason=ended,
                error=None if error is None else _failure(error),
            )

        try:
            for turn in range(self.max_steps):
                mode = requested(stop)
                if mode is not None:
                    return finish(said, mode)
                if turn:
                    drain()
                streamed.clear()
                after = await asyncio.to_thread(marked)
                asking = asyncio.ensure_future(ask())
                if await wait([asking], stop):
                    partial = "".join(streamed)
                    if partial:
                        messages.append(Message(role="assistant", content=partial))
                    return finish(partial or said, "now")
                reply = asking.result()
                ended = reply.stop_reason
                said = reply.text
                calls += 1
                if reply.usage is not None:
                    usage = usage + reply.usage
                if reply.model:
                    model = reply.model
                messages.append(_assistant(reply, after))
                asked = list(reply.tool_calls)
                ran.clear()
                if not streaming and reply.text:
                    await delta(reply.text)
                if reply.stop_reason != "tool_use" or not reply.tool_calls:
                    for call in reply.tool_calls:
                        not_run(call, f"the reply ended with {reply.stop_reason}")
                    return finish(reply.text, "now" if requested(stop) == "now" else None)
                output = next(
                    (c for c in reply.tool_calls if expect is not None and c.name == OUTPUT_TOOL),
                    None,
                )
                before = (
                    reply.tool_calls
                    if output is None
                    else reply.tool_calls[: reply.tool_calls.index(output)]
                )
                stopped = await run_batch(before)
                if output is None or expect is None:
                    continue
                if stopped:
                    for call in reply.tool_calls[reply.tool_calls.index(output) :]:
                        answer(call, await cut(call, time.monotonic()))
                    return finish(said, "now")
                try:
                    parsed = expect.model_validate(output.args)
                except ValidationError as exc:
                    errors = f"invalid; fix and call {OUTPUT_TOOL} again: {exc.errors(include_url=False)}"
                    messages.append(
                        Message(role="tool", tool_call_id=output.id, content=errors, is_error=True)
                    )
                    steps.append(
                        Step(
                            tool=OUTPUT_TOOL,
                            args=output.args,
                            result=errors,
                            is_error=True,
                            at=_now(),
                            duration_ms=0,
                        )
                    )
                    await run_batch(reply.tool_calls[reply.tool_calls.index(output) + 1 :])
                    continue
                messages.append(Message(role="tool", tool_call_id=output.id, content="accepted"))
                steps.append(
                    Step(
                        tool=OUTPUT_TOOL,
                        args=output.args,
                        result="accepted",
                        is_error=False,
                        at=_now(),
                        duration_ms=0,
                    )
                )
                for call in reply.tool_calls[reply.tool_calls.index(output) + 1 :]:
                    not_run(call, f"{OUTPUT_TOOL} ended the run")
                return finish(parsed.model_dump_json())
            mode = requested(stop)
            if mode is not None:
                return finish(said, mode)
            raise RunawayError(f"engine: run exceeded {self.max_steps} steps")
        except Exception as exc:
            for call in asked:
                built = next((step for done, step in ran if done is call), None)
                if built is not None and not any(built is kept for kept in steps):
                    answer(call, built)
            children.extend(delegated.values())
            options.failed.append(finish("", error=exc))
            raise
