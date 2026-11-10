import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from fnmatch import fnmatch
from functools import partial
from itertools import count
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, NoReturn, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError

from rfnry_engine import chain
from rfnry_engine.bundle import build_bundle
from rfnry_engine.compiled import compile_folder
from rfnry_engine.errors import (
    ConfigurationError,
    InputError,
    ModelError,
    NotFoundError,
    RunawayError,
    WorkerError,
)
from rfnry_engine.fingerprint import questions_fingerprint, tools_fingerprint
from rfnry_engine.foreground import (
    CALL,
    FOREGROUND,
    RECORDING,
    Foreground,
    Lent,
    halted,
    hooks,
    returned,
    taking,
)
from rfnry_engine.frontmatter import plainly
from rfnry_engine.manifest import Kind, ModelEntry, read_manifest
from rfnry_engine.models import (
    Answer,
    ChoiceAnswer,
    Content,
    Decided,
    DecisionModel,
    GenerationModel,
    HistoryLike,
    Message,
    ModelChoice,
    Part,
    Provider,
    Question,
    QuestionLike,
    Reply,
    RunModel,
    State,
    StopReason,
    StreamingGenerationModel,
    Text,
    ToolCall,
    ToolSpec,
    Usage,
    VoiceModel,
    as_decided,
    as_reply,
    settle,
)
from rfnry_engine.records import (
    CallInfo,
    CommandOutput,
    Decision,
    EventHandler,
    Fingerprint,
    Generation,
    MessagesTaken,
    Pipe,
    ReasoningDelta,
    Run,
    RunStarted,
    Spoken,
    Step,
    TextDelta,
    ToolCallStarted,
    ToolResultReady,
    Voice,
    VoiceStart,
)
from rfnry_engine.stop import Stop, Stopped, cancel, requested, wait
from rfnry_engine.tools.declared import chosen, smallest
from rfnry_engine.tools.delegate import (
    DELEGATE_TOOL,
    AgentArgs,
    child_result,
    enter,
    folder_reach,
    permit,
    run_arguments,
    run_result,
)
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Result, Tool, Tools, ToolsLike
from rfnry_engine.tools.workspace import WORKSPACE_NAMES, WORKSPACE_TOOLS, Workspace
from rfnry_engine.tree import (
    OUTPUT_TOOL,
    TAKEN,
    AgentSpec,
    Skill,
    as_question,
    foreign,
    no_such_skill,
    skill_text,
)
from rfnry_engine.workers.pool import Script, Workers

STOPPED = "stopped by the user"
NO_TOOLS = tools_fingerprint([])

Approve = Callable[[str, ToolCall], Awaitable[bool]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Options:
    history: list[Message]
    context: str | None
    expect: type[BaseModel] | None
    on_event: EventHandler | None
    tools: ToolsLike | None
    workspace: Workspace | None
    models: Mapping[str, RunModel]
    stop: Stop | None = None
    inbox: asyncio.Queue[Any] | None = None
    skill: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    approve: Approve | None = None
    questions: Mapping[str, Question] = field(default_factory=dict)
    start: int = 0
    source: Sequence[object] | HistoryLike | None = None
    failed: list[Generation | Decision] = field(default_factory=list)


class AgentInfo(BaseModel):
    name: str
    kind: Kind
    model: str
    description: str
    delegate_to: list[str]
    files: list[str]
    tools: list[str]
    skills: dict[str, str]
    public: dict[str, str]
    worker: str
    modes: dict[str, AgentInfo]


MAX_WORKER_PROCESSES = 16


class Recorder(Protocol):
    async def record(self, run: Run) -> None: ...


PART = TypeAdapter[Part](Part)
STATE = TypeAdapter[State](State)
HISTORY = TypeAdapter(list[Message])
SETTINGS = TypeAdapter[dict[str, JsonValue] | None](dict[str, JsonValue] | None)
QUESTIONS = TypeAdapter(dict[str, Annotated[Question, Field(discriminator="type")]])
PART_TYPES = ("text", "image", "audio", "document")
SCALARS = (str, int)
CONTAINERS = (dict, list)
SENT = {"prefix": None, "usage": None, "model": None, "settings": None}
_LEAVE = object()
MAX_TAGS = 32
MAX_TAG_KEY = 64
MAX_TAG_VALUE = 256


def _looks_like_part(item: object) -> bool:
    return isinstance(item, BaseModel) and getattr(item, "type", None) in PART_TYPES


def _part(item: object) -> Part | None:
    if _looks_like_part(item):
        return PART.validate_python(item, from_attributes=True)
    return None


def _at(steps: list[object]) -> str:
    return "input" + "".join(f"[{step!r}]" for step in steps[1:])


def _unfit(input: object) -> str | None:
    if isinstance(input, str):
        return None
    if not isinstance(input, CONTAINERS):
        return type(input).__name__
    trail: list[object] = []
    path: set[int] = set()
    seen: set[int] = set()
    pending: list[tuple[Any, object]] = [(input, None)]
    while pending:
        value, step = pending.pop()
        if value is _LEAVE:
            path.discard(cast(int, step))
            trail.pop()
            continue
        if not isinstance(value, CONTAINERS):
            return f"{type(value).__name__} at {_at([*trail, step])}"
        held = id(value)
        if held in seen:
            if held in path:
                return f"a cycle at {_at([*trail, step])}"
            continue
        seen.add(held)
        inner: list[tuple[Any, object]] | None = None
        pairs = value.items() if isinstance(value, dict) else enumerate(value)
        keyed = isinstance(value, dict)
        parts = value is input and not keyed
        for key, item in pairs:
            if keyed and not isinstance(key, str):
                return f"the key {key!r} at {_at([*trail, step])}"
            if item is None or isinstance(item, SCALARS):
                continue
            if isinstance(item, float):
                if not math.isfinite(item):
                    return f"{item!r} at {_at([*trail, step, key])}"
                continue
            if parts and _looks_like_part(item):
                continue
            if inner is None:
                inner = []
            inner.append((item, key))
        if inner is not None:
            path.add(held)
            trail.append(step)
            pending.append((_LEAVE, held))
            pending.extend(inner)
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
        given = HISTORY.validate_python(list(value or []), from_attributes=True)
    except ValidationError as exc:
        raise InputError(
            "engine: history does not fit the contract: "
            f"{exc.errors(include_url=False, include_input=False)}"
        ) from exc
    return [_handed(message) for message in given]


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
    return {qid: _fresh(question) for qid, question in asked.items()}


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


def _assistant(
    reply: Reply, prefix: str | None, model: str | None, settings: dict[str, JsonValue] | None
) -> Message:
    return Message(
        role="assistant",
        content=reply.text,
        tool_calls=[_call(call) for call in reply.tool_calls],
        reasoning=_copied(reply.reasoning),
        prefix=prefix,
        usage=reply.usage,
        model=reply.model or model,
        settings=reply.settings if reply.settings is not None else settings,
    )


def _invoked(skill: Skill, said: Content) -> list[Part]:
    rest: list[Part] = said if isinstance(said, list) else [Text(text=said)] if said else []
    return [skill_text(skill), *rest]


def _skilled(spec: AgentSpec, skill: str | None) -> str | None:
    if skill is None:
        return None
    if spec.kind == "decision":
        raise InputError(
            f"engine: decision agent {spec.name!r} takes no skill; skills are for generation agents"
        )
    if skill not in spec.skills:
        raise NotFoundError(f"engine: {no_such_skill(spec, skill)}")
    return skill


def _command(spec: AgentSpec, text: str) -> tuple[str, str] | None:
    if not text.startswith("/"):
        return None
    word, *rest = text.split(maxsplit=1)
    if word[1:] not in spec.public:
        return None
    return word[1:], rest[0] if rest else ""


def _slashed(spec: AgentSpec, input: State) -> tuple[str, State] | None:
    if isinstance(input, str):
        return _command(spec, input)
    parts = _parts(input)
    if not parts or not isinstance(parts[0], Text):
        return None
    read = _command(spec, parts[0].text)
    if read is None:
        return None
    named, rest = read
    kept = [Text(text=rest), *parts[1:]] if rest else parts[1:]
    return named, kept or ""


def _copied(value: Any) -> Any:
    if not isinstance(value, CONTAINERS):
        return value
    top = dict(value) if isinstance(value, dict) else list(value)
    made: dict[int, Any] = {id(value): top}
    pending: list[Any] = [top]
    while pending:
        target = pending.pop()
        for key, item in target.items() if isinstance(target, dict) else enumerate(target):
            if isinstance(item, CONTAINERS):
                held = made.get(id(item))
                if held is None:
                    held = made[id(item)] = dict(item) if isinstance(item, dict) else list(item)
                    pending.append(held)
                target[key] = held
    return top


def _given(state: State) -> State:
    held: State = _copied(state)
    if isinstance(held, list):
        for index, item in enumerate(held):
            if isinstance(item, BaseModel):
                held[index] = item.model_copy()
    return held


def _fresh[M: BaseModel](model: M) -> M:
    return model.model_copy(
        update={
            name: _copied(value)
            for name, value in vars(model).items()
            if isinstance(value, CONTAINERS)
        }
    )


def _own(content: Content) -> Content:
    return content if isinstance(content, str) else [part.model_copy() for part in content]


def _call(call: ToolCall) -> ToolCall:
    return call.model_copy(update={"args": _copied(call.args)})


def _handed(message: Message, sent: bool = False) -> Message:
    update: dict[str, Any] = {
        "content": _own(message.content),
        "tool_calls": [_call(call) for call in message.tool_calls],
    }
    if message.reasoning is not None:
        update["reasoning"] = _copied(message.reasoning)
    if sent:
        update.update(SENT)
    elif message.settings is not None:
        update["settings"] = _copied(message.settings)
    return message.model_copy(update=update)


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _now() -> datetime:
    return datetime.now(UTC)


def _caller(scope: str) -> CallInfo | None:
    call = CALL.get()
    return call if call is not None and call.scope == scope else None


def _declared(model: object) -> str | None:
    named = getattr(model, "model", None)
    return named if isinstance(named, str) and named else None


def _settings(model: object) -> dict[str, JsonValue] | None:
    try:
        return SETTINGS.validate_python(getattr(model, "settings", None))
    except ValidationError:
        return None


def _lowered(folder: int | None, ceiling: int) -> int:
    return ceiling if folder is None else min(folder, ceiling)


def _slots(cap: int | None) -> AbstractAsyncContextManager[Any]:
    return nullcontext() if cap is None else asyncio.Semaphore(cap)


def _kind_clash(spec: AgentSpec, verb: str) -> NoReturn:
    raise ConfigurationError(
        f"engine: agent {spec.name!r} is a {spec.kind} agent, and model {spec.model!r} has no {verb}()"
    )


def _models(models: Mapping[str, RunModel]) -> dict[str, RunModel]:
    if not isinstance(models, Mapping):
        raise ConfigurationError("engine: models must be a mapping of model name to adapter")
    for name, model in models.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise ConfigurationError(
                f"engine: model name {name!r} must be non-blank text without padding"
            )
        if isinstance(model, ModelChoice):
            continue
        if not isinstance(model, GenerationModel | DecisionModel | VoiceModel):
            raise ConfigurationError(
                f"engine: model {name!r} has none of generate(), decide() and open(); pass an adapter that has one, or a ModelChoice"
            )
    return dict(models)


def _providers(providers: Mapping[str, Provider] | None) -> dict[str, Provider]:
    if providers is None:
        return {}
    if not isinstance(providers, Mapping):
        raise ConfigurationError(
            "engine: providers must be a mapping of provider name to a function of (model, effort) that returns an adapter"
        )
    for name, provider in providers.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise ConfigurationError(
                f"engine: provider name {name!r} must be non-blank text without padding"
            )
        if not callable(provider):
            raise ConfigurationError(
                f"engine: provider {name!r} cannot be called; pass a function of (model, effort) that returns an adapter"
            )
    return dict(providers)


def _not_held(name: str, provider: str | None, held: Collection[str]) -> NoReturn:
    raise ConfigurationError(
        f"engine: model {name!r} names provider {provider!r}, which Engine(providers=) does not hold; it holds {', '.join(sorted(held)) or 'none'}"
    )


def _check_providers(entries: Mapping[str, ModelEntry], held: Collection[str]) -> None:
    for name, entry in entries.items():
        if entry.provider not in held:
            _not_held(name, entry.provider, held)


def _of_its_kind(spec: AgentSpec, model: object) -> None:
    if spec.kind == "generation" and not isinstance(model, GenerationModel):
        _kind_clash(spec, "generate")
    if spec.kind == "decision" and not isinstance(model, DecisionModel):
        _kind_clash(spec, "decide")
    if spec.kind == "voice" and not isinstance(model, VoiceModel):
        raise ConfigurationError(
            f"engine: agent {spec.name!r} is a voice agent, which needs a live model, and model {spec.model!r} has no open()"
        )


def _with_modes(agents: Mapping[str, AgentSpec]) -> list[AgentSpec]:
    return [held for spec in agents.values() for held in (spec, *spec.modes.values())]


def _check_names(
    agents: Mapping[str, AgentSpec],
    models: Mapping[str, RunModel],
) -> None:
    used = sorted({spec.model for spec in _with_modes(agents)})
    for name in sorted(models):
        if name not in used:
            raise ConfigurationError(
                f"engine: no agent runs on model {name!r}; the agents of rfnry.yaml run on {', '.join(used)}"
            )


def _check_processes(workers: Workers, ceiling: int, raise_it: str) -> None:
    if workers.processes > ceiling:
        raise ConfigurationError(
            f"engine: the workers of this folder run {workers.processes} processes in all, over max_worker_processes={ceiling}; lower processes: under workers:, or {raise_it}"
        )


def _info(name: str, spec: AgentSpec) -> AgentInfo:
    return AgentInfo(
        name=name,
        kind=spec.kind,
        model=spec.model,
        description=spec.description,
        delegate_to=list(spec.agents),
        files=list(spec.reachable),
        tools=list(chosen(spec)),
        skills={called: skill.description for called, skill in spec.skills.items()},
        public={called: spec.skills[called].description for called in spec.public},
        worker=spec.worker,
        modes={mode: _info(name, one) for mode, one in spec.modes.items()},
    )


def _infos(agents: Mapping[str, AgentSpec]) -> dict[str, AgentInfo]:
    return {name: _info(name, spec) for name, spec in agents.items()}


@dataclass
class _ToolCalls:
    spec: AgentSpec
    run_id: str
    scope: str
    tags: Mapping[str, str]
    surface: ToolsLike
    deadline: float | None
    own_deadline: frozenset[str]
    tool_slots: AbstractAsyncContextManager[Any]
    approve: Approve | None
    on_event: EventHandler | None
    stop: Stop | None = None
    shell_slots: AbstractAsyncContextManager[Any] = field(default_factory=nullcontext)
    delegate_slots: AbstractAsyncContextManager[Any] = field(default_factory=nullcontext)
    lend: Callable[[ToolCall, Foreground], None] | None = None
    delegate: Callable[[ToolCall], Awaitable[tuple[Content, bool, str | None, bool]]] | None = None
    ran: list[tuple[ToolCall, Step]] = field(default_factory=list)
    approving: set[int] = field(default_factory=set)
    foregrounds: dict[int, Foreground] = field(default_factory=dict)
    speaking: set[asyncio.Task[Result]] = field(default_factory=set)
    over: bool = False
    runner: asyncio.Task[Any] | None = field(init=False)
    calm: int = field(init=False)

    def __post_init__(self) -> None:
        self.runner = asyncio.current_task()
        self.calm = 0 if self.runner is None else self.runner.cancelling()

    async def emit(self, event: Any) -> None:
        if self.on_event is not None:
            await self.on_event(event)

    def called(self, call: ToolCall, by: str | None = None) -> CallInfo:
        return CallInfo(
            run_id=self.run_id,
            agent=self.spec.name,
            scope=self.scope,
            tool=call.name,
            call_id=call.id,
            by=by,
            tags=self.tags,
        )

    async def hand_over(self, call: ToolCall, step: Step) -> Step:
        if self.runner is not None and self.runner.cancelling() > self.calm:
            raise asyncio.CancelledError("engine: the run was cancelled")
        self.ran.append((call, step))
        if self.on_event is None:
            return step
        token = CALL.set(self.called(call, step.by))
        try:
            await self.emit(
                ToolResultReady(
                    run_id=self.run_id,
                    agent=self.spec.name,
                    call_id=call.id,
                    step=step.model_copy(
                        update={"args": _copied(step.args), "result": _own(step.result)}
                    ),
                    by=step.by,
                )
            )
        finally:
            CALL.reset(token)
        return step

    def ended(
        self, call: ToolCall, began: float, result: str, by: str | None, *, stopped: bool
    ) -> Step:
        here = self.foregrounds.get(id(call))
        return Step(
            tool=call.name,
            args=call.args,
            result=result,
            is_error=True,
            at=_now(),
            duration_ms=int((time.monotonic() - began) * 1000),
            stopped=stopped,
            approved=None if here is None else here.approved,
            by=by,
        )

    def stopped(self, call: ToolCall, began: float, by: str | None = None) -> Step:
        return self.ended(call, began, STOPPED, by, stopped=True)

    async def cut(self, call: ToolCall, began: float, by: str | None = None) -> Step:
        return await self.hand_over(call, self.stopped(call, began, by))

    async def run(self, call: ToolCall, by: str | None = None) -> Step:
        CALL.set(self.called(call, by))
        if self.on_event is not None:
            await self.emit(
                ToolCallStarted(
                    run_id=self.run_id,
                    agent=self.spec.name,
                    call_id=call.id,
                    tool=call.name,
                    args=_copied(call.args),
                    by=by,
                )
            )
        began = time.monotonic()
        if requested(self.stop) == "now":
            return await self.cut(call, began, by)
        child_run_id: str | None = None
        cut_short = False
        content: Content
        approved = None
        if self.approve is not None:
            asking = asyncio.ensure_future(self.approve(self.spec.name, _call(call)))
            self.approving.add(id(call))
            try:
                pressed = await wait([asking], self.stop)
            finally:
                self.approving.discard(id(call))
            if pressed:
                return await self.cut(call, began, by)
            approved = asking.result()
        if approved is not None and not isinstance(approved, bool):
            raise ConfigurationError(f"engine: approve must return True or False; got {approved!r}")
        here = Foreground(self.stop, self.on_event, asyncio.current_task(), approved=approved)
        if self.lend is not None:
            self.lend(call, here)
        self.foregrounds[id(call)] = here
        FOREGROUND.set(here)
        try:
            if approved is False:
                content, is_error = f"NotApproved: {call.name} was not approved", True
            elif call.name == DELEGATE_TOOL and self.delegate is not None:
                async with self.delegate_slots:
                    content, is_error, child_run_id, cut_short = await self.delegate(call)
            else:
                limit = None if call.name in self.own_deadline else self.deadline
                async with self.shell_slots if call.name == "Shell" else self.tool_slots:
                    deadline = asyncio.timeout(limit)
                    try:
                        async with deadline:
                            here.result = await self.surface.dispatch(call.name, _copied(call.args))
                        content, is_error = _own(_content(here.result)), False
                    except Exception as exc:
                        if deadline.expired():
                            content = (
                                f"TimeoutError: engine: {call.name} timed out after {limit:g}s"
                            )
                        else:
                            content = _failure(exc)
                        is_error = True
                if here.stopped or here.pressed:
                    content, is_error, cut_short = STOPPED, True, True
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
            stopped=cut_short,
            approved=approved,
            by=by,
        )
        if here.failed is not None:
            self.ran.append((call, step))
            raise here.failed
        return await self.hand_over(call, step)

    async def settle(self, call: ToolCall, step: Step) -> None:
        if any(call is done for done, _ in self.ran):
            return
        try:
            await self.hand_over(call, step)
        except Exception:
            logger.error(
                "engine: on_event failed on the last step of call %s of run %s",
                call.id,
                self.run_id,
                exc_info=True,
            )

    async def spoken_call(self, call: ToolCall) -> Result:
        began = time.monotonic()
        try:
            step = await self.run(call)
        except asyncio.CancelledError:
            await self.settle(call, self.stopped(call, began))
            raise
        except Exception as exc:
            await self.settle(call, self.ended(call, began, _failure(exc), None, stopped=False))
            raise
        finally:
            here = self.foregrounds.pop(id(call), None)
        if step.is_error:
            raise InputError(f"engine: {step.result}")
        return cast(Result, cast(Foreground, here).result)

    async def speak(self, name: str, args: dict[str, Any]) -> Result:
        if self.over:
            raise InputError(
                f"engine: the session of agent {self.spec.name!r} has ended; a tool call cannot start after it"
            )
        call = ToolCall(id=uuid4().hex, name=name, args=_copied(args))
        running = asyncio.ensure_future(self.spoken_call(call))
        self.speaking.add(running)
        running.add_done_callback(self.speaking.discard)
        return await running


class _SpokenTools(Tools):
    def __init__(self, calls: _ToolCalls, specs: Sequence[ToolSpec]) -> None:
        super().__init__(
            Tool(
                spec.name,
                spec.description,
                body=partial(calls.speak, spec.name),
                input_schema=_copied(spec.input_schema),
            )
            for spec in specs
        )
        self._speak = calls.speak

    async def dispatch(self, name: str, args: dict[str, Any]) -> Result:
        return await self._speak(name, args)


def _spoken(agent: str, value: object) -> Spoken:
    if not isinstance(value, Spoken):
        raise InputError(
            f"engine: the session of agent {agent!r} returned a value of type {type(value).__name__}, not a Spoken; return Spoken(transcript=...)"
        )
    return value


def check_folder(
    root: Path, max_worker_processes: int = MAX_WORKER_PROCESSES
) -> dict[str, AgentInfo]:
    root = Path(root).resolve()
    manifest = read_manifest(root)
    agents = compile_folder(root, manifest)
    _check_processes(
        Workers(agents, manifest.workers, None),
        max_worker_processes,
        "pass --max-worker-processes",
    )
    return _infos(agents)


class Engine:
    def __init__(
        self,
        root: Path | str,
        *,
        providers: Mapping[str, Provider] | None = None,
        refinery: Recorder | None = None,
        max_model_calls: int = 200,
        max_depth: int = 5,
        tool_timeout: float | None = 30.0,
        max_worker_processes: int = MAX_WORKER_PROCESSES,
        worker_result_bytes: int | None = 33_554_432,
        history_walk: int = 10_000,
        history_chars: int = 100_000,
    ) -> None:
        if tool_timeout is not None and tool_timeout <= 0:
            raise ConfigurationError(
                "engine: tool_timeout must be positive, or None for no deadline"
            )
        if worker_result_bytes is not None and (
            not isinstance(worker_result_bytes, int)
            or isinstance(worker_result_bytes, bool)
            or worker_result_bytes <= 0
        ):
            raise ConfigurationError(
                "engine: worker_result_bytes must be a positive number of bytes, or None for no limit"
            )
        for name, cap in (
            ("max_model_calls", max_model_calls),
            ("max_depth", max_depth),
            ("max_worker_processes", max_worker_processes),
        ):
            if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
                raise ConfigurationError(f"engine: {name} must be a positive integer")
        if history_walk <= 0:
            raise ConfigurationError("engine: history_walk must be positive")
        if history_chars <= 0:
            raise ConfigurationError("engine: history_chars must be positive")
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ConfigurationError(f"engine: root is not a directory: {root}")
        self.refinery = refinery
        self.max_model_calls = max_model_calls
        self.max_depth = max_depth
        self.tool_timeout = tool_timeout
        self.worker_result_bytes = worker_result_bytes
        self.history_walk = history_walk
        self.history_chars = history_chars
        self._providers: Mapping[str, Provider] = MappingProxyType(_providers(providers))
        manifest = read_manifest(self.root)
        self._agents = compile_folder(self.root, manifest)
        _check_providers(manifest.models, self._providers)
        self._entries: Mapping[str, ModelEntry] = MappingProxyType(manifest.models)
        self._models: Mapping[str, RunModel] = MappingProxyType(
            {
                name: ModelChoice(entry.provider, entry.model, entry.effort)
                for name, entry in manifest.models.items()
            }
        )
        self._workers = Workers(self._agents, manifest.workers, worker_result_bytes)
        _check_processes(self._workers, max_worker_processes, "raise Engine(max_worker_processes=)")
        self._closed = False
        logger.info("engine: agents loaded [%s]", ", ".join(self._agents))

    async def close(self) -> None:
        self._closed = True
        await self._workers.close()

    def _open(self) -> None:
        if self._closed:
            raise InputError("engine: this engine was closed; build a new Engine")

    async def __aenter__(self) -> Engine:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def models(self) -> Mapping[str, ModelEntry]:
        return self._entries

    @property
    def agents(self) -> dict[str, AgentInfo]:
        return _infos(self._agents)

    def _run_models(self, models: Mapping[str, RunModel] | None) -> Mapping[str, RunModel]:
        if models is None:
            return self._models
        given = _models(models)
        _check_names(self._agents, given)
        return {
            **self._models,
            **{name: self._merged(name, value) for name, value in given.items()},
        }

    def _merged(self, name: str, value: RunModel) -> RunModel:
        if not isinstance(value, ModelChoice):
            return value
        entry = self._entries[name]
        merged = ModelChoice(
            provider=entry.provider if value.provider is None else value.provider,
            model=entry.model if value.model is None else value.model,
            effort=entry.effort if value.effort is None else value.effort,
        )
        if merged.provider not in self._providers:
            _not_held(name, merged.provider, self._providers)
        return merged

    def _built(self, held: RunModel | None) -> Any:
        if isinstance(held, ModelChoice):
            return self._providers[cast(str, held.provider)](held.model, held.effort)
        return held

    def _adapter(
        self,
        spec: AgentSpec,
        models: Mapping[str, RunModel],
    ) -> GenerationModel | DecisionModel | VoiceModel:
        model = self._built(models[spec.model])
        _of_its_kind(spec, model)
        return cast(GenerationModel | DecisionModel | VoiceModel, model)

    def _refuse_voice(self, agent: str) -> None:
        held = self._agents.get(agent)
        if held is not None and held.kind == "voice":
            raise InputError(f"engine: agent {agent!r} is a voice agent; engine.talk runs it")

    def _spec(self, agent: str, mode: str | None = None) -> AgentSpec:
        spec = self._agents.get(agent)
        if spec is None:
            raise NotFoundError(f"engine: agent {agent!r} was not found")
        modes = ", ".join(sorted(spec.modes))
        if mode is not None:
            if mode not in spec.modes:
                have = f"its modes are {modes}" if modes else "it has no modes"
                raise NotFoundError(f"engine: agent {agent!r} has no mode {mode!r}; {have}")
            return spec.modes[mode]
        if spec.needs_mode:
            raise InputError(f"engine: agent {agent!r} needs a mode; its modes are {modes}")
        return spec

    async def run(
        self,
        agent: str,
        scope: str,
        input: State,
        *,
        mode: str | None = None,
        skill: str | None = None,
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
        models: Mapping[str, GenerationModel | DecisionModel | ModelChoice] | None = None,
    ) -> Generation | Decision:
        return await self._start(
            agent,
            scope,
            input,
            mode=mode,
            skill=skill,
            history=history,
            start=start,
            context=context,
            expect=expect,
            on_event=on_event,
            tools=tools,
            workspace=workspace,
            record=record,
            stop=stop,
            inbox=inbox,
            tags=tags,
            approve=approve,
            models=models,
            slash=True,
        )

    async def _start(
        self,
        agent: str,
        scope: str,
        input: State,
        *,
        mode: str | None = None,
        questions: Mapping[str, QuestionLike] | None = None,
        skill: str | None = None,
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
        models: Mapping[str, RunModel] | None = None,
        slash: bool = False,
    ) -> Generation | Decision:
        self._open()
        if not scope:
            raise InputError("engine: scope is required")
        if workspace is not None and workspace.scope != scope:
            raise InputError(
                f"engine: workspace belongs to scope {workspace.scope!r}, not {scope!r}"
            )
        unfit = _unfit(input)
        if unfit is not None:
            raise InputError(
                f"engine: input must be a string, a JSON object or a list; got {unfit}"
            )
        input = _given(input)
        self._refuse_voice(agent)
        spec = self._spec(agent, mode)
        asked = _asked(spec, questions)
        slashed = _slashed(spec, input) if slash else None
        if slashed is not None:
            if skill is not None:
                raise InputError(
                    f"engine: a run invokes one skill, by skill= or by an input that starts with /<name>; got skill={skill!r} and /{slashed[0]}"
                )
            skill, input = slashed
        invoked = _skilled(spec, skill)
        tagged = _tags(tags)
        merged = self._run_models(models)
        held = history if history is None or isinstance(history, HistoryLike) else list(history)
        given = await _window(held, start)
        _paired(given, start)
        options = _Options(
            history=given,
            context=context,
            expect=expect,
            on_event=on_event,
            tools=tools,
            workspace=workspace,
            models=merged,
            stop=stop,
            inbox=inbox,
            skill=invoked,
            tags=tagged,
            approve=approve,
            questions=asked,
            start=start,
            source=held,
        )
        recording = RECORDING.set(record)
        try:
            run = await self._run(spec, scope, input, options, None)
        except Exception:
            if record and options.failed:
                with taking(stop):
                    await self._record(options.failed[-1])
            raise
        finally:
            RECORDING.reset(recording)
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
            logger.warning("engine: refinery.record failed for run %s", run.run_id, exc_info=True)

    def tool(
        self,
        agent: str,
        *,
        scope: str,
        name: str | None = None,
        context: str | None = None,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        tags: Mapping[str, str] | None = None,
        approve: Approve | None = None,
        on_event: EventHandler | None = None,
        stop: Callable[[], Stop | None] | None = None,
        models: Mapping[str, GenerationModel | DecisionModel | ModelChoice] | None = None,
    ) -> Tools:
        if not scope:
            raise InputError("engine: scope is required")
        if workspace is not None and workspace.scope != scope:
            raise InputError(
                f"engine: workspace belongs to scope {workspace.scope!r}, not {scope!r}"
            )
        self._refuse_voice(agent)
        spec = self._spec(agent)
        if not spec.description.strip():
            raise ConfigurationError(
                f"engine: agent {agent!r} has no description; write description: on its entry in rfnry.yaml, or in its instructions file's frontmatter, to use it as a tool"
            )
        tagged = _tags(tags)
        merged = self._run_models(models)

        async def body(args: dict[str, Any]) -> Result:
            watching, held = hooks(on_event, stop)
            run = await self._start(
                agent,
                scope,
                args["input"],
                questions=args.get("questions"),
                context=context,
                tools=tools,
                workspace=workspace,
                tags=tagged,
                approve=approve,
                on_event=watching,
                stop=held,
                record=RECORDING.get(),
                models=merged,
            )
            return run_result(returned(run))

        schema = {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "What to hand the agent"},
                **run_arguments([spec]),
            },
            "required": ["input"],
        }
        return Tools(
            [
                Tool(
                    name or spec.name,
                    spec.description,
                    body=body,
                    args=AgentArgs,
                    input_schema=schema,
                )
            ]
        )

    async def talk(
        self,
        agent: str,
        scope: str,
        session: Callable[[VoiceStart], Awaitable[Spoken]],
        *,
        mode: str | None = None,
        context: str | None = None,
        tools: ToolsLike | None = None,
        models: Mapping[str, VoiceModel | ModelChoice] | None = None,
        tags: Mapping[str, str] | None = None,
        approve: Approve | None = None,
        on_event: EventHandler | None = None,
        record: bool = True,
    ) -> Voice:
        self._open()
        if not scope:
            raise InputError("engine: scope is required")
        held = self._agents.get(agent)
        if held is not None and held.kind != "voice":
            raise InputError(
                f"engine: agent {agent!r} is a {held.kind} agent; engine.run runs it, and engine.talk only a voice agent"
            )
        spec = self._spec(agent, mode)
        tagged = _tags(tags)
        model_ = cast(VoiceModel, self._adapter(spec, self._run_models(models)))
        surface = EngineTools(self, agent=agent, mode=mode, scope=scope, tools=tools, tags=tagged)
        specs = [ToolSpec.model_validate(_copied(d)) for d in surface.definitions()]
        fingerprint = Fingerprint(
            agent=spec.fingerprint, tools=tools_fingerprint(specs), questions=None
        )
        extra = (context or "").strip()
        instructions = f"{spec.body}\n\n<<<CONTEXT>>>\n{extra}" if extra else spec.body
        run_id = uuid4().hex
        started = time.monotonic()
        at = _now()
        tool_calls = _ToolCalls(
            spec=spec,
            run_id=run_id,
            scope=scope,
            tags=tagged,
            surface=surface,
            deadline=smallest(spec.tool_timeout, self.tool_timeout),
            own_deadline=surface._in_workers,
            tool_slots=_slots(spec.tool_parallel),
            approve=approve,
            on_event=on_event,
        )

        def voice(spoken: Spoken | None = None, error: Exception | None = None) -> Voice:
            return Voice(
                run_id=run_id,
                agent=spec.name,
                scope=scope,
                tags=dict(tagged),
                mode=spec.mode,
                context=context,
                at=at,
                duration_ms=int((time.monotonic() - started) * 1000),
                model=(spoken.model if spoken is not None else None) or _declared(model_),
                fingerprint=fingerprint,
                instructions=instructions,
                tools=specs,
                steps=[step for _, step in tool_calls.ran],
                transcript=[]
                if spoken is None
                else [line.model_copy() for line in spoken.transcript],
                usage=None if spoken is None else spoken.usage,
                error=None if error is None else _failure(error),
            )

        start = VoiceStart(
            run_id=run_id,
            model=model_,
            instructions=instructions,
            tools=_SpokenTools(tool_calls, specs),
            mode=spec.mode,
        )
        recording = RECORDING.set(record)
        try:
            try:
                spoken = _spoken(agent, await session(start))
            finally:
                tool_calls.over = True
                await cancel(tuple(tool_calls.speaking))
        except Exception as exc:
            if record:
                await self._record(voice(error=exc))
            raise
        finally:
            RECORDING.reset(recording)
        run = voice(spoken)
        if record:
            await self._record(run)
        return run

    async def _run(
        self,
        spec: AgentSpec,
        scope: str,
        input: State,
        options: _Options,
        parent_run_id: str | None,
        routed: bool = False,
    ) -> Generation | Decision:
        self._open()
        reach = folder_reach(spec.max_depth)
        async with enter(spec.name, _lowered(reach, self.max_depth), reach, routed=routed):
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
        caller = _caller(scope)
        model_ = cast(DecisionModel, self._adapter(spec, options.models))
        folder = {*spec.questions, *spec.scripted}
        clash = [qid for qid in options.questions if qid in folder]
        if clash:
            raise InputError(
                f"engine: decision agent {spec.name!r} already asks {', '.join(map(repr, clash))} from its folder"
            )
        if not folder and not options.questions:
            hint = "give it questions: on its entry, or ask it through Delegate"
            raise InputError(
                f"engine: decision agent {spec.name!r} has no questions to ask; {hint}"
            )
        own = {qid: _fresh(question) for qid, question in spec.questions.items()}
        questions = {**own, **options.questions}
        state = _state(spec, input, getattr(model_, "media_types", None))
        run_id = uuid4().hex
        started = time.monotonic()
        at = _now()
        decided: Decided | None = None
        answers: dict[str, Answer] = {}
        routed: Generation | Decision | None = None
        routing: list[Generation | Decision] = []

        def decision(error: Exception | None = None) -> Decision:
            return Decision(
                run_id=run_id,
                parent_run_id=parent_run_id,
                call=caller,
                agent=spec.name,
                scope=scope,
                tags=dict(options.tags),
                mode=spec.mode,
                at=at,
                duration_ms=int((time.monotonic() - started) * 1000),
                model=(decided.model if decided is not None else None) or _declared(model_),
                settings=decided.settings
                if decided is not None and decided.settings is not None
                else _settings(model_),
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
            if spec.scripted:
                filled = await self._filled(spec, scope, run_id, state, options.tags)
                asked = dict(sorted({**own, **filled}.items()))
                questions = {**asked, **options.questions}
            decided = as_decided(
                await model_.decide(
                    _given(state), {qid: _fresh(question) for qid, question in questions.items()}
                )
            )
            answers = {
                qid: _fresh(answer) for qid, answer in settle(questions, decided.answers).items()
            }
            route = spec.route
            if route is not None:
                answer = answers[route.by]
                target = route.to.get(answer.choice) if isinstance(answer, ChoiceAnswer) else None
                floor = route.min_confidence
                if target is not None and (floor is None or answer.confidence >= floor):
                    forwarded = replace(options, questions={}, failed=routing)
                    routed = await self._run(
                        self._agents[target],
                        scope,
                        _given(input),
                        forwarded,
                        run_id,
                        routed=True,
                    )
        except Exception as exc:
            options.failed.append(decision(exc))
            raise
        return decision()

    async def _filled(
        self, spec: AgentSpec, scope: str, run_id: str, state: State, tags: Mapping[str, str]
    ) -> dict[str, Question]:
        pool = self._workers.pool(spec)
        given = STATE.dump_python(state, mode="json")

        async def fill(qid: str, front: dict[str, JsonValue], script: Script) -> Question:
            call: dict[str, JsonValue] = {
                "scope": scope,
                "run_id": run_id,
                "agent": spec.name,
                "question": qid,
                "tags": dict(tags),
            }
            named = f"{script.path.name} of decision agent {spec.name!r}"
            try:
                held = await pool.call(script.key, given, call, self.tool_timeout)
            except WorkerError as exc:
                raise WorkerError(
                    f"engine: {named} could not fill question {qid!r}: {exc}"
                ) from exc
            such = 'such as {"options": {...}}'
            if not isinstance(held, dict):
                kind = type(held).__name__
                shown = "None" if held is None else f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"
                raise ModelError(
                    f"engine: {named} returned {shown}, not a dict; it returns the fields it fills in {qid}.json, {such}"
                )
            taken = sorted(key for key in held if key in front)
            if taken:
                raise ModelError(
                    f"engine: {named} returned {taken[0]!r}, which question {qid!r} sets in {qid}.json; a script returns only what its question's JSON leaves out, {such}"
                )
            other = foreign(front["type"], held)
            if other is not None:
                raise ModelError(
                    f"engine: {named} returned {other!r}, which question {qid!r}, a {front['type']}, does not take; {TAKEN}"
                )
            try:
                return as_question({**front, **held})
            except ValidationError as exc:
                raise ModelError(
                    f"engine: question {qid!r} of decision agent {spec.name!r} does not fit once {script.path.name} fills it: {plainly(exc, 'the question')}"
                ) from exc

        try:
            async with asyncio.TaskGroup() as group:
                filling = {
                    qid: group.create_task(fill(qid, front, script))
                    for qid, (front, script) in spec.scripted.items()
                }
        except ExceptionGroup as failed:
            raise failed.exceptions[0] from None
        return {qid: task.result() for qid, task in filling.items()}

    async def _generation(
        self,
        spec: AgentSpec,
        scope: str,
        input: State,
        options: _Options,
        parent_run_id: str | None,
    ) -> Generation:
        caller = _caller(scope)
        model_ = cast(GenerationModel, self._adapter(spec, options.models))
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
            mode=spec.mode,
            scope=scope,
            tools=options.tools,
            workspace=options.workspace,
            tags=options.tags,
            approve=options.approve,
            history=options.source,
            start=options.start,
            invoked=options.skill,
            models={**options.models, spec.model: model_},
        )
        room = options.workspace if spec.workspace else None
        barred = WORKSPACE_NAMES | {DELEGATE_TOOL, OUTPUT_TOOL}
        reachable = surface.without(*barred)
        bridging = room is not None and room._bridging and bool(reachable)
        own_deadline = surface._in_workers | (WORKSPACE_TOOLS if room is not None else frozenset())
        variables = {"RFNRY_TREE": room.tree} if room is not None and room.tree is not None else {}
        specs = [ToolSpec.model_validate(_copied(d)) for d in surface.definitions()]
        if room is not None:
            told = room._told(spec.included_folders, bridging)
            specs = [
                s.model_copy(update={"description": s.description + told})
                if s.name == "Shell"
                else s
                for s in specs
            ]
        if expect is not None:
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
        system = build_bundle(
            spec,
            tools=specs,
            agents=targets,
            context=options.context,
        )
        preload = [Message(role="user", content=list(spec.preload))] if spec.preload else []
        chained = chain.Chain(chain.opening(_declared(model_), system, specs, spec.preload_digest))
        history = await chained.settle(options.history)
        opening = _content(input)
        if options.skill is not None:
            opening = _invoked(spec.skills[options.skill], opening)
        messages = [*history, Message(role="user", content=opening)]
        sent = list(messages)
        steps: list[Step] = []
        children: list[Generation | Decision] = []
        delegated: dict[str, Generation | Decision] = {}
        tool_slots = _slots(spec.tool_parallel)
        shell_slots = _slots(None) if bridging else tool_slots
        delegate_slots = _slots(spec.delegate_parallel)
        asked: list[ToolCall] = []
        usage = Usage()
        calls = 0
        model = _declared(model_)
        settings = _settings(model_)
        streaming = on_event is not None and hasattr(model_, "generate_stream")
        streamed: list[str] = []
        said = ""
        ended: StopReason | None = None
        most_calls = _lowered(spec.max_model_calls, self.max_model_calls)

        async def show(call: ToolCall, here: Foreground, stream: Pipe, text: str) -> None:
            if here.failed is not None:
                return
            try:
                await emit(
                    CommandOutput(
                        run_id=run_id, agent=spec.name, call_id=call.id, stream=stream, text=text
                    )
                )
            except Exception as exc:
                here.failed = exc

        async def delta(text: str, *, reasoning: bool = False) -> None:
            if reasoning:
                await emit(ReasoningDelta(run_id=run_id, agent=spec.name, turn=turn, text=text))
                return
            streamed.append(text)
            await emit(TextDelta(run_id=run_id, agent=spec.name, turn=turn, text=text))

        async def delegate(call: ToolCall) -> tuple[Content, bool, str | None, bool]:
            if requested(stop) == "now":
                return STOPPED, True, None, True
            failed: list[Generation | Decision] = []
            try:
                args = permit(spec, call.args)
                target = self._agents[args.agent]
                forwarded = replace(
                    options,
                    questions=_asked(target, args.questions),
                    history=[],
                    expect=None,
                    inbox=None,
                    skill=None,
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

        async def bridged(
            shell: ToolCall,
            here: Foreground,
            numbers: Iterator[int],
            name: str,
            args: dict[str, Any],
        ) -> dict[str, Any]:
            if name not in reachable:
                return {"error": _failure(InputError(f"tools: unknown tool {name!r}"))}
            call = ToolCall(id=f"{shell.id}.{next(numbers)}", name=name, args=args)
            began = time.monotonic()
            calling = asyncio.ensure_future(call_tool(call, shell.id))
            while not calling.done():
                try:
                    await asyncio.wait([calling])
                except asyncio.CancelledError:
                    own = foregrounds.get(id(call))
                    if runner is not None and runner.cancelling() > calm:
                        calling.cancel()
                    elif requested(stop) == "now" and own is not None and own.live > 0:
                        own.pressed = True
                    elif not any(done is call for done, _ in ran):
                        calling.cancel()
            cut_step = stopped(call, began, shell.id) if calling.cancelled() else None
            foregrounds.pop(id(call), None)
            if cut_step is not None:
                steps.append(cut_step)
                try:
                    await hand_over(call, cut_step)
                except Exception as exc:
                    here.failed = here.failed or exc
                return {"error": STOPPED}
            failed = calling.exception()
            if isinstance(failed, Exception):
                steps.extend(built for done, built in ran if done is call)
                here.failed = here.failed or failed
                return {"error": _failure(failed)}
            step = calling.result()
            steps.append(step)
            shown = (
                step.result
                if isinstance(step.result, str)
                else [part.model_dump(mode="json") for part in step.result]
            )
            return {"error" if step.is_error else "result": shown}

        def lend(call: ToolCall, here: Foreground) -> None:
            if call.name != "Shell":
                return
            numbers = count(1)
            here.lent = Lent(
                env=variables,
                said=None
                if on_event is None
                else lambda stream, text: show(call, here, stream, text),
                call=(lambda name, args: bridged(call, here, numbers, name, args))
                if bridging
                else None,
            )

        tool_calls = _ToolCalls(
            spec=spec,
            run_id=run_id,
            scope=scope,
            tags=options.tags,
            surface=surface,
            deadline=smallest(spec.tool_timeout, self.tool_timeout),
            own_deadline=own_deadline,
            tool_slots=tool_slots,
            approve=options.approve,
            on_event=on_event,
            stop=stop,
            shell_slots=shell_slots,
            delegate_slots=delegate_slots,
            lend=lend if room is not None else None,
            delegate=delegate if spec.agents else None,
        )
        emit, hand_over, stopped, cut, call_tool = (
            tool_calls.emit,
            tool_calls.hand_over,
            tool_calls.stopped,
            tool_calls.cut,
            tool_calls.run,
        )
        ran, approving, foregrounds, runner, calm = (
            tool_calls.ran,
            tool_calls.approving,
            tool_calls.foregrounds,
            tool_calls.runner,
            tool_calls.calm,
        )

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

        async def answer_all(
            batch: list[ToolCall], tasks: Sequence[asyncio.Task[Step] | None], began: float
        ) -> None:
            raised: list[BaseException] = []
            held: list[Exception] = []
            for call, task in zip(batch, tasks, strict=True):
                try:
                    if task is None or task.cancelled():
                        await cut(call, began)
                    elif (failed := task.exception()) is not None:
                        raised.append(failed)
                except Exception as exc:
                    raised.append(exc)
                built = next((step for done, step in ran if done is call), None)
                if built is not None:
                    answer(call, built)
                here = foregrounds.pop(id(call), None)
                if here is not None and here.failed is not None and here.failed not in raised:
                    held.append(here.failed)
            failures = [*raised, *held]
            for error in failures[1:]:
                logger.error(
                    "engine: a further call of the same batch failed: %s", error, exc_info=error
                )
            if failures:
                raise failures[0]

        async def run_batch(batch: list[ToolCall]) -> bool:
            began = time.monotonic()
            if requested(stop) == "now":
                await answer_all(batch, [None] * len(batch), began)
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

            cut_short = await wait(tasks, stop, finishing)
            await answer_all(batch, tasks, began)
            return cut_short

        def drain() -> list[Message]:
            taken: list[Message] = []
            while inbox is not None and not inbox.empty():
                taken.append(Message(role="user", content=_own(_content(inbox.get_nowait()))))
                inbox.task_done()
            messages.extend(taken)
            return taken

        async def ask() -> Reply:
            sending = [_handed(message, sent=True) for message in (*preload, *messages)]
            offered = [_fresh(tool) for tool in specs]
            if streaming:
                return as_reply(
                    await cast(StreamingGenerationModel, model_).generate_stream(
                        system, sending, offered, delta
                    )
                )
            return as_reply(await model_.generate(system, sending, offered))

        def finish(
            reply_text: str, stopped: Stopped | None = None, error: Exception | None = None
        ) -> Generation:
            return Generation(
                run_id=run_id,
                parent_run_id=parent_run_id,
                call=caller,
                agent=spec.name,
                scope=scope,
                tags=dict(options.tags),
                mode=spec.mode,
                skill=options.skill,
                context=options.context,
                at=at,
                duration_ms=int((time.monotonic() - started) * 1000),
                model=model,
                settings=settings,
                fingerprint=fingerprint,
                input=input,
                system=system,
                preload=[part.model_copy() for part in spec.preload],
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
            await emit(RunStarted(run_id=run_id, agent=spec.name, call=CALL.get()))
            for turn in range(most_calls):
                mode = requested(stop)
                if mode is not None:
                    return finish(said, mode)
                taken = drain() if turn else []
                if taken:
                    await emit(
                        MessagesTaken(
                            run_id=run_id,
                            agent=spec.name,
                            turn=turn,
                            messages=[_handed(message) for message in taken],
                        )
                    )
                    mode = requested(stop)
                    if mode is not None:
                        return finish(said, mode)
                streamed.clear()
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
                settings = reply.settings if reply.settings is not None else _settings(model_)
                prefix = await chained.stamp(messages) if reply.reasoning is not None else None
                assistant = _assistant(reply, prefix, _declared(model_), _settings(model_))
                messages.append(assistant)
                asked = list(assistant.tool_calls)
                ran.clear()
                if not streaming and reply.text:
                    await delta(reply.text)
                if reply.stop_reason != "tool_use" or not asked:
                    for call in asked:
                        not_run(call, f"the reply ended with {reply.stop_reason}")
                    return finish(reply.text, "now" if requested(stop) == "now" else None)
                rest = list(asked)
                while rest:
                    at_output = next(
                        (
                            n
                            for n, c in enumerate(rest)
                            if expect is not None and c.name == OUTPUT_TOOL
                        ),
                        len(rest),
                    )
                    pressed = await run_batch(rest[:at_output])
                    rest = rest[at_output:]
                    if not rest or expect is None:
                        break
                    if pressed:
                        await answer_all(rest, [None] * len(rest), time.monotonic())
                        return finish(said, "now")
                    output, rest = rest[0], rest[1:]
                    try:
                        parsed = expect.model_validate(_copied(output.args))
                    except ValidationError as exc:
                        errors = f"invalid; fix and call {OUTPUT_TOOL} again: {exc.errors(include_url=False)}"
                        messages.append(
                            Message(
                                role="tool", tool_call_id=output.id, content=errors, is_error=True
                            )
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
                        continue
                    messages.append(
                        Message(role="tool", tool_call_id=output.id, content="accepted")
                    )
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
                    for call in rest:
                        not_run(call, f"{OUTPUT_TOOL} ended the run")
                    return finish(parsed.model_dump_json())
            mode = requested(stop)
            if mode is not None:
                return finish(said, mode)
            raise RunawayError(f"engine: run exceeded {most_calls} model calls")
        except Exception as exc:
            for call in asked:
                built = next((step for done, step in ran if done is call), None)
                if built is not None and not any(built is kept for kept in steps):
                    answer(call, built)
            children.extend(sorted(delegated.values(), key=lambda run: (run.at, run.run_id)))
            options.failed.append(finish("", error=exc))
            raise
