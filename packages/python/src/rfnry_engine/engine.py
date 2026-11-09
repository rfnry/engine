import asyncio
import json
import logging
import math
import posixpath
import time
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from fnmatch import fnmatch
from itertools import count, permutations
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, NoReturn, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError

from rfnry_engine import chain
from rfnry_engine.bundle import build_bundle
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
    Step,
    TextDelta,
    ToolCallStarted,
    ToolResultReady,
)
from rfnry_engine.stop import Stop, Stopped, requested, wait
from rfnry_engine.tools.declared import chosen
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
from rfnry_engine.tools.workspace import WORKSPACE_NAMES, WORKSPACE_TOOLS, Workspace
from rfnry_engine.tree import (
    OUTPUT_TOOL,
    TAKEN,
    AgentSpec,
    Kind,
    as_question,
    foreign,
    load_tree,
    validate_tree,
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
    stop: Stop | None = None
    inbox: asyncio.Queue[Any] | None = None
    skill: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    approve: Approve | None = None
    questions: Mapping[str, Question] = field(default_factory=dict)
    start: int = 0
    source: Sequence[object] | HistoryLike | None = None
    failed: list[Run] = field(default_factory=list)


class AgentInfo(BaseModel):
    name: str
    path: str
    kind: Kind
    model: str
    description: str
    agents: list[str]
    reads: list[str]
    tools: list[str]
    skills: dict[str, str]
    public_skills: dict[str, str]


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


def _invoked(path: str, body: str, said: Content) -> list[Part]:
    rest: list[Part] = said if isinstance(said, list) else [Text(text=said)] if said else []
    return [Text(text=f"<<<SKILL {path}>>>\n{body}"), *rest]


def _skilled(spec: AgentSpec, skill: str | None) -> str | None:
    if skill is None:
        return None
    if spec.kind == "decision":
        raise InputError(
            f"engine: decision agent {spec.name!r} takes no skill; skills are for generation agents"
        )
    if skill not in spec.skills:
        have = (
            f"its skills are {', '.join(spec.skills)}"
            if spec.skills
            else "it has no skills; a skill is skills/<name>.md or skills/<name>/SKILL.md in its folder or a folder above it, and skills: keeps the ones it lists"
        )
        raise NotFoundError(f"engine: agent {spec.name!r} has no skill {skill!r}; {have}")
    return skill


def _command(spec: AgentSpec, text: str) -> tuple[str, str] | None:
    if not text.startswith("/"):
        return None
    word, *rest = text.split(maxsplit=1)
    if word[1:] not in spec.public_skills:
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


def _not_given(spec: AgentSpec) -> NoReturn:
    raise ConfigurationError(
        f"engine: agent {spec.name!r} runs on model {spec.model!r}, which this engine was not given"
    )


def _kind_clash(spec: AgentSpec, verb: str) -> NoReturn:
    raise ConfigurationError(
        f"engine: agent {spec.name!r} is a {spec.kind} agent, and model {spec.model!r} has no {verb}()"
    )


def _models(
    models: Mapping[str, GenerationModel | DecisionModel],
) -> dict[str, GenerationModel | DecisionModel]:
    if not isinstance(models, Mapping):
        raise ConfigurationError("engine: models must be a mapping of model name to adapter")
    for name, model in models.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise ConfigurationError(
                f"engine: model name {name!r} must be non-blank text without padding"
            )
        if not isinstance(model, GenerationModel | DecisionModel):
            raise ConfigurationError(
                f"engine: model {name!r} has neither generate() nor decide(); pass an adapter that has one"
            )
    return dict(models)


def _check_models(
    agents: Mapping[str, AgentSpec], models: Mapping[str, GenerationModel | DecisionModel]
) -> None:
    used = sorted({spec.model for spec in agents.values()})
    for name in sorted(models):
        if name not in used:
            raise ConfigurationError(
                f"engine: no agent runs on model {name!r}; the tree's agents run on {', '.join(used)}"
            )
    for spec in agents.values():
        model = models.get(spec.model)
        if model is None:
            continue
        if spec.kind == "generation" and not isinstance(model, GenerationModel):
            _kind_clash(spec, "generate")
        if spec.kind == "decision" and not isinstance(model, DecisionModel):
            _kind_clash(spec, "decide")


def _groups(agents: Mapping[str, AgentSpec], workers: Mapping[str, int] | None) -> dict[str, int]:
    if workers is None:
        return {}
    if not isinstance(workers, Mapping):
        raise ConfigurationError(
            "engine: workers must map folders of the tree to their number of processes"
        )
    folders = {
        "/".join(parts[:n])
        for spec in agents.values()
        if spec.path != "."
        for parts in [spec.path.split("/")]
        for n in range(1, len(parts) + 1)
    }
    for name, processes in workers.items():
        if name not in folders:
            raise ConfigurationError(
                f"engine: workers names {name!r}, which holds no agent of the tree; name a folder that holds agents by its path from the root"
            )
        if not isinstance(processes, int) or isinstance(processes, bool) or processes < 1:
            raise ConfigurationError(
                f"engine: workers gives {name!r} {processes!r} processes; give each folder 1 or more"
            )
    for one, other in permutations(workers, 2):
        if one.startswith(f"{other}/"):
            raise ConfigurationError(
                f"engine: workers names {one!r} inside {other!r}, which it names too; keep one"
            )
    return dict(workers)


class Engine:
    def __init__(
        self,
        root: Path | str,
        *,
        models: Mapping[str, GenerationModel | DecisionModel],
        workers: Mapping[str, int] | None = None,
        refinery: Recorder | None = None,
        max_steps: int = 200,
        max_depth: int = 5,
        tool_timeout: float | None = 30.0,
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
        if max_steps <= 0:
            raise ConfigurationError("engine: max_steps must be positive")
        if max_depth <= 0:
            raise ConfigurationError("engine: max_depth must be positive")
        if history_walk <= 0:
            raise ConfigurationError("engine: history_walk must be positive")
        if history_chars <= 0:
            raise ConfigurationError("engine: history_chars must be positive")
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ConfigurationError(f"engine: root is not a directory: {root}")
        self._models = MappingProxyType(_models(models))
        self.refinery = refinery
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.tool_timeout = tool_timeout
        self.worker_result_bytes = worker_result_bytes
        self.history_walk = history_walk
        self.history_chars = history_chars
        self._agents = load_tree(
            self.root,
            {name: getattr(model, "media_types", None) for name, model in self._models.items()},
        )
        validate_tree(self._agents)
        _check_models(self._agents, self._models)
        self._workers = Workers(self._agents, _groups(self._agents, workers), worker_result_bytes)
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
    def models(self) -> Mapping[str, GenerationModel | DecisionModel]:
        return self._models

    @property
    def agents(self) -> dict[str, AgentInfo]:
        return {
            name: AgentInfo(
                name=name,
                path=spec.path,
                kind=spec.kind,
                model=spec.model,
                description=spec.description,
                agents=spec.agents,
                reads=list(spec.reads),
                tools=list(chosen(spec)),
                skills={called: skill.description for called, skill in spec.skills.items()},
                public_skills={
                    called: spec.skills[called].description for called in spec.public_skills
                },
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
    ) -> Run:
        return await self._start(
            agent,
            scope,
            input,
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
            slash=True,
        )

    async def _start(
        self,
        agent: str,
        scope: str,
        input: State,
        *,
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
        slash: bool = False,
    ) -> Run:
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
        spec = self._spec(agent)
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
    ) -> Tools:
        if not scope:
            raise InputError("engine: scope is required")
        if workspace is not None and workspace.scope != scope:
            raise InputError(
                f"engine: workspace belongs to scope {workspace.scope!r}, not {scope!r}"
            )
        spec = self._spec(agent)
        if not spec.description.strip():
            raise ConfigurationError(
                f"engine: agent {agent!r} has no description: in its AGENT.md; "
                "add one to use it as a tool"
            )
        tagged = _tags(tags)

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

    async def _run(
        self,
        spec: AgentSpec,
        scope: str,
        input: State,
        options: _Options,
        parent_run_id: str | None,
        routed: bool = False,
    ) -> Run:
        self._open()
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
        caller = _caller(scope)
        model_ = self._models.get(spec.model)
        if model_ is None:
            _not_given(spec)
        model_ = cast(DecisionModel, model_)
        self._check_routes(spec, set())
        folder = {*spec.questions, *spec.scripted}
        clash = [qid for qid in options.questions if qid in folder]
        if clash:
            raise InputError(
                f"engine: decision agent {spec.name!r} already asks {', '.join(map(repr, clash))} from its folder"
            )
        if not folder and not options.questions:
            hint = "give it questions/, or ask it through Delegate"
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
        routed: Run | None = None
        routing: list[Run] = []

        def decision(error: Exception | None = None) -> Decision:
            return Decision(
                run_id=run_id,
                parent_run_id=parent_run_id,
                call=caller,
                agent=spec.name,
                scope=scope,
                tags=dict(options.tags),
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
        pool = self._workers.pool(spec.name)
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

    def _check_routes(self, spec: AgentSpec, seen: set[str]) -> None:
        seen.add(spec.name)
        for target in spec.route.to.values() if spec.route is not None else []:
            routed = self._agents[target]
            if routed.model not in self._models:
                raise ConfigurationError(
                    f"engine: agent {spec.name!r} routes to agent {target!r}, which runs on model {routed.model!r}, which this engine was not given"
                )
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
        caller = _caller(scope)
        model_ = self._models.get(spec.model)
        if model_ is None:
            _not_given(spec)
        model_ = cast(GenerationModel, model_)
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
        room = options.workspace if spec.workspace else None
        barred = WORKSPACE_NAMES | {DELEGATE_TOOL, OUTPUT_TOOL}
        reachable = surface.without(*barred)
        bridging = room is not None and room._bridging and bool(reachable)
        own_deadline = surface._in_workers | (WORKSPACE_TOOLS if room is not None else frozenset())
        variables = (
            {
                "RFNRY_TREE": room.tree,
                "RFNRY_AGENT": room.tree
                if spec.path == "."
                else posixpath.join(room.tree, spec.path),
            }
            if room is not None and room.tree is not None
            else {}
        )
        specs = [ToolSpec.model_validate(_copied(d)) for d in surface.definitions()]
        if room is not None:
            told = room._told(list(spec.reads), bridging)
            specs = [
                s.model_copy(update={"description": s.description + told})
                if s.name == "Shell"
                else s
                for s in specs
            ]
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
            skill = spec.skills[options.skill]
            opening = _invoked(skill.path, skill.body, opening)
        messages = [*history, Message(role="user", content=opening)]
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
        settings = _settings(model_)
        streaming = on_event is not None and hasattr(model_, "generate_stream")
        streamed: list[str] = []
        said = ""
        ended: StopReason | None = None
        runner = asyncio.current_task()
        calm = 0 if runner is None else runner.cancelling()

        async def emit(event: Any) -> None:
            if on_event is not None:
                await on_event(event)

        def called(call: ToolCall, by: str | None = None) -> CallInfo:
            return CallInfo(
                run_id=run_id,
                agent=spec.name,
                scope=scope,
                tool=call.name,
                call_id=call.id,
                by=by,
                tags=options.tags,
            )

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

        async def hand_over(call: ToolCall, step: Step) -> Step:
            if runner is not None and runner.cancelling() > calm:
                raise asyncio.CancelledError("engine: the run was cancelled")
            ran.append((call, step))
            if on_event is None:
                return step
            token = CALL.set(called(call, step.by))
            try:
                await emit(
                    ToolResultReady(
                        run_id=run_id,
                        agent=spec.name,
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

        async def delta(text: str, *, reasoning: bool = False) -> None:
            if reasoning:
                await emit(ReasoningDelta(run_id=run_id, agent=spec.name, turn=turn, text=text))
                return
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
            foregrounds.pop(id(call), None)
            if calling.cancelled():
                step = stopped(call, began, shell.id)
                steps.append(step)
                try:
                    await hand_over(call, step)
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

        async def call_tool(call: ToolCall, by: str | None = None) -> Step:
            CALL.set(called(call, by))
            if on_event is not None:
                await emit(
                    ToolCallStarted(
                        run_id=run_id,
                        agent=spec.name,
                        call_id=call.id,
                        tool=call.name,
                        args=_copied(call.args),
                        by=by,
                    )
                )
            began = time.monotonic()
            if requested(stop) == "now":
                return await cut(call, began, by)
            child_run_id: str | None = None
            cut_short = False
            content: Content
            approved = None
            if options.approve is not None:
                asking = asyncio.ensure_future(options.approve(spec.name, _call(call)))
                approving.add(id(call))
                try:
                    halted = await wait([asking], stop)
                finally:
                    approving.discard(id(call))
                if halted:
                    return await cut(call, began, by)
                approved = asking.result()
            if approved is not None and not isinstance(approved, bool):
                raise ConfigurationError(
                    f"engine: approve must return True or False; got {approved!r}"
                )
            here = Foreground(stop, on_event, asyncio.current_task())
            if room is not None and call.name == "Shell":
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
            foregrounds[id(call)] = here
            FOREGROUND.set(here)
            try:
                if approved is False:
                    content, is_error = f"NotApproved: {call.name} was not approved", True
                elif call.name == DELEGATE_TOOL and spec.agents:
                    content, is_error, child_run_id, cut_short = await delegate(call)
                else:
                    limit = None if call.name in own_deadline else self.tool_timeout
                    deadline = asyncio.timeout(limit)
                    try:
                        async with deadline:
                            result = await surface.dispatch(call.name, _copied(call.args))
                        content, is_error = _own(_content(result)), False
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
                ran.append((call, step))
                raise here.failed
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

        def stopped(call: ToolCall, began: float, by: str | None = None) -> Step:
            return Step(
                tool=call.name,
                args=call.args,
                result=STOPPED,
                is_error=True,
                at=_now(),
                duration_ms=int((time.monotonic() - began) * 1000),
                stopped=True,
                by=by,
            )

        async def cut(call: ToolCall, began: float, by: str | None = None) -> Step:
            return await hand_over(call, stopped(call, began, by))

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
            for turn in range(self.max_steps):
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
            raise RunawayError(f"engine: run exceeded {self.max_steps} steps")
        except Exception as exc:
            for call in asked:
                built = next((step for done, step in ran if done is call), None)
                if built is not None and not any(built is kept for kept in steps):
                    answer(call, built)
            children.extend(sorted(delegated.values(), key=lambda run: (run.at, run.run_id)))
            options.failed.append(finish("", error=exc))
            raise
