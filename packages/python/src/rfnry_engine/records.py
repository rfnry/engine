from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from rfnry_engine.errors import InputError
from rfnry_engine.models import (
    Answer,
    Content,
    Message,
    Part,
    Question,
    State,
    StopReason,
    Text,
    ToolSpec,
    Usage,
)
from rfnry_engine.stop import Stopped

PARTS = TypeAdapter(list[Part])
READ_BACK = {"read_back": True}


def _reading_back(info: ValidationInfo) -> bool:
    return isinstance(info.context, dict) and info.context.get("read_back") is True


def _written(value: object) -> list[Part] | None:
    if not isinstance(value, list):
        return None
    try:
        parts = PARTS.validate_python(value)
    except ValidationError, InputError:
        return None
    return parts if PARTS.dump_python(parts, mode="json") == value else None


class Step(BaseModel):
    tool: str
    args: dict[str, Any]
    result: Content
    is_error: bool
    at: datetime
    duration_ms: int
    child_run_id: str | None = None
    stopped: bool = False
    approved: bool | None = None
    by: str | None = None


class CallInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    agent: str
    scope: str
    tool: str
    call_id: str
    by: str | None = None
    tags: Mapping[str, str] = Field(default_factory=dict, validate_default=True)

    @field_validator("tags", mode="after")
    @classmethod
    def _held(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("tags")
    def _shown(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    def __hash__(self) -> int:
        return hash(
            (self.run_id, self.agent, self.scope, self.tool, self.call_id, self.by)
            + (frozenset(self.tags.items()),)
        )

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self).model_validate, (self.model_dump(),))


class RunStarted(BaseModel):
    run_id: str
    agent: str
    call: CallInfo | None


class TextDelta(BaseModel):
    run_id: str
    agent: str
    turn: int
    text: str


class ReasoningDelta(BaseModel):
    run_id: str
    agent: str
    turn: int
    text: str


class MessagesTaken(BaseModel):
    run_id: str
    agent: str
    turn: int
    messages: list[Message]


class ToolCallStarted(BaseModel):
    run_id: str
    agent: str
    call_id: str
    tool: str
    args: dict[str, Any]
    by: str | None = None


class ToolResultReady(BaseModel):
    run_id: str
    agent: str
    call_id: str
    step: Step
    by: str | None = None


Pipe = Literal["stdout", "stderr"]


class CommandOutput(BaseModel):
    run_id: str
    agent: str
    call_id: str
    stream: Pipe
    text: str


Event = (
    RunStarted
    | TextDelta
    | ReasoningDelta
    | MessagesTaken
    | ToolCallStarted
    | ToolResultReady
    | CommandOutput
)
EventHandler = Callable[[Event], Awaitable[None]]


class Fingerprint(BaseModel):
    agent: str
    tools: str
    questions: str | None


class Generation(BaseModel):
    kind: Literal["generation"] = "generation"
    run_id: str
    parent_run_id: str | None = None
    call: CallInfo | None = None
    agent: str
    scope: str
    tags: dict[str, str] = Field(default_factory=dict)
    mode: str | None = None
    skill: str | None = None
    context: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    settings: dict[str, JsonValue] | None = None
    fingerprint: Fingerprint
    input: State
    system: str
    preload: list[Part] = Field(default_factory=list)
    start: int = 0
    messages: list[Message]
    conversation: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    steps: list[Step]
    calls: int
    reply: str
    usage: Usage
    children: list[Run] = Field(default_factory=list)
    stopped: Stopped | None = None
    stop_reason: StopReason | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _input_reads_back(self, info: ValidationInfo) -> Self:
        if not _reading_back(info) or not isinstance(self.input, list) or not self.messages:
            return self
        sent = self.messages[-1].content
        if self.skill is not None and isinstance(sent, list):
            sent = sent[1:]
        if isinstance(sent, list) and PARTS.dump_python(sent, mode="json") == self.input:
            self.input = list(sent)
        return self

    @property
    def added(self) -> list[Message]:
        return self.conversation[max(len(self.messages) - 1, 0) :]


class Decision(BaseModel):
    kind: Literal["decision"] = "decision"
    run_id: str
    parent_run_id: str | None = None
    call: CallInfo | None = None
    agent: str
    scope: str
    tags: dict[str, str] = Field(default_factory=dict)
    mode: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    settings: dict[str, JsonValue] | None = None
    fingerprint: Fingerprint
    state: State
    questions: dict[str, Question]
    runtime: list[str] = Field(default_factory=list)
    answers: dict[str, Answer]
    usage: Usage
    routed_to: Run | None = None
    error: str | None = None

    @property
    def added(self) -> list[Message]:
        return [] if self.routed_to is None else self.routed_to.added

    @model_validator(mode="after")
    def _state_reads_back(self, info: ValidationInfo) -> Self:
        if not _reading_back(info):
            return self
        parts = _written(self.state)
        if parts is not None and any(not isinstance(part, Text) for part in parts):
            self.state = parts
        return self


Run = Generation | Decision

Generation.model_rebuild()
Decision.model_rebuild()
