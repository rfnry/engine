from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_serializer,
    model_validator,
)

from rfnry_engine.errors import InputError, ModelError

State = str | dict[str, Any] | list[Any]


class Text(BaseModel):
    type: Literal["text"] = "text"
    text: str


class _Media(BaseModel):
    model_config = ConfigDict(val_json_bytes="base64")

    type: str
    media_type: str
    data: bytes | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _data_or_url(self) -> Self:
        if (self.data is None) == (self.url is None):
            raise InputError(f"models: {self.type} carries exactly one of data or url")
        return self

    @field_serializer("data", when_used="json-unless-none")
    def _base64(self, data: bytes) -> str:
        return base64.b64encode(data).decode()


class Image(_Media):
    type: Literal["image"] = "image"


class Audio(_Media):
    type: Literal["audio"] = "audio"


class Document(_Media):
    type: Literal["document"] = "document"
    name: str | None = None


Part = Annotated[Text | Image | Audio | Document, Field(discriminator="type")]
Content = str | list[Part]


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: Content = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    reasoning: Any = None
    is_error: bool = False


class Usage(BaseModel):
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_creation: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)

    model_config = {"extra": "forbid", "frozen": True}

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_creation=self.cache_creation + other.cache_creation,
            cache_read=self.cache_read + other.cache_read,
        )


StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


class Reply(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    stop_reason: StopReason = "end_turn"
    model: str | None = None
    reasoning: Any = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None]

    @model_validator(mode="after")
    def _at_least_two(self) -> Choice:
        if len(self.options) < 2:
            raise ValueError("models: a choice needs at least two options")
        return self


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    levels: list[str | dict[str, JsonValue] | list[JsonValue]]

    @model_validator(mode="after")
    def _at_least_two(self) -> Score:
        if len(self.levels) < 2:
            raise ValueError("models: a score needs at least two levels")
        return self


class Check(BaseModel):
    type: Literal["check"] = "check"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    yes: str | dict[str, JsonValue] | list[JsonValue] | None = None
    no: str | dict[str, JsonValue] | list[JsonValue] | None = None


Question = Choice | Score | Check


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = 0.0


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: list[float]
    confidence: float = 0.0


class CheckAnswer(BaseModel):
    type: Literal["check"] = "check"
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = 0.0


Answer = ChoiceAnswer | ScoreAnswer | CheckAnswer


class Decided(BaseModel):
    answers: dict[str, Answer]
    usage: Usage | None = None
    model: str | None = None


def as_decided(value: object) -> Decided:
    try:
        if isinstance(value, Mapping):
            return Decided.model_validate({"answers": value}, from_attributes=True)
        return Decided.model_validate(value, from_attributes=True)
    except ValidationError as exc:
        raise ModelError(
            f"models: the decision model's answers do not fit the contract: {exc.errors(include_url=False)}"
        ) from exc


def as_reply(value: object) -> Reply:
    try:
        return Reply.model_validate(value, from_attributes=True)
    except ValidationError as exc:
        raise ModelError(
            f"models: the generation model's reply does not fit the contract: {exc.errors(include_url=False)}"
        ) from exc


def confidence(probabilities: list[float]) -> float:
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, (n * max(probabilities) - 1) / (n - 1))


def settle(questions: dict[str, Question], answers: dict[str, Answer]) -> dict[str, Answer]:
    settled: dict[str, Answer] = {}
    for qid, question in questions.items():
        answer = answers.get(qid)
        if answer is None:
            raise ModelError(f"models: no answer for question {qid!r}")
        if answer.type != question.type:
            raise ModelError(
                f"models: answer to {qid!r} is {answer.type}, question is {question.type}"
            )
        if isinstance(answer, ChoiceAnswer) and isinstance(question, Choice):
            if set(answer.probabilities) != set(question.options):
                raise ModelError(f"models: probabilities for {qid!r} do not cover the options")
            if answer.choice not in question.options:
                raise ModelError(f"models: choice {answer.choice!r} is not an option of {qid!r}")
            probabilities = list(answer.probabilities.values())
        elif isinstance(answer, ScoreAnswer) and isinstance(question, Score):
            if len(answer.probabilities) != len(question.levels):
                raise ModelError(f"models: probabilities for {qid!r} do not cover the levels")
            if not 0.0 <= answer.score <= len(question.levels) - 1:
                raise ModelError(f"models: score for {qid!r} lies outside its levels")
            probabilities = list(answer.probabilities)
        elif isinstance(answer, CheckAnswer):
            probabilities = [answer.probability, 1.0 - answer.probability]
        else:
            raise ModelError(f"models: answer to {qid!r} does not match its question")
        if not all(0.0 <= p <= 1.0 for p in probabilities):
            raise ModelError(f"models: probabilities for {qid!r} must lie between 0 and 1")
        if abs(sum(probabilities) - 1.0) > 1e-3:
            raise ModelError(f"models: probabilities for {qid!r} do not sum to 1")
        settled[qid] = answer.model_copy(update={"confidence": confidence(probabilities)})
    return settled


class ToolCallLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def args(self) -> dict[str, Any]: ...


class UsageLike(Protocol):
    @property
    def input(self) -> int: ...
    @property
    def output(self) -> int: ...
    @property
    def cache_creation(self) -> int: ...
    @property
    def cache_read(self) -> int: ...


class ReplyLike(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def tool_calls(self) -> Sequence[ToolCallLike]: ...
    @property
    def usage(self) -> UsageLike | None: ...
    @property
    def stop_reason(self) -> StopReason: ...
    @property
    def model(self) -> str | None: ...
    @property
    def reasoning(self) -> Any: ...


class ChoiceAnswerLike(Protocol):
    @property
    def type(self) -> Literal["choice"]: ...
    @property
    def choice(self) -> str: ...
    @property
    def probabilities(self) -> dict[str, float]: ...


class ScoreAnswerLike(Protocol):
    @property
    def type(self) -> Literal["score"]: ...
    @property
    def score(self) -> float: ...
    @property
    def probabilities(self) -> list[float]: ...


class CheckAnswerLike(Protocol):
    @property
    def type(self) -> Literal["check"]: ...
    @property
    def probability(self) -> float: ...


AnswerLike = ChoiceAnswerLike | ScoreAnswerLike | CheckAnswerLike


class ChoiceLike(Protocol):
    @property
    def type(self) -> Literal["choice"]: ...
    @property
    def instructions(self) -> str | dict[str, JsonValue] | list[JsonValue]: ...
    @property
    def options(self) -> Mapping[str, str | dict[str, JsonValue] | list[JsonValue] | None]: ...


class ScoreLike(Protocol):
    @property
    def type(self) -> Literal["score"]: ...
    @property
    def instructions(self) -> str | dict[str, JsonValue] | list[JsonValue]: ...
    @property
    def levels(self) -> Sequence[str | dict[str, JsonValue] | list[JsonValue]]: ...


class CheckLike(Protocol):
    @property
    def type(self) -> Literal["check"]: ...
    @property
    def instructions(self) -> str | dict[str, JsonValue] | list[JsonValue]: ...
    @property
    def yes(self) -> str | dict[str, JsonValue] | list[JsonValue] | None: ...
    @property
    def no(self) -> str | dict[str, JsonValue] | list[JsonValue] | None: ...


QuestionLike = ChoiceLike | ScoreLike | CheckLike


class DecidedLike(Protocol):
    @property
    def answers(self) -> Mapping[str, AnswerLike]: ...
    @property
    def usage(self) -> UsageLike | None: ...
    @property
    def model(self) -> str | None: ...


@runtime_checkable
class GenerationModel(Protocol):
    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ReplyLike: ...


@runtime_checkable
class StreamingGenerationModel(Protocol):
    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], Awaitable[None]],
    ) -> ReplyLike: ...


@runtime_checkable
class DecisionModel(Protocol):
    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> Mapping[str, AnswerLike] | DecidedLike: ...


class TextDelta(BaseModel):
    run_id: str
    agent: str
    text: str


class ToolCallStarted(BaseModel):
    run_id: str
    agent: str
    call_id: str
    tool: str


class ToolResultReady(BaseModel):
    run_id: str
    agent: str
    call_id: str
    tool: str
    is_error: bool


Event = TextDelta | ToolCallStarted | ToolResultReady
EventHandler = Callable[[Event], Awaitable[None]]
