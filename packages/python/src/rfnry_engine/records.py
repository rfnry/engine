from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from rfnry_engine.errors import InputError
from rfnry_engine.models import (
    Answer,
    Content,
    Message,
    Part,
    Question,
    State,
    Text,
    ToolSpec,
    Usage,
)
from rfnry_engine.stop import Stopped

PARTS = TypeAdapter(list[Part])


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


class Fingerprint(BaseModel):
    agent: str
    tools: str
    questions: str | None = None


class Generation(BaseModel):
    kind: Literal["generation"] = "generation"
    run_id: str
    parent_run_id: str | None = None
    agent: str
    scope: str
    tags: dict[str, str] = Field(default_factory=dict)
    task: str | None = None
    context: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    fingerprint: Fingerprint | None = None
    input: State
    system: str
    messages: list[Message]
    conversation: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    steps: list[Step]
    calls: int
    reply: str
    usage: Usage
    children: list[Run] = Field(default_factory=list)
    stopped: Stopped | None = None


class Decision(BaseModel):
    kind: Literal["decision"] = "decision"
    run_id: str
    parent_run_id: str | None = None
    agent: str
    scope: str
    tags: dict[str, str] = Field(default_factory=dict)
    task: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    fingerprint: Fingerprint | None = None
    state: State
    questions: dict[str, Question]
    runtime: list[str] = Field(default_factory=list)
    answers: dict[str, Answer]
    usage: Usage
    routed_to: Run | None = None

    @field_validator("state", mode="before")
    @classmethod
    def _media_reads_back_as_parts(cls, value: object) -> object:
        if not isinstance(value, list) or not value:
            return value
        try:
            parts = PARTS.validate_python(value, from_attributes=True)
        except (ValidationError, InputError):
            return value
        return parts if any(not isinstance(part, Text) for part in parts) else value


Run = Generation | Decision

Generation.model_rebuild()
Decision.model_rebuild()
