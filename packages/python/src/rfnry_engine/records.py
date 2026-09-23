from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from rfnry_engine.models import Answer, Content, Message, Question, State, Usage


class Step(BaseModel):
    tool: str
    args: dict[str, Any]
    result: Content
    is_error: bool
    at: datetime
    duration_ms: int
    child_run_id: str | None = None


class Generation(BaseModel):
    kind: Literal["generation"] = "generation"
    run_id: str
    parent_run_id: str | None = None
    agent: str
    scope: str
    task: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    input: State
    system: str
    messages: list[Message]
    conversation: list[Message]
    steps: list[Step]
    calls: int
    reply: str
    usage: Usage
    children: list[Run] = Field(default_factory=list)


class Decision(BaseModel):
    kind: Literal["decision"] = "decision"
    run_id: str
    parent_run_id: str | None = None
    agent: str
    scope: str
    at: datetime
    duration_ms: int
    model: str | None
    state: State
    questions: dict[str, Question]
    answers: dict[str, Answer]
    usage: Usage
    routed_to: Run | None = None


Run = Generation | Decision

Generation.model_rebuild()
Decision.model_rebuild()
