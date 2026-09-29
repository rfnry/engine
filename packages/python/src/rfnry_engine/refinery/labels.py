from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from rfnry_engine.models import Answer
from rfnry_engine.records import Run


class Review(BaseModel):
    stance: Literal["approval", "objection"] | None = None
    comment: str | None = None
    correction: str | dict[str, Any] | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> Review:
        if self.stance is None and not self.comment and self.correction is None:
            raise ValueError("labels: a review needs a stance, a comment or a correction")
        return self


class Outcome(BaseModel):
    status: Literal["succeeded", "failed"]
    detail: str | None = None


class Label(BaseModel):
    label_id: str
    run_id: str
    scope: str
    at: datetime
    source: Literal["human", "world", "judge", "teacher"]
    review: Review | None = None
    outcome: Outcome | None = None
    answers: dict[str, Answer] | None = None


class Labeled(BaseModel):
    judged: int = 0
    taught: int = 0
    errors: dict[str, str] = Field(default_factory=dict)


class Replay(BaseModel):
    recorded: Run
    labels: list[Label]
    candidate: Run | None = None
    error: str | None = None


class Exported(BaseModel):
    keys: list[str]
    rows: dict[str, int]
    watermark: datetime | None


class Conflict(BaseModel):
    run_id: str
    corrections: list[tuple[str, datetime, str]]
