from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from rfnry_engine.errors import InputError, ModelError
from rfnry_engine.models import (
    Answer,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Document,
    Part,
    Question,
    Score,
    ScoreAnswer,
    Text,
)
from rfnry_engine.refinery.storage import RunRow

PART = TypeAdapter[Part](Part)
PART_TYPES = ("text", "image", "audio", "document")


def placeholder(part: Part) -> str:
    if isinstance(part, Text):
        return part.text
    if isinstance(part, Document) and part.name is not None:
        return f"[document: {part.media_type}, {json.dumps(part.name)}]"
    return f"[{part.type}: {part.media_type}]"


def without_media(value: Any) -> Any:
    if isinstance(value, list):
        return [without_media(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") in PART_TYPES:
        try:
            part = PART.validate_python(value)
        except (ValidationError, InputError):
            part = None
        if part is not None and part.model_dump(mode="json") == value:
            return placeholder(part)
    return {key: without_media(item) for key, item in value.items()}


def judge_state(row: RunRow) -> dict[str, Any]:
    body = row.body
    return {
        "input": without_media(body["input"]),
        "reply": body["reply"],
        "steps": [
            {
                "tool": s["tool"],
                "args": without_media(s["args"]),
                "result": without_media(s["result"]),
            }
            for s in body["steps"]
        ],
    }


class TeacherAnswers(BaseModel):
    answers: dict[str, str | float | bool]
    rationale: str


def teacher_prompt(row: RunRow) -> str:
    body = row.body
    lines = [
        "Answer every question about the state below. For a choice give the option key, "
        "for a score give the level index starting at 0, for a check give true or false.",
        "",
        "STATE:",
        json.dumps(without_media(body["state"]), indent=2, ensure_ascii=False),
        "",
    ]
    for question_id, question in body["questions"].items():
        lines.append(f"QUESTION {question_id} ({question['type']}): {question['instructions']}")
        if question["type"] == "choice":
            lines.extend(f"  - {key}: {text}" for key, text in question["options"].items())
        if question["type"] == "score":
            lines.extend(f"  {index}: {level}" for index, level in enumerate(question["levels"]))
    return "\n".join(lines)


def teacher_answers(questions: dict[str, Question], taught: TeacherAnswers) -> dict[str, Answer]:
    answers: dict[str, Answer] = {}
    for question_id, question in questions.items():
        value = taught.answers.get(question_id)
        if value is None:
            raise ModelError(f"judge: teacher gave no answer for {question_id!r}")
        if isinstance(question, Choice):
            key = str(value)
            if key not in question.options:
                raise ModelError(f"judge: teacher chose {key!r}, not an option of {question_id!r}")
            answers[question_id] = ChoiceAnswer(
                choice=key,
                probabilities={k: 1.0 if k == key else 0.0 for k in question.options},
                confidence=1.0,
            )
        elif isinstance(question, Score):
            if isinstance(value, bool | str) or not float(value).is_integer():
                raise ModelError(f"judge: teacher gave {value!r}, not a level of {question_id!r}")
            index = int(value)
            if not 0 <= index < len(question.levels):
                raise ModelError(f"judge: teacher gave level {index} for {question_id!r}")
            answers[question_id] = ScoreAnswer(
                score=float(index),
                probabilities=[1.0 if i == index else 0.0 for i in range(len(question.levels))],
                confidence=1.0,
            )
        else:
            if not isinstance(value, bool):
                raise ModelError(
                    f"judge: teacher gave {value!r}, not true or false for {question_id!r}"
                )
            answers[question_id] = CheckAnswer(probability=1.0 if value else 0.0, confidence=1.0)
    return answers
