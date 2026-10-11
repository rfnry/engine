from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rfnry_engine.records import Fingerprint
from rfnry_engine.refinery.labels import Label
from rfnry_engine.refinery.storage import RunRow

DATASETS = ("generation", "preference", "decision")


@runtime_checkable
class Bucket(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...


def export_key(scope: str, agent: str, dataset: str, stamp: str) -> str:
    return f"{scope}/{agent}/{dataset}/{stamp}.jsonl"


def _fingerprint(body: dict[str, Any]) -> dict[str, Any] | None:
    printed = body.get("fingerprint")
    return None if printed is None else Fingerprint.model_validate(printed).model_dump(mode="json")


def _label(label: Label) -> dict[str, Any]:
    return {
        "source": label.source,
        "stance": label.review.stance if label.review else None,
        "comment": label.review.comment if label.review else None,
        "correction": label.review.correction if label.review else None,
        "outcome": label.outcome.status if label.outcome else None,
        "detail": label.outcome.detail if label.outcome else None,
        "answers": {k: v.model_dump() for k, v in label.answers.items()} if label.answers else None,
        "at": label.at.isoformat(),
    }


def generation_row(row: RunRow, labels: list[Label], *, holdout: bool) -> dict[str, Any]:
    body = row.body
    return {
        "run_id": row.run_id,
        "holdout": holdout,
        "agent": row.agent,
        "task": row.task,
        "context": body.get("context"),
        "tags": body.get("tags", {}),
        "at": row.at.isoformat(),
        "model": row.model,
        "fingerprint": _fingerprint(body),
        "system": body["system"],
        "boot": body.get("boot", []),
        "start": body.get("start", 0),
        "messages": body["messages"],
        "tools": body.get("tools", []),
        "conversation": body.get("conversation", []),
        "steps": body["steps"],
        "reply": body["reply"],
        "stopped": body.get("stopped"),
        "stop_reason": body.get("stop_reason"),
        "usage": body["usage"],
        "labels": [_label(label) for label in labels],
    }


def preference_rows(row: RunRow, labels: list[Label], *, holdout: bool) -> list[dict[str, Any]]:
    return [
        {
            "run_id": row.run_id,
            "holdout": holdout,
            "tags": row.body.get("tags", {}),
            "fingerprint": _fingerprint(row.body),
            "prompt": {
                "system": row.body["system"],
                "boot": row.body.get("boot", []),
                "messages": row.body["messages"],
            },
            "chosen": label.review.correction,
            "rejected": row.body["reply"],
            "label_id": label.label_id,
            "at": label.at.isoformat(),
        }
        for label in labels
        if label.source == "human"
        and label.review is not None
        and isinstance(label.review.correction, str)
    ]


def _value(answer: dict[str, Any]) -> Any:
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "score":
        return answer["score"]
    return answer["probability"]


def _truth_value(answer: dict[str, Any]) -> Any:
    if answer["type"] == "check":
        return answer["probability"] >= 0.5
    return _value(answer)


def _probabilities(answer: dict[str, Any]) -> Any:
    if answer["type"] == "check":
        return {"yes": answer["probability"], "no": round(1.0 - answer["probability"], 12)}
    return answer["probabilities"]


def _truth(question_id: str, labels: list[Label]) -> tuple[Any, str | None]:
    for label in labels:
        if (
            label.source == "human"
            and label.review is not None
            and isinstance(label.review.correction, dict)
            and question_id in label.review.correction
        ):
            return label.review.correction[question_id], "human"
    for label in labels:
        if label.source == "teacher" and label.answers is not None and question_id in label.answers:
            return _truth_value(label.answers[question_id].model_dump()), "teacher"
    return None, None


def decision_rows(row: RunRow, labels: list[Label], *, holdout: bool) -> list[dict[str, Any]]:
    body = row.body
    outcome = next(
        (label.outcome.status for label in reversed(labels) if label.outcome is not None), None
    )
    rows = []
    for question_id, question in body["questions"].items():
        answer = body["answers"][question_id]
        truth, truth_source = _truth(question_id, labels)
        rows.append(
            {
                "run_id": row.run_id,
                "holdout": holdout,
                "agent": row.agent,
                "task": row.task,
                "tags": body.get("tags", {}),
                "at": row.at.isoformat(),
                "model": row.model,
                "fingerprint": _fingerprint(body),
                "state": body["state"],
                "question_id": question_id,
                "question": question,
                "answer": _value(answer),
                "probabilities": _probabilities(answer),
                "confidence": answer["confidence"],
                "truth": truth,
                "truth_source": truth_source,
                "outcome": outcome,
            }
        )
    return rows
