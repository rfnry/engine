from typing import Any, Protocol, runtime_checkable

from rfnry_engine.refinery.labels import Label
from rfnry_engine.refinery.storage import RunRow

DATASETS = ("generation", "preference", "decision")


@runtime_checkable
class Bucket(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...


def export_key(scope: str, agent: str, dataset: str, stamp: str) -> str:
    return f"{scope}/{agent}/{dataset}/{stamp}.jsonl"


def _messages(body: dict[str, Any]) -> list[Any]:
    messages: list[Any] = body["conversation"][: body["sent"]]
    return messages


def _fingerprint(body: dict[str, Any]) -> dict[str, Any]:
    fingerprint: dict[str, Any] = body["fingerprint"]
    return fingerprint


def _label(label: Label) -> dict[str, Any]:
    return {
        "source": label.source,
        "stance": label.review.stance if label.review else None,
        "comment": label.review.comment if label.review else None,
        "correction": label.review.correction if label.review else None,
        "outcome": label.outcome.status if label.outcome else None,
        "detail": label.outcome.detail if label.outcome else None,
        "answers": {k: v.model_dump() for k, v in label.answers.items()} if label.answers else None,
        "model": label.model,
        "settings": label.settings,
        "at": label.at.isoformat(),
    }


def generation_row(row: RunRow, labels: list[Label], *, holdout: bool) -> dict[str, Any]:
    body = row.body
    return {
        "run_id": row.run_id,
        "parent_run_id": row.parent_run_id,
        "call": body["call"],
        "holdout": holdout,
        "agent": row.agent,
        "task": row.task,
        "skill": body["skill"],
        "context": body["context"],
        "tags": body["tags"],
        "at": row.at.isoformat(),
        "model": row.model,
        "settings": body["settings"],
        "fingerprint": _fingerprint(body),
        "system": body["system"],
        "preload": body["preload"],
        "start": body["start"],
        "messages": _messages(body),
        "tools": body["tools"],
        "conversation": body["conversation"],
        "steps": body["steps"],
        "reply": body["reply"],
        "stopped": body["stopped"],
        "stop_reason": body["stop_reason"],
        "usage": body["usage"],
        "labels": [_label(label) for label in labels],
    }


def preference_rows(row: RunRow, labels: list[Label], *, holdout: bool) -> list[dict[str, Any]]:
    return [
        {
            "run_id": row.run_id,
            "parent_run_id": row.parent_run_id,
            "call": row.body["call"],
            "holdout": holdout,
            "tags": row.body["tags"],
            "fingerprint": _fingerprint(row.body),
            "model": row.model,
            "settings": row.body["settings"],
            "prompt": {
                "system": row.body["system"],
                "preload": row.body["preload"],
                "messages": _messages(row.body),
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


def _truth(question_id: str, labels: list[Label]) -> tuple[Any, Label | None]:
    for label in labels:
        if (
            label.source == "human"
            and label.review is not None
            and isinstance(label.review.correction, dict)
            and question_id in label.review.correction
        ):
            return label.review.correction[question_id], label
    for label in labels:
        if label.source == "teacher" and label.answers is not None and question_id in label.answers:
            return _truth_value(label.answers[question_id].model_dump()), label
    return None, None


def decision_rows(row: RunRow, labels: list[Label], *, holdout: bool) -> list[dict[str, Any]]:
    body = row.body
    outcome = next(
        (label.outcome.status for label in reversed(labels) if label.outcome is not None), None
    )
    rows = []
    for question_id, question in body["questions"].items():
        answer = body["answers"][question_id]
        truth, label = _truth(question_id, labels)
        rows.append(
            {
                "run_id": row.run_id,
                "parent_run_id": row.parent_run_id,
                "call": body["call"],
                "holdout": holdout,
                "agent": row.agent,
                "tags": body["tags"],
                "at": row.at.isoformat(),
                "model": row.model,
                "settings": body["settings"],
                "fingerprint": _fingerprint(body),
                "state": body["state"],
                "question_id": question_id,
                "question": question,
                "answer": _value(answer),
                "probabilities": _probabilities(answer),
                "confidence": answer["confidence"],
                "truth": truth,
                "truth_source": label.source if label else None,
                "truth_model": label.model if label else None,
                "truth_settings": label.settings if label else None,
                "outcome": outcome,
            }
        )
    return rows
