import json

from rfnry_engine.refinery.labels import Conflict, Label


def _text(label: Label) -> str | None:
    if label.source != "human" or label.review is None or label.review.correction is None:
        return None
    correction = label.review.correction
    text = (
        correction.strip()
        if isinstance(correction, str)
        else json.dumps(correction, sort_keys=True)
    )
    return text or None


def find_conflicts(labels: list[Label]) -> list[Conflict]:
    by_run: dict[str, list[tuple[Label, str]]] = {}
    for label in labels:
        text = _text(label)
        if text is not None:
            by_run.setdefault(label.run_id, []).append((label, text))
    conflicts: list[Conflict] = []
    for run_id, group in sorted(by_run.items()):
        if len({text for _, text in group}) < 2:
            continue
        ordered = sorted(group, key=lambda item: (item[0].at, item[0].label_id))
        conflicts.append(
            Conflict(
                run_id=run_id,
                corrections=[(label.label_id, label.at, text) for label, text in ordered],
            )
        )
    return conflicts
