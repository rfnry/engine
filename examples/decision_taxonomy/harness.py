import asyncio
import json
from pathlib import Path

from rfnry_engine import Choice, Decision, DecisionModel, Engine, Usage

TAXONOMY = json.loads(Path("taxonomy.json").read_text())
CRITERIA = ("what", "not_for", "examples")
AT_ONCE = 6
LEVELS = (
    "the in-scope items the check let through",
    "those right at level 1",
    "those right at level 2",
)


def topic(path: list[str], children: dict) -> Choice:
    return Choice(
        instructions={
            "question": "Which of these is `message` about?",
            "within": " > ".join(path) or "everything this store's support desk handles",
            "focus": "The sender's main request, not every topic the message mentions",
        },
        options={
            key: {field: node[field] for field in CRITERIA if field in node}
            for key, node in children.items()
        },
    )


async def walk(engine: Engine, message: str) -> tuple[list[str] | None, list[Decision]]:
    children, path, runs = TAXONOMY, [], []
    while children:
        run = await engine.run(
            "classifier",
            scope="demo",
            input={"message": message},
            questions={"topic": topic(path, children)},
        )
        runs.append(run)
        if not path and run.answers["in_scope"].probability < 0.5:
            return None, runs
        chosen = run.answers["topic"].choice
        path.append(chosen)
        children = children[chosen].get("children", {})
    return path, runs


def build(decision: DecisionModel) -> Engine:
    return Engine("agents", decision=decision)


async def classify_all(
    engine: Engine, items: list[dict]
) -> list[tuple[list[str] | None, list[Decision]]]:
    gate = asyncio.Semaphore(AT_ONCE)

    async def classify(item: dict) -> tuple[list[str] | None, list[Decision]]:
        async with gate:
            return await walk(engine, item["message"])

    return await asyncio.gather(*(classify(item) for item in items))


def grade(results: list[tuple[list[str] | None, list[Decision]]], items: list[dict]) -> dict:
    said = [got for got, _ in results]
    labels = [item["path"] for item in items]
    runs = [run for _, walked in results for run in walked]
    passed = [(got, label) for got, label in zip(said, labels, strict=True) if got and label]
    levels = []
    for depth in range(len(LEVELS)):
        reached = [(got, label) for got, label in passed if got[:depth] == label[:depth]]
        levels.append((sum(got[depth] == label[depth] for got, label in reached), len(reached)))
    return {
        "runs": runs,
        "scoped": sum(
            (got is None) == (label is None) for got, label in zip(said, labels, strict=True)
        ),
        "levels": levels,
        "exact": sum(got == label for got, label in zip(said, labels, strict=True)),
        "usage": sum((run.usage for run in runs), Usage()),
        "mean": sum(run.duration_ms for run in runs) / len(runs),
    }
