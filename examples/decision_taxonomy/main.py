import asyncio
import json
import os
import sys
import time
from pathlib import Path

from rfnry_engine import Choice, Decision, Engine, Usage

from providers import LocalDecisionProvider, TypeSafeProvider

TAXONOMY = json.loads(Path("taxonomy.json").read_text())
ITEMS = json.loads(Path("items.json").read_text())
CRITERIA = ("what", "not_for", "examples")
URL = os.environ.get("LOCAL_DECISION_URL", "http://localhost:11434/v1")
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


def shown(path: list[str] | None) -> str:
    return "out of scope" if path is None else " > ".join(path)


def rate(right: int, n: int) -> str:
    return f"{right}/{n}".rjust(7) + (f"  {right / n:4.0%}" if n else "     -")


async def main() -> None:
    if sys.argv[1:2] == ["local"]:
        decision = LocalDecisionProvider(sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b", URL)
    else:
        decision = TypeSafeProvider()
    engine = Engine("agents", decision=decision)
    gate = asyncio.Semaphore(AT_ONCE)

    async def classify(item: dict) -> tuple[list[str] | None, list[Decision]]:
        async with gate:
            return await walk(engine, item["message"])

    started = time.monotonic()
    results = await asyncio.gather(*(classify(item) for item in ITEMS))
    wall = time.monotonic() - started
    said = [got for got, _ in results]
    labels = [item["path"] for item in ITEMS]
    runs = [run for _, walked in results for run in walked]
    model = runs[0].model or getattr(decision, "model", "")

    print(f"decision_taxonomy · {model} · {len(ITEMS)} synthetic messages, labelled by hand")
    print("each call asks the folder's in_scope check and one runtime question, topic, whose")
    print("options are the children of the node chosen last\n")

    scoped = sum((got is None) == (label is None) for got, label in zip(said, labels, strict=True))
    print(f"in scope   {rate(scoped, len(ITEMS))}   the check on each item's first call")
    passed = [(got, label) for got, label in zip(said, labels, strict=True) if got and label]
    for depth, of in enumerate(LEVELS):
        reached = [(got, label) for got, label in passed if got[:depth] == label[:depth]]
        right = sum(got[depth] == label[depth] for got, label in reached)
        print(f"level {depth + 1}    {rate(right, len(reached))}   of {of}")
    exact = sum(got == label for got, label in zip(said, labels, strict=True))
    print(f"end to end {rate(exact, len(ITEMS))}   a leaf, or out of scope, against the label\n")

    usage = sum((run.usage for run in runs), Usage())
    mean = sum(run.duration_ms for run in runs) / len(runs)
    print(f"{len(runs)} calls, {mean:.0f} ms each on average, {AT_ONCE} items at a time, ", end="")
    print(f"{wall:.1f} s in all · {usage.input} tokens in, {usage.output} out\n")

    print("wrong:")
    for i, ((got, walked_runs), item) in enumerate(zip(results, ITEMS, strict=True), 1):
        if got != item["path"]:
            p = walked_runs[0].answers["in_scope"].probability
            print(f"#{i:<3}{shown(item['path']):<34} said {shown(got)} (in_scope {p:.2f})")
            print(f"    {item['message']}")

    print(f"\none item, one call per level: {ITEMS[0]['message']}")
    for run in results[0][1]:
        answer, p = run.answers["topic"], run.answers["in_scope"].probability
        print(f"  asked {sorted(run.questions)}, runtime {run.runtime}, ", end="")
        print(f"fingerprint.questions {run.fingerprint.questions}")
        print(f"    topic {answer.choice} at {answer.confidence:.2f}, in_scope {p:.2f}")


asyncio.run(main())
