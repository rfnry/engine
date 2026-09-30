import asyncio
import json
import os
import sys
import time
from pathlib import Path

from harness import AT_ONCE, LEVELS, build, classify_all, grade
from providers import LocalDecisionProvider, TypeSafeProvider

ITEMS = json.loads(Path("items.json").read_text())
URL = os.environ.get("LOCAL_DECISION_URL", "http://localhost:11434/v1")


def shown(path: list[str] | None) -> str:
    return "out of scope" if path is None else " > ".join(path)


def rate(right: int, n: int) -> str:
    return f"{right}/{n}".rjust(7) + (f"  {right / n:4.0%}" if n else "     -")


async def main() -> None:
    if sys.argv[1:2] == ["local"]:
        decision = LocalDecisionProvider(sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b", URL)
    else:
        decision = TypeSafeProvider()
    engine = build(decision)

    started = time.monotonic()
    results = await classify_all(engine, ITEMS)
    wall = time.monotonic() - started
    graded = grade(results, ITEMS)
    runs = graded["runs"]
    model = runs[0].model or getattr(decision, "model", "")

    print(f"decision_taxonomy · {model} · {len(ITEMS)} synthetic messages, labelled by hand")
    print("each call asks the folder's in_scope check and one runtime question, topic, whose")
    print("options are the children of the node chosen last\n")

    scoped, exact = graded["scoped"], graded["exact"]
    print(f"in scope   {rate(scoped, len(ITEMS))}   the check on each item's first call")
    for depth, ((right, reached), of) in enumerate(zip(graded["levels"], LEVELS, strict=True)):
        print(f"level {depth + 1}    {rate(right, reached)}   of {of}")
    print(f"end to end {rate(exact, len(ITEMS))}   a leaf, or out of scope, against the label\n")

    usage, mean = graded["usage"], graded["mean"]
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
