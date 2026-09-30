import asyncio
import json
import os
import sys
from pathlib import Path

from harness import BUCKETS, QUESTIONS, RECORDED, build, judge, measure
from providers import LocalDecisionProvider

URL = os.environ.get("LOCAL_DECISION_URL", "http://localhost:11434/v1")
MODELS = {"llama3.1:8b": {}, "qwen3:8b": {"reasoning_effort": "none"}}


def row(label: str, *cells: str) -> None:
    print(f"{label:<26}" + "".join(f"{cell:>18}" for cell in cells))


async def main() -> None:
    cases = json.loads(Path("cases.json").read_text())
    models = sys.argv[1:] or list(MODELS)
    stats = {}
    for name in models:
        decision = LocalDecisionProvider(name, URL, MODELS.get(name))
        stats[name] = await measure(build(decision), cases)
    columns = stats | RECORDED

    n = len(cases)
    print(f"{n} labelled cases on the router decision agent, local models through {URL}\n")
    row("", *(name[:17] for name in columns))
    row("", *(column["kind"] for column in columns.values()))
    for q in QUESTIONS:
        row(f"accuracy: {q}", *(f"{column['accuracy'][q]}/{n}" for column in columns.values()))
    row("mean latency per case", *(f"{column['latency']:.0f} ms" for column in columns.values()))
    for label, field in (("tokens in", "input"), ("tokens out", "output")):
        row(label, *(str(getattr(column["usage"], field)) for column in columns.values()))
    for i, (label, _, _) in enumerate(BUCKETS):
        cells = []
        for column in columns.values():
            right, asked = column["buckets"][i]
            cells.append(f"{right}/{asked} {right / asked:.0%}" if asked else "0/0 -")
        row(f"confidence {label}", *cells)

    print("\nthe vendor columns are decision_vs_generation's run of 2026-09-28")
    print("confidence is the probability of the answer given; for the local models it is a")
    print("softmax over the option letters' log-probabilities, which nothing has calibrated\n")
    print("wrong at a confidence of 0.9 or more:")
    for name, column in stats.items():
        for i, q, (answer, confidence), case in column["misses"]:
            if confidence >= 0.9:
                print(f"#{i} {q}: {name} said {answer} at {confidence:.2f}, right {case[q]}")
                print(f"   {case['message']}")

    for name in models:
        decision = LocalDecisionProvider(name, URL, MODELS.get(name))
        hits = await judge(decision)
        print(
            f"\nmemory's judge on {decision.model}: {len(hits.hits)} hits, "
            f"{hits.usage.input} tokens in, {hits.usage.output} out"
        )
        for hit in hits.hits:
            j = hit.judge
            print(
                f"relevant={j.relevant:.2f} evidence={j.evidence:.2f} "
                f"contradicts={j.contradicts:.2f} injection={j.injection:.2f}  {hit.text[:50]}"
            )


asyncio.run(main())
