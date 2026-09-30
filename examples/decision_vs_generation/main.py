import asyncio
import json
from pathlib import Path

from harness import ASSUMED_USD_PER_MILLION_TOKENS, BUCKETS, QUESTIONS, build, compare, grade
from providers import AnthropicProvider, TypeSafeProvider


def row(label: str, left: str, right: str) -> None:
    print(f"{label:<26}{left:>18}{right:>18}")


async def main() -> None:
    cases = json.loads(Path("cases.json").read_text())
    engine = build(AnthropicProvider(), TypeSafeProvider())
    results = await compare(engine, cases)
    stats = grade(results, cases)

    d, g = stats["decision"], stats["generation"]
    n = len(cases)
    print(f"{n} labelled cases, each asked of both models\n")
    row("", "decision", "generation")
    row("model", d["model"][:17], g["model"][:17])
    for q in QUESTIONS:
        row(f"accuracy: {q}", f"{d['accuracy'][q]}/{n}", f"{g['accuracy'][q]}/{n}")
    row("mean latency per case", f"{d['latency']:.0f} ms", f"{g['latency']:.0f} ms")
    for label, field in (("tokens in", "input"), ("tokens out", "output")):
        row(label, str(getattr(d["usage"], field)), str(getattr(g["usage"], field)))
    row("estimated cost", f"${d['cost']:.5f}", f"${g['cost']:.5f}")
    for i, (label, _, _) in enumerate(BUCKETS):
        cells = []
        for kind in (d, g):
            bucket = kind["buckets"][i]
            share = f"{sum(bucket) / len(bucket):.0%}" if bucket else "-"
            cells.append(f"{sum(bucket)}/{len(bucket)} {share}")
        row(f"confidence {label}", *cells)

    print(f"\nprices assumed, usd per million tokens (in, out): {ASSUMED_USD_PER_MILLION_TOKENS}")
    print("decision confidence is the probability of the answer it gave\n")
    for i, case in enumerate(cases, 1):
        for q in QUESTIONS:
            left = results["decision"][i - 1][1][q][0]
            right = results["generation"][i - 1][1][q][0]
            if left != right:
                print(f"#{i} {q}: decision {left}, generation {right}, right {case[q]}")
                print(f"   {case['message']}")


asyncio.run(main())
