import asyncio
import json
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field
from rfnry_engine import Engine, Usage

from providers import AnthropicProvider, TypeSafeProvider

ASSUMED_USD_PER_MILLION_TOKENS = {"decision": (0.042, 0.0), "generation": (3.0, 15.0)}
BUCKETS = [(">= 0.9", 0.9, 1.01), ("0.7-0.9", 0.7, 0.9), ("< 0.7", 0.0, 0.7)]
QUESTIONS = ["team", "urgent"]


class Verdict(BaseModel):
    team: Literal["billing", "technical", "account", "other"]
    team_confidence: float = Field(ge=0, le=1)
    urgent: bool
    urgent_confidence: float = Field(ge=0, le=1)


async def decide(engine: Engine, message: str):
    run = await engine.run("router", scope="demo", input=message)
    team, urgent = run.answers["team"], run.answers["urgent"]
    p = urgent.probability
    return run, {
        "team": (team.choice, team.probabilities[team.choice]),
        "urgent": (p >= 0.5, max(p, 1 - p)),
    }


async def generate(engine: Engine, message: str):
    run = await engine.run("triager", scope="demo", input=message, expect=Verdict)
    verdict = Verdict.model_validate_json(run.reply)
    return run, {
        "team": (verdict.team, verdict.team_confidence),
        "urgent": (verdict.urgent, verdict.urgent_confidence),
    }


def cost(usage: Usage, prices: tuple[float, float]) -> float:
    tokens_in = usage.input + usage.cache_creation + usage.cache_read
    return (tokens_in * prices[0] + usage.output * prices[1]) / 1_000_000


def row(label: str, left: str, right: str) -> None:
    print(f"{label:<26}{left:>18}{right:>18}")


async def main() -> None:
    cases = json.loads(Path("cases.json").read_text())
    engine = Engine("agents", generation=AnthropicProvider(), decision=TypeSafeProvider())
    results = {"decision": [], "generation": []}
    for case in cases:
        for kind, ask in (("decision", decide), ("generation", generate)):
            results[kind].append(await ask(engine, case["message"]))

    stats = {}
    for kind, runs in results.items():
        graded = [
            (answers[q][1], answers[q][0] == case[q])
            for (_, answers), case in zip(runs, cases, strict=True)
            for q in QUESTIONS
        ]
        usage = sum((run.usage for run, _ in runs), Usage())
        stats[kind] = {
            "model": runs[0][0].model or kind,
            "accuracy": {
                q: sum(a[q][0] == c[q] for (_, a), c in zip(runs, cases, strict=True))
                for q in QUESTIONS
            },
            "latency": mean(run.duration_ms for run, _ in runs),
            "usage": usage,
            "cost": cost(usage, ASSUMED_USD_PER_MILLION_TOKENS[kind]),
            "buckets": [[ok for c, ok in graded if low <= c < high] for _, low, high in BUCKETS],
        }

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
