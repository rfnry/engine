from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field
from rfnry_engine import DecisionModel, Engine, GenerationModel, Usage

ASSUMED_USD_PER_MILLION_TOKENS = {"decision": (0.042, 0.0), "generation": (3.0, 15.0)}
BUCKETS = [(">= 0.9", 0.9, 1.01), ("0.7-0.9", 0.7, 0.9), ("< 0.7", 0.0, 0.7)]
QUESTIONS = ["team", "urgent"]


class Verdict(BaseModel):
    team: Literal["billing", "technical", "account", "other"]
    team_confidence: float = Field(ge=0, le=1)
    urgent: bool
    urgent_confidence: float = Field(ge=0, le=1)


def build(generation: GenerationModel, decision: DecisionModel) -> Engine:
    return Engine("agents", generation=generation, decision=decision)


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


async def compare(engine: Engine, cases: list[dict]) -> dict:
    results = {"decision": [], "generation": []}
    for case in cases:
        for kind, ask in (("decision", decide), ("generation", generate)):
            results[kind].append(await ask(engine, case["message"]))
    return results


def cost(usage: Usage, prices: tuple[float, float]) -> float:
    tokens_in = usage.input + usage.cache_creation + usage.cache_read
    return (tokens_in * prices[0] + usage.output * prices[1]) / 1_000_000


def grade(results: dict, cases: list[dict]) -> dict:
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
    return stats
