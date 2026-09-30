import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from rfnry_engine import Choice, Engine, InMemoryStorage, Refinery, Review

from providers import decision, generation

TAXONOMY = json.loads(Path("data/taxonomy.json").read_text())
CONTRACTS = {path.stem: path.read_text() for path in Path("data/contracts").glob("*.md")}
CRITERIA = ("what", "not_for", "examples")


class Issue(BaseModel):
    clause: int
    kind: str
    risk: Literal["low", "medium", "high"]
    problem: str
    redline: str


class Issues(BaseModel):
    issues: list[Issue]


class Folder:
    def __init__(self, root: str) -> None:
        self.root = Path(root)

    async def put(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


refinery = Refinery(InMemoryStorage())
engine = Engine("agents", generation=generation(), decision=decision(), refinery=refinery)
watermarks: dict[str, datetime | None] = {}


def kind(path: list[str], children: dict) -> Choice:
    return Choice(
        instructions={
            "question": "Which kind of clause is `clause`?",
            "within": " > ".join(path) or "any clause of a commercial contract",
        },
        options={key: {field: node[field] for field in CRITERIA} for key, node in children.items()},
    )


async def classify(firm: str, clause: str) -> dict:
    children, path, levels = TAXONOMY["children"], [], []
    while children:
        run = await engine.run(
            "clauses",
            scope=firm,
            input={"clause": clause},
            questions={"kind": kind(path, children)},
        )
        answer = run.answers["kind"]
        path.append(answer.choice)
        levels.append(
            {"kind": answer.choice, "confidence": answer.confidence, "run_id": run.run_id}
        )
        children = children[answer.choice].get("children", {})
    return {"path": path, "levels": levels}


async def review(firm: str, contract: str) -> dict:
    run = await engine.run("reviewer", scope=firm, input=CONTRACTS[contract], expect=Issues)
    return {
        "issues": Issues.model_validate_json(run.reply).model_dump()["issues"],
        "asked": [{"agent": c.agent, "task": c.task, "run_id": c.run_id} for c in run.children],
        "run_id": run.run_id,
    }


async def review_run(firm: str, run_id: str, feedback: Review) -> dict:
    label = await refinery.review(firm, run_id, feedback)
    return label.model_dump(mode="json", exclude_none=True)


async def export(firm: str) -> dict:
    exported = await refinery.export(firm, Folder("datasets"), since=watermarks.get(firm))
    watermarks[firm] = exported.watermark
    return exported.model_dump(mode="json")
