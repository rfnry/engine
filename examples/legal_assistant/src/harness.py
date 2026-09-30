from datetime import datetime
from pathlib import Path

from rfnry_engine import Engine, InMemoryStorage, Refinery, Review

from providers import AnthropicProvider
from utils import Folder

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = {path.stem: path.read_text() for path in (ROOT / "data" / "contracts").glob("*.md")}

refinery = Refinery(InMemoryStorage())
engine = Engine(ROOT / "agents", generation=AnthropicProvider(), refinery=refinery)
datasets = Folder(ROOT / "datasets")
watermarks: dict[str, datetime | None] = {}


async def run(firm: str, contract: str, question: str) -> dict:
    engine_run = await engine.run(
        "reviewer", scope=firm, input=question, context=CONTRACTS[contract]
    )
    return {"reply": engine_run.reply, "run_id": engine_run.run_id}


async def review(firm: str, run_id: str, feedback: Review) -> dict:
    label = await refinery.review(firm, run_id, feedback)
    return label.model_dump(mode="json", exclude_none=True)


async def export(firm: str) -> dict:
    exported = await refinery.export(firm, datasets, since=watermarks.get(firm))
    watermarks[firm] = exported.watermark
    return exported.model_dump(mode="json")
