import json
from pathlib import Path

from rfnry_engine import Engine

from providers import AnthropicProvider

ROOT = Path(__file__).resolve().parent.parent
PLANTS = json.loads((ROOT / "data" / "plants.json").read_text())["plants"]

engine = Engine(ROOT / "agents", generation=AnthropicProvider())


async def ask(plant: str, question: str) -> dict:
    facts = json.dumps(PLANTS[plant], indent=2, ensure_ascii=False)
    run = await engine.run("maintenance", scope=plant, input=question, context=facts)
    return {"reply": run.reply, "run_id": run.run_id}
