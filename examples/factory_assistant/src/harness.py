import json
from pathlib import Path

from rfnry_engine import Engine

from providers import AnthropicProvider

ROOT = Path(__file__).resolve().parent.parent
PLANTS = json.loads((ROOT / "data" / "plants.json").read_text())["plants"]

engine = Engine(ROOT / "agents", models={"strong": AnthropicProvider()}, workers={"maintenance": 2})


async def run(plant: str, question: str) -> dict:
    facts = json.dumps(PLANTS[plant], indent=2, ensure_ascii=False)
    engine_run = await engine.run("maintenance", scope=plant, input=question, context=facts)
    return {"reply": engine_run.reply, "run_id": engine_run.run_id}
