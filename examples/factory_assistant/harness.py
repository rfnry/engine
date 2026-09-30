import json
from pathlib import Path

from rfnry_engine import Engine

from providers import AnthropicProvider

PLANTS = json.loads(Path("data/plants.json").read_text())["plants"]

engine = Engine("agents", generation=AnthropicProvider())


async def ask(plant: str, question: str, task: str | None) -> dict:
    facts = json.dumps(PLANTS[plant], indent=2, ensure_ascii=False)
    run = await engine.run("maintenance", scope=plant, input=question, task=task, context=facts)
    return {"reply": run.reply, "run_id": run.run_id}
