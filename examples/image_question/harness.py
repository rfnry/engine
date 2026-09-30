from pathlib import Path

from rfnry_engine import DecisionModel, Engine, GenerationModel, Image, Run, Text

PHOTO = Image(media_type="image/png", data=Path("photo.png").read_bytes())
QUESTION = Text(text="Which shapes are in this picture, and what colour is each?")


def build(generation: GenerationModel, decision: DecisionModel) -> Engine:
    return Engine("agents", generation=generation, decision=decision)


async def describe(engine: Engine) -> Run:
    return await engine.run("describer", scope="demo", input=[QUESTION, PHOTO])


async def inspect(engine: Engine) -> Run:
    return await engine.run("inspector", scope="demo", input=[QUESTION, PHOTO])
