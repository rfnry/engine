import asyncio
from pathlib import Path

from rfnry_engine import Engine, Image, InputError, Text

from providers import AnthropicProvider, GoogleProvider, TypeSafeProvider

PHOTO = Image(media_type="image/png", data=Path("photo.png").read_bytes())
QUESTION = Text(text="Which shapes are in this picture, and what colour is each?")


async def main() -> None:
    for model in (AnthropicProvider(), GoogleProvider()):
        engine = Engine("agents", generation=model, decision=TypeSafeProvider())
        run = await engine.run("describer", scope="demo", input=[QUESTION, PHOTO])
        print(f"{run.model} · {run.usage.input} in / {run.usage.output} out")
        print(f"  {run.reply}\n")

    try:
        await engine.run("inspector", scope="demo", input=[QUESTION, PHOTO])
    except InputError as error:
        print(f"inspector refused: {error}")


asyncio.run(main())
