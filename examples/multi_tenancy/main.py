import asyncio

from harness import absent, ask, build, forget, rows
from providers import AnthropicProvider


async def main() -> None:
    refinery, engine = build(AnthropicProvider())

    for scope, context in [
        ("acme", "Acme pays yearly."),
        ("globex", "Globex is on a trial until October."),
    ]:
        run = await ask(engine, scope, "How many seats do we have, and how do we pay?", context)
        print(f"{scope}: {run.reply}")

    for scope in ("acme", "globex"):
        print(f"{scope} rows: {await rows(refinery, scope)}")

    print(f"deleted from acme: {await forget(refinery, 'acme')}")
    for scope in ("acme", "globex"):
        print(f"{scope} absent: {await absent(refinery, scope)}")


asyncio.run(main())
