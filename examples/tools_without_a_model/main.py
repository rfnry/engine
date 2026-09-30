import asyncio

from rfnry_engine import InputError

from harness import build, combine
from providers import TypeSafeProvider


async def main() -> None:
    tools = build(TypeSafeProvider())

    print([d["name"] for d in tools.definitions()])
    print(await tools.dispatch("Grep", {"pattern": "30 days"}))
    print(await tools.dispatch("word_count", {"path": "notes/returns.md"}))
    print(
        await tools.dispatch(
            "Delegate", {"agent": "classifier", "input": "I returned it twice and still no refund!"}
        )
    )

    for name, args in [
        ("Read", {"path": "../../providers.py"}),
        ("word_count", {"path": "/etc/passwd"}),
    ]:
        try:
            await tools.dispatch(name, args)
        except InputError as error:
            print(f"refused: {error}")

    combined = combine(tools)
    print(combined.names, combined.without("Grep").names)


asyncio.run(main())
