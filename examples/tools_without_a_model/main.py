import asyncio

from rfnry_engine import Engine, EngineTools, InputError, Jail, Tool, Tools

from providers import TypeSafeProvider


async def word_count(args):
    return str(len(Jail("agents/librarian").resolve(args["path"]).read_text().split()))


async def today(args):
    return "2026-09-23"


async def main() -> None:
    engine = Engine(
        "agents",
        decision=TypeSafeProvider(),
        tools=[
            Tool("word_count", "Count the words in a file of the agent directory", body=word_count)
        ],
    )
    tools = EngineTools(engine, agent="librarian", scope="demo")

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

    combined = tools.only("Read", "Grep") + Tools([Tool("today", "Today's date", body=today)])
    print(combined.names, combined.without("Grep").names)


asyncio.run(main())
