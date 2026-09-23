import asyncio

from rfnry_engine import Engine, InMemoryStorage, Refinery, Tool, Tools

from providers import AnthropicProvider

ACCOUNTS = {"acme": {"plan": "enterprise", "seats": 250}, "globex": {"plan": "starter", "seats": 3}}


def account_tools(scope: str) -> Tools:
    async def account(args):
        return ACCOUNTS[scope]

    return Tools([Tool("account", "This tenant's plan and seat count", body=account)])


async def main() -> None:
    refinery = Refinery(InMemoryStorage())
    engine = Engine("agents", generation=AnthropicProvider(), refinery=refinery)

    for scope, context in [
        ("acme", "Acme pays yearly."),
        ("globex", "Globex is on a trial until October."),
    ]:
        run = await engine.run(
            "assistant",
            scope=scope,
            input="How many seats do we have, and how do we pay?",
            context=context,
            tools=account_tools(scope),
        )
        print(f"{scope}: {run.reply}")

    for scope in ("acme", "globex"):
        print(f"{scope} rows: {len(await refinery.storage.runs(scope))}")

    print(f"deleted from acme: {await refinery.delete_scope('acme')}")
    for scope in ("acme", "globex"):
        print(f"{scope} absent: {await refinery.verify_absent(scope)}")


asyncio.run(main())
