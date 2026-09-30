from rfnry_engine import Engine, GenerationModel, InMemoryStorage, Refinery, Run, Tool, Tools

ACCOUNTS = {"acme": {"plan": "enterprise", "seats": 250}, "globex": {"plan": "starter", "seats": 3}}


def account_tools(scope: str) -> Tools:
    async def account(args):
        return ACCOUNTS[scope]

    return Tools([Tool("account", "This tenant's plan and seat count", body=account)])


def build(generation: GenerationModel) -> tuple[Refinery, Engine]:
    refinery = Refinery(InMemoryStorage())
    return refinery, Engine("agents", generation=generation, refinery=refinery)


async def ask(engine: Engine, scope: str, question: str, context: str) -> Run:
    return await engine.run(
        "assistant",
        scope=scope,
        input=question,
        context=context,
        tools=account_tools(scope),
    )


async def rows(refinery: Refinery, scope: str) -> int:
    return len(await refinery.storage.runs(scope))


async def forget(refinery: Refinery, scope: str) -> int:
    return await refinery.delete_scope(scope)


async def absent(refinery: Refinery, scope: str) -> bool:
    return await refinery.verify_absent(scope)
