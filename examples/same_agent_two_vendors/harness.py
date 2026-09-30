from rfnry_engine import Engine, GenerationModel, Run


def build(generation: GenerationModel) -> Engine:
    return Engine("agents", generation=generation)


async def ask(engine: Engine, question: str) -> Run:
    return await engine.run("concierge", scope="demo", input=question)
