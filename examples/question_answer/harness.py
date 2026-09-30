from rfnry_engine import Engine, GenerationModel, Message, Run


def build(generation: GenerationModel) -> Engine:
    return Engine("agents", generation=generation)


async def ask(engine: Engine, question: str, history: list[Message] | None = None) -> Run:
    return await engine.run("assistant", scope="demo", input=question, history=history)
