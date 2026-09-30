from rfnry_engine import DecisionModel, Engine, EngineTools, Jail, Tool, Tools


async def word_count(args):
    return str(len(Jail("agents/librarian").resolve(args["path"]).read_text().split()))


async def today(args):
    return "2026-09-23"


def build(decision: DecisionModel) -> EngineTools:
    engine = Engine(
        "agents",
        decision=decision,
        tools=[
            Tool("word_count", "Count the words in a file of the agent directory", body=word_count)
        ],
    )
    return EngineTools(engine, agent="librarian", scope="demo")


def combine(tools: EngineTools) -> Tools:
    return tools.only("Read", "Grep") + Tools([Tool("today", "Today's date", body=today)])
