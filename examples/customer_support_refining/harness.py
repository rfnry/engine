import json
from pathlib import Path

from datasets import Features, Json, List, Value, disable_progress_bars, load_dataset
from rfnry_engine import (
    Conflict,
    DecisionModel,
    Engine,
    Exported,
    GenerationModel,
    InMemoryStorage,
    Labeled,
    Outcome,
    Refinery,
    Review,
    Run,
    State,
    Tool,
)


async def lookup_order(args):
    return {"order_id": args["order_id"], "amount": 40.0, "charges": 2}


class Folder:
    def __init__(self, root):
        self.root = Path(root)

    async def put(self, key, data):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        (self.root / key).write_bytes(data)


def text(content):
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if part["type"] == "text")


def function(tool):
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


def trl_sft(row):
    names = {}
    messages = [{"role": "system", "content": row["system"]}]
    for message in row["conversation"]:
        turn = {"role": message["role"], "content": text(message["content"])}
        if message["tool_calls"]:
            turn["tool_calls"] = [
                {"type": "function", "function": {"name": call["name"], "arguments": call["args"]}}
                for call in message["tool_calls"]
            ]
            names.update({call["id"]: call["name"] for call in message["tool_calls"]})
        if message["role"] == "tool":
            turn["name"] = names[message["tool_call_id"]]
        messages.append(turn)
    return {"messages": messages, "tools": [function(tool) for tool in row["tools"]]}


def trl_dpo(row):
    prompt = [{"role": "system", "content": row["prompt"]["system"]}]
    prompt += [
        {"role": m["role"], "content": text(m["content"])} for m in row["prompt"]["messages"]
    ]
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def openai_chat(row):
    messages = [{"role": "system", "content": row["system"]}]
    for message in row["conversation"]:
        turn = {"role": message["role"], "content": text(message["content"])}
        if message["tool_calls"]:
            turn["content"] = turn["content"] or None
            turn["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                }
                for call in message["tool_calls"]
            ]
        if message["role"] == "tool":
            turn["tool_call_id"] = message["tool_call_id"]
        messages.append(turn)
    return {"messages": messages, "tools": [function(tool) for tool in row["tools"]]}


TURN = {"role": Value("string"), "content": Value("string")}
SFT = Features(
    {
        "messages": List({**TURN, "name": Value("string"), "tool_calls": List(Json())}),
        "tools": List(Json()),
    }
)
DPO = Features({"prompt": List(TURN), "chosen": List(TURN), "rejected": List(TURN)})


def rows(keys, dataset):
    return [
        json.loads(line)
        for key in keys
        if f"/{dataset}/" in key
        for line in Path("datasets", key).read_text().splitlines()
    ]


def write(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def build(generation: GenerationModel, decision: DecisionModel) -> tuple[Refinery, Engine]:
    refinery = Refinery(InMemoryStorage())
    engine = Engine(
        "agents",
        generation=generation,
        decision=decision,
        refinery=refinery,
        tools=[Tool("lookup_order", body=lookup_order)],
    )
    return refinery, engine


async def ask(engine: Engine, agent: str, input: State, tags: dict[str, str]) -> Run:
    return await engine.run(agent, "acme", input, tags=tags)


async def feedback(refinery: Refinery, refund: Run, late: Run, triage: Run) -> None:
    await refinery.review(
        "acme",
        refund.run_id,
        Review(
            stance="approval",
            correction="Yes: the second $40 charge is refunded in full within five business days.",
        ),
    )
    await refinery.review(
        "acme", refund.run_id, Review(correction="Refund the duplicate $40 charge today.")
    )
    await refinery.review("acme", triage.run_id, Review(correction={"urgent": True, "severity": 3}))
    await refinery.outcome(
        "acme", late.run_id, Outcome(status="failed", detail="Customer escalated.")
    )


async def label(refinery: Refinery, engine: Engine) -> Labeled:
    return await refinery.label("acme", engine, judge="judge", teacher="teacher")


async def conflicts(refinery: Refinery) -> list[Conflict]:
    return await refinery.conflicts("acme")


async def export(refinery: Refinery) -> Exported:
    return await refinery.export("acme", Folder("datasets"))


def convert(keys: list[str]) -> None:
    generation, preference = rows(keys, "generation"), rows(keys, "preference")
    write("datasets/trl/sft.jsonl", [trl_sft(row) for row in generation])
    write("datasets/trl/dpo.jsonl", [trl_dpo(row) for row in preference])
    write("datasets/openai/chat.jsonl", [openai_chat(row) for row in generation])


def load(path, features):
    disable_progress_bars()
    return load_dataset("json", data_files=path, features=features, split="train")
