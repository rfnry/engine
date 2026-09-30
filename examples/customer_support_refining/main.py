import asyncio
import json

from rfnry_engine import is_holdout

from harness import DPO, SFT, ask, build, conflicts, convert, export, feedback, label, load
from providers import AnthropicProvider, TypeSafeProvider


def clip(value):
    if isinstance(value, str):
        return value if len(value) <= 60 else value[:60] + "…"
    if isinstance(value, list):
        return [clip(item) for item in value]
    if isinstance(value, dict):
        return {key: clip(item) for key, item in value.items() if item is not None}
    return value


def show(path, features):
    dataset = load(path, features)
    print(f"{path}: {dataset.num_rows} rows · columns {dataset.column_names}")
    print(f"  {json.dumps(clip(dataset[0]), ensure_ascii=False)}")


async def main() -> None:
    refinery, engine = build(AnthropicProvider(), TypeSafeProvider())

    arm = {"arm": "a"}
    refund = await ask(
        engine, "support", "Order A-1 was charged twice. Can I get my money back?", arm
    )
    late = await ask(engine, "support", "Order A-2 still has not shipped. Where is it?", arm)
    triage = await ask(engine, "triage", {"message": "Checkout is down for every customer."}, arm)
    await feedback(refinery, refund, late, triage)

    labeled = await label(refinery, engine)
    print(f"judged {labeled.judged} · taught {labeled.taught} · errors {labeled.errors}")
    for conflict in await conflicts(refinery):
        print(f"conflict on {conflict.run_id}: {[text for _, _, text in conflict.corrections]}")

    exported = await export(refinery)
    print(f"rows {exported.rows} · watermark {exported.watermark}")
    for key in exported.keys:
        print(f"  datasets/{key}")
    print(f"refund run is holdout: {is_holdout(refund.run_id)}")

    convert(exported.keys)
    show("datasets/trl/sft.jsonl", SFT)
    show("datasets/trl/dpo.jsonl", DPO)


asyncio.run(main())
