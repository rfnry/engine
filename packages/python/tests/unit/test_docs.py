import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import rfnry_engine
from rfnry_engine import (
    CheckAnswer,
    Engine,
    Image,
    InMemoryStorage,
    Message,
    Outcome,
    Refinery,
    Reply,
    Review,
    State,
    Text,
    ToolSpec,
    is_holdout,
)

from .fakes import Fixed

DOCS = Path(__file__).parents[4] / "docs" / "engine.md"
GUIDE = Path(__file__).parents[4] / "docs" / "evaluation.md"


def test_every_public_name_is_documented() -> None:
    text = DOCS.read_text(encoding="utf-8")
    missing = [name for name in rfnry_engine.__all__ if name not in text]
    assert missing == []


def test_the_evaluation_guide_compiles_and_names_real_verbs() -> None:
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(encoding="utf-8"), re.S)
    blocks = [textwrap.dedent(block) for block in blocks]
    assert blocks
    for block in blocks:
        compile(block, "evaluation.md", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    text = GUIDE.read_text(encoding="utf-8")
    for verb in re.findall(r"refinery\.(\w+)\(", text):
        assert hasattr(rfnry_engine.Refinery, verb), verb
    for field in re.findall(r"Review\((\w+)=", text) + re.findall(r"Outcome\((\w+)=", text):
        assert (
            field in rfnry_engine.Review.model_fields or field in rfnry_engine.Outcome.model_fields
        ), field


class Seeing:
    def __init__(self) -> None:
        self.seen: list[Message] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.seen.append(messages[-1])
        return Reply(text="Refund the second charge.")


async def test_the_holdout_replay_grades_the_candidate_on_what_each_run_was_given(
    tmp_path: Path,
) -> None:
    for path, text in {
        "support/AGENT.md": "---\nname: support\nkind: generation\n---\nYou help.\n",
        "grader/AGENT.md": "---\nname: grader\nkind: decision\n---\n",
        "grader/questions/matches.md": "---\ntype: check\n---\nDoes the reply match?\n",
    }.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    refinery = Refinery(InMemoryStorage())
    graded = Fixed({"matches": CheckAnswer(probability=0.9)})
    engine = Engine(tmp_path, generation=Seeing(), decision=graded)
    held = [f"run-{i}" for i in range(200) if is_holdout(f"run-{i}")][:4]
    kept = next(f"run-{i}" for i in range(200) if not is_holdout(f"run-{i}"))
    parts = [Text(text="Charged twice?"), Image(media_type="image/png", data=b"\x89PNG\r\n\x1a\n")]
    inputs: dict[str, State] = {
        held[0]: list(parts),
        held[1]: "Is it down?",
        held[2]: "Where is A-1?",
        held[3]: "Unlabeled.",
        kept: "Hello",
    }
    for run_id, given in inputs.items():
        run = await engine.run("support", "acme", given)
        await refinery.record(run.model_copy(update={"run_id": run_id}))
    await refinery.review("acme", held[0], Review(stance="approval"))
    await refinery.review("acme", held[0], Review(correction="Refund it."))
    await refinery.review("acme", held[1], Review(stance="approval"))
    await refinery.outcome("acme", held[2], Outcome(status="succeeded"))
    await refinery.review("acme", kept, Review(correction="Hi."))
    candidate_model = Seeing()
    namespace: dict[str, Any] = {
        "refinery": refinery,
        "engine": engine,
        "candidate": Engine(tmp_path, generation=candidate_model),
    }
    [block] = [
        textwrap.dedent(b)
        for b in re.findall(r"```python\n(.*?)```", GUIDE.read_text(encoding="utf-8"), re.S)
        if b.startswith("async def read_only")
    ]
    code = compile(block, "evaluation.md", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    await eval(code, namespace)
    assert [m.content for m in candidate_model.seen] == [parts, "Is it down?", "Where is A-1?"]
    assert [json.loads(json.dumps(state)) for state, _ in graded.calls] == [
        {"reference": "Refund it.", "reply": "Refund the second charge."},
        {"reference": "Refund the second charge.", "reply": "Refund the second charge."},
    ]
    assert namespace["grades"] == [0.9, 0.9] and namespace["failures"] == []
