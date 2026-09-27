import ast
import re
import textwrap
from pathlib import Path

import rfnry_engine

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
