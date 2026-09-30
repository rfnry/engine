import ast
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[4] / "examples"


@pytest.mark.parametrize(
    "path", sorted(EXAMPLES.glob("*/src/*.py")), ids=lambda p: p.parents[1].name + "/" + p.name
)
def test_every_example_compiles(path: Path) -> None:
    compile(
        path.read_text(encoding="utf-8"), str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    )
