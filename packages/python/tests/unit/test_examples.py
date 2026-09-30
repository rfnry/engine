import ast
import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.models import Choice

EXAMPLES = Path(__file__).parents[4] / "examples"
TAXONOMY = EXAMPLES / "legal_assistant" / "data" / "taxonomy.json"


@pytest.mark.parametrize(
    "path", sorted(EXAMPLES.glob("*/*.py")), ids=lambda p: p.parent.name + "/" + p.name
)
def test_every_example_compiles(path: Path) -> None:
    compile(
        path.read_text(encoding="utf-8"), str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    )


def leaves(children: dict[str, Any], path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    criteria = ("what", "not_for", "examples")
    Choice(
        instructions="Which?",
        options={key: {c: node[c] for c in criteria} for key, node in children.items()},
    )
    found: set[tuple[str, ...]] = set()
    for key, node in children.items():
        found |= leaves(node["children"], (*path, key)) if "children" in node else {(*path, key)}
    return found


def test_every_level_of_the_legal_taxonomy_is_a_choice_with_structured_criteria() -> None:
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    known = leaves(taxonomy["children"])
    assert len(known) == 11
    assert {len(path) for path in known} == {1, 2, 3}
    assert ("liability", "limitation", "cap") in known and ("general",) in known
