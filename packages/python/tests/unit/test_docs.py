from pathlib import Path

import rfnry_engine

DOCS = Path(__file__).parents[4] / "docs" / "engine.md"


def test_every_public_name_is_documented() -> None:
    text = DOCS.read_text(encoding="utf-8")
    missing = [name for name in rfnry_engine.__all__ if name not in text]
    assert missing == []
