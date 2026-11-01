import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def short_root() -> Iterator[Path]:
    base = Path(tempfile.mkdtemp(dir="/tmp"))
    root = base / "work"
    root.mkdir()
    assert len(str(root / ".rfnry-00000000.sock")) < 100
    yield root
    shutil.rmtree(base)
