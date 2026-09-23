from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
