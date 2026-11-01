import importlib
import os
from typing import Any

import pytest


@pytest.fixture(scope="session")
def adapters() -> Any:
    path = os.environ.get("RFNRY_LIVE_ADAPTERS")
    if not path:
        pytest.fail(
            "tests: the live tier needs RFNRY_LIVE_ADAPTERS, the module path of your live adapters"
        )
    return importlib.import_module(path)
