from pathlib import Path

import pytest

from rfnry_engine import Engine

from .fakes import Scripted
from .trees import provided, solo


def test_the_engine_takes_no_name(tmp_path: Path) -> None:
    solo(tmp_path)
    with pytest.raises(TypeError):
        Engine(tmp_path, name="a", providers=provided({"strong": Scripted()}))  # type: ignore[call-arg]
