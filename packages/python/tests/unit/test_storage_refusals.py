import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from rfnry_engine.errors import StorageError
from rfnry_engine.refinery.postgres import PostgresStorage


class Failing:
    def __init__(self, error: Exception) -> None:
        self.error = error

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        raise self.error
        yield


async def test_without_asyncpg_a_connection_error_or_a_timeout_is_a_storage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "asyncpg", None)
    for error, said in (
        (ConnectionRefusedError(111, "refused"), r"\[Errno 111\] refused"),
        (TimeoutError("acquire timed out"), "acquire timed out"),
    ):
        with pytest.raises(
            StorageError, match=f"^refinery/postgres: storage rejected count_scope: {said}$"
        ) as raised:
            await PostgresStorage(Failing(error)).count_scope("a")
        assert raised.value.__cause__ is error


async def test_without_asyncpg_any_other_error_reaches_the_caller_as_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "asyncpg", None)
    error = RuntimeError("pool broke")
    with pytest.raises(RuntimeError, match="^pool broke$") as raised:
        await PostgresStorage(Failing(error)).count_scope("a")
    assert raised.value is error


async def test_a_refusal_without_text_is_named_by_its_type() -> None:
    with pytest.raises(
        StorageError, match="^refinery/postgres: storage rejected count_scope: TimeoutError$"
    ):
        await PostgresStorage(Failing(TimeoutError())).count_scope("a")
