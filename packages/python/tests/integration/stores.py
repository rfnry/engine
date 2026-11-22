from collections.abc import Callable, Sequence
from typing import Any

from pydantic import TypeAdapter

from rfnry_engine.models import Message

TABLE = (
    "CREATE TABLE IF NOT EXISTS store_messages (scope text, conversation text, position int, "
    "body jsonb, PRIMARY KEY (scope, conversation, position))"
)
MESSAGES = TypeAdapter(list[Message])


def whole(conversation: str, length: int) -> int:
    return 0


class PostgresStore:
    def __init__(self, pool: Any, window: Callable[[str, int], int] = whole) -> None:
        self.pool = pool
        self.windows = window

    async def length(self, scope: str, conversation: str) -> int:
        held: int = await self.pool.fetchval(
            "SELECT count(*) FROM store_messages WHERE scope = $1 AND conversation = $2",
            scope,
            conversation,
        )
        return held

    async def read(
        self, scope: str, conversation: str, start: int, stop: int | None = None
    ) -> Sequence[object]:
        rows = await self.pool.fetch(
            "SELECT body::text AS body FROM store_messages "
            "WHERE scope = $1 AND conversation = $2 AND position >= $3 "
            "AND ($4::int IS NULL OR position < $4) ORDER BY position",
            scope,
            conversation,
            start,
            stop,
        )
        return MESSAGES.validate_json("[" + ",".join(row["body"] for row in rows) + "]")

    async def append(
        self, scope: str, conversation: str, start: int, messages: Sequence[Message]
    ) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            length: int = await connection.fetchval(
                "SELECT count(*) FROM store_messages WHERE scope = $1 AND conversation = $2",
                scope,
                conversation,
            )
            if length == start + len(messages):
                rows = await connection.fetch(
                    "SELECT body::text AS body FROM store_messages "
                    "WHERE scope = $1 AND conversation = $2 AND position >= $3 ORDER BY position",
                    scope,
                    conversation,
                    start,
                )
                landed = "[" + ",".join(row["body"] for row in rows) + "]"
                if MESSAGES.validate_json(landed) == list(messages):
                    return
            if length != start:
                raise ValueError(
                    f"store: conversation {conversation!r} holds {length} messages, "
                    f"so an append at {start} is stale"
                )
            await connection.execute(
                "INSERT INTO store_messages (scope, conversation, position, body) "
                "SELECT $1, $2, $3 + given.n - 1, given.body::jsonb "
                "FROM unnest($4::text[]) WITH ORDINALITY AS given(body, n)",
                scope,
                conversation,
                start,
                [message.model_dump_json() for message in messages],
            )

    async def window(self, scope: str, conversation: str, length: int) -> int:
        return self.windows(conversation, length)


async def held(store: PostgresStore, scope: str, conversation: str) -> list[Message]:
    return list(MESSAGES.validate_python(await store.read(scope, conversation, 0)))
