import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence

import asyncpg
from rfnry_engine import Message

BRIEF = (
    asyncpg.ConnectionDoesNotExistError,
    asyncpg.ConnectionFailureError,
    asyncpg.AdminShutdownError,
    asyncpg.CannotConnectNowError,
    asyncpg.SerializationError,
    asyncpg.DeadlockDetectedError,
    OSError,
)
SIZE = 80_000
FIRST_WAIT = 0.05
LAST_WAIT = 1.0

WRITE = """
insert into messages (scope, conversation, position, body, prompt)
select $1, $2, $3::integer + given.n::integer - 1, given.body::json, given.prompt
from unnest($4::text[], $5::integer[]) with ordinality as given (body, prompt, n)
where $3::integer = 0 or exists (
    select 1 from messages
    where scope = $1 and conversation = $2 and position = $3::integer - 1
)
"""
READ = """
select body::text from messages
where scope = $1 and conversation = $2 and position >= $3
    and ($4::integer is null or position < $4::integer)
order by position
"""
LAST_PROMPT = """
select prompt from messages
where scope = $1 and conversation = $2 and position >= $3 and position < $4
    and prompt is not null
order by position desc
limit 1
"""
MOVE = """
insert into windows (scope, conversation, start, moves, since) values ($1, $2, $3, 1, $4)
on conflict (scope, conversation) do update
set start = excluded.start, moves = windows.moves + 1, since = excluded.since
where windows.start < excluded.start
"""


class StoreError(Exception):
    pass


def _whole(messages: Sequence[Message]) -> bool:
    called: set[str] = set()
    for message in messages:
        if message.role == "tool" and message.tool_call_id not in called:
            return False
        called.update(call.id for call in message.tool_calls)
    return True


def _prompt(message: Message) -> int | None:
    usage = message.usage
    if message.role != "assistant" or usage is None:
        return None
    return usage.input + usage.cache_read + usage.cache_creation


def _held(rows: Sequence[asyncpg.Record]) -> list[Message]:
    return [Message.model_validate_json(row["body"]) for row in rows]


class Store:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        size: int = SIZE,
        keep: float = 1 / 3,
        attempts: int = 4,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        self.pool = pool
        self.size = size
        self.keep = keep
        self.attempts = attempts
        self.sleep = sleep

    async def length(self, scope: str, conversation: str) -> int:
        return await self._retried(
            lambda conn: conn.fetchval(
                "select count(*) from messages where scope = $1 and conversation = $2",
                scope,
                conversation,
            )
        )

    async def read(
        self, scope: str, conversation: str, start: int, stop: int | None = None
    ) -> list[Message]:
        return _held(
            await self._retried(lambda conn: conn.fetch(READ, scope, conversation, start, stop))
        )

    async def append(
        self, scope: str, conversation: str, start: int, messages: Sequence[Message]
    ) -> None:
        given = list(messages)
        if not given:
            return
        bodies = [message.model_dump_json() for message in given]
        prompts = [_prompt(message) for message in given]

        async def write(conn: asyncpg.Connection) -> None:
            try:
                status = await conn.execute(WRITE, scope, conversation, start, bodies, prompts)
            except asyncpg.UniqueViolationError:
                held = await conn.fetch(READ, scope, conversation, start, start + len(bodies))
                if [row["body"] for row in held] != bodies:
                    raise StoreError(
                        f"store: conversation {conversation!r} already holds a message at "
                        f"position {start}; this writer is stale"
                    ) from None
                return
            if status == "INSERT 0 0":
                raise StoreError(
                    f"store: conversation {conversation!r} holds fewer than {start} messages; "
                    f"an append at position {start} leaves a gap"
                )

        await self._retried(write)

    async def window(self, scope: str, conversation: str, length: int) -> int:
        async def find(conn: asyncpg.Connection) -> int:
            row = await conn.fetchrow(
                "select start, since from windows where scope = $1 and conversation = $2",
                scope,
                conversation,
            )
            start, since = (row["start"], row["since"]) if row else (0, 0)
            prompt = await conn.fetchval(
                LAST_PROMPT, scope, conversation, max(start, since), length
            )
            if prompt is None or prompt <= self.size:
                return start
            last = _held(await conn.fetch(READ, scope, conversation, length - 1, length))
            if not last or last[0].role != "user":
                return start
            first = max(start + 1, start + math.floor((length - start) * (1 - self.keep)))
            held = _held(await conn.fetch(READ, scope, conversation, first, length))
            moved = next(
                (
                    first + n
                    for n, message in enumerate(held)
                    if message.role == "user" and _whole(held[n:])
                ),
                None,
            )
            if moved is None:
                return start
            await conn.execute(MOVE, scope, conversation, moved, length)
            return moved

        return await self._retried(find)

    async def moves(self, scope: str, conversation: str) -> int:
        found = await self._retried(
            lambda conn: conn.fetchval(
                "select moves from windows where scope = $1 and conversation = $2",
                scope,
                conversation,
            )
        )
        return found or 0

    async def _retried[T](self, work: Callable[[asyncpg.Connection], Awaitable[T]]) -> T:
        wait = FIRST_WAIT
        attempt = 1
        while True:
            dropped = False
            try:
                async with self.pool.acquire() as conn:
                    try:
                        return await work(conn)
                    except asyncpg.InterfaceError:
                        dropped = conn.is_closed()
                        raise
            except Exception as error:
                if not (dropped or isinstance(error, BRIEF)):
                    raise
                if attempt >= self.attempts:
                    raise StoreError(
                        f"store: the database failed {attempt} times in a row; "
                        f"last: {type(error).__name__}: {error}"
                    ) from error
                await self.sleep(wait)
                wait = min(wait * 2, LAST_WAIT)
                attempt += 1
