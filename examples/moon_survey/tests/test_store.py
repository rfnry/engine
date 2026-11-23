import asyncio
import contextlib
import os
import secrets

import asyncpg
import pytest
from rfnry_engine import ConversationStore, Message, Notice, ToolCall, Usage

import setup
from store import Store, StoreError


@pytest.fixture
def scope():
    return f"t{secrets.token_hex(4)}"


def said(text):
    return Message(role="user", content=text)


def told(job):
    return Message(
        role="user",
        content=f"[helper {job} · surveyor · finished]",
        notice=Notice(job=job, agent="surveyor", run_id="r", kind="finished"),
    )


def reply(*calls, usage=None):
    return Message(
        role="assistant",
        tool_calls=[ToolCall(id=call, name="Shell", args={"command": "ls"}) for call in calls],
        usage=usage,
    )


def result(call):
    return Message(role="tool", tool_call_id=call, content="ok")


def prompt(tokens, output=10):
    return Usage(input=tokens, output=output)


class Refusing:
    def __init__(self, pool, *faults):
        self.pool, self.faults, self.asked = pool, list(faults), 0

    @contextlib.asynccontextmanager
    async def acquire(self):
        self.asked += 1
        if self.faults:
            raise self.faults.pop(0)
        async with self.pool.acquire() as conn:
            yield conn


class Dropping:
    def __init__(self, pool, fault, *, on=None, after=False, closed=True):
        self.pool, self.fault, self.on = pool, fault, on
        self.after, self.closed = after, closed

    @contextlib.asynccontextmanager
    async def acquire(self):
        async with self.pool.acquire() as conn:
            yield Dropped(conn, self)


class Dropped:
    def __init__(self, conn, owner):
        self.conn, self.owner = conn, owner

    def is_closed(self):
        return self.owner.closed

    def __getattr__(self, name):
        method = getattr(self.conn, name)

        async def called(*args):
            if self.owner.on not in (None, name):
                return await method(*args)
            fault, self.owner.fault = self.owner.fault, None
            if fault is None:
                return await method(*args)
            if self.owner.after:
                await method(*args)
            raise fault

        return called


class Unreleased:
    def __init__(self, pool, fault):
        self.pool, self.fault = pool, fault

    @contextlib.asynccontextmanager
    async def acquire(self):
        async with self.pool.acquire() as conn:
            yield conn
        fault, self.fault = self.fault, None
        if fault is not None:
            raise fault


def sleeper():
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    return slept, sleep


async def test_append_then_read_round_trips_every_field(pool, scope):
    store = Store(pool)
    assert isinstance(store, ConversationStore)
    messages = [
        said("survey N00-15/18"),
        Message(
            role="assistant",
            content="Reading the region first.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="region",
                    args={"region": "N00-15/18", "deep": {"list": [1, 2.5, None, True]}},
                )
            ],
            reasoning=[{"type": "thinking", "thinking": "…", "signature": "s"}],
            usage=Usage(input=120, output=40, cache_creation=30, cache_read=900),
            model="claude-haiku-5-5",
            prefix="0123456789abcdef",
            settings={"effort": "low", "max_tokens": 4096, "temperature": 0.7},
        ),
        Message(role="tool", tool_call_id="call_1", content="no such region", is_error=True),
        told("d1"),
    ]
    await store.append(scope, "c1", 0, messages)
    back = await store.read(scope, "c1", 0)
    assert back == messages
    assert back[1].reasoning == [{"type": "thinking", "thinking": "…", "signature": "s"}]
    assert back[1].tool_calls == messages[1].tool_calls
    assert back[1].usage == Usage(input=120, output=40, cache_creation=30, cache_read=900)
    assert (back[1].model, back[1].prefix) == ("claude-haiku-5-5", "0123456789abcdef")
    assert back[1].settings == {"effort": "low", "max_tokens": 4096, "temperature": 0.7}
    assert back[2].is_error is True
    assert back[3].notice == Notice(job="d1", agent="surveyor", run_id="r", kind="finished")
    assert await store.read(scope, "c1", 1, 3) == messages[1:3]


async def test_a_result_holding_a_nul_character_round_trips(pool, scope):
    store = Store(pool)
    messages = [said("dump it"), reply("call_1", usage=prompt(10)), result("call_1")]
    messages[2] = messages[2].model_copy(update={"content": "ELF\x00\x01 ퟿"})
    await store.append(scope, "c1", 0, messages)
    assert await store.read(scope, "c1", 0) == messages


async def test_length_counts_what_was_appended(pool, scope):
    store = Store(pool)
    assert await store.length(scope, "c1") == 0
    await store.append(scope, "c1", 0, [said("a"), reply()])
    await store.append(scope, "c1", 2, [said("b")])
    assert await store.length(scope, "c1") == 3
    assert await store.length(scope, "c1/d1") == 0


async def test_a_scope_sees_only_its_own_conversations(pool, scope):
    store = Store(pool, size=100)
    other = f"{scope}-other"
    await store.append(scope, "c1", 0, [said("a"), reply("x", usage=prompt(500)), result("x")])
    await store.append(scope, "c1", 3, [said("b")])
    await store.append(other, "c1", 0, [said("z")])
    assert await store.read(other, "c1", 0) == [said("z")]
    assert await store.window(scope, "c1", 4) == 3
    assert (await store.moves(scope, "c1"), await store.moves(other, "c1")) == (1, 0)
    assert await store.window(other, "c1", 1) == 0


async def test_a_stale_start_is_refused(pool, scope):
    store = Store(pool)
    await store.append(scope, "c1", 0, [said("a"), reply()])
    with pytest.raises(StoreError) as caught:
        await store.append(scope, "c1", 1, [said("b")])
    assert str(caught.value) == (
        "store: conversation 'c1' already holds a message at position 1; this writer is stale"
    )
    assert await store.read(scope, "c1", 0) == [said("a"), reply()]


async def test_a_writer_whose_messages_only_begin_with_what_is_held_is_stale(pool, scope):
    store = Store(pool)
    await store.append(scope, "c1", 0, [said("a"), reply()])
    with pytest.raises(StoreError) as caught:
        await store.append(scope, "c1", 0, [said("a"), reply(), said("b")])
    assert str(caught.value) == (
        "store: conversation 'c1' already holds a message at position 0; this writer is stale"
    )
    assert await store.length(scope, "c1") == 2


async def test_an_append_past_the_length_is_refused(pool, scope):
    store = Store(pool)
    await store.append(scope, "c1", 0, [said("a")])
    with pytest.raises(StoreError) as caught:
        await store.append(scope, "c1", 3, [said("b")])
    assert str(caught.value) == (
        "store: conversation 'c1' holds fewer than 3 messages; an append at position 3 leaves a gap"
    )
    assert await store.length(scope, "c1") == 1


async def test_a_retry_whose_messages_already_sit_at_start_returns(pool, scope):
    store = Store(pool)
    messages = [said("a"), reply("x", usage=prompt(5)), result("x")]
    await store.append(scope, "c1", 0, messages[:2])
    await store.append(scope, "c1", 0, messages[:2])
    assert await store.length(scope, "c1") == 2
    await store.append(scope, "c1", 2, messages[2:])
    await store.append(scope, "c1", 2, messages[2:])
    assert await store.read(scope, "c1", 0) == messages


async def test_two_writers_with_the_same_start_cannot_both_land(pool, scope):
    store = Store(pool)
    first = [said("first"), reply(), said("first again")]
    second = [said("second"), reply(), said("second again")]
    outcomes = await asyncio.gather(
        store.append(scope, "c1", 0, first),
        store.append(scope, "c1", 0, second),
        return_exceptions=True,
    )
    refused = [outcome for outcome in outcomes if isinstance(outcome, StoreError)]
    assert len(refused) == 1 and outcomes.count(None) == 1
    landed = first if outcomes[0] is None else second
    assert await store.read(scope, "c1", 0) == landed


async def test_a_brief_error_is_retried(pool, scope):
    slept, sleep = sleeper()
    flaky = Refusing(pool, asyncpg.ConnectionDoesNotExistError("connection was closed"))
    store = Store(flaky, sleep=sleep)
    await store.append(scope, "c1", 0, [said("a")])
    assert await store.length(scope, "c1") == 1
    assert (flaky.asked, slept) == (3, [0.05])

    slept.clear()
    down = Refusing(pool, *[asyncpg.ConnectionDoesNotExistError("connection was closed")] * 9)
    with pytest.raises(StoreError) as caught:
        await Store(down, sleep=sleep).append(scope, "c1", 1, [said("b")])
    assert str(caught.value) == (
        "store: the database failed 4 times in a row; "
        "last: ConnectionDoesNotExistError: connection was closed"
    )
    assert (down.asked, slept) == (4, [0.05, 0.1, 0.2])
    assert await store.length(scope, "c1") == 1


async def test_the_backoff_doubles_up_to_a_second(pool, scope):
    slept, sleep = sleeper()
    down = Refusing(pool, *[ConnectionRefusedError("refused")] * 9)
    with pytest.raises(StoreError):
        await Store(down, attempts=8, sleep=sleep).length(scope, "c1")
    assert slept == [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0]


@pytest.mark.parametrize(
    "fault",
    [
        asyncpg.ConnectionDoesNotExistError("connection was closed"),
        asyncpg.ConnectionFailureError("connection failure"),
        asyncpg.AdminShutdownError("terminating connection due to administrator command"),
        asyncpg.CannotConnectNowError("the database system is starting up"),
        asyncpg.SerializationError("could not serialize access"),
        asyncpg.DeadlockDetectedError("deadlock detected"),
        asyncpg.InterfaceError("connection is closed"),
        ConnectionResetError("reset"),
        OSError("network is unreachable"),
    ],
    ids=lambda fault: type(fault).__name__,
)
async def test_every_brief_error_is_retried(pool, scope, fault):
    flakies = [Dropping(pool, fault)]
    if not isinstance(fault, asyncpg.InterfaceError):
        flakies.append(Refusing(pool, fault))
    for flaky in flakies:
        slept, sleep = sleeper()
        store = Store(flaky, sleep=sleep)
        conversation = f"c-{id(flaky)}"
        await store.append(scope, conversation, 0, [said("a")])
        assert await store.read(scope, conversation, 0) == [said("a")]
        assert slept == [0.05]


async def test_an_interface_error_on_a_live_connection_or_a_closed_pool_is_not_retried(pool, scope):
    misused = Dropping(
        pool, asyncpg.InterfaceError("another operation is in progress"), closed=False
    )
    closed = Refusing(pool, asyncpg.InterfaceError("pool is closing"))
    for flaky in (misused, closed):
        slept, sleep = sleeper()
        with pytest.raises(asyncpg.InterfaceError):
            await Store(flaky, sleep=sleep).append(scope, "c1", 0, [said("a")])
        assert slept == []
    assert await Store(pool).length(scope, "c1") == 0


@pytest.mark.parametrize(
    "fault",
    [
        asyncpg.UndefinedTableError('relation "messages" does not exist'),
        asyncpg.ProtocolViolationError("unexpected message type"),
        asyncpg.ConnectionRejectionError("the server rejected the connection"),
        asyncpg.PostgresConnectionError("connection exception"),
    ],
    ids=lambda fault: type(fault).__name__,
)
async def test_an_error_that_is_not_brief_is_not_retried(pool, scope, fault):
    slept, sleep = sleeper()
    with pytest.raises(type(fault)):
        await Store(Dropping(pool, fault), sleep=sleep).length(scope, "c1")
    with pytest.raises(type(fault)):
        await Store(Refusing(pool, fault), sleep=sleep).length(scope, "c1")
    assert slept == []


async def test_a_commit_whose_answer_was_lost_is_found_not_written_twice(pool, scope):
    slept, sleep = sleeper()
    lost = Dropping(pool, asyncpg.ConnectionDoesNotExistError("connection was closed"), after=True)
    store = Store(lost, sleep=sleep)
    messages = [said("a"), reply("x", usage=prompt(7)), result("x")]
    await store.append(scope, "c1", 0, messages)
    assert await store.read(scope, "c1", 0) == messages
    assert slept == [0.05]


async def test_a_commit_whose_release_failed_is_kept_once(pool, scope):
    slept, sleep = sleeper()
    unreleased = Unreleased(pool, asyncpg.ConnectionDoesNotExistError("connection was closed"))
    store = Store(unreleased, sleep=sleep)
    messages = [said("a"), reply("x", usage=prompt(8)), result("x")]
    await store.append(scope, "c1", 0, messages)
    assert await store.read(scope, "c1", 0) == messages
    assert slept == [0.05]


async def test_a_repeat_whose_check_was_cut_is_not_written_again(pool, scope):
    messages = [said("a"), reply("x", usage=prompt(9)), result("x")]
    await Store(pool).append(scope, "c1", 0, messages)
    slept, sleep = sleeper()
    cut = Dropping(pool, asyncpg.ConnectionDoesNotExistError("connection was closed"), on="fetch")
    await Store(cut, sleep=sleep).append(scope, "c1", 0, messages)
    assert await Store(pool).read(scope, "c1", 0) == messages
    assert slept == [0.05]


async def test_a_lost_answer_is_found_even_when_json_changes_a_value(pool, scope):
    shaped = Message(
        role="assistant",
        reasoning=({"type": "thinking", "thinking": "t", "signature": "s"},),
        settings={"budget": 1e16},
        usage=prompt(3),
    )
    slept, sleep = sleeper()
    lost = Dropping(pool, asyncpg.ConnectionDoesNotExistError("connection was closed"), after=True)
    await Store(lost, sleep=sleep).append(scope, "c1", 0, [said("a"), shaped])
    back = await Store(pool).read(scope, "c1", 0)
    assert back[1].reasoning == [{"type": "thinking", "thinking": "t", "signature": "s"}]
    assert back[1].model_dump_json() == shaped.model_dump_json()
    assert slept == [0.05]


async def test_the_window_holds_until_the_size_then_jumps_to_a_person_message(pool, scope):
    store = Store(pool, size=1000)
    steps = [
        [said("survey the north")],
        [reply("a", usage=prompt(100)), result("a")],
        [told("d1")],
        [reply("b", usage=Usage(input=50, cache_read=300, cache_creation=100)), result("b")],
        [
            reply(
                "c", "d", usage=Usage(input=100, output=10_000, cache_read=850, cache_creation=50)
            ),
            result("c"),
            result("d"),
        ],
        [reply("h", usage=prompt(200)), result("h")],
        [said("from the lead: hurry")],
    ]
    length = 0
    for step in steps:
        await store.append(scope, "c1", length, step)
        length += len(step)
        assert await store.window(scope, "c1", length) == 0
    assert (length, await store.moves(scope, "c1")) == (12, 0)

    await store.append(scope, "c1", 12, [reply("e", usage=prompt(1001)), result("e")])
    assert await store.window(scope, "c1", 14) == 0
    await store.append(scope, "c1", 14, [said("from the lead: next")])
    assert await store.window(scope, "c1", 15) == 11
    assert await store.moves(scope, "c1") == 1

    later = [
        [reply("f", usage=prompt(300)), result("f")],
        [told("d2")],
        [reply("g", usage=Usage(input=100, cache_read=800, cache_creation=100)), result("g")],
    ]
    length = 15
    for step in later:
        await store.append(scope, "c1", length, step)
        length += len(step)
        assert await store.window(scope, "c1", length) == 11
    assert await store.moves(scope, "c1") == 1


async def test_the_window_measures_only_replies_sent_since_it_moved(pool, scope):
    store = Store(pool, size=1000)
    held = [
        said("survey"),
        reply("x", usage=prompt(100)),
        result("x"),
        told("d1"),
        reply("y", usage=prompt(100)),
        result("y"),
        told("d2"),
        told("d3"),
        told("d4"),
        told("d5"),
        reply("z", usage=prompt(2000)),
        result("z"),
        told("d6"),
    ]
    await store.append(scope, "c1", 0, held)
    assert await store.window(scope, "c1", 13) == 8
    assert await store.window(scope, "c1", 13) == 8
    await store.append(
        scope,
        "c1",
        13,
        [Message(role="assistant", content="cut off"), said("the process restarted; carry on")],
    )
    assert await store.window(scope, "c1", 15) == 8
    assert await store.moves(scope, "c1") == 1

    await store.append(scope, "c1", 15, [reply("w", usage=prompt(1500)), result("w")])
    assert await store.window(scope, "c1", 17) == 8
    await store.append(scope, "c1", 17, [told("d7")])
    assert await store.window(scope, "c1", 18) == 14
    assert await store.moves(scope, "c1") == 2


async def test_the_window_holds_through_a_tool_loop_and_moves_at_the_next_person_message(
    pool, scope
):
    store = Store(pool, size=1000)
    people = [said("survey"), *(told(f"d{n}") for n in range(1, 6))]
    await store.append(scope, "c1", 0, [*people, reply("x", usage=prompt(2000)), result("x")])
    assert await store.window(scope, "c1", 8) == 0
    await store.append(scope, "c1", 8, [reply("y", usage=prompt(2100)), result("y")])
    assert await store.window(scope, "c1", 10) == 0
    assert await store.moves(scope, "c1") == 0

    await store.append(scope, "c1", 10, [told("d6")])
    assert await store.window(scope, "c1", 11) == 10
    assert await store.moves(scope, "c1") == 1


async def test_the_window_never_moves_while_the_last_message_is_a_reply(pool, scope):
    store = Store(pool, size=1000)
    await store.append(
        scope, "c1", 0, [said("survey"), told("d1"), Message(role="assistant", content="waiting")]
    )
    await store.append(scope, "c1", 3, [said("go on")])
    await store.append(scope, "c1", 4, [reply("x", usage=prompt(2000))])
    assert await store.window(scope, "c1", 5) == 0
    await store.append(scope, "c1", 5, [result("x"), told("d2")])
    assert await store.window(scope, "c1", 7) == 6


async def test_the_window_measures_the_latest_reply_that_carries_usage(pool, scope):
    store = Store(pool, size=1000)
    held = [
        said("survey"),
        reply("x", usage=prompt(2000)),
        result("x"),
        Message(role="assistant", content="cut off"),
        said("carry on"),
    ]
    await store.append(scope, "c1", 0, held)
    assert await store.window(scope, "c1", 5) == 4


async def test_the_window_reads_past_a_nul_in_a_result(pool, scope):
    store = Store(pool, size=1000)
    nul = Message(role="tool", tool_call_id="x", content="ELF\x00\x01")
    await store.append(scope, "c1", 0, [said("dump it"), reply("x", usage=prompt(100)), nul])
    await store.append(scope, "c1", 3, [said("again")])
    assert await store.window(scope, "c1", 4) == 0

    later = nul.model_copy(update={"tool_call_id": "y"})
    await store.append(scope, "c1", 4, [reply("y", usage=prompt(2000)), later, told("d1")])
    assert await store.window(scope, "c1", 7) == 6
    assert await store.moves(scope, "c1") == 1
    assert (await store.read(scope, "c1", 0))[2].content == "ELF\x00\x01"


async def test_the_window_never_splits_a_call_from_its_result(pool, scope):
    store = Store(pool, size=1000)
    held = [
        said("survey"),
        reply("x", usage=prompt(100)),
        result("x"),
        reply("y", usage=prompt(2000)),
        said("inside the batch"),
        result("y"),
        told("d1"),
    ]
    await store.append(scope, "c1", 0, held)
    assert await store.window(scope, "c1", 7) == 6


async def test_the_window_holds_without_a_person_message_to_start_at(pool, scope):
    store = Store(pool, size=1000)
    held = [said("survey"), reply("x", "y", "z", usage=prompt(5000)), *map(result, "xyz")]
    await store.append(scope, "c1", 0, held)
    assert await store.window(scope, "c1", 5) == 0
    assert await store.moves(scope, "c1") == 0


async def test_the_tables_are_made_once_in_their_schema(pool):
    fresh = f"moon_test_{secrets.token_hex(6)}"
    async with pool.acquire() as conn:
        try:
            await setup.create_tables(conn, fresh)
            await setup.create_tables(conn, fresh)
            names = await conn.fetchval(
                "select array_agg(table_name::text order by table_name) "
                "from information_schema.tables where table_schema = $1",
                fresh,
            )
            assert {"messages", "windows"} <= set(names)
            assert await conn.fetchval("select current_schema()") != fresh
        finally:
            await conn.execute(f"drop schema if exists {fresh} cascade")


async def test_the_tables_give_an_older_surveys_table_its_error_column_once(pool):
    fresh = f"moon_test_{secrets.token_hex(6)}"
    async with pool.acquire() as conn:
        try:
            await conn.execute(f"create schema {fresh}")
            await conn.execute(
                f"create table {fresh}.surveys (scope text not null, survey text not null,"
                " cap double precision not null, second text not null, regions text not null,"
                " mode text not null, shape text not null, crews integer not null,"
                " status text not null, primary key (scope, survey))"
            )
            await setup.create_tables(conn, fresh)
            await setup.create_tables(conn, fresh)
            columns = await conn.fetch(
                "select column_name::text as name, data_type::text as type"
                " from information_schema.columns"
                " where table_schema = $1 and table_name = 'surveys' and column_name = 'error'",
                fresh,
            )
            assert [(c["name"], c["type"]) for c in columns] == [("error", "text")]
        finally:
            await conn.execute(f"drop schema if exists {fresh} cascade")


def test_the_tables_command_needs_the_examples_url(monkeypatch, capsys):
    monkeypatch.delenv("RFNRY_EXAMPLES_POSTGRES_URL", raising=False)
    assert setup.main(["tables"]) == 1
    assert capsys.readouterr().err.strip() == "setup: RFNRY_EXAMPLES_POSTGRES_URL is not set"


async def test_the_tables_command_makes_the_tables(pool, monkeypatch, capsys):
    fresh = f"moon_test_{secrets.token_hex(6)}"
    made = setup.create_tables
    monkeypatch.setattr(setup, "create_tables", lambda conn: made(conn, fresh))
    monkeypatch.setenv("RFNRY_EXAMPLES_POSTGRES_URL", os.environ["RFNRY_TEST_POSTGRES_URL"])
    try:
        assert await asyncio.to_thread(setup.main, ["tables"]) == 0
        assert capsys.readouterr().out.strip() == "tables ok"
        assert await pool.fetchval("select to_regclass($1)::text", f"{fresh}.windows")
    finally:
        await pool.execute(f"drop schema if exists {fresh} cascade")
