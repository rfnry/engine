import json
import os
from datetime import UTC, datetime
from email.utils import format_datetime
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from rfnry_engine import (
    Check,
    CheckAnswer,
    Message,
    Notice,
    Reply,
    ToolCall,
    ToolSpec,
)

import providers
from providers import (
    Gemini,
    Haiku,
    Limiter,
    ProviderError,
    ProviderLog,
    Retry,
    TypeSafe,
)

REQUEST = httpx.Request("POST", "https://example.invalid/v1/messages")
SPEC = ToolSpec(name="fetch", description="fetch a url", input_schema={"type": "object"})


class Time:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Logged:
    def __init__(self) -> None:
        self.entries: list[tuple] = []

    async def __call__(self, provider, model, status, wait, headers) -> None:
        self.entries.append((provider, model, status, wait, headers))


def retry(time: Time, logged: Logged, **kw) -> Retry:
    return Retry(
        kw.pop("attempts", 50),
        kw.pop("deadline", 1800.0),
        time.sleep,
        logged,
        clock=time.clock,
        wall=lambda: 1000.0,
        rng=lambda: 0.0,
        **kw,
    )


def status_error(status: int, headers: dict | None = None) -> anthropic.APIStatusError:
    response = httpx.Response(status, headers=headers or {}, request=REQUEST)
    error = anthropic.APIStatusError("stub", response=response, body=None)
    return error


def answer(content, stop="end_turn", usage=None):
    return anthropic.types.Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-haiku-5-5",
        content=content,
        stop_reason=stop,
        stop_sequence=None,
        usage=usage or anthropic.types.Usage(input_tokens=10, output_tokens=5),
    )


def text_block(text):
    return anthropic.types.TextBlock(type="text", text=text)


class Messages:
    def __init__(self, script: list) -> None:
        self.script, self.calls = list(script), []

    async def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def haiku(script, time=None, logged=None, effort="medium", **kw):
    time, logged = time or Time(), logged or Logged()
    client = SimpleNamespace(messages=Messages(script))
    return Haiku(client, "claude-haiku-5-5", effort, retry(time, logged, **kw)), client.messages


async def test_haiku_sends_the_effort_the_folder_names_and_caches_its_prefix():
    model, sent = haiku(
        [
            answer(
                [text_block("done")],
                usage=anthropic.types.Usage(
                    input_tokens=7,
                    output_tokens=3,
                    cache_creation_input_tokens=100,
                    cache_read_input_tokens=200,
                ),
            )
        ],
        effort="high",
    )
    reply = await model.generate("be brief", [Message(role="user", content="hi")], [SPEC])
    call = sent.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] == 16000
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["system"] == "be brief"
    assert call["tools"] == [
        {"name": "fetch", "description": "fetch a url", "input_schema": {"type": "object"}}
    ]
    assert call["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert reply.text == "done"
    assert reply.model == "claude-haiku-5-5"
    assert (reply.usage.input, reply.usage.output) == (7, 3)
    assert (reply.usage.cache_creation, reply.usage.cache_read) == (100, 200)
    assert reply.settings == {"max_tokens": 16000, "thinking": "adaptive", "effort": "high"}
    assert reply.stop_reason == "end_turn"


async def test_haiku_leaves_effort_out_when_the_folder_names_none():
    model, sent = haiku([answer([text_block("x")])], effort=None)
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert "output_config" not in sent.calls[0]
    assert "tools" not in sent.calls[0]


async def test_haiku_keeps_its_thinking_as_reasoning():
    thinking = anthropic.types.ThinkingBlock(type="thinking", thinking="hmm", signature="sig")
    use = anthropic.types.ToolUseBlock(type="tool_use", id="tu_1", name="fetch", input={"u": "x"})
    model, sent = haiku(
        [
            answer([thinking, text_block("looking"), use], stop="tool_use"),
            answer([text_block("ok")]),
        ]
    )
    first = await model.generate("s", [Message(role="user", content="go")], [SPEC])
    assert first.stop_reason == "tool_use"
    assert first.tool_calls == [ToolCall(id="tu_1", name="fetch", args={"u": "x"})]
    assert first.reasoning == [{"type": "thinking", "thinking": "hmm", "signature": "sig"}]
    json.dumps(first.reasoning)
    history = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            content=first.text,
            tool_calls=first.tool_calls,
            reasoning=first.reasoning,
        ),
        Message(role="tool", content="boom", tool_call_id="tu_1", is_error=True),
        Message(
            role="user",
            content="a helper finished",
            notice=Notice(job="j", agent="a", run_id="r", kind="finished"),
        ),
    ]
    await model.generate("s", history, [SPEC])
    turns = sent.calls[1]["messages"]
    assert turns[1]["content"] == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "looking"},
        {"type": "tool_use", "id": "tu_1", "name": "fetch", "input": {"u": "x"}},
    ]
    assert len(turns) == 3
    assert turns[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "boom", "is_error": True},
        {"type": "text", "text": "a helper finished"},
    ]


async def test_a_reply_without_thinking_has_no_reasoning():
    model, _ = haiku([answer([text_block("x")])])
    reply = await model.generate("s", [Message(role="user", content="hi")], [])
    assert reply.reasoning is None


async def test_a_429_waits_what_retry_after_names_and_is_logged():
    time, logged = Time(), Logged()
    headers = {
        "retry-after": "7",
        "anthropic-ratelimit-requests-remaining": "0",
        "request-id": "req_1",
        "x-api-key": "sk-secret",
        "content-type": "application/json",
    }
    model, sent = haiku([status_error(429, headers), answer([text_block("ok")])], time, logged)
    reply = await model.generate("s", [Message(role="user", content="hi")], [])
    assert reply.text == "ok"
    assert time.slept == [7.0]
    assert len(sent.calls) == 2
    [(provider, name, status, wait, kept)] = logged.entries
    assert (provider, name, status, wait) == ("anthropic", "claude-haiku-5-5", 429, 7.0)
    assert kept == {
        "retry-after": "7",
        "anthropic-ratelimit-requests-remaining": "0",
        "request-id": "req_1",
    }
    assert "sk-secret" not in json.dumps(logged.entries)


async def test_retry_after_may_be_an_http_date():
    time, logged = Time(), Logged()
    when = format_datetime(datetime.fromtimestamp(1042, UTC), usegmt=True)
    model, _ = haiku(
        [status_error(429, {"retry-after": when}), answer([text_block("ok")])], time, logged
    )
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert time.slept == [42.0]


async def test_without_retry_after_the_wait_backs_off():
    time, logged = Time(), Logged()
    model, _ = haiku(
        [status_error(429), status_error(429), status_error(429), answer([text_block("ok")])],
        time,
        logged,
    )
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert time.slept == [1.0, 2.0, 4.0]


async def test_jitter_is_added_to_the_wait():
    time, logged = Time(), Logged()
    client = SimpleNamespace(
        messages=Messages([status_error(429, {"retry-after": "10"}), answer([text_block("ok")])])
    )
    spread = Retry(50, 1800.0, time.sleep, logged, clock=time.clock, rng=lambda: 0.5)
    await Haiku(client, "claude-haiku-5-5", None, spread).generate(
        "s", [Message(role="user", content="hi")], []
    )
    assert time.slept == [10.0 + 0.5 * (1 + 0.5 * 10.0)]


@pytest.mark.parametrize("status", [529, 500, 502, 503])
async def test_a_529_and_a_5xx_are_retried(status):
    time, logged = Time(), Logged()
    model, sent = haiku([status_error(status), answer([text_block("ok")])], time, logged)
    reply = await model.generate("s", [Message(role="user", content="hi")], [])
    assert reply.text == "ok"
    assert len(sent.calls) == 2
    assert logged.entries[0][2] == status


async def test_a_connection_error_is_retried():
    time, logged = Time(), Logged()
    cut = anthropic.APIConnectionError(request=REQUEST)
    model, sent = haiku([cut, answer([text_block("ok")])], time, logged)
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert len(sent.calls) == 2
    assert logged.entries[0][2] is None


async def test_a_400_is_not_retried():
    model, sent = haiku([status_error(400), answer([text_block("ok")])])
    with pytest.raises(anthropic.APIStatusError):
        await model.generate("s", [Message(role="user", content="hi")], [])
    assert len(sent.calls) == 1


async def test_retries_end_at_the_deadline():
    time, logged = Time(), Logged()
    last = status_error(429, {"retry-after": "100"})
    model, sent = haiku(
        [
            status_error(429, {"retry-after": "100"}),
            status_error(429, {"retry-after": "100"}),
            status_error(429, {"retry-after": "100"}),
            last,
        ],
        time,
        logged,
        deadline=250.0,
    )
    with pytest.raises(anthropic.APIStatusError) as raised:
        await model.generate("s", [Message(role="user", content="hi")], [])
    assert raised.value is last
    assert time.slept == [100.0, 100.0, 50.0]
    assert len(sent.calls) == 4


async def test_retries_end_at_the_attempts():
    model, sent = haiku([status_error(500), status_error(500)], attempts=2)
    with pytest.raises(anthropic.APIStatusError):
        await model.generate("s", [Message(role="user", content="hi")], [])
    assert len(sent.calls) == 2


async def test_a_refusal_fails_the_call():
    model, _ = haiku([answer([], stop="refusal")])
    with pytest.raises(ProviderError, match="providers: claude-haiku-5-5 refused the request"):
        await model.generate("s", [Message(role="user", content="hi")], [])


def test_a_missing_key_names_the_variable_and_nothing_else(monkeypatch):
    for name, build in [
        ("ANTHROPIC_API_KEY", providers.anthropic),
        ("GEMINI_API_KEY", providers.gemini),
        ("TYPESAFE_API_KEY", providers.typesafe),
    ]:
        monkeypatch.delenv(name, raising=False)
        with pytest.raises(ProviderError) as raised:
            build(None, None)
        assert str(raised.value) == f"providers: {name} is not set"


def test_keys_are_read_when_a_model_is_built_and_each_is_built_once(monkeypatch):
    providers.built.clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-1")
    monkeypatch.setenv("GEMINI_API_KEY", "k-2")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k-3")
    first = providers.anthropic("claude-haiku-5-5", "high")
    assert providers.anthropic("claude-haiku-5-5", "high") is first
    assert providers.anthropic("claude-haiku-5-5", "low") is not first
    assert providers.gemini(None, None) is providers.gemini(None, None)
    assert providers.typesafe("jev-latest", None) is providers.typesafe("jev-latest", None)
    assert first.model == "claude-haiku-5-5"
    providers.built.clear()


class Completions:
    def __init__(self, script: list) -> None:
        self.script, self.calls = list(script), []

    async def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def completion(content="", calls=None, finish="stop", prompt=100, cached=None, out=9):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return SimpleNamespace(
        model="models/other",
        choices=[
            SimpleNamespace(
                finish_reason=finish,
                message=SimpleNamespace(content=content, tool_calls=calls),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=out, prompt_tokens_details=details
        ),
    )


async def test_gemini_maps_a_completion():
    call = SimpleNamespace(
        type="function", id="", function=SimpleNamespace(name="fetch", arguments='{"u": "x"}')
    )
    chat = Completions([completion("", [call], "tool_calls", prompt=100, cached=30)])
    time = Time()
    model = Gemini(
        SimpleNamespace(chat=SimpleNamespace(completions=chat)),
        "gemini-3.5-flash-lite",
        "low",
        retry(time, Logged()),
        Limiter(15, time.clock, time.sleep),
    )
    history = [
        Message(role="user", content="go"),
        Message(role="assistant", tool_calls=[ToolCall(id="c0", name="fetch", args={})]),
        Message(role="tool", content="body", tool_call_id="c0"),
    ]
    reply = await model.generate("sys", history, [SPEC])
    assert reply.model == "gemini-3.5-flash-lite"
    assert reply.stop_reason == "tool_use"
    assert reply.tool_calls == [ToolCall(id="call_0", name="fetch", args={"u": "x"})]
    assert (reply.usage.input, reply.usage.cache_read, reply.usage.output) == (70, 30, 9)
    sent = chat.calls[0]
    assert sent["reasoning_effort"] == "low"
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["messages"][3]["name"] == "fetch"


async def test_gemini_ignores_an_effort_the_api_does_not_name():
    chat = Completions([completion("hi")])
    time = Time()
    model = Gemini(
        SimpleNamespace(chat=SimpleNamespace(completions=chat)),
        "gemini-3.5-flash-lite",
        "max",
        retry(time, Logged()),
        Limiter(15, time.clock, time.sleep),
    )
    reply = await model.generate("s", [Message(role="user", content="hi")], [])
    assert "reasoning_effort" not in chat.calls[0]
    assert reply.usage.input == 100 and reply.usage.cache_read == 0


async def test_gemini_keeps_to_fifteen_a_minute():
    time = Time()
    limiter = Limiter(15, time.clock, time.sleep)
    stamps = []
    for _ in range(31):
        await limiter.wait()
        stamps.append(time.now)
    assert stamps[-1] - stamps[0] == pytest.approx(30 * 4.0)
    window = [t for t in stamps if t < stamps[0] + 60]
    assert len(window) <= 15
    time.now += 1000
    before = len(time.slept)
    await limiter.wait()
    assert len(time.slept) == before


async def test_gemini_retries_pass_the_limiter_each_time():
    time = Time()
    err = anthropic.APIStatusError("x", response=httpx.Response(429, request=REQUEST), body=None)
    chat = Completions([err, completion("ok")])
    model = Gemini(
        SimpleNamespace(chat=SimpleNamespace(completions=chat)),
        "gemini-3.5-flash-lite",
        None,
        retry(time, Logged()),
        Limiter(15, time.clock, time.sleep),
    )
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert len(chat.calls) == 2
    assert sum(time.slept) >= 4.0


class Posts:
    def __init__(self, script: list) -> None:
        self.script, self.calls = list(script), []

    async def post(self, url, json):
        self.calls.append((url, json))
        return self.script.pop(0)


def reply_of(status, body=None, headers=None):
    return httpx.Response(status, json=body, headers=headers or {}, request=REQUEST)


async def test_typesafe_maps_a_check_answer():
    client = Posts(
        [
            reply_of(503),
            reply_of(
                200,
                {
                    "answers": {"safe": {"noul": 0.8}},
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                    "model": "other",
                },
            ),
        ]
    )
    time = Time()
    model = TypeSafe(client, "jev-latest", retry(time, Logged()))
    decided = await model.decide("state", {"safe": Check(instructions="is it safe?")})
    assert decided.answers == {"safe": CheckAnswer(probability=0.8)}
    assert decided.model == "jev-latest"
    assert (decided.usage.input, decided.usage.output) == (12, 1)
    assert len(client.calls) == 2
    assert client.calls[0][1]["questions"]["safe"] == {
        "type": "noul",
        "instructions": "is it safe?",
    }
    assert time.slept == [1.0]


async def test_the_gate_gives_up_sooner_than_the_models():
    client = Posts([reply_of(503) for _ in range(500)])
    time = Time()
    model = TypeSafe(
        client,
        "jev-latest",
        Retry(
            1000, providers.GATE_DEADLINE, time.sleep, Logged(), clock=time.clock, rng=lambda: 0.0
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await model.decide("s", {"safe": Check(instructions="x")})
    assert sum(time.slept) <= providers.GATE_DEADLINE
    assert providers.GATE_DEADLINE < providers.DEADLINE


async def test_the_log_writes_a_row_a_without_a_pool_only_logs(pool, caplog):
    log = ProviderLog()
    await log("anthropic", "claude-haiku-5-5", 429, 1.0, {"retry-after": "1"})
    log.use(pool)
    await log("anthropic", "claude-haiku-5-5", 429, 7.5, {"retry-after": "7"})
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "select provider, model, status, wait, headers::text from provider_log"
        )
    assert [(r["provider"], r["status"], r["wait"]) for r in rows] == [("anthropic", 429, 7.5)]
    assert json.loads(rows[0]["headers"]) == {"retry-after": "7"}


@pytest.mark.live
async def test_the_live_providers_answer_once():
    from rfnry_engine import Decided

    ask = [Message(role="user", content="Reply with the single word: ready")]
    haiku_reply = await providers.anthropic("claude-haiku-5-5", "low").generate(
        "Be brief.", ask, []
    )
    assert isinstance(haiku_reply, Reply)
    assert haiku_reply.model == "claude-haiku-5-5" and haiku_reply.text
    assert haiku_reply.usage.output > 0
    gemini_reply = await providers.gemini(providers.GEMINI_MODEL, None).generate(
        "Be brief.", ask, []
    )
    assert gemini_reply.model == providers.GEMINI_MODEL and gemini_reply.text
    decided = await providers.typesafe("jev-latest", None).decide(
        "The sky is blue.", {"true": Check(instructions="Is the state about the sky?")}
    )
    assert isinstance(decided, Decided)
    assert 0.0 <= decided.answers["true"].probability <= 1.0
    assert os.environ.get("ANTHROPIC_API_KEY") not in {haiku_reply.text, gemini_reply.text}
