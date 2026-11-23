import json
from datetime import UTC, datetime
from email.utils import format_datetime
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from rfnry_engine import (
    Check,
    CheckAnswer,
    Message,
    Notice,
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
    blocks = [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "looking"},
        {"type": "tool_use", "id": "tu_1", "name": "fetch", "input": {"u": "x"}},
    ]
    assert first.reasoning == blocks
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
    assert turns[1]["content"] == blocks
    assert len(turns) == 3
    assert turns[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "boom", "is_error": True},
        {"type": "text", "text": "a helper finished"},
    ]


async def test_an_assistant_turn_the_engine_stripped_is_built_without_thinking():
    model, sent = haiku([answer([text_block("ok")])])
    history = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            content="looking",
            tool_calls=[ToolCall(id="tu_1", name="fetch", args={"u": "x"})],
        ),
        Message(role="tool", content="body", tool_call_id="tu_1"),
    ]
    await model.generate("s", history, [SPEC])
    assert sent.calls[0]["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "tu_1", "name": "fetch", "input": {"u": "x"}},
        ],
    }


async def test_an_empty_assistant_turn_is_left_out():
    model, sent = haiku([answer([text_block("ok")])])
    history = [
        Message(role="user", content="a"),
        Message(role="assistant", content=""),
        Message(role="user", content="b"),
    ]
    await model.generate("s", history, [])
    assert sent.calls[0]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        }
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
    assert time.slept == [15.0]


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


async def test_a_400_is_not_retried_and_is_a_provider_error_with_its_cause():
    cause = status_error(400, {"x-api-key": "sk-secret"})
    model, sent = haiku([cause, answer([text_block("ok")])])
    with pytest.raises(ProviderError) as raised:
        await model.generate("s", [Message(role="user", content="hi")], [])
    assert str(raised.value).startswith("providers: claude-haiku-5-5 answered 400: ")
    assert raised.value.__cause__ is cause
    assert "sk-secret" not in str(raised.value)
    assert len(sent.calls) == 1


async def test_retries_end_at_the_deadline_and_the_last_429_is_logged():
    time, logged = Time(), Logged()
    last = status_error(429, {"retry-after": "100"})
    model, sent = haiku(
        [
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
    assert time.slept == [100.0, 100.0]
    assert len(sent.calls) == 3
    assert [entry[3] for entry in logged.entries] == [100.0, 100.0, 0.0]


async def test_successive_429s_with_the_same_retry_after_wait_longer_each_time():
    time, logged = Time(), Logged()
    script = [status_error(429, {"retry-after": "1"}) for _ in range(5)]
    model, _ = haiku([*script, answer([text_block("ok")])], time, logged)
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert time.slept == [1.0, 2.0, 4.0, 8.0, 16.0]
    slower = Time()
    script = [status_error(429, {"retry-after": "7"}) for _ in range(6)]
    model, _ = haiku([*script, answer([text_block("ok")])], slower, Logged())
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert slower.slept == sorted(slower.slept)
    assert slower.slept[0] == 7.0 and slower.slept[-1] > 7.0


async def test_a_retry_after_that_is_not_a_number_or_a_date_backs_off():
    time = Time()
    model, _ = haiku([status_error(429, {"retry-after": "soon"}), answer([text_block("ok")])], time)
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert time.slept == [1.0]


async def test_retry_after_ms_is_honoured():
    time = Time()
    model, _ = haiku(
        [status_error(429, {"retry-after-ms": "1500"}), answer([text_block("ok")])], time
    )
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert time.slept == [1.5]


@pytest.mark.parametrize("status", [408, 409])
async def test_a_408_and_a_409_are_retried(status):
    model, sent = haiku([status_error(status), answer([text_block("ok")])])
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert len(sent.calls) == 2


async def test_x_should_retry_decides_over_the_status():
    model, sent = haiku([status_error(400, {"x-should-retry": "true"}), answer([text_block("ok")])])
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert len(sent.calls) == 2
    model, sent = haiku(
        [status_error(500, {"x-should-retry": "false"}), answer([text_block("ok")])]
    )
    with pytest.raises(ProviderError, match="answered 500"):
        await model.generate("s", [Message(role="user", content="hi")], [])
    assert len(sent.calls) == 1


class Failing:
    async def __call__(self, *args) -> None:
        raise RuntimeError("the pool is gone")


async def test_a_log_that_cannot_write_neither_fails_nor_replaces_the_error():
    time = Time()
    client = SimpleNamespace(
        messages=Messages([status_error(429, {"retry-after": "3"}), answer([text_block("ok")])])
    )
    model = Haiku(client, "claude-haiku-5-5", None, retry(time, Failing()))
    reply = await model.generate("s", [Message(role="user", content="hi")], [])
    assert reply.text == "ok" and time.slept == [3.0]
    last = status_error(429, {"retry-after": "100"})
    client = SimpleNamespace(messages=Messages([last]))
    model = Haiku(client, "claude-haiku-5-5", None, retry(Time(), Failing(), deadline=50.0))
    with pytest.raises(anthropic.APIStatusError) as raised:
        await model.generate("s", [Message(role="user", content="hi")], [])
    assert raised.value is last


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
    monkeypatch.setattr(providers, "built", {})
    monkeypatch.setattr(providers, "limiters", {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-1")
    monkeypatch.setenv("GEMINI_API_KEY", "k-2")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k-3")
    first = providers.anthropic("claude-haiku-5-5", "high")
    assert providers.anthropic("claude-haiku-5-5", "high") is first
    assert providers.anthropic("claude-haiku-5-5", "low") is not first
    assert providers.gemini(None, None) is providers.gemini(None, None)
    assert providers.typesafe("jev-latest", "high") is providers.typesafe("jev-latest", None)
    assert providers.gemini(None, "low").limiter is providers.gemini(None, "high").limiter
    assert providers.gemini(None, "low").limiter is not providers.gemini("other", "low").limiter
    assert first.model == "claude-haiku-5-5"


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
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {"name": "fetch", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c0", "content": "body"},
    ]


SIGNATURE = {"google": {"thought_signature": "Eo4BCosBAWkUfROEHBhSjkAMW22p79"}}


def gemini_over(chat):
    time = Time()
    return Gemini(
        SimpleNamespace(chat=SimpleNamespace(completions=chat)),
        "gemini-3.5-flash-lite",
        None,
        retry(time, Logged()),
        Limiter(15, time.clock, time.sleep),
    )


async def test_gemini_keeps_the_signed_tool_calls_and_sends_them_back_unchanged():
    call = SimpleNamespace(
        type="function",
        id="call_1",
        function=SimpleNamespace(name="echo", arguments='{"text":"ready"}'),
        extra_content=SIGNATURE,
    )
    chat = Completions([completion(None, [call], "tool_calls"), completion("done")])
    model = gemini_over(chat)
    first = await model.generate("sys", [Message(role="user", content="go")], [SPEC])
    assert first.reasoning == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"text":"ready"}'},
            "extra_content": SIGNATURE,
        }
    ]
    assert json.loads(json.dumps(first.reasoning)) == first.reasoning
    history = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            tool_calls=first.tool_calls,
            reasoning=first.reasoning,
        ),
        Message(role="tool", content="ready", tool_call_id="call_1"),
    ]
    await model.generate("sys", history, [SPEC])
    sent = chat.calls[1]["messages"][2]
    assert sent == {"role": "assistant", "content": None, "tool_calls": first.reasoning}


async def test_gemini_replays_without_the_signature_when_the_engine_stripped_the_reasoning():
    chat = Completions([completion("done")])
    model = gemini_over(chat)
    history = [
        Message(role="user", content="go"),
        Message(role="assistant", tool_calls=[ToolCall(id="c0", name="echo", args={"a": 1})]),
        Message(role="tool", content="ready", tool_call_id="c0"),
    ]
    await model.generate("sys", history, [SPEC])
    assert chat.calls[0]["messages"][2]["tool_calls"] == [
        {"id": "c0", "type": "function", "function": {"name": "echo", "arguments": '{"a": 1}'}}
    ]


async def test_a_gemini_reply_without_signatures_has_no_reasoning():
    call = SimpleNamespace(
        type="function", id="c0", function=SimpleNamespace(name="echo", arguments="{}")
    )
    chat = Completions([completion(None, [call], "tool_calls")])
    reply = await gemini_over(chat).generate("sys", [Message(role="user", content="go")], [SPEC])
    assert reply.reasoning is None


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


async def test_gemini_leaves_out_an_empty_turn():
    chat = Completions([completion("hi")])
    time = Time()
    model = Gemini(
        SimpleNamespace(chat=SimpleNamespace(completions=chat)),
        "gemini-3.5-flash-lite",
        None,
        retry(time, Logged()),
        Limiter(15, time.clock, time.sleep),
    )
    history = [Message(role="user", content="a"), Message(role="assistant", content="")]
    await model.generate("s", history, [])
    assert chat.calls[0]["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
    ]


INFO = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}


@pytest.mark.parametrize("body", [[{"error": {"details": [INFO]}}], {"error": {"details": [INFO]}}])
async def test_gemini_reads_the_retry_delay_from_the_body_when_no_header_names_one(body):
    time = Time()
    err = openai.APIStatusError("x", response=httpx.Response(429, request=REQUEST), body=body)
    chat = Completions([err, completion("ok")])
    model = Gemini(
        SimpleNamespace(chat=SimpleNamespace(completions=chat)),
        "gemini-3.5-flash-lite",
        None,
        retry(time, Logged()),
        Limiter(15, time.clock, time.sleep),
    )
    await model.generate("s", [Message(role="user", content="hi")], [])
    assert time.slept == [37.0]


async def test_gemini_retries_pass_the_limiter_each_time():
    time = Time()
    err = openai.APIStatusError("x", response=httpx.Response(429, request=REQUEST), body=None)
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
    other = await TypeSafe(
        Posts([reply_of(200, {"answers": {"safe": {"noul": 0.1}}})]), "jev-2"
    ).decide("state", {"safe": Check(instructions="is it safe?")})
    assert other.model == "jev-2"
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


async def test_the_log_writes_a_row_with_its_scope_and_survey(pool):
    log = ProviderLog()
    log.use(pool, survey="s1")
    await log("anthropic", "logged-model", 429, 7.5, {"retry-after": "7"})
    log.use(pool)
    await log("gemini", "logged-model", None, 1.0, {})
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "select scope, survey, provider, status, wait, headers::text as headers"
            " from provider_log where model = 'logged-model' order by id"
        )
    assert [(r["scope"], r["survey"], r["provider"], r["status"], r["wait"]) for r in rows] == [
        ("moon", "s1", "anthropic", 429, 7.5),
        ("moon", None, "gemini", None, 1.0),
    ]
    assert json.loads(rows[0]["headers"]) == {"retry-after": "7"}


async def test_without_a_pool_the_log_only_warns(caplog):
    with caplog.at_level("WARNING", logger="providers"):
        await ProviderLog()("anthropic", "claude-haiku-5-5", 429, 1.0, {"retry-after": "1"})
    assert "answered 429" in caplog.text


ECHO = ToolSpec(
    name="echo",
    description="Returns the text it is given.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)


async def round_trip(model):
    ask = Message(
        role="user", content="Call the echo tool with the text ready, then say what it returned."
    )
    system = "Use the tools you are given."
    first = await model.generate(system, [ask], [ECHO])
    assert first.tool_calls and first.stop_reason == "tool_use" and first.usage.output > 0
    call = first.tool_calls[0]
    history = [
        ask,
        Message(
            role="assistant",
            content=first.text,
            tool_calls=first.tool_calls,
            reasoning=first.reasoning,
        ),
        Message(role="tool", content="ready", tool_call_id=call.id),
    ]
    final = await model.generate(system, history, [ECHO])
    assert final.text and final.stop_reason == "end_turn"
    return first, final


@pytest.mark.live
async def test_the_live_providers_answer_once():
    first, final = await round_trip(providers.anthropic("claude-haiku-5-5", "low"))
    assert first.model == final.model == "claude-haiku-5-5"
    first, final = await round_trip(providers.gemini(providers.GEMINI_MODEL, None))
    assert first.model == final.model == providers.GEMINI_MODEL
    decided = await providers.typesafe("jev-latest", None).decide(
        "The sky is blue.", {"true": Check(instructions="Is the state about the sky?")}
    )
    assert decided.model == "jev-latest"
    assert 0.0 <= decided.answers["true"].probability <= 1.0
