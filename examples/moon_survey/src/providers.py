import asyncio
import json
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import asyncpg
import httpx
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import AsyncAnthropic
from openai import APIConnectionError as OpenAIConnectionError
from openai import AsyncOpenAI
from rfnry_engine import (
    Answer,
    CheckAnswer,
    ChoiceAnswer,
    Content,
    Decided,
    Message,
    QuestionLike,
    Reply,
    ScoreAnswer,
    State,
    Text,
    ToolCall,
    ToolSpec,
    Usage,
)

HAIKU_MODEL = "claude-haiku-5-5"
GEMINI_MODEL = "gemini-3.5-flash-lite"
DECISION_MODEL = "jev-latest"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
MAX_TOKENS = 16000
ATTEMPTS = 1000
DEADLINE = 1800.0
GATE_DEADLINE = 120.0
BACKOFF_CAP = 60.0
GEMINI_PER_MINUTE = 15
GEMINI_EFFORTS = ("minimal", "low", "medium", "high")
TRANSIENT = (httpx.TransportError, AnthropicConnectionError, OpenAIConnectionError, TimeoutError)

T = TypeVar("T")
Log = Callable[[str, str, int | None, float, dict[str, str]], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]

logger = logging.getLogger("providers")


class ProviderError(Exception):
    pass


def key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ProviderError(f"providers: {name} is not set")
    return value


def lowered(response: Any) -> dict[str, str]:
    headers = getattr(response, "headers", None)
    if headers is None:
        return {}
    return {name.lower(): str(value) for name, value in headers.items()}


def kept_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.startswith("retry-after") or "ratelimit" in name or name.endswith("request-id")
    }


def status_of(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status


def should_retry(error: BaseException, headers: Mapping[str, str]) -> bool:
    flagged = headers.get("x-should-retry")
    if flagged in ("true", "false"):
        return flagged == "true"
    status = status_of(error)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    return isinstance(error, TRANSIENT)


def body_delay(error: BaseException) -> float | None:
    body = getattr(error, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
        body = body["error"]
    details = body.get("details") if isinstance(body, Mapping) else None
    for detail in details if isinstance(details, list) else []:
        delay = detail.get("retryDelay") if isinstance(detail, Mapping) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return max(0.0, float(delay[:-1]))
            except ValueError:
                continue
    return None


class ProviderLog:
    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None
        self.scope = "moon"
        self.survey: str | None = None

    def use(
        self, pool: asyncpg.Pool | None, scope: str = "moon", survey: str | None = None
    ) -> None:
        self.pool, self.scope, self.survey = pool, scope, survey

    def set_survey(self, survey: str | None) -> None:
        self.survey = survey

    async def __call__(
        self, provider: str, model: str, status: int | None, wait: float, headers: dict[str, str]
    ) -> None:
        logger.warning(
            "providers: %s %s answered %s, waiting %.1fs %s", provider, model, status, wait, headers
        )
        if self.pool is None:
            return
        await self.pool.execute(
            "insert into provider_log (scope, survey, provider, model, status, wait, headers)"
            " values ($1, $2, $3, $4, $5, $6, $7)",
            self.scope,
            self.survey,
            provider,
            model,
            status,
            wait,
            json.dumps(headers),
        )


provider_log = ProviderLog()


class Retry:
    def __init__(
        self,
        attempts: int = ATTEMPTS,
        deadline: float = DEADLINE,
        sleep: Sleep = asyncio.sleep,
        log: Log = provider_log,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.attempts, self.deadline, self.sleep, self.log = attempts, deadline, sleep, log
        self.clock, self.wall, self.rng = clock, wall, rng

    def named(self, headers: Mapping[str, str], error: BaseException) -> float | None:
        millis = headers.get("retry-after-ms")
        try:
            if millis is not None:
                return max(0.0, float(millis) / 1000)
        except ValueError:
            pass
        value = headers.get("retry-after")
        if value is not None:
            try:
                return max(0.0, float(value))
            except ValueError:
                pass
            try:
                return max(0.0, parsedate_to_datetime(value).timestamp() - self.wall())
            except TypeError, ValueError:
                pass
        return body_delay(error)

    def wait_for(self, named: float | None, attempt: int) -> float:
        base = max(named or 0.0, min(BACKOFF_CAP, 2.0**attempt))
        return base + self.rng() * base

    async def note(
        self, provider: str, model: str, status: int | None, wait: float, headers: dict[str, str]
    ) -> None:
        try:
            await self.log(provider, model, status, wait, headers)
        except Exception as failure:
            logger.warning("providers: the log did not write: %s", failure)

    async def call(self, provider: str, model: str, attempt: Callable[[], Awaitable[T]]) -> T:
        started = self.clock()
        tried = 0
        while True:
            try:
                return await attempt()
            except Exception as error:
                tried += 1
                headers = lowered(getattr(error, "response", None))
                status = status_of(error)
                if not should_retry(error, headers):
                    if status is not None:
                        raise ProviderError(
                            f"providers: {model} answered {status}: {error}"
                        ) from error
                    raise
                kept = kept_headers(headers)
                left = self.deadline - (self.clock() - started)
                named = self.named(headers, error)
                if tried >= self.attempts or left <= 0 or (named is not None and named > left):
                    await self.note(provider, model, status, 0.0, kept)
                    raise
                wait = min(self.wait_for(named, tried - 1), left)
                await self.note(provider, model, status, wait, kept)
                await self.sleep(wait)


class Limiter:
    def __init__(
        self,
        per_minute: int = GEMINI_PER_MINUTE,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.interval = 60.0 / per_minute
        self.clock, self.sleep = clock, sleep
        self.next = 0.0

    async def wait(self) -> None:
        now = self.clock()
        start = max(now, self.next)
        self.next = start + self.interval
        if start > now:
            await self.sleep(start - now)


def text_of(content: Content) -> str:
    if isinstance(content, str):
        return content
    if not all(isinstance(part, Text) for part in content):
        raise ProviderError("providers: text only")
    return "".join(part.text for part in content)


class Haiku:
    stops = {
        "end_turn": "end_turn",
        "stop_sequence": "end_turn",
        "tool_use": "tool_use",
        "max_tokens": "max_tokens",
    }

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str = HAIKU_MODEL,
        effort: str | None = None,
        retry: Retry | None = None,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.client, self.model, self.effort = client, model, effort
        self.retry = retry or Retry()
        self.max_tokens = max_tokens

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "adaptive"},
            "cache_control": {"type": "ephemeral"},
            "system": system,
            "messages": self.turns(messages),
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        if tools:
            request["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        response = await self.retry.call(
            "anthropic", self.model, lambda: self.client.messages.create(**request)
        )
        if response.stop_reason == "refusal":
            raise ProviderError(f"providers: {self.model} refused the request")
        blocks = [block.model_dump(exclude_none=True) for block in response.content]
        thinking = any(b["type"] in ("thinking", "redacted_thinking") for b in blocks)
        usage = response.usage
        return Reply(
            text="".join(b["text"] for b in blocks if b["type"] == "text"),
            tool_calls=[
                ToolCall(id=b["id"], name=b["name"], args=b["input"])
                for b in blocks
                if b["type"] == "tool_use"
            ],
            usage=Usage(
                input=usage.input_tokens,
                output=usage.output_tokens,
                cache_creation=usage.cache_creation_input_tokens or 0,
                cache_read=usage.cache_read_input_tokens or 0,
            ),
            stop_reason=self.stops.get(response.stop_reason or "", "error"),
            model=self.model,
            settings={
                "max_tokens": self.max_tokens,
                "thinking": "adaptive",
                "effort": self.effort,
            },
            reasoning=blocks if thinking else None,
        )

    def turns(self, messages: list[Message]) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": text_of(message.content),
                }
                if message.is_error:
                    result["is_error"] = True
                role, content = "user", [result]
            elif message.role == "assistant":
                calls = [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                    for c in message.tool_calls
                ]
                role = "assistant"
                content = message.reasoning or [*self.text(message.content), *calls]
            else:
                role, content = "user", self.text(message.content)
            if not content:
                continue
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] = turns[-1]["content"] + content
            else:
                turns.append({"role": role, "content": content})
        for turn in turns:
            if turn["role"] == "user":
                turn["content"].sort(key=lambda block: block["type"] != "tool_result")
        return turns

    def text(self, content: Content) -> list[dict[str, Any]]:
        text = text_of(content)
        return [{"type": "text", "text": text}] if text else []


class Gemini:
    stops = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = GEMINI_MODEL,
        effort: str | None = None,
        retry: Retry | None = None,
        limiter: Limiter | None = None,
    ) -> None:
        self.client, self.model, self.effort = client, model, effort
        self.retry = retry or Retry()
        self.limiter = limiter or Limiter()

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *self.turns(messages)],
        }
        settings = None
        if self.effort in GEMINI_EFFORTS:
            request["reasoning_effort"] = self.effort
            settings = {"reasoning_effort": self.effort}
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

        async def attempt() -> Any:
            await self.limiter.wait()
            return await self.client.chat.completions.create(**request)

        response = await self.retry.call("gemini", self.model, attempt)
        choice = response.choices[0]
        calls = [c for c in choice.message.tool_calls or [] if c.type == "function"]
        usage = response.usage
        cached = (
            (usage.prompt_tokens_details.cached_tokens or 0)
            if usage and usage.prompt_tokens_details
            else 0
        )
        return Reply(
            text=choice.message.content or "",
            tool_calls=[
                ToolCall(
                    id=c.id or f"call_{n}",
                    name=c.function.name,
                    args=json.loads(c.function.arguments or "{}"),
                )
                for n, c in enumerate(calls)
            ],
            usage=Usage(
                input=(usage.prompt_tokens - cached) if usage else 0,
                output=usage.completion_tokens if usage else 0,
                cache_read=cached,
            ),
            stop_reason="tool_use" if calls else self.stops.get(choice.finish_reason, "error"),
            model=self.model,
            settings=settings,
        )

    def turns(self, messages: list[Message]) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for message in messages:
            text = text_of(message.content)
            if not text and not message.tool_calls and message.role != "tool":
                continue
            if message.role == "tool":
                turns.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": text,
                    }
                )
            elif message.role == "assistant":
                turn: dict[str, Any] = {"role": "assistant", "content": text or None}
                if message.tool_calls:
                    turn["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.args)},
                        }
                        for c in message.tool_calls
                    ]
                turns.append(turn)
            else:
                turns.append({"role": "user", "content": text})
        return turns


class TypeSafe:
    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str = DECISION_MODEL,
        retry: Retry | None = None,
    ) -> None:
        self.client, self.model = client, model
        self.retry = retry or Retry(deadline=GATE_DEADLINE)

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> Decided:
        body = {
            "state": state,
            "model": self.model,
            "questions": {qid: self.question(q) for qid, q in questions.items()},
        }

        async def attempt() -> httpx.Response:
            response = await self.client.post(TYPESAFE_URL, json=body)
            response.raise_for_status()
            return response

        response = await self.retry.call("typesafe", self.model, attempt)
        data = response.json()
        usage = data.get("usage") or {}
        return Decided(
            answers={qid: self.answer(q, data["answers"][qid]) for qid, q in questions.items()},
            usage=Usage(input=usage.get("input_tokens", 0), output=usage.get("output_tokens", 0)),
            model=self.model,
        )

    def question(self, question: QuestionLike) -> dict[str, Any]:
        if question.type == "choice":
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": dict(question.options),
            }
        if question.type == "score":
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": list(question.levels),
            }
        body: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
        if question.yes is not None or question.no is not None:
            body["criteria"] = {"true": question.yes, "false": question.no}
        return body

    def answer(self, question: QuestionLike, raw: dict[str, Any]) -> Answer:
        if question.type == "choice":
            probabilities = self.normalized(
                {o: raw["probabilities"].get(o, 0.0) for o in question.options}
            )
            choice = raw.get("choice")
            if choice not in probabilities:
                choice = max(probabilities, key=lambda o: probabilities[o])
            return ChoiceAnswer(choice=choice, probabilities=probabilities)
        if question.type == "score":
            levels = self.normalized(
                {str(i): raw["probabilities"].get(str(i), 0.0) for i in range(len(question.levels))}
            )
            return ScoreAnswer(score=float(raw["score"]), probabilities=list(levels.values()))
        return CheckAnswer(probability=min(1.0, max(0.0, float(raw["noul"]))))

    def normalized(self, probabilities: dict[str, float]) -> dict[str, float]:
        total = sum(float(value) for value in probabilities.values())
        if total <= 0:
            return {k: 1 / len(probabilities) for k in probabilities}
        return {k: float(v) / total for k, v in probabilities.items()}


built: dict[tuple[str, str | None, str | None], Any] = {}
limiters: dict[str, Limiter] = {}


def anthropic(model: str | None, effort: str | None) -> Haiku:
    name = model or HAIKU_MODEL
    cached = built.get(("anthropic", name, effort))
    if cached is None:
        client = AsyncAnthropic(api_key=key("ANTHROPIC_API_KEY"), max_retries=0)
        cached = built[("anthropic", name, effort)] = Haiku(client, name, effort)
    return cached


def gemini(model: str | None, effort: str | None) -> Gemini:
    name = model or GEMINI_MODEL
    cached = built.get(("gemini", name, effort))
    if cached is None:
        client = AsyncOpenAI(base_url=GEMINI_URL, api_key=key("GEMINI_API_KEY"), max_retries=0)
        limiter = limiters.setdefault(name, Limiter())
        cached = built[("gemini", name, effort)] = Gemini(client, name, effort, limiter=limiter)
    return cached


def typesafe(model: str | None, effort: str | None) -> TypeSafe:
    name = model or DECISION_MODEL
    cached = built.get(("typesafe", name, None))
    if cached is None:
        bearer = {"Authorization": f"Bearer {key('TYPESAFE_API_KEY')}"}
        client = httpx.AsyncClient(timeout=60, headers=bearer)
        cached = built[("typesafe", name, None)] = TypeSafe(client, name)
    return cached
