import asyncio
import json
import math
from collections.abc import Mapping
from string import ascii_uppercase
from typing import Any

import httpx
from rfnry_engine import (
    Answer,
    CheckAnswer,
    ChoiceAnswer,
    Decided,
    QuestionLike,
    ScoreAnswer,
    State,
    Usage,
)


class LocalDecisionProvider:
    retried = (429, 500, 502, 503)
    system = (
        "You read a state and answer one question about it by choosing one of the lettered "
        "options. Reply with the letter alone."
    )

    def __init__(
        self,
        model: str = "llama3.1:8b",
        url: str = "http://localhost:11434/v1",
        body: dict[str, Any] | None = None,
        attempts: int = 4,
    ) -> None:
        self.model = model
        self.url = f"{url.rstrip('/')}/chat/completions"
        self.body = body or {}
        self.attempts = attempts
        self.client = httpx.AsyncClient(timeout=120)

    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> Decided:
        asked = await asyncio.gather(*(self._ask(state, q) for q in questions.values()))
        return Decided(
            answers={qid: answer for qid, (answer, _, _) in zip(questions, asked, strict=True)},
            usage=sum((usage for _, usage, _ in asked), Usage()),
            model=asked[0][2] if asked else self.model,
        )

    async def _ask(self, state: State, question: QuestionLike) -> tuple[Answer, Usage, str]:
        options = self._options(question)
        if len(options) > len(ascii_uppercase):
            raise ValueError(f"local: {len(options)} options, more than the 26 letters")
        letters = ascii_uppercase[: len(options)]
        listed = "\n".join(
            f"{letter}) {option}" for letter, option in zip(letters, options, strict=True)
        )
        allowed = f"{', '.join(letters[:-1])} or {letters[-1]}"
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        prompt = (
            f"{text}\n\nQuestion: {self._shown(question.instructions)}\n\n{listed}\n\n"
            f"Answer with one letter: {allowed}."
        )
        data = await self._complete(prompt)
        probabilities = self._probabilities(data, letters)
        usage = data.get("usage") or {}
        spent = Usage(input=usage.get("prompt_tokens", 0), output=usage.get("completion_tokens", 0))
        return self._answer(question, probabilities), spent, data.get("model") or self.model

    def _options(self, question: QuestionLike) -> list[str]:
        if question.type == "choice":
            return [
                key if meaning is None else f"{key}: {self._shown(meaning)}"
                for key, meaning in question.options.items()
            ]
        if question.type == "score":
            return [self._shown(level) for level in question.levels]
        return [
            f"yes: {self._shown(question.yes)}" if question.yes else "yes",
            f"no: {self._shown(question.no)}" if question.no else "no",
        ]

    def _shown(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    async def _complete(self, prompt: str) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
        } | self.body
        for attempt in range(self.attempts):
            response = await self.client.post(self.url, json=body)
            if response.status_code not in self.retried or attempt == self.attempts - 1:
                break
            await asyncio.sleep(float(response.headers.get("retry-after", 2**attempt)))
        response.raise_for_status()
        return response.json()

    def _probabilities(self, data: dict[str, Any], letters: str) -> list[float]:
        content = (data["choices"][0].get("logprobs") or {}).get("content") or []
        if not content or not content[0].get("top_logprobs"):
            raise ValueError("local: the server returned no top_logprobs for the answer token")
        candidates = content[0]["top_logprobs"]
        mass = dict.fromkeys(letters, 0.0)
        for candidate in candidates:
            token = candidate["token"].strip()
            if token in mass:
                mass[token] += math.exp(candidate["logprob"])
        floor = math.exp(min(candidate["logprob"] for candidate in candidates))
        weights = [mass[letter] or floor for letter in letters]
        total = sum(weights)
        if total <= 0:
            return [1 / len(letters)] * len(letters)
        return [weight / total for weight in weights]

    def _answer(self, question: QuestionLike, probabilities: list[float]) -> Answer:
        if question.type == "choice":
            by_option = dict(zip(question.options, probabilities, strict=True))
            choice = max(by_option, key=lambda option: by_option[option])
            return ChoiceAnswer(choice=choice, probabilities=by_option)
        if question.type == "score":
            score = sum(level * p for level, p in enumerate(probabilities))
            return ScoreAnswer(score=score, probabilities=probabilities)
        return CheckAnswer(probability=probabilities[0])
