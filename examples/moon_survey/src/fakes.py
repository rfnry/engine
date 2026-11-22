import asyncio
import hashlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rfnry_engine import (
    CheckAnswer,
    Decided,
    DecisionModel,
    GenerationModel,
    Message,
    Reply,
    ToolCall,
    ToolSpec,
    Usage,
)

from providers import ProviderError, provider_log, text_of

Sleep = Callable[[float], Awaitable[object]]
Log = Callable[[str, str, int | None, float, dict[str, str]], Awaitable[object]]

MODEL = "fake"
SCHEMA = (
    Path(__file__).resolve().parent.parent
    / "agents"
    / "surveyor"
    / "skills"
    / "dossier"
    / "dossier.schema.json"
)
TREE = "$RFNRY_TREE/surveyor/skills/dossier"
WEH_PER_H = 9 / 10_000
SIGMAS = 2
MEASURED = ("TiO2", "FeO", "Al2O3", "Th")
ENDED = ("finished", "failed", "lost", "stopped")
ROUNDS = 3
POLAR = 75


class FakeError(Exception):
    pass


def seeded(key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))


@dataclass(frozen=True)
class Profile:
    source: str
    latency: Mapping[str, list[float]]
    size: Mapping[str, list[int]]
    scale: float = 1.0
    path: str = field(default="", compare=False)

    @classmethod
    def load(cls, path: Path | str, scale: float = 1.0) -> Profile:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            source=raw["source"],
            latency={name: list(found) for name, found in raw["latency"].items()},
            size={name: list(found) for name, found in raw["size"].items()},
            scale=scale,
            path=str(path),
        )

    def delay(self, role: str, key: str) -> float:
        samples = self.latency.get(role)
        if not samples:
            return 0
        return seeded(f"latency|{role}|{key}").choice(samples) * self.scale

    def chars(self, role: str, key: str) -> int:
        samples = self.size.get(role)
        if not samples:
            return 0
        return seeded(f"size|{role}|{key}").choice(samples)


class Limits:
    def __init__(
        self,
        per_minute: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        log: Log = provider_log,
    ) -> None:
        self.interval = None if per_minute is None else 60.0 / per_minute
        self.clock, self.sleep, self.log = clock, sleep, log
        self.next = 0.0

    async def wait(self) -> None:
        if self.interval is None:
            return
        now = self.clock()
        start = max(now, self.next)
        self.next = start + self.interval
        if start > now:
            wait = start - now
            await self.log(MODEL, MODEL, 429, wait, {"retry-after": f"{wait:.3f}"})
            await self.sleep(wait)


def spec_chars(spec: ToolSpec) -> int:
    return len(spec.name) + len(spec.description) + len(json.dumps(spec.input_schema))


def call_chars(call: ToolCall) -> int:
    return len(json.dumps({"name": call.name, "args": call.args}))


def message_chars(message: Message) -> int:
    return len(text_of(message.content)) + sum(call_chars(c) for c in message.tool_calls)


def usage_of(system: str, messages: list[Message], tools: list[ToolSpec], reply: Reply) -> Usage:
    sent = len(system) + sum(message_chars(m) for m in messages) + sum(map(spec_chars, tools))
    made = len(reply.text) + sum(call_chars(c) for c in reply.tool_calls)
    return Usage(input=sent // 4, output=made // 4)


def first_input(messages: list[Message]) -> Any:
    for message in messages:
        if message.role == "user" and message.notice is None:
            text = text_of(message.content)
            try:
                return json.loads(text)
            except ValueError as error:
                raise FakeError(f"fakes: the input is not JSON: {error}") from error
    raise FakeError("fakes: the conversation holds no input")


def one_line(text: str, limit: int = 300) -> str:
    return " ".join(text.split())[:limit]


@dataclass
class Exchange:
    call: ToolCall
    result: Message | None

    @property
    def error(self) -> bool:
        return self.result is not None and self.result.is_error

    @property
    def body(self) -> str:
        return "" if self.result is None else text_of(self.result.content)

    def data(self) -> Any:
        try:
            return json.loads(self.body)
        except ValueError:
            return None


def exchanges(messages: list[Message]) -> list[Exchange]:
    results = {m.tool_call_id: m for m in messages if m.role == "tool"}
    return [
        Exchange(call, results.get(call.id))
        for m in messages
        if m.role == "assistant"
        for call in m.tool_calls
    ]


def failure(found: list[Exchange]) -> str | None:
    for item in found:
        if item.error:
            return f"{item.call.name} failed: {one_line(item.body)}"
    return None


Script = tuple[str, list[tuple[str, dict[str, Any]]]]


class Fake:
    role = ""

    def __init__(self, profile: Profile, sleep: Sleep, limits: Limits | None = None) -> None:
        self.profile, self.sleep, self.limits = profile, sleep, limits

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        if self.limits is not None:
            await self.limits.wait()
        key = f"{text_of(messages[0].content) if messages else ''}|{len(messages)}"
        await self.sleep(self.profile.delay(self.role, key))
        text, planned = self.script(messages)
        turn = sum(1 for m in messages if m.role == "assistant")
        calls = [
            ToolCall(id=f"{self.role}-{turn}-{n}", name=name, args=args)
            for n, (name, args) in enumerate(planned)
        ]
        if calls:
            size = self.profile.chars(self.role, key)
            text = (
                (text + " " + "thinking about the survey. " * (size // 27 + 1))[:size]
                if size
                else text
            )
        reply = Reply(
            text=text,
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            model=MODEL,
        )
        return reply.model_copy(update={"usage": usage_of(system, messages, tools, reply)})

    def script(self, messages: list[Message]) -> Script:
        raise NotImplementedError


class Surveyor(Fake):
    role = "surveyor"

    def __init__(self, profile: Profile, sleep: Sleep, web: Any) -> None:
        super().__init__(profile, sleep)
        self.pages = sorted(getattr(web, "pages", {}))

    def script(self, messages: list[Message]) -> Script:
        region = first_input(messages)["region"]
        found = exchanges(messages)
        stopped = failure(found)
        if stopped is not None:
            return f"{region}: stopped, {stopped}", []
        by_name: dict[str, list[Exchange]] = {}
        for item in found:
            by_name.setdefault(item.call.name, []).append(item)
        claimed = by_name.get("claim")
        if not claimed:
            return "", [("claim", {"region": region})]
        answer = claimed[0].data() or {}
        if answer.get("claimed") is not True:
            return f"{region}: not mine, held by {answer.get('by')}", []
        if "region" not in by_name:
            return "", [("region", {"region": region})]
        around = by_name["region"][0].data() or {}
        neighbours = list(around.get("neighbours", []))
        if neighbours and "notes" not in by_name:
            return "", [("notes", {"region": n}) for n in neighbours]
        if self.pages and "fetch" not in by_name:
            return "", [("fetch", {"url": self.pages[0]})]
        shells = by_name.get("Shell", [])
        measure = next((s for s in shells if "measure.py" in s.call.args["command"]), None)
        if measure is None:
            command = f'python "{TREE}/measure.py" {region}'
            return "", [("Shell", {"command": command})]
        if (output := self.outcome(measure, "measure")) is not None:
            return f"{region}: stopped, {output}", []
        if "Write" not in by_name:
            dossier = self.dossier(region, json.loads(measure.data()["stdout"]), around)
            return "", [("Write", {"path": "dossier.json", "content": json.dumps(dossier)})]
        check = next((s for s in shells if "check.py" in s.call.args["command"]), None)
        if check is None:
            return "", [("Shell", {"command": f'python "{TREE}/check.py" dossier.json'})]
        if (output := self.outcome(check, "check")) is not None:
            return f"{region}: stopped, {output}", []
        if "note" not in by_name:
            text = "simulated from the Lunar Prospector data in data/lp; checked clean"
            return "", [("note", {"region": region, "kind": "finding", "text": text})]
        return f"{region}: dossier written, checked and noted", []

    def outcome(self, item: Exchange, what: str) -> str | None:
        data = item.data() or {}
        if data.get("exit_code") == 0 and not data.get("timed_out"):
            if what == "check" and data.get("stdout"):
                return f"check: {one_line(data['stdout'])}"
            return None
        return f"{what}: {one_line(str(data.get('stderr') or data.get('stdout') or item.body))}"

    def dossier(self, region: str, measured: dict[str, Any], around: dict[str, Any]) -> dict:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))["properties"]["values"][
            "properties"
        ]
        values: dict[str, Any] = {}
        found = measured["values"]

        def ranged(name: str, mean: float, error: float, source: str) -> dict[str, Any]:
            bounds = schema[name]["properties"]
            low = max(bounds["low"]["minimum"], mean - SIGMAS * error)
            high = min(bounds["high"]["maximum"], mean + SIGMAS * error)
            return {
                "low": low,
                "high": high,
                "confidence": "medium",
                "sources": [source],
            }

        hydrogen = found.get("H")
        if hydrogen is None:
            values["water_ice"] = {"unknown": "no neutron-spectrometer pixel falls in the region"}
        else:
            weh = hydrogen["mean"] * WEH_PER_H
            values["water_ice"] = ranged(
                "water_ice",
                weh,
                hydrogen["error"] * WEH_PER_H,
                "Lunar Prospector neutron spectrometer, data/lp",
            )
        for name in MEASURED:
            value = found.get(name)
            if value is None:
                values[name] = {"unknown": "no gamma-ray pixel falls in the region"}
            else:
                source = "Lunar Prospector gamma-ray spectrometer, data/lp"
                values[name] = ranged(name, value["mean"], value["error"], source)
        values["He3"] = {"unknown": "no orbiter measures helium-3 and the fake model has no paper"}
        values = {name: values[name] for name in schema}
        return {
            "region": region,
            "terrain": self.terrain(around, found.get("FeO")),
            "values": values,
            "notes": "Simulated by a fake model from the Lunar Prospector data in data/lp.",
            "simulated": True,
        }

    def terrain(self, around: dict[str, Any], iron: dict[str, Any] | None) -> str:
        centre = around.get("centre") or [0, 0]
        if abs(centre[0]) >= POLAR:
            return "polar"
        if iron is None:
            return "mixed"
        if iron["mean"] >= 15:
            return "mare"
        return "highland" if iron["mean"] <= 8 else "mixed"


class Crosscheck(Fake):
    def __init__(self, role: str, profile: Profile, sleep: Sleep, limits: Limits | None) -> None:
        super().__init__(profile, sleep, limits)
        self.role = role

    def script(self, messages: list[Message]) -> Script:
        region = first_input(messages)["region"]
        found = exchanges(messages)
        stopped = failure(found)
        if stopped is not None:
            return f"{region}: stopped, {stopped}", []
        by_name = {item.call.name: item for item in found}
        if "notes" not in by_name:
            return "", [("notes", {"region": region})]
        seen = (by_name["notes"].data() or {}).get("notes", [])
        if "note" not in by_name:
            text = f"agree: read {len(seen)} notes on the region and found nothing to dispute"
            return "", [("note", {"region": region, "kind": "agree", "text": text})]
        return f"{region}: checked, agree", []


class Team(Fake):
    def __init__(self, role: str, profile: Profile, sleep: Sleep) -> None:
        super().__init__(profile, sleep)
        self.role = role

    def plan(self, given: Any) -> tuple[list[str], list[tuple[str, dict[str, Any]]]]:
        if self.role == "lead":
            shares = given["shares"]
            if isinstance(shares, int):
                regions = given["regions"]
                shares = [regions[n::shares] for n in range(shares)]
            second = set(given.get("second", ()))
            return [r for share in shares for r in share], [
                (
                    "captain",
                    {"share": list(share), "second": [r for r in share if r in second]},
                )
                for share in shares
            ]
        regions = given["share"] if "share" in given else given["regions"]
        return list(regions), [("surveyor", {"region": r}) for r in regions]

    def script(self, messages: list[Message]) -> Script:
        given = first_input(messages)
        regions, firsts = self.plan(given)
        second = set(given.get("second", ()))
        found = exchanges(messages)
        delegated = [e for e in found if e.call.name == "Delegate"]
        if not delegated:
            return "", [
                (
                    "Delegate",
                    {"agent": agent, "input": json.dumps(args), "background": True},
                )
                for agent, args in firsts
            ]
        ended = {
            m.notice.job: m.notice.kind
            for m in messages
            if m.notice is not None and m.notice.kind in ENDED
        }
        jobs: dict[str, tuple[str, str]] = {}
        refused = 0
        for item in delegated:
            data = item.data()
            if item.error or not isinstance(data, dict) or "job" not in data:
                refused += 1
                continue
            jobs[data["job"]] = (item.call.args["agent"], item.call.args["input"])
        if self.role != "lead":
            checked = {
                json.loads(e.call.args["input"])["region"]
                for e in delegated
                if e.call.args["agent"].startswith("crosscheck")
            }
            todo = [
                json.loads(text)["region"]
                for job, (agent, text) in jobs.items()
                if agent == "surveyor"
                and ended.get(job) == "finished"
                and json.loads(text)["region"] not in checked
            ]
            if todo:
                return "", [
                    (
                        "Delegate",
                        {
                            "agent": "crosscheck_second" if r in second else "crosscheck",
                            "input": json.dumps({"region": r}),
                            "background": True,
                        },
                    )
                    for r in todo
                ]
        running = [job for job in jobs if job not in ended]
        if running:
            return f"waiting for {len(running)} helpers", []
        counts = self.counts(jobs, ended, refused)
        if not any(e.call.name == "note" for e in found):
            return "", [("note", {"region": regions[0], "kind": "summary", "text": counts})]
        stopped = failure([e for e in found if e.call.name == "note"])
        return stopped or f"survey done: {counts}", []

    def counts(self, jobs: dict[str, tuple[str, str]], ended: dict[str, str], refused: int) -> str:
        parts = []
        for agent in dict.fromkeys(a for a, _ in jobs.values()):
            kinds = [ended[job] for job, (a, _) in jobs.items() if a == agent]
            name = "surveyors" if agent == "surveyor" else f"{agent}s"
            failed = kinds.count("failed") + (refused if agent in ("surveyor", "captain") else 0)
            parts.append(
                f"{len(kinds) + (refused if agent in ('surveyor', 'captain') else 0)} {name}: "
                f"{kinds.count('finished')} finished, {failed} failed, "
                f"{kinds.count('lost')} lost, {kinds.count('stopped')} stopped"
            )
        if not parts and refused:
            parts.append(f"{refused} helpers: 0 finished, {refused} failed, 0 lost, 0 stopped")
        return "; ".join(parts)


class Boss(Fake):
    role = "boss"

    def script(self, messages: list[Message]) -> Script:
        found = exchanges(messages)
        stopped = failure(found)
        if stopped is not None:
            return f"boss: stopped, {stopped}", []
        progresses = [n for n, e in enumerate(found) if e.call.name == "progress"]
        if not progresses:
            return "", [("progress", {})]
        last = progresses[-1]
        latest = found[last].data() or {}
        after = found[last + 1 :]
        starts = [e for e in after if e.call.name == "start"]
        free = list(latest.get("free", []))
        if free and not starts:
            return "", [("start", {"region": r}) for r in free]
        if starts:
            jobs = [(e.data() or {}).get("job") for e in starts]
            state: dict[str, bool] = {}
            for e in after:
                if e.call.name == "check":
                    state[e.call.args["job"]] = (e.data() or {}).get("running") is True
            open_ = [j for j in jobs if j is not None and state.get(j, True)]
            if open_:
                return "", [("check", {"job": j}) for j in open_]
            if len(progresses) < ROUNDS:
                return "", [("progress", {})]
        return f"boss: done, {json.dumps(latest.get('counts', latest))}", []


class FakeGate:
    def __init__(self, refuse: frozenset[str] | set[str] = frozenset(), down: bool = False) -> None:
        self.refuse, self.down = frozenset(refuse), down

    def tool_of(self, state: Any) -> str | None:
        if isinstance(state, dict):
            for name in ("tool", "name"):
                if isinstance(state.get(name), str):
                    return state[name]
            call = state.get("call")
            if isinstance(call, dict):
                return self.tool_of(call)
        return None

    async def decide(self, state: Any, questions: Mapping[str, Any]) -> Decided:
        if self.down:
            raise ProviderError("providers: the fake gate is down")
        probability = 0.0 if self.tool_of(state) in self.refuse else 1.0
        sent = len(json.dumps(state, default=str)) + len(json.dumps(list(questions)))
        made = sum(len(name) + 12 for name in questions)
        return Decided(
            answers={name: CheckAnswer(probability=probability) for name in questions},
            usage=Usage(input=sent // 4, output=made // 4),
            model=MODEL,
        )


def fakes(
    profile: Profile,
    limits: Limits,
    *,
    web: Any,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, GenerationModel | DecisionModel]:
    return {
        "lead": Team("lead", profile, sleep),
        "flat": Team("flat", profile, sleep),
        "captain": Team("captain", profile, sleep),
        "surveyor": Surveyor(profile, sleep, web),
        "crosscheck": Crosscheck("crosscheck", profile, sleep, None),
        "crosscheck_second": Crosscheck("crosscheck_second", profile, sleep, limits),
        "boss": Boss(profile, sleep),
        "gate": FakeGate(),
    }
