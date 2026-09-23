# An engine agent as a tool — implementation plan

> For agentic workers: execute task by task, tests first. Steps use checkboxes (`- [ ]`).

**Goal:** `engine.tool(agent, *, scope, ...)` hands any runtime — a live model in rfnry-senses, a
loop of the consumer's own, another engine run — an engine agent as one tool it can call.

**Architecture:** one public method on `Engine`. It resolves the agent and the scope up front and
returns a `Tools` holding one `Tool` whose body is `self.run(...)`. No new module, no new public
name, no new record shape: the call is an ordinary top-level run, recorded like any other.

**Tech stack:** Python 3.12, pydantic v2, pytest (asyncio auto), mypy strict, ruff; `uv run poe`.

**Spec:** `rfnry/senses/docs/plans/2026-09-24-senses-design.md` §10 ("Engine addition: an agent as
a tool"), used by §7 (the two-minds pattern).

## Decisions

```python
engine.tool(agent, *, scope, name=None, description=None, context=None, tools=None, workspace=None) -> Tools
```

1. **Input schema: `{"input": str}`, required, no parts.** The caller is a model emitting JSON tool
   arguments; it has no way to produce an `Image` or a `Document` part, and a runtime that does
   hold media calls `engine.run(input=parts)` directly. Validated by a pydantic args model, so a
   bad call is `InputError: tools: invalid arguments for consult: ...` before any model is called.
2. **Name and description default to the tree:** the agent's name and its `AGENT.md`
   `description`. `name=` and `description=` override them (`"consult"`, `"Ask the reasoning
   mind."`). An empty string is taken as given for `description`; only `None` falls back.
3. **Returns: the reply or the answers.** A generation agent's tool returns `run.reply` (a `str`).
   A decision agent's returns its answers as a dict, `{qid: answer.model_dump()}` — the same
   content `Delegate` gives a model, as a dict instead of a JSON string because `Result` carries a
   dict and the runtime serialises it. A decision agent that routes returns its own answers, not
   the routed agent's reply — `Delegate` parity; the routed run is in the record.
4. **Returns a `Tools`, not a `Tool` (deviation from §10).** §10 says `-> Tool`, but §7 writes
   `consult + ThreadTools(...)` and `Session(tools=...)` takes a `ToolsLike`. A `Tool` has no
   `__add__` and is not a `ToolsLike`, so neither line would work. Giving `Tool` an `__add__`
   changes the tools block that `test_family.py` holds byte-identical with rfnry-memory — a
   two-repository change for one operator. A `Tools` of one tool already composes (`+`, `only`,
   `without`), passes wherever a `ToolsLike` goes (`run(tools=)`, a senses `Session`), and
   dispatches by name. The senses design's §10 signature is corrected to `-> Tools` when senses is
   planned.
5. **`tools=` and `workspace=` ride along (deviation from §10).** The docs' rule is that tools
   belonging to a tenant go on the call that names the tenant; `engine.tool` is that call. A
   consulted agent that needs `MemoryTools` for its tenant, or declares `workspace: true`, gets
   them exactly as `Delegate` dispatched through `EngineTools` forwards them. `history`, `task`,
   `expect` and `on_event` are not taken: a tool call is one stateless exchange, and the caller
   keeps its own conversation.
6. **It records, always.** The body calls `self.run(...)` with the default `record=True`; there is
   no `record=` parameter — a replay or evaluation calls `run(record=False)` directly, never
   through a model's tool call. The run is top-level: `parent_run_id` is `None`.
7. **Deadlines and nesting.** From a live model the call is a top-level run: the engine puts no
   deadline on it (the runtime owns its own), and it is bounded by `max_steps` like any run. Mounted
   inside another engine run through `run(tools=)`, it is a tool like any other: under
   `tool_timeout`, recorded as its own top-level run, absent from the parent's `children`, its step
   without a `child_run_id`. Lineage and the deadline exemption are what `agents:` and `Delegate`
   are for. The loop and depth guards still hold across the boundary, because the delegation stack
   is a context variable the tool call inherits: `billing` calling itself through `engine.tool` is
   `DelegationLoopError: delegate: loop at billing -> billing`, read by the model as a tool error.
8. **Errors, rule 9.** At construction: an empty scope is `InputError("engine: scope is
   required")`, an unknown agent `NotFoundError("engine: agent 'ghost' was not found")` — found
   when mounting, not on a live model's first call. Bad arguments: `InputError("tools: invalid
   arguments for <name>: ...")`. Everything `run` raises (a missing model's `ConfigurationError`,
   `RunawayError`, the nesting errors) propagates out of `dispatch` unchanged, for the runtime to
   report; inside an engine run it becomes the model's tool error with its type in front. A
   duplicate name on `+` is the existing `ConfigurationError("tools: duplicate tool name
   'consult'")`.

## Pre-change checklist

1. **Wanted two generations ahead?** Yes. Stronger live models consult deeper agents more, not
   less; the method is the contract between a runtime and an agent, not a workaround.
2. **More or less useful as models improve?** More — substrate: a contract (agent as a callable
   tool) and a record (every consultation is a recorded run under its scope).
3. **If less, a structural boundary?** Not applicable; it is substrate.
4. **Exists because a model is weak at X?** No. It lets a fast live model reach a slower, deeper
   agent; nothing polices or re-scores either.
5. **Generation only, decision only, both?** All three. A generation agent's tool needs a
   generation model and returns text; a decision agent's needs a decision model and returns
   answers; an engine with one model mounts the agents of that kind, and mounting the other kind
   fails on call with the `ConfigurationError` `run` already raises.

## Global constraints

- `scope` required and non-empty (engineering rule 8); no default scope.
- No comments or docstrings (rule 4). Error messages start with `engine:` / `tools:` (rule 9).
- No new public name; `__all__` and `test_public_api.EXPECTED` unchanged. `Engine`'s public
  members become `{"agents", "run", "tool"}`.
- `tools/tools.py` untouched, so the family byte-identity test stays green without touching
  rfnry-memory.
- `docs/engine.md` changes in the same commit as the code (rule 5). Not a breaking change.

## Review focus

- Decisions 4 and 5 deviate from §10's signature. If the reviewer prefers `-> Tool`, the
  alternative is `Tool.__add__` in engine and memory together, plus `Tools([tool])` wherever a
  `ToolsLike` is needed.
- The tool inside another run is not a child (decision 7). Confirm that is wanted rather than
  threading a `parent_run_id` through — it keeps one nesting mechanism, `Delegate`.
- The decision return is a dict of `model_dump()`s, not `Answer` objects: what a model reads.

## Task 1 — `Engine.tool`

**Files:** create `packages/python/tests/unit/test_engine_tool.py`; modify
`packages/python/src/rfnry_engine/engine.py`, `packages/python/tests/unit/test_public_api.py`.

- [ ] **Step 1: write the failing tests** — `tests/unit/test_engine_tool.py`:

```python
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, NotFoundError
from rfnry_engine.models import Answer, CheckAnswer, Reply, ScoreAnswer, ToolCall
from rfnry_engine.records import Generation, Run
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import TOOLS, Fixed, Scripted, text_reply

TRIAGE: dict[str, Answer] = {"urgent": CheckAnswer(probability=0.9), "severity": ScoreAnswer(score=3, probabilities=[0, 0, 0, 1])}


class Recorded:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


def calling(name: str, text: str) -> Reply:
    return Reply(tool_calls=[ToolCall(id="c1", name=name, args={"input": text})], stop_reason="tool_use")


async def test_a_generation_agent_as_a_tool_replies_and_records(fixtures: Path) -> None:
    llm = Scripted(text_reply("Refunded."))
    recorder = Recorded()
    engine = Engine(fixtures, generation=llm, tools=TOOLS, refinery=recorder)
    billing = engine.tool("billing", scope="acme", context="Acme is on the enterprise plan.")
    [definition] = billing.definitions()
    assert definition["name"] == "billing" and definition["description"] == "Answers billing questions"
    assert definition["input_schema"]["required"] == ["input"]
    assert await billing.dispatch("billing", {"input": "Charged twice"}) == "Refunded."
    [run] = recorder.runs
    assert isinstance(run, Generation) and run.agent == "billing" and run.scope == "acme"
    assert run.parent_run_id is None and run.input == "Charged twice"
    assert "Acme is on the enterprise plan." in llm.calls[0][0]


async def test_a_decision_agent_as_a_tool_returns_its_answers(fixtures: Path) -> None:
    engine = Engine(fixtures, decision=Fixed(TRIAGE))
    judge = engine.tool("triage", scope="acme", name="judge", description="Judge urgency.")
    assert judge.names == ["judge"] and judge.definitions()[0]["description"] == "Judge urgency."
    answers = await judge.dispatch("judge", {"input": "It is down"})
    assert isinstance(answers, dict) and set(answers) == {"urgent", "severity"}
    assert answers["urgent"]["probability"] == 0.9 and answers["severity"]["score"] == 3


async def test_the_tool_composes_and_refuses_bad_calls(fixtures: Path) -> None:
    engine = Engine(fixtures, generation=Scripted(), tools=TOOLS)

    async def ping(args: dict[str, Any]) -> str:
        return "pong"

    surface = engine.tool("billing", scope="acme", name="consult") + Tools([Tool("ping", "Ping", body=ping)])
    assert surface.names == ["consult", "ping"]
    with pytest.raises(InputError, match="^tools: invalid arguments for consult"):
        await surface.dispatch("consult", {})
    with pytest.raises(ConfigurationError, match="^tools: duplicate tool name 'consult'"):
        surface + engine.tool("support", scope="acme", name="consult")
    with pytest.raises(InputError, match="^engine: scope is required"):
        engine.tool("billing", scope="")
    with pytest.raises(NotFoundError, match="^engine: agent 'ghost' was not found"):
        engine.tool("ghost", scope="acme")


async def test_inside_a_run_it_is_its_own_top_level_run(fixtures: Path) -> None:
    llm = Scripted(calling("consult", "Charged twice"), text_reply("Refunded."), text_reply("Done."))
    recorder = Recorded()
    engine = Engine(fixtures, generation=llm, tools=TOOLS, refinery=recorder)
    run = await engine.run("support", "acme", "Help", tools=engine.tool("billing", scope="acme", name="consult"))
    assert isinstance(run, Generation) and run.reply == "Done." and run.children == []
    assert run.steps[0].result == "Refunded." and run.steps[0].child_run_id is None
    assert [r.agent for r in recorder.runs] == ["billing", "support"]
    assert recorder.runs[0].parent_run_id is None


async def test_the_loop_guard_holds_across_the_tool(fixtures: Path) -> None:
    llm = Scripted(calling("billing", "again"), text_reply("done"))
    engine = Engine(fixtures, generation=llm, tools=TOOLS)
    run = await engine.run("billing", "acme", "hi", tools=engine.tool("billing", scope="acme"))
    assert isinstance(run, Generation) and run.steps[0].is_error
    assert run.steps[0].result == "DelegationLoopError: delegate: loop at billing -> billing"
```

In `tests/unit/test_public_api.py`, `test_the_engine_shows_its_agents_and_hides_its_internals`
becomes:

```python
    assert public == {"agents", "run", "tool"}
```

- [ ] **Step 2: run them and watch them fail** —
  `cd packages/python && uv run pytest -q tests/unit/test_engine_tool.py tests/unit/test_public_api.py`.
  Expected: `AttributeError: 'Engine' object has no attribute 'tool'`, and the public-API set
  mismatch.

- [ ] **Step 3: implement** — in `engine.py`, import `Field` from pydantic and `Result`, `Tools`
  from `rfnry_engine.tools.tools`; add the args model beside `AgentInfo`:

```python
class AgentInput(BaseModel):
    input: str = Field(description="What to hand the agent")
```

  and the method after `run`:

```python
    def tool(
        self,
        agent: str,
        *,
        scope: str,
        name: str | None = None,
        description: str | None = None,
        context: str | None = None,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
    ) -> Tools:
        if not scope:
            raise InputError("engine: scope is required")
        spec = self._spec(agent)

        async def body(args: dict[str, Any]) -> Result:
            run = await self.run(agent, scope, args["input"], context=context, tools=tools, workspace=workspace)
            if isinstance(run, Generation):
                return run.reply
            return {qid: answer.model_dump() for qid, answer in run.answers.items()}

        described = spec.description if description is None else description
        return Tools([Tool(name or spec.name, described, body=body, args=AgentInput)])
```

- [ ] **Step 4: run them and watch them pass** — the same command; expected all green.

## Task 2 — the contract in `docs/engine.md`

**Files:** modify `docs/engine.md` (Tools section only). No At-a-glance change: no public name is
added; `tool` is a method of `Engine`, listed under "Run things" already.

- [ ] **Step 1: add, at the end of `## Tools`, after the `Delegate dispatched this way` paragraph:**

````markdown
An agent is itself a tool. `engine.tool(agent, scope=...)` gives any runtime — a live voice model,
a loop of your own — one tool that runs the agent: named for the agent and described by its
`AGENT.md` unless you say otherwise, taking `{"input": str}`, returning a generation agent's reply
or a decision agent's answers as a dict:

```python
consult = engine.tool("billing", scope="acme", name="consult", description="Ask the billing desk.",
                      context="Acme is on the enterprise plan.")
reply = await consult.dispatch("consult", {"input": "Was I charged twice?"})
answers = await engine.tool("triage", scope="acme").dispatch("triage", {"input": "It is down"})
answers["urgent"]["probability"]                           # 0.91
surface = consult + MemoryTools(memory, corpus_id="acme")
```

It is a `Tools` of one, so it adds to any other. Each call is a top-level run under the scope it
was made with, recorded like any run; `context=`, `tools=` and `workspace=` ride every call.
Mounted inside another run with `run(tools=)` it is a tool like any other, under `tool_timeout`,
and its run is not the caller's child — to nest with lineage, name the agent in `agents:`. A loop
through it is still refused.
````

- [ ] **Step 2: run the docs tests** —
  `uv run pytest -q tests/unit/test_docs_examples.py tests/unit/test_docs.py`.
  The new block runs under `test_docs_example_runs` with the default `Setup()`: the docs tree has
  `billing` (generation, the `Model` fake replies `"ok"`) and `triage` (decision, `Answering`
  gives `urgent` 0.91), and `MemoryTools`/`memory` are in the namespace. Its first line is unique,
  so `test_every_named_block_is_in_the_docs` holds; no `SETUPS`, `ILLUSTRATIVE` or `SIGNATURES`
  entry is needed.

## Gate

From `packages/python`, no skips:

```bash
uv run poe check:fix && uv run poe check && uv run poe typecheck && uv run poe test
RFNRY_TEST_POSTGRES_URL=postgresql://harness:harness@127.0.0.1:55435/harness uv run poe test:integration
```

The Postgres tier is unaffected by design (no storage change) and runs to prove it. The family
test runs against the sibling `../memory` checkout and must stay green — it does, because
`tools/tools.py` is untouched.

## Commit

One commit, code and docs together:

```
feat: engine.tool — an engine agent as a tool

engine.tool(agent, *, scope, name, description, context, tools, workspace) returns a
Tools of one tool taking {"input": str}; its body is engine.run, recorded as a
top-level run; it returns a generation agent's reply or a decision agent's answers.
A Tools, not a Tool, so it composes with + and passes as a ToolsLike without
changing the tools block shared with rfnry-memory. Not breaking.

Claude-Session: https://claude.ai/code/session_01GevsYJaAyM8ARLvEnXsoAB
```

Push to `origin main`. Then the senses design's §10 signature reads `-> Tools`, and its §7 example
gains the required `scope=` (`engine.tool("core", scope=scope, name="consult", ...)`).
