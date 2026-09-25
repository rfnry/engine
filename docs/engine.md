# rfnry-engine

rfnry-engine runs agents that are directories of markdown.

An agent is one of two kinds. A **generation agent** puts a language model in a loop with tools
until it has an answer. A **decision agent** asks a decision model a set of typed questions about a
state and gets typed, probabilistic answers back in one call. Agents nest in both directions: a
generation agent can delegate to any agent, a decision agent can route to any agent. Opt-in, a
refinery records what your agents did and exports it as datasets, with every label and its
source — from people, the world and models. What you do with them — evaluate, improve prompts and
rules, train or distill models, audit — is yours.

## Philosophy

**The tree is the agent.** Everything a person needs to author, review or diff about an agent is a
markdown file under version control. The library reads the tree; it never writes it.

**Two kinds of model, one or both.** Open-ended work — conversation, tool use, writing — is
generation. Closed-set judgement — which of these, how much, is this true — is decision. A
generation model does the first well and the second expensively; a decision model does the second
in milliseconds with calibrated probabilities and cannot do the first at all. The library treats
them as two contracts and lets you build an agent from either or both.

**The library runs; it does not take over.** You bring the models, the conversation history, the
bodies behind tools, the storage connection, the bucket. There is no background loop, no plugin
registry, no configuration language. Every verb is a function you call.

**Nothing is hidden.** A run returns a record: what was sent, what came back, every tool call with
its timing, every model call with its usage, every child run. That record is what you feed your own
telemetry, and it is what the refinery stores and exports — everything that could be useful, for
any purpose, without assuming one.

**Tenancy by construction.** Every run names its scope. Every stored row carries it. There is no
default scope and nothing reads across scopes. Multi-tenancy is the default.

## Models

The library speaks to models through two protocols. Any object with the right method is a model;
adapters to vendors are yours to write and never ship here.

```python
class GenerationModel(Protocol):
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ReplyLike: ...

class DecisionModel(Protocol):
    async def decide(self, state: State, questions: dict[str, Question]) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

Models are matched by shape, statically and at runtime: `ReplyLike`, `DecidedLike` and `AnswerLike` are
protocols over the fields of `Reply`, `Decided` and the answers below, and a result is read by those
fields, not its class. An adapter typed against another library's copy of these contracts —
rfnry-memory's — fits here unchanged, so one adapter serves both libraries; a result whose fields
do not fit is a `ModelError`.

`Message`, `ToolSpec`, `ToolCall` and `Reply` are the message shapes the loop exchanges with a
generation model. An assistant turn is a `Message(role="assistant", content=…, tool_calls=[…])`:
its text in `content`, its calls as `ToolCall`s. A tool result is a
`Message(role="tool", tool_call_id=…, content=…)` whose `is_error` says the call failed, for
adapters whose vendor marks failed tool results. A `Reply`
whose `stop_reason` is `"tool_use"` and whose `tool_calls` is non-empty means "run these and call
me again"; anything else is final:

```python
Reply(text="", tool_calls=[ToolCall(id="c1", name="Read", args={"path": "rules/refunds.md"})], stop_reason="tool_use")
Reply(text="Refunded the second charge.", stop_reason="end_turn", usage=Usage(input=812, output=41), model="…")
```

A message's `content` is a string or a list of parts — one vocabulary for everything a model can
see, shared by every library in the family:

```python
Text(text="Is this invoice charged twice?")
Image(media_type="image/png", data=png)
Document(media_type="application/pdf", name="invoice.pdf", url="https://example.com/invoice.pdf")
Audio(media_type="audio/wav", data=wav)
```

`Part` is any of the four and `Content` is `str | list[Part]`. A media part carries exactly one of
`data` (bytes) or `url`; both or neither is an `InputError`. Bytes travel as base64 in JSON. The
library never fetches a `url` — it carries it to your adapter, which decides; turning parts into a
vendor's blocks, and refusing what the vendor cannot take, is the adapter's job. There is no video
part: frames travel as images. Parts are matched by shape too, so another family library's copy of
them is accepted as it is.

Structured output is always a forced tool call, so an adapter needs nothing beyond tool support. An
adapter that also implements `generate_stream(system, messages, tools, on_delta)` streams text when
the caller asks for events.

A decision model answers three kinds of question:

```python
Choice(instructions="Which department should handle this?", options={"billing": "…", "technical": "…"})
Score(instructions="How urgent is the request?", levels=["Can wait days", "Today", "Within the hour", "Outage"])
Check(instructions="Does the user ask to speak with a person?")
```

and returns, respectively, a `ChoiceAnswer` (`choice`, `probabilities`), a `ScoreAnswer` (`score`,
`probabilities`), or a `CheckAnswer` (`probability`). Each carries a `confidence` in `[0, 1]` that
the library computes from the distribution's peakedness, `(n·p_max − 1)/(n − 1)` — the same
formula for every adapter and every type, so a threshold means the same thing everywhere. A
generation model can stand in as a decision model through an adapter that asks it for
probabilities; the calibration is worse, the code is the same. `decide` may return the answers
alone, or a `Decided(answers, usage, model)` when the adapter can report what the call cost and
which model answered; the record carries both.

Fakes are enough to run everything in tests:

```python
class Scripted:
    def __init__(self, *replies): self.replies = list(replies)
    async def generate(self, system, messages, tools): return self.replies.pop(0)

class Fixed:
    async def decide(self, state, questions):
        return {"intent": ChoiceAnswer(choice="billing", probabilities={"billing": 0.8, "technical": 0.15, "other": 0.05}),
                "urgent": CheckAnswer(probability=0.91)}
```

## Getting started

```python
engine = Engine(root=Path("agents"), generation=llm, tools=tools)

run = await engine.run("support", scope="acme", input="My invoice was charged twice.")
print(run.reply)
```

A root of agent directories, a generation model, and `tools` — the bodies behind the tools the
agents declare, one list for the whole tree ([Tools](#tools) builds it). One call runs an agent for
a tenant and returns its record.

## Core features

### An agent is a directory

```
agents/
  support/
    AGENT.md              who the agent is
    INDEX.md              a map of the directory, read at boot
    rules/_index.md       what the agent must do — the index is read at boot, the files on demand
    rules/refunds.md
    skills/_index.md      what the agent can do — the index at boot, a skill's body when loaded
    skills/escalate.md
    knowledge/_index.md   what the agent knows — the index at boot, the files on demand
    tools/lookup_order.md
    tools/read_pdf.md
    tasks/triage.md       a named task, chosen per run
```

`AGENT.md` makes a directory an agent. Its frontmatter names the kind and what the agent may call;
its body is the persona:

```markdown
---
name: support
kind: generation
description: Handles technical support conversations
agents: [triage]
---
You are the support desk for Acme. Be brief, be exact, cite the rule you are applying.
```

At boot the model receives `AGENT.md`, `INDEX.md`, the three `_index.md` files, the list of tools it
has, the descriptions of the agents it may call, the task if one was named, and any context the
caller passes. Everything else it reads when it needs it, with the `Read`, `Grep`, `Glob`, `LS` and
`Skill` tools, and it can read only inside its own directory. Write the maps with full paths —
`rules/refunds.md`, not `refunds.md` — because a model opens exactly what the map names.
`engine.agents` lists the tree as the engine loaded it: each name to an `AgentInfo` with its
`kind`, `description` and the `agents` it may call.

### Running an agent

`engine.run(agent, scope, input)` names the agent, the tenant and the input, and returns a record.
For a generation agent that record is a `Generation`: the reply, the system prompt as sent, the messages as sent to the first model call, the whole conversation, every tool
call as a `Step` with its result and timing, the model calls and their summed usage, and any child
runs. It is the whole observability payload; hand it to your own telemetry:

```python
run.run_id, run.model, run.calls, run.usage.input, run.usage.output, run.duration_ms
for step in run.steps:
    step.tool, step.args, step.is_error, step.duration_ms, step.child_run_id
run.children                                         # runs made through Delegate
```

Keep the conversation yourself and pass it back. `run.conversation` is the history you passed, the
input, every assistant turn — its text in `content`, its calls in `tool_calls`, its `reasoning` as
the model returned it — every tool result with its `is_error`, and the final assistant turn, so the
next run continues from what the agent did, not only what it said:

```python
run = await engine.run("support", scope="acme", input="My invoice was charged twice.")
next = await engine.run("support", scope="acme", input="And the second charge?", history=run.conversation)
```

The input is a string, a JSON object, or a list of parts — what the user sent, as they sent it:

```python
parts = [Text(text="Is this invoice charged twice?"), Image(media_type="image/png", data=png)]
run = await engine.run("support", scope="acme", input=parts)
```

A list of parts becomes the first user message as it is; parts in `history` reach the model
unchanged, and a later run continues from them. History, like input, is read by shape: a message or
part from another library is read by its fields.

Pass `record=False` to run without the refinery recording it — a replay, an evaluation, a dry run.

Ask for a typed reply and the model is made to produce one:

```python
class Verdict(BaseModel):
    refund: bool
    reason: str

run = await engine.run("support", scope="acme", input=message, expect=Verdict)
verdict = Verdict.model_validate_json(run.reply)
```

Name a task to put `tasks/<task>.md` in front of the model, pass `context=` to add facts about this
tenant that do not belong in the tree, and pass `on_event=` to receive text deltas, tool starts and
tool results as they happen:

```python
async def show(event):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)

run = await engine.run("support", scope="acme", input=message, task="triage",
                       context="Acme is on the enterprise plan. Refunds up to $500 need no approval.",
                       on_event=show)
```

### Decision agents

A decision agent is a directory of questions:

```
agents/
  triage/
    AGENT.md
    questions/urgent.md
    questions/severity.md
```

```markdown
---
type: check
yes: The service is down or data is at risk
no: A question, a request, or a problem with a workaround
---
Does the message need a person right now?
```

```markdown
---
type: score
levels: [Can wait days, Today, Within the hour, Outage]
---
How severe is the problem?
```

A `check` may say with `yes:` and `no:` what each answer means; a `score` lists `levels:` in order;
a `choice` lists `options:`, each key with what it means.

Run it with a state instead of a message. The state can be a string, an object or a list. A
decision model reads text: a list of parts made only of `Text` is joined with newlines into the
state, and any other part is an `InputError` before the model is called
(`engine: decision agent 'triage' takes text only; got image`).

```python
engine = Engine(root=Path("agents"), decision=dm)

run = await engine.run("triage", scope="acme", input={"message": text, "plan": "enterprise"})
run.answers["urgent"].probability     # 0.91
run.answers["severity"].score         # 3.0
run.answers["severity"].confidence    # 0.87
```

Every question in the directory is answered in the same call, in parallel and in isolation: adding
a question changes neither the others' answers nor, noticeably, the latency. An engine with only a
decision model needs no tool bodies; the tree's generation agents are the ones it cannot run.

Everything beyond the answers — weighing several, confirming with a person below a threshold,
escalating — is your code over the returned `Decision`. The library gives you the answers; it does
not make the call.

### Nesting

`agents:` in a generation agent's frontmatter gives it a `Delegate` tool listing those agents by
name and description. The model calls it with an agent and an input; the child runs under the same
scope and its reply — text for a generation agent, answers for a decision agent — comes back as the
tool result. The child's record is attached to the parent's `children`, and the step that called it
carries the child's `run_id`. A decision agent used this way is how a generation agent asks a fast,
calibrated question mid-task: is this destructive, which skill applies, is the user done.

Depth is capped (`max_depth`, default 5) and a cycle is refused; both are reported to the model as
tool errors, and the run continues.

### Both kinds together

```
agents/
  frontdesk/    kind: decision   route: {by: intent, to: {billing: billing, technical: support}}
  support/      kind: generation agents: [triage]
  billing/      kind: generation
  triage/       kind: decision   questions: urgent (check), severity (score)
```

A decision agent can route. One choice question, one table, an optional confidence floor:

```markdown
---
name: frontdesk
kind: decision
description: Routes an incoming message
route:
  by: intent
  to: {billing: billing, technical: support}
  min_confidence: 0.6
---
```

```markdown
---
type: choice
options:
  billing: Payments, refunds, invoices
  technical: The product misbehaves
  other: Anything else
---
Which department should handle this message?
```

When the chosen option maps to an agent and confidence clears the floor, that agent runs with
everything the run was given — the same input, parts included, `history`, `context`, `expect`, `tools`, `workspace`
and `on_event`; a decision agent itself takes no `task`. The routed run is attached:

```python
engine = Engine(root, generation=llm, decision=dm, tools=tools)

run = await engine.run("frontdesk", scope="acme", input=message, history=history)
run.answers["intent"].choice            # "technical"
run.routed_to.agent                     # "support" — the generation agent that answered
run.routed_to.children[0].agent         # "triage" — the decision agent support consulted mid-task
```

One message, four model calls, one record. With only `generation=` the same tree runs its
generation agents and refuses the decision ones; with only `decision=` the reverse, and a decision
agent whose route reaches a generation agent is refused before its questions are asked. Nothing
about an agent changes when the other kind of model is absent — only which agents you can run.

### Workspace

> `Workspace` gives the model a real shell on the machine where it runs. It can do anything that
> machine's user can do — read files, reach the network, delete things. Run it where that is
> acceptable. Keep the agent directory outside the workspace root.

A workspace is a directory you own where an agent writes code, runs it, and tries again. An agent
opts in with one line in its `AGENT.md`, and a run supplies the directory:

```
agents/coder/AGENT.md
---
name: coder
kind: generation
description: Writes and runs code to finish a task.
workspace: true
---
```

```python
from rfnry_engine import Workspace

run = await engine.run("coder", scope="acme", input="make the failing test pass",
                       workspace=Workspace("/srv/tenants/acme/work"))
```

The agent then has three more tools: `Shell(command)` runs a command in the directory — its
description tells the model it is already there, so relative paths work — and returns
`exit_code`, `stdout` and `stderr`, each capped at 30,000 bytes with head and tail kept; `Write(path, content)` writes a file exactly; `Edit(path, old, new)`
replaces one exact occurrence. `Write` and `Edit` stay inside the directory; `Shell` is a shell and
can reach anything the service's user can. They are for exact edits and a clean record, not a
security boundary — the isolation is wherever you run the service.

An agent without `workspace: true` never sees these tools, even when the run has a workspace; an
agent with it, run without one, is a `ConfigurationError`, and so is a decision agent that declares it. Delegated and routed agents receive the
run's workspace on the same terms. Commands run with only `PATH`, `HOME` and `LANG` — the service's
own environment, where its keys live, is not passed unless you pass `env=`. Every call is bounded by
`Engine(tool_timeout=)`, and when it expires the command and everything it started are killed — except
a process that leaves the command's process group with `setsid`, which is out of reach and, if it keeps
the output open, holds the call until the deadline. You own the directory and its lifetime: a
temporary one per run, or one kept for a project.

### Refining

The refinery records and exports what your agents did, with every label and its source. What you do
with it — evaluate, improve prompts and rules, train or distill models, audit — is yours.

Attach a refinery and every run is recorded. A recording that fails is logged as a warning and never
fails the run.

```python
refinery = Refinery(PostgresStorage(pool))
await refinery.initialize()
engine = Engine(root, generation=llm, decision=dm, tools=tools, refinery=refinery)
```

Labels come from three places. A person reviews a run — the button under a reply, the correction a
supervisor types, the right answer to a question that was decided wrong:

```python
await refinery.review("acme", run.run_id, Review(stance="objection", correction="Refund the second charge in full."))
await refinery.review("acme", decision.run_id, Review(correction={"intent": "technical"}))
```

The world reports whether the action worked:

```python
await refinery.outcome("acme", run.run_id, Outcome(status="failed", detail="Customer escalated."))
```

And models label what people did not. A **judge** is a decision agent in your tree whose questions
are about an episode; it sees the input, the reply and every tool call with its result, with each image,
audio or document replaced by a placeholder naming it — `[image: image/jpeg]`, `[document: application/pdf, "policy.pdf"]` —
while the recorded run and the exported datasets keep the media:

```
agents/judge/
  AGENT.md                 kind: decision
  questions/grounded.md    type: check   Is every claim in the reply supported by a tool result?
  questions/rule.md        type: check   Does the reply apply the refund rule as written?
  questions/done.md        type: score   levels: [Not addressed, Partly, Fully, Fully and confirmed]
```

A **teacher** is a generation agent that answers a decision agent's questions with a rationale, so
a decision made without ground truth still gets one. Both run when you say so, over runs they have
not labeled yet:

```python
labeled = await refinery.label("acme", engine, judge="judge", teacher="teacher")
labeled.judged, labeled.taught, labeled.errors
```

Every label carries its source. A judge's score is exported as a judge's score, never as a person's.

Export writes datasets to any object with a `put(key, bytes)`:

```python
class Folder:
    def __init__(self, root): self.root = Path(root)
    async def put(self, key, data):
        (self.root / key).parent.mkdir(parents=True, exist_ok=True)
        (self.root / key).write_bytes(data)

exported = await refinery.export("acme", Folder("datasets"), since=last_watermark)
exported.keys, exported.rows, exported.watermark
```

Keys are `{scope}/{agent}/{dataset}/{timestamp}.jsonl`, so repeated exports add
files and never overwrite. `since` selects every run that is new or got a label after it, written in full
with all its labels; the watermark covers both, so a review of a run you already exported reaches the
next export. Rows are keyed by `run_id`: when a run appears again, its latest row supersedes the
earlier ones — dedupe by `run_id`, keep the latest.

Every row carries `"holdout": true|false` — the rows kept aside so any measurement on them is
honest. It is derived from a hash of the run id (`holdout_percent=20` by default), so a run's flag
never changes between exports; `is_holdout(run_id)` gives the same answer in your own code. Three
datasets:

| dataset | one row per | what it holds |
| --- | --- | --- |
| `generation` | generation run | system, messages, steps, reply, usage, every label |
| `preference` | human correction | prompt, chosen (the correction), rejected (the reply) |
| `decision` | question of a decision run | state, question, answer, probabilities, confidence, truth, outcome |

The first two are a generation agent's episodes and the corrections people made to them. Parts
are written inline, their bytes as base64, so a row is self-contained; strip the media from a row
you do not want it in. The third
is a decision agent's answers beside what turned out true: a calibrated model is one whose 0.8 was
right 80% of the time, and this is the table that says whether it was:

```json
{"run_id": "…", "holdout": false, "agent": "frontdesk", "question_id": "intent", "answer": "billing",
 "probabilities": {"billing": 0.8, "technical": 0.15, "other": 0.05}, "confidence": 0.7,
 "truth": "technical", "truth_source": "human", "outcome": "failed"}
```

`refinery.conflicts(scope)` lists runs that carry two different human corrections — a dataset can
hold either, not both, and the library reports rather than guesses.

## Tools

One concept, one word. A **tool** is a name, a description, an input schema and a body. A
**`Tools`** is a set of them: what a model receives, and what your code can call the same way.

```python
class Tools:
    def definitions(self) -> list[dict]                 # [{name, description, input_schema}]
    async def dispatch(self, name: str, args: dict) -> str | dict | list[Part]
    def __add__(self, other) -> Tools                    # union; a duplicate name is refused
    def only(self, *names) -> Tools
    def without(self, *names) -> Tools
```

An agent's tools are declared in its directory. `Read Grep Glob LS Skill` come with every generation agent,
`Delegate` comes with `agents:`; everything else is a file under `tools/`:

```markdown
---
name: lookup_order
description: Fetch an order by id
input:
  order_id: {type: string, description: The order id, required: true}
---
```

```markdown
---
name: read_pdf
---
```

The first declares the contract the model sees and leaves the body to code. The second adopts a
code tool as it is — description and schema included. Bodies and code tools go on the engine, one
list for the whole tree. A body returns a string, a dict (the model reads it as JSON) or a list of
parts, which reach the model as the tool result — a PDF the model reads itself, an image it looks
at. `Jail` resolves a path the way the builtins do, and refuses one that leaves the directory:

```python
class PdfArgs(BaseModel):
    path: str

async def read_pdf(args):
    path = Jail(root / "support").resolve(args["path"])
    return [Document(media_type="application/pdf", name=path.name, data=path.read_bytes())]

tools = [
    Tool("lookup_order", body=lookup_order),
    Tool("read_pdf", "Read a PDF in the agent directory", body=read_pdf, args=PdfArgs),
]
engine = Engine(root, generation=llm, tools=tools)
```

A tool the engine was given that no agent declares is a `ConfigurationError`; so is a declaration
with no body, when the engine has a generation model to run it. So an agent's whole tool surface
is readable from its directory, and the engine cannot hand a model something the tree does not
show — except the `Tools` a caller passes to `run(tools=)`, which ride that one run.

Every tool call inside a run has a deadline, `Engine(tool_timeout=30.0)` by default and `None` to
turn it off. When it passes, the model gets the error `timed out after 30s` and the run continues.
Bodies are async and the deadline cancels them, so a body that runs blocking work in a thread must
put its own limit on that work: cancelling the await does not stop the thread. `Delegate` has no
deadline. The child run it starts is bounded by its own tool calls and `max_steps`.

The builtins are bounded whatever the deadline. They read UTF-8 text only: `Read` refuses a binary
file and `Grep` skips it. Neither reads a file over 5 MB. `Grep` stops after scanning 50 MB and
notes the cut. A search that runs past 10 s in one call is an `InputError` the model can act on. `Glob` and
`Grep` walk at most 100,000 entries. For other formats, declare a tool, as `read_pdf` does.

Tools that belong to a tenant go on the call that names the tenant:

```python
run = await engine.run("support", scope="acme", input=text,
                       tools=MemoryTools(memory, corpus_id="acme"))
```

That is how rfnry-memory mounts: its `MemoryTools` is a `Tools`, and neither library imports the
other. The set an agent runs with is itself a first-class object, so you can drive it without a
model — from a test, or from a loop of your own:

```python
surface = EngineTools(engine, agent="support", scope="acme")
surface.definitions()                                        # what the model would see
await surface.dispatch("Read", {"path": "rules/refunds.md"})
await surface.dispatch("Delegate", {"agent": "triage", "input": "It is down"})

combined = MemoryTools(memory, corpus_id="acme") + surface.only("Read", "Grep")
```

A `Delegate` dispatched this way runs the child as its own top-level run, with the same scope,
`tools=` and `workspace=` a `Delegate` inside a run would give it.

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

## Storage

The refinery is the only part of the library that stores anything. Two storages ship: `InMemoryStorage()`
for tests and small things, `PostgresStorage(pool)` for everything else, taking an asyncpg pool you
built. `initialize()` creates two tables, `engine_runs` and `engine_labels`, and is safe to run
again. The tables are prefixed so the same database can hold rfnry-memory's:

```python
pool = await asyncpg.create_pool(dsn)
memory = Memory(rfnry_memory.PostgresStorage(pool), embedding=emb)
refinery = Refinery(rfnry_engine.PostgresStorage(pool))
```

## Tenancy

`scope` is the tenant. It is required on every run and every refinery verb, it is the first column
of every table, and no read accepts its absence. Tools pinned to a tenant are passed on the call.
`refinery.delete_scope(scope)` removes everything; `refinery.verify_absent(scope)` proves it.

## At a glance

Everything the package exports, by role.

- **Run things:** `Engine`, `AgentInfo`.
- **What a run returns:** `Generation`, `Decision`, `Run`, `Step`; events `TextDelta`, `ToolCallStarted`, `ToolResultReady`, `Event`.
- **Speak to models:** `GenerationModel`, `StreamingGenerationModel`, `DecisionModel`; `Message`, `ToolSpec`, `ToolCall`, `Reply`, `Usage`, `State`; parts `Text`, `Image`, `Audio`, `Document`, `Part`, `Content`; `Choice`, `Score`, `Check`, `Question`; `ChoiceAnswer`, `ScoreAnswer`, `CheckAnswer`, `Answer`, `Decided`; `confidence`.
- **Give models tools:** `Tool`, `Tools`, `EngineTools`, `Jail`, `Workspace`.
- **Refine:** `Refinery`, `Review`, `Outcome`, `Label`, `Labeled`, `Exported`, `Conflict`, `Bucket`, `is_holdout`.
- **Store:** `Storage`, `InMemoryStorage`, `PostgresStorage`, `RunRow`, `LabelRow`.
- **Fail loudly:** `EngineException`, `ConfigurationError`, `NotFoundError`, `InputError`, `ModelError`, `RunawayError`, `DelegationLoopError`, `DelegationDepthError`, `StorageError`.

## Errors

`ConfigurationError` for a tree, a model, a tool or a workspace the engine cannot use — a bad
`AGENT.md`, a tool declared without a body or given without a declaration, a run of an agent whose
kind of model is absent, an unknown task, a missing workspace, a `Workspace` root that is not a
directory, a `max_steps`, `max_depth` or `tool_timeout` that is not positive. `NotFoundError` for
an agent the tree does not have or a run the refinery does not.
`InputError` for a call the library refuses: an empty scope, a `task` for a decision agent, media
given to a decision agent, a media part with both or neither of `data` and `url`, a
correction of the wrong shape, a tool called with bad arguments or a path outside its directory. `ModelError` when an adapter breaks the contract — a
result whose fields do not fit, a missing answer, probabilities that do not cover the options or do not sum to 1. `RunawayError` when
a loop exceeds `max_steps` (default 200). `DelegationLoopError` and `DelegationDepthError` for
nesting: through `Delegate` they reach the model as tool errors, through a route they raise.
`StorageError` for the database. Inside a run, a tool's error is the model's to read, never raised.
All under `EngineException` — the root takes the `Exception` suffix, every child ends in `Error`,
the same shape as rfnry-memory's. Every message names the module that raised it, then says what went
wrong in lowercase: `engine: agent 'ghost' was not found`, `workspace: path outside the workspace: ../x`.
A tool's error reaches the model with its type in front: `InputError: tools: unknown tool 'Ghost'`.
