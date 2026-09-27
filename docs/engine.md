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

Structured output is a tool call — the model is offered an `Output` tool — so an adapter needs
nothing beyond tool support. An adapter that also implements
`generate_stream(system, messages, tools, on_delta)` streams text when the caller asks for events. An adapter that has `media_types`, the file types its model takes —
`{"image/*", "application/pdf"}` — lets `Load` refuse the rest with an error the model can act on,
instead of the next model call failing; without it, `Load` hands the model any file.

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
    knowledge/rates.pdf   knowledge is text, images or PDFs
    tools/lookup_order.md
    tools/fetch_invoice.md
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
caller passes. Everything else it reads when it needs it, with the `Read`, `Grep`, `Glob`, `LS`,
`Skill` and `Load` tools, and it can read only inside its own directory. `Read` returns a text file
as text. `Load` hands the model any other file as it is — an image, a PDF, a recording, a
spreadsheet — so a price list, a diagram or a scanned form sits in `knowledge/` as it came, the
model reads it with its own understanding of images and documents, there is no parser in between,
and nothing is loaded until the model asks. Write the maps with full paths —
`rules/refunds.md`, not `refunds.md` — because a model opens exactly what the map names.
`engine.agents` lists the tree as the engine loaded it: each name to an `AgentInfo` with its
`kind`, `description` and the `agents` it may call.

### Running an agent

`engine.run(agent, scope, input)` names the agent, the tenant and the input, and returns a record.
For a generation agent that record is a `Generation`: the reply, the system prompt as sent, the messages as sent to the first model call, the whole conversation, the tools the model was offered (`run.tools`, each a `ToolSpec`), every tool
call as a `Step` with its result and timing, the model calls and their summed usage, and any child
runs. It is the whole observability payload; hand it to your own telemetry:

```python
run.run_id, run.model, run.calls, run.usage.input, run.usage.output, run.duration_ms
for step in run.steps:
    step.tool, step.args, step.is_error, step.duration_ms, step.child_run_id
run.children                                         # runs made through Delegate, in step order
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

Pass `tags=` to mark a run with what you will group it by later — the commit of the tree, an
experiment arm, a release. Keys and values are strings — at most 32 tags, keys of 1 to 64
characters, values up to 256 — anything else is an `InputError`. The record and every exported row
carry them; the model never sees them. Delegated and routed runs inherit the tags of the run that
started them, so a version or an arm covers the whole tree of runs:

```python
run = await engine.run("support", scope="acme", input=message, tags={"commit": "4f2c9e1", "arm": "b"})
run.tags                                             # {"commit": "4f2c9e1", "arm": "b"}
```

Ask for a typed reply and the model is offered an `Output` tool shaped like it. When the model
calls it with input that fits, the run ends and `run.reply` is that input as JSON; input that does
not fit goes back to the model with what is wrong. A model that answers in text instead ends the
run with that text as `run.reply`, which does not parse:

```python
class Verdict(BaseModel):
    refund: bool
    reason: str

run = await engine.run("support", scope="acme", input=message, expect=Verdict)
verdict = Verdict.model_validate_json(run.reply)
```

An accepted `Output` ends the run. A call the model made after it in the same turn is not run; it is
answered with an error saying so, so every call in `run.conversation` has its result.

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

A person can stop or steer a run while it works. Pass `stop=Stop()` and call `stop.now()` to cancel
the model call in flight and every tool still running, or `stop.after_step()` to let the current
tool calls finish and end before the next model call. Pass `inbox=asyncio.Queue()` and put a string
or a list of parts on it whenever the person says something: at each step boundary — after the
tool results, before the next model call — the run takes everything waiting and appends each as a
`role="user"` message, in order:

```python
stop, inbox = Stop(), asyncio.Queue()
work = asyncio.create_task(engine.run("support", scope="acme", input=message, stop=stop, inbox=inbox))
inbox.put_nowait("Also check the March invoice.")
stop.after_step()
run = await work
run.stopped, run.reply, run.steps
```

A stopped run returns, never raises: `run.stopped` is `"now"` or `"after_step"` (`None` for a run
that ended on its own), `run.steps` are the steps it took, `run.reply` is the last text the model
produced — text streamed before `now()` included — or `""`, and it is recorded like any run. A tool
call cut off by `now()` is answered with the error `stopped by the user` and its `Step` has
`stopped=True`, so every tool call in `run.conversation` has its result and the next run continues
from it. A stop set before the run starts ends it before the first model call; one set after it
ended does nothing. A stop reaches delegated and routed runs, which end the same way at their own
next boundary. With `after_step()` pending the inbox is not read. Whatever is still in the inbox
when the run ends stays there — the engine never starts a run; you do, with those messages as its
input. A decision run never reads the inbox: its questions are one call, and a stop or an inbox
rides to the run it routes to. "Pause, please" is a message in the inbox: the model reads it at the
next boundary and wraps up in its own words.

A run that ended continues from its record, on any process, after any restart. With a refinery
attached every run is recorded with its conversation: keep the run id beside what you showed — a
message's metadata, say — read the run back and pass its conversation on:

```python
last = await refinery.run("acme", run.run_id)
next = await engine.run("support", scope="acme", input="And the second charge?", history=last.conversation)
```

`refinery.run(scope, run_id)` returns the run as `engine.run` returned it, its delegated and routed
runs back in place; a child's id returns that child and the runs under it. A run id the scope does
not have is a `NotFoundError`. A deploy is a stop: on SIGTERM, ask every run in flight to stop after its step.
Each ends at its next boundary and is recorded with `stopped="after_step"`; the next message, on
whichever task takes it, reads the run back and continues:

```python
import signal

running = set()

async def answer(message):
    stop = Stop()
    running.add(stop)
    try:
        return await engine.run("support", scope="acme", input=message, stop=stop)
    finally:
        running.discard(stop)

def shutdown():
    for stop in running:
        stop.after_step()

asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, shutdown)
```

Rewinding is slicing. Cut `run.conversation` just before a `role="user"` message and pass it as the
next run's `history`; files the agent changed in a `Workspace` are yours to restore — a snapshot per
turn, say. The abandoned run stays recorded, and a review of it turns the rewind into preference
data:

```python
turns = [i for i, m in enumerate(run.conversation) if m.role == "user"]
retry = await engine.run("support", scope="acme", input="Refund only the second charge.",
                         history=run.conversation[: turns[-1]])
await refinery.review("acme", run.run_id, Review(stance="objection", correction=retry.reply))
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
everything the run was given — the same input, parts included, `history`, `context`, `expect`, `tools`, `workspace`,
`on_event`, `stop`, `inbox` and `tags`; a decision agent itself takes no `task`. The routed run is attached:

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

> `Workspace` gives the model a real shell, and **a shell is not confined by its directory**. Without
> a sandbox it can do anything the user running the service can do: read every other tenant's
> workspace, read the service's configuration and any secret on disk, reach the database and the
> network, delete things. A document an agent reads can tell it to try. In a service that holds more
> than one tenant, or anything the model must not see, run every command inside a sandbox with
> `shell=` — the recipe is below — or run the service where there is nothing else to reach. Keep the
> agent directory outside the workspace.

A workspace is a directory where an agent writes code, runs it, reads what happened and tries again —
the loop a coding assistant runs in a developer's terminal. An agent opts in with one line in its
`AGENT.md`, and a run supplies the directory:

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
                       workspace=Workspace("/srv/tenants/acme/work", unsandboxed=True, timeout=600))
```

Every workspace says whether its shell is sandboxed. `sandbox=` names the sandbox each command runs
inside; `unsandboxed=True` says out loud that this machine holds nothing the model must not reach — a
developer's laptop, a container of its own. A workspace with neither refuses to start.

The agent then has three more tools. `Shell(command)` runs a command and returns `exit_code`,
`stdout`, `stderr` and `timed_out`. `Write(path, content)` writes a file exactly, and
`Edit(path, old, new)` replaces one exact occurrence — helpers that send only what changes and leave
a clean step in the record; the shell can do both too. `Load` takes `where="workspace"` to hand
the model a file from the directory, so an agent looks at the chart it just plotted. Each command starts fresh in the directory,
so a `cd` or a variable does not carry over, and the tool's description tells the model so, along
with its shell and its limits.

`Workspace(root, sandbox= or unsandboxed=True, shell=, timeout=, output_bytes=, env=)`:

| argument | default | what it does |
| --- | --- | --- |
| `sandbox` | none | the command each `Shell` call runs inside — bubblewrap, a container — before the shell |
| `unsandboxed` | `False` | `True` runs the shell on the machine itself, reaching whatever the service's user can |
| `shell` | `["bash", "-c"]` | the shell the command is given to, last; the model is told its name. Without a sandbox it must be on `PATH` |
| `timeout` | `120` seconds | no `Shell`, `Write` or `Edit` call runs longer; a command still running is killed with everything it started, and the call returns what it printed with `timed_out: true` |
| `output_bytes` | `30_000` | the cap on `stdout` and on `stderr`, each keeping its first and last half |
| `env` | `PATH`, `HOME` (the directory), `LANG` | the only environment commands see, so the service's own keys never reach them |

The workspace's `timeout` replaces `Engine(tool_timeout=)` for its three tools, so a ten-minute
build and a thirty-second API call live in the same engine. An unsandboxed workspace without bash on
`PATH` is a `ConfigurationError` that says so; pass `shell=["sh", "-c"]` on a system that has only a
POSIX shell, such as Alpine.

`Write`, `Edit` and `Load` stay inside the directory: the engine runs them itself and refuses a path
that leaves it. `Shell` reaches what the process running it can reach, so the real boundary is
`sandbox=` or the machine. On Linux, bubblewrap — the sandbox Claude Code uses for its own shell —
builds each command a world of its own: the system's programs read-only, this tenant's directory
writable, an empty `/tmp`, and nothing else — no other tenant, no service configuration, no network —
and it kills every process when the command ends:

```python
sandboxed = Workspace("/srv/tenants/acme/work", sandbox=[
    "bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
    "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
    "--tmpfs", "/tmp", "--bind", "/srv/tenants/acme/work", "/srv/tenants/acme/work",
    "--chdir", "/srv/tenants/acme/work", "--unshare-all", "--die-with-parent", "--new-session",
])
```

Binding the whole system read-only (`--ro-bind / /`) is not enough: a command could still read
every other tenant's directory and every secret on disk. Mount only what commands need — the
recipe above suits a system whose programs live under `/usr`; add a read-only `/etc` path when a tool
needs one — and one workspace per tenant, each with its own sandbox.

A container works too, one per tenant, with two details. Killing `docker exec` does not reach inside the container,
so the time limit goes inside the wrapper, and the container starts with `--init` so killed processes
are reaped. Mount the directory at the same path, so `Write` and the shell see the same files:

```python
contained = Workspace("/srv/tenants/acme/work", timeout=600, sandbox=[
    "docker", "exec", "-i", "-w", "/srv/tenants/acme/work", "sandbox",
    "timeout", "-s", "KILL", "600",
])
```

Without a sandbox, the kill reaches every process the command started except one that leaves its
process group with `setsid`; bubblewrap's own process namespace closes that gap.

An agent without `workspace: true` never sees these tools, even when the run has a workspace; an
agent with it, run without one, is a `ConfigurationError`, and so is a decision agent that declares
it. Delegated and routed agents receive the run's workspace on the same terms. `stop.now()` kills a
running command the same way the timeout does. You own the directory and its lifetime: a temporary
one per run, or one kept for a project.

### Approving tool calls

By default an agent calls any tool it has. Pass `approve` and every tool call waits for your answer
first — the shell, a `Write`, a refund, a `Delegate`. It receives the agent's name and the call, and
returns `True` to run it or `False` to refuse:

```python
async def approve(agent, call):
    if call.name in ("Write", "Edit"):
        return True
    return await ask_a_person(agent, call)

run = await engine.run("coder", scope="acme", input="make the failing test pass",
                       workspace=Workspace("/srv/tenants/acme/work", unsandboxed=True), approve=approve)
```

Where the person is — a thread, a dashboard, a chat app — is yours; the engine only waits. "Allow
everything" is no `approve`; "ask every time" is one that always asks; "allow edits, ask for the
rest" is the function above; "always allow this command" is a set the function keeps. A refused
call does not run: the model reads `NotApproved: Shell was not approved` and carries on, and a
reason reaches it through the `inbox`. Each step records `approved` — `True`, `False`, or `None`
when nobody was asked — and the exported rows keep it, so a person's refusals become labels.

Time spent waiting is not a tool's time: the deadline starts when the call runs. `stop.now()` ends
a wait. Delegated and routed runs are asked through the same function, with their own agent's name;
`engine.tool()` and `EngineTools` take `approve=` for the runs they start. An `approve` that returns
anything but `True` or `False` is a `ConfigurationError`. The run waits in memory, so an approval
that takes hours is better refused now and continued later from `run.conversation`.

### Refining

The refinery records and exports what your agents did, with every label and its source. What you do
with it — evaluate, improve prompts and rules, train or distill models, audit — is yours.
[`evaluation.md`](evaluation.md) is a guide to doing it: capturing the right answer, the correct
label and what really happened, and the three levels of improvement they feed.

Attach a refinery and every run is recorded. A recording that fails is logged as a warning and never
fails the run. `refinery.run(scope, run_id)` reads a recorded run back, its delegated and routed
runs in place, to continue it anywhere — see [Running an agent](#running-an-agent).

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

A judge or teacher the tree does not have is a `NotFoundError`, and one of the wrong kind a
`ConfigurationError`, before any run is labeled. A run that could not be labeled — the model
failed, or the teacher answered a choice with no option, a score with no level index, a check with
anything but true or false — is named in `labeled.errors` with why, and stays unlabeled for the
next call. Every label carries its source. A
judge's score is exported as a judge's score, never as a person's.

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
files and never overwrite. `since` selects every run recorded or labeled after it, written in full
with all its labels. A run is recorded when it ends, so a long run still working at one export is in
the next, whenever it started; the watermark covers both, so a review of a run you already exported
reaches the next export. Rows are keyed by `run_id`: when a run appears again, its latest row
supersedes the earlier ones — dedupe by `run_id`, keep the latest.

Every row carries `"holdout": true|false` — the rows kept aside so any measurement on them is
honest. It is derived from a hash of the run id (`holdout_percent=20` by default), so a run's flag
never changes between exports; `is_holdout(run_id)` gives the same answer in your own code. Three
datasets:

| dataset | one row per | what it holds |
| --- | --- | --- |
| `generation` | generation run | tags, system, messages, tools, conversation, steps, reply, stopped, usage, every label |
| `preference` | human correction | tags, prompt, chosen (the correction), rejected (the reply) |
| `decision` | question of a decision run | tags, state, question, answer, probabilities, confidence, truth, outcome |

Every row carries its run's `tags`, `{}` for a run that had none or was recorded before tags were
kept; group or filter the rows by them in your own code.

The first two are a generation agent's episodes and the corrections people made to them. A
generation row's `tools` is what the model was offered — `{name, description, input_schema}` each —
and `conversation` is `run.conversation` in order: history, input, assistant turns with their
`tool_calls` (`{id, name, args}`), tool results with `is_error`, the final reply. Together they fit
standard fine-tuning and eval formats with a few lines of mapping; the library writes no vendor's
format. A run recorded before `tools` was kept exports `"tools": []`. Parts
are written inline, their bytes as base64, so a row is self-contained; strip the media from a row
you do not want it in. The third
is a decision agent's answers beside what turned out true: a calibrated model is one whose 0.8 was
right 80% of the time, and this is the table that says whether it was:

```json
{"run_id": "…", "holdout": false, "agent": "frontdesk", "tags": {"arm": "b"}, "question_id": "intent", "answer": "billing",
 "probabilities": {"billing": 0.8, "technical": 0.15, "other": 0.05}, "confidence": 0.7,
 "truth": "technical", "truth_source": "human", "outcome": "failed"}
```

`refinery.conflicts(scope)` lists runs that carry two different human corrections — a dataset can
hold either, not both, and the library reports rather than guesses.

### Agents kept in a repository

The engine reads a folder; a new version of the agents is a new `Engine`. Keep the tree in a git
repository — people who do not write code edit it in the host's web UI, and every change is a
reviewed pull request — check each commit out into a folder of its own, build an engine from it,
and swap which engine new runs use. A run in flight finishes on the engine it started with. Rolling
back is loading an older commit, and since a commit's folder never changes, a half-written
checkout is never live. Agents read their files as they work, so remove an old folder only once
its runs have finished. The checkout is yours — `git worktree add`, a clone, an archive:

```python
def load(commit):
    return commit, Engine(checkout(commit), generation=llm, decision=dm, tools=tools, refinery=refinery)

live = load("4f2c9e1")
commit, engine = live
run = await engine.run("support", scope="acme", input=message, tags={"commit": commit})

live = load("a81d07b")

arms = {"a": load("4f2c9e1"), "b": load("a81d07b")}
arm = "ab"[hashlib.sha256(b"user-42").digest()[0] % 2]
commit, engine = arms[arm]
run = await engine.run("support", scope="acme", input=message, tags={"commit": commit, "arm": arm})
```

Tag each run with the commit it came from and the record, its delegated and routed runs, and every
exported row say which version of the tree answered. The same tag is an A/B test: load two commits
side by side, send each user to one by a stable hash of their id, tag the run with its arm, and
group the exported rows by `tags["arm"]` to compare the labels and outcomes of whole versions of
the tree.

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
    names: list[str]                                     # every tool's name, in order
```

An agent's tools are declared in its directory. `Read Grep Glob LS Skill Load` come with every generation agent,
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
name: fetch_invoice
---
```

The first declares the contract the model sees and leaves the body to code. The second adopts a
code tool as it is — description and schema included. Bodies and code tools go on the engine, one
list for the whole tree. A body returns a string, a dict (the model reads it as JSON) or a list of
parts, which reach the model as the tool result — a PDF the model reads itself, an image it looks
at. `Load` already does that for files in the agent's directory; a tool does it for what lives
elsewhere, such as your own store:

```python
class InvoiceArgs(BaseModel):
    invoice_id: str

async def fetch_invoice(args):
    data = await invoices.get(args["invoice_id"])
    return [Document(media_type="application/pdf", name=f"{args['invoice_id']}.pdf", data=data)]

tools = [
    Tool("lookup_order", body=lookup_order),
    Tool("fetch_invoice", "Fetch an invoice PDF by id", body=fetch_invoice, args=InvoiceArgs),
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
put its own limit on that work: cancelling the await does not stop the thread. A workspace's
`Shell`, `Write` and `Edit` keep the workspace's own `timeout` instead. `Delegate` has no
deadline. The child run it starts is bounded by its own tool calls and `max_steps`.

The builtins are bounded whatever the deadline. `Read` returns UTF-8 text and refuses anything else
with a pointer to `Load`; `offset` and `limit` select lines. `Load` takes any file up to 20 MB and
hands it over as an image, audio or a document part, whichever its type says. The type comes from
the file's name, and for PNG, JPEG, GIF, WebP and PDF the first bytes must agree — a `.png` that is
not one is refused rather than sent — while a file with no known name is typed by its bytes or sent
as `application/octet-stream`. What the model can take is the adapter's `media_types`, and past
that the vendor's. `Grep` searches text only and skips the rest; `Glob` and `LS` list every file.
Neither `Read` nor `Grep` reads a file over 5 MB. `Grep` stops after scanning 50 MB and
notes the cut. A search that runs past 10 s in one call is an `InputError` the model can act on. `Glob` and
`Grep` walk at most 100,000 entries. A tool that reads the
agent's directory itself resolves paths with `Jail`, which refuses one that leaves the directory,
as the builtins do.

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

A `Delegate` dispatched this way runs the child as its own top-level run, with the scope, `tools=`,
`workspace=` and `tags=` given to `EngineTools` — what a `Delegate` inside a run would give it.

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
was made with, recorded like any run; `context=`, `tools=`, `workspace=` and `tags=` ride every call.
Mounted inside another run with `run(tools=)` it is a tool like any other, under `tool_timeout`,
and its run is not the caller's child — to nest with lineage, name the agent in `agents:`. A loop
through it is still refused.

## Storage

The refinery is the only part of the library that stores anything. Two storages ship: `InMemoryStorage()`
for tests and small things, `PostgresStorage(pool)` for everything else, taking an asyncpg pool you
built. `initialize()` creates two tables, `engine_runs` and `engine_labels`, and is safe to run
again. A run's row keeps `at`, when the run started, and `recorded`, when the refinery stored it —
one stamp for a run and every child run under it — and `runs(since=)` reads `recorded`. The tables
are prefixed so the same database can hold rfnry-memory's:

```python
pool = await asyncpg.create_pool(dsn)
memory = Memory(rfnry_memory.PostgresStorage(pool), embedding=emb)
refinery = Refinery(rfnry_engine.PostgresStorage(pool))
```

A storage is any object with the methods below. `scope` comes first in every one, and each reads
and writes that scope only. `refinery.storage` is the storage the refinery was given, to read its
rows back yourself, and `refinery.record(run)` stores a run and every child run under it — what an
engine with a refinery does after each run not made with `record=False`:

```python
class Storage(Protocol):
    async def initialize(self) -> None
    async def put_run(self, scope: str, row: RunRow) -> None
    async def get_run(self, scope: str, run_id: str) -> RunRow | None
    async def runs(self, scope: str, *, run_ids: list[str] | None = None, kind: str | None = None,
                   since: datetime | None = None, unlabeled_by: str | None = None,
                   limit: int | None = 1000) -> list[RunRow]       # newest first
    async def put_label(self, scope: str, row: LabelRow) -> None
    async def labels(self, scope: str, *, run_ids: list[str] | None = None,
                     source: str | None = None, since: datetime | None = None) -> list[LabelRow]
    async def delete_scope(self, scope: str) -> int                 # the rows it removed
    async def count_scope(self, scope: str) -> int

class Refinery:
    storage: Storage
    async def record(self, run: Run) -> None

class RunRow(BaseModel):   scope, run_id, parent_run_id, agent, kind, task, at, recorded, duration_ms, model, body
class LabelRow(BaseModel): scope, label_id, run_id, source, at, body
class Exported(BaseModel): keys, rows, watermark
class Labeled(BaseModel):  judged, taught, errors
class Conflict(BaseModel): run_id, corrections
```

A row's `body` is the run's record, or the label, as JSON, with a run's children and routed run
left out: each is a row of its own, joined by `parent_run_id`. `runs` returns the newest first and
`labels` the oldest first; `unlabeled_by="judge"` keeps the runs no label from that source names.
`Exported.rows` counts the rows written per dataset; `Conflict.corrections` lists each correction
as `(label_id, at, text)`, oldest first.

## Tenancy

`scope` is the tenant. It is required on every run and every refinery verb, it is the first column
of every table, and no read accepts its absence. Tools pinned to a tenant are passed on the call.
`refinery.delete_scope(scope)` removes everything; `refinery.verify_absent(scope)` proves it.

Inside a tenant, erase by tag or by run id. A support desk serves every customer in one scope; tag
each run with its customer, and when one asks to be forgotten:

```python
run = await engine.run("support", scope="acme", input=message, tags={"customer": "c-81"})
deleted = await refinery.delete_runs("acme", tags={"customer": "c-81"})
deleted = await refinery.delete_runs("acme", run_ids=[run.run_id])
await refinery.verify_absent("acme", tags={"customer": "c-81"})
```

`delete_runs` takes `run_ids` or `tags`, exactly one, and removes the runs they match, every run
delegated or routed from them, and every label on any of them, in one transaction; it returns the
number of rows removed. A tag matches by containment: `{"customer": "c-81"}` matches a run tagged
`{"customer": "c-81", "arm": "b"}`. `verify_absent` takes the same selectors; without them it checks
the whole scope. Datasets you exported before are files you hold — drop the customer's rows from
them yourself; every row carries its run's tags.

## At a glance

Everything the package exports, by role.

- **Run things:** `Engine`, `AgentInfo`, `Stop`.
- **What a run returns:** `Generation`, `Decision`, `Run`, `Step`; events `TextDelta`, `ToolCallStarted`, `ToolResultReady`, `Event`.
- **Speak to models:** `GenerationModel`, `StreamingGenerationModel`, `DecisionModel`; `Message`, `ToolSpec`, `ToolCall`, `Reply`, `Usage`, `State`; parts `Text`, `Image`, `Audio`, `Document`, `Part`, `Content`; `Choice`, `Score`, `Check`, `Question`; `ChoiceAnswer`, `ScoreAnswer`, `CheckAnswer`, `Answer`, `Decided`; the shapes a result is read by, `ReplyLike`, `DecidedLike`, `AnswerLike`; `confidence`.
- **Give models tools:** `Tool`, `Tools`, `EngineTools`, `Jail`, `Workspace`.
- **Refine:** `Refinery`, `Review`, `Outcome`, `Label`, `Labeled`, `Exported`, `Conflict`, `Bucket`, `is_holdout`.
- **Store:** `Storage`, `InMemoryStorage`, `PostgresStorage`, `RunRow`, `LabelRow`.
- **Fail loudly:** `EngineException`, `ConfigurationError`, `NotFoundError`, `InputError`, `ModelError`, `RunawayError`, `DelegationLoopError`, `DelegationDepthError`, `StorageError`.

## Errors

`ConfigurationError` for a tree, a model, a tool or a workspace the engine cannot use — a bad
`AGENT.md`, an unknown key in the frontmatter of an `AGENT.md` or a tool file
(`frontmatter: unknown key 'route.min_confidance' in agents/frontdesk/AGENT.md`), a tool
declared without a body or given without a declaration, a run of an agent whose kind of model is
absent, an unknown task, a missing workspace, a `Workspace` root that is not a
directory, a `Workspace` with neither `sandbox=` nor `unsandboxed=True` or with both, an unsandboxed
shell not on `PATH`, a `sandbox` or `shell` that is not a list of strings, a `max_steps`,
`max_depth`, `tool_timeout`, `timeout` or `output_bytes` that is not positive, an `approve` that
returns anything but `True` or `False`, a judge that is not a decision agent or a teacher that is
not a generation agent. `NotFoundError` for an agent the tree does not have — a judge or a teacher
included — or a run the refinery does not.
`InputError` for a call the library refuses: an empty scope, a `task` for a decision agent, tags
outside their limits, media given to a decision agent, a media part with both or neither of `data` and `url`, a
correction of the wrong shape or for a question the run was not asked, a `delete_runs` or
`verify_absent` given both `run_ids` and `tags` or an empty one, a `delete_runs` given neither, a
tool called with bad arguments or a path outside its directory. `ModelError` when an adapter breaks the contract — a
result whose fields do not fit, a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1. `RunawayError` when
a loop exceeds `max_steps` (default 200). `DelegationLoopError` and `DelegationDepthError` for
nesting: through `Delegate` they reach the model as tool errors, through a route they raise.
`StorageError` for the database. Inside a run, a tool's error is the model's to read, never raised.
All under `EngineException` — the root takes the `Exception` suffix, every child ends in `Error`,
the same shape as rfnry-memory's. Every message names the module that raised it, then says what went
wrong in lowercase: `engine: agent 'ghost' was not found`, `workspace: path outside the workspace: ../x`.
A tool's error reaches the model with its type in front: `InputError: tools: unknown tool 'Ghost'`.
