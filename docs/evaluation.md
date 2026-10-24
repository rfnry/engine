# Evaluating and improving agents with rfnry

This guide is about knowing whether your agents do the job, and making them better week by week. It
uses the refinery in rfnry-engine, and, where they fit, rfnry-thread for the conversation and
rfnry-memory for knowledge. [`engine.md`](engine.md) is the reference for every verb used here.

## Golden data

An evaluation is only as good as the truth it compares against. In rfnry that truth comes from three
places, and none of them is a model:

- **The right answer.** A person who knows the work writes what the agent should have said: the
  support lead's reply, the team leader's correction in the thread.
- **The correct label.** A person's verdict on what the agent did: approved or not, the right
  department or not, urgent or not.
- **What really happened.** The world's answer: the refund cleared, the ticket reopened, the machine
  started after the fix.

This is golden data. It belongs to the company that produced it, whatever model wrote the draft the
person corrected, and it stays true when the model changes.

Models help with coverage. A **judge** — a decision agent — answers questions about every run, so
the runs nobody reviewed still carry a score; a **teacher** — a generation agent — answers the
questions a decision run was asked. Both are recorded with their source, never as a person's. They
tell you where to look; golden data tells you what is true.

## Capturing the signals

### Keep the run id next to what you show

Every run has a `run_id`. Keep it with whatever the user sees — the suggestion in the support tool,
the message in the thread — so a click, an edit or an outcome later can be attached to the run that
produced it:

```python
run = await engine.run("support", scope="acme", input=ticket_text)
await suggestions.save(ticket_id, text=run.reply, run_id=run.run_id)
```

### What people do with the answer

The strongest signal costs the user nothing: what they did with the agent's work. A support agent
who sends the suggestion unchanged approved it; one who edits it before sending wrote the right
answer; one who discards it objected.

```python
async def on_sent(run_id, suggested, sent):
    if sent == suggested:
        await refinery.review("acme", run_id, Review(stance="approval"))
    else:
        await refinery.review("acme", run_id, Review(correction=sent))

async def on_discarded(run_id):
    await refinery.review("acme", run_id, Review(stance="objection"))
```

### Buttons and corrections

An explicit verdict is a button under the answer. Keep it to one click, and ask why only when the
answer is thumbs down, with a few reasons to choose from and room to write:

```python
async def on_feedback(run_id, up, reason=None):
    stance = "approval" if up else "objection"
    await refinery.review("acme", run_id, Review(stance=stance, comment=reason))
```

A correction is the most valuable label there is: the right answer, written by someone who knows.
In a thread, a team leader who replies to the agent's message with the fix is writing one:

```python
await refinery.review("acme", run_id, Review(stance="objection", correction=leader_text))
```

A decision run takes a correction per question — the department it should have chosen:

```python
await refinery.review("acme", decision.run_id, Review(correction={"intent": "technical"}))
```

### What really happened

Outcomes arrive later, from your systems: the ticket closed or reopened, the order refunded, the
customer escalated. Attach them to the run:

```python
await refinery.outcome("acme", run_id, Outcome(status="failed", detail="Ticket reopened in 3 days."))
```

### How the market does it

The patterns are the same everywhere; rfnry records each of them with its source.

| Pattern | Seen in | In rfnry |
| --- | --- | --- |
| Thumbs up or down under each answer, reasons on thumbs down | Chat assistants, support copilots | `Review(stance=, comment=)` |
| Used as is, edited or discarded | Support copilots, coding assistants | `Review(stance=)` or `Review(correction=)` |
| Resolution, reopen, satisfaction score | Support platforms | `Outcome(status=, detail=)` |
| A review queue: a person scores a sample of runs against a rubric | Evaluation platforms | `Review` on the sampled runs |
| A model scores every run | Evaluation platforms | `refinery.label(judge=)` |

Clicks come mostly from the very happy and the very annoyed. A review queue — a person going
through a random sample every week with the same few questions — is the unbiased picture, and the
one to trust when the two disagree.

### Judges, teachers and conflicts

Run the judge over runs it has not labeled yet, on a schedule of your choosing:

```python
labeled = await refinery.label("acme", engine, judge="judge", teacher="teacher")
```

Check a judge against people before relying on it: on runs that have both a judge score and a
person's verdict, count how often they agree. When two people corrected the same run differently,
`refinery.conflicts("acme")` lists it; a person decides which one stands.

## Three levels of improvement

### Level 1: improve the agent

Most gains come from the agent, not the model. Corrections show which rule is wrong, which knowledge
is missing, which tool fails. The loop:

1. Export and read the runs that people objected to, corrected, or whose outcome failed.
2. Change the folder: a rule, a skill, a knowledge file. With rfnry-memory, a correction that is a
   fact can go straight into the corpus, and the next answer uses it:

   ```python
   await memory.remember("factory-a", "Press 4 hydraulic oil is ISO VG 46, not 68.", subject="press:4")
   ```

3. Replay the holdout on the new folder and compare, as below. Promote the change when it wins.
   The runs it records from then on carry the new folder's fingerprint, so the two versions never
   mix in a later comparison.

This works with any model, frontier or local, because nothing is trained.

### Replaying the holdout

About one run in five is held out, by a hash of its run id, so it never moves between exports.
Held-out runs with golden labels are your test set. `refinery.replay` runs a candidate — an
`Engine` on a new folder or a new model, with the same tools — on what each of them was given,
records nothing, and yields the recorded run, its labels and the candidate's run side by side.
Grade the candidate's replies against the right answers with a decision agent of your own:

```python
async def read_only(agent, call):
    return call.name in {"Read", "Grep", "Glob", "LS", "Load", "Delegate", "lookup_order"}

grades, failures = [], []
async for pair in refinery.replay("acme", candidate, agent="support", holdout=True, labeled=True,
                                  approve=read_only):
    if pair.candidate is None:
        failures.append(pair.error)
        continue
    review = next((label.review for label in reversed(pair.labels) if label.review), None)
    if review is None or not (review.correction or review.stance == "approval"):
        continue
    reference = review.correction or pair.recorded.reply
    grade = await engine.run("grader", scope="acme", record=False,
                             input={"reference": reference, "reply": pair.candidate.reply})
    grades.append(grade.answers["matches"].probability)
```

**Replay runs the candidate's tools for real.** A tool that refunds refunds again; a tool that sends
a message sends it. `read_only` lets through only the tools that read — the builtins, `Delegate`,
whose runs it is asked about too, and a lookup — and the model reads any other call as refused; or
replay with `tools=` stubs that only read, which win over the folder's tools of the same names.

Each replay is given the recorded input — an image replays as an image — the history before it (with
`conversation=`, the conversation before it, and `History`), the task and the context, so the
candidate answers the question the recorded run answered. The latest review decides the reference: a
person's correction is the right answer, an approval makes the recorded reply one, and a run labeled
only by its outcome has no text to grade against, so it is skipped here and its outcome counted on
its own. `grader` is a decision agent of yours whose check `matches` asks whether the reply says
what the reference says; it reads text, so it is given the two texts and not the question's parts.
Its probability is a score. The comparison is the mean over the pairs graded, next to the same
number for the engine you run today replayed the same way, with `failures` counted as failures.
`pair.recorded.fingerprint` and `pair.candidate.fingerprint` say what changed between the two: the
folder, the tools, or neither.

A decision agent replays the same way, and its fingerprint has a third part. `fingerprint.questions`
covers the questions a run asked — a word, an option or their order changed changes it — so two
versions of a classifier are told apart even when its questions come at run time, as a taxonomy's
levels or a list of candidates do. A replay asks the runtime questions again as recorded and the
folder's from the candidate, so a changed folder question shows in `questions` and in `agent`, and
the same runtime questions on a new model show in neither. When you would rather see it live, route a slice of traffic to the candidate with
`tags={"arm": "candidate"}` and compare the golden labels the two arms collect.

### Level 2: choose models on evidence

When a new model ships, replay the same holdout with it. Build an engine on the candidate adapter —
the same folder, its tools in it, the candidate under the name the agent's `model:` gives it
(`Engine(root, models={"strong": candidate_llm})`) — and compare the grade rate, the whole tree's usage and model calls (`spent` in
[Orchestrating](engine.md#orchestrating)) and `pair.candidate.duration_ms` against what you run today.
Not `pair.candidate.usage` alone: a candidate that delegates spends in its children. When the two fingerprints match, the tree is the same, so the model
is the only thing that differs, and `pair.candidate.model` names it. Editing an agent's `model:` is a tree change, and moves its fingerprint. The answer to "is the local
model good enough for this task?" is a number measured on your own labeled questions, not a
benchmark.

### Level 3: train your own models

Training is worth it when a task has enough golden data and a reason to own the model: privacy,
cost, running without a connection. Train only on what may be trained on. Every row records the
model that produced each run and the source of every label, so the filter is mechanical:

| Part of a row | Written by | Train on it |
| --- | --- | --- |
| The user's message, your documents | Your company and its customers | Yes, within your data-protection duties |
| A correction, a verified decision (`truth_source: "human"`) | A person | Yes |
| An outcome | The world | Yes, as a reward |
| A reply from an open-weight model whose license allows it | That model | Yes |
| A reply from a frontier API model | That provider | Only if the provider's terms allow it |
| A judge's or teacher's answer | A model | Only if that model's terms allow it |

Frontier providers restrict training other models on their outputs; check each provider's terms
before any of its text enters a training set. Decision models are the easiest start: a decision row
whose state is the customer's message and whose truth a person set is training data you own. Every
new checkpoint replays the holdout before it replaces the old one, and the old one stays for
rollback.

A customer who asks to be forgotten is removed by tag: with every run tagged with its customer,
`refinery.delete_runs(scope, tags={"customer": customer_id})` deletes their runs, the runs those
started and every label on them — a dataset or a checkpoint built before still holds them, so
rebuild it from a fresh export.

Retrieval and training do different jobs. Facts that change — manuals, prices, procedures — belong
in rfnry-memory, where a correction takes effect on the next question. Training teaches how to
answer: the terms, the format, when to say the manuals do not cover it.

## Deciding with evidence

A comparison is worth what the decision it supports is worth. Each of these rules is cheap, and each
one stops a specific mistake.

**Write the rule before the numbers.** Decide what would make you ship before running the replay —
"promote the new folder if the holdout grade rate rises by three points and outcome success does not
fall" — and keep it with the run. A rule chosen after seeing the numbers fits the numbers. When the
rule turns out to be ambiguous, measure every reading rather than pick the one that wins.

**Name the denominator.** Every rate says what it is a share of: "71% of the 212 held-out runs with a
human label", not "71%". When the population changes, rates over it stop being comparable even if the
agent did not change. Put `n` next to every number.

**Compare like with like.** A run carries the fingerprint of the agent it ran, of the tools it was
offered and, for a decision run, of the questions it asked, and its model; a memory corpus has a
fingerprint of its own. Compare two versions on the
same labeled runs, and name what differs between them: the folder, the tools, the model, the corpus.
When two things changed, you measured both at once.

**A slice does not speak for the whole.** A result on one tag, one route or the first fifty runs is a
result about that slice. Report it as one, and never let it carry the verdict for the whole set.

**Keep what failed.** A candidate that lost is a result. Keep its replay, its rule and its numbers,
reproducible by name, so nobody tries the same change again from memory.

**Measure what the system uses.** A metric computed over an input the system does not use invents
failures the system does not have: retrieval scored on the raw ticket text when the agent searches
with its own words, or a missing document counted against an answer the agent took from a tool.
Score the recorded inputs — the queries the agent issued, the steps it took.

**Know the noise before believing a difference.** Replay the engine you run today with `repeat=3`:
each held-out run is answered three times, and the three attempts come one after another. If the
count of passing runs moves by three between attempts of the same thing, a candidate that passes two
more is noise. With `passes(pair)` your grade of one pair, true or false — the grader above over a
threshold, a failed pair a fail:

```python
graded = {}
async for pair in refinery.replay("acme", engine, agent="support", holdout=True, labeled=True,
                                  approve=read_only, repeat=3):
    graded.setdefault(pair.recorded.run_id, {})[pair.attempt] = await passes(pair)

moved = sum(len(set(grades.values())) > 1 for grades in graded.values())
passing = [sum(grades[attempt] for grades in graded.values()) for attempt in range(3)]
swing = max(passing) - min(passing)
```

`moved` counts the runs whose grade differs across attempts; `swing` is how far the count of passing
runs moves between the best attempt and the worst. The arithmetic is yours, because the metric is: a
rate, a mean score or a count of outcomes is read across attempts the same way. A candidate replays
with `repeat=` too, so its own spread shows beside the one it is compared against. On a small
labeled set, say how many cases a difference is.

Applied to retrieval, these rules mean printing the rule before any query runs, giving every rate its
`n`, and stamping every table with the fingerprint of the corpus it was measured on.

## What to watch

- **Approval and correction rates** per agent and per version: the share of runs people accepted,
  and the share they rewrote.
- **Outcome success**: the share of runs whose outcome succeeded.
- **Holdout grade rate**: the number every change and every candidate model is compared on.
- **Judge agreement**: how often the judge agrees with people, before trusting its scores.
- **Calibration** of decision models: whether answers given at 0.8 are right about 80% of the time,
  from the decision dataset's probabilities next to the truth.

## Two shapes

**A support copilot.** Agents read the ticket and draft a solution; people send, edit or discard it.
Record what they did with each draft, add a thumbs down with reasons, and attach the ticket's outcome
when it closes or reopens. Level 1 fixes the rules the corrections point at. Level 2 tells you
whether a cheaper model drafts as well. Level 3, when the volume is there, starts with the triage
decisions people verified.

**A knowledge assistant over manuals.** The manuals go into rfnry-memory, one corpus per site. Each
question is a thread; the user marks the answer right or wrong, and a team leader can reply with the
correction for the agent and the person who asked. Right and wrong are reviews, the leader's reply
is a correction, and a correction that is a fact is remembered in the corpus at once. Level 2
answers the question the customer asks first — can a local model answer our questions as well — on
the questions they have already labeled. Level 3 trains the local model on the leaders' corrections
and on the answers it gave that people marked right.

## What rfnry does not do

The buttons, the review queue screen and the dashboards are your product's; rfnry records what they
send. Training is a separate step with the tools your team chooses; rfnry gives it rows that say
where every part came from.
