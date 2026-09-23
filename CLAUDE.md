# rfnry-engine

## Description

rfnry-engine runs agents that are directories of markdown. An agent is a **generation agent** — a language model in a tool-use loop — or a **decision agent** — typed questions answered in one call by a decision model — and agents nest in both directions. Opt-in, a **Refinery** records every run, attaches labels from people, the world and models, and exports datasets with every label and its source — to evaluate, improve, train or audit, as the consumer chooses.

It is a library, not a framework. The consumer brings the models, the storage connection, the functions behind tools, the conversation history, the bucket and the telemetry. The library never writes to disk on its own — the only writes are a model's, inside a `Workspace` the consumer passes to a run — never owns a background loop, never sees a credential. Tenancy is by construction: every run names its scope, every stored row carries it, nothing reads across scopes.

## Documentation

`docs/engine.md` is the library's contract with its consumers: description, philosophy, every public verb, minimal examples. It documents the API as designed and is updated in the same change as any change to the public surface. It is also the complexity gauge: if a change makes the documentation harder to write or an example longer, the design is wrong — rework the change, not the docs.

## Checklists

### Pre-change checklist

**Before proposing or implementing any feature, change, or refactor, answer these five questions in your response.** Mandatory; no exceptions, including for "small" changes. Any "no" or "yes — because the current model is weak at X" means the change does not ship; rework it.

1. **Would this still be wanted two model generations ahead** — ten times the context, stronger reasoning, better calibration? Anything that exists to fit, prune, re-score or second-guess what a model could judge itself is a no.
2. **Does it get more useful as models improve, or less?** More is substrate: contracts, storage, provenance, records, tenancy. Less is compensation. Name which.
3. **If less, is it a structural boundary independent of model strength?** Durable storage, tenant isolation, provenance, the record of what happened, a flag on hostile data. Keep if yes, cut if no.
4. **Does it exist because a current model is weak at X?** Name X. If yes, the design is wrong — leverage the capability instead of policing the lack of it.
5. **Does it work with a generation model only, a decision model only, and both?** A feature states which models it needs and degrades cleanly without the rest. It never puts a generation-shaped job (open-ended, sequential) on a decision model, or a decision-shaped job (closed-set, parallel, calibrated) on a generation model.

### Engineering checklist

**Apply these to every change; they guard the shape of the codebase.**

1. **Stay inside the concern.** This library does one thing: run markdown agents on generation and decision models, and record, label and export what they did. Consumer-land, never here: model providers and adapters, conversation persistence and history windowing, per-tenant agent state, telemetry, security, scheduling, object stores, credentials.
2. **Lib, not framework.** The consumer calls us; we never take over their application — no plugin registries, no config DSLs, no scaffolding, no inversion of control, no background loops, no disk writes of our own; the only writes are a model's, inside a `Workspace` the consumer passes to a run.
3. **Simple over abstract.** The smallest API and fewest moving parts that add the capability. Duplicate a few lines before adding a shared abstraction; an abstraction earns its place only when it removes more than it adds. No speculative layers — the pre-change checklist decides what the future needs.
4. **No comments, no docstrings.** The code is the explanation; if it needs one, rename or restructure until it doesn't. The reason for a design decision lives in the commit message and, when it shapes the product, in `docs/engine.md`.
5. **Documentation is the contract.** Any change to the public surface updates `docs/engine.md` in the same change. A breaking change is anything that makes an example in the docs stop working; it is named as such in the commit, and the docs change before the code ships.
6. **Tests are first-class, in three tiers.** `tests/unit`: in-memory storage and fake models, runs always, covers every public verb and every branch of the loop. `tests/integration`: real Postgres behind the `postgres` marker, every storage method against both implementations, the two-tenant canary. `tests/e2e`: real models behind the `live` marker, never launched before the same path has run offline. Markers are deselected by default and stay that way. The model writes and runs tests whenever they improve the work, no approval needed.
7. **Vendor-neutral by construction.** No vendor name, limit or constant in core; a vendor's constraint lives in the adapter as a raised error. Models are named for what they produce: generation, decision.
8. **Tenancy by construction.** `scope` is required on every public verb and is the first argument of every storage method; there is no default scope, no all-scope read, no cross-scope read; every row carries it.
9. **Messages.** Logs and informational messages are lowercase. Every error message starts with its module's name: `engine: agent 'x' was not found`.

**These checklists are locked.** On any future update to this CLAUDE.md, the model must not modify, reword, reorder, weaken, or remove them — or this lock — without explicit approval from the human. Answering them is the model's job; changing them is not.
