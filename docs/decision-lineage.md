# Durable policy-decision lineage

This document fixes the first persistent recommendation/review contract for
GWorker. It is deliberately narrower than a timer UI: the goal is to make one
adaptive choice and its later explicit review replayable without trusting a
caller-supplied propensity.

## Scope

The policy log uses the same permission-hardened SQLite file as the session
event journal. It records only bounded policy context:

- decision UUID and database-owned monotonic sequence;
- policy fingerprint and non-secret RNG seed;
- coarse task kind, self-reported energy, available seconds, and optional
  previous focus duration;
- selected template and the exact IEEE-754 propensity in canonical hexadecimal
  form;
- the ordered reviewed-decision IDs supplied to the policy;
- explicit duration fit and objective-completed feedback.

It does not store objective text, free-form review text, host paths, wall-clock
timestamps, or a claim that the recommendation improved human productivity.

## Relational record

Journal schema version 2 adds three append-only tables:

1. `policy_decisions` owns the decision UUID and contiguous sequence, context,
   RNG seed, selected template, propensity, bounded-history count, and digest.
2. `policy_decision_history` stores the exact ordered reviewed-decision
   sequence used for one recommendation.
3. `policy_reviews` stores at most one closed-enum review for a decision.

Opening an exact version-1 journal performs one explicit transactional
migration: create the three policy tables, update the version row, and commit.
Existing `events` rows are neither rewritten nor re-encoded. Unknown versions
and a damaged version-2 table set fail closed; `CREATE IF NOT EXISTS` is not
used to conceal missing version-2 tables.

Propensities are text, not SQLite `REAL` values. Encoding them with
`float.hex()` and requiring the canonical spelling preserves the exact binary
value passed to propensity-aware evaluation.

## Recommendation transaction

`SQLiteEventStore.recommend()` owns one `BEGIN IMMEDIATE` transaction:

1. verify the policy fingerprint and replay every stored decision;
2. load reviewed decisions in increasing decision sequence;
3. select the bounded tail used by the configured policy window;
4. allocate the next contiguous decision sequence;
5. call the production policy with `random.Random(recorded_seed)`;
6. insert the selected result and every history edge;
7. commit once.

A competing writer either observes the complete decision or receives a
conflict. It cannot publish the row without its history edges.

## Review transaction

`SQLiteEventStore.record_review()` accepts only a decision UUID plus closed
feedback fields. It reloads the stored recommendation, recomputes it from the
recorded context, seed, and exact history, and constructs the
`ReviewedDecision` through `Recommendation.review()`.

The caller cannot supply a policy ID, template, sequence, or propensity.
Missing decisions, duplicate reviews, invalid enums, and a recomputation
mismatch fail before commit.

## Replay and tamper boundary

Replay processes decisions in sequence order. For each row it requires:

- contiguous sequences and canonical UUIDs;
- a matching current policy fingerprint;
- contiguous, increasing history positions with no duplicate or future review;
- a history count and SHA-256 digest matching the ordered edges;
- an exact selected template and canonical propensity match after recomputation;
- a review derived from that exact recomputed recommendation.

SQLite foreign keys and transactions protect ordinary consistency. The digest
detects accidental history loss or reordering. As with the existing session
journal, this is not cryptographic authenticity against a hostile process
running as the same Unix user: such a process can rewrite both data and
digests.

## CLI boundary

The first CLI exposes only:

- `recommend` with explicit bounded context;
- `review` with a decision UUID and closed feedback;
- `verify` through replayed decision/review counts.

The CLI never treats a timer ending as positive feedback. Only `review`
changes policy history. A separate fixed recording harness invokes those
public CLI commands with fixed UUIDs, seeds, synthetic context, and a private
disposable journal. It is CLI evidence, not a locked-evaluation result.

## Failure cases required before publication

- orphan and duplicate review;
- duplicate decision UUID and competing sequence allocation;
- changed template or propensity;
- missing, duplicated, reordered, or future history edge;
- non-canonical float, UUID, or enum storage;
- transaction interruption followed by reopen;
- database replacement, symlink, hard-link, and permission regressions;
- replay after process restart;
- CLI output and error messages free of objectives, credentials, and host
  paths.

## Not in this milestone

- timer interaction or background notifications;
- objective storage in the policy log;
- a graphical desktop UI;
- cross-device synchronization;
- encryption beyond the host's disk protection;
- offline propensity evaluation or benchmark results;
- any execution of the locked synthetic evaluation namespace.
