# One-step off-policy replay diagnostics

Status: pure aggregate arithmetic is implemented in `gworker.offline`; verified
journal extraction, the canonical machine codec, CLI integration, and
synthetic publication evidence remain separate milestones.

GWorker records the probability of every selected duration and can reconstruct
the complete score vector that produced it. That is enough to ask a narrow
offline question:

> How well is a declared stochastic reweighting of the logged score vectors
> supported by the reviewed decisions in this journal?

It is not enough to estimate a sequential policy's effect on productivity.
Changing one duration can change later context, reviews, and available actions,
so this milestone deliberately evaluates one logged decision at a time while
holding its observed history fixed.

## Data boundary

The diagnostic starts from one read transaction in
`SQLiteEventStore`. Before any row is released, the store must run the existing
schema, SQLite, policy-fingerprint, history-edge, recommendation, propensity,
review, and focus-link checks.

An internal eligible row contains only:

- database-owned decision sequence;
- coarse `FocusContext`;
- feasible template IDs and their replayed score/probability decomposition;
- selected template and exact replayed behavior propensity;
- closed duration-fit and objective-completed review fields;
- the ordered-history digest already bound to the decision.

Objective text, session timestamps, event bodies, journal paths, host metadata,
and unreviewed decisions do not enter the reweighted summaries. Aggregate
output identifies the verified snapshot by a canonical digest. Real-journal
row data is never public output: decision sequences, contexts, actions, reviews,
and row digests remain inside the private computation. The only committed
row-level evidence will come from the fixed synthetic journal described below.

## Declared target distribution

For reviewed decision \(i\), let \(A_i\) be its replayed feasible templates and
\(s_i(a)\) the score reconstructed by the production policy replay. A target is
declared by a finite positive temperature \(\tau\) and a probability floor
\(\epsilon\), with the bound below required for every row:

\[
0 \le \epsilon < 1 / |A_i|
\]

The target probability for template \(a\) is:

\[
q_i(a) = \epsilon +
  (1 - \epsilon |A_i|)
  \frac{\exp((s_i(a)-\max_{u \in A_i}s_i(u))/\tau)}
       {\sum_{u \in A_i}\exp((s_i(u)-\max_{v \in A_i}s_i(v))/\tau)}
\]

This is a score-temperature sensitivity target. It reweights the exact score
vector available at that historical decision; it does not simulate how a
different learning policy would have changed later history.

Every report must also include a behavior-replay negative control. For that
control, \(q_i\) is the exact recomputed behavior distribution and every
importance weight must equal one bit-for-bit after canonical normalization. A
uniform-over-feasible target is the declared non-adaptive baseline.

## Descriptive reweighting diagnostics

For the selected action \(a_i\), replay supplies the behavior probability
\(b_i(a_i)\). The target weight and bounded explicit-review reward are:

\[
w_i = q_i(a_i) / b_i(a_i)
\]

\[
y_i = 0.8\,1[\text{fit = just-right}]
    + 0.2\,1[\text{objective completed}]
\]

These weights condition on the subset of decisions that received an explicit
review. GWorker does not assume that missing reviews are random, does not model
a review propensity, and does not use \(q_i/b_i\) to correct review selection.
The formulas below are finite-snapshot descriptive reweightings of the reviewed
rows—not unbiased estimates of one-step or sequential target-policy value.

For \(n\) eligible reviewed decisions, define the declared clipped weight:

\[
\bar{w}_i = \min(w_i, c), \qquad c \ge 1
\]

The report contains:

- observed behavior mean, \(\frac{1}{n}\sum_i y_i\);
- raw inverse-propensity summary, \(\frac{1}{n}\sum_i w_i y_i\);
- raw self-normalized summary,
  \(\frac{\sum_i w_i y_i}{\sum_i w_i}\);
- clipped inverse-propensity summary,
  \(\frac{1}{n}\sum_i \bar{w}_i y_i\);
- clipped self-normalized summary,
  \(\frac{\sum_i \bar{w}_i y_i}{\sum_i \bar{w}_i}\);
- raw effective sample size,
  \((\sum_i w_i)^2 / \sum_i w_i^2\);
- clipped effective sample size,
  \((\sum_i \bar{w}_i)^2 / \sum_i \bar{w}_i^2\);
- mean and maximum raw weight, minimum selected-action behavior propensity,
  clipped-row count, removed raw weight mass
  \(\sum_i (w_i-\bar{w}_i)\), and raw/clipped effective-sample-size ratios;
- per-template reviewed count, target mass, raw weighted reward contribution,
  clipped weighted reward contribution, and support diagnostics.

For template \(a\), those four aggregates are respectively
\(\sum_i 1[a_i=a]\), \(\sum_i q_i(a)\) (with zero for an infeasible template),
\(\frac{1}{n}\sum_i 1[a_i=a]w_i y_i\), and
\(\frac{1}{n}\sum_i 1[a_i=a]\bar{w}_i y_i\). Raw and clipped
effective-sample-size ratios divide their corresponding ESS by \(n\).

All sums use deterministic ordering by decision sequence and `math.fsum`.
Every probability and weight is finite, range-checked, and accompanied by its
canonical hexadecimal representation in the machine document. Decimal
rendering is presentation only. When \(n=0\), the report is
`insufficient-reviews`, records zero counts, and leaves every division-based
summary explicitly unavailable rather than dividing by zero.

The behavior negative control is an executable invariant: its raw and clipped
weights must all be exactly one because \(c \ge 1\). Its mean weight, both raw
and clipped inverse-propensity summaries, both self-normalized summaries, and
the observed behavior mean must agree exactly under the report's canonical
arithmetic.

## Readiness states

Producing arithmetic is not the same as having useful support. A report has one
of three states:

| State | Meaning |
| --- | --- |
| `insufficient-reviews` | Fewer than the declared minimum number of eligible reviews |
| `unstable-support` | Review count passes, but a raw effective-sample-size or raw maximum-weight guardrail fails |
| `reportable` | All declared count and raw-support guardrails pass; this does not mean the target is effective |

The pure core defaults to at least 12 reviews, an effective-sample-size ratio
computed from raw weights of at least 0.25, and a raw maximum weight no greater
than 10.0. Its declared clipping sensitivity defaults to 5.0. All four values
are explicit `ReplayConfig` inputs and will enter the canonical report
fingerprint. Clipped weights never determine readiness and cannot hide poor raw
support. A non-reportable result remains visible with its exact aggregate
diagnostics; it is never silently dropped or promoted to a policy comparison.

Clipping changes the descriptive summary. The raw inverse-propensity and
self-normalized values therefore remain present whenever arithmetic is valid,
and the UI must show the clipped-row count and removed raw weight mass next to
both clipped values.

## Canonical report

The planned machine document is `gworker-offline-replay-v1`. The implemented
pure value objects intentionally do not serialize themselves; the future codec
will bind:

- exact behavior policy fingerprint;
- target kind and all numeric target parameters;
- reward definition and support thresholds;
- total, reviewed-eligible, and excluded counts by closed reason code;
- verified snapshot digest and ordered diagnostic-row root;
- aggregate descriptive summaries and raw/clipped support diagnostics;
- behavior-negative-control results;
- explicit `review_selection_corrected: false` and
  `target_policy_value_estimated: false`;
- explicit booleans for `sequential_policy_value_estimated`,
  `causal_effect_estimated`, and `locked_evaluation_used`, all `false`.

Encoding will reject duplicate or unknown fields, non-canonical identifiers,
non-finite numbers, reordered rows, inconsistent sufficient statistics, and
any document that does not round-trip to identical canonical bytes.

The first CLI is aggregate-only. It will not print or export row-level
sequences, digests, contexts, actions, reviews, objective text, UUIDs, journal
paths, or timestamps. A future private row export is outside this contract and
would require a separate privacy and descriptor-pinned I/O review.

## Evidence plan

Committed evidence will be generated only from a fixed synthetic journal:

1. a genuine CLI transcript showing behavior-control and candidate-target
   status;
2. a sequence plot of behavior propensity, target propensity, raw weight, and
   the declared clip boundary;
3. an estimator-decomposition diagram tying each aggregate to its sufficient
   statistics;
4. a support panel with effective sample size and per-template coverage.

The generator must call the public replay API, bind every source and output
checksum, and reproduce exact bytes in `--check` mode. Synthetic row-level
labels may appear in these committed artifacts because their identities and
reviews are authored fixtures, never copied from a user's journal. It must not
import or invoke `gworker.evaluation`, the publication runner, the held-out
population, or any locked-result codec.

## Interpretation boundary

These diagnostics describe overlap and finite-snapshot reweighting of explicit
self-reports among reviewed decisions in one local journal. `reportable` means
only that the declared raw support guardrails pass. It does not establish that
a duration caused an outcome, that the target distribution has a policy-value
estimate, that it would improve future sessions, or that the reward represents
productivity, health, or workplace performance.

Missing reviews may be systematically different from recorded reviews. The
diagnostic neither estimates nor corrects that selection process. The reward is
a declared convenience encoding of two closed self-report fields, not a
validated measurement scale. Confidence intervals are intentionally outside
the first implementation; adding them requires a separately reviewed
missingness, dependence, and repeated-measures model rather than treating one
person's sequential decisions as independent observations.
