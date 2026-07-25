# Synthetic evaluation protocol v3

This document freezes GWorker's first locked benchmark before evaluation
results are generated. It evaluates software behavior in declared synthetic
environments. It does **not** measure human productivity, establish causal
effects, or validate a health or workplace intervention.

Version 2 replaced the draft v1 path-conditioned headline with a common
availability-only comparison set. An adversarial pre-run review showed that
the old metric could make a poor strategy look better after it trapped itself
at an edge of the one-step movement guardrail. No locked `eval` result was
generated under v1.

Version 3 reserves a fresh locked RNG namespace after a pre-run unit test was
found to generate a small v2 `eval` environment while checking split
separation. That test never ran the benchmark or reported a result, but it now
uses the dedicated `test` split. The v3 `eval` namespace remains untouched
until the committed report runner passes its source-provenance gate.

## Primary question

Does the default adaptive policy reduce common availability-only expected
regret relative to the predeclared `fixed-25` comparator across the seven
primary personas and two availability modes?

The primary endpoint is the macro-average of:

```text
max expected reward among actions allowed by the current time budget
    - expected reward of its selected action
```

The comparison set deliberately ignores the strategy's previous action. It is
therefore identical for every strategy at a given environment decision. The
strategy still has to obey both availability and one-step movement guardrails
when it selects an action.

Each primary persona and availability mode receives equal weight. The primary
paired contrast is `adaptive - fixed-25`; lower regret is better. All other
contrasts are secondary and descriptive. Mean expected reward of the selected
action is reported beside regret as a direction check.

## Locked population

- Evaluator version: `synthetic-eval-v3`
- Locked evaluator ID:
  `synthetic-eval-v3.caf02b5aced57470dfa96c2952df7402dbd5944b7d394a5a95729ef3e6d09045`
- Locked design ID:
  `synthetic-eval-v3-design.503fe85fb1ff862383812f74eb9b6ce89fdc2a7811f44a83d53ca42c1634ebfd`
- Locked policy ID:
  `hierarchical-softmax-ucb-v1.8c10875dd38a025d`
- Horizon: 288 decisions
- Drift decision for abrupt personas: 137
- Environment seeds: integers 0 through 127
- Adaptive sampling replicas: 4 per environment seed
- Statistical unit: environment seed after averaging the 4 adaptive replicas
- Availability modes:
  - `unconstrained`: every complete focus/break template fits;
  - `guardrailed`: each 12-decision block contains exactly three budgets of
    18, 30, 48, and 60 minutes in a seeded order.
- Context balance: every 12-decision block contains each of the
  `4 TaskKind × 3 EnergyLevel` combinations exactly once in a seeded order.

The seven primary personas are `stable-25`, `stable-contextual`,
`boundary-short`, `boundary-long`, `abrupt-up`, `abrupt-down`, and
`gradual-up`. `cyclic` and `noisy-selective` are mandatory secondary stress
personas and cannot be removed after results are viewed.

## Latent preferences

Task offsets in minutes are:

```text
admin -8, learning 0, creative +4, deep_work +8
```

Energy offsets are:

```text
low -7, medium 0, high +7
```

All latent preferences are clipped to 15–50 minutes.

| Persona | Latent preference |
| --- | --- |
| `stable-25` | 25 |
| `stable-contextual` | 30 + task offset + energy offset |
| `boundary-short` | 17 |
| `boundary-long` | 48 |
| `abrupt-up` | 22 + 0.5 × energy offset, then 43 + 0.5 × energy offset |
| `abrupt-down` | reverse of `abrupt-up` |
| `gradual-up` | linear 22 → 43 between decisions 73 and 216 |
| `cyclic` | 32 + 12 × sin(2π(decision − 1) / 73) |
| `noisy-selective` | stable contextual preference with wider noise |

The main noise standard deviation is 5 minutes; `noisy-selective` uses
10 minutes. The perceived ideal is:

```text
latent preference + sigma × inverse_normal_cdf(U_fit)
```

With tolerance `τ = 6` minutes, a selected duration is `TOO_SHORT`,
`JUST_RIGHT`, or `TOO_LONG` according to its position around that perceived
ideal. The analytic right-fit probability is:

```text
Φ((action + τ − preference) / sigma)
    − Φ((action − τ − preference) / sigma)
```

Completion probability is:

```text
clip(
    0.20 + task adjustment
    + 0.55 × min(action / preference, 1)
    − 0.20 × max((action − preference) / preference, 0),
    0.05,
    0.95,
)
```

Task adjustments are `admin +0.08`, `learning 0`, `creative −0.02`, and
`deep_work −0.05`. Realized and expected rewards use the production policy's
fixed 80% right-fit / 20% completion weights.

## Paired randomization and leakage controls

No Python `hash()` values or shared mutable random-number streams are used.
Context order, availability order, fit noise, completion, review availability,
and policy sampling derive from disjoint SHA-256 namespaces containing the
evaluator version, split, persona, availability mode, environment seed,
decision, and component.

One fit uniform and one completion uniform are shared across all potential
actions at a decision. Consequently, two algorithms selecting the same action
in the same environment receive the same outcome. Policy sampling uses a
separate seed for every adaptive replica and decision, so algorithm execution
order cannot change result bytes.

The oracle receives analytic expected rewards but never realized outcomes.
Latent preferences and oracle choices are never passed to the adaptive policy.
Only context, prior reviewed decisions, and a per-decision sampling generator
enter `recommend()`.

The evaluation split is distinct from the non-heldout `dev` and `test`
namespaces. Default policy parameters may not be tuned after viewing locked
`eval` results. An algorithm change requires a `POLICY_FAMILY` bump, updated
golden replay vectors, and a new evaluator version.

## Mandatory comparators

Every strategy follows the same complete focus-plus-break availability and
one-step movement guardrails:

- `fixed-15`
- `fixed-25` (primary comparator)
- `fixed-40`
- `fixed-50`
- `last-choice` (25-minute cold start)
- `myopic-oracle` using expected reward and shorter-action tie-breaking

All four fixed policies remain visible. A best fixed policy selected after
viewing results may be reported only as a post-hoc descriptive comparison.

Each strategy has its own previous-action path. Per-step regret therefore uses
the following exact decomposition:

```text
common regret =
    max expected reward over availability-only actions
    - selected expected reward

conditional choice regret =
    max expected reward over this strategy's path-feasible actions
    - selected expected reward

path opportunity cost =
    availability-only oracle reward
    - path-feasible oracle reward

common regret = conditional choice regret + path opportunity cost
```

Only common regret is the primary comparison. Conditional choice regret and
path opportunity cost are secondary diagnostics that distinguish a poor
choice from a poor path without rewarding self-locking.

## Explicit reviews

Primary personas have 100% review availability. The `noisy-selective` stress
persona uses:

```text
0.90 if JUST_RIGHT else 0.65
minus 0.15 if incomplete
clipped to [0.40, 0.90]
```

An absent review creates no learning event and is never converted into a
negative outcome. Adaptive histories are created only through
`Recommendation.review()`.

## Required metrics

Headline and secondary metrics are:

- common mean and cumulative expected regret;
- conditional choice regret and path opportunity cost;
- expected reward of the selected action;
- realized mean reward, right-fit rate, and completion rate;
- mean arm-index distance from both common and conditional oracles;
- template exposure counts;
- exact/task/global evidence-bucket shares;
- review rate;
- feasible-set size and each guardrail reason rate;
- maximum arm-index transition;
- multiclass Brier score and fixed-bin propensity calibration;
- minimum propensity, maximum inverse weight, rate below 0.05, and inverse
  propensity effective-sample-size ratio.

Calibration bins are locked to:

```text
[0,.025), [.025,.05), [.05,.1), [.1,.2),
[.2,.4), [.4,.6), [.6,.8), [.8,1]
```

Propensity diagnostics describe simulator behavior only. They are not a claim
that directly constructed real-world reviews have journal-verified propensity
provenance.

All adaptive counts are additive across the four within-seed replicas. Reports
must derive rates, Brier scores, calibration summaries, inverse-propensity ESS,
and recovered-only lag from their pooled sufficient statistics and show the
actual denominators. They may not average already-computed replica ratios.

## Drift recovery

Recovery metrics apply only to `abrupt-up` and `abrupt-down` and use common
availability-only regret:

- pre-drift regret: final 48 decisions before drift;
- early post-drift AUC: first 48 post-drift regrets;
- late regret: final 48 decisions;
- threshold: `max(pre-drift mean + 0.02, 0.05)`;
- recovery lag: first post-drift start at which three consecutive 12-decision
  blocks are all at or below the threshold.

Unrecovered trajectories are right-censored at the remaining post-drift
horizon plus one. Reports must show recovery rate, recovered-only lag, and the
conservative censored lag. Recovery-time language is not used for gradual or
cyclic personas. Early post-drift AUC and late regret are the main drift
summaries; threshold-based recovery lag is descriptive because the threshold
depends on pre-drift performance.

## Uncertainty and reporting

The locked report uses 5,000 stratified paired cluster-bootstrap resamples.
Adaptive replicas are averaged before resampling and never become separate
statistical units. For primary persona `p`, availability mode `m`, environment
seed `s`, and comparator `c`, define:

```text
d[m,p,s,c] = adaptive common regret - comparator common regret
D[m,s,c]   = mean over the seven primary personas of d[m,p,s,c]
delta[c]   = equal-weight mean over modes of mean over seeds of D[m,s,c]
```

The registered primary estimate is `delta[fixed-25]`. The calculation is
cell-first: the paired difference is formed before personas are averaged.
Stress personas do not enter the primary estimate.

Each availability mode has its own independently generated `5000 × 128` index
matrix. Within a mode, the exact same bootstrap row is reused for every
persona, strategy, metric, and trace. Modes do not share rows. For bootstrap
replicate `b`, each mode independently resamples 128 seed positions with
replacement; the two resulting mode means are then averaged with equal weight.
Personas and modes themselves are fixed benchmark strata and are not
resampled.

Every resampled index is generated without a mutable RNG. UTF-8 bytes are
hashed for this unit-separator-delimited document:

```text
paired-seed-bootstrap-v1
evaluator_id
availability_mode
bootstrap_index in base 10
draw_position in base 10
rejection_attempt in base 10
```

The first eight digest bytes are interpreted as an unsigned big-endian
integer. The integer is accepted only below
`2^64 - (2^64 mod seed_count)` and is then reduced modulo `seed_count`;
otherwise `rejection_attempt` is incremented. The seed positions refer to the
declared configuration order.

The bootstrap definition digest is SHA-256 over a unit-separator-delimited
UTF-8 document containing, in order, the bootstrap version, evaluator ID,
comma-separated seed IDs, comma-separated mode IDs, and resample count. A mode
index digest is SHA-256 over:

```text
version ASCII
|| NUL || "mode-indices" || NUL
|| uint16be(mode byte length) || mode ASCII
|| uint64be(resample count) || uint32be(seed count)
|| every row-major index as uint32be
```

The combined digest replaces the mode rows with each raw 32-byte mode digest:

```text
version ASCII
|| NUL || "all-indices" || NUL
|| uint32be(seed count) || uint64be(resample count)
|| uint16be(mode count)
|| for each declared mode:
   uint16be(mode byte length) || mode ASCII || raw mode digest
```

The locked pre-run digests are:

```text
definition  0ec40605618d9532544b868463cf2c6ec75f8a5818a660054eb6d15f25c62e71
combined    a89de114bb2299c37192beafc724e4c993ee8484ad7d2cc7c76e6d1da8979318
unconstrained c7d76cd354ee4aae29fc752b40e853def5aa1fb51893aeefa5b8a3d624c8d8b8
guardrailed   bee073df927c228c651cdfb42e27968be95d59294ac04054679b8d9a2dd3069e
```

Percentile 95% endpoints use Hyndman-Fan Type 7: after sorting `B` bootstrap
estimates, `h = (B - 1) × p`, `i = floor(h)`, and the quantile linearly
interpolates between values `i` and `i + 1`. The observed estimate is not added
as a 5,001st draw.

For seed-cluster Monte Carlo standard error, first calculate the sample
variance `s_m²` of `D[m,s,c]` across the 128 seeds within each mode. With
`M = 2` and `n = 128`, the exact registered formula is:

```text
SE(delta[c]) = sqrt((1 / M²) × sum over m of (s_m² / n))
```

Availability modes use separate randomization namespaces. Persona and strategy
contrasts within a mode are paired by seed, but matching numeric seed IDs
between modes are not treated as paired observations and a difference between
availability modes must not be described as a strictly paired mode effect.

The report must include:

- the primary macro contrast and interval;
- every persona × availability mode, including losses;
- all fixed baselines;
- abrupt-up and abrupt-down recovery separately;
- calibration and guardrail diagnostics;
- pointwise—not simultaneous—uncertainty labels on time-series bands;
- the exact denominator and synthetic-only caveat near every visual.

The report manifest must also record the full evaluator design ID, population
ID, locked policy ID, source commit, clean source-tree digest, and artifact
checksums. The evaluator itself does not inspect Git; the future report runner
supplies and verifies source provenance before publication.

No failed seed or outlier may be removed. A hard invariant failure invalidates
the complete locked run and requires a new evaluator version after the defect
is fixed.

## Hard failure conditions

The run is invalid if any trajectory produces:

- an action outside the complete-budget feasible set;
- an undocumented jump of more than one template;
- a non-finite action probability;
- action probabilities whose sum differs from 1 by more than `1e-12`;
- a probability below the configured floor;
- a duplicate recommendation UUID or non-increasing reviewed sequence;
- a policy ID different from the locked default;
- an incomplete or duplicate scenario, seed, strategy, trace, or required
  output;
- a sufficient-statistic denominator that does not reconcile with its rate,
  exposure, calibration bin, or replica count.

## Interpretation boundary

Even a statistically precise synthetic advantage establishes only that the
policy adapts under these frozen equations and seed distribution. It does not
show that a duration causes better work, that the personas resemble users, or
that the reward represents wellbeing. Negative results and subgroup failures
must be published with the same visibility as positive results.
