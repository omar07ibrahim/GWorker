"""Locked synthetic evaluation for the adaptive duration policy.

The benchmark pre-generates context and coherent potential outcomes before a
strategy runs. Every strategy sees the same exogenous values for a given
persona, availability mode, environment seed, and decision. The result is a
software-behavior benchmark, not evidence about people or causal productivity.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, fields
from enum import StrEnum
from statistics import NormalDist
from uuid import NAMESPACE_URL, UUID, uuid5

from .policy import (
    COMPLETION_REWARD_WEIGHT,
    DEFAULT_TEMPLATES,
    FIT_REWARD_WEIGHT,
    POLICY_ID,
    DurationFit,
    EnergyLevel,
    EvidenceBucket,
    FocusContext,
    FocusTemplate,
    HierarchicalSoftmaxUCB,
    ReviewedDecision,
    TaskKind,
)

EVALUATOR_VERSION = "synthetic-eval-v3"
RESULT_SCHEMA_VERSION = "synthetic-eval-result-v1"
LOCKED_POLICY_ID = "hierarchical-softmax-ucb-v1.8c10875dd38a025d"
LOCKED_POPULATION_ID = (
    "synthetic-eval-v3-population."
    "dc3f496427d8b78e742eed763ad494902660e50aa4ad45000008854edcaaab83"
)
LOCKED_EVALUATION_RUN_KEY = "synthetic-eval-v3-eval"
ALLOWED_SPLITS = ("dev", "test", "eval")
CONTEXT_BLOCK_SIZE = len(TaskKind) * len(EnergyLevel)
DEFAULT_HORIZON = 288
DEFAULT_DRIFT_DECISION = 137
PREFERENCE_TOLERANCE_MINUTES = 6.0
MAX_SEED = 2**32 - 1
MAX_HORIZON = 10_000
CALIBRATION_EDGES = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)
TASK_OFFSETS = {
    TaskKind.ADMIN: -8.0,
    TaskKind.LEARNING: 0.0,
    TaskKind.CREATIVE: 4.0,
    TaskKind.DEEP_WORK: 8.0,
}
ENERGY_OFFSETS = {
    EnergyLevel.LOW: -7.0,
    EnergyLevel.MEDIUM: 0.0,
    EnergyLevel.HIGH: 7.0,
}
COMPLETION_TASK_ADJUSTMENTS = {
    TaskKind.ADMIN: 0.08,
    TaskKind.LEARNING: 0.0,
    TaskKind.CREATIVE: -0.02,
    TaskKind.DEEP_WORK: -0.05,
}
COMPLETION_BASE_PROBABILITY = 0.20
COMPLETION_COVERAGE_WEIGHT = 0.55
COMPLETION_OVERRUN_PENALTY = 0.20
COMPLETION_PROBABILITY_BOUNDS = (0.05, 0.95)
SELECTIVE_REVIEW_FIT_PROBABILITY = 0.90
SELECTIVE_REVIEW_MISFIT_PROBABILITY = 0.65
SELECTIVE_REVIEW_INCOMPLETE_PENALTY = 0.15
SELECTIVE_REVIEW_PROBABILITY_BOUNDS = (0.40, 0.90)
LOW_PROPENSITY_THRESHOLD = 0.05
PROBABILITY_FLOOR_BAND_MULTIPLIER = 1.05
RECOVERY_REGRET_MARGIN = 0.02
RECOVERY_REGRET_FLOOR = 0.05
DRIFT_SUMMARY_WINDOW = 48
_NORMAL = NormalDist()


class EvaluationInputError(ValueError):
    """Raised when an evaluation definition is invalid."""


class EvaluationInvariantError(RuntimeError):
    """Raised when a trajectory violates the locked protocol."""


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationInputError(f"{field} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise EvaluationInputError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise EvaluationInputError(f"{field} must be finite")
    return 0.0 if result == 0 else result


def _bounded_integer(
    value: object,
    field: str,
    *,
    lower: int,
    upper: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationInputError(f"{field} must be an integer")
    if not lower <= value <= upper:
        raise EvaluationInputError(f"{field} must be between {lower} and {upper}")
    return value


def _validate_seed(seed: object) -> int:
    return _bounded_integer(
        seed,
        "environment seed",
        lower=0,
        upper=MAX_SEED,
    )


class AvailabilityMode(StrEnum):
    """Exogenous time-budget regimes."""

    UNCONSTRAINED = "unconstrained"
    GUARDRAILED = "guardrailed"


class Strategy(StrEnum):
    """Predeclared policies and baselines."""

    ADAPTIVE = "adaptive"
    FIXED_15 = "fixed-15"
    FIXED_25 = "fixed-25"
    FIXED_40 = "fixed-40"
    FIXED_50 = "fixed-50"
    LAST_CHOICE = "last-choice"
    MYOPIC_ORACLE = "myopic-oracle"


@dataclass(frozen=True, slots=True)
class Persona:
    """One frozen latent-preference process."""

    persona_id: str
    label: str
    primary: bool
    sigma_minutes: float = 5.0
    selective_reviews: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.persona_id, str)
            or not self.persona_id
            or len(self.persona_id) > 64
            or not all(
                character.isascii()
                and (character.islower() or character.isdigit() or character == "-")
                for character in self.persona_id
            )
        ):
            raise EvaluationInputError(
                "persona_id must contain lowercase ASCII letters, digits, or dash"
            )
        if (
            not isinstance(self.label, str)
            or not self.label
            or self.label != self.label.strip()
            or any(ord(character) < 32 for character in self.label)
        ):
            raise EvaluationInputError("label must be non-empty single-line text")
        if not isinstance(self.primary, bool):
            raise EvaluationInputError("primary must be boolean")
        sigma = _finite_float(self.sigma_minutes, "sigma_minutes")
        if not 0.1 <= sigma <= 60:
            raise EvaluationInputError("sigma_minutes must be in [0.1, 60]")
        object.__setattr__(self, "sigma_minutes", sigma)
        if not isinstance(self.selective_reviews, bool):
            raise EvaluationInputError("selective_reviews must be boolean")

    @property
    def is_abrupt(self) -> bool:
        """Return whether recovery-time metrics apply."""

        return self.persona_id in {"abrupt-up", "abrupt-down"}


DEFAULT_PERSONAS = (
    Persona("stable-25", "Stable 25-minute preference", True),
    Persona("stable-contextual", "Stable contextual preference", True),
    Persona("boundary-short", "Short boundary preference", True),
    Persona("boundary-long", "Long boundary preference", True),
    Persona("abrupt-up", "Abrupt upward drift", True),
    Persona("abrupt-down", "Abrupt downward drift", True),
    Persona("gradual-up", "Gradual upward drift", True),
    Persona("cyclic", "Cyclic stress", False),
    Persona(
        "noisy-selective",
        "Noisy selective-review stress",
        False,
        sigma_minutes=10.0,
        selective_reviews=True,
    ),
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete predeclared population and recovery rules."""

    split: str = "eval"
    environment_seeds: tuple[int, ...] = tuple(range(128))
    policy_replicas: int = 4
    personas: tuple[Persona, ...] = DEFAULT_PERSONAS
    availability_modes: tuple[AvailabilityMode, ...] = tuple(AvailabilityMode)
    horizon: int = DEFAULT_HORIZON
    drift_decision: int = DEFAULT_DRIFT_DECISION
    recovery_block_size: int = 12
    recovery_blocks: int = 3

    def __post_init__(self) -> None:
        try:
            environment_seeds = tuple(self.environment_seeds)
            personas = tuple(self.personas)
            availability_modes = tuple(self.availability_modes)
        except TypeError as error:
            raise EvaluationInputError(
                "population fields must be finite iterables"
            ) from error
        object.__setattr__(self, "environment_seeds", environment_seeds)
        object.__setattr__(self, "personas", personas)
        object.__setattr__(self, "availability_modes", availability_modes)
        if self.split not in ALLOWED_SPLITS:
            raise EvaluationInputError("split must be 'dev', 'test', or 'eval'")
        if not 1 <= len(self.environment_seeds) <= 10_000:
            raise EvaluationInputError(
                "environment_seeds must contain between 1 and 10000 values"
            )
        if len(set(self.environment_seeds)) != len(self.environment_seeds):
            raise EvaluationInputError("environment_seeds must be unique")
        for seed in self.environment_seeds:
            _validate_seed(seed)
        _bounded_integer(
            self.policy_replicas,
            "policy_replicas",
            lower=1,
            upper=64,
        )
        if not self.personas or not all(
            isinstance(persona, Persona) for persona in self.personas
        ):
            raise EvaluationInputError("personas must contain at least one Persona")
        persona_ids = [persona.persona_id for persona in self.personas]
        if len(set(persona_ids)) != len(persona_ids):
            raise EvaluationInputError("persona_id values must be unique")
        if not self.availability_modes or not all(
            isinstance(mode, AvailabilityMode) for mode in self.availability_modes
        ):
            raise EvaluationInputError(
                "availability_modes must contain AvailabilityMode values"
            )
        if len(set(self.availability_modes)) != len(self.availability_modes):
            raise EvaluationInputError("availability_modes must be unique")
        _bounded_integer(
            self.horizon,
            "horizon",
            lower=CONTEXT_BLOCK_SIZE * 2,
            upper=MAX_HORIZON,
        )
        if self.horizon % CONTEXT_BLOCK_SIZE:
            raise EvaluationInputError(
                f"horizon must be divisible by {CONTEXT_BLOCK_SIZE}"
            )
        _bounded_integer(
            self.drift_decision,
            "drift_decision",
            lower=2,
            upper=self.horizon,
        )
        _bounded_integer(
            self.recovery_block_size,
            "recovery_block_size",
            lower=1,
            upper=self.horizon,
        )
        _bounded_integer(
            self.recovery_blocks,
            "recovery_blocks",
            lower=1,
            upper=32,
        )
        required_post_drift = self.recovery_block_size * self.recovery_blocks
        if self.horizon - self.drift_index < required_post_drift:
            raise EvaluationInputError(
                "horizon must leave enough post-drift recovery decisions"
            )

    @property
    def drift_index(self) -> int:
        """Return the zero-based index of the first post-drift decision."""

        return self.drift_decision - 1


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


_EVALUATION_PERMIT_SECRET = object()


class _AuthorizedEvaluationRun:
    """Reusable capability scoped to one already-claimed locked run."""

    __slots__ = ("_claim_sha256", "_secret")

    def __init__(self, *, claim_sha256: str, secret: object) -> None:
        if secret is not _EVALUATION_PERMIT_SECRET:
            raise EvaluationInputError("locked run authorization is private")
        self._claim_sha256 = claim_sha256
        self._secret = secret

    def validate(self) -> None:
        if self._secret is not _EVALUATION_PERMIT_SECRET:
            raise EvaluationInputError("locked run authorization is invalid")


class _LockedEvaluationPermit:
    """Single-use in-process capability issued after the durable runner claim."""

    __slots__ = ("_claim_sha256", "_consumed", "_secret")

    def __init__(self, *, claim_sha256: str, secret: object) -> None:
        if secret is not _EVALUATION_PERMIT_SECRET:
            raise EvaluationInputError("locked evaluation permit is private")
        self._claim_sha256 = claim_sha256
        self._consumed = False
        self._secret = secret

    def consume(self) -> _AuthorizedEvaluationRun:
        if self._secret is not _EVALUATION_PERMIT_SECRET or self._consumed:
            raise EvaluationInputError("locked evaluation permit is invalid or used")
        self._consumed = True
        return _AuthorizedEvaluationRun(
            claim_sha256=self._claim_sha256,
            secret=self._secret,
        )


def _issue_locked_evaluation_permit(
    *,
    run_key: str,
    claim_sha256: str,
) -> _LockedEvaluationPermit:
    """Issue the private capability consumed by the publication runner."""

    if run_key != LOCKED_EVALUATION_RUN_KEY:
        raise EvaluationInputError("locked evaluation run key is invalid")
    if (
        not isinstance(claim_sha256, str)
        or len(claim_sha256) != 64
        or any(character not in "0123456789abcdef" for character in claim_sha256)
    ):
        raise EvaluationInputError("locked evaluation claim digest is invalid")
    return _LockedEvaluationPermit(
        claim_sha256=claim_sha256,
        secret=_EVALUATION_PERMIT_SECRET,
    )


def _authorize_eval_component(
    config: ExperimentConfig,
    authorization: _AuthorizedEvaluationRun | None,
) -> None:
    if config.split != "eval":
        if authorization is not None:
            raise EvaluationInputError(
                "locked run authorization cannot enter dev or test namespaces"
            )
        return
    if config != DEFAULT_EXPERIMENT_CONFIG:
        raise EvaluationInputError(
            "eval requires the exact locked experiment configuration"
        )
    if not isinstance(authorization, _AuthorizedEvaluationRun):
        raise EvaluationInputError(
            "eval generation requires publication-runner authorization"
        )
    authorization.validate()


@dataclass(frozen=True, slots=True)
class PotentialOutcome:
    """Coherent potential outcome for one template."""

    template: FocusTemplate
    fit: DurationFit
    completed: bool
    fit_probability: float
    completion_probability: float

    @property
    def realized_reward(self) -> float:
        """Return the production policy's bounded explicit-feedback reward."""

        return FIT_REWARD_WEIGHT * (
            1.0 if self.fit is DurationFit.JUST_RIGHT else 0.0
        ) + COMPLETION_REWARD_WEIGHT * (1.0 if self.completed else 0.0)

    @property
    def expected_reward(self) -> float:
        """Return the known synthetic expectation used only by the oracle."""

        return (
            FIT_REWARD_WEIGHT * self.fit_probability
            + COMPLETION_REWARD_WEIGHT * self.completion_probability
        )


@dataclass(frozen=True, slots=True)
class EnvironmentStep:
    """One strategy-independent decision and all potential outcomes."""

    decision: int
    task_kind: TaskKind
    energy: EnergyLevel
    available_seconds: int
    latent_preference_minutes: float
    review_uniform: float
    outcomes: tuple[PotentialOutcome, ...]

    def __post_init__(self) -> None:
        try:
            outcomes = tuple(self.outcomes)
        except TypeError as error:
            raise EvaluationInputError("outcomes must be a finite iterable") from error
        object.__setattr__(self, "outcomes", outcomes)

    def outcome_for(self, template_id: str) -> PotentialOutcome:
        """Return the pre-generated potential outcome for an action."""

        for outcome in self.outcomes:
            if outcome.template.template_id == template_id:
                return outcome
        raise EvaluationInvariantError(f"potential outcome missing for {template_id}")


@dataclass(frozen=True, slots=True)
class SyntheticEnvironment:
    """A paired persona/mode/seed trajectory."""

    evaluator_id: str
    design_id: str
    policy_id: str
    split: str
    persona: Persona
    availability_mode: AvailabilityMode
    environment_seed: int
    steps: tuple[EnvironmentStep, ...]

    def __post_init__(self) -> None:
        try:
            steps = tuple(self.steps)
        except TypeError as error:
            raise EvaluationInputError("steps must be a finite iterable") from error
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """One deterministic baseline or stochastic-policy replica."""

    mean_common_expected_regret: float
    mean_conditional_expected_regret: float
    mean_path_opportunity_cost: float
    mean_expected_reward: float
    mean_realized_reward: float
    right_fit_rate: float
    completion_rate: float
    mean_common_oracle_distance: float
    mean_conditional_oracle_distance: float
    cumulative_common_expected_regret: float
    cumulative_conditional_expected_regret: float
    cumulative_path_opportunity_cost: float
    availability_guardrail_rate: float
    one_step_guardrail_rate: float
    availability_override_rate: float
    mean_feasible_set_size: float
    maximum_arm_transition: int
    review_rate: float
    minimum_propensity: float | None
    maximum_inverse_propensity: float | None
    low_propensity_rate: float | None
    inverse_propensity_ess_ratio: float | None
    multiclass_brier_score: float | None
    exact_bucket_rate: float | None
    task_bucket_rate: float | None
    global_bucket_rate: float | None
    probability_floor_rate: float | None
    action_count: int
    availability_guardrail_count: int
    one_step_guardrail_count: int
    availability_override_count: int
    feasible_set_size_sum: float
    review_count: int
    selected_propensity_count: int | None
    low_propensity_count: int | None
    inverse_propensity_sum: float | None
    inverse_propensity_squared_sum: float | None
    brier_score_sum: float | None
    brier_score_count: int | None
    evidence_bucket_counts: tuple[int, ...]
    probability_floor_count: int | None
    arm_probability_count: int | None
    template_exposures: tuple[int, ...]
    calibration_predicted_sums: tuple[float, ...]
    calibration_observed_sums: tuple[float, ...]
    calibration_counts: tuple[int, ...]
    pre_drift_regret: float | None
    early_post_drift_auc: float | None
    late_regret: float | None
    recovery_lag: float | None
    recovery_rate: float | None
    conservative_recovery_lag: float | None
    recovered_count: int | None
    recovered_lag_sum: float | None
    common_regret_trace: tuple[float, ...]
    conditional_regret_trace: tuple[float, ...]
    path_opportunity_cost_trace: tuple[float, ...]

    def __post_init__(self) -> None:
        tuple_fields = (
            "evidence_bucket_counts",
            "template_exposures",
            "calibration_predicted_sums",
            "calibration_observed_sums",
            "calibration_counts",
            "common_regret_trace",
            "conditional_regret_trace",
            "path_opportunity_cost_trace",
        )
        for field in tuple_fields:
            try:
                values = tuple(getattr(self, field))
            except TypeError as error:
                raise EvaluationInputError(
                    f"{field} must be a finite iterable"
                ) from error
            object.__setattr__(self, field, values)


@dataclass(frozen=True, slots=True)
class ClusterSummary:
    """Seed-level statistical unit after averaging policy replicas."""

    persona_id: str
    persona_primary: bool
    availability_mode: AvailabilityMode
    environment_seed: int
    strategy: Strategy
    policy_replica_count: int
    metrics: TrajectoryMetrics


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """Seed-level abrupt-drift trace for pointwise uncertainty."""

    persona_id: str
    availability_mode: AvailabilityMode
    environment_seed: int
    strategy: Strategy
    common_expected_regret: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            common_expected_regret = tuple(self.common_expected_regret)
        except TypeError as error:
            raise EvaluationInputError(
                "common_expected_regret must be a finite iterable"
            ) from error
        object.__setattr__(
            self,
            "common_expected_regret",
            common_expected_regret,
        )


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Complete locked result at seed grain."""

    schema_version: str
    config: ExperimentConfig
    evaluator_id: str
    design_id: str
    policy_id: str
    cluster_summaries: tuple[ClusterSummary, ...]
    abrupt_traces: tuple[TraceRecord, ...]
    hard_failure_count: int

    def __post_init__(self) -> None:
        try:
            cluster_summaries = tuple(self.cluster_summaries)
            abrupt_traces = tuple(self.abrupt_traces)
        except TypeError as error:
            raise EvaluationInputError(
                "result collections must be finite iterables"
            ) from error
        object.__setattr__(self, "cluster_summaries", cluster_summaries)
        object.__setattr__(self, "abrupt_traces", abrupt_traces)


def _uniform(*parts: object) -> float:
    document = "\x1f".join(str(part) for part in parts).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(document).digest()[:8], "big")
    return ((integer >> 11) + 0.5) / 2**53


def _sampling_seed(*parts: object) -> int:
    document = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(document).digest()[:8], "big")


def _template_index(template: FocusTemplate) -> int:
    for index, candidate in enumerate(DEFAULT_TEMPLATES):
        if candidate.template_id == template.template_id:
            return index
    raise EvaluationInvariantError(
        f"template outside locked set: {template.template_id}"
    )


def _task_offset(task_kind: TaskKind) -> float:
    return TASK_OFFSETS[task_kind]


def _energy_offset(energy: EnergyLevel) -> float:
    return ENERGY_OFFSETS[energy]


def latent_preference_minutes(
    persona: Persona,
    *,
    task_kind: TaskKind,
    energy: EnergyLevel,
    decision: int,
    config: ExperimentConfig,
) -> float:
    """Return the frozen latent target without observing an action."""

    if not isinstance(persona, Persona):
        raise EvaluationInputError("persona must be a Persona")
    _bounded_integer(
        decision,
        "decision",
        lower=1,
        upper=config.horizon,
    )
    task_offset = _task_offset(task_kind)
    energy_offset = _energy_offset(energy)
    if persona.persona_id == "stable-25":
        preference = 25.0
    elif persona.persona_id == "stable-contextual":
        preference = 30.0 + task_offset + energy_offset
    elif persona.persona_id == "boundary-short":
        preference = 17.0
    elif persona.persona_id == "boundary-long":
        preference = 48.0
    elif persona.persona_id == "abrupt-up":
        preference = (
            22.0 + 0.5 * energy_offset
            if decision < config.drift_decision
            else 43.0 + 0.5 * energy_offset
        )
    elif persona.persona_id == "abrupt-down":
        preference = (
            43.0 + 0.5 * energy_offset
            if decision < config.drift_decision
            else 22.0 + 0.5 * energy_offset
        )
    elif persona.persona_id == "gradual-up":
        progress = min(1.0, max(0.0, (decision - 73) / (216 - 73)))
        preference = 22.0 + progress * (43.0 - 22.0)
    elif persona.persona_id == "cyclic":
        preference = 32.0 + 12.0 * math.sin(2.0 * math.pi * (decision - 1) / 73.0)
    elif persona.persona_id == "noisy-selective":
        preference = 30.0 + task_offset + energy_offset
    else:
        raise EvaluationInputError(f"unsupported persona_id: {persona.persona_id}")
    return min(50.0, max(15.0, preference))


def _fit_probability(
    *,
    focus_minutes: float,
    preference_minutes: float,
    sigma_minutes: float,
) -> float:
    upper = (
        focus_minutes + PREFERENCE_TOLERANCE_MINUTES - preference_minutes
    ) / sigma_minutes
    lower = (
        focus_minutes - PREFERENCE_TOLERANCE_MINUTES - preference_minutes
    ) / sigma_minutes
    return _NORMAL.cdf(upper) - _NORMAL.cdf(lower)


def _completion_probability(
    *,
    focus_minutes: float,
    preference_minutes: float,
    task_kind: TaskKind,
) -> float:
    task_adjustment = COMPLETION_TASK_ADJUSTMENTS[task_kind]
    probability = (
        COMPLETION_BASE_PROBABILITY
        + task_adjustment
        + COMPLETION_COVERAGE_WEIGHT * min(focus_minutes / preference_minutes, 1.0)
        - COMPLETION_OVERRUN_PENALTY
        * max((focus_minutes - preference_minutes) / preference_minutes, 0.0)
    )
    lower, upper = COMPLETION_PROBABILITY_BOUNDS
    return min(upper, max(lower, probability))


def _context_block(
    *,
    split: str,
    persona_id: str,
    availability_mode: AvailabilityMode,
    environment_seed: int,
    block: int,
) -> tuple[tuple[TaskKind, EnergyLevel], ...]:
    contexts = tuple(
        (task_kind, energy) for task_kind in TaskKind for energy in EnergyLevel
    )
    return tuple(
        sorted(
            contexts,
            key=lambda item: _uniform(
                EVALUATOR_VERSION,
                split,
                persona_id,
                availability_mode.value,
                environment_seed,
                block,
                item[0].value,
                item[1].value,
                "context-order",
            ),
        )
    )


def _availability_block(
    *,
    split: str,
    persona_id: str,
    availability_mode: AvailabilityMode,
    environment_seed: int,
    block: int,
) -> tuple[int, ...]:
    if availability_mode is AvailabilityMode.UNCONSTRAINED:
        return (60 * 60,) * CONTEXT_BLOCK_SIZE
    budgets = (18 * 60, 30 * 60, 48 * 60, 60 * 60) * 3
    indexed = tuple(enumerate(budgets))
    return tuple(
        value
        for index, value in sorted(
            indexed,
            key=lambda item: _uniform(
                EVALUATOR_VERSION,
                split,
                persona_id,
                availability_mode.value,
                environment_seed,
                block,
                item[0],
                "availability-order",
            ),
        )
    )


def generate_environment(
    *,
    persona: Persona,
    availability_mode: AvailabilityMode,
    environment_seed: int,
    config: ExperimentConfig,
    _eval_authorization: _AuthorizedEvaluationRun | None = None,
) -> SyntheticEnvironment:
    """Generate paired contexts and common-random-number outcomes."""

    if not isinstance(config, ExperimentConfig):
        raise EvaluationInputError("config must be an ExperimentConfig")
    _authorize_eval_component(config, _eval_authorization)
    if not isinstance(persona, Persona):
        raise EvaluationInputError("persona must be a Persona")
    if not isinstance(availability_mode, AvailabilityMode):
        raise EvaluationInputError("availability_mode must be an AvailabilityMode")
    if persona not in config.personas:
        raise EvaluationInputError("persona is not present in config")
    if availability_mode not in config.availability_modes:
        raise EvaluationInputError("availability_mode is not present in config")
    if POLICY_ID != LOCKED_POLICY_ID:
        raise EvaluationInvariantError("loaded policy differs from locked default")
    seed = _validate_seed(environment_seed)
    if seed not in config.environment_seeds:
        raise EvaluationInputError("environment_seed is not present in config")
    steps: list[EnvironmentStep] = []
    for block in range(config.horizon // CONTEXT_BLOCK_SIZE):
        contexts = _context_block(
            split=config.split,
            persona_id=persona.persona_id,
            availability_mode=availability_mode,
            environment_seed=environment_seed,
            block=block,
        )
        budgets = _availability_block(
            split=config.split,
            persona_id=persona.persona_id,
            availability_mode=availability_mode,
            environment_seed=environment_seed,
            block=block,
        )
        for offset, ((task_kind, energy), available_seconds) in enumerate(
            zip(contexts, budgets, strict=True)
        ):
            decision = block * CONTEXT_BLOCK_SIZE + offset + 1
            preference = latent_preference_minutes(
                persona,
                task_kind=task_kind,
                energy=energy,
                decision=decision,
                config=config,
            )
            fit_uniform = _uniform(
                EVALUATOR_VERSION,
                config.split,
                persona.persona_id,
                availability_mode.value,
                environment_seed,
                decision,
                "fit",
            )
            completion_uniform = _uniform(
                EVALUATOR_VERSION,
                config.split,
                persona.persona_id,
                availability_mode.value,
                environment_seed,
                decision,
                "completion",
            )
            review_uniform = _uniform(
                EVALUATOR_VERSION,
                config.split,
                persona.persona_id,
                availability_mode.value,
                environment_seed,
                decision,
                "review",
            )
            perceived_ideal = preference + persona.sigma_minutes * _NORMAL.inv_cdf(
                fit_uniform
            )
            outcomes: list[PotentialOutcome] = []
            for template in DEFAULT_TEMPLATES:
                focus_minutes = template.focus_seconds / 60.0
                if focus_minutes < perceived_ideal - PREFERENCE_TOLERANCE_MINUTES:
                    fit = DurationFit.TOO_SHORT
                elif focus_minutes > perceived_ideal + PREFERENCE_TOLERANCE_MINUTES:
                    fit = DurationFit.TOO_LONG
                else:
                    fit = DurationFit.JUST_RIGHT
                fit_probability = _fit_probability(
                    focus_minutes=focus_minutes,
                    preference_minutes=preference,
                    sigma_minutes=persona.sigma_minutes,
                )
                completion_probability = _completion_probability(
                    focus_minutes=focus_minutes,
                    preference_minutes=preference,
                    task_kind=task_kind,
                )
                outcomes.append(
                    PotentialOutcome(
                        template=template,
                        fit=fit,
                        completed=completion_uniform < completion_probability,
                        fit_probability=fit_probability,
                        completion_probability=completion_probability,
                    )
                )
            steps.append(
                EnvironmentStep(
                    decision=decision,
                    task_kind=task_kind,
                    energy=energy,
                    available_seconds=available_seconds,
                    latent_preference_minutes=preference,
                    review_uniform=review_uniform,
                    outcomes=tuple(outcomes),
                )
            )
    return SyntheticEnvironment(
        evaluator_id=evaluator_fingerprint(config),
        design_id=evaluator_design_fingerprint(),
        policy_id=LOCKED_POLICY_ID,
        split=config.split,
        persona=persona,
        availability_mode=availability_mode,
        environment_seed=seed,
        steps=tuple(steps),
    )


def _nearest_feasible(
    feasible: tuple[FocusTemplate, ...],
    desired_index: int,
) -> FocusTemplate:
    return min(
        feasible,
        key=lambda template: (
            abs(_template_index(template) - desired_index),
            _template_index(template),
        ),
    )


def _baseline_choice(
    strategy: Strategy,
    *,
    feasible: tuple[FocusTemplate, ...],
    previous: FocusTemplate | None,
    step: EnvironmentStep,
) -> FocusTemplate:
    fixed_targets = {
        Strategy.FIXED_15: 0,
        Strategy.FIXED_25: 1,
        Strategy.FIXED_40: 2,
        Strategy.FIXED_50: 3,
    }
    if strategy in fixed_targets:
        return _nearest_feasible(feasible, fixed_targets[strategy])
    if strategy is Strategy.LAST_CHOICE:
        desired = _template_index(previous) if previous is not None else 1
        return _nearest_feasible(feasible, desired)
    if strategy is Strategy.MYOPIC_ORACLE:
        return max(
            feasible,
            key=lambda template: (
                step.outcome_for(template.template_id).expected_reward,
                -_template_index(template),
            ),
        )
    raise EvaluationInputError(f"unsupported baseline strategy: {strategy}")


def _review_probability(
    persona: Persona,
    outcome: PotentialOutcome,
) -> float:
    if not persona.selective_reviews:
        return 1.0
    probability = (
        SELECTIVE_REVIEW_FIT_PROBABILITY
        if outcome.fit is DurationFit.JUST_RIGHT
        else SELECTIVE_REVIEW_MISFIT_PROBABILITY
    )
    if not outcome.completed:
        probability -= SELECTIVE_REVIEW_INCOMPLETE_PENALTY
    lower, upper = SELECTIVE_REVIEW_PROBABILITY_BOUNDS
    return min(upper, max(lower, probability))


def _calibration_bin(probability: float) -> int:
    for index in range(len(CALIBRATION_EDGES) - 1):
        lower = CALIBRATION_EDGES[index]
        upper = CALIBRATION_EDGES[index + 1]
        if lower <= probability < upper:
            return index
    if probability == 1.0:
        return len(CALIBRATION_EDGES) - 2
    raise EvaluationInvariantError(
        f"probability outside calibration bins: {probability}"
    )


def _mean(values: list[float] | tuple[float, ...]) -> float:
    if not values:
        raise EvaluationInvariantError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _drift_metrics(
    regrets: tuple[float, ...],
    *,
    persona: Persona,
    config: ExperimentConfig,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    if not persona.is_abrupt:
        return None, None, None, None, None, None
    drift = config.drift_index
    pre = _mean(regrets[max(0, drift - DRIFT_SUMMARY_WINDOW) : drift])
    early = math.fsum(
        regrets[drift : min(config.horizon, drift + DRIFT_SUMMARY_WINDOW)]
    )
    late = _mean(regrets[-DRIFT_SUMMARY_WINDOW:])
    threshold = max(pre + RECOVERY_REGRET_MARGIN, RECOVERY_REGRET_FLOOR)
    needed = config.recovery_block_size * config.recovery_blocks
    recovery_lag: int | None = None
    for start in range(drift, config.horizon - needed + 1):
        within_threshold = True
        for block in range(config.recovery_blocks):
            block_start = start + block * config.recovery_block_size
            block_values = regrets[
                block_start : block_start + config.recovery_block_size
            ]
            if _mean(block_values) > threshold:
                within_threshold = False
                break
        if within_threshold:
            recovery_lag = start - drift
            break
    censored_lag = config.horizon - drift + 1
    return (
        pre,
        early,
        late,
        float(recovery_lag) if recovery_lag is not None else None,
        1.0 if recovery_lag is not None else 0.0,
        float(recovery_lag if recovery_lag is not None else censored_lag),
    )


def simulate_trajectory(
    environment: SyntheticEnvironment,
    strategy: Strategy,
    *,
    policy_replica: int,
    config: ExperimentConfig,
    _eval_authorization: _AuthorizedEvaluationRun | None = None,
) -> TrajectoryMetrics:
    """Run one trajectory with no access to future or latent state."""

    if not isinstance(environment, SyntheticEnvironment):
        raise EvaluationInputError("environment must be a SyntheticEnvironment")
    if not isinstance(strategy, Strategy):
        raise EvaluationInputError("strategy must be a Strategy")
    if not isinstance(config, ExperimentConfig):
        raise EvaluationInputError("config must be an ExperimentConfig")
    _authorize_eval_component(config, _eval_authorization)
    expected_evaluator_id = evaluator_fingerprint(config)
    expected_design_id = evaluator_design_fingerprint()
    if environment.evaluator_id != expected_evaluator_id:
        raise EvaluationInputError(
            "environment evaluator_id does not match the supplied config"
        )
    if environment.design_id != expected_design_id:
        raise EvaluationInputError(
            "environment design_id does not match the loaded evaluator"
        )
    if environment.policy_id != LOCKED_POLICY_ID:
        raise EvaluationInputError(
            "environment policy_id does not match the locked default"
        )
    if environment.split != config.split:
        raise EvaluationInputError("environment split does not match config")
    if type(environment.persona) is not Persona:
        raise EvaluationInputError("environment persona has an invalid type")
    if type(environment.availability_mode) is not AvailabilityMode:
        raise EvaluationInputError("environment availability mode has an invalid type")
    if isinstance(environment.environment_seed, bool) or not isinstance(
        environment.environment_seed, int
    ):
        raise EvaluationInputError("environment seed has an invalid type")
    if environment.persona not in config.personas:
        raise EvaluationInputError("environment persona is not present in config")
    if environment.availability_mode not in config.availability_modes:
        raise EvaluationInputError(
            "environment availability mode is not present in config"
        )
    if environment.environment_seed not in config.environment_seeds:
        raise EvaluationInputError("environment seed is not present in config")
    if len(environment.steps) != config.horizon:
        raise EvaluationInputError("environment horizon disagrees with config")
    if tuple(step.decision for step in environment.steps) != tuple(
        range(1, config.horizon + 1)
    ):
        raise EvaluationInvariantError(
            "environment decisions must be contiguous and one-based"
        )
    expected_templates = tuple(template.template_id for template in DEFAULT_TEMPLATES)
    for step in environment.steps:
        actual_templates = tuple(
            outcome.template.template_id for outcome in step.outcomes
        )
        if actual_templates != expected_templates:
            raise EvaluationInvariantError(
                f"decision {step.decision} has an invalid potential-outcome set"
            )
        for outcome in step.outcomes:
            if not (
                math.isfinite(outcome.fit_probability)
                and 0.0 <= outcome.fit_probability <= 1.0
                and math.isfinite(outcome.completion_probability)
                and 0.0 <= outcome.completion_probability <= 1.0
            ):
                raise EvaluationInvariantError(
                    f"decision {step.decision} has an invalid potential outcome"
                )
    _bounded_integer(
        policy_replica,
        "policy_replica",
        lower=0,
        upper=config.policy_replicas - 1,
    )
    if strategy is not Strategy.ADAPTIVE and policy_replica != 0:
        raise EvaluationInputError("baseline policy_replica must be zero")

    policy = HierarchicalSoftmaxUCB()
    if policy.policy_id != LOCKED_POLICY_ID:
        raise EvaluationInvariantError("loaded policy differs from locked default")
    history: list[ReviewedDecision] = []
    recommendation_ids: set[UUID] = set()
    previous: FocusTemplate | None = None
    common_regrets: list[float] = []
    conditional_regrets: list[float] = []
    path_opportunity_costs: list[float] = []
    expected_rewards: list[float] = []
    realized_rewards: list[float] = []
    fits: list[float] = []
    completions: list[float] = []
    common_oracle_distances: list[float] = []
    conditional_oracle_distances: list[float] = []
    feasible_sizes: list[float] = []
    availability_count = 0
    step_count = 0
    override_count = 0
    maximum_transition = 0
    review_count = 0
    selected_propensities: list[float] = []
    brier_scores: list[float] = []
    bucket_counts = {bucket: 0 for bucket in EvidenceBucket}
    template_exposures = [0] * len(DEFAULT_TEMPLATES)
    calibration_predicted = [0.0] * (len(CALIBRATION_EDGES) - 1)
    calibration_observed = [0.0] * (len(CALIBRATION_EDGES) - 1)
    calibration_counts = [0] * (len(CALIBRATION_EDGES) - 1)
    floor_probability_count = 0
    arm_probability_count = 0

    for step in environment.steps:
        availability_context = FocusContext(
            task_kind=step.task_kind,
            energy=step.energy,
            available_seconds=step.available_seconds,
        )
        availability_feasible, _ = policy.feasible_templates(availability_context)
        context = FocusContext(
            task_kind=step.task_kind,
            energy=step.energy,
            available_seconds=step.available_seconds,
            previous_focus_seconds=(
                previous.focus_seconds if previous is not None else None
            ),
        )
        feasible, guardrail_reasons = policy.feasible_templates(context)
        feasible_sizes.append(float(len(feasible)))
        availability_count += "availability_guardrail" in guardrail_reasons
        step_count += "one_step_guardrail" in guardrail_reasons
        override_count += "availability_overrode_step" in guardrail_reasons
        if not set(feasible).issubset(availability_feasible):
            raise EvaluationInvariantError(
                "path-dependent feasible set exceeds availability-only set"
            )
        common_oracle = max(
            availability_feasible,
            key=lambda template: (
                step.outcome_for(template.template_id).expected_reward,
                -_template_index(template),
            ),
        )
        conditional_oracle = max(
            feasible,
            key=lambda template: (
                step.outcome_for(template.template_id).expected_reward,
                -_template_index(template),
            ),
        )

        selected_propensity: float | None = None
        if strategy is Strategy.ADAPTIVE:
            recommendation = policy.recommend(
                context,
                history,
                decision_id=uuid5(
                    NAMESPACE_URL,
                    (
                        f"gworker/{EVALUATOR_VERSION}/{config.split}/"
                        f"{environment.persona.persona_id}/"
                        f"{environment.availability_mode.value}/"
                        f"{environment.environment_seed}/{policy_replica}/"
                        f"{step.decision}"
                    ),
                ),
                decision_sequence=step.decision,
                rng=random.Random(
                    _sampling_seed(
                        EVALUATOR_VERSION,
                        config.split,
                        environment.persona.persona_id,
                        environment.availability_mode.value,
                        environment.environment_seed,
                        policy_replica,
                        step.decision,
                        "policy-sampling",
                    )
                ),
            )
            if recommendation.decision_id in recommendation_ids:
                raise EvaluationInvariantError(
                    f"duplicate recommendation UUID: {recommendation.decision_id}"
                )
            recommendation_ids.add(recommendation.decision_id)
            if recommendation.policy_id != LOCKED_POLICY_ID:
                raise EvaluationInvariantError(
                    "recommendation policy_id differs from locked default"
                )
            selected = recommendation.template
            selected_propensity = recommendation.propensity
            selected_propensities.append(selected_propensity)
            bucket_counts[recommendation.bucket] += 1
            probability_sum = math.fsum(
                arm.probability for arm in recommendation.arm_scores
            )
            if not math.isclose(
                probability_sum,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise EvaluationInvariantError(
                    f"action probabilities sum to {probability_sum}"
                )
            brier = 0.0
            for arm in recommendation.arm_scores:
                probability = arm.probability
                if (
                    not math.isfinite(probability)
                    or probability < policy.config.minimum_probability
                ):
                    raise EvaluationInvariantError(
                        f"invalid action probability: {probability}"
                    )
                observed = 1.0 if arm.template == selected else 0.0
                brier += (probability - observed) ** 2
                bin_index = _calibration_bin(probability)
                calibration_predicted[bin_index] += probability
                calibration_observed[bin_index] += observed
                calibration_counts[bin_index] += 1
                arm_probability_count += 1
                floor_probability_count += probability <= (
                    policy.config.minimum_probability
                    * PROBABILITY_FLOOR_BAND_MULTIPLIER
                )
            brier_scores.append(brier)
        else:
            selected = _baseline_choice(
                strategy,
                feasible=feasible,
                previous=previous,
                step=step,
            )

        if selected not in feasible:
            raise EvaluationInvariantError(
                f"{strategy.value} selected an infeasible template"
            )
        if selected not in availability_feasible:
            raise EvaluationInvariantError(
                f"{strategy.value} selected outside the availability-only set"
            )
        selected_index = _template_index(selected)
        template_exposures[selected_index] += 1
        if previous is not None:
            transition = abs(selected_index - _template_index(previous))
            maximum_transition = max(maximum_transition, transition)
            if transition > 1 and "availability_overrode_step" not in guardrail_reasons:
                raise EvaluationInvariantError(
                    f"undocumented one-step violation: {transition}"
                )

        outcome = step.outcome_for(selected.template_id)
        common_oracle_outcome = step.outcome_for(common_oracle.template_id)
        conditional_oracle_outcome = step.outcome_for(conditional_oracle.template_id)
        common_regret = max(
            0.0,
            common_oracle_outcome.expected_reward - outcome.expected_reward,
        )
        conditional_regret = max(
            0.0,
            conditional_oracle_outcome.expected_reward - outcome.expected_reward,
        )
        path_opportunity_cost = max(
            0.0,
            common_oracle_outcome.expected_reward
            - conditional_oracle_outcome.expected_reward,
        )
        if not math.isclose(
            common_regret,
            conditional_regret + path_opportunity_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise EvaluationInvariantError(
                "common regret does not equal choice regret plus path cost"
            )
        common_regrets.append(common_regret)
        conditional_regrets.append(conditional_regret)
        path_opportunity_costs.append(path_opportunity_cost)
        expected_rewards.append(outcome.expected_reward)
        realized_rewards.append(outcome.realized_reward)
        fits.append(1.0 if outcome.fit is DurationFit.JUST_RIGHT else 0.0)
        completions.append(1.0 if outcome.completed else 0.0)
        common_oracle_distances.append(
            float(abs(selected_index - _template_index(common_oracle)))
        )
        conditional_oracle_distances.append(
            float(abs(selected_index - _template_index(conditional_oracle)))
        )

        if strategy is Strategy.ADAPTIVE:
            review_probability = _review_probability(
                environment.persona,
                outcome,
            )
            if step.review_uniform < review_probability:
                review = recommendation.review(
                    fit=outcome.fit,
                    objective_completed=outcome.completed,
                )
                if history and (
                    review.decision_sequence <= history[-1].decision_sequence
                ):
                    raise EvaluationInvariantError(
                        "reviewed decision sequences are not strictly increasing"
                    )
                history.append(review)
                review_count += 1
        previous = selected

    horizon = len(environment.steps)
    drift_values = _drift_metrics(
        tuple(common_regrets),
        persona=environment.persona,
        config=config,
    )
    minimum_propensity: float | None = None
    maximum_inverse_propensity: float | None = None
    low_propensity_rate: float | None = None
    ess_ratio: float | None = None
    brier_score: float | None = None
    exact_rate: float | None = None
    task_rate: float | None = None
    global_rate: float | None = None
    floor_rate: float | None = None
    selected_propensity_count: int | None = None
    low_propensity_count: int | None = None
    inverse_propensity_sum: float | None = None
    inverse_propensity_squared_sum: float | None = None
    brier_score_sum: float | None = None
    brier_score_count: int | None = None
    probability_floor_count: int | None = None
    arm_probability_total: int | None = None
    if strategy is Strategy.ADAPTIVE:
        weights = [1.0 / value for value in selected_propensities]
        selected_propensity_count = len(selected_propensities)
        low_propensity_count = sum(
            value < LOW_PROPENSITY_THRESHOLD for value in selected_propensities
        )
        inverse_propensity_sum = math.fsum(weights)
        inverse_propensity_squared_sum = math.fsum(value * value for value in weights)
        brier_score_sum = math.fsum(brier_scores)
        brier_score_count = len(brier_scores)
        probability_floor_count = floor_probability_count
        arm_probability_total = arm_probability_count
        minimum_propensity = min(selected_propensities)
        maximum_inverse_propensity = max(weights)
        low_propensity_rate = low_propensity_count / selected_propensity_count
        ess_ratio = inverse_propensity_sum**2 / (
            selected_propensity_count * inverse_propensity_squared_sum
        )
        brier_score = brier_score_sum / brier_score_count
        exact_rate = bucket_counts[EvidenceBucket.EXACT] / selected_propensity_count
        task_rate = bucket_counts[EvidenceBucket.TASK] / selected_propensity_count
        global_rate = bucket_counts[EvidenceBucket.GLOBAL] / selected_propensity_count
        floor_rate = floor_probability_count / arm_probability_count

    recovered_count: int | None = None
    recovered_lag_sum: float | None = None
    if drift_values[4] is not None:
        recovered_count = int(drift_values[4])
        recovered_lag_sum = (
            float(drift_values[3]) if drift_values[3] is not None else 0.0
        )

    return TrajectoryMetrics(
        mean_common_expected_regret=_mean(common_regrets),
        mean_conditional_expected_regret=_mean(conditional_regrets),
        mean_path_opportunity_cost=_mean(path_opportunity_costs),
        mean_expected_reward=_mean(expected_rewards),
        mean_realized_reward=_mean(realized_rewards),
        right_fit_rate=_mean(fits),
        completion_rate=_mean(completions),
        mean_common_oracle_distance=_mean(common_oracle_distances),
        mean_conditional_oracle_distance=_mean(conditional_oracle_distances),
        cumulative_common_expected_regret=math.fsum(common_regrets),
        cumulative_conditional_expected_regret=math.fsum(conditional_regrets),
        cumulative_path_opportunity_cost=math.fsum(path_opportunity_costs),
        availability_guardrail_rate=availability_count / horizon,
        one_step_guardrail_rate=step_count / horizon,
        availability_override_rate=override_count / horizon,
        mean_feasible_set_size=_mean(feasible_sizes),
        maximum_arm_transition=maximum_transition,
        review_rate=(review_count / horizon if strategy is Strategy.ADAPTIVE else 0.0),
        minimum_propensity=minimum_propensity,
        maximum_inverse_propensity=maximum_inverse_propensity,
        low_propensity_rate=low_propensity_rate,
        inverse_propensity_ess_ratio=ess_ratio,
        multiclass_brier_score=brier_score,
        exact_bucket_rate=exact_rate,
        task_bucket_rate=task_rate,
        global_bucket_rate=global_rate,
        probability_floor_rate=floor_rate,
        action_count=horizon,
        availability_guardrail_count=availability_count,
        one_step_guardrail_count=step_count,
        availability_override_count=override_count,
        feasible_set_size_sum=math.fsum(feasible_sizes),
        review_count=review_count,
        selected_propensity_count=selected_propensity_count,
        low_propensity_count=low_propensity_count,
        inverse_propensity_sum=inverse_propensity_sum,
        inverse_propensity_squared_sum=inverse_propensity_squared_sum,
        brier_score_sum=brier_score_sum,
        brier_score_count=brier_score_count,
        evidence_bucket_counts=tuple(
            bucket_counts[bucket] for bucket in EvidenceBucket
        ),
        probability_floor_count=probability_floor_count,
        arm_probability_count=arm_probability_total,
        template_exposures=tuple(template_exposures),
        calibration_predicted_sums=tuple(calibration_predicted),
        calibration_observed_sums=tuple(calibration_observed),
        calibration_counts=tuple(calibration_counts),
        pre_drift_regret=drift_values[0],
        early_post_drift_auc=drift_values[1],
        late_regret=drift_values[2],
        recovery_lag=drift_values[3],
        recovery_rate=drift_values[4],
        conservative_recovery_lag=drift_values[5],
        recovered_count=recovered_count,
        recovered_lag_sum=recovered_lag_sum,
        common_regret_trace=tuple(common_regrets),
        conditional_regret_trace=tuple(conditional_regrets),
        path_opportunity_cost_trace=tuple(path_opportunity_costs),
    )


_OPTIONAL_METRIC_FIELDS = (
    "pre_drift_regret",
    "early_post_drift_auc",
    "late_regret",
    "conservative_recovery_lag",
)


def _aggregate_metrics(
    values: tuple[TrajectoryMetrics, ...],
) -> TrajectoryMetrics:
    if not values:
        raise EvaluationInvariantError("cannot aggregate zero trajectories")

    def average(field: str) -> float:
        return _mean([float(getattr(value, field)) for value in values])

    def optional_average(field: str) -> float | None:
        items = [getattr(value, field) for value in values]
        if all(item is None for item in items):
            return None
        if any(item is None for item in items):
            raise EvaluationInvariantError(
                f"replicas disagree on optional metric presence: {field}"
            )
        return _mean([float(item) for item in items])

    optional = {field: optional_average(field) for field in _OPTIONAL_METRIC_FIELDS}

    def sequence_values(field: str) -> list[tuple[float | int, ...]]:
        sequences: list[tuple[float | int, ...]] = [
            getattr(value, field) for value in values
        ]
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1:
            raise EvaluationInvariantError(
                f"replicas disagree on sequence length: {field}"
            )
        return sequences

    def average_sequence(field: str) -> tuple[float, ...]:
        sequences = sequence_values(field)
        return tuple(
            _mean([float(sequence[index]) for sequence in sequences])
            for index in range(len(sequences[0]))
        )

    def sum_float_sequence(field: str) -> tuple[float, ...]:
        sequences = sequence_values(field)
        return tuple(
            math.fsum(float(sequence[index]) for sequence in sequences)
            for index in range(len(sequences[0]))
        )

    def sum_int_sequence(field: str) -> tuple[int, ...]:
        sequences = sequence_values(field)
        return tuple(
            sum(int(sequence[index]) for sequence in sequences)
            for index in range(len(sequences[0]))
        )

    action_count = sum(value.action_count for value in values)
    availability_guardrail_count = sum(
        value.availability_guardrail_count for value in values
    )
    one_step_guardrail_count = sum(value.one_step_guardrail_count for value in values)
    availability_override_count = sum(
        value.availability_override_count for value in values
    )
    feasible_set_size_sum = math.fsum(value.feasible_set_size_sum for value in values)
    review_count = sum(value.review_count for value in values)

    selected_count_items = [value.selected_propensity_count for value in values]
    has_propensity = all(item is not None for item in selected_count_items)
    if not has_propensity and any(item is not None for item in selected_count_items):
        raise EvaluationInvariantError(
            "replicas disagree on propensity diagnostic presence"
        )

    selected_propensity_count: int | None = None
    low_propensity_count: int | None = None
    inverse_propensity_sum: float | None = None
    inverse_propensity_squared_sum: float | None = None
    brier_score_sum: float | None = None
    brier_score_count: int | None = None
    probability_floor_count: int | None = None
    arm_probability_count: int | None = None
    minimum_propensity: float | None = None
    maximum_inverse_propensity: float | None = None
    low_propensity_rate: float | None = None
    inverse_propensity_ess_ratio: float | None = None
    multiclass_brier_score: float | None = None
    exact_bucket_rate: float | None = None
    task_bucket_rate: float | None = None
    global_bucket_rate: float | None = None
    probability_floor_rate: float | None = None

    evidence_bucket_counts = sum_int_sequence("evidence_bucket_counts")
    if has_propensity:
        selected_propensity_count = sum(
            item for item in selected_count_items if item is not None
        )
        low_propensity_count = sum(value.low_propensity_count or 0 for value in values)
        inverse_propensity_sum = math.fsum(
            value.inverse_propensity_sum or 0.0 for value in values
        )
        inverse_propensity_squared_sum = math.fsum(
            value.inverse_propensity_squared_sum or 0.0 for value in values
        )
        brier_score_sum = math.fsum(value.brier_score_sum or 0.0 for value in values)
        brier_score_count = sum(value.brier_score_count or 0 for value in values)
        probability_floor_count = sum(
            value.probability_floor_count or 0 for value in values
        )
        arm_probability_count = sum(
            value.arm_probability_count or 0 for value in values
        )
        minima = [
            value.minimum_propensity
            for value in values
            if value.minimum_propensity is not None
        ]
        maxima = [
            value.maximum_inverse_propensity
            for value in values
            if value.maximum_inverse_propensity is not None
        ]
        if not minima or not maxima:
            raise EvaluationInvariantError(
                "propensity extrema are missing from adaptive replicas"
            )
        minimum_propensity = min(minima)
        maximum_inverse_propensity = max(maxima)
        if (
            selected_propensity_count <= 0
            or inverse_propensity_squared_sum <= 0
            or brier_score_count <= 0
            or arm_probability_count <= 0
        ):
            raise EvaluationInvariantError(
                "adaptive sufficient statistics have empty denominators"
            )
        low_propensity_rate = low_propensity_count / selected_propensity_count
        inverse_propensity_ess_ratio = inverse_propensity_sum**2 / (
            selected_propensity_count * inverse_propensity_squared_sum
        )
        multiclass_brier_score = brier_score_sum / brier_score_count
        exact_bucket_rate = evidence_bucket_counts[0] / selected_propensity_count
        task_bucket_rate = evidence_bucket_counts[1] / selected_propensity_count
        global_bucket_rate = evidence_bucket_counts[2] / selected_propensity_count
        probability_floor_rate = probability_floor_count / arm_probability_count

    recovered_items = [value.recovered_count for value in values]
    has_recovery = all(item is not None for item in recovered_items)
    if not has_recovery and any(item is not None for item in recovered_items):
        raise EvaluationInvariantError(
            "replicas disagree on recovery diagnostic presence"
        )
    recovered_count: int | None = None
    recovered_lag_sum: float | None = None
    recovery_lag: float | None = None
    recovery_rate: float | None = None
    if has_recovery:
        recovered_count = sum(item for item in recovered_items if item is not None)
        recovered_lag_sum = math.fsum(
            value.recovered_lag_sum or 0.0 for value in values
        )
        recovery_rate = recovered_count / len(values)
        recovery_lag = recovered_lag_sum / recovered_count if recovered_count else None

    return TrajectoryMetrics(
        mean_common_expected_regret=average("mean_common_expected_regret"),
        mean_conditional_expected_regret=average("mean_conditional_expected_regret"),
        mean_path_opportunity_cost=average("mean_path_opportunity_cost"),
        mean_expected_reward=average("mean_expected_reward"),
        mean_realized_reward=average("mean_realized_reward"),
        right_fit_rate=average("right_fit_rate"),
        completion_rate=average("completion_rate"),
        mean_common_oracle_distance=average("mean_common_oracle_distance"),
        mean_conditional_oracle_distance=average("mean_conditional_oracle_distance"),
        cumulative_common_expected_regret=average("cumulative_common_expected_regret"),
        cumulative_conditional_expected_regret=average(
            "cumulative_conditional_expected_regret"
        ),
        cumulative_path_opportunity_cost=average("cumulative_path_opportunity_cost"),
        availability_guardrail_rate=(availability_guardrail_count / action_count),
        one_step_guardrail_rate=one_step_guardrail_count / action_count,
        availability_override_rate=(availability_override_count / action_count),
        mean_feasible_set_size=feasible_set_size_sum / action_count,
        maximum_arm_transition=max(value.maximum_arm_transition for value in values),
        review_rate=review_count / action_count,
        minimum_propensity=minimum_propensity,
        maximum_inverse_propensity=maximum_inverse_propensity,
        low_propensity_rate=low_propensity_rate,
        inverse_propensity_ess_ratio=inverse_propensity_ess_ratio,
        multiclass_brier_score=multiclass_brier_score,
        exact_bucket_rate=exact_bucket_rate,
        task_bucket_rate=task_bucket_rate,
        global_bucket_rate=global_bucket_rate,
        probability_floor_rate=probability_floor_rate,
        action_count=action_count,
        availability_guardrail_count=availability_guardrail_count,
        one_step_guardrail_count=one_step_guardrail_count,
        availability_override_count=availability_override_count,
        feasible_set_size_sum=feasible_set_size_sum,
        review_count=review_count,
        selected_propensity_count=selected_propensity_count,
        low_propensity_count=low_propensity_count,
        inverse_propensity_sum=inverse_propensity_sum,
        inverse_propensity_squared_sum=inverse_propensity_squared_sum,
        brier_score_sum=brier_score_sum,
        brier_score_count=brier_score_count,
        evidence_bucket_counts=evidence_bucket_counts,
        probability_floor_count=probability_floor_count,
        arm_probability_count=arm_probability_count,
        template_exposures=sum_int_sequence("template_exposures"),
        calibration_predicted_sums=sum_float_sequence("calibration_predicted_sums"),
        calibration_observed_sums=sum_float_sequence("calibration_observed_sums"),
        calibration_counts=sum_int_sequence("calibration_counts"),
        pre_drift_regret=optional["pre_drift_regret"],
        early_post_drift_auc=optional["early_post_drift_auc"],
        late_regret=optional["late_regret"],
        recovery_lag=recovery_lag,
        recovery_rate=recovery_rate,
        conservative_recovery_lag=optional["conservative_recovery_lag"],
        recovered_count=recovered_count,
        recovered_lag_sum=recovered_lag_sum,
        common_regret_trace=average_sequence("common_regret_trace"),
        conditional_regret_trace=average_sequence("conditional_regret_trace"),
        path_opportunity_cost_trace=average_sequence("path_opportunity_cost_trace"),
    )


def evaluation_design_manifest() -> dict[str, object]:
    """Return the canonical equations and guardrails hashed into the design."""

    return {
        "availability": {
            "guardrailed_block_seconds": [1080, 1800, 2880, 3600] * 3,
            "unconstrained_seconds": 3600,
        },
        "calibration_edges": list(CALIBRATION_EDGES),
        "completion": {
            "base_probability": COMPLETION_BASE_PROBABILITY,
            "coverage_weight": COMPLETION_COVERAGE_WEIGHT,
            "overrun_penalty": COMPLETION_OVERRUN_PENALTY,
            "probability_bounds": list(COMPLETION_PROBABILITY_BOUNDS),
            "task_adjustments": {
                task.value: adjustment
                for task, adjustment in COMPLETION_TASK_ADJUSTMENTS.items()
            },
        },
        "contexts": {
            "block_size": CONTEXT_BLOCK_SIZE,
            "energy_levels": [energy.value for energy in EnergyLevel],
            "energy_offsets": {
                energy.value: offset for energy, offset in ENERGY_OFFSETS.items()
            },
            "task_kinds": [task.value for task in TaskKind],
            "task_offsets": {
                task.value: offset for task, offset in TASK_OFFSETS.items()
            },
        },
        "drift_metrics": {
            "recovery_regret_floor": RECOVERY_REGRET_FLOOR,
            "recovery_regret_margin": RECOVERY_REGRET_MARGIN,
            "summary_window": DRIFT_SUMMARY_WINDOW,
        },
        "enum_values": {
            "availability_modes": [mode.value for mode in AvailabilityMode],
            "duration_fit": [fit.value for fit in DurationFit],
            "evidence_buckets": [bucket.value for bucket in EvidenceBucket],
            "strategies": [strategy.value for strategy in Strategy],
        },
        "fit": {
            "distribution": "standard-normal-cdf",
            "preference_bounds_minutes": [15.0, 50.0],
            "tolerance_minutes": PREFERENCE_TOLERANCE_MINUTES,
        },
        "input_bounds": {
            "maximum_horizon": MAX_HORIZON,
            "maximum_seed": MAX_SEED,
        },
        "oracle": {
            "conditional_metric": (
                "max expected reward over availability-and-path feasible actions "
                "minus selected expected reward"
            ),
            "path_opportunity_cost": (
                "availability-only oracle reward minus conditional oracle reward"
            ),
            "primary_metric": (
                "availability-only oracle expected reward minus selected "
                "expected reward"
            ),
            "tie_break": "shorter-template-index",
        },
        "personas": {
            "abrupt-down": "43+0.5*energy then 22+0.5*energy",
            "abrupt-up": "22+0.5*energy then 43+0.5*energy",
            "boundary-long": "48",
            "boundary-short": "17",
            "cyclic": "32+12*sin(2*pi*(decision-1)/73)",
            "gradual-up": "linear 22 to 43 over decisions 73 through 216",
            "noisy-selective": "30+task+energy",
            "stable-25": "25",
            "stable-contextual": "30+task+energy",
        },
        "policy_id": LOCKED_POLICY_ID,
        "randomization": {
            "digest": "sha256",
            "fit_and_completion_uniforms_shared_across_arms": True,
            "policy_sampling_stream": "per-replica-per-decision",
            "uniform_conversion": "sha256-first-64-bits-to-open-53-bit-float",
        },
        "review": {
            "fit_probability": SELECTIVE_REVIEW_FIT_PROBABILITY,
            "incomplete_penalty": SELECTIVE_REVIEW_INCOMPLETE_PENALTY,
            "misfit_probability": SELECTIVE_REVIEW_MISFIT_PROBABILITY,
            "probability_bounds": list(SELECTIVE_REVIEW_PROBABILITY_BOUNDS),
        },
        "reward": {
            "completion_weight": COMPLETION_REWARD_WEIGHT,
            "right_fit_weight": FIT_REWARD_WEIGHT,
        },
        "strategy_rules": {
            "fixed_target_indices": {
                "fixed-15": 0,
                "fixed-25": 1,
                "fixed-40": 2,
                "fixed-50": 3,
            },
            "last_choice_initial_index": 1,
            "movement_limit_indices": 1,
        },
        "splits": {
            "allowed": list(ALLOWED_SPLITS),
            "locked": "eval",
        },
        "templates": [
            {
                "break_seconds": template.break_seconds,
                "focus_seconds": template.focus_seconds,
                "template_id": template.template_id,
            }
            for template in DEFAULT_TEMPLATES
        ],
        "thresholds": {
            "low_propensity": LOW_PROPENSITY_THRESHOLD,
            "probability_floor_band_multiplier": (PROBABILITY_FLOOR_BAND_MULTIPLIER),
        },
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "version": EVALUATOR_VERSION,
    }


def _config_manifest(config: ExperimentConfig) -> dict[str, object]:
    return {
        "availability_modes": [mode.value for mode in config.availability_modes],
        "drift_decision": config.drift_decision,
        "environment_seeds": list(config.environment_seeds),
        "horizon": config.horizon,
        "personas": [
            {
                "label": persona.label,
                "persona_id": persona.persona_id,
                "primary": persona.primary,
                "selective_reviews": persona.selective_reviews,
                "sigma_minutes": persona.sigma_minutes,
            }
            for persona in config.personas
        ],
        "policy_replicas": config.policy_replicas,
        "recovery_block_size": config.recovery_block_size,
        "recovery_blocks": config.recovery_blocks,
        "split": config.split,
    }


def population_manifest(config: ExperimentConfig) -> dict[str, object]:
    """Return the declared population independently of evaluator equations."""

    if not isinstance(config, ExperimentConfig):
        raise EvaluationInputError("config must be an ExperimentConfig")
    return _config_manifest(config)


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def population_fingerprint(config: ExperimentConfig) -> str:
    """Return a canonical identity for one declared benchmark population."""

    digest = _canonical_digest(population_manifest(config))
    return f"{EVALUATOR_VERSION}-population.{digest}"


def evaluator_design_fingerprint() -> str:
    """Return a full SHA-256 identity for all evaluator equations."""

    return (
        f"{EVALUATOR_VERSION}-design.{_canonical_digest(evaluation_design_manifest())}"
    )


def evaluator_fingerprint(config: ExperimentConfig) -> str:
    """Return a canonical identifier for the design and declared population."""

    if not isinstance(config, ExperimentConfig):
        raise EvaluationInputError("config must be an ExperimentConfig")
    digest = _canonical_digest(
        {
            "config": _config_manifest(config),
            "design_id": evaluator_design_fingerprint(),
        }
    )
    return f"{EVALUATOR_VERSION}.{digest}"


def _metrics_are_finite(metrics: TrajectoryMetrics) -> None:
    for definition in fields(metrics):
        value = getattr(metrics, definition.name)
        if value is None:
            continue
        items = value if isinstance(value, tuple) else (value,)
        for item in items:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise EvaluationInvariantError(
                    f"metric {definition.name} is not numeric"
                )
            if not math.isfinite(float(item)):
                raise EvaluationInvariantError(
                    f"metric {definition.name} is not finite"
                )


def _require_close(
    actual: float,
    expected: float,
    *,
    field: str,
    tolerance: float = 1e-12,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise EvaluationInvariantError(
            f"{field} does not reconcile: {actual} != {expected}"
        )


def _require_integer_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationInvariantError(f"{field} must be an integer")
    if value < 0:
        raise EvaluationInvariantError(f"{field} must not be negative")
    return value


def _validate_summary_metrics(
    summary: ClusterSummary,
    *,
    persona: Persona,
    config: ExperimentConfig,
) -> None:
    metrics = summary.metrics
    _metrics_are_finite(metrics)
    scalar_count_fields = (
        "maximum_arm_transition",
        "action_count",
        "availability_guardrail_count",
        "one_step_guardrail_count",
        "availability_override_count",
        "review_count",
        "selected_propensity_count",
        "low_propensity_count",
        "brier_score_count",
        "probability_floor_count",
        "arm_probability_count",
        "recovered_count",
    )
    for field in scalar_count_fields:
        value = getattr(metrics, field)
        if value is not None:
            _require_integer_count(value, field=field)
    for field in (
        "evidence_bucket_counts",
        "template_exposures",
        "calibration_counts",
    ):
        for value in getattr(metrics, field):
            _require_integer_count(value, field=field)

    horizon = config.horizon
    replicas = summary.policy_replica_count
    expected_action_count = horizon * replicas
    if metrics.action_count != expected_action_count:
        raise EvaluationInvariantError(
            "action_count disagrees with horizon and replica count"
        )

    unit_interval_fields = (
        "mean_common_expected_regret",
        "mean_conditional_expected_regret",
        "mean_path_opportunity_cost",
        "mean_expected_reward",
        "mean_realized_reward",
        "right_fit_rate",
        "completion_rate",
        "availability_guardrail_rate",
        "one_step_guardrail_rate",
        "availability_override_rate",
        "review_rate",
    )
    for field in unit_interval_fields:
        value = float(getattr(metrics, field))
        if not 0.0 <= value <= 1.0:
            raise EvaluationInvariantError(f"{field} is outside [0, 1]")

    optional_unit_interval_fields = (
        "minimum_propensity",
        "low_propensity_rate",
        "inverse_propensity_ess_ratio",
        "exact_bucket_rate",
        "task_bucket_rate",
        "global_bucket_rate",
        "probability_floor_rate",
        "recovery_rate",
    )
    for field in optional_unit_interval_fields:
        value = getattr(metrics, field)
        if value is not None and not 0.0 <= value <= 1.0:
            raise EvaluationInvariantError(f"{field} is outside [0, 1]")

    if not 0.0 <= (metrics.multiclass_brier_score or 0.0) <= 2.0:
        raise EvaluationInvariantError("multiclass_brier_score is outside [0, 2]")
    maximum_distance = len(DEFAULT_TEMPLATES) - 1
    if not 0.0 <= metrics.mean_common_oracle_distance <= maximum_distance:
        raise EvaluationInvariantError("mean_common_oracle_distance is out of range")
    if not 0.0 <= metrics.mean_conditional_oracle_distance <= maximum_distance:
        raise EvaluationInvariantError(
            "mean_conditional_oracle_distance is out of range"
        )
    if not 0 <= metrics.maximum_arm_transition <= maximum_distance:
        raise EvaluationInvariantError("maximum_arm_transition is out of range")
    if not 1.0 <= metrics.mean_feasible_set_size <= len(DEFAULT_TEMPLATES):
        raise EvaluationInvariantError("mean_feasible_set_size is out of range")

    trace_fields = (
        metrics.common_regret_trace,
        metrics.conditional_regret_trace,
        metrics.path_opportunity_cost_trace,
    )
    if any(len(trace) != horizon for trace in trace_fields):
        raise EvaluationInvariantError("regret trace length disagrees with horizon")
    for common, conditional, path_cost in zip(*trace_fields, strict=True):
        if not (
            0.0 <= common <= 1.0
            and 0.0 <= conditional <= 1.0
            and 0.0 <= path_cost <= 1.0
        ):
            raise EvaluationInvariantError("regret trace value is outside [0, 1]")
        _require_close(
            common,
            conditional + path_cost,
            field="pointwise regret decomposition",
        )

    regret_reconciliations = (
        (
            metrics.common_regret_trace,
            metrics.cumulative_common_expected_regret,
            metrics.mean_common_expected_regret,
            "common regret",
        ),
        (
            metrics.conditional_regret_trace,
            metrics.cumulative_conditional_expected_regret,
            metrics.mean_conditional_expected_regret,
            "conditional regret",
        ),
        (
            metrics.path_opportunity_cost_trace,
            metrics.cumulative_path_opportunity_cost,
            metrics.mean_path_opportunity_cost,
            "path opportunity cost",
        ),
    )
    for trace, cumulative, mean, label in regret_reconciliations:
        _require_close(
            cumulative,
            math.fsum(trace),
            field=f"{label} cumulative",
        )
        _require_close(
            mean,
            cumulative / horizon,
            field=f"{label} mean",
        )
    _require_close(
        metrics.mean_common_expected_regret,
        metrics.mean_conditional_expected_regret + metrics.mean_path_opportunity_cost,
        field="mean regret decomposition",
    )
    _require_close(
        metrics.cumulative_common_expected_regret,
        metrics.cumulative_conditional_expected_regret
        + metrics.cumulative_path_opportunity_cost,
        field="cumulative regret decomposition",
    )

    count_rate_pairs = (
        (
            metrics.availability_guardrail_count,
            metrics.availability_guardrail_rate,
            "availability guardrail",
        ),
        (
            metrics.one_step_guardrail_count,
            metrics.one_step_guardrail_rate,
            "one-step guardrail",
        ),
        (
            metrics.availability_override_count,
            metrics.availability_override_rate,
            "availability override",
        ),
        (metrics.review_count, metrics.review_rate, "review"),
    )
    for count, rate, label in count_rate_pairs:
        if not 0 <= count <= metrics.action_count:
            raise EvaluationInvariantError(f"{label} count is out of range")
        _require_close(
            rate,
            count / metrics.action_count,
            field=f"{label} rate",
        )
    _require_close(
        metrics.mean_feasible_set_size,
        metrics.feasible_set_size_sum / metrics.action_count,
        field="mean feasible set size",
    )

    if len(metrics.template_exposures) != len(DEFAULT_TEMPLATES):
        raise EvaluationInvariantError("template exposure vector has wrong length")
    if sum(metrics.template_exposures) != metrics.action_count:
        raise EvaluationInvariantError("template exposures do not sum to action_count")
    if len(metrics.evidence_bucket_counts) != len(EvidenceBucket):
        raise EvaluationInvariantError("evidence bucket count vector has wrong length")
    calibration_length = len(CALIBRATION_EDGES) - 1
    if not (
        len(metrics.calibration_predicted_sums)
        == len(metrics.calibration_observed_sums)
        == len(metrics.calibration_counts)
        == calibration_length
    ):
        raise EvaluationInvariantError("calibration vectors have an invalid length")
    for predicted, observed, count in zip(
        metrics.calibration_predicted_sums,
        metrics.calibration_observed_sums,
        metrics.calibration_counts,
        strict=True,
    ):
        if not (0.0 <= predicted <= count and 0.0 <= observed <= count):
            raise EvaluationInvariantError(
                "calibration sufficient statistic is out of range"
            )

    propensity_fields = (
        metrics.selected_propensity_count,
        metrics.low_propensity_count,
        metrics.inverse_propensity_sum,
        metrics.inverse_propensity_squared_sum,
        metrics.brier_score_sum,
        metrics.brier_score_count,
        metrics.probability_floor_count,
        metrics.arm_probability_count,
        metrics.minimum_propensity,
        metrics.maximum_inverse_propensity,
        metrics.low_propensity_rate,
        metrics.inverse_propensity_ess_ratio,
        metrics.multiclass_brier_score,
        metrics.exact_bucket_rate,
        metrics.task_bucket_rate,
        metrics.global_bucket_rate,
        metrics.probability_floor_rate,
    )
    if summary.strategy is Strategy.ADAPTIVE:
        if any(value is None for value in propensity_fields):
            raise EvaluationInvariantError(
                "adaptive propensity diagnostics are incomplete"
            )
        selected_count = metrics.selected_propensity_count
        low_count = metrics.low_propensity_count
        weight_sum = metrics.inverse_propensity_sum
        squared_weight_sum = metrics.inverse_propensity_squared_sum
        brier_sum = metrics.brier_score_sum
        brier_count = metrics.brier_score_count
        floor_count = metrics.probability_floor_count
        arm_count = metrics.arm_probability_count
        assert selected_count is not None
        assert low_count is not None
        assert weight_sum is not None
        assert squared_weight_sum is not None
        assert brier_sum is not None
        assert brier_count is not None
        assert floor_count is not None
        assert arm_count is not None
        if selected_count != metrics.action_count:
            raise EvaluationInvariantError(
                "selected propensity count disagrees with action_count"
            )
        if not 0 <= low_count <= selected_count:
            raise EvaluationInvariantError("low propensity count is out of range")
        if brier_count != selected_count or not 0.0 <= brier_sum <= 2 * brier_count:
            raise EvaluationInvariantError("Brier sufficient statistics are invalid")
        if not 0 <= floor_count <= arm_count:
            raise EvaluationInvariantError("probability floor counts are invalid")
        if sum(metrics.evidence_bucket_counts) != selected_count:
            raise EvaluationInvariantError(
                "evidence bucket counts do not sum to action_count"
            )
        if sum(metrics.calibration_counts) != arm_count:
            raise EvaluationInvariantError(
                "calibration counts do not sum to arm_probability_count"
            )
        _require_close(
            math.fsum(metrics.calibration_predicted_sums),
            float(selected_count),
            field="calibration predicted total",
            tolerance=1e-9,
        )
        _require_close(
            math.fsum(metrics.calibration_observed_sums),
            float(selected_count),
            field="calibration observed total",
        )
        if any(
            not float(value).is_integer() for value in metrics.calibration_observed_sums
        ):
            raise EvaluationInvariantError("calibration observed sums must be integral")
        _require_close(
            float(arm_count),
            metrics.feasible_set_size_sum,
            field="arm probability count",
        )
        if weight_sum <= 0 or squared_weight_sum <= 0:
            raise EvaluationInvariantError("inverse propensity sums must be positive")
        _require_close(
            metrics.low_propensity_rate or 0.0,
            low_count / selected_count,
            field="low propensity rate",
        )
        _require_close(
            metrics.inverse_propensity_ess_ratio or 0.0,
            weight_sum**2 / (selected_count * squared_weight_sum),
            field="inverse propensity ESS ratio",
        )
        _require_close(
            metrics.multiclass_brier_score or 0.0,
            brier_sum / brier_count,
            field="Brier score",
        )
        for index, field in enumerate(
            ("exact_bucket_rate", "task_bucket_rate", "global_bucket_rate")
        ):
            value = getattr(metrics, field)
            assert value is not None
            _require_close(
                value,
                metrics.evidence_bucket_counts[index] / selected_count,
                field=field,
            )
        _require_close(
            metrics.probability_floor_rate or 0.0,
            floor_count / arm_count,
            field="probability floor rate",
        )
        if (
            metrics.minimum_propensity is None
            or metrics.maximum_inverse_propensity is None
            or metrics.minimum_propensity <= 0
            or metrics.maximum_inverse_propensity < 1
        ):
            raise EvaluationInvariantError("propensity extrema are invalid")
        _require_close(
            metrics.maximum_inverse_propensity,
            1.0 / metrics.minimum_propensity,
            field="propensity extrema",
        )
    else:
        if any(value is not None for value in propensity_fields):
            raise EvaluationInvariantError(
                "baseline contains adaptive propensity diagnostics"
            )
        if (
            any(metrics.evidence_bucket_counts)
            or any(metrics.calibration_counts)
            or any(metrics.calibration_predicted_sums)
            or any(metrics.calibration_observed_sums)
        ):
            raise EvaluationInvariantError(
                "baseline contains adaptive sufficient statistics"
            )

    recovery_fields = (
        metrics.pre_drift_regret,
        metrics.early_post_drift_auc,
        metrics.late_regret,
        metrics.recovery_rate,
        metrics.conservative_recovery_lag,
        metrics.recovered_count,
        metrics.recovered_lag_sum,
    )
    if persona.is_abrupt:
        if any(value is None for value in recovery_fields):
            raise EvaluationInvariantError(
                "abrupt persona recovery diagnostics are incomplete"
            )
        recovered_count = metrics.recovered_count
        recovered_lag_sum = metrics.recovered_lag_sum
        assert recovered_count is not None
        assert recovered_lag_sum is not None
        if not 0 <= recovered_count <= replicas:
            raise EvaluationInvariantError("recovered_count is out of range")
        recomputed = _drift_metrics(
            metrics.common_regret_trace,
            persona=persona,
            config=config,
        )
        assert recomputed[0] is not None
        assert recomputed[1] is not None
        assert recomputed[2] is not None
        assert metrics.pre_drift_regret is not None
        assert metrics.early_post_drift_auc is not None
        assert metrics.late_regret is not None
        _require_close(
            metrics.pre_drift_regret,
            recomputed[0],
            field="pre-drift regret",
        )
        _require_close(
            metrics.early_post_drift_auc,
            recomputed[1],
            field="early post-drift AUC",
        )
        _require_close(
            metrics.late_regret,
            recomputed[2],
            field="late regret",
        )
        _require_close(
            metrics.recovery_rate or 0.0,
            recovered_count / replicas,
            field="recovery rate",
        )
        needed = config.recovery_block_size * config.recovery_blocks
        maximum_recovered_lag = config.horizon - needed - config.drift_index
        if not (0.0 <= recovered_lag_sum <= recovered_count * maximum_recovered_lag):
            raise EvaluationInvariantError(
                "recovered_lag_sum is outside the searchable recovery window"
            )
        if recovered_count:
            if metrics.recovery_lag is None:
                raise EvaluationInvariantError(
                    "recovered trajectories require a recovery lag"
                )
            _require_close(
                metrics.recovery_lag,
                recovered_lag_sum / recovered_count,
                field="recovered-only lag",
            )
        elif metrics.recovery_lag is not None or recovered_lag_sum != 0.0:
            raise EvaluationInvariantError(
                "unrecovered trajectories have a recovered-only lag"
            )
        censored_lag = config.horizon - config.drift_index + 1
        expected_conservative_lag = (
            recovered_lag_sum + (replicas - recovered_count) * censored_lag
        ) / replicas
        assert metrics.conservative_recovery_lag is not None
        _require_close(
            metrics.conservative_recovery_lag,
            expected_conservative_lag,
            field="conservative recovery lag",
        )
        if replicas == 1:
            expected_lag = recomputed[3]
            expected_rate = recomputed[4]
            expected_conservative = recomputed[5]
            if metrics.recovery_lag != expected_lag:
                raise EvaluationInvariantError(
                    "single-replica recovery lag disagrees with common trace"
                )
            assert expected_rate is not None
            assert expected_conservative is not None
            _require_close(
                metrics.recovery_rate or 0.0,
                expected_rate,
                field="single-replica recovery rate",
            )
            _require_close(
                metrics.conservative_recovery_lag,
                expected_conservative,
                field="single-replica conservative recovery lag",
            )
    elif any(value is not None for value in (*recovery_fields, metrics.recovery_lag)):
        raise EvaluationInvariantError(
            "non-abrupt persona contains recovery diagnostics"
        )


def validate_experiment_result(result: ExperimentResult) -> None:
    """Fail closed unless a result exactly matches its declared Cartesian run."""

    if not isinstance(result, ExperimentResult):
        raise EvaluationInputError("result must be an ExperimentResult")
    if result.schema_version != RESULT_SCHEMA_VERSION:
        raise EvaluationInvariantError("result schema_version is invalid")
    config = result.config
    if result.evaluator_id != evaluator_fingerprint(config):
        raise EvaluationInvariantError("result evaluator_id is invalid")
    if result.design_id != evaluator_design_fingerprint():
        raise EvaluationInvariantError("result design_id is invalid")
    if result.policy_id != LOCKED_POLICY_ID or POLICY_ID != LOCKED_POLICY_ID:
        raise EvaluationInvariantError("result policy_id is not the locked default")
    _require_integer_count(
        result.hard_failure_count,
        field="hard_failure_count",
    )
    if result.hard_failure_count != 0:
        raise EvaluationInvariantError("hard_failure_count must be zero")

    personas = {persona.persona_id: persona for persona in config.personas}
    expected_keys = {
        (persona.persona_id, mode, seed, strategy)
        for persona in config.personas
        for mode in config.availability_modes
        for seed in config.environment_seeds
        for strategy in Strategy
    }
    summaries: dict[
        tuple[str, AvailabilityMode, int, Strategy],
        ClusterSummary,
    ] = {}
    for summary in result.cluster_summaries:
        key = (
            summary.persona_id,
            summary.availability_mode,
            summary.environment_seed,
            summary.strategy,
        )
        if key in summaries:
            raise EvaluationInvariantError(f"duplicate cluster summary: {key}")
        summaries[key] = summary
        persona = personas.get(summary.persona_id)
        if persona is None:
            raise EvaluationInvariantError("summary references an unknown persona")
        if (
            type(summary.persona_primary) is not bool
            or summary.persona_primary is not persona.primary
        ):
            raise EvaluationInvariantError(
                "summary persona_primary disagrees with config"
            )
        if type(summary.availability_mode) is not AvailabilityMode:
            raise EvaluationInvariantError(
                "summary availability_mode has an invalid type"
            )
        if type(summary.strategy) is not Strategy:
            raise EvaluationInvariantError("summary strategy has an invalid type")
        _require_integer_count(
            summary.environment_seed,
            field="summary environment_seed",
        )
        _require_integer_count(
            summary.policy_replica_count,
            field="policy_replica_count",
        )
        expected_replicas = (
            config.policy_replicas if summary.strategy is Strategy.ADAPTIVE else 1
        )
        if summary.policy_replica_count != expected_replicas:
            raise EvaluationInvariantError("summary policy_replica_count is invalid")
        _validate_summary_metrics(
            summary,
            persona=persona,
            config=config,
        )
    actual_keys = set(summaries)
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        unexpected = len(actual_keys - expected_keys)
        raise EvaluationInvariantError(
            f"cluster Cartesian set is incomplete: missing={missing}, "
            f"unexpected={unexpected}"
        )

    abrupt_ids = {
        persona.persona_id for persona in config.personas if persona.is_abrupt
    }
    expected_trace_keys = {
        (persona_id, mode, seed, strategy)
        for persona_id in abrupt_ids
        for mode in config.availability_modes
        for seed in config.environment_seeds
        for strategy in Strategy
    }
    traces: dict[
        tuple[str, AvailabilityMode, int, Strategy],
        TraceRecord,
    ] = {}
    for trace in result.abrupt_traces:
        if type(trace.availability_mode) is not AvailabilityMode:
            raise EvaluationInvariantError(
                "trace availability_mode has an invalid type"
            )
        if type(trace.strategy) is not Strategy:
            raise EvaluationInvariantError("trace strategy has an invalid type")
        _require_integer_count(
            trace.environment_seed,
            field="trace environment_seed",
        )
        key = (
            trace.persona_id,
            trace.availability_mode,
            trace.environment_seed,
            trace.strategy,
        )
        if key in traces:
            raise EvaluationInvariantError(f"duplicate abrupt trace: {key}")
        traces[key] = trace
        if len(trace.common_expected_regret) != config.horizon:
            raise EvaluationInvariantError("abrupt trace length disagrees with horizon")
        matched_summary = summaries.get(key)
        if matched_summary is None:
            raise EvaluationInvariantError(
                "abrupt trace has no matching cluster summary"
            )
        if trace.common_expected_regret != matched_summary.metrics.common_regret_trace:
            raise EvaluationInvariantError(
                "abrupt trace disagrees with cluster common regret"
            )
    actual_trace_keys = set(traces)
    if actual_trace_keys != expected_trace_keys:
        missing = len(expected_trace_keys - actual_trace_keys)
        unexpected = len(actual_trace_keys - expected_trace_keys)
        raise EvaluationInvariantError(
            f"abrupt trace set is incomplete: missing={missing}, "
            f"unexpected={unexpected}"
        )


def _authorize_experiment_run(
    config: ExperimentConfig,
    permit: _LockedEvaluationPermit | None,
) -> _AuthorizedEvaluationRun | None:
    if config.split != "eval":
        if permit is not None:
            raise EvaluationInputError(
                "locked evaluation permits cannot authorize dev or test runs"
            )
        return None
    if config != DEFAULT_EXPERIMENT_CONFIG:
        raise EvaluationInputError(
            "eval requires the exact locked experiment configuration"
        )
    if not isinstance(permit, _LockedEvaluationPermit):
        raise EvaluationInputError(
            "eval requires a single-use publication-runner permit"
        )
    return permit.consume()


def run_experiment(
    config: ExperimentConfig,
    *,
    _eval_permit: _LockedEvaluationPermit | None = None,
) -> ExperimentResult:
    """Execute every declared seed without filtering failures or outliers."""

    if not isinstance(config, ExperimentConfig):
        raise EvaluationInputError("config must be an ExperimentConfig")
    authorization = _authorize_experiment_run(config, _eval_permit)
    if POLICY_ID != LOCKED_POLICY_ID:
        raise EvaluationInvariantError("loaded policy differs from locked default")
    summaries: list[ClusterSummary] = []
    traces: list[TraceRecord] = []
    policy_id = HierarchicalSoftmaxUCB().policy_id
    if policy_id != LOCKED_POLICY_ID:
        raise EvaluationInvariantError("runtime policy differs from locked default")
    evaluator_id = evaluator_fingerprint(config)
    design_id = evaluator_design_fingerprint()
    for persona in config.personas:
        for mode in config.availability_modes:
            for environment_seed in config.environment_seeds:
                environment = generate_environment(
                    persona=persona,
                    availability_mode=mode,
                    environment_seed=environment_seed,
                    config=config,
                    _eval_authorization=authorization,
                )
                for strategy in Strategy:
                    replica_count = (
                        config.policy_replicas if strategy is Strategy.ADAPTIVE else 1
                    )
                    trajectories = tuple(
                        simulate_trajectory(
                            environment,
                            strategy,
                            policy_replica=replica,
                            config=config,
                            _eval_authorization=authorization,
                        )
                        for replica in range(replica_count)
                    )
                    metrics = _aggregate_metrics(trajectories)
                    summaries.append(
                        ClusterSummary(
                            persona_id=persona.persona_id,
                            persona_primary=persona.primary,
                            availability_mode=mode,
                            environment_seed=environment_seed,
                            strategy=strategy,
                            policy_replica_count=replica_count,
                            metrics=metrics,
                        )
                    )
                    if persona.is_abrupt:
                        traces.append(
                            TraceRecord(
                                persona_id=persona.persona_id,
                                availability_mode=mode,
                                environment_seed=environment_seed,
                                strategy=strategy,
                                common_expected_regret=(metrics.common_regret_trace),
                            )
                        )
    result = ExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION,
        config=config,
        evaluator_id=evaluator_id,
        design_id=design_id,
        policy_id=policy_id,
        cluster_summaries=tuple(summaries),
        abrupt_traces=tuple(traces),
        hard_failure_count=0,
    )
    validate_experiment_result(result)
    return result
