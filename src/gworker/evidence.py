"""Publication evidence derived from one complete synthetic experiment.

The statistical report intentionally contains only the registered contrasts and
the compact strategy cells.  This module preserves the additive sufficient
statistics, drift diagnostics, and pointwise trace intervals needed by the
publication renderer.  Every collection has a fixed semantic order so a codec
can serialize it without making presentation or selection decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import cast

from .evaluation import (
    CALIBRATION_EDGES,
    DEFAULT_EXPERIMENT_CONFIG,
    DRIFT_SUMMARY_WINDOW,
    LOCKED_DESIGN_ID,
    LOCKED_EVALUATOR_ID,
    LOCKED_MINIMUM_PROPENSITY,
    LOCKED_POLICY_ID,
    LOCKED_POPULATION_ID,
    LOW_PROPENSITY_THRESHOLD,
    RESULT_SCHEMA_VERSION,
    AvailabilityMode,
    ClusterSummary,
    ExperimentConfig,
    ExperimentResult,
    Strategy,
    evaluator_design_fingerprint,
    evaluator_fingerprint,
    population_fingerprint,
    validate_experiment_result,
)
from .policy import (
    COMPLETION_REWARD_WEIGHT,
    DEFAULT_TEMPLATES,
    FIT_REWARD_WEIGHT,
)
from .reporting import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    LOCKED_BOOTSTRAP_INDICES_SHA256,
    LOCKED_BOOTSTRAP_MODE_INDICES_SHA256,
    LOCKED_BOOTSTRAP_NAMESPACE_SHA256,
    NORMAL_95_CRITICAL_VALUE,
    BootstrapPlan,
    IntervalEstimate,
    StatisticalReport,
    bounded_seed_normal_interval,
    build_bootstrap_plan,
    build_statistical_report,
    seed_cluster_standard_error,
    stratified_bootstrap_interval,
    stratified_seed_standard_error,
)

PUBLICATION_EVIDENCE_SCHEMA_VERSION = "gworker-publication-evidence-v1"
LOCKED_EVIDENCE_KIND = "locked"
FIXTURE_EVIDENCE_KIND = "fixture"
PRIMARY_SCOPE = "primary-macro"
SCENARIO_SCOPE = "scenario"
TRACE_INTERVAL_KIND = "pointwise-normal-95-clipped"
REGISTERED_ENDPOINT = "adaptive-minus-fixed-25-common-expected-regret"
CLAIM_STATUSES = (
    "adaptive-lower-regret",
    "not-distinguished",
    "adaptive-higher-regret",
)
RECOVERY_CONTRAST_METRICS = (
    "pre-drift-regret",
    "early-post-drift-auc",
    "late-regret",
    "recovery-rate",
    "conservative-recovery-lag",
)


class EvidenceInputError(ValueError):
    """Raised when an evidence value object is malformed."""


class EvidenceInvariantError(RuntimeError):
    """Raised when complete inputs cannot produce coherent evidence."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceInputError(f"{field} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise EvidenceInputError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise EvidenceInputError(f"{field} must be finite")
    return 0.0 if result == 0 else result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceInputError(f"{field} must be an integer >= {minimum}")
    return value


def _safe_text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 160
        or len(value.splitlines()) != 1
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EvidenceInputError(f"{field} must be bounded single-line text")
    return value


def _require_close(
    actual: float,
    expected: float,
    *,
    field: str,
    tolerance: float = 1e-12,
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise EvidenceInputError(f"{field} does not reconcile")


def _relation_to_zero(interval: IntervalEstimate) -> str:
    if interval.upper_95 < 0:
        return "below-zero"
    if interval.lower_95 > 0:
        return "above-zero"
    return "includes-zero"


def _claim_status(interval: IntervalEstimate) -> str:
    if interval.upper_95 < 0:
        return "adaptive-lower-regret"
    if interval.lower_95 > 0:
        return "adaptive-higher-regret"
    return "not-distinguished"


@dataclass(frozen=True, slots=True)
class EvidenceScope:
    """One fixed reporting grain."""

    kind: str
    persona_id: str | None
    persona_primary: bool | None
    availability_mode: AvailabilityMode | None

    def __post_init__(self) -> None:
        if self.kind == PRIMARY_SCOPE:
            if (
                self.persona_id is not None
                or self.persona_primary is not None
                or self.availability_mode is not None
            ):
                raise EvidenceInputError(
                    "primary-macro scope cannot contain scenario dimensions"
                )
            return
        if self.kind != SCENARIO_SCOPE:
            raise EvidenceInputError("evidence scope kind is invalid")
        _safe_text(self.persona_id, "persona_id")
        if type(self.persona_primary) is not bool:
            raise EvidenceInputError("scenario persona_primary must be boolean")
        if type(self.availability_mode) is not AvailabilityMode:
            raise EvidenceInputError(
                "scenario availability_mode must be an AvailabilityMode"
            )


@dataclass(frozen=True, slots=True)
class EvidenceCardinalities:
    """Expected or observed inventory for a publication evidence bundle."""

    cluster_summaries: int
    abrupt_traces: int
    raw_trace_points: int
    trajectories: int
    decisions: int
    scenario_strategy_rows: int
    macro_strategy_rows: int
    macro_contrasts: int
    scenario_contrasts: int
    adaptive_diagnostic_scopes: int
    calibration_rows: int
    template_exposure_rows: int
    recovery_rows: int
    recovery_contrast_rows: int
    trace_series: int
    trace_points: int

    def __post_init__(self) -> None:
        for definition in fields(self):
            _integer(getattr(self, definition.name), definition.name)


def expected_publication_cardinalities(
    config: ExperimentConfig,
) -> EvidenceCardinalities:
    """Return the exact inventory implied by a declared population."""

    if not isinstance(config, ExperimentConfig):
        raise EvidenceInputError("config must be an ExperimentConfig")
    persona_count = len(config.personas)
    mode_count = len(config.availability_modes)
    seed_count = len(config.environment_seeds)
    strategy_count = len(Strategy)
    comparator_count = strategy_count - 1
    template_count = len(DEFAULT_TEMPLATES)
    abrupt_count = sum(persona.is_abrupt for persona in config.personas)
    scenario_count = persona_count * mode_count
    scenario_seed_count = scenario_count * seed_count
    trajectory_count = scenario_seed_count * (config.policy_replicas + comparator_count)
    scenario_strategy_rows = scenario_count * strategy_count
    trace_series = abrupt_count * mode_count * strategy_count
    return EvidenceCardinalities(
        cluster_summaries=scenario_seed_count * strategy_count,
        abrupt_traces=trace_series * seed_count,
        raw_trace_points=trace_series * seed_count * config.horizon,
        trajectories=trajectory_count,
        decisions=trajectory_count * config.horizon,
        scenario_strategy_rows=scenario_strategy_rows,
        macro_strategy_rows=strategy_count,
        macro_contrasts=comparator_count,
        scenario_contrasts=scenario_count * comparator_count,
        adaptive_diagnostic_scopes=scenario_count + 1,
        calibration_rows=(scenario_count + 1) * (len(CALIBRATION_EDGES) - 1),
        template_exposure_rows=(
            (scenario_strategy_rows + strategy_count) * template_count
        ),
        recovery_rows=trace_series,
        recovery_contrast_rows=abrupt_count * mode_count * comparator_count,
        trace_series=trace_series,
        trace_points=trace_series * config.horizon,
    )


@dataclass(frozen=True, slots=True)
class RunCompleteness:
    """A fail-closed comparison of the declared and observed inventories."""

    expected: EvidenceCardinalities
    actual: EvidenceCardinalities
    hard_failure_count: int

    def __post_init__(self) -> None:
        if (
            type(self.expected) is not EvidenceCardinalities
            or type(self.actual) is not EvidenceCardinalities
        ):
            raise EvidenceInputError("completeness cardinalities are invalid")
        _integer(self.hard_failure_count, "hard_failure_count")
        if self.hard_failure_count != 0:
            raise EvidenceInputError("hard_failure_count must be zero")
        if self.actual != self.expected:
            raise EvidenceInputError("publication evidence inventory is incomplete")


@dataclass(frozen=True, slots=True)
class StrategyMetricEvidence:
    """Absolute metrics and additive counts for one strategy and scope."""

    scope: EvidenceScope
    strategy: Strategy
    persona_count: int
    mode_count: int
    seed_count_per_cell: int
    trajectory_count: int
    action_count: int
    horizon: int
    mean_common_expected_regret: float
    mean_conditional_expected_regret: float
    mean_path_opportunity_cost: float
    mean_cumulative_common_expected_regret: float
    mean_cumulative_conditional_expected_regret: float
    mean_cumulative_path_opportunity_cost: float
    mean_expected_reward: float
    mean_realized_reward: float
    right_fit_rate: float
    completion_rate: float
    mean_common_oracle_distance: float
    mean_conditional_oracle_distance: float
    common_regret_standard_error: float | None
    availability_guardrail_count: int
    one_step_guardrail_count: int
    availability_override_count: int
    feasible_set_size_sum: float
    review_count: int
    availability_guardrail_rate: float
    one_step_guardrail_rate: float
    availability_override_rate: float
    mean_feasible_set_size: float
    review_rate: float
    maximum_arm_transition: int

    def __post_init__(self) -> None:
        if type(self.scope) is not EvidenceScope:
            raise EvidenceInputError("strategy evidence scope is invalid")
        if type(self.strategy) is not Strategy:
            raise EvidenceInputError("strategy must be a Strategy")
        for field in (
            "persona_count",
            "mode_count",
            "seed_count_per_cell",
            "trajectory_count",
            "action_count",
            "horizon",
        ):
            _integer(getattr(self, field), field, minimum=1)
        if self.action_count != self.trajectory_count * self.horizon:
            raise EvidenceInputError(
                "action_count does not match trajectory_count and horizon"
            )
        unit_fields = (
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
        for field in unit_fields:
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0:
                raise EvidenceInputError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, value)
        cumulative_fields = (
            "mean_cumulative_common_expected_regret",
            "mean_cumulative_conditional_expected_regret",
            "mean_cumulative_path_opportunity_cost",
        )
        for field in cumulative_fields:
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= self.horizon:
                raise EvidenceInputError(f"{field} is outside the horizon")
            object.__setattr__(self, field, value)
        maximum_distance = len(DEFAULT_TEMPLATES) - 1
        for field in (
            "mean_common_oracle_distance",
            "mean_conditional_oracle_distance",
        ):
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= maximum_distance:
                raise EvidenceInputError(f"{field} is out of range")
            object.__setattr__(self, field, value)
        standard_error = self.common_regret_standard_error
        if standard_error is not None:
            standard_error = _finite(
                standard_error,
                "common_regret_standard_error",
            )
            if standard_error < 0:
                raise EvidenceInputError(
                    "common_regret_standard_error must not be negative"
                )
            object.__setattr__(
                self,
                "common_regret_standard_error",
                standard_error,
            )
        if (self.seed_count_per_cell == 1) != (
            self.common_regret_standard_error is None
        ):
            raise EvidenceInputError(
                "common regret standard error presence disagrees with seed count"
            )
        for field in (
            "availability_guardrail_count",
            "one_step_guardrail_count",
            "availability_override_count",
            "review_count",
        ):
            value = _integer(getattr(self, field), field)
            if value > self.action_count:
                raise EvidenceInputError(f"{field} exceeds action_count")
        feasible_sum = _finite(self.feasible_set_size_sum, "feasible_set_size_sum")
        feasible_mean = _finite(
            self.mean_feasible_set_size,
            "mean_feasible_set_size",
        )
        if not 1.0 <= feasible_mean <= len(DEFAULT_TEMPLATES):
            raise EvidenceInputError("mean_feasible_set_size is out of range")
        if not feasible_sum.is_integer():
            raise EvidenceInputError("feasible_set_size_sum must be integer-valued")
        object.__setattr__(self, "feasible_set_size_sum", feasible_sum)
        object.__setattr__(self, "mean_feasible_set_size", feasible_mean)
        transition = _integer(
            self.maximum_arm_transition,
            "maximum_arm_transition",
        )
        if transition > maximum_distance:
            raise EvidenceInputError("maximum_arm_transition is out of range")
        _require_close(
            self.mean_common_expected_regret,
            self.mean_conditional_expected_regret + self.mean_path_opportunity_cost,
            field="mean regret decomposition",
        )
        _require_close(
            self.mean_cumulative_common_expected_regret,
            self.mean_cumulative_conditional_expected_regret
            + self.mean_cumulative_path_opportunity_cost,
            field="cumulative regret decomposition",
        )
        for mean_field, cumulative_field in (
            (
                "mean_common_expected_regret",
                "mean_cumulative_common_expected_regret",
            ),
            (
                "mean_conditional_expected_regret",
                "mean_cumulative_conditional_expected_regret",
            ),
            (
                "mean_path_opportunity_cost",
                "mean_cumulative_path_opportunity_cost",
            ),
        ):
            _require_close(
                getattr(self, cumulative_field),
                getattr(self, mean_field) * self.horizon,
                field=cumulative_field,
                tolerance=1e-10,
            )
        for count_field, rate_field in (
            ("availability_guardrail_count", "availability_guardrail_rate"),
            ("one_step_guardrail_count", "one_step_guardrail_rate"),
            ("availability_override_count", "availability_override_rate"),
            ("review_count", "review_rate"),
        ):
            _require_close(
                getattr(self, rate_field),
                getattr(self, count_field) / self.action_count,
                field=rate_field,
            )
        if (
            self.one_step_guardrail_count + self.availability_override_count
            != self.action_count - self.trajectory_count
        ):
            raise EvidenceInputError("guardrail decision counts do not reconcile")
        if self.availability_override_count > self.availability_guardrail_count:
            raise EvidenceInputError(
                "availability overrides exceed availability guardrails"
            )
        if self.scope.kind == SCENARIO_SCOPE:
            expected_availability_guardrails = (
                0
                if self.scope.availability_mode is AvailabilityMode.UNCONSTRAINED
                else 3 * self.action_count // 4
            )
            if self.availability_guardrail_count != expected_availability_guardrails:
                raise EvidenceInputError(
                    "availability guardrail count disagrees with the frozen schedule"
                )
        if (self.maximum_arm_transition > 1) != (self.availability_override_count > 0):
            raise EvidenceInputError(
                "maximum arm transition disagrees with availability overrides"
            )
        _require_close(
            self.mean_feasible_set_size,
            self.feasible_set_size_sum / self.action_count,
            field="mean_feasible_set_size",
        )
        _require_close(
            self.mean_realized_reward,
            FIT_REWARD_WEIGHT * self.right_fit_rate
            + COMPLETION_REWARD_WEIGHT * self.completion_rate,
            field="realized reward identity",
        )
        if self.strategy is not Strategy.ADAPTIVE and (
            self.review_count != 0 or self.review_rate != 0.0
        ):
            raise EvidenceInputError("baseline strategy contains adaptive reviews")
        if self.strategy is Strategy.MYOPIC_ORACLE and (
            self.mean_conditional_expected_regret != 0.0
            or self.mean_cumulative_conditional_expected_regret != 0.0
            or self.mean_conditional_oracle_distance != 0.0
        ):
            raise EvidenceInputError(
                "myopic oracle must have zero conditional regret and distance"
            )


@dataclass(frozen=True, slots=True)
class AdaptiveDiagnosticEvidence:
    """Pooled adaptive-policy sufficient statistics for one scope."""

    scope: EvidenceScope
    trajectory_count: int
    action_count: int
    review_count: int
    selected_propensity_count: int
    low_propensity_count: int
    inverse_propensity_sum: float
    inverse_propensity_squared_sum: float
    brier_score_sum: float
    brier_score_count: int
    exact_evidence_count: int
    task_evidence_count: int
    global_evidence_count: int
    probability_floor_count: int
    arm_probability_count: int
    minimum_propensity: float
    maximum_inverse_propensity: float
    review_rate: float
    low_propensity_rate: float
    inverse_propensity_ess_ratio: float
    multiclass_brier_score: float
    exact_evidence_rate: float
    task_evidence_rate: float
    global_evidence_rate: float
    probability_floor_rate: float

    def __post_init__(self) -> None:
        if type(self.scope) is not EvidenceScope:
            raise EvidenceInputError("adaptive diagnostic scope is invalid")
        for field in (
            "trajectory_count",
            "action_count",
            "selected_propensity_count",
            "brier_score_count",
            "arm_probability_count",
        ):
            _integer(getattr(self, field), field, minimum=1)
        for field in (
            "review_count",
            "low_propensity_count",
            "exact_evidence_count",
            "task_evidence_count",
            "global_evidence_count",
            "probability_floor_count",
        ):
            _integer(getattr(self, field), field)
        if self.selected_propensity_count != self.action_count:
            raise EvidenceInputError(
                "selected_propensity_count must equal action_count"
            )
        if self.brier_score_count != self.selected_propensity_count:
            raise EvidenceInputError(
                "brier_score_count must equal selected_propensity_count"
            )
        if self.review_count > self.action_count:
            raise EvidenceInputError("review_count exceeds action_count")
        if self.low_propensity_count > self.selected_propensity_count:
            raise EvidenceInputError(
                "low_propensity_count exceeds selected_propensity_count"
            )
        if self.probability_floor_count > self.arm_probability_count:
            raise EvidenceInputError(
                "probability_floor_count exceeds arm_probability_count"
            )
        if (
            self.exact_evidence_count
            + self.task_evidence_count
            + self.global_evidence_count
            != self.selected_propensity_count
        ):
            raise EvidenceInputError(
                "evidence counts do not sum to selected_propensity_count"
            )
        inverse_sum = _finite(
            self.inverse_propensity_sum,
            "inverse_propensity_sum",
        )
        inverse_squared_sum = _finite(
            self.inverse_propensity_squared_sum,
            "inverse_propensity_squared_sum",
        )
        brier_sum = _finite(self.brier_score_sum, "brier_score_sum")
        minimum = _finite(self.minimum_propensity, "minimum_propensity")
        maximum = _finite(
            self.maximum_inverse_propensity,
            "maximum_inverse_propensity",
        )
        if inverse_sum <= 0 or inverse_squared_sum <= 0:
            raise EvidenceInputError("inverse propensity sums must be positive")
        if (
            not LOCKED_MINIMUM_PROPENSITY <= minimum <= 1
            or not 1 <= maximum <= 1.0 / LOCKED_MINIMUM_PROPENSITY
        ):
            raise EvidenceInputError("propensity extrema are out of range")
        impossible_weight_sums = (
            inverse_sum < self.selected_propensity_count,
            inverse_squared_sum < inverse_sum,
            inverse_sum < self.selected_propensity_count - 1 + maximum,
            inverse_squared_sum < self.selected_propensity_count - 1 + maximum**2,
            inverse_sum**2 > self.selected_propensity_count * inverse_squared_sum
            and not math.isclose(
                inverse_sum**2,
                self.selected_propensity_count * inverse_squared_sum,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ),
            inverse_squared_sum > maximum * inverse_sum
            and not math.isclose(
                inverse_squared_sum,
                maximum * inverse_sum,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ),
            inverse_sum > self.selected_propensity_count * maximum,
            inverse_squared_sum > self.selected_propensity_count * maximum**2,
        )
        if any(impossible_weight_sums):
            raise EvidenceInputError("inverse propensity sums violate weight bounds")
        if (self.low_propensity_count > 0) != (minimum < LOW_PROPENSITY_THRESHOLD):
            raise EvidenceInputError(
                "low propensity count disagrees with minimum propensity"
            )
        if not 0 <= brier_sum <= 2 * self.brier_score_count:
            raise EvidenceInputError("brier_score_sum is out of range")
        for field in (
            "review_rate",
            "low_propensity_rate",
            "inverse_propensity_ess_ratio",
            "exact_evidence_rate",
            "task_evidence_rate",
            "global_evidence_rate",
            "probability_floor_rate",
        ):
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0:
                raise EvidenceInputError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, value)
        brier_score = _finite(
            self.multiclass_brier_score,
            "multiclass_brier_score",
        )
        if not 0.0 <= brier_score <= 2.0:
            raise EvidenceInputError("multiclass_brier_score must be in [0, 2]")
        object.__setattr__(self, "multiclass_brier_score", brier_score)
        _require_close(
            maximum,
            1.0 / minimum,
            field="propensity extrema",
        )
        object.__setattr__(self, "inverse_propensity_sum", inverse_sum)
        object.__setattr__(
            self,
            "inverse_propensity_squared_sum",
            inverse_squared_sum,
        )
        object.__setattr__(self, "brier_score_sum", brier_sum)
        object.__setattr__(self, "minimum_propensity", minimum)
        object.__setattr__(self, "maximum_inverse_propensity", maximum)
        identities = (
            ("review_rate", self.review_count / self.action_count),
            (
                "low_propensity_rate",
                self.low_propensity_count / self.selected_propensity_count,
            ),
            (
                "inverse_propensity_ess_ratio",
                inverse_sum**2 / (self.selected_propensity_count * inverse_squared_sum),
            ),
            (
                "multiclass_brier_score",
                brier_sum / self.brier_score_count,
            ),
            (
                "exact_evidence_rate",
                self.exact_evidence_count / self.selected_propensity_count,
            ),
            (
                "task_evidence_rate",
                self.task_evidence_count / self.selected_propensity_count,
            ),
            (
                "global_evidence_rate",
                self.global_evidence_count / self.selected_propensity_count,
            ),
            (
                "probability_floor_rate",
                self.probability_floor_count / self.arm_probability_count,
            ),
        )
        for field, expected in identities:
            _require_close(getattr(self, field), expected, field=field)


@dataclass(frozen=True, slots=True)
class CalibrationBinEvidence:
    """One fixed probability bin with additive calibration statistics."""

    scope: EvidenceScope
    bin_index: int
    lower_bound: float
    upper_bound: float
    upper_inclusive: bool
    predicted_sum: float
    observed_sum: float
    count: int
    mean_predicted_probability: float | None
    observed_frequency: float | None

    def __post_init__(self) -> None:
        if type(self.scope) is not EvidenceScope:
            raise EvidenceInputError("calibration scope is invalid")
        index = _integer(self.bin_index, "bin_index")
        if index >= len(CALIBRATION_EDGES) - 1:
            raise EvidenceInputError("calibration bin_index is out of range")
        lower = _finite(self.lower_bound, "lower_bound")
        upper = _finite(self.upper_bound, "upper_bound")
        if lower != CALIBRATION_EDGES[index] or upper != CALIBRATION_EDGES[index + 1]:
            raise EvidenceInputError("calibration edges differ from the lock")
        if type(self.upper_inclusive) is not bool or self.upper_inclusive is not (
            index == len(CALIBRATION_EDGES) - 2
        ):
            raise EvidenceInputError("calibration upper-bound closure is invalid")
        count = _integer(self.count, "count")
        predicted = _finite(self.predicted_sum, "predicted_sum")
        observed = _finite(self.observed_sum, "observed_sum")
        if not (0.0 <= predicted <= count and 0.0 <= observed <= count):
            raise EvidenceInputError("calibration sufficient statistics are invalid")
        if not observed.is_integer():
            raise EvidenceInputError("calibration observed_sum must be integral")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "predicted_sum", predicted)
        object.__setattr__(self, "observed_sum", observed)
        if count == 0:
            if (
                self.mean_predicted_probability is not None
                or self.observed_frequency is not None
            ):
                raise EvidenceInputError(
                    "empty calibration bins require null derived values"
                )
            return
        if self.mean_predicted_probability is None or self.observed_frequency is None:
            raise EvidenceInputError(
                "non-empty calibration bins require derived values"
            )
        mean_predicted = _finite(
            self.mean_predicted_probability,
            "mean_predicted_probability",
        )
        frequency = _finite(self.observed_frequency, "observed_frequency")
        if not 0.0 <= mean_predicted <= 1.0 or not 0.0 <= frequency <= 1.0:
            raise EvidenceInputError("calibration derived value is out of range")
        if mean_predicted < lower or (
            mean_predicted > upper if self.upper_inclusive else mean_predicted >= upper
        ):
            raise EvidenceInputError(
                "mean_predicted_probability is outside its calibration bin"
            )
        _require_close(
            mean_predicted,
            predicted / count,
            field="mean_predicted_probability",
        )
        _require_close(
            frequency,
            observed / count,
            field="observed_frequency",
        )
        object.__setattr__(
            self,
            "mean_predicted_probability",
            mean_predicted,
        )
        object.__setattr__(self, "observed_frequency", frequency)


@dataclass(frozen=True, slots=True)
class TemplateExposureEvidence:
    """One selected-template count with its explicit denominator."""

    scope: EvidenceScope
    strategy: Strategy
    template_id: str
    focus_seconds: int
    break_seconds: int
    exposure_count: int
    action_count: int
    exposure_rate: float

    def __post_init__(self) -> None:
        if type(self.scope) is not EvidenceScope:
            raise EvidenceInputError("template exposure scope is invalid")
        if type(self.strategy) is not Strategy:
            raise EvidenceInputError("template exposure strategy is invalid")
        _safe_text(self.template_id, "template_id")
        _integer(self.focus_seconds, "focus_seconds", minimum=1)
        _integer(self.break_seconds, "break_seconds", minimum=1)
        count = _integer(self.exposure_count, "exposure_count")
        action_count = _integer(self.action_count, "action_count", minimum=1)
        if count > action_count:
            raise EvidenceInputError("exposure_count exceeds action_count")
        rate = _finite(self.exposure_rate, "exposure_rate")
        if not 0.0 <= rate <= 1.0:
            raise EvidenceInputError("exposure_rate must be in [0, 1]")
        _require_close(rate, count / action_count, field="exposure_rate")
        object.__setattr__(self, "exposure_rate", rate)


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    """Absolute abrupt-drift metrics pooled through replica sufficient stats."""

    persona_id: str
    availability_mode: AvailabilityMode
    strategy: Strategy
    seed_count: int
    trajectory_count: int
    pre_drift_regret: float
    early_post_drift_auc: float
    late_regret: float
    recovered_count: int
    recovered_lag_sum: float
    recovery_rate: float
    recovered_only_lag: float | None
    conservative_recovery_lag: float
    censor_lag: int
    maximum_recovered_lag: int

    def __post_init__(self) -> None:
        _safe_text(self.persona_id, "persona_id")
        if type(self.availability_mode) is not AvailabilityMode:
            raise EvidenceInputError("recovery availability mode is invalid")
        if type(self.strategy) is not Strategy:
            raise EvidenceInputError("recovery strategy is invalid")
        _integer(self.seed_count, "seed_count", minimum=1)
        trajectory_count = _integer(
            self.trajectory_count,
            "trajectory_count",
            minimum=1,
        )
        for field in ("pre_drift_regret", "late_regret"):
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0:
                raise EvidenceInputError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, value)
        early = _finite(self.early_post_drift_auc, "early_post_drift_auc")
        if early < 0:
            raise EvidenceInputError("early_post_drift_auc must not be negative")
        object.__setattr__(self, "early_post_drift_auc", early)
        recovered = _integer(self.recovered_count, "recovered_count")
        if recovered > trajectory_count:
            raise EvidenceInputError("recovered_count exceeds trajectory_count")
        lag_sum = _finite(self.recovered_lag_sum, "recovered_lag_sum")
        if lag_sum < 0 or not lag_sum.is_integer():
            raise EvidenceInputError(
                "recovered_lag_sum must be a non-negative integer-valued sum"
            )
        object.__setattr__(self, "recovered_lag_sum", lag_sum)
        censor = _integer(self.censor_lag, "censor_lag", minimum=1)
        maximum = _integer(
            self.maximum_recovered_lag,
            "maximum_recovered_lag",
        )
        if maximum >= censor or lag_sum > recovered * maximum:
            raise EvidenceInputError("recovery lag sufficient statistics are invalid")
        recovery_rate = _finite(self.recovery_rate, "recovery_rate")
        conservative = _finite(
            self.conservative_recovery_lag,
            "conservative_recovery_lag",
        )
        if not 0.0 <= recovery_rate <= 1.0 or not 0.0 <= conservative <= censor:
            raise EvidenceInputError("recovery derived value is out of range")
        object.__setattr__(self, "recovery_rate", recovery_rate)
        object.__setattr__(
            self,
            "conservative_recovery_lag",
            conservative,
        )
        _require_close(
            recovery_rate,
            recovered / trajectory_count,
            field="recovery_rate",
        )
        if recovered:
            if self.recovered_only_lag is None:
                raise EvidenceInputError(
                    "recovered trajectories require recovered_only_lag"
                )
            lag = _finite(self.recovered_only_lag, "recovered_only_lag")
            _require_close(
                lag,
                lag_sum / recovered,
                field="recovered_only_lag",
            )
            object.__setattr__(self, "recovered_only_lag", lag)
        elif self.recovered_only_lag is not None or lag_sum != 0.0:
            raise EvidenceInputError("unrecovered rows require null recovered_only_lag")
        _require_close(
            conservative,
            (lag_sum + (trajectory_count - recovered) * censor) / trajectory_count,
            field="conservative_recovery_lag",
        )


@dataclass(frozen=True, slots=True)
class RecoveryMetricContrast:
    """One paired adaptive-minus-comparator drift contrast."""

    metric: str
    interval: IntervalEstimate
    interval_relation_to_zero: str

    def __post_init__(self) -> None:
        if self.metric not in RECOVERY_CONTRAST_METRICS:
            raise EvidenceInputError("recovery contrast metric is invalid")
        if type(self.interval) is not IntervalEstimate:
            raise EvidenceInputError("recovery contrast interval is invalid")
        if self.interval_relation_to_zero != _relation_to_zero(self.interval):
            raise EvidenceInputError(
                "recovery contrast relation disagrees with its interval"
            )


@dataclass(frozen=True, slots=True)
class RecoveryContrastEvidence:
    """All registered drift contrasts for one abrupt scenario/comparator."""

    persona_id: str
    availability_mode: AvailabilityMode
    comparator: Strategy
    metrics: tuple[RecoveryMetricContrast, ...]

    def __post_init__(self) -> None:
        _safe_text(self.persona_id, "persona_id")
        if type(self.availability_mode) is not AvailabilityMode:
            raise EvidenceInputError("recovery contrast mode is invalid")
        if (
            type(self.comparator) is not Strategy
            or self.comparator is Strategy.ADAPTIVE
        ):
            raise EvidenceInputError("recovery comparator is invalid")
        try:
            metrics = tuple(self.metrics)
        except TypeError as error:
            raise EvidenceInputError(
                "recovery contrast metrics must be finite"
            ) from error
        object.__setattr__(self, "metrics", metrics)
        if (
            any(type(metric) is not RecoveryMetricContrast for metric in metrics)
            or tuple(metric.metric for metric in metrics) != RECOVERY_CONTRAST_METRICS
        ):
            raise EvidenceInputError(
                "recovery contrast metric set is incomplete or reordered"
            )


@dataclass(frozen=True, slots=True)
class TracePointEvidence:
    """One point on an unsmoothed abrupt-drift seed-level trace."""

    persona_id: str
    availability_mode: AvailabilityMode
    strategy: Strategy
    decision: int
    interval_kind: str
    interval: IntervalEstimate

    def __post_init__(self) -> None:
        _safe_text(self.persona_id, "persona_id")
        if type(self.availability_mode) is not AvailabilityMode:
            raise EvidenceInputError("trace availability mode is invalid")
        if type(self.strategy) is not Strategy:
            raise EvidenceInputError("trace strategy is invalid")
        _integer(self.decision, "decision", minimum=1)
        if self.interval_kind != TRACE_INTERVAL_KIND:
            raise EvidenceInputError("trace interval kind is invalid")
        if type(self.interval) is not IntervalEstimate:
            raise EvidenceInputError("trace interval is invalid")
        for field in ("point_estimate", "lower_95", "upper_95"):
            value = getattr(self.interval, field)
            if not 0.0 <= value <= 1.0:
                raise EvidenceInputError("trace interval value is outside [0, 1]")
        standard_error = self.interval.seed_cluster_standard_error
        if standard_error is None:
            expected_lower = self.interval.point_estimate
            expected_upper = self.interval.point_estimate
        else:
            margin = NORMAL_95_CRITICAL_VALUE * standard_error
            expected_lower = max(0.0, self.interval.point_estimate - margin)
            expected_upper = min(1.0, self.interval.point_estimate + margin)
        _require_close(
            self.interval.lower_95,
            expected_lower,
            field="trace interval lower bound",
        )
        _require_close(
            self.interval.upper_95,
            expected_upper,
            field="trace interval upper bound",
        )


@dataclass(frozen=True, slots=True)
class RegisteredClaimEvidence:
    """Mechanical interpretation of only the pre-registered primary interval."""

    endpoint: str
    status: str
    interval: IntervalEstimate

    def __post_init__(self) -> None:
        if self.endpoint != REGISTERED_ENDPOINT:
            raise EvidenceInputError("registered endpoint is invalid")
        if self.status not in CLAIM_STATUSES:
            raise EvidenceInputError("registered claim status is invalid")
        if type(self.interval) is not IntervalEstimate:
            raise EvidenceInputError("registered claim interval is invalid")
        if self.status != _claim_status(self.interval):
            raise EvidenceInputError(
                "registered claim status disagrees with its interval"
            )


@dataclass(frozen=True, slots=True)
class PublicationEvidence:
    """Complete renderer-independent evidence for one frozen result."""

    schema_version: str
    evidence_kind: str
    result_schema_version: str
    config: ExperimentConfig
    evaluator_id: str
    design_id: str
    population_id: str
    policy_id: str
    completeness: RunCompleteness
    statistics: StatisticalReport
    strategy_metrics: tuple[StrategyMetricEvidence, ...]
    adaptive_diagnostics: tuple[AdaptiveDiagnosticEvidence, ...]
    calibration_bins: tuple[CalibrationBinEvidence, ...]
    template_exposures: tuple[TemplateExposureEvidence, ...]
    recovery_metrics: tuple[RecoveryEvidence, ...]
    recovery_contrasts: tuple[RecoveryContrastEvidence, ...]
    trace_points: tuple[TracePointEvidence, ...]

    def __post_init__(self) -> None:
        collection_names = (
            "strategy_metrics",
            "adaptive_diagnostics",
            "calibration_bins",
            "template_exposures",
            "recovery_metrics",
            "recovery_contrasts",
            "trace_points",
        )
        for name in collection_names:
            try:
                values = tuple(getattr(self, name))
            except TypeError as error:
                raise EvidenceInputError(
                    f"{name} must be a finite collection"
                ) from error
            object.__setattr__(self, name, values)
        _validate_publication_evidence(self)

    @property
    def registered_claim(self) -> RegisteredClaimEvidence:
        """Derive the only claim status from the unrounded primary interval."""

        interval = self.statistics.primary_contrast.interval
        return RegisteredClaimEvidence(
            endpoint=REGISTERED_ENDPOINT,
            status=_claim_status(interval),
            interval=interval,
        )


def _scenario_scopes(config: ExperimentConfig) -> tuple[EvidenceScope, ...]:
    return tuple(
        EvidenceScope(
            kind=SCENARIO_SCOPE,
            persona_id=persona.persona_id,
            persona_primary=persona.primary,
            availability_mode=mode,
        )
        for persona in config.personas
        for mode in config.availability_modes
    )


def _primary_scope() -> EvidenceScope:
    return EvidenceScope(
        kind=PRIMARY_SCOPE,
        persona_id=None,
        persona_primary=None,
        availability_mode=None,
    )


def _validate_publication_evidence(evidence: PublicationEvidence) -> None:
    if evidence.schema_version != PUBLICATION_EVIDENCE_SCHEMA_VERSION:
        raise EvidenceInputError("publication evidence schema version is invalid")
    if evidence.result_schema_version != RESULT_SCHEMA_VERSION:
        raise EvidenceInputError("publication result schema version is invalid")
    if type(evidence.config) is not ExperimentConfig:
        raise EvidenceInputError("publication evidence config is invalid")
    config = evidence.config
    if evidence.evaluator_id != evaluator_fingerprint(config):
        raise EvidenceInputError("publication evaluator_id is invalid")
    if evidence.design_id != evaluator_design_fingerprint():
        raise EvidenceInputError("publication design_id is invalid")
    if evidence.population_id != population_fingerprint(config):
        raise EvidenceInputError("publication population_id is invalid")
    if evidence.policy_id != LOCKED_POLICY_ID:
        raise EvidenceInputError("publication policy_id is invalid")
    if evidence.evidence_kind == LOCKED_EVIDENCE_KIND:
        if (
            config != DEFAULT_EXPERIMENT_CONFIG
            or evidence.evaluator_id != LOCKED_EVALUATOR_ID
            or evidence.design_id != LOCKED_DESIGN_ID
            or evidence.population_id != LOCKED_POPULATION_ID
        ):
            raise EvidenceInputError("locked evidence identities are invalid")
    elif evidence.evidence_kind == FIXTURE_EVIDENCE_KIND:
        if config.split == "eval":
            raise EvidenceInputError("fixture evidence cannot use eval")
    else:
        raise EvidenceInputError("publication evidence kind is invalid")
    if type(evidence.completeness) is not RunCompleteness:
        raise EvidenceInputError("publication completeness is invalid")
    expected_cardinalities = expected_publication_cardinalities(config)
    if evidence.completeness.expected != expected_cardinalities:
        raise EvidenceInputError("publication expected inventory is invalid")
    if type(evidence.statistics) is not StatisticalReport:
        raise EvidenceInputError("publication statistics are invalid")
    statistics = evidence.statistics
    if (
        statistics.evaluator_id != evidence.evaluator_id
        or statistics.design_id != evidence.design_id
        or statistics.policy_id != evidence.policy_id
    ):
        raise EvidenceInputError("publication statistics identities disagree")
    if any(
        contrast.interval.seed_count_per_stratum != len(config.environment_seeds)
        or contrast.interval.stratum_count != len(config.availability_modes)
        for contrast in statistics.macro_contrasts
    ):
        raise EvidenceInputError("macro contrast population is invalid")
    if any(
        contrast.interval.seed_count_per_stratum != len(config.environment_seeds)
        or contrast.interval.stratum_count != 1
        for contrast in statistics.cell_contrasts
    ):
        raise EvidenceInputError("cell contrast population is invalid")
    if evidence.evidence_kind == LOCKED_EVIDENCE_KIND and (
        statistics.bootstrap.resample_count != DEFAULT_BOOTSTRAP_RESAMPLES
        or statistics.bootstrap.namespace_sha256 != LOCKED_BOOTSTRAP_NAMESPACE_SHA256
        or statistics.bootstrap.indices_sha256 != LOCKED_BOOTSTRAP_INDICES_SHA256
        or statistics.bootstrap.mode_indices_sha256
        != LOCKED_BOOTSTRAP_MODE_INDICES_SHA256
    ):
        raise EvidenceInputError("locked evidence bootstrap identity is invalid")
    typed_collections = (
        (evidence.strategy_metrics, StrategyMetricEvidence, "strategy_metrics"),
        (
            evidence.adaptive_diagnostics,
            AdaptiveDiagnosticEvidence,
            "adaptive_diagnostics",
        ),
        (evidence.calibration_bins, CalibrationBinEvidence, "calibration_bins"),
        (
            evidence.template_exposures,
            TemplateExposureEvidence,
            "template_exposures",
        ),
        (evidence.recovery_metrics, RecoveryEvidence, "recovery_metrics"),
        (
            evidence.recovery_contrasts,
            RecoveryContrastEvidence,
            "recovery_contrasts",
        ),
        (evidence.trace_points, TracePointEvidence, "trace_points"),
    )
    for collection, expected_type, name in typed_collections:
        if any(type(row) is not expected_type for row in collection):
            raise EvidenceInputError(f"{name} contains an invalid member")

    scenario_scopes = _scenario_scopes(config)
    all_scopes = (_primary_scope(), *scenario_scopes)
    expected_strategy_keys = tuple(
        (scope, strategy) for scope in all_scopes for strategy in Strategy
    )
    actual_strategy_keys = tuple(
        (row.scope, row.strategy) for row in evidence.strategy_metrics
    )
    if actual_strategy_keys != expected_strategy_keys:
        raise EvidenceInputError("strategy evidence set is incomplete or reordered")
    for row in evidence.strategy_metrics:
        if (
            row.seed_count_per_cell != len(config.environment_seeds)
            or row.horizon != config.horizon
        ):
            raise EvidenceInputError("strategy evidence population is inconsistent")
        if row.scope.kind == PRIMARY_SCOPE:
            expected_personas = sum(persona.primary for persona in config.personas)
            expected_modes = len(config.availability_modes)
        else:
            expected_personas = 1
            expected_modes = 1
        if row.persona_count != expected_personas or row.mode_count != expected_modes:
            raise EvidenceInputError("strategy evidence scope counts are invalid")
        expected_replicas = (
            config.policy_replicas if row.strategy is Strategy.ADAPTIVE else 1
        )
        expected_trajectories = (
            row.persona_count
            * row.mode_count
            * row.seed_count_per_cell
            * expected_replicas
        )
        if row.trajectory_count != expected_trajectories:
            raise EvidenceInputError("strategy trajectory_count is invalid")

    strategy_index = {
        (row.scope, row.strategy): row for row in evidence.strategy_metrics
    }
    primary_scenario_scopes = tuple(
        scope for scope in scenario_scopes if scope.persona_primary is True
    )
    mean_strategy_fields = (
        "mean_common_expected_regret",
        "mean_conditional_expected_regret",
        "mean_path_opportunity_cost",
        "mean_cumulative_common_expected_regret",
        "mean_cumulative_conditional_expected_regret",
        "mean_cumulative_path_opportunity_cost",
        "mean_expected_reward",
        "mean_realized_reward",
        "right_fit_rate",
        "completion_rate",
        "mean_common_oracle_distance",
        "mean_conditional_oracle_distance",
    )
    additive_strategy_fields = (
        "trajectory_count",
        "action_count",
        "availability_guardrail_count",
        "one_step_guardrail_count",
        "availability_override_count",
        "review_count",
    )
    for strategy in Strategy:
        macro_row = strategy_index[(_primary_scope(), strategy)]
        source_rows = tuple(
            strategy_index[(scope, strategy)] for scope in primary_scenario_scopes
        )
        for field in additive_strategy_fields:
            if getattr(macro_row, field) != sum(
                getattr(source, field) for source in source_rows
            ):
                raise EvidenceInputError(f"macro strategy {field} does not reconcile")
        trajectory_count = sum(source.trajectory_count for source in source_rows)
        for field in mean_strategy_fields:
            expected = (
                math.fsum(
                    getattr(source, field) * source.trajectory_count
                    for source in source_rows
                )
                / trajectory_count
            )
            _require_close(
                getattr(macro_row, field),
                expected,
                field=f"macro strategy {field}",
                tolerance=1e-10,
            )
        _require_close(
            macro_row.feasible_set_size_sum,
            math.fsum(source.feasible_set_size_sum for source in source_rows),
            field="macro strategy feasible_set_size_sum",
            tolerance=1e-9,
        )
        if macro_row.maximum_arm_transition != max(
            source.maximum_arm_transition for source in source_rows
        ):
            raise EvidenceInputError(
                "macro strategy maximum_arm_transition does not reconcile"
            )

    expected_diagnostic_scopes = all_scopes
    if tuple(row.scope for row in evidence.adaptive_diagnostics) != (
        expected_diagnostic_scopes
    ):
        raise EvidenceInputError(
            "adaptive diagnostic scope set is incomplete or reordered"
        )
    diagnostic_index = {row.scope: row for row in evidence.adaptive_diagnostics}
    for scope, diagnostic in diagnostic_index.items():
        strategy_row = strategy_index[(scope, Strategy.ADAPTIVE)]
        if (
            diagnostic.trajectory_count != strategy_row.trajectory_count
            or diagnostic.action_count != strategy_row.action_count
            or diagnostic.review_count != strategy_row.review_count
        ):
            raise EvidenceInputError(
                "adaptive diagnostics disagree with strategy evidence"
            )
        _require_close(
            diagnostic.review_rate,
            strategy_row.review_rate,
            field="adaptive review rate",
        )
        _require_close(
            float(diagnostic.arm_probability_count),
            strategy_row.feasible_set_size_sum,
            field="adaptive arm probability count",
            tolerance=1e-9,
        )
    macro_diagnostic = diagnostic_index[_primary_scope()]
    source_diagnostics = tuple(
        diagnostic_index[scope] for scope in primary_scenario_scopes
    )
    additive_diagnostic_fields = (
        "trajectory_count",
        "action_count",
        "review_count",
        "selected_propensity_count",
        "low_propensity_count",
        "brier_score_count",
        "exact_evidence_count",
        "task_evidence_count",
        "global_evidence_count",
        "probability_floor_count",
        "arm_probability_count",
    )
    for field in additive_diagnostic_fields:
        if getattr(macro_diagnostic, field) != sum(
            getattr(source, field) for source in source_diagnostics
        ):
            raise EvidenceInputError(f"macro adaptive {field} does not reconcile")
    for field in (
        "inverse_propensity_sum",
        "inverse_propensity_squared_sum",
        "brier_score_sum",
    ):
        _require_close(
            getattr(macro_diagnostic, field),
            math.fsum(getattr(source, field) for source in source_diagnostics),
            field=f"macro adaptive {field}",
            tolerance=1e-9,
        )
    if macro_diagnostic.minimum_propensity != min(
        source.minimum_propensity for source in source_diagnostics
    ) or macro_diagnostic.maximum_inverse_propensity != max(
        source.maximum_inverse_propensity for source in source_diagnostics
    ):
        raise EvidenceInputError("macro adaptive propensity extrema do not reconcile")

    expected_calibration_keys = tuple(
        (scope, bin_index)
        for scope in expected_diagnostic_scopes
        for bin_index in range(len(CALIBRATION_EDGES) - 1)
    )
    actual_calibration_keys = tuple(
        (row.scope, row.bin_index) for row in evidence.calibration_bins
    )
    if actual_calibration_keys != expected_calibration_keys:
        raise EvidenceInputError("calibration set is incomplete or reordered")
    for scope in expected_diagnostic_scopes:
        diagnostic = diagnostic_index[scope]
        scope_calibration_rows = tuple(
            calibration_row
            for calibration_row in evidence.calibration_bins
            if calibration_row.scope == scope
        )
        if (
            sum(calibration_row.count for calibration_row in scope_calibration_rows)
            != diagnostic.arm_probability_count
        ):
            raise EvidenceInputError(
                "calibration counts disagree with arm_probability_count"
            )
        _require_close(
            math.fsum(
                calibration_row.predicted_sum
                for calibration_row in scope_calibration_rows
            ),
            float(diagnostic.selected_propensity_count),
            field="calibration predicted total",
            tolerance=1e-9,
        )
        _require_close(
            math.fsum(
                calibration_row.observed_sum
                for calibration_row in scope_calibration_rows
            ),
            float(diagnostic.selected_propensity_count),
            field="calibration observed total",
        )
    calibration_index = {
        (row.scope, row.bin_index): row for row in evidence.calibration_bins
    }
    for bin_index in range(len(CALIBRATION_EDGES) - 1):
        macro_bin = calibration_index[(_primary_scope(), bin_index)]
        source_bins = tuple(
            calibration_index[(scope, bin_index)] for scope in primary_scenario_scopes
        )
        if macro_bin.count != sum(source.count for source in source_bins):
            raise EvidenceInputError("macro calibration count does not reconcile")
        _require_close(
            macro_bin.predicted_sum,
            math.fsum(source.predicted_sum for source in source_bins),
            field="macro calibration predicted_sum",
            tolerance=1e-9,
        )
        _require_close(
            macro_bin.observed_sum,
            math.fsum(source.observed_sum for source in source_bins),
            field="macro calibration observed_sum",
        )

    expected_exposure_keys = tuple(
        (scope, strategy, template.template_id)
        for scope in all_scopes
        for strategy in Strategy
        for template in DEFAULT_TEMPLATES
    )
    actual_exposure_keys = tuple(
        (row.scope, row.strategy, row.template_id)
        for row in evidence.template_exposures
    )
    if actual_exposure_keys != expected_exposure_keys:
        raise EvidenceInputError("template exposure set is incomplete or reordered")
    template_index = {template.template_id: template for template in DEFAULT_TEMPLATES}
    for exposure_row in evidence.template_exposures:
        template = template_index[exposure_row.template_id]
        if (
            exposure_row.focus_seconds != template.focus_seconds
            or exposure_row.break_seconds != template.break_seconds
        ):
            raise EvidenceInputError("template exposure metadata is invalid")
    for scope in all_scopes:
        for strategy in Strategy:
            strategy_row = strategy_index[(scope, strategy)]
            strategy_exposure_rows = tuple(
                exposure_row
                for exposure_row in evidence.template_exposures
                if exposure_row.scope == scope and exposure_row.strategy is strategy
            )
            if (
                any(
                    exposure_row.action_count != strategy_row.action_count
                    for exposure_row in strategy_exposure_rows
                )
                or sum(
                    exposure_row.exposure_count
                    for exposure_row in strategy_exposure_rows
                )
                != strategy_row.action_count
            ):
                raise EvidenceInputError(
                    "template exposures disagree with action_count"
                )
    exposure_index = {
        (row.scope, row.strategy, row.template_id): row
        for row in evidence.template_exposures
    }
    for strategy in Strategy:
        for template in DEFAULT_TEMPLATES:
            macro_exposure = exposure_index[
                (_primary_scope(), strategy, template.template_id)
            ]
            source_exposures = tuple(
                exposure_index[(scope, strategy, template.template_id)]
                for scope in primary_scenario_scopes
            )
            if macro_exposure.exposure_count != sum(
                source.exposure_count for source in source_exposures
            ) or macro_exposure.action_count != sum(
                source.action_count for source in source_exposures
            ):
                raise EvidenceInputError("macro template exposure does not reconcile")

    abrupt_personas = tuple(persona for persona in config.personas if persona.is_abrupt)
    expected_recovery_keys = tuple(
        (persona.persona_id, mode, strategy)
        for persona in abrupt_personas
        for mode in config.availability_modes
        for strategy in Strategy
    )
    actual_recovery_keys = tuple(
        (row.persona_id, row.availability_mode, row.strategy)
        for row in evidence.recovery_metrics
    )
    if actual_recovery_keys != expected_recovery_keys:
        raise EvidenceInputError("recovery evidence set is incomplete or reordered")
    for recovery_row in evidence.recovery_metrics:
        if recovery_row.seed_count != len(config.environment_seeds):
            raise EvidenceInputError("recovery seed_count is invalid")
        expected_replicas = (
            config.policy_replicas if recovery_row.strategy is Strategy.ADAPTIVE else 1
        )
        if (
            recovery_row.trajectory_count
            != len(config.environment_seeds) * expected_replicas
        ):
            raise EvidenceInputError("recovery trajectory_count is invalid")
        if (
            recovery_row.censor_lag != config.horizon - config.drift_index + 1
            or recovery_row.maximum_recovered_lag
            != config.horizon
            - config.recovery_block_size * config.recovery_blocks
            - config.drift_index
        ):
            raise EvidenceInputError("recovery window metadata is invalid")

    comparators = tuple(
        strategy for strategy in Strategy if strategy is not Strategy.ADAPTIVE
    )
    expected_recovery_contrast_keys = tuple(
        (persona.persona_id, mode, comparator)
        for persona in abrupt_personas
        for mode in config.availability_modes
        for comparator in comparators
    )
    actual_recovery_contrast_keys = tuple(
        (row.persona_id, row.availability_mode, row.comparator)
        for row in evidence.recovery_contrasts
    )
    if actual_recovery_contrast_keys != expected_recovery_contrast_keys:
        raise EvidenceInputError("recovery contrast set is incomplete or reordered")
    if any(
        metric.interval.seed_count_per_stratum != len(config.environment_seeds)
        or metric.interval.stratum_count != 1
        for contrast in evidence.recovery_contrasts
        for metric in contrast.metrics
    ):
        raise EvidenceInputError("recovery contrast population is invalid")
    recovery_interval_limits = {
        "pre-drift-regret": 1.0,
        "early-post-drift-auc": float(
            min(
                DRIFT_SUMMARY_WINDOW,
                config.horizon - config.drift_index,
            )
        ),
        "late-regret": 1.0,
        "recovery-rate": 1.0,
        "conservative-recovery-lag": float(config.horizon - config.drift_index + 1),
    }
    for contrast in evidence.recovery_contrasts:
        for metric in contrast.metrics:
            limit = recovery_interval_limits[metric.metric]
            if any(
                not -limit <= value <= limit
                for value in (
                    metric.interval.point_estimate,
                    metric.interval.lower_95,
                    metric.interval.upper_95,
                )
            ):
                raise EvidenceInputError(
                    f"recovery contrast {metric.metric} is out of range"
                )

    expected_trace_keys = tuple(
        (persona.persona_id, mode, strategy, decision)
        for persona in abrupt_personas
        for mode in config.availability_modes
        for strategy in Strategy
        for decision in range(1, config.horizon + 1)
    )
    actual_trace_keys = tuple(
        (row.persona_id, row.availability_mode, row.strategy, row.decision)
        for row in evidence.trace_points
    )
    if actual_trace_keys != expected_trace_keys:
        raise EvidenceInputError("trace point set is incomplete or reordered")
    if any(
        row.interval.seed_count_per_stratum != len(config.environment_seeds)
        or row.interval.stratum_count != 1
        for row in evidence.trace_points
    ):
        raise EvidenceInputError("trace interval population is invalid")
    recovery_index = {
        (row.persona_id, row.availability_mode, row.strategy): row
        for row in evidence.recovery_metrics
    }
    recovery_field_by_metric = {
        "pre-drift-regret": "pre_drift_regret",
        "early-post-drift-auc": "early_post_drift_auc",
        "late-regret": "late_regret",
        "recovery-rate": "recovery_rate",
        "conservative-recovery-lag": "conservative_recovery_lag",
    }
    for contrast in evidence.recovery_contrasts:
        adaptive = recovery_index[
            (
                contrast.persona_id,
                contrast.availability_mode,
                Strategy.ADAPTIVE,
            )
        ]
        comparator = recovery_index[
            (
                contrast.persona_id,
                contrast.availability_mode,
                contrast.comparator,
            )
        ]
        for metric in contrast.metrics:
            field = recovery_field_by_metric[metric.metric]
            _require_close(
                metric.interval.point_estimate,
                getattr(adaptive, field) - getattr(comparator, field),
                field=f"recovery contrast {metric.metric}",
                tolerance=1e-10,
            )
    trace_series_index = {
        (persona.persona_id, mode, strategy): tuple(
            row
            for row in evidence.trace_points
            if row.persona_id == persona.persona_id
            and row.availability_mode is mode
            and row.strategy is strategy
        )
        for persona in abrupt_personas
        for mode in config.availability_modes
        for strategy in Strategy
    }
    for key, recovery in recovery_index.items():
        points = tuple(row.interval.point_estimate for row in trace_series_index[key])
        drift = config.drift_index
        pre_values = points[max(0, drift - DRIFT_SUMMARY_WINDOW) : drift]
        early_values = points[drift : min(config.horizon, drift + DRIFT_SUMMARY_WINDOW)]
        late_values = points[-DRIFT_SUMMARY_WINDOW:]
        _require_close(
            recovery.pre_drift_regret,
            math.fsum(pre_values) / len(pre_values),
            field="trace-derived pre-drift regret",
            tolerance=1e-10,
        )
        _require_close(
            recovery.early_post_drift_auc,
            math.fsum(early_values),
            field="trace-derived early post-drift AUC",
            tolerance=1e-10,
        )
        _require_close(
            recovery.late_regret,
            math.fsum(late_values) / len(late_values),
            field="trace-derived late regret",
            tolerance=1e-10,
        )

    actual_cardinalities = EvidenceCardinalities(
        cluster_summaries=evidence.completeness.actual.cluster_summaries,
        abrupt_traces=evidence.completeness.actual.abrupt_traces,
        raw_trace_points=evidence.completeness.actual.raw_trace_points,
        trajectories=evidence.completeness.actual.trajectories,
        decisions=evidence.completeness.actual.decisions,
        scenario_strategy_rows=sum(
            row.scope.kind == SCENARIO_SCOPE for row in evidence.strategy_metrics
        ),
        macro_strategy_rows=sum(
            row.scope.kind == PRIMARY_SCOPE for row in evidence.strategy_metrics
        ),
        macro_contrasts=len(statistics.macro_contrasts),
        scenario_contrasts=len(statistics.cell_contrasts),
        adaptive_diagnostic_scopes=len(evidence.adaptive_diagnostics),
        calibration_rows=len(evidence.calibration_bins),
        template_exposure_rows=len(evidence.template_exposures),
        recovery_rows=len(evidence.recovery_metrics),
        recovery_contrast_rows=len(evidence.recovery_contrasts),
        trace_series=len(
            {
                (row.persona_id, row.availability_mode, row.strategy)
                for row in evidence.trace_points
            }
        ),
        trace_points=len(evidence.trace_points),
    )
    if actual_cardinalities != evidence.completeness.actual:
        raise EvidenceInputError(
            "publication evidence row inventory disagrees with completeness"
        )

    statistical_cells = {
        (cell.persona_id, cell.availability_mode, cell.strategy): cell
        for cell in statistics.strategy_cells
    }
    for row in evidence.strategy_metrics:
        if row.scope.kind != SCENARIO_SCOPE:
            continue
        assert row.scope.persona_id is not None
        assert row.scope.availability_mode is not None
        cell = statistical_cells[
            (row.scope.persona_id, row.scope.availability_mode, row.strategy)
        ]
        if (
            cell.seed_count != row.seed_count_per_cell
            or cell.persona_primary is not row.scope.persona_primary
        ):
            raise EvidenceInputError(
                "strategy evidence disagrees with statistical cell identity"
            )
        for field in (
            "mean_common_expected_regret",
            "mean_conditional_expected_regret",
            "mean_path_opportunity_cost",
            "mean_expected_reward",
            "mean_realized_reward",
            "right_fit_rate",
            "completion_rate",
        ):
            _require_close(
                getattr(row, field),
                getattr(cell, field),
                field=f"strategy cell {field}",
            )
        if row.common_regret_standard_error != cell.common_regret_standard_error:
            raise EvidenceInputError(
                "strategy cell common_regret_standard_error disagrees"
            )
    expected_comparators = tuple(
        strategy for strategy in Strategy if strategy is not Strategy.ADAPTIVE
    )
    if (
        tuple(contrast.comparator for contrast in statistics.macro_contrasts)
        != expected_comparators
    ):
        raise EvidenceInputError("macro contrast order is invalid")
    statistical_cell_index = {
        (
            contrast.persona_id,
            contrast.availability_mode,
            contrast.comparator,
        ): contrast
        for contrast in statistics.cell_contrasts
    }
    scenario_scope_index = {
        (scope.persona_id, scope.availability_mode): scope for scope in scenario_scopes
    }
    for cell_contrast in statistics.cell_contrasts:
        scope = scenario_scope_index[
            (cell_contrast.persona_id, cell_contrast.availability_mode)
        ]
        adaptive_row = strategy_index[(scope, Strategy.ADAPTIVE)]
        comparator_row = strategy_index[(scope, cell_contrast.comparator)]
        _require_close(
            cell_contrast.interval.point_estimate,
            adaptive_row.mean_common_expected_regret
            - comparator_row.mean_common_expected_regret,
            field="cell contrast point estimate",
            tolerance=1e-10,
        )
    for macro_contrast in statistics.macro_contrasts:
        source_points = tuple(
            statistical_cell_index[
                (
                    persona.persona_id,
                    mode,
                    macro_contrast.comparator,
                )
            ].interval.point_estimate
            for persona in config.personas
            if persona.primary
            for mode in config.availability_modes
        )
        _require_close(
            macro_contrast.interval.point_estimate,
            math.fsum(source_points) / len(source_points),
            field="macro contrast point estimate",
            tolerance=1e-10,
        )
    if tuple(
        (
            cell.persona_id,
            cell.availability_mode,
            cell.strategy,
        )
        for cell in statistics.strategy_cells
    ) != tuple(
        (persona.persona_id, mode, strategy)
        for persona in config.personas
        for mode in config.availability_modes
        for strategy in Strategy
    ):
        raise EvidenceInputError("statistical strategy-cell order is invalid")
    if tuple(
        (
            contrast.persona_id,
            contrast.availability_mode,
            contrast.comparator,
        )
        for contrast in statistics.cell_contrasts
    ) != tuple(
        (persona.persona_id, mode, comparator)
        for persona in config.personas
        for mode in config.availability_modes
        for comparator in expected_comparators
    ):
        raise EvidenceInputError("statistical cell-contrast order is invalid")


def _summary_index(
    result: ExperimentResult,
) -> dict[tuple[str, AvailabilityMode, int, Strategy], ClusterSummary]:
    return {
        (
            summary.persona_id,
            summary.availability_mode,
            summary.environment_seed,
            summary.strategy,
        ): summary
        for summary in result.cluster_summaries
    }


def _scope_summaries(
    *,
    scope: EvidenceScope,
    strategy: Strategy,
    config: ExperimentConfig,
    index: dict[tuple[str, AvailabilityMode, int, Strategy], ClusterSummary],
) -> tuple[ClusterSummary, ...]:
    if scope.kind == PRIMARY_SCOPE:
        return tuple(
            index[(persona.persona_id, mode, seed, strategy)]
            for persona in config.personas
            if persona.primary
            for mode in config.availability_modes
            for seed in config.environment_seeds
        )
    assert scope.persona_id is not None
    assert scope.availability_mode is not None
    return tuple(
        index[(scope.persona_id, scope.availability_mode, seed, strategy)]
        for seed in config.environment_seeds
    )


def _weighted_metric(
    summaries: tuple[ClusterSummary, ...],
    field: str,
) -> float:
    trajectory_count = sum(summary.policy_replica_count for summary in summaries)
    if trajectory_count < 1:
        raise EvidenceInvariantError("metric aggregation has no trajectories")
    return (
        math.fsum(
            float(getattr(summary.metrics, field)) * summary.policy_replica_count
            for summary in summaries
        )
        / trajectory_count
    )


def _strategy_common_regret_standard_error(
    *,
    scope: EvidenceScope,
    strategy: Strategy,
    config: ExperimentConfig,
    index: dict[tuple[str, AvailabilityMode, int, Strategy], ClusterSummary],
) -> float | None:
    if scope.kind == SCENARIO_SCOPE:
        assert scope.persona_id is not None
        assert scope.availability_mode is not None
        return seed_cluster_standard_error(
            tuple(
                index[
                    (
                        scope.persona_id,
                        scope.availability_mode,
                        seed,
                        strategy,
                    )
                ].metrics.mean_common_expected_regret
                for seed in config.environment_seeds
            )
        )
    primary_personas = tuple(persona for persona in config.personas if persona.primary)
    values_by_mode = {
        mode: tuple(
            math.fsum(
                index[
                    (
                        persona.persona_id,
                        mode,
                        seed,
                        strategy,
                    )
                ].metrics.mean_common_expected_regret
                for persona in primary_personas
            )
            / len(primary_personas)
            for seed in config.environment_seeds
        )
        for mode in config.availability_modes
    }
    return stratified_seed_standard_error(values_by_mode)


def _required_metric(summary: ClusterSummary, field: str) -> float:
    value = getattr(summary.metrics, field)
    if value is None:
        raise EvidenceInvariantError(f"required recovery metric is missing: {field}")
    return float(value)


def _build_strategy_metric(
    *,
    scope: EvidenceScope,
    strategy: Strategy,
    summaries: tuple[ClusterSummary, ...],
    config: ExperimentConfig,
    index: dict[tuple[str, AvailabilityMode, int, Strategy], ClusterSummary],
) -> StrategyMetricEvidence:
    if not summaries:
        raise EvidenceInvariantError("strategy evidence scope is empty")
    trajectory_count = sum(summary.policy_replica_count for summary in summaries)
    action_count = sum(summary.metrics.action_count for summary in summaries)
    availability_count = sum(
        summary.metrics.availability_guardrail_count for summary in summaries
    )
    step_count = sum(summary.metrics.one_step_guardrail_count for summary in summaries)
    override_count = sum(
        summary.metrics.availability_override_count for summary in summaries
    )
    feasible_sum = math.fsum(
        summary.metrics.feasible_set_size_sum for summary in summaries
    )
    review_count = sum(summary.metrics.review_count for summary in summaries)
    persona_count = (
        sum(persona.primary for persona in config.personas)
        if scope.kind == PRIMARY_SCOPE
        else 1
    )
    mode_count = len(config.availability_modes) if scope.kind == PRIMARY_SCOPE else 1
    return StrategyMetricEvidence(
        scope=scope,
        strategy=strategy,
        persona_count=persona_count,
        mode_count=mode_count,
        seed_count_per_cell=len(config.environment_seeds),
        trajectory_count=trajectory_count,
        action_count=action_count,
        horizon=config.horizon,
        mean_common_expected_regret=_weighted_metric(
            summaries,
            "mean_common_expected_regret",
        ),
        mean_conditional_expected_regret=_weighted_metric(
            summaries,
            "mean_conditional_expected_regret",
        ),
        mean_path_opportunity_cost=_weighted_metric(
            summaries,
            "mean_path_opportunity_cost",
        ),
        mean_cumulative_common_expected_regret=_weighted_metric(
            summaries,
            "cumulative_common_expected_regret",
        ),
        mean_cumulative_conditional_expected_regret=_weighted_metric(
            summaries,
            "cumulative_conditional_expected_regret",
        ),
        mean_cumulative_path_opportunity_cost=_weighted_metric(
            summaries,
            "cumulative_path_opportunity_cost",
        ),
        mean_expected_reward=_weighted_metric(summaries, "mean_expected_reward"),
        mean_realized_reward=_weighted_metric(summaries, "mean_realized_reward"),
        right_fit_rate=_weighted_metric(summaries, "right_fit_rate"),
        completion_rate=_weighted_metric(summaries, "completion_rate"),
        mean_common_oracle_distance=_weighted_metric(
            summaries,
            "mean_common_oracle_distance",
        ),
        mean_conditional_oracle_distance=_weighted_metric(
            summaries,
            "mean_conditional_oracle_distance",
        ),
        common_regret_standard_error=_strategy_common_regret_standard_error(
            scope=scope,
            strategy=strategy,
            config=config,
            index=index,
        ),
        availability_guardrail_count=availability_count,
        one_step_guardrail_count=step_count,
        availability_override_count=override_count,
        feasible_set_size_sum=feasible_sum,
        review_count=review_count,
        availability_guardrail_rate=availability_count / action_count,
        one_step_guardrail_rate=step_count / action_count,
        availability_override_rate=override_count / action_count,
        mean_feasible_set_size=feasible_sum / action_count,
        review_rate=review_count / action_count,
        maximum_arm_transition=max(
            summary.metrics.maximum_arm_transition for summary in summaries
        ),
    )


def _build_adaptive_diagnostic(
    *,
    scope: EvidenceScope,
    summaries: tuple[ClusterSummary, ...],
) -> AdaptiveDiagnosticEvidence:
    if not summaries or any(
        summary.strategy is not Strategy.ADAPTIVE for summary in summaries
    ):
        raise EvidenceInvariantError("adaptive diagnostic scope is invalid")

    def integer_sum(field: str) -> int:
        values = tuple(getattr(summary.metrics, field) for summary in summaries)
        if any(value is None for value in values):
            raise EvidenceInvariantError(
                f"adaptive diagnostic count is missing: {field}"
            )
        return sum(cast(int, value) for value in values)

    def float_sum(field: str) -> float:
        values = tuple(getattr(summary.metrics, field) for summary in summaries)
        if any(value is None for value in values):
            raise EvidenceInvariantError(f"adaptive diagnostic sum is missing: {field}")
        return math.fsum(cast(float, value) for value in values)

    action_count = sum(summary.metrics.action_count for summary in summaries)
    review_count = sum(summary.metrics.review_count for summary in summaries)
    selected_count = integer_sum("selected_propensity_count")
    low_count = integer_sum("low_propensity_count")
    inverse_sum = float_sum("inverse_propensity_sum")
    inverse_squared_sum = float_sum("inverse_propensity_squared_sum")
    brier_sum = float_sum("brier_score_sum")
    brier_count = integer_sum("brier_score_count")
    floor_count = integer_sum("probability_floor_count")
    arm_count = integer_sum("arm_probability_count")
    evidence_counts = tuple(
        sum(summary.metrics.evidence_bucket_counts[index] for summary in summaries)
        for index in range(3)
    )
    minima = tuple(summary.metrics.minimum_propensity for summary in summaries)
    maxima = tuple(summary.metrics.maximum_inverse_propensity for summary in summaries)
    if any(value is None for value in (*minima, *maxima)):
        raise EvidenceInvariantError("adaptive propensity extrema are missing")
    minimum = min(cast(tuple[float, ...], minima))
    maximum = max(cast(tuple[float, ...], maxima))
    return AdaptiveDiagnosticEvidence(
        scope=scope,
        trajectory_count=sum(summary.policy_replica_count for summary in summaries),
        action_count=action_count,
        review_count=review_count,
        selected_propensity_count=selected_count,
        low_propensity_count=low_count,
        inverse_propensity_sum=inverse_sum,
        inverse_propensity_squared_sum=inverse_squared_sum,
        brier_score_sum=brier_sum,
        brier_score_count=brier_count,
        exact_evidence_count=evidence_counts[0],
        task_evidence_count=evidence_counts[1],
        global_evidence_count=evidence_counts[2],
        probability_floor_count=floor_count,
        arm_probability_count=arm_count,
        minimum_propensity=minimum,
        maximum_inverse_propensity=maximum,
        review_rate=review_count / action_count,
        low_propensity_rate=low_count / selected_count,
        inverse_propensity_ess_ratio=(
            inverse_sum**2 / (selected_count * inverse_squared_sum)
        ),
        multiclass_brier_score=brier_sum / brier_count,
        exact_evidence_rate=evidence_counts[0] / selected_count,
        task_evidence_rate=evidence_counts[1] / selected_count,
        global_evidence_rate=evidence_counts[2] / selected_count,
        probability_floor_rate=floor_count / arm_count,
    )


def _build_calibration_rows(
    *,
    scope: EvidenceScope,
    summaries: tuple[ClusterSummary, ...],
) -> tuple[CalibrationBinEvidence, ...]:
    rows: list[CalibrationBinEvidence] = []
    for bin_index in range(len(CALIBRATION_EDGES) - 1):
        predicted = math.fsum(
            summary.metrics.calibration_predicted_sums[bin_index]
            for summary in summaries
        )
        observed = math.fsum(
            summary.metrics.calibration_observed_sums[bin_index]
            for summary in summaries
        )
        count = sum(
            summary.metrics.calibration_counts[bin_index] for summary in summaries
        )
        rows.append(
            CalibrationBinEvidence(
                scope=scope,
                bin_index=bin_index,
                lower_bound=CALIBRATION_EDGES[bin_index],
                upper_bound=CALIBRATION_EDGES[bin_index + 1],
                upper_inclusive=bin_index == len(CALIBRATION_EDGES) - 2,
                predicted_sum=predicted,
                observed_sum=observed,
                count=count,
                mean_predicted_probability=(predicted / count if count else None),
                observed_frequency=observed / count if count else None,
            )
        )
    return tuple(rows)


def _build_exposure_rows(
    *,
    strategy_row: StrategyMetricEvidence,
    summaries: tuple[ClusterSummary, ...],
) -> tuple[TemplateExposureEvidence, ...]:
    rows: list[TemplateExposureEvidence] = []
    for template_index, template in enumerate(DEFAULT_TEMPLATES):
        count = sum(
            summary.metrics.template_exposures[template_index] for summary in summaries
        )
        rows.append(
            TemplateExposureEvidence(
                scope=strategy_row.scope,
                strategy=strategy_row.strategy,
                template_id=template.template_id,
                focus_seconds=template.focus_seconds,
                break_seconds=template.break_seconds,
                exposure_count=count,
                action_count=strategy_row.action_count,
                exposure_rate=count / strategy_row.action_count,
            )
        )
    return tuple(rows)


def _build_recovery_row(
    *,
    persona_id: str,
    mode: AvailabilityMode,
    strategy: Strategy,
    summaries: tuple[ClusterSummary, ...],
    config: ExperimentConfig,
) -> RecoveryEvidence:
    trajectory_count = sum(summary.policy_replica_count for summary in summaries)
    recovered_count = sum(
        cast(int, summary.metrics.recovered_count) for summary in summaries
    )
    recovered_lag_sum = math.fsum(
        cast(float, summary.metrics.recovered_lag_sum) for summary in summaries
    )
    censor_lag = config.horizon - config.drift_index + 1
    maximum_recovered_lag = (
        config.horizon
        - config.recovery_block_size * config.recovery_blocks
        - config.drift_index
    )

    def weighted_required(field: str) -> float:
        return (
            math.fsum(
                _required_metric(summary, field) * summary.policy_replica_count
                for summary in summaries
            )
            / trajectory_count
        )

    return RecoveryEvidence(
        persona_id=persona_id,
        availability_mode=mode,
        strategy=strategy,
        seed_count=len(config.environment_seeds),
        trajectory_count=trajectory_count,
        pre_drift_regret=weighted_required("pre_drift_regret"),
        early_post_drift_auc=weighted_required("early_post_drift_auc"),
        late_regret=weighted_required("late_regret"),
        recovered_count=recovered_count,
        recovered_lag_sum=recovered_lag_sum,
        recovery_rate=recovered_count / trajectory_count,
        recovered_only_lag=(
            recovered_lag_sum / recovered_count if recovered_count else None
        ),
        conservative_recovery_lag=weighted_required("conservative_recovery_lag"),
        censor_lag=censor_lag,
        maximum_recovered_lag=maximum_recovered_lag,
    )


def _build_recovery_contrast(
    *,
    persona_id: str,
    mode: AvailabilityMode,
    comparator: Strategy,
    config: ExperimentConfig,
    index: dict[tuple[str, AvailabilityMode, int, Strategy], ClusterSummary],
    plan: BootstrapPlan,
) -> RecoveryContrastEvidence:
    field_by_metric = {
        "pre-drift-regret": "pre_drift_regret",
        "early-post-drift-auc": "early_post_drift_auc",
        "late-regret": "late_regret",
        "recovery-rate": "recovery_rate",
        "conservative-recovery-lag": "conservative_recovery_lag",
    }
    metrics: list[RecoveryMetricContrast] = []
    for metric in RECOVERY_CONTRAST_METRICS:
        field = field_by_metric[metric]
        effects = tuple(
            _required_metric(
                index[(persona_id, mode, seed, Strategy.ADAPTIVE)],
                field,
            )
            - _required_metric(
                index[(persona_id, mode, seed, comparator)],
                field,
            )
            for seed in config.environment_seeds
        )
        interval = stratified_bootstrap_interval({mode: effects}, plan)
        metrics.append(
            RecoveryMetricContrast(
                metric=metric,
                interval=interval,
                interval_relation_to_zero=_relation_to_zero(interval),
            )
        )
    return RecoveryContrastEvidence(
        persona_id=persona_id,
        availability_mode=mode,
        comparator=comparator,
        metrics=tuple(metrics),
    )


def _build_trace_points(
    *,
    result: ExperimentResult,
) -> tuple[TracePointEvidence, ...]:
    config = result.config
    trace_index = {
        (
            trace.persona_id,
            trace.availability_mode,
            trace.environment_seed,
            trace.strategy,
        ): trace
        for trace in result.abrupt_traces
    }
    rows: list[TracePointEvidence] = []
    for persona in config.personas:
        if not persona.is_abrupt:
            continue
        for mode in config.availability_modes:
            for strategy in Strategy:
                seed_traces = tuple(
                    trace_index[
                        (persona.persona_id, mode, seed, strategy)
                    ].common_expected_regret
                    for seed in config.environment_seeds
                )
                for decision_index in range(config.horizon):
                    values = tuple(trace[decision_index] for trace in seed_traces)
                    interval = bounded_seed_normal_interval(
                        values,
                        lower_bound=0.0,
                        upper_bound=1.0,
                    )
                    rows.append(
                        TracePointEvidence(
                            persona_id=persona.persona_id,
                            availability_mode=mode,
                            strategy=strategy,
                            decision=decision_index + 1,
                            interval_kind=TRACE_INTERVAL_KIND,
                            interval=interval,
                        )
                    )
    return tuple(rows)


def _collect_publication_evidence(
    result: ExperimentResult,
    *,
    resample_count: int,
    evidence_kind: str,
) -> PublicationEvidence:
    """Build the complete fixed-order evidence bundle without rendering it."""

    validate_experiment_result(result)
    config = result.config
    statistics = build_statistical_report(
        result,
        resample_count=resample_count,
    )
    plan = build_bootstrap_plan(
        config.environment_seeds,
        availability_modes=config.availability_modes,
        evaluator_id=result.evaluator_id,
        resample_count=resample_count,
    )
    if (
        plan.namespace_sha256 != statistics.bootstrap.namespace_sha256
        or plan.indices_sha256 != statistics.bootstrap.indices_sha256
    ):
        raise EvidenceInvariantError(
            "publication bootstrap plan disagrees with statistical report"
        )
    index = _summary_index(result)
    all_scopes = (_primary_scope(), *_scenario_scopes(config))

    strategy_rows: list[StrategyMetricEvidence] = []
    exposure_rows: list[TemplateExposureEvidence] = []
    for scope in all_scopes:
        for strategy in Strategy:
            summaries = _scope_summaries(
                scope=scope,
                strategy=strategy,
                config=config,
                index=index,
            )
            strategy_row = _build_strategy_metric(
                scope=scope,
                strategy=strategy,
                summaries=summaries,
                config=config,
                index=index,
            )
            strategy_rows.append(strategy_row)
            exposure_rows.extend(
                _build_exposure_rows(
                    strategy_row=strategy_row,
                    summaries=summaries,
                )
            )

    diagnostic_rows: list[AdaptiveDiagnosticEvidence] = []
    calibration_rows: list[CalibrationBinEvidence] = []
    for scope in all_scopes:
        summaries = _scope_summaries(
            scope=scope,
            strategy=Strategy.ADAPTIVE,
            config=config,
            index=index,
        )
        diagnostic_rows.append(
            _build_adaptive_diagnostic(scope=scope, summaries=summaries)
        )
        calibration_rows.extend(
            _build_calibration_rows(scope=scope, summaries=summaries)
        )

    recovery_rows: list[RecoveryEvidence] = []
    recovery_contrast_rows: list[RecoveryContrastEvidence] = []
    comparators = tuple(
        strategy for strategy in Strategy if strategy is not Strategy.ADAPTIVE
    )
    for persona in config.personas:
        if not persona.is_abrupt:
            continue
        for mode in config.availability_modes:
            for strategy in Strategy:
                summaries = tuple(
                    index[(persona.persona_id, mode, seed, strategy)]
                    for seed in config.environment_seeds
                )
                recovery_rows.append(
                    _build_recovery_row(
                        persona_id=persona.persona_id,
                        mode=mode,
                        strategy=strategy,
                        summaries=summaries,
                        config=config,
                    )
                )
            for comparator in comparators:
                recovery_contrast_rows.append(
                    _build_recovery_contrast(
                        persona_id=persona.persona_id,
                        mode=mode,
                        comparator=comparator,
                        config=config,
                        index=index,
                        plan=plan,
                    )
                )

    trace_points = _build_trace_points(result=result)
    actual_cardinalities = EvidenceCardinalities(
        cluster_summaries=len(result.cluster_summaries),
        abrupt_traces=len(result.abrupt_traces),
        raw_trace_points=sum(
            len(trace.common_expected_regret) for trace in result.abrupt_traces
        ),
        trajectories=sum(
            summary.policy_replica_count for summary in result.cluster_summaries
        ),
        decisions=sum(
            summary.metrics.action_count for summary in result.cluster_summaries
        ),
        scenario_strategy_rows=sum(
            row.scope.kind == SCENARIO_SCOPE for row in strategy_rows
        ),
        macro_strategy_rows=sum(
            row.scope.kind == PRIMARY_SCOPE for row in strategy_rows
        ),
        macro_contrasts=len(statistics.macro_contrasts),
        scenario_contrasts=len(statistics.cell_contrasts),
        adaptive_diagnostic_scopes=len(diagnostic_rows),
        calibration_rows=len(calibration_rows),
        template_exposure_rows=len(exposure_rows),
        recovery_rows=len(recovery_rows),
        recovery_contrast_rows=len(recovery_contrast_rows),
        trace_series=len(
            {
                (row.persona_id, row.availability_mode, row.strategy)
                for row in trace_points
            }
        ),
        trace_points=len(trace_points),
    )
    expected_cardinalities = expected_publication_cardinalities(config)
    completeness = RunCompleteness(
        expected=expected_cardinalities,
        actual=actual_cardinalities,
        hard_failure_count=result.hard_failure_count,
    )
    return PublicationEvidence(
        schema_version=PUBLICATION_EVIDENCE_SCHEMA_VERSION,
        evidence_kind=evidence_kind,
        result_schema_version=result.schema_version,
        config=config,
        evaluator_id=result.evaluator_id,
        design_id=result.design_id,
        population_id=population_fingerprint(config),
        policy_id=result.policy_id,
        completeness=completeness,
        statistics=statistics,
        strategy_metrics=tuple(strategy_rows),
        adaptive_diagnostics=tuple(diagnostic_rows),
        calibration_bins=tuple(calibration_rows),
        template_exposures=tuple(exposure_rows),
        recovery_metrics=tuple(recovery_rows),
        recovery_contrasts=tuple(recovery_contrast_rows),
        trace_points=trace_points,
    )


def build_fixture_publication_evidence(
    result: ExperimentResult,
    *,
    resample_count: int,
) -> PublicationEvidence:
    """Build non-publishable evidence for a dev/test fixture."""

    if not isinstance(result, ExperimentResult):
        raise EvidenceInputError("result must be an ExperimentResult")
    if result.config.split == "eval":
        raise EvidenceInputError("fixture evidence cannot use the eval namespace")
    return _collect_publication_evidence(
        result,
        resample_count=resample_count,
        evidence_kind=FIXTURE_EVIDENCE_KIND,
    )


def build_publication_evidence(result: ExperimentResult) -> PublicationEvidence:
    """Build only the exact locked-v3 publication evidence."""

    if not isinstance(result, ExperimentResult):
        raise EvidenceInputError("result must be an ExperimentResult")
    if result.config != DEFAULT_EXPERIMENT_CONFIG:
        raise EvidenceInputError(
            "publication evidence requires the exact locked population"
        )
    literal_identities = (
        (result.evaluator_id, LOCKED_EVALUATOR_ID, "evaluator"),
        (result.design_id, LOCKED_DESIGN_ID, "design"),
        (
            population_fingerprint(result.config),
            LOCKED_POPULATION_ID,
            "population",
        ),
        (result.policy_id, LOCKED_POLICY_ID, "policy"),
    )
    for actual, expected, label in literal_identities:
        if actual != expected:
            raise EvidenceInputError(f"locked {label} identity is invalid")
    if result.schema_version != RESULT_SCHEMA_VERSION:
        raise EvidenceInputError("locked result schema version is invalid")
    evidence = _collect_publication_evidence(
        result,
        resample_count=DEFAULT_BOOTSTRAP_RESAMPLES,
        evidence_kind=LOCKED_EVIDENCE_KIND,
    )
    bootstrap = evidence.statistics.bootstrap
    if (
        bootstrap.namespace_sha256 != LOCKED_BOOTSTRAP_NAMESPACE_SHA256
        or bootstrap.indices_sha256 != LOCKED_BOOTSTRAP_INDICES_SHA256
        or bootstrap.mode_indices_sha256 != LOCKED_BOOTSTRAP_MODE_INDICES_SHA256
    ):
        raise EvidenceInvariantError("locked bootstrap digests are invalid")
    return evidence
