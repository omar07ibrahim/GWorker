"""Pure one-step replay diagnostics for reviewed policy decisions.

This module performs finite-snapshot descriptive arithmetic only. It does not
read journals, estimate a sequential policy value, correct review selection,
or import the locked synthetic evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .policy import (
    COMPLETION_REWARD_WEIGHT,
    FIT_REWARD_WEIGHT,
    MAX_DECISION_SEQUENCE,
    MAX_TEMPLATES,
    DurationFit,
)

MAX_REPLAY_ROWS = 100_000
PROBABILITY_SUM_TOLERANCE = 1e-12


class ReplayInputError(ValueError):
    """Raised when a replay value or finite snapshot is invalid."""


class ReplayInvariantError(RuntimeError):
    """Raised when an internal negative-control invariant fails."""


class ReplayTargetKind(StrEnum):
    """Supported one-step target distributions."""

    BEHAVIOR = "behavior"
    SCORE_TEMPERATURE = "score-temperature"
    UNIFORM = "uniform"


class ReplayReadiness(StrEnum):
    """Descriptive support state for one replay summary."""

    INSUFFICIENT_REVIEWS = "insufficient-reviews"
    UNSTABLE_SUPPORT = "unstable-support"
    REPORTABLE = "reportable"


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayInputError(f"{field} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise ReplayInputError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ReplayInputError(f"{field} must be finite")
    if result == 0:
        return 0.0
    return result


def _positive_integer(value: object, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayInputError(f"{field} must be an integer")
    if not 1 <= value <= maximum:
        raise ReplayInputError(f"{field} must be between 1 and {maximum}")
    return value


def _template_identifier(value: object, field: str = "template_id") -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 80:
        raise ReplayInputError(f"{field} must contain 1 to 80 characters")
    if not all(
        character.isascii()
        and (character.islower() or character.isdigit() or character in ".-_")
        for character in value
    ):
        raise ReplayInputError(
            f"{field} may contain lowercase ASCII letters, digits, dot, dash, "
            "and underscore"
        )
    return value


def _checked_sum(values: Sequence[float], field: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ReplayInputError(f"{field} exceeds finite arithmetic") from error
    if not math.isfinite(result):
        raise ReplayInputError(f"{field} exceeds finite arithmetic")
    if result == 0:
        return 0.0
    return result


@dataclass(frozen=True, slots=True)
class ReplayArm:
    """One feasible action reconstructed at a historical decision."""

    template_id: str
    score: float
    behavior_probability: float

    def __post_init__(self) -> None:
        _template_identifier(self.template_id)
        score = _finite_float(self.score, "score")
        probability = _finite_float(
            self.behavior_probability,
            "behavior_probability",
        )
        if not 0 < probability <= 1:
            raise ReplayInputError("behavior_probability must be in (0, 1]")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "behavior_probability", probability)


@dataclass(frozen=True, slots=True)
class ReplayRow:
    """Privacy-minimal reviewed row supplied by verified policy replay."""

    decision_sequence: int
    selected_template_id: str
    fit: DurationFit
    objective_completed: bool
    arms: tuple[ReplayArm, ...]

    def __post_init__(self) -> None:
        _positive_integer(
            self.decision_sequence,
            "decision_sequence",
            maximum=MAX_DECISION_SEQUENCE,
        )
        _template_identifier(self.selected_template_id, "selected_template_id")
        if not isinstance(self.fit, DurationFit):
            raise ReplayInputError("fit must be a DurationFit")
        if not isinstance(self.objective_completed, bool):
            raise ReplayInputError("objective_completed must be boolean")
        if not isinstance(self.arms, tuple):
            raise ReplayInputError("arms must be a tuple")
        if not 1 <= len(self.arms) <= MAX_TEMPLATES:
            raise ReplayInputError(
                f"arms must contain between 1 and {MAX_TEMPLATES} items"
            )
        if not all(isinstance(arm, ReplayArm) for arm in self.arms):
            raise ReplayInputError("arms must contain ReplayArm instances")
        identifiers = tuple(arm.template_id for arm in self.arms)
        if len(set(identifiers)) != len(identifiers):
            raise ReplayInputError("arms must have unique template_id values")
        if self.selected_template_id not in identifiers:
            raise ReplayInputError("selected_template_id must identify a feasible arm")
        probability_sum = _checked_sum(
            tuple(arm.behavior_probability for arm in self.arms),
            "behavior probability sum",
        )
        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_SUM_TOLERANCE,
        ):
            raise ReplayInputError("behavior probabilities must sum to one")

    @property
    def reward(self) -> float:
        """Return the declared bounded explicit-review reward."""

        fit_reward = 1.0 if self.fit is DurationFit.JUST_RIGHT else 0.0
        completion_reward = 1.0 if self.objective_completed else 0.0
        return (
            FIT_REWARD_WEIGHT * fit_reward
            + COMPLETION_REWARD_WEIGHT * completion_reward
        )

    @property
    def selected_arm(self) -> ReplayArm:
        """Return the selected arm using the exact stored probability object."""

        return next(
            arm
            for arm in self.arms
            if arm.template_id == self.selected_template_id
        )


@dataclass(frozen=True, slots=True)
class ReplayTarget:
    """A declared one-step target distribution."""

    kind: ReplayTargetKind
    temperature: float | None = None
    probability_floor: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ReplayTargetKind):
            raise ReplayInputError("kind must be a ReplayTargetKind")
        if self.kind is ReplayTargetKind.SCORE_TEMPERATURE:
            if self.temperature is None or self.probability_floor is None:
                raise ReplayInputError(
                    "score-temperature target requires temperature and "
                    "probability_floor"
                )
            temperature = _finite_float(self.temperature, "temperature")
            floor = _finite_float(self.probability_floor, "probability_floor")
            if temperature <= 0:
                raise ReplayInputError("temperature must be positive")
            if not 0 <= floor < 1:
                raise ReplayInputError("probability_floor must be in [0, 1)")
            object.__setattr__(self, "temperature", temperature)
            object.__setattr__(self, "probability_floor", floor)
            return
        if self.temperature is not None or self.probability_floor is not None:
            raise ReplayInputError(
                "behavior and uniform targets do not accept numeric parameters"
            )

    @classmethod
    def behavior(cls) -> ReplayTarget:
        """Return the exact behavior-replay negative-control target."""

        return cls(kind=ReplayTargetKind.BEHAVIOR)

    @classmethod
    def uniform(cls) -> ReplayTarget:
        """Return the uniform-over-feasible non-adaptive target."""

        return cls(kind=ReplayTargetKind.UNIFORM)

    @classmethod
    def score_temperature(
        cls,
        *,
        temperature: float,
        probability_floor: float = 0.0,
    ) -> ReplayTarget:
        """Return a stable softmax reweighting of replayed scores."""

        return cls(
            kind=ReplayTargetKind.SCORE_TEMPERATURE,
            temperature=temperature,
            probability_floor=probability_floor,
        )


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Support thresholds and the declared clipping sensitivity."""

    minimum_reviews: int = 12
    minimum_raw_ess_ratio: float = 0.25
    maximum_raw_weight: float = 10.0
    clip_weight: float = 5.0

    def __post_init__(self) -> None:
        _positive_integer(
            self.minimum_reviews,
            "minimum_reviews",
            maximum=MAX_REPLAY_ROWS,
        )
        minimum_ess = _finite_float(
            self.minimum_raw_ess_ratio,
            "minimum_raw_ess_ratio",
        )
        if not 0 <= minimum_ess <= 1:
            raise ReplayInputError("minimum_raw_ess_ratio must be in [0, 1]")
        maximum_weight = _finite_float(
            self.maximum_raw_weight,
            "maximum_raw_weight",
        )
        if maximum_weight < 1:
            raise ReplayInputError("maximum_raw_weight must be at least one")
        clip_weight = _finite_float(self.clip_weight, "clip_weight")
        if clip_weight < 1:
            raise ReplayInputError("clip_weight must be at least one")
        object.__setattr__(self, "minimum_raw_ess_ratio", minimum_ess)
        object.__setattr__(self, "maximum_raw_weight", maximum_weight)
        object.__setattr__(self, "clip_weight", clip_weight)


@dataclass(frozen=True, slots=True)
class TemplateReplaySummary:
    """Sufficient statistics for one template across reviewed rows."""

    template_id: str
    feasible_count: int
    reviewed_count: int
    target_mass: float
    raw_weight_sum: float
    clipped_weight_sum: float
    raw_reward_contribution: float | None
    clipped_reward_contribution: float | None
    minimum_selected_behavior_probability: float | None
    maximum_raw_weight: float | None


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Aggregate-only descriptive diagnostics for one declared target."""

    target: ReplayTarget
    readiness: ReplayReadiness
    reviewed_count: int
    observed_behavior_mean: float | None
    raw_inverse_propensity: float | None
    raw_self_normalized: float | None
    clipped_inverse_propensity: float | None
    clipped_self_normalized: float | None
    raw_effective_sample_size: float | None
    clipped_effective_sample_size: float | None
    mean_raw_weight: float | None
    maximum_raw_weight: float | None
    minimum_selected_behavior_probability: float | None
    clipped_row_count: int
    removed_raw_weight_mass: float
    raw_effective_sample_size_ratio: float | None
    clipped_effective_sample_size_ratio: float | None
    templates: tuple[TemplateReplaySummary, ...]


@dataclass(frozen=True, slots=True)
class ReplayNonClaims:
    """Explicitly false interpretation flags carried by every report."""

    review_selection_corrected: bool = False
    target_policy_value_estimated: bool = False
    sequential_policy_value_estimated: bool = False
    causal_effect_estimated: bool = False
    locked_evaluation_used: bool = False

    def __post_init__(self) -> None:
        flags = (
            self.review_selection_corrected,
            self.target_policy_value_estimated,
            self.sequential_policy_value_estimated,
            self.causal_effect_estimated,
            self.locked_evaluation_used,
        )
        if not all(isinstance(flag, bool) for flag in flags):
            raise ReplayInputError("replay interpretation flags must be boolean")
        if any(flags):
            raise ReplayInputError("replay interpretation flags must remain false")


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Candidate diagnostics plus an executable behavior negative control."""

    candidate: ReplaySummary
    behavior_control: ReplaySummary
    config: ReplayConfig
    nonclaims: ReplayNonClaims


def _score_probabilities(
    row: ReplayRow,
    target: ReplayTarget,
) -> tuple[float, ...]:
    assert target.temperature is not None
    assert target.probability_floor is not None
    count = len(row.arms)
    floor = target.probability_floor
    if not floor < 1.0 / count:
        raise ReplayInputError(
            "probability_floor must be below one divided by every row's "
            "feasible arm count"
        )
    maximum = max(arm.score for arm in row.arms)
    weights: list[float] = []
    for arm in row.arms:
        delta = arm.score - maximum
        if delta == 0:
            weights.append(1.0)
            continue
        try:
            scaled = delta / target.temperature
        except OverflowError:
            scaled = -math.inf
        weights.append(0.0 if scaled == -math.inf else math.exp(scaled))
    denominator = _checked_sum(tuple(weights), "softmax denominator")
    remaining_mass = 1.0 - floor * count
    probabilities = tuple(
        floor + remaining_mass * weight / denominator for weight in weights
    )
    probability_sum = _checked_sum(probabilities, "target probability sum")
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise ReplayInvariantError("score target probabilities do not sum to one")
    return probabilities


def _target_probabilities(
    row: ReplayRow,
    target: ReplayTarget,
) -> tuple[float, ...]:
    if target.kind is ReplayTargetKind.BEHAVIOR:
        return tuple(arm.behavior_probability for arm in row.arms)
    if target.kind is ReplayTargetKind.UNIFORM:
        probability = 1.0 / len(row.arms)
        return tuple(probability for _ in row.arms)
    return _score_probabilities(row, target)


def _effective_sample_size(weights: tuple[float, ...]) -> float | None:
    maximum = max(weights, default=0.0)
    if maximum == 0:
        return None
    scaled = tuple(weight / maximum for weight in weights)
    numerator = _checked_sum(scaled, "scaled weight sum")
    denominator = _checked_sum(
        tuple(weight * weight for weight in scaled),
        "scaled squared-weight sum",
    )
    if denominator == 0:
        return None
    return numerator * numerator / denominator


def _readiness(
    *,
    count: int,
    raw_ess_ratio: float | None,
    maximum_raw_weight: float | None,
    config: ReplayConfig,
) -> ReplayReadiness:
    if count < config.minimum_reviews:
        return ReplayReadiness.INSUFFICIENT_REVIEWS
    if (
        raw_ess_ratio is None
        or maximum_raw_weight is None
        or raw_ess_ratio < config.minimum_raw_ess_ratio
        or maximum_raw_weight > config.maximum_raw_weight
    ):
        return ReplayReadiness.UNSTABLE_SUPPORT
    return ReplayReadiness.REPORTABLE


def _summarize(
    rows: tuple[ReplayRow, ...],
    target: ReplayTarget,
    config: ReplayConfig,
) -> ReplaySummary:
    rewards: list[float] = []
    raw_weights: list[float] = []
    clipped_weights: list[float] = []
    target_by_row: list[dict[str, float]] = []
    template_ids: set[str] = set()

    for row in rows:
        probabilities = _target_probabilities(row, target)
        target_map = {
            arm.template_id: probability
            for arm, probability in zip(row.arms, probabilities, strict=True)
        }
        selected = row.selected_arm
        target_probability = target_map[selected.template_id]
        try:
            raw_weight = target_probability / selected.behavior_probability
        except OverflowError as error:
            raise ReplayInputError(
                "importance weight exceeds finite arithmetic"
            ) from error
        if not math.isfinite(raw_weight):
            raise ReplayInputError(
                "importance weight exceeds finite arithmetic"
            )
        rewards.append(row.reward)
        raw_weights.append(raw_weight)
        clipped_weights.append(min(raw_weight, config.clip_weight))
        target_by_row.append(target_map)
        template_ids.update(target_map)

    count = len(rows)
    if count == 0:
        return ReplaySummary(
            target=target,
            readiness=ReplayReadiness.INSUFFICIENT_REVIEWS,
            reviewed_count=0,
            observed_behavior_mean=None,
            raw_inverse_propensity=None,
            raw_self_normalized=None,
            clipped_inverse_propensity=None,
            clipped_self_normalized=None,
            raw_effective_sample_size=None,
            clipped_effective_sample_size=None,
            mean_raw_weight=None,
            maximum_raw_weight=None,
            minimum_selected_behavior_probability=None,
            clipped_row_count=0,
            removed_raw_weight_mass=0.0,
            raw_effective_sample_size_ratio=None,
            clipped_effective_sample_size_ratio=None,
            templates=(),
        )

    raw_tuple = tuple(raw_weights)
    clipped_tuple = tuple(clipped_weights)
    reward_tuple = tuple(rewards)
    raw_reward_terms = tuple(
        weight * reward
        for weight, reward in zip(raw_tuple, reward_tuple, strict=True)
    )
    clipped_reward_terms = tuple(
        weight * reward
        for weight, reward in zip(clipped_tuple, reward_tuple, strict=True)
    )
    reward_sum = _checked_sum(reward_tuple, "reward sum")
    raw_weight_sum = _checked_sum(raw_tuple, "raw weight sum")
    clipped_weight_sum = _checked_sum(clipped_tuple, "clipped weight sum")
    raw_reward_sum = _checked_sum(raw_reward_terms, "raw weighted reward sum")
    clipped_reward_sum = _checked_sum(
        clipped_reward_terms,
        "clipped weighted reward sum",
    )
    removed_mass = _checked_sum(
        tuple(
            raw_weight - clipped_weight
            for raw_weight, clipped_weight in zip(
                raw_tuple,
                clipped_tuple,
                strict=True,
            )
        ),
        "removed raw weight mass",
    )
    raw_ess = _effective_sample_size(raw_tuple)
    clipped_ess = _effective_sample_size(clipped_tuple)
    raw_ess_ratio = None if raw_ess is None else raw_ess / count
    clipped_ess_ratio = None if clipped_ess is None else clipped_ess / count
    maximum_raw_weight = max(raw_tuple)

    templates: list[TemplateReplaySummary] = []
    for template_id in sorted(template_ids):
        feasible = tuple(
            target_map[template_id]
            for target_map in target_by_row
            if template_id in target_map
        )
        selected_indices = tuple(
            index
            for index, row in enumerate(rows)
            if row.selected_template_id == template_id
        )
        selected_raw = tuple(raw_tuple[index] for index in selected_indices)
        selected_clipped = tuple(clipped_tuple[index] for index in selected_indices)
        selected_raw_reward = tuple(
            raw_reward_terms[index] for index in selected_indices
        )
        selected_clipped_reward = tuple(
            clipped_reward_terms[index] for index in selected_indices
        )
        selected_behavior = tuple(
            rows[index].selected_arm.behavior_probability
            for index in selected_indices
        )
        templates.append(
            TemplateReplaySummary(
                template_id=template_id,
                feasible_count=len(feasible),
                reviewed_count=len(selected_indices),
                target_mass=_checked_sum(feasible, "template target mass"),
                raw_weight_sum=_checked_sum(
                    selected_raw,
                    "template raw weight sum",
                ),
                clipped_weight_sum=_checked_sum(
                    selected_clipped,
                    "template clipped weight sum",
                ),
                raw_reward_contribution=(
                    _checked_sum(
                        selected_raw_reward,
                        "template raw reward sum",
                    )
                    / count
                ),
                clipped_reward_contribution=(
                    _checked_sum(
                        selected_clipped_reward,
                        "template clipped reward sum",
                    )
                    / count
                ),
                minimum_selected_behavior_probability=(
                    min(selected_behavior) if selected_behavior else None
                ),
                maximum_raw_weight=(
                    max(selected_raw) if selected_raw else None
                ),
            )
        )

    return ReplaySummary(
        target=target,
        readiness=_readiness(
            count=count,
            raw_ess_ratio=raw_ess_ratio,
            maximum_raw_weight=maximum_raw_weight,
            config=config,
        ),
        reviewed_count=count,
        observed_behavior_mean=reward_sum / count,
        raw_inverse_propensity=raw_reward_sum / count,
        raw_self_normalized=(
            raw_reward_sum / raw_weight_sum if raw_weight_sum > 0 else None
        ),
        clipped_inverse_propensity=clipped_reward_sum / count,
        clipped_self_normalized=(
            clipped_reward_sum / clipped_weight_sum
            if clipped_weight_sum > 0
            else None
        ),
        raw_effective_sample_size=raw_ess,
        clipped_effective_sample_size=clipped_ess,
        mean_raw_weight=raw_weight_sum / count,
        maximum_raw_weight=maximum_raw_weight,
        minimum_selected_behavior_probability=min(
            row.selected_arm.behavior_probability for row in rows
        ),
        clipped_row_count=sum(
            raw_weight > config.clip_weight for raw_weight in raw_tuple
        ),
        removed_raw_weight_mass=removed_mass,
        raw_effective_sample_size_ratio=raw_ess_ratio,
        clipped_effective_sample_size_ratio=clipped_ess_ratio,
        templates=tuple(templates),
    )


def _verify_behavior_control(summary: ReplaySummary) -> None:
    if summary.reviewed_count == 0:
        return
    expected = summary.observed_behavior_mean
    if (
        summary.mean_raw_weight != 1.0
        or summary.maximum_raw_weight != 1.0
        or summary.clipped_row_count != 0
        or summary.removed_raw_weight_mass != 0.0
        or summary.raw_effective_sample_size != summary.reviewed_count
        or summary.clipped_effective_sample_size != summary.reviewed_count
        or summary.raw_effective_sample_size_ratio != 1.0
        or summary.clipped_effective_sample_size_ratio != 1.0
        or summary.raw_inverse_propensity != expected
        or summary.raw_self_normalized != expected
        or summary.clipped_inverse_propensity != expected
        or summary.clipped_self_normalized != expected
    ):
        raise ReplayInvariantError("behavior replay negative control is not exact")


def build_replay_report(
    rows: Sequence[ReplayRow],
    *,
    target: ReplayTarget,
    config: ReplayConfig | None = None,
) -> ReplayReport:
    """Build aggregate one-step diagnostics over an ordered reviewed snapshot."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ReplayInputError("rows must be a sequence of ReplayRow instances")
    if len(rows) > MAX_REPLAY_ROWS:
        raise ReplayInputError(f"rows must contain at most {MAX_REPLAY_ROWS} items")
    snapshot: list[ReplayRow] = []
    previous_sequence: int | None = None
    for row in rows:
        if len(snapshot) >= MAX_REPLAY_ROWS:
            raise ReplayInputError(
                f"rows must contain at most {MAX_REPLAY_ROWS} items"
            )
        if not isinstance(row, ReplayRow):
            raise ReplayInputError("rows must contain ReplayRow instances")
        if (
            previous_sequence is not None
            and row.decision_sequence <= previous_sequence
        ):
            raise ReplayInputError(
                "decision_sequence values must be strictly increasing"
            )
        previous_sequence = row.decision_sequence
        snapshot.append(row)
    if not isinstance(target, ReplayTarget):
        raise ReplayInputError("target must be a ReplayTarget")
    if config is None:
        replay_config = ReplayConfig()
    elif isinstance(config, ReplayConfig):
        replay_config = config
    else:
        raise ReplayInputError("config must be a ReplayConfig")

    ordered_rows = tuple(snapshot)
    candidate = _summarize(ordered_rows, target, replay_config)
    behavior_control = _summarize(
        ordered_rows,
        ReplayTarget.behavior(),
        replay_config,
    )
    _verify_behavior_control(behavior_control)
    return ReplayReport(
        candidate=candidate,
        behavior_control=behavior_control,
        config=replay_config,
        nonclaims=ReplayNonClaims(),
    )
