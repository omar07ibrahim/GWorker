"""Explainable contextual-bandit policy for bounded focus templates."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

POLICY_FAMILY = "hierarchical-softmax-ucb-v1"
MAX_AVAILABLE_SECONDS = 24 * 60 * 60
MAX_DECISION_SEQUENCE = 2**63 - 1
MAX_TEMPLATES = 32
FIT_REWARD_WEIGHT = 0.8
COMPLETION_REWARD_WEIGHT = 0.2


class PolicyInputError(ValueError):
    """Raised when policy context, history, or configuration is invalid."""


class NoFeasibleTemplate(PolicyInputError):
    """Raised when the available time cannot fit any complete template."""


class TaskKind(StrEnum):
    """Explicit, coarse task categories; no task content is inspected."""

    DEEP_WORK = "deep_work"
    ADMIN = "admin"
    LEARNING = "learning"
    CREATIVE = "creative"


class EnergyLevel(StrEnum):
    """Self-reported context retained only after an explicit review."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DurationFit(StrEnum):
    """Direction of explicit post-session duration feedback."""

    TOO_SHORT = "too_short"
    JUST_RIGHT = "just_right"
    TOO_LONG = "too_long"


class EvidenceBucket(StrEnum):
    """Hierarchy level used for a recommendation."""

    EXACT = "exact"
    TASK = "task"
    GLOBAL = "global"


def _positive_integer(value: object, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyInputError(f"{field} must be an integer")
    if not 1 <= value <= maximum:
        raise PolicyInputError(f"{field} must be between 1 and {maximum}")
    return value


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyInputError(f"{field} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise PolicyInputError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise PolicyInputError(f"{field} must be finite")
    if result == 0:
        return 0.0
    return result


def _safe_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 80:
        raise PolicyInputError(f"{field} must contain 1 to 80 characters")
    if not all(
        character.isascii()
        and (character.islower() or character.isdigit() or character in ".-_")
        for character in value
    ):
        raise PolicyInputError(
            f"{field} may contain lowercase ASCII letters, digits, dot, dash, "
            "and underscore"
        )
    return value


def _non_nil_uuid(value: object, field: str) -> UUID:
    if not isinstance(value, UUID):
        raise PolicyInputError(f"{field} must be a UUID")
    if value.int == 0:
        raise PolicyInputError(f"{field} must not be the nil UUID")
    return value


@dataclass(frozen=True, slots=True)
class FocusTemplate:
    """One policy arm with a complete focus-and-break budget."""

    template_id: str
    focus_seconds: int
    break_seconds: int

    def __post_init__(self) -> None:
        _safe_identifier(self.template_id, "template_id")
        _positive_integer(
            self.focus_seconds,
            "focus_seconds",
            maximum=MAX_AVAILABLE_SECONDS,
        )
        _positive_integer(
            self.break_seconds,
            "break_seconds",
            maximum=MAX_AVAILABLE_SECONDS,
        )
        if self.focus_seconds + self.break_seconds > MAX_AVAILABLE_SECONDS:
            raise PolicyInputError("template duration must not exceed 24 hours")

    @property
    def total_seconds(self) -> int:
        """Return the full time commitment used by availability guardrails."""

        return self.focus_seconds + self.break_seconds


DEFAULT_TEMPLATES = (
    FocusTemplate("focus-15", 15 * 60, 3 * 60),
    FocusTemplate("focus-25", 25 * 60, 5 * 60),
    FocusTemplate("focus-40", 40 * 60, 8 * 60),
    FocusTemplate("focus-50", 50 * 60, 10 * 60),
)


@dataclass(frozen=True, slots=True)
class FocusContext:
    """Minimal explicit context supplied for one recommendation."""

    task_kind: TaskKind
    energy: EnergyLevel
    available_seconds: int
    previous_focus_seconds: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_kind, TaskKind):
            raise PolicyInputError("task_kind must be a TaskKind")
        if not isinstance(self.energy, EnergyLevel):
            raise PolicyInputError("energy must be an EnergyLevel")
        _positive_integer(
            self.available_seconds,
            "available_seconds",
            maximum=MAX_AVAILABLE_SECONDS,
        )
        if self.previous_focus_seconds is not None:
            _positive_integer(
                self.previous_focus_seconds,
                "previous_focus_seconds",
                maximum=MAX_AVAILABLE_SECONDS,
            )


@dataclass(frozen=True, slots=True)
class ReviewedDecision:
    """A structurally valid decision with explicit feedback.

    Direct construction does not prove that ``propensity`` came from a matching
    recommendation. Durable provenance belongs to the future journal
    integration; callers should normally use :meth:`Recommendation.review`.
    """

    decision_id: UUID
    decision_sequence: int
    policy_id: str
    context: FocusContext
    template_id: str
    propensity: float
    fit: DurationFit
    objective_completed: bool

    def __post_init__(self) -> None:
        _non_nil_uuid(self.decision_id, "decision_id")
        _positive_integer(
            self.decision_sequence,
            "decision_sequence",
            maximum=MAX_DECISION_SEQUENCE,
        )
        _safe_identifier(self.policy_id, "policy_id")
        if not isinstance(self.context, FocusContext):
            raise PolicyInputError("context must be a FocusContext")
        _safe_identifier(self.template_id, "template_id")
        propensity = _finite_float(self.propensity, "propensity")
        if not 0 < propensity <= 1:
            raise PolicyInputError("propensity must be in (0, 1]")
        object.__setattr__(self, "propensity", propensity)
        if not isinstance(self.fit, DurationFit):
            raise PolicyInputError("fit must be a DurationFit")
        if not isinstance(self.objective_completed, bool):
            raise PolicyInputError("objective_completed must be boolean")

    @property
    def reward(self) -> float:
        """Return a bounded fit-first reward in the interval [0, 1]."""

        fit_reward = 1.0 if self.fit is DurationFit.JUST_RIGHT else 0.0
        completion_reward = 1.0 if self.objective_completed else 0.0
        return (
            FIT_REWARD_WEIGHT * fit_reward
            + COMPLETION_REWARD_WEIGHT * completion_reward
        )


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Controls bounded memory, confidence bonuses, and exploration."""

    window_size: int = 48
    minimum_exact_reviews: int = 6
    minimum_task_reviews: int = 12
    prior_mean: float = 0.55
    prior_weight: float = 2.0
    direction_weight: float = 0.15
    exploration_weight: float = 0.35
    temperature: float = 0.20
    minimum_probability: float = 0.02

    def __post_init__(self) -> None:
        _positive_integer(self.window_size, "window_size", maximum=10_000)
        _positive_integer(
            self.minimum_exact_reviews,
            "minimum_exact_reviews",
            maximum=self.window_size,
        )
        _positive_integer(
            self.minimum_task_reviews,
            "minimum_task_reviews",
            maximum=self.window_size,
        )
        if self.minimum_exact_reviews > self.minimum_task_reviews:
            raise PolicyInputError(
                "minimum_exact_reviews must not exceed minimum_task_reviews"
            )
        prior_mean = _finite_float(self.prior_mean, "prior_mean")
        if not 0 <= prior_mean <= 1:
            raise PolicyInputError("prior_mean must be in [0, 1]")
        object.__setattr__(self, "prior_mean", prior_mean)
        prior_weight = _finite_float(self.prior_weight, "prior_weight")
        if not 0.01 <= prior_weight <= 10_000:
            raise PolicyInputError("prior_weight must be in [0.01, 10000]")
        object.__setattr__(self, "prior_weight", prior_weight)
        direction = _finite_float(self.direction_weight, "direction_weight")
        if not 0 <= direction <= 0.5:
            raise PolicyInputError("direction_weight must be in [0, 0.5]")
        object.__setattr__(self, "direction_weight", direction)
        exploration = _finite_float(
            self.exploration_weight,
            "exploration_weight",
        )
        if not 0 <= exploration <= 5:
            raise PolicyInputError("exploration_weight must be in [0, 5]")
        object.__setattr__(self, "exploration_weight", exploration)
        temperature = _finite_float(self.temperature, "temperature")
        if not 0.01 <= temperature <= 5:
            raise PolicyInputError("temperature must be in [0.01, 5]")
        object.__setattr__(self, "temperature", temperature)
        minimum_probability = _finite_float(
            self.minimum_probability,
            "minimum_probability",
        )
        if not 0.000_001 <= minimum_probability <= 0.25:
            raise PolicyInputError("minimum_probability must be in [0.000001, 0.25]")
        object.__setattr__(
            self,
            "minimum_probability",
            minimum_probability,
        )


@dataclass(frozen=True, slots=True)
class ArmScore:
    """Reader-facing decomposition for one feasible policy arm."""

    template: FocusTemplate
    review_count: int
    posterior_mean: float
    directional_adjustment: float
    exploration_bonus: float
    score: float
    probability: float


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Selected arm plus the exact evidence and logged propensity."""

    decision_id: UUID
    decision_sequence: int
    policy_id: str
    context: FocusContext
    template: FocusTemplate
    propensity: float
    bucket: EvidenceBucket
    bucket_label: str
    evidence_count: int
    arm_scores: tuple[ArmScore, ...]
    reason_codes: tuple[str, ...]

    def review(
        self,
        *,
        fit: DurationFit,
        objective_completed: bool,
    ) -> ReviewedDecision:
        """Attach explicit feedback to this exact logged decision."""

        return ReviewedDecision(
            decision_id=self.decision_id,
            decision_sequence=self.decision_sequence,
            policy_id=self.policy_id,
            context=self.context,
            template_id=self.template.template_id,
            propensity=self.propensity,
            fit=fit,
            objective_completed=objective_completed,
        )


@dataclass(frozen=True, slots=True)
class _UnnormalizedArmScore:
    template: FocusTemplate
    review_count: int
    posterior_mean: float
    directional_adjustment: float
    exploration_bonus: float
    score: float


def _policy_identifier(
    config: PolicyConfig,
    templates: tuple[FocusTemplate, ...],
) -> str:
    payload = {
        "algorithm": POLICY_FAMILY,
        "config": {
            "direction_weight": config.direction_weight,
            "exploration_weight": config.exploration_weight,
            "minimum_exact_reviews": config.minimum_exact_reviews,
            "minimum_probability": config.minimum_probability,
            "minimum_task_reviews": config.minimum_task_reviews,
            "prior_mean": config.prior_mean,
            "prior_weight": config.prior_weight,
            "temperature": config.temperature,
            "window_size": config.window_size,
        },
        "templates": [
            {
                "break_seconds": template.break_seconds,
                "focus_seconds": template.focus_seconds,
                "template_id": template.template_id,
            }
            for template in templates
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    fingerprint = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{POLICY_FAMILY}.{fingerprint}"


POLICY_ID = _policy_identifier(PolicyConfig(), DEFAULT_TEMPLATES)


class HierarchicalSoftmaxUCB:
    """Sliding-window contextual UCB with softmax exploration.

    Exact task-and-energy evidence is preferred once sufficiently populated,
    followed by task-only evidence and finally the global window. Availability
    and one-step movement constraints are applied before scoring.
    """

    def __init__(
        self,
        *,
        config: PolicyConfig | None = None,
        templates: Sequence[FocusTemplate] = DEFAULT_TEMPLATES,
    ) -> None:
        if config is not None and not isinstance(config, PolicyConfig):
            raise PolicyInputError("config must be a PolicyConfig")
        if isinstance(templates, (str, bytes)) or not isinstance(
            templates,
            Sequence,
        ):
            raise PolicyInputError(
                "templates must be a sequence of FocusTemplate instances"
            )
        self.config = config if config is not None else PolicyConfig()
        self.templates = tuple(templates)
        if not 2 <= len(self.templates) <= MAX_TEMPLATES:
            raise PolicyInputError(
                f"between 2 and {MAX_TEMPLATES} focus templates are required"
            )
        if not all(isinstance(template, FocusTemplate) for template in self.templates):
            raise PolicyInputError("templates must contain FocusTemplate instances")
        if self.config.minimum_probability * len(self.templates) >= 1:
            raise PolicyInputError(
                "minimum_probability multiplied by template count must be below 1"
            )
        identifiers = [template.template_id for template in self.templates]
        durations = [template.focus_seconds for template in self.templates]
        if len(set(identifiers)) != len(identifiers):
            raise PolicyInputError("template_id values must be unique")
        if len(set(durations)) != len(durations):
            raise PolicyInputError("focus durations must be unique")
        if durations != sorted(durations):
            raise PolicyInputError("templates must be ordered by focus duration")
        self._template_by_id = {
            template.template_id: template for template in self.templates
        }
        self._index_by_duration = {
            template.focus_seconds: index
            for index, template in enumerate(self.templates)
        }
        self.policy_id = _policy_identifier(self.config, self.templates)

    def _validate_history(
        self,
        reviews: Sequence[ReviewedDecision],
    ) -> tuple[ReviewedDecision, ...]:
        if isinstance(reviews, (str, bytes)) or not isinstance(reviews, Sequence):
            raise PolicyInputError(
                "history must be a sequence of ReviewedDecision instances"
            )
        validated: list[ReviewedDecision] = []
        decision_ids: set[UUID] = set()
        first_index = max(0, len(reviews) - self.config.window_size)
        previous_sequence: int | None = None
        for index in range(first_index, len(reviews)):
            review = reviews[index]
            if not isinstance(review, ReviewedDecision):
                raise PolicyInputError(
                    "history must contain ReviewedDecision instances"
                )
            if (
                previous_sequence is not None
                and review.decision_sequence <= previous_sequence
            ):
                raise PolicyInputError(
                    "history decision_sequence values must be strictly increasing"
                )
            previous_sequence = review.decision_sequence
            if review.decision_id in decision_ids:
                raise PolicyInputError(
                    f"history repeats decision_id: {review.decision_id}"
                )
            decision_ids.add(review.decision_id)
            if review.policy_id != self.policy_id:
                raise PolicyInputError(
                    "history policy_id does not match this policy configuration"
                )
            if review.template_id not in self._template_by_id:
                raise PolicyInputError(
                    f"history references unknown template_id: {review.template_id}"
                )
            template = self._template_by_id[review.template_id]
            feasible, _ = self.feasible_templates(review.context)
            if template not in feasible:
                raise PolicyInputError(
                    "history contains a template that violated policy guardrails"
                )
            validated.append(review)
        return tuple(validated)

    def _select_bucket(
        self,
        context: FocusContext,
        reviews: tuple[ReviewedDecision, ...],
    ) -> tuple[EvidenceBucket, str, tuple[ReviewedDecision, ...]]:
        exact = tuple(
            review
            for review in reviews
            if review.context.task_kind is context.task_kind
            and review.context.energy is context.energy
        )
        if len(exact) >= self.config.minimum_exact_reviews:
            label = f"exact:{context.task_kind.value}:{context.energy.value}"
            return EvidenceBucket.EXACT, label, exact

        task = tuple(
            review
            for review in reviews
            if review.context.task_kind is context.task_kind
        )
        if len(task) >= self.config.minimum_task_reviews:
            label = f"task:{context.task_kind.value}"
            return EvidenceBucket.TASK, label, task

        return EvidenceBucket.GLOBAL, "global", reviews

    def feasible_templates(
        self,
        context: FocusContext,
    ) -> tuple[tuple[FocusTemplate, ...], tuple[str, ...]]:
        """Return arms allowed by availability and one-step guardrails."""

        if not isinstance(context, FocusContext):
            raise PolicyInputError("context must be a FocusContext")
        available = tuple(
            template
            for template in self.templates
            if template.total_seconds <= context.available_seconds
        )
        if not available:
            minimum = min(template.total_seconds for template in self.templates)
            raise NoFeasibleTemplate(
                f"available_seconds cannot fit the shortest template ({minimum})"
            )

        reasons: list[str] = []
        if len(available) != len(self.templates):
            reasons.append("availability_guardrail")

        if context.previous_focus_seconds is None:
            return available, tuple(reasons)
        previous_index = self._index_by_duration.get(context.previous_focus_seconds)
        if previous_index is None:
            raise PolicyInputError(
                "previous_focus_seconds must match a configured template"
            )
        stepped = tuple(
            template
            for template in available
            if abs(self._index_by_duration[template.focus_seconds] - previous_index)
            <= 1
        )
        if stepped:
            reasons.append("one_step_guardrail")
            return stepped, tuple(reasons)
        reasons.append("availability_overrode_step")
        return available, tuple(reasons)

    def _score_arms(
        self,
        templates: tuple[FocusTemplate, ...],
        reviews: tuple[ReviewedDecision, ...],
    ) -> tuple[_UnnormalizedArmScore, ...]:
        total = len(reviews)
        log_term = math.log(total + self.config.prior_weight + 1.0)
        scores: list[_UnnormalizedArmScore] = []
        for template in templates:
            arm_reviews = tuple(
                review
                for review in reviews
                if review.template_id == template.template_id
            )
            reward_sum = math.fsum(review.reward for review in arm_reviews)
            count = len(arm_reviews)
            denominator = self.config.prior_weight + count
            posterior_mean = (
                self.config.prior_mean * self.config.prior_weight + reward_sum
            ) / denominator
            direction_sum = math.fsum(
                self._direction_signal(template, review) for review in reviews
            )
            directional_adjustment = (
                self.config.direction_weight
                * direction_sum
                / (total + self.config.prior_weight)
            )
            exploration_bonus = self.config.exploration_weight * math.sqrt(
                log_term / denominator
            )
            scores.append(
                _UnnormalizedArmScore(
                    template=template,
                    review_count=count,
                    posterior_mean=posterior_mean,
                    directional_adjustment=directional_adjustment,
                    exploration_bonus=exploration_bonus,
                    score=(posterior_mean + directional_adjustment + exploration_bonus),
                )
            )
        return tuple(scores)

    def _direction_signal(
        self,
        candidate: FocusTemplate,
        review: ReviewedDecision,
    ) -> float:
        reviewed_template = self._template_by_id[review.template_id]
        if candidate.focus_seconds == reviewed_template.focus_seconds:
            return 0.0
        candidate_is_longer = candidate.focus_seconds > reviewed_template.focus_seconds
        if review.fit is DurationFit.TOO_SHORT:
            return 1.0 if candidate_is_longer else -1.0
        if review.fit is DurationFit.TOO_LONG:
            return -1.0 if candidate_is_longer else 1.0
        return 0.0

    def _normalize(
        self,
        raw_scores: tuple[_UnnormalizedArmScore, ...],
    ) -> tuple[ArmScore, ...]:
        maximum = max(item.score for item in raw_scores)
        weights = tuple(
            math.exp((item.score - maximum) / self.config.temperature)
            for item in raw_scores
        )
        denominator = math.fsum(weights)
        floor = self.config.minimum_probability
        remaining_mass = 1.0 - floor * len(raw_scores)
        return tuple(
            ArmScore(
                template=item.template,
                review_count=item.review_count,
                posterior_mean=item.posterior_mean,
                directional_adjustment=item.directional_adjustment,
                exploration_bonus=item.exploration_bonus,
                score=item.score,
                probability=floor + remaining_mass * weight / denominator,
            )
            for item, weight in zip(raw_scores, weights, strict=True)
        )

    @staticmethod
    def _sample(
        arms: tuple[ArmScore, ...],
        rng: random.Random,
    ) -> ArmScore:
        draw = _finite_float(rng.random(), "rng.random()")
        if not 0 <= draw < 1:
            raise PolicyInputError("rng.random() must return a value in [0, 1)")
        cumulative = 0.0
        for arm in arms:
            cumulative += arm.probability
            if draw < cumulative:
                return arm
        return arms[-1]

    def recommend(
        self,
        context: FocusContext,
        history: Sequence[ReviewedDecision],
        *,
        decision_id: UUID,
        decision_sequence: int,
        rng: random.Random,
    ) -> Recommendation:
        """Choose one feasible arm and expose its exact action probability."""

        if not isinstance(context, FocusContext):
            raise PolicyInputError("context must be a FocusContext")
        if not isinstance(rng, random.Random):
            raise PolicyInputError("rng must be an instance of random.Random")
        _non_nil_uuid(decision_id, "decision_id")
        _positive_integer(
            decision_sequence,
            "decision_sequence",
            maximum=MAX_DECISION_SEQUENCE,
        )
        window = self._validate_history(history)
        if any(review.decision_id == decision_id for review in window):
            raise PolicyInputError("decision_id is already present in history")
        if window and decision_sequence <= window[-1].decision_sequence:
            raise PolicyInputError(
                "decision_sequence must be newer than the history tail"
            )
        bucket, bucket_label, evidence = self._select_bucket(context, window)
        feasible, guardrail_reasons = self.feasible_templates(context)
        arm_scores = self._normalize(self._score_arms(feasible, evidence))
        selected = self._sample(arm_scores, rng)

        reasons = [f"{bucket.value}_evidence", "bounded_exploration"]
        if not evidence:
            reasons.append("cold_start")
        reasons.extend(guardrail_reasons)
        return Recommendation(
            decision_id=decision_id,
            decision_sequence=decision_sequence,
            policy_id=self.policy_id,
            context=context,
            template=selected.template,
            propensity=selected.probability,
            bucket=bucket,
            bucket_label=bucket_label,
            evidence_count=len(evidence),
            arm_scores=arm_scores,
            reason_codes=tuple(reasons),
        )
