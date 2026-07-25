"""Seed-level statistics and source provenance for synthetic evaluation reports.

The module is intentionally independent from presentation. It turns a validated
``ExperimentResult`` into frozen scalar estimates and proves which clean Git
tree supplied the running evaluator. The locked runner and SVG renderer build
on these primitives in a later milestone.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import stat
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .evaluation import (
    AvailabilityMode,
    ClusterSummary,
    ExperimentResult,
    Strategy,
    validate_experiment_result,
)

REPORT_SCHEMA_VERSION = "gworker-statistical-report-v1"
BOOTSTRAP_VERSION = "paired-seed-bootstrap-v1"
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_CONFIDENCE = 0.95
MAX_BOOTSTRAP_DRAW_COUNT = 5_000_000
LOCKED_AUTHOR_NAME = "Omar Ibrahim"
LOCKED_AUTHOR_EMAIL = "31526072+omar07ibrahim@users.noreply.github.com"


class ReportingInputError(ValueError):
    """Raised when a report definition or source checkout is invalid."""


class ReportingInvariantError(RuntimeError):
    """Raised when validated evaluation evidence fails report invariants."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportingInputError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReportingInputError(f"{field} must be finite")
    return 0.0 if result == 0 else result


def _safe_identifier(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 160
        or len(value.splitlines()) != 1
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ReportingInputError(f"{field} must be bounded single-line text")
    return value


def _sha256_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReportingInputError(f"{field} must be a SHA-256 hex digest")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ReportingInvariantError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def type7_quantile(values: Sequence[float], probability: float) -> float:
    """Return the Hyndman-Fan type-7 sample quantile used by the protocol."""

    if not values:
        raise ReportingInputError("quantile values must not be empty")
    validated = tuple(_finite(value, "quantile value") for value in values)
    quantile = _finite(probability, "probability")
    if not 0.0 <= quantile <= 1.0:
        raise ReportingInputError("probability must be in [0, 1]")
    ordered = sorted(validated)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    fraction = position - lower_index
    if fraction == 0.0:
        return ordered[lower_index]
    return ordered[lower_index] + fraction * (
        ordered[lower_index + 1] - ordered[lower_index]
    )


def seed_cluster_standard_error(values: Sequence[float]) -> float | None:
    """Return the standard error of a paired seed-level mean."""

    validated = tuple(_finite(value, "seed-level value") for value in values)
    if not validated:
        raise ReportingInputError("seed-level values must not be empty")
    if len(validated) == 1:
        return None
    center = _mean(validated)
    variance = math.fsum((value - center) ** 2 for value in validated) / (
        len(validated) - 1
    )
    return math.sqrt(variance / len(validated))


def _mode_indices_digest(
    mode: AvailabilityMode,
    indices: tuple[tuple[int, ...], ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(BOOTSTRAP_VERSION.encode("ascii"))
    digest.update(b"\x00mode-indices\x00")
    mode_bytes = mode.value.encode("ascii")
    digest.update(len(mode_bytes).to_bytes(2, "big"))
    digest.update(mode_bytes)
    digest.update(len(indices).to_bytes(8, "big"))
    digest.update(len(indices[0]).to_bytes(4, "big"))
    for row in indices:
        for index in row:
            digest.update(index.to_bytes(4, "big"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModeBootstrapPlan:
    """One mode-specific matrix reused across its personas and strategies."""

    availability_mode: AvailabilityMode
    indices_sha256: str
    indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.availability_mode) is not AvailabilityMode:
            raise ReportingInputError("availability_mode must be an AvailabilityMode")
        try:
            indices = tuple(tuple(row) for row in self.indices)
        except TypeError as error:
            raise ReportingInputError(
                "bootstrap indices must be a finite matrix"
            ) from error
        object.__setattr__(self, "indices", indices)
        if (
            not indices
            or not indices[0]
            or any(len(row) != len(indices[0]) for row in indices)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index <= 0xFFFF_FFFF
                for row in indices
                for index in row
            )
        ):
            raise ReportingInputError(
                "bootstrap indices must be a non-empty rectangular integer matrix"
            )
        digest = _sha256_digest(self.indices_sha256, "indices_sha256")
        if digest != _mode_indices_digest(self.availability_mode, indices):
            raise ReportingInputError("mode bootstrap index digest is inconsistent")


def _combined_indices_digest(
    *,
    width: int,
    resample_count: int,
    mode_plans: tuple[ModeBootstrapPlan, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(BOOTSTRAP_VERSION.encode("ascii"))
    digest.update(b"\x00all-indices\x00")
    digest.update(width.to_bytes(4, "big"))
    digest.update(resample_count.to_bytes(8, "big"))
    digest.update(len(mode_plans).to_bytes(2, "big"))
    for plan in mode_plans:
        mode_bytes = plan.availability_mode.value.encode("ascii")
        digest.update(len(mode_bytes).to_bytes(2, "big"))
        digest.update(mode_bytes)
        digest.update(bytes.fromhex(plan.indices_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    """Independent mode strata with shared within-mode resample indices."""

    version: str
    evaluator_id: str
    environment_seeds: tuple[int, ...]
    availability_modes: tuple[AvailabilityMode, ...]
    resample_count: int
    namespace_sha256: str
    indices_sha256: str
    mode_plans: tuple[ModeBootstrapPlan, ...]

    def __post_init__(self) -> None:
        try:
            seeds = tuple(self.environment_seeds)
            modes = tuple(self.availability_modes)
            mode_plans = tuple(self.mode_plans)
        except TypeError as error:
            raise ReportingInputError(
                "bootstrap plan collections must be finite"
            ) from error
        object.__setattr__(self, "environment_seeds", seeds)
        object.__setattr__(self, "availability_modes", modes)
        object.__setattr__(self, "mode_plans", mode_plans)
        if self.version != BOOTSTRAP_VERSION:
            raise ReportingInputError("bootstrap version is not supported")
        identifier = _safe_identifier(self.evaluator_id, "evaluator_id")
        if (
            isinstance(self.resample_count, bool)
            or not isinstance(self.resample_count, int)
            or not 1 <= self.resample_count <= 100_000
        ):
            raise ReportingInputError(
                "resample_count must be an integer between 1 and 100000"
            )
        width = len(self.environment_seeds)
        if (
            width < 1
            or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in self.environment_seeds
            )
            or len(set(self.environment_seeds)) != width
        ):
            raise ReportingInputError(
                "bootstrap population must contain unique integer seeds"
            )
        if (
            not self.availability_modes
            or len(set(self.availability_modes)) != len(self.availability_modes)
            or any(
                type(mode) is not AvailabilityMode for mode in self.availability_modes
            )
        ):
            raise ReportingInputError("availability mode strata are invalid")
        if any(not isinstance(plan, ModeBootstrapPlan) for plan in self.mode_plans):
            raise ReportingInputError("mode plans contain an invalid member")
        if tuple(plan.availability_mode for plan in self.mode_plans) != (
            self.availability_modes
        ):
            raise ReportingInputError("mode plans do not match declared strata")
        if (
            width * self.resample_count * len(self.availability_modes)
            > MAX_BOOTSTRAP_DRAW_COUNT
        ):
            raise ReportingInputError(
                "bootstrap plan exceeds the safe draw-count limit"
            )
        for plan in self.mode_plans:
            if len(plan.indices) != self.resample_count:
                raise ReportingInputError("bootstrap row count is inconsistent")
            if any(len(row) != width for row in plan.indices):
                raise ReportingInputError("bootstrap rows have an invalid width")
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < width
                for row in plan.indices
                for index in row
            ):
                raise ReportingInputError("bootstrap index is out of range")
        namespace_digest = _sha256_digest(
            self.namespace_sha256,
            "namespace_sha256",
        )
        if namespace_digest != _bootstrap_namespace_digest(
            self.environment_seeds,
            self.availability_modes,
            evaluator_id=identifier,
            resample_count=self.resample_count,
        ):
            raise ReportingInputError("bootstrap namespace digest is inconsistent")
        digest = _sha256_digest(self.indices_sha256, "indices_sha256")
        expected_digest = _combined_indices_digest(
            width=width,
            resample_count=self.resample_count,
            mode_plans=self.mode_plans,
        )
        if digest != expected_digest:
            raise ReportingInputError("combined bootstrap index digest is inconsistent")

    def for_mode(self, mode: AvailabilityMode) -> ModeBootstrapPlan:
        """Return the precomputed matrix for one declared mode."""

        for plan in self.mode_plans:
            if plan.availability_mode is mode:
                return plan
        raise ReportingInvariantError(f"bootstrap plan is missing mode {mode.value}")


def _bootstrap_namespace_digest(
    seeds: tuple[int, ...],
    modes: tuple[AvailabilityMode, ...],
    *,
    evaluator_id: str,
    resample_count: int,
) -> str:
    document = "\x1f".join(
        (
            BOOTSTRAP_VERSION,
            evaluator_id,
            ",".join(str(seed) for seed in seeds),
            ",".join(mode.value for mode in modes),
            str(resample_count),
        )
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def build_bootstrap_plan(
    environment_seeds: Sequence[int],
    *,
    availability_modes: Sequence[AvailabilityMode],
    evaluator_id: str,
    resample_count: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> BootstrapPlan:
    """Build independent mode matrices shared within every mode stratum."""

    seeds = tuple(environment_seeds)
    if (
        not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ReportingInputError("environment_seeds must be unique integer values")
    if (
        isinstance(resample_count, bool)
        or not isinstance(resample_count, int)
        or not 1 <= resample_count <= 100_000
    ):
        raise ReportingInputError(
            "resample_count must be an integer between 1 and 100000"
        )
    identifier = _safe_identifier(evaluator_id, "evaluator_id")
    modes = tuple(availability_modes)
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(type(mode) is not AvailabilityMode for mode in modes)
    ):
        raise ReportingInputError(
            "availability_modes must contain unique AvailabilityMode values"
        )
    width = len(seeds)
    if width * resample_count * len(modes) > MAX_BOOTSTRAP_DRAW_COUNT:
        raise ReportingInputError("bootstrap plan exceeds the safe draw-count limit")
    namespace_sha256 = _bootstrap_namespace_digest(
        seeds,
        modes,
        evaluator_id=identifier,
        resample_count=resample_count,
    )

    def draw_index(
        mode: AvailabilityMode,
        bootstrap_index: int,
        draw_position: int,
    ) -> int:
        rejection_limit = 2**64 - (2**64 % width)
        attempt = 0
        while True:
            document = "\x1f".join(
                (
                    BOOTSTRAP_VERSION,
                    identifier,
                    mode.value,
                    str(bootstrap_index),
                    str(draw_position),
                    str(attempt),
                )
            ).encode("utf-8")
            integer = int.from_bytes(
                hashlib.sha256(document).digest()[:8],
                "big",
            )
            if integer < rejection_limit:
                return integer % width
            attempt += 1

    mode_plans: list[ModeBootstrapPlan] = []
    for mode in modes:
        rows = tuple(
            tuple(
                draw_index(mode, bootstrap_index, draw_position)
                for draw_position in range(width)
            )
            for bootstrap_index in range(resample_count)
        )
        mode_plans.append(
            ModeBootstrapPlan(
                availability_mode=mode,
                indices_sha256=_mode_indices_digest(mode, rows),
                indices=rows,
            )
        )
    frozen_mode_plans = tuple(mode_plans)
    return BootstrapPlan(
        version=BOOTSTRAP_VERSION,
        evaluator_id=identifier,
        environment_seeds=seeds,
        availability_modes=modes,
        resample_count=resample_count,
        namespace_sha256=namespace_sha256,
        indices_sha256=_combined_indices_digest(
            width=width,
            resample_count=resample_count,
            mode_plans=frozen_mode_plans,
        ),
        mode_plans=frozen_mode_plans,
    )


@dataclass(frozen=True, slots=True)
class IntervalEstimate:
    """Point estimate and predeclared uncertainty at seed grain."""

    point_estimate: float
    lower_95: float
    upper_95: float
    seed_cluster_standard_error: float | None
    seed_count_per_stratum: int
    stratum_count: int

    def __post_init__(self) -> None:
        point = _finite(self.point_estimate, "point_estimate")
        lower = _finite(self.lower_95, "lower_95")
        upper = _finite(self.upper_95, "upper_95")
        if lower > upper:
            raise ReportingInputError("interval bounds are reversed")
        standard_error = self.seed_cluster_standard_error
        if standard_error is not None:
            standard_error = _finite(
                standard_error,
                "seed_cluster_standard_error",
            )
            if standard_error < 0:
                raise ReportingInputError(
                    "seed_cluster_standard_error must not be negative"
                )
            object.__setattr__(
                self,
                "seed_cluster_standard_error",
                standard_error,
            )
        if (
            isinstance(self.seed_count_per_stratum, bool)
            or not isinstance(self.seed_count_per_stratum, int)
            or self.seed_count_per_stratum < 1
        ):
            raise ReportingInputError(
                "seed_count_per_stratum must be a positive integer"
            )
        if (
            isinstance(self.stratum_count, bool)
            or not isinstance(self.stratum_count, int)
            or self.stratum_count < 1
        ):
            raise ReportingInputError("stratum_count must be a positive integer")
        object.__setattr__(self, "point_estimate", point)
        object.__setattr__(self, "lower_95", lower)
        object.__setattr__(self, "upper_95", upper)


def stratified_seed_standard_error(
    values_by_mode: dict[AvailabilityMode, tuple[float, ...]],
) -> float | None:
    if not values_by_mode:
        raise ReportingInvariantError("uncertainty requires at least one stratum")
    if any(type(mode) is not AvailabilityMode for mode in values_by_mode):
        raise ReportingInputError("uncertainty strata must use AvailabilityMode keys")
    validated = {
        mode: tuple(_finite(value, "seed-level value") for value in values)
        for mode, values in values_by_mode.items()
    }
    widths = {len(values) for values in validated.values()}
    if len(widths) != 1:
        raise ReportingInvariantError("mode strata have unequal seed counts")
    width = next(iter(widths))
    if width < 1:
        raise ReportingInputError("uncertainty strata must not be empty")
    if width == 1:
        return None
    variance_terms: list[float] = []
    for values in validated.values():
        center = _mean(values)
        sample_variance = math.fsum((value - center) ** 2 for value in values) / (
            width - 1
        )
        variance_terms.append(sample_variance / width)
    mode_count = len(validated)
    return math.sqrt(math.fsum(variance_terms) / mode_count**2)


def stratified_bootstrap_interval(
    values_by_mode: dict[AvailabilityMode, tuple[float, ...]],
    plan: BootstrapPlan,
) -> IntervalEstimate:
    modes = tuple(values_by_mode)
    expected_modes = tuple(
        mode for mode in plan.availability_modes if mode in values_by_mode
    )
    if not modes or modes != expected_modes:
        raise ReportingInvariantError(
            "uncertainty strata do not align with the bootstrap plan"
        )
    width = len(plan.environment_seeds)
    validated: dict[AvailabilityMode, tuple[float, ...]] = {}
    for mode, values in values_by_mode.items():
        validated_values = tuple(_finite(value, "seed value") for value in values)
        if len(validated_values) != width:
            raise ReportingInvariantError(
                "seed values do not align with the bootstrap population"
            )
        validated[mode] = validated_values
    estimates = tuple(
        _mean(
            tuple(
                math.fsum(
                    validated[mode][index]
                    for index in plan.for_mode(mode).indices[bootstrap_index]
                )
                / width
                for mode in modes
            )
        )
        for bootstrap_index in range(plan.resample_count)
    )
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    point = _mean(tuple(_mean(validated[mode]) for mode in modes))
    lower = type7_quantile(estimates, tail)
    upper = type7_quantile(estimates, 1.0 - tail)
    return IntervalEstimate(
        point_estimate=point,
        lower_95=lower,
        upper_95=upper,
        seed_cluster_standard_error=stratified_seed_standard_error(validated),
        seed_count_per_stratum=width,
        stratum_count=len(validated),
    )


def _relation_to_zero(interval: IntervalEstimate) -> str:
    if interval.upper_95 < 0:
        return "below-zero"
    if interval.lower_95 > 0:
        return "above-zero"
    return "includes-zero"


@dataclass(frozen=True, slots=True)
class ContrastEstimate:
    """Adaptive-minus-comparator common-regret contrast."""

    scope: str
    persona_id: str | None
    availability_mode: AvailabilityMode | None
    comparator: Strategy
    interval: IntervalEstimate
    interval_relation_to_zero: str

    def __post_init__(self) -> None:
        if self.scope not in {"primary-macro", "macro", "cell"}:
            raise ReportingInputError("contrast scope is invalid")
        if type(self.comparator) is not Strategy:
            raise ReportingInputError("contrast comparator must be a Strategy")
        if self.comparator is Strategy.ADAPTIVE:
            raise ReportingInputError("adaptive cannot compare with itself")
        if not isinstance(self.interval, IntervalEstimate):
            raise ReportingInputError("contrast interval must be an IntervalEstimate")
        if self.scope == "cell":
            if self.persona_id is None or self.availability_mode is None:
                raise ReportingInputError(
                    "cell contrast requires persona and availability mode"
                )
            _safe_identifier(self.persona_id, "persona_id")
            if type(self.availability_mode) is not AvailabilityMode:
                raise ReportingInputError(
                    "cell availability_mode must be an AvailabilityMode"
                )
        elif self.persona_id is not None or self.availability_mode is not None:
            raise ReportingInputError("macro contrast must not carry cell identifiers")
        if self.interval_relation_to_zero not in {
            "below-zero",
            "includes-zero",
            "above-zero",
        }:
            raise ReportingInputError("interval relation is invalid")
        if self.interval_relation_to_zero != _relation_to_zero(self.interval):
            raise ReportingInputError(
                "interval relation does not match interval bounds"
            )


@dataclass(frozen=True, slots=True)
class StrategyCellEstimate:
    """Absolute seed-level means for one persona, mode, and strategy."""

    persona_id: str
    persona_primary: bool
    availability_mode: AvailabilityMode
    strategy: Strategy
    seed_count: int
    mean_common_expected_regret: float
    mean_conditional_expected_regret: float
    mean_path_opportunity_cost: float
    mean_expected_reward: float
    mean_realized_reward: float
    right_fit_rate: float
    completion_rate: float
    common_regret_standard_error: float | None

    def __post_init__(self) -> None:
        _safe_identifier(self.persona_id, "persona_id")
        if type(self.persona_primary) is not bool:
            raise ReportingInputError("persona_primary must be boolean")
        if type(self.availability_mode) is not AvailabilityMode:
            raise ReportingInputError("availability_mode must be an AvailabilityMode")
        if type(self.strategy) is not Strategy:
            raise ReportingInputError("strategy must be a Strategy")
        if (
            isinstance(self.seed_count, bool)
            or not isinstance(self.seed_count, int)
            or self.seed_count < 1
        ):
            raise ReportingInputError("seed_count must be positive")
        unit_fields = (
            "mean_common_expected_regret",
            "mean_conditional_expected_regret",
            "mean_path_opportunity_cost",
            "mean_expected_reward",
            "mean_realized_reward",
            "right_fit_rate",
            "completion_rate",
        )
        for field in unit_fields:
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0:
                raise ReportingInputError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, value)
        standard_error = self.common_regret_standard_error
        if standard_error is not None:
            standard_error = _finite(
                standard_error,
                "common_regret_standard_error",
            )
            if standard_error < 0:
                raise ReportingInputError(
                    "common_regret_standard_error must not be negative"
                )
            object.__setattr__(
                self,
                "common_regret_standard_error",
                standard_error,
            )
        if not math.isclose(
            self.mean_common_expected_regret,
            self.mean_conditional_expected_regret + self.mean_path_opportunity_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ReportingInputError(
                "strategy-cell regret decomposition is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class BootstrapMetadata:
    """Persistable identity for the shared resample matrix."""

    version: str
    resample_count: int
    confidence: float
    namespace_sha256: str
    indices_sha256: str
    mode_indices_sha256: tuple[tuple[AvailabilityMode, str], ...]

    def __post_init__(self) -> None:
        try:
            mode_digests = tuple(tuple(item) for item in self.mode_indices_sha256)
        except TypeError as error:
            raise ReportingInputError(
                "mode index digests must be a finite collection"
            ) from error
        object.__setattr__(self, "mode_indices_sha256", mode_digests)
        if self.version != BOOTSTRAP_VERSION:
            raise ReportingInputError("bootstrap metadata version is invalid")
        if (
            isinstance(self.resample_count, bool)
            or not isinstance(self.resample_count, int)
            or not 1 <= self.resample_count <= 100_000
        ):
            raise ReportingInputError("bootstrap metadata resample count is invalid")
        confidence = _finite(self.confidence, "bootstrap confidence")
        if confidence != BOOTSTRAP_CONFIDENCE:
            raise ReportingInputError("bootstrap metadata confidence is invalid")
        object.__setattr__(self, "confidence", confidence)
        _sha256_digest(self.namespace_sha256, "namespace_sha256")
        _sha256_digest(self.indices_sha256, "indices_sha256")
        if (
            not mode_digests
            or any(len(item) != 2 for item in mode_digests)
            or any(type(item[0]) is not AvailabilityMode for item in mode_digests)
            or len({item[0] for item in mode_digests}) != len(mode_digests)
        ):
            raise ReportingInputError("mode index digest identities are invalid")
        for _mode, digest in mode_digests:
            _sha256_digest(digest, "mode indices digest")


@dataclass(frozen=True, slots=True)
class StatisticalReport:
    """Frozen report-ready statistics with no presentation decisions."""

    schema_version: str
    evaluator_id: str
    design_id: str
    policy_id: str
    bootstrap: BootstrapMetadata
    primary_contrast: ContrastEstimate
    macro_contrasts: tuple[ContrastEstimate, ...]
    cell_contrasts: tuple[ContrastEstimate, ...]
    strategy_cells: tuple[StrategyCellEstimate, ...]

    def __post_init__(self) -> None:
        try:
            macro_contrasts = tuple(self.macro_contrasts)
            cell_contrasts = tuple(self.cell_contrasts)
            strategy_cells = tuple(self.strategy_cells)
        except TypeError as error:
            raise ReportingInputError("report collections must be finite") from error
        object.__setattr__(self, "macro_contrasts", macro_contrasts)
        object.__setattr__(self, "cell_contrasts", cell_contrasts)
        object.__setattr__(self, "strategy_cells", strategy_cells)
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ReportingInputError("report schema version is invalid")
        for field in ("evaluator_id", "design_id", "policy_id"):
            _safe_identifier(getattr(self, field), field)
        if not isinstance(self.bootstrap, BootstrapMetadata):
            raise ReportingInputError("report bootstrap metadata is invalid")
        if not isinstance(self.primary_contrast, ContrastEstimate):
            raise ReportingInputError("primary contrast is invalid")
        if any(
            not isinstance(contrast, ContrastEstimate)
            for contrast in (*macro_contrasts, *cell_contrasts)
        ) or any(not isinstance(cell, StrategyCellEstimate) for cell in strategy_cells):
            raise ReportingInputError("report collection member has an invalid type")
        if (
            self.primary_contrast.scope != "primary-macro"
            or self.primary_contrast.comparator is not Strategy.FIXED_25
            or self.primary_contrast not in macro_contrasts
        ):
            raise ReportingInputError(
                "primary contrast must be the registered fixed-25 macro contrast"
            )
        expected_comparators = {
            strategy for strategy in Strategy if strategy is not Strategy.ADAPTIVE
        }
        if (
            any(
                contrast.scope not in {"primary-macro", "macro"}
                for contrast in macro_contrasts
            )
            or len(macro_contrasts) != len(expected_comparators)
            or {contrast.comparator for contrast in macro_contrasts}
            != expected_comparators
        ):
            raise ReportingInputError("macro contrast set is incomplete or duplicated")
        if any(contrast.scope != "cell" for contrast in cell_contrasts):
            raise ReportingInputError("cell contrast collection has a non-cell member")
        cell_keys = tuple(
            (
                contrast.persona_id,
                contrast.availability_mode,
                contrast.comparator,
            )
            for contrast in cell_contrasts
        )
        if len(set(cell_keys)) != len(cell_keys):
            raise ReportingInputError("cell contrast set contains duplicates")
        strategy_keys = tuple(
            (cell.persona_id, cell.availability_mode, cell.strategy)
            for cell in strategy_cells
        )
        if len(set(strategy_keys)) != len(strategy_keys):
            raise ReportingInputError("strategy cell set contains duplicates")
        scenario_keys = {
            (cell.persona_id, cell.availability_mode) for cell in strategy_cells
        }
        if not scenario_keys:
            raise ReportingInputError("strategy cell set must not be empty")
        if set(strategy_keys) != {
            (persona_id, mode, strategy)
            for persona_id, mode in scenario_keys
            for strategy in Strategy
        }:
            raise ReportingInputError("strategy cell Cartesian set is incomplete")
        if set(cell_keys) != {
            (persona_id, mode, comparator)
            for persona_id, mode in scenario_keys
            for comparator in expected_comparators
        }:
            raise ReportingInputError("cell contrast Cartesian set is incomplete")


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


def _cell_metric_values(
    index: dict[tuple[str, AvailabilityMode, int, Strategy], ClusterSummary],
    *,
    persona_id: str,
    mode: AvailabilityMode,
    strategy: Strategy,
    seeds: tuple[int, ...],
    metric: str,
) -> tuple[float, ...]:
    return tuple(
        float(
            getattr(
                index[(persona_id, mode, seed, strategy)].metrics,
                metric,
            )
        )
        for seed in seeds
    )


def _contrast(
    *,
    scope: str,
    persona_id: str | None,
    mode: AvailabilityMode | None,
    comparator: Strategy,
    seed_effects_by_mode: dict[AvailabilityMode, tuple[float, ...]],
    plan: BootstrapPlan,
) -> ContrastEstimate:
    interval = stratified_bootstrap_interval(seed_effects_by_mode, plan)
    return ContrastEstimate(
        scope=scope,
        persona_id=persona_id,
        availability_mode=mode,
        comparator=comparator,
        interval=interval,
        interval_relation_to_zero=_relation_to_zero(interval),
    )


def build_statistical_report(
    result: ExperimentResult,
    *,
    resample_count: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> StatisticalReport:
    """Build every predeclared scalar comparison from a validated result."""

    validate_experiment_result(result)
    config = result.config
    seeds = config.environment_seeds
    plan = build_bootstrap_plan(
        seeds,
        availability_modes=config.availability_modes,
        evaluator_id=result.evaluator_id,
        resample_count=resample_count,
    )
    index = _summary_index(result)
    primary_personas = tuple(persona for persona in config.personas if persona.primary)
    if not primary_personas:
        raise ReportingInvariantError("at least one primary persona is required")
    comparators = tuple(
        strategy for strategy in Strategy if strategy is not Strategy.ADAPTIVE
    )
    for persona in config.personas:
        for mode in config.availability_modes:
            for seed in seeds:
                adaptive_metrics = index[
                    (
                        persona.persona_id,
                        mode,
                        seed,
                        Strategy.ADAPTIVE,
                    )
                ].metrics
                for comparator in comparators:
                    comparator_metrics = index[
                        (
                            persona.persona_id,
                            mode,
                            seed,
                            comparator,
                        )
                    ].metrics
                    direction_residual = (
                        adaptive_metrics.mean_common_expected_regret
                        - comparator_metrics.mean_common_expected_regret
                        + adaptive_metrics.mean_expected_reward
                        - comparator_metrics.mean_expected_reward
                    )
                    if not math.isclose(
                        direction_residual,
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ReportingInvariantError(
                            "common-regret and expected-reward contrasts "
                            "do not have opposite signs"
                        )

    macro_contrasts: list[ContrastEstimate] = []
    for comparator in comparators:
        effects_by_mode: dict[AvailabilityMode, tuple[float, ...]] = {}
        for mode in config.availability_modes:
            mode_effects: list[float] = []
            for seed in seeds:
                persona_effects = [
                    (
                        index[
                            (
                                persona.persona_id,
                                mode,
                                seed,
                                Strategy.ADAPTIVE,
                            )
                        ].metrics.mean_common_expected_regret
                        - index[
                            (
                                persona.persona_id,
                                mode,
                                seed,
                                comparator,
                            )
                        ].metrics.mean_common_expected_regret
                    )
                    for persona in primary_personas
                ]
                mode_effects.append(_mean(persona_effects))
            effects_by_mode[mode] = tuple(mode_effects)
        scope = "primary-macro" if comparator is Strategy.FIXED_25 else "macro"
        macro_contrasts.append(
            _contrast(
                scope=scope,
                persona_id=None,
                mode=None,
                comparator=comparator,
                seed_effects_by_mode=effects_by_mode,
                plan=plan,
            )
        )

    cell_contrasts: list[ContrastEstimate] = []
    for persona in config.personas:
        for mode in config.availability_modes:
            adaptive = _cell_metric_values(
                index,
                persona_id=persona.persona_id,
                mode=mode,
                strategy=Strategy.ADAPTIVE,
                seeds=seeds,
                metric="mean_common_expected_regret",
            )
            for comparator in comparators:
                baseline = _cell_metric_values(
                    index,
                    persona_id=persona.persona_id,
                    mode=mode,
                    strategy=comparator,
                    seeds=seeds,
                    metric="mean_common_expected_regret",
                )
                effects = tuple(
                    adaptive_value - baseline_value
                    for adaptive_value, baseline_value in zip(
                        adaptive,
                        baseline,
                        strict=True,
                    )
                )
                cell_contrasts.append(
                    _contrast(
                        scope="cell",
                        persona_id=persona.persona_id,
                        mode=mode,
                        comparator=comparator,
                        seed_effects_by_mode={mode: effects},
                        plan=plan,
                    )
                )

    strategy_cells: list[StrategyCellEstimate] = []
    for persona in config.personas:
        for mode in config.availability_modes:
            for strategy in Strategy:
                summaries = tuple(
                    index[(persona.persona_id, mode, seed, strategy)] for seed in seeds
                )
                common_values = tuple(
                    summary.metrics.mean_common_expected_regret for summary in summaries
                )
                strategy_cells.append(
                    StrategyCellEstimate(
                        persona_id=persona.persona_id,
                        persona_primary=persona.primary,
                        availability_mode=mode,
                        strategy=strategy,
                        seed_count=len(seeds),
                        mean_common_expected_regret=_mean(common_values),
                        mean_conditional_expected_regret=_mean(
                            tuple(
                                summary.metrics.mean_conditional_expected_regret
                                for summary in summaries
                            )
                        ),
                        mean_path_opportunity_cost=_mean(
                            tuple(
                                summary.metrics.mean_path_opportunity_cost
                                for summary in summaries
                            )
                        ),
                        mean_expected_reward=_mean(
                            tuple(
                                summary.metrics.mean_expected_reward
                                for summary in summaries
                            )
                        ),
                        mean_realized_reward=_mean(
                            tuple(
                                summary.metrics.mean_realized_reward
                                for summary in summaries
                            )
                        ),
                        right_fit_rate=_mean(
                            tuple(
                                summary.metrics.right_fit_rate for summary in summaries
                            )
                        ),
                        completion_rate=_mean(
                            tuple(
                                summary.metrics.completion_rate for summary in summaries
                            )
                        ),
                        common_regret_standard_error=(
                            seed_cluster_standard_error(common_values)
                        ),
                    )
                )

    primary = next(
        contrast
        for contrast in macro_contrasts
        if contrast.comparator is Strategy.FIXED_25
    )
    return StatisticalReport(
        schema_version=REPORT_SCHEMA_VERSION,
        evaluator_id=result.evaluator_id,
        design_id=result.design_id,
        policy_id=result.policy_id,
        bootstrap=BootstrapMetadata(
            version=plan.version,
            resample_count=plan.resample_count,
            confidence=BOOTSTRAP_CONFIDENCE,
            namespace_sha256=plan.namespace_sha256,
            indices_sha256=plan.indices_sha256,
            mode_indices_sha256=tuple(
                (
                    mode_plan.availability_mode,
                    mode_plan.indices_sha256,
                )
                for mode_plan in plan.mode_plans
            ),
        ),
        primary_contrast=primary,
        macro_contrasts=tuple(macro_contrasts),
        cell_contrasts=tuple(cell_contrasts),
        strategy_cells=tuple(strategy_cells),
    )


@dataclass(frozen=True, slots=True)
class SourceFileIdentity:
    """One loaded source file proven equal to its committed bytes."""

    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        relative = _validated_relative_source_path(self.relative_path)
        object.__setattr__(self, "relative_path", relative)
        _sha256_digest(self.sha256, "source file sha256")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Clean Git and runtime identity captured before a locked run."""

    source_commit: str
    source_tree: str
    git_object_format: str
    source_archive_sha256: str
    source_commit_time: str
    branch: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    python_implementation: str
    python_version: str
    python_cache_tag: str
    platform: str
    loaded_sources: tuple[SourceFileIdentity, ...]
    clean_pre_run: bool

    def __post_init__(self) -> None:
        try:
            loaded_sources = tuple(self.loaded_sources)
        except TypeError as error:
            raise ReportingInputError("loaded_sources must be finite") from error
        object.__setattr__(self, "loaded_sources", loaded_sources)
        if type(self.clean_pre_run) is not bool or not self.clean_pre_run:
            raise ReportingInputError("clean_pre_run must be true")
        if self.git_object_format not in {"sha1", "sha256"}:
            raise ReportingInputError("git_object_format is invalid")
        expected_object_length = 40 if self.git_object_format == "sha1" else 64
        for field in ("source_commit", "source_tree"):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or len(value) != expected_object_length
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ReportingInputError(f"{field} is not a valid object digest")
        _sha256_digest(self.source_archive_sha256, "source_archive_sha256")
        expected_identity = (
            LOCKED_AUTHOR_NAME,
            LOCKED_AUTHOR_EMAIL,
            LOCKED_AUTHOR_NAME,
            LOCKED_AUTHOR_EMAIL,
        )
        actual_identity = (
            self.author_name,
            self.author_email,
            self.committer_name,
            self.committer_email,
        )
        if actual_identity != expected_identity:
            raise ReportingInputError("source identity must belong to Omar Ibrahim")
        for field in (
            "source_commit_time",
            "branch",
            "python_implementation",
            "python_version",
            "python_cache_tag",
            "platform",
        ):
            _safe_identifier(getattr(self, field), field)
        if (
            not loaded_sources
            or any(
                not isinstance(source, SourceFileIdentity) for source in loaded_sources
            )
            or tuple(
                sorted(
                    loaded_sources,
                    key=lambda source: source.relative_path,
                )
            )
            != loaded_sources
            or len({source.relative_path for source in loaded_sources})
            != len(loaded_sources)
        ):
            raise ReportingInputError(
                "loaded_sources must be a non-empty sorted unique source set"
            )


def _git(
    repo_root: Path,
    arguments: Sequence[str],
) -> bytes:
    environment = os.environ.copy()
    for field in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
    ):
        environment.pop(field, None)
    for field in tuple(environment):
        if field == "GIT_CONFIG_COUNT" or field.startswith(
            ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
        ):
            environment.pop(field, None)
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    try:
        completed = subprocess.run(
            (
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ),
            cwd=repo_root,
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=environment,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        command = " ".join(arguments)
        raise ReportingInputError(f"git {command} could not execute") from error
    if completed.returncode != 0:
        command = " ".join(arguments)
        raise ReportingInputError(f"git {command} failed")
    return completed.stdout


def _decode_git_line(value: bytes, field: str) -> str:
    try:
        decoded = value.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise ReportingInputError(f"{field} is not UTF-8") from error
    return _safe_identifier(decoded, field)


def _validated_relative_source_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ReportingInputError("loaded source path is unsafe")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", "..", ".git"} for part in path.parts)
    ):
        raise ReportingInputError("loaded source path is unsafe")
    return path.as_posix()


def _source_file_state(path: Path) -> tuple[int, int, int, int, int, int, int]:
    metadata = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size < 0
    ):
        raise ReportingInputError("loaded source must be a single-link regular file")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def capture_source_provenance(
    repo_root: Path,
    *,
    loaded_sources: Sequence[tuple[str, Path]] = (),
) -> SourceProvenance:
    """Capture a clean, stable Git snapshot and verify loaded source bytes."""

    if not isinstance(repo_root, Path):
        raise ReportingInputError("repo_root must be a Path")
    try:
        declared_sources = tuple(loaded_sources)
    except TypeError as error:
        raise ReportingInputError("loaded_sources must be finite") from error
    if not declared_sources:
        raise ReportingInputError("loaded_sources must not be empty")
    root = repo_root.resolve(strict=True)
    discovered = Path(
        _decode_git_line(
            _git(root, ("rev-parse", "--show-toplevel")),
            "Git top-level path",
        )
    ).resolve(strict=True)
    if discovered != root:
        raise ReportingInputError("repo_root is not the Git top level")
    if _git(root, ("status", "--porcelain=v1", "--untracked-files=all")):
        raise ReportingInputError("source checkout must be clean before evaluation")

    object_format = _decode_git_line(
        _git(root, ("rev-parse", "--show-object-format")),
        "Git object format",
    )
    if object_format not in {"sha1", "sha256"}:
        raise ReportingInputError("Git object format is unsupported")
    commit = _decode_git_line(
        _git(root, ("rev-parse", "HEAD")),
        "source commit",
    )
    tree = _decode_git_line(
        _git(root, ("rev-parse", "HEAD^{tree}")),
        "source tree",
    )
    identity_bytes = _git(
        root,
        (
            "show",
            "-s",
            "--format=%an%x00%ae%x00%cn%x00%ce",
            "HEAD",
        ),
    ).rstrip(b"\n")
    try:
        identity = tuple(
            item.decode("utf-8", errors="strict")
            for item in identity_bytes.split(b"\x00")
        )
    except UnicodeDecodeError as error:
        raise ReportingInputError("commit identity is not UTF-8") from error
    if len(identity) != 4:
        raise ReportingInputError("commit identity has an invalid shape")
    expected_identity = (
        LOCKED_AUTHOR_NAME,
        LOCKED_AUTHOR_EMAIL,
        LOCKED_AUTHOR_NAME,
        LOCKED_AUTHOR_EMAIL,
    )
    if identity != expected_identity:
        raise ReportingInputError("HEAD author and committer must be Omar Ibrahim")

    source_identities: list[SourceFileIdentity] = []
    source_verifications: list[tuple[Path, str]] = []
    seen_paths: set[str] = set()
    for relative_value, actual_path in declared_sources:
        relative = _validated_relative_source_path(relative_value)
        if relative in seen_paths:
            raise ReportingInputError(f"loaded source repeats {relative}")
        seen_paths.add(relative)
        if not isinstance(actual_path, Path):
            raise ReportingInputError("loaded source location must be a Path")
        candidate_path = root.joinpath(*PurePosixPath(relative).parts)
        expected_path = candidate_path.resolve(strict=True)
        if expected_path != candidate_path or not expected_path.is_relative_to(root):
            raise ReportingInputError(
                f"loaded source must not traverse a symlink: {relative}"
            )
        if actual_path.resolve(strict=True) != expected_path:
            raise ReportingInputError(
                f"loaded source path does not match checkout: {relative}"
            )
        state_before = _source_file_state(expected_path)
        actual_bytes = expected_path.read_bytes()
        if _source_file_state(expected_path) != state_before:
            raise ReportingInputError(f"loaded source changed while read: {relative}")
        committed_bytes = _git(root, ("show", f"HEAD:{relative}"))
        if actual_bytes != committed_bytes:
            raise ReportingInputError(
                f"loaded source bytes differ from HEAD: {relative}"
            )
        source_identities.append(
            SourceFileIdentity(
                relative_path=relative,
                sha256=hashlib.sha256(actual_bytes).hexdigest(),
            )
        )
        source_verifications.append(
            (
                expected_path,
                hashlib.sha256(actual_bytes).hexdigest(),
            )
        )

    archive_digest = hashlib.sha256(
        _git(root, ("archive", "--format=tar", "HEAD"))
    ).hexdigest()
    commit_time = _decode_git_line(
        _git(root, ("show", "-s", "--format=%cI", "HEAD")),
        "source commit time",
    )
    branch = _decode_git_line(
        _git(root, ("rev-parse", "--abbrev-ref", "HEAD")),
        "branch",
    )

    # Close the ordinary race between the initial cleanliness check and the
    # source/archive reads. A hostile same-user process remains out of scope.
    if _git(root, ("status", "--porcelain=v1", "--untracked-files=all")):
        raise ReportingInputError("source checkout changed during capture")
    if (
        _decode_git_line(
            _git(root, ("rev-parse", "HEAD")),
            "source commit",
        )
        != commit
        or _decode_git_line(
            _git(root, ("rev-parse", "HEAD^{tree}")),
            "source tree",
        )
        != tree
    ):
        raise ReportingInputError("Git HEAD changed during provenance capture")
    for path, expected_digest in source_verifications:
        state_before = _source_file_state(path)
        current_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            _source_file_state(path) != state_before
            or current_digest != expected_digest
        ):
            raise ReportingInputError("loaded source changed during provenance capture")

    cache_tag = sys.implementation.cache_tag
    if cache_tag is None:
        raise ReportingInvariantError("Python runtime has no cache tag")
    return SourceProvenance(
        source_commit=commit,
        source_tree=tree,
        git_object_format=object_format,
        source_archive_sha256=archive_digest,
        source_commit_time=commit_time,
        branch=branch,
        author_name=identity[0],
        author_email=identity[1],
        committer_name=identity[2],
        committer_email=identity[3],
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        python_cache_tag=cache_tag,
        platform=platform.platform(aliased=True, terse=True),
        loaded_sources=tuple(
            sorted(
                source_identities,
                key=lambda item: item.relative_path,
            )
        ),
        clean_pre_run=True,
    )
