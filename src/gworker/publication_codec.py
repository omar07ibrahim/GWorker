"""Strict canonical JSON codecs for publication reports and evidence.

The codec has no persistence responsibilities: callers own temporary files,
``fsync``, and atomic replacement. Each document is a deterministic UTF-8 JSON
envelope with this shape::

    {
      "codec_version": "gworker-publication-json-v1",
      "document_type": "statistical-report" | "publication-evidence",
      "payload": { ... exact closed dataclass schema ... }
    }

Objects use sorted keys and no insignificant whitespace. Floats are finite
binary64 values rendered by Python's deterministic JSON number encoder;
negative zero is normalized to ``0.0``. Enums use their frozen string values,
tuples use arrays, and optional values use ``null``.

Decode requires the expected SHA-256 of the exact content. Before constructing
domain objects it enforces byte, nesting, container, string, integer, and
collection bounds; rejects duplicate, missing, and unknown keys; and checks
exact scalar kinds (including the bool/int distinction). A successful decode
must re-encode byte-for-byte to the supplied canonical content.
"""

from __future__ import annotations

import hashlib
import json
import math
import types
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Final, TypeVar, cast, get_args, get_origin, get_type_hints

from .evaluation import (
    LOCKED_POLICY_ID,
    AvailabilityMode,
    EvaluationInputError,
    ExperimentConfig,
    Persona,
    Strategy,
    evaluator_design_fingerprint,
    evaluator_fingerprint,
    validate_experiment_config,
)
from .evidence import (
    RECOVERY_CONTRAST_METRICS,
    AdaptiveDiagnosticEvidence,
    CalibrationBinEvidence,
    EvidenceCardinalities,
    EvidenceScope,
    PublicationEvidence,
    RecoveryContrastEvidence,
    RecoveryEvidence,
    RecoveryMetricContrast,
    RunCompleteness,
    StrategyMetricEvidence,
    TemplateExposureEvidence,
    TracePointEvidence,
)
from .reporting import (
    BOOTSTRAP_CONFIDENCE,
    BootstrapMetadata,
    ContrastEstimate,
    IntervalEstimate,
    ReportingInputError,
    StatisticalReport,
    StrategyCellEstimate,
    build_bootstrap_plan,
)

PUBLICATION_JSON_CODEC_VERSION: Final = "gworker-publication-json-v1"
STATISTICAL_REPORT_DOCUMENT_TYPE: Final = "statistical-report"
PUBLICATION_EVIDENCE_DOCUMENT_TYPE: Final = "publication-evidence"

_MAX_REPORT_BYTES: Final = 2 * 1024 * 1024
_MAX_EVIDENCE_BYTES: Final = 16 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 32
_MAX_JSON_CONTAINERS: Final = 50_000
_MAX_JSON_DELIMITERS: Final = 500_000
_MAX_STRING_TOKEN_BYTES: Final = 8_192
_MAX_TEXT_BYTES: Final = 4_096
_MAX_NUMBER_TOKEN_BYTES: Final = 64
_MIN_JSON_INTEGER: Final = -(2**63)
_MAX_JSON_INTEGER: Final = 2**63 - 1

_REPORT_MACRO_CONTRAST_LIMIT: Final = 6
_REPORT_CELL_CONTRAST_LIMIT: Final = 108
_REPORT_STRATEGY_CELL_LIMIT: Final = 126
_EVIDENCE_STRATEGY_METRIC_LIMIT: Final = 133
_EVIDENCE_ADAPTIVE_DIAGNOSTIC_LIMIT: Final = 19
_EVIDENCE_CALIBRATION_LIMIT: Final = 152
_EVIDENCE_TEMPLATE_EXPOSURE_LIMIT: Final = 532
_EVIDENCE_RECOVERY_LIMIT: Final = 28
_EVIDENCE_RECOVERY_CONTRAST_LIMIT: Final = 24
_EVIDENCE_TRACE_POINT_LIMIT: Final = 8_064
_CONFIG_SEED_LIMIT: Final = 128
_CONFIG_PERSONA_LIMIT: Final = 9
_CONFIG_MODE_LIMIT: Final = 2
_MODE_DIGEST_LIMIT: Final = 2
_RECOVERY_METRIC_LIMIT: Final = 5


class PublicationCodecError(ValueError):
    """Raised when canonical publication content fails an integrity check."""


@dataclass(frozen=True, slots=True)
class CanonicalJsonDocument:
    """Exact canonical content and its SHA-256 identity."""

    content: bytes
    content_sha256: str

    def __post_init__(self) -> None:
        if type(self.content) is not bytes:
            raise PublicationCodecError("canonical content must be exact bytes")
        _validate_sha256(self.content_sha256, field="content_sha256")
        if compute_content_sha256(self.content) != self.content_sha256:
            raise PublicationCodecError("content_sha256 does not match content")


RootModel = StatisticalReport | PublicationEvidence
RootModelT = TypeVar("RootModelT", StatisticalReport, PublicationEvidence)

_MODEL_SCHEMAS: Final[dict[type[object], tuple[tuple[str, object], ...]]] = {
    Persona: (
        ("persona_id", str),
        ("label", str),
        ("primary", bool),
        ("sigma_minutes", float),
        ("selective_reviews", bool),
    ),
    ExperimentConfig: (
        ("split", str),
        ("environment_seeds", tuple[int, ...]),
        ("policy_replicas", int),
        ("personas", tuple[Persona, ...]),
        ("availability_modes", tuple[AvailabilityMode, ...]),
        ("horizon", int),
        ("drift_decision", int),
        ("recovery_block_size", int),
        ("recovery_blocks", int),
    ),
    IntervalEstimate: (
        ("point_estimate", float),
        ("lower_95", float),
        ("upper_95", float),
        ("seed_cluster_standard_error", float | None),
        ("seed_count_per_stratum", int),
        ("stratum_count", int),
    ),
    ContrastEstimate: (
        ("scope", str),
        ("persona_id", str | None),
        ("availability_mode", AvailabilityMode | None),
        ("comparator", Strategy),
        ("interval", IntervalEstimate),
        ("interval_relation_to_zero", str),
    ),
    StrategyCellEstimate: (
        ("persona_id", str),
        ("persona_primary", bool),
        ("availability_mode", AvailabilityMode),
        ("strategy", Strategy),
        ("seed_count", int),
        ("mean_common_expected_regret", float),
        ("mean_conditional_expected_regret", float),
        ("mean_path_opportunity_cost", float),
        ("mean_expected_reward", float),
        ("mean_realized_reward", float),
        ("right_fit_rate", float),
        ("completion_rate", float),
        ("common_regret_standard_error", float | None),
    ),
    BootstrapMetadata: (
        ("version", str),
        ("resample_count", int),
        ("confidence", float),
        ("namespace_sha256", str),
        ("indices_sha256", str),
        ("mode_indices_sha256", tuple[tuple[AvailabilityMode, str], ...]),
    ),
    StatisticalReport: (
        ("schema_version", str),
        ("evaluator_id", str),
        ("design_id", str),
        ("policy_id", str),
        ("bootstrap", BootstrapMetadata),
        ("primary_contrast", ContrastEstimate),
        ("macro_contrasts", tuple[ContrastEstimate, ...]),
        ("cell_contrasts", tuple[ContrastEstimate, ...]),
        ("strategy_cells", tuple[StrategyCellEstimate, ...]),
    ),
    EvidenceScope: (
        ("kind", str),
        ("persona_id", str | None),
        ("persona_primary", bool | None),
        ("availability_mode", AvailabilityMode | None),
    ),
    EvidenceCardinalities: (
        ("cluster_summaries", int),
        ("abrupt_traces", int),
        ("raw_trace_points", int),
        ("trajectories", int),
        ("decisions", int),
        ("scenario_strategy_rows", int),
        ("macro_strategy_rows", int),
        ("macro_contrasts", int),
        ("scenario_contrasts", int),
        ("adaptive_diagnostic_scopes", int),
        ("calibration_rows", int),
        ("template_exposure_rows", int),
        ("recovery_rows", int),
        ("recovery_contrast_rows", int),
        ("trace_series", int),
        ("trace_points", int),
    ),
    RunCompleteness: (
        ("expected", EvidenceCardinalities),
        ("actual", EvidenceCardinalities),
        ("hard_failure_count", int),
    ),
    StrategyMetricEvidence: (
        ("scope", EvidenceScope),
        ("strategy", Strategy),
        ("persona_count", int),
        ("mode_count", int),
        ("seed_count_per_cell", int),
        ("trajectory_count", int),
        ("action_count", int),
        ("horizon", int),
        ("mean_common_expected_regret", float),
        ("mean_conditional_expected_regret", float),
        ("mean_path_opportunity_cost", float),
        ("mean_cumulative_common_expected_regret", float),
        ("mean_cumulative_conditional_expected_regret", float),
        ("mean_cumulative_path_opportunity_cost", float),
        ("mean_expected_reward", float),
        ("mean_realized_reward", float),
        ("right_fit_rate", float),
        ("completion_rate", float),
        ("mean_common_oracle_distance", float),
        ("mean_conditional_oracle_distance", float),
        ("common_regret_standard_error", float | None),
        ("availability_guardrail_count", int),
        ("one_step_guardrail_count", int),
        ("availability_override_count", int),
        ("feasible_set_size_sum", float),
        ("review_count", int),
        ("availability_guardrail_rate", float),
        ("one_step_guardrail_rate", float),
        ("availability_override_rate", float),
        ("mean_feasible_set_size", float),
        ("review_rate", float),
        ("maximum_arm_transition", int),
    ),
    AdaptiveDiagnosticEvidence: (
        ("scope", EvidenceScope),
        ("trajectory_count", int),
        ("action_count", int),
        ("review_count", int),
        ("selected_propensity_count", int),
        ("low_propensity_count", int),
        ("inverse_propensity_sum", float),
        ("inverse_propensity_squared_sum", float),
        ("brier_score_sum", float),
        ("brier_score_count", int),
        ("exact_evidence_count", int),
        ("task_evidence_count", int),
        ("global_evidence_count", int),
        ("probability_floor_count", int),
        ("arm_probability_count", int),
        ("minimum_propensity", float),
        ("maximum_inverse_propensity", float),
        ("review_rate", float),
        ("low_propensity_rate", float),
        ("inverse_propensity_ess_ratio", float),
        ("multiclass_brier_score", float),
        ("exact_evidence_rate", float),
        ("task_evidence_rate", float),
        ("global_evidence_rate", float),
        ("probability_floor_rate", float),
    ),
    CalibrationBinEvidence: (
        ("scope", EvidenceScope),
        ("bin_index", int),
        ("lower_bound", float),
        ("upper_bound", float),
        ("upper_inclusive", bool),
        ("predicted_sum", float),
        ("observed_sum", float),
        ("count", int),
        ("mean_predicted_probability", float | None),
        ("observed_frequency", float | None),
    ),
    TemplateExposureEvidence: (
        ("scope", EvidenceScope),
        ("strategy", Strategy),
        ("template_id", str),
        ("focus_seconds", int),
        ("break_seconds", int),
        ("exposure_count", int),
        ("action_count", int),
        ("exposure_rate", float),
    ),
    RecoveryEvidence: (
        ("persona_id", str),
        ("availability_mode", AvailabilityMode),
        ("strategy", Strategy),
        ("seed_count", int),
        ("trajectory_count", int),
        ("pre_drift_regret", float),
        ("early_post_drift_auc", float),
        ("late_regret", float),
        ("recovered_count", int),
        ("recovered_lag_sum", float),
        ("recovery_rate", float),
        ("recovered_only_lag", float | None),
        ("conservative_recovery_lag", float),
        ("censor_lag", int),
        ("maximum_recovered_lag", int),
    ),
    RecoveryMetricContrast: (
        ("metric", str),
        ("interval", IntervalEstimate),
        ("interval_relation_to_zero", str),
    ),
    RecoveryContrastEvidence: (
        ("persona_id", str),
        ("availability_mode", AvailabilityMode),
        ("comparator", Strategy),
        ("metrics", tuple[RecoveryMetricContrast, ...]),
    ),
    TracePointEvidence: (
        ("persona_id", str),
        ("availability_mode", AvailabilityMode),
        ("strategy", Strategy),
        ("decision", int),
        ("interval_kind", str),
        ("interval", IntervalEstimate),
    ),
    PublicationEvidence: (
        ("schema_version", str),
        ("evidence_kind", str),
        ("result_schema_version", str),
        ("config", ExperimentConfig),
        ("evaluator_id", str),
        ("design_id", str),
        ("population_id", str),
        ("policy_id", str),
        ("completeness", RunCompleteness),
        ("statistics", StatisticalReport),
        ("strategy_metrics", tuple[StrategyMetricEvidence, ...]),
        ("adaptive_diagnostics", tuple[AdaptiveDiagnosticEvidence, ...]),
        ("calibration_bins", tuple[CalibrationBinEvidence, ...]),
        ("template_exposures", tuple[TemplateExposureEvidence, ...]),
        ("recovery_metrics", tuple[RecoveryEvidence, ...]),
        ("recovery_contrasts", tuple[RecoveryContrastEvidence, ...]),
        ("trace_points", tuple[TracePointEvidence, ...]),
    ),
}

_COLLECTION_LIMITS: Final[dict[tuple[type[object], str], int]] = {
    (ExperimentConfig, "environment_seeds"): _CONFIG_SEED_LIMIT,
    (ExperimentConfig, "personas"): _CONFIG_PERSONA_LIMIT,
    (ExperimentConfig, "availability_modes"): _CONFIG_MODE_LIMIT,
    (BootstrapMetadata, "mode_indices_sha256"): _MODE_DIGEST_LIMIT,
    (StatisticalReport, "macro_contrasts"): _REPORT_MACRO_CONTRAST_LIMIT,
    (StatisticalReport, "cell_contrasts"): _REPORT_CELL_CONTRAST_LIMIT,
    (StatisticalReport, "strategy_cells"): _REPORT_STRATEGY_CELL_LIMIT,
    (
        PublicationEvidence,
        "strategy_metrics",
    ): _EVIDENCE_STRATEGY_METRIC_LIMIT,
    (
        PublicationEvidence,
        "adaptive_diagnostics",
    ): _EVIDENCE_ADAPTIVE_DIAGNOSTIC_LIMIT,
    (PublicationEvidence, "calibration_bins"): _EVIDENCE_CALIBRATION_LIMIT,
    (
        PublicationEvidence,
        "template_exposures",
    ): _EVIDENCE_TEMPLATE_EXPOSURE_LIMIT,
    (PublicationEvidence, "recovery_metrics"): _EVIDENCE_RECOVERY_LIMIT,
    (
        PublicationEvidence,
        "recovery_contrasts",
    ): _EVIDENCE_RECOVERY_CONTRAST_LIMIT,
    (PublicationEvidence, "trace_points"): _EVIDENCE_TRACE_POINT_LIMIT,
    (RecoveryContrastEvidence, "metrics"): _RECOVERY_METRIC_LIMIT,
}

_AVAILABILITY_MODE_SCHEMA: Final = (
    ("UNCONSTRAINED", "unconstrained"),
    ("GUARDRAILED", "guardrailed"),
)
_STRATEGY_SCHEMA: Final = (
    ("ADAPTIVE", "adaptive"),
    ("FIXED_15", "fixed-15"),
    ("FIXED_25", "fixed-25"),
    ("FIXED_40", "fixed-40"),
    ("FIXED_50", "fixed-50"),
    ("LAST_CHOICE", "last-choice"),
    ("MYOPIC_ORACLE", "myopic-oracle"),
)
_RECOVERY_METRIC_SCHEMA: Final = (
    "pre-drift-regret",
    "early-post-drift-auc",
    "late-regret",
    "recovery-rate",
    "conservative-recovery-lag",
)


def _resolved_schema(model: type[object]) -> tuple[tuple[str, object], ...]:
    hints = cast(dict[str, object], get_type_hints(model))
    return tuple((field.name, hints[field.name]) for field in fields(cast(Any, model)))


def _is_variadic_tuple(annotation: object) -> bool:
    return (
        get_origin(annotation) is tuple
        and len(get_args(annotation)) == 2
        and get_args(annotation)[1] is Ellipsis
    )


if (
    any(_resolved_schema(model) != schema for model, schema in _MODEL_SCHEMAS.items())
    or {
        (model, name)
        for model, schema in _MODEL_SCHEMAS.items()
        for name, annotation in schema
        if _is_variadic_tuple(annotation)
    }
    != set(_COLLECTION_LIMITS)
    or tuple((member.name, member.value) for member in AvailabilityMode)
    != _AVAILABILITY_MODE_SCHEMA
    or tuple((member.name, member.value) for member in Strategy) != _STRATEGY_SCHEMA
    or RECOVERY_CONTRAST_METRICS != _RECOVERY_METRIC_SCHEMA
):
    raise RuntimeError(
        "publication schema changed without a publication JSON codec version update"
    )


def _validate_report_context(
    report: StatisticalReport,
    config: ExperimentConfig,
) -> None:
    """Bind one report to its full experiment and canonical row order."""

    if type(report) is not StatisticalReport:
        raise PublicationCodecError("report must be an exact StatisticalReport")
    if type(config) is not ExperimentConfig:
        raise PublicationCodecError("config must be an exact ExperimentConfig")
    try:
        validate_experiment_config(config)
    except EvaluationInputError as error:
        raise PublicationCodecError("report config failed closed validation") from error
    if (
        report.evaluator_id != evaluator_fingerprint(config)
        or report.design_id != evaluator_design_fingerprint()
        or report.policy_id != LOCKED_POLICY_ID
    ):
        raise PublicationCodecError("report identifiers disagree with its config")

    try:
        plan = build_bootstrap_plan(
            config.environment_seeds,
            availability_modes=config.availability_modes,
            evaluator_id=report.evaluator_id,
            resample_count=report.bootstrap.resample_count,
        )
        expected_bootstrap = BootstrapMetadata(
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
        )
    except ReportingInputError as error:
        raise PublicationCodecError(
            "report bootstrap cannot be rebuilt from its config"
        ) from error
    if report.bootstrap != expected_bootstrap:
        raise PublicationCodecError("report bootstrap disagrees with its config")

    comparators = tuple(
        strategy for strategy in Strategy if strategy is not Strategy.ADAPTIVE
    )
    expected_cell_contrasts = tuple(
        (persona.persona_id, mode, comparator)
        for persona in config.personas
        for mode in config.availability_modes
        for comparator in comparators
    )
    actual_cell_contrasts = tuple(
        (
            contrast.persona_id,
            contrast.availability_mode,
            contrast.comparator,
        )
        for contrast in report.cell_contrasts
    )
    if actual_cell_contrasts != expected_cell_contrasts:
        raise PublicationCodecError(
            "report cell contrasts are not in canonical config order"
        )

    expected_strategy_cells = tuple(
        (
            persona.persona_id,
            persona.primary,
            mode,
            strategy,
            len(config.environment_seeds),
        )
        for persona in config.personas
        for mode in config.availability_modes
        for strategy in Strategy
    )
    actual_strategy_cells = tuple(
        (
            cell.persona_id,
            cell.persona_primary,
            cell.availability_mode,
            cell.strategy,
            cell.seed_count,
        )
        for cell in report.strategy_cells
    )
    if actual_strategy_cells != expected_strategy_cells:
        raise PublicationCodecError(
            "report strategy cells are not in canonical config order"
        )

    seed_count = len(config.environment_seeds)
    mode_count = len(config.availability_modes)
    primary_personas = tuple(persona for persona in config.personas if persona.primary)
    if not primary_personas:
        raise PublicationCodecError("report config has no primary persona")
    cells = {
        (cell.persona_id, cell.availability_mode, cell.strategy): cell
        for cell in report.strategy_cells
    }

    for contrast in report.macro_contrasts:
        interval = contrast.interval
        if (
            interval.seed_count_per_stratum != seed_count
            or interval.stratum_count != mode_count
        ):
            raise PublicationCodecError(
                "report macro interval population disagrees with its config"
            )
        mode_effects = tuple(
            math.fsum(
                cells[
                    (
                        persona.persona_id,
                        mode,
                        Strategy.ADAPTIVE,
                    )
                ].mean_common_expected_regret
                - cells[
                    (
                        persona.persona_id,
                        mode,
                        contrast.comparator,
                    )
                ].mean_common_expected_regret
                for persona in primary_personas
            )
            / len(primary_personas)
            for mode in config.availability_modes
        )
        expected_point = math.fsum(mode_effects) / mode_count
        if not math.isclose(
            interval.point_estimate,
            expected_point,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PublicationCodecError(
                "report macro point estimate disagrees with strategy cells"
            )

    for contrast in report.cell_contrasts:
        interval = contrast.interval
        if interval.seed_count_per_stratum != seed_count or interval.stratum_count != 1:
            raise PublicationCodecError(
                "report cell interval population disagrees with its config"
            )
        persona_id = contrast.persona_id
        mode = contrast.availability_mode
        if persona_id is None or mode is None:
            raise PublicationCodecError("report cell contrast identity is incomplete")
        adaptive = cells[
            (
                persona_id,
                mode,
                Strategy.ADAPTIVE,
            )
        ]
        comparator = cells[
            (
                persona_id,
                mode,
                contrast.comparator,
            )
        ]
        expected_point = (
            adaptive.mean_common_expected_regret
            - comparator.mean_common_expected_regret
        )
        if not math.isclose(
            interval.point_estimate,
            expected_point,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PublicationCodecError(
                "report cell point estimate disagrees with strategy cells"
            )


def _validate_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationCodecError(f"{field} must be a lowercase SHA-256 digest")
    return value


def compute_content_sha256(content: bytes) -> str:
    """Return the SHA-256 hex identity of exact content bytes."""

    if type(content) is not bytes:
        raise PublicationCodecError("content must be exact bytes")
    return hashlib.sha256(content).hexdigest()


def _finite_float(value: object, *, path: str) -> float:
    if type(value) is not float:
        raise PublicationCodecError(f"{path} must be an exact float")
    result = value
    if not math.isfinite(result):
        raise PublicationCodecError(f"{path} must be finite")
    return 0.0 if result == 0.0 else result


def _bounded_integer(value: object, *, path: str) -> int:
    if type(value) is not int:
        raise PublicationCodecError(f"{path} must be an exact integer")
    result = value
    if not _MIN_JSON_INTEGER <= result <= _MAX_JSON_INTEGER:
        raise PublicationCodecError(f"{path} is outside the JSON integer bound")
    return result


def _bounded_text(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise PublicationCodecError(f"{path} must be exact text")
    result = value
    try:
        encoded = result.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise PublicationCodecError(f"{path} is not valid UTF-8 text") from error
    if len(encoded) > _MAX_TEXT_BYTES:
        raise PublicationCodecError(f"{path} exceeds the text byte bound")
    return result


def _optional_member(annotation: object) -> object | None:
    if get_origin(annotation) is not types.UnionType:
        return None
    arguments = get_args(annotation)
    if len(arguments) != 2 or type(None) not in arguments:
        raise RuntimeError(f"unsupported publication union: {annotation!r}")
    return cast(
        object,
        arguments[0] if arguments[1] is type(None) else arguments[1],
    )


def _encode_value(
    value: object,
    annotation: object,
    *,
    path: str,
    collection_limit: int | None = None,
) -> object:
    optional = _optional_member(annotation)
    if optional is not None:
        if value is None:
            return None
        return _encode_value(value, optional, path=path)
    if annotation is float:
        return _finite_float(value, path=path)
    if annotation is int:
        return _bounded_integer(value, path=path)
    if annotation is bool:
        if type(value) is not bool:
            raise PublicationCodecError(f"{path} must be an exact boolean")
        return value
    if annotation is str:
        return _bounded_text(value, path=path)
    if annotation in (AvailabilityMode, Strategy):
        if type(value) is not annotation:
            raise PublicationCodecError(f"{path} has an invalid enum type")
        return _bounded_text(cast(Enum, value).value, path=path)
    if annotation in _MODEL_SCHEMAS:
        return _encode_model(value, annotation, path=path)
    if get_origin(annotation) is tuple:
        if type(value) is not tuple:
            raise PublicationCodecError(f"{path} must be an exact tuple")
        items = cast(tuple[object, ...], value)
        arguments = get_args(annotation)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            if collection_limit is None:
                raise RuntimeError(f"{path} has no collection bound")
            if len(items) > collection_limit:
                raise PublicationCodecError(f"{path} exceeds its collection bound")
            return [
                _encode_value(item, arguments[0], path=f"{path}[{index}]")
                for index, item in enumerate(items)
            ]
        if len(items) != len(arguments):
            raise PublicationCodecError(f"{path} has a noncanonical tuple width")
        return [
            _encode_value(item, item_type, path=f"{path}[{index}]")
            for index, (item, item_type) in enumerate(
                zip(items, arguments, strict=True)
            )
        ]
    raise RuntimeError(f"unsupported publication type at {path}: {annotation!r}")


def _encode_model(value: object, model: type[object], *, path: str) -> object:
    if type(value) is not model:
        raise PublicationCodecError(f"{path} must be an exact {model.__name__}")
    return {
        name: _encode_value(
            getattr(value, name),
            annotation,
            path=f"{path}.{name}",
            collection_limit=_COLLECTION_LIMITS.get((model, name)),
        )
        for name, annotation in _MODEL_SCHEMAS[model]
    }


def _decode_value(
    value: object,
    annotation: object,
    *,
    path: str,
    collection_limit: int | None = None,
) -> object:
    optional = _optional_member(annotation)
    if optional is not None:
        if value is None:
            return None
        return _decode_value(value, optional, path=path)
    if annotation is float:
        return _finite_float(value, path=path)
    if annotation is int:
        return _bounded_integer(value, path=path)
    if annotation is bool:
        if type(value) is not bool:
            raise PublicationCodecError(f"{path} must be an exact boolean")
        return value
    if annotation is str:
        return _bounded_text(value, path=path)
    if annotation in (AvailabilityMode, Strategy):
        text = _bounded_text(value, path=path)
        try:
            return cast(type[Enum], annotation)(text)
        except ValueError as error:
            raise PublicationCodecError(f"{path} has an invalid enum value") from error
    if annotation in _MODEL_SCHEMAS:
        return _decode_model(value, annotation, path=path)
    if get_origin(annotation) is tuple:
        if type(value) is not list:
            raise PublicationCodecError(f"{path} must be a JSON array")
        items = cast(list[object], value)
        arguments = get_args(annotation)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            if collection_limit is None:
                raise RuntimeError(f"{path} has no collection bound")
            if len(items) > collection_limit:
                raise PublicationCodecError(f"{path} exceeds its collection bound")
            return tuple(
                _decode_value(item, arguments[0], path=f"{path}[{index}]")
                for index, item in enumerate(items)
            )
        if len(items) != len(arguments):
            raise PublicationCodecError(f"{path} has a noncanonical tuple width")
        return tuple(
            _decode_value(item, item_type, path=f"{path}[{index}]")
            for index, (item, item_type) in enumerate(
                zip(items, arguments, strict=True)
            )
        )
    raise RuntimeError(f"unsupported publication type at {path}: {annotation!r}")


def _decode_model(value: object, model: type[object], *, path: str) -> object:
    if type(value) is not dict:
        raise PublicationCodecError(f"{path} must be a JSON object")
    mapping = cast(dict[str, object], value)
    schema = _MODEL_SCHEMAS[model]
    expected_keys = {name for name, _annotation in schema}
    actual_keys = set(mapping)
    missing = expected_keys - actual_keys
    unknown = actual_keys - expected_keys
    if missing:
        raise PublicationCodecError(
            f"{path} is missing keys: {','.join(sorted(missing))}"
        )
    if unknown:
        raise PublicationCodecError(
            f"{path} has unknown keys: {','.join(sorted(unknown))}"
        )
    values = {
        name: _decode_value(
            mapping[name],
            annotation,
            path=f"{path}.{name}",
            collection_limit=_COLLECTION_LIMITS.get((model, name)),
        )
        for name, annotation in schema
    }
    try:
        return model(**values)
    except PublicationCodecError:
        raise
    except (
        AssertionError,
        AttributeError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise PublicationCodecError(
            f"{path} fails the {model.__name__} contract"
        ) from error


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationCodecError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_integer(token: str) -> int:
    if len(token) > _MAX_NUMBER_TOKEN_BYTES:
        raise PublicationCodecError("JSON integer token exceeds its byte bound")
    value = int(token)
    if not _MIN_JSON_INTEGER <= value <= _MAX_JSON_INTEGER:
        raise PublicationCodecError("JSON integer is outside its bound")
    return value


def _parse_float(token: str) -> float:
    if len(token) > _MAX_NUMBER_TOKEN_BYTES:
        raise PublicationCodecError("JSON float token exceeds its byte bound")
    value = float(token)
    if not math.isfinite(value):
        raise PublicationCodecError("JSON float must be finite")
    return 0.0 if value == 0.0 else value


def _reject_constant(token: str) -> object:
    raise PublicationCodecError(f"non-finite JSON constant is forbidden: {token}")


def _check_lexical_bounds(content: bytes) -> None:
    depth = 0
    container_count = 0
    delimiter_count = 0
    string_bytes = 0
    in_string = False
    escaped = False
    for byte in content:
        if in_string:
            string_bytes += 1
            if string_bytes > _MAX_STRING_TOKEN_BYTES:
                raise PublicationCodecError("JSON string token exceeds its byte bound")
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
            string_bytes = 0
        elif byte in (0x7B, 0x5B):
            depth += 1
            container_count += 1
            if depth > _MAX_JSON_DEPTH:
                raise PublicationCodecError("JSON nesting exceeds its depth bound")
            if container_count > _MAX_JSON_CONTAINERS:
                raise PublicationCodecError("JSON container count exceeds its bound")
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise PublicationCodecError("JSON containers are unbalanced")
        elif byte in (0x2C, 0x3A):
            delimiter_count += 1
            if delimiter_count > _MAX_JSON_DELIMITERS:
                raise PublicationCodecError("JSON scalar count exceeds its bound")
    if in_string or depth != 0:
        raise PublicationCodecError("JSON content is structurally incomplete")


def _parse_json(content: bytes) -> object:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise PublicationCodecError("content is not valid UTF-8") from error
    try:
        return cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_object_without_duplicates,
                parse_int=_parse_integer,
                parse_float=_parse_float,
                parse_constant=_reject_constant,
            ),
        )
    except PublicationCodecError:
        raise
    except (RecursionError, TypeError, ValueError) as error:
        raise PublicationCodecError("content is not valid bounded JSON") from error


def _json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return text.encode("utf-8", errors="strict")
    except (TypeError, UnicodeError, ValueError) as error:
        raise PublicationCodecError(
            "value cannot be encoded as canonical JSON"
        ) from error


def _canonical_content(
    value: RootModel,
    *,
    model: type[object],
    document_type: str,
    maximum_bytes: int,
) -> CanonicalJsonDocument:
    payload = _encode_model(value, model, path="payload")
    rebuilt = _decode_model(payload, model, path="payload")
    if rebuilt != value:
        raise PublicationCodecError(
            "publication object changes under closed-schema copy"
        )
    content = _json_bytes(
        {
            "codec_version": PUBLICATION_JSON_CODEC_VERSION,
            "document_type": document_type,
            "payload": payload,
        }
    )
    if len(content) > maximum_bytes:
        raise PublicationCodecError("canonical content exceeds its byte bound")
    _check_lexical_bounds(content)
    return CanonicalJsonDocument(
        content=content,
        content_sha256=compute_content_sha256(content),
    )


def _decode_content(
    content: bytes,
    *,
    expected_content_sha256: str,
    model: type[RootModelT],
    document_type: str,
    maximum_bytes: int,
) -> RootModelT:
    if type(content) is not bytes:
        raise PublicationCodecError("content must be exact bytes")
    if len(content) > maximum_bytes:
        raise PublicationCodecError("content exceeds its byte bound")
    expected_sha256 = _validate_sha256(
        expected_content_sha256,
        field="expected_content_sha256",
    )
    if compute_content_sha256(content) != expected_sha256:
        raise PublicationCodecError("content SHA-256 does not match")
    _check_lexical_bounds(content)
    parsed = _parse_json(content)
    if type(parsed) is not dict:
        raise PublicationCodecError("canonical document must be a JSON object")
    envelope = cast(dict[str, object], parsed)
    expected_envelope_keys = {"codec_version", "document_type", "payload"}
    missing = expected_envelope_keys - set(envelope)
    unknown = set(envelope) - expected_envelope_keys
    if missing:
        raise PublicationCodecError(
            f"document envelope is missing keys: {','.join(sorted(missing))}"
        )
    if unknown:
        raise PublicationCodecError(
            f"document envelope has unknown keys: {','.join(sorted(unknown))}"
        )
    if envelope["codec_version"] != PUBLICATION_JSON_CODEC_VERSION:
        raise PublicationCodecError("publication JSON codec version is invalid")
    if envelope["document_type"] != document_type:
        raise PublicationCodecError("publication document type is invalid")
    decoded = _decode_model(envelope["payload"], model, path="payload")
    if type(decoded) is not model:
        raise PublicationCodecError("decoded publication root has an invalid type")
    result = decoded
    canonical = _canonical_content(
        result,
        model=model,
        document_type=document_type,
        maximum_bytes=maximum_bytes,
    )
    if canonical.content != content:
        raise PublicationCodecError("content is valid JSON but not canonical")
    return result


def encode_statistical_report(
    report: StatisticalReport,
    *,
    config: ExperimentConfig,
) -> CanonicalJsonDocument:
    """Return canonical content for one report bound to its full config."""

    document = _canonical_content(
        report,
        model=StatisticalReport,
        document_type=STATISTICAL_REPORT_DOCUMENT_TYPE,
        maximum_bytes=_MAX_REPORT_BYTES,
    )
    _validate_report_context(report, config)
    return document


def decode_statistical_report(
    content: bytes,
    *,
    expected_content_sha256: str,
    config: ExperimentConfig,
) -> StatisticalReport:
    """Verify and decode one report against its externally bound config."""

    report = _decode_content(
        content,
        expected_content_sha256=expected_content_sha256,
        model=StatisticalReport,
        document_type=STATISTICAL_REPORT_DOCUMENT_TYPE,
        maximum_bytes=_MAX_REPORT_BYTES,
    )
    _validate_report_context(report, config)
    return report


def encode_publication_evidence(
    evidence: PublicationEvidence,
) -> CanonicalJsonDocument:
    """Return canonical content and SHA-256 for exact publication evidence."""

    if type(evidence) is not PublicationEvidence:
        raise PublicationCodecError("evidence must be an exact PublicationEvidence")
    document = _canonical_content(
        evidence,
        model=PublicationEvidence,
        document_type=PUBLICATION_EVIDENCE_DOCUMENT_TYPE,
        maximum_bytes=_MAX_EVIDENCE_BYTES,
    )
    _validate_report_context(evidence.statistics, evidence.config)
    return document


def decode_publication_evidence(
    content: bytes,
    *,
    expected_content_sha256: str,
) -> PublicationEvidence:
    """Verify and decode one canonical publication-evidence document."""

    evidence = _decode_content(
        content,
        expected_content_sha256=expected_content_sha256,
        model=PublicationEvidence,
        document_type=PUBLICATION_EVIDENCE_DOCUMENT_TYPE,
        maximum_bytes=_MAX_EVIDENCE_BYTES,
    )
    _validate_report_context(evidence.statistics, evidence.config)
    return evidence
