"""Deterministic streaming storage for :class:`ExperimentResult`.

The closed binary layout is ``evaluation-result-binary-v1``::

    canonical header
    result metadata
    experiment configuration
    cluster count, abrupt-trace count
    canonical cluster summaries
      summary identity
      TrajectoryMetrics fields in their declared schema order
    trailer magic, SHA-256(header + payload)

Unsigned integers are big-endian, floats are big-endian IEEE-754 binary64,
booleans are one byte, and text is length-prefixed canonical UTF-8. Optional
values have a one-byte absence/presence tag. Variable-length vectors carry a
32-bit count which is checked against the configuration before allocation.

``TraceRecord.common_expected_regret`` is deliberately not written a second
time. A valid trace is identical to its matching summary's
``TrajectoryMetrics.common_regret_trace``; the decoder reconstructs exact
``TraceRecord`` objects from that tuple after verifying the payload checksum.

The codec uses bounded chunks for long float vectors. It therefore never
constructs a giant JSON document or a second full-result byte string.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import fields
from typing import Any, BinaryIO, Final, Literal, cast, get_type_hints

from .evaluation import (
    CALIBRATION_EDGES,
    LOCKED_POLICY_ID,
    MAX_HORIZON,
    MAX_SEED,
    RESULT_SCHEMA_VERSION,
    AvailabilityMode,
    ClusterSummary,
    EvaluationInputError,
    EvaluationInvariantError,
    ExperimentConfig,
    ExperimentResult,
    Persona,
    Strategy,
    TraceRecord,
    TrajectoryMetrics,
    evaluator_design_fingerprint,
    evaluator_fingerprint,
    validate_experiment_config,
    validate_experiment_result,
)
from .policy import DEFAULT_TEMPLATES, EvidenceBucket

CODEC_VERSION: Final = "evaluation-result-binary-v1"

_CANONICAL_HEADER: Final = (
    b"GWORKER-EVALUATION-RESULT\x00" + CODEC_VERSION.encode("ascii") + b"\x00"
)
_TRAILER_MAGIC: Final = b"GWORKER-SHA256\x00"
_SHA256_SIZE: Final = hashlib.sha256().digest_size
_TRAILER_SIZE: Final = len(_TRAILER_MAGIC) + _SHA256_SIZE

_FLOAT_CHUNK_VALUES: Final = 4_096
_MAX_TEXT_BYTES: Final = 4_096
_MAX_ID_BYTES: Final = 256
_MAX_PERSONAS: Final = 256
# The locked run needs 16,128 summaries and 13,934,592 stored trace floats.
# These limits leave narrow format headroom without permitting an attacker to
# request allocations on the scale of the evaluator's theoretical maxima.
_MAX_CLUSTER_SUMMARIES: Final = 20_000
_MAX_TOTAL_TRACE_VALUES: Final = 15_000_000
_EVIDENCE_BUCKET_WIDTH: Final = 3
_TEMPLATE_WIDTH: Final = 4
_CALIBRATION_WIDTH: Final = 8

_U8 = struct.Struct(">B")
_U16 = struct.Struct(">H")
_U32 = struct.Struct(">I")
_U64 = struct.Struct(">Q")
_F64 = struct.Struct(">d")

MetricKind = Literal[
    "float",
    "optional-float",
    "int",
    "optional-int",
    "float-vector",
    "int-vector",
]

# This declaration is the on-disk TrajectoryMetrics schema. An import-time
# assertion below makes a source-level field addition fail closed until the
# codec version and this declaration are intentionally revised.
_METRIC_SCHEMA: Final[tuple[tuple[str, MetricKind], ...]] = (
    ("mean_common_expected_regret", "float"),
    ("mean_conditional_expected_regret", "float"),
    ("mean_path_opportunity_cost", "float"),
    ("mean_expected_reward", "float"),
    ("mean_realized_reward", "float"),
    ("right_fit_rate", "float"),
    ("completion_rate", "float"),
    ("mean_common_oracle_distance", "float"),
    ("mean_conditional_oracle_distance", "float"),
    ("cumulative_common_expected_regret", "float"),
    ("cumulative_conditional_expected_regret", "float"),
    ("cumulative_path_opportunity_cost", "float"),
    ("availability_guardrail_rate", "float"),
    ("one_step_guardrail_rate", "float"),
    ("availability_override_rate", "float"),
    ("mean_feasible_set_size", "float"),
    ("maximum_arm_transition", "int"),
    ("review_rate", "float"),
    ("minimum_propensity", "optional-float"),
    ("maximum_inverse_propensity", "optional-float"),
    ("low_propensity_rate", "optional-float"),
    ("inverse_propensity_ess_ratio", "optional-float"),
    ("multiclass_brier_score", "optional-float"),
    ("exact_bucket_rate", "optional-float"),
    ("task_bucket_rate", "optional-float"),
    ("global_bucket_rate", "optional-float"),
    ("probability_floor_rate", "optional-float"),
    ("action_count", "int"),
    ("availability_guardrail_count", "int"),
    ("one_step_guardrail_count", "int"),
    ("availability_override_count", "int"),
    ("feasible_set_size_sum", "float"),
    ("review_count", "int"),
    ("selected_propensity_count", "optional-int"),
    ("low_propensity_count", "optional-int"),
    ("inverse_propensity_sum", "optional-float"),
    ("inverse_propensity_squared_sum", "optional-float"),
    ("brier_score_sum", "optional-float"),
    ("brier_score_count", "optional-int"),
    ("evidence_bucket_counts", "int-vector"),
    ("probability_floor_count", "optional-int"),
    ("arm_probability_count", "optional-int"),
    ("template_exposures", "int-vector"),
    ("calibration_predicted_sums", "float-vector"),
    ("calibration_observed_sums", "float-vector"),
    ("calibration_counts", "int-vector"),
    ("pre_drift_regret", "optional-float"),
    ("early_post_drift_auc", "optional-float"),
    ("late_regret", "optional-float"),
    ("recovery_lag", "optional-float"),
    ("recovery_rate", "optional-float"),
    ("conservative_recovery_lag", "optional-float"),
    ("recovered_count", "optional-int"),
    ("recovered_lag_sum", "optional-float"),
    ("common_regret_trace", "float-vector"),
    ("conditional_regret_trace", "float-vector"),
    ("path_opportunity_cost_trace", "float-vector"),
)

_METRIC_KIND_TYPES: Final[dict[MetricKind, object]] = {
    "float": float,
    "optional-float": float | None,
    "int": int,
    "optional-int": int | None,
    "float-vector": tuple[float, ...],
    "int-vector": tuple[int, ...],
}
_VECTOR_LENGTHS: Final[dict[str, int | None]] = {
    "evidence_bucket_counts": _EVIDENCE_BUCKET_WIDTH,
    "template_exposures": _TEMPLATE_WIDTH,
    "calibration_predicted_sums": _CALIBRATION_WIDTH,
    "calibration_observed_sums": _CALIBRATION_WIDTH,
    "calibration_counts": _CALIBRATION_WIDTH,
    "common_regret_trace": None,
    "conditional_regret_trace": None,
    "path_opportunity_cost_trace": None,
}
_CLOSED_DATACLASS_SCHEMAS: Final[
    tuple[tuple[type[object], tuple[tuple[str, object], ...]], ...]
] = (
    (
        Persona,
        (
            ("persona_id", str),
            ("label", str),
            ("primary", bool),
            ("sigma_minutes", float),
            ("selective_reviews", bool),
        ),
    ),
    (
        ExperimentConfig,
        (
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
    ),
    (
        ClusterSummary,
        (
            ("persona_id", str),
            ("persona_primary", bool),
            ("availability_mode", AvailabilityMode),
            ("environment_seed", int),
            ("strategy", Strategy),
            ("policy_replica_count", int),
            ("metrics", TrajectoryMetrics),
        ),
    ),
    (
        TraceRecord,
        (
            ("persona_id", str),
            ("availability_mode", AvailabilityMode),
            ("environment_seed", int),
            ("strategy", Strategy),
            ("common_expected_regret", tuple[float, ...]),
        ),
    ),
    (
        ExperimentResult,
        (
            ("schema_version", str),
            ("config", ExperimentConfig),
            ("evaluator_id", str),
            ("design_id", str),
            ("policy_id", str),
            ("cluster_summaries", tuple[ClusterSummary, ...]),
            ("abrupt_traces", tuple[TraceRecord, ...]),
            ("hard_failure_count", int),
        ),
    ),
)
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
_METRIC_TYPE_HINTS: Final[dict[str, object]] = cast(
    dict[str, object],
    get_type_hints(TrajectoryMetrics),
)


def _resolved_schema(model: type[object]) -> tuple[tuple[str, object], ...]:
    hints = cast(dict[str, object], get_type_hints(model))
    return tuple((field.name, hints[field.name]) for field in fields(cast(Any, model)))


if (
    tuple(name for name, _kind in _METRIC_SCHEMA)
    != tuple(field.name for field in fields(TrajectoryMetrics))
    or any(
        _METRIC_TYPE_HINTS.get(name) != _METRIC_KIND_TYPES[kind]
        for name, kind in _METRIC_SCHEMA
    )
    or {name for name, kind in _METRIC_SCHEMA if kind in ("float-vector", "int-vector")}
    != set(_VECTOR_LENGTHS)
    or any(
        _resolved_schema(model) != expected
        for model, expected in _CLOSED_DATACLASS_SCHEMAS
    )
    or tuple((member.name, member.value) for member in AvailabilityMode)
    != _AVAILABILITY_MODE_SCHEMA
    or tuple((member.name, member.value) for member in Strategy) != _STRATEGY_SCHEMA
    or len(EvidenceBucket) != _EVIDENCE_BUCKET_WIDTH
    or len(DEFAULT_TEMPLATES) != _TEMPLATE_WIDTH
    or len(CALIBRATION_EDGES) - 1 != _CALIBRATION_WIDTH
):
    raise RuntimeError(
        "evaluation schema changed without a result codec version update"
    )


class ResultCodecError(ValueError):
    """Raised when a result cannot be encoded or pass integrity checks."""


class _HashingWriter:
    __slots__ = ("_digest", "_stream")

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()

    def write(self, payload: bytes) -> None:
        try:
            written = self._stream.write(payload)
        except (OSError, ValueError, TypeError) as error:
            raise ResultCodecError("result destination rejected a write") from error
        if type(written) is not int or written != len(payload):
            raise ResultCodecError("result destination performed a short write")
        self._digest.update(payload)

    def content_sha256(self) -> str:
        return self._digest.hexdigest()

    def content_digest(self) -> bytes:
        return self._digest.digest()


class _HashingReader:
    __slots__ = ("_digest", "_stream")

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()

    def read_exact(self, size: int, *, field: str) -> bytes:
        if size < 0:
            raise ResultCodecError(f"{field} has an invalid byte length")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            try:
                chunk = self._stream.read(remaining)
            except (OSError, ValueError, TypeError) as error:
                raise ResultCodecError("result source rejected a read") from error
            if not isinstance(chunk, bytes):
                raise ResultCodecError("result source must be a binary stream")
            if not chunk:
                raise ResultCodecError(f"truncated result while reading {field}")
            if len(chunk) > remaining:
                raise ResultCodecError(
                    "result source returned more bytes than requested"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        self._digest.update(payload)
        return payload

    def content_digest(self) -> bytes:
        return self._digest.digest()


def _write_direct(stream: BinaryIO, payload: bytes, *, field: str) -> None:
    try:
        written = stream.write(payload)
    except (OSError, ValueError, TypeError) as error:
        raise ResultCodecError(f"result destination rejected {field}") from error
    if type(written) is not int or written != len(payload):
        raise ResultCodecError(f"result destination short-wrote {field}")


def _read_direct(stream: BinaryIO, size: int, *, field: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = stream.read(remaining)
        except (OSError, ValueError, TypeError) as error:
            raise ResultCodecError(f"result source rejected {field}") from error
        if not isinstance(chunk, bytes):
            raise ResultCodecError("result source must be a binary stream")
        if not chunk:
            raise ResultCodecError(f"truncated result while reading {field}")
        if len(chunk) > remaining:
            raise ResultCodecError("result source returned more bytes than requested")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _canonical_float(value: object, *, field: str) -> float:
    if type(value) is not float:
        raise ResultCodecError(f"{field} must be an exact float")
    result = value
    if not math.isfinite(result):
        raise ResultCodecError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result


def _bounded_int(
    value: object,
    *,
    field: str,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise ResultCodecError(f"{field} must be an exact integer")
    result = value
    if not 0 <= result <= maximum:
        raise ResultCodecError(f"{field} is outside its binary range")
    return result


def _write_u8(writer: _HashingWriter, value: object, *, field: str) -> None:
    writer.write(_U8.pack(_bounded_int(value, field=field, maximum=2**8 - 1)))


def _write_u16(writer: _HashingWriter, value: object, *, field: str) -> None:
    writer.write(_U16.pack(_bounded_int(value, field=field, maximum=2**16 - 1)))


def _write_u32(writer: _HashingWriter, value: object, *, field: str) -> None:
    writer.write(_U32.pack(_bounded_int(value, field=field, maximum=2**32 - 1)))


def _write_u64(writer: _HashingWriter, value: object, *, field: str) -> None:
    writer.write(_U64.pack(_bounded_int(value, field=field, maximum=2**64 - 1)))


def _write_float(writer: _HashingWriter, value: object, *, field: str) -> None:
    writer.write(_F64.pack(_canonical_float(value, field=field)))


def _write_bool(writer: _HashingWriter, value: object, *, field: str) -> None:
    if type(value) is not bool:
        raise ResultCodecError(f"{field} must be an exact boolean")
    writer.write(b"\x01" if value else b"\x00")


def _write_text(
    writer: _HashingWriter,
    value: object,
    *,
    field: str,
    maximum: int,
) -> None:
    encoded = _encoded_text(value, field=field, maximum=maximum)
    _write_u16(writer, len(encoded), field=f"{field} byte length")
    writer.write(encoded)


def _encoded_text(value: object, *, field: str, maximum: int) -> bytes:
    if type(value) is not str:
        raise ResultCodecError(f"{field} must be exact text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ResultCodecError(f"{field} is not canonical UTF-8 text") from error
    if not encoded or len(encoded) > maximum:
        raise ResultCodecError(f"{field} has an invalid encoded length")
    return encoded


def _write_optional_float(
    writer: _HashingWriter,
    value: object,
    *,
    field: str,
) -> None:
    if value is None:
        writer.write(b"\x00")
        return
    writer.write(b"\x01")
    _write_float(writer, value, field=field)


def _write_optional_int(
    writer: _HashingWriter,
    value: object,
    *,
    field: str,
) -> None:
    if value is None:
        writer.write(b"\x00")
        return
    writer.write(b"\x01")
    _write_u64(writer, value, field=field)


def _write_int_vector(
    writer: _HashingWriter,
    value: object,
    *,
    field: str,
    expected_length: int,
) -> None:
    if type(value) is not tuple:
        raise ResultCodecError(f"{field} must be an exact tuple")
    vector = cast(tuple[object, ...], value)
    if len(vector) != expected_length:
        raise ResultCodecError(f"{field} has a noncanonical length")
    _write_u32(writer, len(vector), field=f"{field} length")
    for item in vector:
        _write_u64(writer, item, field=f"{field} value")


def _write_float_vector(
    writer: _HashingWriter,
    value: object,
    *,
    field: str,
    expected_length: int,
) -> None:
    if type(value) is not tuple:
        raise ResultCodecError(f"{field} must be an exact tuple")
    vector = cast(tuple[object, ...], value)
    if len(vector) != expected_length:
        raise ResultCodecError(f"{field} has a noncanonical length")
    _write_u32(writer, len(vector), field=f"{field} length")
    for offset in range(0, len(vector), _FLOAT_CHUNK_VALUES):
        chunk = tuple(
            _canonical_float(item, field=f"{field} value")
            for item in vector[offset : offset + _FLOAT_CHUNK_VALUES]
        )
        writer.write(struct.pack(f">{len(chunk)}d", *chunk))


def _read_u8(reader: _HashingReader, *, field: str) -> int:
    return cast(int, _U8.unpack(reader.read_exact(_U8.size, field=field))[0])


def _read_u16(reader: _HashingReader, *, field: str) -> int:
    return cast(int, _U16.unpack(reader.read_exact(_U16.size, field=field))[0])


def _read_u32(reader: _HashingReader, *, field: str) -> int:
    return cast(int, _U32.unpack(reader.read_exact(_U32.size, field=field))[0])


def _read_u64(reader: _HashingReader, *, field: str) -> int:
    return cast(int, _U64.unpack(reader.read_exact(_U64.size, field=field))[0])


def _read_float(reader: _HashingReader, *, field: str) -> float:
    value = cast(float, _F64.unpack(reader.read_exact(_F64.size, field=field))[0])
    if not math.isfinite(value):
        raise ResultCodecError(f"{field} must be finite")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ResultCodecError(f"{field} contains noncanonical negative zero")
    return value


def _read_bool(reader: _HashingReader, *, field: str) -> bool:
    value = _read_u8(reader, field=field)
    if value not in (0, 1):
        raise ResultCodecError(f"{field} has a noncanonical boolean")
    return bool(value)


def _read_text(
    reader: _HashingReader,
    *,
    field: str,
    maximum: int,
) -> str:
    size = _read_u16(reader, field=f"{field} byte length")
    if size == 0 or size > maximum:
        raise ResultCodecError(f"{field} has an invalid encoded length")
    payload = reader.read_exact(size, field=field)
    try:
        value = payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ResultCodecError(f"{field} is not canonical UTF-8 text") from error
    if value.encode("utf-8") != payload:
        raise ResultCodecError(f"{field} has a noncanonical UTF-8 encoding")
    return value


def _read_optional_float(
    reader: _HashingReader,
    *,
    field: str,
) -> float | None:
    tag = _read_u8(reader, field=f"{field} presence")
    if tag == 0:
        return None
    if tag != 1:
        raise ResultCodecError(f"{field} has a noncanonical presence tag")
    return _read_float(reader, field=field)


def _read_optional_int(
    reader: _HashingReader,
    *,
    field: str,
) -> int | None:
    tag = _read_u8(reader, field=f"{field} presence")
    if tag == 0:
        return None
    if tag != 1:
        raise ResultCodecError(f"{field} has a noncanonical presence tag")
    return _read_u64(reader, field=field)


def _read_int_vector(
    reader: _HashingReader,
    *,
    field: str,
    expected_length: int,
) -> tuple[int, ...]:
    count = _read_u32(reader, field=f"{field} length")
    if count != expected_length:
        raise ResultCodecError(f"{field} has a noncanonical length")
    return tuple(_read_u64(reader, field=f"{field} value") for _ in range(count))


def _read_float_vector(
    reader: _HashingReader,
    *,
    field: str,
    expected_length: int,
) -> tuple[float, ...]:
    count = _read_u32(reader, field=f"{field} length")
    if count != expected_length:
        raise ResultCodecError(f"{field} has a noncanonical length")
    result: list[float] = []
    remaining = count
    while remaining:
        chunk_size = min(remaining, _FLOAT_CHUNK_VALUES)
        payload = reader.read_exact(
            chunk_size * _F64.size,
            field=f"{field} values",
        )
        values = struct.unpack(f">{chunk_size}d", payload)
        for value in values:
            if not math.isfinite(value):
                raise ResultCodecError(f"{field} must contain finite values")
            if value == 0.0 and math.copysign(1.0, value) < 0.0:
                raise ResultCodecError(f"{field} contains noncanonical negative zero")
        result.extend(values)
        remaining -= chunk_size
    return tuple(result)


def _vector_length(field: str, *, horizon: int) -> int:
    try:
        declared = _VECTOR_LENGTHS[field]
    except KeyError as error:
        raise RuntimeError(f"missing vector bound for {field}") from error
    return horizon if declared is None else declared


def _write_metrics(
    writer: _HashingWriter,
    metrics: TrajectoryMetrics,
    *,
    horizon: int,
) -> None:
    if type(metrics) is not TrajectoryMetrics:
        raise ResultCodecError("metrics must be an exact TrajectoryMetrics")
    for field, kind in _METRIC_SCHEMA:
        value = getattr(metrics, field)
        if kind == "float":
            _write_float(writer, value, field=field)
        elif kind == "optional-float":
            _write_optional_float(writer, value, field=field)
        elif kind == "int":
            _write_u64(writer, value, field=field)
        elif kind == "optional-int":
            _write_optional_int(writer, value, field=field)
        elif kind == "float-vector":
            _write_float_vector(
                writer,
                value,
                field=field,
                expected_length=_vector_length(field, horizon=horizon),
            )
        elif kind == "int-vector":
            _write_int_vector(
                writer,
                value,
                field=field,
                expected_length=_vector_length(field, horizon=horizon),
            )
        else:  # pragma: no cover - guarded by MetricKind and import assertion
            raise RuntimeError(f"unsupported metric field kind: {kind}")


def _read_metrics(
    reader: _HashingReader,
    *,
    horizon: int,
) -> TrajectoryMetrics:
    values: dict[str, object] = {}
    for field, kind in _METRIC_SCHEMA:
        if kind == "float":
            value: object = _read_float(reader, field=field)
        elif kind == "optional-float":
            value = _read_optional_float(reader, field=field)
        elif kind == "int":
            value = _read_u64(reader, field=field)
        elif kind == "optional-int":
            value = _read_optional_int(reader, field=field)
        elif kind == "float-vector":
            value = _read_float_vector(
                reader,
                field=field,
                expected_length=_vector_length(field, horizon=horizon),
            )
        elif kind == "int-vector":
            value = _read_int_vector(
                reader,
                field=field,
                expected_length=_vector_length(field, horizon=horizon),
            )
        else:  # pragma: no cover - guarded by MetricKind and import assertion
            raise RuntimeError(f"unsupported metric field kind: {kind}")
        values[field] = value
    return TrajectoryMetrics(**values)  # type: ignore[arg-type]


def _write_config(writer: _HashingWriter, config: ExperimentConfig) -> None:
    if type(config) is not ExperimentConfig:
        raise ResultCodecError("config must be an exact ExperimentConfig")
    _write_text(
        writer,
        config.split,
        field="config split",
        maximum=_MAX_ID_BYTES,
    )
    _write_u32(
        writer,
        len(config.environment_seeds),
        field="environment seed count",
    )
    for seed in config.environment_seeds:
        _write_u32(writer, seed, field="environment seed")
    _write_u16(writer, config.policy_replicas, field="policy replicas")
    if len(config.personas) > _MAX_PERSONAS:
        raise ResultCodecError("persona count exceeds the codec bound")
    _write_u16(writer, len(config.personas), field="persona count")
    for persona in config.personas:
        if type(persona) is not Persona:
            raise ResultCodecError("config contains a non-exact Persona")
        _write_text(
            writer,
            persona.persona_id,
            field="persona id",
            maximum=_MAX_ID_BYTES,
        )
        _write_text(
            writer,
            persona.label,
            field="persona label",
            maximum=_MAX_TEXT_BYTES,
        )
        _write_bool(writer, persona.primary, field="persona primary")
        _write_float(writer, persona.sigma_minutes, field="persona sigma")
        _write_bool(
            writer,
            persona.selective_reviews,
            field="persona selective reviews",
        )
    _write_u8(
        writer,
        len(config.availability_modes),
        field="availability mode count",
    )
    for mode in config.availability_modes:
        if type(mode) is not AvailabilityMode:
            raise ResultCodecError("config contains an invalid availability mode")
        _write_text(
            writer,
            mode.value,
            field="availability mode",
            maximum=_MAX_ID_BYTES,
        )
    _write_u32(writer, config.horizon, field="horizon")
    _write_u32(writer, config.drift_decision, field="drift decision")
    _write_u32(writer, config.recovery_block_size, field="recovery block size")
    _write_u32(writer, config.recovery_blocks, field="recovery blocks")


def _read_config(reader: _HashingReader) -> ExperimentConfig:
    split = _read_text(reader, field="config split", maximum=_MAX_ID_BYTES)
    seed_count = _read_u32(reader, field="environment seed count")
    if not 1 <= seed_count <= 10_000:
        raise ResultCodecError("environment seed count exceeds the codec bound")
    seeds = tuple(
        _read_u32(reader, field="environment seed") for _ in range(seed_count)
    )
    if any(seed > MAX_SEED for seed in seeds):
        raise ResultCodecError("environment seed exceeds the evaluator bound")
    policy_replicas = _read_u16(reader, field="policy replicas")
    persona_count = _read_u16(reader, field="persona count")
    if not 1 <= persona_count <= _MAX_PERSONAS:
        raise ResultCodecError("persona count exceeds the codec bound")
    personas = tuple(
        Persona(
            persona_id=_read_text(
                reader,
                field="persona id",
                maximum=_MAX_ID_BYTES,
            ),
            label=_read_text(
                reader,
                field="persona label",
                maximum=_MAX_TEXT_BYTES,
            ),
            primary=_read_bool(reader, field="persona primary"),
            sigma_minutes=_read_float(reader, field="persona sigma"),
            selective_reviews=_read_bool(
                reader,
                field="persona selective reviews",
            ),
        )
        for _ in range(persona_count)
    )
    mode_count = _read_u8(reader, field="availability mode count")
    if not 1 <= mode_count <= len(AvailabilityMode):
        raise ResultCodecError("availability mode count exceeds the codec bound")
    modes = tuple(
        AvailabilityMode(
            _read_text(
                reader,
                field="availability mode",
                maximum=_MAX_ID_BYTES,
            )
        )
        for _ in range(mode_count)
    )
    horizon = _read_u32(reader, field="horizon")
    if horizon > MAX_HORIZON:
        raise ResultCodecError("horizon exceeds the evaluator bound")
    return ExperimentConfig(
        split=split,
        environment_seeds=seeds,
        policy_replicas=policy_replicas,
        personas=personas,
        availability_modes=modes,
        horizon=horizon,
        drift_decision=_read_u32(reader, field="drift decision"),
        recovery_block_size=_read_u32(reader, field="recovery block size"),
        recovery_blocks=_read_u32(reader, field="recovery blocks"),
    )


def _canonical_summaries(
    result: ExperimentResult,
) -> tuple[ClusterSummary, ...]:
    config = result.config
    persona_rank = {
        persona.persona_id: index for index, persona in enumerate(config.personas)
    }
    mode_rank = {mode: index for index, mode in enumerate(config.availability_modes)}
    seed_rank = {
        environment_seed: index
        for index, environment_seed in enumerate(config.environment_seeds)
    }
    strategy_rank = {strategy: index for index, strategy in enumerate(Strategy)}

    def ordinal(summary: ClusterSummary) -> tuple[int, int, int, int]:
        try:
            return (
                persona_rank[summary.persona_id],
                mode_rank[summary.availability_mode],
                seed_rank[summary.environment_seed],
                strategy_rank[summary.strategy],
            )
        except KeyError as error:  # pragma: no cover - preflight validation owns this
            raise ResultCodecError(
                "summary identity falls outside the declared Cartesian set"
            ) from error

    return tuple(sorted(result.cluster_summaries, key=ordinal))


def _write_summary(
    writer: _HashingWriter,
    summary: ClusterSummary,
    *,
    horizon: int,
) -> None:
    if type(summary) is not ClusterSummary:
        raise ResultCodecError("summary must be an exact ClusterSummary")
    _write_text(
        writer,
        summary.persona_id,
        field="summary persona id",
        maximum=_MAX_ID_BYTES,
    )
    _write_bool(writer, summary.persona_primary, field="summary persona primary")
    if type(summary.availability_mode) is not AvailabilityMode:
        raise ResultCodecError("summary availability mode has an invalid type")
    _write_text(
        writer,
        summary.availability_mode.value,
        field="summary availability mode",
        maximum=_MAX_ID_BYTES,
    )
    _write_u32(writer, summary.environment_seed, field="summary environment seed")
    if type(summary.strategy) is not Strategy:
        raise ResultCodecError("summary strategy has an invalid type")
    _write_text(
        writer,
        summary.strategy.value,
        field="summary strategy",
        maximum=_MAX_ID_BYTES,
    )
    _write_u16(
        writer,
        summary.policy_replica_count,
        field="summary policy replica count",
    )
    _write_metrics(writer, summary.metrics, horizon=horizon)


def _read_summary(
    reader: _HashingReader,
    *,
    horizon: int,
) -> ClusterSummary:
    return ClusterSummary(
        persona_id=_read_text(
            reader,
            field="summary persona id",
            maximum=_MAX_ID_BYTES,
        ),
        persona_primary=_read_bool(reader, field="summary persona primary"),
        availability_mode=AvailabilityMode(
            _read_text(
                reader,
                field="summary availability mode",
                maximum=_MAX_ID_BYTES,
            )
        ),
        environment_seed=_read_u32(reader, field="summary environment seed"),
        strategy=Strategy(
            _read_text(
                reader,
                field="summary strategy",
                maximum=_MAX_ID_BYTES,
            )
        ),
        policy_replica_count=_read_u16(
            reader,
            field="summary policy replica count",
        ),
        metrics=_read_metrics(reader, horizon=horizon),
    )


def _expected_cardinalities(config: ExperimentConfig) -> tuple[int, int]:
    cluster_count = (
        len(config.personas)
        * len(config.availability_modes)
        * len(config.environment_seeds)
        * len(Strategy)
    )
    abrupt_count = (
        sum(persona.is_abrupt for persona in config.personas)
        * len(config.availability_modes)
        * len(config.environment_seeds)
        * len(Strategy)
    )
    if cluster_count > _MAX_CLUSTER_SUMMARIES:
        raise ResultCodecError("cluster cardinality exceeds the codec bound")
    trace_values = cluster_count * config.horizon * 3
    if trace_values > _MAX_TOTAL_TRACE_VALUES:
        raise ResultCodecError("trace cardinality exceeds the codec bound")
    return cluster_count, abrupt_count


def _preflight_metrics(metrics: object, *, horizon: int) -> None:
    if type(metrics) is not TrajectoryMetrics:
        raise ResultCodecError("metrics must be an exact TrajectoryMetrics")
    for field, kind in _METRIC_SCHEMA:
        value = getattr(metrics, field)
        if kind == "float":
            _canonical_float(value, field=field)
        elif kind == "optional-float":
            if value is not None:
                _canonical_float(value, field=field)
        elif kind == "int":
            _bounded_int(value, field=field, maximum=2**64 - 1)
        elif kind == "optional-int":
            if value is not None:
                _bounded_int(value, field=field, maximum=2**64 - 1)
        elif kind in ("float-vector", "int-vector"):
            if type(value) is not tuple:
                raise ResultCodecError(f"{field} must be an exact tuple")
            vector = cast(tuple[object, ...], value)
            if len(vector) != _vector_length(field, horizon=horizon):
                raise ResultCodecError(f"{field} has a noncanonical length")
            if kind == "float-vector":
                for item in vector:
                    _canonical_float(item, field=f"{field} value")
            else:
                for item in vector:
                    _bounded_int(
                        item,
                        field=f"{field} value",
                        maximum=2**64 - 1,
                    )
        else:  # pragma: no cover - guarded by MetricKind and import assertion
            raise RuntimeError(f"unsupported metric field kind: {kind}")


def _preflight_config(config: object) -> ExperimentConfig:
    if type(config) is not ExperimentConfig:
        raise ResultCodecError("config must be an exact ExperimentConfig")
    _encoded_text(
        config.split,
        field="config split",
        maximum=_MAX_ID_BYTES,
    )
    if type(config.environment_seeds) is not tuple:
        raise ResultCodecError("environment seeds must be an exact tuple")
    if not 1 <= len(config.environment_seeds) <= 10_000:
        raise ResultCodecError("environment seed count exceeds the codec bound")
    _bounded_int(
        len(config.environment_seeds),
        field="environment seed count",
        maximum=2**32 - 1,
    )
    for seed in config.environment_seeds:
        _bounded_int(seed, field="environment seed", maximum=MAX_SEED)
    _bounded_int(
        config.policy_replicas,
        field="policy replicas",
        maximum=2**16 - 1,
    )
    if type(config.personas) is not tuple:
        raise ResultCodecError("personas must be an exact tuple")
    if not 1 <= len(config.personas) <= _MAX_PERSONAS:
        raise ResultCodecError("persona count exceeds the codec bound")
    for persona in config.personas:
        if type(persona) is not Persona:
            raise ResultCodecError("config contains a non-exact Persona")
        _encoded_text(
            persona.persona_id,
            field="persona id",
            maximum=_MAX_ID_BYTES,
        )
        _encoded_text(
            persona.label,
            field="persona label",
            maximum=_MAX_TEXT_BYTES,
        )
        if type(persona.primary) is not bool:
            raise ResultCodecError("persona primary must be an exact boolean")
        _canonical_float(persona.sigma_minutes, field="persona sigma")
        if type(persona.selective_reviews) is not bool:
            raise ResultCodecError("persona selective reviews must be an exact boolean")
    if type(config.availability_modes) is not tuple:
        raise ResultCodecError("availability modes must be an exact tuple")
    if not 1 <= len(config.availability_modes) <= len(AvailabilityMode):
        raise ResultCodecError("availability mode count exceeds the codec bound")
    _bounded_int(
        len(config.availability_modes),
        field="availability mode count",
        maximum=2**8 - 1,
    )
    for mode in config.availability_modes:
        if type(mode) is not AvailabilityMode:
            raise ResultCodecError("config contains an invalid availability mode")
        _encoded_text(
            mode.value,
            field="availability mode",
            maximum=_MAX_ID_BYTES,
        )
    for field in (
        "horizon",
        "drift_decision",
        "recovery_block_size",
        "recovery_blocks",
    ):
        _bounded_int(
            getattr(config, field),
            field=field.replace("_", " "),
            maximum=2**32 - 1,
        )
    try:
        validate_experiment_config(config)
    except EvaluationInputError as error:
        raise ResultCodecError("config failed closed validation") from error
    return config


def _preflight_result_layout(result: object) -> ExperimentResult:
    if type(result) is not ExperimentResult:
        raise ResultCodecError("result must be an exact ExperimentResult")
    _encoded_text(
        result.schema_version,
        field="result schema version",
        maximum=_MAX_ID_BYTES,
    )
    _encoded_text(
        result.evaluator_id,
        field="result evaluator id",
        maximum=_MAX_ID_BYTES,
    )
    _encoded_text(
        result.design_id,
        field="result design id",
        maximum=_MAX_ID_BYTES,
    )
    _encoded_text(
        result.policy_id,
        field="result policy id",
        maximum=_MAX_ID_BYTES,
    )
    _bounded_int(
        result.hard_failure_count,
        field="hard failure count",
        maximum=2**32 - 1,
    )
    config = _preflight_config(result.config)
    expected_clusters, expected_abrupt = _expected_cardinalities(config)
    if type(result.cluster_summaries) is not tuple:
        raise ResultCodecError("cluster summaries must be an exact tuple")
    if len(result.cluster_summaries) != expected_clusters:
        raise ResultCodecError("cluster cardinality disagrees with the config")
    for summary in result.cluster_summaries:
        if type(summary) is not ClusterSummary:
            raise ResultCodecError("summary must be an exact ClusterSummary")
        _encoded_text(
            summary.persona_id,
            field="summary persona id",
            maximum=_MAX_ID_BYTES,
        )
        if type(summary.persona_primary) is not bool:
            raise ResultCodecError("summary persona primary must be an exact boolean")
        if type(summary.availability_mode) is not AvailabilityMode:
            raise ResultCodecError("summary availability mode has an invalid type")
        _bounded_int(
            summary.environment_seed,
            field="summary environment seed",
            maximum=MAX_SEED,
        )
        if type(summary.strategy) is not Strategy:
            raise ResultCodecError("summary strategy has an invalid type")
        _bounded_int(
            summary.policy_replica_count,
            field="summary policy replica count",
            maximum=2**16 - 1,
        )
        _preflight_metrics(summary.metrics, horizon=config.horizon)
    if type(result.abrupt_traces) is not tuple:
        raise ResultCodecError("abrupt traces must be an exact tuple")
    if len(result.abrupt_traces) != expected_abrupt:
        raise ResultCodecError("abrupt trace cardinality disagrees with the config")
    for trace in result.abrupt_traces:
        if type(trace) is not TraceRecord:
            raise ResultCodecError("trace must be an exact TraceRecord")
        _encoded_text(
            trace.persona_id,
            field="trace persona id",
            maximum=_MAX_ID_BYTES,
        )
        if type(trace.availability_mode) is not AvailabilityMode:
            raise ResultCodecError("trace availability mode has an invalid type")
        _bounded_int(
            trace.environment_seed,
            field="trace environment seed",
            maximum=MAX_SEED,
        )
        if type(trace.strategy) is not Strategy:
            raise ResultCodecError("trace strategy has an invalid type")
        if type(trace.common_expected_regret) is not tuple:
            raise ResultCodecError("trace values must be an exact tuple")
        if len(trace.common_expected_regret) != config.horizon:
            raise ResultCodecError("trace values have a noncanonical length")
        for value in trace.common_expected_regret:
            _canonical_float(value, field="trace value")
    return result


def _validate_for_write(result: ExperimentResult) -> None:
    _preflight_result_layout(result)
    try:
        validate_experiment_result(result)
    except (
        AttributeError,
        EvaluationInputError,
        EvaluationInvariantError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise ResultCodecError("experiment result failed validation") from error


def write_experiment_result(
    result: ExperimentResult,
    destination: BinaryIO,
) -> str:
    """Stream a validated result and return its content SHA-256 hex digest.

    Validation and bound checks happen before the first byte is written.
    ``destination`` remains open and positioned immediately after the trailer.
    """

    _validate_for_write(result)
    writer = _HashingWriter(destination)
    writer.write(_CANONICAL_HEADER)
    _write_text(
        writer,
        result.schema_version,
        field="result schema version",
        maximum=_MAX_ID_BYTES,
    )
    _write_text(
        writer,
        result.evaluator_id,
        field="result evaluator id",
        maximum=_MAX_ID_BYTES,
    )
    _write_text(
        writer,
        result.design_id,
        field="result design id",
        maximum=_MAX_ID_BYTES,
    )
    _write_text(
        writer,
        result.policy_id,
        field="result policy id",
        maximum=_MAX_ID_BYTES,
    )
    _write_u32(
        writer,
        result.hard_failure_count,
        field="hard failure count",
    )
    _write_config(writer, result.config)
    cluster_count, abrupt_count = _expected_cardinalities(result.config)
    _write_u32(writer, cluster_count, field="cluster count")
    _write_u32(writer, abrupt_count, field="abrupt trace count")
    for summary in _canonical_summaries(result):
        _write_summary(writer, summary, horizon=result.config.horizon)
    content_digest = writer.content_digest()
    _write_direct(destination, _TRAILER_MAGIC, field="checksum trailer magic")
    _write_direct(destination, content_digest, field="checksum trailer digest")
    return writer.content_sha256()


def _read_payload(source: BinaryIO) -> ExperimentResult:
    reader = _HashingReader(source)
    header = reader.read_exact(len(_CANONICAL_HEADER), field="canonical header")
    if header != _CANONICAL_HEADER:
        raise ResultCodecError("result header or codec version is noncanonical")
    schema_version = _read_text(
        reader,
        field="result schema version",
        maximum=_MAX_ID_BYTES,
    )
    evaluator_id = _read_text(
        reader,
        field="result evaluator id",
        maximum=_MAX_ID_BYTES,
    )
    design_id = _read_text(
        reader,
        field="result design id",
        maximum=_MAX_ID_BYTES,
    )
    policy_id = _read_text(
        reader,
        field="result policy id",
        maximum=_MAX_ID_BYTES,
    )
    hard_failure_count = _read_u32(reader, field="hard failure count")
    config = _read_config(reader)

    if schema_version != RESULT_SCHEMA_VERSION:
        raise ResultCodecError("result schema version is invalid")
    if evaluator_id != evaluator_fingerprint(config):
        raise ResultCodecError("result evaluator id disagrees with its config")
    if design_id != evaluator_design_fingerprint():
        raise ResultCodecError("result design id is invalid")
    if policy_id != LOCKED_POLICY_ID:
        raise ResultCodecError("result policy id is invalid")
    if hard_failure_count != 0:
        raise ResultCodecError("result hard failure count is nonzero")

    expected_clusters, expected_abrupt = _expected_cardinalities(config)
    cluster_count = _read_u32(reader, field="cluster count")
    abrupt_count = _read_u32(reader, field="abrupt trace count")
    if cluster_count != expected_clusters:
        raise ResultCodecError("cluster cardinality disagrees with the config")
    if abrupt_count != expected_abrupt:
        raise ResultCodecError("abrupt trace cardinality disagrees with the config")

    expected_keys = (
        (
            persona.persona_id,
            persona.primary,
            mode,
            environment_seed,
            strategy,
        )
        for persona in config.personas
        for mode in config.availability_modes
        for environment_seed in config.environment_seeds
        for strategy in Strategy
    )
    summaries: list[ClusterSummary] = []
    for expected_key in expected_keys:
        summary = _read_summary(reader, horizon=config.horizon)
        actual_key = (
            summary.persona_id,
            summary.persona_primary,
            summary.availability_mode,
            summary.environment_seed,
            summary.strategy,
        )
        if actual_key != expected_key:
            raise ResultCodecError(
                "cluster summary ordinal disagrees with the declared Cartesian order"
            )
        summaries.append(summary)

    content_digest = reader.content_digest()
    trailer_magic = _read_direct(
        source,
        len(_TRAILER_MAGIC),
        field="checksum trailer magic",
    )
    if trailer_magic != _TRAILER_MAGIC:
        raise ResultCodecError("checksum trailer magic is invalid")
    recorded_content_digest = _read_direct(
        source,
        _SHA256_SIZE,
        field="checksum trailer digest",
    )
    try:
        trailing = source.read(1)
    except (OSError, ValueError, TypeError) as error:
        raise ResultCodecError(
            "result source rejected the trailing-data check"
        ) from error
    if not isinstance(trailing, bytes):
        raise ResultCodecError("result source must be a binary stream")
    if trailing:
        raise ResultCodecError("result contains trailing data")
    if recorded_content_digest != content_digest:
        raise ResultCodecError("result payload checksum does not match")

    # Derived objects are constructed only after checksum and exact-EOF checks.
    abrupt_personas = {
        persona.persona_id for persona in config.personas if persona.is_abrupt
    }
    traces = tuple(
        TraceRecord(
            persona_id=summary.persona_id,
            availability_mode=summary.availability_mode,
            environment_seed=summary.environment_seed,
            strategy=summary.strategy,
            common_expected_regret=summary.metrics.common_regret_trace,
        )
        for summary in summaries
        if summary.persona_id in abrupt_personas
    )
    if len(traces) != abrupt_count:
        raise ResultCodecError("reconstructed abrupt trace cardinality is invalid")

    result = ExperimentResult(
        schema_version=schema_version,
        config=config,
        evaluator_id=evaluator_id,
        design_id=design_id,
        policy_id=policy_id,
        cluster_summaries=tuple(summaries),
        abrupt_traces=traces,
        hard_failure_count=hard_failure_count,
    )
    return result


def read_experiment_result(source: BinaryIO) -> ExperimentResult:
    """Verify integrity, decode, reconstruct traces, and validate one result.

    ``source`` must contain exactly one complete container and is left open at
    EOF. Any malformed representation, failed checksum, or semantic invariant
    raises :class:`ResultCodecError`.
    """

    try:
        result = _read_payload(source)
        validate_experiment_result(result)
    except ResultCodecError:
        raise
    except (
        EvaluationInputError,
        EvaluationInvariantError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise ResultCodecError("decoded experiment result failed validation") from error
    return result
