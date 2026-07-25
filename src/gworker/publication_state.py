"""Durable single-use state for the locked publication run.

The state journal is intentionally small and separate from generated
artifacts.  A private run directory contains an advisory ``lock`` file and a
private ``states`` directory.  State documents are created once, in this exact
order::

    000-prepared.json
    001-evaluating.json
    002-evaluated.json
    003-materialized.json
    004-sealed.json

Every document is strict canonical JSON and repeats the complete locked
identity.  It also commits to the previous document's SHA-256 and to the exact
artifact/cardinality inventory permitted at that stage.

``begin_evaluation`` appends and fsyncs ``EVALUATING`` before issuing the
private evaluator permit.  Only that live store instance may append
``EVALUATED``.  Reopening a run whose last durable state is ``EVALUATING`` can
inspect it, but cannot resume or replace the computation; that run is burned.
Later materialization-only stages may safely continue after a process restart.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
from contextlib import suppress
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Final, NoReturn, cast

from .evaluation import (
    DEFAULT_EXPERIMENT_CONFIG,
    LOCKED_DESIGN_ID,
    LOCKED_EVALUATION_RUN_KEY,
    LOCKED_EVALUATOR_ID,
    LOCKED_POLICY_ID,
    LOCKED_POPULATION_ID,
    ExperimentConfig,
    _issue_locked_evaluation_permit,
    _LockedEvaluationPermit,
    evaluator_design_fingerprint,
    evaluator_fingerprint,
    population_fingerprint,
    validate_experiment_config,
)
from .evidence import (
    EvidenceCardinalities,
    expected_publication_cardinalities,
)
from .policy import POLICY_ID

STATE_SCHEMA_VERSION: Final = "gworker-publication-state-v2"
LOCK_FILE_NAME: Final = "lock"
STATE_DIRECTORY_NAME: Final = "states"
SOURCE_PROVENANCE_FILE_NAME: Final = "source-provenance.json"

_PLATFORM_PATH_TYPE: Final = type(Path())
_PATH_RAW_COMPONENTS_SLOT: Final = (
    "_raw_paths" if hasattr(_PLATFORM_PATH_TYPE(), "_raw_paths") else "_parts"
)
_MAX_STATE_BYTES: Final = 64 * 1024
_MAX_STATE_CONTAINERS: Final = 256
_MAX_JSON_DEPTH: Final = 16
_MAX_JSON_STRING_BYTES: Final = 4 * 1024
_MAX_ARTIFACTS: Final = 64
_MAX_ARTIFACT_NAME_BYTES: Final = 192
_MAX_ARTIFACT_BYTES: Final = 16 * 1024 * 1024 * 1024
_MAX_CARDINALITY: Final = 2**63 - 1
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_COMPONENT_PATTERN: Final = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?\Z"
)

_RESULT_ARTIFACT_NAME: Final = "result.bin"
_REPORT_ARTIFACT_NAME: Final = "report.json"
_EVIDENCE_ARTIFACT_NAME: Final = "evidence.json"
_MANIFEST_ARTIFACT_NAME: Final = "manifest.json"
_MISSING: Final = object()


class PublicationStateError(RuntimeError):
    """Base class for locked publication state failures."""


class PublicationStateSecurityError(PublicationStateError):
    """Raised when a path, descriptor, owner, or mode is unsafe."""


class PublicationStateCorrupt(PublicationStateError):
    """Raised when the append-only state journal is malformed or tampered."""


class PublicationStateConflict(PublicationStateError):
    """Raised when a caller requests an illegal state transition."""


class PublicationRunBusy(PublicationStateError):
    """Raised when another process holds the run's exclusive lock."""


class PublicationRunExists(PublicationStateError):
    """Raised when initialization encounters an already claimed run."""


class PublicationStateIOError(PublicationStateError):
    """Raised when durable state I/O cannot be completed."""


class PublicationStage(StrEnum):
    """Closed lifecycle for one held-out publication run."""

    PREPARED = "prepared"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    MATERIALIZED = "materialized"
    SEALED = "sealed"


_STAGES: Final = tuple(PublicationStage)
_STATE_FILE_NAMES: Final = tuple(
    f"{sequence:03d}-{stage.value}.json" for sequence, stage in enumerate(_STAGES)
)
_STATE_FILE_NAME_SET: Final = frozenset(_STATE_FILE_NAMES)
_EVIDENCE_CARDINALITY_NAMES: Final = tuple(
    sorted(definition.name for definition in fields(EvidenceCardinalities))
)
_RESULT_CARDINALITY_NAMES: Final = (
    "abrupt_traces",
    "cluster_summaries",
    "hard_failure_count",
)


def _exact_integer(
    value: object,
    *,
    field: str,
    lower: int,
    upper: int,
    error_type: type[PublicationStateError] = PublicationStateConflict,
) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise error_type(f"{field} must be an integer between {lower} and {upper}")
    return value


def _validate_sha256(
    value: object,
    *,
    field: str,
    allow_none: bool = False,
    error_type: type[PublicationStateError] = PublicationStateConflict,
) -> str | None:
    if allow_none and value is None:
        return None
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise error_type(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_artifact_name(
    value: object,
    *,
    error_type: type[PublicationStateError] = PublicationStateConflict,
) -> str:
    if type(value) is not str:
        raise error_type("artifact name must be exact text")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise error_type("artifact name must be ASCII") from error
    if not encoded or len(encoded) > _MAX_ARTIFACT_NAME_BYTES:
        raise error_type("artifact name length is invalid")
    if value.startswith("/") or "\\" in value:
        raise error_type("artifact name must be a safe relative POSIX path")
    components = value.split("/")
    if any(
        component in {"", ".", ".."}
        or _ARTIFACT_COMPONENT_PATTERN.fullmatch(component) is None
        for component in components
    ):
        raise error_type("artifact name has an unsafe path component")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    """Digest and exact byte cardinality for one publication artifact."""

    name: str
    content_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        _validate_artifact_binding(
            self,
            error_type=PublicationStateConflict,
        )


@dataclass(frozen=True, slots=True)
class CardinalityBinding:
    """One named non-negative inventory count."""

    name: str
    value: int

    def __post_init__(self) -> None:
        _validate_cardinality_binding(
            self,
            error_type=PublicationStateConflict,
        )


def _validate_artifact_binding(
    binding: ArtifactBinding,
    *,
    error_type: type[PublicationStateError],
) -> None:
    _validate_artifact_name(
        getattr(binding, "name", _MISSING),
        error_type=error_type,
    )
    _validate_sha256(
        getattr(binding, "content_sha256", _MISSING),
        field="artifact content_sha256",
        error_type=error_type,
    )
    _exact_integer(
        getattr(binding, "byte_count", _MISSING),
        field="artifact byte_count",
        lower=1,
        upper=_MAX_ARTIFACT_BYTES,
        error_type=error_type,
    )


def _validate_cardinality_binding(
    binding: CardinalityBinding,
    *,
    error_type: type[PublicationStateError],
) -> None:
    name = getattr(binding, "name", _MISSING)
    if (
        type(name) is not str
        or not name
        or len(name) > 64
        or not name.isascii()
        or any(
            not (character.islower() or character.isdigit() or character == "_")
            for character in name
        )
    ):
        raise error_type("cardinality name is invalid")
    _exact_integer(
        getattr(binding, "value", _MISSING),
        field=f"cardinality {name}",
        lower=0,
        upper=_MAX_CARDINALITY,
        error_type=error_type,
    )


def _validate_record_structure(
    record: PublicationStateRecord,
    *,
    error_type: type[PublicationStateError],
) -> None:
    if type(getattr(record, "schema_version", _MISSING)) is not str:
        raise error_type("state schema_version must be exact text")
    if type(getattr(record, "run_key", _MISSING)) is not str:
        raise error_type("state run_key must be exact text")
    _exact_integer(
        getattr(record, "sequence", _MISSING),
        field="state sequence",
        lower=0,
        upper=len(_STAGES) - 1,
        error_type=error_type,
    )
    if type(getattr(record, "stage", _MISSING)) is not PublicationStage:
        raise error_type("state stage must be exact")
    _validate_sha256(
        getattr(record, "previous_record_sha256", _MISSING),
        field="previous_record_sha256",
        allow_none=True,
        error_type=error_type,
    )
    _validate_sha256(
        getattr(record, "config_sha256", _MISSING),
        field="config_sha256",
        error_type=error_type,
    )
    for name in ("evaluator_id", "design_id", "population_id", "policy_id"):
        value = getattr(record, name, _MISSING)
        if type(value) is not str or not value or len(value) > 256:
            raise error_type(f"{name} is invalid")
    artifact_values = getattr(record, "artifacts", _MISSING)
    if type(artifact_values) is not tuple or any(
        type(item) is not ArtifactBinding for item in artifact_values
    ):
        raise error_type("artifacts must be an exact binding tuple")
    cardinality_values = getattr(record, "cardinalities", _MISSING)
    if type(cardinality_values) is not tuple or any(
        type(item) is not CardinalityBinding for item in cardinality_values
    ):
        raise error_type("cardinalities must be an exact binding tuple")
    artifacts = cast(tuple[ArtifactBinding, ...], artifact_values)
    cardinalities = cast(tuple[CardinalityBinding, ...], cardinality_values)
    for artifact_binding in artifacts:
        _validate_artifact_binding(artifact_binding, error_type=error_type)
    for cardinality_binding in cardinalities:
        _validate_cardinality_binding(cardinality_binding, error_type=error_type)


@dataclass(frozen=True, slots=True)
class PublicationStateRecord:
    """One immutable canonical state document."""

    schema_version: str
    run_key: str
    sequence: int
    stage: PublicationStage
    previous_record_sha256: str | None
    config_sha256: str
    evaluator_id: str
    design_id: str
    population_id: str
    policy_id: str
    artifacts: tuple[ArtifactBinding, ...]
    cardinalities: tuple[CardinalityBinding, ...]

    def __post_init__(self) -> None:
        _validate_record(self, error_type=PublicationStateConflict)


@dataclass(frozen=True, slots=True)
class PublicationRunInspection:
    """Completely replayed state records and their exact content digests."""

    records: tuple[PublicationStateRecord, ...]
    record_sha256s: tuple[str, ...]

    @property
    def current(self) -> PublicationStateRecord:
        """Return the last durable state."""

        if not self.records:
            raise PublicationStateCorrupt("publication state journal is empty")
        return self.records[-1]

    @property
    def current_sha256(self) -> str:
        """Return the SHA-256 of the last canonical state document."""

        if not self.record_sha256s:
            raise PublicationStateCorrupt("publication state journal is empty")
        return self.record_sha256s[-1]


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int


def _locked_config_sha256() -> str:
    prefix = "synthetic-eval-v3-population."
    if not LOCKED_POPULATION_ID.startswith(prefix):
        raise PublicationStateCorrupt("locked population identity is malformed")
    digest = LOCKED_POPULATION_ID.removeprefix(prefix)
    validated = _validate_sha256(
        digest,
        field="locked config SHA-256",
        error_type=PublicationStateCorrupt,
    )
    assert validated is not None
    return validated


def _validate_loaded_locked_identities() -> None:
    try:
        validate_experiment_config(DEFAULT_EXPERIMENT_CONFIG)
    except Exception as error:
        raise PublicationStateCorrupt(
            "loaded locked experiment configuration is invalid"
        ) from error
    if (
        DEFAULT_EXPERIMENT_CONFIG.split != "eval"
        or POLICY_ID != LOCKED_POLICY_ID
        or evaluator_fingerprint(DEFAULT_EXPERIMENT_CONFIG) != LOCKED_EVALUATOR_ID
        or evaluator_design_fingerprint() != LOCKED_DESIGN_ID
        or population_fingerprint(DEFAULT_EXPERIMENT_CONFIG) != LOCKED_POPULATION_ID
    ):
        raise PublicationStateCorrupt("loaded locked identities disagree")


def _artifact_names(record: PublicationStateRecord) -> tuple[str, ...]:
    return tuple(binding.name for binding in record.artifacts)


def _cardinality_names(record: PublicationStateRecord) -> tuple[str, ...]:
    return tuple(binding.name for binding in record.cardinalities)


def _expected_cardinality_values() -> dict[str, int]:
    expected = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
    return {
        definition.name: getattr(expected, definition.name)
        for definition in fields(expected)
    }


def _expected_result_cardinality_values() -> dict[str, int]:
    expected = _expected_cardinality_values()
    return {
        "abrupt_traces": expected["abrupt_traces"],
        "cluster_summaries": expected["cluster_summaries"],
        "hard_failure_count": 0,
    }


def _validate_record(
    record: PublicationStateRecord,
    *,
    error_type: type[PublicationStateError],
) -> None:
    _validate_record_structure(record, error_type=error_type)
    _validate_loaded_locked_identities()
    expected_sequence = _STAGES.index(record.stage)
    if record.schema_version != STATE_SCHEMA_VERSION:
        raise error_type("publication state schema version is invalid")
    if record.run_key != LOCKED_EVALUATION_RUN_KEY:
        raise error_type("publication state run key is invalid")
    if record.sequence != expected_sequence:
        raise error_type("publication state sequence and stage disagree")
    if (record.sequence == 0) != (record.previous_record_sha256 is None):
        raise error_type("publication state previous digest is invalid")
    identities = (
        (record.config_sha256, _locked_config_sha256(), "config"),
        (record.evaluator_id, LOCKED_EVALUATOR_ID, "evaluator"),
        (record.design_id, LOCKED_DESIGN_ID, "design"),
        (record.population_id, LOCKED_POPULATION_ID, "population"),
        (record.policy_id, LOCKED_POLICY_ID, "policy"),
    )
    for actual, expected, label in identities:
        if actual != expected:
            raise error_type(f"publication state {label} identity is invalid")

    if len(record.artifacts) > _MAX_ARTIFACTS:
        raise error_type("publication state has too many artifacts")
    artifact_names = _artifact_names(record)
    cardinality_names = _cardinality_names(record)
    if artifact_names != tuple(sorted(artifact_names)) or len(
        set(artifact_names)
    ) != len(artifact_names):
        raise error_type("publication artifacts are not uniquely canonical")
    if cardinality_names != tuple(sorted(cardinality_names)) or len(
        set(cardinality_names)
    ) != len(cardinality_names):
        raise error_type("publication cardinalities are not uniquely canonical")

    if record.stage in {PublicationStage.PREPARED, PublicationStage.EVALUATING}:
        if artifact_names != (SOURCE_PROVENANCE_FILE_NAME,):
            raise error_type(f"{record.stage.value} state must bind source provenance")
        if record.cardinalities:
            raise error_type(f"{record.stage.value} state cannot bind cardinalities")
        return

    result_values = _expected_result_cardinality_values()
    if record.stage is PublicationStage.EVALUATED:
        if artifact_names != (
            _RESULT_ARTIFACT_NAME,
            SOURCE_PROVENANCE_FILE_NAME,
        ):
            raise error_type(
                "evaluated state must bind result.bin and source provenance"
            )
        if cardinality_names != _RESULT_CARDINALITY_NAMES:
            raise error_type("evaluated state cardinalities are invalid")
        if {
            binding.name: binding.value for binding in record.cardinalities
        } != result_values:
            raise error_type("evaluated state cardinality values are invalid")
        return

    required_artifacts = (
        _EVIDENCE_ARTIFACT_NAME,
        _REPORT_ARTIFACT_NAME,
        _RESULT_ARTIFACT_NAME,
        SOURCE_PROVENANCE_FILE_NAME,
    )
    expected_cardinalities = {
        **_expected_cardinality_values(),
        "hard_failure_count": 0,
    }
    expected_cardinality_names = tuple(sorted(expected_cardinalities))
    if (
        cardinality_names != expected_cardinality_names
        or {binding.name: binding.value for binding in record.cardinalities}
        != expected_cardinalities
    ):
        raise error_type("publication evidence cardinalities are invalid")

    if record.stage is PublicationStage.MATERIALIZED:
        if artifact_names != required_artifacts:
            raise error_type("materialized state artifact inventory is invalid")
        return

    if record.stage is not PublicationStage.SEALED:
        raise error_type("publication state stage is unknown")
    if not all(name in artifact_names for name in required_artifacts):
        raise error_type("sealed state is missing a materialized artifact")
    if _MANIFEST_ARTIFACT_NAME not in artifact_names:
        raise error_type("sealed state is missing manifest.json")
    rendered_names = tuple(
        name
        for name in artifact_names
        if name not in {*required_artifacts, _MANIFEST_ARTIFACT_NAME}
    )
    if not rendered_names or any(
        not name.startswith("artifacts/") for name in rendered_names
    ):
        raise error_type("sealed state must bind rendered artifacts")


def _record_payload(record: PublicationStateRecord) -> dict[str, object]:
    return {
        "artifacts": [
            {
                "byte_count": artifact.byte_count,
                "content_sha256": artifact.content_sha256,
                "name": artifact.name,
            }
            for artifact in record.artifacts
        ],
        "cardinalities": [
            {"name": cardinality.name, "value": cardinality.value}
            for cardinality in record.cardinalities
        ],
        "config_sha256": record.config_sha256,
        "design_id": record.design_id,
        "evaluator_id": record.evaluator_id,
        "policy_id": record.policy_id,
        "population_id": record.population_id,
        "previous_record_sha256": record.previous_record_sha256,
        "run_key": record.run_key,
        "schema_version": record.schema_version,
        "sequence": record.sequence,
        "stage": record.stage.value,
    }


def encode_state_record(record: PublicationStateRecord) -> bytes:
    """Return the only accepted JSON representation of a state record."""

    if type(record) is not PublicationStateRecord:
        raise PublicationStateConflict(
            "state record must be an exact PublicationStateRecord"
        )
    _validate_record(record, error_type=PublicationStateConflict)
    try:
        content = json.dumps(
            _record_payload(record),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise PublicationStateConflict("cannot encode publication state") from error
    if len(content) > _MAX_STATE_BYTES:
        raise PublicationStateConflict("canonical publication state is too large")
    return content


def _reject_float(_token: str) -> NoReturn:
    raise PublicationStateCorrupt("publication state cannot contain JSON floats")


def _reject_constant(_token: str) -> NoReturn:
    raise PublicationStateCorrupt("publication state cannot contain JSON constants")


def _parse_json_integer(token: str) -> int:
    if len(token) > 19:
        raise PublicationStateCorrupt("publication state integer is too long")
    try:
        value = int(token)
    except ValueError as error:
        raise PublicationStateCorrupt("publication state integer is invalid") from error
    if not -(2**63) <= value <= _MAX_CARDINALITY:
        raise PublicationStateCorrupt("publication state integer is out of range")
    return value


def _decode_json(content: bytes) -> object:
    if type(content) is not bytes:
        raise PublicationStateCorrupt("publication state content must be exact bytes")
    if not content or len(content) > _MAX_STATE_BYTES:
        raise PublicationStateCorrupt("publication state byte length is invalid")
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as error:
        raise PublicationStateCorrupt(
            "publication state must be canonical ASCII"
        ) from error
    container_count = 0

    def closed_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal container_count
        container_count += 1
        if container_count > _MAX_STATE_CONTAINERS:
            raise PublicationStateCorrupt("publication state contains too many objects")
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PublicationStateCorrupt(
                    f"publication state contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=closed_object,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_json_integer,
        )
    except PublicationStateCorrupt:
        raise
    except (RecursionError, ValueError, json.JSONDecodeError) as error:
        raise PublicationStateCorrupt("publication state JSON is invalid") from error
    total_containers = 0
    stack: list[tuple[object, int]] = [(decoded, 0)]
    while stack:
        value, depth = stack.pop()
        if isinstance(value, (dict, list)):
            total_containers += 1
            if total_containers > _MAX_STATE_CONTAINERS or depth > _MAX_JSON_DEPTH:
                raise PublicationStateCorrupt(
                    "publication state JSON structure exceeds bounds"
                )
            children = value.values() if isinstance(value, dict) else value
            stack.extend((child, depth + 1) for child in children)
        elif isinstance(value, str):
            try:
                string_bytes = value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise PublicationStateCorrupt(
                    "publication state JSON string is not valid Unicode"
                ) from error
            if len(string_bytes) > _MAX_JSON_STRING_BYTES:
                raise PublicationStateCorrupt(
                    "publication state JSON string exceeds bounds"
                )
    return decoded


_ROOT_KEYS: Final = frozenset(
    {
        "artifacts",
        "cardinalities",
        "config_sha256",
        "design_id",
        "evaluator_id",
        "policy_id",
        "population_id",
        "previous_record_sha256",
        "run_key",
        "schema_version",
        "sequence",
        "stage",
    }
)
_ARTIFACT_KEYS: Final = frozenset({"byte_count", "content_sha256", "name"})
_CARDINALITY_KEYS: Final = frozenset({"name", "value"})


def _exact_object(
    value: object,
    *,
    keys: frozenset[str],
    field: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise PublicationStateCorrupt(f"{field} must be an object")
    mapping = value
    if set(mapping) != keys:
        raise PublicationStateCorrupt(f"{field} has missing or unknown keys")
    return mapping


def _exact_list(value: object, *, field: str) -> list[object]:
    if type(value) is not list:
        raise PublicationStateCorrupt(f"{field} must be an array")
    return value


def _exact_text(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise PublicationStateCorrupt(f"{field} must be exact text")
    return value


def _decoded_integer(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise PublicationStateCorrupt(f"{field} must be an exact integer")
    return value


def _decoded_optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _exact_text(value, field=field)


def decode_state_record(content: bytes) -> PublicationStateRecord:
    """Decode, validate, and byte-for-byte replay one canonical state record."""

    payload = _exact_object(_decode_json(content), keys=_ROOT_KEYS, field="state")
    artifacts_raw = _exact_list(payload["artifacts"], field="artifacts")
    cardinalities_raw = _exact_list(
        payload["cardinalities"],
        field="cardinalities",
    )
    if len(artifacts_raw) > _MAX_ARTIFACTS:
        raise PublicationStateCorrupt("publication state has too many artifacts")
    if len(cardinalities_raw) > len(_EVIDENCE_CARDINALITY_NAMES) + 1:
        raise PublicationStateCorrupt("publication state has too many cardinalities")

    try:
        artifacts = tuple(
            ArtifactBinding(
                name=_exact_text(
                    _exact_object(item, keys=_ARTIFACT_KEYS, field="artifact")["name"],
                    field="artifact name",
                ),
                content_sha256=_exact_text(
                    _exact_object(item, keys=_ARTIFACT_KEYS, field="artifact")[
                        "content_sha256"
                    ],
                    field="artifact content_sha256",
                ),
                byte_count=_decoded_integer(
                    _exact_object(
                        item,
                        keys=_ARTIFACT_KEYS,
                        field="artifact",
                    )["byte_count"],
                    field="artifact byte_count",
                ),
            )
            for item in artifacts_raw
        )
        cardinalities = tuple(
            CardinalityBinding(
                name=_exact_text(
                    _exact_object(
                        item,
                        keys=_CARDINALITY_KEYS,
                        field="cardinality",
                    )["name"],
                    field="cardinality name",
                ),
                value=_decoded_integer(
                    _exact_object(
                        item,
                        keys=_CARDINALITY_KEYS,
                        field="cardinality",
                    )["value"],
                    field="cardinality value",
                ),
            )
            for item in cardinalities_raw
        )
        stage_text = _exact_text(payload["stage"], field="stage")
        record = PublicationStateRecord(
            schema_version=_exact_text(
                payload["schema_version"],
                field="schema_version",
            ),
            run_key=_exact_text(payload["run_key"], field="run_key"),
            sequence=_decoded_integer(payload["sequence"], field="sequence"),
            stage=PublicationStage(stage_text),
            previous_record_sha256=_decoded_optional_text(
                payload["previous_record_sha256"],
                field="previous_record_sha256",
            ),
            config_sha256=_exact_text(
                payload["config_sha256"],
                field="config_sha256",
            ),
            evaluator_id=_exact_text(payload["evaluator_id"], field="evaluator_id"),
            design_id=_exact_text(payload["design_id"], field="design_id"),
            population_id=_exact_text(
                payload["population_id"],
                field="population_id",
            ),
            policy_id=_exact_text(payload["policy_id"], field="policy_id"),
            artifacts=artifacts,
            cardinalities=cardinalities,
        )
    except PublicationStateCorrupt:
        raise
    except PublicationStateError as error:
        raise PublicationStateCorrupt(str(error)) from error
    except (ValueError, TypeError) as error:
        raise PublicationStateCorrupt("publication state fields are invalid") from error
    if encode_state_record(record) != content:
        raise PublicationStateCorrupt("publication state JSON is not canonical")
    return record


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        raise PublicationStateSecurityError(
            "publication state requires Linux secure-open flags"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _file_open_flags(*, write: bool, create_exclusive: bool = False) -> int:
    required = ("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required):
        raise PublicationStateSecurityError(
            "publication state requires Linux secure-open flags"
        )
    flags = (
        (os.O_WRONLY if write else os.O_RDONLY)
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | os.O_NONBLOCK
    )
    if create_exclusive:
        flags |= os.O_CREAT | os.O_EXCL
    return flags


def _lock_open_flags(*, create_exclusive: bool) -> int:
    flags = _file_open_flags(write=True, create_exclusive=create_exclusive)
    return (flags & ~os.O_WRONLY) | os.O_RDWR


def _best_effort_close(descriptor: int) -> None:
    with suppress(OSError):
        os.close(descriptor)


def _close_descriptor(descriptor: int, *, field: str) -> None:
    try:
        os.close(descriptor)
    except OSError as error:
        raise PublicationStateIOError(f"cannot close {field}") from error


def _file_identity(information: os.stat_result) -> _FileIdentity:
    return _FileIdentity(information.st_dev, information.st_ino)


def _validate_directory(
    information: os.stat_result,
    *,
    field: str,
    private: bool,
    expected: _FileIdentity | None = None,
) -> _FileIdentity:
    if not stat.S_ISDIR(information.st_mode):
        raise PublicationStateSecurityError(f"{field} must be a directory")
    if private:
        if information.st_uid != os.geteuid():
            raise PublicationStateSecurityError(f"{field} has the wrong owner")
        if stat.S_IMODE(information.st_mode) != 0o700:
            raise PublicationStateSecurityError(f"{field} mode must be 0700")
    identity = _file_identity(information)
    if expected is not None and identity != expected:
        raise PublicationStateSecurityError(f"{field} identity changed")
    return identity


def _validate_private_file(
    information: os.stat_result,
    *,
    field: str,
    expected: _FileIdentity | None = None,
) -> _FileIdentity:
    if not stat.S_ISREG(information.st_mode):
        raise PublicationStateSecurityError(f"{field} must be a regular file")
    if information.st_uid != os.geteuid():
        raise PublicationStateSecurityError(f"{field} has the wrong owner")
    if stat.S_IMODE(information.st_mode) != 0o600:
        raise PublicationStateSecurityError(f"{field} mode must be 0600")
    if information.st_nlink != 1:
        raise PublicationStateSecurityError(f"{field} must not have hard links")
    identity = _file_identity(information)
    if expected is not None and identity != expected:
        raise PublicationStateSecurityError(f"{field} identity changed")
    return identity


def _fstat(descriptor: int, *, field: str) -> os.stat_result:
    try:
        return os.fstat(descriptor)
    except OSError as error:
        raise PublicationStateIOError(f"cannot inspect open {field}") from error


def _stat_at(
    directory_descriptor: int,
    name: str,
    *,
    field: str,
) -> os.stat_result:
    try:
        return os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise PublicationStateSecurityError(f"cannot inspect {field}") from error


def _fsync(descriptor: int, *, field: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise PublicationStateIOError(f"cannot fsync {field}") from error


def _validate_run_path(path: str | Path) -> Path:
    if type(path) is str:
        raw = path
    elif type(path) is _PLATFORM_PATH_TYPE:
        try:
            raw_components = object.__getattribute__(
                path,
                _PATH_RAW_COMPONENTS_SLOT,
            )
        except AttributeError as error:
            raise PublicationStateSecurityError(
                "publication run path must be text or Path"
            ) from error
        if type(raw_components) is not list:
            raise PublicationStateSecurityError(
                "publication run path must be text or Path"
            )
        components = tuple(raw_components)
        if any(type(component) is not str for component in components):
            raise PublicationStateSecurityError(
                "publication run path must be text or Path"
            )
        rebuilt = _PLATFORM_PATH_TYPE(*components)
        raw = rebuilt.as_posix()
    else:
        raise PublicationStateSecurityError("publication run path must be text or Path")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PublicationStateSecurityError(
            "publication run path must be valid Unicode"
        ) from error
    if (
        not raw
        or "\x00" in raw
        or raw.startswith("//")
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
        or not os.path.isabs(raw)
        or os.path.normpath(raw) != raw
        or raw == os.path.sep
    ):
        raise PublicationStateSecurityError(
            "publication run path must be canonical and absolute"
        )
    candidate = Path(raw)
    if candidate.name != LOCKED_EVALUATION_RUN_KEY:
        raise PublicationStateSecurityError(
            "publication run directory must use the locked run key"
        )
    return candidate


def _open_absolute_directory(path: Path) -> int:
    try:
        descriptor = os.open(os.path.sep, _directory_open_flags())
    except OSError as error:
        raise PublicationStateSecurityError(
            "cannot open filesystem root safely"
        ) from error
    try:
        for component in path.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise PublicationStateSecurityError(
                    "publication path must contain only real directories"
                ) from error
            _best_effort_close(descriptor)
            descriptor = next_descriptor
        _validate_directory(
            _fstat(descriptor, field="publication parent path"),
            field="publication parent",
            private=True,
        )
        return descriptor
    except BaseException:
        _best_effort_close(descriptor)
        raise


def _open_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    field: str,
) -> tuple[int, _FileIdentity]:
    try:
        descriptor = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise PublicationStateSecurityError(f"cannot safely open {field}") from error
    try:
        descriptor_identity = _validate_directory(
            _fstat(descriptor, field=field),
            field=field,
            private=True,
        )
        path_identity = _validate_directory(
            _stat_at(parent_descriptor, name, field=field),
            field=field,
            private=True,
        )
        if descriptor_identity != path_identity:
            raise PublicationStateSecurityError(f"{field} changed while opening")
        return descriptor, descriptor_identity
    except BaseException:
        _best_effort_close(descriptor)
        raise


def _create_or_open_run_directory(
    run_path: Path,
    *,
    create: bool,
) -> tuple[int, _FileIdentity, int, _FileIdentity]:
    parent_descriptor = _open_absolute_directory(run_path.parent)
    try:
        try:
            information = os.stat(
                run_path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if not create:
                raise PublicationStateSecurityError(
                    "publication run directory does not exist"
                ) from None
            try:
                os.mkdir(run_path.name, mode=0o700, dir_fd=parent_descriptor)
            except OSError as error:
                raise PublicationStateIOError(
                    "cannot create publication run directory"
                ) from error
            _fsync(parent_descriptor, field="publication parent directory")
        except OSError as error:
            raise PublicationStateSecurityError(
                "cannot inspect publication run directory"
            ) from error
        else:
            _validate_directory(
                information,
                field="publication run directory",
                private=True,
            )
        run_descriptor, run_identity = _open_directory_at(
            parent_descriptor,
            run_path.name,
            field="publication run directory",
        )
        parent_identity = _validate_directory(
            _fstat(parent_descriptor, field="publication parent"),
            field="publication parent",
            private=True,
        )
        return (
            parent_descriptor,
            parent_identity,
            run_descriptor,
            run_identity,
        )
    except BaseException:
        _best_effort_close(parent_descriptor)
        raise


def _open_private_file_at(
    directory_descriptor: int,
    name: str,
    *,
    field: str,
    lock_file: bool = False,
    create_exclusive: bool = False,
) -> tuple[int, _FileIdentity]:
    flags = (
        _lock_open_flags(create_exclusive=create_exclusive)
        if lock_file
        else _file_open_flags(write=False, create_exclusive=False)
    )
    try:
        descriptor = os.open(
            name,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
    except FileExistsError:
        raise PublicationRunExists("publication run is already claimed") from None
    except OSError as error:
        raise PublicationStateSecurityError(f"cannot safely open {field}") from error
    try:
        descriptor_information = _fstat(descriptor, field=field)
        descriptor_identity = _validate_private_file(
            descriptor_information,
            field=field,
        )
        path_identity = _validate_private_file(
            _stat_at(directory_descriptor, name, field=field),
            field=field,
        )
        if descriptor_identity != path_identity:
            raise PublicationStateSecurityError(f"{field} changed while opening")
        if lock_file and descriptor_information.st_size != 0:
            raise PublicationStateSecurityError("publication lock must be empty")
        return descriptor, descriptor_identity
    except BaseException:
        _best_effort_close(descriptor)
        raise


def _acquire_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            raise PublicationRunBusy(
                "another process holds the publication run lock"
            ) from error
        raise PublicationStateIOError("cannot acquire publication run lock") from error


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    offset = 0
    while offset < len(content):
        try:
            written = os.write(descriptor, view[offset:])
        except OSError as error:
            raise PublicationStateIOError("cannot write publication state") from error
        if type(written) is not int or written <= 0:
            raise PublicationStateIOError("publication state write made no progress")
        offset += written


def _read_state_file(
    directory_descriptor: int,
    name: str,
) -> tuple[bytes, str]:
    descriptor, identity = _open_private_file_at(
        directory_descriptor,
        name,
        field=f"state file {name}",
    )
    try:
        information = _fstat(descriptor, field=f"state file {name}")
        if not 1 <= information.st_size <= _MAX_STATE_BYTES:
            raise PublicationStateCorrupt(f"state file {name} has invalid size")
        chunks: list[bytes] = []
        remaining = information.st_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, 16 * 1024))
            except OSError as error:
                raise PublicationStateIOError(
                    f"cannot read state file {name}"
                ) from error
            if not chunk:
                raise PublicationStateCorrupt(f"state file {name} is truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        try:
            trailing = os.read(descriptor, 1)
        except OSError as error:
            raise PublicationStateIOError(
                f"cannot finish reading state file {name}"
            ) from error
        if trailing:
            raise PublicationStateCorrupt(f"state file {name} changed while reading")
        final_information = _fstat(descriptor, field=f"state file {name}")
        _validate_private_file(
            final_information,
            field=f"state file {name}",
            expected=identity,
        )
        _validate_private_file(
            _stat_at(
                directory_descriptor,
                name,
                field=f"state file {name}",
            ),
            field=f"state file {name}",
            expected=identity,
        )
        if final_information.st_size != information.st_size:
            raise PublicationStateCorrupt(f"state file {name} changed while reading")
        content = b"".join(chunks)
        return content, hashlib.sha256(content).hexdigest()
    finally:
        _close_descriptor(descriptor, field=f"state file {name}")


class PublicationStateStore:
    """Exclusive descriptor- and process-bound handle to one locked run."""

    __slots__ = (
        "_closed",
        "_live_evaluation_claim",
        "_live_evaluation_permit",
        "_lock_descriptor",
        "_lock_identity",
        "_owner_pid",
        "_parent_descriptor",
        "_parent_identity",
        "_run_descriptor",
        "_run_identity",
        "_run_path",
        "_state_descriptor",
        "_state_identity",
    )

    def __init__(
        self,
        *,
        run_path: Path,
        parent_descriptor: int,
        parent_identity: _FileIdentity,
        run_descriptor: int,
        run_identity: _FileIdentity,
        lock_descriptor: int,
        lock_identity: _FileIdentity,
        state_descriptor: int,
        state_identity: _FileIdentity,
    ) -> None:
        self._run_path = run_path
        self._parent_descriptor = parent_descriptor
        self._parent_identity = parent_identity
        self._run_descriptor = run_descriptor
        self._run_identity = run_identity
        self._lock_descriptor = lock_descriptor
        self._lock_identity = lock_identity
        self._owner_pid = os.getpid()
        self._state_descriptor = state_descriptor
        self._state_identity = state_identity
        self._live_evaluation_claim: str | None = None
        self._live_evaluation_permit: _LockedEvaluationPermit | None = None
        self._closed = False

    @classmethod
    def initialize(
        cls,
        run_directory: str | Path,
        *,
        source_provenance: ArtifactBinding,
        config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
    ) -> PublicationStateStore:
        """Claim a new locked run and durably append ``PREPARED``.

        The absolute parent directory must already exist, be owned by the
        effective user, and have mode ``0700``.  A pre-existing lock or states
        directory is treated as a previous claim, never reset.
        """

        _validate_loaded_locked_identities()
        try:
            validate_experiment_config(config)
        except Exception as error:
            raise PublicationStateConflict(
                "publication config failed closed validation"
            ) from error
        if config != DEFAULT_EXPERIMENT_CONFIG:
            raise PublicationStateConflict(
                "publication state requires the exact locked config"
            )
        if type(source_provenance) is not ArtifactBinding:
            raise PublicationStateConflict(
                "publication state requires source provenance"
            )
        _validate_artifact_binding(
            source_provenance,
            error_type=PublicationStateConflict,
        )
        if source_provenance.name != SOURCE_PROVENANCE_FILE_NAME:
            raise PublicationStateConflict(
                "publication state requires source-provenance.json"
            )
        run_path = _validate_run_path(run_directory)
        parent_fd, parent_id, run_fd, run_id = _create_or_open_run_directory(
            run_path,
            create=True,
        )
        lock_fd = -1
        state_fd = -1
        try:
            lock_fd, lock_id = _open_private_file_at(
                run_fd,
                LOCK_FILE_NAME,
                field="publication lock",
                lock_file=True,
                create_exclusive=True,
            )
            _fsync(lock_fd, field="publication lock")
            _fsync(run_fd, field="publication run directory")
            _acquire_lock(lock_fd)
            try:
                os.mkdir(
                    STATE_DIRECTORY_NAME,
                    mode=0o700,
                    dir_fd=run_fd,
                )
            except FileExistsError:
                raise PublicationRunExists(
                    "publication states directory already exists"
                ) from None
            except OSError as error:
                raise PublicationStateIOError(
                    "cannot create publication states directory"
                ) from error
            _fsync(run_fd, field="publication run directory")
            state_fd, state_id = _open_directory_at(
                run_fd,
                STATE_DIRECTORY_NAME,
                field="publication states directory",
            )
            store = cls(
                run_path=run_path,
                parent_descriptor=parent_fd,
                parent_identity=parent_id,
                run_descriptor=run_fd,
                run_identity=run_id,
                lock_descriptor=lock_fd,
                lock_identity=lock_id,
                state_descriptor=state_fd,
                state_identity=state_id,
            )
            prepared = store._new_record(
                stage=PublicationStage.PREPARED,
                previous_sha256=None,
                artifacts=(source_provenance,),
                cardinalities=(),
            )
            store._write_record(prepared)
            store.inspect()
            return store
        except BaseException:
            if state_fd >= 0:
                _best_effort_close(state_fd)
            if lock_fd >= 0:
                _best_effort_close(lock_fd)
            _best_effort_close(run_fd)
            _best_effort_close(parent_fd)
            raise

    @classmethod
    def open(cls, run_directory: str | Path) -> PublicationStateStore:
        """Acquire an existing run for replay and legal continuation."""

        _validate_loaded_locked_identities()
        run_path = _validate_run_path(run_directory)
        parent_fd, parent_id, run_fd, run_id = _create_or_open_run_directory(
            run_path,
            create=False,
        )
        lock_fd = -1
        state_fd = -1
        try:
            lock_fd, lock_id = _open_private_file_at(
                run_fd,
                LOCK_FILE_NAME,
                field="publication lock",
                lock_file=True,
            )
            _acquire_lock(lock_fd)
            state_fd, state_id = _open_directory_at(
                run_fd,
                STATE_DIRECTORY_NAME,
                field="publication states directory",
            )
            store = cls(
                run_path=run_path,
                parent_descriptor=parent_fd,
                parent_identity=parent_id,
                run_descriptor=run_fd,
                run_identity=run_id,
                lock_descriptor=lock_fd,
                lock_identity=lock_id,
                state_descriptor=state_fd,
                state_identity=state_id,
            )
            store.inspect()
            return store
        except BaseException:
            if state_fd >= 0:
                _best_effort_close(state_fd)
            if lock_fd >= 0:
                _best_effort_close(lock_fd)
            _best_effort_close(run_fd)
            _best_effort_close(parent_fd)
            raise

    def __enter__(self) -> PublicationStateStore:
        self._require_open()
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        self.close()

    @property
    def run_directory(self) -> Path:
        """Return the canonical path used to bind this descriptor set."""

        self._require_open()
        return self._run_path

    def _require_open(self) -> None:
        if self._closed:
            raise PublicationStateConflict("publication state store is closed")
        owner_pid = getattr(self, "_owner_pid", None)
        if type(owner_pid) is not int or owner_pid != os.getpid():
            raise PublicationStateConflict(
                "publication state store cannot be used after fork"
            )

    def _validate_handles(self) -> None:
        self._require_open()
        _validate_directory(
            _fstat(self._parent_descriptor, field="publication parent"),
            field="publication parent",
            private=True,
            expected=self._parent_identity,
        )
        _validate_directory(
            _stat_at(
                self._parent_descriptor,
                self._run_path.name,
                field="publication run directory",
            ),
            field="publication run directory",
            private=True,
            expected=self._run_identity,
        )
        _validate_directory(
            _fstat(self._run_descriptor, field="publication run directory"),
            field="publication run directory",
            private=True,
            expected=self._run_identity,
        )
        _validate_private_file(
            _fstat(self._lock_descriptor, field="publication lock"),
            field="publication lock",
            expected=self._lock_identity,
        )
        _validate_private_file(
            _stat_at(
                self._run_descriptor,
                LOCK_FILE_NAME,
                field="publication lock",
            ),
            field="publication lock",
            expected=self._lock_identity,
        )
        _validate_directory(
            _fstat(
                self._state_descriptor,
                field="publication states directory",
            ),
            field="publication states directory",
            private=True,
            expected=self._state_identity,
        )
        _validate_directory(
            _stat_at(
                self._run_descriptor,
                STATE_DIRECTORY_NAME,
                field="publication states directory",
            ),
            field="publication states directory",
            private=True,
            expected=self._state_identity,
        )

    def inspect(self) -> PublicationRunInspection:
        """Replay the complete journal and reject any non-prefix entry set."""

        self._validate_handles()
        try:
            entries = os.listdir(self._state_descriptor)
        except OSError as error:
            raise PublicationStateIOError(
                "cannot list publication states directory"
            ) from error
        if any(type(entry) is not str for entry in entries):
            raise PublicationStateCorrupt("publication state filename is invalid")
        unknown = set(entries) - _STATE_FILE_NAME_SET
        if unknown:
            raise PublicationStateCorrupt(
                "publication states directory contains trailing entries"
            )
        present = tuple(name for name in _STATE_FILE_NAMES if name in entries)
        if not present:
            raise PublicationStateCorrupt("publication state journal is empty")
        if present != _STATE_FILE_NAMES[: len(present)] or len(entries) != len(present):
            raise PublicationStateCorrupt(
                "publication state journal has a gap or reordered file"
            )

        records: list[PublicationStateRecord] = []
        digests: list[str] = []
        for sequence, name in enumerate(present):
            content, digest = _read_state_file(self._state_descriptor, name)
            record = decode_state_record(content)
            if record.sequence != sequence or record.stage is not _STAGES[sequence]:
                raise PublicationStateCorrupt(
                    "publication state filename and content disagree"
                )
            expected_previous = None if sequence == 0 else digests[-1]
            if record.previous_record_sha256 != expected_previous:
                raise PublicationStateCorrupt(
                    "publication state SHA-256 chain is invalid"
                )
            if records:
                self._validate_transition(records[-1], record)
            records.append(record)
            digests.append(digest)
        self._validate_handles()
        try:
            final_entries = os.listdir(self._state_descriptor)
        except OSError as error:
            raise PublicationStateIOError(
                "cannot relist publication states directory"
            ) from error
        if sorted(final_entries) != sorted(entries):
            raise PublicationStateCorrupt(
                "publication states directory changed during replay"
            )
        return PublicationRunInspection(tuple(records), tuple(digests))

    @staticmethod
    def _validate_transition(
        previous: PublicationStateRecord,
        current: PublicationStateRecord,
    ) -> None:
        if current.sequence != previous.sequence + 1:
            raise PublicationStateCorrupt(
                "publication state transition is not adjacent"
            )
        for field in (
            "schema_version",
            "run_key",
            "config_sha256",
            "evaluator_id",
            "design_id",
            "population_id",
            "policy_id",
        ):
            if getattr(current, field) != getattr(previous, field):
                raise PublicationStateCorrupt(
                    f"publication state transition changed {field}"
                )
        previous_artifacts = {binding.name: binding for binding in previous.artifacts}
        current_artifacts = {binding.name: binding for binding in current.artifacts}
        if any(
            current_artifacts.get(name) != binding
            for name, binding in previous_artifacts.items()
        ):
            raise PublicationStateCorrupt(
                "publication state transition changed an artifact binding"
            )
        if previous.stage is PublicationStage.EVALUATED:
            previous_counts = {
                binding.name: binding.value for binding in previous.cardinalities
            }
            current_counts = {
                binding.name: binding.value for binding in current.cardinalities
            }
            for name, value in previous_counts.items():
                if current_counts.get(name) != value:
                    raise PublicationStateCorrupt(
                        "publication state transition changed result cardinality"
                    )
        elif previous.stage is PublicationStage.MATERIALIZED:
            if current.cardinalities != previous.cardinalities:
                raise PublicationStateCorrupt(
                    "sealed state changed publication cardinalities"
                )

    def _new_record(
        self,
        *,
        stage: PublicationStage,
        previous_sha256: str | None,
        artifacts: tuple[ArtifactBinding, ...],
        cardinalities: tuple[CardinalityBinding, ...],
    ) -> PublicationStateRecord:
        return PublicationStateRecord(
            schema_version=STATE_SCHEMA_VERSION,
            run_key=LOCKED_EVALUATION_RUN_KEY,
            sequence=_STAGES.index(stage),
            stage=stage,
            previous_record_sha256=previous_sha256,
            config_sha256=_locked_config_sha256(),
            evaluator_id=LOCKED_EVALUATOR_ID,
            design_id=LOCKED_DESIGN_ID,
            population_id=LOCKED_POPULATION_ID,
            policy_id=LOCKED_POLICY_ID,
            artifacts=artifacts,
            cardinalities=cardinalities,
        )

    def _write_record(self, record: PublicationStateRecord) -> str:
        self._validate_handles()
        content = encode_state_record(record)
        name = _STATE_FILE_NAMES[record.sequence]
        try:
            descriptor = os.open(
                name,
                _file_open_flags(write=True, create_exclusive=True),
                0o600,
                dir_fd=self._state_descriptor,
            )
        except FileExistsError:
            raise PublicationStateConflict(
                "publication state transition was already appended"
            ) from None
        except OSError as error:
            raise PublicationStateIOError(
                "cannot create publication state file"
            ) from error
        try:
            identity = _validate_private_file(
                _fstat(descriptor, field=f"state file {name}"),
                field=f"state file {name}",
            )
            _write_all(descriptor, content)
            _fsync(descriptor, field=f"state file {name}")
        except BaseException:
            _best_effort_close(descriptor)
            raise
        _close_descriptor(descriptor, field=f"state file {name}")
        path_identity = _validate_private_file(
            _stat_at(
                self._state_descriptor,
                name,
                field=f"state file {name}",
            ),
            field=f"state file {name}",
        )
        if path_identity != identity:
            raise PublicationStateSecurityError(
                f"state file {name} changed after creation"
            )
        _fsync(self._state_descriptor, field="publication states directory")
        return hashlib.sha256(content).hexdigest()

    def _append(
        self,
        *,
        expected_current: PublicationStage,
        stage: PublicationStage,
        artifacts: tuple[ArtifactBinding, ...],
        cardinalities: tuple[CardinalityBinding, ...],
    ) -> PublicationRunInspection:
        inspection = self.inspect()
        if inspection.current.stage is not expected_current:
            raise PublicationStateConflict(
                f"cannot enter {stage.value} from {inspection.current.stage.value}"
            )
        if _STAGES.index(stage) != inspection.current.sequence + 1:
            raise PublicationStateConflict("publication state transition is not next")
        record = self._new_record(
            stage=stage,
            previous_sha256=inspection.current_sha256,
            artifacts=artifacts,
            cardinalities=cardinalities,
        )
        self._validate_transition(inspection.current, record)
        self._write_record(record)
        replayed = self.inspect()
        if replayed.current != record:
            raise PublicationStateCorrupt(
                "durable publication state differs after append"
            )
        return replayed

    def begin_evaluation(self) -> _LockedEvaluationPermit:
        """Durably claim held-out computation, then issue its one-use permit."""

        inspection = self.inspect()
        replayed = self._append(
            expected_current=PublicationStage.PREPARED,
            stage=PublicationStage.EVALUATING,
            artifacts=inspection.current.artifacts,
            cardinalities=(),
        )
        claim_sha256 = replayed.current_sha256
        permit = _issue_locked_evaluation_permit(
            run_key=LOCKED_EVALUATION_RUN_KEY,
            claim_sha256=claim_sha256,
        )
        if type(permit) is not _LockedEvaluationPermit:
            raise PublicationStateCorrupt(
                "evaluator issued a non-canonical locked permit"
            )
        self._live_evaluation_claim = claim_sha256
        self._live_evaluation_permit = permit
        return permit

    def record_evaluated(
        self,
        *,
        result: ArtifactBinding,
        cluster_summaries: int,
        abrupt_traces: int,
        hard_failure_count: int,
    ) -> PublicationStateRecord:
        """Bind a complete result from this handle's live evaluation claim."""

        inspection = self.inspect()
        if inspection.current.stage is not PublicationStage.EVALUATING:
            raise PublicationStateConflict(
                "evaluated state requires the durable evaluating state"
            )
        permit = self._live_evaluation_permit
        permit_claim = (
            getattr(permit, "_claim_sha256", None)
            if type(permit) is _LockedEvaluationPermit
            else None
        )
        if (
            self._live_evaluation_claim is None
            or self._live_evaluation_claim != inspection.current_sha256
            or type(permit) is not _LockedEvaluationPermit
            or permit_claim != inspection.current_sha256
        ):
            raise PublicationStateConflict(
                "an evaluating run cannot resume after reopening"
            )
        if getattr(permit, "_consumed", None) is not True:
            raise PublicationStateConflict(
                "evaluation permit must be consumed before binding a result"
            )
        if type(result) is not ArtifactBinding:
            raise PublicationStateConflict("evaluated state requires result.bin")
        _validate_artifact_binding(
            result,
            error_type=PublicationStateConflict,
        )
        if result.name != _RESULT_ARTIFACT_NAME:
            raise PublicationStateConflict("evaluated state requires result.bin")
        supplied = {
            "abrupt_traces": _exact_integer(
                abrupt_traces,
                field="abrupt_traces",
                lower=0,
                upper=_MAX_CARDINALITY,
            ),
            "cluster_summaries": _exact_integer(
                cluster_summaries,
                field="cluster_summaries",
                lower=0,
                upper=_MAX_CARDINALITY,
            ),
            "hard_failure_count": _exact_integer(
                hard_failure_count,
                field="hard_failure_count",
                lower=0,
                upper=_MAX_CARDINALITY,
            ),
        }
        if supplied != _expected_result_cardinality_values():
            raise PublicationStateConflict(
                "evaluated result inventory is not the locked inventory"
            )
        cardinalities = tuple(
            CardinalityBinding(name, supplied[name])
            for name in _RESULT_CARDINALITY_NAMES
        )
        artifacts = tuple(
            sorted(
                (*inspection.current.artifacts, result),
                key=lambda binding: binding.name,
            )
        )
        replayed = self._append(
            expected_current=PublicationStage.EVALUATING,
            stage=PublicationStage.EVALUATED,
            artifacts=artifacts,
            cardinalities=cardinalities,
        )
        self._live_evaluation_claim = None
        self._live_evaluation_permit = None
        return replayed.current

    def record_materialized(
        self,
        *,
        report: ArtifactBinding,
        evidence: ArtifactBinding,
        cardinalities: EvidenceCardinalities,
    ) -> PublicationStateRecord:
        """Bind canonical report/evidence documents and their full inventory."""

        inspection = self.inspect()
        if inspection.current.stage is not PublicationStage.EVALUATED:
            raise PublicationStateConflict(
                "materialized state requires evaluated state"
            )
        if type(report) is not ArtifactBinding:
            raise PublicationStateConflict("materialized state requires report.json")
        _validate_artifact_binding(
            report,
            error_type=PublicationStateConflict,
        )
        if report.name != _REPORT_ARTIFACT_NAME:
            raise PublicationStateConflict("materialized state requires report.json")
        if type(evidence) is not ArtifactBinding:
            raise PublicationStateConflict("materialized state requires evidence.json")
        _validate_artifact_binding(
            evidence,
            error_type=PublicationStateConflict,
        )
        if evidence.name != _EVIDENCE_ARTIFACT_NAME:
            raise PublicationStateConflict("materialized state requires evidence.json")
        if type(cardinalities) is not EvidenceCardinalities:
            raise PublicationStateConflict(
                "materialized cardinalities must be exact EvidenceCardinalities"
            )
        expected = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
        if cardinalities != expected:
            raise PublicationStateConflict(
                "materialized evidence inventory is not locked"
            )
        artifacts = tuple(
            sorted(
                (*inspection.current.artifacts, report, evidence),
                key=lambda binding: binding.name,
            )
        )
        cardinality_values = {
            definition.name: getattr(cardinalities, definition.name)
            for definition in fields(cardinalities)
        }
        cardinality_values["hard_failure_count"] = 0
        bound_cardinalities = tuple(
            CardinalityBinding(name, cardinality_values[name])
            for name in sorted(cardinality_values)
        )
        replayed = self._append(
            expected_current=PublicationStage.EVALUATED,
            stage=PublicationStage.MATERIALIZED,
            artifacts=artifacts,
            cardinalities=bound_cardinalities,
        )
        return replayed.current

    def seal(
        self,
        *,
        manifest: ArtifactBinding,
        rendered_artifacts: tuple[ArtifactBinding, ...],
    ) -> PublicationStateRecord:
        """Bind the final manifest and at least one rendered output."""

        inspection = self.inspect()
        if inspection.current.stage is not PublicationStage.MATERIALIZED:
            raise PublicationStateConflict("sealed state requires materialized state")
        if type(manifest) is not ArtifactBinding:
            raise PublicationStateConflict("sealed state requires manifest.json")
        _validate_artifact_binding(
            manifest,
            error_type=PublicationStateConflict,
        )
        if manifest.name != _MANIFEST_ARTIFACT_NAME:
            raise PublicationStateConflict("sealed state requires manifest.json")
        if (
            type(rendered_artifacts) is not tuple
            or not rendered_artifacts
            or any(type(item) is not ArtifactBinding for item in rendered_artifacts)
        ):
            raise PublicationStateConflict(
                "sealed state requires an exact non-empty artifact tuple"
            )
        for rendered_artifact in rendered_artifacts:
            _validate_artifact_binding(
                rendered_artifact,
                error_type=PublicationStateConflict,
            )
        if any(
            not artifact.name.startswith("artifacts/")
            for artifact in rendered_artifacts
        ):
            raise PublicationStateConflict(
                "rendered artifacts must be beneath artifacts/"
            )
        artifacts = tuple(
            sorted(
                (*inspection.current.artifacts, manifest, *rendered_artifacts),
                key=lambda binding: binding.name,
            )
        )
        if len({artifact.name for artifact in artifacts}) != len(artifacts):
            raise PublicationStateConflict("sealed artifact names must be unique")
        replayed = self._append(
            expected_current=PublicationStage.MATERIALIZED,
            stage=PublicationStage.SEALED,
            artifacts=artifacts,
            cardinalities=inspection.current.cardinalities,
        )
        return replayed.current

    def close(self) -> None:
        """Release every held descriptor; the advisory lock releases last."""

        if self._closed:
            return
        self._closed = True
        self._live_evaluation_claim = None
        self._live_evaluation_permit = None
        owner_pid = getattr(self, "_owner_pid", None)
        inherited_after_fork = type(owner_pid) is not int or owner_pid != os.getpid()
        first_error: OSError | None = None
        for descriptor in (
            self._state_descriptor,
            self._run_descriptor,
            self._parent_descriptor,
        ):
            try:
                os.close(descriptor)
            except OSError as error:
                if first_error is None:
                    first_error = error
        if not inherited_after_fork:
            try:
                fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            except OSError as error:
                if first_error is None:
                    first_error = error
        try:
            os.close(self._lock_descriptor)
        except OSError as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise PublicationStateIOError(
                "cannot completely close publication state store"
            ) from first_error
