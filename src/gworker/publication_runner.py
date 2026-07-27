"""Fail-closed coordinator for the single locked publication evaluation.

The runner owns no configurable experiment surface.  It always targets the
literal ``synthetic-eval-v3-eval`` key beneath the repository's private
``.gworker/publication`` directory and calls the evaluator with
``DEFAULT_EXPERIMENT_CONFIG`` only.

Evaluation is deliberately irreversible.  ``EVALUATING`` is durable before
the private permit is used; reopening at that stage reports a burned run and
never retries it.  Result, report, and evidence artifacts are published with
Linux descriptor-relative operations, verified from independently reopened
bytes, and only then bound into the append-only publication state.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, Final, NoReturn, cast

from . import codec as _codec_source
from . import domain as _domain_source
from . import evaluation as _evaluation_source
from . import evidence as _evidence_source
from . import policy as _policy_source
from . import publication_codec as _publication_codec_source
from . import publication_state as _publication_state_source
from . import reporting as _reporting_source
from . import resource_preflight as _resource_preflight_source
from . import result_codec as _result_codec_source
from . import storage as _storage_source
from .evaluation import (
    DEFAULT_EXPERIMENT_CONFIG,
    LOCKED_DESIGN_ID,
    LOCKED_EVALUATION_RUN_KEY,
    LOCKED_EVALUATOR_ID,
    LOCKED_POLICY_ID,
    LOCKED_POPULATION_ID,
    RESULT_SCHEMA_VERSION,
    ExperimentResult,
    _LockedEvaluationPermit,
    evaluator_design_fingerprint,
    evaluator_fingerprint,
    population_fingerprint,
    run_experiment,
    validate_experiment_config,
)
from .evidence import (
    EvidenceCardinalities,
    PublicationEvidence,
    build_publication_evidence,
    expected_publication_cardinalities,
)
from .policy import POLICY_ID
from .publication_codec import (
    CanonicalJsonDocument,
    decode_publication_evidence,
    decode_statistical_report,
    encode_publication_evidence,
    encode_statistical_report,
)
from .publication_state import (
    SOURCE_PROVENANCE_FILE_NAME,
    STATE_SCHEMA_VERSION,
    ArtifactBinding,
    PublicationRunBusy,
    PublicationRunExists,
    PublicationRunInspection,
    PublicationStage,
    PublicationStateConflict,
    PublicationStateCorrupt,
    PublicationStateIOError,
    PublicationStateRecord,
    PublicationStateSecurityError,
    PublicationStateStore,
)
from .reporting import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    SourceFileIdentity,
    SourceProvenance,
    StatisticalReport,
    build_statistical_report,
    capture_source_provenance,
)
from .resource_preflight import (
    CapacityAssessment,
    ResourceCapacityError,
    assess_publication_capacity,
    capture_resource_snapshot,
    require_publication_capacity,
)
from .result_codec import read_experiment_result, write_experiment_result

_EXPECTED_RUN_KEY: Final = "synthetic-eval-v3-eval"
_EXPECTED_EVALUATOR_ID: Final = (
    "synthetic-eval-v3.caf02b5aced57470dfa96c2952df7402dbd5944b7d394a5a95729ef3e6d09045"
)
_EXPECTED_DESIGN_ID: Final = (
    "synthetic-eval-v3-design."
    "503fe85fb1ff862383812f74eb9b6ce89fdc2a7811f44a83d53ca42c1634ebfd"
)
_EXPECTED_POPULATION_ID: Final = (
    "synthetic-eval-v3-population."
    "dc3f496427d8b78e742eed763ad494902660e50aa4ad45000008854edcaaab83"
)
_EXPECTED_POLICY_ID: Final = "hierarchical-softmax-ucb-v1.8c10875dd38a025d"
_EXPECTED_RESULT_SCHEMA: Final = "synthetic-eval-result-v1"
_EXPECTED_BOOTSTRAP_RESAMPLES: Final = 5_000

_PRIVATE_DIRECTORY_MODE: Final = 0o700
_PRIVATE_FILE_MODE: Final = 0o600
_PLATFORM_PATH_TYPE: Final = type(Path())
_PATH_RAW_COMPONENTS_SLOT: Final = (
    "_raw_paths" if hasattr(_PLATFORM_PATH_TYPE(), "_raw_paths") else "_parts"
)
_RESULT_NAME: Final = "result.bin"
_REPORT_NAME: Final = "report.json"
_EVIDENCE_NAME: Final = "evidence.json"
_PROVENANCE_NAME: Final = SOURCE_PROVENANCE_FILE_NAME
_MANIFEST_NAME: Final = "manifest.json"
_STATE_ENTRIES: Final = frozenset({"lock", "states"})
_TOP_LEVEL_MATERIALIZED: Final = frozenset({_RESULT_NAME, _REPORT_NAME, _EVIDENCE_NAME})
_MAX_RESULT_BYTES: Final = 16 * 1024 * 1024 * 1024
_MAX_REPORT_BYTES: Final = 2 * 1024 * 1024
_MAX_EVIDENCE_BYTES: Final = 16 * 1024 * 1024
_MAX_PROVENANCE_BYTES: Final = 64 * 1024
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_RENAME_NOREPLACE: Final = 1

_RUNTIME_MODULES: Final[tuple[tuple[str, ModuleType], ...]] = (
    ("src/gworker/__init__.py", sys.modules[__package__]),
    ("src/gworker/codec.py", _codec_source),
    ("src/gworker/domain.py", _domain_source),
    ("src/gworker/evaluation.py", _evaluation_source),
    ("src/gworker/evidence.py", _evidence_source),
    ("src/gworker/policy.py", _policy_source),
    ("src/gworker/publication_codec.py", _publication_codec_source),
    ("src/gworker/publication_state.py", _publication_state_source),
    ("src/gworker/reporting.py", _reporting_source),
    ("src/gworker/resource_preflight.py", _resource_preflight_source),
    ("src/gworker/result_codec.py", _result_codec_source),
    ("src/gworker/storage.py", _storage_source),
)
_RUNNER_SOURCE_PATH: Final = "src/gworker/publication_runner.py"
_EXPECTED_SOURCE_PATHS: Final = tuple(
    sorted((*tuple(path for path, _module in _RUNTIME_MODULES), _RUNNER_SOURCE_PATH))
)


class PublicationRunnerError(RuntimeError):
    """Base class for deterministic publication-runner failures."""


class PublicationRunnerSecurityError(PublicationRunnerError):
    """Raised when a repository or artifact path is unsafe."""


class PublicationRunnerIOError(PublicationRunnerError):
    """Raised when durable publication I/O cannot be completed."""


class PublicationRunnerConflict(PublicationRunnerError):
    """Raised when immutable publication bytes conflict."""


class PublicationRunBurned(PublicationRunnerError):
    """Raised when a durable evaluation claim has no bound result."""


@dataclass(frozen=True, slots=True)
class PublicationRunnerStatus:
    """Path-free public status for the fixed publication run."""

    stage: str
    disposition: str
    artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.stage) is not str
            or type(self.disposition) is not str
            or type(self.artifacts) is not tuple
            or any(type(item) is not str for item in self.artifacts)
        ):
            raise PublicationRunnerError("runner status is malformed")

    def as_dict(self) -> dict[str, object]:
        """Return deterministic CLI-safe fields without filesystem paths."""

        return {
            "artifacts": list(self.artifacts),
            "disposition": self.disposition,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class PublicationPreflight:
    """Redacted resource preflight summary safe for CLI output."""

    ready: bool
    failure_codes: tuple[str, ...]
    effective_memory_bytes: int
    effective_swap_bytes: int
    filesystem_available_bytes: int
    filesystem_available_inodes: int
    nofile_soft_limit: int | None

    def as_dict(self) -> dict[str, object]:
        """Return counters and stable failure codes, never cgroup paths."""

        return {
            "effective_memory_bytes": self.effective_memory_bytes,
            "effective_swap_bytes": self.effective_swap_bytes,
            "failure_codes": list(self.failure_codes),
            "filesystem_available_bytes": self.filesystem_available_bytes,
            "filesystem_available_inodes": self.filesystem_available_inodes,
            "nofile_soft_limit": self.nofile_soft_limit,
            "ready": self.ready,
        }


@dataclass(frozen=True, slots=True)
class _ArtifactObservation:
    identity: tuple[int, int]
    byte_count: int
    content_sha256: str


class _DescriptorWriter:
    """Unbuffered exact-write adapter used by the streaming result codec."""

    __slots__ = ("_descriptor",)

    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    def write(self, payload: bytes) -> int:
        if type(payload) is not bytes:
            raise PublicationRunnerIOError("artifact write requires exact bytes")
        total = 0
        while total < len(payload):
            try:
                written = os.write(self._descriptor, payload[total:])
            except OSError as error:
                raise PublicationRunnerIOError(
                    "cannot write publication artifact"
                ) from error
            remaining = len(payload) - total
            if type(written) is not int or not 0 < written <= remaining:
                raise PublicationRunnerIOError(
                    "publication artifact write made no progress"
                )
            total += written
        return total


class _DescriptorReader:
    """Unbuffered descriptor adapter used by the streaming result codec."""

    __slots__ = ("_descriptor",)

    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    def read(self, size: int = -1) -> bytes:
        if type(size) is not int or size < 0:
            raise PublicationRunnerIOError("publication artifact reads must be bounded")
        try:
            return os.read(self._descriptor, size)
        except OSError as error:
            raise PublicationRunnerIOError(
                "cannot read publication artifact"
            ) from error


def _directory_flags() -> int:
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise PublicationRunnerSecurityError(
            "publication runner requires Linux directory flags"
        )
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _file_flags(*, create: bool = False) -> int:
    required = ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required):
        raise PublicationRunnerSecurityError(
            "publication runner requires Linux file flags"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    if create:
        flags = (
            os.O_WRONLY
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
            | os.O_NONBLOCK
            | os.O_CREAT
            | os.O_EXCL
        )
    return flags


def _pending_update_flags() -> int:
    required = ("O_APPEND", "O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required):
        raise PublicationRunnerSecurityError(
            "publication runner requires Linux append flags"
        )
    return os.O_RDWR | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _close_after_failure(descriptor: int) -> None:
    with suppress(OSError):
        os.close(descriptor)


def _close_descriptor(descriptor: int, field: str) -> None:
    try:
        os.close(descriptor)
    except OSError as error:
        raise PublicationRunnerIOError(f"cannot close {field}") from error


def _close_descriptors(
    descriptors: tuple[tuple[int, str], ...],
    *,
    retained_on_failure: int = -1,
) -> None:
    active_error = sys.exc_info()[0] is not None
    first_error: PublicationRunnerIOError | None = None
    for descriptor, field in descriptors:
        if descriptor < 0:
            continue
        try:
            _close_descriptor(descriptor, field)
        except PublicationRunnerIOError as error:
            if first_error is None:
                first_error = error
    if first_error is not None and not active_error:
        if retained_on_failure >= 0:
            _close_after_failure(retained_on_failure)
        raise first_error


def _fsync(descriptor: int, field: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise PublicationRunnerIOError(f"cannot fsync {field}") from error


def _fstat(descriptor: int, field: str) -> os.stat_result:
    try:
        return os.fstat(descriptor)
    except OSError as error:
        raise PublicationRunnerIOError(f"cannot inspect open {field}") from error


def _stat_at(directory: int, name: str, field: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory, follow_symlinks=False)
    except OSError as error:
        raise PublicationRunnerSecurityError(f"cannot inspect {field}") from error


def _validate_private_directory(
    metadata: os.stat_result,
    field: str,
) -> tuple[int, int]:
    if not stat.S_ISDIR(metadata.st_mode):
        raise PublicationRunnerSecurityError(f"{field} must be a directory")
    if metadata.st_uid != os.geteuid():
        raise PublicationRunnerSecurityError(f"{field} has the wrong owner")
    if stat.S_IMODE(metadata.st_mode) != _PRIVATE_DIRECTORY_MODE:
        raise PublicationRunnerSecurityError(f"{field} mode must be 0700")
    return metadata.st_dev, metadata.st_ino


def _validate_private_file(
    metadata: os.stat_result,
    field: str,
) -> tuple[int, int]:
    if not stat.S_ISREG(metadata.st_mode):
        raise PublicationRunnerSecurityError(f"{field} must be a regular file")
    if metadata.st_uid != os.geteuid():
        raise PublicationRunnerSecurityError(f"{field} has the wrong owner")
    if stat.S_IMODE(metadata.st_mode) != _PRIVATE_FILE_MODE:
        raise PublicationRunnerSecurityError(f"{field} mode must be 0600")
    if metadata.st_nlink != 1:
        raise PublicationRunnerSecurityError(f"{field} must not have hard links")
    if metadata.st_size < 0:
        raise PublicationRunnerSecurityError(f"{field} size is invalid")
    return metadata.st_dev, metadata.st_ino


def _open_directory_at(
    parent: int,
    name: str,
    field: str,
    *,
    private: bool,
) -> int:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent)
    except OSError as error:
        raise PublicationRunnerSecurityError(f"cannot open {field}") from error
    try:
        metadata = _fstat(descriptor, field)
        if private:
            identity = _validate_private_directory(metadata, field)
        else:
            if not stat.S_ISDIR(metadata.st_mode):
                raise PublicationRunnerSecurityError(f"{field} must be a directory")
            identity = (metadata.st_dev, metadata.st_ino)
        path_metadata = _stat_at(parent, name, field)
        path_identity = (
            _validate_private_directory(path_metadata, field)
            if private
            else (path_metadata.st_dev, path_metadata.st_ino)
        )
        if not stat.S_ISDIR(path_metadata.st_mode) or path_identity != identity:
            raise PublicationRunnerSecurityError(f"{field} identity changed")
        return descriptor
    except BaseException:
        _close_after_failure(descriptor)
        raise


def _open_repo_root(repo_root: Path) -> int:
    try:
        return os.open(repo_root, _directory_flags())
    except OSError as error:
        raise PublicationRunnerSecurityError("cannot open repository root") from error


def _canonical_repo_root(repo_root: Path) -> Path:
    if type(repo_root) is not _PLATFORM_PATH_TYPE:
        raise PublicationRunnerSecurityError("repository root must be an exact Path")
    try:
        raw_components = object.__getattribute__(
            repo_root,
            _PATH_RAW_COMPONENTS_SLOT,
        )
    except AttributeError as error:
        raise PublicationRunnerSecurityError(
            "repository root must be a canonical absolute Path"
        ) from error
    if type(raw_components) is not list:
        raise PublicationRunnerSecurityError(
            "repository root must be a canonical absolute Path"
        )
    components = tuple(raw_components)
    if any(type(component) is not str for component in components):
        raise PublicationRunnerSecurityError(
            "repository root must be a canonical absolute Path"
        )
    candidate = _PLATFORM_PATH_TYPE(*components)
    raw = candidate.as_posix()
    if (
        not candidate.is_absolute()
        or raw.startswith("//")
        or ".." in candidate.parts
        or any(
            ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in raw
        )
    ):
        raise PublicationRunnerSecurityError(
            "repository root must be canonical and absolute"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PublicationRunnerSecurityError(
            "repository root does not exist"
        ) from error
    if resolved != candidate:
        raise PublicationRunnerSecurityError(
            "repository root must not traverse symlinks"
        )
    descriptor = _open_repo_root(candidate)
    _close_descriptor(descriptor, "repository root")
    return candidate


def _entry_metadata_or_none(directory: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise PublicationRunnerSecurityError(
            "cannot inspect publication entry"
        ) from error


def _open_existing_run_directory(repo_root: Path) -> int | None:
    root_descriptor = _open_repo_root(repo_root)
    gworker_descriptor = -1
    publication_descriptor = -1
    run_descriptor = -1
    try:
        if _entry_metadata_or_none(root_descriptor, ".gworker") is None:
            return None
        gworker_descriptor = _open_directory_at(
            root_descriptor,
            ".gworker",
            "private state directory",
            private=True,
        )
        if _entry_metadata_or_none(gworker_descriptor, "publication") is None:
            return None
        publication_descriptor = _open_directory_at(
            gworker_descriptor,
            "publication",
            "publication parent directory",
            private=True,
        )
        if _entry_metadata_or_none(publication_descriptor, _EXPECTED_RUN_KEY) is None:
            return None
        run_descriptor = _open_directory_at(
            publication_descriptor,
            _EXPECTED_RUN_KEY,
            "publication run directory",
            private=True,
        )
        return run_descriptor
    finally:
        _close_descriptors(
            (
                (publication_descriptor, "publication parent directory"),
                (gworker_descriptor, "private state directory"),
                (root_descriptor, "repository root"),
            ),
            retained_on_failure=run_descriptor,
        )


def _mkdir_private_at(parent: int, name: str, field: str) -> int:
    metadata = _entry_metadata_or_none(parent, name)
    if metadata is None:
        try:
            os.mkdir(name, _PRIVATE_DIRECTORY_MODE, dir_fd=parent)
        except FileExistsError:
            pass
        except OSError as error:
            raise PublicationRunnerIOError(f"cannot create {field}") from error
    descriptor = _open_directory_at(parent, name, field, private=True)
    try:
        _fsync(parent, f"{field} parent")
    except BaseException:
        _close_after_failure(descriptor)
        raise
    return descriptor


def _ensure_publication_parent(repo_root: Path) -> Path:
    root_descriptor = _open_repo_root(repo_root)
    gworker_descriptor = -1
    publication_descriptor = -1
    try:
        gworker_descriptor = _mkdir_private_at(
            root_descriptor,
            ".gworker",
            "private state directory",
        )
        publication_descriptor = _mkdir_private_at(
            gworker_descriptor,
            "publication",
            "publication parent directory",
        )
    finally:
        _close_descriptors(
            (
                (publication_descriptor, "publication parent directory"),
                (gworker_descriptor, "private state directory"),
                (root_descriptor, "repository root"),
            )
        )
    return repo_root / ".gworker" / "publication"


def _create_provenance_run_directory(repo_root: Path) -> int:
    root_descriptor = _open_repo_root(repo_root)
    gworker_descriptor = -1
    publication_descriptor = -1
    run_descriptor = -1
    try:
        gworker_descriptor = _open_directory_at(
            root_descriptor,
            ".gworker",
            "private state directory",
            private=True,
        )
        publication_descriptor = _open_directory_at(
            gworker_descriptor,
            "publication",
            "publication parent directory",
            private=True,
        )
        try:
            os.mkdir(
                _EXPECTED_RUN_KEY,
                _PRIVATE_DIRECTORY_MODE,
                dir_fd=publication_descriptor,
            )
        except FileExistsError:
            pass
        except OSError as error:
            raise PublicationRunnerIOError(
                "cannot create publication run directory"
            ) from error
        _fsync(publication_descriptor, "publication parent directory")
        run_descriptor = _open_directory_at(
            publication_descriptor,
            _EXPECTED_RUN_KEY,
            "publication run directory",
            private=True,
        )
        return run_descriptor
    finally:
        _close_descriptors(
            (
                (publication_descriptor, "publication parent directory"),
                (gworker_descriptor, "private state directory"),
                (root_descriptor, "repository root"),
            ),
            retained_on_failure=run_descriptor,
        )


def _run_path(repo_root: Path) -> Path:
    return repo_root / ".gworker" / "publication" / _EXPECTED_RUN_KEY


def _module_file(module: ModuleType, field: str) -> Path:
    location = getattr(module, "__file__", None)
    if type(location) is not str:
        raise PublicationRunnerSecurityError(f"{field} has no source file")
    return Path(location)


def _loaded_sources() -> tuple[tuple[str, Path], ...]:
    runner_location = globals().get("__file__")
    if type(runner_location) is not str:
        raise PublicationRunnerSecurityError("publication runner has no source file")
    sources = [(path, _module_file(module, path)) for path, module in _RUNTIME_MODULES]
    sources.append((_RUNNER_SOURCE_PATH, Path(runner_location)))
    return tuple(sorted(sources, key=lambda item: item[0]))


def _validate_locked_runtime() -> None:
    literal = (
        (LOCKED_EVALUATION_RUN_KEY, _EXPECTED_RUN_KEY, "run key"),
        (LOCKED_EVALUATOR_ID, _EXPECTED_EVALUATOR_ID, "evaluator"),
        (LOCKED_DESIGN_ID, _EXPECTED_DESIGN_ID, "design"),
        (LOCKED_POPULATION_ID, _EXPECTED_POPULATION_ID, "population"),
        (LOCKED_POLICY_ID, _EXPECTED_POLICY_ID, "locked policy"),
        (POLICY_ID, _EXPECTED_POLICY_ID, "runtime policy"),
        (RESULT_SCHEMA_VERSION, _EXPECTED_RESULT_SCHEMA, "result schema"),
        (
            SOURCE_PROVENANCE_FILE_NAME,
            "source-provenance.json",
            "source provenance filename",
        ),
        (
            STATE_SCHEMA_VERSION,
            "gworker-publication-state-v2",
            "publication state schema",
        ),
    )
    if any(actual != expected for actual, expected, _field in literal):
        failed = next(
            field for actual, expected, field in literal if actual != expected
        )
        raise PublicationRunnerConflict(f"loaded locked {failed} identity is invalid")
    if DEFAULT_BOOTSTRAP_RESAMPLES != _EXPECTED_BOOTSTRAP_RESAMPLES:
        raise PublicationRunnerConflict("loaded bootstrap count is invalid")
    try:
        validate_experiment_config(DEFAULT_EXPERIMENT_CONFIG)
    except Exception as error:
        raise PublicationRunnerConflict(
            "loaded experiment configuration is invalid"
        ) from error
    if (
        DEFAULT_EXPERIMENT_CONFIG.split != "eval"
        or evaluator_fingerprint(DEFAULT_EXPERIMENT_CONFIG) != _EXPECTED_EVALUATOR_ID
        or evaluator_design_fingerprint() != _EXPECTED_DESIGN_ID
        or population_fingerprint(DEFAULT_EXPERIMENT_CONFIG) != _EXPECTED_POPULATION_ID
    ):
        raise PublicationRunnerConflict("loaded locked identities disagree")


def _capture_locked_source(repo_root: Path) -> SourceProvenance:
    _validate_locked_runtime()
    provenance = capture_source_provenance(
        repo_root,
        loaded_sources=_loaded_sources(),
    )
    actual_paths = tuple(source.relative_path for source in provenance.loaded_sources)
    if actual_paths != _EXPECTED_SOURCE_PATHS:
        raise PublicationRunnerConflict("loaded source inventory is invalid")
    return provenance


def _source_provenance_payload(
    provenance: SourceProvenance,
) -> dict[str, object]:
    return {
        "author_email": provenance.author_email,
        "author_name": provenance.author_name,
        "branch": provenance.branch,
        "clean_pre_run": provenance.clean_pre_run,
        "committer_email": provenance.committer_email,
        "committer_name": provenance.committer_name,
        "git_object_format": provenance.git_object_format,
        "loaded_sources": [
            {
                "relative_path": source.relative_path,
                "sha256": source.sha256,
            }
            for source in provenance.loaded_sources
        ],
        "platform": provenance.platform,
        "python_cache_tag": provenance.python_cache_tag,
        "python_implementation": provenance.python_implementation,
        "python_version": provenance.python_version,
        "source_archive_sha256": provenance.source_archive_sha256,
        "source_commit": provenance.source_commit,
        "source_commit_time": provenance.source_commit_time,
        "source_tree": provenance.source_tree,
    }


def _encode_source_provenance(provenance: SourceProvenance) -> bytes:
    if type(provenance) is not SourceProvenance:
        raise PublicationRunnerConflict(
            "source provenance must be an exact SourceProvenance"
        )
    try:
        content = json.dumps(
            _source_provenance_payload(provenance),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PublicationRunnerConflict(
            "source provenance cannot be encoded"
        ) from error
    if not 0 < len(content) <= _MAX_PROVENANCE_BYTES:
        raise PublicationRunnerConflict("source provenance byte length is invalid")
    return content


def _decode_source_provenance(content: bytes) -> SourceProvenance:
    if (
        type(content) is not bytes
        or not content
        or len(content) > _MAX_PROVENANCE_BYTES
    ):
        raise PublicationRunnerConflict("source provenance byte length is invalid")
    object_count = 0

    def closed_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal object_count
        object_count += 1
        if object_count > 64:
            raise PublicationRunnerConflict("source provenance has too many objects")
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PublicationRunnerConflict(
                    "source provenance contains duplicate keys"
                )
            result[key] = value
        return result

    def reject_number(_token: str) -> NoReturn:
        raise PublicationRunnerConflict("source provenance contains a number")

    try:
        decoded = json.loads(
            content.decode("ascii"),
            object_pairs_hook=closed_object,
            parse_constant=reject_number,
            parse_float=reject_number,
            parse_int=reject_number,
        )
    except PublicationRunnerConflict:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise PublicationRunnerConflict("source provenance JSON is invalid") from error
    expected_keys = {
        "author_email",
        "author_name",
        "branch",
        "clean_pre_run",
        "committer_email",
        "committer_name",
        "git_object_format",
        "loaded_sources",
        "platform",
        "python_cache_tag",
        "python_implementation",
        "python_version",
        "source_archive_sha256",
        "source_commit",
        "source_commit_time",
        "source_tree",
    }
    if type(decoded) is not dict or set(decoded) != expected_keys:
        raise PublicationRunnerConflict("source provenance fields are invalid")
    document = cast(dict[str, object], decoded)
    text_fields = expected_keys - {"clean_pre_run", "loaded_sources"}
    if any(type(document[name]) is not str for name in text_fields):
        raise PublicationRunnerConflict("source provenance text field is invalid")
    if document["clean_pre_run"] is not True:
        raise PublicationRunnerConflict("source provenance clean flag is invalid")
    loaded_value = document["loaded_sources"]
    if type(loaded_value) is not list or not loaded_value:
        raise PublicationRunnerConflict(
            "source provenance loaded source set is invalid"
        )
    loaded_sources: list[SourceFileIdentity] = []
    for value in loaded_value:
        if (
            type(value) is not dict
            or set(value) != {"relative_path", "sha256"}
            or type(value["relative_path"]) is not str
            or type(value["sha256"]) is not str
        ):
            raise PublicationRunnerConflict(
                "source provenance source identity is invalid"
            )
        try:
            loaded_sources.append(
                SourceFileIdentity(
                    relative_path=value["relative_path"],
                    sha256=value["sha256"],
                )
            )
        except ValueError as error:
            raise PublicationRunnerConflict(
                "source provenance source identity is invalid"
            ) from error
    try:
        provenance = SourceProvenance(
            source_commit=cast(str, document["source_commit"]),
            source_tree=cast(str, document["source_tree"]),
            git_object_format=cast(str, document["git_object_format"]),
            source_archive_sha256=cast(str, document["source_archive_sha256"]),
            source_commit_time=cast(str, document["source_commit_time"]),
            branch=cast(str, document["branch"]),
            author_name=cast(str, document["author_name"]),
            author_email=cast(str, document["author_email"]),
            committer_name=cast(str, document["committer_name"]),
            committer_email=cast(str, document["committer_email"]),
            python_implementation=cast(str, document["python_implementation"]),
            python_version=cast(str, document["python_version"]),
            python_cache_tag=cast(str, document["python_cache_tag"]),
            platform=cast(str, document["platform"]),
            loaded_sources=tuple(loaded_sources),
            clean_pre_run=True,
        )
    except ValueError as error:
        raise PublicationRunnerConflict(
            "source provenance values are invalid"
        ) from error
    if _encode_source_provenance(provenance) != content:
        raise PublicationRunnerConflict("source provenance JSON is not canonical")
    if (
        tuple(source.relative_path for source in provenance.loaded_sources)
        != _EXPECTED_SOURCE_PATHS
    ):
        raise PublicationRunnerConflict("source provenance inventory is invalid")
    return provenance


def _resource_assessment(repo_root: Path) -> CapacityAssessment:
    snapshot = capture_resource_snapshot(repo_root)
    return assess_publication_capacity(snapshot)


def _preflight_from_assessment(
    assessment: CapacityAssessment,
) -> PublicationPreflight:
    snapshot = assessment.snapshot
    return PublicationPreflight(
        ready=assessment.ready,
        failure_codes=tuple(failure.value for failure in assessment.failures),
        effective_memory_bytes=snapshot.effective_available_memory_bytes,
        effective_swap_bytes=snapshot.effective_available_swap_bytes,
        filesystem_available_bytes=snapshot.filesystem.available_bytes,
        filesystem_available_inodes=snapshot.filesystem.available_inodes,
        nofile_soft_limit=snapshot.limits.nofile_soft,
    )


def publication_preflight(repo_root: Path) -> PublicationPreflight:
    """Verify committed source first, then return redacted capacity diagnostics."""

    root = _canonical_repo_root(repo_root)
    _capture_locked_source(root)
    return _preflight_from_assessment(_resource_assessment(root))


def _list_directory(descriptor: int, field: str) -> tuple[str, ...]:
    try:
        entries = os.listdir(descriptor)
    except OSError as error:
        raise PublicationRunnerIOError(f"cannot list {field}") from error
    if any(type(entry) is not str for entry in entries):
        raise PublicationRunnerSecurityError(f"{field} contains an invalid name")
    return tuple(sorted(entries))


def _open_artifact(
    directory: int,
    name: str,
    field: str,
) -> tuple[int, tuple[int, int], int]:
    try:
        descriptor = os.open(name, _file_flags(), dir_fd=directory)
    except OSError as error:
        raise PublicationRunnerSecurityError(f"cannot open {field}") from error
    try:
        metadata = _fstat(descriptor, field)
        identity = _validate_private_file(metadata, field)
        path_identity = _validate_private_file(_stat_at(directory, name, field), field)
        if path_identity != identity:
            raise PublicationRunnerSecurityError(f"{field} identity changed")
        return descriptor, identity, metadata.st_size
    except BaseException:
        _close_after_failure(descriptor)
        raise


def _hash_open_artifact(
    descriptor: int,
    *,
    identity: tuple[int, int],
    initial_size: int,
    maximum_bytes: int,
    field: str,
) -> _ArtifactObservation:
    if initial_size <= 0 or initial_size > maximum_bytes:
        raise PublicationRunnerSecurityError(f"{field} byte length is invalid")
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        try:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
        except OSError as error:
            raise PublicationRunnerIOError(f"cannot read {field}") from error
        if not chunk:
            break
        byte_count += len(chunk)
        if byte_count > maximum_bytes:
            raise PublicationRunnerSecurityError(f"{field} is too large")
        digest.update(chunk)
    final_metadata = _fstat(descriptor, field)
    final_identity = _validate_private_file(final_metadata, field)
    if (
        final_identity != identity
        or final_metadata.st_size != initial_size
        or byte_count != initial_size
    ):
        raise PublicationRunnerSecurityError(f"{field} changed while read")
    return _ArtifactObservation(identity, byte_count, digest.hexdigest())


def _observe_artifact_at(
    directory: int,
    name: str,
    *,
    maximum_bytes: int,
    field: str,
) -> _ArtifactObservation:
    descriptor, identity, size = _open_artifact(directory, name, field)
    try:
        observation = _hash_open_artifact(
            descriptor,
            identity=identity,
            initial_size=size,
            maximum_bytes=maximum_bytes,
            field=field,
        )
    except BaseException:
        _close_after_failure(descriptor)
        raise
    _close_descriptor(descriptor, field)
    if _validate_private_file(_stat_at(directory, name, field), field) != identity:
        raise PublicationRunnerSecurityError(f"{field} changed after read")
    return observation


def _binding_maximum(name: str) -> int:
    pending_names = {
        f".{_RESULT_NAME}.pending": _RESULT_NAME,
        f".{_REPORT_NAME}.pending": _REPORT_NAME,
        f".{_EVIDENCE_NAME}.pending": _EVIDENCE_NAME,
        f".{_PROVENANCE_NAME}.pending": _PROVENANCE_NAME,
    }
    name = pending_names.get(name, name)
    if name == _RESULT_NAME:
        return _MAX_RESULT_BYTES
    if name == _REPORT_NAME:
        return _MAX_REPORT_BYTES
    if name == _EVIDENCE_NAME:
        return _MAX_EVIDENCE_BYTES
    if name == _PROVENANCE_NAME:
        return _MAX_PROVENANCE_BYTES
    return _MAX_RESULT_BYTES


def _open_artifact_parent(
    run_descriptor: int,
    name: str,
) -> tuple[int, str, tuple[int, ...]]:
    components = name.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise PublicationRunnerSecurityError("artifact binding path is unsafe")
    current = run_descriptor
    opened: list[int] = []
    try:
        for component in components[:-1]:
            current = _open_directory_at(
                current,
                component,
                "rendered artifact directory",
                private=True,
            )
            opened.append(current)
        return current, components[-1], tuple(opened)
    except BaseException:
        for descriptor in reversed(opened):
            _close_after_failure(descriptor)
        raise


def _close_opened_directories(descriptors: tuple[int, ...]) -> None:
    first_error: PublicationRunnerIOError | None = None
    for descriptor in reversed(descriptors):
        try:
            _close_descriptor(descriptor, "rendered artifact directory")
        except PublicationRunnerIOError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _verify_binding(run_descriptor: int, binding: ArtifactBinding) -> None:
    parent, basename, opened = _open_artifact_parent(run_descriptor, binding.name)
    try:
        observed = _observe_artifact_at(
            parent,
            basename,
            maximum_bytes=_binding_maximum(binding.name),
            field=f"artifact {binding.name}",
        )
    finally:
        _close_opened_directories(opened)
    if (
        observed.byte_count != binding.byte_count
        or observed.content_sha256 != binding.content_sha256
    ):
        raise PublicationRunnerConflict(
            f"artifact {binding.name} disagrees with durable state"
        )


def _verify_binding_at(
    directory: int,
    basename: str,
    binding: ArtifactBinding,
) -> None:
    observed = _observe_artifact_at(
        directory,
        basename,
        maximum_bytes=_binding_maximum(binding.name),
        field=f"artifact {binding.name}",
    )
    if (
        observed.byte_count != binding.byte_count
        or observed.content_sha256 != binding.content_sha256
    ):
        raise PublicationRunnerConflict(
            f"artifact {binding.name} disagrees with durable state"
        )


def _expected_tree(
    bindings: tuple[ArtifactBinding, ...],
) -> dict[str, object]:
    root: dict[str, object] = {}
    for binding in bindings:
        current = root
        components = binding.name.split("/")
        for component in components[:-1]:
            child = current.setdefault(component, {})
            if type(child) is not dict:
                raise PublicationRunnerConflict("artifact tree has a path collision")
            current = cast(dict[str, object], child)
        if components[-1] in current:
            raise PublicationRunnerConflict("artifact tree repeats a path")
        current[components[-1]] = binding
    return root


def _verify_tree(
    directory: int,
    tree: dict[str, object],
    *,
    root: bool,
) -> None:
    expected = set(tree)
    if root:
        expected.update(_STATE_ENTRIES)
    entries = set(_list_directory(directory, "publication run directory"))
    if entries != expected:
        raise PublicationRunnerConflict(
            "publication run directory contains unknown or trailing entries"
        )
    for name, child in tree.items():
        if type(child) is dict:
            descriptor = _open_directory_at(
                directory,
                name,
                "rendered artifact directory",
                private=True,
            )
            try:
                _verify_tree(
                    descriptor,
                    cast(dict[str, object], child),
                    root=False,
                )
            finally:
                _close_descriptor(descriptor, "rendered artifact directory")
        else:
            if type(child) is not ArtifactBinding:
                raise PublicationRunnerConflict("artifact tree member is invalid")
            _verify_binding_at(directory, name, child)


def _verify_all_bound_artifacts(
    run_descriptor: int,
    record: PublicationStateRecord,
) -> None:
    _verify_tree(run_descriptor, _expected_tree(record.artifacts), root=True)


def _validate_partial_layout(
    run_descriptor: int,
    stage: PublicationStage,
) -> None:
    entries = set(_list_directory(run_descriptor, "publication run directory"))
    extra = entries - _STATE_ENTRIES
    if stage is PublicationStage.PREPARED:
        allowed: tuple[frozenset[str], ...] = (frozenset({_PROVENANCE_NAME}),)
    elif stage is PublicationStage.EVALUATING:
        allowed = (
            frozenset({_PROVENANCE_NAME}),
            frozenset({_PROVENANCE_NAME, _RESULT_NAME}),
            frozenset({_PROVENANCE_NAME, f".{_RESULT_NAME}.pending"}),
        )
    elif stage is PublicationStage.EVALUATED:
        evaluated = frozenset({_PROVENANCE_NAME, _RESULT_NAME})
        allowed = (
            evaluated,
            evaluated | {f".{_REPORT_NAME}.pending"},
            evaluated | {_REPORT_NAME},
            evaluated | {_REPORT_NAME, f".{_EVIDENCE_NAME}.pending"},
            frozenset({*_TOP_LEVEL_MATERIALIZED, _PROVENANCE_NAME}),
        )
    else:
        raise PublicationRunnerConflict("partial layout stage is invalid")
    if frozenset(extra) not in allowed:
        raise PublicationRunnerConflict(
            "publication run directory contains unknown or trailing entries"
        )
    for name in extra:
        maximum = _binding_maximum(name)
        if stage is PublicationStage.EVALUATING and name == f".{_RESULT_NAME}.pending":
            _validate_unbound_pending(
                run_descriptor,
                name,
                maximum_bytes=maximum,
                field="pending result.bin",
            )
            continue
        if stage is PublicationStage.EVALUATED and name in {
            f".{_REPORT_NAME}.pending",
            f".{_EVIDENCE_NAME}.pending",
        }:
            _validate_unbound_pending(
                run_descriptor,
                name,
                maximum_bytes=maximum,
                field=f"artifact {name}",
            )
            continue
        _observe_artifact_at(
            run_descriptor,
            name,
            maximum_bytes=maximum,
            field=f"artifact {name}",
        )


def _status_from_inspection(
    inspection: PublicationRunInspection,
) -> PublicationRunnerStatus:
    stage = inspection.current.stage
    disposition = {
        PublicationStage.PREPARED: "prepared",
        PublicationStage.EVALUATING: "burned",
        PublicationStage.EVALUATED: "materialization-resumable",
        PublicationStage.MATERIALIZED: "ready-for-rendering",
        PublicationStage.SEALED: "sealed",
    }[stage]
    return PublicationRunnerStatus(
        stage=stage.value,
        disposition=disposition,
        artifacts=tuple(binding.name for binding in inspection.current.artifacts),
    )


def _inspect_open_store(
    store: PublicationStateStore,
    run_descriptor: int,
) -> PublicationRunInspection:
    inspection = store.inspect()
    stage = inspection.current.stage
    if stage in {
        PublicationStage.PREPARED,
        PublicationStage.EVALUATING,
        PublicationStage.EVALUATED,
    }:
        _validate_partial_layout(run_descriptor, stage)
        for binding in inspection.current.artifacts:
            _verify_binding(run_descriptor, binding)
    else:
        _verify_all_bound_artifacts(run_descriptor, inspection.current)
    _recorded_source, observed_binding = _read_preserved_source(run_descriptor)
    if observed_binding != _source_binding(inspection.current):
        raise PublicationRunnerConflict(
            "source provenance disagrees with durable state"
        )
    return inspection


def publication_status(repo_root: Path) -> PublicationRunnerStatus:
    """Replay state and verify every bound artifact without changing the run."""

    root = _canonical_repo_root(repo_root)
    _validate_locked_runtime()
    run_descriptor = _open_existing_run_directory(root)
    if run_descriptor is None:
        return PublicationRunnerStatus("unclaimed", "not-started", ())
    try:
        entries = set(_list_directory(run_descriptor, "publication run directory"))
        bootstrap_layouts: tuple[set[str], ...] = (
            set(),
            {f".{_PROVENANCE_NAME}.pending"},
            {_PROVENANCE_NAME},
        )
        if entries in bootstrap_layouts:
            if entries == {f".{_PROVENANCE_NAME}.pending"}:
                _validate_unbound_pending(
                    run_descriptor,
                    f".{_PROVENANCE_NAME}.pending",
                    maximum_bytes=_MAX_PROVENANCE_BYTES,
                    field="pending source provenance",
                )
                stage = "provenance-pending"
            elif entries == {_PROVENANCE_NAME}:
                _read_preserved_source(run_descriptor)
                stage = "provenance-captured"
            else:
                stage = "provenance-bootstrap"
            return PublicationRunnerStatus(
                stage,
                "claim-pending",
                tuple(sorted(entries)),
            )
        if not _STATE_ENTRIES.issubset(entries):
            raise PublicationRunnerConflict(
                "publication run directory is an invalid incomplete claim"
            )
        with PublicationStateStore.open(_run_path(root)) as store:
            inspection = _inspect_open_store(store, run_descriptor)
            return _status_from_inspection(inspection)
    finally:
        _close_descriptor(run_descriptor, "publication run directory")


def _create_pending_artifact(
    run_descriptor: int,
    pending_name: str,
    field: str,
) -> tuple[int, tuple[int, int]]:
    try:
        descriptor = os.open(
            pending_name,
            _file_flags(create=True),
            _PRIVATE_FILE_MODE,
            dir_fd=run_descriptor,
        )
    except FileExistsError:
        raise PublicationRunnerConflict(
            "a previous incomplete artifact write remains"
        ) from None
    except OSError as error:
        raise PublicationRunnerIOError(f"cannot create pending {field}") from error
    try:
        identity = _validate_private_file(_fstat(descriptor, field), field)
        if (
            _validate_private_file(
                _stat_at(run_descriptor, pending_name, field),
                field,
            )
            != identity
        ):
            raise PublicationRunnerSecurityError(f"{field} identity changed")
        return descriptor, identity
    except BaseException:
        _close_after_failure(descriptor)
        raise


def _complete_pending_artifact(
    run_descriptor: int,
    *,
    pending_name: str,
    expected_content: bytes,
    maximum_bytes: int,
    field: str,
) -> tuple[int, int]:
    try:
        descriptor = os.open(
            pending_name,
            _pending_update_flags(),
            dir_fd=run_descriptor,
        )
    except OSError as error:
        raise PublicationRunnerSecurityError(f"cannot reopen {field}") from error
    try:
        metadata = _fstat(descriptor, field)
        identity = _validate_private_file(metadata, field)
        size = metadata.st_size
        if size > len(expected_content) or size > maximum_bytes:
            raise PublicationRunnerConflict(
                f"{field} is not a prefix of regenerated bytes"
            )
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            try:
                chunk = os.read(
                    descriptor,
                    min(remaining, _HASH_CHUNK_BYTES),
                )
            except OSError as error:
                raise PublicationRunnerIOError(f"cannot read {field}") from error
            if not chunk:
                raise PublicationRunnerConflict(f"{field} was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        try:
            trailing = os.read(descriptor, 1)
        except OSError as error:
            raise PublicationRunnerIOError(f"cannot finish reading {field}") from error
        if trailing:
            raise PublicationRunnerConflict(f"{field} grew while read")
        prefix = b"".join(chunks)
        if not expected_content.startswith(prefix):
            raise PublicationRunnerConflict(
                f"{field} is not a prefix of regenerated bytes"
            )
        suffix = expected_content[len(prefix) :]
        if suffix:
            _DescriptorWriter(descriptor).write(suffix)
        _fsync(descriptor, field)
        final_metadata = _fstat(descriptor, field)
        if _validate_private_file(
            final_metadata, field
        ) != identity or final_metadata.st_size != len(expected_content):
            raise PublicationRunnerSecurityError(f"{field} changed while completed")
    except BaseException:
        _close_after_failure(descriptor)
        raise
    _close_descriptor(descriptor, field)
    if (
        _validate_private_file(
            _stat_at(run_descriptor, pending_name, field),
            field,
        )
        != identity
    ):
        raise PublicationRunnerSecurityError(f"{field} changed after completion")
    return identity


def _validate_unbound_pending(
    run_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
    field: str,
) -> None:
    descriptor, _identity, size = _open_artifact(run_descriptor, name, field)
    try:
        if size > maximum_bytes:
            raise PublicationRunnerSecurityError(f"{field} is too large")
    finally:
        _close_descriptor(descriptor, field)


def _rename_noreplace(
    directory: int,
    source_name: str,
    destination_name: str,
) -> None:
    try:
        library = ctypes.CDLL(None, use_errno=True)
    except (OSError, TypeError, ValueError) as error:
        raise PublicationRunnerSecurityError("cannot load Linux renameat2") from error
    function = getattr(library, "renameat2", None)
    if function is None:
        raise PublicationRunnerSecurityError(
            "Linux renameat2 is required for immutable artifact publication"
        )
    try:
        result = int(
            function(
                ctypes.c_int(directory),
                ctypes.c_char_p(source_name.encode("ascii")),
                ctypes.c_int(directory),
                ctypes.c_char_p(destination_name.encode("ascii")),
                ctypes.c_uint(_RENAME_NOREPLACE),
            )
        )
    except (OSError, TypeError, ValueError, ctypes.ArgumentError) as error:
        raise PublicationRunnerIOError("cannot invoke Linux renameat2") from error
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise PublicationRunnerConflict("publication artifact already exists")
        raise PublicationRunnerIOError(
            "cannot atomically publish publication artifact"
        ) from OSError(error_number, os.strerror(error_number))


def _publish_pending(
    run_descriptor: int,
    *,
    pending_name: str,
    final_name: str,
    identity: tuple[int, int],
    field: str,
) -> None:
    if _entry_metadata_or_none(run_descriptor, final_name) is not None:
        raise PublicationRunnerConflict("publication artifact already exists")
    _rename_noreplace(run_descriptor, pending_name, final_name)
    _fsync(run_descriptor, "publication run directory")
    if (
        _validate_private_file(
            _stat_at(run_descriptor, final_name, field),
            field,
        )
        != identity
    ):
        raise PublicationRunnerSecurityError(f"{field} identity changed on publish")


def _read_artifact_bytes(
    run_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
    field: str,
) -> bytes:
    descriptor, identity, size = _open_artifact(run_descriptor, name, field)
    if size <= 0 or size > maximum_bytes:
        _close_after_failure(descriptor)
        raise PublicationRunnerSecurityError(f"{field} byte length is invalid")
    chunks: list[bytes] = []
    remaining = size
    try:
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, _HASH_CHUNK_BYTES))
            except OSError as error:
                raise PublicationRunnerIOError(f"cannot read {field}") from error
            if not chunk:
                raise PublicationRunnerConflict(f"{field} was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        try:
            trailing = os.read(descriptor, 1)
        except OSError as error:
            raise PublicationRunnerIOError(f"cannot finish reading {field}") from error
        if trailing:
            raise PublicationRunnerConflict(f"{field} grew while read")
        final_metadata = _fstat(descriptor, field)
        if (
            _validate_private_file(final_metadata, field) != identity
            or final_metadata.st_size != size
        ):
            raise PublicationRunnerSecurityError(f"{field} changed while read")
    except BaseException:
        _close_after_failure(descriptor)
        raise
    _close_descriptor(descriptor, field)
    return b"".join(chunks)


def _read_preserved_source(
    run_descriptor: int,
) -> tuple[SourceProvenance, ArtifactBinding]:
    content = _read_artifact_bytes(
        run_descriptor,
        _PROVENANCE_NAME,
        maximum_bytes=_MAX_PROVENANCE_BYTES,
        field="artifact source-provenance.json",
    )
    provenance = _decode_source_provenance(content)
    digest = hashlib.sha256(content).hexdigest()
    return provenance, ArtifactBinding(_PROVENANCE_NAME, digest, len(content))


def _preserve_source(
    run_descriptor: int,
    provenance: SourceProvenance,
) -> ArtifactBinding:
    content = _encode_source_provenance(provenance)
    published_binding = _publish_immutable_content(
        run_descriptor,
        name=_PROVENANCE_NAME,
        content=content,
        content_sha256=hashlib.sha256(content).hexdigest(),
        maximum_bytes=_MAX_PROVENANCE_BYTES,
    )
    recorded, reopened_binding = _read_preserved_source(run_descriptor)
    if recorded != provenance or reopened_binding != published_binding:
        raise PublicationRunnerConflict(
            "reopened source provenance differs from captured source"
        )
    return reopened_binding


def _require_preserved_source(
    run_descriptor: int,
    current: SourceProvenance,
    *,
    expected_binding: ArtifactBinding | None,
) -> ArtifactBinding:
    recorded, binding = _read_preserved_source(run_descriptor)
    if recorded != current:
        raise PublicationRunnerConflict(
            "current clean checkout differs from evaluation source provenance"
        )
    if expected_binding is not None and binding != expected_binding:
        raise PublicationRunnerConflict(
            "source provenance disagrees with durable state"
        )
    return binding


def _artifact_matches_content(
    run_descriptor: int,
    name: str,
    content: bytes,
    digest: str,
    maximum_bytes: int,
) -> ArtifactBinding | None:
    metadata = _entry_metadata_or_none(run_descriptor, name)
    if metadata is None:
        return None
    actual = _read_artifact_bytes(
        run_descriptor,
        name,
        maximum_bytes=maximum_bytes,
        field=f"artifact {name}",
    )
    if actual != content or hashlib.sha256(actual).hexdigest() != digest:
        raise PublicationRunnerConflict(
            f"existing artifact {name} differs from regenerated bytes"
        )
    return ArtifactBinding(name, digest, len(actual))


def _publish_immutable_content(
    run_descriptor: int,
    *,
    name: str,
    content: bytes,
    content_sha256: str,
    maximum_bytes: int,
) -> ArtifactBinding:
    if type(name) is not str or name not in {
        _PROVENANCE_NAME,
        _REPORT_NAME,
        _EVIDENCE_NAME,
    }:
        raise PublicationRunnerSecurityError("immutable artifact name is not permitted")
    if (
        type(content) is not bytes
        or not 0 < len(content) <= maximum_bytes
        or hashlib.sha256(content).hexdigest() != content_sha256
    ):
        raise PublicationRunnerConflict(f"canonical {name} content identity is invalid")
    pending_name = f".{name}.pending"
    final_metadata = _entry_metadata_or_none(run_descriptor, name)
    pending_metadata = _entry_metadata_or_none(run_descriptor, pending_name)
    if final_metadata is not None:
        if pending_metadata is not None:
            raise PublicationRunnerConflict(
                f"artifact {name} has a trailing pending file"
            )
        existing = _artifact_matches_content(
            run_descriptor,
            name,
            content,
            content_sha256,
            maximum_bytes,
        )
        assert existing is not None
        _fsync(run_descriptor, "publication run directory")
        return existing

    if pending_metadata is not None:
        identity = _complete_pending_artifact(
            run_descriptor,
            pending_name=pending_name,
            expected_content=content,
            maximum_bytes=maximum_bytes,
            field=f"pending {name}",
        )
    else:
        descriptor, identity = _create_pending_artifact(
            run_descriptor,
            pending_name,
            f"pending {name}",
        )
        try:
            writer = _DescriptorWriter(descriptor)
            written = writer.write(content)
            if written != len(content):
                raise PublicationRunnerIOError(f"pending {name} write was short")
            _fsync(descriptor, f"pending {name}")
            final_pending_metadata = _fstat(descriptor, f"pending {name}")
            if _validate_private_file(
                final_pending_metadata,
                f"pending {name}",
            ) != identity or final_pending_metadata.st_size != len(content):
                raise PublicationRunnerSecurityError(f"pending {name} changed")
        except BaseException:
            _close_after_failure(descriptor)
            raise
        _close_descriptor(descriptor, f"pending {name}")

    _publish_pending(
        run_descriptor,
        pending_name=pending_name,
        final_name=name,
        identity=identity,
        field=f"artifact {name}",
    )
    observed = _observe_artifact_at(
        run_descriptor,
        name,
        maximum_bytes=maximum_bytes,
        field=f"artifact {name}",
    )
    if observed.byte_count != len(content) or observed.content_sha256 != content_sha256:
        raise PublicationRunnerConflict(f"published {name} failed verification")
    actual = _read_artifact_bytes(
        run_descriptor,
        name,
        maximum_bytes=maximum_bytes,
        field=f"artifact {name}",
    )
    if actual != content:
        raise PublicationRunnerConflict(f"published {name} bytes changed")
    return ArtifactBinding(name, observed.content_sha256, observed.byte_count)


def _publish_json_document(
    run_descriptor: int,
    *,
    name: str,
    document: CanonicalJsonDocument,
    maximum_bytes: int,
) -> ArtifactBinding:
    return _publish_immutable_content(
        run_descriptor,
        name=name,
        content=document.content,
        content_sha256=document.content_sha256,
        maximum_bytes=maximum_bytes,
    )


def _read_result(
    run_descriptor: int,
    binding: ArtifactBinding,
) -> ExperimentResult:
    _verify_binding(run_descriptor, binding)
    descriptor, identity, size = _open_artifact(
        run_descriptor,
        _RESULT_NAME,
        "artifact result.bin",
    )
    if size != binding.byte_count or size > _MAX_RESULT_BYTES:
        _close_after_failure(descriptor)
        raise PublicationRunnerConflict("result.bin byte count is invalid")
    try:
        source = cast(BinaryIO, _DescriptorReader(descriptor))
        result = read_experiment_result(source)
        metadata = _fstat(descriptor, "artifact result.bin")
        if (
            _validate_private_file(metadata, "artifact result.bin") != identity
            or metadata.st_size != size
        ):
            raise PublicationRunnerSecurityError("result.bin changed while decoded")
    except BaseException:
        _close_after_failure(descriptor)
        raise
    _close_descriptor(descriptor, "artifact result.bin")
    _verify_binding(run_descriptor, binding)
    _validate_locked_result(result)
    return result


def _validate_locked_result(result: ExperimentResult) -> None:
    if type(result) is not ExperimentResult:
        raise PublicationRunnerConflict(
            "locked evaluator returned a non-canonical result"
        )
    if (
        result.config != DEFAULT_EXPERIMENT_CONFIG
        or result.schema_version != _EXPECTED_RESULT_SCHEMA
        or result.evaluator_id != _EXPECTED_EVALUATOR_ID
        or result.design_id != _EXPECTED_DESIGN_ID
        or result.policy_id != _EXPECTED_POLICY_ID
        or result.hard_failure_count != 0
    ):
        raise PublicationRunnerConflict("locked evaluator result identity is invalid")


def _run_locked_experiment(
    permit: _LockedEvaluationPermit,
) -> ExperimentResult:
    """The only evaluator call site; it exposes no configuration parameter."""

    if type(permit) is not _LockedEvaluationPermit:
        raise PublicationRunnerConflict("locked evaluation permit is invalid")
    return run_experiment(
        DEFAULT_EXPERIMENT_CONFIG,
        _eval_permit=permit,
    )


def _persist_result(
    run_descriptor: int,
    result: ExperimentResult,
) -> ArtifactBinding:
    if _entry_metadata_or_none(run_descriptor, _RESULT_NAME) is not None:
        raise PublicationRunnerConflict("result.bin already exists before evaluation")
    pending_name = f".{_RESULT_NAME}.pending"
    descriptor, identity = _create_pending_artifact(
        run_descriptor,
        pending_name,
        "pending result.bin",
    )
    try:
        destination = cast(BinaryIO, _DescriptorWriter(descriptor))
        codec_payload_sha256 = write_experiment_result(result, destination)
        if (
            type(codec_payload_sha256) is not str
            or len(codec_payload_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in codec_payload_sha256
            )
        ):
            raise PublicationRunnerConflict(
                "result codec returned an invalid payload digest"
            )
        _fsync(descriptor, "pending result.bin")
        metadata = _fstat(descriptor, "pending result.bin")
        if (
            _validate_private_file(metadata, "pending result.bin") != identity
            or not 0 < metadata.st_size <= _MAX_RESULT_BYTES
        ):
            raise PublicationRunnerSecurityError("pending result.bin is invalid")
    except BaseException:
        _close_after_failure(descriptor)
        raise
    _close_descriptor(descriptor, "pending result.bin")
    _publish_pending(
        run_descriptor,
        pending_name=pending_name,
        final_name=_RESULT_NAME,
        identity=identity,
        field="artifact result.bin",
    )
    observed = _observe_artifact_at(
        run_descriptor,
        _RESULT_NAME,
        maximum_bytes=_MAX_RESULT_BYTES,
        field="artifact result.bin",
    )
    decoded = _read_result(
        run_descriptor,
        ArtifactBinding(
            _RESULT_NAME,
            observed.content_sha256,
            observed.byte_count,
        ),
    )
    if decoded != result:
        raise PublicationRunnerConflict(
            "reopened result.bin differs from evaluator output"
        )
    return ArtifactBinding(
        _RESULT_NAME,
        observed.content_sha256,
        observed.byte_count,
    )


def _result_binding(record: PublicationStateRecord) -> ArtifactBinding:
    matches = tuple(
        binding for binding in record.artifacts if binding.name == _RESULT_NAME
    )
    if len(matches) != 1:
        raise PublicationRunnerConflict("durable state has no unique result.bin")
    return matches[0]


def _source_binding(record: PublicationStateRecord) -> ArtifactBinding:
    matches = tuple(
        binding for binding in record.artifacts if binding.name == _PROVENANCE_NAME
    )
    if len(matches) != 1:
        raise PublicationRunnerConflict("durable state has no unique source provenance")
    return matches[0]


def _evaluate_once(
    store: PublicationStateStore,
    run_descriptor: int,
) -> PublicationStateRecord:
    permit = store.begin_evaluation()
    result = _run_locked_experiment(permit)
    _validate_locked_result(result)
    binding = _persist_result(run_descriptor, result)
    record = store.record_evaluated(
        result=binding,
        cluster_summaries=len(result.cluster_summaries),
        abrupt_traces=len(result.abrupt_traces),
        hard_failure_count=result.hard_failure_count,
    )
    _verify_binding(run_descriptor, binding)
    return record


def _verify_json_round_trip(
    run_descriptor: int,
    *,
    report: StatisticalReport,
    evidence: PublicationEvidence,
    report_binding: ArtifactBinding,
    evidence_binding: ArtifactBinding,
) -> None:
    report_bytes = _read_artifact_bytes(
        run_descriptor,
        _REPORT_NAME,
        maximum_bytes=_MAX_REPORT_BYTES,
        field="artifact report.json",
    )
    decoded_report = decode_statistical_report(
        report_bytes,
        expected_content_sha256=report_binding.content_sha256,
        config=DEFAULT_EXPERIMENT_CONFIG,
    )
    if decoded_report != report:
        raise PublicationRunnerConflict(
            "reopened report.json differs from regenerated report"
        )
    evidence_bytes = _read_artifact_bytes(
        run_descriptor,
        _EVIDENCE_NAME,
        maximum_bytes=_MAX_EVIDENCE_BYTES,
        field="artifact evidence.json",
    )
    decoded_evidence = decode_publication_evidence(
        evidence_bytes,
        expected_content_sha256=evidence_binding.content_sha256,
    )
    if decoded_evidence != evidence:
        raise PublicationRunnerConflict(
            "reopened evidence.json differs from regenerated evidence"
        )


def _materialize(
    store: PublicationStateStore,
    run_descriptor: int,
    record: PublicationStateRecord,
) -> PublicationStateRecord:
    result = _read_result(run_descriptor, _result_binding(record))
    report = build_statistical_report(
        result,
        resample_count=_EXPECTED_BOOTSTRAP_RESAMPLES,
    )
    evidence = build_publication_evidence(result)
    if report != evidence.statistics:
        raise PublicationRunnerConflict(
            "independent statistical report disagrees with publication evidence"
        )
    report_document = encode_statistical_report(
        report,
        config=DEFAULT_EXPERIMENT_CONFIG,
    )
    evidence_document = encode_publication_evidence(evidence)
    report_binding = _publish_json_document(
        run_descriptor,
        name=_REPORT_NAME,
        document=report_document,
        maximum_bytes=_MAX_REPORT_BYTES,
    )
    evidence_binding = _publish_json_document(
        run_descriptor,
        name=_EVIDENCE_NAME,
        document=evidence_document,
        maximum_bytes=_MAX_EVIDENCE_BYTES,
    )
    _verify_json_round_trip(
        run_descriptor,
        report=report,
        evidence=evidence,
        report_binding=report_binding,
        evidence_binding=evidence_binding,
    )
    cardinalities = evidence.completeness.actual
    if (
        type(cardinalities) is not EvidenceCardinalities
        or cardinalities
        != expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
    ):
        raise PublicationRunnerConflict(
            "publication evidence cardinalities are invalid"
        )
    materialized = store.record_materialized(
        report=report_binding,
        evidence=evidence_binding,
        cardinalities=cardinalities,
    )
    _verify_all_bound_artifacts(run_descriptor, materialized)
    return materialized


def _double_gate_for_evaluation(repo_root: Path) -> SourceProvenance:
    before = _capture_locked_source(repo_root)
    assessment = _resource_assessment(repo_root)
    require_publication_capacity(assessment.snapshot)
    after = _capture_locked_source(repo_root)
    if after != before:
        raise PublicationRunnerConflict(
            "source provenance changed during resource preflight"
        )
    return after


def _continue_existing(
    repo_root: Path,
    store: PublicationStateStore,
    run_descriptor: int,
) -> PublicationRunnerStatus:
    inspection = _inspect_open_store(store, run_descriptor)
    stage = inspection.current.stage
    if stage is PublicationStage.EVALUATING:
        raise PublicationRunBurned(
            "the durable evaluating claim is burned and cannot be resumed"
        )
    if stage is PublicationStage.PREPARED:
        current_source = _double_gate_for_evaluation(repo_root)
        _require_preserved_source(
            run_descriptor,
            current_source,
            expected_binding=_source_binding(inspection.current),
        )
        recaptured = _capture_locked_source(repo_root)
        _require_preserved_source(
            run_descriptor,
            recaptured,
            expected_binding=_source_binding(inspection.current),
        )
        evaluated = _evaluate_once(store, run_descriptor)
        _materialize(
            store,
            run_descriptor,
            evaluated,
        )
        return _status_from_inspection(store.inspect())
    if stage is PublicationStage.EVALUATED:
        current_source = _capture_locked_source(repo_root)
        _require_preserved_source(
            run_descriptor,
            current_source,
            expected_binding=_source_binding(inspection.current),
        )
        materialized = _materialize(
            store,
            run_descriptor,
            inspection.current,
        )
        _verify_all_bound_artifacts(run_descriptor, materialized)
        return _status_from_inspection(store.inspect())
    current_source = _capture_locked_source(repo_root)
    _require_preserved_source(
        run_descriptor,
        current_source,
        expected_binding=_source_binding(inspection.current),
    )
    return _status_from_inspection(inspection)


def run_publication(repo_root: Path) -> PublicationRunnerStatus:
    """Advance only legal stages of the fixed publication run.

    A fresh or ``PREPARED`` run passes a clean-source gate, a read-only
    resource gate, and the same source gate again before ``EVALUATING`` is
    appended.  ``EVALUATING`` is terminally burned on reopen.  ``EVALUATED``
    may only regenerate and bind deterministic publication documents.
    """

    root = _canonical_repo_root(repo_root)
    _validate_locked_runtime()
    run_descriptor = _open_existing_run_directory(root)
    if run_descriptor is not None:
        try:
            entries = set(_list_directory(run_descriptor, "publication run directory"))
            bootstrap_layouts: tuple[set[str], ...] = (
                set(),
                {f".{_PROVENANCE_NAME}.pending"},
                {_PROVENANCE_NAME},
            )
            if entries in bootstrap_layouts:
                current_source = _double_gate_for_evaluation(root)
                source_binding = _preserve_source(
                    run_descriptor,
                    current_source,
                )
                with PublicationStateStore.initialize(
                    _run_path(root),
                    source_provenance=source_binding,
                ) as store:
                    inspection = _inspect_open_store(store, run_descriptor)
                    if inspection.current.stage is not PublicationStage.PREPARED:
                        raise PublicationStateConflict(
                            "recovered publication run was not prepared"
                        )
                    recaptured = _capture_locked_source(root)
                    _require_preserved_source(
                        run_descriptor,
                        recaptured,
                        expected_binding=_source_binding(inspection.current),
                    )
                    evaluated = _evaluate_once(store, run_descriptor)
                    _materialize(store, run_descriptor, evaluated)
                    return _status_from_inspection(store.inspect())
            if not _STATE_ENTRIES.issubset(entries):
                raise PublicationRunnerConflict(
                    "publication run directory is an invalid incomplete claim"
                )
            with PublicationStateStore.open(_run_path(root)) as store:
                return _continue_existing(root, store, run_descriptor)
        finally:
            _close_descriptor(run_descriptor, "publication run directory")

    current_source = _double_gate_for_evaluation(root)
    _ensure_publication_parent(root)
    created_run_descriptor = _create_provenance_run_directory(root)
    try:
        source_binding = _preserve_source(created_run_descriptor, current_source)
        with PublicationStateStore.initialize(
            _run_path(root),
            source_provenance=source_binding,
        ) as store:
            inspection = _inspect_open_store(store, created_run_descriptor)
            if inspection.current.stage is not PublicationStage.PREPARED:
                raise PublicationStateConflict("fresh publication run was not prepared")
            recaptured = _capture_locked_source(root)
            _require_preserved_source(
                created_run_descriptor,
                recaptured,
                expected_binding=_source_binding(inspection.current),
            )
            evaluated = _evaluate_once(store, created_run_descriptor)
            _materialize(store, created_run_descriptor, evaluated)
            return _status_from_inspection(store.inspect())
    finally:
        _close_descriptor(
            created_run_descriptor,
            "publication run directory",
        )


def _safe_error_payload(error: BaseException) -> tuple[int, dict[str, object]]:
    if isinstance(error, ResourceCapacityError):
        return (
            2,
            {
                "error": "resource-capacity",
                "ok": False,
                **_preflight_from_assessment(error.assessment).as_dict(),
            },
        )
    if isinstance(error, PublicationRunBurned):
        return 3, {"error": "evaluation-burned", "ok": False}
    if isinstance(
        error,
        (PublicationRunnerSecurityError, PublicationStateSecurityError),
    ):
        return 4, {"error": "unsafe-publication-path", "ok": False}
    if isinstance(error, PublicationRunBusy):
        return 5, {"error": "publication-busy", "ok": False}
    if isinstance(
        error,
        (
            PublicationRunnerConflict,
            PublicationStateConflict,
            PublicationStateCorrupt,
            PublicationRunExists,
        ),
    ):
        return 3, {"error": "publication-conflict", "ok": False}
    if isinstance(error, (PublicationRunnerIOError, PublicationStateIOError)):
        return 1, {"error": "publication-io", "ok": False}
    return 1, {"error": "publication-failed", "ok": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gworker.publication_runner",
        description="Inspect or advance the fixed locked publication run.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="canonical absolute repository root (default: current directory)",
    )
    parser.add_argument(
        "command",
        choices=("status", "preflight", "run"),
    )
    return parser


def _write_cli_payload(payload: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic path-free command-line interface."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "status":
            payload = {"ok": True, **publication_status(arguments.repo_root).as_dict()}
            code = 0
        elif arguments.command == "preflight":
            preflight = publication_preflight(arguments.repo_root)
            payload = {"ok": preflight.ready, **preflight.as_dict()}
            code = 0 if preflight.ready else 2
        else:
            payload = {
                "ok": True,
                **run_publication(arguments.repo_root).as_dict(),
            }
            code = 0
    except (
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        code, payload = _safe_error_payload(error)
    _write_cli_payload(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
