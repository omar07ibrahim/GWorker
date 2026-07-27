"""Read-only resource preflight for the single-use publication run.

The publication runner must establish conservative local capacity before it
durably claims the locked evaluation key.  This module only reads Linux
process, cgroup, filesystem, pressure, and rlimit state.  It never reserves,
reclaims, configures, or writes a system resource.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import math
import os
import resource
import stat
import struct
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Final

GIBIBYTE: Final = 1 << 30
MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES: Final = 8 * GIBIBYTE
MIN_PUBLICATION_AVAILABLE_SWAP_BYTES: Final = 1 * GIBIBYTE
MIN_PUBLICATION_AVAILABLE_DISK_BYTES: Final = 8 * GIBIBYTE
MIN_PUBLICATION_AVAILABLE_INODES: Final = 10_000
MIN_PUBLICATION_NOFILE_SOFT_LIMIT: Final = 256

_MAX_RESOURCE_FILE_BYTES: Final = 1 << 20
_MAX_COUNTER: Final = 2**64 - 1
_PLATFORM_PATH_TYPE: Final = type(Path())
_PATH_RAW_COMPONENTS_SLOT: Final = (
    "_raw_paths" if hasattr(_PLATFORM_PATH_TYPE(), "_raw_paths") else "_parts"
)
_CGROUP_NS_INIT_INO: Final = 0xEFFFFFFB
_CLONE_NEWCGROUP: Final = 0x02000000
_NS_GET_NSTYPE: Final = 0xB703
_CGROUP_FILES: Final = (
    "memory.max",
    "memory.current",
    "memory.swap.max",
    "memory.swap.current",
)


class ResourceInputError(ValueError):
    """Raised when an in-memory resource value object is malformed."""


class ResourceSnapshotError(RuntimeError):
    """Raised when the host resource snapshot cannot be captured exactly."""


class ResourceCapacityError(RuntimeError):
    """Raised when a valid snapshot does not satisfy publication gates."""

    def __init__(self, assessment: CapacityAssessment) -> None:
        if type(assessment) is not CapacityAssessment:
            raise ResourceInputError(
                "capacity error requires an exact CapacityAssessment"
            )
        validate_resource_snapshot(assessment.snapshot)
        if assessment.failures != _capacity_failures(assessment.snapshot):
            raise ResourceInputError("capacity assessment was mutated")
        if assessment.ready:
            raise ResourceInputError("capacity error requires a failing assessment")
        self.assessment = assessment
        details = "; ".join(assessment.diagnostics)
        super().__init__(f"publication resource preflight failed: {details}")


def _exact_counter(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ResourceInputError(f"{field} must be an exact integer")
    minimum = 1 if positive else 0
    if not minimum <= value <= _MAX_COUNTER:
        qualifier = "positive" if positive else "non-negative"
        raise ResourceInputError(f"{field} must be a bounded {qualifier} counter")
    return value


def _optional_counter(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _exact_counter(value, field)


def _pressure_average(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ResourceInputError(f"{field} must be an exact finite float")
    if not 0.0 <= value <= 100.0:
        raise ResourceInputError(f"{field} must be in [0, 100]")
    return 0.0 if value == 0.0 else value


def _canonical_cgroup_path(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 4_096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ResourceInputError(f"{field} must be bounded cgroup path text")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or value.startswith("//")
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ResourceInputError(f"{field} must be a canonical absolute cgroup path")
    return value


def _validate_absolute_path(value: object, field: str) -> Path:
    if type(value) is not _PLATFORM_PATH_TYPE:
        raise ResourceInputError(f"{field} must be a canonical absolute Path")
    try:
        raw_components = object.__getattribute__(
            value,
            _PATH_RAW_COMPONENTS_SLOT,
        )
    except AttributeError as error:
        raise ResourceInputError(
            f"{field} must be a canonical absolute Path"
        ) from error
    if type(raw_components) is not list:
        raise ResourceInputError(f"{field} must be a canonical absolute Path")
    components = tuple(raw_components)
    if any(type(component) is not str for component in components):
        raise ResourceInputError(f"{field} must be a canonical absolute Path")
    path = _PLATFORM_PATH_TYPE(*components)
    path_text = path.as_posix()
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path_text.startswith("//")
        or any(
            ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in path_text
        )
    ):
        raise ResourceInputError(f"{field} must be a canonical absolute Path")
    return path


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Host memory and swap counters from ``/proc/meminfo``."""

    total_bytes: int
    available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int

    def __post_init__(self) -> None:
        total = _exact_counter(self.total_bytes, "memory total", positive=True)
        available = _exact_counter(self.available_bytes, "memory available")
        swap_total = _exact_counter(self.swap_total_bytes, "swap total")
        swap_free = _exact_counter(self.swap_free_bytes, "swap free")
        if available > total:
            raise ResourceInputError("available memory exceeds total memory")
        if swap_free > swap_total:
            raise ResourceInputError("free swap exceeds total swap")


@dataclass(frozen=True, slots=True)
class CgroupLevelSnapshot:
    """Memory-controller counters for one cgroup ancestor."""

    path: str
    memory_limit_bytes: int | None
    memory_current_bytes: int
    swap_limit_bytes: int | None
    swap_current_bytes: int

    def __post_init__(self) -> None:
        _canonical_cgroup_path(self.path, "cgroup level path")
        _optional_counter(self.memory_limit_bytes, "cgroup memory limit")
        _exact_counter(self.memory_current_bytes, "cgroup memory current")
        _optional_counter(self.swap_limit_bytes, "cgroup swap limit")
        _exact_counter(self.swap_current_bytes, "cgroup swap current")

    @property
    def memory_headroom_bytes(self) -> int | None:
        """Return finite memory headroom, or ``None`` for an unlimited level."""

        if self.memory_limit_bytes is None:
            return None
        return max(0, self.memory_limit_bytes - self.memory_current_bytes)

    @property
    def swap_headroom_bytes(self) -> int | None:
        """Return finite swap headroom, or ``None`` for an unlimited level."""

        if self.swap_limit_bytes is None:
            return None
        return max(0, self.swap_limit_bytes - self.swap_current_bytes)


@dataclass(frozen=True, slots=True)
class CgroupSnapshot:
    """Controller-enabled ancestor chain for the current cgroup-v2 process."""

    self_path: str
    levels: tuple[CgroupLevelSnapshot, ...]

    def __post_init__(self) -> None:
        self_path = _canonical_cgroup_path(self.self_path, "self cgroup path")
        if type(self.levels) is not tuple:
            raise ResourceInputError("cgroup levels must be an exact tuple")
        if any(type(level) is not CgroupLevelSnapshot for level in self.levels):
            raise ResourceInputError("cgroup levels contain an invalid member")
        if not self.levels:
            if self_path != "/":
                raise ResourceInputError(
                    "only the native root cgroup may omit memory controller levels"
                )
            return
        paths = tuple(level.path for level in self.levels)
        if len(set(paths)) != len(paths) or paths[-1] != self_path:
            raise ResourceInputError("cgroup level path set is invalid")
        if paths[0] != "/" and PurePosixPath(paths[0]).parent.as_posix() != "/":
            raise ResourceInputError(
                "cgroup levels must start at the native root or its direct child"
            )
        for parent, child in pairwise(paths):
            if PurePosixPath(child).parent.as_posix() != parent:
                raise ResourceInputError(
                    "cgroup levels are not an immediate ancestor chain"
                )

    @property
    def memory_headroom_bytes(self) -> int | None:
        """Return the tightest finite memory limit across all ancestors."""

        finite = tuple(
            headroom
            for level in self.levels
            if (headroom := level.memory_headroom_bytes) is not None
        )
        return min(finite) if finite else None

    @property
    def swap_headroom_bytes(self) -> int | None:
        """Return the tightest finite swap limit across all ancestors."""

        finite = tuple(
            headroom
            for level in self.levels
            if (headroom := level.swap_headroom_bytes) is not None
        )
        return min(finite) if finite else None


@dataclass(frozen=True, slots=True)
class FilesystemSnapshot:
    """Capacity visible to the current user at the future publication path."""

    total_bytes: int
    available_bytes: int
    total_inodes: int
    available_inodes: int

    def __post_init__(self) -> None:
        total = _exact_counter(self.total_bytes, "filesystem total", positive=True)
        available = _exact_counter(self.available_bytes, "filesystem available")
        total_inodes = _exact_counter(
            self.total_inodes,
            "filesystem inode total",
            positive=True,
        )
        available_inodes = _exact_counter(
            self.available_inodes,
            "filesystem inode available",
        )
        if available > total:
            raise ResourceInputError("available filesystem bytes exceed total bytes")
        if available_inodes > total_inodes:
            raise ResourceInputError("available inodes exceed total inodes")


@dataclass(frozen=True, slots=True)
class PressureLine:
    """One Linux PSI ``some`` or ``full`` line."""

    avg10: float
    avg60: float
    avg300: float
    total_microseconds: int

    def __post_init__(self) -> None:
        for field in ("avg10", "avg60", "avg300"):
            value = _pressure_average(getattr(self, field), f"pressure {field}")
            object.__setattr__(self, field, value)
        _exact_counter(
            self.total_microseconds,
            "pressure total microseconds",
        )


@dataclass(frozen=True, slots=True)
class PressureResourceSnapshot:
    """Both PSI classes for one resource."""

    some: PressureLine
    full: PressureLine

    def __post_init__(self) -> None:
        if type(self.some) is not PressureLine or type(self.full) is not PressureLine:
            raise ResourceInputError("pressure resource lines have invalid types")


@dataclass(frozen=True, slots=True)
class PressureSnapshot:
    """Host memory and I/O pressure observations."""

    memory: PressureResourceSnapshot
    io: PressureResourceSnapshot

    def __post_init__(self) -> None:
        if (
            type(self.memory) is not PressureResourceSnapshot
            or type(self.io) is not PressureResourceSnapshot
        ):
            raise ResourceInputError("pressure snapshot contains an invalid resource")


@dataclass(frozen=True, slots=True)
class ProcessLimitSnapshot:
    """Soft and hard ``RLIMIT_NOFILE``; ``None`` represents infinity."""

    nofile_soft: int | None
    nofile_hard: int | None

    def __post_init__(self) -> None:
        soft = _optional_counter(self.nofile_soft, "nofile soft limit")
        hard = _optional_counter(self.nofile_hard, "nofile hard limit")
        if soft is None and hard is not None:
            raise ResourceInputError("infinite nofile soft limit exceeds hard limit")
        if soft is not None and hard is not None and soft > hard:
            raise ResourceInputError("nofile soft limit exceeds hard limit")


def _snapshot_integrity_sha256(snapshot: ResourceSnapshot) -> str:
    digest = hashlib.sha256(b"gworker-resource-snapshot-v1\x00")

    def add_integer(value: int) -> None:
        digest.update(value.to_bytes(8, "big"))

    def add_optional_integer(value: int | None) -> None:
        if value is None:
            digest.update(b"\x00")
        else:
            digest.update(b"\x01")
            add_integer(value)

    def add_float(value: float) -> None:
        digest.update(struct.pack(">d", 0.0 if value == 0.0 else value))

    def add_text(value: str) -> None:
        encoded = value.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)

    memory = snapshot.memory
    for value in (
        memory.total_bytes,
        memory.available_bytes,
        memory.swap_total_bytes,
        memory.swap_free_bytes,
    ):
        add_integer(value)

    cgroup = snapshot.cgroup
    add_text(cgroup.self_path)
    add_integer(len(cgroup.levels))
    for level in cgroup.levels:
        add_text(level.path)
        add_optional_integer(level.memory_limit_bytes)
        add_integer(level.memory_current_bytes)
        add_optional_integer(level.swap_limit_bytes)
        add_integer(level.swap_current_bytes)

    filesystem = snapshot.filesystem
    for value in (
        filesystem.total_bytes,
        filesystem.available_bytes,
        filesystem.total_inodes,
        filesystem.available_inodes,
    ):
        add_integer(value)

    for pressure in (snapshot.pressure.memory, snapshot.pressure.io):
        for line in (pressure.some, pressure.full):
            add_float(line.avg10)
            add_float(line.avg60)
            add_float(line.avg300)
            add_integer(line.total_microseconds)

    add_optional_integer(snapshot.limits.nofile_soft)
    add_optional_integer(snapshot.limits.nofile_hard)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Complete read-only publication preflight observation."""

    memory: MemorySnapshot
    cgroup: CgroupSnapshot
    filesystem: FilesystemSnapshot
    pressure: PressureSnapshot
    limits: ProcessLimitSnapshot
    _integrity_sha256: str = dataclass_field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_resource_snapshot_type_shape(self)
        try:
            integrity_sha256 = _snapshot_integrity_sha256(self)
        except (
            AttributeError,
            OverflowError,
            TypeError,
            UnicodeError,
            struct.error,
        ) as error:
            raise ResourceInputError(
                "resource snapshot members are not sealable"
            ) from error
        object.__setattr__(self, "_integrity_sha256", integrity_sha256)

    @property
    def effective_available_memory_bytes(self) -> int:
        """Return host availability clamped by every finite cgroup limit."""

        cgroup_headroom = self.cgroup.memory_headroom_bytes
        if cgroup_headroom is None:
            return self.memory.available_bytes
        return min(self.memory.available_bytes, cgroup_headroom)

    @property
    def effective_available_swap_bytes(self) -> int:
        """Return host free swap clamped by every finite cgroup swap limit."""

        cgroup_headroom = self.cgroup.swap_headroom_bytes
        if cgroup_headroom is None:
            return self.memory.swap_free_bytes
        return min(self.memory.swap_free_bytes, cgroup_headroom)


def _validate_resource_snapshot_type_shape(snapshot: ResourceSnapshot) -> None:
    """Reject incomplete values and runtime subclasses before value operations."""

    def require_exact_int(
        value: object,
        field: str,
        *,
        optional: bool = False,
    ) -> None:
        if optional and value is None:
            return
        if type(value) is not int:
            raise ResourceInputError(
                f"resource snapshot {field} has an invalid scalar type"
            )

    def require_exact_str(value: object, field: str) -> None:
        if type(value) is not str:
            raise ResourceInputError(
                f"resource snapshot {field} has an invalid scalar type"
            )

    def require_exact_float(value: object, field: str) -> None:
        if type(value) is not float:
            raise ResourceInputError(
                f"resource snapshot {field} has an invalid scalar type"
            )

    if type(snapshot) is not ResourceSnapshot:
        raise ResourceInputError("snapshot must be an exact ResourceSnapshot")

    try:
        memory = snapshot.memory
        if type(memory) is not MemorySnapshot:
            raise ResourceInputError("resource snapshot memory has an invalid type")

        cgroup = snapshot.cgroup
        if type(cgroup) is not CgroupSnapshot:
            raise ResourceInputError("resource snapshot cgroup has an invalid type")
        levels = cgroup.levels
        if type(levels) is not tuple:
            raise ResourceInputError(
                "resource snapshot cgroup levels have an invalid type"
            )
        if any(type(level) is not CgroupLevelSnapshot for level in levels):
            raise ResourceInputError(
                "resource snapshot cgroup levels contain an invalid type"
            )

        filesystem = snapshot.filesystem
        if type(filesystem) is not FilesystemSnapshot:
            raise ResourceInputError("resource snapshot filesystem has an invalid type")

        pressure = snapshot.pressure
        if type(pressure) is not PressureSnapshot:
            raise ResourceInputError("resource snapshot pressure has an invalid type")
        memory_pressure = pressure.memory
        if type(memory_pressure) is not PressureResourceSnapshot:
            raise ResourceInputError(
                "resource snapshot memory pressure has an invalid type"
            )
        io_pressure = pressure.io
        if type(io_pressure) is not PressureResourceSnapshot:
            raise ResourceInputError(
                "resource snapshot I/O pressure has an invalid type"
            )
        pressure_lines: list[tuple[str, PressureLine]] = []
        for field, resource_pressure in (
            ("memory", memory_pressure),
            ("I/O", io_pressure),
        ):
            some = resource_pressure.some
            if type(some) is not PressureLine:
                raise ResourceInputError(
                    f"resource snapshot {field} pressure some has an invalid type"
                )
            full = resource_pressure.full
            if type(full) is not PressureLine:
                raise ResourceInputError(
                    f"resource snapshot {field} pressure full has an invalid type"
                )
            pressure_lines.extend(
                (
                    (f"{field} pressure some", some),
                    (f"{field} pressure full", full),
                )
            )

        limits = snapshot.limits
        if type(limits) is not ProcessLimitSnapshot:
            raise ResourceInputError("resource snapshot limits have an invalid type")

        for field, value in (
            ("memory total", memory.total_bytes),
            ("memory available", memory.available_bytes),
            ("swap total", memory.swap_total_bytes),
            ("swap free", memory.swap_free_bytes),
        ):
            require_exact_int(value, field)

        require_exact_str(cgroup.self_path, "self cgroup path")
        for level in levels:
            require_exact_str(level.path, "cgroup level path")
            require_exact_int(
                level.memory_limit_bytes,
                "cgroup memory limit",
                optional=True,
            )
            require_exact_int(
                level.memory_current_bytes,
                "cgroup memory current",
            )
            require_exact_int(
                level.swap_limit_bytes,
                "cgroup swap limit",
                optional=True,
            )
            require_exact_int(
                level.swap_current_bytes,
                "cgroup swap current",
            )

        for field, value in (
            ("filesystem total", filesystem.total_bytes),
            ("filesystem available", filesystem.available_bytes),
            ("filesystem inode total", filesystem.total_inodes),
            ("filesystem inode available", filesystem.available_inodes),
        ):
            require_exact_int(value, field)

        for field, line in pressure_lines:
            require_exact_float(line.avg10, f"{field} avg10")
            require_exact_float(line.avg60, f"{field} avg60")
            require_exact_float(line.avg300, f"{field} avg300")
            require_exact_int(
                line.total_microseconds,
                f"{field} total microseconds",
            )

        require_exact_int(
            limits.nofile_soft,
            "nofile soft limit",
            optional=True,
        )
        require_exact_int(
            limits.nofile_hard,
            "nofile hard limit",
            optional=True,
        )
    except AttributeError as error:
        raise ResourceInputError(
            "resource snapshot type shape is incomplete"
        ) from error


def validate_resource_snapshot(snapshot: ResourceSnapshot) -> None:
    """Reject a malformed or low-level-mutated frozen resource snapshot."""

    _validate_resource_snapshot_type_shape(snapshot)
    try:
        rebuilt_memory = MemorySnapshot(
            total_bytes=snapshot.memory.total_bytes,
            available_bytes=snapshot.memory.available_bytes,
            swap_total_bytes=snapshot.memory.swap_total_bytes,
            swap_free_bytes=snapshot.memory.swap_free_bytes,
        )
        rebuilt_levels = tuple(
            CgroupLevelSnapshot(
                path=level.path,
                memory_limit_bytes=level.memory_limit_bytes,
                memory_current_bytes=level.memory_current_bytes,
                swap_limit_bytes=level.swap_limit_bytes,
                swap_current_bytes=level.swap_current_bytes,
            )
            for level in snapshot.cgroup.levels
        )
        rebuilt_cgroup = CgroupSnapshot(
            self_path=snapshot.cgroup.self_path,
            levels=rebuilt_levels,
        )
        rebuilt_filesystem = FilesystemSnapshot(
            total_bytes=snapshot.filesystem.total_bytes,
            available_bytes=snapshot.filesystem.available_bytes,
            total_inodes=snapshot.filesystem.total_inodes,
            available_inodes=snapshot.filesystem.available_inodes,
        )

        def rebuild_pressure_line(line: PressureLine) -> PressureLine:
            return PressureLine(
                avg10=line.avg10,
                avg60=line.avg60,
                avg300=line.avg300,
                total_microseconds=line.total_microseconds,
            )

        rebuilt_pressure = PressureSnapshot(
            memory=PressureResourceSnapshot(
                some=rebuild_pressure_line(snapshot.pressure.memory.some),
                full=rebuild_pressure_line(snapshot.pressure.memory.full),
            ),
            io=PressureResourceSnapshot(
                some=rebuild_pressure_line(snapshot.pressure.io.some),
                full=rebuild_pressure_line(snapshot.pressure.io.full),
            ),
        )
        rebuilt_limits = ProcessLimitSnapshot(
            nofile_soft=snapshot.limits.nofile_soft,
            nofile_hard=snapshot.limits.nofile_hard,
        )
        rebuilt = ResourceSnapshot(
            memory=rebuilt_memory,
            cgroup=rebuilt_cgroup,
            filesystem=rebuilt_filesystem,
            pressure=rebuilt_pressure,
            limits=rebuilt_limits,
        )
    except (AttributeError, ResourceInputError, TypeError, struct.error) as error:
        raise ResourceInputError("snapshot failed closed validation") from error
    recorded = getattr(snapshot, "_integrity_sha256", None)
    if (
        type(recorded) is not str
        or len(recorded) != 64
        or not recorded.isascii()
        or any(character not in "0123456789abcdef" for character in recorded)
    ):
        raise ResourceInputError("snapshot integrity seal does not match")
    if not hmac.compare_digest(recorded, rebuilt._integrity_sha256):
        raise ResourceInputError("snapshot integrity seal does not match")


@dataclass(frozen=True, slots=True)
class ResourcePaths:
    """Absolute Linux resource paths, injectable only for fixture capture."""

    meminfo: Path
    self_cgroup: Path
    mountinfo: Path
    cgroup_namespace: Path
    cgroup_root: Path
    memory_pressure: Path
    io_pressure: Path

    def __post_init__(self) -> None:
        for field in (
            "meminfo",
            "self_cgroup",
            "mountinfo",
            "cgroup_namespace",
            "cgroup_root",
            "memory_pressure",
            "io_pressure",
        ):
            value = getattr(self, field)
            validated = _validate_absolute_path(value, field)
            object.__setattr__(self, field, validated)


def validate_resource_paths(paths: ResourcePaths) -> ResourcePaths:
    """Return a clean exact copy of fixture/default resource paths."""

    if type(paths) is not ResourcePaths:
        raise ResourceInputError("paths must be an exact ResourcePaths")
    try:
        rebuilt = ResourcePaths(
            meminfo=paths.meminfo,
            self_cgroup=paths.self_cgroup,
            mountinfo=paths.mountinfo,
            cgroup_namespace=paths.cgroup_namespace,
            cgroup_root=paths.cgroup_root,
            memory_pressure=paths.memory_pressure,
            io_pressure=paths.io_pressure,
        )
    except (AttributeError, ResourceInputError, TypeError) as error:
        raise ResourceInputError("paths failed closed validation") from error
    return rebuilt


DEFAULT_RESOURCE_PATHS: Final = ResourcePaths(
    meminfo=Path("/proc/meminfo"),
    self_cgroup=Path("/proc/self/cgroup"),
    mountinfo=Path("/proc/self/mountinfo"),
    cgroup_namespace=Path("/proc/self/ns/cgroup"),
    cgroup_root=Path("/sys/fs/cgroup"),
    memory_pressure=Path("/proc/pressure/memory"),
    io_pressure=Path("/proc/pressure/io"),
)


class CapacityFailure(StrEnum):
    """Stable diagnostic codes in mandatory evaluation order."""

    MEMORY = "memory-headroom"
    SWAP = "swap-headroom"
    DISK = "filesystem-bytes"
    INODES = "filesystem-inodes"
    NOFILE = "nofile-soft-limit"


_CAPACITY_FAILURE_ORDER: Final = tuple(CapacityFailure)


def _capacity_failures(snapshot: ResourceSnapshot) -> tuple[CapacityFailure, ...]:
    failures: list[CapacityFailure] = []
    if (
        snapshot.effective_available_memory_bytes
        < MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES
    ):
        failures.append(CapacityFailure.MEMORY)
    if (
        snapshot.memory.swap_total_bytes > 0
        and snapshot.effective_available_swap_bytes
        < MIN_PUBLICATION_AVAILABLE_SWAP_BYTES
    ):
        failures.append(CapacityFailure.SWAP)
    if snapshot.filesystem.available_bytes < MIN_PUBLICATION_AVAILABLE_DISK_BYTES:
        failures.append(CapacityFailure.DISK)
    if snapshot.filesystem.available_inodes < MIN_PUBLICATION_AVAILABLE_INODES:
        failures.append(CapacityFailure.INODES)
    if (
        snapshot.limits.nofile_soft is not None
        and snapshot.limits.nofile_soft < MIN_PUBLICATION_NOFILE_SOFT_LIMIT
    ):
        failures.append(CapacityFailure.NOFILE)
    return tuple(failures)


@dataclass(frozen=True, slots=True)
class CapacityAssessment:
    """Deterministic pass/fail projection of one immutable snapshot."""

    snapshot: ResourceSnapshot
    failures: tuple[CapacityFailure, ...]

    def __post_init__(self) -> None:
        if type(self.snapshot) is not ResourceSnapshot:
            raise ResourceInputError("assessment snapshot has an invalid type")
        validate_resource_snapshot(self.snapshot)
        if type(self.failures) is not tuple:
            raise ResourceInputError("assessment failures must be an exact tuple")
        if any(type(failure) is not CapacityFailure for failure in self.failures):
            raise ResourceInputError("assessment failures contain an invalid member")
        if (
            len(set(self.failures)) != len(self.failures)
            or tuple(
                failure
                for failure in _CAPACITY_FAILURE_ORDER
                if failure in self.failures
            )
            != self.failures
        ):
            raise ResourceInputError("assessment failures are duplicated or reordered")
        if self.failures != _capacity_failures(self.snapshot):
            raise ResourceInputError(
                "assessment failures do not match the resource snapshot"
            )

    @property
    def ready(self) -> bool:
        """Return whether all literal publication capacity gates pass."""

        return not self.failures

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return stable, precise diagnostics in protocol order."""

        snapshot = self.snapshot
        values = {
            CapacityFailure.MEMORY: (
                snapshot.effective_available_memory_bytes,
                MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES,
            ),
            CapacityFailure.SWAP: (
                snapshot.effective_available_swap_bytes,
                MIN_PUBLICATION_AVAILABLE_SWAP_BYTES,
            ),
            CapacityFailure.DISK: (
                snapshot.filesystem.available_bytes,
                MIN_PUBLICATION_AVAILABLE_DISK_BYTES,
            ),
            CapacityFailure.INODES: (
                snapshot.filesystem.available_inodes,
                MIN_PUBLICATION_AVAILABLE_INODES,
            ),
            CapacityFailure.NOFILE: (
                snapshot.limits.nofile_soft,
                MIN_PUBLICATION_NOFILE_SOFT_LIMIT,
            ),
        }
        diagnostics: list[str] = []
        for failure in self.failures:
            actual, required = values[failure]
            diagnostics.append(
                f"{failure.value}: actual={actual} required-at-least={required}"
            )
        return tuple(diagnostics)


def _read_bounded_descriptor(descriptor: int, field: str) -> str:
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            payload = source.read(_MAX_RESOURCE_FILE_BYTES + 1)
    except OSError as error:
        raise ResourceSnapshotError(f"cannot read {field}") from error
    if len(payload) > _MAX_RESOURCE_FILE_BYTES:
        raise ResourceSnapshotError(f"{field} exceeds the read bound")
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ResourceSnapshotError(f"{field} must be ASCII") from error
    if not text or "\x00" in text:
        raise ResourceSnapshotError(f"{field} is empty or contains NUL")
    return text


def _fstat_descriptor(descriptor: int, field: str) -> os.stat_result:
    try:
        return os.fstat(descriptor)
    except OSError as error:
        raise ResourceSnapshotError(f"cannot inspect opened {field}") from error


def _close_descriptor(descriptor: int, field: str) -> None:
    try:
        os.close(descriptor)
    except OSError as error:
        raise ResourceSnapshotError(f"cannot close {field}") from error


def _close_descriptor_after_failure(descriptor: int) -> None:
    # Preserve the original fail-closed diagnostic after making the
    # best-effort cleanup attempt.
    with suppress(OSError):
        os.close(descriptor)


def _device_identity(device: int) -> tuple[int, int]:
    return os.major(device), os.minor(device)


def _open_initial_cgroup_namespace(path: Path) -> tuple[int, tuple[int, int]]:
    field = "current cgroup namespace"
    try:
        information = os.stat(path, follow_symlinks=True)
    except OSError as error:
        raise ResourceSnapshotError(f"cannot inspect {field}") from error
    if not stat.S_ISREG(information.st_mode):
        raise ResourceSnapshotError(f"{field} handle must be a regular nsfs file")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as error:
        raise ResourceSnapshotError(f"cannot open {field}") from error
    try:
        opened_information = _fstat_descriptor(descriptor, field)
        if (
            not stat.S_ISREG(opened_information.st_mode)
            or opened_information.st_dev != information.st_dev
            or opened_information.st_ino != information.st_ino
        ):
            raise ResourceSnapshotError(f"{field} changed during inspection")
        try:
            namespace_type = fcntl.ioctl(descriptor, _NS_GET_NSTYPE)
        except OSError as error:
            raise ResourceSnapshotError(
                f"{field} handle is not a Linux namespace"
            ) from error
        if namespace_type != _CLONE_NEWCGROUP:
            raise ResourceSnapshotError(f"{field} handle is not a cgroup namespace")
        if opened_information.st_ino != _CGROUP_NS_INIT_INO:
            raise ResourceSnapshotError(
                "publication preflight requires the initial cgroup namespace"
            )
    except BaseException:
        _close_descriptor_after_failure(descriptor)
        raise
    return descriptor, (opened_information.st_dev, opened_information.st_ino)


def _revalidate_initial_cgroup_namespace(
    path: Path,
    held_descriptor: int,
    expected_identity: tuple[int, int],
) -> None:
    repeated_descriptor, repeated_identity = _open_initial_cgroup_namespace(path)
    try:
        held_information = _fstat_descriptor(
            held_descriptor,
            "held cgroup namespace",
        )
        if (
            not stat.S_ISREG(held_information.st_mode)
            or (held_information.st_dev, held_information.st_ino) != expected_identity
            or repeated_identity != expected_identity
        ):
            raise ResourceSnapshotError(
                "current cgroup namespace changed during capture"
            )
    except BaseException:
        _close_descriptor_after_failure(repeated_descriptor)
        raise
    _close_descriptor(repeated_descriptor, "repeated cgroup namespace")


def _read_bounded_ascii(path: Path, field: str) -> str:
    try:
        information = os.lstat(path)
    except OSError as error:
        raise ResourceSnapshotError(f"cannot inspect {field}") from error
    if not stat.S_ISREG(information.st_mode):
        raise ResourceSnapshotError(f"{field} must be a regular read-only source")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        raise ResourceSnapshotError(f"cannot read {field}") from error
    try:
        opened_information = _fstat_descriptor(descriptor, field)
        if (
            not stat.S_ISREG(opened_information.st_mode)
            or opened_information.st_dev != information.st_dev
            or opened_information.st_ino != information.st_ino
        ):
            raise ResourceSnapshotError(f"{field} changed during inspection")
        text = _read_bounded_descriptor(descriptor, field)
    except BaseException:
        _close_descriptor_after_failure(descriptor)
        raise
    _close_descriptor(descriptor, field)
    return text


def _open_bound_directory(path: Path, field: str) -> int:
    try:
        information = os.lstat(path)
    except OSError as error:
        raise ResourceSnapshotError(f"cannot inspect {field}") from error
    if not stat.S_ISDIR(information.st_mode):
        raise ResourceSnapshotError(f"{field} must be a directory")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise ResourceSnapshotError(f"cannot open {field}") from error
    try:
        opened_information = _fstat_descriptor(descriptor, field)
        if (
            not stat.S_ISDIR(opened_information.st_mode)
            or opened_information.st_dev != information.st_dev
            or opened_information.st_ino != information.st_ino
        ):
            raise ResourceSnapshotError(f"{field} changed during inspection")
    except BaseException:
        _close_descriptor_after_failure(descriptor)
        raise
    return descriptor


def _open_bound_directory_at(
    parent_descriptor: int,
    name: str,
    field: str,
    *,
    expected_device: tuple[int, int] | None = None,
) -> int:
    try:
        information = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ResourceSnapshotError(f"cannot inspect {field}") from error
    if not stat.S_ISDIR(information.st_mode):
        raise ResourceSnapshotError(f"{field} must be a directory")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ResourceSnapshotError(f"cannot open {field}") from error
    try:
        opened_information = _fstat_descriptor(descriptor, field)
        if (
            not stat.S_ISDIR(opened_information.st_mode)
            or opened_information.st_dev != information.st_dev
            or opened_information.st_ino != information.st_ino
        ):
            raise ResourceSnapshotError(f"{field} changed during inspection")
        if (
            expected_device is not None
            and _device_identity(opened_information.st_dev) != expected_device
        ):
            raise ResourceSnapshotError(
                f"{field} crosses out of the cgroup2 filesystem"
            )
    except BaseException:
        _close_descriptor_after_failure(descriptor)
        raise
    return descriptor


def _read_bounded_ascii_at(
    directory_descriptor: int,
    name: str,
    field: str,
    *,
    expected_device: tuple[int, int] | None = None,
) -> str:
    try:
        information = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ResourceSnapshotError(f"cannot inspect {field}") from error
    if not stat.S_ISREG(information.st_mode):
        raise ResourceSnapshotError(f"{field} must be a regular read-only source")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise ResourceSnapshotError(f"cannot read {field}") from error
    try:
        opened_information = _fstat_descriptor(descriptor, field)
        if (
            not stat.S_ISREG(opened_information.st_mode)
            or opened_information.st_dev != information.st_dev
            or opened_information.st_ino != information.st_ino
        ):
            raise ResourceSnapshotError(f"{field} changed during inspection")
        if (
            expected_device is not None
            and _device_identity(opened_information.st_dev) != expected_device
        ):
            raise ResourceSnapshotError(
                f"{field} crosses out of the cgroup2 filesystem"
            )
        text = _read_bounded_descriptor(descriptor, field)
    except BaseException:
        _close_descriptor_after_failure(descriptor)
        raise
    _close_descriptor(descriptor, field)
    return text


def _directory_entry_exists(
    directory_descriptor: int,
    name: str,
    field: str,
) -> bool:
    try:
        os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ResourceSnapshotError(f"cannot inspect {field}") from error
    return True


def _decimal_counter(value: str, field: str, *, scale: int = 1) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise ResourceSnapshotError(f"{field} must be an unsigned decimal counter")
    if type(scale) is not int or scale <= 0 or scale > _MAX_COUNTER:
        raise RuntimeError("counter scale must be a bounded positive integer")
    if len(value) > 1 and value.startswith("0"):
        raise ResourceSnapshotError(f"{field} must use canonical decimal notation")
    maximum = _MAX_COUNTER // scale
    maximum_text = str(maximum)
    if len(value) > len(maximum_text) or (
        len(value) == len(maximum_text) and value > maximum_text
    ):
        raise ResourceSnapshotError(f"{field} exceeds the counter bound")
    return int(value) * scale


def _parse_meminfo(text: str) -> MemorySnapshot:
    fields: dict[str, int | None] = {
        "MemTotal": None,
        "MemAvailable": None,
        "SwapTotal": None,
        "SwapFree": None,
    }
    for line in text.splitlines():
        if ":" not in line:
            raise ResourceSnapshotError("proc meminfo contains a malformed line")
        name, raw_value = line.split(":", 1)
        if name not in fields:
            continue
        if fields[name] is not None:
            raise ResourceSnapshotError(f"proc meminfo repeats {name}")
        parts = raw_value.split()
        if len(parts) != 2 or parts[1] != "kB":
            raise ResourceSnapshotError(f"proc meminfo {name} must use kB")
        fields[name] = _decimal_counter(
            parts[0],
            f"proc meminfo {name}",
            scale=1_024,
        )
    missing = tuple(name for name, value in fields.items() if value is None)
    if missing:
        raise ResourceSnapshotError(f"proc meminfo is missing {','.join(missing)}")
    total = fields["MemTotal"]
    available = fields["MemAvailable"]
    swap_total = fields["SwapTotal"]
    swap_free = fields["SwapFree"]
    if total is None or available is None or swap_total is None or swap_free is None:
        raise ResourceSnapshotError("proc meminfo required counter is absent")
    try:
        return MemorySnapshot(
            total_bytes=total,
            available_bytes=available,
            swap_total_bytes=swap_total,
            swap_free_bytes=swap_free,
        )
    except ResourceInputError as error:
        raise ResourceSnapshotError("proc meminfo counters are inconsistent") from error


def _parse_self_cgroup(text: str) -> str:
    matches: list[str] = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            raise ResourceSnapshotError("proc self cgroup contains a malformed line")
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            matches.append(path)
        elif "memory" in controllers.split(","):
            raise ResourceSnapshotError(
                "proc self cgroup contains an unsupported v1 memory membership"
            )
    if len(matches) != 1:
        raise ResourceSnapshotError(
            "proc self cgroup must contain one unified-v2 membership"
        )
    try:
        return _canonical_cgroup_path(matches[0], "self cgroup path")
    except ResourceInputError as error:
        raise ResourceSnapshotError("self cgroup path is invalid") from error


def _decode_mountinfo_path(value: str, field: str) -> str:
    escapes = {
        r"\040": " ",
        r"\011": "\t",
        r"\012": "\n",
        r"\134": "\\",
    }
    decoded: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            decoded.append(value[index])
            index += 1
            continue
        escape = value[index : index + 4]
        if escape not in escapes:
            raise ResourceSnapshotError(f"{field} contains an invalid escape")
        decoded.append(escapes[escape])
        index += 4
    path_text = "".join(decoded)
    path = PurePosixPath(path_text)
    if (
        not path_text
        or len(path_text) > 4_096
        or not path.is_absolute()
        or path_text.startswith("//")
        or path.as_posix() != path_text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ResourceSnapshotError(f"{field} is not a canonical absolute path")
    return path_text


def _parse_cgroup2_mount(
    text: str,
    cgroup_root: Path,
) -> tuple[int, int, int]:
    cgroup2_mounts: list[tuple[int, int, int, str, str]] = []
    mount_records: list[tuple[int, str]] = []
    mount_ids: set[int] = set()
    legacy_cgroup_found = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        try:
            separator = fields.index("-", 6)
        except ValueError as error:
            raise ResourceSnapshotError(
                f"proc self mountinfo line {line_number} is malformed"
            ) from error
        if separator < 6 or len(fields) < separator + 4:
            raise ResourceSnapshotError(
                f"proc self mountinfo line {line_number} is malformed"
            )
        mount_id = _decimal_counter(fields[0], "mountinfo mount id")
        if mount_id in mount_ids:
            raise ResourceSnapshotError("proc self mountinfo repeats a mount id")
        mount_ids.add(mount_id)
        mount_point = _decode_mountinfo_path(
            fields[4],
            f"mountinfo line {line_number} mount point",
        )
        mount_records.append((mount_id, mount_point))
        filesystem_type = fields[separator + 1]
        if filesystem_type == "cgroup":
            legacy_cgroup_found = True
            continue
        if filesystem_type != "cgroup2":
            continue
        device = fields[2].split(":")
        if len(device) != 2:
            raise ResourceSnapshotError("cgroup2 mount device is malformed")
        device_major = _decimal_counter(device[0], "cgroup2 mount device major")
        device_minor = _decimal_counter(device[1], "cgroup2 mount device minor")
        mount_root = _decode_mountinfo_path(
            fields[3],
            "cgroup2 mount root",
        )
        cgroup2_mounts.append(
            (
                mount_id,
                device_major,
                device_minor,
                mount_root,
                mount_point,
            )
        )
    if legacy_cgroup_found:
        raise ResourceSnapshotError(
            "legacy cgroup mount makes the memory hierarchy ambiguous"
        )
    if len(cgroup2_mounts) != 1:
        raise ResourceSnapshotError(
            "proc self mountinfo must contain exactly one cgroup2 mount"
        )
    mount_id, device_major, device_minor, mount_root, mount_point = cgroup2_mounts[0]
    if mount_root != "/":
        raise ResourceSnapshotError(
            "cgroup2 mount does not expose the full ancestor hierarchy"
        )
    if mount_point != cgroup_root.as_posix():
        raise ResourceSnapshotError(
            "cgroup2 mount point does not match the configured cgroup root"
        )
    configured_root = PurePosixPath(mount_point)
    for candidate_id, candidate_text in mount_records:
        if candidate_id == mount_id:
            continue
        candidate = PurePosixPath(candidate_text)
        if candidate == configured_root or configured_root in candidate.parents:
            raise ResourceSnapshotError(
                "a nested mount makes the cgroup2 hierarchy ambiguous"
            )
    return mount_id, device_major, device_minor


def _single_cgroup_value(text: str, field: str) -> str:
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0] or lines[0] != lines[0].strip():
        raise ResourceSnapshotError(f"{field} must contain one canonical value")
    return lines[0]


def _parse_cgroup_limit(text: str, field: str) -> int | None:
    value = _single_cgroup_value(text, field)
    if value == "max":
        return None
    return _decimal_counter(value, field)


def _parse_cgroup_current(text: str, field: str) -> int:
    return _decimal_counter(_single_cgroup_value(text, field), field)


def _cgroup_prefixes(self_path: str) -> tuple[str, ...]:
    path = PurePosixPath(self_path)
    if self_path == "/":
        return ("/",)
    parts = path.parts[1:]
    return (
        "/",
        *(("/" + "/".join(parts[:index])) for index in range(1, len(parts) + 1)),
    )


def _capture_cgroup_hierarchy(paths: ResourcePaths) -> CgroupSnapshot:
    cgroup2_mount = _parse_cgroup2_mount(
        _read_bounded_ascii(paths.mountinfo, "proc self mountinfo"),
        paths.cgroup_root,
    )
    self_path = _parse_self_cgroup(
        _read_bounded_ascii(paths.self_cgroup, "proc self cgroup")
    )
    levels: list[CgroupLevelSnapshot] = []
    directory_descriptor = _open_bound_directory(
        paths.cgroup_root,
        "cgroup root",
    )
    directory_descriptor_open = True
    try:
        root_information = _fstat_descriptor(directory_descriptor, "cgroup root")
        expected_device = (cgroup2_mount[1], cgroup2_mount[2])
        if _device_identity(root_information.st_dev) != expected_device:
            raise ResourceSnapshotError(
                "configured cgroup root does not match the cgroup2 mount device"
            )
        for relative in _cgroup_prefixes(self_path):
            if relative != "/":
                component = PurePosixPath(relative).name
                child_descriptor = _open_bound_directory_at(
                    directory_descriptor,
                    component,
                    f"cgroup ancestor {relative}",
                    expected_device=expected_device,
                )
                directory_descriptor_open = False
                try:
                    os.close(directory_descriptor)
                except OSError as error:
                    _close_descriptor_after_failure(child_descriptor)
                    raise ResourceSnapshotError(
                        "cannot close cgroup ancestor"
                    ) from error
                directory_descriptor = child_descriptor
                directory_descriptor_open = True
            present = tuple(
                _directory_entry_exists(
                    directory_descriptor,
                    name,
                    f"cgroup {name}",
                )
                for name in _CGROUP_FILES
            )
            if not any(present):
                if relative == "/":
                    # The native global cgroup-v2 root intentionally has no
                    # memory controller resource-control files.
                    continue
                raise ResourceSnapshotError(
                    "non-root cgroup ancestor has no memory controller"
                )
            if not all(present):
                raise ResourceSnapshotError(
                    "cgroup memory controller files are incomplete"
                )
            memory_limit, memory_current, swap_limit, swap_current = (
                _read_bounded_ascii_at(
                    directory_descriptor,
                    name,
                    f"cgroup {name}",
                    expected_device=expected_device,
                )
                for name in _CGROUP_FILES
            )
            try:
                levels.append(
                    CgroupLevelSnapshot(
                        path=relative,
                        memory_limit_bytes=_parse_cgroup_limit(
                            memory_limit,
                            "cgroup memory.max",
                        ),
                        memory_current_bytes=_parse_cgroup_current(
                            memory_current,
                            "cgroup memory.current",
                        ),
                        swap_limit_bytes=_parse_cgroup_limit(
                            swap_limit,
                            "cgroup memory.swap.max",
                        ),
                        swap_current_bytes=_parse_cgroup_current(
                            swap_current,
                            "cgroup memory.swap.current",
                        ),
                    )
                )
            except ResourceInputError as error:
                raise ResourceSnapshotError("cgroup counters are invalid") from error
    finally:
        if directory_descriptor_open:
            try:
                os.close(directory_descriptor)
            except OSError as error:
                raise ResourceSnapshotError("cannot close cgroup ancestor") from error
    repeated_self_path = _parse_self_cgroup(
        _read_bounded_ascii(paths.self_cgroup, "proc self cgroup")
    )
    if repeated_self_path != self_path:
        raise ResourceSnapshotError("cgroup membership changed during capture")
    repeated_cgroup2_mount = _parse_cgroup2_mount(
        _read_bounded_ascii(paths.mountinfo, "proc self mountinfo"),
        paths.cgroup_root,
    )
    if repeated_cgroup2_mount != cgroup2_mount:
        raise ResourceSnapshotError("cgroup2 mount changed during capture")
    if (levels and levels[-1].path != self_path) or (not levels and self_path != "/"):
        raise ResourceSnapshotError("current cgroup has no complete memory controller")
    try:
        return CgroupSnapshot(self_path=self_path, levels=tuple(levels))
    except ResourceInputError as error:
        raise ResourceSnapshotError("cgroup hierarchy is inconsistent") from error


def _capture_cgroup(paths: ResourcePaths) -> CgroupSnapshot:
    namespace_descriptor, namespace_identity = _open_initial_cgroup_namespace(
        paths.cgroup_namespace
    )
    try:
        snapshot = _capture_cgroup_hierarchy(paths)
        _revalidate_initial_cgroup_namespace(
            paths.cgroup_namespace,
            namespace_descriptor,
            namespace_identity,
        )
    except BaseException:
        _close_descriptor_after_failure(namespace_descriptor)
        raise
    _close_descriptor(namespace_descriptor, "held cgroup namespace")
    return snapshot


def _parse_pressure_line(line: str, label: str) -> PressureLine:
    parts = line.split()
    if not parts or parts[0] != label:
        raise ResourceSnapshotError(f"PSI {label} line is malformed")
    values: dict[str, str] = {}
    for item in parts[1:]:
        if item.count("=") != 1:
            raise ResourceSnapshotError(f"PSI {label} field is malformed")
        name, value = item.split("=", 1)
        if name in values:
            raise ResourceSnapshotError(f"PSI {label} repeats {name}")
        values[name] = value
    expected = {"avg10", "avg60", "avg300", "total"}
    if set(values) != expected:
        raise ResourceSnapshotError(f"PSI {label} fields are incomplete")
    try:
        averages = tuple(float(values[name]) for name in ("avg10", "avg60", "avg300"))
    except ValueError as error:
        raise ResourceSnapshotError(f"PSI {label} average is invalid") from error
    try:
        return PressureLine(
            avg10=averages[0],
            avg60=averages[1],
            avg300=averages[2],
            total_microseconds=_decimal_counter(
                values["total"],
                f"PSI {label} total",
            ),
        )
    except ResourceInputError as error:
        raise ResourceSnapshotError(f"PSI {label} values are invalid") from error


def _parse_pressure(text: str, field: str) -> PressureResourceSnapshot:
    lines: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        label = parts[0] if parts else ""
        if label not in {"some", "full"}:
            raise ResourceSnapshotError(f"{field} PSI has an unknown line")
        if label in lines:
            raise ResourceSnapshotError(f"{field} PSI repeats {label}")
        lines[label] = line
    if set(lines) != {"some", "full"}:
        raise ResourceSnapshotError(f"{field} PSI requires some and full lines")
    return PressureResourceSnapshot(
        some=_parse_pressure_line(lines["some"], "some"),
        full=_parse_pressure_line(lines["full"], "full"),
    )


def _statvfs_counter(value: object, field: str, *, positive: bool = False) -> int:
    try:
        return _exact_counter(value, field, positive=positive)
    except ResourceInputError as error:
        raise ResourceSnapshotError(f"statvfs {field} is invalid") from error


def _capture_filesystem(capacity_path: Path) -> FilesystemSnapshot:
    descriptor = _open_bound_directory(
        capacity_path,
        "publication capacity path",
    )
    try:
        try:
            values = os.fstatvfs(descriptor)
        except OSError as error:
            raise ResourceSnapshotError(
                "cannot inspect publication filesystem"
            ) from error
    finally:
        try:
            os.close(descriptor)
        except OSError as error:
            raise ResourceSnapshotError(
                "cannot close publication capacity path"
            ) from error
    fragment_size = _statvfs_counter(
        values.f_frsize,
        "fragment size",
        positive=True,
    )
    block_count = _statvfs_counter(values.f_blocks, "block count", positive=True)
    available_blocks = _statvfs_counter(
        values.f_bavail,
        "available block count",
    )
    inode_count = _statvfs_counter(values.f_files, "inode count", positive=True)
    available_inodes = _statvfs_counter(
        values.f_favail,
        "available inode count",
    )
    try:
        return FilesystemSnapshot(
            total_bytes=_exact_counter(
                block_count * fragment_size,
                "filesystem total",
                positive=True,
            ),
            available_bytes=_exact_counter(
                available_blocks * fragment_size,
                "filesystem available",
            ),
            total_inodes=inode_count,
            available_inodes=available_inodes,
        )
    except ResourceInputError as error:
        raise ResourceSnapshotError("filesystem counters are inconsistent") from error


def _rlimit_value(value: object, field: str) -> int | None:
    if type(value) is not int:
        raise ResourceSnapshotError(f"{field} must be an integer")
    if value == resource.RLIM_INFINITY:
        return None
    try:
        return _exact_counter(value, field)
    except ResourceInputError as error:
        raise ResourceSnapshotError(f"{field} is invalid") from error


def _capture_limits() -> ProcessLimitSnapshot:
    try:
        values = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError) as error:
        raise ResourceSnapshotError("cannot inspect RLIMIT_NOFILE") from error
    if type(values) is not tuple or len(values) != 2:
        raise ResourceSnapshotError("RLIMIT_NOFILE has an invalid shape")
    try:
        return ProcessLimitSnapshot(
            nofile_soft=_rlimit_value(values[0], "nofile soft limit"),
            nofile_hard=_rlimit_value(values[1], "nofile hard limit"),
        )
    except ResourceInputError as error:
        raise ResourceSnapshotError("RLIMIT_NOFILE values are inconsistent") from error


def capture_resource_snapshot(
    capacity_path: Path,
    *,
    paths: ResourcePaths = DEFAULT_RESOURCE_PATHS,
) -> ResourceSnapshot:
    """Capture all publication capacity inputs without changing host state.

    ``capacity_path`` must already exist. The future runner should pass the
    repository root or another existing ancestor on the same filesystem as its
    private publication directory.
    """

    capacity_path = _validate_absolute_path(capacity_path, "capacity_path")
    paths = validate_resource_paths(paths)
    memory = _parse_meminfo(_read_bounded_ascii(paths.meminfo, "proc meminfo"))
    cgroup = _capture_cgroup(paths)
    filesystem = _capture_filesystem(capacity_path)
    pressure = PressureSnapshot(
        memory=_parse_pressure(
            _read_bounded_ascii(paths.memory_pressure, "memory pressure"),
            "memory",
        ),
        io=_parse_pressure(
            _read_bounded_ascii(paths.io_pressure, "I/O pressure"),
            "I/O",
        ),
    )
    limits = _capture_limits()
    return ResourceSnapshot(
        memory=memory,
        cgroup=cgroup,
        filesystem=filesystem,
        pressure=pressure,
        limits=limits,
    )


def assess_publication_capacity(snapshot: ResourceSnapshot) -> CapacityAssessment:
    """Apply only the literal conservative gates frozen before publication."""

    if type(snapshot) is not ResourceSnapshot:
        raise ResourceInputError("snapshot must be an exact ResourceSnapshot")
    validate_resource_snapshot(snapshot)
    return CapacityAssessment(
        snapshot=snapshot,
        failures=_capacity_failures(snapshot),
    )


def require_publication_capacity(
    snapshot: ResourceSnapshot,
) -> CapacityAssessment:
    """Return a passing assessment or raise one deterministic fail-closed error."""

    assessment = assess_publication_capacity(snapshot)
    if not assessment.ready:
        raise ResourceCapacityError(assessment)
    return assessment
