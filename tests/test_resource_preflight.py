from __future__ import annotations

import fcntl
import os
import resource
import stat
import unittest
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PosixPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import ClassVar, Literal, SupportsIndex, cast
from unittest.mock import patch

import gworker.resource_preflight as preflight
from gworker.resource_preflight import (
    GIBIBYTE,
    MIN_PUBLICATION_AVAILABLE_DISK_BYTES,
    MIN_PUBLICATION_AVAILABLE_INODES,
    MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES,
    MIN_PUBLICATION_AVAILABLE_SWAP_BYTES,
    MIN_PUBLICATION_NOFILE_SOFT_LIMIT,
    CapacityAssessment,
    CapacityFailure,
    CgroupLevelSnapshot,
    CgroupSnapshot,
    FilesystemSnapshot,
    MemorySnapshot,
    PressureLine,
    PressureResourceSnapshot,
    PressureSnapshot,
    ProcessLimitSnapshot,
    ResourceCapacityError,
    ResourceInputError,
    ResourcePaths,
    ResourceSnapshot,
    ResourceSnapshotError,
    assess_publication_capacity,
    capture_resource_snapshot,
    require_publication_capacity,
)


@dataclass(frozen=True, slots=True)
class ExtendedMemorySnapshot(MemorySnapshot):
    payload: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtendedResourceSnapshot(ResourceSnapshot):
    payload: tuple[str, ...] = ()


class StatefulPressureLine(PressureLine):
    __slots__ = ()

    field_reads: ClassVar[int] = 0

    def __getattribute__(self, name: str) -> object:
        if name in {"avg10", "avg60", "avg300", "total_microseconds"}:
            type(self).field_reads += 1
        return super().__getattribute__(name)


class StatefulInt(int):
    to_bytes_reads: ClassVar[int] = 0

    def to_bytes(
        self,
        length: SupportsIndex = 1,
        byteorder: Literal["little", "big"] = "big",
        *,
        signed: bool = False,
    ) -> bytes:
        type(self).to_bytes_reads += 1
        return super().to_bytes(length, byteorder, signed=signed)


class StatefulStr(str):
    encode_reads: ClassVar[int] = 0

    def encode(
        self,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> bytes:
        type(self).encode_reads += 1
        return super().encode(encoding, errors)


class StatefulFloat(float):
    float_reads: ClassVar[int] = 0

    def __float__(self) -> float:
        type(self).float_reads += 1
        return super().__float__()


class StatefulPathComponent(str):
    path_reads: ClassVar[int] = 0

    def __getattribute__(self, name: str) -> object:
        if name in {"startswith", "endswith", "replace"}:
            type(self).path_reads += 1
        return super().__getattribute__(name)


class NulPath(PosixPath):
    fspath_reads: ClassVar[int] = 0

    def __fspath__(self) -> str:
        type(self).fspath_reads += 1
        return f"{super().__fspath__()}\x00"


class StatefulRedirectPath(PosixPath):
    path_reads: ClassVar[int] = 0

    def as_posix(self) -> str:
        type(self).path_reads += 1
        if type(self).path_reads == 1:
            return super().as_posix()
        return "/redirected"

    def is_absolute(self) -> bool:
        type(self).path_reads += 1
        return super().is_absolute()

    @property
    def parts(self) -> tuple[str, ...]:
        type(self).path_reads += 1
        return super().parts

    def __fspath__(self) -> str:
        type(self).path_reads += 1
        return "/redirected\x00"


class ResourcePreflightFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.proc = self.root / "proc"
        self.cgroup_root = self.root / "cgroup"
        self.capacity_path = self.root / "capacity"
        self.meminfo = self.proc / "meminfo"
        self.self_cgroup = self.proc / "self" / "cgroup"
        self.mountinfo = self.proc / "self" / "mountinfo"
        self.cgroup_namespace = self.proc / "self" / "ns" / "cgroup"
        self.memory_pressure = self.proc / "pressure" / "memory"
        self.io_pressure = self.proc / "pressure" / "io"
        self.parent_cgroup = self.cgroup_root / "a.slice"
        self.leaf_cgroup = self.parent_cgroup / "job.scope"
        for directory in (
            self.self_cgroup.parent,
            self.cgroup_namespace.parent,
            self.memory_pressure.parent,
            self.leaf_cgroup,
            self.capacity_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.paths = ResourcePaths(
            meminfo=self.meminfo,
            self_cgroup=self.self_cgroup,
            mountinfo=self.mountinfo,
            cgroup_namespace=self.cgroup_namespace,
            cgroup_root=self.cgroup_root,
            memory_pressure=self.memory_pressure,
            io_pressure=self.io_pressure,
        )
        self._write_default_fixture()
        self.production_cgroup_namespace_inode = preflight._CGROUP_NS_INIT_INO
        fixture_namespace_inode = os.stat(self.cgroup_namespace).st_ino
        self.namespace_inode_patcher = patch.object(
            preflight,
            "_CGROUP_NS_INIT_INO",
            fixture_namespace_inode,
        )
        self.namespace_type_patcher = patch.object(
            fcntl,
            "ioctl",
            return_value=preflight._CLONE_NEWCGROUP,
        )
        self.namespace_inode_patcher.start()
        self.namespace_type_patcher.start()

    def tearDown(self) -> None:
        self.namespace_type_patcher.stop()
        self.namespace_inode_patcher.stop()
        self.temporary_directory.cleanup()

    def _write_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="ascii")

    def _write_meminfo(
        self,
        *,
        total_kibibytes: int = 32 * 1_024 * 1_024,
        available_kibibytes: int = 8 * 1_024 * 1_024,
        swap_total_kibibytes: int = 2 * 1_024 * 1_024,
        swap_free_kibibytes: int = 1 * 1_024 * 1_024,
    ) -> None:
        self._write_text(
            self.meminfo,
            (
                f"MemTotal: {total_kibibytes} kB\n"
                f"MemAvailable: {available_kibibytes} kB\n"
                f"SwapTotal: {swap_total_kibibytes} kB\n"
                f"SwapFree: {swap_free_kibibytes} kB\n"
                "Cached: 123 kB\n"
            ),
        )

    def _write_pressure(self) -> None:
        payload = (
            "some avg10=0.00 avg60=0.01 avg300=0.02 total=12\n"
            "full avg10=0.00 avg60=0.00 avg300=0.00 total=3\n"
        )
        self._write_text(self.memory_pressure, payload)
        self._write_text(self.io_pressure, payload)

    @staticmethod
    def _mountinfo_path(value: str) -> str:
        return (
            value.replace("\\", r"\134")
            .replace(" ", r"\040")
            .replace("\t", r"\011")
            .replace("\n", r"\012")
        )

    def _write_mountinfo(
        self,
        *,
        mount_root: str = "/",
        mount_point: Path | None = None,
        cgroup2_entries: int = 1,
        include_legacy_cgroup: bool = False,
        nested_mount_point: Path | None = None,
        device: tuple[int, int] | None = None,
    ) -> None:
        if device is None:
            device_number = os.stat(self.cgroup_root).st_dev
            device = (os.major(device_number), os.minor(device_number))
        configured_mount_point = (
            self.cgroup_root if mount_point is None else mount_point
        )
        lines = ["10 1 0:1 / /proc rw - proc proc rw\n"]
        for index in range(cgroup2_entries):
            lines.append(
                f"{36 + index} 30 {device[0]}:{device[1]} "
                f"{self._mountinfo_path(mount_root)} "
                f"{self._mountinfo_path(configured_mount_point.as_posix())} "
                "rw - cgroup2 cgroup2 rw\n"
            )
        if include_legacy_cgroup:
            lines.append(
                "50 30 0:50 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
            )
        if nested_mount_point is not None:
            lines.append(
                f"60 30 0:60 / "
                f"{self._mountinfo_path(nested_mount_point.as_posix())} "
                "rw - tmpfs tmpfs rw\n"
            )
        self._write_text(self.mountinfo, "".join(lines))

    def _write_cgroup_level(
        self,
        directory: Path,
        *,
        memory_max: str = "max",
        memory_current: str = "0",
        swap_max: str = "max",
        swap_current: str = "0",
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        values = {
            "memory.max": memory_max,
            "memory.current": memory_current,
            "memory.swap.max": swap_max,
            "memory.swap.current": swap_current,
        }
        for name, value in values.items():
            self._write_text(directory / name, f"{value}\n")

    def _write_default_fixture(self) -> None:
        self._write_meminfo()
        self._write_text(self.self_cgroup, "0::/a.slice/job.scope\n")
        self._write_text(self.cgroup_namespace, "fixture namespace handle\n")
        self._write_mountinfo()
        self._write_pressure()
        self._write_cgroup_level(self.parent_cgroup)
        self._write_cgroup_level(self.leaf_cgroup)

    def _statvfs(
        self,
        **overrides: object,
    ) -> SimpleNamespace:
        fragment_size = 4_096
        values: dict[str, object] = {
            "f_frsize": fragment_size,
            "f_blocks": 16 * GIBIBYTE // fragment_size,
            "f_bavail": MIN_PUBLICATION_AVAILABLE_DISK_BYTES // fragment_size,
            "f_files": 20_000,
            "f_favail": MIN_PUBLICATION_AVAILABLE_INODES,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _capture(
        self,
        *,
        statvfs_result: object | None = None,
        rlimit_result: object = (MIN_PUBLICATION_NOFILE_SOFT_LIMIT, 1_024),
    ) -> ResourceSnapshot:
        filesystem = self._statvfs() if statvfs_result is None else statvfs_result
        with (
            patch.object(os, "fstatvfs", return_value=filesystem),
            patch.object(
                resource,
                "getrlimit",
                return_value=rlimit_result,
            ),
        ):
            return capture_resource_snapshot(
                self.capacity_path,
                paths=self.paths,
            )

    def _fixture_inventory(self) -> tuple[tuple[str, str, bytes | None], ...]:
        entries: list[tuple[str, str, bytes | None]] = []
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root).as_posix()
            if path.is_dir():
                entries.append((relative, "directory", None))
            else:
                entries.append((relative, "file", path.read_bytes()))
        return tuple(entries)

    def test_capture_is_deterministic_read_only_and_passes_exact_boundaries(
        self,
    ) -> None:
        before = self._fixture_inventory()

        first = self._capture()
        second = self._capture()

        self.assertEqual(first, second)
        self.assertEqual(self._fixture_inventory(), before)
        self.assertEqual(
            first.memory,
            MemorySnapshot(
                total_bytes=32 * GIBIBYTE,
                available_bytes=MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES,
                swap_total_bytes=2 * GIBIBYTE,
                swap_free_bytes=MIN_PUBLICATION_AVAILABLE_SWAP_BYTES,
            ),
        )
        self.assertEqual(
            tuple(level.path for level in first.cgroup.levels),
            ("/a.slice", "/a.slice/job.scope"),
        )
        self.assertIsNone(first.cgroup.memory_headroom_bytes)
        self.assertIsNone(first.cgroup.swap_headroom_bytes)
        self.assertEqual(
            first.filesystem.available_bytes,
            MIN_PUBLICATION_AVAILABLE_DISK_BYTES,
        )
        self.assertEqual(
            first.filesystem.available_inodes,
            MIN_PUBLICATION_AVAILABLE_INODES,
        )
        self.assertEqual(first.pressure.memory.some.total_microseconds, 12)
        self.assertEqual(
            first.limits.nofile_soft,
            MIN_PUBLICATION_NOFILE_SOFT_LIMIT,
        )
        assessment = assess_publication_capacity(first)
        self.assertTrue(assessment.ready)
        self.assertEqual(assessment.failures, ())
        self.assertEqual(assessment.diagnostics, ())
        self.assertIs(require_publication_capacity(first).snapshot, first)

    def test_each_gate_is_inclusive_and_one_below_fails(self) -> None:
        baseline = self._capture()
        below_memory = replace(
            baseline,
            memory=replace(
                baseline.memory,
                available_bytes=MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES - 1,
            ),
        )
        below_swap = replace(
            baseline,
            memory=replace(
                baseline.memory,
                swap_free_bytes=MIN_PUBLICATION_AVAILABLE_SWAP_BYTES - 1,
            ),
        )
        below_disk = replace(
            baseline,
            filesystem=replace(
                baseline.filesystem,
                available_bytes=MIN_PUBLICATION_AVAILABLE_DISK_BYTES - 1,
            ),
        )
        below_inodes = replace(
            baseline,
            filesystem=replace(
                baseline.filesystem,
                available_inodes=MIN_PUBLICATION_AVAILABLE_INODES - 1,
            ),
        )
        below_nofile = replace(
            baseline,
            limits=replace(
                baseline.limits,
                nofile_soft=MIN_PUBLICATION_NOFILE_SOFT_LIMIT - 1,
            ),
        )
        cases = (
            (below_memory, CapacityFailure.MEMORY),
            (below_swap, CapacityFailure.SWAP),
            (below_disk, CapacityFailure.DISK),
            (below_inodes, CapacityFailure.INODES),
            (below_nofile, CapacityFailure.NOFILE),
        )
        for snapshot, expected in cases:
            with self.subTest(failure=expected):
                assessment = assess_publication_capacity(snapshot)
                self.assertFalse(assessment.ready)
                self.assertEqual(assessment.failures, (expected,))

        unlimited_nofile = replace(
            baseline,
            limits=ProcessLimitSnapshot(nofile_soft=None, nofile_hard=None),
        )
        self.assertTrue(assess_publication_capacity(unlimited_nofile).ready)

    def test_all_failure_diagnostics_have_stable_order_and_exact_values(
        self,
    ) -> None:
        baseline = self._capture()
        snapshot = replace(
            baseline,
            memory=replace(
                baseline.memory,
                available_bytes=MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES - 1,
                swap_free_bytes=MIN_PUBLICATION_AVAILABLE_SWAP_BYTES - 1,
            ),
            filesystem=replace(
                baseline.filesystem,
                available_bytes=MIN_PUBLICATION_AVAILABLE_DISK_BYTES - 1,
                available_inodes=MIN_PUBLICATION_AVAILABLE_INODES - 1,
            ),
            limits=replace(
                baseline.limits,
                nofile_soft=MIN_PUBLICATION_NOFILE_SOFT_LIMIT - 1,
            ),
        )

        assessment = assess_publication_capacity(snapshot)

        self.assertEqual(
            assessment.failures,
            (
                CapacityFailure.MEMORY,
                CapacityFailure.SWAP,
                CapacityFailure.DISK,
                CapacityFailure.INODES,
                CapacityFailure.NOFILE,
            ),
        )
        expected = (
            "memory-headroom: actual=8589934591 required-at-least=8589934592",
            "swap-headroom: actual=1073741823 required-at-least=1073741824",
            "filesystem-bytes: actual=8589934591 required-at-least=8589934592",
            "filesystem-inodes: actual=9999 required-at-least=10000",
            "nofile-soft-limit: actual=255 required-at-least=256",
        )
        self.assertEqual(assessment.diagnostics, expected)
        with self.assertRaises(ResourceCapacityError) as raised:
            require_publication_capacity(snapshot)
        self.assertEqual(raised.exception.assessment, assessment)
        self.assertIs(raised.exception.assessment.snapshot, snapshot)
        self.assertEqual(
            str(raised.exception),
            "publication resource preflight failed: " + "; ".join(expected),
        )

    def test_host_without_swap_skips_only_the_swap_gate(self) -> None:
        self._write_meminfo(
            swap_total_kibibytes=0,
            swap_free_kibibytes=0,
        )
        self._write_cgroup_level(
            self.parent_cgroup,
            swap_max="0",
            swap_current="0",
        )
        self._write_cgroup_level(
            self.leaf_cgroup,
            swap_max="0",
            swap_current="0",
        )

        snapshot = self._capture()

        self.assertEqual(snapshot.effective_available_swap_bytes, 0)
        self.assertTrue(assess_publication_capacity(snapshot).ready)

    def test_tightest_finite_cgroup_ancestor_clamps_host_headroom(self) -> None:
        self._write_meminfo(
            available_kibibytes=16 * 1_024 * 1_024,
        )
        memory_current = 2 * GIBIBYTE
        swap_current = GIBIBYTE
        self._write_cgroup_level(
            self.parent_cgroup,
            memory_max=str(memory_current + MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES - 1),
            memory_current=str(memory_current),
            swap_max=str(swap_current + MIN_PUBLICATION_AVAILABLE_SWAP_BYTES - 1),
            swap_current=str(swap_current),
        )

        snapshot = self._capture()

        self.assertEqual(
            snapshot.effective_available_memory_bytes,
            MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES - 1,
        )
        self.assertEqual(
            snapshot.effective_available_swap_bytes,
            MIN_PUBLICATION_AVAILABLE_SWAP_BYTES - 1,
        )
        self.assertEqual(
            assess_publication_capacity(snapshot).failures,
            (CapacityFailure.MEMORY, CapacityFailure.SWAP),
        )

        self._write_cgroup_level(
            self.parent_cgroup,
            memory_max="1",
            memory_current="2",
            swap_max="1",
            swap_current="2",
        )
        exhausted = self._capture()
        self.assertEqual(exhausted.cgroup.memory_headroom_bytes, 0)
        self.assertEqual(exhausted.cgroup.swap_headroom_bytes, 0)

    def test_unlimited_cgroup_uses_host_availability(self) -> None:
        snapshot = self._capture()

        self.assertTrue(
            all(
                level.memory_limit_bytes is None and level.swap_limit_bytes is None
                for level in snapshot.cgroup.levels
            )
        )
        self.assertEqual(
            snapshot.effective_available_memory_bytes,
            snapshot.memory.available_bytes,
        )
        self.assertEqual(
            snapshot.effective_available_swap_bytes,
            snapshot.memory.swap_free_bytes,
        )

        self._write_text(self.self_cgroup, "0::/\n")
        self._write_cgroup_level(self.cgroup_root)
        root_snapshot = self._capture()
        self.assertEqual(
            root_snapshot.cgroup,
            CgroupSnapshot(
                self_path="/",
                levels=(
                    CgroupLevelSnapshot(
                        path="/",
                        memory_limit_bytes=None,
                        memory_current_bytes=0,
                        swap_limit_bytes=None,
                        swap_current_bytes=0,
                    ),
                ),
            ),
        )

    def test_native_root_without_controller_files_uses_host_availability(
        self,
    ) -> None:
        self._write_text(self.self_cgroup, "0::/\n")

        snapshot = self._capture()

        self.assertEqual(
            snapshot.cgroup,
            CgroupSnapshot(self_path="/", levels=()),
        )
        self.assertEqual(
            snapshot.effective_available_memory_bytes,
            snapshot.memory.available_bytes,
        )
        self.assertEqual(
            snapshot.effective_available_swap_bytes,
            snapshot.memory.swap_free_bytes,
        )

    def test_visible_namespace_root_limit_is_included_for_descendants(self) -> None:
        current = GIBIBYTE
        self._write_cgroup_level(
            self.cgroup_root,
            memory_max=str(current + MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES - 1),
            memory_current=str(current),
        )

        snapshot = self._capture()

        self.assertEqual(
            tuple(level.path for level in snapshot.cgroup.levels),
            ("/", "/a.slice", "/a.slice/job.scope"),
        )
        self.assertEqual(
            snapshot.effective_available_memory_bytes,
            MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES - 1,
        )

    def test_meminfo_fixture_parse_failures_are_closed(self) -> None:
        invalid_payloads = (
            (
                "missing",
                "MemTotal: 1 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
            ),
            (
                "duplicate",
                (
                    "MemTotal: 2 kB\nMemTotal: 2 kB\n"
                    "MemAvailable: 1 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
                ),
            ),
            (
                "wrong-unit",
                (
                    "MemTotal: 2 MB\nMemAvailable: 1 kB\n"
                    "SwapTotal: 0 kB\nSwapFree: 0 kB\n"
                ),
            ),
            (
                "signed-counter",
                (
                    "MemTotal: 2 kB\nMemAvailable: -1 kB\n"
                    "SwapTotal: 0 kB\nSwapFree: 0 kB\n"
                ),
            ),
            (
                "inconsistent",
                (
                    "MemTotal: 1 kB\nMemAvailable: 2 kB\n"
                    "SwapTotal: 0 kB\nSwapFree: 0 kB\n"
                ),
            ),
            (
                "overflow",
                (
                    f"MemTotal: {'9' * 5_000} kB\nMemAvailable: 1 kB\n"
                    "SwapTotal: 0 kB\nSwapFree: 0 kB\n"
                ),
            ),
            (
                "malformed-line",
                (
                    "MemTotal: 2 kB\nMemAvailable: 1 kB\n"
                    "SwapTotal: 0 kB\nSwapFree: 0 kB\nmalformed\n"
                ),
            ),
        )
        for name, payload in invalid_payloads:
            with self.subTest(case=name):
                self._write_text(self.meminfo, payload)
                with self.assertRaises(ResourceSnapshotError):
                    self._capture()

        self.meminfo.write_bytes(b"MemTotal: 2 kB\n\xff\n")
        with self.assertRaisesRegex(ResourceSnapshotError, "ASCII"):
            self._capture()
        self.meminfo.write_bytes(b"MemTotal: 2 kB\n\x00\n")
        with self.assertRaisesRegex(ResourceSnapshotError, "NUL"):
            self._capture()

    def test_nonregular_proc_source_is_rejected_without_following_it(self) -> None:
        target = self.root / "outside-meminfo"
        target.write_text(self.meminfo.read_text(encoding="ascii"), encoding="ascii")
        self.meminfo.unlink()
        self.meminfo.symlink_to(target)

        with self.assertRaisesRegex(ResourceSnapshotError, "regular"):
            self._capture()

    def test_source_reads_are_missing_empty_and_size_bounded(self) -> None:
        self.meminfo.unlink()
        with self.assertRaisesRegex(ResourceSnapshotError, "cannot inspect"):
            self._capture()

        self.meminfo.write_bytes(b"")
        with self.assertRaisesRegex(ResourceSnapshotError, "empty"):
            self._capture()

        self.meminfo.write_bytes(b"x" * (preflight._MAX_RESOURCE_FILE_BYTES + 1))
        with self.assertRaisesRegex(ResourceSnapshotError, "read bound"):
            self._capture()

    def test_fstat_failures_are_closed_and_cleanup_opened_descriptors(
        self,
    ) -> None:
        def assert_closed(operation: Callable[[], object]) -> None:
            with (
                patch.object(os, "fstat", side_effect=OSError("fstat fault")),
                patch.object(os, "close", wraps=os.close) as close,
                self.assertRaisesRegex(
                    ResourceSnapshotError,
                    "cannot inspect opened",
                ),
            ):
                operation()
            close.assert_called_once()

        assert_closed(
            lambda: preflight._read_bounded_ascii(
                self.meminfo,
                "proc meminfo",
            )
        )
        assert_closed(
            lambda: preflight._open_bound_directory(
                self.cgroup_root,
                "cgroup root",
            )
        )

        parent_descriptor = os.open(
            self.cgroup_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            assert_closed(
                lambda: preflight._open_bound_directory_at(
                    parent_descriptor,
                    "a.slice",
                    "cgroup ancestor /a.slice",
                )
            )
        finally:
            os.close(parent_descriptor)

        parent_descriptor = os.open(
            self.parent_cgroup,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            assert_closed(
                lambda: preflight._read_bounded_ascii_at(
                    parent_descriptor,
                    "memory.max",
                    "cgroup memory.max",
                )
            )
        finally:
            os.close(parent_descriptor)

    def test_source_file_opens_are_nonblocking(self) -> None:
        with patch.object(os, "open", wraps=os.open) as opened:
            preflight._read_bounded_ascii(self.meminfo, "proc meminfo")
        self.assertTrue(opened.call_args.args[1] & os.O_NONBLOCK)

        parent_descriptor = os.open(
            self.parent_cgroup,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            with patch.object(os, "open", wraps=os.open) as opened_at:
                preflight._read_bounded_ascii_at(
                    parent_descriptor,
                    "memory.max",
                    "cgroup memory.max",
                )
            self.assertTrue(opened_at.call_args.args[1] & os.O_NONBLOCK)
        finally:
            os.close(parent_descriptor)

    def test_cgroup_membership_and_controller_parse_failures_are_closed(
        self,
    ) -> None:
        invalid_memberships = (
            "0::/a.slice/../job.scope\n",
            "0::/a.slice/job.scope\n0::/other.scope\n",
            "broken\n",
            "1:name=systemd:/a.slice/job.scope\n",
        )
        for payload in invalid_memberships:
            with self.subTest(membership=payload):
                self._write_text(self.self_cgroup, payload)
                with self.assertRaises(ResourceSnapshotError):
                    self._capture()

        self._write_text(self.self_cgroup, "0::/a.slice/job.scope\n")
        self._write_text(self.parent_cgroup / "memory.max", "infinity\n")
        with self.assertRaises(ResourceSnapshotError):
            self._capture()

        self._write_default_fixture()
        (self.leaf_cgroup / "memory.swap.current").unlink()
        with self.assertRaisesRegex(ResourceSnapshotError, "incomplete"):
            self._capture()

        self._write_default_fixture()
        for name in preflight._CGROUP_FILES:
            (self.leaf_cgroup / name).unlink()
        with self.assertRaisesRegex(ResourceSnapshotError, "memory controller"):
            self._capture()

        self._write_default_fixture()
        for name in preflight._CGROUP_FILES:
            (self.parent_cgroup / name).unlink()
        with self.assertRaisesRegex(ResourceSnapshotError, "memory controller"):
            self._capture()

        self._write_default_fixture()
        target = self.root / "memory-limit"
        self._write_text(target, "max\n")
        cgroup_limit = self.leaf_cgroup / "memory.max"
        cgroup_limit.unlink()
        cgroup_limit.symlink_to(target)
        with self.assertRaisesRegex(ResourceSnapshotError, "regular"):
            self._capture()

    def test_hybrid_v1_memory_membership_is_rejected_at_native_v2_root(
        self,
    ) -> None:
        self._write_text(
            self.self_cgroup,
            "0::/\n7:cpuset,memory:/actually-limited\n",
        )

        with self.assertRaisesRegex(ResourceSnapshotError, "v1 memory"):
            self._capture()

    def test_cgroup_namespace_must_be_initial_bound_and_stable(self) -> None:
        self.assertEqual(
            self.production_cgroup_namespace_inode,
            0xEFFFFFFB,
        )
        with patch.object(os, "open", wraps=os.open) as opened_namespace:
            descriptor, identity = preflight._open_initial_cgroup_namespace(
                self.cgroup_namespace
            )
        self.assertTrue(opened_namespace.call_args.args[1] & os.O_NONBLOCK)
        try:
            self.assertEqual(identity[1], os.fstat(descriptor).st_ino)
        finally:
            os.close(descriptor)

        with (
            patch.object(
                preflight,
                "_CGROUP_NS_INIT_INO",
                self.production_cgroup_namespace_inode,
            ),
            self.assertRaisesRegex(ResourceSnapshotError, "initial cgroup namespace"),
        ):
            preflight._open_initial_cgroup_namespace(self.cgroup_namespace)

        with (
            patch.object(fcntl, "ioctl", return_value=0),
            self.assertRaisesRegex(ResourceSnapshotError, "not a cgroup namespace"),
        ):
            preflight._open_initial_cgroup_namespace(self.cgroup_namespace)

        syscall_failures = (
            ("stat", os, "stat", "cannot inspect"),
            ("open", os, "open", "cannot open"),
            ("fstat", os, "fstat", "cannot inspect opened"),
            ("ioctl", fcntl, "ioctl", "not a Linux namespace"),
        )
        for name, owner, attribute, message in syscall_failures:
            with (
                self.subTest(syscall=name),
                patch.object(owner, attribute, side_effect=OSError("fault")),
                self.assertRaisesRegex(ResourceSnapshotError, message),
            ):
                preflight._open_initial_cgroup_namespace(self.cgroup_namespace)

        original = preflight._read_bounded_ascii
        membership_reads = 0

        def replace_namespace(path: Path, field: str) -> str:
            nonlocal membership_reads
            text = original(path, field)
            if path == self.self_cgroup:
                membership_reads += 1
                if membership_reads == 2:
                    held_path = self.cgroup_namespace.with_name("held-cgroup")
                    self.cgroup_namespace.rename(held_path)
                    self._write_text(
                        self.cgroup_namespace,
                        "replacement namespace handle\n",
                    )
            return text

        with (
            patch.object(
                preflight,
                "_read_bounded_ascii",
                side_effect=replace_namespace,
            ),
            self.assertRaisesRegex(ResourceSnapshotError, "initial cgroup namespace"),
        ):
            self._capture()

    def test_mountinfo_must_prove_one_complete_cgroup2_hierarchy(self) -> None:
        cases = (
            (
                "hidden-ancestors",
                {"mount_root": "/hidden.slice"},
                "full ancestor hierarchy",
            ),
            (
                "missing-cgroup2",
                {"cgroup2_entries": 0},
                "exactly one cgroup2",
            ),
            (
                "ambiguous-cgroup2",
                {"cgroup2_entries": 2},
                "exactly one cgroup2",
            ),
            (
                "legacy-hybrid",
                {"include_legacy_cgroup": True},
                "ambiguous",
            ),
            (
                "wrong-mountpoint",
                {"mount_point": self.root / "other-cgroup"},
                "configured cgroup root",
            ),
            (
                "wrong-device",
                {"device": (99, 99)},
                "mount device",
            ),
            (
                "nested-mount",
                {"nested_mount_point": self.parent_cgroup},
                "nested mount",
            ),
        )
        for name, options, message in cases:
            with self.subTest(case=name):
                self._write_mountinfo(**options)
                with self.assertRaisesRegex(ResourceSnapshotError, message):
                    self._capture()
                self._write_mountinfo()

    def test_cgroup_descendants_remain_on_the_cgroup2_device(self) -> None:
        device = os.stat(self.cgroup_root).st_dev
        wrong_device = (os.major(device), os.minor(device) + 1)

        root_descriptor = os.open(
            self.cgroup_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            with self.assertRaisesRegex(ResourceSnapshotError, "crosses"):
                preflight._open_bound_directory_at(
                    root_descriptor,
                    "a.slice",
                    "cgroup ancestor /a.slice",
                    expected_device=wrong_device,
                )
        finally:
            os.close(root_descriptor)

        parent_descriptor = os.open(
            self.parent_cgroup,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            with self.assertRaisesRegex(ResourceSnapshotError, "crosses"):
                preflight._read_bounded_ascii_at(
                    parent_descriptor,
                    "memory.max",
                    "cgroup memory.max",
                    expected_device=wrong_device,
                )
        finally:
            os.close(parent_descriptor)

    def test_cgroup_parent_close_failure_is_not_retried(self) -> None:
        real_close = os.close
        calls: list[int] = []
        faulted_descriptor: int | None = None
        fault_call_index: int | None = None

        def close_first_directory_with_error(descriptor: int) -> None:
            nonlocal fault_call_index, faulted_descriptor
            calls.append(descriptor)
            if faulted_descriptor is None and stat.S_ISDIR(
                os.fstat(descriptor).st_mode
            ):
                faulted_descriptor = descriptor
                fault_call_index = len(calls) - 1
                real_close(descriptor)
                raise OSError("ambiguous close result")
            real_close(descriptor)

        with (
            patch.object(os, "close", side_effect=close_first_directory_with_error),
            self.assertRaisesRegex(
                ResourceSnapshotError,
                "cannot close cgroup ancestor",
            ),
        ):
            preflight._capture_cgroup_hierarchy(self.paths)
        self.assertIsNotNone(faulted_descriptor)
        if fault_call_index is None:
            self.fail("the cgroup parent close was not exercised")
        self.assertNotIn(faulted_descriptor, calls[fault_call_index + 1 :])

    def test_cgroup_mountinfo_must_remain_stable_during_capture(self) -> None:
        original = preflight._read_bounded_ascii
        mountinfo_reads = 0

        def changing_mountinfo(path: Path, field: str) -> str:
            nonlocal mountinfo_reads
            text = original(path, field)
            if path == self.mountinfo:
                mountinfo_reads += 1
                if mountinfo_reads == 2:
                    return text.replace("36 30 ", "37 30 ", 1)
            return text

        with (
            patch.object(
                preflight,
                "_read_bounded_ascii",
                side_effect=changing_mountinfo,
            ),
            self.assertRaisesRegex(ResourceSnapshotError, "mount changed"),
        ):
            self._capture()

    def test_cgroup_membership_must_remain_stable_during_capture(self) -> None:
        original = preflight._read_bounded_ascii
        membership_reads = 0

        def changing_membership(path: Path, field: str) -> str:
            nonlocal membership_reads
            if path == self.self_cgroup:
                membership_reads += 1
                if membership_reads == 2:
                    return "0::/moved.scope\n"
            return original(path, field)

        with (
            patch.object(
                preflight,
                "_read_bounded_ascii",
                side_effect=changing_membership,
            ),
            self.assertRaisesRegex(ResourceSnapshotError, "membership changed"),
        ):
            self._capture()

    def test_psi_fixture_parse_failures_are_closed(self) -> None:
        invalid_payloads = (
            (
                "\t\n"
                "some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
            ),
            "some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n",
            (
                "some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
                "some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
            ),
            (
                "other avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
            ),
            (
                "some avg10=0.00 avg60=0.00 avg300=0.00\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
            ),
            (
                "some avg10=nan avg60=0.00 avg300=0.00 total=1\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
            ),
            (
                "some avg10=100.01 avg60=0.00 avg300=0.00 total=1\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
            ),
            (
                "some avg10=0.00 avg60=0.00 avg300=0.00 total=-1\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
            ),
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self._write_text(self.memory_pressure, payload)
                with self.assertRaises(ResourceSnapshotError):
                    self._capture()

    def test_statvfs_failures_and_inconsistent_counters_are_closed(self) -> None:
        with (
            patch.object(os, "fstatvfs", side_effect=OSError("no stat")),
            patch.object(
                resource,
                "getrlimit",
                return_value=(256, 1_024),
            ),
            self.assertRaisesRegex(ResourceSnapshotError, "filesystem"),
        ):
            capture_resource_snapshot(self.capacity_path, paths=self.paths)

        invalid_values = (
            self._statvfs(f_frsize=0),
            self._statvfs(f_blocks=0),
            self._statvfs(f_bavail=cast(int, self._statvfs().f_blocks) + 1),
            self._statvfs(f_files=0),
            self._statvfs(f_favail=20_001),
            self._statvfs(f_bavail=True),
            self._statvfs(f_frsize=2**63, f_blocks=3),
        )
        for values in invalid_values:
            with (
                self.subTest(values=values),
                self.assertRaises(ResourceSnapshotError),
            ):
                self._capture(statvfs_result=values)

    def test_rlimit_failures_are_closed_and_infinity_is_supported(self) -> None:
        invalid_values: tuple[object, ...] = (
            [256, 1_024],
            (256,),
            (True, 1_024),
            (-2, 1_024),
            (2_048, 1_024),
            (resource.RLIM_INFINITY, 1_024),
        )
        for values in invalid_values:
            with (
                self.subTest(values=values),
                self.assertRaises(ResourceSnapshotError),
            ):
                self._capture(rlimit_result=values)

        snapshot = self._capture(
            rlimit_result=(resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
        self.assertEqual(
            snapshot.limits,
            ProcessLimitSnapshot(nofile_soft=None, nofile_hard=None),
        )
        self.assertTrue(assess_publication_capacity(snapshot).ready)

        with (
            patch.object(os, "fstatvfs", return_value=self._statvfs()),
            patch.object(
                resource,
                "getrlimit",
                side_effect=OSError("no rlimit"),
            ),
            self.assertRaisesRegex(ResourceSnapshotError, "RLIMIT_NOFILE"),
        ):
            capture_resource_snapshot(self.capacity_path, paths=self.paths)

    def test_value_objects_require_exact_closed_types(self) -> None:
        baseline = self._capture()
        extended_memory = ExtendedMemorySnapshot(
            total_bytes=baseline.memory.total_bytes,
            available_bytes=baseline.memory.available_bytes,
            swap_total_bytes=baseline.memory.swap_total_bytes,
            swap_free_bytes=baseline.memory.swap_free_bytes,
        )

        with self.assertRaisesRegex(ResourceInputError, "exact integer"):
            MemorySnapshot(
                total_bytes=cast(int, True),
                available_bytes=0,
                swap_total_bytes=0,
                swap_free_bytes=0,
            )
        with self.assertRaisesRegex(ResourceInputError, "exact finite float"):
            PressureLine(
                avg10=cast(float, 0),
                avg60=0.0,
                avg300=0.0,
                total_microseconds=0,
            )
        with self.assertRaisesRegex(ResourceInputError, "invalid type"):
            replace(baseline, memory=cast(MemorySnapshot, extended_memory))
        with self.assertRaisesRegex(ResourceInputError, "exact tuple"):
            CgroupSnapshot(
                self_path="/a.slice",
                levels=cast(tuple[CgroupLevelSnapshot, ...], []),
            )
        with self.assertRaisesRegex(ResourceInputError, "exact tuple"):
            CapacityAssessment(
                snapshot=baseline,
                failures=cast(tuple[CapacityFailure, ...], []),
            )
        with self.assertRaisesRegex(ResourceInputError, "reordered"):
            CapacityAssessment(
                snapshot=baseline,
                failures=(CapacityFailure.DISK, CapacityFailure.MEMORY),
            )
        with self.assertRaisesRegex(ResourceInputError, "reordered"):
            CapacityAssessment(
                snapshot=baseline,
                failures=(CapacityFailure.MEMORY, CapacityFailure.MEMORY),
            )
        with self.assertRaisesRegex(ResourceInputError, "do not match"):
            CapacityAssessment(
                snapshot=baseline,
                failures=(CapacityFailure.MEMORY,),
            )
        with self.assertRaisesRegex(ResourceInputError, "exact ResourceSnapshot"):
            ExtendedResourceSnapshot(
                memory=baseline.memory,
                cgroup=baseline.cgroup,
                filesystem=baseline.filesystem,
                pressure=baseline.pressure,
                limits=baseline.limits,
            )

    def test_recursive_snapshot_shape_rejects_stateful_subclasses_before_reads(
        self,
    ) -> None:
        baseline = self._capture()
        stateful_line = StatefulPressureLine(
            avg10=baseline.pressure.memory.some.avg10,
            avg60=baseline.pressure.memory.some.avg60,
            avg300=baseline.pressure.memory.some.avg300,
            total_microseconds=baseline.pressure.memory.some.total_microseconds,
        )
        unsafe_memory_pressure = replace(baseline.pressure.memory)
        object.__setattr__(unsafe_memory_pressure, "some", stateful_line)
        unsafe_pressure = replace(
            baseline.pressure,
            memory=unsafe_memory_pressure,
        )

        StatefulPressureLine.field_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "some has an invalid type"):
            ResourceSnapshot(
                memory=baseline.memory,
                cgroup=baseline.cgroup,
                filesystem=baseline.filesystem,
                pressure=unsafe_pressure,
                limits=baseline.limits,
            )
        self.assertEqual(StatefulPressureLine.field_reads, 0)

        mutated = replace(baseline)
        object.__setattr__(mutated, "pressure", unsafe_pressure)
        StatefulPressureLine.field_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "some has an invalid type"):
            assess_publication_capacity(mutated)
        self.assertEqual(StatefulPressureLine.field_reads, 0)

    def test_recursive_snapshot_shape_rejects_stateful_scalar_subclasses(
        self,
    ) -> None:
        baseline = self._capture()

        unsafe_memory = replace(baseline.memory)
        object.__setattr__(
            unsafe_memory,
            "total_bytes",
            cast(int, StatefulInt(unsafe_memory.total_bytes)),
        )
        StatefulInt.to_bytes_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "invalid scalar type"):
            ResourceSnapshot(
                memory=unsafe_memory,
                cgroup=baseline.cgroup,
                filesystem=baseline.filesystem,
                pressure=baseline.pressure,
                limits=baseline.limits,
            )
        self.assertEqual(StatefulInt.to_bytes_reads, 0)
        mutated_memory_snapshot = replace(baseline)
        object.__setattr__(mutated_memory_snapshot, "memory", unsafe_memory)
        StatefulInt.to_bytes_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "invalid scalar type"):
            assess_publication_capacity(mutated_memory_snapshot)
        self.assertEqual(StatefulInt.to_bytes_reads, 0)

        unsafe_cgroup = replace(baseline.cgroup)
        object.__setattr__(
            unsafe_cgroup,
            "self_path",
            cast(str, StatefulStr(unsafe_cgroup.self_path)),
        )
        StatefulStr.encode_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "invalid scalar type"):
            ResourceSnapshot(
                memory=baseline.memory,
                cgroup=unsafe_cgroup,
                filesystem=baseline.filesystem,
                pressure=baseline.pressure,
                limits=baseline.limits,
            )
        self.assertEqual(StatefulStr.encode_reads, 0)
        mutated_cgroup_snapshot = replace(baseline)
        object.__setattr__(mutated_cgroup_snapshot, "cgroup", unsafe_cgroup)
        StatefulStr.encode_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "invalid scalar type"):
            assess_publication_capacity(mutated_cgroup_snapshot)
        self.assertEqual(StatefulStr.encode_reads, 0)

        unsafe_line = replace(baseline.pressure.memory.some)
        object.__setattr__(
            unsafe_line,
            "avg10",
            cast(float, StatefulFloat(unsafe_line.avg10)),
        )
        unsafe_memory_pressure = replace(
            baseline.pressure.memory,
            some=unsafe_line,
        )
        unsafe_pressure = replace(
            baseline.pressure,
            memory=unsafe_memory_pressure,
        )
        StatefulFloat.float_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "invalid scalar type"):
            ResourceSnapshot(
                memory=baseline.memory,
                cgroup=baseline.cgroup,
                filesystem=baseline.filesystem,
                pressure=unsafe_pressure,
                limits=baseline.limits,
            )
        self.assertEqual(StatefulFloat.float_reads, 0)
        mutated_pressure_snapshot = replace(baseline)
        object.__setattr__(mutated_pressure_snapshot, "pressure", unsafe_pressure)
        StatefulFloat.float_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "invalid scalar type"):
            assess_publication_capacity(mutated_pressure_snapshot)
        self.assertEqual(StatefulFloat.float_reads, 0)

    def test_recursive_snapshot_shape_rejects_missing_slots_fail_closed(
        self,
    ) -> None:
        baseline = self._capture()

        empty_snapshot = object.__new__(ResourceSnapshot)
        with self.assertRaisesRegex(ResourceInputError, "shape is incomplete"):
            preflight.validate_resource_snapshot(empty_snapshot)

        missing_top_level = replace(baseline)
        object.__delattr__(missing_top_level, "memory")

        missing_memory = replace(baseline.memory)
        object.__delattr__(missing_memory, "total_bytes")
        missing_memory_snapshot = replace(baseline)
        object.__setattr__(missing_memory_snapshot, "memory", missing_memory)

        missing_cgroup = replace(baseline.cgroup)
        object.__delattr__(missing_cgroup, "levels")
        missing_cgroup_snapshot = replace(baseline)
        object.__setattr__(missing_cgroup_snapshot, "cgroup", missing_cgroup)

        missing_memory_pressure = replace(baseline.pressure.memory)
        object.__delattr__(missing_memory_pressure, "some")
        missing_pressure = replace(baseline.pressure)
        object.__setattr__(
            missing_pressure,
            "memory",
            missing_memory_pressure,
        )
        missing_pressure_snapshot = replace(baseline)
        object.__setattr__(
            missing_pressure_snapshot,
            "pressure",
            missing_pressure,
        )

        for name, snapshot in (
            ("top-level", missing_top_level),
            ("memory-scalar", missing_memory_snapshot),
            ("cgroup-levels", missing_cgroup_snapshot),
            ("pressure-line", missing_pressure_snapshot),
        ):
            with (
                self.subTest(case=name),
                self.assertRaisesRegex(ResourceInputError, "shape is incomplete"),
            ):
                assess_publication_capacity(snapshot)

        with self.assertRaisesRegex(ResourceInputError, "shape is incomplete"):
            ResourceSnapshot(
                memory=missing_memory,
                cgroup=baseline.cgroup,
                filesystem=baseline.filesystem,
                pressure=baseline.pressure,
                limits=baseline.limits,
            )

    def test_public_capacity_boundaries_reject_low_level_snapshot_mutation(
        self,
    ) -> None:
        baseline = self._capture()
        failing = replace(
            baseline,
            memory=replace(
                baseline.memory,
                available_bytes=MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES - 1,
            ),
        )
        self.assertFalse(assess_publication_capacity(failing).ready)

        object.__setattr__(
            failing.memory,
            "available_bytes",
            MIN_PUBLICATION_AVAILABLE_MEMORY_BYTES,
        )
        with self.assertRaisesRegex(ResourceInputError, "integrity seal"):
            assess_publication_capacity(failing)
        with self.assertRaisesRegex(ResourceInputError, "integrity seal"):
            require_publication_capacity(failing)

        for invalid_seal in ("é", "A" * 64, "a" * 63, "g" * 64):
            with self.subTest(invalid_seal=invalid_seal):
                invalidly_sealed = self._capture()
                object.__setattr__(
                    invalidly_sealed,
                    "_integrity_sha256",
                    invalid_seal,
                )
                with self.assertRaisesRegex(ResourceInputError, "integrity seal"):
                    assess_publication_capacity(invalidly_sealed)

        invalid_pressure = replace(baseline.pressure.memory.some)
        object.__setattr__(invalid_pressure, "avg10", object())
        invalid_pressure_resource = replace(
            baseline.pressure.memory,
            some=invalid_pressure,
        )
        invalid_pressure_snapshot = replace(
            baseline.pressure,
            memory=invalid_pressure_resource,
        )
        with self.assertRaisesRegex(ResourceInputError, "invalid scalar type"):
            ResourceSnapshot(
                memory=baseline.memory,
                cgroup=baseline.cgroup,
                filesystem=baseline.filesystem,
                pressure=invalid_pressure_snapshot,
                limits=baseline.limits,
            )

        invalid_memory = replace(baseline.memory)
        object.__setattr__(
            invalid_memory,
            "available_bytes",
            invalid_memory.total_bytes + 1,
        )
        sealed_invalid = ResourceSnapshot(
            memory=invalid_memory,
            cgroup=baseline.cgroup,
            filesystem=baseline.filesystem,
            pressure=baseline.pressure,
            limits=baseline.limits,
        )
        with self.assertRaisesRegex(ResourceInputError, "closed validation"):
            assess_publication_capacity(sealed_invalid)

    def test_hierarchy_and_nested_value_invariants_fail_closed(self) -> None:
        level_a = CgroupLevelSnapshot(
            path="/a",
            memory_limit_bytes=None,
            memory_current_bytes=0,
            swap_limit_bytes=None,
            swap_current_bytes=0,
        )
        level_c = replace(level_a, path="/a/b/c")
        with self.assertRaisesRegex(ResourceInputError, "ancestor chain"):
            CgroupSnapshot(self_path="/a/b/c", levels=(level_a, level_c))
        with self.assertRaisesRegex(ResourceInputError, "path set"):
            CgroupSnapshot(self_path="/a", levels=(level_a, level_a))
        suffix_only = replace(level_a, path="/a/b")
        with self.assertRaisesRegex(ResourceInputError, "native root"):
            CgroupSnapshot(self_path="/a/b", levels=(suffix_only,))
        with self.assertRaisesRegex(ResourceInputError, "canonical"):
            replace(level_a, path="/a/../b")
        with self.assertRaisesRegex(ResourceInputError, "canonical"):
            replace(level_a, path="//a")
        with self.assertRaisesRegex(ResourceInputError, "exceeds"):
            MemorySnapshot(
                total_bytes=1,
                available_bytes=2,
                swap_total_bytes=0,
                swap_free_bytes=0,
            )
        with self.assertRaisesRegex(ResourceInputError, "exceed"):
            FilesystemSnapshot(
                total_bytes=1,
                available_bytes=0,
                total_inodes=1,
                available_inodes=2,
            )
        with self.assertRaisesRegex(ResourceInputError, "exceeds"):
            ProcessLimitSnapshot(nofile_soft=2, nofile_hard=1)
        line = PressureLine(
            avg10=0.0,
            avg60=0.0,
            avg300=0.0,
            total_microseconds=0,
        )
        with self.assertRaisesRegex(ResourceInputError, "invalid types"):
            PressureResourceSnapshot(
                some=line,
                full=cast(PressureLine, object()),
            )
        resource_pressure = PressureResourceSnapshot(some=line, full=line)
        with self.assertRaisesRegex(ResourceInputError, "invalid resource"):
            PressureSnapshot(
                memory=resource_pressure,
                io=cast(PressureResourceSnapshot, object()),
            )

    def test_public_capture_rejects_nonexact_or_relative_paths(self) -> None:
        with self.assertRaisesRegex(ResourceInputError, "absolute Path"):
            ResourcePaths(
                meminfo=Path("meminfo"),
                self_cgroup=self.self_cgroup,
                mountinfo=self.mountinfo,
                cgroup_namespace=self.cgroup_namespace,
                cgroup_root=self.cgroup_root,
                memory_pressure=self.memory_pressure,
                io_pressure=self.io_pressure,
            )
        for suffix in ("\x00", "\n", "\x7f", "\x80", "\ud800"):
            with (
                self.subTest(path_suffix=repr(suffix)),
                self.assertRaisesRegex(
                    ResourceInputError,
                    "canonical absolute Path",
                ),
            ):
                replace(
                    self.paths,
                    meminfo=Path(f"{self.meminfo.as_posix()}{suffix}"),
                )

        nul_capacity_path = NulPath(self.capacity_path)
        NulPath.fspath_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "capacity_path"):
            capture_resource_snapshot(
                cast(Path, nul_capacity_path),
                paths=self.paths,
            )
        self.assertEqual(NulPath.fspath_reads, 0)

        stateful_capacity_path = StatefulRedirectPath(self.capacity_path)
        StatefulRedirectPath.path_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "capacity_path"):
            capture_resource_snapshot(
                cast(Path, stateful_capacity_path),
                paths=self.paths,
            )
        self.assertEqual(StatefulRedirectPath.path_reads, 0)

        empty_path = object.__new__(type(Path()))
        with self.assertRaisesRegex(ResourceInputError, "capacity_path"):
            capture_resource_snapshot(empty_path, paths=self.paths)
        with self.assertRaisesRegex(ResourceInputError, "canonical absolute Path"):
            ResourcePaths(
                meminfo=empty_path,
                self_cgroup=self.self_cgroup,
                mountinfo=self.mountinfo,
                cgroup_namespace=self.cgroup_namespace,
                cgroup_root=self.cgroup_root,
                memory_pressure=self.memory_pressure,
                io_pressure=self.io_pressure,
            )

        stateful_component_path = Path(self.capacity_path)
        stateful_components = cast(
            list[str],
            object.__getattribute__(
                stateful_component_path,
                preflight._PATH_RAW_COMPONENTS_SLOT,
            ),
        )
        stateful_components[0] = cast(
            str,
            StatefulPathComponent(stateful_components[0]),
        )
        StatefulPathComponent.path_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "capacity_path"):
            capture_resource_snapshot(
                stateful_component_path,
                paths=self.paths,
            )
        self.assertEqual(StatefulPathComponent.path_reads, 0)

        configured_stateful_component = Path(self.meminfo)
        configured_components = cast(
            list[str],
            object.__getattribute__(
                configured_stateful_component,
                preflight._PATH_RAW_COMPONENTS_SLOT,
            ),
        )
        configured_components[0] = cast(
            str,
            StatefulPathComponent(configured_components[0]),
        )
        StatefulPathComponent.path_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "canonical absolute Path"):
            ResourcePaths(
                meminfo=configured_stateful_component,
                self_cgroup=self.self_cgroup,
                mountinfo=self.mountinfo,
                cgroup_namespace=self.cgroup_namespace,
                cgroup_root=self.cgroup_root,
                memory_pressure=self.memory_pressure,
                io_pressure=self.io_pressure,
            )
        self.assertEqual(StatefulPathComponent.path_reads, 0)

        low_level_component_paths = replace(self.paths)
        low_level_component_path = Path(self.meminfo)
        low_level_components = cast(
            list[str],
            object.__getattribute__(
                low_level_component_path,
                preflight._PATH_RAW_COMPONENTS_SLOT,
            ),
        )
        low_level_components[0] = cast(
            str,
            StatefulPathComponent(low_level_components[0]),
        )
        object.__setattr__(
            low_level_component_paths,
            "meminfo",
            low_level_component_path,
        )
        StatefulPathComponent.path_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "closed validation"):
            capture_resource_snapshot(
                self.capacity_path,
                paths=low_level_component_paths,
            )
        self.assertEqual(StatefulPathComponent.path_reads, 0)

        NulPath.fspath_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "canonical absolute Path"):
            replace(
                self.paths,
                meminfo=cast(Path, NulPath(self.meminfo)),
            )
        self.assertEqual(NulPath.fspath_reads, 0)

        configured_redirect = StatefulRedirectPath(self.meminfo)
        StatefulRedirectPath.path_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "canonical absolute Path"):
            ResourcePaths(
                meminfo=cast(Path, configured_redirect),
                self_cgroup=self.self_cgroup,
                mountinfo=self.mountinfo,
                cgroup_namespace=self.cgroup_namespace,
                cgroup_root=self.cgroup_root,
                memory_pressure=self.memory_pressure,
                io_pressure=self.io_pressure,
            )
        self.assertEqual(StatefulRedirectPath.path_reads, 0)

        mutated_redirect_paths = replace(self.paths)
        low_level_redirect = StatefulRedirectPath(self.meminfo)
        object.__setattr__(
            mutated_redirect_paths,
            "meminfo",
            cast(Path, low_level_redirect),
        )
        StatefulRedirectPath.path_reads = 0
        with self.assertRaisesRegex(ResourceInputError, "closed validation"):
            capture_resource_snapshot(
                self.capacity_path,
                paths=mutated_redirect_paths,
            )
        self.assertEqual(StatefulRedirectPath.path_reads, 0)

        with self.assertRaisesRegex(ResourceInputError, "capacity_path"):
            capture_resource_snapshot(Path("relative"), paths=self.paths)
        with self.assertRaisesRegex(ResourceInputError, "capacity_path"):
            capture_resource_snapshot(
                Path(f"{self.capacity_path.as_posix()}\x00"),
                paths=self.paths,
            )
        with self.assertRaisesRegex(ResourceInputError, "capacity_path"):
            capture_resource_snapshot(
                Path(f"{self.capacity_path.as_posix()}\ud800"),
                paths=self.paths,
            )
        with self.assertRaisesRegex(ResourceInputError, "exact ResourcePaths"):
            capture_resource_snapshot(
                self.capacity_path,
                paths=cast(ResourcePaths, object()),
            )
        mutated_paths = replace(self.paths)
        object.__setattr__(mutated_paths, "meminfo", Path("relative"))
        with self.assertRaisesRegex(ResourceInputError, "closed validation"):
            capture_resource_snapshot(
                self.capacity_path,
                paths=mutated_paths,
            )

        capacity_file = self.root / "capacity-file"
        capacity_file.write_bytes(b"not a directory")
        with self.assertRaisesRegex(ResourceSnapshotError, "must be a directory"):
            capture_resource_snapshot(capacity_file, paths=self.paths)

        capacity_link = self.root / "capacity-link"
        capacity_link.symlink_to(self.capacity_path, target_is_directory=True)
        with self.assertRaisesRegex(ResourceSnapshotError, "must be a directory"):
            capture_resource_snapshot(capacity_link, paths=self.paths)


if __name__ == "__main__":
    unittest.main()
