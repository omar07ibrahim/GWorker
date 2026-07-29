from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import io
import json
import stat
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import verify_distribution


def _record(files: dict[str, bytes], record_path: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name in sorted(files):
        digest = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).rstrip(
            b"="
        )
        writer.writerow((name, f"sha256={digest.decode('ascii')}", len(files[name])))
    writer.writerow((record_path, "", ""))
    return output.getvalue().encode()


def _wheel_files() -> dict[str, bytes]:
    config = verify_distribution._load_project_config(verify_distribution.ROOT)
    prefix = f"{config.dist_info}/"
    files = verify_distribution._expected_runtime_files(verify_distribution.ROOT)
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {config.name}\n"
        f"Version: {config.version}\n"
        f"Requires-Python: {config.requires_python}\n"
        "Provides-Extra: dev\n"
        'Requires-Dist: build==1.5.0; extra == "dev"\n'
        'Requires-Dist: coverage[toml]==7.15.2; extra == "dev"\n'
        'Requires-Dist: mypy==2.3.0; extra == "dev"\n'
        'Requires-Dist: Pillow==12.3.0; extra == "dev"\n'
        'Requires-Dist: pip-audit==2.10.1; extra == "dev"\n'
        'Requires-Dist: ruff==0.16.0; extra == "dev"\n'
        "\n"
    ).encode()
    files.update(
        {
            f"{prefix}METADATA": metadata,
            f"{prefix}WHEEL": (
                b"Wheel-Version: 1.0\n"
                b"Generator: synthetic-test-fixture\n"
                b"Root-Is-Purelib: true\n"
                b"Tag: py3-none-any\n"
                b"\n"
            ),
            f"{prefix}entry_points.txt": (
                b"[console_scripts]\ngworker = gworker.cli:main\n"
            ),
            f"{prefix}top_level.txt": b"gworker\n",
        }
    )
    record_path = f"{prefix}RECORD"
    files[record_path] = _record(files, record_path)
    return files


def _wheel_bytes(
    files: dict[str, bytes],
    *,
    special_members: tuple[tuple[str, int, bytes], ...] = (),
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, content)
        for name, mode, content in special_members:
            info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, content)
    return output.getvalue()


def _sdist_files() -> dict[str, bytes]:
    config = verify_distribution._load_project_config(verify_distribution.ROOT)
    metadata = _wheel_files()[f"{config.dist_info}/METADATA"]
    scopes = verify_distribution._expected_sdist_scopes(verify_distribution.ROOT)
    files = {
        path: content
        for scoped_files in scopes.values()
        for path, content in scoped_files.items()
    }
    files.update(
        {
            "PKG-INFO": metadata,
            "setup.cfg": b"[egg_info]\n",
            f"{config.egg_info}/PKG-INFO": metadata,
            f"{config.egg_info}/SOURCES.txt": b"fixture\n",
        }
    )
    return files


def _sdist_bytes(
    files: dict[str, bytes],
    *,
    special_members: tuple[tarfile.TarInfo, ...] = (),
) -> bytes:
    config = verify_distribution._load_project_config(verify_distribution.ROOT)
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT
    ) as archive:
        root = tarfile.TarInfo(config.sdist_root)
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        archive.addfile(root)
        for relative, content in sorted(files.items()):
            info = tarfile.TarInfo(f"{config.sdist_root}/{relative}")
            info.size = len(content)
            info.mode = 0o755 if relative.startswith("scripts/") else 0o644
            archive.addfile(info, io.BytesIO(content))
        for member in special_members:
            archive.addfile(member, io.BytesIO(b"x") if member.isreg() else None)
    return output.getvalue()


class DistributionVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = verify_distribution._load_project_config(verify_distribution.ROOT)
        cls.base_wheel_files = _wheel_files()
        cls.base_sdist_files = _sdist_files()
        cls.base_wheel = _wheel_bytes(cls.base_wheel_files)
        cls.base_sdist = _sdist_bytes(cls.base_sdist_files)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.primary = root / "primary"
        self.rebuild = root / "rebuild"
        self.primary.mkdir()
        self.rebuild.mkdir()
        self._write_artifacts(self.base_wheel, self.base_sdist)

    def _write_artifacts(self, wheel: bytes, sdist: bytes) -> None:
        (self.primary / self.config.wheel_name).write_bytes(wheel)
        (self.primary / self.config.sdist_name).write_bytes(sdist)
        (self.rebuild / self.config.wheel_name).write_bytes(wheel)

    def _replace_wheels(self, wheel: bytes) -> None:
        (self.primary / self.config.wheel_name).write_bytes(wheel)
        (self.rebuild / self.config.wheel_name).write_bytes(wheel)

    def test_valid_pair_reports_exact_contract_and_no_sdist_reproducibility(
        self,
    ) -> None:
        report = verify_distribution.verify_distribution(
            self.primary,
            self.rebuild,
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["wheel_reproducibility"]["byte_for_byte"])  # type: ignore[index]
        sdist = report["sdist_verification"]
        assert isinstance(sdist, dict)
        reproducibility = sdist["byte_reproducibility"]
        assert isinstance(reproducibility, dict)
        self.assertEqual(reproducibility["status"], "not-checked")
        self.assertFalse(reproducibility["claimed"])
        self.assertEqual(
            sdist["scope_files"],
            {
                "config": 4,
                "docs": len(
                    verify_distribution._repository_files(
                        verify_distribution.ROOT,
                        "docs",
                        suffixes=verify_distribution.DOC_SUFFIXES,
                    )
                ),
                "scripts": len(
                    verify_distribution._repository_files(
                        verify_distribution.ROOT,
                        "scripts",
                        suffixes=frozenset({".py"}),
                    )
                ),
                "src": len(
                    verify_distribution._repository_files(
                        verify_distribution.ROOT,
                        "src/gworker",
                    )
                ),
                "tests": len(
                    verify_distribution._repository_files(
                        verify_distribution.ROOT,
                        "tests",
                        suffixes=frozenset({".py"}),
                    )
                ),
            },
        )

    def test_json_cli_is_canonical_and_read_only(self) -> None:
        before = {
            path: path.read_bytes()
            for directory in (self.primary, self.rebuild)
            for path in directory.iterdir()
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = verify_distribution.main(
                [str(self.primary), str(self.rebuild), "--json"]
            )

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertEqual(rendered.count("\n"), 1)
        self.assertEqual(
            rendered,
            verify_distribution._canonical_json(json.loads(rendered)),
        )
        self.assertEqual(
            before,
            {
                path: path.read_bytes()
                for directory in (self.primary, self.rebuild)
                for path in directory.iterdir()
            },
        )

    def test_extra_primary_entry_fails_closed(self) -> None:
        (self.primary / "unexpected.txt").write_text("not an artifact")
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "exactly one wheel and one sdist",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_non_identical_rebuilt_wheel_is_rejected(self) -> None:
        rebuilt = self.rebuild / self.config.wheel_name
        rebuilt.write_bytes(rebuilt.read_bytes() + b"different")
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "not byte-for-byte identical",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_runtime_content_must_match_repository_source(self) -> None:
        files = dict(self.base_wheel_files)
        files["gworker/__init__.py"] = b'"""tampered."""\n'
        record_path = f"{self.config.dist_info}/RECORD"
        files.pop(record_path)
        files[record_path] = _record(files, record_path)
        self._replace_wheels(_wheel_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "differs from repository source",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_every_record_hash_is_verified(self) -> None:
        files = dict(self.base_wheel_files)
        record_path = f"{self.config.dist_info}/RECORD"
        files[record_path] = files[record_path].replace(
            b"sha256=",
            b"sha256=A",
            1,
        )
        self._replace_wheels(_wheel_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "RECORD hash mismatch",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_unconditional_runtime_dependency_is_rejected(self) -> None:
        files = dict(self.base_wheel_files)
        metadata_path = f"{self.config.dist_info}/METADATA"
        files[metadata_path] = files[metadata_path].replace(
            b"\n\n",
            b"\nRequires-Dist: requests>=2\n\n",
            1,
        )
        record_path = f"{self.config.dist_info}/RECORD"
        files.pop(record_path)
        files[record_path] = _record(files, record_path)
        self._replace_wheels(_wheel_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "Requires-Dist headers",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_undeclared_conditional_dependency_is_rejected(self) -> None:
        files = dict(self.base_wheel_files)
        metadata_path = f"{self.config.dist_info}/METADATA"
        files[metadata_path] = files[metadata_path].replace(
            b"\n\n",
            b'\nRequires-Dist: not-declared==99; extra == "dev"\n\n',
            1,
        )
        record_path = f"{self.config.dist_info}/RECORD"
        files.pop(record_path)
        files[record_path] = _record(files, record_path)
        self._replace_wheels(_wheel_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "Requires-Dist headers",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_provided_extras_must_match_pyproject(self) -> None:
        files = dict(self.base_wheel_files)
        metadata_path = f"{self.config.dist_info}/METADATA"
        files[metadata_path] = files[metadata_path].replace(
            b"Provides-Extra: dev\n",
            b"Provides-Extra: dev\nProvides-Extra: hidden\n",
            1,
        )
        record_path = f"{self.config.dist_info}/RECORD"
        files.pop(record_path)
        files[record_path] = _record(files, record_path)
        self._replace_wheels(_wheel_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "Provides-Extra headers",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_missing_declared_dependency_is_rejected(self) -> None:
        files = dict(self.base_wheel_files)
        metadata_path = f"{self.config.dist_info}/METADATA"
        files[metadata_path] = files[metadata_path].replace(
            b'Requires-Dist: build==1.5.0; extra == "dev"\n',
            b"",
            1,
        )
        record_path = f"{self.config.dist_info}/RECORD"
        files.pop(record_path)
        files[record_path] = _record(files, record_path)
        self._replace_wheels(_wheel_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "Requires-Dist headers",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_duplicate_declared_dependency_is_rejected(self) -> None:
        files = dict(self.base_wheel_files)
        metadata_path = f"{self.config.dist_info}/METADATA"
        requirement = b'Requires-Dist: build==1.5.0; extra == "dev"\n'
        files[metadata_path] = files[metadata_path].replace(
            requirement,
            requirement * 2,
            1,
        )
        record_path = f"{self.config.dist_info}/RECORD"
        files.pop(record_path)
        files[record_path] = _record(files, record_path)
        self._replace_wheels(_wheel_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "Requires-Dist headers",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_unsafe_wheel_path_is_rejected(self) -> None:
        wheel = _wheel_bytes(
            self.base_wheel_files,
            special_members=(("../escape", stat.S_IFREG | 0o644, b"x"),),
        )
        self._replace_wheels(wheel)
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "unsafe archive member path",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_wheel_symlink_is_rejected(self) -> None:
        wheel = _wheel_bytes(
            self.base_wheel_files,
            special_members=(("gworker/link", stat.S_IFLNK | 0o777, b"cli.py"),),
        )
        self._replace_wheels(wheel)
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "non-regular archive type",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_missing_sdist_script_is_rejected(self) -> None:
        files = dict(self.base_sdist_files)
        missing = next(name for name in files if name.startswith("scripts/"))
        files.pop(missing)
        (self.primary / self.config.sdist_name).write_bytes(_sdist_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "missing required repository file",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_sdist_root_metadata_must_match_verified_wheel(self) -> None:
        files = dict(self.base_sdist_files)
        files["PKG-INFO"] = files["PKG-INFO"].replace(
            f"Name: {self.config.name}".encode(),
            b"Name: unrelated",
            1,
        )
        (self.primary / self.config.sdist_name).write_bytes(_sdist_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "differs from verified wheel METADATA",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_sdist_egg_metadata_must_match_verified_wheel(self) -> None:
        files = dict(self.base_sdist_files)
        egg_metadata = f"{self.config.egg_info}/PKG-INFO"
        files[egg_metadata] = files[egg_metadata].replace(
            f"Version: {self.config.version}".encode(),
            b"Version: 999",
            1,
        )
        (self.primary / self.config.sdist_name).write_bytes(_sdist_bytes(files))
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "differs from verified wheel METADATA",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_sdist_traversal_is_rejected(self) -> None:
        unsafe = tarfile.TarInfo(f"{self.config.sdist_root}/../escape")
        unsafe.size = 1
        unsafe.mode = 0o644
        sdist = _sdist_bytes(
            self.base_sdist_files,
            special_members=(unsafe,),
        )
        (self.primary / self.config.sdist_name).write_bytes(sdist)
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "unsafe archive member path",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_sdist_symlink_is_rejected(self) -> None:
        unsafe = tarfile.TarInfo(f"{self.config.sdist_root}/docs/link")
        unsafe.type = tarfile.SYMTYPE
        unsafe.linkname = "../../README.md"
        unsafe.mode = 0o777
        sdist = _sdist_bytes(
            self.base_sdist_files,
            special_members=(unsafe,),
        )
        (self.primary / self.config.sdist_name).write_bytes(sdist)
        with self.assertRaisesRegex(
            verify_distribution.VerificationError,
            "link or special archive type",
        ):
            verify_distribution.verify_distribution(self.primary, self.rebuild)

    def test_manifest_recursively_includes_evidence_and_python_sources(
        self,
    ) -> None:
        manifest = (verify_distribution.ROOT / "MANIFEST.in").read_text()
        self.assertIn("recursive-include scripts *.py", manifest)
        self.assertIn(
            "recursive-include docs *.md *.json *.svg *.txt *.gif *.png",
            manifest,
        )
        self.assertIn("recursive-include tests *.py", manifest)
        self.assertIn("include .github/workflows/ci.yml", manifest)


if __name__ == "__main__":
    unittest.main()
