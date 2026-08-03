#!/usr/bin/env python3
"""Fail-closed, read-only verification for GWorker distribution archives."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import re
import stat
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "gworker-distribution-verification-v1"
MAX_ARCHIVE_FILES = 10_000
MAX_ARCHIVE_FILE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 128 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 2_000
DOC_SUFFIXES = frozenset({".gif", ".json", ".md", ".png", ".svg", ".txt"})
CONFIG_PATHS = (
    ".github/workflows/ci.yml",
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
)
GENERATED_SDIST_ROOT_FILES = frozenset({"PKG-INFO", "setup.cfg"})
GENERATED_EGG_INFO_FILES = frozenset(
    {
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "entry_points.txt",
        "requires.txt",
        "top_level.txt",
    }
)
SAFE_PROJECT_COMPONENT = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
REQUIRES_PYTHON_CLAUSE = re.compile(r"\A(?:~=|==|!=|<=|>=|<|>)\d+(?:\.\d+)*(?:\.\*)?\Z")


class VerificationError(RuntimeError):
    """Raised when an artifact violates the distribution contract."""


class _CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


@dataclass(frozen=True)
class ProjectConfig:
    """The packaging fields that the verifier binds to repository state."""

    name: str
    version: str
    requires_python: str
    metadata_requires_python: str
    console_scripts: Mapping[str, str]
    dependencies: tuple[str, ...]
    optional_dependencies: Mapping[str, tuple[str, ...]]

    @property
    def distribution_token(self) -> str:
        return re.sub(r"[-_.]+", "_", self.name)

    @property
    def normalized_name(self) -> str:
        return re.sub(r"[-_.]+", "-", self.name).lower()

    @property
    def version_token(self) -> str:
        return self.version.replace("-", "_")

    @property
    def wheel_name(self) -> str:
        return f"{self.distribution_token}-{self.version_token}-py3-none-any.whl"

    @property
    def sdist_name(self) -> str:
        return f"{self.distribution_token}-{self.version}.tar.gz"

    @property
    def dist_info(self) -> str:
        return f"{self.distribution_token}-{self.version_token}.dist-info"

    @property
    def sdist_root(self) -> str:
        return f"{self.distribution_token}-{self.version}"

    @property
    def egg_info(self) -> str:
        return f"src/{self.distribution_token}.egg-info"


@dataclass(frozen=True)
class SelectedArtifacts:
    """The only accepted artifacts in the primary and rebuild directories."""

    primary_wheel: Path
    primary_sdist: Path
    rebuilt_wheel: Path


@dataclass(frozen=True)
class LoadedArchive:
    """Validated regular-file contents from an archive."""

    files: Mapping[str, bytes]
    directories: frozenset[str]


def _fail(message: str) -> NoReturn:
    raise VerificationError(message)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(document: Mapping[str, object]) -> str:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _required_string(
    table: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        _fail(f"{context}.{key} must be a non-empty string")
    return value


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail(f"{context} must be a string-keyed table")
    return value


def _canonical_metadata_requires_python(value: str) -> str:
    """Return the deterministic clause order emitted into package metadata."""

    clauses = value.split(",")
    if any(REQUIRES_PYTHON_CLAUSE.fullmatch(clause) is None for clause in clauses):
        _fail(
            "project.requires-python must contain comma-separated, "
            "whitespace-free release clauses"
        )
    if len(set(clauses)) != len(clauses):
        _fail("project.requires-python must not contain duplicate clauses")
    return ",".join(sorted(clauses))


def _load_project_config(repo_root: Path) -> ProjectConfig:
    config_path = repo_root / "pyproject.toml"
    if config_path.is_symlink() or not config_path.is_file():
        _fail("repository pyproject.toml must be a regular file")
    try:
        with config_path.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        _fail(f"cannot parse repository pyproject.toml: {error}")

    project = _mapping(document.get("project"), context="project")
    name = _required_string(project, "name", context="project")
    version = _required_string(project, "version", context="project")
    requires_python = _required_string(
        project,
        "requires-python",
        context="project",
    )
    metadata_requires_python = _canonical_metadata_requires_python(requires_python)
    if SAFE_PROJECT_COMPONENT.fullmatch(name) is None:
        _fail("project.name contains an unsupported filename character")
    if SAFE_PROJECT_COMPONENT.fullmatch(version) is None:
        _fail("project.version contains an unsupported filename character")

    scripts_value = project.get("scripts", {})
    scripts = _mapping(scripts_value, context="project.scripts")
    if not scripts:
        _fail("project.scripts must declare at least one console entry point")
    console_scripts: dict[str, str] = {}
    for entry_name, target in scripts.items():
        if not isinstance(target, str) or not target:
            _fail(f"project.scripts.{entry_name} must be a non-empty string")
        console_scripts[entry_name] = target

    optional_value = project.get("optional-dependencies", {})
    optional = _mapping(
        optional_value,
        context="project.optional-dependencies",
    )
    optional_dependencies: dict[str, tuple[str, ...]] = {}
    normalized_extras: set[str] = set()
    for extra, dependencies in optional.items():
        if SAFE_PROJECT_COMPONENT.fullmatch(extra) is None:
            _fail(
                f"project.optional-dependencies extra {extra!r} "
                "contains an unsupported character"
            )
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) and dependency for dependency in dependencies
        ):
            _fail(
                "project.optional-dependencies."
                f"{extra} must be a list of non-empty strings"
            )
        normalized_extra = re.sub(r"[-_.]+", "-", extra).lower()
        if normalized_extra in normalized_extras:
            _fail(
                "project.optional-dependencies contains extras that normalize "
                f"to the same name: {extra!r}"
            )
        validated_dependencies = tuple(
            dependency for dependency in dependencies if isinstance(dependency, str)
        )
        if len(set(validated_dependencies)) != len(validated_dependencies):
            _fail(
                "project.optional-dependencies."
                f"{extra} must not contain duplicate requirements"
            )
        if any(";" in dependency for dependency in validated_dependencies):
            _fail(
                "project.optional-dependencies environment markers are outside "
                "this repository's distribution contract"
            )
        normalized_extras.add(normalized_extra)
        optional_dependencies[normalized_extra] = validated_dependencies

    dependencies_value = project.get("dependencies", [])
    if not isinstance(dependencies_value, list) or not all(
        isinstance(dependency, str) and dependency for dependency in dependencies_value
    ):
        _fail("project.dependencies must be a list of non-empty strings")
    dependencies = tuple(
        dependency for dependency in dependencies_value if isinstance(dependency, str)
    )
    if len(set(dependencies)) != len(dependencies):
        _fail("project.dependencies must not contain duplicate requirements")
    if any(";" in dependency for dependency in dependencies):
        _fail(
            "project.dependencies environment markers are outside this "
            "repository's distribution contract"
        )

    return ProjectConfig(
        name=name,
        version=version,
        requires_python=requires_python,
        metadata_requires_python=metadata_requires_python,
        console_scripts=console_scripts,
        dependencies=dependencies,
        optional_dependencies=optional_dependencies,
    )


def _select_artifacts(
    primary_dir: Path,
    rebuild_dir: Path,
    config: ProjectConfig,
) -> SelectedArtifacts:
    primary_files = _directory_files(primary_dir, label="primary")
    rebuild_files = _directory_files(rebuild_dir, label="rebuild")

    if len(primary_files) != 2:
        _fail(
            "primary distribution directory must contain exactly one wheel "
            "and one sdist"
        )
    if len(rebuild_files) != 1:
        _fail("rebuild distribution directory must contain exactly one wheel")

    primary_wheels = [path for path in primary_files if path.name.endswith(".whl")]
    primary_sdists = [path for path in primary_files if path.name.endswith(".tar.gz")]
    rebuild_wheels = [path for path in rebuild_files if path.name.endswith(".whl")]
    if len(primary_wheels) != 1 or len(primary_sdists) != 1:
        _fail(
            "primary distribution directory must contain exactly one .whl "
            "and one .tar.gz"
        )
    if len(rebuild_wheels) != 1:
        _fail("rebuild distribution directory must contain exactly one .whl")

    primary_wheel = primary_wheels[0]
    primary_sdist = primary_sdists[0]
    rebuilt_wheel = rebuild_wheels[0]
    if primary_wheel.name != config.wheel_name:
        _fail(
            f"primary wheel name must be {config.wheel_name!r}, "
            f"found {primary_wheel.name!r}"
        )
    if rebuilt_wheel.name != config.wheel_name:
        _fail(
            f"rebuilt wheel name must be {config.wheel_name!r}, "
            f"found {rebuilt_wheel.name!r}"
        )
    if primary_sdist.name != config.sdist_name:
        _fail(f"sdist name must be {config.sdist_name!r}, found {primary_sdist.name!r}")
    return SelectedArtifacts(
        primary_wheel=primary_wheel,
        primary_sdist=primary_sdist,
        rebuilt_wheel=rebuilt_wheel,
    )


def _directory_files(directory: Path, *, label: str) -> tuple[Path, ...]:
    if directory.is_symlink() or not directory.is_dir():
        _fail(f"{label} distribution path must be a real directory")
    try:
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError as error:
        _fail(f"cannot inspect {label} distribution directory: {error}")
    if not entries:
        _fail(f"{label} distribution directory is empty")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            _fail(
                f"{label} distribution directory contains a non-regular "
                f"entry: {entry.name!r}"
            )
    return tuple(entries)


def _validate_archive_name(name: str, *, directory: bool) -> str:
    if not name or "\x00" in name or "\\" in name:
        _fail(f"unsafe archive member name: {name!r}")
    if name.startswith("/") or re.match(r"\A[A-Za-z]:", name):
        _fail(f"unsafe absolute archive member name: {name!r}")
    if directory:
        if not name.endswith("/"):
            _fail(f"directory archive member must end with '/': {name!r}")
        candidate = name[:-1]
    else:
        if name.endswith("/"):
            _fail(f"file archive member must not end with '/': {name!r}")
        candidate = name
    parts = candidate.split("/")
    if not candidate or any(part in {"", ".", ".."} for part in parts):
        _fail(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(candidate)
    if path.is_absolute() or str(path) != candidate:
        _fail(f"non-canonical archive member path: {name!r}")
    return candidate


def _load_wheel(path: Path) -> LoadedArchive:
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    total_size = 0
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_FILES:
                _fail("wheel contains too many archive members")
            for member in members:
                is_directory = member.is_dir()
                name = _validate_archive_name(
                    member.filename,
                    directory=is_directory,
                )
                if name in files or name in directories:
                    _fail(f"wheel contains duplicate archive member {name!r}")
                if member.flag_bits & 0x1:
                    _fail(f"wheel contains encrypted archive member {name!r}")
                mode = (member.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                allowed_type = stat.S_IFDIR if is_directory else stat.S_IFREG
                if file_type not in {0, allowed_type}:
                    _fail(f"wheel contains a non-regular archive type at {name!r}")
                if is_directory:
                    directories.add(name)
                    continue
                if member.file_size > MAX_ARCHIVE_FILE_BYTES:
                    _fail(f"wheel member exceeds the size limit: {name!r}")
                total_size += member.file_size
                if total_size > MAX_ARCHIVE_TOTAL_BYTES:
                    _fail("wheel uncompressed contents exceed the size limit")
                if (member.file_size > 0 and member.compress_size == 0) or (
                    member.compress_size > 0
                    and member.file_size
                    > member.compress_size * MAX_ZIP_COMPRESSION_RATIO
                ):
                    _fail(f"wheel member has an unsafe compression ratio: {name!r}")
                content = archive.read(member)
                if len(content) != member.file_size:
                    _fail(f"wheel member size changed while reading: {name!r}")
                files[name] = content
    except VerificationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        _fail(f"cannot safely read wheel {path.name!r}: {error}")
    return LoadedArchive(files=files, directories=frozenset(directories))


def _load_sdist(path: Path, *, expected_root: str) -> LoadedArchive:
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    total_size = 0
    root_seen = False
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > MAX_ARCHIVE_FILES:
                _fail("sdist contains too many archive members")
            for member in members:
                is_directory = member.isdir()
                if not (is_directory or member.isreg()):
                    _fail(
                        "sdist contains a link or special archive type at "
                        f"{member.name!r}"
                    )
                raw_name = member.name + ("/" if is_directory else "")
                name = _validate_archive_name(
                    raw_name,
                    directory=is_directory,
                )
                if member.mode & 0o7000:
                    _fail(f"sdist member has unsafe special permission bits: {name!r}")
                if name == expected_root and is_directory:
                    root_seen = True
                elif not name.startswith(f"{expected_root}/"):
                    _fail(
                        "sdist member escapes the expected top-level directory: "
                        f"{name!r}"
                    )
                if name in files or name in directories:
                    _fail(f"sdist contains duplicate archive member {name!r}")
                if is_directory:
                    directories.add(name)
                    continue
                if member.size > MAX_ARCHIVE_FILE_BYTES:
                    _fail(f"sdist member exceeds the size limit: {name!r}")
                total_size += member.size
                if total_size > MAX_ARCHIVE_TOTAL_BYTES:
                    _fail("sdist contents exceed the size limit")
                extracted = archive.extractfile(member)
                if extracted is None:
                    _fail(f"cannot read regular sdist member {name!r}")
                content = extracted.read(MAX_ARCHIVE_FILE_BYTES + 1)
                if len(content) != member.size:
                    _fail(f"sdist member size changed while reading: {name!r}")
                files[name] = content
    except VerificationError:
        raise
    except (OSError, tarfile.TarError) as error:
        _fail(f"cannot safely read sdist {path.name!r}: {error}")
    if not root_seen:
        _fail("sdist is missing its explicit top-level directory")
    return LoadedArchive(files=files, directories=frozenset(directories))


def _repository_files(
    repo_root: Path,
    directory: str,
    *,
    suffixes: frozenset[str] | None = None,
) -> dict[str, bytes]:
    root = repo_root / directory
    if root.is_symlink() or not root.is_dir():
        _fail(f"repository directory {directory!r} must be a real directory")
    expected: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        relative_parts = path.relative_to(root).parts
        if "__pycache__" in relative_parts:
            continue
        if path.is_symlink():
            _fail(f"repository scope contains a symlink: {path.relative_to(repo_root)}")
        if not path.is_file():
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        relative = path.relative_to(repo_root).as_posix()
        try:
            expected[relative] = path.read_bytes()
        except OSError as error:
            _fail(f"cannot read repository file {relative!r}: {error}")
    if not expected:
        _fail(f"repository scope {directory!r} has no required files")
    return expected


def _expected_runtime_files(repo_root: Path) -> dict[str, bytes]:
    scoped = _repository_files(repo_root, "src/gworker")
    unexpected = sorted(
        relative
        for relative in scoped
        if not (relative.endswith(".py") or relative.endswith("/py.typed"))
    )
    if unexpected:
        _fail(
            "src/gworker contains unsupported runtime files: "
            + ", ".join(repr(path) for path in unexpected)
        )
    return {
        relative.removeprefix("src/"): content for relative, content in scoped.items()
    }


def _single_header(message: Message, name: str) -> str:
    values = message.get_all(name, [])
    if len(values) != 1:
        _fail(f"wheel METADATA must contain exactly one {name!r} header")
    value = values[0]
    if not isinstance(value, str) or not value:
        _fail(f"wheel METADATA {name!r} header must be non-empty")
    return value


def _parse_metadata(content: bytes) -> Message:
    try:
        message = BytesParser(policy=policy.default).parsebytes(content)
    except (UnicodeDecodeError, ValueError) as error:
        _fail(f"cannot parse wheel METADATA: {error}")
    if message.defects:
        _fail(f"wheel METADATA contains parser defects: {message.defects!r}")
    return message


def _verify_metadata(
    content: bytes,
    config: ProjectConfig,
) -> dict[str, object]:
    message = _parse_metadata(content)
    metadata_name = _single_header(message, "Name")
    if re.sub(r"[-_.]+", "-", metadata_name).lower() != config.normalized_name:
        _fail(f"wheel METADATA Name is {metadata_name!r}, expected {config.name!r}")
    metadata_version = _single_header(message, "Version")
    if metadata_version != config.version:
        _fail(
            "wheel METADATA Version is "
            f"{metadata_version!r}, expected {config.version!r}"
        )
    requires_python = _single_header(message, "Requires-Python")
    if requires_python != config.metadata_requires_python:
        _fail(
            "wheel METADATA Requires-Python is "
            f"{requires_python!r}, expected {config.metadata_requires_python!r}"
        )

    provided_extras = message.get_all("Provides-Extra", [])
    if not all(isinstance(extra, str) for extra in provided_extras):
        _fail("wheel METADATA contains a non-text Provides-Extra header")
    expected_extras = list(config.optional_dependencies)
    if provided_extras != expected_extras:
        _fail(
            "wheel METADATA Provides-Extra headers are "
            f"{provided_extras!r}, expected {expected_extras!r}"
        )

    requirements = message.get_all("Requires-Dist", [])
    if not all(isinstance(requirement, str) for requirement in requirements):
        _fail("wheel METADATA contains a non-text Requires-Dist header")
    expected_requirements = list(config.dependencies)
    expected_requirements.extend(
        f'{dependency}; extra == "{extra}"'
        for extra, dependencies in config.optional_dependencies.items()
        for dependency in dependencies
    )
    if requirements != expected_requirements:
        _fail(
            "wheel METADATA Requires-Dist headers are "
            f"{requirements!r}, expected {expected_requirements!r}"
        )

    return {
        "name": metadata_name,
        "requires_dist_total": len(requirements),
        "requires_python": requires_python,
        "unconditional_runtime_requires_dist": len(config.dependencies),
        "version": metadata_version,
        "conditional_extras": expected_extras,
    }


def _verify_wheel_headers(content: bytes) -> dict[str, object]:
    message = BytesParser(policy=policy.default).parsebytes(content)
    if message.defects:
        _fail(f"WHEEL metadata contains parser defects: {message.defects!r}")
    if _single_header(message, "Root-Is-Purelib").lower() != "true":
        _fail("wheel must declare Root-Is-Purelib: true")
    tags = message.get_all("Tag", [])
    if tags != ["py3-none-any"]:
        _fail(f"wheel must declare exactly one py3-none-any tag, found {tags!r}")
    wheel_version = _single_header(message, "Wheel-Version")
    if wheel_version != "1.0":
        _fail(f"unsupported Wheel-Version {wheel_version!r}")
    return {
        "root_is_purelib": True,
        "tag": "py3-none-any",
        "wheel_version": wheel_version,
    }


def _verify_entry_points(
    content: bytes,
    expected: Mapping[str, str],
) -> dict[str, str]:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        _fail(f"entry_points.txt is not UTF-8: {error}")
    parser = _CaseSensitiveConfigParser(
        interpolation=None,
        strict=True,
    )
    try:
        parser.read_string(text)
    except configparser.Error as error:
        _fail(f"cannot parse wheel entry_points.txt: {error}")
    if not parser.has_section("console_scripts"):
        _fail("wheel entry_points.txt is missing [console_scripts]")
    actual = dict(parser.items("console_scripts"))
    if actual != dict(expected):
        _fail(f"wheel console entry points are {actual!r}, expected {dict(expected)!r}")
    return actual


def _urlsafe_sha256(content: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _verify_record(
    files: Mapping[str, bytes],
    *,
    record_path: str,
) -> int:
    content = files[record_path]
    try:
        text = content.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        _fail(f"cannot parse wheel RECORD: {error}")
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            _fail(f"wheel RECORD row must have three columns: {row!r}")
        path, digest, size = row
        _validate_archive_name(path, directory=False)
        if path in records:
            _fail(f"wheel RECORD contains duplicate path {path!r}")
        records[path] = (digest, size)
    if set(records) != set(files):
        missing = sorted(set(files) - set(records))
        extra = sorted(set(records) - set(files))
        _fail(f"wheel RECORD inventory mismatch; missing={missing!r}, extra={extra!r}")
    for path, file_content in files.items():
        digest, size = records[path]
        if path == record_path:
            if digest or size:
                _fail("wheel RECORD must leave its own hash and size empty")
            continue
        expected_digest = _urlsafe_sha256(file_content)
        if digest != expected_digest:
            _fail(f"wheel RECORD hash mismatch for {path!r}")
        if size != str(len(file_content)):
            _fail(f"wheel RECORD size mismatch for {path!r}")
    return len(records)


def _verify_wheel(
    archive: LoadedArchive,
    *,
    config: ProjectConfig,
    repo_root: Path,
) -> dict[str, object]:
    files = archive.files
    runtime = _expected_runtime_files(repo_root)
    runtime_names = {
        name for name in files if name == "gworker" or name.startswith("gworker/")
    }
    if runtime_names != set(runtime):
        missing = sorted(set(runtime) - runtime_names)
        extra = sorted(runtime_names - set(runtime))
        _fail(f"wheel runtime inventory mismatch; missing={missing!r}, extra={extra!r}")
    for name, expected_content in runtime.items():
        if files[name] != expected_content:
            _fail(f"wheel runtime file differs from repository source: {name!r}")

    prefix = f"{config.dist_info}/"
    unexpected = sorted(
        name for name in files if name not in runtime and not name.startswith(prefix)
    )
    if unexpected:
        _fail(f"wheel contains unexpected top-level files: {unexpected!r}")
    foreign_dist_info = sorted(
        name for name in files if ".dist-info/" in name and not name.startswith(prefix)
    )
    if foreign_dist_info:
        _fail(f"wheel contains foreign .dist-info files: {foreign_dist_info!r}")

    metadata_path = f"{prefix}METADATA"
    wheel_path = f"{prefix}WHEEL"
    entry_points_path = f"{prefix}entry_points.txt"
    record_path = f"{prefix}RECORD"
    for required in (
        metadata_path,
        wheel_path,
        entry_points_path,
        record_path,
    ):
        if required not in files:
            _fail(f"wheel is missing required metadata file {required!r}")

    metadata = _verify_metadata(files[metadata_path], config)
    wheel_headers = _verify_wheel_headers(files[wheel_path])
    entry_points = _verify_entry_points(
        files[entry_points_path],
        config.console_scripts,
    )
    record_entries = _verify_record(files, record_path=record_path)
    return {
        "archive_directories": len(archive.directories),
        "archive_files": len(files),
        "console_scripts": entry_points,
        "metadata": metadata,
        "record_entries": record_entries,
        "runtime_files": len(runtime),
        "wheel": wheel_headers,
    }


def _expected_sdist_scopes(
    repo_root: Path,
) -> dict[str, dict[str, bytes]]:
    config: dict[str, bytes] = {}
    for relative in CONFIG_PATHS:
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            _fail(f"repository config file {relative!r} must be a regular file")
        config[relative] = path.read_bytes()
    return {
        "config": config,
        "src": _repository_files(repo_root, "src/gworker"),
        "tests": _repository_files(
            repo_root,
            "tests",
            suffixes=frozenset({".py"}),
        ),
        "scripts": _repository_files(
            repo_root,
            "scripts",
            suffixes=frozenset({".py"}),
        ),
        "docs": _repository_files(
            repo_root,
            "docs",
            suffixes=DOC_SUFFIXES,
        ),
    }


def _verify_sdist(
    archive: LoadedArchive,
    *,
    config: ProjectConfig,
    expected_metadata: bytes,
    repo_root: Path,
) -> dict[str, object]:
    root_prefix = f"{config.sdist_root}/"
    relative_files = {
        name.removeprefix(root_prefix): content
        for name, content in archive.files.items()
    }
    if len(relative_files) != len(archive.files):
        _fail("sdist contains a file outside its expected top-level directory")

    scopes = _expected_sdist_scopes(repo_root)
    required: dict[str, bytes] = {}
    for scoped_files in scopes.values():
        overlap = set(required) & set(scoped_files)
        if overlap:
            _fail(f"internal sdist scope overlap: {sorted(overlap)!r}")
        required.update(scoped_files)

    for relative, expected_content in required.items():
        actual_content = relative_files.get(relative)
        if actual_content is None:
            _fail(f"sdist is missing required repository file {relative!r}")
        if actual_content != expected_content:
            _fail(f"sdist repository file differs from source: {relative!r}")

    generated_allowed = set(GENERATED_SDIST_ROOT_FILES)
    generated_allowed.update(
        f"{config.egg_info}/{name}" for name in GENERATED_EGG_INFO_FILES
    )
    unexpected = sorted(set(relative_files) - set(required) - generated_allowed)
    if unexpected:
        _fail(f"sdist contains unexpected files: {unexpected!r}")
    metadata_paths = ("PKG-INFO", f"{config.egg_info}/PKG-INFO")
    for metadata_path in metadata_paths:
        metadata_content = relative_files.get(metadata_path)
        if metadata_content is None:
            _fail(f"sdist is missing generated {metadata_path}")
        if metadata_content != expected_metadata:
            _fail(
                f"sdist generated {metadata_path} differs from verified wheel METADATA"
            )

    scope_counts = {scope: len(scoped_files) for scope, scoped_files in scopes.items()}
    return {
        "archive_directories": len(archive.directories),
        "archive_files": len(archive.files),
        "required_repository_files": len(required),
        "scope_files": scope_counts,
        "metadata": {
            "matches_verified_wheel": True,
            "paths": list(metadata_paths),
            "sha256": _sha256(expected_metadata),
        },
        "byte_reproducibility": {
            "checked": False,
            "claimed": False,
            "status": "not-checked",
            "reason": (
                "Only the wheel is rebuilt and compared byte-for-byte; "
                "sdist byte reproducibility is not checked or claimed."
            ),
        },
    }


def _files_equal(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as first_stream, second.open("rb") as second_stream:
        while True:
            first_chunk = first_stream.read(1024 * 1024)
            second_chunk = second_stream.read(1024 * 1024)
            if first_chunk != second_chunk:
                return False
            if not first_chunk:
                return True


def verify_distribution(
    primary_dir: Path,
    rebuild_dir: Path,
    *,
    repo_root: Path = ROOT,
) -> dict[str, object]:
    """Verify artifact safety, content, metadata, and wheel reproducibility."""

    config = _load_project_config(repo_root)
    selected = _select_artifacts(primary_dir, rebuild_dir, config)

    primary_wheel_sha256 = _file_sha256(selected.primary_wheel)
    rebuilt_wheel_sha256 = _file_sha256(selected.rebuilt_wheel)
    if primary_wheel_sha256 != rebuilt_wheel_sha256 or not _files_equal(
        selected.primary_wheel,
        selected.rebuilt_wheel,
    ):
        _fail("rebuilt wheel is not byte-for-byte identical to the primary wheel")

    primary_wheel = _load_wheel(selected.primary_wheel)
    rebuilt_wheel = _load_wheel(selected.rebuilt_wheel)
    if primary_wheel.files != rebuilt_wheel.files:
        _fail("rebuilt wheel archive contents differ from the primary wheel")
    wheel_report = _verify_wheel(
        primary_wheel,
        config=config,
        repo_root=repo_root,
    )
    sdist = _load_sdist(
        selected.primary_sdist,
        expected_root=config.sdist_root,
    )
    sdist_report = _verify_sdist(
        sdist,
        config=config,
        expected_metadata=primary_wheel.files[f"{config.dist_info}/METADATA"],
        repo_root=repo_root,
    )

    wheel_size = selected.primary_wheel.stat().st_size
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "project": {
            "name": config.name,
            "version": config.version,
        },
        "artifacts": {
            "primary_wheel": {
                "bytes": wheel_size,
                "file": selected.primary_wheel.name,
                "sha256": primary_wheel_sha256,
            },
            "rebuilt_wheel": {
                "bytes": selected.rebuilt_wheel.stat().st_size,
                "file": selected.rebuilt_wheel.name,
                "sha256": rebuilt_wheel_sha256,
            },
            "sdist": {
                "bytes": selected.primary_sdist.stat().st_size,
                "file": selected.primary_sdist.name,
                "sha256": _file_sha256(selected.primary_sdist),
            },
        },
        "wheel_reproducibility": {
            "byte_for_byte": True,
            "checked": True,
            "sha256": primary_wheel_sha256,
        },
        "wheel_verification": wheel_report,
        "sdist_verification": sdist_report,
    }


def render_human(report: Mapping[str, object]) -> str:
    """Render a concise terminal summary without weakening the JSON report."""

    artifacts = _mapping(report["artifacts"], context="artifacts")
    primary = _mapping(
        artifacts["primary_wheel"],
        context="artifacts.primary_wheel",
    )
    sdist = _mapping(artifacts["sdist"], context="artifacts.sdist")
    wheel = _mapping(
        report["wheel_verification"],
        context="wheel_verification",
    )
    sdist_verification = _mapping(
        report["sdist_verification"],
        context="sdist_verification",
    )
    scopes = _mapping(
        sdist_verification["scope_files"],
        context="sdist_verification.scope_files",
    )
    lines = [
        "GWorker distribution verification: PASS",
        (
            f"wheel  {primary['file']}  sha256={primary['sha256']}  "
            "(rebuilt byte-for-byte)"
        ),
        (
            f"        runtime={wheel['runtime_files']} files  "
            f"RECORD={wheel['record_entries']} entries  tag=py3-none-any"
        ),
        (
            f"sdist  {sdist['file']}  "
            f"repository files={sdist_verification['required_repository_files']}"
        ),
        (
            "        scopes "
            + ", ".join(f"{name}={scopes[name]}" for name in sorted(scopes))
        ),
        "sdist byte reproducibility: NOT CHECKED OR CLAIMED",
    ]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only verification of one primary wheel+sdist and one "
            "independently rebuilt wheel."
        )
    )
    parser.add_argument(
        "primary_dist",
        type=Path,
        help="directory containing exactly one wheel and one sdist",
    )
    parser.add_argument(
        "rebuild_dist",
        type=Path,
        help="directory containing exactly one independently rebuilt wheel",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a canonical single-line JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = verify_distribution(
            arguments.primary_dist,
            arguments.rebuild_dist,
        )
    except VerificationError as error:
        if arguments.json:
            print(
                _canonical_json(
                    {
                        "error": str(error),
                        "ok": False,
                        "schema_version": SCHEMA_VERSION,
                    }
                ),
                end="",
            )
        else:
            print(f"distribution verification failed: {error}", file=sys.stderr)
        return 1
    output = _canonical_json(report) if arguments.json else render_human(report) + "\n"
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
