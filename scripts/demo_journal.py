#!/usr/bin/env python3
"""Deterministic SQLite journal, replay, and safe tamper-copy demo."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict
from uuid import UUID

from gworker.codec import event_document
from gworker.domain import (
    BreakCompleted,
    BreakStarted,
    DomainEvent,
    FocusCompleted,
    FocusStarted,
    InterruptionKind,
    InterruptionRecorded,
    SessionPlanned,
    SessionState,
)
from gworker.policy import POLICY_ID
from gworker.storage import CorruptJournal, JournalError, SQLiteEventStore

SCHEMA_VERSION = "gworker-journal-demo-v1"
VISUAL_DEMO_COMPONENTS = (".gworker", "visual-demo")
LIVE_DIRECTORY_NAME = "live"
TAMPER_DIRECTORY_NAME = "tamper-copy"
DATABASE_NAME = "events.sqlite3"
SESSION_ID = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e1000")
BASE_TIME = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)
SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


class DemoPathError(ValueError):
    """Raised before a demo path can escape or traverse a symlink."""


class DemoInvariantError(RuntimeError):
    """Raised when a real journal operation contradicts the demo contract."""


class _EventFields(TypedDict):
    event_id: UUID
    occurred_at: datetime
    sequence: int
    session_id: UUID


def _close_resources(
    *,
    connections: Sequence[sqlite3.Connection | None] = (),
    descriptors: Sequence[int | None] = (),
) -> None:
    active_exception = sys.exc_info()[0] is not None
    first_failure: Exception | None = None
    seen_connections: set[int] = set()
    for connection in connections:
        if connection is None or id(connection) in seen_connections:
            continue
        seen_connections.add(id(connection))
        try:
            connection.close()
        except Exception as exc:
            if first_failure is None:
                first_failure = exc

    seen_descriptors: set[int] = set()
    for descriptor in descriptors:
        if descriptor is None or descriptor in seen_descriptors:
            continue
        seen_descriptors.add(descriptor)
        try:
            os.close(descriptor)
        except OSError as exc:
            if first_failure is None:
                first_failure = exc

    if first_failure is not None and not active_exception:
        raise first_failure


def _directory_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _database_flags() -> int:
    flags = os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _validate_directory_stat(
    information: os.stat_result,
    *,
    field: str,
    private: bool,
) -> None:
    if not stat.S_ISDIR(information.st_mode):
        raise DemoPathError(f"{field} must be a directory")
    if information.st_uid != os.geteuid():
        raise DemoPathError(f"{field} must be owned by the current user")
    if private and stat.S_IMODE(information.st_mode) & 0o077:
        raise DemoPathError(f"{field} must not be accessible by group or others")


def _canonical_repo_root(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DemoPathError("repo root must be an existing directory") from exc
    try:
        information = os.lstat(root)
    except OSError as exc:
        raise DemoPathError("cannot inspect repo root") from exc
    _validate_directory_stat(information, field="repo root", private=False)

    pyproject = root / "pyproject.toml"
    package = root / "src" / "gworker"
    try:
        pyproject_info = os.lstat(pyproject)
        package_info = os.lstat(package)
    except OSError as exc:
        raise DemoPathError(
            "repo root does not contain the GWorker source tree"
        ) from exc
    if stat.S_ISLNK(pyproject_info.st_mode) or not stat.S_ISREG(pyproject_info.st_mode):
        raise DemoPathError("repo pyproject.toml must be a regular file")
    if stat.S_ISLNK(package_info.st_mode) or not stat.S_ISDIR(package_info.st_mode):
        raise DemoPathError("repo src/gworker must be a directory")
    if not _validate_existing_chain(root, Path("src", "gworker")):
        raise DemoPathError("repo src/gworker must be an existing canonical path")
    return root


def _validate_existing_chain(root: Path, relative: Path) -> bool:
    current = root
    exists = True
    for component in relative.parts:
        current /= component
        try:
            information = os.lstat(current)
        except FileNotFoundError:
            exists = False
            break
        except OSError as exc:
            raise DemoPathError("cannot inspect workspace path") from exc
        if stat.S_ISLNK(information.st_mode):
            raise DemoPathError("workspace path must not contain symlinks")
        if not stat.S_ISDIR(information.st_mode):
            raise DemoPathError("workspace path must contain only directories")
    return exists


def _workspace_location(
    root: Path,
    value: str | Path,
) -> tuple[Path, tuple[str, ...], bool]:
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise DemoPathError("workspace path must not contain '..'")
    candidate = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(candidate))
    base = root.joinpath(*VISUAL_DEMO_COMPONENTS)
    try:
        below_base = lexical.relative_to(base)
    except ValueError as exc:
        raise DemoPathError(
            "workspace must be beneath <repo-root>/.gworker/visual-demo"
        ) from exc
    if not below_base.parts:
        raise DemoPathError("workspace must be a strict descendant of visual-demo")
    if not all(SAFE_COMPONENT.fullmatch(component) for component in below_base.parts):
        raise DemoPathError("workspace components must use safe lowercase names")

    relative = Path(*VISUAL_DEMO_COMPONENTS, *below_base.parts)
    exists = _validate_existing_chain(root, relative)
    return lexical, relative.parts, exists


def _open_relative_directory(
    root: Path,
    components: Sequence[str],
    *,
    private_from: int,
) -> int:
    try:
        root_descriptor = os.open(root, _directory_flags())
    except OSError as exc:
        raise DemoPathError("cannot safely open repo root") from exc
    descriptors = [root_descriptor]
    try:
        for index, component in enumerate(components):
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptors[-1],
                )
            except OSError as exc:
                raise DemoPathError("cannot safely open workspace directory") from exc
            descriptors.append(child)
            _validate_directory_stat(
                os.fstat(child),
                field="workspace directory",
                private=index >= private_from,
            )
    except Exception:
        _close_resources(descriptors=tuple(reversed(descriptors)))
        raise
    result = descriptors[-1]
    try:
        _close_resources(descriptors=tuple(reversed(descriptors[:-1])))
    except Exception:
        _close_resources(descriptors=(result,))
        raise
    return result


def _ensure_private_chain(root: Path, components: Sequence[str]) -> None:
    try:
        root_descriptor = os.open(root, _directory_flags())
    except OSError as exc:
        raise DemoPathError("cannot safely open repo root") from exc
    descriptors = [root_descriptor]
    try:
        for component in components:
            try:
                os.mkdir(component, mode=0o700, dir_fd=descriptors[-1])
            except FileExistsError:
                pass
            except OSError as exc:
                raise DemoPathError("cannot create demo workspace") from exc
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptors[-1],
                )
            except OSError as exc:
                raise DemoPathError("cannot safely open demo workspace") from exc
            descriptors.append(child)
            _validate_directory_stat(
                os.fstat(child),
                field="demo workspace",
                private=True,
            )
    finally:
        _close_resources(descriptors=tuple(reversed(descriptors)))


def _ensure_private_child_directory(parent: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent)
        os.fsync(parent)
    except FileExistsError:
        pass
    except OSError as exc:
        raise DemoPathError("cannot create demo database directory") from exc
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent)
    except OSError as exc:
        raise DemoPathError("cannot safely open demo database directory") from exc
    try:
        _validate_directory_stat(
            os.fstat(descriptor),
            field="demo database directory",
            private=True,
        )
    except Exception:
        _close_resources(descriptors=(descriptor,))
        raise
    return descriptor


def _verify_directory_identity(path: Path, descriptor: int, *, field: str) -> None:
    try:
        path_information = os.lstat(path)
        descriptor_information = os.fstat(descriptor)
    except OSError as exc:
        raise DemoPathError(f"cannot revalidate {field}") from exc
    if stat.S_ISLNK(path_information.st_mode):
        raise DemoPathError(f"{field} must not be a symlink")
    _validate_directory_stat(path_information, field=field, private=True)
    _validate_directory_stat(descriptor_information, field=field, private=True)
    if (
        path_information.st_dev,
        path_information.st_ino,
    ) != (
        descriptor_information.st_dev,
        descriptor_information.st_ino,
    ):
        raise DemoPathError(f"{field} identity changed")


def _open_database_at(directory: int, name: str) -> int:
    try:
        descriptor = os.open(name, _database_flags(), dir_fd=directory)
    except OSError as exc:
        raise DemoPathError("cannot safely open demo database") from exc
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            raise DemoPathError("demo database must be a regular file")
        if information.st_uid != os.geteuid():
            raise DemoPathError("demo database must be user-owned")
        if stat.S_IMODE(information.st_mode) & 0o077:
            raise DemoPathError("demo database must be private")
        if information.st_nlink != 1:
            raise DemoPathError("demo database must not have hard links")
    except Exception:
        _close_resources(descriptors=(descriptor,))
        raise
    return descriptor


def _descriptor_path(descriptor: int) -> str:
    proc_path = Path("/proc/self/fd") / str(descriptor)
    if not proc_path.exists():
        raise DemoPathError("pinned SQLite operations require Linux /proc/self/fd")
    return str(proc_path)


def _assert_reset_tree_safe(descriptor: int) -> None:
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise DemoPathError("cannot inspect reset workspace") from exc
    for name in names:
        try:
            information = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise DemoPathError("cannot inspect reset workspace entry") from exc
        if stat.S_ISLNK(information.st_mode):
            raise DemoPathError("reset workspace must not contain symlinks")
        if information.st_uid != os.geteuid():
            raise DemoPathError("reset workspace entries must be user-owned")
        if stat.S_ISDIR(information.st_mode):
            try:
                child = os.open(name, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise DemoPathError("cannot safely inspect reset directory") from exc
            try:
                _validate_directory_stat(
                    os.fstat(child),
                    field="reset directory",
                    private=True,
                )
                _assert_reset_tree_safe(child)
            finally:
                _close_resources(descriptors=(child,))
        elif not stat.S_ISREG(information.st_mode):
            raise DemoPathError(
                "reset workspace may contain only directories and regular files"
            )
        elif information.st_nlink != 1:
            raise DemoPathError("reset workspace files must not have hard links")


def _reset_workspace(root: Path, components: Sequence[str]) -> None:
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise DemoPathError("this platform cannot reset workspaces symlink-safely")
    parent_components = components[:-1]
    leaf = components[-1]
    parent = _open_relative_directory(
        root,
        parent_components,
        private_from=0,
    )
    try:
        try:
            workspace = os.open(leaf, _directory_flags(), dir_fd=parent)
        except OSError as exc:
            raise DemoPathError("cannot safely open reset workspace") from exc
        try:
            _validate_directory_stat(
                os.fstat(workspace),
                field="reset workspace",
                private=True,
            )
            _assert_reset_tree_safe(workspace)
        finally:
            _close_resources(descriptors=(workspace,))
        try:
            shutil.rmtree(leaf, dir_fd=parent)
            os.fsync(parent)
        except OSError as exc:
            raise DemoPathError("cannot reset demo workspace") from exc
    finally:
        _close_resources(descriptors=(parent,))


def _directory_is_empty(root: Path, components: Sequence[str]) -> bool:
    descriptor = _open_relative_directory(root, components, private_from=0)
    try:
        return not os.listdir(descriptor)
    except OSError as exc:
        raise DemoPathError("cannot inspect demo workspace") from exc
    finally:
        _close_resources(descriptors=(descriptor,))


def prepare_workspace(
    repo_root: str | Path,
    workspace: str | Path,
    *,
    reset: bool,
) -> tuple[Path, Path]:
    """Validate and optionally clear one tightly scoped demo workspace."""

    root = _canonical_repo_root(repo_root)
    location, components, exists = _workspace_location(root, workspace)

    if reset and exists:
        _reset_workspace(root, components)
        exists = False
    if exists and not _directory_is_empty(root, components):
        raise DemoPathError("workspace is not empty; pass --reset to replace it")

    _ensure_private_chain(root, components)
    if not _directory_is_empty(root, components):
        raise DemoPathError("workspace must be empty before the demo starts")
    return root, location


def _events() -> tuple[DomainEvent, ...]:
    metadata = (
        ("018f4f69-e7a2-7f84-8c2d-9f531c4e0001", 1, 0),
        ("018f4f69-e7a2-7f84-8c2d-9f531c4e0002", 2, 5),
        ("018f4f69-e7a2-7f84-8c2d-9f531c4e0003", 3, 750),
        ("018f4f69-e7a2-7f84-8c2d-9f531c4e0004", 4, 2_405),
        ("018f4f69-e7a2-7f84-8c2d-9f531c4e0005", 5, 2_410),
        ("018f4f69-e7a2-7f84-8c2d-9f531c4e0006", 6, 2_890),
    )

    def fields(index: int) -> _EventFields:
        event_id, sequence, seconds = metadata[index]
        return {
            "event_id": UUID(event_id),
            "occurred_at": BASE_TIME + timedelta(seconds=seconds),
            "sequence": sequence,
            "session_id": SESSION_ID,
        }

    return (
        SessionPlanned(
            **fields(0),
            objective="Synthetic architecture review",
            policy_id=POLICY_ID,
            target_break_seconds=480,
            target_focus_seconds=2_400,
        ),
        FocusStarted(**fields(1)),
        InterruptionRecorded(
            **fields(2),
            elapsed_seconds=18,
            kind=InterruptionKind.NOTIFICATION,
        ),
        FocusCompleted(**fields(3), elapsed_seconds=2_382),
        BreakStarted(**fields(4)),
        BreakCompleted(**fields(5), elapsed_seconds=480),
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _state_document(state: SessionState) -> dict[str, object]:
    return {
        "actual_break_seconds": state.actual_break_seconds,
        "actual_focus_seconds": state.actual_focus_seconds,
        "focus_completion_ratio": state.focus_completion_ratio,
        "interruption_count": state.interruption_count,
        "interruption_seconds": state.interruption_seconds,
        "is_terminal": state.is_terminal,
        "last_occurred_at": _timestamp(state.last_occurred_at),
        "phase": state.phase.value,
        "revision": state.revision,
        "session_id": str(state.session_id),
    }


def _permissions(path: Path) -> dict[str, object]:
    information = os.lstat(path)
    return {
        "hard_link_count": information.st_nlink,
        "mode": f"{stat.S_IMODE(information.st_mode):04o}",
        "owned_by_current_user": information.st_uid == os.geteuid(),
        "private": stat.S_IMODE(information.st_mode) & 0o077 == 0,
    }


def _sqlite_quick_check(descriptor: int) -> str:
    connection = sqlite3.connect(_descriptor_path(descriptor))
    try:
        rows = connection.execute("PRAGMA quick_check").fetchall()
    finally:
        _close_resources(connections=(connection,))
    return ",".join(str(row[0]) for row in rows)


def _make_tamper_copy(
    destination: Path,
    *,
    source_directory: int,
    destination_directory: int,
) -> SQLiteEventStore:
    source_descriptor = _open_database_at(source_directory, DATABASE_NAME)
    destination_descriptor: int | None = None
    source_connection: sqlite3.Connection | None = None
    destination_connection: sqlite3.Connection | None = None
    try:
        tamper_store = SQLiteEventStore(destination)
        destination_descriptor = _open_database_at(
            destination_directory,
            DATABASE_NAME,
        )
        source_connection = sqlite3.connect(_descriptor_path(source_descriptor))
        destination_connection = sqlite3.connect(
            _descriptor_path(destination_descriptor)
        )
        source_connection.backup(destination_connection)
        destination_connection.execute(
            """
            UPDATE events
            SET event_json = ?
            WHERE session_id = ? AND sequence = 3
            """,
            ('{"schema_version":1}', str(SESSION_ID)),
        )
        destination_connection.commit()
        return tamper_store
    finally:
        _close_resources(
            connections=(destination_connection, source_connection),
            descriptors=(destination_descriptor, source_descriptor),
        )


def build_demo(
    repo_root: str | Path,
    workspace: str | Path,
    *,
    reset: bool,
) -> dict[str, object]:
    """Write, reopen, replay, verify, and tamper only inside the workspace."""

    root, location = prepare_workspace(repo_root, workspace, reset=reset)
    workspace_components = location.relative_to(root).parts
    workspace_descriptor = _open_relative_directory(
        root,
        workspace_components,
        private_from=0,
    )
    live_descriptor: int | None = None
    tamper_descriptor: int | None = None
    try:
        _verify_directory_identity(
            location,
            workspace_descriptor,
            field="demo workspace",
        )
        live_descriptor = _ensure_private_child_directory(
            workspace_descriptor,
            LIVE_DIRECTORY_NAME,
        )
        tamper_descriptor = _ensure_private_child_directory(
            workspace_descriptor,
            TAMPER_DIRECTORY_NAME,
        )
        live_directory = location / LIVE_DIRECTORY_NAME
        tamper_directory = location / TAMPER_DIRECTORY_NAME
        live_path = live_directory / DATABASE_NAME
        tamper_path = tamper_directory / DATABASE_NAME
        _verify_directory_identity(
            live_directory,
            live_descriptor,
            field="live database directory",
        )
        _verify_directory_identity(
            tamper_directory,
            tamper_descriptor,
            field="tamper database directory",
        )

        store = SQLiteEventStore(live_path)
        _verify_directory_identity(
            live_directory,
            live_descriptor,
            field="live database directory",
        )
        events = _events()
        for event in events:
            store.append(event)

        reopened = SQLiteEventStore(live_path)
        loaded = reopened.load(SESSION_ID)
        state = reopened.replay(SESSION_ID)
        verification = reopened.verify()
        if loaded != list(events):
            raise DemoInvariantError("reopened journal did not reproduce exact events")

        tamper_store = _make_tamper_copy(
            tamper_path,
            source_directory=live_descriptor,
            destination_directory=tamper_descriptor,
        )
        _verify_directory_identity(
            tamper_directory,
            tamper_descriptor,
            field="tamper database directory",
        )
        tamper_database = _open_database_at(tamper_descriptor, DATABASE_NAME)
        try:
            tamper_quick_check = _sqlite_quick_check(tamper_database)
        finally:
            _close_resources(descriptors=(tamper_database,))
        try:
            tamper_store.verify()
        except CorruptJournal as exc:
            tamper_detection = {
                "detected": True,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        else:
            raise DemoInvariantError("synthetic logical tamper was not detected")

        _verify_directory_identity(
            location,
            workspace_descriptor,
            field="demo workspace",
        )
    finally:
        _close_resources(
            descriptors=(
                tamper_descriptor,
                live_descriptor,
                workspace_descriptor,
            )
        )

    workspace_relative = location.relative_to(root).as_posix()
    return {
        "events": [event_document(event) for event in loaded],
        "fixture_kind": "deterministic-synthetic-journal-demo",
        "fixture_notice": (
            "Synthetic local journal data only; the tamper test modifies "
            "a separate copy."
        ),
        "journal": {
            "database": f"{workspace_relative}/{LIVE_DIRECTORY_NAME}/{DATABASE_NAME}",
            "database_permissions": _permissions(live_path),
            "directory_permissions": _permissions(live_path.parent),
            "event_count": verification.event_count,
            "reopen_matches_written_events": loaded == list(events),
            "session_count": verification.session_count,
            "sqlite_quick_check": verification.sqlite_check,
            "storage_api": "SQLiteEventStore",
        },
        "replay": _state_document(state),
        "schema_version": SCHEMA_VERSION,
        "tamper_copy": {
            "database": (
                f"{workspace_relative}/{TAMPER_DIRECTORY_NAME}/{DATABASE_NAME}"
            ),
            "database_permissions": _permissions(tamper_path),
            "detection": tamper_detection,
            "mutation": (
                "sequence 3 event_json replaced with incomplete synthetic JSON"
            ),
            "separate_from_live_database": not os.path.samestat(
                os.stat(live_path), os.stat(tamper_path)
            ),
            "sqlite_quick_check_before_domain_replay": tamper_quick_check,
        },
        "verification": {
            "fixed_timestamps": True,
            "fixed_uuids": True,
            "live_database_unchanged_by_tamper_demo": (
                reopened.load(SESSION_ID) == loaded
                and reopened.verify() == verification
            ),
            "real_reopen_replay": True,
        },
        "workspace": workspace_relative,
    }


def canonical_json(document: dict[str, object], *, pretty: bool) -> str:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
    )


def render_human(document: dict[str, object]) -> str:
    journal = document["journal"]
    replay = document["replay"]
    tamper = document["tamper_copy"]
    assert isinstance(journal, dict)
    assert isinstance(replay, dict)
    assert isinstance(tamper, dict)
    detection = tamper["detection"]
    assert isinstance(detection, dict)
    events = document["events"]
    assert isinstance(events, list)

    lines = [
        "GWorker journal demo | real SQLite store, synthetic events",
        f"Workspace: {document['workspace']}",
        f"Database:  {journal['database']}",
        (
            f"Security:  file {journal['database_permissions']['mode']}, "
            f"directory {journal['directory_permissions']['mode']}, "
            f"owner=current-user"
        ),
        "",
        "Append-only event stream",
        " seq  occurred_at                  event_type",
    ]
    for event in events:
        assert isinstance(event, dict)
        lines.append(
            f" {int(event['sequence']):>3}  "
            f"{event['occurred_at']!s:<27}  "
            f"{event['event_type']}"
        )
    lines.extend(
        [
            "",
            (
                f"Reopen + replay: phase={replay['phase']}, "
                f"revision={replay['revision']}, terminal={replay['is_terminal']}"
            ),
            (
                f"Durations: focus={replay['actual_focus_seconds']}s, "
                f"break={replay['actual_break_seconds']}s, "
                f"interruptions={replay['interruption_count']}"
            ),
            (
                f"Integrity: PRAGMA quick_check={journal['sqlite_quick_check']}, "
                f"sessions={journal['session_count']}, "
                f"events={journal['event_count']}"
            ),
            "",
            "Safe synthetic tamper copy",
            f"Copy:      {tamper['database']}",
            (
                "SQLite:    quick_check="
                f"{tamper['sqlite_quick_check_before_domain_replay']}"
            ),
            (
                f"Replay:    detected={str(detection['detected']).lower()} "
                f"({detection['error_type']}: {detection['error']})"
            ),
            "Live journal remains verified and unchanged.",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise GWorker's real SQLite journal in a scoped workspace."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="canonical GWorker repository root (default: current directory)",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="strict descendant of <repo-root>/.gworker/visual-demo",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="symlink-safely clear only the validated demo workspace",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit canonical, pretty JSON instead of terminal text",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        document = build_demo(
            arguments.repo_root,
            arguments.workspace,
            reset=arguments.reset,
        )
    except (
        DemoInvariantError,
        DemoPathError,
        JournalError,
        OSError,
        sqlite3.Error,
    ) as exc:
        parser.exit(2, f"demo-journal: error: {exc}\n")
    output = (
        canonical_json(document, pretty=True)
        if arguments.json
        else render_human(document)
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
