"""Transactional SQLite event journal for GWorker."""

from __future__ import annotations

import os
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from .codec import EventCodecError, decode_event, encode_event
from .domain import (
    DomainEvent,
    InvalidTransition,
    SessionState,
    apply_event,
    reduce_events,
)

SCHEMA_VERSION = 1
DATABASE_NAME = "events.sqlite3"

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (
        typeof(sequence) = 'integer' AND sequence > 0
    ),
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    event_json TEXT NOT NULL,
    PRIMARY KEY (session_id, sequence)
) WITHOUT ROWID;
"""


class JournalError(RuntimeError):
    """Base class for durable journal failures."""


class JournalSecurityError(JournalError):
    """Raised when the database path has unsafe file properties."""


class JournalConflict(JournalError):
    """Raised when an append races or violates aggregate ordering."""


class CorruptJournal(JournalError):
    """Raised when stored bytes cannot produce a valid event stream."""


@dataclass(frozen=True, slots=True)
class JournalVerification:
    """Summary returned after a complete journal replay."""

    session_count: int
    event_count: int
    sqlite_check: str


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int


def default_journal_path() -> Path:
    """Return the XDG-compatible path without creating any files."""

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    xdg_candidate = Path(xdg_data_home).expanduser() if xdg_data_home else None
    base = (
        xdg_candidate
        if xdg_candidate is not None and xdg_candidate.is_absolute()
        else Path.home() / ".local" / "share"
    )
    return base / "gworker" / DATABASE_NAME


def _canonical_uuid(value: UUID, field: str) -> str:
    if not isinstance(value, UUID):
        raise TypeError(f"{field} must be a UUID")
    if value.int == 0:
        raise ValueError(f"{field} must not be the nil UUID")
    return str(value)


class SQLiteEventStore:
    """Append-only session journal with fail-closed deterministic replay."""

    def __init__(self, path: str | Path) -> None:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise JournalSecurityError("journal path must be absolute")
        self.path = candidate
        self._parent_identity, self._identity = self._prepare_file()
        self._initialize_schema()

    def _validate_private_parent(self) -> _FileIdentity:
        current = Path(self.path.anchor)
        for component in self.path.parent.parts[1:]:
            current /= component
            try:
                information = os.lstat(current)
            except OSError as exc:
                raise JournalSecurityError(
                    "cannot inspect journal directory path"
                ) from exc
            if stat.S_ISLNK(information.st_mode):
                raise JournalSecurityError(
                    "journal directory path must not contain symlinks"
                )
            if not stat.S_ISDIR(information.st_mode):
                raise JournalSecurityError(
                    "journal directory path must contain only directories"
                )

        try:
            parent = os.lstat(self.path.parent)
        except OSError as exc:
            raise JournalSecurityError("cannot inspect journal directory") from exc
        if not stat.S_ISDIR(parent.st_mode):
            raise JournalSecurityError("journal parent must be a directory")
        if parent.st_uid != os.geteuid():
            raise JournalSecurityError("journal directory must be owned by the user")
        if stat.S_IMODE(parent.st_mode) & 0o077:
            raise JournalSecurityError(
                "journal directory must not be accessible by group or others"
            )
        return _FileIdentity(parent.st_dev, parent.st_ino)

    @staticmethod
    def _validate_directory_information(
        information: os.stat_result,
        expected: _FileIdentity | None = None,
    ) -> _FileIdentity:
        if not stat.S_ISDIR(information.st_mode):
            raise JournalSecurityError("journal parent must be a directory")
        if information.st_uid != os.geteuid():
            raise JournalSecurityError("journal directory must be owned by the user")
        if stat.S_IMODE(information.st_mode) & 0o077:
            raise JournalSecurityError(
                "journal directory must not be accessible by group or others"
            )
        identity = _FileIdentity(information.st_dev, information.st_ino)
        if expected is not None and identity != expected:
            raise JournalSecurityError("journal directory identity changed")
        return identity

    @staticmethod
    def _validate_file_information(
        information: os.stat_result,
        expected: _FileIdentity | None = None,
    ) -> _FileIdentity:
        if not stat.S_ISREG(information.st_mode):
            raise JournalSecurityError("journal path must be a regular file")
        if information.st_uid != os.geteuid():
            raise JournalSecurityError("journal file must be owned by the user")
        if stat.S_IMODE(information.st_mode) & 0o077:
            raise JournalSecurityError(
                "journal file must not be accessible by group or others"
            )
        if information.st_nlink != 1:
            raise JournalSecurityError("journal file must not have hard links")
        identity = _FileIdentity(
            device=information.st_dev,
            inode=information.st_ino,
        )
        if expected is not None and identity != expected:
            raise JournalSecurityError("journal path identity changed")
        return identity

    @staticmethod
    def _open_flags(*, create: bool) -> int:
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return flags

    @staticmethod
    def _directory_flags() -> int:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return flags

    def _open_parent(
        self,
        expected: _FileIdentity | None,
    ) -> tuple[int, _FileIdentity]:
        path_identity = self._validate_private_parent()
        try:
            descriptor = os.open(
                self.path.parent,
                self._directory_flags(),
            )
        except OSError as exc:
            raise JournalSecurityError("cannot safely open journal directory") from exc
        try:
            descriptor_identity = self._validate_directory_information(
                os.fstat(descriptor),
                expected,
            )
            if path_identity != descriptor_identity:
                raise JournalSecurityError("journal directory changed while opening")
        except Exception:
            os.close(descriptor)
            raise
        return descriptor, descriptor_identity

    def _prepare_file(self) -> tuple[_FileIdentity, _FileIdentity]:
        parent_was_missing = not self.path.parent.exists()
        try:
            self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        except OSError as exc:
            raise JournalSecurityError("cannot create journal directory") from exc
        if parent_was_missing:
            try:
                self.path.parent.chmod(0o700)
            except OSError as exc:
                raise JournalSecurityError("cannot secure journal directory") from exc

        directory_descriptor, parent_identity = self._open_parent(expected=None)
        try:
            existing = os.stat(
                self.path.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            os.close(directory_descriptor)
            raise JournalSecurityError("cannot inspect journal path") from exc
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                os.close(directory_descriptor)
                raise JournalSecurityError("journal path must not be a symlink")
            try:
                self._validate_file_information(existing)
            except Exception:
                os.close(directory_descriptor)
                raise
        try:
            descriptor = os.open(
                self.path.name,
                self._open_flags(create=True),
                0o600,
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            os.close(directory_descriptor)
            raise JournalSecurityError("cannot safely open journal file") from exc
        try:
            identity = self._validate_file_information(os.fstat(descriptor))
            path_identity = self._validate_file_information(
                os.stat(
                    self.path.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            )
            if path_identity != identity:
                raise JournalSecurityError("journal path changed while opening")
            return parent_identity, identity
        finally:
            os.close(descriptor)
            os.close(directory_descriptor)

    def _open_expected_file(self, directory_descriptor: int) -> int:
        try:
            descriptor = os.open(
                self.path.name,
                self._open_flags(create=False),
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise JournalSecurityError("cannot safely reopen journal file") from exc
        try:
            descriptor_identity = self._validate_file_information(
                os.fstat(descriptor),
                self._identity,
            )
            path_identity = self._validate_file_information(
                os.stat(
                    self.path.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                ),
                self._identity,
            )
            if path_identity != descriptor_identity:
                raise JournalSecurityError("journal path changed while reopening")
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    def _confirm_expected_file(self, directory_descriptor: int) -> None:
        descriptor = self._open_expected_file(directory_descriptor)
        os.close(descriptor)

    def _connect(self) -> tuple[sqlite3.Connection, int]:
        directory_descriptor, _ = self._open_parent(self._parent_identity)
        try:
            anchor = self._open_expected_file(directory_descriptor)
        except Exception:
            os.close(directory_descriptor)
            raise
        connection: sqlite3.Connection | None = None
        try:
            proc_directory = Path("/proc/self/fd") / str(directory_descriptor)
            if not proc_directory.is_dir():
                raise JournalSecurityError(
                    "secure journal connections require Linux /proc/self/fd"
                )
            database_path = proc_directory / self.path.name
            connection = sqlite3.connect(
                database_path,
                isolation_level=None,
                timeout=5.0,
            )
            self._confirm_expected_file(directory_descriptor)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA trusted_schema = OFF")
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            connection.execute("PRAGMA synchronous = FULL")
            self._confirm_expected_file(directory_descriptor)
        except sqlite3.Error as exc:
            if connection is not None:
                self._close_connection(connection, directory_descriptor)
            else:
                os.close(directory_descriptor)
            raise JournalError("cannot configure SQLite journal") from exc
        except Exception:
            if connection is not None:
                self._close_connection(connection, directory_descriptor)
            else:
                os.close(directory_descriptor)
            raise
        finally:
            os.close(anchor)
        if mode != "wal":
            connection.close()
            os.close(directory_descriptor)
            raise JournalError(f"SQLite refused WAL mode: {mode}")
        return connection, directory_descriptor

    @staticmethod
    def _close_connection(
        connection: sqlite3.Connection,
        directory_descriptor: int,
    ) -> None:
        try:
            connection.close()
        finally:
            os.close(directory_descriptor)

    @staticmethod
    def _require_schema_version(connection: sqlite3.Connection) -> None:
        try:
            row = connection.execute(
                """
                SELECT value
                FROM journal_metadata
                WHERE key = 'schema_version'
                """
            ).fetchone()
        except sqlite3.Error as exc:
            raise CorruptJournal("cannot read journal schema version") from exc
        if row is None or row["value"] != str(SCHEMA_VERSION):
            stored = "missing" if row is None else row["value"]
            raise CorruptJournal(f"unsupported journal schema version: {stored}")

    def _initialize_schema(self) -> None:
        connection, directory_descriptor = self._connect()
        try:
            connection.executescript(_CREATE_SCHEMA)
            connection.execute(
                """
                INSERT OR IGNORE INTO journal_metadata (key, value)
                VALUES ('schema_version', ?)
                """,
                (str(SCHEMA_VERSION),),
            )
            self._require_schema_version(connection)
        except sqlite3.Error as exc:
            raise JournalError("cannot initialize SQLite journal") from exc
        finally:
            self._close_connection(connection, directory_descriptor)

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> DomainEvent:
        try:
            event = decode_event(row["event_json"])
        except EventCodecError as exc:
            raise CorruptJournal(
                f"cannot decode stored event at {row['session_id']}:{row['sequence']}"
            ) from exc
        expected = (
            str(event.session_id),
            event.sequence,
            str(event.event_id),
            event.KIND,
        )
        stored = (
            row["session_id"],
            row["sequence"],
            row["event_id"],
            row["event_type"],
        )
        if stored != expected:
            location = f"{row['session_id']}:{row['sequence']}"
            raise CorruptJournal(f"indexed event fields disagree at {location}")
        return event

    def _load_with_connection(
        self,
        connection: sqlite3.Connection,
        session_id: UUID,
    ) -> list[DomainEvent]:
        identifier = _canonical_uuid(session_id, "session_id")
        try:
            rows = connection.execute(
                """
                SELECT session_id, sequence, event_id, event_type, event_json
                FROM events
                WHERE session_id = ?
                ORDER BY sequence
                """,
                (identifier,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise CorruptJournal("cannot read session events") from exc
        return [self._decode_row(row) for row in rows]

    def append(self, event: DomainEvent) -> SessionState:
        """Validate and atomically append one next event."""

        encoded = encode_event(event)
        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._require_schema_version(connection)
            existing = self._load_with_connection(connection, event.session_id)
            try:
                state = reduce_events(existing) if existing else None
            except InvalidTransition as exc:
                raise CorruptJournal(
                    f"invalid existing session stream: {event.session_id}"
                ) from exc
            try:
                projected = apply_event(state, event)
            except InvalidTransition as exc:
                raise JournalConflict(str(exc)) from exc
            connection.execute(
                """
                INSERT INTO events (
                    session_id,
                    sequence,
                    event_id,
                    event_type,
                    event_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(event.session_id),
                    event.sequence,
                    str(event.event_id),
                    event.KIND,
                    encoded,
                ),
            )
            connection.execute("COMMIT")
            return projected
        except sqlite3.IntegrityError as exc:
            self._rollback(connection)
            raise JournalConflict("event sequence or event_id already exists") from exc
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        if connection.in_transaction:
            connection.execute("ROLLBACK")

    def load(self, session_id: UUID) -> list[DomainEvent]:
        """Load and validate one complete session stream."""

        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN")
            self._require_schema_version(connection)
            events = self._load_with_connection(connection, session_id)
            connection.execute("COMMIT")
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)
        if events:
            try:
                reduce_events(events)
            except InvalidTransition as exc:
                raise CorruptJournal(f"invalid session stream: {session_id}") from exc
        return events

    def replay(self, session_id: UUID) -> SessionState:
        """Project one stored session or fail if it does not exist."""

        events = self.load(session_id)
        if not events:
            raise KeyError(f"session not found: {session_id}")
        return reduce_events(events)

    def session_ids(self) -> list[UUID]:
        """List sessions in stable identifier order."""

        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN")
            self._require_schema_version(connection)
            rows = connection.execute(
                "SELECT DISTINCT session_id FROM events ORDER BY session_id"
            ).fetchall()
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            self._rollback(connection)
            raise CorruptJournal("cannot list journal sessions") from exc
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)

        identifiers: list[UUID] = []
        for row in rows:
            raw = row["session_id"]
            try:
                identifier = UUID(raw)
            except (TypeError, ValueError) as exc:
                raise CorruptJournal("stored session_id is not a UUID") from exc
            if str(identifier) != raw:
                raise CorruptJournal("stored session_id is not canonical")
            identifiers.append(identifier)
        return identifiers

    def verify(self) -> JournalVerification:
        """Run SQLite integrity checks and replay every stored session."""

        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN")
            self._require_schema_version(connection)
            check_rows = connection.execute("PRAGMA quick_check").fetchall()
            check_values = [row[0] for row in check_rows]
            if check_values != ["ok"]:
                raise CorruptJournal(
                    f"SQLite quick_check failed: {','.join(check_values)}"
                )
            rows = connection.execute(
                """
                SELECT session_id, sequence, event_id, event_type, event_json
                FROM events
                ORDER BY session_id, sequence
                """
            ).fetchall()
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            self._rollback(connection)
            raise CorruptJournal("cannot verify SQLite journal") from exc
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)

        session_events: dict[UUID, list[DomainEvent]] = {}
        for row in rows:
            event = self._decode_row(row)
            session_events.setdefault(event.session_id, []).append(event)
        for session_id, events in session_events.items():
            try:
                reduce_events(events)
            except InvalidTransition as exc:
                raise CorruptJournal(f"invalid session stream: {session_id}") from exc
        return JournalVerification(
            session_count=len(session_events),
            event_count=len(rows),
            sqlite_check="ok",
        )
