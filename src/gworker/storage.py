"""Transactional SQLite event journal for GWorker."""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

from .codec import EventCodecError, decode_event, encode_event
from .domain import (
    DomainEvent,
    InvalidTransition,
    SessionPhase,
    SessionPlanned,
    SessionState,
    apply_event,
    reduce_events,
)
from .policy import (
    MAX_AVAILABLE_SECONDS,
    MAX_DECISION_SEQUENCE,
    MAX_RNG_SEED,
    DurationFit,
    EnergyLevel,
    FocusContext,
    HierarchicalSoftmaxUCB,
    PolicyInputError,
    Recommendation,
    ReviewedDecision,
    TaskKind,
)

SCHEMA_VERSION = 3
DATABASE_NAME = "events.sqlite3"

_CREATE_SCHEMA_V1 = (
    """
    CREATE TABLE journal_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE events (
        session_id TEXT NOT NULL,
        sequence INTEGER NOT NULL CHECK (
            typeof(sequence) = 'integer' AND sequence > 0
        ),
        event_id TEXT NOT NULL UNIQUE,
        event_type TEXT NOT NULL,
        event_json TEXT NOT NULL,
        PRIMARY KEY (session_id, sequence)
    ) WITHOUT ROWID
    """,
)

_CREATE_SCHEMA_V2 = (
    """
    CREATE TABLE policy_decisions (
        decision_id TEXT PRIMARY KEY,
        decision_sequence INTEGER NOT NULL UNIQUE CHECK (
            typeof(decision_sequence) = 'integer'
            AND decision_sequence BETWEEN 1 AND 9223372036854775807
        ),
        policy_id TEXT NOT NULL,
        rng_seed INTEGER NOT NULL CHECK (
            typeof(rng_seed) = 'integer'
            AND rng_seed BETWEEN 0 AND 9223372036854775807
        ),
        task_kind TEXT NOT NULL,
        energy TEXT NOT NULL,
        available_seconds INTEGER NOT NULL CHECK (
            typeof(available_seconds) = 'integer'
        ),
        previous_focus_seconds INTEGER CHECK (
            previous_focus_seconds IS NULL
            OR typeof(previous_focus_seconds) = 'integer'
        ),
        template_id TEXT NOT NULL,
        propensity_hex TEXT NOT NULL,
        history_count INTEGER NOT NULL CHECK (
            typeof(history_count) = 'integer' AND history_count >= 0
        ),
        history_sha256 TEXT NOT NULL
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE policy_reviews (
        decision_id TEXT PRIMARY KEY
            REFERENCES policy_decisions(decision_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT,
        fit TEXT NOT NULL,
        objective_completed INTEGER NOT NULL CHECK (
            typeof(objective_completed) = 'integer'
            AND objective_completed IN (0, 1)
        )
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE policy_decision_history (
        decision_id TEXT NOT NULL
            REFERENCES policy_decisions(decision_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT,
        position INTEGER NOT NULL CHECK (
            typeof(position) = 'integer' AND position >= 0
        ),
        reviewed_decision_id TEXT NOT NULL
            REFERENCES policy_reviews(decision_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT,
        PRIMARY KEY (decision_id, position),
        UNIQUE (decision_id, reviewed_decision_id)
    ) WITHOUT ROWID
    """,
    """
    CREATE INDEX policy_decisions_policy_sequence
    ON policy_decisions(policy_id, decision_sequence)
    """,
    """
    CREATE INDEX policy_history_reviewed_decision
    ON policy_decision_history(reviewed_decision_id)
    """,
)

_CREATE_SCHEMA_V3 = (
    """
    CREATE TABLE focus_session_links (
        decision_id TEXT PRIMARY KEY
            REFERENCES policy_decisions(decision_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT,
        planned_event_id TEXT NOT NULL UNIQUE
            REFERENCES events(event_id)
            ON UPDATE RESTRICT
            ON DELETE RESTRICT
    ) WITHOUT ROWID
    """,
)

_EXPECTED_TABLE_COLUMNS = {
    "journal_metadata": ("key", "value"),
    "events": (
        "session_id",
        "sequence",
        "event_id",
        "event_type",
        "event_json",
    ),
    "policy_decisions": (
        "decision_id",
        "decision_sequence",
        "policy_id",
        "rng_seed",
        "task_kind",
        "energy",
        "available_seconds",
        "previous_focus_seconds",
        "template_id",
        "propensity_hex",
        "history_count",
        "history_sha256",
    ),
    "policy_reviews": ("decision_id", "fit", "objective_completed"),
    "policy_decision_history": (
        "decision_id",
        "position",
        "reviewed_decision_id",
    ),
    "focus_session_links": ("decision_id", "planned_event_id"),
}

_EXPECTED_SCHEMA_SQL_V1 = {
    "journal_metadata": _CREATE_SCHEMA_V1[0],
    "events": _CREATE_SCHEMA_V1[1],
}

_EXPECTED_SCHEMA_SQL_V2 = {
    **_EXPECTED_SCHEMA_SQL_V1,
    "policy_decisions": _CREATE_SCHEMA_V2[0],
    "policy_reviews": _CREATE_SCHEMA_V2[1],
    "policy_decision_history": _CREATE_SCHEMA_V2[2],
    "policy_decisions_policy_sequence": _CREATE_SCHEMA_V2[3],
    "policy_history_reviewed_decision": _CREATE_SCHEMA_V2[4],
}

_EXPECTED_SCHEMA_SQL_V3 = {
    **_EXPECTED_SCHEMA_SQL_V2,
    "focus_session_links": _CREATE_SCHEMA_V3[0],
}

_EXPECTED_SCHEMA_SQL_BY_VERSION = {
    1: _EXPECTED_SCHEMA_SQL_V1,
    2: _EXPECTED_SCHEMA_SQL_V2,
    3: _EXPECTED_SCHEMA_SQL_V3,
}

_EXPECTED_TABLES_BY_VERSION = {
    1: ("journal_metadata", "events"),
    2: (
        "journal_metadata",
        "events",
        "policy_decisions",
        "policy_reviews",
        "policy_decision_history",
    ),
    3: tuple(_EXPECTED_TABLE_COLUMNS),
}


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
class PolicyJournalVerification:
    """Summary returned after replaying one policy's durable decisions."""

    policy_id: str
    decision_count: int
    review_count: int
    history_edge_count: int
    sqlite_check: str


@dataclass(frozen=True, slots=True)
class FocusSessionLink:
    """Immutable provenance link between a decision and one planned session."""

    decision_id: UUID
    decision_sequence: int
    session_id: UUID
    planned_event_id: UUID
    policy_id: str
    template_id: str


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _StoredPolicyDecision:
    decision_id: UUID
    decision_sequence: int
    policy_id: str
    rng_seed: int
    context: FocusContext
    template_id: str
    propensity: float
    history_count: int
    history_sha256: str


@dataclass(frozen=True, slots=True)
class _StoredPolicyReview:
    fit: DurationFit
    objective_completed: bool


@dataclass(frozen=True, slots=True)
class _PolicyJournalSnapshot:
    decisions: tuple[_StoredPolicyDecision, ...]
    decisions_by_id: Mapping[UUID, _StoredPolicyDecision]
    reviews: Mapping[UUID, _StoredPolicyReview]
    histories: Mapping[UUID, tuple[UUID, ...]]


@dataclass(frozen=True, slots=True)
class _PolicyReplay:
    recommendations: tuple[Recommendation, ...]
    reviews: tuple[ReviewedDecision, ...]
    history_edge_count: int
    sqlite_check: str
    snapshot: _PolicyJournalSnapshot
    links: _FocusLinkSnapshot


@dataclass(frozen=True, slots=True)
class _FocusLinkSnapshot:
    links: tuple[FocusSessionLink, ...]
    by_decision: Mapping[UUID, FocusSessionLink]
    by_planned_event: Mapping[UUID, FocusSessionLink]
    by_session: Mapping[UUID, FocusSessionLink]


def default_journal_path() -> Path:
    """Return the XDG-compatible path without creating any files."""

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    try:
        xdg_candidate = Path(xdg_data_home).expanduser() if xdg_data_home else None
    except (OSError, RuntimeError) as exc:
        raise JournalSecurityError("cannot determine the journal home") from exc
    if xdg_candidate is not None and xdg_candidate.is_absolute():
        base = xdg_candidate
    else:
        try:
            base = Path.home() / ".local" / "share"
        except (OSError, RuntimeError) as exc:
            raise JournalSecurityError("cannot determine the journal home") from exc
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
        try:
            candidate = Path(path).expanduser()
        except RuntimeError as exc:
            raise JournalSecurityError("cannot expand journal path") from exc
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
    def _require_schema_version(
        connection: sqlite3.Connection,
        *,
        allowed: tuple[int, ...] = (SCHEMA_VERSION,),
    ) -> int:
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
        if row is None:
            raise CorruptJournal("unsupported journal schema version: missing")
        stored = row["value"]
        if not isinstance(stored, str):
            raise CorruptJournal("journal schema version is not text")
        for version in allowed:
            if stored == str(version):
                return version
        raise CorruptJournal(f"unsupported journal schema version: {stored}")

    @staticmethod
    def _table_names(connection: sqlite3.Connection) -> set[str]:
        try:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        except sqlite3.Error as exc:
            raise CorruptJournal("cannot inspect journal schema") from exc
        return {row["name"] for row in rows}

    @staticmethod
    def _require_table_columns(
        connection: sqlite3.Connection,
        tables: tuple[str, ...],
    ) -> None:
        available = SQLiteEventStore._table_names(connection)
        for table in tables:
            if table not in available:
                raise CorruptJournal(f"journal schema is missing table: {table}")
            try:
                rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
            except sqlite3.Error as exc:
                raise CorruptJournal(f"cannot inspect journal table: {table}") from exc
            columns = tuple(row["name"] for row in rows)
            if columns != _EXPECTED_TABLE_COLUMNS[table]:
                raise CorruptJournal(f"journal table has unexpected columns: {table}")

    @staticmethod
    def _require_schema_layout(
        connection: sqlite3.Connection,
        *,
        version: int,
    ) -> None:
        expected_sql = _EXPECTED_SCHEMA_SQL_BY_VERSION.get(version)
        if expected_sql is None:
            raise CorruptJournal("cannot validate unknown journal schema")
        expected_names = set(expected_sql)
        try:
            rows = connection.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type IN ('table', 'index', 'trigger', 'view')
                    AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
        except sqlite3.Error as exc:
            raise CorruptJournal("cannot inspect canonical journal schema") from exc
        actual_names = {row["name"] for row in rows}
        if actual_names != expected_names:
            raise CorruptJournal("journal schema contains unexpected objects")
        for row in rows:
            name = row["name"]
            stored_sql = row["sql"]
            expected_definition = expected_sql[name]
            if not isinstance(stored_sql, str):
                raise CorruptJournal(f"journal schema object has no SQL: {name}")
            if " ".join(stored_sql.split()) != " ".join(expected_definition.split()):
                raise CorruptJournal(
                    f"journal schema object has unexpected definition: {name}"
                )

    @staticmethod
    def _create_statements(
        connection: sqlite3.Connection,
        statements: tuple[str, ...],
    ) -> None:
        for statement in statements:
            connection.execute(statement)

    @staticmethod
    def _insert_focus_link(
        connection: sqlite3.Connection,
        decision_id: str,
        planned_event_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO focus_session_links (
                decision_id,
                planned_event_id
            )
            VALUES (?, ?)
            """,
            (decision_id, planned_event_id),
        )

    def _initialize_schema(self) -> None:
        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            tables = self._table_names(connection)
            if not tables:
                self._create_statements(connection, _CREATE_SCHEMA_V1)
                self._create_statements(connection, _CREATE_SCHEMA_V2)
                self._create_statements(connection, _CREATE_SCHEMA_V3)
                connection.execute(
                    """
                    INSERT INTO journal_metadata (key, value)
                    VALUES ('schema_version', ?)
                    """,
                    (str(SCHEMA_VERSION),),
                )
            else:
                if "journal_metadata" not in tables:
                    raise CorruptJournal(
                        "journal schema is missing table: journal_metadata"
                    )
                stored_version = self._require_schema_version(
                    connection,
                    allowed=(1, 2, SCHEMA_VERSION),
                )
                if stored_version == 1:
                    if tables != set(_EXPECTED_TABLES_BY_VERSION[1]):
                        raise CorruptJournal(
                            "version 1 journal contains unexpected tables"
                        )
                    self._require_table_columns(
                        connection,
                        ("journal_metadata", "events"),
                    )
                    self._require_schema_layout(connection, version=1)
                    self._create_statements(connection, _CREATE_SCHEMA_V2)
                    self._create_statements(connection, _CREATE_SCHEMA_V3)
                    connection.execute(
                        """
                        UPDATE journal_metadata
                        SET value = ?
                        WHERE key = 'schema_version'
                        """,
                        (str(SCHEMA_VERSION),),
                    )
                elif stored_version == 2:
                    if tables != set(_EXPECTED_TABLES_BY_VERSION[2]):
                        raise CorruptJournal(
                            "version 2 journal contains unexpected tables"
                        )
                    self._require_table_columns(
                        connection,
                        _EXPECTED_TABLES_BY_VERSION[2],
                    )
                    self._require_schema_layout(connection, version=2)
                    self._create_statements(connection, _CREATE_SCHEMA_V3)
                    connection.execute(
                        """
                        UPDATE journal_metadata
                        SET value = ?
                        WHERE key = 'schema_version'
                        """,
                        (str(SCHEMA_VERSION),),
                    )
                elif tables != set(_EXPECTED_TABLES_BY_VERSION[3]):
                    raise CorruptJournal(
                        "version 3 journal has unexpected or missing tables"
                    )
            self._require_table_columns(
                connection,
                tuple(_EXPECTED_TABLE_COLUMNS),
            )
            self._require_schema_layout(connection, version=SCHEMA_VERSION)
            self._require_schema_version(connection)
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            self._rollback(connection)
            raise JournalError("cannot initialize SQLite journal") from exc
        except Exception:
            self._rollback(connection)
            raise
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

    @staticmethod
    def _require_policy(
        policy: HierarchicalSoftmaxUCB,
    ) -> HierarchicalSoftmaxUCB:
        if type(policy) is not HierarchicalSoftmaxUCB:
            raise TypeError("policy must be an exact HierarchicalSoftmaxUCB")
        checked = HierarchicalSoftmaxUCB(
            config=policy.config,
            templates=policy.templates,
        )
        if checked.policy_id != policy.policy_id:
            raise PolicyInputError(
                "policy_id does not match the current policy configuration"
            )
        return checked

    @staticmethod
    def _decode_uuid_text(value: object, field: str) -> UUID:
        if not isinstance(value, str):
            raise CorruptJournal(f"stored {field} is not text")
        try:
            identifier = UUID(value)
        except ValueError as exc:
            raise CorruptJournal(f"stored {field} is not a UUID") from exc
        if identifier.int == 0:
            raise CorruptJournal(f"stored {field} must not be the nil UUID")
        if str(identifier) != value:
            raise CorruptJournal(f"stored {field} is not canonical")
        return identifier

    @staticmethod
    def _decode_integer(
        value: object,
        field: str,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CorruptJournal(f"stored {field} is not an integer")
        if not minimum <= value <= maximum:
            raise CorruptJournal(f"stored {field} is out of bounds")
        return value

    @staticmethod
    def _decode_propensity(value: object) -> float:
        if not isinstance(value, str):
            raise CorruptJournal("stored propensity_hex is not text")
        try:
            propensity = float.fromhex(value)
        except (OverflowError, ValueError) as exc:
            raise CorruptJournal("stored propensity_hex is invalid") from exc
        if (
            not math.isfinite(propensity)
            or not 0 < propensity <= 1
            or propensity.hex() != value
        ):
            raise CorruptJournal("stored propensity_hex is not canonical")
        return propensity

    @staticmethod
    def _decode_identifier_text(value: object, field: str) -> str:
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 80
            or not all(
                character.isascii()
                and (character.islower() or character.isdigit() or character in ".-_")
                for character in value
            )
        ):
            raise CorruptJournal(f"stored {field} is not a canonical identifier")
        return value

    @staticmethod
    def _history_digest(decision_ids: tuple[UUID, ...]) -> str:
        digest = hashlib.sha256()
        for position, decision_id in enumerate(decision_ids):
            digest.update(position.to_bytes(8, "big"))
            digest.update(decision_id.bytes)
        return digest.hexdigest()

    @staticmethod
    def _decode_decision_row(
        row: sqlite3.Row,
    ) -> _StoredPolicyDecision:
        decision_id = SQLiteEventStore._decode_uuid_text(
            row["decision_id"],
            "decision_id",
        )
        sequence = SQLiteEventStore._decode_integer(
            row["decision_sequence"],
            "decision_sequence",
            minimum=1,
            maximum=MAX_DECISION_SEQUENCE,
        )
        stored_policy_id = row["policy_id"]
        SQLiteEventStore._decode_identifier_text(stored_policy_id, "policy_id")
        rng_seed = SQLiteEventStore._decode_integer(
            row["rng_seed"],
            "rng_seed",
            minimum=0,
            maximum=MAX_RNG_SEED,
        )
        task_kind_raw = row["task_kind"]
        energy_raw = row["energy"]
        if not isinstance(task_kind_raw, str):
            raise CorruptJournal("stored task_kind is not text")
        if not isinstance(energy_raw, str):
            raise CorruptJournal("stored energy is not text")
        try:
            task_kind = TaskKind(task_kind_raw)
            energy = EnergyLevel(energy_raw)
        except ValueError as exc:
            raise CorruptJournal("stored policy context enum is invalid") from exc
        available_seconds = SQLiteEventStore._decode_integer(
            row["available_seconds"],
            "available_seconds",
            minimum=1,
            maximum=MAX_AVAILABLE_SECONDS,
        )
        previous_raw = row["previous_focus_seconds"]
        previous_focus_seconds = (
            None
            if previous_raw is None
            else SQLiteEventStore._decode_integer(
                previous_raw,
                "previous_focus_seconds",
                minimum=1,
                maximum=MAX_AVAILABLE_SECONDS,
            )
        )
        try:
            context = FocusContext(
                task_kind=task_kind,
                energy=energy,
                available_seconds=available_seconds,
                previous_focus_seconds=previous_focus_seconds,
            )
        except PolicyInputError as exc:
            raise CorruptJournal("stored policy context is invalid") from exc
        template_id = row["template_id"]
        SQLiteEventStore._decode_identifier_text(template_id, "template_id")
        propensity = SQLiteEventStore._decode_propensity(row["propensity_hex"])
        history_count = SQLiteEventStore._decode_integer(
            row["history_count"],
            "history_count",
            minimum=0,
            maximum=MAX_DECISION_SEQUENCE,
        )
        history_sha256 = row["history_sha256"]
        if (
            not isinstance(history_sha256, str)
            or len(history_sha256) != 64
            or any(character not in "0123456789abcdef" for character in history_sha256)
        ):
            raise CorruptJournal("stored history_sha256 is not canonical")
        return _StoredPolicyDecision(
            decision_id=decision_id,
            decision_sequence=sequence,
            policy_id=stored_policy_id,
            rng_seed=rng_seed,
            context=context,
            template_id=template_id,
            propensity=propensity,
            history_count=history_count,
            history_sha256=history_sha256,
        )

    @staticmethod
    def _decode_review_row(row: sqlite3.Row) -> tuple[UUID, _StoredPolicyReview]:
        decision_id = SQLiteEventStore._decode_uuid_text(
            row["decision_id"],
            "review decision_id",
        )
        fit_raw = row["fit"]
        if not isinstance(fit_raw, str):
            raise CorruptJournal("stored review fit is not text")
        try:
            fit = DurationFit(fit_raw)
        except ValueError as exc:
            raise CorruptJournal("stored review fit is invalid") from exc
        completed_raw = row["objective_completed"]
        if (
            isinstance(completed_raw, bool)
            or not isinstance(completed_raw, int)
            or completed_raw not in (0, 1)
        ):
            raise CorruptJournal("stored objective_completed is not canonical")
        return decision_id, _StoredPolicyReview(
            fit=fit,
            objective_completed=bool(completed_raw),
        )

    def _load_policy_snapshot(
        self,
        connection: sqlite3.Connection,
    ) -> _PolicyJournalSnapshot:
        try:
            decision_rows = connection.execute(
                """
                SELECT
                    decision_id,
                    decision_sequence,
                    policy_id,
                    rng_seed,
                    task_kind,
                    energy,
                    available_seconds,
                    previous_focus_seconds,
                    template_id,
                    propensity_hex,
                    history_count,
                    history_sha256
                FROM policy_decisions
                ORDER BY decision_sequence
                """
            ).fetchall()
            review_rows = connection.execute(
                """
                SELECT decision_id, fit, objective_completed
                FROM policy_reviews
                ORDER BY decision_id
                """
            ).fetchall()
            edge_rows = connection.execute(
                """
                SELECT decision_id, position, reviewed_decision_id
                FROM policy_decision_history
                ORDER BY decision_id, position
                """
            ).fetchall()
        except sqlite3.Error as exc:
            raise CorruptJournal("cannot read policy journal relations") from exc

        decisions: list[_StoredPolicyDecision] = []
        decisions_by_id: dict[UUID, _StoredPolicyDecision] = {}
        for expected_sequence, row in enumerate(decision_rows, start=1):
            decision = self._decode_decision_row(row)
            if decision.decision_sequence != expected_sequence:
                raise CorruptJournal(
                    "stored global decision_sequence values are not contiguous"
                )
            if decision.decision_id in decisions_by_id:
                raise CorruptJournal("stored policy decisions repeat decision_id")
            decisions.append(decision)
            decisions_by_id[decision.decision_id] = decision

        reviews: dict[UUID, _StoredPolicyReview] = {}
        for row in review_rows:
            decision_id, review = self._decode_review_row(row)
            if decision_id not in decisions_by_id:
                raise CorruptJournal("stored policy review is orphaned")
            if decision_id in reviews:
                raise CorruptJournal("stored policy reviews repeat decision_id")
            reviews[decision_id] = review

        grouped_edges: dict[UUID, list[UUID]] = {}
        for edge in edge_rows:
            target_id = self._decode_uuid_text(
                edge["decision_id"],
                "history decision_id",
            )
            reviewed_id = self._decode_uuid_text(
                edge["reviewed_decision_id"],
                "reviewed_decision_id",
            )
            target = decisions_by_id.get(target_id)
            if target is None:
                raise CorruptJournal("stored policy history target is orphaned")
            reviewed = decisions_by_id.get(reviewed_id)
            if reviewed is None or reviewed_id not in reviews:
                raise CorruptJournal("stored policy history references no review")
            if target.policy_id != reviewed.policy_id:
                raise CorruptJournal("stored policy history crosses policy_id")
            if reviewed.decision_sequence >= target.decision_sequence:
                raise CorruptJournal("stored policy history contains a future decision")
            group = grouped_edges.setdefault(target_id, [])
            position = self._decode_integer(
                edge["position"],
                "history position",
                minimum=0,
                maximum=MAX_DECISION_SEQUENCE,
            )
            if position != len(group):
                raise CorruptJournal(
                    "stored policy history positions are not contiguous"
                )
            if reviewed_id in group:
                raise CorruptJournal("stored policy history repeats a decision")
            if (
                group
                and reviewed.decision_sequence
                <= decisions_by_id[group[-1]].decision_sequence
            ):
                raise CorruptJournal(
                    "stored policy history is not in decision sequence order"
                )
            group.append(reviewed_id)

        histories: dict[UUID, tuple[UUID, ...]] = {}
        for decision in decisions:
            history = tuple(grouped_edges.get(decision.decision_id, ()))
            if len(history) != decision.history_count:
                raise CorruptJournal("stored policy history count disagrees")
            if self._history_digest(history) != decision.history_sha256:
                raise CorruptJournal("stored policy history digest disagrees")
            histories[decision.decision_id] = history

        return _PolicyJournalSnapshot(
            decisions=tuple(decisions),
            decisions_by_id=MappingProxyType(decisions_by_id),
            reviews=MappingProxyType(reviews),
            histories=MappingProxyType(histories),
        )

    def _recompute_decision(
        self,
        snapshot: _PolicyJournalSnapshot,
        policy: HierarchicalSoftmaxUCB,
        decision: _StoredPolicyDecision,
        cache: dict[UUID, Recommendation],
        active: set[UUID],
    ) -> Recommendation:
        if decision.policy_id != policy.policy_id:
            raise CorruptJournal("stored policy_id does not match policy")
        cached = cache.get(decision.decision_id)
        if cached is not None:
            return cached
        if decision.decision_id in active:
            raise CorruptJournal("stored policy history contains a cycle")
        active.add(decision.decision_id)
        try:
            history_ids = snapshot.histories[decision.decision_id]
            if len(history_ids) > policy.config.window_size:
                raise CorruptJournal(
                    "stored policy history exceeds the configured window"
                )
            reviews: list[ReviewedDecision] = []
            for history_id in history_ids:
                history_decision = snapshot.decisions_by_id[history_id]
                history_recommendation = self._recompute_decision(
                    snapshot,
                    policy,
                    history_decision,
                    cache,
                    active,
                )
                stored_review = snapshot.reviews.get(history_id)
                if stored_review is None:
                    raise CorruptJournal("stored policy history references no review")
                try:
                    reviews.append(
                        history_recommendation.review(
                            fit=stored_review.fit,
                            objective_completed=stored_review.objective_completed,
                        )
                    )
                except PolicyInputError as exc:
                    raise CorruptJournal("stored policy review is invalid") from exc
            try:
                recommendation = policy.recommend_seeded(
                    decision.context,
                    tuple(reviews),
                    decision_id=decision.decision_id,
                    decision_sequence=decision.decision_sequence,
                    rng_seed=decision.rng_seed,
                )
            except PolicyInputError as exc:
                raise CorruptJournal(
                    "stored policy decision cannot be recomputed"
                ) from exc
            if recommendation.template.template_id != decision.template_id:
                raise CorruptJournal(
                    "stored template_id disagrees with policy recomputation"
                )
            if recommendation.propensity.hex() != decision.propensity.hex():
                raise CorruptJournal(
                    "stored propensity_hex disagrees with policy recomputation"
                )
            cache[decision.decision_id] = recommendation
            return recommendation
        finally:
            active.remove(decision.decision_id)

    def _load_focus_links(
        self,
        connection: sqlite3.Connection,
        snapshot: _PolicyJournalSnapshot,
        recommendations: Mapping[UUID, Recommendation],
    ) -> _FocusLinkSnapshot:
        try:
            rows = connection.execute(
                """
                SELECT
                    links.decision_id AS link_decision_id,
                    links.planned_event_id AS link_planned_event_id,
                    events.session_id AS session_id,
                    events.sequence AS sequence,
                    events.event_id AS event_id,
                    events.event_type AS event_type,
                    events.event_json AS event_json
                FROM focus_session_links AS links
                LEFT JOIN events
                    ON events.event_id = links.planned_event_id
                ORDER BY links.decision_id
                """
            ).fetchall()
        except sqlite3.Error as exc:
            raise CorruptJournal("cannot read focus-session links") from exc

        links: list[FocusSessionLink] = []
        by_decision: dict[UUID, FocusSessionLink] = {}
        by_planned_event: dict[UUID, FocusSessionLink] = {}
        by_session: dict[UUID, FocusSessionLink] = {}
        for row in rows:
            decision_id = self._decode_uuid_text(
                row["link_decision_id"],
                "link decision_id",
            )
            planned_event_id = self._decode_uuid_text(
                row["link_planned_event_id"],
                "link planned_event_id",
            )
            decision = snapshot.decisions_by_id.get(decision_id)
            if decision is None or row["event_json"] is None:
                raise CorruptJournal("stored focus-session link is orphaned")
            try:
                event = self._decode_row(row)
            except CorruptJournal as exc:
                raise CorruptJournal("linked planned event is invalid") from exc
            if event.event_id != planned_event_id:
                raise CorruptJournal(
                    "link planned_event_id disagrees with the referenced event"
                )
            if not isinstance(event, SessionPlanned) or event.sequence != 1:
                raise CorruptJournal(
                    "focus-session link must reference a sequence-1 plan"
                )
            if event.policy_id != decision.policy_id:
                raise CorruptJournal(
                    "linked event policy_id disagrees with policy decision"
                )
            recommendation = recommendations.get(decision_id)
            if recommendation is not None:
                if recommendation.policy_id != event.policy_id:
                    raise CorruptJournal(
                        "linked recommendation policy_id disagrees with event"
                    )
                if (
                    event.target_focus_seconds != recommendation.template.focus_seconds
                    or event.target_break_seconds
                    != recommendation.template.break_seconds
                ):
                    raise CorruptJournal(
                        "linked plan durations disagree with policy recomputation"
                    )
            link = FocusSessionLink(
                decision_id=decision_id,
                decision_sequence=decision.decision_sequence,
                session_id=event.session_id,
                planned_event_id=planned_event_id,
                policy_id=decision.policy_id,
                template_id=decision.template_id,
            )
            if (
                decision_id in by_decision
                or planned_event_id in by_planned_event
                or event.session_id in by_session
            ):
                raise CorruptJournal("stored focus-session links are not one-to-one")
            links.append(link)
            by_decision[decision_id] = link
            by_planned_event[planned_event_id] = link
            by_session[event.session_id] = link

        return _FocusLinkSnapshot(
            links=tuple(links),
            by_decision=MappingProxyType(by_decision),
            by_planned_event=MappingProxyType(by_planned_event),
            by_session=MappingProxyType(by_session),
        )

    @staticmethod
    def _policy_integrity_check(connection: sqlite3.Connection) -> str:
        try:
            check_values = [
                row[0] for row in connection.execute("PRAGMA quick_check").fetchall()
            ]
            foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        except sqlite3.Error as exc:
            raise CorruptJournal("cannot check policy journal integrity") from exc
        if check_values != ["ok"]:
            raise CorruptJournal(f"SQLite quick_check failed: {','.join(check_values)}")
        if foreign_key_rows:
            raise CorruptJournal("SQLite foreign_key_check failed")
        return "ok"

    def _replay_policy_with_connection(
        self,
        connection: sqlite3.Connection,
        policy: HierarchicalSoftmaxUCB,
    ) -> _PolicyReplay:
        self._require_schema_version(connection)
        self._require_table_columns(
            connection,
            tuple(_EXPECTED_TABLE_COLUMNS),
        )
        self._require_schema_layout(connection, version=SCHEMA_VERSION)
        sqlite_check = self._policy_integrity_check(connection)
        snapshot = self._load_policy_snapshot(connection)
        cache: dict[UUID, Recommendation] = {}
        recommendations: list[Recommendation] = []
        reviews: list[ReviewedDecision] = []
        edge_count = 0
        for decision in snapshot.decisions:
            if decision.policy_id != policy.policy_id:
                continue
            recommendation = self._recompute_decision(
                snapshot,
                policy,
                decision,
                cache,
                set(),
            )
            recommendations.append(recommendation)
            edge_count += len(snapshot.histories[recommendation.decision_id])
            stored_review = snapshot.reviews.get(recommendation.decision_id)
            if stored_review is not None:
                try:
                    reviews.append(
                        recommendation.review(
                            fit=stored_review.fit,
                            objective_completed=stored_review.objective_completed,
                        )
                    )
                except PolicyInputError as exc:
                    raise CorruptJournal("stored policy review is invalid") from exc
        recommendation_by_id = {
            recommendation.decision_id: recommendation
            for recommendation in recommendations
        }
        links = self._load_focus_links(
            connection,
            snapshot,
            recommendation_by_id,
        )
        return _PolicyReplay(
            recommendations=tuple(recommendations),
            reviews=tuple(reviews),
            history_edge_count=edge_count,
            sqlite_check=sqlite_check,
            snapshot=snapshot,
            links=links,
        )

    def recommend(
        self,
        policy: HierarchicalSoftmaxUCB,
        context: FocusContext,
        *,
        decision_id: UUID,
        rng_seed: int,
    ) -> Recommendation:
        """Atomically allocate and persist one reproducible policy decision."""

        checked_policy = self._require_policy(policy)
        identifier = _canonical_uuid(decision_id, "decision_id")
        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._replay_policy_with_connection(
                connection,
                checked_policy,
            )
            if decision_id in replay.snapshot.decisions_by_id:
                raise JournalConflict("decision_id already exists")
            maximum = len(replay.snapshot.decisions)
            if maximum == MAX_DECISION_SEQUENCE:
                raise JournalConflict("decision sequence space is exhausted")
            sequence = maximum + 1
            history = replay.reviews[-checked_policy.config.window_size :]
            recommendation = checked_policy.recommend_seeded(
                context,
                history,
                decision_id=decision_id,
                decision_sequence=sequence,
                rng_seed=rng_seed,
            )
            history_ids = tuple(review.decision_id for review in history)
            connection.execute(
                """
                INSERT INTO policy_decisions (
                    decision_id,
                    decision_sequence,
                    policy_id,
                    rng_seed,
                    task_kind,
                    energy,
                    available_seconds,
                    previous_focus_seconds,
                    template_id,
                    propensity_hex,
                    history_count,
                    history_sha256
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    sequence,
                    checked_policy.policy_id,
                    rng_seed,
                    context.task_kind.value,
                    context.energy.value,
                    context.available_seconds,
                    context.previous_focus_seconds,
                    recommendation.template.template_id,
                    recommendation.propensity.hex(),
                    len(history_ids),
                    self._history_digest(history_ids),
                ),
            )
            connection.executemany(
                """
                INSERT INTO policy_decision_history (
                    decision_id,
                    position,
                    reviewed_decision_id
                )
                VALUES (?, ?, ?)
                """,
                (
                    (identifier, position, str(reviewed_id))
                    for position, reviewed_id in enumerate(history_ids)
                ),
            )
            connection.execute("COMMIT")
            return recommendation
        except sqlite3.IntegrityError as exc:
            self._rollback(connection)
            raise JournalConflict("policy decision conflicts with journal") from exc
        except sqlite3.OperationalError as exc:
            self._rollback(connection)
            if self._is_lock_error(exc):
                raise JournalConflict("policy journal is locked by a writer") from exc
            raise JournalError("cannot persist policy decision") from exc
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)

    def link_focus_session(
        self,
        policy: HierarchicalSoftmaxUCB,
        decision_id: UUID,
        *,
        session_id: UUID,
    ) -> FocusSessionLink:
        """Atomically link an unreviewed decision to an existing planned session."""

        checked_policy = self._require_policy(policy)
        decision_identifier = _canonical_uuid(decision_id, "decision_id")
        _canonical_uuid(session_id, "session_id")
        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._replay_policy_with_connection(
                connection,
                checked_policy,
            )
            decision = replay.snapshot.decisions_by_id.get(decision_id)
            if decision is None:
                raise JournalConflict("policy decision does not exist")
            if decision.policy_id != checked_policy.policy_id:
                raise JournalConflict("policy decision belongs to a different policy")
            if decision_id in replay.snapshot.reviews:
                raise JournalConflict("policy decision is already reviewed")
            if decision_id in replay.links.by_decision:
                raise JournalConflict("policy decision is already linked")
            recommendation = next(
                (
                    candidate
                    for candidate in replay.recommendations
                    if candidate.decision_id == decision_id
                ),
                None,
            )
            if recommendation is None:
                raise CorruptJournal(
                    "policy decision is missing from recomputed history"
                )

            events = self._load_with_connection(connection, session_id)
            if not events:
                raise JournalConflict("focus session does not exist")
            try:
                state = reduce_events(events)
            except InvalidTransition as exc:
                raise CorruptJournal("target focus session cannot be replayed") from exc
            if (
                len(events) != 1
                or state.phase is not SessionPhase.PLANNED
                or state.revision != 1
                or not isinstance(events[0], SessionPlanned)
            ):
                raise JournalConflict(
                    "focus session must be an unstarted revision-1 plan"
                )
            planned = events[0]
            if planned.event_id in replay.links.by_planned_event:
                raise JournalConflict("focus session is already linked")
            if planned.policy_id != checked_policy.policy_id:
                raise JournalConflict("focus session belongs to a different policy")
            if (
                planned.target_focus_seconds != recommendation.template.focus_seconds
                or planned.target_break_seconds != recommendation.template.break_seconds
            ):
                raise JournalConflict(
                    "focus session durations do not match the recommendation"
                )

            self._insert_focus_link(
                connection,
                decision_identifier,
                str(planned.event_id),
            )
            connection.execute("COMMIT")
            return FocusSessionLink(
                decision_id=decision_id,
                decision_sequence=decision.decision_sequence,
                session_id=session_id,
                planned_event_id=planned.event_id,
                policy_id=decision.policy_id,
                template_id=decision.template_id,
            )
        except sqlite3.IntegrityError as exc:
            self._rollback(connection)
            raise JournalConflict("focus-session link conflicts with journal") from exc
        except sqlite3.OperationalError as exc:
            self._rollback(connection)
            if self._is_lock_error(exc):
                raise JournalConflict(
                    "focus-session journal is locked by a writer"
                ) from exc
            raise JournalError("cannot persist focus-session link") from exc
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)

    def focus_session_link(
        self,
        policy: HierarchicalSoftmaxUCB,
        *,
        session_id: UUID,
    ) -> FocusSessionLink | None:
        """Return the validated link derived from one session's planned event."""

        checked_policy = self._require_policy(policy)
        _canonical_uuid(session_id, "session_id")
        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN")
            replay = self._replay_policy_with_connection(
                connection,
                checked_policy,
            )
            link = replay.links.by_session.get(session_id)
            if link is not None:
                if link.policy_id != checked_policy.policy_id:
                    raise JournalConflict(
                        "focus session is linked to a different policy"
                    )
                events = self._load_with_connection(connection, session_id)
                try:
                    state = reduce_events(events)
                except InvalidTransition as exc:
                    raise CorruptJournal(
                        "linked focus session cannot be replayed"
                    ) from exc
                if (
                    not events
                    or state.session_id != session_id
                    or events[0].event_id != link.planned_event_id
                ):
                    raise CorruptJournal("linked focus session disagrees with its plan")
            connection.execute("COMMIT")
            return link
        except sqlite3.OperationalError as exc:
            self._rollback(connection)
            if self._is_lock_error(exc):
                raise JournalConflict(
                    "focus-session journal is locked by a writer"
                ) from exc
            raise JournalError("cannot read focus-session link") from exc
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)

    def record_review(
        self,
        policy: HierarchicalSoftmaxUCB,
        decision_id: UUID,
        *,
        fit: DurationFit,
        objective_completed: bool,
    ) -> ReviewedDecision:
        """Validate and atomically attach feedback to one frozen decision."""

        checked_policy = self._require_policy(policy)
        identifier = _canonical_uuid(decision_id, "decision_id")
        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._replay_policy_with_connection(
                connection,
                checked_policy,
            )
            by_id = {
                recommendation.decision_id: recommendation
                for recommendation in replay.recommendations
            }
            recommendation = by_id.get(decision_id)
            if recommendation is None:
                stored = replay.snapshot.decisions_by_id.get(decision_id)
                message = (
                    "policy decision does not exist"
                    if stored is None
                    else "policy decision belongs to a different policy"
                )
                raise JournalConflict(message)
            if decision_id in replay.snapshot.reviews:
                raise JournalConflict("policy decision is already reviewed")
            reviewed = recommendation.review(
                fit=fit,
                objective_completed=objective_completed,
            )
            connection.execute(
                """
                INSERT INTO policy_reviews (
                    decision_id,
                    fit,
                    objective_completed
                )
                VALUES (?, ?, ?)
                """,
                (
                    identifier,
                    reviewed.fit.value,
                    int(reviewed.objective_completed),
                ),
            )
            connection.execute("COMMIT")
            return reviewed
        except sqlite3.IntegrityError as exc:
            self._rollback(connection)
            raise JournalConflict("policy review conflicts with journal") from exc
        except sqlite3.OperationalError as exc:
            self._rollback(connection)
            if self._is_lock_error(exc):
                raise JournalConflict("policy journal is locked by a writer") from exc
            raise JournalError("cannot persist policy review") from exc
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)

    def reviewed_decisions(
        self,
        policy: HierarchicalSoftmaxUCB,
    ) -> tuple[ReviewedDecision, ...]:
        """Replay all reviewed decisions for one exact policy configuration."""

        checked_policy = self._require_policy(policy)
        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN")
            replay = self._replay_policy_with_connection(
                connection,
                checked_policy,
            )
            connection.execute("COMMIT")
            return replay.reviews
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)

    def verify_policy_history(
        self,
        policy: HierarchicalSoftmaxUCB,
    ) -> PolicyJournalVerification:
        """Replay one complete policy journal and verify every frozen edge."""

        checked_policy = self._require_policy(policy)
        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN")
            replay = self._replay_policy_with_connection(
                connection,
                checked_policy,
            )
            connection.execute("COMMIT")
        except Exception:
            self._rollback(connection)
            raise
        finally:
            self._close_connection(connection, directory_descriptor)
        return PolicyJournalVerification(
            policy_id=checked_policy.policy_id,
            decision_count=len(replay.recommendations),
            review_count=len(replay.reviews),
            history_edge_count=replay.history_edge_count,
            sqlite_check=replay.sqlite_check,
        )

    def append(self, event: DomainEvent) -> SessionState:
        """Validate and atomically append one next event."""

        encoded = encode_event(event)
        connection, directory_descriptor = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._require_schema_version(connection)
            self._require_table_columns(
                connection,
                tuple(_EXPECTED_TABLE_COLUMNS),
            )
            self._require_schema_layout(connection, version=SCHEMA_VERSION)
            self._policy_integrity_check(connection)
            policy_snapshot = self._load_policy_snapshot(connection)
            self._load_focus_links(connection, policy_snapshot, {})
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

    @staticmethod
    def _is_lock_error(error: sqlite3.OperationalError) -> bool:
        code = getattr(error, "sqlite_errorcode", None)
        return isinstance(code, int) and (code & 0xFF) in {
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
        }

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
            self._require_table_columns(
                connection,
                tuple(_EXPECTED_TABLE_COLUMNS),
            )
            self._require_schema_layout(connection, version=SCHEMA_VERSION)
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
            foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_rows:
                raise CorruptJournal("SQLite foreign_key_check failed")
            policy_snapshot = self._load_policy_snapshot(connection)
            self._load_focus_links(connection, policy_snapshot, {})
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
