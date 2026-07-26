from __future__ import annotations

import os
import sqlite3
import stat
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from gworker.codec import encode_event
from gworker.domain import (
    BreakCompleted,
    BreakStarted,
    FocusCompleted,
    FocusStarted,
    SessionPhase,
    SessionPlanned,
)
from gworker.storage import (
    CorruptJournal,
    JournalConflict,
    JournalSecurityError,
    SQLiteEventStore,
    default_journal_path,
)

SESSION_ID = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e1000")
OTHER_SESSION_ID = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e2000")
BASE_TIME = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)


def metadata(
    sequence: int,
    *,
    session_id: UUID = SESSION_ID,
    event_suffix: int | None = None,
) -> dict[str, object]:
    return {
        "event_id": UUID(f"018f4f69-e7a2-7f84-8c2d-{event_suffix or sequence:012d}"),
        "session_id": session_id,
        "sequence": sequence,
        "occurred_at": BASE_TIME + timedelta(seconds=sequence),
    }


def plan(
    *,
    session_id: UUID = SESSION_ID,
    event_suffix: int = 1,
) -> SessionPlanned:
    return SessionPlanned(
        **metadata(1, session_id=session_id, event_suffix=event_suffix),
        objective="Exercise durable replay",
        target_focus_seconds=1_500,
        target_break_seconds=300,
        policy_id="fixed-baseline-v1",
    )


class SQLiteEventStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = self.root / "private" / "events.sqlite3"
        self.store = SQLiteEventStore(self.database)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_creation_is_explicit_and_private(self) -> None:
        self.assertTrue(self.database.is_file())
        self.assertEqual(
            stat.S_IMODE(self.database.stat().st_mode),
            0o600,
        )
        self.assertEqual(
            stat.S_IMODE(self.database.parent.stat().st_mode),
            0o700,
        )

    def test_append_load_and_replay_survive_reopen(self) -> None:
        events = [
            plan(),
            FocusStarted(**metadata(2)),
            FocusCompleted(**metadata(3), elapsed_seconds=1_490),
            BreakStarted(**metadata(4)),
            BreakCompleted(**metadata(5), elapsed_seconds=300),
        ]
        for event in events:
            self.store.append(event)

        reopened = SQLiteEventStore(self.database)
        self.assertEqual(reopened.load(SESSION_ID), events)
        state = reopened.replay(SESSION_ID)
        self.assertEqual(state.phase, SessionPhase.COMPLETED)
        self.assertEqual(state.actual_focus_seconds, 1_490)

        report = reopened.verify()
        self.assertEqual(report.sqlite_check, "ok")
        self.assertEqual(report.session_count, 1)
        self.assertEqual(report.event_count, 5)

    def test_empty_journal_verifies(self) -> None:
        report = self.store.verify()
        self.assertEqual(report.session_count, 0)
        self.assertEqual(report.event_count, 0)

    def test_missing_session_raises_key_error(self) -> None:
        with self.assertRaisesRegex(KeyError, str(OTHER_SESSION_ID)):
            self.store.replay(OTHER_SESSION_ID)

    def test_append_rejects_gap_and_rolls_back(self) -> None:
        self.store.append(plan())
        with self.assertRaisesRegex(JournalConflict, "expected sequence 2"):
            self.store.append(FocusStarted(**metadata(3)))
        self.assertEqual(self.store.load(SESSION_ID), [plan()])

    def test_event_id_is_unique_across_sessions(self) -> None:
        self.store.append(plan())
        with self.assertRaisesRegex(JournalConflict, "event_id already exists"):
            self.store.append(
                plan(
                    session_id=OTHER_SESSION_ID,
                    event_suffix=1,
                )
            )
        self.assertEqual(self.store.session_ids(), [SESSION_ID])

    def test_two_writers_cannot_append_the_same_revision(self) -> None:
        self.store.append(plan())
        barrier = threading.Barrier(2)
        outcomes: list[str] = []

        def append_competing(event_suffix: int) -> None:
            contender = SQLiteEventStore(self.database)
            barrier.wait()
            try:
                contender.append(
                    FocusStarted(
                        **metadata(2, event_suffix=event_suffix),
                    )
                )
            except JournalConflict:
                outcomes.append("conflict")
            else:
                outcomes.append("committed")

        threads = [
            threading.Thread(target=append_competing, args=(20,)),
            threading.Thread(target=append_competing, args=(21,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(sorted(outcomes), ["committed", "conflict"])
        self.assertEqual(len(self.store.load(SESSION_ID)), 2)

    def test_corrupt_event_json_fails_closed(self) -> None:
        self.store.append(plan())
        connection = sqlite3.connect(self.database)
        connection.execute(
            "UPDATE events SET event_json = ?",
            ('{"schema_version":1}',),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "cannot decode stored event"):
            self.store.load(SESSION_ID)
        with self.assertRaisesRegex(CorruptJournal, "cannot decode stored event"):
            self.store.verify()

    def test_indexed_fields_must_match_event_document(self) -> None:
        self.store.append(plan())
        connection = sqlite3.connect(self.database)
        connection.execute(
            "UPDATE events SET event_type = ?",
            ("focus_started",),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "indexed event fields disagree"):
            self.store.load(SESSION_ID)

    def test_rejects_relative_symlink_and_public_database_paths(self) -> None:
        with self.assertRaisesRegex(JournalSecurityError, "absolute"):
            SQLiteEventStore(Path("relative.sqlite3"))

        target = self.root / "target.sqlite3"
        target.touch(mode=0o600)
        link = self.root / "link.sqlite3"
        link.symlink_to(target)
        with self.assertRaisesRegex(JournalSecurityError, "symlink"):
            SQLiteEventStore(link)

        public = self.root / "public.sqlite3"
        public.touch(mode=0o644)
        public.chmod(0o644)
        with self.assertRaisesRegex(JournalSecurityError, "group or others"):
            SQLiteEventStore(public)

    def test_rejects_unsafe_or_symlinked_parent_directory(self) -> None:
        unsafe = self.root / "unsafe"
        unsafe.mkdir(mode=0o700)
        unsafe.chmod(0o777)
        with self.assertRaisesRegex(JournalSecurityError, "journal directory"):
            SQLiteEventStore(unsafe / "events.sqlite3")

        writable_ancestor = self.root / "writable-ancestor"
        writable_ancestor.mkdir(mode=0o700)
        private_child = writable_ancestor / "private"
        private_child.mkdir(mode=0o700)
        writable_ancestor.chmod(0o777)
        pinned = SQLiteEventStore(private_child / "events.sqlite3")
        self.assertEqual(pinned.session_ids(), [])

        target = self.root / "target-directory"
        target.mkdir(mode=0o700)
        link = self.root / "linked-directory"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(JournalSecurityError, "must not contain symlinks"):
            SQLiteEventStore(link / "events.sqlite3")

    def test_rejects_hardlinked_database(self) -> None:
        source = self.root / "hardlink-source.sqlite3"
        source.touch(mode=0o600)
        alias = self.root / "hardlink-alias.sqlite3"
        os.link(source, alias)

        with self.assertRaisesRegex(JournalSecurityError, "must not have hard links"):
            SQLiteEventStore(alias)

    def test_rejects_path_rebinding_after_initialization(self) -> None:
        victim = SQLiteEventStore(
            self.root / "victim" / "events.sqlite3",
        )
        source_path = self.root / "source" / "events.sqlite3"
        source = SQLiteEventStore(source_path)
        preserved = source_path.with_name("preserved.sqlite3")
        source_path.rename(preserved)
        source_path.symlink_to(victim.path)

        with self.assertRaisesRegex(JournalSecurityError, "reopen journal file"):
            source.append(plan())
        self.assertEqual(victim.session_ids(), [])

    def test_rejects_regular_file_replacement_after_initialization(self) -> None:
        source_path = self.root / "replace" / "events.sqlite3"
        source = SQLiteEventStore(source_path)
        source_path.rename(source_path.with_name("preserved.sqlite3"))
        source_path.touch(mode=0o600)

        with self.assertRaisesRegex(JournalSecurityError, "identity changed"):
            source.session_ids()

    def test_rejects_parent_replacement_after_initialization(self) -> None:
        source_path = self.root / "parent-replace" / "events.sqlite3"
        source = SQLiteEventStore(source_path)
        preserved = source_path.parent.with_name("preserved-parent")
        source_path.parent.rename(preserved)
        source_path.parent.mkdir(mode=0o700)

        with self.assertRaisesRegex(
            JournalSecurityError,
            "directory identity changed",
        ):
            source.session_ids()

    def test_revalidates_permissions_and_link_count_each_operation(self) -> None:
        public_parent_store = SQLiteEventStore(
            self.root / "parent-mode" / "events.sqlite3"
        )
        public_parent_store.path.parent.chmod(0o750)
        with self.assertRaisesRegex(JournalSecurityError, "group or others"):
            public_parent_store.verify()

        hardlinked_store = SQLiteEventStore(
            self.root / "late-hardlink" / "events.sqlite3"
        )
        os.link(
            hardlinked_store.path,
            hardlinked_store.path.with_name("alias.sqlite3"),
        )
        with self.assertRaisesRegex(JournalSecurityError, "hard links"):
            hardlinked_store.verify()

    def test_rejects_directory_as_database(self) -> None:
        directory = self.root / "not-a-file"
        directory.mkdir()
        with self.assertRaisesRegex(JournalSecurityError, "regular file"):
            SQLiteEventStore(directory)

    def test_default_path_honors_xdg_without_writing(self) -> None:
        xdg = self.root / "xdg-data"
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg)}):
            path = default_journal_path()
        self.assertEqual(path, xdg / "gworker" / "events.sqlite3")
        self.assertFalse(xdg.exists())

    def test_default_path_falls_back_to_home_without_writing(self) -> None:
        home = self.root / "home"
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("gworker.storage.Path.home", return_value=home),
        ):
            path = default_journal_path()
        self.assertEqual(
            path,
            home / ".local" / "share" / "gworker" / "events.sqlite3",
        )
        self.assertFalse(home.exists())

    def test_relative_xdg_path_is_ignored(self) -> None:
        home = self.root / "fallback-home"
        with (
            patch.dict(os.environ, {"XDG_DATA_HOME": "relative/data"}, clear=True),
            patch("gworker.storage.Path.home", return_value=home),
        ):
            path = default_journal_path()
        self.assertEqual(
            path,
            home / ".local" / "share" / "gworker" / "events.sqlite3",
        )

    def test_unexpandable_paths_fail_as_journal_security_errors(self) -> None:
        with (
            patch(
                "gworker.storage.Path.expanduser",
                side_effect=RuntimeError("injected home failure"),
            ),
            self.assertRaisesRegex(JournalSecurityError, "expand journal path"),
        ):
            SQLiteEventStore(Path("~/events.sqlite3"))

        with (
            patch.dict(
                os.environ,
                {"XDG_DATA_HOME": "~/private-data"},
                clear=True,
            ),
            patch(
                "gworker.storage.Path.expanduser",
                side_effect=RuntimeError("injected home failure"),
            ),
            self.assertRaisesRegex(JournalSecurityError, "journal home"),
        ):
            default_journal_path()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "gworker.storage.Path.home",
                side_effect=OSError("injected home failure"),
            ),
            self.assertRaisesRegex(JournalSecurityError, "journal home"),
        ):
            default_journal_path()

    def test_session_lookup_requires_a_non_nil_uuid(self) -> None:
        with self.assertRaisesRegex(TypeError, "session_id must be a UUID"):
            self.store.load("not-a-uuid")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "must not be the nil UUID"):
            self.store.load(UUID(int=0))

    def test_rejects_unsupported_stored_schema_version(self) -> None:
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            UPDATE journal_metadata
            SET value = '99'
            WHERE key = 'schema_version'
            """
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "schema version: 99"):
            self.store.append(plan())
        with self.assertRaisesRegex(CorruptJournal, "schema version: 99"):
            self.store.verify()

        with self.assertRaisesRegex(CorruptJournal, "schema version: 99"):
            SQLiteEventStore(self.database)

        connection = sqlite3.connect(self.database)
        event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        connection.close()
        self.assertEqual(event_count, 0)

    def test_session_listing_rejects_invalid_identifier(self) -> None:
        self.store.append(plan())
        connection = sqlite3.connect(self.database)
        connection.execute(
            "UPDATE events SET session_id = 'not-a-uuid'",
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "not a UUID"):
            self.store.session_ids()

        connection = sqlite3.connect(self.database)
        connection.execute(
            "UPDATE events SET session_id = ?",
            (str(SESSION_ID).upper(),),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "not canonical"):
            self.store.session_ids()

    def test_load_rejects_a_structurally_valid_sequence_gap(self) -> None:
        self.store.append(plan())
        gap_event = SessionPlanned(
            **{
                **metadata(2),
                "objective": "Exercise durable replay",
                "target_focus_seconds": 1_500,
                "target_break_seconds": 300,
                "policy_id": "fixed-baseline-v1",
            }
        )
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            UPDATE events
            SET sequence = 2, event_id = ?, event_json = ?
            """,
            (str(gap_event.event_id), encode_event(gap_event)),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "invalid session stream"):
            self.store.load(SESSION_ID)
        with self.assertRaisesRegex(CorruptJournal, "invalid session stream"):
            self.store.verify()
        with self.assertRaisesRegex(CorruptJournal, "invalid existing session stream"):
            self.store.append(FocusStarted(**metadata(3)))

    def test_missing_schema_or_events_table_fails_closed(self) -> None:
        connection = sqlite3.connect(self.database)
        connection.execute("DROP TABLE journal_metadata")
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "read journal schema version"):
            self.store.load(SESSION_ID)

        missing_row = SQLiteEventStore(
            self.root / "missing-schema-row" / "events.sqlite3"
        )
        connection = sqlite3.connect(missing_row.path)
        connection.execute("DELETE FROM journal_metadata")
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "schema version: missing"):
            missing_row.verify()

        replacement = SQLiteEventStore(self.root / "missing-events" / "events.sqlite3")
        connection = sqlite3.connect(replacement.path)
        connection.execute("DROP TABLE events")
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "read session events"):
            replacement.load(SESSION_ID)
        with self.assertRaisesRegex(CorruptJournal, "list journal sessions"):
            replacement.session_ids()
        with self.assertRaisesRegex(CorruptJournal, "missing table: events"):
            replacement.verify()


if __name__ == "__main__":
    unittest.main()
