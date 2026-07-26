from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from gworker.codec import encode_event
from gworker.domain import (
    AbandonReason,
    BreakCompleted,
    BreakStarted,
    FocusCompleted,
    FocusStarted,
    SessionAbandoned,
    SessionPhase,
    SessionPlanned,
)
from gworker.policy import (
    DurationFit,
    EnergyLevel,
    FocusContext,
    HierarchicalSoftmaxUCB,
    PolicyConfig,
    TaskKind,
)
from gworker.storage import (
    CorruptJournal,
    FocusSessionLink,
    JournalConflict,
    JournalError,
    SQLiteEventStore,
)

BASE_TIME = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)


def identifier(namespace: int, suffix: int) -> UUID:
    return UUID(f"018f4f69-e7a2-7f84-8c2d-{namespace:04d}{suffix:08d}")


def decision_id(suffix: int) -> UUID:
    return identifier(1000, suffix)


def session_id(suffix: int) -> UUID:
    return identifier(2000, suffix)


def event_id(suffix: int) -> UUID:
    return identifier(3000, suffix)


def context() -> FocusContext:
    return FocusContext(
        task_kind=TaskKind.DEEP_WORK,
        energy=EnergyLevel.HIGH,
        available_seconds=3_600,
    )


def planned_event(
    recommendation,
    suffix: int,
    *,
    objective: str = "Exercise explicit policy provenance",
    policy_id: str | None = None,
    focus_delta: int = 0,
) -> SessionPlanned:
    return SessionPlanned(
        event_id=event_id(suffix * 10 + 1),
        session_id=session_id(suffix),
        sequence=1,
        occurred_at=BASE_TIME + timedelta(minutes=suffix),
        objective=objective,
        target_focus_seconds=(recommendation.template.focus_seconds + focus_delta),
        target_break_seconds=recommendation.template.break_seconds,
        policy_id=policy_id or recommendation.policy_id,
    )


def event_metadata(suffix: int, sequence: int) -> dict[str, object]:
    return {
        "event_id": event_id(suffix * 10 + sequence),
        "session_id": session_id(suffix),
        "sequence": sequence,
        "occurred_at": BASE_TIME + timedelta(minutes=suffix, seconds=sequence),
    }


class FocusSessionLinkageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = self.root / "private" / "events.sqlite3"
        self.store = SQLiteEventStore(self.database)
        self.policy = HierarchicalSoftmaxUCB()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def persist_pair(
        self,
        suffix: int,
        *,
        seed: int | None = None,
        objective: str = "Exercise explicit policy provenance",
        policy=None,
        focus_delta: int = 0,
        plan_policy_id: str | None = None,
    ):
        selected_policy = policy or self.policy
        recommendation = self.store.recommend(
            selected_policy,
            context(),
            decision_id=decision_id(suffix),
            rng_seed=seed if seed is not None else suffix,
        )
        event = planned_event(
            recommendation,
            suffix,
            objective=objective,
            policy_id=plan_policy_id,
            focus_delta=focus_delta,
        )
        self.store.append(event)
        return recommendation, event

    def test_new_journal_has_exact_schema_v3_link_table(self) -> None:
        connection = sqlite3.connect(self.database)
        version = connection.execute(
            "SELECT value FROM journal_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
        columns = tuple(
            row[1]
            for row in connection.execute("PRAGMA table_info(focus_session_links)")
        )
        sql = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'focus_session_links'
            """
        ).fetchone()[0]
        foreign_keys = {
            (row[2], row[3], row[4], row[5], row[6])
            for row in connection.execute(
                "PRAGMA foreign_key_list(focus_session_links)"
            )
        }
        connection.close()

        self.assertEqual(version, "3")
        self.assertEqual(columns, ("decision_id", "planned_event_id"))
        self.assertIn("WITHOUT ROWID", sql)
        self.assertNotIn("session_id", sql)
        self.assertEqual(
            foreign_keys,
            {
                (
                    "policy_decisions",
                    "decision_id",
                    "decision_id",
                    "RESTRICT",
                    "RESTRICT",
                ),
                (
                    "events",
                    "planned_event_id",
                    "event_id",
                    "RESTRICT",
                    "RESTRICT",
                ),
            },
        )

    def test_exact_v1_to_v3_migration_preserves_event_bytes(self) -> None:
        legacy = self.root / "legacy-v1" / "events.sqlite3"
        legacy.parent.mkdir(mode=0o700)
        event = SessionPlanned(
            event_id=event_id(1),
            session_id=session_id(1),
            sequence=1,
            occurred_at=BASE_TIME,
            objective="Preserve canonical event bytes",
            target_focus_seconds=1_500,
            target_break_seconds=300,
            policy_id="fixed-baseline-v1",
        )
        encoded = encode_event(event)
        connection = sqlite3.connect(legacy)
        connection.executescript(
            """
            CREATE TABLE journal_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE events (
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL CHECK (
                    typeof(sequence) = 'integer' AND sequence > 0
                ),
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                event_json TEXT NOT NULL,
                PRIMARY KEY (session_id, sequence)
            ) WITHOUT ROWID;
            INSERT INTO journal_metadata (key, value)
            VALUES ('schema_version', '1');
            """
        )
        connection.execute(
            """
            INSERT INTO events (
                session_id, sequence, event_id, event_type, event_json
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
        connection.commit()
        connection.close()
        legacy.chmod(0o600)

        migrated = SQLiteEventStore(legacy)

        connection = sqlite3.connect(legacy)
        version = connection.execute(
            "SELECT value FROM journal_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
        preserved = connection.execute("SELECT event_json FROM events").fetchone()[0]
        link_count = connection.execute(
            "SELECT COUNT(*) FROM focus_session_links"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(version, "3")
        self.assertEqual(preserved, encoded)
        self.assertEqual(migrated.load(event.session_id), [event])
        self.assertEqual(link_count, 0)

    def test_exact_v2_to_v3_migration_preserves_rows_without_backfill(self) -> None:
        recommendation, event = self.persist_pair(1, seed=11)
        connection = sqlite3.connect(self.database)
        before = (
            connection.execute("SELECT * FROM policy_decisions").fetchall(),
            connection.execute("SELECT * FROM events").fetchall(),
        )
        connection.executescript(
            """
            DROP TABLE focus_session_links;
            UPDATE journal_metadata
            SET value = '2'
            WHERE key = 'schema_version';
            """
        )
        connection.commit()
        connection.close()

        migrated = SQLiteEventStore(self.database)

        connection = sqlite3.connect(self.database)
        after = (
            connection.execute("SELECT * FROM policy_decisions").fetchall(),
            connection.execute("SELECT * FROM events").fetchall(),
        )
        version = connection.execute(
            "SELECT value FROM journal_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
        link_count = connection.execute(
            "SELECT COUNT(*) FROM focus_session_links"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(version, "3")
        self.assertEqual(after, before)
        self.assertEqual(link_count, 0)
        self.assertEqual(migrated.load(event.session_id), [event])
        self.assertEqual(recommendation.decision_sequence, 1)

    def test_v2_migration_failure_rolls_back_to_exact_v2(self) -> None:
        connection = sqlite3.connect(self.database)
        connection.executescript(
            """
            DROP TABLE focus_session_links;
            UPDATE journal_metadata
            SET value = '2'
            WHERE key = 'schema_version';
            """
        )
        connection.commit()
        connection.close()

        def fail_after_create(
            connection: sqlite3.Connection,
            statements: tuple[str, ...],
        ) -> None:
            connection.execute(statements[0])
            raise sqlite3.OperationalError("injected migration failure")

        with (
            patch.object(
                SQLiteEventStore,
                "_create_statements",
                side_effect=fail_after_create,
            ),
            self.assertRaisesRegex(JournalError, "initialize"),
        ):
            SQLiteEventStore(self.database)

        connection = sqlite3.connect(self.database)
        version = connection.execute(
            "SELECT value FROM journal_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        connection.close()
        self.assertEqual(version, "2")
        self.assertNotIn("focus_session_links", tables)
        SQLiteEventStore(self.database)

    def test_unknown_hybrid_and_altered_v2_schemas_fail_closed(self) -> None:
        cases = (
            (
                "hybrid",
                """
                UPDATE journal_metadata SET value = '2'
                WHERE key = 'schema_version'
                """,
                "version 2 journal contains unexpected tables",
            ),
            (
                "altered",
                """
                DROP TABLE focus_session_links;
                DROP INDEX policy_history_reviewed_decision;
                CREATE INDEX policy_history_reviewed_decision
                ON policy_decision_history(reviewed_decision_id DESC);
                UPDATE journal_metadata SET value = '2'
                WHERE key = 'schema_version'
                """,
                "unexpected definition",
            ),
        )
        for name, mutation, expected in cases:
            with self.subTest(name=name):
                path = self.root / name / "events.sqlite3"
                SQLiteEventStore(path)
                connection = sqlite3.connect(path)
                connection.executescript(mutation)
                connection.commit()
                connection.close()
                with self.assertRaisesRegex(CorruptJournal, expected):
                    SQLiteEventStore(path)

    def test_successful_link_reopens_and_derives_session_identity(self) -> None:
        recommendation, event = self.persist_pair(1, seed=73)

        link = self.store.link_focus_session(
            self.policy,
            recommendation.decision_id,
            session_id=event.session_id,
        )

        self.assertIsInstance(link, FocusSessionLink)
        self.assertEqual(
            link,
            FocusSessionLink(
                decision_id=recommendation.decision_id,
                decision_sequence=recommendation.decision_sequence,
                session_id=event.session_id,
                planned_event_id=event.event_id,
                policy_id=recommendation.policy_id,
                template_id=recommendation.template.template_id,
            ),
        )
        reopened = SQLiteEventStore(self.database)
        self.assertEqual(
            reopened.focus_session_link(
                self.policy,
                session_id=event.session_id,
            ),
            link,
        )
        self.assertEqual(reopened.replay(event.session_id).phase, SessionPhase.PLANNED)
        self.assertEqual(reopened.verify().event_count, 1)
        self.assertEqual(
            reopened.verify_policy_history(self.policy).decision_count,
            1,
        )

    def test_unlinked_decision_and_session_remain_valid(self) -> None:
        _, event = self.persist_pair(1)

        self.assertIsNone(
            self.store.focus_session_link(
                self.policy,
                session_id=event.session_id,
            )
        )
        self.assertEqual(self.store.verify().session_count, 1)
        self.assertEqual(
            self.store.verify_policy_history(self.policy).review_count,
            0,
        )

    def test_link_requires_existing_unreviewed_decision_and_revision_one_plan(
        self,
    ) -> None:
        recommendation, event = self.persist_pair(1)
        with self.assertRaisesRegex(JournalConflict, "does not exist"):
            self.store.link_focus_session(
                self.policy,
                decision_id(99),
                session_id=event.session_id,
            )
        with self.assertRaisesRegex(JournalConflict, "does not exist"):
            self.store.link_focus_session(
                self.policy,
                recommendation.decision_id,
                session_id=session_id(99),
            )

        self.store.append(FocusStarted(**event_metadata(1, 2)))
        with self.assertRaisesRegex(JournalConflict, "unstarted revision-1"):
            self.store.link_focus_session(
                self.policy,
                recommendation.decision_id,
                session_id=event.session_id,
            )

        second, second_event = self.persist_pair(2)
        self.store.record_review(
            self.policy,
            second.decision_id,
            fit=DurationFit.JUST_RIGHT,
            objective_completed=True,
        )
        with self.assertRaisesRegex(JournalConflict, "already reviewed"):
            self.store.link_focus_session(
                self.policy,
                second.decision_id,
                session_id=second_event.session_id,
            )

    def test_policy_duration_and_exact_policy_mismatches_are_conflicts(self) -> None:
        recommendation, event = self.persist_pair(1, focus_delta=1)
        with self.assertRaisesRegex(JournalConflict, "durations"):
            self.store.link_focus_session(
                self.policy,
                recommendation.decision_id,
                session_id=event.session_id,
            )

        other_policy = HierarchicalSoftmaxUCB(config=PolicyConfig(prior_mean=0.6))
        other, other_event = self.persist_pair(2, policy=other_policy)
        with self.assertRaisesRegex(JournalConflict, "different policy"):
            self.store.link_focus_session(
                self.policy,
                other.decision_id,
                session_id=other_event.session_id,
            )

        own, foreign_event = self.persist_pair(
            3,
            plan_policy_id=other_policy.policy_id,
        )
        with self.assertRaisesRegex(JournalConflict, "different policy"):
            self.store.link_focus_session(
                self.policy,
                own.decision_id,
                session_id=foreign_event.session_id,
            )

    def test_one_to_one_consumption_rejects_both_duplicate_sides(self) -> None:
        first, first_event = self.persist_pair(1)
        second, second_event = self.persist_pair(2)
        self.store.link_focus_session(
            self.policy,
            first.decision_id,
            session_id=first_event.session_id,
        )

        with self.assertRaisesRegex(JournalConflict, "decision is already linked"):
            self.store.link_focus_session(
                self.policy,
                first.decision_id,
                session_id=second_event.session_id,
            )
        with self.assertRaisesRegex(JournalConflict, "session is already linked"):
            self.store.link_focus_session(
                self.policy,
                second.decision_id,
                session_id=first_event.session_id,
            )

        connection = sqlite3.connect(self.database)
        rows = connection.execute(
            "SELECT decision_id, planned_event_id FROM focus_session_links"
        ).fetchall()
        connection.close()
        self.assertEqual(rows, [(str(first.decision_id), str(first_event.event_id))])

    def test_injected_insert_failure_rolls_back_without_partial_link(self) -> None:
        recommendation, event = self.persist_pair(1)
        original = SQLiteEventStore._insert_focus_link

        def insert_then_fail(
            connection: sqlite3.Connection,
            stored_decision_id: str,
            planned_event_id: str,
        ) -> None:
            original(connection, stored_decision_id, planned_event_id)
            raise sqlite3.OperationalError("injected link failure")

        with (
            patch.object(
                SQLiteEventStore,
                "_insert_focus_link",
                side_effect=insert_then_fail,
            ),
            self.assertRaisesRegex(JournalError, "persist focus-session link"),
        ):
            self.store.link_focus_session(
                self.policy,
                recommendation.decision_id,
                session_id=event.session_id,
            )

        connection = sqlite3.connect(self.database)
        count = connection.execute(
            "SELECT COUNT(*) FROM focus_session_links"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(count, 0)
        reopened = SQLiteEventStore(self.database)
        self.assertIsNone(
            reopened.focus_session_link(
                self.policy,
                session_id=event.session_id,
            )
        )

    def test_session_progress_never_infers_a_review(self) -> None:
        recommendation, event = self.persist_pair(1)
        self.store.link_focus_session(
            self.policy,
            recommendation.decision_id,
            session_id=event.session_id,
        )
        self.store.append(FocusStarted(**event_metadata(1, 2)))
        self.store.append(
            FocusCompleted(
                **event_metadata(1, 3),
                elapsed_seconds=recommendation.template.focus_seconds,
            )
        )
        self.store.append(BreakStarted(**event_metadata(1, 4)))
        self.store.append(
            BreakCompleted(
                **event_metadata(1, 5),
                elapsed_seconds=recommendation.template.break_seconds,
            )
        )
        second = self.store.recommend(
            self.policy,
            context(),
            decision_id=decision_id(2),
            rng_seed=2,
        )

        verification = self.store.verify_policy_history(self.policy)
        self.assertEqual(verification.review_count, 0)
        self.assertEqual(verification.history_edge_count, 0)
        self.assertEqual(second.decision_sequence, 2)
        self.assertEqual(
            self.store.focus_session_link(
                self.policy,
                session_id=event.session_id,
            ).decision_id,
            recommendation.decision_id,
        )

    def test_abandonment_replay_and_verification_never_infer_a_review(self) -> None:
        recommendation, event = self.persist_pair(1)
        link = self.store.link_focus_session(
            self.policy,
            recommendation.decision_id,
            session_id=event.session_id,
        )
        self.store.append(
            SessionAbandoned(
                **event_metadata(1, 2),
                reason=AbandonReason.PRIORITY_CHANGED,
            )
        )

        reopened = SQLiteEventStore(self.database)
        self.assertEqual(
            reopened.replay(event.session_id).phase,
            SessionPhase.ABANDONED,
        )
        self.assertEqual(
            reopened.focus_session_link(
                self.policy,
                session_id=event.session_id,
            ),
            link,
        )
        self.assertEqual(reopened.verify().event_count, 2)
        verification = reopened.verify_policy_history(self.policy)
        self.assertEqual(verification.review_count, 0)
        self.assertEqual(verification.history_edge_count, 0)

    def test_explicit_review_after_link_preserves_lineage(self) -> None:
        recommendation, event = self.persist_pair(1, seed=17)
        link = self.store.link_focus_session(
            self.policy,
            recommendation.decision_id,
            session_id=event.session_id,
        )
        reviewed = self.store.record_review(
            self.policy,
            recommendation.decision_id,
            fit=DurationFit.TOO_SHORT,
            objective_completed=True,
        )
        second = self.store.recommend(
            self.policy,
            context(),
            decision_id=decision_id(2),
            rng_seed=29,
        )

        expected = self.policy.recommend_seeded(
            context(),
            (reviewed,),
            decision_id=second.decision_id,
            decision_sequence=2,
            rng_seed=29,
        )
        self.assertEqual(second, expected)
        self.assertEqual(
            self.store.focus_session_link(
                self.policy,
                session_id=event.session_id,
            ),
            link,
        )
        verification = self.store.verify_policy_history(self.policy)
        self.assertEqual(verification.review_count, 1)
        self.assertEqual(verification.history_edge_count, 1)

    def test_link_scalar_and_parent_corruption_fail_closed(self) -> None:
        cases = ("noncanonical", "missing-parent", "wrong-event-id")
        for index, mutation in enumerate(cases, start=1):
            with self.subTest(mutation=mutation):
                path = self.root / f"corrupt-{mutation}" / "events.sqlite3"
                store = SQLiteEventStore(path)
                policy = HierarchicalSoftmaxUCB()
                recommendation = store.recommend(
                    policy,
                    context(),
                    decision_id=decision_id(index),
                    rng_seed=index,
                )
                event = planned_event(recommendation, index)
                store.append(event)
                store.link_focus_session(
                    policy,
                    recommendation.decision_id,
                    session_id=event.session_id,
                )
                connection = sqlite3.connect(path)
                connection.execute("PRAGMA foreign_keys = OFF")
                if mutation == "noncanonical":
                    uppercase = str(event.event_id).upper()
                    connection.execute(
                        "UPDATE events SET event_id = ?",
                        (uppercase,),
                    )
                    connection.execute(
                        "UPDATE focus_session_links SET planned_event_id = ?",
                        (uppercase,),
                    )
                    expected = "link planned_event_id is not canonical"
                elif mutation == "missing-parent":
                    connection.execute(
                        "DELETE FROM events WHERE event_id = ?",
                        (str(event.event_id),),
                    )
                    expected = "foreign_key_check"
                else:
                    replacement = str(event_id(9000 + index))
                    connection.execute(
                        "UPDATE events SET event_id = ?",
                        (replacement,),
                    )
                    connection.execute(
                        "UPDATE focus_session_links SET planned_event_id = ?",
                        (replacement,),
                    )
                    expected = "linked planned event is invalid"
                connection.commit()
                connection.close()

                with self.assertRaisesRegex(CorruptJournal, expected):
                    store.verify_policy_history(policy)

    def test_linked_event_type_sequence_policy_and_duration_corruption_fail_closed(
        self,
    ) -> None:
        mutations = ("type", "sequence", "policy", "duration")
        for index, mutation in enumerate(mutations, start=1):
            with self.subTest(mutation=mutation):
                path = self.root / f"event-{mutation}" / "events.sqlite3"
                store = SQLiteEventStore(path)
                policy = HierarchicalSoftmaxUCB()
                recommendation = store.recommend(
                    policy,
                    context(),
                    decision_id=decision_id(100 + index),
                    rng_seed=index,
                )
                event = planned_event(recommendation, 100 + index)
                store.append(event)
                store.link_focus_session(
                    policy,
                    recommendation.decision_id,
                    session_id=event.session_id,
                )
                connection = sqlite3.connect(path)
                if mutation == "type":
                    connection.execute("UPDATE events SET event_type = 'focus_started'")
                    expected = "linked planned event is invalid"
                elif mutation == "sequence":
                    changed = SessionPlanned(
                        event_id=event.event_id,
                        session_id=event.session_id,
                        sequence=2,
                        occurred_at=event.occurred_at,
                        objective=event.objective,
                        target_focus_seconds=event.target_focus_seconds,
                        target_break_seconds=event.target_break_seconds,
                        policy_id=event.policy_id,
                    )
                    connection.execute(
                        "UPDATE events SET sequence = 2, event_json = ?",
                        (encode_event(changed),),
                    )
                    expected = "sequence-1"
                elif mutation == "policy":
                    changed = SessionPlanned(
                        event_id=event.event_id,
                        session_id=event.session_id,
                        sequence=1,
                        occurred_at=event.occurred_at,
                        objective=event.objective,
                        target_focus_seconds=event.target_focus_seconds,
                        target_break_seconds=event.target_break_seconds,
                        policy_id="foreign-policy-v1",
                    )
                    connection.execute(
                        "UPDATE events SET event_json = ?",
                        (encode_event(changed),),
                    )
                    expected = "policy_id disagrees"
                else:
                    changed = SessionPlanned(
                        event_id=event.event_id,
                        session_id=event.session_id,
                        sequence=1,
                        occurred_at=event.occurred_at,
                        objective=event.objective,
                        target_focus_seconds=event.target_focus_seconds + 1,
                        target_break_seconds=event.target_break_seconds,
                        policy_id=event.policy_id,
                    )
                    connection.execute(
                        "UPDATE events SET event_json = ?",
                        (encode_event(changed),),
                    )
                    expected = "durations disagree"
                connection.commit()
                connection.close()

                with self.assertRaisesRegex(CorruptJournal, expected):
                    store.verify_policy_history(policy)

    def test_link_api_validates_exact_policy_and_uuid_inputs_before_mutation(
        self,
    ) -> None:
        recommendation, event = self.persist_pair(1)

        class SubclassedPolicy(HierarchicalSoftmaxUCB):
            pass

        with self.assertRaisesRegex(TypeError, "exact"):
            self.store.link_focus_session(
                SubclassedPolicy(),
                recommendation.decision_id,
                session_id=event.session_id,
            )
        with self.assertRaisesRegex(TypeError, "decision_id must be a UUID"):
            self.store.link_focus_session(
                self.policy,
                "not-a-uuid",  # type: ignore[arg-type]
                session_id=event.session_id,
            )
        with self.assertRaisesRegex(ValueError, "session_id must not be the nil UUID"):
            self.store.link_focus_session(
                self.policy,
                recommendation.decision_id,
                session_id=UUID(int=0),
            )
        connection = sqlite3.connect(self.database)
        count = connection.execute(
            "SELECT COUNT(*) FROM focus_session_links"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(count, 0)

    def test_competing_links_consume_one_decision_once(self) -> None:
        recommendation, first_event = self.persist_pair(1)
        second_event = planned_event(recommendation, 2)
        self.store.append(second_event)
        barrier = threading.Barrier(2)
        outcomes: list[str] = []

        def link(session: UUID) -> None:
            contender = SQLiteEventStore(self.database)
            barrier.wait()
            try:
                contender.link_focus_session(
                    self.policy,
                    recommendation.decision_id,
                    session_id=session,
                )
            except JournalConflict:
                outcomes.append("conflict")
            else:
                outcomes.append("committed")

        threads = [
            threading.Thread(target=link, args=(first_event.session_id,)),
            threading.Thread(target=link, args=(second_event.session_id,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(sorted(outcomes), ["committed", "conflict"])
        connection = sqlite3.connect(self.database)
        count = connection.execute(
            "SELECT COUNT(*) FROM focus_session_links"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(count, 1)

    def test_link_serializes_with_start_and_review(self) -> None:
        recommendation, event = self.persist_pair(1)
        barrier = threading.Barrier(2)
        outcomes: list[str] = []

        def link() -> None:
            contender = SQLiteEventStore(self.database)
            barrier.wait()
            try:
                contender.link_focus_session(
                    self.policy,
                    recommendation.decision_id,
                    session_id=event.session_id,
                )
            except JournalConflict:
                outcomes.append("link-conflict")
            else:
                outcomes.append("link-committed")

        def start() -> None:
            contender = SQLiteEventStore(self.database)
            barrier.wait()
            contender.append(FocusStarted(**event_metadata(1, 2)))
            outcomes.append("start-committed")

        threads = [threading.Thread(target=link), threading.Thread(target=start)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertIn(
            sorted(outcomes),
            (
                ["link-committed", "start-committed"],
                ["link-conflict", "start-committed"],
            ),
        )
        self.assertEqual(self.store.replay(event.session_id).revision, 2)
        connection = sqlite3.connect(self.database)
        link_count = connection.execute(
            "SELECT COUNT(*) FROM focus_session_links"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(link_count, int("link-committed" in outcomes))

        second, second_event = self.persist_pair(2)
        review_barrier = threading.Barrier(2)
        review_outcomes: list[str] = []

        def link_second() -> None:
            contender = SQLiteEventStore(self.database)
            review_barrier.wait()
            try:
                contender.link_focus_session(
                    self.policy,
                    second.decision_id,
                    session_id=second_event.session_id,
                )
            except JournalConflict:
                review_outcomes.append("link-conflict")
            else:
                review_outcomes.append("link-committed")

        def review_second() -> None:
            contender = SQLiteEventStore(self.database)
            review_barrier.wait()
            contender.record_review(
                self.policy,
                second.decision_id,
                fit=DurationFit.JUST_RIGHT,
                objective_completed=True,
            )
            review_outcomes.append("review-committed")

        threads = [
            threading.Thread(target=link_second),
            threading.Thread(target=review_second),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertIn(
            sorted(review_outcomes),
            (
                ["link-committed", "review-committed"],
                ["link-conflict", "review-committed"],
            ),
        )
        self.assertEqual(
            self.store.verify_policy_history(self.policy).review_count,
            1,
        )

    def test_active_writer_lock_maps_to_path_private_conflict(self) -> None:
        recommendation, event = self.persist_pair(1)
        blocker = sqlite3.connect(self.database, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaisesRegex(
                JournalConflict, "locked by a writer"
            ) as caught:
                self.store.link_focus_session(
                    self.policy,
                    recommendation.decision_id,
                    session_id=event.session_id,
                )
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        message = str(caught.exception)
        self.assertNotIn(str(self.database), message)
        self.assertNotIn(str(event.session_id), message)

    def test_errors_do_not_expose_objective_path_or_secret_like_text(self) -> None:
        secret = "credential-marker-123456789"
        recommendation, event = self.persist_pair(
            1,
            objective=f"Private objective {secret}",
            focus_delta=1,
        )
        with self.assertRaises(JournalConflict) as caught:
            self.store.link_focus_session(
                self.policy,
                recommendation.decision_id,
                session_id=event.session_id,
            )
        message = str(caught.exception)
        self.assertNotIn(secret, message)
        self.assertNotIn(event.objective, message)
        self.assertNotIn(str(self.database), message)


if __name__ == "__main__":
    unittest.main()
