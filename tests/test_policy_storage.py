from __future__ import annotations

import hashlib
import sqlite3
import stat
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from gworker.codec import encode_event
from gworker.domain import SessionPlanned
from gworker.policy import (
    MAX_RNG_SEED,
    DurationFit,
    EnergyLevel,
    FocusContext,
    HierarchicalSoftmaxUCB,
    PolicyConfig,
    PolicyInputError,
    TaskKind,
)
from gworker.storage import (
    CorruptJournal,
    JournalConflict,
    JournalError,
    SQLiteEventStore,
)

POLICY_DECISION_PREFIX = "018f4f69-e7a2-7f84-8c2d"
SESSION_ID = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e0001")
EVENT_ID = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e0002")


def decision_id(suffix: int) -> UUID:
    return UUID(f"{POLICY_DECISION_PREFIX}-{suffix:012d}")


def context(
    *,
    task_kind: TaskKind = TaskKind.DEEP_WORK,
    energy: EnergyLevel = EnergyLevel.HIGH,
) -> FocusContext:
    return FocusContext(
        task_kind=task_kind,
        energy=energy,
        available_seconds=3_600,
    )


class SQLitePolicyStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = self.root / "private" / "events.sqlite3"
        self.store = SQLiteEventStore(self.database)
        self.policy = HierarchicalSoftmaxUCB()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def recommend(
        self,
        suffix: int,
        *,
        seed: int,
        recommendation_context: FocusContext | None = None,
    ):
        return self.store.recommend(
            self.policy,
            recommendation_context or context(),
            decision_id=decision_id(suffix),
            rng_seed=seed,
        )

    def review(
        self,
        suffix: int,
        *,
        fit: DurationFit = DurationFit.JUST_RIGHT,
        completed: bool = True,
    ):
        return self.store.record_review(
            self.policy,
            decision_id(suffix),
            fit=fit,
            objective_completed=completed,
        )

    def test_migrates_v1_transactionally_and_preserves_events(self) -> None:
        legacy_path = self.root / "legacy" / "events.sqlite3"
        legacy_path.parent.mkdir(mode=0o700)
        event = SessionPlanned(
            event_id=EVENT_ID,
            session_id=SESSION_ID,
            sequence=1,
            occurred_at=datetime(2026, 7, 25, 9, 0, tzinfo=UTC),
            objective="Preserve this event during migration",
            target_focus_seconds=1_500,
            target_break_seconds=300,
            policy_id="fixed-baseline-v1",
        )
        connection = sqlite3.connect(legacy_path)
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
                encode_event(event),
            ),
        )
        connection.commit()
        connection.close()
        legacy_path.chmod(0o600)

        migrated = SQLiteEventStore(legacy_path)

        self.assertEqual(migrated.load(SESSION_ID), [event])
        connection = sqlite3.connect(legacy_path)
        version = connection.execute(
            """
            SELECT value FROM journal_metadata WHERE key = 'schema_version'
            """
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
        self.assertEqual(version, "3")
        self.assertEqual(
            tables,
            {
                "events",
                "focus_session_links",
                "journal_metadata",
                "policy_decision_history",
                "policy_decisions",
                "policy_reviews",
            },
        )
        self.assertEqual(
            stat.S_IMODE(legacy_path.stat().st_mode),
            0o600,
        )

    def test_failed_migration_rolls_back_and_can_be_reopened(self) -> None:
        legacy_path = self.root / "migration-rollback" / "events.sqlite3"
        legacy_path.parent.mkdir(mode=0o700)
        connection = sqlite3.connect(legacy_path)
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
        connection.commit()
        connection.close()
        legacy_path.chmod(0o600)

        def fail_after_first_statement(
            connection: sqlite3.Connection,
            statements: tuple[str, ...],
        ) -> None:
            connection.execute(statements[0])
            raise sqlite3.OperationalError("injected migration failure")

        with (
            patch.object(
                SQLiteEventStore,
                "_create_statements",
                side_effect=fail_after_first_statement,
            ),
            self.assertRaisesRegex(JournalError, "initialize"),
        ):
            SQLiteEventStore(legacy_path)

        connection = sqlite3.connect(legacy_path)
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
        self.assertEqual(version, "1")
        self.assertEqual(tables, {"events", "journal_metadata"})
        reopened = SQLiteEventStore(legacy_path)
        self.assertEqual(reopened.verify_policy_history(self.policy).decision_count, 0)

    def test_fixed_seed_review_survives_reopen_and_next_decision_is_exact(
        self,
    ) -> None:
        first = self.recommend(1, seed=73)
        expected_first = self.policy.recommend_seeded(
            context(),
            (),
            decision_id=decision_id(1),
            decision_sequence=1,
            rng_seed=73,
        )
        self.assertEqual(first, expected_first)

        reopened = SQLiteEventStore(self.database)
        first_review = reopened.record_review(
            self.policy,
            first.decision_id,
            fit=DurationFit.TOO_SHORT,
            objective_completed=True,
        )
        second = reopened.recommend(
            self.policy,
            context(),
            decision_id=decision_id(2),
            rng_seed=91,
        )
        expected_second = self.policy.recommend_seeded(
            context(),
            (first_review,),
            decision_id=decision_id(2),
            decision_sequence=2,
            rng_seed=91,
        )

        self.assertEqual(second, expected_second)
        reopened_again = SQLiteEventStore(self.database)
        self.assertEqual(
            reopened_again.reviewed_decisions(self.policy),
            (first_review,),
        )
        verification = reopened_again.verify_policy_history(self.policy)
        self.assertEqual(verification.decision_count, 2)
        self.assertEqual(verification.review_count, 1)
        self.assertEqual(verification.history_edge_count, 1)
        self.assertEqual(verification.sqlite_check, "ok")

    def test_delayed_review_is_visible_only_to_later_recommendations(self) -> None:
        self.recommend(1, seed=1)
        self.recommend(2, seed=2)
        first_review = self.review(1, fit=DurationFit.TOO_LONG)
        third = self.recommend(3, seed=3)

        expected = self.policy.recommend_seeded(
            context(),
            (first_review,),
            decision_id=decision_id(3),
            decision_sequence=3,
            rng_seed=3,
        )
        self.assertEqual(third, expected)

        connection = sqlite3.connect(self.database)
        edges = connection.execute(
            """
            SELECT decision_id, position, reviewed_decision_id
            FROM policy_decision_history
            ORDER BY decision_id, position
            """
        ).fetchall()
        connection.close()
        self.assertEqual(
            edges,
            [(str(decision_id(3)), 0, str(decision_id(1)))],
        )

    def test_duplicate_decision_id_rolls_back_without_sequence_gap(self) -> None:
        first = self.recommend(1, seed=1)
        with self.assertRaisesRegex(JournalConflict, "decision_id already exists"):
            self.store.recommend(
                self.policy,
                context(),
                decision_id=first.decision_id,
                rng_seed=999,
            )
        second = self.recommend(2, seed=2)
        self.assertEqual((first.decision_sequence, second.decision_sequence), (1, 2))

    def test_review_of_missing_decision_is_an_orphan_conflict(self) -> None:
        with self.assertRaisesRegex(JournalConflict, "does not exist"):
            self.review(404)
        self.assertEqual(
            self.store.verify_policy_history(self.policy).review_count,
            0,
        )

    def test_duplicate_review_is_a_conflict_and_preserves_the_first(self) -> None:
        recommendation = self.recommend(1, seed=1)
        first = self.review(1, fit=DurationFit.TOO_SHORT, completed=False)

        with self.assertRaisesRegex(JournalConflict, "already reviewed"):
            self.review(1, fit=DurationFit.TOO_LONG, completed=True)

        self.assertEqual(first.decision_id, recommendation.decision_id)
        self.assertEqual(self.store.reviewed_decisions(self.policy), (first,))

    def test_review_with_wrong_policy_is_a_conflict(self) -> None:
        other_policy = HierarchicalSoftmaxUCB(config=PolicyConfig(prior_mean=0.6))
        self.store.recommend(
            other_policy,
            context(),
            decision_id=decision_id(1),
            rng_seed=1,
        )
        with self.assertRaisesRegex(JournalConflict, "different policy"):
            self.review(1)

    def test_forged_or_subclassed_policy_fails_before_mutation(self) -> None:
        forged = HierarchicalSoftmaxUCB(config=PolicyConfig(prior_mean=0.6))
        forged.policy_id = self.policy.policy_id
        with self.assertRaisesRegex(PolicyInputError, "does not match"):
            self.store.recommend(
                forged,
                context(),
                decision_id=decision_id(1),
                rng_seed=1,
            )

        class SubclassedPolicy(HierarchicalSoftmaxUCB):
            pass

        with self.assertRaisesRegex(TypeError, "exact"):
            self.store.verify_policy_history(SubclassedPolicy())
        self.assertEqual(
            self.store.verify_policy_history(self.policy).decision_count,
            0,
        )

    def test_concurrent_writers_allocate_distinct_contiguous_sequences(self) -> None:
        barrier = threading.Barrier(2)
        outcomes: list[tuple[int, UUID]] = []
        errors: list[BaseException] = []

        def write(suffix: int) -> None:
            contender = SQLiteEventStore(self.database)
            barrier.wait()
            try:
                recommendation = contender.recommend(
                    self.policy,
                    context(),
                    decision_id=decision_id(suffix),
                    rng_seed=suffix,
                )
            except BaseException as exc:
                errors.append(exc)
            else:
                outcomes.append(
                    (
                        recommendation.decision_sequence,
                        recommendation.decision_id,
                    )
                )

        threads = [
            threading.Thread(target=write, args=(10,)),
            threading.Thread(target=write, args=(11,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(errors)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(
            sorted(sequence for sequence, _ in outcomes),
            [1, 2],
        )
        self.assertEqual(
            {identifier for _, identifier in outcomes},
            {decision_id(10), decision_id(11)},
        )

    def test_global_sequence_is_contiguous_across_distinct_policy_ids(self) -> None:
        other_policy = HierarchicalSoftmaxUCB(config=PolicyConfig(prior_mean=0.6))
        first = self.recommend(1, seed=1)
        second = self.store.recommend(
            other_policy,
            context(),
            decision_id=decision_id(2),
            rng_seed=2,
        )
        third = self.recommend(3, seed=3)

        self.assertEqual(
            (
                first.decision_sequence,
                second.decision_sequence,
                third.decision_sequence,
            ),
            (1, 2, 3),
        )
        self.assertEqual(
            self.store.verify_policy_history(self.policy).decision_count,
            2,
        )
        self.assertEqual(
            self.store.verify_policy_history(other_policy).decision_count,
            1,
        )

    def test_frozen_history_uses_only_the_configured_review_window(self) -> None:
        policy = HierarchicalSoftmaxUCB(
            config=PolicyConfig(
                minimum_exact_reviews=2,
                minimum_task_reviews=2,
                window_size=2,
            )
        )
        for suffix in range(1, 4):
            recommendation = self.store.recommend(
                policy,
                context(),
                decision_id=decision_id(suffix),
                rng_seed=suffix,
            )
            self.store.record_review(
                policy,
                recommendation.decision_id,
                fit=DurationFit.JUST_RIGHT,
                objective_completed=True,
            )
        fourth = self.store.recommend(
            policy,
            context(),
            decision_id=decision_id(4),
            rng_seed=4,
        )

        connection = sqlite3.connect(self.database)
        history = connection.execute(
            """
            SELECT position, reviewed_decision_id
            FROM policy_decision_history
            WHERE decision_id = ?
            ORDER BY position
            """,
            (str(fourth.decision_id),),
        ).fetchall()
        connection.close()
        self.assertEqual(
            history,
            [
                (0, str(decision_id(2))),
                (1, str(decision_id(3))),
            ],
        )

    def test_oversized_stored_history_fails_closed(self) -> None:
        policy = HierarchicalSoftmaxUCB(
            config=PolicyConfig(
                minimum_exact_reviews=2,
                minimum_task_reviews=2,
                window_size=2,
            )
        )
        for suffix in range(1, 4):
            recommendation = self.store.recommend(
                policy,
                context(),
                decision_id=decision_id(suffix),
                rng_seed=suffix,
            )
            self.store.record_review(
                policy,
                recommendation.decision_id,
                fit=DurationFit.JUST_RIGHT,
                objective_completed=True,
            )
        fourth = self.store.recommend(
            policy,
            context(),
            decision_id=decision_id(4),
            rng_seed=4,
        )
        history_ids = tuple(decision_id(suffix) for suffix in range(1, 4))
        digest = hashlib.sha256()
        for position, identifier in enumerate(history_ids):
            digest.update(position.to_bytes(8, "big"))
            digest.update(identifier.bytes)

        connection = sqlite3.connect(self.database)
        connection.execute(
            "DELETE FROM policy_decision_history WHERE decision_id = ?",
            (str(fourth.decision_id),),
        )
        connection.executemany(
            """
            INSERT INTO policy_decision_history (
                decision_id, position, reviewed_decision_id
            )
            VALUES (?, ?, ?)
            """,
            (
                (str(fourth.decision_id), position, str(identifier))
                for position, identifier in enumerate(history_ids)
            ),
        )
        connection.execute(
            """
            UPDATE policy_decisions
            SET history_count = 3, history_sha256 = ?
            WHERE decision_id = ?
            """,
            (digest.hexdigest(), str(fourth.decision_id)),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "configured window"):
            self.store.verify_policy_history(policy)

    def test_propensity_tampering_fails_closed(self) -> None:
        self.recommend(1, seed=17)
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            UPDATE policy_decisions
            SET propensity_hex = '0x1.0000000000000p-1'
            WHERE decision_id = ?
            """,
            (str(decision_id(1)),),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "policy recomputation"):
            self.store.verify_policy_history(self.policy)
        with self.assertRaisesRegex(CorruptJournal, "policy recomputation"):
            self.review(1)

    def test_noncanonical_and_overflowing_propensity_fail_closed(self) -> None:
        self.recommend(1, seed=17)
        connection = sqlite3.connect(self.database)
        original = connection.execute(
            """
            SELECT propensity_hex
            FROM policy_decisions
            WHERE decision_id = ?
            """,
            (str(decision_id(1)),),
        ).fetchone()[0]
        for tampered in ("0x1p-2", "0x1p+999999999"):
            with self.subTest(tampered=tampered):
                connection.execute(
                    """
                    UPDATE policy_decisions
                    SET propensity_hex = ?
                    WHERE decision_id = ?
                    """,
                    (tampered, str(decision_id(1))),
                )
                connection.commit()
                with self.assertRaisesRegex(
                    CorruptJournal,
                    "propensity_hex",
                ):
                    self.store.verify_policy_history(self.policy)
                connection.execute(
                    """
                    UPDATE policy_decisions
                    SET propensity_hex = ?
                    WHERE decision_id = ?
                    """,
                    (original, str(decision_id(1))),
                )
                connection.commit()
        connection.close()

    def test_decision_scalar_tampering_fails_closed(self) -> None:
        cases = (
            (
                "UPDATE policy_decisions SET decision_id = ?",
                sqlite3.Binary(b"not-text"),
                "decision_id is not text",
            ),
            (
                "UPDATE policy_decisions SET decision_id = ?",
                "not-a-uuid",
                "decision_id is not a UUID",
            ),
            (
                "UPDATE policy_decisions SET decision_id = ?",
                str(UUID(int=0)),
                "decision_id must not be the nil UUID",
            ),
            (
                "UPDATE policy_decisions SET policy_id = ?",
                "Bad-Policy",
                "policy_id is not a canonical identifier",
            ),
            (
                "UPDATE policy_decisions SET task_kind = ?",
                sqlite3.Binary(b"deep_work"),
                "task_kind is not text",
            ),
            (
                "UPDATE policy_decisions SET energy = ?",
                sqlite3.Binary(b"high"),
                "energy is not text",
            ),
            (
                "UPDATE policy_decisions SET available_seconds = ?",
                0,
                "available_seconds is out of bounds",
            ),
            (
                "UPDATE policy_decisions SET previous_focus_seconds = ?",
                0,
                "previous_focus_seconds is out of bounds",
            ),
            (
                "UPDATE policy_decisions SET template_id = ?",
                "Bad-Template",
                "template_id is not a canonical identifier",
            ),
            (
                "UPDATE policy_decisions SET propensity_hex = ?",
                sqlite3.Binary(b"0x1.0p-1"),
                "propensity_hex is not text",
            ),
            (
                "UPDATE policy_decisions SET propensity_hex = ?",
                "nan",
                "propensity_hex is not canonical",
            ),
            (
                "UPDATE policy_decisions SET history_sha256 = ?",
                "g" * 64,
                "history_sha256 is not canonical",
            ),
        )

        for index, (statement, tampered, expected) in enumerate(cases):
            with self.subTest(column=statement, value=tampered):
                path = self.root / f"scalar-{index}" / "events.sqlite3"
                store = SQLiteEventStore(path)
                policy = HierarchicalSoftmaxUCB()
                store.recommend(
                    policy,
                    context(),
                    decision_id=decision_id(100 + index),
                    rng_seed=index,
                )
                connection = sqlite3.connect(path)
                connection.execute(statement, (tampered,))
                connection.commit()
                connection.close()

                with self.assertRaisesRegex(CorruptJournal, expected):
                    store.verify_policy_history(policy)

    def test_review_scalar_tampering_fails_closed(self) -> None:
        cases = (
            (sqlite3.Binary(b"just_right"), "review fit is not text"),
            ("NOT_A_FIT", "review fit is invalid"),
        )

        for index, (tampered, expected) in enumerate(cases):
            with self.subTest(value=tampered):
                path = self.root / f"review-scalar-{index}" / "events.sqlite3"
                store = SQLiteEventStore(path)
                policy = HierarchicalSoftmaxUCB()
                recommendation = store.recommend(
                    policy,
                    context(),
                    decision_id=decision_id(200 + index),
                    rng_seed=index,
                )
                store.record_review(
                    policy,
                    recommendation.decision_id,
                    fit=DurationFit.JUST_RIGHT,
                    objective_completed=True,
                )
                connection = sqlite3.connect(path)
                connection.execute(
                    "UPDATE policy_reviews SET fit = ?",
                    (tampered,),
                )
                connection.commit()
                connection.close()

                with self.assertRaisesRegex(CorruptJournal, expected):
                    store.reviewed_decisions(policy)

    def test_check_constraint_tampering_fails_quick_check(self) -> None:
        self.recommend(1, seed=17)
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("UPDATE policy_decisions SET rng_seed = -1")
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "quick_check failed"):
            self.store.verify_policy_history(self.policy)

    def test_selected_template_tampering_fails_closed(self) -> None:
        recommendation = self.recommend(1, seed=17)
        replacement = (
            "focus-25"
            if recommendation.template.template_id != "focus-25"
            else "focus-40"
        )
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            UPDATE policy_decisions
            SET template_id = ?
            WHERE decision_id = ?
            """,
            (replacement, str(decision_id(1))),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "template_id"):
            self.store.verify_policy_history(self.policy)

    def test_noncanonical_uuid_and_context_enum_fail_closed(self) -> None:
        self.recommend(1, seed=17)
        canonical = str(decision_id(1))
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            UPDATE policy_decisions
            SET decision_id = ?
            WHERE decision_id = ?
            """,
            (canonical.upper(), canonical),
        )
        connection.commit()
        with self.assertRaisesRegex(CorruptJournal, "not canonical"):
            self.store.verify_policy_history(self.policy)
        connection.execute(
            """
            UPDATE policy_decisions
            SET decision_id = ?
            WHERE decision_id = ?
            """,
            (canonical, canonical.upper()),
        )
        connection.execute(
            """
            UPDATE policy_decisions
            SET task_kind = 'DEEP_WORK'
            WHERE decision_id = ?
            """,
            (canonical,),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "context enum"):
            self.store.verify_policy_history(self.policy)

    def test_history_edge_tampering_fails_digest_validation(self) -> None:
        self.recommend(1, seed=1)
        self.review(1)
        self.recommend(2, seed=2)
        self.review(2)
        self.recommend(3, seed=3)

        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            DELETE FROM policy_decision_history
            WHERE decision_id = ? AND position = 1
            """,
            (str(decision_id(3)),),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "history count"):
            self.store.verify_policy_history(self.policy)

    def test_reordered_history_edges_fail_closed(self) -> None:
        self.recommend(1, seed=1)
        self.review(1)
        self.recommend(2, seed=2)
        self.review(2)
        self.recommend(3, seed=3)
        temporary = decision_id(999)
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            UPDATE policy_decision_history
            SET reviewed_decision_id = ?
            WHERE decision_id = ? AND position = 0
            """,
            (str(temporary), str(decision_id(3))),
        )
        connection.execute(
            """
            UPDATE policy_decision_history
            SET reviewed_decision_id = ?
            WHERE decision_id = ? AND position = 1
            """,
            (str(decision_id(1)), str(decision_id(3))),
        )
        connection.execute(
            """
            UPDATE policy_decision_history
            SET reviewed_decision_id = ?
            WHERE decision_id = ? AND position = 0
            """,
            (str(decision_id(2)), str(decision_id(3))),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "sequence order"):
            self.store.verify_policy_history(self.policy)

    def test_schema_object_tampering_fails_closed(self) -> None:
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            CREATE TRIGGER unexpected_policy_trigger
            BEFORE INSERT ON policy_reviews
            BEGIN
                SELECT RAISE(ABORT, 'unexpected');
            END
            """
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "unexpected objects"):
            self.store.verify_policy_history(self.policy)

    def test_schema_columns_and_index_definitions_are_enforced(self) -> None:
        cases = (
            (
                "unexpected-column",
                "ALTER TABLE policy_decisions ADD COLUMN unexpected TEXT",
                "unexpected columns",
            ),
            (
                "missing-table",
                "DROP TABLE policy_reviews",
                "missing table: policy_reviews",
            ),
            (
                "altered-index",
                """
                DROP INDEX policy_decisions_policy_sequence;
                CREATE INDEX policy_decisions_policy_sequence
                ON policy_decisions(policy_id, decision_sequence DESC);
                """,
                "unexpected definition: policy_decisions_policy_sequence",
            ),
        )

        for name, script, expected in cases:
            with self.subTest(name=name):
                path = self.root / name / "events.sqlite3"
                store = SQLiteEventStore(path)
                connection = sqlite3.connect(path)
                connection.executescript(script)
                connection.commit()
                connection.close()

                with self.assertRaisesRegex(CorruptJournal, expected):
                    store.verify_policy_history(self.policy)

    def test_schema_version_table_sets_are_enforced_on_reopen(self) -> None:
        non_text_version_path = self.root / "non-text-version" / "events.sqlite3"
        non_text_version_store = SQLiteEventStore(non_text_version_path)
        connection = sqlite3.connect(non_text_version_path)
        connection.execute(
            "UPDATE journal_metadata SET value = ?",
            (sqlite3.Binary(b"2"),),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "schema version is not text"):
            non_text_version_store.verify_policy_history(self.policy)

        missing_metadata_path = self.root / "missing-metadata" / "events.sqlite3"
        SQLiteEventStore(missing_metadata_path)
        connection = sqlite3.connect(missing_metadata_path)
        connection.execute("DROP TABLE journal_metadata")
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "missing table: journal_metadata"):
            SQLiteEventStore(missing_metadata_path)

        extra_v2_path = self.root / "extra-v2" / "events.sqlite3"
        SQLiteEventStore(extra_v2_path)
        connection = sqlite3.connect(extra_v2_path)
        connection.execute("CREATE TABLE unexpected (value TEXT)")
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(
            CorruptJournal,
            "version 3 journal has unexpected or missing tables",
        ):
            SQLiteEventStore(extra_v2_path)

        extra_v1_path = self.root / "extra-v1" / "events.sqlite3"
        SQLiteEventStore(extra_v1_path)
        connection = sqlite3.connect(extra_v1_path)
        connection.executescript(
            """
            DROP INDEX policy_history_reviewed_decision;
            DROP INDEX policy_decisions_policy_sequence;
            DROP TABLE policy_decision_history;
            DROP TABLE policy_reviews;
            DROP TABLE policy_decisions;
            UPDATE journal_metadata
            SET value = '1'
            WHERE key = 'schema_version';
            CREATE TABLE unexpected (value TEXT);
            """
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(
            CorruptJournal,
            "version 1 journal contains unexpected tables",
        ):
            SQLiteEventStore(extra_v1_path)

    def test_global_sequence_gap_fails_closed(self) -> None:
        self.recommend(1, seed=1)
        self.recommend(2, seed=2)
        connection = sqlite3.connect(self.database)
        connection.execute(
            "DELETE FROM policy_decisions WHERE decision_id = ?",
            (str(decision_id(1)),),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "not contiguous"):
            self.store.verify_policy_history(self.policy)

    def test_same_or_future_history_edge_fails_closed(self) -> None:
        self.recommend(1, seed=1)
        self.review(1)
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            INSERT INTO policy_decision_history (
                decision_id, position, reviewed_decision_id
            )
            VALUES (?, 0, ?)
            """,
            (str(decision_id(1)), str(decision_id(1))),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "future decision"):
            self.store.verify_policy_history(self.policy)

    def test_history_edges_cannot_cross_policies_or_skip_positions(self) -> None:
        cases = (
            ("cross-policy", "crosses policy_id"),
            ("position-gap", "positions are not contiguous"),
        )

        for name, expected in cases:
            with self.subTest(name=name):
                path = self.root / name / "events.sqlite3"
                store = SQLiteEventStore(path)
                policy = HierarchicalSoftmaxUCB()
                first = store.recommend(
                    policy,
                    context(),
                    decision_id=decision_id(301),
                    rng_seed=1,
                )
                store.record_review(
                    policy,
                    first.decision_id,
                    fit=DurationFit.JUST_RIGHT,
                    objective_completed=True,
                )
                target = store.recommend(
                    policy,
                    context(),
                    decision_id=decision_id(303),
                    rng_seed=3,
                )
                connection = sqlite3.connect(path)
                if name == "cross-policy":
                    other_policy = HierarchicalSoftmaxUCB(
                        config=PolicyConfig(prior_mean=0.6)
                    )
                    other = store.recommend(
                        other_policy,
                        context(),
                        decision_id=decision_id(302),
                        rng_seed=2,
                    )
                    store.record_review(
                        other_policy,
                        other.decision_id,
                        fit=DurationFit.JUST_RIGHT,
                        objective_completed=True,
                    )
                    connection.execute(
                        """
                        UPDATE policy_decision_history
                        SET reviewed_decision_id = ?
                        WHERE decision_id = ?
                        """,
                        (str(other.decision_id), str(target.decision_id)),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE policy_decision_history
                        SET position = 1
                        WHERE decision_id = ?
                        """,
                        (str(target.decision_id),),
                    )
                connection.commit()
                connection.close()

                with self.assertRaisesRegex(CorruptJournal, expected):
                    store.verify_policy_history(policy)

    def test_history_digest_tampering_fails_closed(self) -> None:
        self.recommend(1, seed=1)
        self.review(1)
        target = self.recommend(2, seed=2)
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            UPDATE policy_decisions
            SET history_sha256 = ?
            WHERE decision_id = ?
            """,
            ("0" * 64, str(target.decision_id)),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "history digest disagrees"):
            self.store.verify_policy_history(self.policy)

    def test_invalid_stored_context_fails_policy_recomputation(self) -> None:
        self.recommend(1, seed=17)
        connection = sqlite3.connect(self.database)
        connection.execute("UPDATE policy_decisions SET previous_focus_seconds = 1")
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(CorruptJournal, "cannot be recomputed"):
            self.store.verify_policy_history(self.policy)

    def test_orphan_review_and_duplicate_history_fail_closed(self) -> None:
        self.recommend(1, seed=1)
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO policy_reviews (
                decision_id, fit, objective_completed
            )
            VALUES (?, 'just_right', 1)
            """,
            (str(decision_id(999)),),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "foreign_key_check"):
            self.store.verify_policy_history(self.policy)

        duplicate_path = self.root / "duplicate" / "events.sqlite3"
        duplicate_store = SQLiteEventStore(duplicate_path)
        duplicate_policy = HierarchicalSoftmaxUCB()
        first = duplicate_store.recommend(
            duplicate_policy,
            context(),
            decision_id=decision_id(101),
            rng_seed=1,
        )
        duplicate_store.record_review(
            duplicate_policy,
            first.decision_id,
            fit=DurationFit.JUST_RIGHT,
            objective_completed=True,
        )
        duplicate_store.recommend(
            duplicate_policy,
            context(),
            decision_id=decision_id(102),
            rng_seed=2,
        )
        connection = sqlite3.connect(duplicate_path)
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.executescript(
            """
            ALTER TABLE policy_decision_history RENAME TO old_history;
            CREATE TABLE policy_decision_history (
                decision_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                reviewed_decision_id TEXT NOT NULL
            );
            INSERT INTO policy_decision_history
            SELECT decision_id, position, reviewed_decision_id
            FROM old_history;
            DROP TABLE old_history;
            """
        )
        connection.execute(
            """
            INSERT INTO policy_decision_history (
                decision_id, position, reviewed_decision_id
            )
            VALUES (?, 1, ?)
            """,
            (str(decision_id(102)), str(decision_id(101))),
        )
        connection.execute(
            """
            UPDATE policy_decisions
            SET history_count = 2
            WHERE decision_id = ?
            """,
            (str(decision_id(102)),),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(CorruptJournal, "schema|repeats a decision"):
            duplicate_store.verify_policy_history(duplicate_policy)

    def test_recommendation_exception_rolls_back_allocation_and_edges(self) -> None:
        with (
            patch.object(
                HierarchicalSoftmaxUCB,
                "recommend_seeded",
                side_effect=RuntimeError("injected failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "injected failure"),
        ):
            self.store.recommend(
                self.policy,
                context(),
                decision_id=decision_id(1),
                rng_seed=1,
            )

        connection = sqlite3.connect(self.database)
        counts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "policy_decisions",
                "policy_reviews",
                "policy_decision_history",
            )
        )
        connection.close()
        self.assertEqual(counts, (0, 0, 0))
        self.assertEqual(self.recommend(2, seed=2).decision_sequence, 1)

    def test_operational_errors_roll_back_and_preserve_error_taxonomy(self) -> None:
        with (
            patch.object(
                HierarchicalSoftmaxUCB,
                "recommend_seeded",
                side_effect=sqlite3.OperationalError("injected write failure"),
            ),
            self.assertRaisesRegex(JournalError, "cannot persist policy decision"),
        ):
            self.store.recommend(
                self.policy,
                context(),
                decision_id=decision_id(1),
                rng_seed=1,
            )
        self.assertEqual(
            self.store.verify_policy_history(self.policy).decision_count,
            0,
        )

        recommendation = self.recommend(2, seed=2)
        with (
            patch.object(
                HierarchicalSoftmaxUCB,
                "recommend_seeded",
                side_effect=sqlite3.OperationalError("injected replay failure"),
            ),
            self.assertRaisesRegex(JournalError, "cannot persist policy review"),
        ):
            self.store.record_review(
                self.policy,
                recommendation.decision_id,
                fit=DurationFit.JUST_RIGHT,
                objective_completed=True,
            )
        self.assertEqual(self.store.reviewed_decisions(self.policy), ())

    def test_active_writer_lock_is_a_conflict_for_decisions_and_reviews(self) -> None:
        blocker = sqlite3.connect(self.database, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaisesRegex(JournalConflict, "locked by a writer"):
                self.store.recommend(
                    self.policy,
                    context(),
                    decision_id=decision_id(1),
                    rng_seed=1,
                )
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        self.assertEqual(
            self.store.verify_policy_history(self.policy).decision_count,
            0,
        )

        recommendation = self.recommend(2, seed=2)
        blocker = sqlite3.connect(self.database, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaisesRegex(JournalConflict, "locked by a writer"):
                self.store.record_review(
                    self.policy,
                    recommendation.decision_id,
                    fit=DurationFit.JUST_RIGHT,
                    objective_completed=True,
                )
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        self.assertEqual(self.store.reviewed_decisions(self.policy), ())

    def test_review_validation_rolls_back_and_seed_bounds_are_enforced(self) -> None:
        self.recommend(1, seed=MAX_RNG_SEED)
        with self.assertRaises(PolicyInputError):
            self.store.record_review(
                self.policy,
                decision_id(1),
                fit="just_right",  # type: ignore[arg-type]
                objective_completed=True,
            )
        reviewed = self.review(1)
        self.assertEqual(reviewed.decision_id, decision_id(1))
        with self.assertRaises(PolicyInputError):
            self.store.recommend(
                self.policy,
                context(),
                decision_id=decision_id(2),
                rng_seed=MAX_RNG_SEED + 1,
            )


if __name__ == "__main__":
    unittest.main()
