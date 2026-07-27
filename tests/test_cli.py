from __future__ import annotations

import io
import json
import sqlite3
import stat
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from gworker.cli import run
from gworker.storage import JournalError, JournalSecurityError

DECISION_ID = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e9000")


class CLIWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.journal = self.root / "private" / "events.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = run(
            ("--journal", str(self.journal), *arguments),
            stdout=stdout,
            stderr=stderr,
        )
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_fixed_workflow_survives_reopen_and_omits_the_path(self) -> None:
        recommended = self.invoke(
            "--json",
            "recommend",
            "--task-kind",
            "deep_work",
            "--energy",
            "medium",
            "--available-minutes",
            "60",
            "--decision-id",
            str(DECISION_ID),
            "--seed",
            "20260725",
        )
        self.assertEqual(recommended[0], 0)
        self.assertEqual(recommended[2], "")
        recommendation = json.loads(recommended[1])
        self.assertEqual(recommendation["decision_sequence"], 1)
        self.assertEqual(recommendation["rng_seed"], 20260725)
        self.assertEqual(
            float.fromhex(recommendation["propensity_hex"]),
            recommendation["propensity"],
        )
        self.assertFalse(recommendation["journal_path_disclosed"])
        self.assertNotIn(str(self.root), recommended[1])

        reviewed = self.invoke(
            "--json",
            "review",
            str(DECISION_ID),
            "--fit",
            "just_right",
            "--completed",
        )
        self.assertEqual(reviewed[0], 0)
        self.assertEqual(reviewed[2], "")
        review = json.loads(reviewed[1])
        self.assertEqual(review["decision_id"], str(DECISION_ID))
        self.assertEqual(
            review["propensity"],
            recommendation["propensity"],
        )
        self.assertEqual(
            review["propensity_hex"],
            recommendation["propensity_hex"],
        )
        self.assertNotIn(str(self.root), reviewed[1])

        verified = self.invoke("--json", "verify")
        self.assertEqual(verified[0], 0)
        verification = json.loads(verified[1])
        self.assertEqual(verification["policy_decision_count"], 1)
        self.assertEqual(verification["policy_review_count"], 1)
        self.assertEqual(verification["policy_history_edge_count"], 0)
        self.assertTrue(
            verification["policy_id"].startswith("hierarchical-softmax-ucb-v1.")
        )
        self.assertEqual(verification["sqlite_check"], "ok")
        self.assertNotIn(str(self.root), verified[1])
        self.assertEqual(
            stat.S_IMODE(self.journal.stat().st_mode),
            0o600,
        )

    def test_review_cannot_supply_orphaned_provenance(self) -> None:
        exit_code, stdout, stderr = self.invoke(
            "review",
            str(DECISION_ID),
            "--fit",
            "just_right",
            "--not-completed",
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "gworker: policy journal conflict\n")
        self.assertNotIn(str(self.root), stderr)

    def test_corrupt_stored_text_is_never_echoed(self) -> None:
        initialized = self.invoke("--json", "verify")
        self.assertEqual(initialized[0], 0)
        sentinel = "/home/alice/github_pat_private_value"
        connection = sqlite3.connect(self.journal)
        connection.execute(
            """
            UPDATE journal_metadata
            SET value = ?
            WHERE key = 'schema_version'
            """,
            (sentinel,),
        )
        connection.commit()
        connection.close()

        exit_code, stdout, stderr = self.invoke("verify")
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(
            stderr,
            "gworker: journal is corrupt or incompatible\n",
        )
        self.assertNotIn(sentinel, stderr)
        self.assertNotIn(str(self.root), stderr)

    def test_invalid_arguments_use_injected_stream_and_hide_input(self) -> None:
        sentinel = "/home/alice/github_pat_private_value"
        external_stderr = io.StringIO()
        with redirect_stderr(external_stderr):
            exit_code, stdout, stderr = self.invoke(
                "recommend",
                "--task-kind",
                sentinel,
                "--energy",
                "high",
                "--available-minutes",
                "60",
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(external_stderr.getvalue(), "")
        self.assertIn("error: invalid arguments", stderr)
        self.assertNotIn(sentinel, stderr)
        self.assertNotIn(str(self.root), stderr)

    def test_help_uses_the_injected_output_stream(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = run(("--help",), stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 0)
        self.assertIn("Record and replay", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_operational_errors_use_stable_categories(self) -> None:
        cases = (
            (
                JournalSecurityError("/home/alice/github_pat_private_value"),
                "gworker: journal security check failed\n",
            ),
            (
                JournalError("/home/alice/github_pat_private_value"),
                "gworker: journal operation failed\n",
            ),
        )
        for error, expected in cases:
            with (
                self.subTest(error=type(error).__name__),
                patch("gworker.cli.SQLiteEventStore", side_effect=error),
            ):
                exit_code, stdout, stderr = self.invoke("verify")
                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout, "")
                self.assertEqual(stderr, expected)
                self.assertNotIn("/home/alice", stderr)

    def test_human_output_labels_scope_and_omits_the_path(self) -> None:
        exit_code, stdout, stderr = self.invoke(
            "recommend",
            "--task-kind",
            "learning",
            "--energy",
            "high",
            "--available-minutes",
            "48",
            "--decision-id",
            str(DECISION_ID),
            "--seed",
            "0",
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("GWorker recommendation recorded", stdout)
        self.assertIn("Propensity:", stdout)
        self.assertIn("| exact 0x", stdout)
        self.assertIn("Replay seed: 0", stdout)
        self.assertIn("path omitted", stdout)
        self.assertNotIn(str(self.root), stdout)


if __name__ == "__main__":
    unittest.main()
