from __future__ import annotations

import ast
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import demo_policy_journal


class PolicyJournalDemoTests(unittest.TestCase):
    def test_demo_is_exact_across_two_disposable_journals(self) -> None:
        first = demo_policy_journal.build_demo()
        second = demo_policy_journal.build_demo()

        self.assertEqual(first, second)
        self.assertEqual(first["second_evidence_count"], 1)
        self.assertEqual(first["verification_decision_count"], 2)
        self.assertEqual(first["verification_review_count"], 1)
        self.assertEqual(first["verification_history_edge_count"], 1)
        self.assertEqual(first["verification_sqlite_check"], "ok")
        self.assertEqual(first["workspace_mode"], "0700")
        self.assertEqual(first["journal_mode"], "0600")
        self.assertTrue(first["temporary_workspace_removed"])
        self.assertEqual(
            first["first_propensity_hex"],
            first["review_propensity_hex"],
        )

    def test_human_output_is_path_free_and_labels_the_claim_boundary(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = demo_policy_journal.main(())

        content = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("decisions/reviews/history edges = 2/1/1", content)
        self.assertIn("every CLI call reopened", content)
        self.assertIn("temporary workspace removed", content)
        self.assertIn("not a human outcome or locked evaluation", content)
        self.assertNotIn("/tmp/", content)
        self.assertNotIn("/home/", content)

    def test_harness_has_no_subprocess_or_evaluation_import_surface(self) -> None:
        source = Path(demo_policy_journal.__file__).read_text("utf-8")
        tree = ast.parse(source)
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )

        self.assertNotIn("subprocess", imported_modules)
        self.assertFalse(
            any(
                name.startswith(
                    (
                        "gworker.evaluation",
                        "gworker.publication_runner",
                    )
                )
                for name in imported_modules
            )
        )

    def test_invalid_arguments_do_not_echo_untrusted_bytes(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        sentinel = "/home/alice/github_pat_private_value"

        exit_code = demo_policy_journal.main(
            (sentinel,),
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("error: invalid arguments", stderr.getvalue())
        self.assertNotIn(sentinel, stderr.getvalue())

    def test_expected_failure_is_static_and_path_free(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(
            demo_policy_journal,
            "build_demo",
            side_effect=demo_policy_journal.DemoInvariantError(
                "/home/alice/github_pat_private_value"
            ),
        ):
            exit_code = demo_policy_journal.main(
                (),
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            "gworker demo: workflow verification failed\n",
        )
        self.assertNotIn("/home/alice", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
