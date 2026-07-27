from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import gworker.offline as offline
from scripts import demo_offline_replay as demo

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_OUTPUT = (
    "\n".join(
        (
            "GWorker offline replay | fixture=authored-synthetic-replay-v1",
            "behavior-control | status=reportable | exact=true",
            ("  rows=16 | mean-weight=1.000000 | raw-ess=16.000000 | raw-ips=0.525000"),
            "candidate | target=score-temperature | status=reportable",
            ("  rows=16 | temperature=0.750000 | probability-floor=0.020000"),
            ("  raw-ess=12.111890 | raw-ess-ratio=0.756993 | max-weight=5.191650"),
            ("  clipped-rows=4 | clip=3.000000 | removed-weight-mass=5.445540"),
            "descriptive",
            ("  observed=0.525000 | raw-ips=1.078713 | raw-snips=0.466212"),
            ("  clipped-ips=0.916476 | clipped-snips=0.464406"),
            ("support | minimum-behavior=0.080000 | minimum-reviews=12"),
            ("  minimum-raw-ess-ratio=0.250000 | maximum-raw-weight=10.000000"),
            "template-support",
            ("  focus-15=4/16 | focus-25=4/16 | focus-40=4/16 | focus-50=4/16"),
            "nonclaims",
            (
                "  review-selection-corrected=false"
                " | target-policy-value-estimated=false"
            ),
            (
                "  sequential-policy-value-estimated=false"
                " | causal-effect-estimated=false"
            ),
            "  locked-evaluation-used=false",
            ("boundary | authored synthetic rows; descriptive one-step support only"),
            "  no locked evaluation",
        )
    )
    + "\n"
)


class OfflineReplayFixtureTests(unittest.TestCase):
    def test_fixture_is_complete_ordered_and_authored(self) -> None:
        first = demo.fixed_replay_rows()
        second = demo.fixed_replay_rows()

        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(
            tuple(row.decision_sequence for row in first),
            tuple(range(1, 17)),
        )
        self.assertEqual(
            {arm.template_id for row in first for arm in row.arms},
            {"focus-15", "focus-25", "focus-40", "focus-50"},
        )
        self.assertTrue(all(isinstance(row, offline.ReplayRow) for row in first))
        self.assertTrue(
            all(isinstance(arm, offline.ReplayArm) for row in first for arm in row.arms)
        )

    def test_report_exercises_control_support_and_clipping(self) -> None:
        report = demo.build_fixed_replay_report()
        control = report.behavior_control
        candidate = report.candidate

        self.assertEqual(control.readiness, offline.ReplayReadiness.REPORTABLE)
        self.assertEqual(control.reviewed_count, 16)
        self.assertEqual(control.mean_raw_weight, 1.0)
        self.assertEqual(control.maximum_raw_weight, 1.0)
        self.assertEqual(control.raw_effective_sample_size, 16.0)
        self.assertEqual(control.raw_effective_sample_size_ratio, 1.0)
        self.assertEqual(control.clipped_row_count, 0)
        self.assertEqual(control.removed_raw_weight_mass, 0.0)
        self.assertEqual(
            (
                control.raw_inverse_propensity,
                control.raw_self_normalized,
                control.clipped_inverse_propensity,
                control.clipped_self_normalized,
            ),
            (0.525, 0.525, 0.525, 0.525),
        )

        self.assertEqual(candidate.readiness, offline.ReplayReadiness.REPORTABLE)
        self.assertEqual(candidate.reviewed_count, 16)
        self.assertEqual(candidate.clipped_row_count, 4)
        self.assertGreater(candidate.removed_raw_weight_mass, 0.0)
        self.assertGreater(
            candidate.maximum_raw_weight or 0.0,
            report.config.clip_weight,
        )
        self.assertGreaterEqual(
            candidate.raw_effective_sample_size_ratio or 0.0,
            report.config.minimum_raw_ess_ratio,
        )
        self.assertLessEqual(
            candidate.maximum_raw_weight or float("inf"),
            report.config.maximum_raw_weight,
        )
        self.assertEqual(
            tuple(item.reviewed_count for item in candidate.templates),
            (4, 4, 4, 4),
        )

    def test_every_interpretation_flag_is_false(self) -> None:
        nonclaims = demo.build_fixed_replay_report().nonclaims
        observed = {
            field.name: getattr(nonclaims, field.name) for field in fields(nonclaims)
        }

        self.assertEqual(
            observed,
            {
                "review_selection_corrected": False,
                "target_policy_value_estimated": False,
                "sequential_policy_value_estimated": False,
                "causal_effect_estimated": False,
                "locked_evaluation_used": False,
            },
        )

    def test_renderer_rejects_a_different_report_under_the_fixture_id(
        self,
    ) -> None:
        different = offline.build_replay_report(
            demo.fixed_replay_rows(),
            target=offline.ReplayTarget.uniform(),
            config=demo.fixed_replay_config(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not match the fixed replay fixture",
        ):
            demo.render_fixed_replay_report(different)

    def test_renderer_rejects_a_replay_report_subclass(self) -> None:
        report = demo.build_fixed_replay_report()

        class ReplayReportSubclass(offline.ReplayReport):
            pass

        subclass = ReplayReportSubclass(
            **{field.name: getattr(report, field.name) for field in fields(report)}
        )

        with self.assertRaisesRegex(TypeError, "exact ReplayReport"):
            demo.render_fixed_replay_report(subclass)


class OfflineReplayDemoContractTests(unittest.TestCase):
    def run_demo(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "scripts/demo_offline_replay.py"],
            cwd=ROOT,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": "src",
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_terminal_document_is_exact_and_deterministic(self) -> None:
        first = self.run_demo()
        second = self.run_demo()

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stderr, "")
        self.assertEqual(second.stderr, "")
        self.assertEqual(first.stdout, EXPECTED_OUTPUT)
        self.assertEqual(second.stdout, EXPECTED_OUTPUT)

    def test_terminal_document_contains_no_identity_or_host_metadata(self) -> None:
        document = demo.render_fixed_replay_report(demo.build_fixed_replay_report())

        self.assertEqual(document, EXPECTED_OUTPUT)
        self.assertNotRegex(
            document,
            re.compile(
                r"(?:/home/|/tmp/|\\\\Users\\\\|"
                r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
                r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b|"
                r"\b\d{4}-\d{2}-\d{2}(?:T|\s)|"
                r"\b\d{1,2}:\d{2}(?::\d{2})?\b|"
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b|"
                r"@[A-Za-z0-9.-]+)",
                re.IGNORECASE,
            ),
        )
        self.assertNotIn("decision_sequence", document)
        self.assertNotIn("decision-id", document)
        self.assertNotIn("objective", document)

    def test_demo_imports_only_the_public_offline_gworker_surface(self) -> None:
        source = (ROOT / "scripts" / "demo_offline_replay.py").read_text("utf-8")
        tree = ast.parse(source)
        gworker_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                gworker_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name == "gworker" or alias.name.startswith("gworker.")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and (node.module == "gworker" or node.module.startswith("gworker."))
            ):
                gworker_imports.append(node.module)

        self.assertEqual(gworker_imports, ["gworker.offline"])

    def test_clean_process_loads_no_forbidden_gworker_module(self) -> None:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from scripts import demo_offline_replay as demo;"
                    "demo.assert_safe_import_boundary();"
                    "demo.build_fixed_replay_report();"
                    "demo.assert_safe_import_boundary()"
                ),
            ],
            cwd=ROOT,
            env={
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": "src",
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(probe.stdout, "")
        self.assertEqual(probe.stderr, "")

    def test_import_guard_fails_closed(self) -> None:
        for prefix in demo.FORBIDDEN_GWORKER_PREFIXES:
            for forbidden in (prefix, f"{prefix}.nested", f"{prefix}_runner"):
                with (
                    self.subTest(forbidden=forbidden),
                    self.assertRaisesRegex(
                        demo.DemoImportBoundaryError,
                        "forbidden GWorker module",
                    ),
                ):
                    demo.assert_safe_import_boundary(("gworker.offline", forbidden))

    def test_main_redacts_internal_failures_without_a_traceback(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch.object(
                demo,
                "build_fixed_replay_report",
                side_effect=RuntimeError("/home/private/source.py"),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            return_code = demo.main()

        self.assertEqual(return_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "offline replay demo failed closed\n")


if __name__ == "__main__":
    unittest.main()
