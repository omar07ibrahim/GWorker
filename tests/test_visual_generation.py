from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from dataclasses import fields
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from gworker.domain import SessionPhase
from gworker.evaluation import DEFAULT_EXPERIMENT_CONFIG
from gworker.policy import DurationFit, EvidenceBucket
from scripts.visuals import generate

FROZEN_OUTPUT_SHA256 = {
    "architecture-trust-boundaries.svg": (
        "0f524f08919028213576037b03e30c3297f8616726eca1f2b811b843ee09ec33"
    ),
    "event-replay.svg": (
        "69e412bc5277adaee2cc2624e84d45e05b81179d7a3d8cba170acb03eaccbbad"
    ),
    "guardrail-matrix.svg": (
        "edae712494e97ec8f5fcbd467ca2272a9c17f9e7f87a7d70520cff05f706ed30"
    ),
    "locked-protocol-inventory.svg": (
        "9476097deceb2c329e9dcf9175d2075f1b47487b2d7126fdebaff7405b815767"
    ),
    "policy-score-decomposition.svg": (
        "17a2300379e9cfe0c8838f8c050e62580cfa90878a51e5305a708a708eb65a99"
    ),
    "publication-lifecycle.svg": (
        "27aa1f73c35f0b30e6c1de8a20a746521bc2a46e06c25ec314185037d387770a"
    ),
}
FROZEN_MANIFEST_SHA256 = (
    "b154b238b98fc4e2ba060b302ae6a1683ca2074d4ef9d6e037d6509aee6a735f"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VisualDataTests(unittest.TestCase):
    def test_event_timeline_is_a_real_completed_replay(self) -> None:
        steps, state = generate.build_event_replay()

        self.assertEqual(
            tuple(step.event_kind for step in steps),
            (
                "session_planned",
                "focus_started",
                "interruption_recorded",
                "interruption_recorded",
                "focus_completed",
                "break_started",
                "break_completed",
            ),
        )
        self.assertEqual(
            tuple(step.phase for step in steps),
            (
                "planned",
                "focusing",
                "focusing",
                "focusing",
                "focus_complete",
                "breaking",
                "completed",
            ),
        )
        self.assertEqual(state.phase, SessionPhase.COMPLETED)
        self.assertEqual(state.revision, 7)
        self.assertEqual(state.interruption_count, 2)
        self.assertEqual(state.interruption_seconds, 60)
        self.assertEqual(state.actual_focus_seconds, 1_470)
        self.assertEqual(state.actual_break_seconds, 280)
        self.assertEqual(state.focus_completion_ratio, 0.98)

    def test_policy_visual_uses_sequential_recommend_and_review_calls(self) -> None:
        scenario = generate.build_policy_scenario()
        recommendation = scenario.recommendation

        self.assertEqual(
            scenario.choices_minutes,
            (25, 40, 40, 40, 50, 40, 40, 40, 40, 40, 40, 40),
        )
        self.assertEqual(
            tuple(review.decision_sequence for review in scenario.reviewed_history),
            tuple(range(1, 13)),
        )
        self.assertEqual(
            tuple(review.decision_id for review in scenario.reviewed_history),
            tuple(
                uuid5(
                    NAMESPACE_URL,
                    f"gworker-demo-policy-v1:{sequence}",
                )
                for sequence in range(1, 13)
            ),
        )
        self.assertEqual(
            tuple(review.fit for review in scenario.reviewed_history),
            (
                DurationFit.TOO_SHORT,
                DurationFit.JUST_RIGHT,
                DurationFit.JUST_RIGHT,
                DurationFit.JUST_RIGHT,
                DurationFit.TOO_LONG,
                DurationFit.JUST_RIGHT,
                DurationFit.JUST_RIGHT,
                DurationFit.JUST_RIGHT,
                DurationFit.JUST_RIGHT,
                DurationFit.JUST_RIGHT,
                DurationFit.JUST_RIGHT,
                DurationFit.JUST_RIGHT,
            ),
        )
        self.assertEqual(
            tuple(review.objective_completed for review in scenario.reviewed_history),
            (False, True, True, True, True, True, True, True, True, True, True, True),
        )
        self.assertEqual(recommendation.template.template_id, "focus-40")
        self.assertEqual(recommendation.bucket, EvidenceBucket.EXACT)
        self.assertEqual(recommendation.evidence_count, 12)
        self.assertEqual(recommendation.propensity, 0.7322645461484336)
        self.assertEqual(
            {
                arm.template.template_id: (
                    arm.review_count,
                    arm.probability,
                )
                for arm in recommendation.arm_scores
            },
            {
                "focus-25": (1, 0.11506356348247043),
                "focus-40": (10, 0.7322645461484336),
                "focus-50": (1, 0.15267189036909593),
            },
        )

    def test_guardrail_matrix_comes_from_feasible_template_queries(self) -> None:
        rows = generate.build_guardrail_rows()

        self.assertEqual(
            tuple(row.allowed_template_ids for row in rows),
            (
                ("focus-15",),
                ("focus-15", "focus-25"),
                ("focus-15", "focus-25", "focus-40"),
                ("focus-15",),
                ("focus-25", "focus-40", "focus-50"),
                ("focus-15", "focus-25", "focus-40", "focus-50"),
            ),
        )
        self.assertEqual(
            tuple(row.reason_codes for row in rows),
            (
                ("availability_guardrail",),
                ("availability_guardrail",),
                ("one_step_guardrail",),
                (
                    "availability_guardrail",
                    "availability_overrode_step",
                ),
                ("one_step_guardrail",),
                (),
            ),
        )
        self.assertIn("ALLOW · override", rows[3].cells)
        self.assertIn("BLOCK · budget", rows[0].cells)
        self.assertIn("BLOCK · step", rows[2].cells)

    def test_protocol_inventory_is_expected_and_has_zero_observed_outputs(
        self,
    ) -> None:
        inventory = generate.build_protocol_inventory()
        actual = {
            definition.name: getattr(inventory, definition.name)
            for definition in fields(inventory)
        }

        self.assertEqual(
            actual,
            {
                "cluster_summaries": 16_128,
                "abrupt_traces": 3_584,
                "raw_trace_points": 1_032_192,
                "trajectories": 23_040,
                "decisions": 6_635_520,
                "scenario_strategy_rows": 126,
                "macro_strategy_rows": 7,
                "macro_contrasts": 6,
                "scenario_contrasts": 108,
                "adaptive_diagnostic_scopes": 19,
                "calibration_rows": 152,
                "template_exposure_rows": 532,
                "recovery_rows": 28,
                "recovery_contrast_rows": 24,
                "trace_series": 28,
                "trace_points": 8_064,
            },
        )
        self.assertEqual(len(DEFAULT_EXPERIMENT_CONFIG.environment_seeds), 128)
        self.assertEqual(generate.observed_locked_outcome_artifacts(), ())


class VisualArtifactTests(unittest.TestCase):
    def test_committed_bundle_matches_a_clean_temporary_generation(self) -> None:
        self.assertEqual(generate.check_bundle(), ())
        with tempfile.TemporaryDirectory(
            prefix=".visual-test-",
            dir=generate.VISUAL_ROOT.parent,
        ) as temporary:
            candidate = Path(temporary) / "visuals"
            generate.write_bundle(candidate)
            self.assertEqual(
                generate._bundle_differences(
                    candidate,
                    generate.VISUAL_ROOT,
                ),
                (),
            )

    def test_frozen_output_and_manifest_hashes(self) -> None:
        generated = generate.VISUAL_ROOT / generate.GENERATED_DIRECTORY_NAME
        self.assertEqual(
            {path.name: sha256(path) for path in sorted(generated.glob("*.svg"))},
            FROZEN_OUTPUT_SHA256,
        )
        self.assertEqual(
            sha256(generate.VISUAL_ROOT / generate.MANIFEST_NAME),
            FROZEN_MANIFEST_SHA256,
        )

    def test_manifest_binds_inputs_and_every_svg(self) -> None:
        manifest_path = generate.VISUAL_ROOT / generate.MANIFEST_NAME
        payload = json.loads(manifest_path.read_bytes())

        self.assertEqual(
            payload["schema_version"],
            "gworker-visual-manifest-v1",
        )
        self.assertEqual(payload["command"], generate.GENERATION_COMMAND)
        self.assertEqual(
            payload["tool"],
            {"name": generate.TOOL_NAME, "version": generate.TOOL_VERSION},
        )
        self.assertEqual(
            payload["python"],
            {
                "requires": ">=3.11",
                "stdlib_only": True,
                "validated_minor_versions": ["3.11", "3.12"],
            },
        )
        self.assertEqual(
            payload["observations"],
            {
                "locked_outcome_artifact_count": 0,
                "locked_result_visuals_generated": False,
            },
        )
        for input_record in payload["inputs"]:
            self.assertEqual(
                input_record["sha256"],
                sha256(generate.ROOT / input_record["path"]),
            )
        output_records = {
            Path(record["path"]).name: record for record in payload["outputs"]
        }
        self.assertEqual(set(output_records), set(FROZEN_OUTPUT_SHA256))
        for filename, expected_sha256 in FROZEN_OUTPUT_SHA256.items():
            path = generate.VISUAL_ROOT / generate.GENERATED_DIRECTORY_NAME / filename
            record = output_records[filename]
            self.assertEqual(record["sha256"], expected_sha256)
            self.assertEqual(record["byte_count"], path.stat().st_size)
            self.assertTrue(record["title"])

    def test_svgs_are_accessible_self_contained_and_directly_labelled(
        self,
    ) -> None:
        generated = generate.VISUAL_ROOT / generate.GENERATED_DIRECTORY_NAME
        required_labels = {
            "architecture-trust-boundaries.svg": ("NEXT", "same-UID"),
            "event-replay.svg": ("session_planned", "FINAL PROJECTION"),
            "guardrail-matrix.svg": ("ALLOW", "BLOCK · budget"),
            "locked-protocol-inventory.svg": (
                "OBSERVED LOCKED OUTCOME ARTIFACTS",
                "EXPECTATIONS, NOT RESULTS",
            ),
            "policy-score-decomposition.svg": (
                "SELECTED",
                "p = 0.732264546148",
            ),
            "publication-lifecycle.svg": ("RENDER · NEXT", "BURNED"),
        }
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        for filename, labels in required_labels.items():
            with self.subTest(filename=filename):
                content = (generated / filename).read_text("utf-8")
                root = ElementTree.fromstring(content)
                title = root.find("svg:title", namespace)
                description = root.find("svg:desc", namespace)
                self.assertEqual(root.attrib["role"], "img")
                self.assertIn("aria-labelledby", root.attrib)
                self.assertIsNotNone(title)
                self.assertIsNotNone(description)
                self.assertTrue(title is not None and title.text)
                self.assertTrue(description is not None and description.text)
                self.assertNotIn("<script", content)
                self.assertNotIn("<image", content)
                self.assertNotIn(" href=", content)
                for label in labels:
                    self.assertIn(label, content)


if __name__ == "__main__":
    unittest.main()
