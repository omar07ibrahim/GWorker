from __future__ import annotations

import math
import subprocess
import sys
import unittest
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gworker.offline as offline
from gworker.offline import (
    MAX_REPLAY_ROWS,
    ReplayArm,
    ReplayConfig,
    ReplayInputError,
    ReplayNonClaims,
    ReplayReadiness,
    ReplayRow,
    ReplayTarget,
    ReplayTargetKind,
    build_replay_report,
)
from gworker.policy import DurationFit


def arms(
    *,
    first_probability: float = 0.8,
    first_score: float = 1.0,
    second_score: float = 0.0,
) -> tuple[ReplayArm, ...]:
    return (
        ReplayArm("focus-a", first_score, first_probability),
        ReplayArm("focus-b", second_score, 1.0 - first_probability),
    )


def row(
    sequence: int,
    *,
    selected: str = "focus-a",
    fit: DurationFit = DurationFit.JUST_RIGHT,
    completed: bool = True,
    replay_arms: tuple[ReplayArm, ...] | None = None,
) -> ReplayRow:
    return ReplayRow(
        decision_sequence=sequence,
        selected_template_id=selected,
        fit=fit,
        objective_completed=completed,
        arms=replay_arms if replay_arms is not None else arms(),
    )


class UnderreportedRows(Sequence[ReplayRow]):
    def __init__(self, values: tuple[ReplayRow, ...]) -> None:
        self.values = values

    def __len__(self) -> int:
        return 1

    def __getitem__(
        self,
        index: int | slice,
    ) -> ReplayRow | tuple[ReplayRow, ...]:
        return self.values[index]


class ReplayValueTests(unittest.TestCase):
    def test_arm_rejects_unsafe_identifiers_and_nonfinite_values(self) -> None:
        invalid_identifiers = ("", "Focus-a", "focus a", "focus\u202ea", "a" * 81)
        for template_id in invalid_identifiers:
            with (
                self.subTest(template_id=template_id),
                self.assertRaises(ReplayInputError),
            ):
                ReplayArm(template_id, 0.0, 1.0)

        for value in (True, object(), math.nan, math.inf, 10**1_000):
            with (
                self.subTest(score=value),
                self.assertRaises(ReplayInputError),
            ):
                ReplayArm("focus-a", value, 1.0)  # type: ignore[arg-type]

        for probability in (True, 0, -0.1, 1.1, math.nan, math.inf):
            with (
                self.subTest(probability=probability),
                self.assertRaises(ReplayInputError),
            ):
                ReplayArm(  # type: ignore[arg-type]
                    "focus-a",
                    0.0,
                    probability,
                )

    def test_row_is_strict_immutable_and_probability_bound(self) -> None:
        valid = row(1)
        self.assertEqual(valid.reward, 1.0)
        self.assertEqual(valid.selected_arm.template_id, "focus-a")
        with self.assertRaisesRegex(ReplayInputError, "tuple"):
            replace(valid, arms=list(valid.arms))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ReplayInputError, "ReplayArm"):
            replace(valid, arms=(object(),))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ReplayInputError, "unique"):
            replace(valid, arms=(valid.arms[0], valid.arms[0]))
        with self.assertRaisesRegex(ReplayInputError, "feasible arm"):
            replace(valid, selected_template_id="focus-c")
        with self.assertRaisesRegex(ReplayInputError, "sum to one"):
            replace(
                valid,
                arms=(
                    ReplayArm("focus-a", 0.0, 0.4),
                    ReplayArm("focus-b", 0.0, 0.4),
                ),
            )
        with self.assertRaisesRegex(ReplayInputError, "DurationFit"):
            replace(valid, fit="just_right")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ReplayInputError, "boolean"):
            replace(valid, objective_completed=1)  # type: ignore[arg-type]
        for sequence in (0, True, 2**63):
            with (
                self.subTest(sequence=sequence),
                self.assertRaises(ReplayInputError),
            ):
                replace(valid, decision_sequence=sequence)
        with self.assertRaisesRegex(AttributeError, "cannot assign"):
            valid.decision_sequence = 2  # type: ignore[misc]

    def test_reward_uses_only_closed_review_fields(self) -> None:
        cases = (
            (DurationFit.JUST_RIGHT, True, 1.0),
            (DurationFit.JUST_RIGHT, False, 0.8),
            (DurationFit.TOO_SHORT, True, 0.2),
            (DurationFit.TOO_LONG, False, 0.0),
        )
        for fit, completed, expected in cases:
            with self.subTest(fit=fit, completed=completed):
                self.assertEqual(row(1, fit=fit, completed=completed).reward, expected)

    def test_target_parameters_are_kind_specific_and_finite(self) -> None:
        self.assertEqual(
            ReplayTarget.behavior().kind,
            ReplayTargetKind.BEHAVIOR,
        )
        self.assertEqual(ReplayTarget.uniform().kind, ReplayTargetKind.UNIFORM)
        score = ReplayTarget.score_temperature(
            temperature=0.25,
            probability_floor=0.02,
        )
        self.assertEqual(score.kind, ReplayTargetKind.SCORE_TEMPERATURE)
        self.assertEqual(score.temperature, 0.25)
        self.assertEqual(score.probability_floor, 0.02)

        with self.assertRaisesRegex(ReplayInputError, "ReplayTargetKind"):
            ReplayTarget(kind="uniform")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ReplayInputError, "do not accept"):
            ReplayTarget(kind=ReplayTargetKind.UNIFORM, temperature=1.0)
        for temperature in (True, 0, -1, math.nan, math.inf, 10**1_000):
            with (
                self.subTest(temperature=temperature),
                self.assertRaises(ReplayInputError),
            ):
                ReplayTarget.score_temperature(  # type: ignore[arg-type]
                    temperature=temperature,
                )
        for floor in (True, -0.1, 1.0, 1.1, math.nan, math.inf):
            with (
                self.subTest(floor=floor),
                self.assertRaises(ReplayInputError),
            ):
                ReplayTarget.score_temperature(  # type: ignore[arg-type]
                    temperature=1.0,
                    probability_floor=floor,
                )

    def test_config_rejects_ambiguous_or_unbounded_thresholds(self) -> None:
        valid = ReplayConfig()
        invalid = (
            {"minimum_reviews": 0},
            {"minimum_reviews": True},
            {"minimum_reviews": MAX_REPLAY_ROWS + 1},
            {"minimum_raw_ess_ratio": -0.1},
            {"minimum_raw_ess_ratio": 1.1},
            {"minimum_raw_ess_ratio": math.nan},
            {"maximum_raw_weight": 0.99},
            {"maximum_raw_weight": math.inf},
            {"clip_weight": 0.99},
            {"clip_weight": True},
        )
        for changes in invalid:
            with (
                self.subTest(changes=changes),
                self.assertRaises(ReplayInputError),
            ):
                replace(valid, **changes)

    def test_nonclaims_cannot_be_promoted(self) -> None:
        self.assertEqual(
            ReplayNonClaims(),
            ReplayNonClaims(
                review_selection_corrected=False,
                target_policy_value_estimated=False,
                sequential_policy_value_estimated=False,
                causal_effect_estimated=False,
                locked_evaluation_used=False,
            ),
        )
        with self.assertRaisesRegex(ReplayInputError, "remain false"):
            ReplayNonClaims(causal_effect_estimated=True)
        with self.assertRaisesRegex(ReplayInputError, "boolean"):
            ReplayNonClaims(causal_effect_estimated=0)  # type: ignore[arg-type]


class ReplayArithmeticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = (
            row(1),
            row(
                2,
                selected="focus-b",
                fit=DurationFit.TOO_SHORT,
                replay_arms=(
                    ReplayArm("focus-a", 1.0, 0.9),
                    ReplayArm("focus-b", 0.0, 0.1),
                ),
            ),
        )
        self.config = ReplayConfig(
            minimum_reviews=2,
            minimum_raw_ess_ratio=0.5,
            maximum_raw_weight=5.0,
            clip_weight=2.0,
        )

    def test_behavior_negative_control_is_bit_exact(self) -> None:
        report = build_replay_report(
            self.rows,
            target=ReplayTarget.behavior(),
            config=self.config,
        )
        for summary in (report.candidate, report.behavior_control):
            self.assertEqual(summary.readiness, ReplayReadiness.REPORTABLE)
            self.assertEqual(summary.observed_behavior_mean, 0.6)
            self.assertEqual(summary.raw_inverse_propensity, 0.6)
            self.assertEqual(summary.raw_self_normalized, 0.6)
            self.assertEqual(summary.clipped_inverse_propensity, 0.6)
            self.assertEqual(summary.clipped_self_normalized, 0.6)
            self.assertEqual(summary.mean_raw_weight, 1.0)
            self.assertEqual(summary.maximum_raw_weight, 1.0)
            self.assertEqual(summary.raw_effective_sample_size, 2.0)
            self.assertEqual(summary.clipped_effective_sample_size, 2.0)
            self.assertEqual(summary.raw_effective_sample_size_ratio, 1.0)
            self.assertEqual(summary.clipped_effective_sample_size_ratio, 1.0)
            self.assertEqual(summary.clipped_row_count, 0)
            self.assertEqual(summary.removed_raw_weight_mass, 0.0)

    def test_uniform_target_reports_raw_clipped_and_template_statistics(self) -> None:
        candidate = build_replay_report(
            self.rows,
            target=ReplayTarget.uniform(),
            config=self.config,
        ).candidate

        self.assertEqual(candidate.readiness, ReplayReadiness.REPORTABLE)
        self.assertEqual(candidate.reviewed_count, 2)
        self.assertEqual(candidate.observed_behavior_mean, 0.6)
        self.assertEqual(candidate.raw_inverse_propensity, 0.8125)
        self.assertAlmostEqual(
            candidate.raw_self_normalized or -1,
            1.625 / 5.625,
        )
        self.assertEqual(candidate.clipped_inverse_propensity, 0.5125)
        self.assertAlmostEqual(
            candidate.clipped_self_normalized or -1,
            1.025 / 2.625,
        )
        self.assertAlmostEqual(
            candidate.raw_effective_sample_size or -1,
            5.625**2 / (0.625**2 + 5.0**2),
        )
        self.assertAlmostEqual(
            candidate.clipped_effective_sample_size or -1,
            2.625**2 / (0.625**2 + 2.0**2),
        )
        self.assertEqual(candidate.mean_raw_weight, 2.8125)
        self.assertEqual(candidate.maximum_raw_weight, 5.0)
        self.assertEqual(candidate.minimum_selected_behavior_probability, 0.1)
        self.assertEqual(candidate.clipped_row_count, 1)
        self.assertEqual(candidate.removed_raw_weight_mass, 3.0)

        first, second = candidate.templates
        self.assertEqual(
            (first.template_id, second.template_id),
            ("focus-a", "focus-b"),
        )
        self.assertEqual((first.feasible_count, first.reviewed_count), (2, 1))
        self.assertEqual(first.target_mass, 1.0)
        self.assertEqual(first.raw_weight_sum, 0.625)
        self.assertEqual(first.clipped_weight_sum, 0.625)
        self.assertEqual(first.raw_reward_contribution, 0.3125)
        self.assertEqual(first.clipped_reward_contribution, 0.3125)
        self.assertEqual(first.minimum_selected_behavior_probability, 0.8)
        self.assertEqual(first.maximum_raw_weight, 0.625)
        self.assertEqual((second.feasible_count, second.reviewed_count), (2, 1))
        self.assertEqual(second.target_mass, 1.0)
        self.assertEqual(second.raw_weight_sum, 5.0)
        self.assertEqual(second.clipped_weight_sum, 2.0)
        self.assertEqual(second.raw_reward_contribution, 0.5)
        self.assertEqual(second.clipped_reward_contribution, 0.2)
        self.assertEqual(second.minimum_selected_behavior_probability, 0.1)
        self.assertEqual(second.maximum_raw_weight, 5.0)

    def test_score_target_is_stable_and_arm_order_independent(self) -> None:
        target = ReplayTarget.score_temperature(
            temperature=0.7,
            probability_floor=0.02,
        )
        original = build_replay_report(
            self.rows,
            target=target,
            config=self.config,
        )
        reversed_arms = tuple(
            replace(item, arms=tuple(reversed(item.arms))) for item in self.rows
        )
        reordered = build_replay_report(
            reversed_arms,
            target=target,
            config=self.config,
        )
        self.assertEqual(original, reordered)

    def test_score_target_rejects_floor_at_any_row_bound(self) -> None:
        with self.assertRaisesRegex(
            ReplayInputError,
            "every row",
        ):
            build_replay_report(
                (row(1),),
                target=ReplayTarget.score_temperature(
                    temperature=1.0,
                    probability_floor=0.5,
                ),
            )

    def test_underflowed_target_has_unavailable_ratios(self) -> None:
        extreme = row(
            1,
            replay_arms=(
                ReplayArm("focus-a", -1e308, 0.5),
                ReplayArm("focus-b", 1e308, 0.5),
            ),
        )
        candidate = build_replay_report(
            (extreme,),
            target=ReplayTarget.score_temperature(temperature=5e-324),
            config=ReplayConfig(minimum_reviews=1),
        ).candidate
        self.assertEqual(candidate.mean_raw_weight, 0.0)
        self.assertEqual(candidate.raw_inverse_propensity, 0.0)
        self.assertIsNone(candidate.raw_self_normalized)
        self.assertIsNone(candidate.raw_effective_sample_size)
        self.assertIsNone(candidate.raw_effective_sample_size_ratio)
        self.assertEqual(candidate.readiness, ReplayReadiness.UNSTABLE_SUPPORT)

    def test_single_feasible_arm_accepts_a_strictly_smaller_floor(self) -> None:
        single = row(
            1,
            replay_arms=(ReplayArm("focus-a", 4.0, 1.0),),
        )
        report = build_replay_report(
            (single,),
            target=ReplayTarget.score_temperature(
                temperature=1.0,
                probability_floor=math.nextafter(1.0, 0.0),
            ),
            config=ReplayConfig(minimum_reviews=1),
        )
        self.assertEqual(report.candidate.mean_raw_weight, 1.0)
        self.assertEqual(report.candidate.readiness, ReplayReadiness.REPORTABLE)

    def test_support_readiness_uses_raw_not_clipped_weights(self) -> None:
        unstable_by_weight = replace(self.config, maximum_raw_weight=4.9)
        summary = build_replay_report(
            self.rows,
            target=ReplayTarget.uniform(),
            config=unstable_by_weight,
        ).candidate
        self.assertEqual(summary.maximum_raw_weight, 5.0)
        self.assertEqual(summary.clipped_row_count, 1)
        self.assertEqual(summary.readiness, ReplayReadiness.UNSTABLE_SUPPORT)

        insufficient = build_replay_report(
            self.rows,
            target=ReplayTarget.uniform(),
            config=replace(self.config, minimum_reviews=3),
        ).candidate
        self.assertEqual(
            insufficient.readiness,
            ReplayReadiness.INSUFFICIENT_REVIEWS,
        )

    def test_empty_snapshot_never_divides_by_zero(self) -> None:
        report = build_replay_report((), target=ReplayTarget.uniform())
        for summary in (report.candidate, report.behavior_control):
            self.assertEqual(
                summary.readiness,
                ReplayReadiness.INSUFFICIENT_REVIEWS,
            )
            self.assertEqual(summary.reviewed_count, 0)
            self.assertEqual(summary.clipped_row_count, 0)
            self.assertEqual(summary.removed_raw_weight_mass, 0.0)
            self.assertEqual(summary.templates, ())
            self.assertIsNone(summary.observed_behavior_mean)
            self.assertIsNone(summary.raw_inverse_propensity)
            self.assertIsNone(summary.raw_self_normalized)
            self.assertIsNone(summary.raw_effective_sample_size)
            self.assertIsNone(summary.mean_raw_weight)


class ReplaySnapshotBoundaryTests(unittest.TestCase):
    def test_rows_must_be_typed_strictly_increasing_and_bounded(self) -> None:
        target = ReplayTarget.uniform()
        with self.assertRaisesRegex(ReplayInputError, "sequence"):
            build_replay_report("rows", target=target)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ReplayInputError, "ReplayRow"):
            build_replay_report((object(),), target=target)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ReplayInputError, "strictly increasing"):
            build_replay_report((row(2), row(1)), target=target)
        with self.assertRaisesRegex(ReplayInputError, "strictly increasing"):
            build_replay_report((row(1), row(1)), target=target)
        with self.assertRaisesRegex(ReplayInputError, "at most"):
            build_replay_report([row(1)] * (MAX_REPLAY_ROWS + 1), target=target)
        underreported = UnderreportedRows((row(1), row(2), row(3)))
        with (
            patch.object(offline, "MAX_REPLAY_ROWS", 2),
            self.assertRaisesRegex(ReplayInputError, "at most 2"),
        ):
            build_replay_report(
                underreported,
                target=target,
                config=ReplayConfig(minimum_reviews=1),
            )
        with self.assertRaisesRegex(ReplayInputError, "ReplayTarget"):
            build_replay_report((), target=object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ReplayInputError, "ReplayConfig"):
            build_replay_report((), target=target, config=object())  # type: ignore[arg-type]

    def test_nonfinite_importance_weight_fails_closed(self) -> None:
        tiny = math.nextafter(0.0, 1.0)
        dangerous = row(
            1,
            replay_arms=(
                ReplayArm("focus-a", 0.0, tiny),
                ReplayArm("focus-b", 0.0, 1.0),
            ),
        )
        with self.assertRaisesRegex(ReplayInputError, "importance weight"):
            build_replay_report(
                (dangerous,),
                target=ReplayTarget.uniform(),
            )

    def test_offline_import_does_not_load_locked_evaluator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (
            "import sys\n"
            "import gworker.offline\n"
            "raise SystemExit(1 if 'gworker.evaluation' in sys.modules else 0)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env={"PYTHONPATH": str(root / "src")},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
