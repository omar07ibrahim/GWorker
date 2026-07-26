from __future__ import annotations

import math
import random
import unittest
from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID

from gworker.policy import (
    MAX_AVAILABLE_SECONDS,
    MAX_DECISION_SEQUENCE,
    MAX_RNG_SEED,
    MAX_TEMPLATES,
    POLICY_FAMILY,
    POLICY_ID,
    DurationFit,
    EnergyLevel,
    EvidenceBucket,
    FocusContext,
    FocusTemplate,
    HierarchicalSoftmaxUCB,
    NoFeasibleTemplate,
    PolicyConfig,
    PolicyInputError,
    Recommendation,
    ReviewedDecision,
    TaskKind,
)


def decision_id(sequence: int) -> UUID:
    return UUID(int=sequence)


def context(
    *,
    task_kind: TaskKind = TaskKind.DEEP_WORK,
    energy: EnergyLevel = EnergyLevel.MEDIUM,
    available_seconds: int = 3_600,
    previous_focus_seconds: int | None = None,
) -> FocusContext:
    return FocusContext(
        task_kind=task_kind,
        energy=energy,
        available_seconds=available_seconds,
        previous_focus_seconds=previous_focus_seconds,
    )


def reviewed(
    sequence: int,
    *,
    policy: HierarchicalSoftmaxUCB,
    template_id: str = "focus-25",
    review_context: FocusContext | None = None,
    fit: DurationFit = DurationFit.JUST_RIGHT,
    objective_completed: bool = True,
    propensity: float = 0.25,
) -> ReviewedDecision:
    return ReviewedDecision(
        decision_id=decision_id(sequence),
        decision_sequence=sequence,
        policy_id=policy.policy_id,
        context=review_context or context(),
        template_id=template_id,
        propensity=propensity,
        fit=fit,
        objective_completed=objective_completed,
    )


def scores_by_id(recommendation: Recommendation) -> dict[str, float]:
    return {
        arm.template.template_id: arm.posterior_mean
        for arm in recommendation.arm_scores
    }


class CountingHistory(Sequence[ReviewedDecision]):
    def __init__(self, values: list[ReviewedDecision]) -> None:
        self.values = values
        self.accesses = 0

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(
        self,
        index: int | slice,
    ) -> ReviewedDecision | list[ReviewedDecision]:
        self.accesses += 1
        return self.values[index]


class InvalidRandom(random.Random):
    def random(self) -> float:
        return math.nan


class OutOfRangeRandom(random.Random):
    def random(self) -> float:
        return 1.0


class PolicyValueValidationTests(unittest.TestCase):
    def test_template_validates_identifier_and_duration_budget(self) -> None:
        for template_id in ("", "Focus-25", "focus 25", "focus\n25", "a" * 81):
            with (
                self.subTest(template_id=template_id),
                self.assertRaises(PolicyInputError),
            ):
                FocusTemplate(template_id, 1_500, 300)

        for duration in (0, -1, True, 1.5, MAX_AVAILABLE_SECONDS + 1):
            with (
                self.subTest(duration=duration),
                self.assertRaises(PolicyInputError),
            ):
                FocusTemplate("valid", duration, 300)  # type: ignore[arg-type]

        with self.assertRaisesRegex(PolicyInputError, "must not exceed"):
            FocusTemplate("too-long", MAX_AVAILABLE_SECONDS, 1)

    def test_context_requires_bounded_explicit_enums(self) -> None:
        with self.assertRaisesRegex(PolicyInputError, "TaskKind"):
            FocusContext(  # type: ignore[arg-type]
                task_kind="deep_work",
                energy=EnergyLevel.HIGH,
                available_seconds=1_800,
            )
        with self.assertRaisesRegex(PolicyInputError, "EnergyLevel"):
            FocusContext(  # type: ignore[arg-type]
                task_kind=TaskKind.DEEP_WORK,
                energy="high",
                available_seconds=1_800,
            )
        for available in (0, True, 1.5, MAX_AVAILABLE_SECONDS + 1):
            with (
                self.subTest(available=available),
                self.assertRaises(PolicyInputError),
            ):
                context(available_seconds=available)  # type: ignore[arg-type]
        with self.assertRaisesRegex(PolicyInputError, "previous_focus_seconds"):
            context(previous_focus_seconds=0)

    def test_review_requires_auditable_bounded_fields(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        valid = reviewed(1, policy=policy)

        invalid_values = (
            ({"decision_id": UUID(int=0)}, "nil UUID"),
            ({"decision_id": "1"}, "must be a UUID"),
            ({"decision_sequence": 0}, "decision_sequence"),
            ({"decision_sequence": True}, "decision_sequence"),
            (
                {"decision_sequence": MAX_DECISION_SEQUENCE + 1},
                "decision_sequence",
            ),
            ({"policy_id": "UPPER"}, "lowercase ASCII"),
            ({"context": object()}, "FocusContext"),
            ({"template_id": "focus\u202e25"}, "lowercase ASCII"),
            ({"propensity": object()}, "numeric"),
            ({"propensity": 0}, r"in \(0, 1\]"),
            ({"propensity": math.nan}, "finite"),
            ({"propensity": 1.1}, r"in \(0, 1\]"),
            ({"fit": "just_right"}, "DurationFit"),
            ({"objective_completed": 1}, "boolean"),
        )
        for changes, message in invalid_values:
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(PolicyInputError, message),
            ):
                replace(valid, **changes)

    def test_reward_is_fit_first_and_bounded(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        cases = (
            (DurationFit.JUST_RIGHT, True, 1.0),
            (DurationFit.JUST_RIGHT, False, 0.8),
            (DurationFit.TOO_SHORT, True, 0.2),
            (DurationFit.TOO_LONG, False, 0.0),
        )
        for index, (fit, completed, expected) in enumerate(cases, start=1):
            with self.subTest(fit=fit, completed=completed):
                outcome = reviewed(
                    index,
                    policy=policy,
                    fit=fit,
                    objective_completed=completed,
                )
                self.assertEqual(outcome.reward, expected)
                self.assertGreaterEqual(outcome.reward, 0)
                self.assertLessEqual(outcome.reward, 1)

    def test_config_rejects_invalid_hyperparameters(self) -> None:
        invalid_configs = (
            {"window_size": 0},
            {"window_size": True},
            {"minimum_exact_reviews": 13, "minimum_task_reviews": 12},
            {"prior_mean": -0.01},
            {"prior_mean": math.inf},
            {"prior_mean": 10**1_000},
            {"prior_weight": 0},
            {"prior_weight": 0.009},
            {"prior_weight": 10_001},
            {"direction_weight": -0.01},
            {"direction_weight": 0.51},
            {"exploration_weight": 5.1},
            {"temperature": 0.009},
            {"minimum_probability": 0},
            {"minimum_probability": 0.000_000_1},
            {"minimum_probability": 0.251},
        )
        for values in invalid_configs:
            with (
                self.subTest(values=values),
                self.assertRaises(PolicyInputError),
            ):
                PolicyConfig(**values)  # type: ignore[arg-type]


class PolicyConstructionTests(unittest.TestCase):
    def test_default_identifier_is_stable_and_configuration_bound(self) -> None:
        default = HierarchicalSoftmaxUCB()
        changed_config = HierarchicalSoftmaxUCB(
            config=replace(default.config, temperature=0.21)
        )
        changed_template = HierarchicalSoftmaxUCB(
            templates=(
                FocusTemplate("focus-15", 900, 180),
                FocusTemplate("focus-30", 1_800, 360),
            )
        )

        self.assertEqual(default.policy_id, POLICY_ID)
        self.assertEqual(
            POLICY_ID,
            "hierarchical-softmax-ucb-v1.8c10875dd38a025d",
        )
        self.assertRegex(
            default.policy_id,
            rf"^{POLICY_FAMILY}\.[0-9a-f]{{16}}$",
        )
        self.assertNotEqual(default.policy_id, changed_config.policy_id)
        self.assertNotEqual(default.policy_id, changed_template.policy_id)
        self.assertEqual(
            changed_config.policy_id,
            HierarchicalSoftmaxUCB(
                config=replace(default.config, temperature=0.21)
            ).policy_id,
        )

    def test_numerically_equivalent_configs_share_one_identifier(self) -> None:
        integers = PolicyConfig(
            prior_mean=0,
            prior_weight=2,
            direction_weight=0,
            exploration_weight=0,
            temperature=1,
            minimum_probability=0.02,
        )
        floats = PolicyConfig(
            prior_mean=-0.0,
            prior_weight=2.0,
            direction_weight=0.0,
            exploration_weight=0.0,
            temperature=1.0,
            minimum_probability=0.02,
        )

        self.assertEqual(integers, floats)
        self.assertEqual(
            HierarchicalSoftmaxUCB(config=integers).policy_id,
            HierarchicalSoftmaxUCB(config=floats).policy_id,
        )

    def test_constructor_rejects_ambiguous_template_sets(self) -> None:
        valid = FocusTemplate("a", 100, 20)
        cases = (
            (),
            (valid,),
            (valid, FocusTemplate("a", 200, 20)),
            (valid, FocusTemplate("b", 100, 30)),
            (FocusTemplate("later", 200, 20), valid),
            tuple(
                FocusTemplate(f"t-{index}", index + 1, 1)
                for index in range(MAX_TEMPLATES + 1)
            ),
        )
        for templates in cases:
            with (
                self.subTest(template_count=len(templates)),
                self.assertRaises(PolicyInputError),
            ):
                HierarchicalSoftmaxUCB(templates=templates)

        with self.assertRaisesRegex(PolicyInputError, "FocusTemplate"):
            HierarchicalSoftmaxUCB(  # type: ignore[arg-type]
                templates=(valid, object())
            )
        with self.assertRaisesRegex(PolicyInputError, "PolicyConfig"):
            HierarchicalSoftmaxUCB(config=0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(PolicyInputError, "sequence"):
            HierarchicalSoftmaxUCB(templates=None)  # type: ignore[arg-type]

    def test_constructor_rejects_impossible_probability_floor(self) -> None:
        with self.assertRaisesRegex(PolicyInputError, "multiplied"):
            HierarchicalSoftmaxUCB(config=PolicyConfig(minimum_probability=0.25))


class RecommendationTests(unittest.TestCase):
    def test_seeded_path_matches_an_explicit_rng(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        recommendation_context = context()
        identifier = decision_id(1)
        seeded = policy.recommend_seeded(
            recommendation_context,
            (),
            decision_id=identifier,
            decision_sequence=1,
            rng_seed=20_260_725,
        )
        explicit = policy.recommend(
            recommendation_context,
            (),
            decision_id=identifier,
            decision_sequence=1,
            rng=random.Random(20_260_725),
        )
        self.assertEqual(seeded, explicit)

    def test_seeded_path_rejects_unbounded_values(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        for value in (-1, True, 1.5, MAX_RNG_SEED + 1):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(PolicyInputError, "rng_seed"),
            ):
                policy.recommend_seeded(
                    context(),
                    (),
                    decision_id=decision_id(1),
                    decision_sequence=1,
                    rng_seed=value,  # type: ignore[arg-type]
                )

    def test_cold_start_is_seeded_exploration_with_exact_propensity(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        first = policy.recommend(
            context(),
            [],
            decision_id=decision_id(1),
            decision_sequence=1,
            rng=random.Random(2026),
        )
        repeated = policy.recommend(
            context(),
            [],
            decision_id=decision_id(1),
            decision_sequence=1,
            rng=random.Random(2026),
        )

        self.assertEqual(first, repeated)
        self.assertEqual(first.bucket, EvidenceBucket.GLOBAL)
        self.assertEqual(first.evidence_count, 0)
        self.assertIn("cold_start", first.reason_codes)
        self.assertIn("bounded_exploration", first.reason_codes)
        self.assertAlmostEqual(
            math.fsum(arm.probability for arm in first.arm_scores),
            1.0,
        )
        self.assertTrue(
            all(
                arm.probability >= policy.config.minimum_probability
                for arm in first.arm_scores
            )
        )
        selected = next(
            arm for arm in first.arm_scores if arm.template == first.template
        )
        self.assertEqual(first.propensity, selected.probability)

    def test_review_preserves_decision_provenance(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        recommendation = policy.recommend(
            context(),
            [],
            decision_id=decision_id(8),
            decision_sequence=8,
            rng=random.Random(8),
        )

        outcome = recommendation.review(
            fit=DurationFit.JUST_RIGHT,
            objective_completed=True,
        )

        self.assertEqual(outcome.decision_id, recommendation.decision_id)
        self.assertEqual(outcome.policy_id, policy.policy_id)
        self.assertEqual(outcome.context, recommendation.context)
        self.assertEqual(outcome.template_id, recommendation.template.template_id)
        self.assertEqual(outcome.propensity, recommendation.propensity)
        self.assertEqual(outcome.reward, 1.0)

    def test_availability_filters_before_scoring(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        with self.assertRaisesRegex(PolicyInputError, "FocusContext"):
            policy.feasible_templates(object())  # type: ignore[arg-type]
        recommendation = policy.recommend(
            context(available_seconds=18 * 60),
            [],
            decision_id=decision_id(1),
            decision_sequence=1,
            rng=random.Random(1),
        )

        self.assertEqual(recommendation.template.template_id, "focus-15")
        self.assertEqual(len(recommendation.arm_scores), 1)
        self.assertEqual(recommendation.propensity, 1.0)
        self.assertIn("availability_guardrail", recommendation.reason_codes)

        with self.assertRaises(NoFeasibleTemplate):
            policy.recommend(
                context(available_seconds=18 * 60 - 1),
                [],
                decision_id=decision_id(2),
                decision_sequence=2,
                rng=random.Random(2),
            )

        custom = HierarchicalSoftmaxUCB(
            templates=(
                FocusTemplate("short-focus-long-break", 100, 1_000),
                FocusTemplate("long-focus-short-break", 200, 1),
            )
        )
        with self.assertRaisesRegex(NoFeasibleTemplate, r"\(201\)"):
            custom.recommend(
                context(available_seconds=200),
                [],
                decision_id=decision_id(3),
                decision_sequence=3,
                rng=random.Random(3),
            )

    def test_one_step_guardrail_bounds_duration_changes(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        recommendation = policy.recommend(
            context(previous_focus_seconds=25 * 60),
            [],
            decision_id=decision_id(1),
            decision_sequence=1,
            rng=random.Random(1),
        )

        self.assertEqual(
            {arm.template.focus_seconds for arm in recommendation.arm_scores},
            {15 * 60, 25 * 60, 40 * 60},
        )
        self.assertIn("one_step_guardrail", recommendation.reason_codes)
        with self.assertRaisesRegex(PolicyInputError, "configured template"):
            policy.recommend(
                context(previous_focus_seconds=30 * 60),
                [],
                decision_id=decision_id(2),
                decision_sequence=2,
                rng=random.Random(2),
            )

    def test_availability_can_override_step_when_no_adjacent_arm_fits(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        recommendation = policy.recommend(
            context(
                available_seconds=18 * 60,
                previous_focus_seconds=50 * 60,
            ),
            [],
            decision_id=decision_id(1),
            decision_sequence=1,
            rng=random.Random(1),
        )

        self.assertEqual(recommendation.template.template_id, "focus-15")
        self.assertIn(
            "availability_overrode_step",
            recommendation.reason_codes,
        )

    def test_evidence_hierarchy_prefers_exact_then_task_then_global(self) -> None:
        policy = HierarchicalSoftmaxUCB(
            config=PolicyConfig(
                window_size=10,
                minimum_exact_reviews=2,
                minimum_task_reviews=3,
            )
        )
        high = context(energy=EnergyLevel.HIGH)
        low = context(energy=EnergyLevel.LOW)
        admin = context(task_kind=TaskKind.ADMIN)

        exact_history = [
            reviewed(1, policy=policy, review_context=high),
            reviewed(2, policy=policy, review_context=high),
            reviewed(3, policy=policy, review_context=low),
        ]
        exact = policy.recommend(
            high,
            exact_history,
            decision_id=decision_id(10),
            decision_sequence=10,
            rng=random.Random(1),
        )
        self.assertEqual(exact.bucket, EvidenceBucket.EXACT)
        self.assertEqual(exact.evidence_count, 2)
        self.assertEqual(exact.bucket_label, "exact:deep_work:high")

        task_history = [
            reviewed(11, policy=policy, review_context=high),
            reviewed(12, policy=policy, review_context=low),
            reviewed(
                13,
                policy=policy,
                review_context=context(energy=EnergyLevel.MEDIUM),
            ),
        ]
        task = policy.recommend(
            high,
            task_history,
            decision_id=decision_id(20),
            decision_sequence=20,
            rng=random.Random(2),
        )
        self.assertEqual(task.bucket, EvidenceBucket.TASK)
        self.assertEqual(task.evidence_count, 3)
        self.assertEqual(task.bucket_label, "task:deep_work")

        global_result = policy.recommend(
            admin,
            task_history,
            decision_id=decision_id(21),
            decision_sequence=21,
            rng=random.Random(3),
        )
        self.assertEqual(global_result.bucket, EvidenceBucket.GLOBAL)
        self.assertEqual(global_result.evidence_count, 3)

    def test_sliding_window_forgets_stale_preferences(self) -> None:
        policy = HierarchicalSoftmaxUCB(
            config=PolicyConfig(
                window_size=4,
                minimum_exact_reviews=4,
                minimum_task_reviews=4,
                prior_weight=1,
                exploration_weight=0,
            )
        )
        old_short = [
            reviewed(index, policy=policy, template_id="focus-15")
            for index in range(1, 5)
        ]
        recent_long = [
            reviewed(index, policy=policy, template_id="focus-50")
            for index in range(5, 9)
        ]

        stale = policy.recommend(
            context(),
            old_short,
            decision_id=decision_id(20),
            decision_sequence=20,
            rng=random.Random(1),
        )
        adapted = policy.recommend(
            context(),
            old_short + recent_long,
            decision_id=decision_id(21),
            decision_sequence=21,
            rng=random.Random(1),
        )

        stale_scores = scores_by_id(stale)
        adapted_scores = scores_by_id(adapted)
        self.assertGreater(stale_scores["focus-15"], stale_scores["focus-50"])
        self.assertGreater(adapted_scores["focus-50"], adapted_scores["focus-15"])
        counts = {
            arm.template.template_id: arm.review_count for arm in adapted.arm_scores
        }
        self.assertEqual(counts["focus-15"], 0)
        self.assertEqual(counts["focus-50"], 4)
        self.assertEqual(adapted.evidence_count, 4)

    def test_directional_feedback_guides_ordered_neighboring_arms(self) -> None:
        policy = HierarchicalSoftmaxUCB(
            config=PolicyConfig(
                window_size=4,
                minimum_exact_reviews=4,
                minimum_task_reviews=4,
                prior_weight=1,
                direction_weight=0.3,
                exploration_weight=0,
            )
        )
        too_short = [
            reviewed(
                index,
                policy=policy,
                template_id="focus-25",
                fit=DurationFit.TOO_SHORT,
                objective_completed=False,
            )
            for index in range(1, 5)
        ]
        too_long = [
            reviewed(
                index,
                policy=policy,
                template_id="focus-25",
                fit=DurationFit.TOO_LONG,
                objective_completed=False,
            )
            for index in range(5, 9)
        ]

        longer = policy.recommend(
            context(),
            too_short,
            decision_id=decision_id(20),
            decision_sequence=20,
            rng=random.Random(1),
        )
        shorter = policy.recommend(
            context(),
            too_long,
            decision_id=decision_id(21),
            decision_sequence=21,
            rng=random.Random(1),
        )
        long_adjustments = {
            arm.template.template_id: arm.directional_adjustment
            for arm in longer.arm_scores
        }
        short_adjustments = {
            arm.template.template_id: arm.directional_adjustment
            for arm in shorter.arm_scores
        }

        self.assertGreater(
            long_adjustments["focus-40"],
            long_adjustments["focus-15"],
        )
        self.assertGreater(
            short_adjustments["focus-15"],
            short_adjustments["focus-40"],
        )
        self.assertEqual(long_adjustments["focus-25"], 0)
        self.assertEqual(short_adjustments["focus-25"], 0)

    def test_golden_replay_vector_requires_an_algorithm_version_bump(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        history = [
            reviewed(
                index,
                policy=policy,
                template_id="focus-25",
                fit=DurationFit.TOO_SHORT,
                objective_completed=False,
            )
            for index in range(1, 5)
        ]

        result = policy.recommend(
            context(),
            history,
            decision_id=decision_id(10),
            decision_sequence=10,
            rng=random.Random(42),
        )

        expected = {
            "focus-15": (
                0.55,
                -0.1,
                0.34523469789353145,
                0.1601659450936364,
            ),
            "focus-25": (
                0.18333333333333335,
                0.0,
                0.19932134576242952,
                0.03781297187272354,
            ),
            "focus-40": (
                0.55,
                0.1,
                0.34523469789353145,
                0.40101054151682003,
            ),
            "focus-50": (
                0.55,
                0.1,
                0.34523469789353145,
                0.40101054151682003,
            ),
        }
        self.assertEqual(result.template.template_id, "focus-50")
        self.assertEqual(result.bucket, EvidenceBucket.GLOBAL)
        for arm in result.arm_scores:
            posterior, direction, exploration, probability = expected[
                arm.template.template_id
            ]
            self.assertAlmostEqual(arm.posterior_mean, posterior, places=14)
            self.assertAlmostEqual(
                arm.directional_adjustment,
                direction,
                places=14,
            )
            self.assertAlmostEqual(
                arm.exploration_bonus,
                exploration,
                places=14,
            )
            self.assertAlmostEqual(arm.probability, probability, places=14)

    def test_history_fails_closed_on_provenance_errors(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        valid = reviewed(1, policy=policy)
        other_policy = HierarchicalSoftmaxUCB(
            config=replace(policy.config, temperature=0.21)
        )
        invalid_histories = (
            [replace(valid, policy_id=other_policy.policy_id)],
            [replace(valid, template_id="unknown")],
            [valid, valid],
            [valid, replace(valid, decision_sequence=2)],
            [
                replace(
                    valid,
                    context=context(available_seconds=18 * 60),
                    template_id="focus-50",
                )
            ],
            [
                replace(
                    valid,
                    context=context(previous_focus_seconds=50 * 60),
                    template_id="focus-15",
                )
            ],
            [object()],
        )
        for history in invalid_histories:
            with (
                self.subTest(history=history),
                self.assertRaises(PolicyInputError),
            ):
                policy.recommend(
                    context(),
                    history,  # type: ignore[arg-type]
                    decision_id=decision_id(50),
                    decision_sequence=50,
                    rng=random.Random(1),
                )

        with self.assertRaisesRegex(PolicyInputError, "sequence"):
            policy.recommend(
                context(),
                "not-history",  # type: ignore[arg-type]
                decision_id=decision_id(51),
                decision_sequence=51,
                rng=random.Random(1),
            )

    def test_recommendation_rejects_reused_or_invalid_decision_id(self) -> None:
        policy = HierarchicalSoftmaxUCB(
            config=PolicyConfig(
                window_size=2,
                minimum_exact_reviews=2,
                minimum_task_reviews=2,
            )
        )
        history = [reviewed(index, policy=policy) for index in range(1, 5)]

        with self.assertRaisesRegex(PolicyInputError, "already present"):
            policy.recommend(
                context(),
                history,
                decision_id=decision_id(4),
                decision_sequence=8,
                rng=random.Random(1),
            )
        with self.assertRaisesRegex(PolicyInputError, "nil UUID"):
            policy.recommend(
                context(),
                history,
                decision_id=UUID(int=0),
                decision_sequence=8,
                rng=random.Random(1),
            )
        with self.assertRaisesRegex(PolicyInputError, "random.Random"):
            policy.recommend(
                context(),
                history,
                decision_id=decision_id(8),
                decision_sequence=8,
                rng=object(),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(PolicyInputError, "newer"):
            policy.recommend(
                context(),
                history,
                decision_id=decision_id(9),
                decision_sequence=4,
                rng=random.Random(1),
            )
        with self.assertRaisesRegex(PolicyInputError, r"rng\.random"):
            policy.recommend(
                context(),
                history,
                decision_id=decision_id(9),
                decision_sequence=9,
                rng=InvalidRandom(),
            )
        with self.assertRaisesRegex(PolicyInputError, r"\[0, 1\)"):
            policy.recommend(
                context(),
                history,
                decision_id=decision_id(9),
                decision_sequence=9,
                rng=OutOfRangeRandom(),
            )
        with self.assertRaisesRegex(PolicyInputError, "FocusContext"):
            policy.recommend(
                object(),  # type: ignore[arg-type]
                history,
                decision_id=decision_id(9),
                decision_sequence=9,
                rng=random.Random(1),
            )

    def test_history_tail_is_bounded_and_order_is_verified(self) -> None:
        policy = HierarchicalSoftmaxUCB(
            config=PolicyConfig(
                window_size=2,
                minimum_exact_reviews=2,
                minimum_task_reviews=2,
            )
        )
        history = CountingHistory(
            [reviewed(index, policy=policy) for index in range(1, 401)]
        )

        policy.recommend(
            context(),
            history,
            decision_id=decision_id(401),
            decision_sequence=401,
            rng=random.Random(1),
        )

        self.assertEqual(history.accesses, policy.config.window_size)
        with self.assertRaisesRegex(PolicyInputError, "strictly increasing"):
            policy.recommend(
                context(),
                list(reversed(history.values[-2:])),
                decision_id=decision_id(402),
                decision_sequence=402,
                rng=random.Random(2),
            )

    def test_only_explicit_reviews_are_accepted_as_learning_history(self) -> None:
        policy = HierarchicalSoftmaxUCB()
        recommendation = policy.recommend(
            context(),
            [],
            decision_id=decision_id(1),
            decision_sequence=1,
            rng=random.Random(1),
        )

        with self.assertRaisesRegex(PolicyInputError, "ReviewedDecision"):
            policy.recommend(
                context(),
                [recommendation],  # type: ignore[list-item]
                decision_id=decision_id(2),
                decision_sequence=2,
                rng=random.Random(2),
            )

    def test_extreme_valid_configurations_keep_probabilities_finite(self) -> None:
        configurations = (
            PolicyConfig(
                prior_weight=0.01,
                direction_weight=0.5,
                exploration_weight=5,
                temperature=0.01,
                minimum_probability=0.000_001,
            ),
            PolicyConfig(
                prior_weight=10_000,
                direction_weight=0,
                exploration_weight=0,
                temperature=5,
                minimum_probability=0.24,
            ),
        )
        for index, configuration in enumerate(configurations, start=1):
            with self.subTest(configuration=configuration):
                policy = HierarchicalSoftmaxUCB(config=configuration)
                recommendation = policy.recommend(
                    context(),
                    [],
                    decision_id=decision_id(index),
                    decision_sequence=index,
                    rng=random.Random(index),
                )
                probabilities = [arm.probability for arm in recommendation.arm_scores]
                self.assertTrue(all(math.isfinite(value) for value in probabilities))
                self.assertTrue(all(value > 0 for value in probabilities))
                self.assertAlmostEqual(math.fsum(probabilities), 1.0)

    def test_seeded_stress_run_remains_bounded_and_replayable(self) -> None:
        policy = HierarchicalSoftmaxUCB(
            config=PolicyConfig(
                window_size=32,
                minimum_exact_reviews=4,
                minimum_task_reviews=8,
            )
        )
        generator = random.Random(4_204)
        history: list[ReviewedDecision] = []
        previous: int | None = None
        task_kinds = tuple(TaskKind)
        energy_levels = tuple(EnergyLevel)
        fits = tuple(DurationFit)

        for sequence in range(1, 129):
            current_context = context(
                task_kind=generator.choice(task_kinds),
                energy=generator.choice(energy_levels),
                available_seconds=generator.choice(
                    (18 * 60, 30 * 60, 48 * 60, 60 * 60)
                ),
                previous_focus_seconds=previous,
            )
            recommendation = policy.recommend(
                current_context,
                history,
                decision_id=decision_id(sequence),
                decision_sequence=sequence,
                rng=generator,
            )

            self.assertTrue(recommendation.arm_scores)
            self.assertAlmostEqual(
                math.fsum(arm.probability for arm in recommendation.arm_scores),
                1.0,
            )
            for arm in recommendation.arm_scores:
                self.assertTrue(math.isfinite(arm.posterior_mean))
                self.assertTrue(math.isfinite(arm.directional_adjustment))
                self.assertTrue(math.isfinite(arm.exploration_bonus))
                self.assertTrue(math.isfinite(arm.score))
                self.assertGreaterEqual(
                    arm.probability,
                    policy.config.minimum_probability,
                )

            history.append(
                recommendation.review(
                    fit=generator.choice(fits),
                    objective_completed=bool(generator.randrange(2)),
                )
            )
            previous = recommendation.template.focus_seconds

        final = policy.recommend(
            context(previous_focus_seconds=previous),
            history,
            decision_id=decision_id(1_000),
            decision_sequence=1_000,
            rng=random.Random(1_000),
        )
        self.assertLessEqual(final.evidence_count, policy.config.window_size)


if __name__ == "__main__":
    unittest.main()
