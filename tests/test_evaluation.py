from __future__ import annotations

import hashlib
import json
import math
import random
import unittest
from collections import Counter
from dataclasses import replace
from unittest.mock import patch
from uuid import UUID

import gworker.evaluation as evaluation
from gworker.evaluation import (
    CALIBRATION_EDGES,
    CONTEXT_BLOCK_SIZE,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PERSONAS,
    LOCKED_POLICY_ID,
    AvailabilityMode,
    EvaluationInputError,
    EvaluationInvariantError,
    ExperimentConfig,
    ExperimentResult,
    Persona,
    Strategy,
    evaluator_design_fingerprint,
    evaluator_fingerprint,
    generate_environment,
    latent_preference_minutes,
    run_experiment,
    simulate_trajectory,
    validate_experiment_result,
)
from gworker.policy import (
    POLICY_ID,
    EnergyLevel,
    FocusContext,
    HierarchicalSoftmaxUCB,
    TaskKind,
)


def small_config(
    *,
    personas: tuple[Persona, ...] = (DEFAULT_PERSONAS[4],),
    availability_modes: tuple[AvailabilityMode, ...] = tuple(AvailabilityMode),
    seeds: tuple[int, ...] = (7,),
    replicas: int = 2,
) -> ExperimentConfig:
    return ExperimentConfig(
        split="dev",
        environment_seeds=seeds,
        policy_replicas=replicas,
        personas=personas,
        availability_modes=availability_modes,
        horizon=24,
        drift_decision=13,
        recovery_block_size=2,
        recovery_blocks=2,
    )


class EvaluationDefinitionTests(unittest.TestCase):
    def test_persona_rejects_unsafe_or_ambiguous_fields(self) -> None:
        invalid = (
            {"persona_id": ""},
            {"persona_id": "UPPER"},
            {"persona_id": "space id"},
            {"label": "\nlabel"},
            {"label": " padded"},
            {"primary": 1},
            {"sigma_minutes": 0},
            {"sigma_minutes": math.inf},
            {"selective_reviews": 1},
        )
        for changes in invalid:
            with (
                self.subTest(changes=changes),
                self.assertRaises(EvaluationInputError),
            ):
                replace(DEFAULT_PERSONAS[0], **changes)

    def test_config_rejects_invalid_population_or_recovery_rules(self) -> None:
        valid = small_config()
        invalid = (
            {"split": "production"},
            {"environment_seeds": ()},
            {"environment_seeds": (1, 1)},
            {"environment_seeds": (True,)},
            {"policy_replicas": 0},
            {"personas": ()},
            {"personas": (DEFAULT_PERSONAS[0], DEFAULT_PERSONAS[0])},
            {"availability_modes": ()},
            {
                "availability_modes": (
                    AvailabilityMode.GUARDRAILED,
                    AvailabilityMode.GUARDRAILED,
                )
            },
            {"horizon": 25},
            {"drift_decision": 1},
            {"recovery_block_size": 20},
        )
        for changes in invalid:
            with (
                self.subTest(changes=changes),
                self.assertRaises(EvaluationInputError),
            ):
                replace(valid, **changes)

        with self.assertRaisesRegex(EvaluationInputError, "finite iterables"):
            ExperimentConfig(
                environment_seeds=None,  # type: ignore[arg-type]
            )

    def test_default_protocol_is_frozen_to_declared_population(self) -> None:
        config = DEFAULT_EXPERIMENT_CONFIG

        self.assertEqual(config.split, "eval")
        self.assertEqual(config.environment_seeds, tuple(range(128)))
        self.assertEqual(config.policy_replicas, 4)
        self.assertEqual(config.horizon, 288)
        self.assertEqual(config.drift_decision, 137)
        self.assertEqual(len(config.personas), 9)
        self.assertEqual(config.availability_modes, tuple(AvailabilityMode))
        self.assertEqual(LOCKED_POLICY_ID, POLICY_ID)
        self.assertEqual(
            evaluator_fingerprint(config),
            (
                "synthetic-eval-v3."
                "caf02b5aced57470dfa96c2952df7402dbd5944b7d394a5a95729ef3e6d09045"
            ),
        )

    def test_config_copies_mutable_population_inputs(self) -> None:
        seeds = [7]
        personas = [DEFAULT_PERSONAS[0]]
        modes = [AvailabilityMode.UNCONSTRAINED]
        config = ExperimentConfig(
            split="dev",
            environment_seeds=seeds,  # type: ignore[arg-type]
            policy_replicas=1,
            personas=personas,  # type: ignore[arg-type]
            availability_modes=modes,  # type: ignore[arg-type]
            horizon=24,
            drift_decision=13,
            recovery_block_size=2,
            recovery_blocks=2,
        )
        fingerprint = evaluator_fingerprint(config)

        seeds.append(8)
        personas.append(DEFAULT_PERSONAS[1])
        modes.append(AvailabilityMode.GUARDRAILED)

        self.assertEqual(config.environment_seeds, (7,))
        self.assertEqual(config.personas, (DEFAULT_PERSONAS[0],))
        self.assertEqual(
            config.availability_modes,
            (AvailabilityMode.UNCONSTRAINED,),
        )
        self.assertEqual(evaluator_fingerprint(config), fingerprint)

    def test_fingerprint_covers_labels_and_equation_constants(self) -> None:
        config = small_config(personas=(DEFAULT_PERSONAS[0],))
        relabeled = replace(
            config,
            personas=(replace(DEFAULT_PERSONAS[0], label="Relabeled persona"),),
        )

        self.assertNotEqual(
            evaluator_fingerprint(config),
            evaluator_fingerprint(relabeled),
        )
        design_id = evaluator_design_fingerprint()
        self.assertEqual(len(design_id.rsplit(".", 1)[1]), 64)
        with patch.object(
            evaluation,
            "PREFERENCE_TOLERANCE_MINUTES",
            7.0,
        ):
            self.assertNotEqual(evaluator_design_fingerprint(), design_id)

    def test_latent_preferences_match_frozen_persona_equations(self) -> None:
        config = replace(DEFAULT_EXPERIMENT_CONFIG, split="dev")
        contextual = DEFAULT_PERSONAS[1]
        abrupt_up = DEFAULT_PERSONAS[4]
        gradual = DEFAULT_PERSONAS[6]

        self.assertEqual(
            latent_preference_minutes(
                contextual,
                task_kind=TaskKind.DEEP_WORK,
                energy=EnergyLevel.HIGH,
                decision=1,
                config=config,
            ),
            45,
        )
        self.assertEqual(
            latent_preference_minutes(
                contextual,
                task_kind=TaskKind.ADMIN,
                energy=EnergyLevel.LOW,
                decision=1,
                config=config,
            ),
            15,
        )
        self.assertEqual(
            latent_preference_minutes(
                abrupt_up,
                task_kind=TaskKind.LEARNING,
                energy=EnergyLevel.MEDIUM,
                decision=136,
                config=config,
            ),
            22,
        )
        self.assertEqual(
            latent_preference_minutes(
                abrupt_up,
                task_kind=TaskKind.LEARNING,
                energy=EnergyLevel.MEDIUM,
                decision=137,
                config=config,
            ),
            43,
        )
        self.assertEqual(
            latent_preference_minutes(
                gradual,
                task_kind=TaskKind.LEARNING,
                energy=EnergyLevel.MEDIUM,
                decision=73,
                config=config,
            ),
            22,
        )
        self.assertEqual(
            latent_preference_minutes(
                gradual,
                task_kind=TaskKind.LEARNING,
                energy=EnergyLevel.MEDIUM,
                decision=216,
                config=config,
            ),
            43,
        )


class EnvironmentGenerationTests(unittest.TestCase):
    def test_generation_is_byte_stable_and_split_separated(self) -> None:
        config = small_config()
        first = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=config,
        )
        repeated = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=config,
        )
        test_config = replace(config, split="test")
        test_environment = generate_environment(
            persona=test_config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=test_config,
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first.steps, test_environment.steps)

    def test_each_block_balances_contexts_and_guardrailed_budgets(self) -> None:
        config = small_config()
        environment = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=config,
        )
        expected_contexts = {
            (task_kind, energy) for task_kind in TaskKind for energy in EnergyLevel
        }

        for start in range(0, config.horizon, CONTEXT_BLOCK_SIZE):
            block = environment.steps[start : start + CONTEXT_BLOCK_SIZE]
            self.assertEqual(
                {(step.task_kind, step.energy) for step in block},
                expected_contexts,
            )
            self.assertEqual(
                Counter(step.available_seconds for step in block),
                Counter(
                    {
                        18 * 60: 3,
                        30 * 60: 3,
                        48 * 60: 3,
                        60 * 60: 3,
                    }
                ),
            )

    def test_unconstrained_mode_fits_every_template(self) -> None:
        config = small_config(availability_modes=(AvailabilityMode.UNCONSTRAINED,))
        environment = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.UNCONSTRAINED,
            environment_seed=7,
            config=config,
        )

        self.assertTrue(
            all(step.available_seconds == 60 * 60 for step in environment.steps)
        )
        policy = HierarchicalSoftmaxUCB()
        for step in environment.steps:
            feasible, _ = policy.feasible_templates(
                FocusContext(
                    task_kind=step.task_kind,
                    energy=step.energy,
                    available_seconds=step.available_seconds,
                )
            )
            self.assertEqual(len(feasible), 4)

    def test_potential_outcomes_are_coherent_and_bounded(self) -> None:
        config = small_config()
        environment = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=config,
        )

        for step in environment.steps:
            self.assertEqual(len(step.outcomes), 4)
            for outcome in step.outcomes:
                self.assertGreaterEqual(outcome.fit_probability, 0)
                self.assertLessEqual(outcome.fit_probability, 1)
                self.assertGreaterEqual(outcome.completion_probability, 0.05)
                self.assertLessEqual(outcome.completion_probability, 0.95)
                self.assertGreaterEqual(outcome.realized_reward, 0)
                self.assertLessEqual(outcome.realized_reward, 1)
                self.assertGreaterEqual(outcome.expected_reward, 0)
                self.assertLessEqual(outcome.expected_reward, 1)

    def test_environment_has_a_golden_canonical_vector(self) -> None:
        config = small_config()
        environment = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=config,
        )
        first = environment.steps[0]
        document = {
            "available_seconds": first.available_seconds,
            "decision": first.decision,
            "energy": first.energy.value,
            "latent_preference_minutes": format(
                first.latent_preference_minutes,
                ".17g",
            ),
            "outcomes": [
                {
                    "completed": outcome.completed,
                    "completion_probability": format(
                        outcome.completion_probability,
                        ".17g",
                    ),
                    "expected_reward": format(
                        outcome.expected_reward,
                        ".17g",
                    ),
                    "fit": outcome.fit.value,
                    "fit_probability": format(
                        outcome.fit_probability,
                        ".17g",
                    ),
                    "template_id": outcome.template.template_id,
                }
                for outcome in first.outcomes
            ],
            "task_kind": first.task_kind.value,
        }
        digest = hashlib.sha256(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        self.assertEqual(
            digest,
            "5d16412d12bd3f95c0bdfdd8216951ce89b5f03612c01ed1c4b8db1ef583382a",
        )

    def test_environment_provenance_is_fail_closed(self) -> None:
        config = small_config()
        environment = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=config,
        )

        with self.assertRaisesRegex(EvaluationInputError, "evaluator_id"):
            simulate_trajectory(
                environment,
                Strategy.FIXED_25,
                policy_replica=0,
                config=replace(config, split="test"),
            )
        with self.assertRaisesRegex(EvaluationInputError, "policy_id"):
            simulate_trajectory(
                replace(environment, policy_id="wrong-policy"),
                Strategy.FIXED_25,
                policy_replica=0,
                config=config,
            )

    def test_generation_rejects_undeclared_or_wrongly_typed_inputs(self) -> None:
        config = small_config(
            availability_modes=(AvailabilityMode.GUARDRAILED,),
        )
        environment = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=config,
        )
        cases = (
            {
                "persona": object(),
                "availability_mode": AvailabilityMode.GUARDRAILED,
                "environment_seed": 7,
            },
            {
                "persona": config.personas[0],
                "availability_mode": "guardrailed",
                "environment_seed": 7,
            },
            {
                "persona": DEFAULT_PERSONAS[0],
                "availability_mode": AvailabilityMode.GUARDRAILED,
                "environment_seed": 7,
            },
            {
                "persona": config.personas[0],
                "availability_mode": AvailabilityMode.UNCONSTRAINED,
                "environment_seed": 7,
            },
            {
                "persona": config.personas[0],
                "availability_mode": AvailabilityMode.GUARDRAILED,
                "environment_seed": 8,
            },
        )
        for arguments in cases:
            with (
                self.subTest(arguments=arguments),
                self.assertRaises(EvaluationInputError),
            ):
                generate_environment(
                    config=config,
                    **arguments,  # type: ignore[arg-type]
                )

        with self.assertRaisesRegex(EvaluationInputError, "Persona"):
            latent_preference_minutes(
                object(),  # type: ignore[arg-type]
                task_kind=TaskKind.ADMIN,
                energy=EnergyLevel.LOW,
                decision=1,
                config=config,
            )
        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "potential-outcome set",
        ):
            simulate_trajectory(
                replace(
                    environment,
                    steps=(
                        replace(
                            environment.steps[0],
                            outcomes=tuple(reversed(environment.steps[0].outcomes)),
                        ),
                        *environment.steps[1:],
                    ),
                ),
                Strategy.FIXED_25,
                policy_replica=0,
                config=config,
            )


class TrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = small_config()
        self.environment = generate_environment(
            persona=self.config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=self.config,
        )

    def test_every_predeclared_strategy_is_bounded(self) -> None:
        for strategy in Strategy:
            with self.subTest(strategy=strategy):
                metrics = simulate_trajectory(
                    self.environment,
                    strategy,
                    policy_replica=0,
                    config=self.config,
                )
                self.assertGreaterEqual(
                    metrics.mean_common_expected_regret,
                    0,
                )
                self.assertLessEqual(metrics.mean_common_expected_regret, 1)
                self.assertGreaterEqual(
                    metrics.mean_conditional_expected_regret,
                    0,
                )
                self.assertLessEqual(
                    metrics.mean_conditional_expected_regret,
                    1,
                )
                self.assertAlmostEqual(
                    metrics.mean_common_expected_regret,
                    metrics.mean_conditional_expected_regret
                    + metrics.mean_path_opportunity_cost,
                )
                self.assertGreaterEqual(metrics.mean_expected_reward, 0)
                self.assertLessEqual(metrics.mean_expected_reward, 1)
                self.assertGreaterEqual(metrics.mean_realized_reward, 0)
                self.assertLessEqual(metrics.mean_realized_reward, 1)
                self.assertGreaterEqual(metrics.right_fit_rate, 0)
                self.assertLessEqual(metrics.right_fit_rate, 1)
                self.assertGreaterEqual(metrics.completion_rate, 0)
                self.assertLessEqual(metrics.completion_rate, 1)
                self.assertEqual(
                    len(metrics.common_regret_trace),
                    self.config.horizon,
                )
                self.assertEqual(
                    sum(metrics.template_exposures),
                    metrics.action_count,
                )
                self.assertEqual(metrics.action_count, self.config.horizon)
                self.assertLessEqual(metrics.maximum_arm_transition, 3)

    def test_myopic_oracle_has_zero_conditional_regret(self) -> None:
        metrics = simulate_trajectory(
            self.environment,
            Strategy.MYOPIC_ORACLE,
            policy_replica=0,
            config=self.config,
        )

        self.assertEqual(metrics.mean_conditional_expected_regret, 0)
        self.assertEqual(metrics.cumulative_conditional_expected_regret, 0)
        self.assertTrue(all(value == 0 for value in metrics.conditional_regret_trace))
        self.assertAlmostEqual(
            metrics.mean_common_expected_regret,
            metrics.mean_path_opportunity_cost,
        )

    def test_common_regret_does_not_reward_a_bad_self_locked_path(self) -> None:
        config = ExperimentConfig(
            split="dev",
            environment_seeds=(0,),
            policy_replicas=1,
            personas=(DEFAULT_PERSONAS[0],),
            availability_modes=(AvailabilityMode.UNCONSTRAINED,),
            horizon=288,
            drift_decision=137,
            recovery_block_size=12,
            recovery_blocks=3,
        )
        environment = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.UNCONSTRAINED,
            environment_seed=0,
            config=config,
        )
        fixed_15 = simulate_trajectory(
            environment,
            Strategy.FIXED_15,
            policy_replica=0,
            config=config,
        )
        fixed_50 = simulate_trajectory(
            environment,
            Strategy.FIXED_50,
            policy_replica=0,
            config=config,
        )

        self.assertGreater(
            fixed_15.mean_expected_reward,
            fixed_50.mean_expected_reward,
        )
        self.assertLess(
            fixed_15.mean_common_expected_regret,
            fixed_50.mean_common_expected_regret,
        )
        self.assertGreater(
            fixed_15.mean_conditional_expected_regret,
            fixed_50.mean_conditional_expected_regret,
        )
        self.assertGreater(fixed_50.mean_path_opportunity_cost, 0.5)

    def test_adaptive_diagnostics_are_complete_and_repeatable(self) -> None:
        first = simulate_trajectory(
            self.environment,
            Strategy.ADAPTIVE,
            policy_replica=1,
            config=self.config,
        )
        repeated = simulate_trajectory(
            self.environment,
            Strategy.ADAPTIVE,
            policy_replica=1,
            config=self.config,
        )

        self.assertEqual(first, repeated)
        self.assertIsNotNone(first.minimum_propensity)
        self.assertIsNotNone(first.maximum_inverse_propensity)
        self.assertIsNotNone(first.inverse_propensity_ess_ratio)
        self.assertIsNotNone(first.multiclass_brier_score)
        self.assertEqual(
            first.selected_propensity_count,
            first.action_count,
        )
        self.assertEqual(
            sum(first.evidence_bucket_counts),
            first.action_count,
        )
        self.assertEqual(
            sum(first.calibration_counts),
            first.arm_probability_count,
        )
        self.assertAlmostEqual(
            (first.exact_bucket_rate or 0)
            + (first.task_bucket_rate or 0)
            + (first.global_bucket_rate or 0),
            1,
        )
        self.assertGreater(sum(first.calibration_counts), self.config.horizon)
        self.assertEqual(
            len(first.calibration_counts),
            len(CALIBRATION_EDGES) - 1,
        )

    def test_selective_reviews_create_no_implicit_negative_records(self) -> None:
        config = small_config(personas=(DEFAULT_PERSONAS[8],))
        environment = generate_environment(
            persona=config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=config,
        )
        metrics = simulate_trajectory(
            environment,
            Strategy.ADAPTIVE,
            policy_replica=0,
            config=config,
        )

        self.assertGreater(metrics.review_rate, 0)
        self.assertLess(metrics.review_rate, 1)

    def test_abrupt_persona_emits_recovery_metrics_only(self) -> None:
        abrupt = simulate_trajectory(
            self.environment,
            Strategy.ADAPTIVE,
            policy_replica=0,
            config=self.config,
        )
        stable_config = small_config(personas=(DEFAULT_PERSONAS[0],))
        stable_environment = generate_environment(
            persona=stable_config.personas[0],
            availability_mode=AvailabilityMode.GUARDRAILED,
            environment_seed=7,
            config=stable_config,
        )
        stable = simulate_trajectory(
            stable_environment,
            Strategy.ADAPTIVE,
            policy_replica=0,
            config=stable_config,
        )

        self.assertIsNotNone(abrupt.pre_drift_regret)
        self.assertIsNotNone(abrupt.early_post_drift_auc)
        self.assertIsNotNone(abrupt.late_regret)
        self.assertIn(abrupt.recovery_rate, (0.0, 1.0))
        self.assertIsNotNone(abrupt.conservative_recovery_lag)
        self.assertIsNone(stable.pre_drift_regret)
        self.assertIsNone(stable.recovery_rate)

    def test_baseline_rejects_a_nonzero_policy_replica(self) -> None:
        with self.assertRaisesRegex(EvaluationInputError, "baseline"):
            simulate_trajectory(
                self.environment,
                Strategy.FIXED_25,
                policy_replica=1,
                config=self.config,
            )


class ExperimentTests(unittest.TestCase):
    def test_locked_eval_requires_an_exact_single_use_runner_permit(self) -> None:
        with patch.object(evaluation, "generate_environment") as generate:
            with self.assertRaisesRegex(EvaluationInputError, "publication-runner"):
                run_experiment(DEFAULT_EXPERIMENT_CONFIG)
            with self.assertRaisesRegex(EvaluationInputError, "exact locked"):
                run_experiment(
                    replace(
                        DEFAULT_EXPERIMENT_CONFIG,
                        environment_seeds=(0,),
                    )
                )
            generate.assert_not_called()

        with patch.object(evaluation, "_context_block") as context_block:
            with self.assertRaisesRegex(EvaluationInputError, "authorization"):
                generate_environment(
                    persona=DEFAULT_EXPERIMENT_CONFIG.personas[0],
                    availability_mode=DEFAULT_EXPERIMENT_CONFIG.availability_modes[0],
                    environment_seed=DEFAULT_EXPERIMENT_CONFIG.environment_seeds[0],
                    config=DEFAULT_EXPERIMENT_CONFIG,
                )
            context_block.assert_not_called()

        dev_config = small_config()
        dev_environment = generate_environment(
            persona=dev_config.personas[0],
            availability_mode=dev_config.availability_modes[0],
            environment_seed=dev_config.environment_seeds[0],
            config=dev_config,
        )
        with patch.object(evaluation, "HierarchicalSoftmaxUCB") as policy:
            with self.assertRaisesRegex(EvaluationInputError, "authorization"):
                simulate_trajectory(
                    dev_environment,
                    Strategy.ADAPTIVE,
                    policy_replica=0,
                    config=DEFAULT_EXPERIMENT_CONFIG,
                )
            policy.assert_not_called()

        for run_key, claim_sha256 in (
            ("wrong-run", "0" * 64),
            (evaluation.LOCKED_EVALUATION_RUN_KEY, "invalid"),
        ):
            with (
                self.subTest(run_key=run_key, claim_sha256=claim_sha256),
                self.assertRaises(EvaluationInputError),
            ):
                evaluation._issue_locked_evaluation_permit(
                    run_key=run_key,
                    claim_sha256=claim_sha256,
                )

        permit = evaluation._issue_locked_evaluation_permit(
            run_key=evaluation.LOCKED_EVALUATION_RUN_KEY,
            claim_sha256="0" * 64,
        )
        with self.assertRaisesRegex(EvaluationInputError, "dev or test"):
            run_experiment(small_config(), _eval_permit=permit)
        authorization = evaluation._authorize_experiment_run(
            DEFAULT_EXPERIMENT_CONFIG,
            permit,
        )
        evaluation._authorize_eval_component(
            DEFAULT_EXPERIMENT_CONFIG,
            authorization,
        )
        with self.assertRaisesRegex(EvaluationInputError, "invalid or used"):
            evaluation._authorize_experiment_run(
                DEFAULT_EXPERIMENT_CONFIG,
                permit,
            )

    def test_small_experiment_is_complete_at_seed_grain(self) -> None:
        config = small_config(
            personas=(DEFAULT_PERSONAS[0], DEFAULT_PERSONAS[4]),
            seeds=(3, 7),
        )
        result = run_experiment(config)
        expected_clusters = (
            len(config.personas)
            * len(config.availability_modes)
            * len(config.environment_seeds)
            * len(Strategy)
        )
        expected_traces = (
            1
            * len(config.availability_modes)
            * len(config.environment_seeds)
            * len(Strategy)
        )

        self.assertEqual(len(result.cluster_summaries), expected_clusters)
        self.assertEqual(len(result.abrupt_traces), expected_traces)
        self.assertEqual(result.hard_failure_count, 0)
        validate_experiment_result(result)
        self.assertEqual(
            {summary.strategy for summary in result.cluster_summaries},
            set(Strategy),
        )
        for summary in result.cluster_summaries:
            expected_replicas = (
                config.policy_replicas if summary.strategy is Strategy.ADAPTIVE else 1
            )
            self.assertEqual(
                summary.policy_replica_count,
                expected_replicas,
            )
            self.assertEqual(
                summary.metrics.action_count,
                config.horizon * expected_replicas,
            )
            self.assertEqual(
                sum(summary.metrics.template_exposures),
                summary.metrics.action_count,
            )

    def test_result_validator_rejects_duplicates_and_bad_denominators(self) -> None:
        config = small_config(
            personas=(DEFAULT_PERSONAS[4],),
            availability_modes=(AvailabilityMode.UNCONSTRAINED,),
        )
        result = run_experiment(config)
        first = result.cluster_summaries[0]

        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "duplicate cluster",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=(
                        *result.cluster_summaries,
                        first,
                    ),
                )
            )

        corrupted = replace(
            first,
            metrics=replace(
                first.metrics,
                template_exposures=(0, 0, 0, 0),
            ),
        )
        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "template exposures",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=(
                        corrupted,
                        *result.cluster_summaries[1:],
                    ),
                )
            )

        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "action_count must be an integer",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=(
                        replace(
                            first,
                            metrics=replace(
                                first.metrics,
                                action_count=float(  # type: ignore[arg-type]
                                    first.metrics.action_count
                                ),
                            ),
                        ),
                        *result.cluster_summaries[1:],
                    ),
                )
            )

        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "calibration predicted total",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=(
                        replace(
                            first,
                            metrics=replace(
                                first.metrics,
                                calibration_predicted_sums=(0.0,)
                                * len(first.metrics.calibration_predicted_sums),
                            ),
                        ),
                        *result.cluster_summaries[1:],
                    ),
                )
            )

        assert first.metrics.arm_probability_count is not None
        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "arm_probability_count",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=(
                        replace(
                            first,
                            metrics=replace(
                                first.metrics,
                                arm_probability_count=(
                                    first.metrics.arm_probability_count + 1
                                ),
                            ),
                        ),
                        *result.cluster_summaries[1:],
                    ),
                )
            )

        assert first.metrics.early_post_drift_auc is not None
        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "early post-drift AUC",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=(
                        replace(
                            first,
                            metrics=replace(
                                first.metrics,
                                early_post_drift_auc=(
                                    first.metrics.early_post_drift_auc + 0.125
                                ),
                            ),
                        ),
                        *result.cluster_summaries[1:],
                    ),
                )
            )

        recovered_index, recovered = next(
            (index, summary)
            for index, summary in enumerate(result.cluster_summaries)
            if summary.policy_replica_count == 1
            and summary.metrics.recovered_count == 1
        )
        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "conservative recovery lag|single-replica recovery",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=(
                        *result.cluster_summaries[:recovered_index],
                        replace(
                            recovered,
                            metrics=replace(
                                recovered.metrics,
                                recovered_count=0,
                                recovered_lag_sum=0.0,
                                recovery_rate=0.0,
                                recovery_lag=None,
                            ),
                        ),
                        *result.cluster_summaries[recovered_index + 1 :],
                    ),
                )
            )

        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "availability_mode has an invalid type",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=(
                        replace(
                            first,
                            availability_mode=first.availability_mode.value,  # type: ignore[arg-type]
                        ),
                        *result.cluster_summaries[1:],
                    ),
                )
            )

        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "environment_seed must be an integer",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=(
                        replace(
                            first,
                            environment_seed=float(  # type: ignore[arg-type]
                                first.environment_seed
                            ),
                        ),
                        *result.cluster_summaries[1:],
                    ),
                )
            )

        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "Cartesian set is incomplete",
        ):
            validate_experiment_result(
                replace(
                    result,
                    cluster_summaries=result.cluster_summaries[1:],
                )
            )

        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "policy_id",
        ):
            validate_experiment_result(replace(result, policy_id="wrong-policy"))

        trace = result.abrupt_traces[0]
        changed_trace = (
            min(1.0, trace.common_expected_regret[0] + 0.001),
            *trace.common_expected_regret[1:],
        )
        with self.assertRaisesRegex(
            EvaluationInvariantError,
            "trace disagrees",
        ):
            validate_experiment_result(
                replace(
                    result,
                    abrupt_traces=(
                        replace(
                            trace,
                            common_expected_regret=changed_trace,
                        ),
                        *result.abrupt_traces[1:],
                    ),
                )
            )

    def test_result_defensively_copies_nested_sequence_inputs(self) -> None:
        config = small_config(
            personas=(DEFAULT_PERSONAS[4],),
            availability_modes=(AvailabilityMode.UNCONSTRAINED,),
            replicas=1,
        )
        result = run_experiment(config)
        summaries = list(result.cluster_summaries)
        traces = list(result.abrupt_traces)
        exposures = list(summaries[0].metrics.template_exposures)
        copied_metrics = replace(
            summaries[0].metrics,
            template_exposures=exposures,  # type: ignore[arg-type]
        )
        summaries[0] = replace(summaries[0], metrics=copied_metrics)
        copied = replace(
            result,
            cluster_summaries=summaries,  # type: ignore[arg-type]
            abrupt_traces=traces,  # type: ignore[arg-type]
        )

        exposures[0] += 100
        summaries.clear()
        traces.clear()

        self.assertIsInstance(copied.cluster_summaries, tuple)
        self.assertIsInstance(copied.abrupt_traces, tuple)
        self.assertIsInstance(
            copied.cluster_summaries[0].metrics.template_exposures,
            tuple,
        )
        validate_experiment_result(copied)

    def test_persona_execution_order_does_not_change_trajectories(self) -> None:
        personas = (DEFAULT_PERSONAS[0], DEFAULT_PERSONAS[4])
        forward = run_experiment(
            small_config(
                personas=personas,
                availability_modes=(AvailabilityMode.GUARDRAILED,),
            )
        )
        reverse = run_experiment(
            small_config(
                personas=tuple(reversed(personas)),
                availability_modes=(AvailabilityMode.GUARDRAILED,),
            )
        )

        def indexed(result: ExperimentResult) -> dict[tuple[str, str], object]:
            return {
                (
                    summary.persona_id,
                    summary.strategy.value,
                ): summary.metrics
                for summary in result.cluster_summaries
            }

        self.assertEqual(indexed(forward), indexed(reverse))

    def test_dev_matrix_executes_every_persona_and_mode(self) -> None:
        config = small_config(
            personas=DEFAULT_PERSONAS,
            seeds=(11,),
            replicas=1,
        )
        result = run_experiment(config)

        self.assertEqual(
            {summary.persona_id for summary in result.cluster_summaries},
            {persona.persona_id for persona in DEFAULT_PERSONAS},
        )
        self.assertEqual(
            {summary.availability_mode for summary in result.cluster_summaries},
            set(AvailabilityMode),
        )

    def test_seeded_sampler_matches_its_declared_cold_start_probability(
        self,
    ) -> None:
        policy = HierarchicalSoftmaxUCB()
        context = FocusContext(
            task_kind=TaskKind.DEEP_WORK,
            energy=EnergyLevel.MEDIUM,
            available_seconds=60 * 60,
        )
        sample_count = 4_096
        counts: Counter[str] = Counter()
        expected: dict[str, float] | None = None

        for seed in range(sample_count):
            recommendation = policy.recommend(
                context,
                [],
                decision_id=UUID(int=seed + 1),
                decision_sequence=1,
                rng=random.Random(seed),
            )
            counts[recommendation.template.template_id] += 1
            if expected is None:
                expected = {
                    arm.template.template_id: arm.probability
                    for arm in recommendation.arm_scores
                }

        assert expected is not None
        for template_id, probability in expected.items():
            observed = counts[template_id] / sample_count
            tolerance = (
                4 * math.sqrt(probability * (1 - probability) / sample_count)
                + 1 / sample_count
            )
            self.assertLessEqual(
                abs(observed - probability),
                tolerance,
            )


if __name__ == "__main__":
    unittest.main()
