from __future__ import annotations

import hashlib
import math
import os
import subprocess
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, cast
from unittest.mock import patch

from gworker.evaluation import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PERSONAS,
    AvailabilityMode,
    ExperimentConfig,
    ExperimentResult,
    Strategy,
    evaluator_fingerprint,
    run_experiment,
)
from gworker.reporting import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_VERSION,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    LOCKED_AUTHOR_EMAIL,
    LOCKED_AUTHOR_NAME,
    LOCKED_BOOTSTRAP_INDICES_SHA256,
    LOCKED_BOOTSTRAP_MODE_INDICES_SHA256,
    LOCKED_BOOTSTRAP_NAMESPACE_SHA256,
    NORMAL_95_CRITICAL_VALUE,
    BootstrapMetadata,
    BootstrapPlan,
    ContrastEstimate,
    IntervalEstimate,
    ModeBootstrapPlan,
    ReportingInputError,
    ReportingInvariantError,
    SourceFileIdentity,
    StatisticalReport,
    bounded_seed_normal_interval,
    build_bootstrap_plan,
    build_statistical_report,
    capture_source_provenance,
    seed_cluster_standard_error,
    stratified_bootstrap_interval,
    stratified_seed_standard_error,
    type7_quantile,
)


@dataclass(frozen=True, slots=True)
class MutableIntervalEstimate(IntervalEstimate):
    payload: list[str]


@dataclass(frozen=True, slots=True)
class MutableContrastEstimate(ContrastEstimate):
    payload: list[str]


def reporting_fixture_config() -> ExperimentConfig:
    return ExperimentConfig(
        split="test",
        environment_seeds=(2, 5, 11, 17),
        policy_replicas=2,
        personas=(
            DEFAULT_PERSONAS[0],
            DEFAULT_PERSONAS[4],
            DEFAULT_PERSONAS[7],
        ),
        availability_modes=tuple(AvailabilityMode),
        horizon=24,
        drift_decision=13,
        recovery_block_size=2,
        recovery_blocks=2,
    )


class QuantileAndBootstrapTests(unittest.TestCase):
    def test_type7_quantile_matches_registered_goldens(self) -> None:
        values = (0.0, 10.0, 20.0, 30.0)

        self.assertEqual(type7_quantile(values, 0), 0)
        self.assertEqual(type7_quantile(values, 0.25), 7.5)
        self.assertEqual(type7_quantile(values, 0.5), 15)
        self.assertEqual(type7_quantile(values, 0.975), 29.25)
        self.assertEqual(type7_quantile(values, 1), 30)
        self.assertEqual(type7_quantile((4.0,), 0.25), 4)

        for invalid_values, probability in (
            ((), 0.5),
            ((0.0, math.inf), 0.5),
            ((0.0, 1.0), -0.1),
            ((0.0, 1.0), 1.1),
        ):
            with (
                self.subTest(
                    values=invalid_values,
                    probability=probability,
                ),
                self.assertRaises(ReportingInputError),
            ):
                type7_quantile(invalid_values, probability)

    def test_seed_standard_errors_preserve_mode_strata(self) -> None:
        values = (-0.3, -0.1, 0.1, 0.3)
        reverse = tuple(reversed(values))

        self.assertAlmostEqual(
            stratified_seed_standard_error(
                {
                    AvailabilityMode.UNCONSTRAINED: values,
                    AvailabilityMode.GUARDRAILED: reverse,
                }
            )
            or 0.0,
            math.sqrt(1 / 120),
        )
        self.assertAlmostEqual(
            seed_cluster_standard_error(values) or 0.0,
            math.sqrt(1 / 60),
        )
        self.assertIsNone(seed_cluster_standard_error((0.2,)))

    def test_bounded_pointwise_normal_interval_has_locked_math(self) -> None:
        values = (0.1, 0.2, 0.3, 0.4)
        interval = bounded_seed_normal_interval(
            values,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        standard_error = seed_cluster_standard_error(values)
        assert standard_error is not None

        self.assertEqual(interval.point_estimate, 0.25)
        self.assertAlmostEqual(
            interval.lower_95,
            0.25 - NORMAL_95_CRITICAL_VALUE * standard_error,
        )
        self.assertAlmostEqual(
            interval.upper_95,
            0.25 + NORMAL_95_CRITICAL_VALUE * standard_error,
        )
        self.assertEqual(interval.seed_count_per_stratum, 4)
        self.assertEqual(interval.stratum_count, 1)
        self.assertEqual(
            bounded_seed_normal_interval(
                (0.0, 0.0, 0.01),
                lower_bound=0.0,
                upper_bound=1.0,
            ).lower_95,
            0.0,
        )
        singleton = bounded_seed_normal_interval(
            (0.2,),
            lower_bound=0.0,
            upper_bound=1.0,
        )
        self.assertEqual(singleton.lower_95, 0.2)
        self.assertEqual(singleton.upper_95, 0.2)

        for invalid_values, bounds in (
            ((), (0.0, 1.0)),
            ((0.2,), (1.0, 0.0)),
            ((1.1,), (0.0, 1.0)),
            ((math.nan,), (0.0, 1.0)),
        ):
            with (
                self.subTest(values=invalid_values, bounds=bounds),
                self.assertRaises(ReportingInputError),
            ):
                bounded_seed_normal_interval(
                    invalid_values,
                    lower_bound=bounds[0],
                    upper_bound=bounds[1],
                )

    def test_bootstrap_plan_is_golden_and_independent_by_mode(self) -> None:
        plan = build_bootstrap_plan(
            (2, 5, 11, 17),
            availability_modes=tuple(AvailabilityMode),
            evaluator_id="fixture-evaluator",
            resample_count=8,
        )
        repeated = build_bootstrap_plan(
            (2, 5, 11, 17),
            availability_modes=tuple(AvailabilityMode),
            evaluator_id="fixture-evaluator",
            resample_count=8,
        )

        self.assertEqual(plan, repeated)
        self.assertEqual(plan.version, BOOTSTRAP_VERSION)
        self.assertNotEqual(
            plan.for_mode(AvailabilityMode.UNCONSTRAINED).indices,
            plan.for_mode(AvailabilityMode.GUARDRAILED).indices,
        )
        self.assertEqual(
            plan.namespace_sha256,
            "efbef6f5168f467238e6c4367b6c4f40332c56e82ca05e22e02d3574cae69745",
        )
        self.assertEqual(
            plan.indices_sha256,
            "fcc3278d52f30875a912cc45b6549016318727af8f3de94ea8b20eb60adf29c0",
        )
        self.assertEqual(
            tuple(item.indices_sha256 for item in plan.mode_plans),
            (
                "588c1f55f30823b486c1698eb6ab6d1ea3d9c102a15d5d80052a6ddf13306b9a",
                "bd386b90c641e1f2a40a04faa5a0f0f3a016edb78bcbc9cbb420f6333bcf2b5d",
            ),
        )

    def test_locked_bootstrap_definition_has_registered_digests(self) -> None:
        config = DEFAULT_EXPERIMENT_CONFIG
        plan = build_bootstrap_plan(
            config.environment_seeds,
            availability_modes=config.availability_modes,
            evaluator_id=evaluator_fingerprint(config),
            resample_count=DEFAULT_BOOTSTRAP_RESAMPLES,
        )

        self.assertEqual(
            plan.namespace_sha256,
            LOCKED_BOOTSTRAP_NAMESPACE_SHA256,
        )
        self.assertEqual(
            plan.indices_sha256,
            LOCKED_BOOTSTRAP_INDICES_SHA256,
        )
        self.assertEqual(
            tuple(
                (item.availability_mode, item.indices_sha256)
                for item in plan.mode_plans
            ),
            LOCKED_BOOTSTRAP_MODE_INDICES_SHA256,
        )

    def test_bootstrap_plans_fail_closed_on_tampering_and_unsafe_size(self) -> None:
        plan = build_bootstrap_plan(
            (2, 5, 11, 17),
            availability_modes=tuple(AvailabilityMode),
            evaluator_id="tamper-fixture",
            resample_count=8,
        )
        changed_rows = list(plan.mode_plans[0].indices)
        changed_row = list(changed_rows[0])
        changed_row[0] = (changed_row[0] + 1) % len(plan.environment_seeds)
        changed_rows[0] = tuple(changed_row)

        with self.assertRaisesRegex(ReportingInputError, "mode bootstrap"):
            replace(
                plan.mode_plans[0],
                indices=tuple(changed_rows),
            )
        with self.assertRaisesRegex(ReportingInputError, "combined bootstrap"):
            replace(plan, indices_sha256="0" * 64)
        with self.assertRaisesRegex(ReportingInputError, "namespace"):
            replace(plan, namespace_sha256="0" * 64)
        with self.assertRaisesRegex(ReportingInputError, "draw-count"):
            build_bootstrap_plan(
                tuple(range(10_000)),
                availability_modes=tuple(AvailabilityMode),
                evaluator_id="oversized-fixture",
                resample_count=100_000,
            )

    def test_bootstrap_definition_rejects_invalid_public_inputs(self) -> None:
        invalid_calls = (
            (
                (),
                (AvailabilityMode.UNCONSTRAINED,),
                "fixture",
                8,
            ),
            (
                (0, 0),
                (AvailabilityMode.UNCONSTRAINED,),
                "fixture",
                8,
            ),
            (
                cast("tuple[int, ...]", (True, 1)),
                (AvailabilityMode.UNCONSTRAINED,),
                "fixture",
                8,
            ),
            (
                (0, 1),
                (),
                "fixture",
                8,
            ),
            (
                (0, 1),
                (
                    AvailabilityMode.UNCONSTRAINED,
                    AvailabilityMode.UNCONSTRAINED,
                ),
                "fixture",
                8,
            ),
            (
                (0, 1),
                (AvailabilityMode.UNCONSTRAINED,),
                "",
                8,
            ),
            (
                (0, 1),
                (AvailabilityMode.UNCONSTRAINED,),
                "fixture",
                0,
            ),
        )
        for seeds, modes, evaluator_id, resample_count in invalid_calls:
            with (
                self.subTest(
                    seeds=seeds,
                    modes=modes,
                    evaluator_id=evaluator_id,
                    resample_count=resample_count,
                ),
                self.assertRaises(ReportingInputError),
            ):
                build_bootstrap_plan(
                    seeds,
                    availability_modes=modes,
                    evaluator_id=evaluator_id,
                    resample_count=resample_count,
                )

        one_mode = build_bootstrap_plan(
            (0, 1),
            availability_modes=(AvailabilityMode.UNCONSTRAINED,),
            evaluator_id="missing-mode-fixture",
            resample_count=4,
        )
        with self.assertRaises(ReportingInvariantError):
            one_mode.for_mode(AvailabilityMode.GUARDRAILED)

    def test_constant_paired_effect_survives_every_resample(self) -> None:
        plan = build_bootstrap_plan(
            (0, 1, 2, 3),
            availability_modes=(AvailabilityMode.UNCONSTRAINED,),
            evaluator_id="constant-effect-fixture",
            resample_count=64,
        )
        interval = stratified_bootstrap_interval(
            {
                AvailabilityMode.UNCONSTRAINED: (
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                )
            },
            plan,
        )

        self.assertAlmostEqual(interval.point_estimate, -0.1)
        self.assertAlmostEqual(interval.lower_95, -0.1)
        self.assertAlmostEqual(interval.upper_95, -0.1)
        self.assertAlmostEqual(interval.seed_cluster_standard_error or 0.0, 0)
        self.assertEqual(interval.seed_count_per_stratum, 4)
        self.assertEqual(interval.stratum_count, 1)

    def test_two_mode_interval_matches_an_independent_manual_oracle(self) -> None:
        plan = build_bootstrap_plan(
            (2, 5, 11, 17),
            availability_modes=tuple(AvailabilityMode),
            evaluator_id="manual-oracle-fixture",
            resample_count=8,
        )
        values: dict[AvailabilityMode, tuple[float, ...]] = {
            AvailabilityMode.UNCONSTRAINED: (-0.3, -0.1, 0.1, 0.3),
            AvailabilityMode.GUARDRAILED: (0.4, 0.2, 0.0, -0.2),
        }
        interval = stratified_bootstrap_interval(values, plan)
        manual_draws: list[float] = []
        for bootstrap_index in range(plan.resample_count):
            mode_means = []
            for mode in tuple(AvailabilityMode):
                indices = plan.for_mode(mode).indices[bootstrap_index]
                mode_means.append(
                    math.fsum(values[mode][index] for index in indices) / len(indices)
                )
            manual_draws.append(math.fsum(mode_means) / len(mode_means))
        ordered = sorted(manual_draws)

        def manual_type7(probability: float) -> float:
            position = (len(ordered) - 1) * probability
            lower = math.floor(position)
            fraction = position - lower
            if fraction == 0:
                return ordered[lower]
            return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])

        self.assertAlmostEqual(
            interval.point_estimate,
            0.05,
        )
        self.assertAlmostEqual(interval.lower_95, manual_type7(0.025))
        self.assertAlmostEqual(interval.upper_95, manual_type7(0.975))

    def test_stratified_standard_error_rejects_invalid_strata(self) -> None:
        for values in (
            {AvailabilityMode.UNCONSTRAINED: ()},
            {AvailabilityMode.UNCONSTRAINED: (math.nan, 0.0)},
            {AvailabilityMode.UNCONSTRAINED: (math.inf, 0.0)},
            {AvailabilityMode.UNCONSTRAINED: (True, 0.0)},
        ):
            with (
                self.subTest(values=values),
                self.assertRaises(
                    (ReportingInputError, RuntimeError),
                ),
            ):
                stratified_seed_standard_error(
                    cast(
                        "dict[AvailabilityMode, tuple[float, ...]]",
                        values,
                    )
                )
        with self.assertRaises(ReportingInvariantError):
            stratified_seed_standard_error({})
        with self.assertRaises(ReportingInvariantError):
            stratified_seed_standard_error(
                {
                    AvailabilityMode.UNCONSTRAINED: (0.0, 1.0),
                    AvailabilityMode.GUARDRAILED: (0.0,),
                }
            )
        with self.assertRaises(ReportingInputError):
            stratified_seed_standard_error(
                cast(
                    "dict[AvailabilityMode, tuple[float, ...]]",
                    {"not-a-mode": (0.0, 1.0)},
                )
            )

    def test_interval_rejects_misaligned_strata(self) -> None:
        plan = build_bootstrap_plan(
            (0, 1),
            availability_modes=tuple(AvailabilityMode),
            evaluator_id="alignment-fixture",
            resample_count=4,
        )
        with self.assertRaises(ReportingInvariantError):
            stratified_bootstrap_interval({}, plan)
        with self.assertRaises(ReportingInvariantError):
            stratified_bootstrap_interval(
                {
                    AvailabilityMode.GUARDRAILED: (0.0, 1.0),
                    AvailabilityMode.UNCONSTRAINED: (0.0, 1.0),
                },
                plan,
            )
        with self.assertRaises(ReportingInvariantError):
            stratified_bootstrap_interval(
                {AvailabilityMode.UNCONSTRAINED: (0.0,)},
                plan,
            )


class ReportValueValidationTests(unittest.TestCase):
    report: ClassVar[StatisticalReport]
    plan: ClassVar[BootstrapPlan]

    @classmethod
    def setUpClass(cls) -> None:
        result = run_experiment(reporting_fixture_config())
        cls.report = build_statistical_report(result, resample_count=8)
        cls.plan = build_bootstrap_plan(
            (0, 1),
            availability_modes=(AvailabilityMode.UNCONSTRAINED,),
            evaluator_id="validation-fixture",
            resample_count=4,
        )

    def test_interval_and_contrast_values_fail_closed(self) -> None:
        interval = IntervalEstimate(
            point_estimate=0.0,
            lower_95=-0.1,
            upper_95=0.1,
            seed_cluster_standard_error=0.02,
            seed_count_per_stratum=4,
            stratum_count=2,
        )
        for interval_changes in (
            {"lower_95": 0.2},
            {"seed_cluster_standard_error": -0.1},
            {"seed_cluster_standard_error": None},
            {"seed_count_per_stratum": 1},
            {"seed_count_per_stratum": 0},
            {"stratum_count": 0},
            {"point_estimate": math.nan},
            {"point_estimate": 10**1_000},
        ):
            with (
                self.subTest(changes=interval_changes),
                self.assertRaises(ReportingInputError),
            ):
                replace(interval, **interval_changes)

        valid = ContrastEstimate(
            scope="macro",
            persona_id=None,
            availability_mode=None,
            comparator=Strategy.FIXED_15,
            interval=interval,
            interval_relation_to_zero="includes-zero",
        )
        for contrast_changes in (
            {"scope": "invalid"},
            {"comparator": Strategy.ADAPTIVE},
            {"interval_relation_to_zero": "invalid"},
            {"interval_relation_to_zero": "above-zero"},
            {"persona_id": "unexpected"},
        ):
            with (
                self.subTest(changes=contrast_changes),
                self.assertRaises(ReportingInputError),
            ):
                replace(valid, **contrast_changes)

        mutable_interval = MutableIntervalEstimate(
            point_estimate=interval.point_estimate,
            lower_95=interval.lower_95,
            upper_95=interval.upper_95,
            seed_cluster_standard_error=interval.seed_cluster_standard_error,
            seed_count_per_stratum=interval.seed_count_per_stratum,
            stratum_count=interval.stratum_count,
            payload=[],
        )
        with self.assertRaisesRegex(ReportingInputError, "IntervalEstimate"):
            replace(valid, interval=mutable_interval)

        cell = self.report.cell_contrasts[0]
        mutable_cell = MutableContrastEstimate(
            scope=cell.scope,
            persona_id=cell.persona_id,
            availability_mode=cell.availability_mode,
            comparator=cell.comparator,
            interval=cell.interval,
            interval_relation_to_zero=cell.interval_relation_to_zero,
            payload=[],
        )
        with self.assertRaisesRegex(ReportingInputError, "invalid type"):
            replace(
                self.report,
                cell_contrasts=(
                    mutable_cell,
                    *self.report.cell_contrasts[1:],
                ),
            )

        for bounds, relation in (
            ((-0.3, -0.1), "below-zero"),
            ((0.1, 0.3), "above-zero"),
        ):
            directional = replace(
                interval,
                lower_95=bounds[0],
                upper_95=bounds[1],
            )
            self.assertEqual(
                replace(
                    valid,
                    interval=directional,
                    interval_relation_to_zero=relation,
                ).interval_relation_to_zero,
                relation,
            )
        with self.assertRaises(ReportingInputError):
            ContrastEstimate(
                scope="cell",
                persona_id=None,
                availability_mode=None,
                comparator=Strategy.FIXED_15,
                interval=interval,
                interval_relation_to_zero="includes-zero",
            )

    def test_strategy_cell_values_fail_closed(self) -> None:
        cell = self.report.strategy_cells[0]
        invalid_changes = (
            {"persona_id": ""},
            {"persona_primary": cast("bool", 1)},
            {"availability_mode": cast("AvailabilityMode", "invalid")},
            {"strategy": cast("Strategy", "invalid")},
            {"seed_count": 0},
            {"mean_expected_reward": 1.1},
            {"common_regret_standard_error": -0.1},
            {"mean_common_expected_regret": (cell.mean_common_expected_regret + 0.01)},
        )
        for changes in invalid_changes:
            with (
                self.subTest(changes=changes),
                self.assertRaises(ReportingInputError),
            ):
                replace(cell, **changes)

    def test_bootstrap_value_objects_reject_malformed_state(self) -> None:
        mode_plan = self.plan.mode_plans[0]
        for mode_changes in (
            {"availability_mode": cast("AvailabilityMode", "invalid")},
            {"indices": ()},
            {"indices": ((0,), (0, 1))},
            {"indices": ((2**32,),)},
            {"indices_sha256": "x"},
            {"indices_sha256": "0" * 64},
        ):
            with (
                self.subTest(changes=mode_changes),
                self.assertRaises(ReportingInputError),
            ):
                replace(mode_plan, **mode_changes)

        shorter = build_bootstrap_plan(
            (0, 1),
            availability_modes=(AvailabilityMode.UNCONSTRAINED,),
            evaluator_id="shorter-fixture",
            resample_count=2,
        )
        wider = build_bootstrap_plan(
            (0, 1, 2),
            availability_modes=(AvailabilityMode.UNCONSTRAINED,),
            evaluator_id="wider-fixture",
            resample_count=4,
        )
        invalid_plan_changes = (
            {"version": "invalid"},
            {"resample_count": 0},
            {"environment_seeds": (0, 0)},
            {"availability_modes": ()},
            {"mode_plans": cast("tuple[ModeBootstrapPlan, ...]", ("invalid",))},
            {"mode_plans": shorter.mode_plans},
            {"mode_plans": wider.mode_plans},
        )
        for plan_changes in invalid_plan_changes:
            with (
                self.subTest(changes=plan_changes),
                self.assertRaises(ReportingInputError),
            ):
                replace(self.plan, **plan_changes)

    def test_metadata_and_report_collections_reject_tampering(self) -> None:
        metadata = self.report.bootstrap
        invalid_metadata_changes = (
            {"version": "invalid"},
            {"resample_count": 0},
            {"confidence": 0.9},
            {"namespace_sha256": "x"},
            {"indices_sha256": "x"},
            {"mode_indices_sha256": ()},
            {
                "mode_indices_sha256": (
                    metadata.mode_indices_sha256[0],
                    metadata.mode_indices_sha256[0],
                )
            },
            {"mode_indices_sha256": ((AvailabilityMode.UNCONSTRAINED, "x"),)},
        )
        for metadata_changes in invalid_metadata_changes:
            with (
                self.subTest(changes=metadata_changes),
                self.assertRaises(ReportingInputError),
            ):
                replace(metadata, **metadata_changes)

        invalid_report_changes = (
            {"schema_version": "invalid"},
            {"evaluator_id": ""},
            {"bootstrap": cast("BootstrapMetadata", "invalid")},
            {"primary_contrast": self.report.macro_contrasts[0]},
            {"macro_contrasts": self.report.macro_contrasts[:-1]},
            {"macro_contrasts": tuple(reversed(self.report.macro_contrasts))},
            {
                "macro_contrasts": (
                    replace(
                        self.report.macro_contrasts[0],
                        scope="primary-macro",
                    ),
                    *self.report.macro_contrasts[1:],
                )
            },
            {"cell_contrasts": self.report.cell_contrasts[:-1]},
            {"strategy_cells": self.report.strategy_cells[:-1]},
            {"strategy_cells": ()},
        )
        for report_changes in invalid_report_changes:
            with (
                self.subTest(changes=report_changes),
                self.assertRaises(ReportingInputError),
            ):
                replace(self.report, **report_changes)


class StatisticalReportTests(unittest.TestCase):
    result: ClassVar[ExperimentResult]

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_experiment(reporting_fixture_config())

    def test_report_preserves_weights_pairing_and_all_visible_cells(self) -> None:
        report = build_statistical_report(
            self.result,
            resample_count=64,
        )
        config = self.result.config
        index = {
            (
                summary.persona_id,
                summary.availability_mode,
                summary.environment_seed,
                summary.strategy,
            ): summary
            for summary in self.result.cluster_summaries
        }
        primary_personas = tuple(
            persona for persona in config.personas if persona.primary
        )
        mode_means: list[float] = []
        for mode in config.availability_modes:
            seed_effects: list[float] = []
            for seed in config.environment_seeds:
                persona_effects = [
                    index[
                        (
                            persona.persona_id,
                            mode,
                            seed,
                            Strategy.ADAPTIVE,
                        )
                    ].metrics.mean_common_expected_regret
                    - index[
                        (
                            persona.persona_id,
                            mode,
                            seed,
                            Strategy.FIXED_25,
                        )
                    ].metrics.mean_common_expected_regret
                    for persona in primary_personas
                ]
                seed_effects.append(math.fsum(persona_effects) / len(persona_effects))
            mode_means.append(math.fsum(seed_effects) / len(seed_effects))
        expected_primary = math.fsum(mode_means) / len(mode_means)

        self.assertAlmostEqual(
            report.primary_contrast.interval.point_estimate,
            expected_primary,
        )
        self.assertEqual(report.primary_contrast.scope, "primary-macro")
        self.assertEqual(report.primary_contrast.comparator, Strategy.FIXED_25)
        self.assertEqual(len(report.macro_contrasts), len(Strategy) - 1)
        self.assertEqual(
            len(report.cell_contrasts),
            len(config.personas) * len(config.availability_modes) * (len(Strategy) - 1),
        )
        self.assertEqual(
            len(report.strategy_cells),
            len(config.personas) * len(config.availability_modes) * len(Strategy),
        )
        self.assertNotEqual(
            report.bootstrap.mode_indices_sha256[0][1],
            report.bootstrap.mode_indices_sha256[1][1],
        )
        stress_cells = [
            cell for cell in report.strategy_cells if cell.persona_id == "cyclic"
        ]
        self.assertTrue(stress_cells)
        self.assertTrue(all(not cell.persona_primary for cell in stress_cells))
        with self.assertRaisesRegex(ReportingInputError, "confidence"):
            replace(
                report.bootstrap,
                confidence=BOOTSTRAP_CONFIDENCE - 0.05,
            )
        with self.assertRaisesRegex(ReportingInputError, "duplicates"):
            replace(
                report,
                strategy_cells=(*report.strategy_cells, report.strategy_cells[0]),
            )

    def test_report_is_independent_of_result_row_order(self) -> None:
        forward = build_statistical_report(
            self.result,
            resample_count=32,
        )
        reverse = build_statistical_report(
            replace(
                self.result,
                cluster_summaries=tuple(reversed(self.result.cluster_summaries)),
                abrupt_traces=tuple(reversed(self.result.abrupt_traces)),
            ),
            resample_count=32,
        )

        self.assertEqual(forward, reverse)


class SourceProvenanceTests(unittest.TestCase):
    def _git(self, root: Path, *arguments: str) -> None:
        subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
        )

    def _temporary_repository(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        scratch = Path.cwd() / ".gworker"
        scratch.mkdir(mode=0o700, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="reporting-source-",
            dir=scratch,
        )
        root = Path(temporary.name)
        self._git(root, "init", "--quiet")
        self._git(root, "config", "user.name", LOCKED_AUTHOR_NAME)
        self._git(root, "config", "user.email", LOCKED_AUTHOR_EMAIL)
        source = root / "runner.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        self._git(root, "add", "runner.py")
        self._git(root, "commit", "--quiet", "-m", "Freeze runner fixture")
        return temporary, source

    def test_clean_source_capture_is_stable_and_source_bound(self) -> None:
        temporary, source = self._temporary_repository()
        self.addCleanup(temporary.cleanup)
        root = source.parent

        first = capture_source_provenance(
            root,
            loaded_sources=(("runner.py", source),),
        )
        repeated = capture_source_provenance(
            root,
            loaded_sources=(("runner.py", source),),
        )
        with patch.dict(
            os.environ,
            {"GIT_CONFIG_PARAMETERS": "malformed"},
        ):
            sanitized = capture_source_provenance(
                root,
                loaded_sources=(("runner.py", source),),
            )

        self.assertEqual(first, repeated)
        self.assertEqual(first, sanitized)
        self.assertTrue(first.clean_pre_run)
        self.assertEqual(first.author_name, LOCKED_AUTHOR_NAME)
        self.assertEqual(first.committer_email, LOCKED_AUTHOR_EMAIL)
        self.assertEqual(
            first.loaded_sources[0].sha256,
            hashlib.sha256(b"VALUE = 1\n").hexdigest(),
        )

        for changes in (
            {"clean_pre_run": False},
            {"git_object_format": "invalid"},
            {"source_commit": "0" * 39},
            {"source_archive_sha256": "x"},
            {"author_name": "Not Omar"},
            {"branch": "bad\nbranch"},
            {"loaded_sources": ()},
            {"loaded_sources": (*first.loaded_sources, *first.loaded_sources)},
            {"loaded_sources": cast("tuple[SourceFileIdentity, ...]", ("invalid",))},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises(ReportingInputError),
            ):
                replace(first, **changes)
        for relative_path, digest in (
            ("../runner.py", "0" * 64),
            ("runner.py", "x"),
        ):
            with self.assertRaises(ReportingInputError):
                SourceFileIdentity(
                    relative_path=relative_path,
                    sha256=digest,
                )

    def test_source_capture_rejects_dirty_or_mismatched_bytes(self) -> None:
        temporary, source = self._temporary_repository()
        self.addCleanup(temporary.cleanup)
        root = source.parent

        source.write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ReportingInputError, "clean"):
            capture_source_provenance(
                root,
                loaded_sources=(("runner.py", source),),
            )

        source.write_text("VALUE = 1\n", encoding="utf-8")
        outside = root / "other.py"
        outside.write_text("VALUE = 1\n", encoding="utf-8")
        self._git(root, "add", "other.py")
        self._git(root, "commit", "--quiet", "-m", "Add alternate source fixture")
        with self.assertRaisesRegex(
            ReportingInputError,
            "loaded source path",
        ):
            capture_source_provenance(
                root,
                loaded_sources=(("runner.py", outside),),
            )

    def test_source_capture_requires_real_declared_source_files(self) -> None:
        temporary, source = self._temporary_repository()
        self.addCleanup(temporary.cleanup)
        root = source.parent

        with self.assertRaisesRegex(ReportingInputError, "must not be empty"):
            capture_source_provenance(root)

        linked = root / "linked.py"
        linked.symlink_to("runner.py")
        self._git(root, "add", "linked.py")
        self._git(root, "commit", "--quiet", "-m", "Add symlink source fixture")
        with self.assertRaisesRegex(ReportingInputError, "symlink"):
            capture_source_provenance(
                root,
                loaded_sources=(("linked.py", linked),),
            )

    def test_source_capture_rejects_invalid_roots_and_declarations(self) -> None:
        temporary, source = self._temporary_repository()
        self.addCleanup(temporary.cleanup)
        root = source.parent
        nested = root / "nested"
        nested.mkdir()

        with self.assertRaises(ReportingInputError):
            capture_source_provenance(
                cast("Path", "not-a-path"),
                loaded_sources=(("runner.py", source),),
            )
        with self.assertRaisesRegex(ReportingInputError, "Git top level"):
            capture_source_provenance(
                nested,
                loaded_sources=(("runner.py", source),),
            )
        with self.assertRaisesRegex(ReportingInputError, "repeats"):
            capture_source_provenance(
                root,
                loaded_sources=(
                    ("runner.py", source),
                    ("runner.py", source),
                ),
            )
        with self.assertRaisesRegex(ReportingInputError, "must be a Path"):
            capture_source_provenance(
                root,
                loaded_sources=(("runner.py", cast("Path", "runner.py")),),
            )
        with self.assertRaisesRegex(ReportingInputError, "unsafe"):
            capture_source_provenance(
                root,
                loaded_sources=(("../runner.py", source),),
            )


if __name__ == "__main__":
    unittest.main()
