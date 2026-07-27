from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import ClassVar, cast

from gworker.evaluation import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PERSONAS,
    LOCKED_POLICY_ID,
    AvailabilityMode,
    ExperimentConfig,
    ExperimentResult,
    Strategy,
    run_experiment,
)
from gworker.evidence import (
    FIXTURE_EVIDENCE_KIND,
    LOCKED_EVIDENCE_KIND,
    PRIMARY_SCOPE,
    PUBLICATION_EVIDENCE_SCHEMA_VERSION,
    RECOVERY_CONTRAST_METRICS,
    REGISTERED_ENDPOINT,
    TRACE_INTERVAL_KIND,
    AdaptiveDiagnosticEvidence,
    CalibrationBinEvidence,
    EvidenceCardinalities,
    EvidenceInputError,
    EvidenceScope,
    PublicationEvidence,
    RecoveryContrastEvidence,
    RecoveryMetricContrast,
    RegisteredClaimEvidence,
    RunCompleteness,
    StrategyMetricEvidence,
    TemplateExposureEvidence,
    TracePointEvidence,
    build_fixture_publication_evidence,
    build_publication_evidence,
    expected_publication_cardinalities,
)
from gworker.reporting import IntervalEstimate, StatisticalReport


@dataclass(frozen=True, slots=True)
class MutableEvidenceScope(EvidenceScope):
    payload: list[str]


@dataclass(frozen=True, slots=True)
class MutableRecoveryMetric(RecoveryMetricContrast):
    payload: list[str]


@dataclass(frozen=True, slots=True)
class MutableIntervalEstimate(IntervalEstimate):
    payload: list[str]


def evidence_fixture_config() -> ExperimentConfig:
    return ExperimentConfig(
        split="test",
        environment_seeds=(2, 5, 11),
        policy_replicas=2,
        personas=(
            DEFAULT_PERSONAS[0],
            DEFAULT_PERSONAS[4],
            DEFAULT_PERSONAS[5],
            DEFAULT_PERSONAS[7],
        ),
        availability_modes=tuple(AvailabilityMode),
        horizon=24,
        drift_decision=13,
        recovery_block_size=2,
        recovery_blocks=2,
    )


class EvidenceCardinalityTests(unittest.TestCase):
    def test_locked_population_inventory_is_frozen_without_running_it(self) -> None:
        counts = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)

        self.assertEqual(counts.cluster_summaries, 16_128)
        self.assertEqual(counts.abrupt_traces, 3_584)
        self.assertEqual(counts.raw_trace_points, 1_032_192)
        self.assertEqual(counts.trajectories, 23_040)
        self.assertEqual(counts.decisions, 6_635_520)
        self.assertEqual(counts.scenario_strategy_rows, 126)
        self.assertEqual(counts.macro_strategy_rows, 7)
        self.assertEqual(counts.macro_contrasts, 6)
        self.assertEqual(counts.scenario_contrasts, 108)
        self.assertEqual(counts.adaptive_diagnostic_scopes, 19)
        self.assertEqual(counts.calibration_rows, 152)
        self.assertEqual(counts.template_exposure_rows, 532)
        self.assertEqual(counts.recovery_rows, 28)
        self.assertEqual(counts.recovery_contrast_rows, 24)
        self.assertEqual(counts.trace_series, 28)
        self.assertEqual(counts.trace_points, 8_064)

    def test_inventory_value_objects_fail_closed(self) -> None:
        expected = expected_publication_cardinalities(evidence_fixture_config())
        self.assertEqual(
            RunCompleteness(
                expected=expected,
                actual=expected,
                hard_failure_count=0,
            ).actual,
            expected,
        )
        with self.assertRaisesRegex(EvidenceInputError, "ExperimentConfig"):
            expected_publication_cardinalities(cast("ExperimentConfig", object()))
        with self.assertRaisesRegex(EvidenceInputError, "integer"):
            replace(expected, decisions=-1)
        with self.assertRaisesRegex(EvidenceInputError, "cardinalities"):
            RunCompleteness(
                expected=cast("EvidenceCardinalities", object()),
                actual=expected,
                hard_failure_count=0,
            )
        with self.assertRaisesRegex(EvidenceInputError, "hard_failure_count"):
            RunCompleteness(
                expected=expected,
                actual=expected,
                hard_failure_count=1,
            )
        with self.assertRaisesRegex(EvidenceInputError, "incomplete"):
            RunCompleteness(
                expected=expected,
                actual=replace(expected, decisions=expected.decisions - 1),
                hard_failure_count=0,
            )


class PublicationEvidenceTests(unittest.TestCase):
    config: ClassVar[ExperimentConfig]
    result: ClassVar[ExperimentResult]
    evidence: ClassVar[PublicationEvidence]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = evidence_fixture_config()
        cls.result = run_experiment(cls.config)
        cls.evidence = build_fixture_publication_evidence(
            cls.result,
            resample_count=16,
        )

    def test_builder_emits_the_complete_fixed_order_fixture(self) -> None:
        evidence = self.evidence
        counts = evidence.completeness.actual

        self.assertEqual(evidence.schema_version, PUBLICATION_EVIDENCE_SCHEMA_VERSION)
        self.assertEqual(evidence.evidence_kind, FIXTURE_EVIDENCE_KIND)
        self.assertEqual(evidence.policy_id, LOCKED_POLICY_ID)
        self.assertEqual(evidence.registered_claim.endpoint, REGISTERED_ENDPOINT)
        self.assertEqual(
            evidence.registered_claim.interval,
            evidence.statistics.primary_contrast.interval,
        )
        self.assertEqual(counts.cluster_summaries, 168)
        self.assertEqual(counts.abrupt_traces, 84)
        self.assertEqual(counts.raw_trace_points, 2_016)
        self.assertEqual(counts.trajectories, 192)
        self.assertEqual(counts.decisions, 4_608)
        self.assertEqual(counts.scenario_strategy_rows, 56)
        self.assertEqual(counts.macro_strategy_rows, 7)
        self.assertEqual(counts.scenario_contrasts, 48)
        self.assertEqual(counts.adaptive_diagnostic_scopes, 9)
        self.assertEqual(counts.calibration_rows, 72)
        self.assertEqual(counts.template_exposure_rows, 252)
        self.assertEqual(counts.recovery_rows, 28)
        self.assertEqual(counts.recovery_contrast_rows, 24)
        self.assertEqual(counts.trace_series, 28)
        self.assertEqual(counts.trace_points, 672)
        self.assertEqual(
            tuple(row.strategy for row in evidence.strategy_metrics[:7]),
            tuple(Strategy),
        )
        self.assertTrue(
            all(
                row.scope.kind == PRIMARY_SCOPE for row in evidence.strategy_metrics[:7]
            )
        )
        self.assertEqual(
            evidence.strategy_metrics[7].scope.persona_id,
            self.config.personas[0].persona_id,
        )

    def test_strategy_counts_and_adaptive_sufficient_statistics_reconcile(self) -> None:
        evidence = self.evidence
        macro_scope = evidence.strategy_metrics[0].scope
        adaptive_strategy = next(
            row
            for row in evidence.strategy_metrics
            if row.scope == macro_scope and row.strategy is Strategy.ADAPTIVE
        )
        diagnostic = evidence.adaptive_diagnostics[0]
        primary_persona_ids = {
            persona.persona_id for persona in self.config.personas if persona.primary
        }
        source = tuple(
            summary
            for summary in self.result.cluster_summaries
            if summary.persona_id in primary_persona_ids
            and summary.strategy is Strategy.ADAPTIVE
        )

        self.assertEqual(
            adaptive_strategy.trajectory_count,
            sum(summary.policy_replica_count for summary in source),
        )
        self.assertEqual(
            diagnostic.action_count,
            sum(summary.metrics.action_count for summary in source),
        )
        self.assertEqual(
            diagnostic.low_propensity_count,
            sum(summary.metrics.low_propensity_count or 0 for summary in source),
        )
        self.assertAlmostEqual(
            diagnostic.inverse_propensity_ess_ratio,
            diagnostic.inverse_propensity_sum**2
            / (
                diagnostic.selected_propensity_count
                * diagnostic.inverse_propensity_squared_sum
            ),
        )
        macro_bins = tuple(
            row for row in evidence.calibration_bins if row.scope == macro_scope
        )
        self.assertEqual(
            sum(row.count for row in macro_bins),
            diagnostic.arm_probability_count,
        )
        self.assertAlmostEqual(
            math.fsum(row.predicted_sum for row in macro_bins),
            diagnostic.selected_propensity_count,
        )
        self.assertAlmostEqual(
            math.fsum(row.observed_sum for row in macro_bins),
            diagnostic.selected_propensity_count,
        )

    def test_exposures_decomposition_and_recovery_are_source_backed(self) -> None:
        evidence = self.evidence
        for strategy_row in evidence.strategy_metrics:
            exposures = tuple(
                row
                for row in evidence.template_exposures
                if row.scope == strategy_row.scope
                and row.strategy is strategy_row.strategy
            )
            self.assertEqual(
                sum(row.exposure_count for row in exposures),
                strategy_row.action_count,
            )
            self.assertAlmostEqual(
                strategy_row.mean_common_expected_regret,
                strategy_row.mean_conditional_expected_regret
                + strategy_row.mean_path_opportunity_cost,
            )
            self.assertAlmostEqual(
                strategy_row.mean_cumulative_common_expected_regret,
                strategy_row.mean_common_expected_regret * self.config.horizon,
            )

        for row in evidence.recovery_metrics:
            self.assertEqual(row.censor_lag, 13)
            self.assertEqual(row.maximum_recovered_lag, 8)
            self.assertAlmostEqual(
                row.recovery_rate,
                row.recovered_count / row.trajectory_count,
            )
            self.assertAlmostEqual(
                row.conservative_recovery_lag,
                (
                    row.recovered_lag_sum
                    + (row.trajectory_count - row.recovered_count) * row.censor_lag
                )
                / row.trajectory_count,
            )

    def test_recovery_contrasts_and_trace_bands_are_complete(self) -> None:
        evidence = self.evidence
        first_contrast = evidence.recovery_contrasts[0]
        self.assertEqual(
            tuple(metric.metric for metric in first_contrast.metrics),
            RECOVERY_CONTRAST_METRICS,
        )
        self.assertTrue(
            all(
                metric.interval.stratum_count == 1
                and metric.interval.seed_count_per_stratum
                == len(self.config.environment_seeds)
                for contrast in evidence.recovery_contrasts
                for metric in contrast.metrics
            )
        )
        first_series = evidence.trace_points[: self.config.horizon]
        self.assertEqual(
            tuple(row.decision for row in first_series),
            tuple(range(1, self.config.horizon + 1)),
        )
        self.assertTrue(
            all(row.interval_kind == TRACE_INTERVAL_KIND for row in first_series)
        )
        trace = next(
            trace
            for trace in self.result.abrupt_traces
            if (
                trace.persona_id,
                trace.availability_mode,
                trace.strategy,
            )
            == (
                first_series[0].persona_id,
                first_series[0].availability_mode,
                first_series[0].strategy,
            )
        )
        same_series = tuple(
            candidate
            for candidate in self.result.abrupt_traces
            if (
                candidate.persona_id,
                candidate.availability_mode,
                candidate.strategy,
            )
            == (
                trace.persona_id,
                trace.availability_mode,
                trace.strategy,
            )
        )
        self.assertAlmostEqual(
            first_series[0].interval.point_estimate,
            math.fsum(item.common_expected_regret[0] for item in same_series)
            / len(same_series),
        )

    def test_input_row_order_does_not_change_evidence(self) -> None:
        reversed_result = replace(
            self.result,
            cluster_summaries=tuple(reversed(self.result.cluster_summaries)),
            abrupt_traces=tuple(reversed(self.result.abrupt_traces)),
        )

        self.assertEqual(
            build_fixture_publication_evidence(
                reversed_result,
                resample_count=16,
            ),
            self.evidence,
        )

    def test_locked_builder_rejects_fixture_data_before_reporting(self) -> None:
        with self.assertRaisesRegex(EvidenceInputError, "exact locked"):
            build_publication_evidence(self.result)
        with self.assertRaisesRegex(EvidenceInputError, "ExperimentResult"):
            build_publication_evidence(cast("ExperimentResult", object()))
        with self.assertRaisesRegex(EvidenceInputError, "ExperimentResult"):
            build_fixture_publication_evidence(
                cast("ExperimentResult", object()),
                resample_count=4,
            )
        with self.assertRaisesRegex(EvidenceInputError, "fixture"):
            build_fixture_publication_evidence(
                replace(
                    self.result,
                    config=replace(self.config, split="eval"),
                ),
                resample_count=4,
            )

    def test_top_level_identity_and_collection_tampering_is_rejected(self) -> None:
        evidence = self.evidence
        altered_exposure = replace(
            evidence.template_exposures[0],
            focus_seconds=evidence.template_exposures[0].focus_seconds + 60,
        )
        invalid_changes = (
            {"schema_version": "invalid"},
            {"evidence_kind": "invalid"},
            {"evidence_kind": LOCKED_EVIDENCE_KIND},
            {"result_schema_version": "invalid"},
            {"config": cast("ExperimentConfig", object())},
            {"evaluator_id": "invalid"},
            {"design_id": "invalid"},
            {"population_id": "invalid"},
            {"policy_id": "invalid"},
            {"completeness": cast("RunCompleteness", object())},
            {"statistics": cast("StatisticalReport", object())},
            {"strategy_metrics": evidence.strategy_metrics[1:]},
            {"adaptive_diagnostics": evidence.adaptive_diagnostics[1:]},
            {"calibration_bins": evidence.calibration_bins[1:]},
            {"template_exposures": evidence.template_exposures[1:]},
            {
                "template_exposures": (
                    altered_exposure,
                    *evidence.template_exposures[1:],
                )
            },
            {"recovery_metrics": evidence.recovery_metrics[1:]},
            {"recovery_contrasts": evidence.recovery_contrasts[1:]},
            {"trace_points": evidence.trace_points[1:]},
        )
        for changes in invalid_changes:
            with (
                self.subTest(changes=tuple(changes)),
                self.assertRaises(EvidenceInputError),
            ):
                replace(evidence, **changes)

        copied = replace(
            evidence,
            strategy_metrics=list(evidence.strategy_metrics),  # type: ignore[arg-type]
        )
        self.assertIsInstance(copied.strategy_metrics, tuple)

    def test_scope_and_strategy_value_objects_reject_malformed_values(self) -> None:
        macro = self.evidence.strategy_metrics[0].scope
        scenario = self.evidence.strategy_metrics[7].scope
        for macro_changes in (
            {"kind": "invalid"},
            {"persona_id": "unexpected"},
            {"availability_mode": AvailabilityMode.UNCONSTRAINED},
        ):
            with (
                self.subTest(changes=macro_changes),
                self.assertRaises(EvidenceInputError),
            ):
                replace(macro, **macro_changes)
        for scenario_changes in (
            {"persona_id": ""},
            {"persona_primary": cast("bool", 1)},
            {"availability_mode": cast("AvailabilityMode", "invalid")},
        ):
            with (
                self.subTest(changes=scenario_changes),
                self.assertRaises(EvidenceInputError),
            ):
                replace(scenario, **scenario_changes)

        strategy = self.evidence.strategy_metrics[0]
        invalid_strategy_changes = (
            {"scope": cast("EvidenceScope", object())},
            {"strategy": cast("Strategy", "invalid")},
            {"trajectory_count": 0},
            {"action_count": strategy.action_count + 1},
            {"mean_expected_reward": math.nan},
            {"mean_expected_reward": 10**1_000},
            {"mean_expected_reward": 1.1},
            {"mean_cumulative_common_expected_regret": strategy.horizon + 1.0},
            {"mean_common_oracle_distance": 4.0},
            {"common_regret_standard_error": -0.1},
            {"common_regret_standard_error": None},
            {"availability_guardrail_count": strategy.action_count + 1},
            {"mean_feasible_set_size": 0.5},
            {"feasible_set_size_sum": strategy.feasible_set_size_sum + 0.5},
            {"maximum_arm_transition": 4},
            {"mean_realized_reward": strategy.mean_realized_reward + 0.001},
            {
                "mean_common_expected_regret": (
                    strategy.mean_common_expected_regret + 0.01
                )
            },
            {
                "mean_cumulative_common_expected_regret": (
                    strategy.mean_cumulative_common_expected_regret + 0.01
                )
            },
            {"availability_guardrail_rate": 1.0},
            {"feasible_set_size_sum": strategy.feasible_set_size_sum + 1.0},
        )
        for strategy_changes in invalid_strategy_changes:
            with (
                self.subTest(changes=strategy_changes),
                self.assertRaises(EvidenceInputError),
            ):
                replace(strategy, **strategy_changes)
        baseline = next(
            row
            for row in self.evidence.strategy_metrics
            if row.scope.kind != PRIMARY_SCOPE and row.strategy is Strategy.FIXED_25
        )
        with self.assertRaisesRegex(EvidenceInputError, "adaptive reviews"):
            replace(
                baseline,
                review_count=1,
                review_rate=1 / baseline.action_count,
            )
        scenario_guardrailed = next(
            row
            for row in self.evidence.strategy_metrics
            if row.scope.kind != PRIMARY_SCOPE
            and row.scope.availability_mode is AvailabilityMode.GUARDRAILED
            and row.strategy is Strategy.FIXED_25
        )
        changed_guardrail_count = scenario_guardrailed.availability_guardrail_count - 1
        with self.assertRaisesRegex(EvidenceInputError, "frozen schedule"):
            replace(
                scenario_guardrailed,
                availability_guardrail_count=changed_guardrail_count,
                availability_guardrail_rate=(
                    changed_guardrail_count / scenario_guardrailed.action_count
                ),
            )
        scenario_unconstrained = next(
            row
            for row in self.evidence.strategy_metrics
            if row.scope.kind != PRIMARY_SCOPE
            and row.scope.availability_mode is AvailabilityMode.UNCONSTRAINED
            and row.strategy is Strategy.ADAPTIVE
        )
        with self.assertRaisesRegex(EvidenceInputError, "maximum arm transition"):
            replace(scenario_unconstrained, maximum_arm_transition=2)
        myopic = next(
            row
            for row in self.evidence.strategy_metrics
            if row.scope.kind != PRIMARY_SCOPE
            and row.strategy is Strategy.MYOPIC_ORACLE
        )
        with self.assertRaisesRegex(EvidenceInputError, "myopic oracle"):
            replace(myopic, mean_conditional_oracle_distance=0.5)

    def test_diagnostic_and_calibration_values_reject_tampering(self) -> None:
        diagnostic = self.evidence.adaptive_diagnostics[0]
        invalid_diagnostic_changes = (
            {"scope": cast("EvidenceScope", object())},
            {"action_count": 0},
            {"selected_propensity_count": (diagnostic.selected_propensity_count - 1)},
            {"brier_score_count": diagnostic.brier_score_count - 1},
            {"review_count": diagnostic.action_count + 1},
            {"low_propensity_count": (diagnostic.selected_propensity_count + 1)},
            {"probability_floor_count": (diagnostic.arm_probability_count + 1)},
            {"exact_evidence_count": diagnostic.exact_evidence_count + 1},
            {"inverse_propensity_sum": 0.0},
            {"minimum_propensity": 0.0},
            {"brier_score_sum": diagnostic.brier_score_count * 3.0},
            {"review_rate": 1.1},
            {"maximum_inverse_propensity": 1.0},
            {"low_propensity_rate": 1.0},
            {
                "minimum_propensity": 0.04,
                "maximum_inverse_propensity": 25.0,
                "low_propensity_count": 1,
                "low_propensity_rate": 1 / diagnostic.selected_propensity_count,
                "inverse_propensity_sum": float(diagnostic.selected_propensity_count),
                "inverse_propensity_squared_sum": float(
                    diagnostic.selected_propensity_count
                ),
                "inverse_propensity_ess_ratio": 1.0,
            },
        )
        for diagnostic_changes in invalid_diagnostic_changes:
            with (
                self.subTest(changes=diagnostic_changes),
                self.assertRaises(EvidenceInputError),
            ):
                replace(diagnostic, **diagnostic_changes)
        high_valid_brier = replace(
            diagnostic,
            brier_score_sum=1.5 * diagnostic.brier_score_count,
            multiclass_brier_score=1.5,
        )
        self.assertEqual(high_valid_brier.multiclass_brier_score, 1.5)

        calibration = next(
            row for row in self.evidence.calibration_bins if row.count > 0
        )
        invalid_calibration_changes = (
            {"scope": cast("EvidenceScope", object())},
            {"bin_index": 8},
            {"lower_bound": 0.5},
            {"upper_inclusive": not calibration.upper_inclusive},
            {"count": -1},
            {"predicted_sum": calibration.count + 1.0},
            {"observed_sum": calibration.observed_sum + 0.5},
            {"mean_predicted_probability": None},
        )
        for calibration_changes in invalid_calibration_changes:
            with (
                self.subTest(changes=calibration_changes),
                self.assertRaises(EvidenceInputError),
            ):
                replace(calibration, **calibration_changes)

        empty = CalibrationBinEvidence(
            scope=calibration.scope,
            bin_index=calibration.bin_index,
            lower_bound=calibration.lower_bound,
            upper_bound=calibration.upper_bound,
            upper_inclusive=calibration.upper_inclusive,
            predicted_sum=0.0,
            observed_sum=0.0,
            count=0,
            mean_predicted_probability=None,
            observed_frequency=None,
        )
        with self.assertRaisesRegex(EvidenceInputError, "empty"):
            replace(empty, observed_frequency=0.0)
        with self.assertRaisesRegex(EvidenceInputError, "outside its calibration bin"):
            CalibrationBinEvidence(
                scope=calibration.scope,
                bin_index=1,
                lower_bound=0.025,
                upper_bound=0.05,
                upper_inclusive=False,
                predicted_sum=0.0,
                observed_sum=0.0,
                count=1,
                mean_predicted_probability=0.0,
                observed_frequency=0.0,
            )

    def test_coordinated_cross_row_tampering_is_rejected(self) -> None:
        evidence = self.evidence
        macro = evidence.strategy_metrics[0]
        delta = 0.001
        altered_macro = replace(
            macro,
            mean_common_expected_regret=(macro.mean_common_expected_regret + delta),
            mean_conditional_expected_regret=(
                macro.mean_conditional_expected_regret + delta
            ),
            mean_cumulative_common_expected_regret=(
                macro.mean_cumulative_common_expected_regret + delta * macro.horizon
            ),
            mean_cumulative_conditional_expected_regret=(
                macro.mean_cumulative_conditional_expected_regret
                + delta * macro.horizon
            ),
        )
        with self.assertRaisesRegex(EvidenceInputError, "macro strategy"):
            replace(
                evidence,
                strategy_metrics=(
                    altered_macro,
                    *evidence.strategy_metrics[1:],
                ),
            )

        diagnostic = evidence.adaptive_diagnostics[0]
        altered_arm_count = diagnostic.arm_probability_count + 1
        altered_diagnostic = replace(
            diagnostic,
            arm_probability_count=altered_arm_count,
            probability_floor_rate=(
                diagnostic.probability_floor_count / altered_arm_count
            ),
        )
        calibration = evidence.calibration_bins[0]
        altered_bin_count = calibration.count + 1
        altered_calibration = replace(
            calibration,
            count=altered_bin_count,
            mean_predicted_probability=(calibration.predicted_sum / altered_bin_count),
            observed_frequency=calibration.observed_sum / altered_bin_count,
        )
        with self.assertRaisesRegex(
            EvidenceInputError,
            "arm probability count",
        ):
            replace(
                evidence,
                adaptive_diagnostics=(
                    altered_diagnostic,
                    *evidence.adaptive_diagnostics[1:],
                ),
                calibration_bins=(
                    altered_calibration,
                    *evidence.calibration_bins[1:],
                ),
            )

        recovery_contrast = evidence.recovery_contrasts[0]
        recovery_metric = recovery_contrast.metrics[0]
        altered_recovery_metric = replace(
            recovery_metric,
            interval=replace(
                recovery_metric.interval,
                seed_count_per_stratum=1,
                seed_cluster_standard_error=None,
            ),
        )
        altered_recovery_contrast = replace(
            recovery_contrast,
            metrics=(
                altered_recovery_metric,
                *recovery_contrast.metrics[1:],
            ),
        )
        with self.assertRaisesRegex(EvidenceInputError, "population"):
            replace(
                evidence,
                recovery_contrasts=(
                    altered_recovery_contrast,
                    *evidence.recovery_contrasts[1:],
                ),
            )

    def test_exposure_recovery_trace_and_claim_values_fail_closed(self) -> None:
        exposure = self.evidence.template_exposures[0]
        for exposure_changes in (
            {"scope": cast("EvidenceScope", object())},
            {"strategy": cast("Strategy", "invalid")},
            {"template_id": ""},
            {"focus_seconds": 0},
            {"exposure_count": exposure.action_count + 1},
            {"exposure_rate": 1.1},
            {"exposure_rate": 0.5},
        ):
            with (
                self.subTest(changes=exposure_changes),
                self.assertRaises(EvidenceInputError),
            ):
                replace(exposure, **exposure_changes)

        recovery = self.evidence.recovery_metrics[0]
        for recovery_changes in (
            {"persona_id": ""},
            {"availability_mode": cast("AvailabilityMode", "invalid")},
            {"strategy": cast("Strategy", "invalid")},
            {"trajectory_count": 0},
            {"pre_drift_regret": 1.1},
            {"early_post_drift_auc": -1.0},
            {"recovered_count": recovery.trajectory_count + 1},
            {"recovered_lag_sum": -1.0},
            {"maximum_recovered_lag": recovery.censor_lag},
            {"recovery_rate": 1.1},
            {"conservative_recovery_lag": recovery.censor_lag + 1.0},
            {"recovery_rate": 0.123},
            {"conservative_recovery_lag": (recovery.conservative_recovery_lag + 0.5)},
        ):
            with (
                self.subTest(changes=recovery_changes),
                self.assertRaises(EvidenceInputError),
            ):
                replace(recovery, **recovery_changes)
        recovered = next(
            row for row in self.evidence.recovery_metrics if row.recovered_count
        )
        with self.assertRaisesRegex(EvidenceInputError, "integer-valued"):
            replace(
                recovered,
                recovered_lag_sum=recovered.recovered_lag_sum + 0.5,
            )

        contrast = self.evidence.recovery_contrasts[0]
        metric = contrast.metrics[0]
        with self.assertRaisesRegex(EvidenceInputError, "metric"):
            replace(metric, metric="invalid")
        wrong_relation = next(
            relation
            for relation in ("below-zero", "includes-zero", "above-zero")
            if relation != metric.interval_relation_to_zero
        )
        with self.assertRaisesRegex(EvidenceInputError, "relation"):
            replace(metric, interval_relation_to_zero=wrong_relation)
        with self.assertRaisesRegex(EvidenceInputError, "comparator"):
            replace(contrast, comparator=Strategy.ADAPTIVE)
        with self.assertRaisesRegex(EvidenceInputError, "reordered"):
            replace(contrast, metrics=tuple(reversed(contrast.metrics)))

        trace = self.evidence.trace_points[0]
        with self.assertRaisesRegex(EvidenceInputError, "decision"):
            replace(trace, decision=0)
        with self.assertRaisesRegex(EvidenceInputError, "kind"):
            replace(trace, interval_kind="simultaneous")
        with self.assertRaisesRegex(EvidenceInputError, "outside"):
            replace(
                trace,
                interval=replace(trace.interval, point_estimate=1.1),
            )
        with self.assertRaisesRegex(EvidenceInputError, "trace interval"):
            replace(
                trace,
                interval=replace(
                    trace.interval,
                    point_estimate=0.9,
                ),
            )

        claim = self.evidence.registered_claim
        with self.assertRaisesRegex(EvidenceInputError, "endpoint"):
            replace(claim, endpoint="post-hoc")
        with self.assertRaisesRegex(EvidenceInputError, "status"):
            replace(claim, status="significant")
        opposite = IntervalEstimate(
            point_estimate=0.2,
            lower_95=0.1,
            upper_95=0.3,
            seed_cluster_standard_error=0.01,
            seed_count_per_stratum=3,
            stratum_count=1,
        )
        with self.assertRaisesRegex(EvidenceInputError, "disagrees"):
            RegisteredClaimEvidence(
                endpoint=REGISTERED_ENDPOINT,
                status="adaptive-lower-regret",
                interval=opposite,
            )
        zero_exposure = next(
            row for row in self.evidence.template_exposures if row.exposure_count == 0
        )
        normalized = replace(zero_exposure, exposure_rate=-0.0)
        self.assertEqual(math.copysign(1.0, normalized.exposure_rate), 1.0)

    def test_collection_members_are_defensively_copied_and_typed(self) -> None:
        contrast = self.evidence.recovery_contrasts[0]
        copied = replace(
            contrast,
            metrics=list(contrast.metrics),  # type: ignore[arg-type]
        )
        self.assertIsInstance(copied.metrics, tuple)
        with self.assertRaisesRegex(EvidenceInputError, "incomplete|reordered"):
            replace(
                contrast,
                metrics=cast(
                    "tuple[RecoveryMetricContrast, ...]",
                    ("invalid",) * len(RECOVERY_CONTRAST_METRICS),
                ),
            )
        with self.assertRaisesRegex(EvidenceInputError, "finite"):
            replace(
                self.evidence,
                strategy_metrics=cast(
                    "tuple[StrategyMetricEvidence, ...]",
                    1,
                ),
            )
        strategy_row = self.evidence.strategy_metrics[0]
        mutable_duck = SimpleNamespace(
            **{
                definition.name: getattr(strategy_row, definition.name)
                for definition in fields(strategy_row)
            }
        )
        with self.assertRaisesRegex(EvidenceInputError, "invalid member"):
            replace(
                self.evidence,
                strategy_metrics=cast(
                    "tuple[StrategyMetricEvidence, ...]",
                    (mutable_duck, *self.evidence.strategy_metrics[1:]),
                ),
            )
        mutable_scope = MutableEvidenceScope(
            kind=strategy_row.scope.kind,
            persona_id=strategy_row.scope.persona_id,
            persona_primary=strategy_row.scope.persona_primary,
            availability_mode=strategy_row.scope.availability_mode,
            payload=[],
        )
        with self.assertRaisesRegex(EvidenceInputError, "scope"):
            replace(strategy_row, scope=mutable_scope)

        metric = contrast.metrics[0]
        mutable_metric = MutableRecoveryMetric(
            metric=metric.metric,
            interval=metric.interval,
            interval_relation_to_zero=metric.interval_relation_to_zero,
            payload=[],
        )
        with self.assertRaisesRegex(EvidenceInputError, "incomplete|reordered"):
            replace(
                contrast,
                metrics=(mutable_metric, *contrast.metrics[1:]),
            )
        mutable_interval = MutableIntervalEstimate(
            point_estimate=metric.interval.point_estimate,
            lower_95=metric.interval.lower_95,
            upper_95=metric.interval.upper_95,
            seed_cluster_standard_error=(metric.interval.seed_cluster_standard_error),
            seed_count_per_stratum=metric.interval.seed_count_per_stratum,
            stratum_count=metric.interval.stratum_count,
            payload=[],
        )
        with self.assertRaisesRegex(EvidenceInputError, "interval"):
            replace(metric, interval=mutable_interval)
        for invalid_type in (
            cast("AdaptiveDiagnosticEvidence", object()),
            cast("CalibrationBinEvidence", object()),
            cast("TemplateExposureEvidence", object()),
            cast("RecoveryContrastEvidence", object()),
            cast("TracePointEvidence", object()),
        ):
            self.assertIsNotNone(invalid_type)


if __name__ == "__main__":
    unittest.main()
