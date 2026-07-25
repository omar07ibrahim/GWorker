from __future__ import annotations

import copy
import json
import math
import unittest
from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar, cast
from unittest.mock import patch

import gworker.publication_codec as codec
from gworker.evaluation import (
    DEFAULT_PERSONAS,
    AvailabilityMode,
    ExperimentConfig,
    ExperimentResult,
    Strategy,
    evaluator_fingerprint,
    run_experiment,
)
from gworker.evidence import (
    PublicationEvidence,
    build_fixture_publication_evidence,
)
from gworker.publication_codec import (
    PUBLICATION_EVIDENCE_DOCUMENT_TYPE,
    PUBLICATION_JSON_CODEC_VERSION,
    STATISTICAL_REPORT_DOCUMENT_TYPE,
    CanonicalJsonDocument,
    PublicationCodecError,
    compute_content_sha256,
    decode_publication_evidence,
    decode_statistical_report,
    encode_publication_evidence,
    encode_statistical_report,
)
from gworker.reporting import (
    BootstrapMetadata,
    StatisticalReport,
    build_bootstrap_plan,
    build_statistical_report,
)


def publication_fixture_config() -> ExperimentConfig:
    return ExperimentConfig(
        split="test",
        environment_seeds=(2, 5),
        policy_replicas=2,
        personas=(DEFAULT_PERSONAS[0], DEFAULT_PERSONAS[4]),
        availability_modes=tuple(AvailabilityMode),
        horizon=24,
        drift_decision=13,
        recovery_block_size=2,
        recovery_blocks=2,
    )


def parsed_document(document: CanonicalJsonDocument) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(document.content))


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def mutate_document(
    document: CanonicalJsonDocument,
    mutation: Callable[[dict[str, Any]], None],
) -> tuple[bytes, str]:
    value = parsed_document(document)
    mutation(value)
    content = canonical_bytes(value)
    return content, compute_content_sha256(content)


class PublicationJsonCodecTests(unittest.TestCase):
    result: ClassVar[ExperimentResult]
    report: ClassVar[StatisticalReport]
    evidence: ClassVar[PublicationEvidence]
    report_document: ClassVar[CanonicalJsonDocument]
    evidence_document: ClassVar[CanonicalJsonDocument]

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_experiment(publication_fixture_config())
        cls.report = build_statistical_report(cls.result, resample_count=8)
        cls.evidence = build_fixture_publication_evidence(
            cls.result,
            resample_count=8,
        )
        cls.report_document = encode_statistical_report(
            cls.report,
            config=cls.result.config,
        )
        cls.evidence_document = encode_publication_evidence(cls.evidence)

    def test_statistical_report_round_trip_is_deterministic(self) -> None:
        repeated = encode_statistical_report(
            self.report,
            config=self.result.config,
        )
        decoded = decode_statistical_report(
            self.report_document.content,
            expected_content_sha256=self.report_document.content_sha256,
            config=self.result.config,
        )

        self.assertEqual(decoded, self.report)
        self.assertEqual(repeated, self.report_document)
        self.assertEqual(
            self.report_document.content_sha256,
            compute_content_sha256(self.report_document.content),
        )
        self.assertLess(len(self.report_document.content), codec._MAX_REPORT_BYTES)
        self.assertTrue(self.report_document.content.startswith(b'{"codec_version":'))
        self.assertNotIn(b"\n", self.report_document.content)
        self.assertNotIn(b": ", self.report_document.content)

    def test_publication_evidence_round_trip_is_deterministic(self) -> None:
        repeated = encode_publication_evidence(self.evidence)
        decoded = decode_publication_evidence(
            self.evidence_document.content,
            expected_content_sha256=self.evidence_document.content_sha256,
        )

        self.assertEqual(decoded, self.evidence)
        self.assertEqual(repeated, self.evidence_document)
        self.assertLess(
            len(self.evidence_document.content),
            codec._MAX_EVIDENCE_BYTES,
        )
        envelope = parsed_document(self.evidence_document)
        self.assertEqual(
            envelope["codec_version"],
            PUBLICATION_JSON_CODEC_VERSION,
        )
        self.assertEqual(
            envelope["document_type"],
            PUBLICATION_EVIDENCE_DOCUMENT_TYPE,
        )

    def test_content_sha_is_exact_and_checked_before_json_decode(self) -> None:
        wrong_sha256 = "0" * 64
        if wrong_sha256 == self.report_document.content_sha256:
            wrong_sha256 = "1" * 64
        with (
            patch.object(codec, "_parse_json") as parse_json,
            self.assertRaisesRegex(PublicationCodecError, "SHA-256"),
        ):
            decode_statistical_report(
                self.report_document.content,
                expected_content_sha256=wrong_sha256,
                config=self.result.config,
            )
        parse_json.assert_not_called()

        with self.assertRaisesRegex(PublicationCodecError, "lowercase SHA-256"):
            decode_statistical_report(
                self.report_document.content,
                expected_content_sha256="INVALID",
                config=self.result.config,
            )
        with self.assertRaisesRegex(PublicationCodecError, "does not match"):
            CanonicalJsonDocument(
                content=self.report_document.content,
                content_sha256=wrong_sha256,
            )
        with self.assertRaisesRegex(PublicationCodecError, "exact bytes"):
            compute_content_sha256(cast(bytes, bytearray(self.report_document.content)))

    def test_rejects_duplicate_unknown_and_missing_keys(self) -> None:
        duplicate = self.report_document.content.replace(
            b'{"codec_version":',
            b'{"codec_version":"duplicate","codec_version":',
            1,
        )
        with self.assertRaisesRegex(PublicationCodecError, "duplicate JSON key"):
            decode_statistical_report(
                duplicate,
                expected_content_sha256=compute_content_sha256(duplicate),
                config=self.result.config,
            )

        def add_unknown(value: dict[str, Any]) -> None:
            value["payload"]["unknown"] = 1

        def remove_required(value: dict[str, Any]) -> None:
            del value["payload"]["policy_id"]

        for mutation, message in (
            (add_unknown, "unknown keys"),
            (remove_required, "missing keys"),
        ):
            content, digest = mutate_document(self.report_document, mutation)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(PublicationCodecError, message),
            ):
                decode_statistical_report(
                    content,
                    expected_content_sha256=digest,
                    config=self.result.config,
                )

    def test_rejects_unknown_or_missing_envelope_keys(self) -> None:
        def add_unknown(value: dict[str, Any]) -> None:
            value["extra"] = None

        def remove_required(value: dict[str, Any]) -> None:
            del value["document_type"]

        for mutation, message in (
            (add_unknown, "envelope has unknown"),
            (remove_required, "envelope is missing"),
        ):
            content, digest = mutate_document(self.report_document, mutation)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(PublicationCodecError, message),
            ):
                decode_statistical_report(
                    content,
                    expected_content_sha256=digest,
                    config=self.result.config,
                )

    def test_rejects_wrong_version_type_and_cross_document_decode(self) -> None:
        def wrong_version(value: dict[str, Any]) -> None:
            value["codec_version"] = "wrong-version"

        def wrong_type(value: dict[str, Any]) -> None:
            value["document_type"] = "wrong-type"

        for mutation, message in (
            (wrong_version, "codec version"),
            (wrong_type, "document type"),
        ):
            content, digest = mutate_document(self.report_document, mutation)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(PublicationCodecError, message),
            ):
                decode_statistical_report(
                    content,
                    expected_content_sha256=digest,
                    config=self.result.config,
                )

        with self.assertRaisesRegex(PublicationCodecError, "document type"):
            decode_publication_evidence(
                self.report_document.content,
                expected_content_sha256=self.report_document.content_sha256,
            )
        report_envelope = parsed_document(self.report_document)
        self.assertEqual(
            report_envelope["document_type"],
            STATISTICAL_REPORT_DOCUMENT_TYPE,
        )

    def test_rejects_noncanonical_whitespace_order_and_negative_zero(self) -> None:
        whitespace = b" " + self.report_document.content
        reordered_value = parsed_document(self.report_document)
        reordered = json.dumps(
            {
                "payload": reordered_value["payload"],
                "document_type": reordered_value["document_type"],
                "codec_version": reordered_value["codec_version"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")
        self.assertIn(b":0.0,", self.report_document.content)
        negative_zero = self.report_document.content.replace(
            b":0.0,",
            b":-0.0,",
            1,
        )
        for content in (whitespace, reordered, negative_zero):
            with (
                self.subTest(content=content[:40]),
                self.assertRaisesRegex(PublicationCodecError, "not canonical"),
            ):
                decode_statistical_report(
                    content,
                    expected_content_sha256=compute_content_sha256(content),
                    config=self.result.config,
                )

    def test_encoder_canonicalizes_negative_zero(self) -> None:
        altered = copy.deepcopy(self.report)
        myopic = next(
            cell
            for cell in altered.strategy_cells
            if cell.strategy is Strategy.MYOPIC_ORACLE
        )
        self.assertEqual(myopic.mean_conditional_expected_regret, 0.0)
        object.__setattr__(
            myopic,
            "mean_conditional_expected_regret",
            -0.0,
        )

        self.assertEqual(
            encode_statistical_report(
                altered,
                config=self.result.config,
            ),
            self.report_document,
        )

    def test_rejects_bool_int_float_confusion_and_invalid_enums(self) -> None:
        def bool_for_int(value: dict[str, Any]) -> None:
            value["payload"]["bootstrap"]["resample_count"] = True

        def int_for_float(value: dict[str, Any]) -> None:
            value["payload"]["bootstrap"]["confidence"] = 1

        def int_for_bool(value: dict[str, Any]) -> None:
            value["payload"]["strategy_cells"][0]["persona_primary"] = 1

        def invalid_enum(value: dict[str, Any]) -> None:
            value["payload"]["strategy_cells"][0]["strategy"] = "unknown"

        cases = (
            (bool_for_int, "exact integer"),
            (int_for_float, "exact float"),
            (int_for_bool, "exact boolean"),
            (invalid_enum, "invalid enum"),
        )
        for mutation, message in cases:
            content, digest = mutate_document(self.report_document, mutation)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(PublicationCodecError, message),
            ):
                decode_statistical_report(
                    content,
                    expected_content_sha256=digest,
                    config=self.result.config,
                )

    def test_rejects_semantically_invalid_but_well_typed_rows(self) -> None:
        def wrong_confidence(value: dict[str, Any]) -> None:
            value["payload"]["bootstrap"]["confidence"] = 0.5

        content, digest = mutate_document(
            self.report_document,
            wrong_confidence,
        )
        with self.assertRaisesRegex(PublicationCodecError, "contract"):
            decode_statistical_report(
                content,
                expected_content_sha256=digest,
                config=self.result.config,
            )

    def test_report_context_binds_bootstrap_and_canonical_row_order(self) -> None:
        def missing_mode_digest(value: dict[str, Any]) -> None:
            value["payload"]["bootstrap"]["mode_indices_sha256"].pop()

        def wrong_namespace(value: dict[str, Any]) -> None:
            value["payload"]["bootstrap"]["namespace_sha256"] = "0" * 64

        def wrong_combined_digest(value: dict[str, Any]) -> None:
            value["payload"]["bootstrap"]["indices_sha256"] = "0" * 64

        def wrong_mode_digest(value: dict[str, Any]) -> None:
            value["payload"]["bootstrap"]["mode_indices_sha256"][0][1] = "0" * 64

        def reversed_mode_digests(value: dict[str, Any]) -> None:
            value["payload"]["bootstrap"]["mode_indices_sha256"].reverse()

        def reversed_cell_contrasts(value: dict[str, Any]) -> None:
            value["payload"]["cell_contrasts"].reverse()

        def reversed_strategy_cells(value: dict[str, Any]) -> None:
            value["payload"]["strategy_cells"].reverse()

        for mutation in (
            missing_mode_digest,
            wrong_namespace,
            wrong_combined_digest,
            wrong_mode_digest,
            reversed_mode_digests,
            reversed_cell_contrasts,
            reversed_strategy_cells,
        ):
            content, digest = mutate_document(self.report_document, mutation)
            with (
                self.subTest(mutation=mutation.__name__),
                self.assertRaisesRegex(
                    PublicationCodecError,
                    "bootstrap disagrees|canonical config order",
                ),
            ):
                decode_statistical_report(
                    content,
                    expected_content_sha256=digest,
                    config=self.result.config,
                )

        reversed_report = replace(
            self.report,
            strategy_cells=tuple(reversed(self.report.strategy_cells)),
        )
        with self.assertRaisesRegex(PublicationCodecError, "canonical config order"):
            encode_statistical_report(
                reversed_report,
                config=self.result.config,
            )
        wrong_config = replace(
            self.result.config,
            environment_seeds=(2, 6),
        )
        with self.assertRaisesRegex(PublicationCodecError, "identifiers disagree"):
            encode_statistical_report(self.report, config=wrong_config)

    def test_report_context_reconciles_population_and_point_estimates(self) -> None:
        def wrong_macro_seed_count(value: dict[str, Any]) -> None:
            value["payload"]["macro_contrasts"][0]["interval"][
                "seed_count_per_stratum"
            ] += 1

        def wrong_macro_strata(value: dict[str, Any]) -> None:
            value["payload"]["macro_contrasts"][0]["interval"]["stratum_count"] = 1

        def wrong_cell_seed_count(value: dict[str, Any]) -> None:
            value["payload"]["cell_contrasts"][0]["interval"][
                "seed_count_per_stratum"
            ] += 1

        def wrong_cell_strata(value: dict[str, Any]) -> None:
            value["payload"]["cell_contrasts"][0]["interval"]["stratum_count"] = 2

        def wrong_macro_point(value: dict[str, Any]) -> None:
            value["payload"]["macro_contrasts"][0]["interval"]["point_estimate"] += (
                0.001
            )

        def wrong_cell_point(value: dict[str, Any]) -> None:
            value["payload"]["cell_contrasts"][0]["interval"]["point_estimate"] += 0.001

        for mutation in (
            wrong_macro_seed_count,
            wrong_macro_strata,
            wrong_cell_seed_count,
            wrong_cell_strata,
            wrong_macro_point,
            wrong_cell_point,
        ):
            content, digest = mutate_document(self.report_document, mutation)
            with (
                self.subTest(mutation=mutation.__name__),
                self.assertRaisesRegex(
                    PublicationCodecError,
                    "interval population|point estimate",
                ),
            ):
                decode_statistical_report(
                    content,
                    expected_content_sha256=digest,
                    config=self.result.config,
                )

        no_primary_config = replace(
            self.result.config,
            personas=tuple(
                replace(persona, primary=False)
                for persona in self.result.config.personas
            ),
        )
        plan = build_bootstrap_plan(
            no_primary_config.environment_seeds,
            availability_modes=no_primary_config.availability_modes,
            evaluator_id=evaluator_fingerprint(no_primary_config),
            resample_count=self.report.bootstrap.resample_count,
        )
        no_primary_report = replace(
            self.report,
            evaluator_id=evaluator_fingerprint(no_primary_config),
            bootstrap=BootstrapMetadata(
                version=plan.version,
                resample_count=plan.resample_count,
                confidence=self.report.bootstrap.confidence,
                namespace_sha256=plan.namespace_sha256,
                indices_sha256=plan.indices_sha256,
                mode_indices_sha256=tuple(
                    (mode_plan.availability_mode, mode_plan.indices_sha256)
                    for mode_plan in plan.mode_plans
                ),
            ),
            strategy_cells=tuple(
                replace(cell, persona_primary=False)
                for cell in self.report.strategy_cells
            ),
        )
        with self.assertRaisesRegex(PublicationCodecError, "no primary persona"):
            encode_statistical_report(
                no_primary_report,
                config=no_primary_config,
            )

    def test_evidence_rejects_report_bootstrap_tampering(self) -> None:
        def missing_mode_digest(value: dict[str, Any]) -> None:
            value["payload"]["statistics"]["bootstrap"]["mode_indices_sha256"].pop()

        def wrong_namespace(value: dict[str, Any]) -> None:
            value["payload"]["statistics"]["bootstrap"]["namespace_sha256"] = "0" * 64

        for mutation in (missing_mode_digest, wrong_namespace):
            content, digest = mutate_document(self.evidence_document, mutation)
            with (
                self.subTest(mutation=mutation.__name__),
                self.assertRaisesRegex(PublicationCodecError, "bootstrap disagrees"),
            ):
                decode_publication_evidence(
                    content,
                    expected_content_sha256=digest,
                )

    def test_encoder_rejects_nonfinite_and_wrong_runtime_types(self) -> None:
        nonfinite = copy.deepcopy(self.report)
        object.__setattr__(nonfinite.bootstrap, "confidence", math.inf)
        with self.assertRaisesRegex(PublicationCodecError, "finite"):
            encode_statistical_report(nonfinite, config=self.result.config)

        wrong_integer = copy.deepcopy(self.report)
        object.__setattr__(
            wrong_integer.strategy_cells[0],
            "seed_count",
            True,
        )
        with self.assertRaisesRegex(PublicationCodecError, "exact integer"):
            encode_statistical_report(wrong_integer, config=self.result.config)

        wrong_float = copy.deepcopy(self.report)
        object.__setattr__(
            wrong_float.strategy_cells[0],
            "mean_common_expected_regret",
            0,
        )
        with self.assertRaisesRegex(PublicationCodecError, "exact float"):
            encode_statistical_report(wrong_float, config=self.result.config)

        with self.assertRaisesRegex(PublicationCodecError, "exact StatisticalReport"):
            encode_statistical_report(
                cast(StatisticalReport, object()),
                config=self.result.config,
            )
        with self.assertRaisesRegex(PublicationCodecError, "PublicationEvidence"):
            encode_publication_evidence(cast(PublicationEvidence, object()))

    def test_rejects_nonfinite_constants_and_noncanonical_numbers(self) -> None:
        nonfinite = self.report_document.content.replace(b":0.95", b":NaN", 1)
        with self.assertRaisesRegex(PublicationCodecError, "non-finite"):
            decode_statistical_report(
                nonfinite,
                expected_content_sha256=compute_content_sha256(nonfinite),
                config=self.result.config,
            )

        long_integer = b"1" * (codec._MAX_NUMBER_TOKEN_BYTES + 1)
        with self.assertRaisesRegex(PublicationCodecError, "integer token"):
            decode_statistical_report(
                long_integer,
                expected_content_sha256=compute_content_sha256(long_integer),
                config=self.result.config,
            )

        def float_for_int(value: dict[str, Any]) -> None:
            value["payload"]["bootstrap"]["resample_count"] = 8.0

        content, digest = mutate_document(self.report_document, float_for_int)
        with self.assertRaisesRegex(PublicationCodecError, "exact integer"):
            decode_statistical_report(
                content,
                expected_content_sha256=digest,
                config=self.result.config,
            )

    def test_collection_and_content_bounds_fail_before_domain_construction(
        self,
    ) -> None:
        def oversized_macro(value: dict[str, Any]) -> None:
            rows = value["payload"]["macro_contrasts"]
            rows.append(copy.deepcopy(rows[0]))

        report_content, report_sha256 = mutate_document(
            self.report_document,
            oversized_macro,
        )
        with self.assertRaisesRegex(PublicationCodecError, "collection bound"):
            decode_statistical_report(
                report_content,
                expected_content_sha256=report_sha256,
                config=self.result.config,
            )

        def oversized_seeds(value: dict[str, Any]) -> None:
            value["payload"]["config"]["environment_seeds"] = list(
                range(codec._CONFIG_SEED_LIMIT + 1)
            )

        evidence_content, evidence_sha256 = mutate_document(
            self.evidence_document,
            oversized_seeds,
        )
        with self.assertRaisesRegex(PublicationCodecError, "collection bound"):
            decode_publication_evidence(
                evidence_content,
                expected_content_sha256=evidence_sha256,
            )

        oversized_content = b" " * (codec._MAX_REPORT_BYTES + 1)
        with self.assertRaisesRegex(PublicationCodecError, "byte bound"):
            decode_statistical_report(
                oversized_content,
                expected_content_sha256=compute_content_sha256(oversized_content),
                config=self.result.config,
            )

    def test_lexical_depth_container_string_and_structure_bounds(self) -> None:
        depth_content = b"[" * (codec._MAX_JSON_DEPTH + 1) + b"]" * (
            codec._MAX_JSON_DEPTH + 1
        )
        container_content = b"[" + b"[]," * codec._MAX_JSON_CONTAINERS + b"[]]"
        string_content = (
            b'{"value":"' + b"x" * (codec._MAX_STRING_TOKEN_BYTES + 1) + b'"}'
        )
        scalar_content = b"[" + b"0," * (codec._MAX_JSON_DELIMITERS + 1) + b"0]"
        incomplete_content = b'{"value":"unterminated'
        cases = (
            (depth_content, "depth bound"),
            (container_content, "container count"),
            (string_content, "string token"),
            (scalar_content, "scalar count"),
            (incomplete_content, "structurally incomplete"),
            (b"}", "unbalanced"),
        )
        for content, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(PublicationCodecError, message),
            ):
                decode_statistical_report(
                    content,
                    expected_content_sha256=compute_content_sha256(content),
                    config=self.result.config,
                )

    def test_rejects_invalid_utf8_trailing_json_and_non_object_roots(self) -> None:
        cases = (
            (b"\xff", "valid UTF-8"),
            (self.report_document.content + b"{}", "valid bounded JSON"),
            (b"[]", "must be a JSON object"),
        )
        for content, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(PublicationCodecError, message),
            ):
                decode_statistical_report(
                    content,
                    expected_content_sha256=compute_content_sha256(content),
                    config=self.result.config,
                )

    def test_frozen_limits_cover_exact_locked_publication_cardinalities(self) -> None:
        self.assertEqual(codec._REPORT_MACRO_CONTRAST_LIMIT, 6)
        self.assertEqual(codec._REPORT_CELL_CONTRAST_LIMIT, 108)
        self.assertEqual(codec._REPORT_STRATEGY_CELL_LIMIT, 126)
        self.assertEqual(codec._EVIDENCE_STRATEGY_METRIC_LIMIT, 133)
        self.assertEqual(codec._EVIDENCE_ADAPTIVE_DIAGNOSTIC_LIMIT, 19)
        self.assertEqual(codec._EVIDENCE_CALIBRATION_LIMIT, 152)
        self.assertEqual(codec._EVIDENCE_TEMPLATE_EXPOSURE_LIMIT, 532)
        self.assertEqual(codec._EVIDENCE_RECOVERY_LIMIT, 28)
        self.assertEqual(codec._EVIDENCE_RECOVERY_CONTRAST_LIMIT, 24)
        self.assertEqual(codec._EVIDENCE_TRACE_POINT_LIMIT, 8_064)
        self.assertEqual(codec._RECOVERY_METRIC_LIMIT, 5)
        self.assertEqual(
            codec._RECOVERY_METRIC_SCHEMA,
            (
                "pre-drift-regret",
                "early-post-drift-auc",
                "late-regret",
                "recovery-rate",
                "conservative-recovery-lag",
            ),
        )

    def test_primitive_encoding_guards_cover_closed_scalar_and_tuple_types(
        self,
    ) -> None:
        with self.assertRaisesRegex(PublicationCodecError, "exact bytes"):
            CanonicalJsonDocument(
                content=cast(bytes, bytearray(b"{}")),
                content_sha256="0" * 64,
            )
        with self.assertRaisesRegex(PublicationCodecError, "integer bound"):
            codec._bounded_integer(2**63, path="integer")
        with self.assertRaisesRegex(PublicationCodecError, "exact text"):
            codec._bounded_text(1, path="text")
        with self.assertRaisesRegex(PublicationCodecError, "valid UTF-8"):
            codec._bounded_text("\ud800", path="text")
        with self.assertRaisesRegex(PublicationCodecError, "text byte bound"):
            codec._bounded_text(
                "x" * (codec._MAX_TEXT_BYTES + 1),
                path="text",
            )
        with self.assertRaisesRegex(PublicationCodecError, "exact boolean"):
            codec._encode_value(1, bool, path="boolean")
        with self.assertRaisesRegex(PublicationCodecError, "enum type"):
            codec._encode_value("adaptive", Strategy, path="strategy")
        with self.assertRaisesRegex(PublicationCodecError, "exact tuple"):
            codec._encode_value(
                [],
                tuple[int, ...],
                path="values",
                collection_limit=1,
            )
        with self.assertRaisesRegex(PublicationCodecError, "collection bound"):
            codec._encode_value(
                (1, 2),
                tuple[int, ...],
                path="values",
                collection_limit=1,
            )
        with self.assertRaisesRegex(RuntimeError, "no collection bound"):
            codec._encode_value(
                (1,),
                tuple[int, ...],
                path="values",
            )
        with self.assertRaisesRegex(PublicationCodecError, "tuple width"):
            codec._encode_value(
                (1,),
                tuple[int, str],
                path="pair",
            )
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            codec._encode_value({}, dict[str, int], path="mapping")
        with self.assertRaisesRegex(PublicationCodecError, "exact StatisticalReport"):
            codec._encode_model(object(), StatisticalReport, path="report")
        self.assertEqual(
            codec._json_bytes({"message": "café"}),
            '{"message":"café"}'.encode(),
        )
        with self.assertRaisesRegex(PublicationCodecError, "canonical JSON"):
            codec._json_bytes({"value": math.nan})

    def test_primitive_decoding_guards_cover_arrays_unions_and_numbers(self) -> None:
        with self.assertRaisesRegex(PublicationCodecError, "JSON array"):
            codec._decode_value(
                {},
                tuple[int, ...],
                path="values",
                collection_limit=1,
            )
        with self.assertRaisesRegex(PublicationCodecError, "collection bound"):
            codec._decode_value(
                [1, 2],
                tuple[int, ...],
                path="values",
                collection_limit=1,
            )
        with self.assertRaisesRegex(RuntimeError, "no collection bound"):
            codec._decode_value(
                [1],
                tuple[int, ...],
                path="values",
            )
        with self.assertRaisesRegex(PublicationCodecError, "tuple width"):
            codec._decode_value(
                [1],
                tuple[int, str],
                path="pair",
            )
        with self.assertRaisesRegex(PublicationCodecError, "must be a JSON object"):
            codec._decode_model([], StatisticalReport, path="report")
        with self.assertRaisesRegex(RuntimeError, "unsupported publication union"):
            codec._optional_member(int | str)
        with self.assertRaisesRegex(PublicationCodecError, "outside its bound"):
            codec._parse_integer(str(2**63))
        with self.assertRaisesRegex(PublicationCodecError, "float token"):
            codec._parse_float("1." + "0" * codec._MAX_NUMBER_TOKEN_BYTES)
        with self.assertRaisesRegex(PublicationCodecError, "finite"):
            codec._parse_float("1e999")
        with self.assertRaisesRegex(PublicationCodecError, "exact bytes"):
            decode_statistical_report(
                cast(bytes, bytearray(b"{}")),
                expected_content_sha256="0" * 64,
                config=self.result.config,
            )

    def test_encoder_enforces_final_document_byte_bound(self) -> None:
        with (
            patch.object(codec, "_MAX_REPORT_BYTES", 1),
            self.assertRaisesRegex(PublicationCodecError, "byte bound"),
        ):
            encode_statistical_report(self.report, config=self.result.config)


if __name__ == "__main__":
    unittest.main()
