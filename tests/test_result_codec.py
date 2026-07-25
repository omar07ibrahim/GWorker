from __future__ import annotations

import hashlib
import io
import math
import struct
import unittest
from dataclasses import replace
from typing import BinaryIO, cast
from unittest.mock import patch

import gworker.result_codec as codec
from gworker.evaluation import (
    CALIBRATION_EDGES,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PERSONAS,
    AvailabilityMode,
    ClusterSummary,
    ExperimentConfig,
    ExperimentResult,
    Strategy,
    TraceRecord,
    evaluator_fingerprint,
    run_experiment,
    validate_experiment_result,
)
from gworker.policy import DEFAULT_TEMPLATES, EvidenceBucket
from gworker.result_codec import (
    CODEC_VERSION,
    ResultCodecError,
    read_experiment_result,
    write_experiment_result,
)


def codec_config() -> ExperimentConfig:
    return ExperimentConfig(
        split="dev",
        environment_seeds=(7,),
        policy_replicas=2,
        personas=(DEFAULT_PERSONAS[0], DEFAULT_PERSONAS[4]),
        availability_modes=(AvailabilityMode.UNCONSTRAINED,),
        horizon=24,
        drift_decision=13,
        recovery_block_size=2,
        recovery_blocks=2,
    )


def encoded(result: ExperimentResult) -> tuple[bytes, str]:
    destination = io.BytesIO()
    content_sha256 = write_experiment_result(result, destination)
    return destination.getvalue(), content_sha256


def canonical_result(result: ExperimentResult) -> ExperimentResult:
    persona_rank = {
        persona.persona_id: index
        for index, persona in enumerate(result.config.personas)
    }
    mode_rank = {
        mode: index for index, mode in enumerate(result.config.availability_modes)
    }
    seed_rank = {
        seed: index for index, seed in enumerate(result.config.environment_seeds)
    }
    strategy_rank = {strategy: index for index, strategy in enumerate(Strategy)}

    def summary_key(summary: ClusterSummary) -> tuple[int, int, int, int]:
        return (
            persona_rank[summary.persona_id],
            mode_rank[summary.availability_mode],
            seed_rank[summary.environment_seed],
            strategy_rank[summary.strategy],
        )

    def trace_key(trace: TraceRecord) -> tuple[int, int, int, int]:
        return (
            persona_rank[trace.persona_id],
            mode_rank[trace.availability_mode],
            seed_rank[trace.environment_seed],
            strategy_rank[trace.strategy],
        )

    return replace(
        result,
        cluster_summaries=tuple(sorted(result.cluster_summaries, key=summary_key)),
        abrupt_traces=tuple(sorted(result.abrupt_traces, key=trace_key)),
    )


def resigned(container: bytes) -> bytes:
    payload = container[: -codec._TRAILER_SIZE]
    return payload + codec._TRAILER_MAGIC + hashlib.sha256(payload).digest()


def replace_once(container: bytes, old: bytes, new: bytes) -> bytes:
    if len(old) != len(new):
        raise AssertionError("test replacement must preserve the binary layout")
    payload = container[: -codec._TRAILER_SIZE]
    if payload.count(old) != 1:
        raise AssertionError(f"expected exactly one occurrence of {old!r}")
    return resigned(payload.replace(old, new, 1) + container[-codec._TRAILER_SIZE :])


def skip_text(payload: bytes, offset: int) -> int:
    size = cast(int, struct.unpack_from(">H", payload, offset)[0])
    return offset + 2 + size


def config_offsets(container: bytes) -> tuple[int, int, int]:
    """Return split, environment-count, and cluster-count offsets."""

    payload = container[: -codec._TRAILER_SIZE]
    offset = len(codec._CANONICAL_HEADER)
    for _ in range(4):
        offset = skip_text(payload, offset)
    offset += 4  # hard_failure_count
    split_offset = offset
    offset = skip_text(payload, offset)
    environment_count_offset = offset
    seed_count = struct.unpack_from(">I", payload, offset)[0]
    offset += 4 + seed_count * 4
    offset += 2  # policy_replicas
    persona_count = struct.unpack_from(">H", payload, offset)[0]
    offset += 2
    for _ in range(persona_count):
        offset = skip_text(payload, offset)
        offset = skip_text(payload, offset)
        offset += 1 + 8 + 1
    mode_count = payload[offset]
    offset += 1
    for _ in range(mode_count):
        offset = skip_text(payload, offset)
    offset += 4 * 4  # horizon and three recovery fields
    return split_offset, environment_count_offset, offset


def first_summary_offsets(container: bytes) -> tuple[int, int]:
    """Return the first summary boolean and first metric offsets."""

    payload = container[: -codec._TRAILER_SIZE]
    _split, _seed_count, cluster_count_offset = config_offsets(container)
    offset = cluster_count_offset + 8
    offset = skip_text(payload, offset)
    boolean_offset = offset
    offset += 1
    offset = skip_text(payload, offset)
    offset += 4
    offset = skip_text(payload, offset)
    offset += 2
    return boolean_offset, offset


def patched_u32(container: bytes, offset: int, value: int) -> bytes:
    mutable = bytearray(container)
    struct.pack_into(">I", mutable, offset, value)
    return resigned(bytes(mutable))


class TinyChunkReader(io.BytesIO):
    def read(self, size: int | None = -1, /) -> bytes:
        if size is None or size < 0:
            size = 3
        return super().read(min(size, 3))


class OversizedChunkReader(io.BytesIO):
    def read(self, size: int | None = -1, /) -> bytes:
        if size is None or size < 0:
            return super().read(size)
        return super().read(size + 1)


class TrailerOversizedReader(io.BytesIO):
    def __init__(self, payload: bytes, trailer_offset: int) -> None:
        super().__init__(payload)
        self._trailer_offset = trailer_offset

    def read(self, size: int | None = -1, /) -> bytes:
        if size is not None and size >= 0 and self.tell() >= self._trailer_offset:
            return super().read(size + 1)
        return super().read(size)


class ShortWriter:
    def write(self, payload: bytes) -> int:
        if not payload:
            return 0
        return len(payload) - 1


class NonIntegerWriter:
    def __init__(self, return_value: bool | float) -> None:
        self._return_value = return_value

    def write(self, payload: bytes) -> bool | float:
        del payload
        return self._return_value


class RaisingBinaryStream:
    def read(self, size: int = -1) -> bytes:
        del size
        raise OSError("read failed")

    def write(self, payload: bytes) -> int:
        del payload
        raise OSError("write failed")


class TextReturningStream:
    def read(self, size: int = -1) -> str:
        del size
        return "not bytes"


class ExperimentResultCodecTests(unittest.TestCase):
    result: ExperimentResult
    container: bytes
    content_sha256: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_experiment(codec_config())
        cls.container, cls.content_sha256 = encoded(cls.result)

    def test_round_trip_is_exact_and_reuses_common_trace_tuples(self) -> None:
        decoded = read_experiment_result(io.BytesIO(self.container))

        self.assertEqual(decoded, canonical_result(self.result))
        validate_experiment_result(decoded)
        expected_ordinals = tuple(
            (
                persona.persona_id,
                mode,
                seed,
                strategy,
            )
            for persona in decoded.config.personas
            for mode in decoded.config.availability_modes
            for seed in decoded.config.environment_seeds
            for strategy in Strategy
        )
        self.assertEqual(
            tuple(
                (
                    summary.persona_id,
                    summary.availability_mode,
                    summary.environment_seed,
                    summary.strategy,
                )
                for summary in decoded.cluster_summaries
            ),
            expected_ordinals,
        )
        summaries = {
            (
                summary.persona_id,
                summary.availability_mode,
                summary.environment_seed,
                summary.strategy,
            ): summary
            for summary in decoded.cluster_summaries
        }
        for trace in decoded.abrupt_traces:
            summary = summaries[
                (
                    trace.persona_id,
                    trace.availability_mode,
                    trace.environment_seed,
                    trace.strategy,
                )
            ]
            self.assertIs(
                trace.common_expected_regret,
                summary.metrics.common_regret_trace,
            )

    def test_repeated_and_shuffled_writes_are_byte_identical(self) -> None:
        repeated, repeated_content_sha256 = encoded(self.result)
        shuffled = replace(
            self.result,
            cluster_summaries=tuple(reversed(self.result.cluster_summaries)),
            abrupt_traces=tuple(reversed(self.result.abrupt_traces)),
        )
        validate_experiment_result(shuffled)
        shuffled_bytes, shuffled_content_sha256 = encoded(shuffled)

        self.assertEqual(repeated, self.container)
        self.assertEqual(shuffled_bytes, self.container)
        self.assertEqual(repeated_content_sha256, self.content_sha256)
        self.assertEqual(shuffled_content_sha256, self.content_sha256)

    def test_canonical_order_preserves_declared_nonlexical_seed_order(self) -> None:
        config = replace(codec_config(), environment_seeds=(17, 2))
        result = run_experiment(config)
        original_bytes, original_digest = encoded(result)
        shuffled_bytes, shuffled_digest = encoded(
            replace(
                result,
                cluster_summaries=tuple(reversed(result.cluster_summaries)),
                abrupt_traces=tuple(reversed(result.abrupt_traces)),
            )
        )

        self.assertEqual(shuffled_bytes, original_bytes)
        self.assertEqual(shuffled_digest, original_digest)
        decoded = read_experiment_result(io.BytesIO(original_bytes))
        first_persona_mode_seeds = tuple(
            summary.environment_seed
            for summary in decoded.cluster_summaries
            if summary.persona_id == config.personas[0].persona_id
            and summary.availability_mode is config.availability_modes[0]
            and summary.strategy is Strategy.ADAPTIVE
        )
        self.assertEqual(first_persona_mode_seeds, (17, 2))

    def test_version_header_and_checksum_are_explicit(self) -> None:
        payload = self.container[: -codec._TRAILER_SIZE]
        trailer = self.container[-codec._TRAILER_SIZE :]

        self.assertIn(CODEC_VERSION.encode("ascii"), codec._CANONICAL_HEADER)
        self.assertTrue(payload.startswith(codec._CANONICAL_HEADER))
        self.assertTrue(trailer.startswith(codec._TRAILER_MAGIC))
        self.assertEqual(
            trailer[len(codec._TRAILER_MAGIC) :],
            hashlib.sha256(payload).digest(),
        )
        self.assertEqual(self.content_sha256, hashlib.sha256(payload).hexdigest())

    def test_closed_schema_widths_enum_order_and_bounds_are_frozen(self) -> None:
        self.assertEqual(codec._EVIDENCE_BUCKET_WIDTH, 3)
        self.assertEqual(codec._TEMPLATE_WIDTH, 4)
        self.assertEqual(codec._CALIBRATION_WIDTH, 8)
        self.assertEqual(len(EvidenceBucket), codec._EVIDENCE_BUCKET_WIDTH)
        self.assertEqual(len(DEFAULT_TEMPLATES), codec._TEMPLATE_WIDTH)
        self.assertEqual(
            len(CALIBRATION_EDGES) - 1,
            codec._CALIBRATION_WIDTH,
        )
        self.assertEqual(
            tuple((member.name, member.value) for member in AvailabilityMode),
            codec._AVAILABILITY_MODE_SCHEMA,
        )
        self.assertEqual(
            tuple((member.name, member.value) for member in Strategy),
            codec._STRATEGY_SCHEMA,
        )
        locked_clusters = (
            len(DEFAULT_EXPERIMENT_CONFIG.personas)
            * len(DEFAULT_EXPERIMENT_CONFIG.availability_modes)
            * len(DEFAULT_EXPERIMENT_CONFIG.environment_seeds)
            * len(Strategy)
        )
        locked_trace_floats = locked_clusters * DEFAULT_EXPERIMENT_CONFIG.horizon * 3
        self.assertEqual(locked_clusters, 16_128)
        self.assertEqual(locked_trace_floats, 13_934_592)
        self.assertLessEqual(locked_clusters, codec._MAX_CLUSTER_SUMMARIES)
        self.assertLessEqual(locked_trace_floats, codec._MAX_TOTAL_TRACE_VALUES)

    def test_reader_supports_incremental_binary_reads(self) -> None:
        decoded = read_experiment_result(TinyChunkReader(self.container))
        self.assertEqual(decoded, canonical_result(self.result))

    def test_reader_rejects_streams_that_overreturn_bytes(self) -> None:
        readers = (
            OversizedChunkReader(self.container),
            TrailerOversizedReader(
                self.container,
                len(self.container) - codec._TRAILER_SIZE,
            ),
        )
        for reader in readers:
            with (
                self.subTest(reader=type(reader).__name__),
                self.assertRaisesRegex(ResultCodecError, "more bytes"),
            ):
                read_experiment_result(reader)

    def test_writer_rejects_a_short_binary_write(self) -> None:
        with self.assertRaisesRegex(ResultCodecError, "short write"):
            write_experiment_result(self.result, cast(BinaryIO, ShortWriter()))

    def test_writer_requires_an_exact_integer_write_count(self) -> None:
        for return_value in (True, float(len(codec._CANONICAL_HEADER))):
            with (
                self.subTest(return_value=return_value),
                self.assertRaisesRegex(ResultCodecError, "short write"),
            ):
                write_experiment_result(
                    self.result,
                    cast(BinaryIO, NonIntegerWriter(return_value)),
                )

    def test_rejects_truncation_at_each_container_region(self) -> None:
        cuts = (
            0,
            len(codec._CANONICAL_HEADER) - 1,
            len(self.container) // 2,
            len(self.container) - codec._TRAILER_SIZE // 2,
            len(self.container) - 1,
        )
        for cut in cuts:
            with (
                self.subTest(cut=cut),
                self.assertRaisesRegex(ResultCodecError, "truncated"),
            ):
                read_experiment_result(io.BytesIO(self.container[:cut]))

    def test_rejects_trailing_data_and_checksum_corruption(self) -> None:
        with self.assertRaisesRegex(ResultCodecError, "trailing data"):
            read_experiment_result(io.BytesIO(self.container + b"\x00"))

        corrupted = bytearray(self.container)
        corrupted[-1] ^= 1
        with (
            patch.object(codec, "TraceRecord") as trace_record,
            self.assertRaisesRegex(ResultCodecError, "checksum"),
        ):
            read_experiment_result(io.BytesIO(corrupted))
        trace_record.assert_not_called()

    def test_rejects_malformed_or_wrong_version_header(self) -> None:
        for target in (
            b"GWORKER",
            CODEC_VERSION.encode("ascii"),
        ):
            malformed = bytearray(self.container)
            offset = malformed.index(target)
            malformed[offset] ^= 1
            with (
                self.subTest(target=target),
                self.assertRaisesRegex(ResultCodecError, "header|version"),
            ):
                read_experiment_result(io.BytesIO(malformed))

    def test_rejects_resigned_wrong_schema_and_nonzero_failure_count(self) -> None:
        schema = self.result.schema_version.encode("utf-8")
        wrong_schema = bytes((schema[0] ^ 1,)) + schema[1:]
        with self.assertRaisesRegex(ResultCodecError, "schema version"):
            read_experiment_result(
                io.BytesIO(replace_once(self.container, schema, wrong_schema))
            )

        split_offset, _seed_count, _cluster_count = config_offsets(self.container)
        with self.assertRaisesRegex(ResultCodecError, "hard failure"):
            read_experiment_result(
                io.BytesIO(patched_u32(self.container, split_offset - 4, 1))
            )

    def test_rejects_invalid_checksum_trailer_magic(self) -> None:
        malformed = bytearray(self.container)
        malformed[-codec._TRAILER_SIZE] ^= 1
        with self.assertRaisesRegex(ResultCodecError, "trailer magic"):
            read_experiment_result(io.BytesIO(malformed))

    def test_rejects_resigned_wrong_result_identifiers(self) -> None:
        identifiers = (
            self.result.evaluator_id,
            self.result.design_id,
            self.result.policy_id,
        )
        for identifier in identifiers:
            old = identifier.encode("utf-8")
            replacement = bytes((old[0] ^ 1,)) + old[1:]
            tampered = replace_once(self.container, old, replacement)
            with (
                self.subTest(identifier=identifier),
                self.assertRaisesRegex(ResultCodecError, "id"),
            ):
                read_experiment_result(io.BytesIO(tampered))

    def test_rejects_resigned_wrong_config_and_oversized_count(self) -> None:
        split_offset, seed_count_offset, _cluster_count_offset = config_offsets(
            self.container
        )
        malformed_split = bytearray(self.container)
        self.assertEqual(
            malformed_split[split_offset : split_offset + 5], b"\x00\x03dev"
        )
        malformed_split[split_offset + 2 : split_offset + 5] = b"bad"
        with self.assertRaises(ResultCodecError):
            read_experiment_result(io.BytesIO(resigned(bytes(malformed_split))))

        oversized = patched_u32(self.container, seed_count_offset, 0xFFFFFFFF)
        with self.assertRaisesRegex(ResultCodecError, "seed count.*bound"):
            read_experiment_result(io.BytesIO(oversized))

    def test_rejects_resigned_wrong_cluster_or_trace_cardinality(self) -> None:
        _split, _seed_count, cluster_count_offset = config_offsets(self.container)
        cluster_count = struct.unpack_from(
            ">I",
            self.container,
            cluster_count_offset,
        )[0]
        abrupt_count = struct.unpack_from(
            ">I",
            self.container,
            cluster_count_offset + 4,
        )[0]
        tampered_values = (
            (cluster_count_offset, cluster_count + 1, "cluster cardinality"),
            (cluster_count_offset + 4, abrupt_count + 1, "abrupt trace cardinality"),
        )
        for offset, value, message in tampered_values:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ResultCodecError, message),
            ):
                read_experiment_result(
                    io.BytesIO(patched_u32(self.container, offset, value))
                )

    def test_rejects_resigned_noncanonical_summary_order(self) -> None:
        with patch.object(
            codec,
            "_canonical_summaries",
            return_value=tuple(reversed(self.result.cluster_summaries)),
        ):
            noncanonical, _content_sha256 = encoded(self.result)

        with self.assertRaisesRegex(ResultCodecError, "ordinal|Cartesian order"):
            read_experiment_result(io.BytesIO(noncanonical))

    def test_resigned_semantic_tamper_reaches_final_result_validation(self) -> None:
        _boolean_offset, first_metric_offset = first_summary_offsets(self.container)
        tampered = bytearray(self.container)
        original = cast(
            float,
            struct.unpack_from(">d", tampered, first_metric_offset)[0],
        )
        struct.pack_into(">d", tampered, first_metric_offset, original + 0.001)

        with self.assertRaisesRegex(ResultCodecError, "decoded.*validation"):
            read_experiment_result(io.BytesIO(resigned(bytes(tampered))))

    def test_rejects_noncanonical_bool_and_nonfinite_float_payloads(self) -> None:
        boolean_offset, first_metric_offset = first_summary_offsets(self.container)

        invalid_bool = bytearray(self.container)
        invalid_bool[boolean_offset] = 2
        with self.assertRaisesRegex(ResultCodecError, "boolean"):
            read_experiment_result(io.BytesIO(resigned(bytes(invalid_bool))))

        invalid_float = bytearray(self.container)
        struct.pack_into(">d", invalid_float, first_metric_offset, math.nan)
        with self.assertRaisesRegex(ResultCodecError, "finite"):
            read_experiment_result(io.BytesIO(resigned(bytes(invalid_float))))

        negative_zero = bytearray(self.container)
        struct.pack_into(">d", negative_zero, first_metric_offset, -0.0)
        with self.assertRaisesRegex(ResultCodecError, "negative zero"):
            read_experiment_result(io.BytesIO(resigned(bytes(negative_zero))))

    def test_encoder_rejects_wrong_ids_config_and_cardinality_before_write(
        self,
    ) -> None:
        invalid_results = (
            replace(self.result, evaluator_id="wrong"),
            replace(
                self.result,
                config=replace(self.result.config, environment_seeds=(8,)),
            ),
            replace(
                self.result,
                cluster_summaries=self.result.cluster_summaries[:-1],
            ),
        )
        for invalid in invalid_results:
            destination = io.BytesIO()
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(ResultCodecError),
            ):
                write_experiment_result(invalid, destination)
            self.assertEqual(destination.getvalue(), b"")

    def test_full_codec_preflight_finishes_before_the_first_write(self) -> None:
        long_persona = replace(
            self.result.config.personas[0],
            label="x" * (codec._MAX_TEXT_BYTES + 1),
        )
        changed_config = replace(
            self.result.config,
            personas=(long_persona, *self.result.config.personas[1:]),
        )
        changed = replace(
            self.result,
            config=changed_config,
            evaluator_id=evaluator_fingerprint(changed_config),
        )
        destination = io.BytesIO()

        with self.assertRaisesRegex(ResultCodecError, "encoded length"):
            write_experiment_result(changed, destination)
        self.assertEqual(destination.getvalue(), b"")

        for limit_name in ("_MAX_CLUSTER_SUMMARIES", "_MAX_TOTAL_TRACE_VALUES"):
            destination = io.BytesIO()
            with (
                self.subTest(limit_name=limit_name),
                patch.object(codec, limit_name, 1),
                self.assertRaisesRegex(ResultCodecError, "cardinality"),
            ):
                write_experiment_result(self.result, destination)
            self.assertEqual(destination.getvalue(), b"")

    def test_writer_rejects_low_level_config_mutation_before_write(self) -> None:
        mutated_persona = replace(self.result.config.personas[0])
        object.__setattr__(mutated_persona, "sigma_minutes", 5)
        mutations: tuple[tuple[str, object], ...] = (
            ("personas", ()),
            ("personas", (mutated_persona, *self.result.config.personas[1:])),
            ("availability_modes", ()),
            ("environment_seeds", ()),
            ("environment_seeds", (7, 7)),
            ("policy_replicas", 0),
            ("policy_replicas", 65),
            ("horizon", 25),
            ("recovery_blocks", 0),
        )
        for field, value in mutations:
            mutated = replace(self.result.config)
            object.__setattr__(mutated, field, value)
            destination = io.BytesIO()

            with (
                self.subTest(field=field, value=value),
                self.assertRaises(ResultCodecError),
            ):
                write_experiment_result(
                    replace(self.result, config=mutated),
                    destination,
                )
            self.assertEqual(destination.getvalue(), b"")

    def test_encoder_rejects_bool_int_confusion_and_nonfinite_metrics(self) -> None:
        first = self.result.cluster_summaries[0]
        invalid_summaries = (
            replace(first, persona_primary=1),  # type: ignore[arg-type]
            replace(
                first,
                metrics=replace(first.metrics, action_count=True),
            ),
            replace(
                first,
                metrics=replace(
                    first.metrics,
                    mean_common_expected_regret=math.inf,
                ),
            ),
        )
        for invalid_summary in invalid_summaries:
            invalid_result = self._replace_summary(first, invalid_summary)
            with (
                self.subTest(summary=invalid_summary),
                self.assertRaises(ResultCodecError),
            ):
                write_experiment_result(invalid_result, io.BytesIO())

    def test_float_fields_have_a_closed_exact_type(self) -> None:
        index, myopic = next(
            (index, summary)
            for index, summary in enumerate(self.result.cluster_summaries)
            if summary.strategy is Strategy.MYOPIC_ORACLE
        )
        integer_zero = replace(
            myopic,
            metrics=replace(
                myopic.metrics,
                mean_conditional_oracle_distance=0,
            ),
        )
        invalid = replace(
            self.result,
            cluster_summaries=(
                *self.result.cluster_summaries[:index],
                integer_zero,
                *self.result.cluster_summaries[index + 1 :],
            ),
        )
        validate_experiment_result(invalid)

        with self.assertRaisesRegex(ResultCodecError, "exact float"):
            write_experiment_result(invalid, io.BytesIO())

    def test_negative_zero_is_canonicalized_on_write(self) -> None:
        index, myopic = next(
            (index, summary)
            for index, summary in enumerate(self.result.cluster_summaries)
            if summary.strategy is Strategy.MYOPIC_ORACLE
        )
        signed_zero = replace(
            myopic,
            metrics=replace(
                myopic.metrics,
                mean_conditional_oracle_distance=-0.0,
            ),
        )
        altered = replace(
            self.result,
            cluster_summaries=(
                *self.result.cluster_summaries[:index],
                signed_zero,
                *self.result.cluster_summaries[index + 1 :],
            ),
        )
        validate_experiment_result(altered)

        altered_bytes, _content_sha256 = encoded(altered)
        self.assertEqual(altered_bytes, self.container)

    def test_primitive_decoders_reject_noncanonical_tags_and_vectors(self) -> None:
        with self.assertRaisesRegex(ResultCodecError, "presence tag"):
            codec._read_optional_float(
                codec._HashingReader(io.BytesIO(b"\x02")),
                field="optional float",
            )
        with self.assertRaisesRegex(ResultCodecError, "presence tag"):
            codec._read_optional_int(
                codec._HashingReader(io.BytesIO(b"\x02")),
                field="optional int",
            )
        with self.assertRaisesRegex(ResultCodecError, "noncanonical length"):
            codec._read_int_vector(
                codec._HashingReader(io.BytesIO(struct.pack(">I", 2))),
                field="integers",
                expected_length=1,
            )
        with self.assertRaisesRegex(ResultCodecError, "noncanonical length"):
            codec._read_float_vector(
                codec._HashingReader(io.BytesIO(struct.pack(">I", 2))),
                field="floats",
                expected_length=1,
            )
        for value, message in ((math.nan, "finite"), (-0.0, "negative zero")):
            raw = struct.pack(">Id", 1, value)
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ResultCodecError, message),
            ):
                codec._read_float_vector(
                    codec._HashingReader(io.BytesIO(raw)),
                    field="floats",
                    expected_length=1,
                )

    def test_primitive_text_decoder_rejects_empty_or_invalid_utf8(self) -> None:
        for raw in (b"\x00\x00", b"\x00\x01\xff"):
            with (
                self.subTest(raw=raw),
                self.assertRaises(ResultCodecError),
            ):
                codec._read_text(
                    codec._HashingReader(io.BytesIO(raw)),
                    field="text",
                    maximum=8,
                )

    def test_primitive_io_and_encoding_guards_fail_closed(self) -> None:
        with self.assertRaisesRegex(ResultCodecError, "invalid byte length"):
            codec._HashingReader(io.BytesIO()).read_exact(-1, field="negative")

        for source in (RaisingBinaryStream(), TextReturningStream()):
            with (
                self.subTest(source=type(source).__name__),
                self.assertRaises(ResultCodecError),
            ):
                codec._HashingReader(cast(BinaryIO, source)).read_exact(
                    1,
                    field="payload",
                )
            with (
                self.subTest(direct_source=type(source).__name__),
                self.assertRaises(ResultCodecError),
            ):
                codec._read_direct(
                    cast(BinaryIO, source),
                    1,
                    field="payload",
                )

        stream = cast(BinaryIO, RaisingBinaryStream())
        with self.assertRaises(ResultCodecError):
            codec._HashingWriter(stream).write(b"x")
        with self.assertRaises(ResultCodecError):
            codec._write_direct(stream, b"x", field="payload")
        with self.assertRaisesRegex(ResultCodecError, "binary range"):
            codec._bounded_int(-1, field="integer", maximum=1)
        with self.assertRaisesRegex(ResultCodecError, "exact text"):
            codec._encoded_text(1, field="text", maximum=8)
        with self.assertRaisesRegex(ResultCodecError, "UTF-8"):
            codec._encoded_text("\ud800", field="text", maximum=8)

    def test_primitive_writer_guards_reject_wrong_shapes(self) -> None:
        writer = codec._HashingWriter(io.BytesIO())
        with self.assertRaisesRegex(ResultCodecError, "exact boolean"):
            codec._write_bool(writer, 1, field="boolean")
        with self.assertRaisesRegex(ResultCodecError, "exact tuple"):
            codec._write_int_vector(
                writer,
                [],
                field="integers",
                expected_length=1,
            )
        with self.assertRaisesRegex(ResultCodecError, "noncanonical length"):
            codec._write_int_vector(
                writer,
                (),
                field="integers",
                expected_length=1,
            )
        with self.assertRaisesRegex(ResultCodecError, "exact tuple"):
            codec._write_float_vector(
                writer,
                [],
                field="floats",
                expected_length=1,
            )
        with self.assertRaisesRegex(ResultCodecError, "noncanonical length"):
            codec._write_float_vector(
                writer,
                (),
                field="floats",
                expected_length=1,
            )

    def _replace_summary(
        self,
        original: ClusterSummary,
        replacement: ClusterSummary,
    ) -> ExperimentResult:
        index = self.result.cluster_summaries.index(original)
        return replace(
            self.result,
            cluster_summaries=(
                *self.result.cluster_summaries[:index],
                replacement,
                *self.result.cluster_summaries[index + 1 :],
            ),
        )


if __name__ == "__main__":
    unittest.main()
