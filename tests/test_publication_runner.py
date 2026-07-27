from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import gworker.publication_runner as runner
from gworker.evaluation import (
    DEFAULT_EXPERIMENT_CONFIG,
    LOCKED_EVALUATION_RUN_KEY,
    ExperimentResult,
)
from gworker.evidence import expected_publication_cardinalities
from gworker.publication_codec import CanonicalJsonDocument
from gworker.publication_state import (
    SOURCE_PROVENANCE_FILE_NAME,
    ArtifactBinding,
    PublicationStage,
    PublicationStateStore,
)
from gworker.reporting import SourceFileIdentity, SourceProvenance
from gworker.resource_preflight import (
    CgroupSnapshot,
    FilesystemSnapshot,
    MemorySnapshot,
    PressureLine,
    PressureResourceSnapshot,
    PressureSnapshot,
    ProcessLimitSnapshot,
    ResourceCapacityError,
    ResourceSnapshot,
    assess_publication_capacity,
)


def source_provenance(*, variant: int = 0) -> SourceProvenance:
    digit = f"{variant % 10:x}"
    return SourceProvenance(
        source_commit=digit * 40,
        source_tree=f"{(variant + 1) % 10:x}" * 40,
        git_object_format="sha1",
        source_archive_sha256=f"{(variant + 2) % 10:x}" * 64,
        source_commit_time=f"2026-07-{20 + variant:02d}T12:00:00+00:00",
        branch="feat/event-sourced-focus-engine",
        author_name="Omar Ibrahim",
        author_email="31526072+omar07ibrahim@users.noreply.github.com",
        committer_name="Omar Ibrahim",
        committer_email="31526072+omar07ibrahim@users.noreply.github.com",
        python_implementation="CPython",
        python_version="3.12.3",
        python_cache_tag="cpython-312",
        platform="Linux-test",
        loaded_sources=tuple(
            SourceFileIdentity(
                relative_path=path,
                sha256=hashlib.sha256(f"{path}:{variant}".encode()).hexdigest(),
            )
            for path in runner._EXPECTED_SOURCE_PATHS
        ),
        clean_pre_run=True,
    )


def resource_assessment(*, ready: bool = True):
    gibibyte = 1 << 30
    available = 16 * gibibyte if ready else 1 * gibibyte
    zero_pressure = PressureResourceSnapshot(
        some=PressureLine(0.0, 0.0, 0.0, 0),
        full=PressureLine(0.0, 0.0, 0.0, 0),
    )
    snapshot = ResourceSnapshot(
        memory=MemorySnapshot(
            total_bytes=32 * gibibyte,
            available_bytes=available,
            swap_total_bytes=0,
            swap_free_bytes=0,
        ),
        cgroup=CgroupSnapshot(self_path="/", levels=()),
        filesystem=FilesystemSnapshot(
            total_bytes=64 * gibibyte,
            available_bytes=32 * gibibyte,
            total_inodes=100_000,
            available_inodes=50_000,
        ),
        pressure=PressureSnapshot(
            memory=zero_pressure,
            io=zero_pressure,
        ),
        limits=ProcessLimitSnapshot(nofile_soft=1024, nofile_hard=4096),
    )
    return assess_publication_capacity(snapshot)


def binding(name: str, content: bytes) -> ArtifactBinding:
    return ArtifactBinding(name, hashlib.sha256(content).hexdigest(), len(content))


def locked_result() -> ExperimentResult:
    counts = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
    result = object.__new__(ExperimentResult)
    object.__setattr__(result, "schema_version", runner._EXPECTED_RESULT_SCHEMA)
    object.__setattr__(result, "config", DEFAULT_EXPERIMENT_CONFIG)
    object.__setattr__(result, "evaluator_id", runner._EXPECTED_EVALUATOR_ID)
    object.__setattr__(result, "design_id", runner._EXPECTED_DESIGN_ID)
    object.__setattr__(result, "policy_id", runner._EXPECTED_POLICY_ID)
    object.__setattr__(
        result,
        "cluster_summaries",
        (None,) * counts.cluster_summaries,
    )
    object.__setattr__(
        result,
        "abrupt_traces",
        (None,) * counts.abrupt_traces,
    )
    object.__setattr__(result, "hard_failure_count", 0)
    return result


class PublicationRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.gworker = self.root / ".gworker"
        self.publication = self.gworker / "publication"
        self.run_directory = self.publication / LOCKED_EVALUATION_RUN_KEY

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_parent(self) -> None:
        self.gworker.mkdir(mode=0o700)
        self.gworker.chmod(0o700)
        self.publication.mkdir(mode=0o700)
        self.publication.chmod(0o700)

    def make_provenance_only(
        self,
        provenance: SourceProvenance | None = None,
    ) -> tuple[SourceProvenance, ArtifactBinding]:
        selected = provenance or source_provenance()
        self.make_parent()
        descriptor = runner._create_provenance_run_directory(self.root)
        try:
            artifact = runner._preserve_source(descriptor, selected)
        finally:
            runner._close_descriptor(descriptor, "test run directory")
        return selected, artifact

    def initialize(
        self,
        provenance: SourceProvenance | None = None,
    ) -> tuple[SourceProvenance, ArtifactBinding, PublicationStateStore]:
        selected, artifact = self.make_provenance_only(provenance)
        store = PublicationStateStore.initialize(
            self.run_directory,
            source_provenance=artifact,
        )
        return selected, artifact, store

    def write_artifact(self, name: str, content: bytes) -> ArtifactBinding:
        path = self.run_directory / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        for parent in (path.parent,):
            parent.chmod(0o700)
        path.write_bytes(content)
        path.chmod(0o600)
        return binding(name, content)

    def fake_evaluate(
        self,
        store: PublicationStateStore,
        _run_descriptor: int,
    ):
        store.begin_evaluation().consume()
        result = self.write_artifact("result.bin", b"synthetic-result")
        counts = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
        return store.record_evaluated(
            result=result,
            cluster_summaries=counts.cluster_summaries,
            abrupt_traces=counts.abrupt_traces,
            hard_failure_count=0,
        )

    def fake_materialize(
        self,
        store: PublicationStateStore,
        _run_descriptor: int,
        _record: object,
    ):
        report = self.write_artifact("report.json", b'{"report":"locked"}')
        evidence = self.write_artifact("evidence.json", b'{"evidence":"locked"}')
        return store.record_materialized(
            report=report,
            evidence=evidence,
            cardinalities=expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG),
        )

    def test_provenance_codec_is_strict_canonical_and_bounded(self) -> None:
        expected = source_provenance()
        content = runner._encode_source_provenance(expected)
        self.assertEqual(runner._decode_source_provenance(content), expected)
        self.assertEqual(
            json.dumps(
                json.loads(content),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            content,
        )

        malformed = (
            b" " + content,
            content.replace(b'"branch":', b'"branch":"duplicate","branch":', 1),
            content.replace(b'"clean_pre_run":true', b'"clean_pre_run":false'),
            content.replace(b'"loaded_sources":[', b'"loaded_sources":[0,', 1),
            content + b"0" * runner._MAX_PROVENANCE_BYTES,
        )
        for payload in malformed:
            with (
                self.subTest(payload=payload[:32]),
                self.assertRaises(runner.PublicationRunnerConflict),
            ):
                runner._decode_source_provenance(payload)

    def test_source_provenance_is_created_once_and_reopened_exactly(self) -> None:
        expected, expected_binding = self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        self.assertIsNotNone(descriptor)
        assert descriptor is not None
        try:
            observed, observed_binding = runner._read_preserved_source(descriptor)
            self.assertEqual(observed, expected)
            self.assertEqual(observed_binding, expected_binding)
            self.assertEqual(
                runner._preserve_source(descriptor, expected),
                expected_binding,
            )
            with self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "differs",
            ):
                runner._preserve_source(descriptor, source_provenance(variant=1))
        finally:
            runner._close_descriptor(descriptor, "test run directory")

        path = self.run_directory / SOURCE_PROVENANCE_FILE_NAME
        self.assertEqual(path.stat().st_nlink, 1)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_provenance_only_crash_has_explicit_status(self) -> None:
        self.make_provenance_only()
        status = runner.publication_status(self.root)
        self.assertEqual(status.stage, "provenance-captured")
        self.assertEqual(status.disposition, "claim-pending")

    def test_empty_bootstrap_crash_is_recoverable_without_locked_eval(self) -> None:
        expected = source_provenance()
        self.make_parent()
        self.run_directory.mkdir(mode=0o700)
        self.run_directory.chmod(0o700)
        self.assertEqual(
            runner.publication_status(self.root).stage,
            "provenance-bootstrap",
        )
        with (
            patch.object(
                runner,
                "_double_gate_for_evaluation",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_capture_locked_source",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_evaluate_once",
                side_effect=self.fake_evaluate,
            ),
            patch.object(
                runner,
                "_materialize",
                side_effect=self.fake_materialize,
            ),
            patch.object(runner, "_run_locked_experiment") as evaluate,
        ):
            status = runner.run_publication(self.root)
        evaluate.assert_not_called()
        self.assertEqual(status.stage, PublicationStage.MATERIALIZED.value)

    def test_prefix_pending_provenance_is_completed_and_recovered(self) -> None:
        expected = source_provenance()
        self.make_parent()
        self.run_directory.mkdir(mode=0o700)
        self.run_directory.chmod(0o700)
        pending = self.run_directory / f".{SOURCE_PROVENANCE_FILE_NAME}.pending"
        canonical = runner._encode_source_provenance(expected)
        pending.write_bytes(canonical[: len(canonical) // 2])
        pending.chmod(0o600)
        self.assertEqual(
            runner.publication_status(self.root).stage,
            "provenance-pending",
        )
        with (
            patch.object(
                runner,
                "_double_gate_for_evaluation",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_capture_locked_source",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_evaluate_once",
                side_effect=self.fake_evaluate,
            ),
            patch.object(
                runner,
                "_materialize",
                side_effect=self.fake_materialize,
            ),
            patch.object(runner, "_run_locked_experiment") as evaluate,
        ):
            status = runner.run_publication(self.root)
        evaluate.assert_not_called()
        self.assertEqual(status.stage, PublicationStage.MATERIALIZED.value)
        self.assertFalse(pending.exists())
        self.assertTrue((self.run_directory / SOURCE_PROVENANCE_FILE_NAME).is_file())

    def test_mismatched_pending_provenance_fails_closed(self) -> None:
        expected = source_provenance()
        self.make_parent()
        self.run_directory.mkdir(mode=0o700)
        self.run_directory.chmod(0o700)
        pending = self.run_directory / f".{SOURCE_PROVENANCE_FILE_NAME}.pending"
        pending.write_bytes(
            runner._encode_source_provenance(source_provenance(variant=1))
        )
        pending.chmod(0o600)
        with (
            patch.object(
                runner,
                "_double_gate_for_evaluation",
                return_value=expected,
            ),
            patch.object(runner, "_run_locked_experiment") as evaluate,
            self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "not a prefix",
            ),
        ):
            runner.run_publication(self.root)
        evaluate.assert_not_called()
        self.assertFalse((self.run_directory / "states").exists())

    def test_zero_byte_pending_provenance_resumes_from_known_prefix(self) -> None:
        expected = source_provenance()
        self.make_parent()
        self.run_directory.mkdir(mode=0o700)
        self.run_directory.chmod(0o700)
        pending = self.run_directory / f".{SOURCE_PROVENANCE_FILE_NAME}.pending"
        pending.write_bytes(b"")
        pending.chmod(0o600)
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        try:
            observed = runner._preserve_source(descriptor, expected)
        finally:
            runner._close_descriptor(descriptor, "test run directory")
        self.assertEqual(observed.name, SOURCE_PROVENANCE_FILE_NAME)
        self.assertFalse(pending.exists())
        self.assertEqual(
            runner._decode_source_provenance(
                (self.run_directory / SOURCE_PROVENANCE_FILE_NAME).read_bytes()
            ),
            expected,
        )

    def test_prepared_status_verifies_state_bound_provenance(self) -> None:
        _provenance, artifact, store = self.initialize()
        store.close()
        status = runner.publication_status(self.root)
        self.assertEqual(status.stage, PublicationStage.PREPARED.value)
        self.assertEqual(status.artifacts, (artifact.name,))

        path = self.run_directory / SOURCE_PROVENANCE_FILE_NAME
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(runner.PublicationRunnerConflict):
            runner.publication_status(self.root)

    def test_unknown_run_entries_and_pending_files_fail_closed(self) -> None:
        self.initialize()[2].close()
        (self.run_directory / "notes.txt").write_text("unexpected")
        (self.run_directory / "notes.txt").chmod(0o600)
        with self.assertRaisesRegex(
            runner.PublicationRunnerConflict,
            "unknown or trailing",
        ):
            runner.publication_status(self.root)

    def test_symlink_and_hardlinked_artifacts_are_rejected(self) -> None:
        _provenance, _artifact, store = self.initialize()
        store.close()
        provenance_path = self.run_directory / SOURCE_PROVENANCE_FILE_NAME
        target = self.root / "target"
        target.write_bytes(provenance_path.read_bytes())
        target.chmod(0o600)
        provenance_path.unlink()
        provenance_path.symlink_to(target)
        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner.publication_status(self.root)

        self.temporary_directory.cleanup()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.gworker = self.root / ".gworker"
        self.publication = self.gworker / "publication"
        self.run_directory = self.publication / LOCKED_EVALUATION_RUN_KEY
        self.initialize()[2].close()
        provenance_path = self.run_directory / SOURCE_PROVENANCE_FILE_NAME
        os.link(provenance_path, self.root / "provenance-copy")
        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner.publication_status(self.root)

    def test_evaluating_reopen_is_burned_without_evaluator_call(self) -> None:
        _provenance, _artifact, store = self.initialize()
        store.begin_evaluation()
        store.close()
        self.assertEqual(
            runner.publication_status(self.root).disposition,
            "burned",
        )
        with (
            patch.object(runner, "_run_locked_experiment") as evaluate,
            self.assertRaises(runner.PublicationRunBurned),
        ):
            runner.run_publication(self.root)
        evaluate.assert_not_called()

    def test_evaluating_with_partial_result_is_still_unmistakably_burned(
        self,
    ) -> None:
        _provenance, _artifact, store = self.initialize()
        store.begin_evaluation()
        store.close()
        pending = self.run_directory / ".result.bin.pending"
        pending.write_bytes(b"")
        pending.chmod(0o600)
        self.assertEqual(
            runner.publication_status(self.root).disposition,
            "burned",
        )
        with (
            patch.object(runner, "_run_locked_experiment") as evaluate,
            self.assertRaises(runner.PublicationRunBurned),
        ):
            runner.run_publication(self.root)
        evaluate.assert_not_called()

    def test_descriptor_writer_retries_short_writes_and_rejects_zero(self) -> None:
        writer = runner._DescriptorWriter(123)
        with patch.object(runner.os, "write", side_effect=(2, 3)) as write:
            self.assertEqual(writer.write(b"abcde"), 5)
        self.assertEqual(write.call_args_list[0].args, (123, b"abcde"))
        self.assertEqual(write.call_args_list[1].args, (123, b"cde"))

        with (
            patch.object(runner.os, "write", return_value=0),
            self.assertRaisesRegex(
                runner.PublicationRunnerIOError,
                "no progress",
            ),
        ):
            writer.write(b"x")

    def test_dirty_source_stops_before_resource_and_claim(self) -> None:
        with (
            patch.object(
                runner,
                "_capture_locked_source",
                side_effect=runner.PublicationRunnerConflict("dirty"),
            ),
            patch.object(runner, "_resource_assessment") as resources,
            patch.object(runner, "_run_locked_experiment") as evaluate,
            self.assertRaises(runner.PublicationRunnerConflict),
        ):
            runner.run_publication(self.root)
        resources.assert_not_called()
        evaluate.assert_not_called()
        self.assertFalse(self.run_directory.exists())

    def test_capacity_failure_creates_no_run_and_issues_no_permit(self) -> None:
        expected = source_provenance()
        with (
            patch.object(
                runner,
                "_capture_locked_source",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_resource_assessment",
                return_value=resource_assessment(ready=False),
            ),
            patch.object(
                PublicationStateStore,
                "initialize",
                wraps=PublicationStateStore.initialize,
            ) as initialize,
            patch.object(runner, "_run_locked_experiment") as evaluate,
            self.assertRaises(ResourceCapacityError),
        ):
            runner.run_publication(self.root)
        initialize.assert_not_called()
        evaluate.assert_not_called()
        self.assertFalse(self.run_directory.exists())

    def test_source_is_rechecked_after_resource_before_claim(self) -> None:
        first = source_provenance()
        changed = source_provenance(variant=1)
        with (
            patch.object(
                runner,
                "_capture_locked_source",
                side_effect=(first, changed),
            ),
            patch.object(
                runner,
                "_resource_assessment",
                return_value=resource_assessment(),
            ),
            self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "changed during",
            ),
        ):
            runner._double_gate_for_evaluation(self.root)

    def test_fresh_coordinator_reaches_materialized_without_locked_eval(self) -> None:
        expected = source_provenance()
        recapture_observed_prepared = False

        def recapture(_root: Path) -> SourceProvenance:
            nonlocal recapture_observed_prepared
            recapture_observed_prepared = (
                (self.run_directory / SOURCE_PROVENANCE_FILE_NAME).is_file()
                and (self.run_directory / "states" / "000-prepared.json").is_file()
                and not (self.run_directory / "states" / "001-evaluating.json").exists()
            )
            return expected

        with (
            patch.object(
                runner,
                "_double_gate_for_evaluation",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_evaluate_once",
                side_effect=self.fake_evaluate,
            ),
            patch.object(
                runner,
                "_materialize",
                side_effect=self.fake_materialize,
            ),
            patch.object(
                runner,
                "_capture_locked_source",
                side_effect=recapture,
            ),
            patch.object(runner, "_run_locked_experiment") as locked_evaluator,
        ):
            status = runner.run_publication(self.root)
        locked_evaluator.assert_not_called()
        self.assertTrue(recapture_observed_prepared)
        self.assertEqual(status.stage, PublicationStage.MATERIALIZED.value)
        self.assertEqual(
            set(status.artifacts),
            {
                "evidence.json",
                "report.json",
                "result.bin",
                SOURCE_PROVENANCE_FILE_NAME,
            },
        )

    def test_prepared_reopen_advances_once_after_immediate_recapture(self) -> None:
        expected, _binding, store = self.initialize()
        store.close()
        with (
            patch.object(
                runner,
                "_double_gate_for_evaluation",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_capture_locked_source",
                return_value=expected,
            ) as recapture,
            patch.object(
                runner,
                "_evaluate_once",
                side_effect=self.fake_evaluate,
            ) as evaluate,
            patch.object(
                runner,
                "_materialize",
                side_effect=self.fake_materialize,
            ),
            patch.object(runner, "_run_locked_experiment") as locked_evaluator,
        ):
            status = runner.run_publication(self.root)
        self.assertEqual(status.stage, PublicationStage.MATERIALIZED.value)
        recapture.assert_called_once_with(self.root)
        evaluate.assert_called_once()
        locked_evaluator.assert_not_called()

    def test_evaluated_and_materialized_reopen_use_original_source(self) -> None:
        expected, _binding, store = self.initialize()
        self.fake_evaluate(store, -1)
        store.close()
        with (
            patch.object(
                runner,
                "_capture_locked_source",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_materialize",
                side_effect=self.fake_materialize,
            ) as materialize,
        ):
            evaluated_status = runner.run_publication(self.root)
        self.assertEqual(
            evaluated_status.stage,
            PublicationStage.MATERIALIZED.value,
        )
        materialize.assert_called_once()

        with (
            patch.object(
                runner,
                "_capture_locked_source",
                return_value=expected,
            ),
            patch.object(runner, "_materialize") as no_materialize,
        ):
            materialized_status = runner.run_publication(self.root)
        self.assertEqual(
            materialized_status.disposition,
            "ready-for-rendering",
        )
        no_materialize.assert_not_called()

    def test_evaluate_once_consumes_permit_and_binds_locked_inventory(self) -> None:
        _expected, _binding, store = self.initialize()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        result = locked_result()

        def execute(permit: object) -> ExperimentResult:
            cast(runner._LockedEvaluationPermit, permit).consume()
            return result

        def persist(
            _descriptor: int,
            _result: ExperimentResult,
        ) -> ArtifactBinding:
            return self.write_artifact("result.bin", b"locked-container")

        try:
            with (
                patch.object(
                    runner,
                    "_run_locked_experiment",
                    side_effect=execute,
                ) as evaluate,
                patch.object(runner, "_persist_result", side_effect=persist),
            ):
                record = runner._evaluate_once(store, descriptor)
        finally:
            runner._close_descriptor(descriptor, "test run directory")
            store.close()
        self.assertEqual(record.stage, PublicationStage.EVALUATED)
        evaluate.assert_called_once()
        self.assertEqual(
            {item.name for item in record.artifacts},
            {"result.bin", SOURCE_PROVENANCE_FILE_NAME},
        )

    def test_materialize_builds_independent_report_and_exact_inventory(self) -> None:
        _expected, _binding, store = self.initialize()
        evaluated = self.fake_evaluate(store, -1)
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        result = locked_result()
        report = object()
        cardinalities = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
        evidence = SimpleNamespace(
            statistics=report,
            completeness=SimpleNamespace(actual=cardinalities),
        )
        report_content = b'{"report":"canonical"}'
        evidence_content = b'{"evidence":"canonical"}'

        def publish(
            _descriptor: int,
            *,
            name: str,
            document: CanonicalJsonDocument,
            maximum_bytes: int,
        ) -> ArtifactBinding:
            self.assertGreater(maximum_bytes, len(document.content))
            return self.write_artifact(name, document.content)

        try:
            with (
                patch.object(runner, "_read_result", return_value=result),
                patch.object(
                    runner,
                    "build_statistical_report",
                    return_value=report,
                ) as build_report,
                patch.object(
                    runner,
                    "build_publication_evidence",
                    return_value=evidence,
                ) as build_evidence,
                patch.object(
                    runner,
                    "encode_statistical_report",
                    return_value=CanonicalJsonDocument(
                        report_content,
                        hashlib.sha256(report_content).hexdigest(),
                    ),
                ),
                patch.object(
                    runner,
                    "encode_publication_evidence",
                    return_value=CanonicalJsonDocument(
                        evidence_content,
                        hashlib.sha256(evidence_content).hexdigest(),
                    ),
                ),
                patch.object(
                    runner,
                    "_publish_json_document",
                    side_effect=publish,
                ),
                patch.object(runner, "_verify_json_round_trip") as round_trip,
            ):
                materialized = runner._materialize(
                    store,
                    descriptor,
                    evaluated,
                )
        finally:
            runner._close_descriptor(descriptor, "test run directory")
            store.close()
        self.assertEqual(materialized.stage, PublicationStage.MATERIALIZED)
        build_report.assert_called_once_with(
            result,
            resample_count=runner._EXPECTED_BOOTSTRAP_RESAMPLES,
        )
        build_evidence.assert_called_once_with(result)
        round_trip.assert_called_once()

    def test_provenance_only_crash_recovers_only_on_exact_source(self) -> None:
        expected, _binding = self.make_provenance_only()
        with (
            patch.object(
                runner,
                "_double_gate_for_evaluation",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_evaluate_once",
                side_effect=self.fake_evaluate,
            ),
            patch.object(
                runner,
                "_materialize",
                side_effect=self.fake_materialize,
            ),
            patch.object(
                runner,
                "_capture_locked_source",
                return_value=expected,
            ),
            patch.object(runner, "_run_locked_experiment") as locked_evaluator,
        ):
            status = runner.run_publication(self.root)
        self.assertEqual(status.stage, PublicationStage.MATERIALIZED.value)
        locked_evaluator.assert_not_called()

    def test_provenance_only_crash_rejects_changed_checkout(self) -> None:
        self.make_provenance_only(source_provenance())
        with (
            patch.object(
                runner,
                "_double_gate_for_evaluation",
                return_value=source_provenance(variant=1),
            ),
            patch.object(runner, "_run_locked_experiment") as evaluate,
            self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "differs",
            ),
        ):
            runner.run_publication(self.root)
        evaluate.assert_not_called()
        self.assertFalse((self.run_directory / "states").exists())

    def test_evaluated_resume_requires_original_clean_source(self) -> None:
        original, _artifact, store = self.initialize()
        self.fake_evaluate(store, -1)
        store.close()
        with (
            patch.object(
                runner,
                "_capture_locked_source",
                return_value=source_provenance(variant=1),
            ),
            patch.object(runner, "_materialize") as materialize,
            self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "differs",
            ),
        ):
            runner.run_publication(self.root)
        materialize.assert_not_called()
        self.assertNotEqual(original, source_provenance(variant=1))

    def test_evaluated_status_accepts_only_declared_materialization_pending(
        self,
    ) -> None:
        _original, _artifact, store = self.initialize()
        self.fake_evaluate(store, -1)
        store.close()
        pending = self.run_directory / ".report.json.pending"
        pending.write_bytes(b'{"complete":"pending"}')
        pending.chmod(0o600)
        self.assertEqual(
            runner.publication_status(self.root).disposition,
            "materialization-resumable",
        )
        unexpected = self.run_directory / ".other.pending"
        unexpected.write_bytes(b"x")
        unexpected.chmod(0o600)
        with self.assertRaises(runner.PublicationRunnerConflict):
            runner.publication_status(self.root)

    def test_json_artifact_publish_is_reusable_only_for_exact_bytes(self) -> None:
        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        content = b'{"canonical":true}'
        document = CanonicalJsonDocument(
            content,
            hashlib.sha256(content).hexdigest(),
        )
        try:
            first = runner._publish_json_document(
                descriptor,
                name="report.json",
                document=document,
                maximum_bytes=runner._MAX_REPORT_BYTES,
            )
            second = runner._publish_json_document(
                descriptor,
                name="report.json",
                document=document,
                maximum_bytes=runner._MAX_REPORT_BYTES,
            )
            self.assertEqual(first, second)
            changed = b'{"canonical":false}'
            with self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "differs",
            ):
                runner._publish_json_document(
                    descriptor,
                    name="report.json",
                    document=CanonicalJsonDocument(
                        changed,
                        hashlib.sha256(changed).hexdigest(),
                    ),
                    maximum_bytes=runner._MAX_REPORT_BYTES,
                )
        finally:
            runner._close_descriptor(descriptor, "test run directory")

    def test_exact_json_pending_file_is_recovered_after_publish_crash(self) -> None:
        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        content = b'{"canonical":true}'
        document = CanonicalJsonDocument(
            content,
            hashlib.sha256(content).hexdigest(),
        )
        try:
            with (
                patch.object(
                    runner,
                    "_rename_noreplace",
                    side_effect=runner.PublicationRunnerIOError("crash"),
                ),
                self.assertRaises(runner.PublicationRunnerIOError),
            ):
                runner._publish_json_document(
                    descriptor,
                    name="report.json",
                    document=document,
                    maximum_bytes=runner._MAX_REPORT_BYTES,
                )
            pending = self.run_directory / ".report.json.pending"
            self.assertTrue(pending.is_file())
            recovered = runner._publish_json_document(
                descriptor,
                name="report.json",
                document=document,
                maximum_bytes=runner._MAX_REPORT_BYTES,
            )
            self.assertEqual(recovered, binding("report.json", content))
            self.assertFalse(pending.exists())
        finally:
            runner._close_descriptor(descriptor, "test run directory")

    def test_prefix_and_empty_json_pending_files_resume_append_only(self) -> None:
        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        content = b'{"canonical":true}'
        document = CanonicalJsonDocument(
            content,
            hashlib.sha256(content).hexdigest(),
        )
        try:
            for prefix_length, name in (
                (0, "report.json"),
                (7, "evidence.json"),
            ):
                with self.subTest(prefix_length=prefix_length):
                    pending = self.run_directory / f".{name}.pending"
                    pending.write_bytes(content[:prefix_length])
                    pending.chmod(0o600)
                    recovered = runner._publish_json_document(
                        descriptor,
                        name=name,
                        document=document,
                        maximum_bytes=runner._MAX_REPORT_BYTES,
                    )
                    self.assertEqual(recovered, binding(name, content))
                    self.assertFalse(pending.exists())
        finally:
            runner._close_descriptor(descriptor, "test run directory")

    def test_mismatched_json_pending_file_is_never_deleted_or_reused(self) -> None:
        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        pending = self.run_directory / ".report.json.pending"
        pending.write_bytes(b'{"wrong":true}')
        pending.chmod(0o600)
        content = b'{"canonical":true}'
        try:
            with self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "not a prefix",
            ):
                runner._publish_json_document(
                    descriptor,
                    name="report.json",
                    document=CanonicalJsonDocument(
                        content,
                        hashlib.sha256(content).hexdigest(),
                    ),
                    maximum_bytes=runner._MAX_REPORT_BYTES,
                )
            self.assertEqual(pending.read_bytes(), b'{"wrong":true}')
            self.assertFalse((self.run_directory / "report.json").exists())
        finally:
            runner._close_descriptor(descriptor, "test run directory")

    def test_publish_faults_leave_no_overwrite_path(self) -> None:
        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        content = b'{"canonical":true}'
        document = CanonicalJsonDocument(
            content,
            hashlib.sha256(content).hexdigest(),
        )
        try:
            with (
                patch.object(runner.os, "write", side_effect=(1, 0)),
                self.assertRaises(runner.PublicationRunnerIOError),
            ):
                runner._publish_json_document(
                    descriptor,
                    name="report.json",
                    document=document,
                    maximum_bytes=runner._MAX_REPORT_BYTES,
                )
            self.assertFalse((self.run_directory / "report.json").exists())
            self.assertTrue((self.run_directory / ".report.json.pending").exists())
        finally:
            runner._close_descriptor(descriptor, "test run directory")

    def test_fsync_rename_open_and_stat_failures_are_normalized(self) -> None:
        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        content = b'{"canonical":true}'
        document = CanonicalJsonDocument(
            content,
            hashlib.sha256(content).hexdigest(),
        )
        try:
            with (
                patch.object(
                    runner,
                    "_fsync",
                    side_effect=runner.PublicationRunnerIOError("fsync"),
                ),
                self.assertRaises(runner.PublicationRunnerIOError),
            ):
                runner._publish_json_document(
                    descriptor,
                    name="report.json",
                    document=document,
                    maximum_bytes=runner._MAX_REPORT_BYTES,
                )
        finally:
            runner._close_descriptor(descriptor, "test run directory")

        self.temporary_directory.cleanup()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.gworker = self.root / ".gworker"
        self.publication = self.gworker / "publication"
        self.run_directory = self.publication / LOCKED_EVALUATION_RUN_KEY
        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        try:
            with (
                patch.object(
                    runner,
                    "_rename_noreplace",
                    side_effect=runner.PublicationRunnerIOError("rename"),
                ),
                self.assertRaises(runner.PublicationRunnerIOError),
            ):
                runner._publish_json_document(
                    descriptor,
                    name="report.json",
                    document=document,
                    maximum_bytes=runner._MAX_REPORT_BYTES,
                )
        finally:
            runner._close_descriptor(descriptor, "test run directory")

        with (
            patch.object(runner.os, "open", side_effect=OSError("open")),
            self.assertRaises(runner.PublicationRunnerSecurityError),
        ):
            runner._open_repo_root(self.root)
        with (
            patch.object(runner.os, "fstat", side_effect=OSError("stat")),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            runner._fstat(0, "test descriptor")

    def test_final_artifact_recovery_reissues_directory_fsync(self) -> None:
        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        content = b'{"canonical":true}'
        document = CanonicalJsonDocument(
            content,
            hashlib.sha256(content).hexdigest(),
        )
        real_fsync = runner._fsync

        def fail_directory_fsync(fd: int, field: str) -> None:
            if field == "publication run directory":
                raise runner.PublicationRunnerIOError("injected directory fsync")
            real_fsync(fd, field)

        try:
            with (
                patch.object(
                    runner,
                    "_fsync",
                    side_effect=fail_directory_fsync,
                ),
                self.assertRaises(runner.PublicationRunnerIOError),
            ):
                runner._publish_json_document(
                    descriptor,
                    name="report.json",
                    document=document,
                    maximum_bytes=runner._MAX_REPORT_BYTES,
                )
            self.assertTrue((self.run_directory / "report.json").is_file())
            with patch.object(
                runner,
                "_fsync",
                wraps=real_fsync,
            ) as recovered_fsync:
                recovered = runner._publish_json_document(
                    descriptor,
                    name="report.json",
                    document=document,
                    maximum_bytes=runner._MAX_REPORT_BYTES,
                )
            self.assertEqual(recovered, binding("report.json", content))
            self.assertIn(
                (descriptor, "publication run directory"),
                tuple(call.args for call in recovered_fsync.call_args_list),
            )
        finally:
            runner._close_descriptor(descriptor, "test run directory")

    def test_existing_private_parent_recovery_reissues_parent_fsync(self) -> None:
        root_descriptor = runner._open_repo_root(self.root)
        real_fsync = runner._fsync
        failed = False

        def fail_after_mkdir(fd: int, field: str) -> None:
            nonlocal failed
            if field == "private state directory parent" and not failed:
                failed = True
                raise runner.PublicationRunnerIOError("injected parent fsync")
            real_fsync(fd, field)

        try:
            with (
                patch.object(runner, "_fsync", side_effect=fail_after_mkdir),
                self.assertRaises(runner.PublicationRunnerIOError),
            ):
                runner._mkdir_private_at(
                    root_descriptor,
                    ".gworker",
                    "private state directory",
                )
            self.assertTrue(self.gworker.is_dir())
            with patch.object(
                runner,
                "_fsync",
                wraps=real_fsync,
            ) as recovered_fsync:
                descriptor = runner._mkdir_private_at(
                    root_descriptor,
                    ".gworker",
                    "private state directory",
                )
            runner._close_descriptor(descriptor, "private state directory")
            self.assertIn(
                (root_descriptor, "private state directory parent"),
                tuple(call.args for call in recovered_fsync.call_args_list),
            )
        finally:
            runner._close_descriptor(root_descriptor, "repository root")

    def test_locked_call_site_has_no_configurable_experiment_surface(self) -> None:
        _provenance, _binding, store = self.initialize()
        permit = store.begin_evaluation()
        sentinel = cast(ExperimentResult, object())
        with patch.object(
            runner,
            "run_experiment",
            return_value=sentinel,
        ) as execute:
            self.assertIs(runner._run_locked_experiment(permit), sentinel)
        execute.assert_called_once_with(
            DEFAULT_EXPERIMENT_CONFIG,
            _eval_permit=permit,
        )
        store.close()

    def test_result_binding_uses_independent_whole_container_digest(self) -> None:
        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        result = cast(ExperimentResult, object())

        def write_container(_result: object, destination: object) -> str:
            cast(runner._DescriptorWriter, destination).write(b"whole-container")
            return "1" * 64

        try:
            with (
                patch.object(
                    runner,
                    "write_experiment_result",
                    side_effect=write_container,
                ),
                patch.object(runner, "_read_result", return_value=result),
            ):
                observed = runner._persist_result(descriptor, result)
        finally:
            runner._close_descriptor(descriptor, "test run directory")
        self.assertEqual(
            observed.content_sha256,
            hashlib.sha256(b"whole-container").hexdigest(),
        )
        self.assertNotEqual(observed.content_sha256, "1" * 64)

    def test_materialized_and_sealed_status_verify_every_bound_file(self) -> None:
        _source, _binding, store = self.initialize()
        evaluated = self.fake_evaluate(store, -1)
        materialized = self.fake_materialize(store, -1, evaluated)
        manifest = self.write_artifact("manifest.json", b'{"manifest":"locked"}')
        rendered = self.write_artifact("artifacts/summary.svg", b"<svg/>")
        sealed = store.seal(
            manifest=manifest,
            rendered_artifacts=(rendered,),
        )
        store.close()
        self.assertEqual(sealed.stage, PublicationStage.SEALED)
        self.assertEqual(
            runner.publication_status(self.root).stage,
            PublicationStage.SEALED.value,
        )

        (self.run_directory / "artifacts" / "summary.svg").write_bytes(
            b"<svg>tampered</svg>"
        )
        with self.assertRaisesRegex(
            runner.PublicationRunnerConflict,
            "disagrees",
        ):
            runner.publication_status(self.root)
        self.assertEqual(materialized.stage, PublicationStage.MATERIALIZED)

    def test_preflight_output_is_redacted(self) -> None:
        expected = source_provenance()
        assessment = resource_assessment()
        with (
            patch.object(
                runner,
                "_capture_locked_source",
                return_value=expected,
            ),
            patch.object(
                runner,
                "_resource_assessment",
                return_value=assessment,
            ),
        ):
            result = runner.publication_preflight(self.root)
        payload = result.as_dict()
        self.assertTrue(payload["ready"])
        self.assertNotIn("cgroup", json.dumps(payload))
        self.assertNotIn(str(self.root), json.dumps(payload))

    def test_cli_status_capacity_and_burned_codes_are_deterministic(self) -> None:
        stdout = io.StringIO()
        with (
            patch.object(
                runner,
                "publication_status",
                return_value=runner.PublicationRunnerStatus(
                    "unclaimed",
                    "not-started",
                    (),
                ),
            ),
            patch.object(runner.sys, "stdout", stdout),
        ):
            self.assertEqual(
                runner.main(["--repo-root", str(self.root), "status"]),
                0,
            )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["stage"], "unclaimed")
        self.assertNotIn(str(self.root), stdout.getvalue())

        stdout = io.StringIO()
        failed = runner._preflight_from_assessment(resource_assessment(ready=False))
        with (
            patch.object(
                runner,
                "publication_preflight",
                return_value=failed,
            ),
            patch.object(runner.sys, "stdout", stdout),
        ):
            self.assertEqual(
                runner.main(["--repo-root", str(self.root), "preflight"]),
                2,
            )
        self.assertFalse(json.loads(stdout.getvalue())["ok"])

        stdout = io.StringIO()
        with (
            patch.object(
                runner,
                "run_publication",
                side_effect=runner.PublicationRunBurned("burned"),
            ),
            patch.object(runner.sys, "stdout", stdout),
        ):
            self.assertEqual(
                runner.main(["--repo-root", str(self.root), "run"]),
                3,
            )
        self.assertEqual(json.loads(stdout.getvalue())["error"], "evaluation-burned")

    def test_cli_has_no_force_reset_or_retry_flags(self) -> None:
        parser = runner._parser()
        for forbidden in ("--force", "--reset", "--retry-evaluation"):
            stderr = io.StringIO()
            with (
                self.subTest(forbidden=forbidden),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args([forbidden, "run"])
            self.assertIn(forbidden, stderr.getvalue())

    def test_result_reader_rejects_tampered_noncanonical_bytes(self) -> None:
        self.make_provenance_only()
        result = self.write_artifact("result.bin", b"not-a-result")
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        try:
            with self.assertRaises(ValueError):
                runner._read_result(descriptor, result)
        finally:
            runner._close_descriptor(descriptor, "test run directory")

    def test_result_reader_revalidates_reopened_locked_identity(self) -> None:
        self.make_provenance_only()
        result_binding = self.write_artifact("result.bin", b"canonical-container")
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        expected = locked_result()
        try:
            with patch.object(
                runner,
                "read_experiment_result",
                return_value=expected,
            ) as decode:
                observed = runner._read_result(descriptor, result_binding)
        finally:
            runner._close_descriptor(descriptor, "test run directory")
        self.assertIs(observed, expected)
        decode.assert_called_once()

    def test_json_round_trip_requires_exact_decoded_objects(self) -> None:
        self.make_provenance_only()
        report_content = b'{"report":"canonical"}'
        evidence_content = b'{"evidence":"canonical"}'
        report_binding = self.write_artifact("report.json", report_content)
        evidence_binding = self.write_artifact("evidence.json", evidence_content)
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        report = cast(runner.StatisticalReport, object())
        evidence = cast(runner.PublicationEvidence, object())
        try:
            with (
                patch.object(
                    runner,
                    "decode_statistical_report",
                    return_value=report,
                ) as decode_report,
                patch.object(
                    runner,
                    "decode_publication_evidence",
                    return_value=evidence,
                ) as decode_evidence,
            ):
                runner._verify_json_round_trip(
                    descriptor,
                    report=report,
                    evidence=evidence,
                    report_binding=report_binding,
                    evidence_binding=evidence_binding,
                )
        finally:
            runner._close_descriptor(descriptor, "test run directory")
        decode_report.assert_called_once_with(
            report_content,
            expected_content_sha256=report_binding.content_sha256,
            config=DEFAULT_EXPERIMENT_CONFIG,
        )
        decode_evidence.assert_called_once_with(
            evidence_content,
            expected_content_sha256=evidence_binding.content_sha256,
        )

    def test_close_failure_is_reported(self) -> None:
        with (
            patch.object(runner.os, "close", side_effect=OSError("close")),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            runner._close_descriptor(123, "test descriptor")

    def test_multi_descriptor_cleanup_attempts_every_close_after_failure(
        self,
    ) -> None:
        self.make_provenance_only()
        real_close = os.close
        closed: list[int] = []
        failed = False

        def fail_first(descriptor: int) -> None:
            nonlocal failed
            closed.append(descriptor)
            if not failed:
                failed = True
                raise OSError("injected first close failure")
            real_close(descriptor)

        with (
            patch.object(runner.os, "close", side_effect=fail_first),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            runner._open_existing_run_directory(self.root)
        self.assertGreaterEqual(len(closed), 4)
        self.assertEqual(len(closed), len(set(closed)))

    def test_status_is_unclaimed_without_mutation(self) -> None:
        status = runner.publication_status(self.root)
        self.assertEqual(status.stage, "unclaimed")
        self.assertFalse(self.gworker.exists())

    def test_unsafe_noncanonical_repo_root_is_rejected(self) -> None:
        candidate = Path(f"{self.root}/..") / self.root.name
        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner.publication_status(candidate)

        link = self.root.parent / f"{self.root.name}-link"
        link.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(link.unlink)
        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner.publication_status(link)

    def test_hostile_low_level_path_instances_fail_before_path_methods(self) -> None:
        path_type = type(Path())
        missing_state = object.__new__(path_type)
        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner._canonical_repo_root(cast(Path, missing_state))

        mutated = Path(self.root)
        object.__setattr__(
            mutated,
            runner._PATH_RAW_COMPONENTS_SLOT,
            [str(self.root), cast(str, object())],
        )
        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner._canonical_repo_root(mutated)

    def test_run_output_does_not_expose_exception_paths(self) -> None:
        stdout = io.StringIO()
        error = runner.PublicationRunnerSecurityError(f"unsafe path: {self.root}")
        with (
            patch.object(runner, "run_publication", side_effect=error),
            patch.object(runner.sys, "stdout", stdout),
        ):
            self.assertEqual(
                runner.main(["--repo-root", str(self.root), "run"]),
                4,
            )
        self.assertNotIn(str(self.root), stdout.getvalue())

    def test_status_rejects_incomplete_state_initialization(self) -> None:
        self.make_parent()
        self.run_directory.mkdir(mode=0o700)
        self.run_directory.chmod(0o700)
        (self.run_directory / "lock").write_bytes(b"")
        (self.run_directory / "lock").chmod(0o600)
        with self.assertRaises(runner.PublicationRunnerConflict):
            runner.publication_status(self.root)

    def test_preflight_failure_payload_contains_codes_not_paths(self) -> None:
        assessment = resource_assessment(ready=False)
        summary = runner._preflight_from_assessment(assessment)
        self.assertEqual(summary.failure_codes, ("memory-headroom",))
        self.assertNotIn("/", json.dumps(summary.as_dict()))

    def test_value_objects_and_descriptor_adapters_fail_closed(self) -> None:
        invalid_statuses = (
            (1, "ready", ()),
            ("ready", 1, ()),
            ("ready", "ready", []),
            ("ready", "ready", (1,)),
        )
        for stage, disposition, artifacts in invalid_statuses:
            with (
                self.subTest(stage=stage, disposition=disposition),
                self.assertRaises(runner.PublicationRunnerError),
            ):
                runner.PublicationRunnerStatus(
                    cast(str, stage),
                    cast(str, disposition),
                    cast(tuple[str, ...], artifacts),
                )

        writer = runner._DescriptorWriter(123)
        with self.assertRaises(runner.PublicationRunnerIOError):
            writer.write(cast(bytes, "not-bytes"))
        with (
            patch.object(runner.os, "write", side_effect=OSError("write")),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            writer.write(b"x")
        with (
            patch.object(runner.os, "write", return_value=2),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            writer.write(b"x")

        reader = runner._DescriptorReader(123)
        for size in (-1, cast(int, True)):
            with (
                self.subTest(size=size),
                self.assertRaises(runner.PublicationRunnerIOError),
            ):
                reader.read(size)
        with (
            patch.object(runner.os, "read", side_effect=OSError("read")),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            reader.read(1)

        with (
            patch.object(runner.os, "fsync", side_effect=OSError("fsync")),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            runner._fsync(123, "test")
        with (
            patch.object(runner.os, "stat", side_effect=OSError("stat")),
            self.assertRaises(runner.PublicationRunnerSecurityError),
        ):
            runner._stat_at(123, "x", "test")
        with (
            patch.object(runner.os, "listdir", side_effect=OSError("list")),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            runner._list_directory(123, "test")
        with (
            patch.object(runner.os, "listdir", return_value=[b"x"]),
            self.assertRaises(runner.PublicationRunnerSecurityError),
        ):
            runner._list_directory(123, "test")
        with (
            patch.object(runner.os, "stat", side_effect=PermissionError("stat")),
            self.assertRaises(runner.PublicationRunnerSecurityError),
        ):
            runner._entry_metadata_or_none(123, "x")

    def test_private_metadata_validators_reject_every_unsafe_shape(self) -> None:
        regular = self.root / "regular"
        regular.write_bytes(b"x")
        regular.chmod(0o600)
        directory = self.root / "directory"
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)
        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner._validate_private_directory(regular.stat(), "test")
        directory.chmod(0o755)
        with self.assertRaisesRegex(
            runner.PublicationRunnerSecurityError,
            "0700",
        ):
            runner._validate_private_directory(directory.stat(), "test")
        directory.chmod(0o700)
        with (
            patch.object(runner.os, "geteuid", return_value=os.geteuid() + 1),
            self.assertRaisesRegex(
                runner.PublicationRunnerSecurityError,
                "owner",
            ),
        ):
            runner._validate_private_directory(directory.stat(), "test")

        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner._validate_private_file(directory.stat(), "test")
        regular.chmod(0o644)
        with self.assertRaisesRegex(
            runner.PublicationRunnerSecurityError,
            "0600",
        ):
            runner._validate_private_file(regular.stat(), "test")
        regular.chmod(0o600)
        linked = self.root / "linked"
        os.link(regular, linked)
        with self.assertRaisesRegex(
            runner.PublicationRunnerSecurityError,
            "hard links",
        ):
            runner._validate_private_file(regular.stat(), "test")
        linked.unlink()
        metadata = regular.stat()
        negative_size = SimpleNamespace(
            st_mode=metadata.st_mode,
            st_uid=metadata.st_uid,
            st_nlink=1,
            st_size=-1,
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
        )
        with self.assertRaisesRegex(
            runner.PublicationRunnerSecurityError,
            "size",
        ):
            runner._validate_private_file(
                cast(os.stat_result, negative_size),
                "test",
            )

    def test_source_identity_and_provenance_defenses_are_closed(self) -> None:
        expected = source_provenance()
        with patch.object(
            runner,
            "capture_source_provenance",
            return_value=expected,
        ) as capture:
            self.assertEqual(runner._capture_locked_source(self.root), expected)
        self.assertEqual(
            tuple(
                path for path, _location in capture.call_args.kwargs["loaded_sources"]
            ),
            runner._EXPECTED_SOURCE_PATHS,
        )

        wrong_inventory = source_provenance()
        object.__setattr__(
            wrong_inventory,
            "loaded_sources",
            (
                SourceFileIdentity(
                    relative_path="src/gworker/wrong.py",
                    sha256="0" * 64,
                ),
            ),
        )
        with (
            patch.object(
                runner,
                "capture_source_provenance",
                return_value=wrong_inventory,
            ),
            self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "inventory",
            ),
        ):
            runner._capture_locked_source(self.root)

        with (
            patch.object(runner, "LOCKED_POLICY_ID", "changed"),
            self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "locked policy",
            ),
        ):
            runner._validate_locked_runtime()
        with (
            patch.object(
                runner,
                "validate_experiment_config",
                side_effect=ValueError("invalid"),
            ),
            self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "configuration",
            ),
        ):
            runner._validate_locked_runtime()
        with (
            patch.object(runner, "evaluator_design_fingerprint", return_value="wrong"),
            self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "disagree",
            ),
        ):
            runner._validate_locked_runtime()

        with self.assertRaises(runner.PublicationRunnerConflict):
            runner._encode_source_provenance(cast(SourceProvenance, object()))
        mutated = source_provenance()
        object.__setattr__(mutated, "branch", object())
        with self.assertRaisesRegex(
            runner.PublicationRunnerConflict,
            "encoded",
        ):
            runner._encode_source_provenance(mutated)

        canonical = runner._encode_source_provenance(expected)
        malformed = (
            canonical.replace(b'"branch":"', b'"branch":1,"ignored":"', 1),
            canonical.replace(b'"branch":"', b'"branch":false,"ignored":"', 1),
            canonical.replace(b'"source_commit":"', b'"unknown":"x","source_commit":"'),
            canonical.replace(b'"loaded_sources":[', b'"loaded_sources":[], "x":[', 1),
            canonical.replace(b'"sha256":"', b'"sha256":"bad', 1),
        )
        for content in malformed:
            with (
                self.subTest(content=content[:40]),
                self.assertRaises(runner.PublicationRunnerConflict),
            ):
                runner._decode_source_provenance(content)

    def test_artifact_helpers_reject_corruption_and_io_faults(self) -> None:
        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner._hash_open_artifact(
                123,
                identity=(1, 1),
                initial_size=0,
                maximum_bytes=1,
                field="test",
            )
        with (
            patch.object(runner.os, "read", side_effect=OSError("read")),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            runner._hash_open_artifact(
                123,
                identity=(1, 1),
                initial_size=1,
                maximum_bytes=1,
                field="test",
            )
        with self.assertRaises(runner.PublicationRunnerSecurityError):
            runner._open_artifact_parent(123, "../unsafe")
        for unsafe_name in ("../report.json", "/tmp/report.json", "nested/x", "\x00"):
            with (
                self.subTest(unsafe_name=unsafe_name),
                self.assertRaises(runner.PublicationRunnerSecurityError),
            ):
                runner._publish_immutable_content(
                    123,
                    name=unsafe_name,
                    content=b"x",
                    content_sha256=hashlib.sha256(b"x").hexdigest(),
                    maximum_bytes=1,
                )
        with (
            patch.object(runner.ctypes, "CDLL", side_effect=OSError("loader")),
            self.assertRaises(runner.PublicationRunnerSecurityError),
        ):
            runner._rename_noreplace(123, "source", "destination")
        with (
            patch.object(
                runner.ctypes,
                "CDLL",
                return_value=SimpleNamespace(renameat2=lambda *_args: -1),
            ),
            patch.object(runner.ctypes, "get_errno", return_value=runner.errno.EEXIST),
            self.assertRaises(runner.PublicationRunnerConflict),
        ):
            runner._rename_noreplace(123, "source", "destination")
        with (
            patch.object(
                runner.ctypes,
                "CDLL",
                return_value=SimpleNamespace(renameat2=lambda *_args: -1),
            ),
            patch.object(runner.ctypes, "get_errno", return_value=runner.errno.EIO),
            self.assertRaises(runner.PublicationRunnerIOError),
        ):
            runner._rename_noreplace(123, "source", "destination")

        self.make_provenance_only()
        descriptor = runner._open_existing_run_directory(self.root)
        assert descriptor is not None
        try:
            first, _identity = runner._create_pending_artifact(
                descriptor,
                ".report.json.pending",
                "pending report",
            )
            runner._close_descriptor(first, "pending report")
            with self.assertRaises(runner.PublicationRunnerConflict):
                runner._create_pending_artifact(
                    descriptor,
                    ".report.json.pending",
                    "pending report",
                )
            with self.assertRaises(runner.PublicationRunnerConflict):
                runner._publish_immutable_content(
                    descriptor,
                    name="evidence.json",
                    content=b"x",
                    content_sha256="0" * 64,
                    maximum_bytes=10,
                )
            final = self.run_directory / "evidence.json"
            final.write_bytes(b"x")
            final.chmod(0o600)
            trailing = self.run_directory / ".evidence.json.pending"
            trailing.write_bytes(b"x")
            trailing.chmod(0o600)
            with self.assertRaisesRegex(
                runner.PublicationRunnerConflict,
                "trailing pending",
            ):
                runner._publish_immutable_content(
                    descriptor,
                    name="evidence.json",
                    content=b"x",
                    content_sha256=hashlib.sha256(b"x").hexdigest(),
                    maximum_bytes=10,
                )
        finally:
            runner._close_descriptor(descriptor, "test run directory")

    def test_locked_result_and_cli_error_classification_fail_closed(self) -> None:
        with self.assertRaises(runner.PublicationRunnerConflict):
            runner._validate_locked_result(cast(ExperimentResult, object()))
        changed = locked_result()
        object.__setattr__(changed, "policy_id", "wrong")
        with self.assertRaisesRegex(
            runner.PublicationRunnerConflict,
            "identity",
        ):
            runner._validate_locked_result(changed)
        with self.assertRaises(runner.PublicationRunnerConflict):
            runner._run_locked_experiment(
                cast(runner._LockedEvaluationPermit, object())
            )
        empty_record = SimpleNamespace(artifacts=())
        with self.assertRaises(runner.PublicationRunnerConflict):
            runner._result_binding(cast(runner.PublicationStateRecord, empty_record))
        with self.assertRaises(runner.PublicationRunnerConflict):
            runner._source_binding(cast(runner.PublicationStateRecord, empty_record))

        classifications = (
            (
                runner.PublicationStateSecurityError("unsafe"),
                4,
                "unsafe-publication-path",
            ),
            (runner.PublicationRunBusy("busy"), 5, "publication-busy"),
            (runner.PublicationStateCorrupt("corrupt"), 3, "publication-conflict"),
            (runner.PublicationStateIOError("io"), 1, "publication-io"),
            (RuntimeError("other"), 1, "publication-failed"),
        )
        for error, expected_code, expected_name in classifications:
            with self.subTest(error=error):
                code, payload = runner._safe_error_payload(error)
                self.assertEqual(code, expected_code)
                self.assertEqual(payload["error"], expected_name)


if __name__ == "__main__":
    unittest.main()
