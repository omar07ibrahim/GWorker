from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import gworker.publication_state as state_module
from gworker.evaluation import (
    DEFAULT_EXPERIMENT_CONFIG,
    LOCKED_EVALUATION_RUN_KEY,
)
from gworker.evidence import expected_publication_cardinalities
from gworker.publication_state import (
    ArtifactBinding,
    CardinalityBinding,
    PublicationRunBusy,
    PublicationRunExists,
    PublicationRunInspection,
    PublicationStage,
    PublicationStateConflict,
    PublicationStateCorrupt,
    PublicationStateError,
    PublicationStateIOError,
    PublicationStateSecurityError,
    PublicationStateStore,
    decode_state_record,
    encode_state_record,
)


def artifact(name: str, content: bytes | None = None) -> ArtifactBinding:
    payload = content if content is not None else f"content:{name}".encode()
    return ArtifactBinding(
        name=name,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
    )


class PublicationStateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.run_directory = self.root / LOCKED_EVALUATION_RUN_KEY

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def initialize(self) -> PublicationStateStore:
        return PublicationStateStore.initialize(self.run_directory)

    @staticmethod
    def evaluate(store: PublicationStateStore) -> None:
        permit = store.begin_evaluation()
        permit.consume()
        expected = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
        store.record_evaluated(
            result=artifact("result.bin"),
            cluster_summaries=expected.cluster_summaries,
            abrupt_traces=expected.abrupt_traces,
            hard_failure_count=0,
        )

    @staticmethod
    def materialize(store: PublicationStateStore) -> None:
        store.record_materialized(
            report=artifact("report.json"),
            evidence=artifact("evidence.json"),
            cardinalities=expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG),
        )

    def state_path(self, sequence: int) -> Path:
        return self.run_directory / "states" / state_module._STATE_FILE_NAMES[sequence]

    def rewrite_state(self, sequence: int, payload: dict[str, object]) -> None:
        self.state_path(sequence).write_bytes(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        )

    def test_initialize_is_private_canonical_and_durable(self) -> None:
        with self.initialize() as store:
            inspection = store.inspect()

        self.assertEqual(inspection.current.stage, PublicationStage.PREPARED)
        self.assertEqual(inspection.current.sequence, 0)
        self.assertIsNone(inspection.current.previous_record_sha256)
        self.assertEqual(
            inspection.current_sha256,
            hashlib.sha256(self.state_path(0).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            stat.S_IMODE(self.run_directory.stat().st_mode),
            0o700,
        )
        self.assertEqual(
            stat.S_IMODE((self.run_directory / "states").stat().st_mode),
            0o700,
        )
        self.assertEqual(
            stat.S_IMODE((self.run_directory / "lock").stat().st_mode),
            0o600,
        )
        self.assertEqual(
            stat.S_IMODE(self.state_path(0).stat().st_mode),
            0o600,
        )
        self.assertEqual(
            encode_state_record(decode_state_record(self.state_path(0).read_bytes())),
            self.state_path(0).read_bytes(),
        )

    def test_full_lifecycle_preserves_chain_artifacts_and_counts(self) -> None:
        with self.initialize() as store:
            self.evaluate(store)
            self.materialize(store)
            sealed = store.seal(
                manifest=artifact("manifest.json"),
                rendered_artifacts=(
                    artifact("artifacts/summary.svg"),
                    artifact("artifacts/evidence.csv"),
                ),
            )
            inspection = store.inspect()

        self.assertEqual(sealed.stage, PublicationStage.SEALED)
        self.assertEqual(len(inspection.records), 5)
        self.assertEqual(
            tuple(record.stage for record in inspection.records),
            tuple(PublicationStage),
        )
        for sequence, record in enumerate(inspection.records):
            expected_previous = (
                None if sequence == 0 else inspection.record_sha256s[sequence - 1]
            )
            self.assertEqual(record.previous_record_sha256, expected_previous)
        self.assertEqual(
            tuple(binding.name for binding in sealed.artifacts),
            (
                "artifacts/evidence.csv",
                "artifacts/summary.svg",
                "evidence.json",
                "manifest.json",
                "report.json",
                "result.bin",
            ),
        )
        self.assertIn(
            CardinalityBinding("hard_failure_count", 0),
            sealed.cardinalities,
        )

    def test_evaluating_is_durable_before_permit_is_issued(self) -> None:
        with self.initialize() as store:
            observed: list[PublicationStage] = []
            real_issue = state_module._issue_locked_evaluation_permit

            def issue(
                *,
                run_key: str,
                claim_sha256: str,
            ) -> state_module._LockedEvaluationPermit:
                self.assertEqual(run_key, LOCKED_EVALUATION_RUN_KEY)
                self.assertRegex(claim_sha256, r"^[0-9a-f]{64}$")
                observed.append(store.inspect().current.stage)
                self.assertTrue(self.state_path(1).is_file())
                return real_issue(
                    run_key=run_key,
                    claim_sha256=claim_sha256,
                )

            with patch.object(
                state_module,
                "_issue_locked_evaluation_permit",
                side_effect=issue,
            ):
                permit = store.begin_evaluation()

        self.assertIs(type(permit), state_module._LockedEvaluationPermit)
        self.assertEqual(observed, [PublicationStage.EVALUATING])

    def test_permit_failure_still_burns_the_durable_run(self) -> None:
        store = self.initialize()
        with (
            patch.object(
                state_module,
                "_issue_locked_evaluation_permit",
                side_effect=RuntimeError("permit failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "permit failure"),
        ):
            store.begin_evaluation()
        self.assertEqual(store.inspect().current.stage, PublicationStage.EVALUATING)
        store.close()

        with PublicationStateStore.open(self.run_directory) as reopened:
            expected = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "cannot resume",
            ):
                reopened.record_evaluated(
                    result=artifact("result.bin"),
                    cluster_summaries=expected.cluster_summaries,
                    abrupt_traces=expected.abrupt_traces,
                    hard_failure_count=0,
                )

    def test_reopened_prepared_run_may_claim_but_evaluating_may_not_resume(
        self,
    ) -> None:
        self.initialize().close()
        with PublicationStateStore.open(self.run_directory) as store:
            store.begin_evaluation()
        with PublicationStateStore.open(self.run_directory) as reopened:
            self.assertEqual(
                reopened.inspect().current.stage,
                PublicationStage.EVALUATING,
            )
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "cannot resume",
            ):
                expected = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
                reopened.record_evaluated(
                    result=artifact("result.bin"),
                    cluster_summaries=expected.cluster_summaries,
                    abrupt_traces=expected.abrupt_traces,
                    hard_failure_count=0,
                )

    def test_post_evaluation_materialization_can_resume(self) -> None:
        store = self.initialize()
        self.evaluate(store)
        store.close()

        with PublicationStateStore.open(self.run_directory) as reopened:
            self.materialize(reopened)
            self.assertEqual(
                reopened.inspect().current.stage,
                PublicationStage.MATERIALIZED,
            )

    def test_only_the_next_transition_is_accepted(self) -> None:
        with self.initialize() as store:
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "requires evaluated",
            ):
                self.materialize(store)
            store.begin_evaluation()
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "cannot enter evaluating",
            ):
                store.begin_evaluation()
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "requires materialized",
            ):
                store.seal(
                    manifest=artifact("manifest.json"),
                    rendered_artifacts=(artifact("artifacts/summary.svg"),),
                )

    def test_result_binding_requires_the_exact_consumed_permit(self) -> None:
        with self.initialize() as store:
            permit = store.begin_evaluation()
            expected = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "must be consumed",
            ):
                store.record_evaluated(
                    result=artifact("result.bin"),
                    cluster_summaries=expected.cluster_summaries,
                    abrupt_traces=expected.abrupt_traces,
                    hard_failure_count=0,
                )
            permit.consume()
            evaluated = store.record_evaluated(
                result=artifact("result.bin"),
                cluster_summaries=expected.cluster_summaries,
                abrupt_traces=expected.abrupt_traces,
                hard_failure_count=0,
            )
            self.assertEqual(evaluated.stage, PublicationStage.EVALUATED)

    def test_deleted_permit_slots_fail_closed(self) -> None:
        expected = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
        for field, pattern in (
            ("_claim_sha256", "cannot resume"),
            ("_consumed", "must be consumed"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
                with PublicationStateStore.initialize(run) as store:
                    permit = store.begin_evaluation()
                    permit.consume()
                    object.__delattr__(permit, field)
                    with self.assertRaisesRegex(
                        PublicationStateConflict,
                        pattern,
                    ):
                        store.record_evaluated(
                            result=artifact("result.bin"),
                            cluster_summaries=expected.cluster_summaries,
                            abrupt_traces=expected.abrupt_traces,
                            hard_failure_count=0,
                        )

    def test_locked_result_cardinality_is_exact(self) -> None:
        with self.initialize() as store:
            store.begin_evaluation().consume()
            expected = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "locked inventory",
            ):
                store.record_evaluated(
                    result=artifact("result.bin"),
                    cluster_summaries=expected.cluster_summaries - 1,
                    abrupt_traces=expected.abrupt_traces,
                    hard_failure_count=0,
                )
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "locked inventory",
            ):
                store.record_evaluated(
                    result=artifact("result.bin"),
                    cluster_summaries=expected.cluster_summaries,
                    abrupt_traces=expected.abrupt_traces,
                    hard_failure_count=1,
                )

    def test_artifact_names_and_stage_inventory_are_closed(self) -> None:
        for name in (
            "../result.bin",
            "/result.bin",
            "artifacts//result.svg",
            r"artifacts\result.svg",
            "artifacts/é.svg",
        ):
            with (
                self.subTest(name=name),
                self.assertRaises(PublicationStateConflict),
            ):
                artifact(name)

        with self.initialize() as store:
            self.evaluate(store)
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "report.json",
            ):
                store.record_materialized(
                    report=artifact("wrong.json"),
                    evidence=artifact("evidence.json"),
                    cardinalities=expected_publication_cardinalities(
                        DEFAULT_EXPERIMENT_CONFIG
                    ),
                )

    def test_seal_requires_unique_rendered_artifacts_beneath_artifacts(
        self,
    ) -> None:
        with self.initialize() as store:
            self.evaluate(store)
            self.materialize(store)
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "non-empty",
            ):
                store.seal(
                    manifest=artifact("manifest.json"),
                    rendered_artifacts=(),
                )
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "beneath artifacts",
            ):
                store.seal(
                    manifest=artifact("manifest.json"),
                    rendered_artifacts=(artifact("summary.svg"),),
                )
            duplicate = artifact("artifacts/summary.svg")
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "unique",
            ):
                store.seal(
                    manifest=artifact("manifest.json"),
                    rendered_artifacts=(duplicate, duplicate),
                )

    def test_exclusive_lock_blocks_a_second_store_and_releases_on_close(
        self,
    ) -> None:
        first = self.initialize()
        with self.assertRaises(PublicationRunBusy):
            PublicationStateStore.open(self.run_directory)
        first.close()

        reopened = PublicationStateStore.open(self.run_directory)
        reopened.close()
        reopened.close()

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_inherited_store_rejects_use_and_child_close_keeps_parent_lock(
        self,
    ) -> None:
        store = self.initialize()
        read_descriptor, write_descriptor = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_descriptor)
            outcomes: list[str] = []
            try:
                store.inspect()
            except PublicationStateConflict:
                outcomes.append("use-rejected")
            except BaseException as error:
                outcomes.append(f"raw:{type(error).__name__}")
            else:
                outcomes.append("use-accepted")
            try:
                store.close()
            except BaseException as error:
                outcomes.append(f"close:{type(error).__name__}")
            else:
                outcomes.append("child-closed")
            os.write(write_descriptor, ",".join(outcomes).encode())
            os.close(write_descriptor)
            os._exit(0)

        os.close(write_descriptor)
        child_outcome = os.read(read_descriptor, 256)
        os.close(read_descriptor)
        waited_pid, wait_status = os.waitpid(child_pid, 0)
        self.assertEqual(waited_pid, child_pid)
        self.assertTrue(os.WIFEXITED(wait_status))
        self.assertEqual(os.WEXITSTATUS(wait_status), 0)
        self.assertEqual(child_outcome, b"use-rejected,child-closed")

        self.assertEqual(store.inspect().current.stage, PublicationStage.PREPARED)
        with self.assertRaises(PublicationRunBusy):
            PublicationStateStore.open(self.run_directory)
        store.close()
        PublicationStateStore.open(self.run_directory).close()

    def test_initialization_never_resets_an_existing_claim(self) -> None:
        self.initialize().close()
        original = self.state_path(0).read_bytes()
        with self.assertRaises(PublicationRunExists):
            self.initialize()
        self.assertEqual(self.state_path(0).read_bytes(), original)

    def test_path_must_be_absolute_canonical_and_use_locked_key(self) -> None:
        candidates = (
            Path(LOCKED_EVALUATION_RUN_KEY),
            Path("/tmp") / "wrong-run",
            f"{self.root}/../{self.root.name}/{LOCKED_EVALUATION_RUN_KEY}",
            f"{self.run_directory}/",
            f"//tmp/{LOCKED_EVALUATION_RUN_KEY}",
            f"{self.root}/bad\n/{LOCKED_EVALUATION_RUN_KEY}",
            f"{self.root}/bad\ud800/{LOCKED_EVALUATION_RUN_KEY}",
        )
        for candidate in candidates:
            with (
                self.subTest(candidate=candidate),
                self.assertRaises(PublicationStateSecurityError),
            ):
                PublicationStateStore.initialize(candidate)

        for candidate in (None, b"/tmp/synthetic-eval-v3-eval"):
            with (
                self.subTest(candidate=candidate),
                self.assertRaises(PublicationStateSecurityError),
            ):
                PublicationStateStore.initialize(cast(str | Path, candidate))

    def test_parent_path_symlink_is_rejected(self) -> None:
        actual = self.root / "actual"
        actual.mkdir(mode=0o700)
        link = self.root / "link"
        link.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(
            PublicationStateSecurityError,
            "real directories",
        ):
            PublicationStateStore.initialize(link / LOCKED_EVALUATION_RUN_KEY)

    def test_public_parent_and_run_modes_are_rejected(self) -> None:
        public_parent = self.root / "public"
        public_parent.mkdir(mode=0o755)
        public_parent.chmod(0o755)
        with self.assertRaisesRegex(
            PublicationStateSecurityError,
            "mode must be 0700",
        ):
            PublicationStateStore.initialize(public_parent / LOCKED_EVALUATION_RUN_KEY)

        self.run_directory.mkdir(mode=0o700)
        self.run_directory.chmod(0o755)
        with self.assertRaisesRegex(
            PublicationStateSecurityError,
            "mode must be 0700",
        ):
            self.initialize()

    def test_wrong_owner_is_rejected_before_state_access(self) -> None:
        with (
            patch.object(os, "geteuid", return_value=os.geteuid() + 1),
            self.assertRaisesRegex(
                PublicationStateSecurityError,
                "wrong owner",
            ),
        ):
            self.initialize()

    def test_lock_state_directory_and_record_symlinks_are_rejected(self) -> None:
        with self.subTest(entry="lock"):
            self.run_directory.mkdir(mode=0o700)
            target = self.root / "lock-target"
            target.touch(mode=0o600)
            (self.run_directory / "lock").symlink_to(target)
            with self.assertRaises(PublicationStateSecurityError):
                PublicationStateStore.open(self.run_directory)

        # Use separate roots because each malformed run is intentionally burned.
        for entry in ("states", "record"):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                run = root / LOCKED_EVALUATION_RUN_KEY
                if entry == "states":
                    run.mkdir(mode=0o700)
                    (run / "lock").touch(mode=0o600)
                    target = root / "states-target"
                    target.mkdir(mode=0o700)
                    (run / "states").symlink_to(target, target_is_directory=True)
                    with self.assertRaises(PublicationStateSecurityError):
                        PublicationStateStore.open(run)
                else:
                    PublicationStateStore.initialize(run).close()
                    state_path = run / "states" / "000-prepared.json"
                    target = root / "state-target"
                    state_path.rename(target)
                    state_path.symlink_to(target)
                    with self.assertRaises(PublicationStateSecurityError):
                        PublicationStateStore.open(run)

    def test_record_public_mode_and_hard_link_are_rejected(self) -> None:
        self.initialize().close()
        self.state_path(0).chmod(0o644)
        with self.assertRaisesRegex(
            PublicationStateSecurityError,
            "mode must be 0600",
        ):
            PublicationStateStore.open(self.run_directory)

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            PublicationStateStore.initialize(run).close()
            os.link(
                run / "states" / "000-prepared.json",
                run / "record-copy",
            )
            with self.assertRaisesRegex(
                PublicationStateSecurityError,
                "hard links",
            ):
                PublicationStateStore.open(run)

    def test_replaced_run_or_states_directory_is_detected_by_identity(
        self,
    ) -> None:
        store = self.initialize()
        moved = self.root / "moved-run"
        self.run_directory.rename(moved)
        self.run_directory.mkdir(mode=0o700)
        with self.assertRaisesRegex(
            PublicationStateSecurityError,
            "identity changed",
        ):
            store.inspect()
        store.close()

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            store = PublicationStateStore.initialize(run)
            moved_states = run / "moved-states"
            (run / "states").rename(moved_states)
            (run / "states").mkdir(mode=0o700)
            with self.assertRaisesRegex(
                PublicationStateSecurityError,
                "identity changed",
            ):
                store.inspect()
            store.close()

    def test_gap_and_trailing_state_entries_are_rejected(self) -> None:
        store = self.initialize()
        store.begin_evaluation()
        store.close()
        self.state_path(0).unlink()
        with self.assertRaisesRegex(
            PublicationStateCorrupt,
            "gap",
        ):
            PublicationStateStore.open(self.run_directory)

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            PublicationStateStore.initialize(run).close()
            (run / "states" / "notes.txt").write_text("not state")
            with self.assertRaisesRegex(
                PublicationStateCorrupt,
                "trailing entries",
            ):
                PublicationStateStore.open(run)

    def test_filename_content_reorder_and_sha_chain_tamper_are_rejected(
        self,
    ) -> None:
        store = self.initialize()
        store.begin_evaluation()
        store.close()
        first = self.state_path(0).read_bytes()
        second = self.state_path(1).read_bytes()
        self.state_path(0).write_bytes(second)
        self.state_path(1).write_bytes(first)
        with self.assertRaisesRegex(
            PublicationStateCorrupt,
            "filename and content",
        ):
            PublicationStateStore.open(self.run_directory)

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            store = PublicationStateStore.initialize(run)
            store.begin_evaluation()
            store.close()
            second_path = run / "states" / "001-evaluating.json"
            payload = json.loads(second_path.read_bytes())
            payload["previous_record_sha256"] = "0" * 64
            second_path.write_bytes(
                json.dumps(
                    payload,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            )
            with self.assertRaisesRegex(
                PublicationStateCorrupt,
                "chain",
            ):
                PublicationStateStore.open(run)

    def test_resigned_identity_and_artifact_tamper_are_rejected(self) -> None:
        self.initialize().close()
        payload = json.loads(self.state_path(0).read_bytes())
        payload["evaluator_id"] = "synthetic-eval-v3." + "0" * 64
        self.rewrite_state(0, payload)
        with self.assertRaisesRegex(
            PublicationStateCorrupt,
            "evaluator identity",
        ):
            PublicationStateStore.open(self.run_directory)

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            store = PublicationStateStore.initialize(run)
            self.evaluate(store)
            self.materialize(store)
            store.close()
            evaluated_path = run / "states" / "002-evaluated.json"
            payload = json.loads(evaluated_path.read_bytes())
            payload["artifacts"][0]["content_sha256"] = "0" * 64
            evaluated_path.write_bytes(
                json.dumps(
                    payload,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            )
            with self.assertRaisesRegex(
                PublicationStateCorrupt,
                "chain",
            ):
                PublicationStateStore.open(run)

    def test_decoder_rejects_duplicate_unknown_missing_and_noncanonical_json(
        self,
    ) -> None:
        self.initialize().close()
        canonical = self.state_path(0).read_bytes()
        malformed = (
            canonical.replace(
                b'"artifacts":[]',
                b'"artifacts":[],"artifacts":[]',
                1,
            ),
            canonical.replace(
                b'"artifacts":[]',
                b'"artifacts":[],"unknown":0',
                1,
            ),
            canonical.replace(b'"artifacts":[],', b"", 1),
            b" " + canonical,
        )
        patterns = (
            "duplicate",
            "missing or unknown",
            "missing or unknown",
            "canonical",
        )
        for content, pattern in zip(malformed, patterns, strict=True):
            with (
                self.subTest(pattern=pattern),
                self.assertRaisesRegex(PublicationStateCorrupt, pattern),
            ):
                decode_state_record(content)

    def test_decoder_rejects_non_ascii_float_boolean_and_long_integer(self) -> None:
        self.initialize().close()
        canonical = self.state_path(0).read_bytes()
        malformed = (
            canonical.replace(b'"sequence":0', b'"sequence":0.0'),
            canonical.replace(b'"sequence":0', b'"sequence":true'),
            canonical.replace(b'"sequence":0', b'"sequence":12345678901234567890'),
            canonical.replace(b'"run_key":"', b'"run_key":"\\u00e9'),
        )
        patterns = ("floats", "exact integer", "too long", "run key")
        for content, pattern in zip(malformed, patterns, strict=True):
            with (
                self.subTest(pattern=pattern),
                self.assertRaisesRegex(PublicationStateCorrupt, pattern),
            ):
                decode_state_record(content)

    def test_low_level_mutated_record_fails_closed_at_encode(self) -> None:
        with self.initialize() as store:
            record = store.inspect().current
        object.__setattr__(record, "config_sha256", "é")
        with self.assertRaises(PublicationStateError):
            encode_state_record(record)

    def test_low_level_mutated_nested_bindings_fail_closed(self) -> None:
        with self.initialize() as store:
            self.evaluate(store)

        artifact_mutations = (
            ("name", "../result.bin", "unsafe path"),
            ("content_sha256", "A" * 64, "lowercase SHA"),
            ("content_sha256", "é", "lowercase SHA"),
            ("byte_count", True, "integer"),
            ("byte_count", 0, "integer"),
        )
        for field, value, pattern in artifact_mutations:
            record = decode_state_record(self.state_path(2).read_bytes())
            object.__setattr__(record.artifacts[0], field, value)
            with (
                self.subTest(field=field, value=value),
                self.assertRaisesRegex(PublicationStateConflict, pattern),
            ):
                encode_state_record(record)

        cardinality_mutations = (
            ("name", "Bad-Name", "name"),
            ("name", 1, "name"),
            ("value", True, "integer"),
            ("value", -1, "integer"),
        )
        for field, value, pattern in cardinality_mutations:
            record = decode_state_record(self.state_path(2).read_bytes())
            object.__setattr__(record.cardinalities[0], field, value)
            with (
                self.subTest(field=field, value=value),
                self.assertRaisesRegex(PublicationStateConflict, pattern),
            ):
                encode_state_record(record)

        for target_kind, field, pattern in (
            ("artifact", "content_sha256", "lowercase SHA"),
            ("cardinality", "value", "integer"),
        ):
            record = decode_state_record(self.state_path(2).read_bytes())
            target = (
                record.artifacts[0]
                if target_kind == "artifact"
                else record.cardinalities[0]
            )
            object.__delattr__(target, field)
            with (
                self.subTest(target_kind=target_kind, field=field),
                self.assertRaisesRegex(PublicationStateConflict, pattern),
            ):
                encode_state_record(record)

    def test_record_structure_and_locked_identity_are_revalidated(self) -> None:
        with self.initialize() as store:
            prepared = store.inspect().current
        mutations = (
            ("schema_version", 1, "schema_version must"),
            ("run_key", 1, "run_key must"),
            ("sequence", True, "sequence must"),
            ("stage", "prepared", "stage must"),
            ("previous_record_sha256", "bad", "lowercase SHA"),
            ("config_sha256", "0" * 64, "config identity"),
            ("evaluator_id", "", "evaluator_id"),
            ("design_id", "wrong", "design identity"),
            ("population_id", "wrong", "population identity"),
            ("policy_id", "wrong", "policy identity"),
            ("artifacts", [], "exact binding tuple"),
            ("cardinalities", [], "exact binding tuple"),
        )
        for field, value, pattern in mutations:
            record = decode_state_record(encode_state_record(prepared))
            object.__setattr__(record, field, value)
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(PublicationStateConflict, pattern),
            ):
                encode_state_record(record)

        deleted = decode_state_record(encode_state_record(prepared))
        object.__delattr__(deleted, "stage")
        with self.assertRaisesRegex(PublicationStateConflict, "stage must"):
            encode_state_record(deleted)

        for changes, pattern in (
            ({"schema_version": "wrong"}, "schema version"),
            ({"run_key": "other-run"}, "run key"),
            ({"sequence": 1}, "sequence and stage"),
            ({"previous_record_sha256": "0" * 64}, "previous digest"),
            (
                {"artifacts": (artifact("result.bin"),)},
                "cannot bind output",
            ),
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(PublicationStateConflict, pattern),
            ):
                replace(prepared, **changes)

    def test_every_stage_rejects_foreign_inventory(self) -> None:
        with self.initialize() as store:
            self.evaluate(store)
            self.materialize(store)
            store.seal(
                manifest=artifact("manifest.json"),
                rendered_artifacts=(artifact("artifacts/summary.svg"),),
            )
            records = store.inspect().records
        evaluated = records[2]
        materialized = records[3]
        sealed = records[4]

        invalid_records = (
            (
                replace(
                    evaluated.artifacts[0],
                    name="report.json",
                ),
                "artifact",
            ),
        )
        for wrong_artifact, _label in invalid_records:
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "only result.bin",
            ):
                replace(evaluated, artifacts=(wrong_artifact,))

        with self.assertRaisesRegex(
            PublicationStateConflict,
            "cardinalities are invalid",
        ):
            replace(evaluated, cardinalities=evaluated.cardinalities[:-1])
        changed_count = replace(
            evaluated.cardinalities[0],
            value=evaluated.cardinalities[0].value + 1,
        )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "cardinality values",
        ):
            replace(
                evaluated,
                cardinalities=(changed_count, *evaluated.cardinalities[1:]),
            )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "artifact inventory",
        ):
            replace(materialized, artifacts=materialized.artifacts[:-1])
        changed_evidence_count = replace(
            materialized.cardinalities[0],
            value=materialized.cardinalities[0].value + 1,
        )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "evidence cardinalities",
        ):
            replace(
                materialized,
                cardinalities=(
                    changed_evidence_count,
                    *materialized.cardinalities[1:],
                ),
            )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "missing a materialized",
        ):
            replace(
                sealed,
                artifacts=tuple(
                    item for item in sealed.artifacts if item.name != "evidence.json"
                ),
            )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "missing manifest",
        ):
            replace(
                sealed,
                artifacts=tuple(
                    item for item in sealed.artifacts if item.name != "manifest.json"
                ),
            )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "rendered artifacts",
        ):
            replace(
                sealed,
                artifacts=tuple(
                    item
                    for item in sealed.artifacts
                    if not item.name.startswith("artifacts/")
                ),
            )

    def test_duplicate_unsorted_and_oversized_binding_sets_are_rejected(
        self,
    ) -> None:
        with self.initialize() as store:
            self.evaluate(store)
            self.materialize(store)
            sealed = store.seal(
                manifest=artifact("manifest.json"),
                rendered_artifacts=(artifact("artifacts/summary.svg"),),
            )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "artifacts are not uniquely canonical",
        ):
            replace(
                sealed,
                artifacts=(sealed.artifacts[1], sealed.artifacts[0]),
            )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "artifacts are not uniquely canonical",
        ):
            replace(
                sealed,
                artifacts=(sealed.artifacts[0], sealed.artifacts[0]),
            )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "cardinalities are not uniquely canonical",
        ):
            replace(
                sealed,
                cardinalities=(
                    sealed.cardinalities[1],
                    sealed.cardinalities[0],
                ),
            )
        too_many = tuple(
            artifact(f"artifacts/render-{index:02d}.svg")
            for index in range(state_module._MAX_ARTIFACTS + 1)
        )
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "too many artifacts",
        ):
            replace(sealed, artifacts=too_many)

    def test_value_object_bounds_are_exact(self) -> None:
        invalid_artifacts = (
            ("", "name length"),
            ("a" * (state_module._MAX_ARTIFACT_NAME_BYTES + 1), "name length"),
        )
        for name, pattern in invalid_artifacts:
            with (
                self.subTest(name_length=len(name)),
                self.assertRaisesRegex(PublicationStateConflict, pattern),
            ):
                artifact(name)
        for digest in ("0" * 63, "A" * 64):
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "lowercase SHA",
            ):
                ArtifactBinding("result.bin", digest, 1)
        for byte_count in (True, 0, state_module._MAX_ARTIFACT_BYTES + 1):
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "byte_count must be an integer",
            ):
                ArtifactBinding("result.bin", "0" * 64, byte_count)
        for name in ("", "UPPER", "dash-name", "é"):
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "cardinality name",
            ):
                CardinalityBinding(name, 0)
        for value in (True, -1, state_module._MAX_CARDINALITY + 1):
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "must be an integer",
            ):
                CardinalityBinding("decisions", value)

    def test_decoder_enforces_content_and_structure_bounds(self) -> None:
        malformed = (
            (cast(bytes, bytearray(b"{}")), "exact bytes"),
            (b"", "byte length"),
            (b"x" * (state_module._MAX_STATE_BYTES + 1), "byte length"),
            (b"\xff", "canonical ASCII"),
            (b'"\\ud800"', "not valid Unicode"),
            (b'{"value":NaN}', "constants"),
            (b"{", "JSON is invalid"),
            (
                (
                    b"["
                    + b",".join(
                        b"[]" for _ in range(state_module._MAX_STATE_CONTAINERS + 1)
                    )
                    + b"]"
                ),
                "structure exceeds bounds",
            ),
            (
                (
                    b"["
                    + b",".join(
                        b"{}" for _ in range(state_module._MAX_STATE_CONTAINERS + 1)
                    )
                    + b"]"
                ),
                "too many objects",
            ),
            (
                json.dumps("x" * (state_module._MAX_JSON_STRING_BYTES + 1)).encode(),
                "string exceeds bounds",
            ),
            (b"9223372036854775808", "out of range"),
        )
        for content, pattern in malformed:
            with (
                self.subTest(pattern=pattern),
                self.assertRaisesRegex(PublicationStateCorrupt, pattern),
            ):
                decode_state_record(content)

    def test_decoder_rejects_oversized_binding_arrays_before_construction(
        self,
    ) -> None:
        self.initialize().close()
        payload = json.loads(self.state_path(0).read_bytes())
        payload["artifacts"] = [{} for _ in range(state_module._MAX_ARTIFACTS + 1)]
        with self.assertRaisesRegex(
            PublicationStateCorrupt,
            "too many artifacts",
        ):
            decode_state_record(
                json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            )
        payload = json.loads(self.state_path(0).read_bytes())
        payload["cardinalities"] = [
            {} for _ in range(len(state_module._EVIDENCE_CARDINALITY_NAMES) + 2)
        ]
        with self.assertRaisesRegex(
            PublicationStateCorrupt,
            "too many cardinalities",
        ):
            decode_state_record(
                json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            )

    def test_inspection_and_encoder_reject_wrong_root_objects(self) -> None:
        empty = PublicationRunInspection((), ())
        with self.assertRaisesRegex(PublicationStateCorrupt, "journal is empty"):
            _ = empty.current
        with self.assertRaisesRegex(PublicationStateCorrupt, "journal is empty"):
            _ = empty.current_sha256
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "exact PublicationStateRecord",
        ):
            encode_state_record(cast(state_module.PublicationStateRecord, object()))

        with self.initialize() as store:
            record = store.inspect().current
        with (
            patch.object(
                state_module,
                "_MAX_STATE_BYTES",
                1,
            ),
            self.assertRaisesRegex(
                PublicationStateConflict,
                "too large",
            ),
        ):
            encode_state_record(record)
        with (
            patch.object(
                json,
                "dumps",
                side_effect=ValueError("injected encoder failure"),
            ),
            patch.object(
                state_module,
                "_validate_loaded_locked_identities",
            ),
            self.assertRaisesRegex(
                PublicationStateConflict,
                "cannot encode",
            ),
        ):
            encode_state_record(record)

    def test_decoder_rejects_wrong_nested_json_kinds(self) -> None:
        self.initialize().close()
        canonical_payload = json.loads(self.state_path(0).read_bytes())
        mutations = (
            ("root", [], "state must be an object"),
            ("artifacts", {}, "artifacts must be an array"),
            ("cardinalities", {}, "cardinalities must be an array"),
            ("run_key", 1, "run_key must be exact text"),
            ("stage", "unknown", "fields are invalid"),
            ("previous_record_sha256", 1, "must be exact text"),
        )
        for field, value, pattern in mutations:
            payload = value if field == "root" else {**canonical_payload, field: value}
            content = json.dumps(
                payload,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(PublicationStateCorrupt, pattern),
            ):
                decode_state_record(content)

        for field, value in (
            ("artifacts", [0]),
            ("cardinalities", [0]),
        ):
            payload = {**canonical_payload, field: value}
            with self.assertRaisesRegex(
                PublicationStateCorrupt,
                "must be an object",
            ):
                decode_state_record(
                    json.dumps(
                        payload,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                )

    def test_fifo_entries_are_rejected_without_blocking(self) -> None:
        self.initialize().close()
        self.state_path(0).unlink()
        os.mkfifo(self.state_path(0), mode=0o600)
        with self.assertRaisesRegex(
            PublicationStateSecurityError,
            "regular file",
        ):
            PublicationStateStore.open(self.run_directory)

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            run.mkdir(mode=0o700)
            (run / "states").mkdir(mode=0o700)
            os.mkfifo(run / "lock", mode=0o600)
            with self.assertRaisesRegex(
                PublicationStateSecurityError,
                "regular file",
            ):
                PublicationStateStore.open(run)

    def test_replay_detects_entry_created_during_final_recheck(self) -> None:
        store = self.initialize()
        real_listdir = os.listdir
        calls = 0

        def changing_listdir(descriptor: int) -> list[str]:
            nonlocal calls
            calls += 1
            entries = real_listdir(descriptor)
            return entries if calls == 1 else [*entries, "001-evaluating.json"]

        with (
            patch.object(os, "listdir", side_effect=changing_listdir),
            self.assertRaisesRegex(
                PublicationStateCorrupt,
                "changed during replay",
            ),
        ):
            store.inspect()
        store.close()

    def test_replay_final_relist_failure_is_normalized(self) -> None:
        store = self.initialize()
        real_listdir = os.listdir
        calls = 0

        def failing_relist(descriptor: int) -> list[str]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected relist failure")
            return real_listdir(descriptor)

        with (
            patch.object(os, "listdir", side_effect=failing_relist),
            self.assertRaisesRegex(PublicationStateIOError, "relist"),
        ):
            store.inspect()
        store.close()

    def test_static_transition_replay_rejects_coordinated_changes(self) -> None:
        with self.initialize() as store:
            self.evaluate(store)
            self.materialize(store)
            store.seal(
                manifest=artifact("manifest.json"),
                rendered_artifacts=(artifact("artifacts/summary.svg"),),
            )
            records = store.inspect().records

        prepared, evaluating, evaluated, materialized, sealed = records
        adjacent = decode_state_record(encode_state_record(evaluating))
        object.__setattr__(adjacent, "sequence", 2)
        with self.assertRaisesRegex(PublicationStateCorrupt, "not adjacent"):
            state_module.PublicationStateStore._validate_transition(
                prepared,
                adjacent,
            )

        changed_identity = decode_state_record(encode_state_record(evaluating))
        object.__setattr__(changed_identity, "policy_id", "changed")
        with self.assertRaisesRegex(PublicationStateCorrupt, "changed policy_id"):
            state_module.PublicationStateStore._validate_transition(
                prepared,
                changed_identity,
            )

        changed_artifact = decode_state_record(encode_state_record(materialized))
        object.__setattr__(
            changed_artifact.artifacts[-1],
            "content_sha256",
            "0" * 64,
        )
        with self.assertRaisesRegex(
            PublicationStateCorrupt,
            "changed an artifact",
        ):
            state_module.PublicationStateStore._validate_transition(
                evaluated,
                changed_artifact,
            )

        changed_result_count = decode_state_record(encode_state_record(materialized))
        result_count = next(
            item
            for item in changed_result_count.cardinalities
            if item.name == "cluster_summaries"
        )
        object.__setattr__(result_count, "value", result_count.value + 1)
        with self.assertRaisesRegex(
            PublicationStateCorrupt,
            "changed result cardinality",
        ):
            state_module.PublicationStateStore._validate_transition(
                evaluated,
                changed_result_count,
            )

        changed_sealed_count = decode_state_record(encode_state_record(sealed))
        object.__setattr__(
            changed_sealed_count.cardinalities[0],
            "value",
            changed_sealed_count.cardinalities[0].value + 1,
        )
        with self.assertRaisesRegex(
            PublicationStateCorrupt,
            "changed publication cardinalities",
        ):
            state_module.PublicationStateStore._validate_transition(
                materialized,
                changed_sealed_count,
            )

    def test_missing_run_nonempty_lock_and_state_directory_claim_fail_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            PublicationStateSecurityError,
            "does not exist",
        ):
            PublicationStateStore.open(self.run_directory)

        self.initialize().close()
        (self.run_directory / "lock").write_bytes(b"claimed")
        with self.assertRaisesRegex(
            PublicationStateSecurityError,
            "lock must be empty",
        ):
            PublicationStateStore.open(self.run_directory)

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            run.mkdir(mode=0o700)
            (run / "states").mkdir(mode=0o700)
            with self.assertRaisesRegex(
                PublicationRunExists,
                "states directory",
            ):
                PublicationStateStore.initialize(run)

    def test_mkdir_stat_lock_and_zero_write_failures_are_normalized(self) -> None:
        real_mkdir = os.mkdir

        def fail_run_mkdir(
            path: object,
            *args: object,
            **kwargs: object,
        ) -> None:
            if path == LOCKED_EVALUATION_RUN_KEY:
                raise OSError("injected mkdir failure")
            real_mkdir(path, *args, **kwargs)

        with (
            patch.object(os, "mkdir", side_effect=fail_run_mkdir),
            self.assertRaisesRegex(PublicationStateIOError, "create publication run"),
        ):
            self.initialize()

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            run.mkdir(mode=0o700)
            real_mkdir = os.mkdir

            def fail_states_mkdir(
                path: object,
                *args: object,
                **kwargs: object,
            ) -> None:
                if path == "states":
                    raise OSError("injected states mkdir failure")
                real_mkdir(path, *args, **kwargs)

            with (
                patch.object(os, "mkdir", side_effect=fail_states_mkdir),
                self.assertRaisesRegex(
                    PublicationStateIOError,
                    "states directory",
                ),
            ):
                PublicationStateStore.initialize(run)

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            run.mkdir(mode=0o700)
            real_flock = state_module.fcntl.flock

            def fail_lock(descriptor: int, operation: int) -> None:
                if operation & state_module.fcntl.LOCK_NB:
                    raise OSError("injected lock failure")
                real_flock(descriptor, operation)

            with (
                patch.object(
                    state_module.fcntl,
                    "flock",
                    side_effect=fail_lock,
                ),
                self.assertRaisesRegex(PublicationStateIOError, "acquire"),
            ):
                PublicationStateStore.initialize(run)

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            store = PublicationStateStore.initialize(run)
            with (
                patch.object(os, "write", return_value=0),
                self.assertRaisesRegex(
                    PublicationStateIOError,
                    "made no progress",
                ),
            ):
                store.begin_evaluation()
            store.close()

    def test_record_evaluated_and_materialized_argument_kinds_are_closed(
        self,
    ) -> None:
        with self.initialize() as store:
            permit = store.begin_evaluation()
            permit.consume()
            expected = expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "requires result.bin",
            ):
                store.record_evaluated(
                    result=artifact("report.json"),
                    cluster_summaries=expected.cluster_summaries,
                    abrupt_traces=expected.abrupt_traces,
                    hard_failure_count=0,
                )
            store.record_evaluated(
                result=artifact("result.bin"),
                cluster_summaries=expected.cluster_summaries,
                abrupt_traces=expected.abrupt_traces,
                hard_failure_count=0,
            )
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "evidence.json",
            ):
                store.record_materialized(
                    report=artifact("report.json"),
                    evidence=artifact("wrong.json"),
                    cardinalities=expected,
                )
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "exact EvidenceCardinalities",
            ):
                store.record_materialized(
                    report=artifact("report.json"),
                    evidence=artifact("evidence.json"),
                    cardinalities=cast(
                        state_module.EvidenceCardinalities,
                        object(),
                    ),
                )
            altered = replace(
                expected,
                trace_points=expected.trace_points - 1,
            )
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "inventory is not locked",
            ):
                store.record_materialized(
                    report=artifact("report.json"),
                    evidence=artifact("evidence.json"),
                    cardinalities=altered,
                )

    def test_seal_rejects_wrong_manifest_and_non_tuple_outputs(self) -> None:
        with self.initialize() as store:
            self.evaluate(store)
            self.materialize(store)
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "manifest.json",
            ):
                store.seal(
                    manifest=artifact("report.json"),
                    rendered_artifacts=(artifact("artifacts/summary.svg"),),
                )
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "exact non-empty",
            ):
                store.seal(
                    manifest=artifact("manifest.json"),
                    rendered_artifacts=cast(
                        tuple[ArtifactBinding, ...],
                        [artifact("artifacts/summary.svg")],
                    ),
                )
            rendered = artifact("artifacts/summary.svg")
            object.__setattr__(rendered, "name", None)
            with self.assertRaisesRegex(
                PublicationStateConflict,
                "exact text",
            ):
                store.seal(
                    manifest=artifact("manifest.json"),
                    rendered_artifacts=(rendered,),
                )

    def test_close_failures_are_normalized_and_release_the_lock(self) -> None:
        store = self.initialize()
        state_descriptor = store._state_descriptor
        real_close = os.close
        failed = False

        def fail_one_close(descriptor: int) -> None:
            nonlocal failed
            if descriptor == state_descriptor and not failed:
                failed = True
                raise OSError("injected close failure")
            real_close(descriptor)

        with (
            patch.object(os, "close", side_effect=fail_one_close),
            self.assertRaisesRegex(PublicationStateIOError, "completely close"),
        ):
            store.close()
        real_close(state_descriptor)
        with PublicationStateStore.open(self.run_directory):
            pass

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / LOCKED_EVALUATION_RUN_KEY
            store = PublicationStateStore.initialize(run)
            real_flock = state_module.fcntl.flock

            def fail_unlock(descriptor: int, operation: int) -> None:
                if operation == state_module.fcntl.LOCK_UN:
                    raise OSError("injected unlock failure")
                real_flock(descriptor, operation)

            with (
                patch.object(
                    state_module.fcntl,
                    "flock",
                    side_effect=fail_unlock,
                ),
                self.assertRaisesRegex(
                    PublicationStateIOError,
                    "completely close",
                ),
            ):
                store.close()

    def test_partial_write_failure_leaves_a_burned_invalid_prefix(self) -> None:
        store = self.initialize()
        real_write = os.write
        calls = 0

        def partial_then_fail(descriptor: int, content: object) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                data = bytes(content)
                return real_write(descriptor, data[:7])
            raise OSError("injected write failure")

        with (
            patch.object(os, "write", side_effect=partial_then_fail),
            self.assertRaisesRegex(PublicationStateIOError, "write"),
        ):
            store.begin_evaluation()
        store.close()
        self.assertTrue(self.state_path(1).exists())
        with self.assertRaises(PublicationStateCorrupt):
            PublicationStateStore.open(self.run_directory)

    def test_file_fsync_failure_never_issues_a_permit(self) -> None:
        store = self.initialize()
        with (
            patch.object(os, "fsync", side_effect=OSError("injected fsync")),
            patch.object(state_module, "_issue_locked_evaluation_permit") as issue,
            self.assertRaisesRegex(PublicationStateIOError, "fsync"),
        ):
            store.begin_evaluation()
        issue.assert_not_called()
        store.close()
        self.assertTrue(self.state_path(1).exists())

    def test_directory_fsync_failure_commits_no_retryable_permission(self) -> None:
        store = self.initialize()
        real_fsync = os.fsync
        calls = 0

        def fail_directory_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected directory fsync")
            real_fsync(descriptor)

        with (
            patch.object(os, "fsync", side_effect=fail_directory_fsync),
            patch.object(state_module, "_issue_locked_evaluation_permit") as issue,
            self.assertRaisesRegex(PublicationStateIOError, "states directory"),
        ):
            store.begin_evaluation()
        issue.assert_not_called()
        self.assertEqual(store.inspect().current.stage, PublicationStage.EVALUATING)
        store.close()

    def test_create_open_list_read_and_fstat_failures_are_normalized(self) -> None:
        store = self.initialize()
        real_open = os.open

        def fail_state_create(path: object, *args: object, **kwargs: object) -> int:
            if path == "001-evaluating.json":
                raise OSError("injected open failure")
            return real_open(path, *args, **kwargs)

        with (
            patch.object(os, "open", side_effect=fail_state_create),
            self.assertRaisesRegex(PublicationStateIOError, "create"),
        ):
            store.begin_evaluation()

        with (
            patch.object(os, "listdir", side_effect=OSError("injected list")),
            self.assertRaisesRegex(PublicationStateIOError, "list"),
        ):
            store.inspect()

        with (
            patch.object(
                state_module,
                "_fstat",
                side_effect=PublicationStateIOError("injected fstat"),
            ),
            self.assertRaisesRegex(PublicationStateIOError, "injected fstat"),
        ):
            store.inspect()
        store.close()

        with (
            patch.object(os, "read", side_effect=OSError("injected read")),
            self.assertRaisesRegex(PublicationStateIOError, "read"),
        ):
            PublicationStateStore.open(self.run_directory)

    def test_closed_store_rejects_operations(self) -> None:
        store = self.initialize()
        store.close()
        with self.assertRaisesRegex(PublicationStateConflict, "closed"):
            store.inspect()

    def test_non_locked_config_is_rejected_before_claim(self) -> None:
        config = replace(DEFAULT_EXPERIMENT_CONFIG, split="test")
        with self.assertRaisesRegex(
            PublicationStateConflict,
            "exact locked config",
        ):
            PublicationStateStore.initialize(
                self.run_directory,
                config=config,
            )
        self.assertFalse(self.run_directory.exists())


if __name__ == "__main__":
    unittest.main()
