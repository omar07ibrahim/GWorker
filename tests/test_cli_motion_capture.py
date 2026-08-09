from __future__ import annotations

import ast
import hashlib
import io
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts.visuals import capture_cli_motion as motion

TEST_ROOT = motion.ROOT / ".gworker" / "cli-motion-tests"
COMMITTED_SOURCE_COMMIT = "910361a6c7c60c1d2c0e7738ec4ac35a7557577b"
COMMITTED_BUNDLE_SHA256 = {
    motion.EVENTS_NAME: (
        "1a69fa39807214f879847deb05080dc685bb6bdfb5f69d0ad4ab424da540d037"
    ),
    motion.GIF_NAME: (
        "1b301f5adbc0df6c07f9ced259f83968a8eab9b3868cdb02791709b746e2b692"
    ),
    motion.POSTER_NAME: (
        "761b3b175120d61375684303be1a19f5b1ec6aa5bebb82ce0a735d7cab1d7a7c"
    ),
    motion.TRANSCRIPT_NAME: (
        "e02708286479a90f870957e2093d2debaf5d7b0980df306e60a010b090de1976"
    ),
    motion.MANIFEST_NAME: (
        "e68b2cdcbcacf7bb554f63c9cb9daccf333ea58f7135e10abf2ffff8738abd00"
    ),
}
COMMITTED_BUNDLE_BYTE_COUNTS = {
    motion.EVENTS_NAME: 7_285,
    motion.GIF_NAME: 268_393,
    motion.POSTER_NAME: 36_505,
    motion.TRANSCRIPT_NAME: 2_107,
    motion.MANIFEST_NAME: 3_849,
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _process_is_running(process_id: int) -> bool:
    try:
        status = Path(f"/proc/{process_id}/stat").read_text("ascii").split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return status not in {"X", "Z"}


def fixture_document() -> dict[str, object]:
    commands = [
        motion._command_record(spec, motion.EXPECTED_STDOUT[spec.capture_id])
        for spec in motion.COMMANDS
    ]
    return motion._events_document(
        source_commit="a" * 40,
        sources=motion._source_records(),
        commands=commands,
        workspace_mode="0700",
        journal_mode="0600",
        removed=True,
    )


def fixture_bundle() -> dict[str, bytes]:
    return motion._bundle_from_events(motion._canonical_json(fixture_document()))


class FixedWorkflowTests(unittest.TestCase):
    def test_allowlist_is_literal_complete_and_safe(self) -> None:
        self.assertEqual(
            tuple(spec.capture_id for spec in motion.COMMANDS),
            (
                "recommend-initial",
                "review-explicit",
                "recommend-reopened",
                "verify-reopened",
            ),
        )
        self.assertEqual(len({spec.display_argv for spec in motion.COMMANDS}), 4)
        for spec in motion.COMMANDS:
            self.assertEqual(
                spec.display_argv[:5],
                (
                    "python",
                    "-m",
                    "gworker.cli",
                    "--journal",
                    "<PRIVATE-JOURNAL>",
                ),
            )
            self.assertNotIn("run", spec.display_argv)
            self.assertNotIn("run_experiment", spec.display_argv)
            self.assertNotIn("publication_runner", spec.display_argv)
        self.assertEqual(
            motion.COMMANDS[-1].runtime_argv(Path("/private/journal"))[-1],
            "verify",
        )

    def test_process_source_uses_pty_without_shell_or_inherited_stdin(self) -> None:
        source = Path(motion.__file__).read_text("utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Popen"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertIsInstance(keywords["shell"], ast.Constant)
        self.assertIs(keywords["shell"].value, False)
        self.assertEqual(
            ast.unparse(keywords["stdin"]),
            "subprocess.DEVNULL",
        )
        self.assertEqual(ast.unparse(keywords["stdout"]), "slave_fd")
        self.assertEqual(
            ast.unparse(keywords["stderr"]),
            "subprocess.PIPE",
        )
        self.assertIsInstance(keywords["start_new_session"], ast.Constant)
        self.assertIs(keywords["start_new_session"].value, True)
        self.assertIn("pty.openpty()", source)
        self.assertIn("termios.TIOCSWINSZ", source)
        self.assertIn("os.killpg(process.pid, signal.SIGKILL)", source)

    def test_real_workflow_uses_private_storage_and_cleans_up(self) -> None:
        records, workspace_mode, journal_mode, removed = motion._capture_workflow()

        self.assertEqual(workspace_mode, "0700")
        self.assertEqual(journal_mode, "0600")
        self.assertIs(removed, True)
        self.assertEqual(len(records), 4)
        for spec, record in zip(motion.COMMANDS, records, strict=True):
            self.assertEqual(
                record,
                motion._command_record(
                    spec,
                    motion.EXPECTED_STDOUT[spec.capture_id],
                ),
            )
            self.assertNotIn(str(motion.ROOT), record["stdout"])
        self.assertEqual(tuple(motion.RUNTIME_ROOT.iterdir()), ())

    def test_cleanup_failure_does_not_hide_the_capture_failure(self) -> None:
        primary = motion.MotionCaptureError("primary capture failure")
        try:
            with (
                patch.object(motion, "_run_pty_process", side_effect=primary),
                patch.object(
                    motion.shutil,
                    "rmtree",
                    side_effect=OSError("simulated cleanup failure"),
                ),
                self.assertRaisesRegex(
                    motion.MotionCaptureError,
                    "primary capture failure",
                ) as raised,
            ):
                motion._capture_workflow()

            self.assertEqual(
                raised.exception.__notes__,
                ["cannot remove motion workspace"],
            )
        finally:
            for workspace in motion.RUNTIME_ROOT.glob("capture-*"):
                motion.shutil.rmtree(workspace)

    def test_timeout_terminates_a_child_in_the_created_process_group(self) -> None:
        TEST_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        child_pid: int | None = None
        with tempfile.TemporaryDirectory(prefix="process-", dir=TEST_ROOT) as temporary:
            pid_path = Path(temporary) / "child.pid"
            program = (
                "import os, pathlib, subprocess, sys, time;"
                "executable = os.readlink('/proc/self/exe');"
                "child = subprocess.Popen("
                "[executable, '-c', 'import time; time.sleep(60)']);"
                "pathlib.Path(sys.argv[1]).write_text("
                "str(child.pid), encoding='ascii');"
                "time.sleep(60)"
            )
            try:
                with (
                    patch.object(motion, "PROCESS_TIMEOUT_SECONDS", 0.5),
                    self.assertRaisesRegex(
                        motion.MotionCaptureError,
                        "timed out",
                    ),
                ):
                    motion._run_pty_process(
                        ("python", "-c", program, str(pid_path)),
                        environment={
                            "PATH": os.defpath,
                            "PYTHONIOENCODING": "utf-8",
                        },
                    )
                child_pid = int(pid_path.read_text("ascii"))
                deadline = time.monotonic() + 2
                while _process_is_running(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(_process_is_running(child_pid))
            finally:
                if child_pid is not None and _process_is_running(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

    def test_terminal_normalization_is_fail_closed(self) -> None:
        self.assertEqual(
            motion._normalize_terminal_output(b"one\r\ntwo\r\n", label="x"),
            "one\ntwo\n",
        )
        rejected = (
            b"bare\rreturn\n",
            b"\x1b[31mred\x1b[0m\n",
            b"nul\x00byte\n",
            b"/home/private/journal.sqlite3\n",
            b"github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456\n",
            b"missing-final-newline",
        )
        for content in rejected:
            with (
                self.subTest(content=content),
                self.assertRaises(motion.MotionCaptureError),
            ):
                motion._normalize_terminal_output(content, label="unsafe")

    def test_stream_limit_and_timeout_are_bounded_constants(self) -> None:
        self.assertEqual(motion.MAX_STREAM_BYTES, 65_536)
        self.assertEqual(motion.PROCESS_TIMEOUT_SECONDS, 30)
        self.assertEqual(
            motion.TERMINAL_CONTRACT,
            {
                "columns": 120,
                "rows": 40,
                "stdout_is_pty": True,
                "stderr_is_separate_pipe": True,
                "crlf_normalization": ("PTY CRLF converted to LF; bare CR rejected"),
            },
        )


class EventContractTests(unittest.TestCase):
    def test_fixture_is_closed_canonical_and_source_bound(self) -> None:
        document = fixture_document()
        encoded = motion._canonical_json(document)
        decoded = motion._load_json_bytes(encoded, label="fixture")

        self.assertEqual(decoded, document)
        motion._validate_events(decoded)
        self.assertEqual(
            decoded["presentation"],
            motion.PRESENTATION_CONTRACT,
        )
        provenance = decoded["provenance"]
        self.assertIsInstance(provenance, dict)
        assert isinstance(provenance, dict)
        self.assertEqual(provenance["sources"], motion._source_records())

    def test_duplicate_unknown_and_tampered_events_are_rejected(self) -> None:
        with self.assertRaises(motion.MotionCaptureError):
            motion._load_json_bytes(b'{"x":1,"x":2}\n', label="duplicate")

        document = fixture_document()
        document["unexpected"] = True
        with self.assertRaises(motion.MotionCaptureError):
            motion._validate_events(document)

        document = fixture_document()
        commands = document["commands"]
        assert isinstance(commands, list)
        command = commands[0]
        assert isinstance(command, dict)
        command["stdout"] = "forged\n"
        with self.assertRaises(motion.MotionCaptureError):
            motion._validate_events(document)

        document = fixture_document()
        provenance = document["provenance"]
        assert isinstance(provenance, dict)
        sources = provenance["sources"]
        assert isinstance(sources, list)
        source = sources[0]
        assert isinstance(source, dict)
        source["sha256"] = "0" * 64
        with self.assertRaises(motion.MotionCaptureError):
            motion._validate_events(document)

    def test_transcript_preserves_every_process_and_claim_boundary(self) -> None:
        transcript = motion._transcript(fixture_document()).decode("utf-8")

        self.assertEqual(transcript.count("\n$ python -m gworker.cli "), 4)
        self.assertEqual(transcript.count("[exit 0]"), 4)
        self.assertIn("fixed presentation timing", transcript.lower())
        self.assertIn("<PRIVATE-JOURNAL>", transcript)
        self.assertIn("workspace 0700; journal 0600", transcript)
        self.assertIn("no human outcome or locked evaluation", transcript)
        self.assertNotIn(str(motion.ROOT), transcript)
        self.assertNotRegex(transcript, r"/home/|/Users/|C:\\Users\\")


class RenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)

    def temporary_output(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(prefix="bundle-", dir=TEST_ROOT)

    def test_render_is_byte_deterministic_and_multiframe(self) -> None:
        self.assertEqual(
            motion.WORKFLOW_TITLE,
            "Recommend > review > reopen > verify",
        )
        first = fixture_bundle()
        second = fixture_bundle()
        self.assertEqual(first, second)

        gif = first[motion.GIF_NAME]
        self.assertTrue(gif.startswith(b"GIF89a"))
        durations: list[int] = []
        frame_hashes: list[str] = []
        with Image.open(io.BytesIO(gif)) as animation:
            self.assertEqual(animation.size, (960, 540))
            self.assertEqual(animation.n_frames, 11)
            self.assertEqual(animation.info["loop"], 0)
            self.assertEqual(
                animation.info["comment"],
                b"GWorker real CLI output; fixed presentation timing",
            )
            for index in range(animation.n_frames):
                animation.seek(index)
                durations.append(animation.info["duration"])
                frame_hashes.append(sha256(animation.convert("RGB").tobytes()))
        self.assertEqual(tuple(durations), motion.FRAME_DURATIONS_MS)
        self.assertEqual(len(set(frame_hashes)), 11)

        with Image.open(io.BytesIO(first[motion.POSTER_NAME])) as poster:
            self.assertEqual(poster.format, "PNG")
            self.assertEqual(poster.size, (960, 540))
            self.assertEqual(poster.mode, "RGB")
            self.assertEqual(poster.info, {})

    def test_manifest_binds_sources_events_and_every_output(self) -> None:
        bundle = fixture_bundle()
        manifest = motion._load_json_bytes(
            bundle[motion.MANIFEST_NAME],
            label="manifest",
        )

        self.assertEqual(manifest["schema_version"], motion.MANIFEST_SCHEMA)
        self.assertEqual(manifest["presentation"], motion.PRESENTATION_CONTRACT)
        tool = manifest["tool"]
        self.assertIsInstance(tool, dict)
        assert isinstance(tool, dict)
        self.assertEqual(tool["pillow_version"], "12.3.0")
        self.assertEqual(tool["name"], motion.TOOL_NAME)
        self.assertEqual(tool["version"], motion.TOOL_VERSION)
        outputs = manifest["outputs"]
        self.assertIsInstance(outputs, dict)
        assert isinstance(outputs, dict)
        self.assertEqual(
            set(outputs),
            {
                motion.EVENTS_NAME,
                motion.TRANSCRIPT_NAME,
                motion.GIF_NAME,
                motion.POSTER_NAME,
            },
        )
        for name, record in outputs.items():
            self.assertIsInstance(record, dict)
            assert isinstance(record, dict)
            self.assertEqual(record["byte_count"], len(bundle[name]))
            self.assertEqual(record["sha256"], sha256(bundle[name]))
        self.assertEqual(
            manifest["claim_boundary"],
            [
                "four fixed synthetic CLI processes against one disposable journal",
                "real PTY stdout with deterministic presentation timing",
                (
                    "not a timer UI, benchmark, latency measurement, human outcome, "
                    "or locked evaluation"
                ),
                (
                    "byte-identical rendering requires the recorded Pillow and "
                    "FreeType versions"
                ),
            ],
        )

    def test_write_check_and_render_are_process_free(self) -> None:
        bundle = fixture_bundle()
        with self.temporary_output() as temporary:
            root = Path(temporary) / "motion"
            motion._write_bundle(bundle, root)
            with patch.object(
                motion,
                "_run_pty_process",
                side_effect=AssertionError("process execution is forbidden"),
            ):
                self.assertEqual(motion.check_bundle(root), ())
                rendered = motion.render_bundle(root)
            self.assertEqual(rendered, bundle)
            self.assertEqual(
                {path.name for path in root.iterdir()},
                set(bundle),
            )

    def test_check_detects_tampering_and_unknown_files(self) -> None:
        bundle = fixture_bundle()
        with self.temporary_output() as temporary:
            root = Path(temporary) / "motion"
            motion._write_bundle(bundle, root)
            transcript = root / motion.TRANSCRIPT_NAME
            transcript.write_bytes(transcript.read_bytes() + b"tamper\n")
            self.assertIn(
                f"{motion.TRANSCRIPT_NAME}: bytes differ",
                motion.check_bundle(root),
            )

            transcript.write_bytes(bundle[motion.TRANSCRIPT_NAME])
            (root / "unexpected.bin").write_bytes(b"x")
            self.assertIn(
                "motion output inventory differs",
                motion.check_bundle(root),
            )

    def test_writer_rejects_nonregular_and_foreign_entries(self) -> None:
        bundle = fixture_bundle()
        with self.temporary_output() as temporary:
            root = Path(temporary) / "motion"
            root.mkdir()
            (root / "foreign.txt").write_text("foreign", encoding="utf-8")
            with self.assertRaises(motion.MotionCaptureError):
                motion._write_bundle(bundle, root)

        with self.temporary_output() as temporary:
            root = Path(temporary) / "motion"
            os.symlink(TEST_ROOT, root)
            with self.assertRaises(motion.MotionCaptureError):
                motion._write_bundle(bundle, root)


class CommittedMotionBundleTests(unittest.TestCase):
    def test_committed_bundle_matches_frozen_bytes_and_source_commit(self) -> None:
        self.assertEqual(motion.check_bundle(), ())
        observed = {
            path.name: path.read_bytes()
            for path in motion.MOTION_ROOT.iterdir()
            if path.is_file()
        }
        self.assertEqual(set(observed), set(COMMITTED_BUNDLE_SHA256))
        for name, content in observed.items():
            self.assertEqual(
                sha256(content),
                COMMITTED_BUNDLE_SHA256[name],
                name,
            )
            self.assertEqual(
                len(content),
                COMMITTED_BUNDLE_BYTE_COUNTS[name],
                name,
            )

        events = motion._load_json_bytes(
            observed[motion.EVENTS_NAME],
            label="committed motion events",
        )
        provenance = events["provenance"]
        self.assertIsInstance(provenance, dict)
        assert isinstance(provenance, dict)
        self.assertEqual(provenance["source_commit"], COMMITTED_SOURCE_COMMIT)
        self.assertEqual(provenance["sources"], motion._source_records())

    def test_committed_check_starts_no_process(self) -> None:
        with (
            patch.object(
                motion,
                "_run_pty_process",
                side_effect=AssertionError("application execution is forbidden"),
            ),
            patch.object(
                motion,
                "_run_git",
                side_effect=AssertionError("Git execution is forbidden"),
            ),
        ):
            self.assertEqual(motion.check_bundle(), ())

    def test_committed_text_has_no_host_path_or_secret_signature(self) -> None:
        text = b"\n".join(
            (motion.MOTION_ROOT / name).read_bytes()
            for name in (
                motion.EVENTS_NAME,
                motion.TRANSCRIPT_NAME,
                motion.MANIFEST_NAME,
            )
        ).decode("utf-8")
        self.assertNotIn(str(motion.ROOT), text)
        self.assertNotRegex(text, r"/home/|/Users/|C:\\Users\\")
        self.assertIsNone(motion.SECRET_PATTERN.search(text))


class RecordBoundaryTests(unittest.TestCase):
    def test_source_bytes_must_match_the_recorded_commit_blobs(self) -> None:
        records = motion._source_records()
        with (
            patch.object(motion, "_run_git", return_value=b"different blob"),
            self.assertRaisesRegex(
                motion.MotionCaptureError,
                "does not match commit blob",
            ),
        ):
            motion._assert_sources_match_commit(records, "a" * 40)

    def test_record_rejects_an_arbitrary_output_root(self) -> None:
        with self.assertRaises(motion.MotionCaptureError):
            motion.record_bundle(TEST_ROOT / "not-the-public-bundle")

    def test_dirty_source_fails_before_application_processes(self) -> None:
        with (
            patch.object(
                motion,
                "_clean_source_commit",
                side_effect=motion.MotionCaptureError("dirty"),
            ),
            patch.object(
                motion,
                "_capture_workflow",
                side_effect=AssertionError("workflow must not start"),
            ),
            self.assertRaisesRegex(motion.MotionCaptureError, "dirty"),
        ):
            motion.record_bundle()

    def test_pillow_version_is_an_exact_rendering_input(self) -> None:
        with (
            patch.object(motion, "PILLOW_VERSION", "0.0.0"),
            self.assertRaisesRegex(
                motion.MotionCaptureError,
                "requires Pillow 12.3.0",
            ),
        ):
            motion._require_pillow()


if __name__ == "__main__":
    unittest.main()
