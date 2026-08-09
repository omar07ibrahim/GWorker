from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import cast
from unittest.mock import call, patch

from scripts.visuals import capture_terminal

TEST_TEMP_ROOT = capture_terminal.ROOT / ".gworker" / "terminal-capture-tests"
FROZEN_BASE_INPUT_COMMIT = "910361a6c7c60c1d2c0e7738ec4ac35a7557577b"
FROZEN_TERMINAL_SHA256 = {
    "durable-policy-workflow.svg": (
        "ad0d49d259a7088fb45a8f12b3f9190a7a703668c30fb5f2ce8d89c367c80c0f"
    ),
    "durable-policy-workflow.txt": (
        "aadff023da66499811b6790764495a02227cc44d3385741279b7f5448ab1b420"
    ),
    "journal-recovery.svg": (
        "08cc240fd28121cd9f09a9029fcc16631232f4b568383cd340e6fe8759c582d7"
    ),
    "journal-recovery.txt": (
        "bc077acc3dba23616efbf2ea6a52c2c092ae00a64d582383c662b24125c8db91"
    ),
    "manifest.json": (
        "f3fcaea19403c28988adc4b75ba48a69ad0fbab28c04abb93714406457794db1"
    ),
    "offline-replay.svg": (
        "90b7a07bcafc74f2c7f33fcbd3beb14926360890a6f189ab45e42c925fd908ad"
    ),
    "offline-replay.txt": (
        "9e90e83110163251da9bd1e2846475d4881a45dcc097a783b6b1b139c1065024"
    ),
    "policy-demo.svg": (
        "9c263b3ce0045268b9eee4d0d8e2683aa933595241e9bf43d0cd7dcfd3cebf06"
    ),
    "policy-demo.txt": (
        "406f72a524554a7d71ed1c5deda699ce2d049fa7862b45deae28614fbeb03f3e"
    ),
    "protocol-inventory.svg": (
        "f207b34179dd8b49d32bb2edc496dadd7b3150719f7ebb5921cbfc3f43c8b510"
    ),
    "protocol-inventory.txt": (
        "306534aa044285aa55e06c86d275a00ea752a660f89caaf4d6f2e231d36ee31a"
    ),
    "publication-preflight.svg": (
        "555a259640293f30c8458041476d088ce15eaef234fa71a7a43ed2e36c24bb1d"
    ),
    "publication-preflight.txt": (
        "00fef974627c434404d5e64163132e03a4a9aeb3150df1a29f11192b29805b9b"
    ),
    "publication-status.svg": (
        "8ca14d43b5b66a17da61d1f8045ba26e09d84f25f55d80133975252e267c8170"
    ),
    "publication-status.txt": (
        "2834937ead3d73afdabee5aa5f3e96ad4bb4d3a0f60a63cf23d145ccce42abd4"
    ),
}
FROZEN_TERMINAL_BYTE_COUNTS = {
    "durable-policy-workflow.svg": 4_402,
    "durable-policy-workflow.txt": 685,
    "journal-recovery.svg": 6_454,
    "journal-recovery.txt": 1_069,
    "manifest.json": 21_178,
    "offline-replay.svg": 5_878,
    "offline-replay.txt": 1_037,
    "policy-demo.svg": 7_606,
    "policy-demo.txt": 1_484,
    "protocol-inventory.svg": 8_096,
    "protocol-inventory.txt": 1_547,
    "publication-preflight.svg": 3_369,
    "publication-preflight.txt": 221,
    "publication-status.svg": 2_603,
    "publication-status.txt": 75,
}
FROZEN_CAPTURE_SOURCE_SHA256 = {
    "scripts/demo_offline_replay.py": (
        "ef6120c5cba53111e5a0eb092437f58f055faba1a1b0c46c4421516119e2fa6d"
    ),
    "scripts/demo_journal.py": (
        "662a875d6b205144228339c7cc6a1eaf903dc66a2373451399bd28fe6c59b3f7"
    ),
    "scripts/demo_policy.py": (
        "a9e4d6a459af30a5d920cead9b20d3a80b67c9a86b7b17ad05a34b419e8faa83"
    ),
    "scripts/demo_policy_journal.py": (
        "7e0c365f1ce2d6ae9536702375efcde23d8a8dfaa2d40145e036eef45ddcdbd7"
    ),
    "scripts/protocol_inventory.py": (
        "7b81d54c902dd837a9d3d7fefa2465dcbd0d778830e7879b1ee156830d1159d4"
    ),
    "scripts/visuals/capture_terminal.py": (
        "c47849ee40a25e96d97c6972b37b972d1fa5378c55ed12203ced73fa1bb4a361"
    ),
    "src/gworker/__init__.py": (
        "9da9d4eaa9708bfd1da9d30879dca161d8ce8903ebb7d0fd176e14e8f7b60ade"
    ),
    "src/gworker/cli.py": (
        "3b3baffd2e7c247929fef414a724d3af7f945aff76f4d9d1d6b637f23c190a31"
    ),
    "src/gworker/codec.py": (
        "0d74b7542d0abca553f945adb150040115698c402055ae9f5e115e00c3cd665d"
    ),
    "src/gworker/domain.py": (
        "847af410217e93820e25e67bbf6c6a85747c4426bc9cd0546a02e43e718195a0"
    ),
    "src/gworker/evaluation.py": (
        "0440be639688c4755929ed3ea3c975388986fcfe3412194c82eaadc7b183d9e5"
    ),
    "src/gworker/evidence.py": (
        "a7e27eec70f12351bb8bb8c95eff414b176a22b0868fe753cb08c0ef6e4863b8"
    ),
    "src/gworker/offline.py": (
        "8f099a90e4d3c8a22788128ec526776a0995161d4f6780584f6720d1d0c83618"
    ),
    "src/gworker/policy.py": (
        "c39b2fb37c0a3db2f64812d0aba2c04c06c9e063bcafee7e71f6ec4b1bb10b13"
    ),
    "src/gworker/publication_codec.py": (
        "3514265191e225ab9a7b0588f254e9587940b1c64b4eda607989834748dad20a"
    ),
    "src/gworker/publication_runner.py": (
        "b12402453cf8ee5b98556e1cad8f572802d2163c6dd803d4994459fd61334ab0"
    ),
    "src/gworker/publication_state.py": (
        "d2de30d2af8e5a172d5980f9b67274ac760f66b00621779ea29496a8a6e1df8d"
    ),
    "src/gworker/reporting.py": (
        "710bbe03becb4321323fdda6b2964431a8eecf25d54a6d1d3d6366ac74295c29"
    ),
    "src/gworker/resource_preflight.py": (
        "75599a900af5691e02cba41a974bb9d4627a2110978f39f3a724d4334f5012c7"
    ),
    "src/gworker/result_codec.py": (
        "1b068c6ed37a0924496df0111a259c8f30106aaa043ac7a1e314e39b6ca0a64d"
    ),
    "src/gworker/storage.py": (
        "748033ee8d23abf2fce3d187ad3376b2f1ea3cac85aa2e8f2c1e3ba9bdb7988f"
    ),
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def private_temporary_directory() -> tempfile.TemporaryDirectory[str]:
    TEST_TEMP_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(TEST_TEMP_ROOT, 0o700)
    return tempfile.TemporaryDirectory(
        prefix="case-",
        dir=TEST_TEMP_ROOT,
    )


def fake_records() -> dict[str, capture_terminal.CaptureRecord]:
    records = {}
    for spec in capture_terminal.COMMANDS:
        exit_code = 2 if spec.capture_id == "publication-preflight" else 0
        if spec.capture_id == "publication-preflight":
            stdout = b'{"failure_codes":["memory-headroom"],"ok":false,"ready":false}\n'
        else:
            stdout = f"{spec.capture_id} genuine fixture output\n".encode()
        records[spec.capture_id] = capture_terminal.CaptureRecord(
            stdout=stdout,
            stderr=b"",
            exit_code=exit_code,
        )
    return records


def fake_source_state(
    commit: str = "a" * 40,
) -> capture_terminal.CaptureSourceState:
    return capture_terminal._snapshot_sources(commit)


def write_test_bundle(root: Path) -> dict[str, object]:
    records = fake_records()
    visuals = {
        spec.capture_id: capture_terminal.render_svg(
            spec,
            records[spec.capture_id],
        )
        for spec in capture_terminal.COMMANDS
    }
    manifest = capture_terminal._manifest(
        records,
        visuals,
        input_commit="a" * 40,
    )
    root.mkdir(parents=True)
    for spec in capture_terminal.COMMANDS:
        (root / spec.transcript_name).write_bytes(records[spec.capture_id].stdout)
        (root / spec.visual_name).write_bytes(visuals[spec.capture_id])
    (root / capture_terminal.MANIFEST_NAME).write_bytes(
        capture_terminal._canonical_json(manifest)
    )
    return manifest


class CommandAllowlistTests(unittest.TestCase):
    def test_allowlist_is_literal_complete_and_read_only_where_required(
        self,
    ) -> None:
        self.assertEqual(
            {spec.capture_id: spec.argv for spec in capture_terminal.COMMANDS},
            {
                "durable-policy-workflow": (
                    "python",
                    "scripts/demo_policy_journal.py",
                ),
                "policy-demo": ("python", "scripts/demo_policy.py"),
                "offline-replay": (
                    "python",
                    "scripts/demo_offline_replay.py",
                ),
                "journal-recovery": (
                    "python",
                    "scripts/demo_journal.py",
                    "--repo-root",
                    ".",
                    "--workspace",
                    ".gworker/visual-demo/terminal-capture",
                    "--reset",
                ),
                "protocol-inventory": (
                    "python",
                    "scripts/protocol_inventory.py",
                ),
                "publication-status": (
                    "python",
                    "-m",
                    "gworker.publication_runner",
                    "status",
                ),
                "publication-preflight": (
                    "python",
                    "-m",
                    "gworker.publication_runner",
                    "preflight",
                ),
            },
        )
        self.assertEqual(
            capture_terminal.COMMAND_BY_ID,
            {spec.capture_id: spec for spec in capture_terminal.COMMANDS},
        )
        self.assertEqual(
            set(capture_terminal.COMMAND_BY_ID["durable-policy-workflow"].source_paths),
            {
                "scripts/demo_policy_journal.py",
                "src/gworker/__init__.py",
                "src/gworker/cli.py",
                "src/gworker/codec.py",
                "src/gworker/domain.py",
                "src/gworker/policy.py",
                "src/gworker/storage.py",
            },
        )
        self.assertEqual(
            set(capture_terminal.COMMAND_BY_ID["offline-replay"].source_paths),
            {
                "scripts/demo_offline_replay.py",
                "src/gworker/__init__.py",
                "src/gworker/codec.py",
                "src/gworker/domain.py",
                "src/gworker/offline.py",
                "src/gworker/policy.py",
                "src/gworker/storage.py",
            },
        )
        self.assertEqual(
            set(capture_terminal.OFFLINE_REPLAY_IMPORT_SOURCES),
            {
                "src/gworker/__init__.py",
                "src/gworker/codec.py",
                "src/gworker/domain.py",
                "src/gworker/offline.py",
                "src/gworker/policy.py",
                "src/gworker/storage.py",
            },
        )
        for capture_id in capture_terminal.COMMAND_BY_ID:
            self.assertTrue(
                set(capture_terminal.CORE_IMPORT_SOURCES).issubset(
                    capture_terminal.COMMAND_BY_ID[capture_id].source_paths
                )
            )
        for capture_id in ("publication-status", "publication-preflight"):
            self.assertEqual(
                set(capture_terminal.COMMAND_BY_ID[capture_id].source_paths),
                set(capture_terminal.PUBLICATION_IMPORT_SOURCES),
            )
        for spec in capture_terminal.COMMANDS:
            self.assertNotIn("run", spec.argv)
            self.assertNotIn("run_experiment", spec.argv)
        preflight = capture_terminal.COMMAND_BY_ID["publication-preflight"]
        self.assertEqual(preflight.expected_exit_codes, (0, 2))
        self.assertTrue(preflight.host_dependent)
        self.assertEqual(
            preflight.host_note,
            "CAPTURED ON THIS HOST · readiness may vary elsewhere",
        )

    def test_journal_is_reset_only_in_the_fixed_private_workspace(self) -> None:
        journal = capture_terminal.COMMAND_BY_ID["journal-recovery"]
        workspace_index = journal.argv.index("--workspace")

        self.assertEqual(
            journal.argv[workspace_index + 1],
            capture_terminal.JOURNAL_WORKSPACE,
        )
        self.assertEqual(journal.argv.count("--reset"), 1)
        self.assertEqual(
            capture_terminal.JOURNAL_WORKSPACE,
            ".gworker/visual-demo/terminal-capture",
        )

    def test_runner_uses_no_shell_and_exact_subprocess_controls(self) -> None:
        spec = capture_terminal.COMMANDS[0]
        completed = subprocess.CompletedProcess(
            args=spec.argv,
            returncode=0,
            stdout=b"real output\n",
            stderr=b"",
        )
        environment = {
            "HOME": "/private/home",
            "TMPDIR": "/private/tmp",
        }
        with (
            patch.object(
                capture_terminal,
                "_minimal_environment",
                return_value=environment,
            ),
            patch.object(
                subprocess,
                "run",
                return_value=completed,
            ) as run,
        ):
            record = capture_terminal._run_allowlisted(spec)

        self.assertEqual(record.stdout, b"real output\n")
        self.assertEqual(record.stderr, b"")
        self.assertEqual(record.exit_code, 0)
        run.assert_called_once_with(
            spec.argv,
            cwd=capture_terminal.ROOT,
            env=environment,
            executable=sys.executable,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=120,
        )
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_imposter_command_is_rejected_before_subprocess(self) -> None:
        original = capture_terminal.COMMANDS[0]
        imposter = capture_terminal.CommandSpec(
            capture_id=original.capture_id,
            title=original.title,
            description=original.description,
            argv=("python", "-c", "print('not allowlisted')"),
            expected_exit_codes=(0,),
            transcript_name=original.transcript_name,
            visual_name=original.visual_name,
            source_paths=original.source_paths,
        )
        with (
            patch.object(subprocess, "run") as run,
            self.assertRaisesRegex(
                capture_terminal.CaptureError,
                "immutable allowlist",
            ),
        ):
            capture_terminal._run_allowlisted(imposter)
        run.assert_not_called()

    def test_unexpected_exit_stderr_and_noncanonical_output_fail_closed(
        self,
    ) -> None:
        spec = capture_terminal.COMMANDS[0]
        cases = (
            (
                subprocess.CompletedProcess(spec.argv, 9, b"failed\n", b""),
                "unexpected exit code",
            ),
            (
                subprocess.CompletedProcess(spec.argv, 0, b"ok\n", b"warning\n"),
                "unexpectedly wrote to stderr",
            ),
            (
                subprocess.CompletedProcess(spec.argv, 0, b"no newline", b""),
                "lacks a final newline",
            ),
            (
                subprocess.CompletedProcess(spec.argv, 0, b"\xff\n", b""),
                "valid UTF-8",
            ),
        )
        for completed, message in cases:
            with (
                self.subTest(message=message),
                patch.object(
                    subprocess,
                    "run",
                    return_value=completed,
                ),
                self.assertRaisesRegex(
                    capture_terminal.CaptureError,
                    message,
                ),
            ):
                capture_terminal._run_allowlisted(spec)

    def test_record_writes_nothing_when_any_command_fails(self) -> None:
        records = [
            capture_terminal.CaptureRecord(
                stdout=f"{index}\n".encode(),
                stderr=b"",
                exit_code=0,
            )
            for index in range(len(capture_terminal.COMMANDS) - 1)
        ]
        side_effects: list[object] = [
            *records,
            capture_terminal.CaptureError("preflight capture failed"),
        ]
        with (
            patch.object(
                capture_terminal,
                "_capture_clean_source_state",
                return_value=fake_source_state(),
            ),
            patch.object(
                capture_terminal,
                "_run_allowlisted",
                side_effect=side_effects,
            ) as run,
            patch.object(capture_terminal, "_atomic_write") as write,
            self.assertRaisesRegex(
                capture_terminal.CaptureError,
                "preflight capture failed",
            ),
        ):
            capture_terminal.record_bundle()

        self.assertEqual(
            run.call_args_list,
            [call(spec) for spec in capture_terminal.COMMANDS],
        )
        write.assert_not_called()

    def test_record_rejects_a_source_change_before_any_write(self) -> None:
        initial = fake_source_state()
        original = initial.files[0]
        changed = capture_terminal.CaptureSourceState(
            commit=initial.commit,
            files=(
                capture_terminal.SourceFileSnapshot(
                    path=original.path,
                    byte_count=original.byte_count,
                    sha256="f" * 64,
                    mode=original.mode,
                ),
                *initial.files[1:],
            ),
        )
        with (
            patch.object(
                capture_terminal,
                "_capture_clean_source_state",
                side_effect=(initial, changed),
            ),
            patch.object(
                capture_terminal,
                "_run_allowlisted",
                side_effect=tuple(fake_records().values()),
            ),
            patch.object(capture_terminal, "_atomic_write") as write,
            self.assertRaisesRegex(
                capture_terminal.CaptureError,
                "source state changed",
            ),
        ):
            capture_terminal.record_bundle()

        write.assert_not_called()

    def test_record_manifest_uses_the_bracketed_source_snapshot(self) -> None:
        source_state = fake_source_state()
        with (
            patch.object(
                capture_terminal,
                "_capture_clean_source_state",
                side_effect=(source_state, source_state),
            ) as capture_state,
            patch.object(
                capture_terminal,
                "_run_allowlisted",
                side_effect=tuple(fake_records().values()),
            ),
            patch.object(capture_terminal, "_atomic_write"),
        ):
            manifest = capture_terminal.record_bundle()

        self.assertEqual(capture_state.call_count, 2)
        capture = cast(dict[str, object], manifest["capture"])
        self.assertEqual(capture["base_input_commit"], source_state.commit)
        command_records = cast(list[dict[str, object]], manifest["commands"])
        observed = {
            cast(str, source["path"]): cast(str, source["sha256"])
            for command in command_records
            for source in cast(list[dict[str, object]], command["sources"])
        }
        expected = {
            snapshot.path: snapshot.sha256
            for snapshot in source_state.files
            if snapshot.path != "scripts/visuals/capture_terminal.py"
        }
        self.assertEqual(observed, expected)

    def test_cli_has_no_arbitrary_command_or_output_path_surface(self) -> None:
        parser = capture_terminal._parser()
        self.assertEqual(parser.parse_args(["record"]).action, "record")
        self.assertEqual(parser.parse_args(["render"]).action, "render")
        self.assertEqual(parser.parse_args(["check"]).action, "check")
        command_error = io.StringIO()
        with redirect_stderr(command_error), self.assertRaises(SystemExit):
            parser.parse_args(["record", "--command", "whoami"])
        output_error = io.StringIO()
        with redirect_stderr(output_error), self.assertRaises(SystemExit):
            parser.parse_args(["record", "--output", "outside"])
        self.assertIn(
            "unrecognized arguments: --command whoami", command_error.getvalue()
        )
        self.assertIn(
            "unrecognized arguments: --output outside", output_error.getvalue()
        )

    def test_source_ast_has_one_safe_process_primitive_and_no_evaluator_call(
        self,
    ) -> None:
        tree = ast.parse(Path(capture_terminal.__file__).read_text("utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        subprocess_calls = [
            (node, node.func)
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ]
        self.assertEqual(len(subprocess_calls), 1)
        subprocess_call, subprocess_function = subprocess_calls[0]
        self.assertEqual(subprocess_function.attr, "run")
        keywords = {
            keyword.arg: keyword.value
            for keyword in subprocess_call.keywords
            if keyword.arg is not None
        }
        self.assertNotIn("shell", keywords)
        self.assertIn("executable", keywords)
        self.assertIn("timeout", keywords)
        called_names = {
            node.func.id for node in calls if isinstance(node.func, ast.Name)
        } | {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
        self.assertTrue(
            {
                "eval",
                "exec",
                "popen",
                "run_experiment",
                "run_publication",
                "system",
            }.isdisjoint(called_names)
        )


class EnvironmentAndSanitizationTests(unittest.TestCase):
    def test_environment_is_minimal_reproducible_and_secret_free(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AWS_SECRET_ACCESS_KEY": "do-not-inherit",
                "GH_TOKEN": "do-not-inherit",
            },
            clear=False,
        ):
            environment = capture_terminal._minimal_environment()

        self.assertEqual(
            set(environment),
            {
                "HOME",
                "LANG",
                "LC_ALL",
                "PATH",
                "PYTHONHASHSEED",
                "PYTHONIOENCODING",
                "PYTHONPATH",
                "TMPDIR",
                "TZ",
            },
        )
        self.assertEqual(environment["LC_ALL"], "C")
        self.assertEqual(environment["LANG"], "C")
        self.assertEqual(environment["TZ"], "UTC")
        self.assertEqual(environment["PYTHONHASHSEED"], "0")
        self.assertEqual(environment["PYTHONPATH"], "src")
        for name in ("HOME", "TMPDIR"):
            location = Path(environment[name])
            self.assertTrue(location.is_relative_to(capture_terminal.ROOT / ".gworker"))
            self.assertEqual(location.stat().st_mode & 0o777, 0o700)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
        self.assertNotIn("GH_TOKEN", environment)

    def test_clean_worktree_check_uses_fixed_git_controls(self) -> None:
        completed = subprocess.CompletedProcess(
            args=capture_terminal.GIT_STATUS_ARGV,
            returncode=0,
            stdout=b"",
            stderr=b"",
        )
        with patch.object(subprocess, "run", return_value=completed) as run:
            capture_terminal._require_clean_worktree()

        arguments = run.call_args
        self.assertEqual(arguments.args, (capture_terminal.GIT_STATUS_ARGV,))
        self.assertEqual(arguments.kwargs["cwd"], capture_terminal.ROOT)
        self.assertEqual(
            arguments.kwargs["executable"],
            capture_terminal.GIT_EXECUTABLE,
        )
        self.assertEqual(arguments.kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(arguments.kwargs["capture_output"])
        self.assertFalse(arguments.kwargs["check"])
        self.assertEqual(arguments.kwargs["timeout"], 30)
        environment = cast(dict[str, str], arguments.kwargs["env"])
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(environment["GIT_OPTIONAL_LOCKS"], "0")
        self.assertNotIn("GH_TOKEN", environment)

    def test_dirty_or_unverifiable_worktree_fails_without_path_leakage(self) -> None:
        cases = (
            subprocess.CompletedProcess(
                capture_terminal.GIT_STATUS_ARGV,
                0,
                b"?? private-name\n",
                b"",
            ),
            subprocess.CompletedProcess(
                capture_terminal.GIT_STATUS_ARGV,
                1,
                b"",
                b"/home/private/repository\n",
            ),
        )
        for completed in cases:
            with (
                self.subTest(returncode=completed.returncode),
                patch.object(
                    capture_terminal,
                    "_run_process",
                    return_value=completed,
                ),
                self.assertRaises(capture_terminal.CaptureError) as raised,
            ):
                capture_terminal._require_clean_worktree()
            self.assertNotIn("private-name", str(raised.exception))
            self.assertNotIn("/home/private", str(raised.exception))

    def test_source_snapshot_is_bracketed_by_clean_checks_and_stable_head(
        self,
    ) -> None:
        expected = fake_source_state()
        with (
            patch.object(
                capture_terminal,
                "_head_commit",
                side_effect=(expected.commit, expected.commit),
            ) as head,
            patch.object(capture_terminal, "_require_clean_worktree") as clean,
            patch.object(
                capture_terminal,
                "_snapshot_sources",
                return_value=expected,
            ) as snapshot,
        ):
            observed = capture_terminal._capture_clean_source_state()

        self.assertEqual(observed, expected)
        self.assertEqual(head.call_count, 2)
        self.assertEqual(clean.call_count, 2)
        snapshot.assert_called_once_with(expected.commit)

        with (
            patch.object(
                capture_terminal,
                "_head_commit",
                side_effect=(expected.commit, "b" * 40),
            ),
            patch.object(capture_terminal, "_require_clean_worktree"),
            patch.object(
                capture_terminal,
                "_snapshot_sources",
                return_value=expected,
            ),
            self.assertRaisesRegex(
                capture_terminal.CaptureError,
                "HEAD changed",
            ),
        ):
            capture_terminal._capture_clean_source_state()

    def test_scrubber_removes_host_paths_hostname_and_secret_signatures(
        self,
    ) -> None:
        hostname = socket.gethostname()
        source = (
            f"root={capture_terminal.ROOT}\n"
            f"runtime={capture_terminal.RUNTIME_ROOT}\n"
            f"host={hostname}\n"
            "token=github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456\n"
            "focus=1470s break=280s propensity=0.732264546148\n"
        ).encode()

        scrubbed = capture_terminal._scrub_output(source).decode()

        self.assertNotIn(str(capture_terminal.ROOT), scrubbed)
        self.assertNotIn(hostname, scrubbed)
        self.assertNotIn("github_pat_", scrubbed)
        self.assertIn("<REPO>", scrubbed)
        self.assertIn("<RUNTIME>", scrubbed)
        self.assertIn("<HOST>", scrubbed)
        self.assertIn("<REDACTED>", scrubbed)
        self.assertIn(
            "focus=1470s break=280s propensity=0.732264546148",
            scrubbed,
        )

    def test_scrubber_rejects_unhandled_paths_and_terminal_controls(self) -> None:
        for content, message in (
            (b"outside=/home/alice/private/data.json\n", "absolute path"),
            (b"\x1b[32mgreen\x1b[0m\n", "ANSI controls"),
            (b"bad\x00value\n", "NUL"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    capture_terminal.CaptureError,
                    message,
                ),
            ):
                capture_terminal._scrub_output(content)


class RenderingTests(unittest.TestCase):
    def test_render_is_byte_deterministic_accessible_and_self_contained(
        self,
    ) -> None:
        spec = capture_terminal.COMMAND_BY_ID["policy-demo"]
        record = capture_terminal.CaptureRecord(
            stdout=(
                b"GWorker policy demo | deterministic synthetic scenario\n"
                b"Decision 13: focus-40 at p=0.732265\n"
            ),
            stderr=b"",
            exit_code=0,
        )

        first = capture_terminal.render_svg(spec, record)
        second = capture_terminal.render_svg(spec, record)

        self.assertEqual(first, second)
        root = ElementTree.fromstring(first)
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        title = root.find("svg:title", namespace)
        description = root.find("svg:desc", namespace)
        labelled_by = root.attrib["aria-labelledby"].split()
        self.assertEqual(root.attrib["role"], "img")
        self.assertIsNotNone(title)
        self.assertIsNotNone(description)
        self.assertIn(title.attrib["id"], labelled_by)  # type: ignore[union-attr]
        self.assertIn(
            description.attrib["id"],  # type: ignore[union-attr]
            labelled_by,
        )
        content = first.decode()
        self.assertIn(spec.title, content)
        self.assertIn("REAL OUTPUT · EXIT 0", content)
        self.assertIn(sha256(record.stdout), content)
        self.assertIn('font-family="monospace"', content)
        for forbidden in ("<script", "<image", " href=", "@import", "url("):
            self.assertNotIn(forbidden, content)
        self.assertEqual(capture_terminal._validate_svg(first, spec), ())

    def test_xml_namespace_is_not_misclassified_as_an_absolute_host_path(
        self,
    ) -> None:
        spec = capture_terminal.COMMAND_BY_ID["publication-status"]
        content = capture_terminal.render_svg(
            spec,
            capture_terminal.CaptureRecord(
                stdout=b'{"ok":true,"stage":"unclaimed"}\n',
                stderr=b"",
                exit_code=0,
            ),
        )

        self.assertIn(b'xmlns="http://www.w3.org/2000/svg"', content)
        self.assertEqual(capture_terminal._validate_svg(content, spec), ())

    def test_absolute_path_in_visible_svg_content_is_rejected(self) -> None:
        spec = capture_terminal.COMMAND_BY_ID["publication-status"]
        content = capture_terminal.render_svg(
            spec,
            capture_terminal.CaptureRecord(
                stdout=b"safe output\n",
                stderr=b"",
                exit_code=0,
            ),
        )
        unsafe = content.replace(b"safe output", b"/home/alice/private/output")

        self.assertIn(
            "SVG contains an unsanitized path or secret",
            capture_terminal._validate_svg(unsafe, spec),
        )

    def test_nonzero_preflight_badge_is_orange_not_success_green(self) -> None:
        spec = capture_terminal.COMMAND_BY_ID["publication-preflight"]
        content = capture_terminal.render_svg(
            spec,
            capture_terminal.CaptureRecord(
                stdout=b'{"ok":false,"ready":false}\n',
                stderr=b"",
                exit_code=2,
            ),
        ).decode()

        self.assertIn("REAL OUTPUT · EXIT 2", content)
        self.assertIn(
            f'fill="{capture_terminal.ORANGE}" font-size="14"',
            content,
        )
        self.assertIn(
            f'stroke="{capture_terminal.ORANGE}" stroke-width="1"',
            content,
        )
        self.assertIn(spec.host_note, content)

    def test_wrapped_display_retains_every_transcript_character(self) -> None:
        line = "key=" + "0123456789" * 24 + " trailing words"
        rendered = capture_terminal._display_lines(
            (line + "\n").encode(),
            width=40,
        )

        self.assertGreater(len(rendered), 1)
        self.assertEqual("".join(rendered), line)


class ManifestAndCheckTests(unittest.TestCase):
    def test_manifest_binds_commands_sources_outputs_and_base_commit(self) -> None:
        records = fake_records()
        visuals = {
            spec.capture_id: capture_terminal.render_svg(
                spec,
                records[spec.capture_id],
            )
            for spec in capture_terminal.COMMANDS
        }

        manifest = capture_terminal._manifest(
            records,
            visuals,
            input_commit="b" * 40,
        )

        self.assertEqual(manifest["schema_version"], capture_terminal.SCHEMA_VERSION)
        capture_record = cast(dict[str, object], manifest["capture"])
        self.assertEqual(capture_record["base_input_commit"], "b" * 40)
        self.assertEqual(
            manifest["safety"],
            {
                "arbitrary_commands_accepted": False,
                "evaluator_executed": False,
                "journal_workspace": capture_terminal.JOURNAL_WORKSPACE,
                "policy_journal_workspace": (
                    "private temporary directory; removed by the fixed harness"
                ),
                "publication_run_executed": False,
                "shell_used": False,
                "synthetic_demo_data_only": True,
            },
        )
        raw_commands = cast(list[dict[str, object]], manifest["commands"])
        command_records = {record["capture_id"]: record for record in raw_commands}
        for spec in capture_terminal.COMMANDS:
            record = command_records[spec.capture_id]
            self.assertEqual(record["argv"], list(spec.argv))
            self.assertEqual(record["exit_code"], records[spec.capture_id].exit_code)
            self.assertEqual(
                record["stderr"],
                {
                    "byte_count": 0,
                    "sha256": capture_terminal.EMPTY_SHA256,
                },
            )
            stdout_record = cast(dict[str, object], record["stdout"])
            visual_record = cast(dict[str, object], record["visual"])
            self.assertEqual(
                stdout_record["sha256"],
                sha256(records[spec.capture_id].stdout),
            )
            self.assertEqual(visual_record["sha256"], sha256(visuals[spec.capture_id]))
            sources = cast(list[dict[str, object]], record["sources"])
            self.assertEqual(
                [source["path"] for source in sources],
                sorted(set(spec.source_paths)),
            )
            for source in sources:
                relative = cast(str, source["path"])
                path = capture_terminal.ROOT / relative
                self.assertEqual(source["sha256"], sha256(path.read_bytes()))
                self.assertEqual(source["byte_count"], path.stat().st_size)

    def test_check_and_render_never_execute_a_command(self) -> None:
        with private_temporary_directory() as temporary:
            root = Path(temporary) / "terminal"
            write_test_bundle(root)
            with patch.object(subprocess, "run") as run:
                self.assertEqual(capture_terminal.check_bundle(root), ())
                capture_terminal.render_bundle(root)
                self.assertEqual(capture_terminal.check_bundle(root), ())
            run.assert_not_called()

    def test_check_detects_exact_byte_hash_and_provenance_changes(self) -> None:
        with private_temporary_directory() as temporary:
            root = Path(temporary) / "terminal"
            write_test_bundle(root)
            spec = capture_terminal.COMMANDS[0]

            (root / spec.transcript_name).write_bytes(b"changed\n")
            differences = capture_terminal.check_bundle(root)

            self.assertTrue(
                any("transcript hash or size differs" in item for item in differences)
            )
            self.assertTrue(
                any(
                    "SVG differs from deterministic render" in item
                    for item in differences
                )
            )

    def test_check_rejects_stale_unmanifested_files(self) -> None:
        with private_temporary_directory() as temporary:
            root = Path(temporary) / "terminal"
            write_test_bundle(root)
            (root / "stale-capture.svg").write_text("<svg/>", "utf-8")

            differences = capture_terminal.check_bundle(root)

            self.assertIn(
                "unexpected terminal bundle entry: stale-capture.svg",
                differences,
            )

    def test_check_rejects_noncanonical_manifest_bytes(self) -> None:
        with private_temporary_directory() as temporary:
            root = Path(temporary) / "terminal"
            manifest = write_test_bundle(root)
            (root / capture_terminal.MANIFEST_NAME).write_text(
                json.dumps(manifest),
                "utf-8",
            )

            self.assertIn(
                "manifest JSON is not canonical",
                capture_terminal.check_bundle(root),
            )

    def test_capture_interpreter_contract_is_bounded_to_supported_minors(
        self,
    ) -> None:
        for version in ("3.11.0", "3.12.3", "3.13.14"):
            with self.subTest(version=version), private_temporary_directory() as temp:
                root = Path(temp) / "terminal"
                manifest = write_test_bundle(root)
                capture = cast(dict[str, object], manifest["capture"])
                interpreter = cast(dict[str, object], capture["interpreter"])
                interpreter["version"] = version
                (root / capture_terminal.MANIFEST_NAME).write_bytes(
                    capture_terminal._canonical_json(manifest)
                )
                self.assertEqual(capture_terminal.check_bundle(root), ())
        for version in ("3.10.14", "3.14.0", "3.13", "3.13.1rc1"):
            with self.subTest(version=version):
                self.assertIsNone(
                    capture_terminal.CAPTURE_PYTHON_VERSION.fullmatch(version)
                )

    def test_check_rejects_tampered_manifest_claims_and_fields(self) -> None:
        cases = (
            ("top-level", "manifest top-level fields differ"),
            ("commit-role", "manifest base input commit role differs"),
            ("environment", "manifest capture environment differs"),
            ("normalization", "manifest capture normalization differs"),
            ("working-directory", "manifest capture working directory differs"),
            ("interpreter", "manifest capture interpreter is unsupported"),
            ("reproduction", "manifest reproduction contract differs"),
            ("safety", "manifest safety declaration differs"),
            ("safety-bool-alias", "manifest safety declaration differs"),
            ("tool", "manifest tool identity differs"),
            ("tool-source", "manifest tool source path differs"),
            (
                "command-fields",
                "durable-policy-workflow: manifest record fields differ",
            ),
            ("command-title", "durable-policy-workflow: title differs from allowlist"),
            (
                "command-description",
                "durable-policy-workflow: description differs from allowlist",
            ),
            (
                "exit-code-bool-alias",
                "durable-policy-workflow: expected exit codes differ",
            ),
            (
                "stderr-count-bool-alias",
                "durable-policy-workflow recorded stderr must be empty",
            ),
            (
                "stdout-count-bool-alias",
                "durable-policy-workflow: transcript hash or size differs",
            ),
            (
                "source-fields",
                "durable-policy-workflow source 0: source record fields differ",
            ),
        )
        for case, expected in cases:
            with self.subTest(case=case), private_temporary_directory() as temporary:
                root = Path(temporary) / "terminal"
                manifest = write_test_bundle(root)
                capture = cast(dict[str, object], manifest["capture"])
                tool = cast(dict[str, object], manifest["tool"])
                commands = cast(list[dict[str, object]], manifest["commands"])
                command = commands[0]
                sources = cast(list[dict[str, object]], command["sources"])
                stderr = cast(dict[str, object], command["stderr"])
                stdout = cast(dict[str, object], command["stdout"])
                if case == "top-level":
                    manifest["unexpected"] = False
                elif case == "commit-role":
                    capture["base_input_commit_role"] = "unverified claim"
                elif case == "environment":
                    capture["environment"] = {"LANG": "C"}
                elif case == "normalization":
                    capture["normalization"] = {}
                elif case == "working-directory":
                    capture["working_directory"] = "elsewhere"
                elif case == "interpreter":
                    capture["interpreter"] = {
                        "implementation": "PyPy",
                        "version": "3.12.3",
                    }
                elif case == "reproduction":
                    manifest["reproduction"] = {"check": "different"}
                elif case == "safety":
                    manifest["safety"] = {"evaluator_executed": True}
                elif case == "safety-bool-alias":
                    safety = cast(dict[str, object], manifest["safety"])
                    safety["evaluator_executed"] = 0
                elif case == "tool":
                    tool["stdlib_only"] = False
                elif case == "tool-source":
                    source = cast(dict[str, object], tool["source"])
                    source["path"] = "README.md"
                elif case == "command-fields":
                    command["unexpected"] = False
                elif case == "command-title":
                    command["title"] = "Unbound title"
                elif case == "command-description":
                    command["description"] = "Unbound description"
                elif case == "exit-code-bool-alias":
                    command["expected_exit_codes"] = [False]
                elif case == "stderr-count-bool-alias":
                    stderr["byte_count"] = False
                elif case == "stdout-count-bool-alias":
                    stdout["byte_count"] = False
                else:
                    sources[0]["unexpected"] = False
                (root / capture_terminal.MANIFEST_NAME).write_bytes(
                    capture_terminal._canonical_json(manifest)
                )

                self.assertIn(
                    expected,
                    capture_terminal.check_bundle(root),
                )

    def test_base_commit_reader_uses_repository_metadata_without_git_process(
        self,
    ) -> None:
        if not (capture_terminal.ROOT / ".git").exists():
            self.skipTest("source distributions intentionally omit Git metadata")
        with patch.object(subprocess, "run") as run:
            commit = capture_terminal._head_commit()

        self.assertRegex(commit, r"^[0-9a-f]{40,64}$")
        run.assert_not_called()


class CommittedTerminalBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = capture_terminal.TERMINAL_ROOT
        self.manifest = cast(
            dict[str, object],
            json.loads((self.root / capture_terminal.MANIFEST_NAME).read_bytes()),
        )
        raw_commands = cast(list[dict[str, object]], self.manifest["commands"])
        self.commands = {
            cast(str, command["capture_id"]): command for command in raw_commands
        }

    def transcript(self, capture_id: str) -> str:
        spec = capture_terminal.COMMAND_BY_ID[capture_id]
        return (self.root / spec.transcript_name).read_text("utf-8")

    def test_frozen_terminal_assets_and_manifest_match_exact_bytes(self) -> None:
        paths = {path.name: path for path in self.root.iterdir() if path.is_file()}

        self.assertEqual(set(paths), set(FROZEN_TERMINAL_SHA256))
        self.assertEqual(
            {name: sha256(path.read_bytes()) for name, path in paths.items()},
            FROZEN_TERMINAL_SHA256,
        )
        self.assertEqual(
            {name: path.stat().st_size for name, path in paths.items()},
            FROZEN_TERMINAL_BYTE_COUNTS,
        )
        for path in paths.values():
            self.assertFalse(path.is_symlink())
            self.assertTrue(path.is_file())
        self.assertEqual(
            capture_terminal._canonical_json(self.manifest),
            paths[capture_terminal.MANIFEST_NAME].read_bytes(),
        )
        self.assertEqual(capture_terminal.check_bundle(), ())

    def test_manifest_binds_exact_base_commit_and_source_bytes(self) -> None:
        capture = cast(dict[str, object], self.manifest["capture"])
        self.assertEqual(
            capture["base_input_commit"],
            FROZEN_BASE_INPUT_COMMIT,
        )
        self.assertEqual(
            capture["base_input_commit_role"],
            "Git HEAD at record time; per-file hashes bind exact source bytes.",
        )
        self.assertEqual(
            capture["environment"],
            {
                "HOME": ".gworker/terminal-capture-runtime/home",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "<OS_DEFPATH>",
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": "src",
                "TMPDIR": ".gworker/terminal-capture-runtime/tmp",
                "TZ": "UTC",
            },
        )
        self.assertEqual(
            capture["interpreter"],
            {"implementation": "CPython", "version": "3.12.13"},
        )

        observed_sources: dict[str, str] = {}
        observed_sizes: dict[str, int] = {}
        for command in self.commands.values():
            sources = cast(list[dict[str, object]], command["sources"])
            for source in sources:
                path = cast(str, source["path"])
                digest = cast(str, source["sha256"])
                byte_count = cast(int, source["byte_count"])
                if path in observed_sources:
                    self.assertEqual(observed_sources[path], digest)
                    self.assertEqual(observed_sizes[path], byte_count)
                observed_sources[path] = digest
                observed_sizes[path] = byte_count

        tool = cast(dict[str, object], self.manifest["tool"])
        tool_source = cast(dict[str, object], tool["source"])
        tool_path = cast(str, tool_source["path"])
        observed_sources[tool_path] = cast(str, tool_source["sha256"])
        observed_sizes[tool_path] = cast(int, tool_source["byte_count"])

        self.assertEqual(observed_sources, FROZEN_CAPTURE_SOURCE_SHA256)
        for relative, expected_sha256 in FROZEN_CAPTURE_SOURCE_SHA256.items():
            source_path = capture_terminal.ROOT / relative
            self.assertEqual(sha256(source_path.read_bytes()), expected_sha256)
            self.assertEqual(source_path.stat().st_size, observed_sizes[relative])

    def test_policy_transcript_proves_evidence_and_preserves_nonclaim(self) -> None:
        content = self.transcript("policy-demo")
        rows = re.findall(
            r"^\s+(\d+)\s+(focus-\d+)\s+\d+\.\d+\s+"
            r"(global|exact)\s+(too_short|just_right|too_long)\s+"
            r"(true|false)$",
            content,
            flags=re.MULTILINE,
        )

        self.assertEqual(len(rows), 12)
        self.assertEqual(tuple(int(row[0]) for row in rows), tuple(range(1, 13)))
        self.assertEqual(
            tuple(row[2] for row in rows),
            ("global",) * 6 + ("exact",) * 6,
        )
        self.assertEqual(rows[0][1:], ("focus-25", "global", "too_short", "false"))
        self.assertEqual(rows[4][1:], ("focus-50", "global", "too_long", "true"))
        self.assertIn("Decision 13: focus-40 at p=0.732265", content)
        self.assertIn("Evidence: exact:deep_work:medium (12 reviews)", content)
        self.assertIn(
            "Reasons: exact_evidence, bounded_exploration, one_step_guardrail",
            content,
        )
        self.assertIn("probabilities sum to 1.000000000000", content)
        self.assertIn(
            "Notice: synthetic API demo, not locked-evaluation evidence.",
            content,
        )
        self.assertNotIn("optimal", content.lower())

    def test_durable_workflow_proves_exact_reopen_lineage_and_cleanup(self) -> None:
        content = self.transcript("durable-policy-workflow")

        self.assertIn(
            "GWorker durable policy journal | deterministic synthetic workflow",
            content,
        )
        self.assertRegex(
            content,
            r"D1 recommend \| sequence 1 \| focus-\d+ \| exact p=0x",
        )
        self.assertIn(
            "D1 review    | fit=just_right | completed=true | "
            "exact propensity preserved",
            content,
        )
        self.assertRegex(
            content,
            r"D2 recommend \| sequence 2 \| focus-\d+ \| "
            r"history evidence=1 \| exact p=0x",
        )
        self.assertIn(
            "Verify       | decisions/reviews/history edges = 2/1/1 | SQLite=ok",
            content,
        )
        self.assertIn("workspace 0700 | journal 0600 | path omitted", content)
        self.assertIn("every CLI call reopened the same disposable journal", content)
        self.assertIn("temporary workspace removed", content)
        self.assertIn("not a human outcome or locked evaluation", content)

    def test_journal_transcript_proves_real_reopen_replay_and_tamper_detection(
        self,
    ) -> None:
        content = self.transcript("journal-recovery")
        event_rows = re.findall(
            r"^\s+(\d+)\s+\d{4}-\d{2}-\d{2}T\S+\s+([a-z_]+)$",
            content,
            flags=re.MULTILINE,
        )

        self.assertEqual(
            event_rows,
            [
                ("1", "session_planned"),
                ("2", "focus_started"),
                ("3", "interruption_recorded"),
                ("4", "focus_completed"),
                ("5", "break_started"),
                ("6", "break_completed"),
            ],
        )
        self.assertIn(
            "Workspace: .gworker/visual-demo/terminal-capture",
            content,
        )
        self.assertIn("file 0600, directory 0700, owner=current-user", content)
        self.assertIn(
            "Reopen + replay: phase=completed, revision=6, terminal=True",
            content,
        )
        self.assertIn(
            "Integrity: PRAGMA quick_check=ok, sessions=1, events=6",
            content,
        )
        self.assertIn("Safe synthetic tamper copy", content)
        self.assertIn(
            "Replay:    detected=true (CorruptJournal:",
            content,
        )
        self.assertIn("Live journal remains verified and unchanged.", content)

    def test_inventory_transcript_is_expectation_only_with_zero_result_data(
        self,
    ) -> None:
        content = self.transcript("protocol-inventory")

        self.assertIn(
            "GWorker locked protocol inventory | no evaluation executed",
            content,
        )
        self.assertIn("split=eval  horizon=288  policy_replicas=4", content)
        self.assertIn(
            "seeds=128 (0..127)  personas=9  availability_modes=2",
            content,
        )
        self.assertIn("decisions                         6,635,520", content)
        self.assertIn("raw_trace_points                  1,032,192", content)
        self.assertIn(
            "Call surface: validate_experiment_config, "
            "expected_publication_cardinalities",
            content,
        )
        self.assertTrue(
            content.endswith("Result data: none (inventory arithmetic only).\n")
        )
        self.assertNotIn("evaluation completed", content.lower())

    def test_offline_replay_transcript_proves_support_and_preserves_nonclaims(
        self,
    ) -> None:
        content = self.transcript("offline-replay")

        self.assertIn(
            "GWorker offline replay | fixture=authored-synthetic-replay-v1",
            content,
        )
        self.assertIn(
            "behavior-control | status=reportable | exact=true",
            content,
        )
        self.assertIn(
            "raw-ess=12.111890 | raw-ess-ratio=0.756993 | max-weight=5.191650",
            content,
        )
        self.assertIn(
            "clipped-rows=4 | clip=3.000000 | removed-weight-mass=5.445540",
            content,
        )
        self.assertIn(
            "focus-15=4/16 | focus-25=4/16 | focus-40=4/16 | focus-50=4/16",
            content,
        )
        self.assertIn("locked-evaluation-used=false", content)
        self.assertTrue(content.endswith("  no locked evaluation\n"))
        self.assertNotIn("causal-effect-estimated=true", content)

    def test_status_and_host_preflight_are_exact_fail_closed_observations(
        self,
    ) -> None:
        status = json.loads(self.transcript("publication-status"))
        preflight = json.loads(self.transcript("publication-preflight"))

        self.assertEqual(
            status,
            {
                "artifacts": [],
                "disposition": "not-started",
                "ok": True,
                "stage": "unclaimed",
            },
        )
        self.assertEqual(
            preflight,
            {
                "effective_memory_bytes": 15_607_062_528,
                "effective_swap_bytes": 3_221_221_376,
                "failure_codes": [],
                "filesystem_available_bytes": 93_422_592_000,
                "filesystem_available_inodes": 18_483_377,
                "nofile_soft_limit": 65_536,
                "ok": True,
                "ready": True,
            },
        )
        status_record = self.commands["publication-status"]
        preflight_record = self.commands["publication-preflight"]
        self.assertEqual(status_record["exit_code"], 0)
        self.assertIs(status_record["host_dependent"], False)
        self.assertEqual(preflight_record["exit_code"], 0)
        self.assertIs(preflight_record["host_dependent"], True)
        self.assertEqual(
            preflight_record["host_note"],
            "CAPTURED ON THIS HOST · readiness may vary elsewhere",
        )
        safety = cast(dict[str, object], self.manifest["safety"])
        self.assertIs(safety["publication_run_executed"], False)
        self.assertIs(safety["evaluator_executed"], False)
        for command in self.commands.values():
            argv = cast(list[str], command["argv"])
            self.assertNotIn("run", argv)
            self.assertNotIn("run_experiment", argv)

    def test_bundle_has_no_host_paths_personal_data_or_secret_signatures(
        self,
    ) -> None:
        forbidden_literals = (
            str(capture_terminal.ROOT),
            "/home/",
            "/Users/",
            "C:\\Users\\",
            "github_pat_",
            "ghp_",
            "AWS_SECRET_ACCESS_KEY",
            "BEGIN PRIVATE KEY",
            "31526072+",
            "@users.noreply.github.com",
        )
        for path in sorted(self.root.iterdir()):
            with self.subTest(path=path.name):
                content = path.read_text("utf-8")
                for forbidden in forbidden_literals:
                    self.assertNotIn(forbidden, content)
                self.assertIsNone(capture_terminal.SECRET_PATTERN.search(content))
                self.assertNotRegex(
                    content,
                    r"\b(?:elapsed|wall[_ -]?time|real\s+\d+m)\b",
                )
                if path.suffix in {".txt", ".json"}:
                    self.assertIsNone(capture_terminal.ABSOLUTE_PATH.search(content))

    def test_every_svg_is_accessible_and_matches_transcript_render_exactly(
        self,
    ) -> None:
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        for spec in capture_terminal.COMMANDS:
            with self.subTest(capture_id=spec.capture_id):
                command = self.commands[spec.capture_id]
                exit_code = cast(int, command["exit_code"])
                transcript = (self.root / spec.transcript_name).read_bytes()
                visual = (self.root / spec.visual_name).read_bytes()
                record = capture_terminal.CaptureRecord(
                    stdout=transcript,
                    stderr=b"",
                    exit_code=exit_code,
                )
                self.assertEqual(
                    visual,
                    capture_terminal.render_svg(spec, record),
                )
                self.assertEqual(
                    capture_terminal._validate_svg(visual, spec),
                    (),
                )
                root = ElementTree.fromstring(visual)
                title = root.find("svg:title", namespace)
                description = root.find("svg:desc", namespace)
                labelled_by = root.attrib["aria-labelledby"].split()
                self.assertEqual(root.attrib["role"], "img")
                self.assertIsNotNone(title)
                self.assertIsNotNone(description)
                self.assertIn(
                    title.attrib["id"],  # type: ignore[union-attr]
                    labelled_by,
                )
                self.assertIn(
                    description.attrib["id"],  # type: ignore[union-attr]
                    labelled_by,
                )
                text = visual.decode("utf-8")
                self.assertIn(spec.title, text)
                self.assertIn(
                    f"REAL OUTPUT · EXIT {exit_code}",
                    text,
                )
                self.assertIn(sha256(transcript), text)

    def test_committed_check_cli_is_subprocess_free(self) -> None:
        output = io.StringIO()
        with (
            patch.object(subprocess, "run") as run,
            redirect_stdout(output),
        ):
            code = capture_terminal.main(["check"])

        self.assertEqual(code, 0)
        self.assertEqual(
            output.getvalue(),
            "verified 7 terminal captures without command execution\n",
        )
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
