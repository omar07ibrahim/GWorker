from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from contextlib import redirect_stderr
from pathlib import Path
from typing import cast
from unittest.mock import call, patch

from scripts.visuals import capture_terminal

TEST_TEMP_ROOT = capture_terminal.ROOT / ".gworker" / "terminal-capture-tests"


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
                "policy-demo": ("python", "scripts/demo_policy.py"),
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

    def test_base_commit_reader_uses_repository_metadata_without_git_process(
        self,
    ) -> None:
        with patch.object(subprocess, "run") as run:
            commit = capture_terminal._head_commit()

        self.assertRegex(commit, r"^[0-9a-f]{40,64}$")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
