#!/usr/bin/env python3
"""Capture and render GWorker's fixed four-process CLI motion evidence.

``record`` is the only mode that starts application processes. It runs four
literal commands with stdout attached to a fixed-size pseudo-terminal, keeps
stderr separate, and writes artifacts only after the complete workflow and
source snapshot have been verified. ``render`` and ``check`` use only the
committed event document; their frame timing is presentation timing, never a
measurement of command latency.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import io
import json
import os
import pty
import re
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import termios
import textwrap
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFont, features
from PIL import __version__ as PILLOW_VERSION

ROOT: Final = Path(__file__).resolve().parents[2]
MOTION_ROOT: Final = ROOT / "docs" / "visuals" / "motion"
RUNTIME_ROOT: Final = ROOT / ".gworker" / "cli-motion-runtime"

EVENTS_NAME: Final = "durable-policy-workflow.events.json"
TRANSCRIPT_NAME: Final = "durable-policy-workflow.txt"
GIF_NAME: Final = "durable-policy-workflow.gif"
POSTER_NAME: Final = "durable-policy-workflow.png"
MANIFEST_NAME: Final = "manifest.json"

EVENTS_SCHEMA: Final = "gworker-cli-motion-events-v1"
MANIFEST_SCHEMA: Final = "gworker-cli-motion-manifest-v1"
TOOL_NAME: Final = "gworker-cli-motion-capture"
TOOL_VERSION: Final = "1"
REQUIRED_PILLOW_VERSION: Final = "12.3.0"

WIDTH: Final = 960
HEIGHT: Final = 540
TERMINAL_COLUMNS: Final = 120
TERMINAL_ROWS: Final = 40
MAX_STREAM_BYTES: Final = 65_536
PROCESS_TIMEOUT_SECONDS: Final = 30
FRAME_DURATIONS_MS: Final = (
    900,
    900,
    1_400,
    800,
    1_300,
    800,
    1_400,
    800,
    1_600,
    1_800,
    3_000,
)

FIRST_DECISION_ID: Final = "018f4f69-e7a2-7f84-8c2d-9f531c4e9101"
SECOND_DECISION_ID: Final = "018f4f69-e7a2-7f84-8c2d-9f531c4e9102"
POLICY_ID: Final = "hierarchical-softmax-ucb-v1.8c10875dd38a025d"

RECORD_COMMAND: Final = (
    "PYTHONPATH=src python scripts/visuals/capture_cli_motion.py record"
)
RENDER_COMMAND: Final = (
    "PYTHONPATH=src python scripts/visuals/capture_cli_motion.py render"
)
CHECK_COMMAND: Final = (
    "PYTHONPATH=src python scripts/visuals/capture_cli_motion.py check"
)

EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()
HEX_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT: Final = re.compile(r"^[0-9a-f]{40,64}$")
ANSI_ESCAPE: Final = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")
ABSOLUTE_PATH: Final = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9_.@+-]+/)+[A-Za-z0-9_.@+-]*"
)
SECRET_PATTERN: Final = re.compile(
    r"(?i)"
    r"(?:github_pat_[A-Za-z0-9_]{20,}"
    r"|gh[opsu]_[A-Za-z0-9]{20,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|sk-[A-Za-z0-9_-]{20,}"
    r"|Bearer[ \t]+[A-Za-z0-9._~+/-]{12,}=*"
    r"|-----BEGIN[ A-Z]+PRIVATE KEY-----)"
)

BACKGROUND: Final = "#07111F"
PANEL: Final = "#0F1B2D"
PANEL_EDGE: Final = "#2E405C"
TEXT: Final = "#F4F7FB"
MUTED: Final = "#9BAFC8"
CYAN: Final = "#36D7E8"
GREEN: Final = "#65E6A8"
PURPLE: Final = "#A78BFA"
ORANGE: Final = "#FDBA74"

SOURCE_PATHS: Final = (
    "docs/cli-motion.md",
    "pyproject.toml",
    "scripts/visuals/capture_cli_motion.py",
    "src/gworker/__init__.py",
    "src/gworker/cli.py",
    "src/gworker/codec.py",
    "src/gworker/domain.py",
    "src/gworker/policy.py",
    "src/gworker/storage.py",
)

CAPTURE_ENVIRONMENT: Final[dict[str, str]] = {
    "HOME": ".gworker/cli-motion-runtime/<capture>/home",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "<OS_DEFPATH>",
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONPATH": "src",
    "TMPDIR": ".gworker/cli-motion-runtime/<capture>/tmp",
    "TZ": "UTC",
}
TERMINAL_CONTRACT: Final[dict[str, object]] = {
    "columns": TERMINAL_COLUMNS,
    "rows": TERMINAL_ROWS,
    "stdout_is_pty": True,
    "stderr_is_separate_pipe": True,
    "crlf_normalization": "PTY CRLF converted to LF; bare CR rejected",
}
PRESENTATION_CONTRACT: Final[dict[str, object]] = {
    "frame_count": len(FRAME_DURATIONS_MS),
    "frame_durations_ms": list(FRAME_DURATIONS_MS),
    "height": HEIGHT,
    "kind": "fixed-presentation-timing",
    "loop": 0,
    "measured_command_latency": False,
    "width": WIDTH,
}
SAFETY_CONTRACT: Final[dict[str, object]] = {
    "arbitrary_commands_accepted": False,
    "evaluator_executed": False,
    "journal_path_published": False,
    "publication_run_executed": False,
    "shell_used": False,
    "stdin_inherited": False,
    "synthetic_demo_data_only": True,
}


@dataclass(frozen=True, slots=True)
class CommandSpec:
    capture_id: str
    title: str
    tail: tuple[str, ...]

    @property
    def display_argv(self) -> tuple[str, ...]:
        return (
            "python",
            "-m",
            "gworker.cli",
            "--journal",
            "<PRIVATE-JOURNAL>",
            *self.tail,
        )

    def runtime_argv(self, journal: Path) -> tuple[str, ...]:
        return (
            "python",
            "-m",
            "gworker.cli",
            "--journal",
            str(journal),
            *self.tail,
        )


COMMANDS: Final = (
    CommandSpec(
        "recommend-initial",
        "Seeded recommendation · no prior reviews",
        (
            "recommend",
            "--task-kind",
            "deep_work",
            "--energy",
            "medium",
            "--available-minutes",
            "60",
            "--decision-id",
            FIRST_DECISION_ID,
            "--seed",
            "20260725",
        ),
    ),
    CommandSpec(
        "review-explicit",
        "Explicit review · propensity preserved",
        (
            "review",
            FIRST_DECISION_ID,
            "--fit",
            "just_right",
            "--completed",
        ),
    ),
    CommandSpec(
        "recommend-reopened",
        "Reopened journal · one reviewed decision",
        (
            "recommend",
            "--task-kind",
            "deep_work",
            "--energy",
            "medium",
            "--available-minutes",
            "60",
            "--previous-focus-minutes",
            "40",
            "--decision-id",
            SECOND_DECISION_ID,
            "--seed",
            "20260726",
        ),
    ),
    CommandSpec(
        "verify-reopened",
        "Reopened journal · exact replay verification",
        ("verify",),
    ),
)

EXPECTED_STDOUT: Final = {
    "recommend-initial": (
        "GWorker recommendation recorded\n"
        f"Decision: {FIRST_DECISION_ID} (sequence 1)\n"
        "Template: focus-40 | 40m focus + 8m break\n"
        "Propensity: 0.250000000000 | exact 0x1.0000000000000p-2\n"
        "Evidence: global (0 reviewed decisions)\n"
        "Reasons: global_evidence, bounded_exploration, cold_start\n"
        "Replay seed: 20260725\n"
        "Journal: private local file (path omitted)\n"
    ),
    "review-explicit": (
        "GWorker review recorded\n"
        f"Decision: {FIRST_DECISION_ID} (sequence 1)\n"
        "Outcome: fit=just_right | completed=true | reward=1.000\n"
        "Provenance: focus-40 at p=0.250000000000\n"
        "Exact propensity: 0x1.0000000000000p-2\n"
        "Journal: private local file (path omitted)\n"
    ),
    "recommend-reopened": (
        "GWorker recommendation recorded\n"
        f"Decision: {SECOND_DECISION_ID} (sequence 2)\n"
        "Template: focus-25 | 25m focus + 5m break\n"
        "Propensity: 0.279643550634 | exact 0x1.1e5ae1020930fp-2\n"
        "Evidence: global (1 reviewed decisions)\n"
        "Reasons: global_evidence, bounded_exploration, one_step_guardrail\n"
        "Replay seed: 20260726\n"
        "Journal: private local file (path omitted)\n"
    ),
    "verify-reopened": (
        "GWorker journal verified\n"
        "Sessions/events: 0 / 0\n"
        "Policy decisions/reviews/history edges: 2 / 1 / 1\n"
        f"Policy: {POLICY_ID}\n"
        "SQLite quick_check: ok\n"
        "Replay: every decision for this policy matched\n"
        "Journal: private local file (path omitted)\n"
    ),
}


class MotionCaptureError(RuntimeError):
    """Raised when motion evidence cannot be captured or verified safely."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise MotionCaptureError("JSON document contains a duplicate field")
        document[key] = value
    return document


def _load_json_bytes(content: bytes, *, label: str) -> dict[str, object]:
    try:
        document = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                MotionCaptureError(f"{label} contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MotionCaptureError(f"{label} is not valid UTF-8 JSON") from error
    if type(document) is not dict:
        raise MotionCaptureError(f"{label} must be a JSON object")
    return document


def _require_pillow() -> None:
    if PILLOW_VERSION != REQUIRED_PILLOW_VERSION:
        raise MotionCaptureError(
            f"motion rendering requires Pillow {REQUIRED_PILLOW_VERSION}"
        )


def _source_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        try:
            metadata = os.lstat(path)
            content = path.read_bytes()
            after = os.lstat(path)
        except OSError as error:
            raise MotionCaptureError(
                f"cannot read motion provenance source: {relative}"
            ) from error
        before_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or before_identity != after_identity
            or len(content) != metadata.st_size
        ):
            raise MotionCaptureError(
                f"motion provenance source is unstable: {relative}"
            )
        records.append(
            {
                "byte_count": len(content),
                "path": relative,
                "sha256": _sha256(content),
            }
        )
    return records


def _git_environment() -> dict[str, str]:
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": str(ROOT / ".gworker" / "git-home"),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "TZ": "UTC",
    }


def _run_git(arguments: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            ("/usr/bin/git", *arguments),
            cwd=ROOT,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MotionCaptureError("cannot inspect the recording source") from error
    if result.returncode != 0 or result.stderr:
        raise MotionCaptureError("cannot inspect the recording source")
    return result.stdout


def _clean_source_commit() -> str:
    status = _run_git(
        (
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
            "--",
        )
    )
    if status:
        raise MotionCaptureError("recording requires a clean committed worktree")
    try:
        commit = _run_git(("rev-parse", "--verify", "HEAD")).decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise MotionCaptureError("source commit is not ASCII") from error
    if not HEX_COMMIT.fullmatch(commit):
        raise MotionCaptureError("source commit is malformed")
    return commit


def _assert_sources_match_commit(
    records: Sequence[Mapping[str, object]],
    source_commit: str,
) -> None:
    if not HEX_COMMIT.fullmatch(source_commit):
        raise MotionCaptureError("source commit is malformed")
    if len(records) != len(SOURCE_PATHS):
        raise MotionCaptureError("motion source inventory is malformed")
    for relative, record in zip(SOURCE_PATHS, records, strict=True):
        if record.get("path") != relative:
            raise MotionCaptureError("motion source inventory is malformed")
        blob = _run_git(("cat-file", "blob", f"{source_commit}:{relative}"))
        if record != {
            "byte_count": len(blob),
            "path": relative,
            "sha256": _sha256(blob),
        }:
            raise MotionCaptureError(
                f"motion source does not match commit blob: {relative}"
            )


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
        metadata = os.lstat(path)
        if not stat.S_ISDIR(metadata.st_mode):
            raise MotionCaptureError("motion runtime component is not a directory")
        os.chmod(path, 0o700)
    except MotionCaptureError:
        raise
    except OSError as error:
        raise MotionCaptureError("cannot prepare private motion runtime") from error


def _prepare_runtime_parent() -> None:
    _ensure_private_directory(ROOT / ".gworker")
    _ensure_private_directory(RUNTIME_ROOT)


def _capture_environment(workspace: Path) -> dict[str, str]:
    home = workspace / "home"
    temporary = workspace / "tmp"
    _ensure_private_directory(home)
    _ensure_private_directory(temporary)
    return {
        "HOME": str(home),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": "src",
        "TMPDIR": str(temporary),
        "TZ": "UTC",
    }


def _terminate_process_group(
    process: subprocess.Popen[bytes],
) -> MotionCaptureError | None:
    failure: MotionCaptureError | None = None
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        failure = MotionCaptureError("cannot terminate motion process group")
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        failure = MotionCaptureError("motion process group survived termination")
    return failure


def _run_pty_process(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
) -> tuple[bytes, bytes, int]:
    """Run one fixed argv with PTY stdout and a separately bounded stderr."""

    master_fd, slave_fd = pty.openpty()
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    process_error: BaseException | None = None
    try:
        fcntl.ioctl(
            slave_fd,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", TERMINAL_ROWS, TERMINAL_COLUMNS, 0, 0),
        )
        process = subprocess.Popen(
            argv,
            executable=sys.executable,
            cwd=ROOT,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave_fd)
        slave_fd = -1
        if process.stderr is None:
            raise MotionCaptureError("motion process has no stderr pipe")
        stderr_fd = process.stderr.fileno()
        os.set_blocking(master_fd, False)
        os.set_blocking(stderr_fd, False)
        selector.register(master_fd, selectors.EVENT_READ, ("stdout", stdout))
        selector.register(stderr_fd, selectors.EVENT_READ, ("stderr", stderr))
        deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MotionCaptureError("motion command timed out")
            ready = selector.select(min(remaining, 0.25))
            if not ready and process.poll() is not None:
                continue
            for key, _events in ready:
                label, buffer = key.data
                try:
                    chunk = os.read(key.fd, 4_096)
                except BlockingIOError:
                    continue
                except OSError as error:
                    if label == "stdout" and error.errno == errno.EIO:
                        chunk = b""
                    else:
                        raise MotionCaptureError(
                            f"cannot read motion {label}"
                        ) from error
                if chunk:
                    buffer.extend(chunk)
                    if len(buffer) > MAX_STREAM_BYTES:
                        raise MotionCaptureError(
                            f"motion {label} exceeded its byte limit"
                        )
                    continue
                selector.unregister(key.fd)

        return_code = process.wait(timeout=5)
        return bytes(stdout), bytes(stderr), return_code
    except subprocess.TimeoutExpired as error:
        process_error = MotionCaptureError("motion command did not exit")
        raise process_error from error
    except BaseException as error:
        process_error = error
        raise
    finally:
        selector.close()
        if process is not None:
            termination_error = _terminate_process_group(process)
            if termination_error is not None:
                if process_error is None:
                    raise termination_error
                process_error.add_note(str(termination_error))
            if process.stderr is not None:
                process.stderr.close()
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)


def _normalize_terminal_output(content: bytes, *, label: str) -> str:
    if b"\r" in content.replace(b"\r\n", b""):
        raise MotionCaptureError(f"{label} contains a bare carriage return")
    normalized = content.replace(b"\r\n", b"\n")
    try:
        text = normalized.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MotionCaptureError(f"{label} is not valid UTF-8") from error
    if not text.endswith("\n"):
        raise MotionCaptureError(f"{label} lacks a final newline")
    if "\x00" in text or ANSI_ESCAPE.search(text):
        raise MotionCaptureError(f"{label} contains terminal control bytes")
    if ABSOLUTE_PATH.search(text):
        raise MotionCaptureError(f"{label} contains an absolute path")
    if SECRET_PATTERN.search(text):
        raise MotionCaptureError(f"{label} contains a credential-shaped value")
    return text


def _command_record(spec: CommandSpec, stdout: str) -> dict[str, object]:
    stdout_bytes = stdout.encode("utf-8")
    return {
        "argv": list(spec.display_argv),
        "capture_id": spec.capture_id,
        "exit_code": 0,
        "stderr": {
            "byte_count": 0,
            "sha256": EMPTY_SHA256,
        },
        "stdout": stdout,
        "stdout_byte_count": len(stdout_bytes),
        "stdout_sha256": _sha256(stdout_bytes),
        "title": spec.title,
    }


def _capture_workflow() -> tuple[list[dict[str, object]], str, str, bool]:
    """Run the fixed workflow and return path-free records after cleanup."""

    _prepare_runtime_parent()
    workspace = Path(tempfile.mkdtemp(prefix="capture-", dir=RUNTIME_ROOT))
    os.chmod(workspace, 0o700)
    journal = workspace / "journal.sqlite3"
    records: list[dict[str, object]] = []
    workspace_mode = ""
    journal_mode = ""
    capture_error: BaseException | None = None
    try:
        environment = _capture_environment(workspace)
        workspace_mode = f"{stat.S_IMODE(os.lstat(workspace).st_mode):04o}"
        for spec in COMMANDS:
            raw_stdout, raw_stderr, exit_code = _run_pty_process(
                spec.runtime_argv(journal),
                environment=environment,
            )
            stdout = _normalize_terminal_output(
                raw_stdout,
                label=f"{spec.capture_id} stdout",
            )
            stderr = (
                _normalize_terminal_output(
                    raw_stderr,
                    label=f"{spec.capture_id} stderr",
                )
                if raw_stderr
                else ""
            )
            if exit_code != 0 or stderr:
                raise MotionCaptureError(
                    f"{spec.capture_id} did not complete with empty stderr"
                )
            if stdout != EXPECTED_STDOUT[spec.capture_id]:
                raise MotionCaptureError(
                    f"{spec.capture_id} output differs from the fixed workflow"
                )
            records.append(_command_record(spec, stdout))

        metadata = os.lstat(journal)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise MotionCaptureError("motion journal identity is unsafe")
        journal_mode = f"{stat.S_IMODE(metadata.st_mode):04o}"
        if workspace_mode != "0700" or journal_mode != "0600":
            raise MotionCaptureError("motion workspace permissions are not private")
    except BaseException as error:
        capture_error = error
        raise
    finally:
        try:
            shutil.rmtree(workspace)
        except OSError as error:
            cleanup_error = MotionCaptureError("cannot remove motion workspace")
            if capture_error is None:
                raise cleanup_error from error
            capture_error.add_note(str(cleanup_error))

    removed = not workspace.exists()
    if not removed:
        raise MotionCaptureError("motion workspace survived cleanup")
    return records, workspace_mode, journal_mode, removed


def _events_document(
    *,
    source_commit: str,
    sources: list[dict[str, object]],
    commands: list[dict[str, object]],
    workspace_mode: str,
    journal_mode: str,
    removed: bool,
) -> dict[str, object]:
    return {
        "capture": {
            "environment": CAPTURE_ENVIRONMENT,
            "terminal": TERMINAL_CONTRACT,
            "workspace": {
                "journal_mode": journal_mode,
                "removed_after_capture": removed,
                "workspace_mode": workspace_mode,
            },
        },
        "commands": commands,
        "presentation": PRESENTATION_CONTRACT,
        "provenance": {
            "source_commit": source_commit,
            "source_commit_role": (
                "clean Git commit executed by record; source bytes matched its blobs"
            ),
            "sources": sources,
        },
        "safety": SAFETY_CONTRACT,
        "schema_version": EVENTS_SCHEMA,
    }


def _validate_source_records(records: object) -> list[dict[str, object]]:
    if type(records) is not list or len(records) != len(SOURCE_PATHS):
        raise MotionCaptureError("motion source inventory is malformed")
    expected = _source_records()
    if records != expected:
        raise MotionCaptureError("motion source bytes differ from provenance")
    return expected


def _validate_events(document: Mapping[str, object]) -> None:
    if set(document) != {
        "capture",
        "commands",
        "presentation",
        "provenance",
        "safety",
        "schema_version",
    }:
        raise MotionCaptureError("motion event fields differ")
    if document.get("schema_version") != EVENTS_SCHEMA:
        raise MotionCaptureError("motion event schema differs")
    if document.get("presentation") != PRESENTATION_CONTRACT:
        raise MotionCaptureError("motion presentation contract differs")
    if document.get("safety") != SAFETY_CONTRACT:
        raise MotionCaptureError("motion safety contract differs")

    capture = document.get("capture")
    if type(capture) is not dict or set(capture) != {
        "environment",
        "terminal",
        "workspace",
    }:
        raise MotionCaptureError("motion capture record is malformed")
    if capture.get("environment") != CAPTURE_ENVIRONMENT:
        raise MotionCaptureError("motion capture environment differs")
    if capture.get("terminal") != TERMINAL_CONTRACT:
        raise MotionCaptureError("motion terminal contract differs")
    if capture.get("workspace") != {
        "journal_mode": "0600",
        "removed_after_capture": True,
        "workspace_mode": "0700",
    }:
        raise MotionCaptureError("motion workspace evidence differs")

    provenance = document.get("provenance")
    if type(provenance) is not dict or set(provenance) != {
        "source_commit",
        "source_commit_role",
        "sources",
    }:
        raise MotionCaptureError("motion provenance is malformed")
    source_commit = provenance.get("source_commit")
    if type(source_commit) is not str or not HEX_COMMIT.fullmatch(source_commit):
        raise MotionCaptureError("motion source commit is malformed")
    if provenance.get("source_commit_role") != (
        "clean Git commit executed by record; source bytes matched its blobs"
    ):
        raise MotionCaptureError("motion source commit role differs")
    _validate_source_records(provenance.get("sources"))

    commands = document.get("commands")
    if type(commands) is not list or len(commands) != len(COMMANDS):
        raise MotionCaptureError("motion command inventory is malformed")
    expected_commands = [
        _command_record(spec, EXPECTED_STDOUT[spec.capture_id]) for spec in COMMANDS
    ]
    if commands != expected_commands:
        raise MotionCaptureError("motion command evidence differs")


def _transcript(document: Mapping[str, object]) -> bytes:
    _validate_events(document)
    commands = document["commands"]
    if type(commands) is not list:
        raise MotionCaptureError("motion command inventory is malformed")
    lines = [
        "# GWorker real CLI process output",
        "# Fixed presentation timing; command latency was not measured.",
        "",
    ]
    for command in commands:
        if type(command) is not dict:
            raise MotionCaptureError("motion command record is malformed")
        argv = command["argv"]
        stdout = command["stdout"]
        if type(argv) is not list or type(stdout) is not str:
            raise MotionCaptureError("motion command payload is malformed")
        lines.append("$ " + " ".join(argv))
        lines.extend(stdout.removesuffix("\n").splitlines())
        lines.extend(("[exit 0]", ""))
    lines.extend(
        (
            "# workspace 0700; journal 0600; workspace removed after capture",
            "# synthetic CLI/storage evidence; no human outcome or locked evaluation",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _draw_header(
    draw: ImageDraw.ImageDraw,
    *,
    frame_index: int,
    title: str,
) -> None:
    draw.rounded_rectangle(
        (24, 22, WIDTH - 24, HEIGHT - 22),
        radius=18,
        fill=PANEL,
        outline=PANEL_EDGE,
        width=2,
    )
    draw.ellipse((48, 43, 60, 55), fill="#F97066")
    draw.ellipse((68, 43, 80, 55), fill=ORANGE)
    draw.ellipse((88, 43, 100, 55), fill=GREEN)
    draw.text(
        (120, 40),
        "GWORKER · REAL CLI MOTION EVIDENCE",
        fill=MUTED,
        font=_font(14),
    )
    draw.text((48, 82), title, fill=TEXT, font=_font(23))
    draw.rounded_rectangle(
        (WIDTH - 184, 78, WIDTH - 48, 110),
        radius=16,
        fill=BACKGROUND,
        outline=CYAN,
        width=1,
    )
    draw.text(
        (WIDTH - 164, 86),
        f"FRAME {frame_index + 1:02d}/11",
        fill=CYAN,
        font=_font(14),
    )


def _wrapped(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


def _draw_terminal_step(
    draw: ImageDraw.ImageDraw,
    *,
    command: Mapping[str, object],
    step_index: int,
    show_output: bool,
) -> None:
    argv = command["argv"]
    stdout = command["stdout"]
    if type(argv) is not list or type(stdout) is not str:
        raise MotionCaptureError("motion render command is malformed")
    draw.text(
        (50, 126),
        f"STEP {step_index + 1}/4 · {command['title']}",
        fill=PURPLE,
        font=_font(15),
    )
    y = 160
    for line in _wrapped("$ " + " ".join(argv), 92):
        draw.text((50, y), line, fill=GREEN, font=_font(15))
        y += 21
    y += 10
    if show_output:
        for line_number, line in enumerate(stdout.removesuffix("\n").splitlines()):
            color = CYAN if line_number == 0 else TEXT
            draw.text((50, y), line, fill=color, font=_font(15))
            y += 22
    else:
        draw.text((50, y), "capturing PTY stdout …", fill=MUTED, font=_font(15))
        y += 22
        draw.text(
            (50, y),
            "stdin closed · stderr separate · shell disabled",
            fill=MUTED,
            font=_font(14),
        )
    draw.text(
        (50, HEIGHT - 56),
        "fixed replay timing · not measured latency",
        fill=MUTED,
        font=_font(13),
    )


def _draw_summary(
    draw: ImageDraw.ImageDraw,
    *,
    final: bool,
    transcript_sha256: str,
) -> None:
    cards = (
        ("01", "recommend", "focus-40 · p=0.250000000000", GREEN),
        ("02", "explicit review", "just_right · reward=1.000", PURPLE),
        ("03", "reopen + recommend", "focus-25 · history=1", CYAN),
        ("04", "reopen + verify", "2 / 1 / 1 · SQLite=ok", ORANGE),
    )
    y = 132
    for number, label, value, color in cards:
        draw.rounded_rectangle(
            (48, y, WIDTH - 48, y + 62),
            radius=10,
            fill=BACKGROUND,
            outline=PANEL_EDGE,
            width=1,
        )
        draw.text((66, y + 15), number, fill=color, font=_font(17))
        draw.text((112, y + 13), label, fill=TEXT, font=_font(17))
        draw.text((470, y + 15), value, fill=color, font=_font(15))
        y += 72
    if final:
        draw.text(
            (50, 430),
            "workspace 0700 · journal 0600 · removed after capture",
            fill=GREEN,
            font=_font(14),
        )
        draw.text(
            (50, 456),
            "synthetic workflow · no human outcome · no locked evaluation",
            fill=ORANGE,
            font=_font(14),
        )
    else:
        draw.text(
            (50, 438),
            "Every card above is validated against the captured process outputs.",
            fill=MUTED,
            font=_font(14),
        )
    draw.text(
        (50, HEIGHT - 50),
        f"transcript sha256 · {transcript_sha256[:32]}…",
        fill=MUTED,
        font=_font(12),
    )


def _render_frames(
    document: Mapping[str, object],
    *,
    transcript_sha256: str,
) -> list[Image.Image]:
    _validate_events(document)
    commands = document["commands"]
    if type(commands) is not list:
        raise MotionCaptureError("motion commands are malformed")
    stages: tuple[tuple[str, int | None, bool], ...] = (
        ("intro", None, False),
        ("step", 0, False),
        ("step", 0, True),
        ("step", 1, False),
        ("step", 1, True),
        ("step", 2, False),
        ("step", 2, True),
        ("step", 3, False),
        ("step", 3, True),
        ("summary", None, False),
        ("summary", None, True),
    )
    frames: list[Image.Image] = []
    for frame_index, (kind, step_index, final) in enumerate(stages):
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        title = (
            "Recommend → review → reopen → verify"
            if kind == "intro"
            else (
                "Durable decision lineage"
                if kind == "summary"
                else "Four separate CLI processes · one private journal"
            )
        )
        _draw_header(draw, frame_index=frame_index, title=title)
        if kind == "intro":
            draw.text(
                (50, 150),
                "A deterministic synthetic workflow captured from production CLI code.",
                fill=TEXT,
                font=_font(17),
            )
            facts = (
                "4 allowlisted processes · PTY stdout · empty stderr",
                "one disposable 0700 workspace · one 0600 SQLite journal",
                "exact IDs, seeds, propensities, reopen lineage, and replay check",
                "presentation timing is fixed; runtime latency is not measured",
            )
            y = 218
            for index, fact in enumerate(facts):
                draw.rounded_rectangle(
                    (52, y - 8, WIDTH - 52, y + 38),
                    radius=8,
                    fill=BACKGROUND,
                    outline=PANEL_EDGE,
                    width=1,
                )
                draw.text(
                    (72, y + 4),
                    f"{index + 1}. {fact}",
                    fill=(CYAN if index == 0 else TEXT),
                    font=_font(15),
                )
                y += 58
        elif kind == "step" and step_index is not None:
            command = commands[step_index]
            if type(command) is not dict:
                raise MotionCaptureError("motion command record is malformed")
            _draw_terminal_step(
                draw,
                command=command,
                step_index=step_index,
                show_output=final,
            )
        else:
            _draw_summary(
                draw,
                final=final,
                transcript_sha256=transcript_sha256,
            )
        frames.append(image)
    return frames


def _render_motion(
    document: Mapping[str, object],
    transcript: bytes,
) -> tuple[bytes, bytes]:
    _require_pillow()
    transcript_sha256 = _sha256(transcript)
    frames = _render_frames(document, transcript_sha256=transcript_sha256)
    if len(frames) != len(FRAME_DURATIONS_MS):
        raise MotionCaptureError("motion frame count differs")

    gif_buffer = io.BytesIO()
    frames[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATIONS_MS,
        loop=0,
        disposal=2,
        optimize=False,
        comment=b"GWorker real CLI output; fixed presentation timing",
    )
    gif = gif_buffer.getvalue()

    poster_buffer = io.BytesIO()
    frames[-1].save(
        poster_buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    poster = poster_buffer.getvalue()

    try:
        with Image.open(io.BytesIO(gif)) as animation:
            if animation.format != "GIF" or getattr(animation, "n_frames", 1) != len(
                frames
            ):
                raise MotionCaptureError("rendered GIF structure differs")
            if animation.size != (WIDTH, HEIGHT) or animation.info.get("loop") != 0:
                raise MotionCaptureError("rendered GIF presentation differs")
    except OSError as error:
        raise MotionCaptureError("rendered GIF is unreadable") from error
    return gif, poster


def _artifact_record(name: str, content: bytes) -> dict[str, object]:
    return {
        "byte_count": len(content),
        "path": f"docs/visuals/motion/{name}",
        "sha256": _sha256(content),
    }


def _manifest(
    document: Mapping[str, object],
    *,
    events: bytes,
    transcript: bytes,
    gif: bytes,
    poster: bytes,
) -> dict[str, object]:
    _validate_events(document)
    provenance = document["provenance"]
    if type(provenance) is not dict:
        raise MotionCaptureError("motion provenance is malformed")
    freetype_version = features.version_module("freetype2")
    return {
        "claim_boundary": [
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
        "outputs": {
            EVENTS_NAME: _artifact_record(EVENTS_NAME, events),
            GIF_NAME: _artifact_record(GIF_NAME, gif),
            POSTER_NAME: _artifact_record(POSTER_NAME, poster),
            TRANSCRIPT_NAME: _artifact_record(TRANSCRIPT_NAME, transcript),
        },
        "presentation": PRESENTATION_CONTRACT,
        "provenance": provenance,
        "reproduction": {
            "check": CHECK_COMMAND,
            "record_real_processes": RECORD_COMMAND,
            "render_without_processes": RENDER_COMMAND,
        },
        "schema_version": MANIFEST_SCHEMA,
        "tool": {
            "freetype_version": freetype_version,
            "name": TOOL_NAME,
            "pillow_version": PILLOW_VERSION,
            "version": TOOL_VERSION,
        },
    }


def _bundle_from_events(events: bytes) -> dict[str, bytes]:
    document = _load_json_bytes(events, label="motion events")
    if _canonical_json(document) != events:
        raise MotionCaptureError("motion events JSON is not canonical")
    _validate_events(document)
    transcript = _transcript(document)
    gif, poster = _render_motion(document, transcript)
    manifest = _canonical_json(
        _manifest(
            document,
            events=events,
            transcript=transcript,
            gif=gif,
            poster=poster,
        )
    )
    return {
        EVENTS_NAME: events,
        TRANSCRIPT_NAME: transcript,
        GIF_NAME: gif,
        POSTER_NAME: poster,
        MANIFEST_NAME: manifest,
    }


def _write_bundle(
    bundle: Mapping[str, bytes],
    output_root: Path = MOTION_ROOT,
) -> None:
    expected = {
        EVENTS_NAME,
        TRANSCRIPT_NAME,
        GIF_NAME,
        POSTER_NAME,
        MANIFEST_NAME,
    }
    if set(bundle) != expected:
        raise MotionCaptureError("motion output inventory differs")
    output_root.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if output_root.exists():
        metadata = os.lstat(output_root)
        if not stat.S_ISDIR(metadata.st_mode):
            raise MotionCaptureError("motion output root is unsafe")
        observed = {path.name for path in output_root.iterdir()}
        if observed - expected:
            raise MotionCaptureError("motion output root contains unknown entries")
    else:
        output_root.mkdir(mode=0o755)
    for name in (*sorted(expected - {MANIFEST_NAME}), MANIFEST_NAME):
        path = output_root / name
        if path.exists() and not path.is_file():
            raise MotionCaptureError("motion output path is unsafe")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.",
            dir=output_root,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(bundle[name])
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def record_bundle(output_root: Path = MOTION_ROOT) -> dict[str, bytes]:
    if output_root != MOTION_ROOT:
        raise MotionCaptureError("recording target is fixed to the motion bundle")
    _require_pillow()
    source_commit = _clean_source_commit()
    sources_before = _source_records()
    _assert_sources_match_commit(sources_before, source_commit)
    commands, workspace_mode, journal_mode, removed = _capture_workflow()
    if _clean_source_commit() != source_commit:
        raise MotionCaptureError("source commit changed during motion capture")
    sources_after = _source_records()
    if sources_after != sources_before:
        raise MotionCaptureError("source bytes changed during motion capture")
    _assert_sources_match_commit(sources_after, source_commit)
    document = _events_document(
        source_commit=source_commit,
        sources=sources_before,
        commands=commands,
        workspace_mode=workspace_mode,
        journal_mode=journal_mode,
        removed=removed,
    )
    events = _canonical_json(document)
    bundle = _bundle_from_events(events)
    _write_bundle(bundle, output_root)
    return bundle


def render_bundle(output_root: Path = MOTION_ROOT) -> dict[str, bytes]:
    try:
        events = (output_root / EVENTS_NAME).read_bytes()
    except OSError as error:
        raise MotionCaptureError("motion events are unavailable") from error
    bundle = _bundle_from_events(events)
    _write_bundle(bundle, output_root)
    return bundle


def check_bundle(output_root: Path = MOTION_ROOT) -> tuple[str, ...]:
    differences: list[str] = []
    expected_names = {
        EVENTS_NAME,
        TRANSCRIPT_NAME,
        GIF_NAME,
        POSTER_NAME,
        MANIFEST_NAME,
    }
    try:
        metadata = os.lstat(output_root)
        if not stat.S_ISDIR(metadata.st_mode):
            return ("motion output root is not a directory",)
        paths = {path.name: path for path in output_root.iterdir()}
    except OSError:
        return ("motion output root is unavailable",)
    if set(paths) != expected_names:
        differences.append("motion output inventory differs")
    for name, path in paths.items():
        try:
            metadata = os.lstat(path)
        except OSError:
            differences.append(f"{name}: cannot inspect artifact")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            differences.append(f"{name}: artifact is not a regular file")
    if differences:
        return tuple(differences)
    try:
        events = paths[EVENTS_NAME].read_bytes()
        expected = _bundle_from_events(events)
    except (OSError, MotionCaptureError) as error:
        return (str(error),)
    for name in sorted(expected_names):
        try:
            actual = paths[name].read_bytes()
        except OSError:
            differences.append(f"{name}: artifact is unreadable")
            continue
        if actual != expected[name]:
            differences.append(f"{name}: bytes differ")
    try:
        manifest = _load_json_bytes(
            paths[MANIFEST_NAME].read_bytes(),
            label="motion manifest",
        )
        if _canonical_json(manifest) != paths[MANIFEST_NAME].read_bytes():
            differences.append("motion manifest JSON is not canonical")
    except (OSError, MotionCaptureError) as error:
        differences.append(str(error))
    return tuple(differences)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record, render, or verify GWorker real CLI motion evidence."
    )
    parser.add_argument(
        "action",
        choices=("record", "render", "check"),
        help=(
            "record runs four fixed CLI processes; render and check execute "
            "no application process"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "record":
            record_bundle()
            print("recorded 4 real CLI processes into 11 motion frames")
            return 0
        if arguments.action == "render":
            render_bundle()
            print("rendered motion evidence without application processes")
            return 0
        differences = check_bundle()
    except MotionCaptureError as error:
        print(f"motion evidence: {error}", file=sys.stderr)
        return 1
    if differences:
        for difference in differences:
            print(f"DIFF: {difference}", file=sys.stderr)
        return 1
    print("verified real CLI motion evidence without application processes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
