#!/usr/bin/env python3
"""Record and render genuine, reproducible GWorker terminal captures.

``record`` is the only mode that starts subprocesses.  It executes five
literal, reviewed command vectors: the public policy demo, the descriptor-safe
SQLite journal demo, the locked protocol inventory, publication status, and
publication preflight.  It never exposes an arbitrary command surface and
never calls the publication ``run`` command or evaluator.

``render`` rebuilds SVGs from committed transcripts.  ``check`` is entirely
read-only and does not rerun even the host-dependent resource preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from xml.sax.saxutils import escape

ROOT: Final = Path(__file__).resolve().parents[2]
TERMINAL_ROOT: Final = ROOT / "docs" / "visuals" / "terminal"
MANIFEST_NAME: Final = "manifest.json"
SCHEMA_VERSION: Final = "gworker-terminal-capture-manifest-v1"
TOOL_NAME: Final = "gworker-terminal-capture"
TOOL_VERSION: Final = "1"
RECORD_COMMAND: Final = (
    "PYTHONPATH=src python scripts/visuals/capture_terminal.py record"
)
RENDER_COMMAND: Final = (
    "PYTHONPATH=src python scripts/visuals/capture_terminal.py render"
)
CHECK_COMMAND: Final = "PYTHONPATH=src python scripts/visuals/capture_terminal.py check"
JOURNAL_WORKSPACE: Final = ".gworker/visual-demo/terminal-capture"
RUNTIME_ROOT: Final = ROOT / ".gworker" / "terminal-capture-runtime"
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

BACKGROUND: Final = "#0B1220"
PANEL: Final = "#101828"
PANEL_EDGE: Final = "#344054"
TEXT: Final = "#F2F4F7"
MUTED: Final = "#98A2B3"
GREEN: Final = "#6CE9A6"
SKY: Final = "#7CD4FD"
ORANGE: Final = "#FDB022"
RED: Final = "#F97066"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One immutable, allowlisted terminal capture."""

    capture_id: str
    title: str
    description: str
    argv: tuple[str, ...]
    expected_exit_codes: tuple[int, ...]
    transcript_name: str
    visual_name: str
    source_paths: tuple[str, ...]
    host_dependent: bool = False
    host_note: str | None = None


@dataclass(frozen=True, slots=True)
class CaptureRecord:
    """Sanitized bytes and exit status produced by an allowlisted command."""

    stdout: bytes
    stderr: bytes
    exit_code: int


COMMANDS: Final[tuple[CommandSpec, ...]] = (
    CommandSpec(
        capture_id="policy-demo",
        title="Adaptive policy · 12 verified reviews",
        description=(
            "Real output from the deterministic public policy demo. It shows "
            "the review history, evidence transition, arm scores, and selected "
            "next focus duration; it is not locked-evaluation evidence."
        ),
        argv=("python", "scripts/demo_policy.py"),
        expected_exit_codes=(0,),
        transcript_name="policy-demo.txt",
        visual_name="policy-demo.svg",
        source_paths=(
            "scripts/demo_policy.py",
            "src/gworker/policy.py",
        ),
    ),
    CommandSpec(
        capture_id="journal-recovery",
        title="SQLite journal · reopen, replay, detect tamper",
        description=(
            "Real output from the descriptor-safe synthetic journal demo. It "
            "writes the production SQLiteEventStore, reopens and replays it, "
            "then detects a logical mutation in a separate copy."
        ),
        argv=(
            "python",
            "scripts/demo_journal.py",
            "--repo-root",
            ".",
            "--workspace",
            JOURNAL_WORKSPACE,
            "--reset",
        ),
        expected_exit_codes=(0,),
        transcript_name="journal-recovery.txt",
        visual_name="journal-recovery.svg",
        source_paths=(
            "scripts/demo_journal.py",
            "src/gworker/codec.py",
            "src/gworker/domain.py",
            "src/gworker/storage.py",
        ),
    ),
    CommandSpec(
        capture_id="protocol-inventory",
        title="Locked protocol · expected inventory, zero results",
        description=(
            "Real output from the read-only locked protocol inventory. It "
            "validates configuration and computes expected cardinalities "
            "without calling the evaluator or publication runner."
        ),
        argv=("python", "scripts/protocol_inventory.py"),
        expected_exit_codes=(0,),
        transcript_name="protocol-inventory.txt",
        visual_name="protocol-inventory.svg",
        source_paths=(
            "scripts/protocol_inventory.py",
            "src/gworker/evaluation.py",
            "src/gworker/evidence.py",
            "src/gworker/policy.py",
            "src/gworker/reporting.py",
        ),
    ),
    CommandSpec(
        capture_id="publication-status",
        title="Publication gate · unspent claim",
        description=(
            "Real path-free output from the publication runner's read-only "
            "status command. No publication run or evaluator is invoked."
        ),
        argv=(
            "python",
            "-m",
            "gworker.publication_runner",
            "status",
        ),
        expected_exit_codes=(0,),
        transcript_name="publication-status.txt",
        visual_name="publication-status.svg",
        source_paths=(
            "src/gworker/publication_runner.py",
            "src/gworker/publication_state.py",
        ),
    ),
    CommandSpec(
        capture_id="publication-preflight",
        title="Publication gate · capacity preflight",
        description=(
            "Real path-free output from the read-only resource preflight. "
            "Capacity is an observation from the capture host, not a portable "
            "success claim, and no publication run or evaluator is invoked."
        ),
        argv=(
            "python",
            "-m",
            "gworker.publication_runner",
            "preflight",
        ),
        expected_exit_codes=(0, 2),
        transcript_name="publication-preflight.txt",
        visual_name="publication-preflight.svg",
        source_paths=(
            "src/gworker/publication_runner.py",
            "src/gworker/resource_preflight.py",
        ),
        host_dependent=True,
        host_note="CAPTURED ON THIS HOST · readiness may vary elsewhere",
    ),
)
COMMAND_BY_ID: Final = {spec.capture_id: spec for spec in COMMANDS}


class CaptureError(RuntimeError):
    """Raised when recording, rendering, or verification is unsafe."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as exc:
        raise CaptureError(f"cannot read provenance source: {path.name}") from exc


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


def _git_directory() -> Path:
    marker = ROOT / ".git"
    if marker.is_dir():
        return marker
    try:
        declaration = marker.read_text("utf-8").strip()
    except OSError as exc:
        raise CaptureError("cannot locate repository metadata") from exc
    prefix = "gitdir: "
    if not declaration.startswith(prefix):
        raise CaptureError("repository metadata declaration is malformed")
    candidate = Path(declaration.removeprefix(prefix))
    location = candidate if candidate.is_absolute() else ROOT / candidate
    try:
        return location.resolve(strict=True)
    except OSError as exc:
        raise CaptureError("repository metadata directory is unavailable") from exc


def _head_commit() -> str:
    git_directory = _git_directory()
    try:
        head = (git_directory / "HEAD").read_text("ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise CaptureError("cannot read repository HEAD") from exc
    if HEX_COMMIT.fullmatch(head):
        return head
    prefix = "ref: "
    if not head.startswith(prefix):
        raise CaptureError("repository HEAD is malformed")
    ref = head.removeprefix(prefix)
    if (
        not ref.startswith("refs/")
        or ".." in Path(ref).parts
        or Path(ref).is_absolute()
    ):
        raise CaptureError("repository HEAD reference is unsafe")
    loose = git_directory / ref
    try:
        commit = loose.read_text("ascii").strip()
    except FileNotFoundError:
        commit = ""
        try:
            lines = (git_directory / "packed-refs").read_text("ascii").splitlines()
        except (OSError, UnicodeError) as exc:
            raise CaptureError("cannot resolve repository HEAD") from exc
        for line in lines:
            if line.startswith(("#", "^")):
                continue
            fields = line.split(" ", 1)
            if len(fields) == 2 and fields[1] == ref:
                commit = fields[0]
                break
    except (OSError, UnicodeError) as exc:
        raise CaptureError("cannot resolve repository HEAD") from exc
    if not HEX_COMMIT.fullmatch(commit):
        raise CaptureError("repository HEAD commit is malformed")
    return commit


def _minimal_environment() -> dict[str, str]:
    home = RUNTIME_ROOT / "home"
    temporary = RUNTIME_ROOT / "tmp"
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    os.chmod(temporary, 0o700)
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


def _scrub_output(content: bytes) -> bytes:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CaptureError("command output must be valid UTF-8") from exc
    replacements = {
        str(ROOT): "<REPO>",
        str(ROOT.resolve()): "<REPO>",
        str(RUNTIME_ROOT): "<RUNTIME>",
        socket.gethostname(): "<HOST>",
    }
    for original in sorted(replacements, key=len, reverse=True):
        if original:
            text = text.replace(original, replacements[original])
    text = SECRET_PATTERN.sub("<REDACTED>", text)
    if ANSI_ESCAPE.search(text):
        raise CaptureError("terminal output unexpectedly contains ANSI controls")
    if "\x00" in text:
        raise CaptureError("terminal output unexpectedly contains NUL")
    if ABSOLUTE_PATH.search(text):
        raise CaptureError("terminal output contains an unredacted absolute path")
    return text.encode("utf-8")


def _run_allowlisted(spec: CommandSpec) -> CaptureRecord:
    if spec not in COMMANDS or COMMAND_BY_ID.get(spec.capture_id) != spec:
        raise CaptureError("command is not in the immutable allowlist")
    if not spec.argv or spec.argv[0] != "python":
        raise CaptureError("allowlisted command must use the pinned interpreter")
    if "run" in spec.argv or "run_experiment" in spec.argv:
        raise CaptureError("publication run and evaluator calls are forbidden")
    completed = subprocess.run(
        spec.argv,
        cwd=ROOT,
        env=_minimal_environment(),
        executable=sys.executable,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=120,
    )
    stdout = _scrub_output(completed.stdout)
    stderr = _scrub_output(completed.stderr)
    if completed.returncode not in spec.expected_exit_codes:
        raise CaptureError(
            f"{spec.capture_id} returned unexpected exit code {completed.returncode}"
        )
    if stderr:
        raise CaptureError(f"{spec.capture_id} unexpectedly wrote to stderr")
    if not stdout.endswith(b"\n"):
        raise CaptureError(f"{spec.capture_id} output lacks a final newline")
    return CaptureRecord(
        stdout=stdout,
        stderr=stderr,
        exit_code=completed.returncode,
    )


def _escape(value: object) -> str:
    return escape(str(value), {'"': "&quot;"})


def _display_lines(content: bytes, *, width: int = 108) -> tuple[str, ...]:
    text = content.decode("utf-8").removesuffix("\n")
    rendered: list[str] = []
    for line in text.splitlines():
        if len(line) <= width:
            rendered.append(line)
            continue
        chunks = textwrap.wrap(
            line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            drop_whitespace=False,
            replace_whitespace=False,
        )
        rendered.extend(chunks or [""])
    return tuple(rendered)


def render_svg(spec: CommandSpec, record: CaptureRecord) -> bytes:
    """Render one self-contained accessible terminal SVG."""

    transcript_sha256 = _sha256(record.stdout)
    exit_color = GREEN if record.exit_code == 0 else ORANGE
    command = "$ " + " ".join(spec.argv)
    command_lines = tuple(
        textwrap.wrap(
            command,
            width=108,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )
    output_lines = _display_lines(record.stdout)
    all_lines = (*command_lines, "", *output_lines, "", f"[exit {record.exit_code}]")
    top = 176 if spec.host_note is None else 204
    line_height = 22
    height = max(360, top + len(all_lines) * line_height + 82)
    title_id = f"{spec.capture_id}-title"
    description_id = f"{spec.capture_id}-description"
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" '
            f'height="{height}" viewBox="0 0 1280 {height}" role="img" '
            f'aria-labelledby="{title_id} {description_id}" '
            f'data-capture-id="{_escape(spec.capture_id)}">'
        ),
        f'<title id="{title_id}">{_escape(spec.title)}</title>',
        (
            f'<desc id="{description_id}">{_escape(spec.description)} '
            "The canonical plain-text transcript is stored beside this SVG."
            "</desc>"
        ),
        (f'<rect x="0" y="0" width="1280" height="{height}" fill="{BACKGROUND}"/>'),
        (
            f'<rect x="34" y="30" width="1212" height="{height - 60}" rx="16" '
            f'fill="{PANEL}" stroke="{PANEL_EDGE}" stroke-width="2"/>'
        ),
        f'<circle cx="64" cy="60" r="7" fill="{RED}"/>',
        f'<circle cx="88" cy="60" r="7" fill="{ORANGE}"/>',
        f'<circle cx="112" cy="60" r="7" fill="{GREEN}"/>',
        (
            f'<text x="640" y="66" fill="{MUTED}" font-size="15" '
            'font-family="monospace" text-anchor="middle">'
            "GWORKER · REPRODUCIBLE TERMINAL EVIDENCE</text>"
        ),
        (
            f'<text x="62" y="112" fill="{TEXT}" font-size="25" '
            f'font-weight="700" font-family="monospace">{_escape(spec.title)}</text>'
        ),
        (
            f'<rect x="1016" y="88" width="196" height="34" rx="17" '
            f'fill="{BACKGROUND}" stroke="{exit_color}" stroke-width="1"/>'
        ),
        (
            f'<text x="1114" y="111" fill="{exit_color}" font-size="14" '
            'font-weight="700" font-family="monospace" text-anchor="middle">'
            f"REAL OUTPUT · EXIT {record.exit_code}</text>"
        ),
        (
            f'<text x="62" y="144" fill="{SKY}" font-size="14" '
            'font-family="monospace">'
            f"TRANSCRIPT SHA-256 · {_escape(transcript_sha256)}</text>"
        ),
    ]
    if spec.host_note is not None:
        parts.extend(
            [
                (
                    f'<rect x="62" y="160" width="1150" height="28" rx="6" '
                    f'fill="{BACKGROUND}" stroke="{ORANGE}" stroke-width="1"/>'
                ),
                (
                    f'<text x="76" y="180" fill="{ORANGE}" font-size="14" '
                    'font-weight="700" font-family="monospace">'
                    f"{_escape(spec.host_note)}</text>"
                ),
            ]
        )
    y = top
    for index, line in enumerate(all_lines):
        color = TEXT
        weight = "400"
        if index < len(command_lines):
            color = GREEN
            weight = "700"
        elif line == f"[exit {record.exit_code}]":
            color = GREEN if record.exit_code == 0 else ORANGE
            weight = "700"
        parts.append(
            f'<text x="62" y="{y}" fill="{color}" font-size="15" '
            f'font-weight="{weight}" font-family="monospace" '
            f'xml:space="preserve">{_escape(line)}</text>'
        )
        y += line_height
    parts.extend(
        [
            (
                f'<line x1="62" y1="{height - 70}" x2="1212" '
                f'y2="{height - 70}" stroke="{PANEL_EDGE}" stroke-width="1"/>'
            ),
            (
                f'<text x="62" y="{height - 43}" fill="{MUTED}" font-size="13" '
                'font-family="monospace">'
                f"Canonical transcript · docs/visuals/terminal/"
                f"{_escape(spec.transcript_name)}</text>"
            ),
            "</svg>",
            "",
        ]
    )
    return "\n".join(parts).encode("utf-8")


def _source_records(paths: Sequence[str]) -> list[dict[str, object]]:
    records = []
    for relative in sorted(set(paths)):
        path = ROOT / relative
        if not path.is_file():
            raise CaptureError(f"provenance source is missing: {relative}")
        records.append(
            {
                "byte_count": path.stat().st_size,
                "path": relative,
                "sha256": _file_sha256(path),
            }
        )
    return records


def _output_record(path: str, content: bytes) -> dict[str, object]:
    return {
        "byte_count": len(content),
        "path": path,
        "sha256": _sha256(content),
    }


def _manifest(
    records: Mapping[str, CaptureRecord],
    visuals: Mapping[str, bytes],
    *,
    input_commit: str,
) -> dict[str, object]:
    commands = []
    for spec in COMMANDS:
        record = records[spec.capture_id]
        transcript_path = f"docs/visuals/terminal/{spec.transcript_name}"
        visual_path = f"docs/visuals/terminal/{spec.visual_name}"
        commands.append(
            {
                "argv": list(spec.argv),
                "capture_id": spec.capture_id,
                "description": spec.description,
                "exit_code": record.exit_code,
                "expected_exit_codes": list(spec.expected_exit_codes),
                "host_dependent": spec.host_dependent,
                "host_note": spec.host_note,
                "sources": _source_records(spec.source_paths),
                "stderr": {
                    "byte_count": len(record.stderr),
                    "sha256": _sha256(record.stderr),
                },
                "stdout": _output_record(transcript_path, record.stdout),
                "title": spec.title,
                "visual": _output_record(
                    visual_path,
                    visuals[spec.capture_id],
                ),
            }
        )
    tool_path = "scripts/visuals/capture_terminal.py"
    return {
        "capture": {
            "base_input_commit": input_commit,
            "base_input_commit_role": (
                "Git HEAD at record time; per-file hashes bind exact source bytes."
            ),
            "environment": {
                "HOME": ".gworker/terminal-capture-runtime/home",
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": "src",
                "TMPDIR": ".gworker/terminal-capture-runtime/tmp",
                "TZ": "UTC",
            },
            "interpreter": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "normalization": {
                "absolute_paths": "<REPO> or <RUNTIME>",
                "ansi_controls": "rejected",
                "hostname": "<HOST>",
                "runtime_timings": "not emitted by allowlisted commands",
                "secret_signatures": "<REDACTED>",
                "semantic_numbers": "preserved verbatim",
            },
            "working_directory": ".",
        },
        "commands": commands,
        "reproduction": {
            "check": CHECK_COMMAND,
            "record": RECORD_COMMAND,
            "render_without_execution": RENDER_COMMAND,
        },
        "safety": {
            "arbitrary_commands_accepted": False,
            "evaluator_executed": False,
            "journal_workspace": JOURNAL_WORKSPACE,
            "publication_run_executed": False,
            "shell_used": False,
            "synthetic_demo_data_only": True,
        },
        "schema_version": SCHEMA_VERSION,
        "tool": {
            "name": TOOL_NAME,
            "source": {
                "byte_count": (ROOT / tool_path).stat().st_size,
                "path": tool_path,
                "sha256": _file_sha256(ROOT / tool_path),
            },
            "stdlib_only": True,
            "version": TOOL_VERSION,
        },
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def record_bundle(output_root: Path = TERMINAL_ROOT) -> dict[str, object]:
    """Execute only the fixed allowlist, then atomically write the bundle."""

    if output_root != TERMINAL_ROOT:
        raise CaptureError("CLI recording target is fixed to the terminal bundle")
    records: dict[str, CaptureRecord] = {}
    for spec in COMMANDS:
        records[spec.capture_id] = _run_allowlisted(spec)
    visuals = {
        spec.capture_id: render_svg(spec, records[spec.capture_id]) for spec in COMMANDS
    }
    manifest = _manifest(records, visuals, input_commit=_head_commit())
    for spec in COMMANDS:
        _atomic_write(
            output_root / spec.transcript_name,
            records[spec.capture_id].stdout,
        )
        _atomic_write(output_root / spec.visual_name, visuals[spec.capture_id])
    _atomic_write(output_root / MANIFEST_NAME, _canonical_json(manifest))
    return manifest


def _load_manifest(output_root: Path = TERMINAL_ROOT) -> dict[str, object]:
    try:
        raw = (output_root / MANIFEST_NAME).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(
            "terminal capture manifest is unavailable or invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise CaptureError("terminal capture manifest must be a JSON object")
    return payload


def _manifest_commands(
    manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    raw_commands = manifest.get("commands")
    if not isinstance(raw_commands, list):
        raise CaptureError("manifest commands must be a list")
    commands: dict[str, dict[str, object]] = {}
    for item in raw_commands:
        if not isinstance(item, dict) or not isinstance(item.get("capture_id"), str):
            raise CaptureError("manifest command record is malformed")
        capture_id = item["capture_id"]
        if capture_id in commands:
            raise CaptureError("manifest capture identifiers must be unique")
        commands[capture_id] = item
    return commands


def _record_from_manifest(
    spec: CommandSpec,
    record: Mapping[str, object],
    output_root: Path,
) -> CaptureRecord:
    stdout_record = record.get("stdout")
    stderr_record = record.get("stderr")
    exit_code = record.get("exit_code")
    if (
        not isinstance(stdout_record, dict)
        or not isinstance(stderr_record, dict)
        or type(exit_code) is not int
    ):
        raise CaptureError(f"{spec.capture_id} manifest record is malformed")
    try:
        stdout = (output_root / spec.transcript_name).read_bytes()
    except OSError as exc:
        raise CaptureError(f"{spec.capture_id} transcript is unavailable") from exc
    if stderr_record != {"byte_count": 0, "sha256": EMPTY_SHA256}:
        raise CaptureError(f"{spec.capture_id} recorded stderr must be empty")
    return CaptureRecord(stdout=stdout, stderr=b"", exit_code=exit_code)


def render_bundle(output_root: Path = TERMINAL_ROOT) -> dict[str, object]:
    """Rebuild only derived SVG and manifest bytes; execute no commands."""

    manifest = _load_manifest(output_root)
    commands = _manifest_commands(manifest)
    if set(commands) != set(COMMAND_BY_ID):
        raise CaptureError("manifest command set does not match the allowlist")
    for spec in COMMANDS:
        existing = commands[spec.capture_id]
        record = _record_from_manifest(spec, existing, output_root)
        visual = render_svg(spec, record)
        _atomic_write(output_root / spec.visual_name, visual)
        existing["visual"] = _output_record(
            f"docs/visuals/terminal/{spec.visual_name}",
            visual,
        )
    _atomic_write(output_root / MANIFEST_NAME, _canonical_json(manifest))
    return manifest


def _validate_source_record(
    record: object,
    *,
    differences: list[str],
    label: str,
) -> None:
    if not isinstance(record, dict):
        differences.append(f"{label}: malformed source record")
        return
    path = record.get("path")
    expected_sha256 = record.get("sha256")
    expected_size = record.get("byte_count")
    if (
        not isinstance(path, str)
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or not isinstance(expected_sha256, str)
        or type(expected_size) is not int
    ):
        differences.append(f"{label}: unsafe source record")
        return
    source = ROOT / path
    try:
        content = source.read_bytes()
    except OSError:
        differences.append(f"{label}: source is missing")
        return
    if _sha256(content) != expected_sha256 or len(content) != expected_size:
        differences.append(f"{label}: source bytes differ from provenance")


def _validate_svg(content: bytes, spec: CommandSpec) -> tuple[str, ...]:
    differences: list[str] = []
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return ("SVG is not well-formed XML",)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    title = root.find("svg:title", namespace)
    description = root.find("svg:desc", namespace)
    if root.attrib.get("role") != "img":
        differences.append("SVG lacks role=img")
    labelled_by = root.attrib.get("aria-labelledby", "").split()
    if len(labelled_by) != 2:
        differences.append("SVG lacks title/description aria-labelledby")
    if title is None or not title.text or title.attrib.get("id") not in labelled_by:
        differences.append("SVG lacks an accessible title")
    if (
        description is None
        or not description.text
        or description.attrib.get("id") not in labelled_by
    ):
        differences.append("SVG lacks an accessible description")
    text = content.decode("utf-8", errors="replace")
    if spec.title not in text:
        differences.append("SVG lacks its direct title label")
    for forbidden in ("<script", "<image", " href=", "@import", "url("):
        if forbidden in text:
            differences.append(f"SVG contains forbidden external surface: {forbidden}")
    user_content = "\n".join(
        value
        for element in root.iter()
        for value in (
            element.text or "",
            element.tail or "",
            *element.attrib.values(),
        )
    )
    if SECRET_PATTERN.search(user_content) or ABSOLUTE_PATH.search(user_content):
        differences.append("SVG contains an unsanitized path or secret")
    return tuple(differences)


def check_bundle(output_root: Path = TERMINAL_ROOT) -> tuple[str, ...]:
    """Verify committed bytes and provenance without starting subprocesses."""

    differences: list[str] = []
    try:
        manifest = _load_manifest(output_root)
        commands = _manifest_commands(manifest)
    except CaptureError as exc:
        return (str(exc),)
    expected_names = {
        MANIFEST_NAME,
        *(spec.transcript_name for spec in COMMANDS),
        *(spec.visual_name for spec in COMMANDS),
    }
    try:
        actual_names = {path.name for path in output_root.iterdir()}
    except OSError:
        return ("terminal capture directory is unavailable",)
    for name in sorted(actual_names - expected_names):
        differences.append(f"unexpected terminal bundle entry: {name}")
    for name in sorted(expected_names - actual_names):
        differences.append(f"missing terminal bundle entry: {name}")
    manifest_path = output_root / MANIFEST_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError:
        return ("terminal capture manifest is missing",)
    if _canonical_json(manifest) != manifest_bytes:
        differences.append("manifest JSON is not canonical")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        differences.append("manifest schema version differs")
    if list(commands) != [spec.capture_id for spec in COMMANDS]:
        differences.append("manifest command order or set differs")
    capture = manifest.get("capture")
    if not isinstance(capture, dict) or not HEX_COMMIT.fullmatch(
        str(capture.get("base_input_commit", ""))
    ):
        differences.append("manifest base input commit is malformed")
    safety = manifest.get("safety")
    expected_safety = {
        "arbitrary_commands_accepted": False,
        "evaluator_executed": False,
        "journal_workspace": JOURNAL_WORKSPACE,
        "publication_run_executed": False,
        "shell_used": False,
        "synthetic_demo_data_only": True,
    }
    if safety != expected_safety:
        differences.append("manifest safety declaration differs")
    tool = manifest.get("tool")
    if not isinstance(tool, dict):
        differences.append("manifest tool record is malformed")
    else:
        if tool.get("name") != TOOL_NAME or tool.get("version") != TOOL_VERSION:
            differences.append("manifest tool identity differs")
        _validate_source_record(
            tool.get("source"),
            differences=differences,
            label="capture tool",
        )
    if set(commands) != set(COMMAND_BY_ID):
        differences.append("manifest command set differs from allowlist")
        return tuple(differences)
    for spec in COMMANDS:
        raw = commands[spec.capture_id]
        prefix = spec.capture_id
        if raw.get("argv") != list(spec.argv):
            differences.append(f"{prefix}: argv differs from allowlist")
        if raw.get("expected_exit_codes") != list(spec.expected_exit_codes):
            differences.append(f"{prefix}: expected exit codes differ")
        if raw.get("host_dependent") is not spec.host_dependent:
            differences.append(f"{prefix}: host-dependency flag differs")
        if raw.get("host_note") != spec.host_note:
            differences.append(f"{prefix}: host note differs")
        exit_code = raw.get("exit_code")
        if type(exit_code) is not int or exit_code not in spec.expected_exit_codes:
            differences.append(f"{prefix}: captured exit code is invalid")
            continue
        sources = raw.get("sources")
        if not isinstance(sources, list):
            differences.append(f"{prefix}: source records are malformed")
        else:
            expected_paths = sorted(set(spec.source_paths))
            actual_paths = [
                item.get("path") for item in sources if isinstance(item, dict)
            ]
            if actual_paths != expected_paths:
                differences.append(f"{prefix}: source path set differs")
            for index, source in enumerate(sources):
                _validate_source_record(
                    source,
                    differences=differences,
                    label=f"{prefix} source {index}",
                )
        try:
            record = _record_from_manifest(spec, raw, output_root)
        except CaptureError as exc:
            differences.append(str(exc))
            continue
        transcript_record = raw.get("stdout")
        expected_transcript = _output_record(
            f"docs/visuals/terminal/{spec.transcript_name}",
            record.stdout,
        )
        if transcript_record != expected_transcript:
            differences.append(f"{prefix}: transcript hash or size differs")
        try:
            sanitized = _scrub_output(record.stdout)
        except CaptureError as exc:
            differences.append(f"{prefix}: {exc}")
        else:
            if sanitized != record.stdout:
                differences.append(f"{prefix}: transcript is not fully sanitized")
        expected_visual = render_svg(spec, record)
        try:
            actual_visual = (output_root / spec.visual_name).read_bytes()
        except OSError:
            differences.append(f"{prefix}: SVG is missing")
            continue
        visual_record = raw.get("visual")
        if visual_record != _output_record(
            f"docs/visuals/terminal/{spec.visual_name}",
            actual_visual,
        ):
            differences.append(f"{prefix}: SVG hash or size differs")
        if actual_visual != expected_visual:
            differences.append(f"{prefix}: SVG differs from deterministic render")
        differences.extend(
            f"{prefix}: {difference}"
            for difference in _validate_svg(actual_visual, spec)
        )
    return tuple(differences)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record, render, or verify GWorker's fixed terminal evidence bundle."
        )
    )
    parser.add_argument(
        "action",
        choices=("record", "render", "check"),
        help=("record runs the fixed allowlist; render and check never start commands"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "record":
            record_bundle()
            print(f"recorded {len(COMMANDS)} genuine terminal captures")
            return 0
        if arguments.action == "render":
            render_bundle()
            print(f"rendered {len(COMMANDS)} terminal SVGs")
            return 0
        differences = check_bundle()
        if differences:
            for difference in differences:
                print(f"DIFF: {difference}", file=sys.stderr)
            return 1
        print(f"verified {len(COMMANDS)} terminal captures without command execution")
        return 0
    except CaptureError as exc:
        print(f"capture error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
