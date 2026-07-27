#!/usr/bin/env python3
"""Deterministic recommend-review-reopen demo through the public CLI handler."""

from __future__ import annotations

import argparse
import io
import json
import stat
import sys
import tempfile
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import NoReturn, TextIO, TypedDict, cast

from gworker.cli import run

SCHEMA_VERSION = "gworker-policy-journal-demo-v1"
FIRST_DECISION_ID = "018f4f69-e7a2-7f84-8c2d-9f531c4e9101"
SECOND_DECISION_ID = "018f4f69-e7a2-7f84-8c2d-9f531c4e9102"


class DemoInvariantError(RuntimeError):
    """Raised when the real CLI/storage workflow contradicts the fixture."""


class _ArgumentParser(argparse.ArgumentParser):
    """Reject arguments without repeating their untrusted bytes."""

    def error(self, message: str) -> NoReturn:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid arguments\n")


class DemoDocument(TypedDict):
    first_propensity_hex: str
    first_template_id: str
    fixture_kind: str
    journal_mode: str
    policy_id: str
    review_propensity_hex: str
    schema_version: str
    second_evidence_count: int
    second_propensity_hex: str
    second_template_id: str
    temporary_workspace_removed: bool
    verification_decision_count: int
    verification_history_edge_count: int
    verification_review_count: int
    verification_sqlite_check: str
    workspace_mode: str


def _json_command(journal: Path, *arguments: str) -> dict[str, object]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = run(
        ("--journal", str(journal), "--json", *arguments),
        stdout=stdout,
        stderr=stderr,
    )
    if exit_code != 0 or stderr.getvalue():
        raise DemoInvariantError("a fixed CLI command failed")
    if str(journal.parent) in stdout.getvalue():
        raise DemoInvariantError("CLI output disclosed the temporary workspace")
    try:
        payload = json.loads(stdout.getvalue())
    except json.JSONDecodeError as error:
        raise DemoInvariantError("CLI output was not valid JSON") from error
    if not isinstance(payload, dict):
        raise DemoInvariantError("CLI output was not a JSON object")
    return cast(dict[str, object], payload)


def _integer(document: dict[str, object], field: str) -> int:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DemoInvariantError(f"{field} was not an integer")
    return value


def _text(document: dict[str, object], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str):
        raise DemoInvariantError(f"{field} was not text")
    return value


def build_demo() -> DemoDocument:
    """Exercise four CLI calls against one disposable private journal."""

    workspace: Path | None = None
    with tempfile.TemporaryDirectory(prefix="gworker-policy-journal-") as temporary:
        workspace = Path(temporary)
        journal = workspace / "events.sqlite3"
        first = _json_command(
            journal,
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
        )
        review = _json_command(
            journal,
            "review",
            FIRST_DECISION_ID,
            "--fit",
            "just_right",
            "--completed",
        )
        first_focus_minutes = _integer(first, "focus_minutes")
        second = _json_command(
            journal,
            "recommend",
            "--task-kind",
            "deep_work",
            "--energy",
            "medium",
            "--available-minutes",
            "60",
            "--previous-focus-minutes",
            str(first_focus_minutes),
            "--decision-id",
            SECOND_DECISION_ID,
            "--seed",
            "20260726",
        )
        verification = _json_command(journal, "verify")

        first_hex = _text(first, "propensity_hex")
        review_hex = _text(review, "propensity_hex")
        if first_hex != review_hex:
            raise DemoInvariantError("review lost the recommendation propensity")
        if (
            _integer(first, "decision_sequence") != 1
            or _integer(second, "decision_sequence") != 2
            or _integer(second, "evidence_count") != 1
        ):
            raise DemoInvariantError("durable decision ordering drifted")
        expected_counts = (2, 1, 1)
        observed_counts = (
            _integer(verification, "policy_decision_count"),
            _integer(verification, "policy_review_count"),
            _integer(verification, "policy_history_edge_count"),
        )
        if observed_counts != expected_counts:
            raise DemoInvariantError("policy lineage counts drifted")
        if _text(verification, "sqlite_check") != "ok":
            raise DemoInvariantError("SQLite verification did not pass")
        policy_id = _text(first, "policy_id")
        if policy_id != _text(review, "policy_id") or policy_id != _text(
            verification,
            "policy_id",
        ):
            raise DemoInvariantError("policy identity drifted across reopen")

        document: DemoDocument = {
            "first_propensity_hex": first_hex,
            "first_template_id": _text(first, "template_id"),
            "fixture_kind": "deterministic-synthetic-cli-storage-demo",
            "journal_mode": f"{stat.S_IMODE(journal.stat().st_mode):04o}",
            "policy_id": policy_id,
            "review_propensity_hex": review_hex,
            "schema_version": SCHEMA_VERSION,
            "second_evidence_count": _integer(second, "evidence_count"),
            "second_propensity_hex": _text(second, "propensity_hex"),
            "second_template_id": _text(second, "template_id"),
            "temporary_workspace_removed": False,
            "verification_decision_count": observed_counts[0],
            "verification_history_edge_count": observed_counts[2],
            "verification_review_count": observed_counts[1],
            "verification_sqlite_check": _text(verification, "sqlite_check"),
            "workspace_mode": f"{stat.S_IMODE(workspace.stat().st_mode):04o}",
        }

    if workspace is None or workspace.exists():
        raise DemoInvariantError("temporary workspace was not removed")
    document["temporary_workspace_removed"] = True
    return document


def render_human(document: DemoDocument) -> str:
    """Render the categorical facts produced by the fixed real workflow."""

    return "\n".join(
        (
            "GWorker durable policy journal | deterministic synthetic workflow",
            f"Policy: {document['policy_id']}",
            (
                "D1 recommend | sequence 1 | "
                f"{document['first_template_id']} | "
                f"exact p={document['first_propensity_hex']}"
            ),
            (
                "D1 review    | fit=just_right | completed=true | "
                "exact propensity preserved"
            ),
            (
                "D2 recommend | sequence 2 | "
                f"{document['second_template_id']} | "
                f"history evidence={document['second_evidence_count']} | "
                f"exact p={document['second_propensity_hex']}"
            ),
            (
                "Verify       | decisions/reviews/history edges = "
                f"{document['verification_decision_count']}/"
                f"{document['verification_review_count']}/"
                f"{document['verification_history_edge_count']} | "
                f"SQLite={document['verification_sqlite_check']}"
            ),
            (
                "Storage      | workspace "
                f"{document['workspace_mode']} | journal "
                f"{document['journal_mode']} | path omitted"
            ),
            "Reopen        | every CLI call reopened the same disposable journal",
            "Cleanup       | temporary workspace removed",
            (
                "Scope         | synthetic CLI/storage evidence only; "
                "not a human outcome or locked evaluation"
            ),
        )
    )


def _parser() -> _ArgumentParser:
    return _ArgumentParser(
        prog="gworker-policy-journal-demo",
        description="Run GWorker's fixed durable policy-journal demonstration.",
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    errors = stderr if stderr is not None else sys.stderr
    try:
        with redirect_stdout(output), redirect_stderr(errors):
            _parser().parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 2
    try:
        document = build_demo()
    except (DemoInvariantError, OSError):
        print("gworker demo: workflow verification failed", file=errors)
        return 1
    print(render_human(document), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
