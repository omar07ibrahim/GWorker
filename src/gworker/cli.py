"""Thin command-line workflow for the durable GWorker policy journal."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Literal, NoReturn, TextIO, TypeAlias, TypedDict
from uuid import UUID, uuid4

from .policy import (
    DEFAULT_TEMPLATES,
    MAX_AVAILABLE_SECONDS,
    MAX_RNG_SEED,
    DurationFit,
    EnergyLevel,
    FocusContext,
    HierarchicalSoftmaxUCB,
    PolicyInputError,
    TaskKind,
)
from .storage import (
    CorruptJournal,
    JournalConflict,
    JournalError,
    JournalSecurityError,
    SQLiteEventStore,
    default_journal_path,
)


class _RecommendationDocument(TypedDict):
    decision_id: str
    decision_sequence: int
    evidence_bucket: str
    evidence_count: int
    focus_minutes: int
    break_minutes: int
    journal_path_disclosed: bool
    policy_id: str
    propensity: float
    propensity_hex: str
    reason_codes: list[str]
    rng_seed: int
    schema_version: Literal["gworker-cli-recommendation-v1"]
    template_id: str


class _ReviewDocument(TypedDict):
    decision_id: str
    decision_sequence: int
    fit: str
    journal_path_disclosed: bool
    objective_completed: bool
    policy_id: str
    propensity: float
    propensity_hex: str
    reward: float
    schema_version: Literal["gworker-cli-review-v1"]
    template_id: str


class _VerificationDocument(TypedDict):
    event_count: int
    journal_path_disclosed: bool
    policy_decision_count: int
    policy_history_edge_count: int
    policy_id: str
    policy_review_count: int
    schema_version: Literal["gworker-cli-verification-v1"]
    session_count: int
    sqlite_check: str


_Document: TypeAlias = _RecommendationDocument | _ReviewDocument | _VerificationDocument


class _ArgumentParser(argparse.ArgumentParser):
    """Argparse surface that never repeats untrusted argument bytes."""

    def error(self, message: str) -> NoReturn:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid arguments\n")


def _canonical_uuid(value: str) -> UUID:
    try:
        identifier = UUID(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a UUID") from error
    if identifier.int == 0 or str(identifier) != value:
        raise argparse.ArgumentTypeError("must be a canonical lowercase non-nil UUID")
    return identifier


def _bounded_seed(value: str) -> int:
    try:
        seed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a decimal integer") from error
    if str(seed) != value or not 0 <= seed <= MAX_RNG_SEED:
        raise argparse.ArgumentTypeError(
            f"must be a canonical integer between 0 and {MAX_RNG_SEED}"
        )
    return seed


def _positive_minutes(value: str) -> int:
    try:
        minutes = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a decimal integer") from error
    maximum = MAX_AVAILABLE_SECONDS // 60
    if str(minutes) != value or not 1 <= minutes <= maximum:
        raise argparse.ArgumentTypeError(
            f"must be a canonical integer between 1 and {maximum}"
        )
    return minutes


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(
        prog="gworker",
        description="Record and replay explainable focus-duration decisions.",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=None,
        help="private absolute SQLite journal path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit canonical reader-facing JSON",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    recommend = commands.add_parser(
        "recommend",
        help="record one seeded recommendation",
    )
    recommend.add_argument(
        "--task-kind",
        choices=[item.value for item in TaskKind],
        required=True,
    )
    recommend.add_argument(
        "--energy",
        choices=[item.value for item in EnergyLevel],
        required=True,
    )
    recommend.add_argument(
        "--available-minutes",
        type=_positive_minutes,
        required=True,
    )
    recommend.add_argument(
        "--previous-focus-minutes",
        type=_positive_minutes,
        choices=[template.focus_seconds // 60 for template in DEFAULT_TEMPLATES],
    )
    recommend.add_argument(
        "--decision-id",
        type=_canonical_uuid,
    )
    recommend.add_argument(
        "--seed",
        type=_bounded_seed,
    )

    review = commands.add_parser(
        "review",
        help="attach explicit feedback to a stored recommendation",
    )
    review.add_argument("decision_id", type=_canonical_uuid)
    review.add_argument(
        "--fit",
        choices=[item.value for item in DurationFit],
        required=True,
    )
    completion = review.add_mutually_exclusive_group(required=True)
    completion.add_argument(
        "--completed",
        action="store_true",
        dest="objective_completed",
    )
    completion.add_argument(
        "--not-completed",
        action="store_false",
        dest="objective_completed",
    )

    commands.add_parser(
        "verify",
        help="replay every event and the current policy fingerprint",
    )
    return parser


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, JournalSecurityError):
        return "journal security check failed"
    if isinstance(error, CorruptJournal):
        return "journal is corrupt or incompatible"
    if isinstance(error, JournalConflict):
        return "policy journal conflict"
    if isinstance(error, JournalError):
        return "journal operation failed"
    return "policy input is invalid"


def _recommend(
    arguments: argparse.Namespace,
    store: SQLiteEventStore,
) -> _RecommendationDocument:
    policy = HierarchicalSoftmaxUCB()
    decision_id = arguments.decision_id or uuid4()
    rng_seed = (
        arguments.seed
        if arguments.seed is not None
        else secrets.randbelow(MAX_RNG_SEED + 1)
    )
    previous = arguments.previous_focus_minutes
    context = FocusContext(
        task_kind=TaskKind(arguments.task_kind),
        energy=EnergyLevel(arguments.energy),
        available_seconds=arguments.available_minutes * 60,
        previous_focus_seconds=None if previous is None else previous * 60,
    )
    recommendation = store.recommend(
        policy,
        context,
        decision_id=decision_id,
        rng_seed=rng_seed,
    )
    return {
        "decision_id": str(recommendation.decision_id),
        "decision_sequence": recommendation.decision_sequence,
        "evidence_bucket": recommendation.bucket.value,
        "evidence_count": recommendation.evidence_count,
        "focus_minutes": recommendation.template.focus_seconds // 60,
        "break_minutes": recommendation.template.break_seconds // 60,
        "journal_path_disclosed": False,
        "policy_id": recommendation.policy_id,
        "propensity": recommendation.propensity,
        "propensity_hex": recommendation.propensity.hex(),
        "reason_codes": list(recommendation.reason_codes),
        "rng_seed": rng_seed,
        "schema_version": "gworker-cli-recommendation-v1",
        "template_id": recommendation.template.template_id,
    }


def _review(
    arguments: argparse.Namespace,
    store: SQLiteEventStore,
) -> _ReviewDocument:
    policy = HierarchicalSoftmaxUCB()
    review = store.record_review(
        policy,
        arguments.decision_id,
        fit=DurationFit(arguments.fit),
        objective_completed=arguments.objective_completed,
    )
    return {
        "decision_id": str(review.decision_id),
        "decision_sequence": review.decision_sequence,
        "fit": review.fit.value,
        "journal_path_disclosed": False,
        "objective_completed": review.objective_completed,
        "policy_id": review.policy_id,
        "propensity": review.propensity,
        "propensity_hex": review.propensity.hex(),
        "reward": review.reward,
        "schema_version": "gworker-cli-review-v1",
        "template_id": review.template_id,
    }


def _verify(store: SQLiteEventStore) -> _VerificationDocument:
    journal = store.verify()
    policy = store.verify_policy_history(HierarchicalSoftmaxUCB())
    return {
        "event_count": journal.event_count,
        "journal_path_disclosed": False,
        "policy_decision_count": policy.decision_count,
        "policy_history_edge_count": policy.history_edge_count,
        "policy_id": policy.policy_id,
        "policy_review_count": policy.review_count,
        "schema_version": "gworker-cli-verification-v1",
        "session_count": journal.session_count,
        "sqlite_check": journal.sqlite_check,
    }


def _render_recommendation(document: _RecommendationDocument) -> str:
    return "\n".join(
        (
            "GWorker recommendation recorded",
            (
                f"Decision: {document['decision_id']} "
                f"(sequence {document['decision_sequence']})"
            ),
            (
                f"Template: {document['template_id']} | "
                f"{document['focus_minutes']}m focus + "
                f"{document['break_minutes']}m break"
            ),
            (
                f"Propensity: {document['propensity']:.12f} | "
                f"exact {document['propensity_hex']}"
            ),
            (
                f"Evidence: {document['evidence_bucket']} "
                f"({document['evidence_count']} reviewed decisions)"
            ),
            f"Reasons: {', '.join(document['reason_codes'])}",
            f"Replay seed: {document['rng_seed']}",
            "Journal: private local file (path omitted)",
        )
    )


def _render_review(document: _ReviewDocument) -> str:
    return "\n".join(
        (
            "GWorker review recorded",
            (
                f"Decision: {document['decision_id']} "
                f"(sequence {document['decision_sequence']})"
            ),
            (
                f"Outcome: fit={document['fit']} | "
                f"completed={str(document['objective_completed']).lower()} | "
                f"reward={document['reward']:.3f}"
            ),
            (
                f"Provenance: {document['template_id']} at "
                f"p={document['propensity']:.12f}"
            ),
            f"Exact propensity: {document['propensity_hex']}",
            "Journal: private local file (path omitted)",
        )
    )


def _render_verification(document: _VerificationDocument) -> str:
    return "\n".join(
        (
            "GWorker journal verified",
            (
                f"Sessions/events: {document['session_count']} / "
                f"{document['event_count']}"
            ),
            (
                "Policy decisions/reviews/history edges: "
                f"{document['policy_decision_count']} / "
                f"{document['policy_review_count']} / "
                f"{document['policy_history_edge_count']}"
            ),
            f"Policy: {document['policy_id']}",
            f"SQLite quick_check: {document['sqlite_check']}",
            "Replay: every decision for this policy matched",
            "Journal: private local file (path omitted)",
        )
    )


def run(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one CLI command and return a process-style exit code."""

    output = stdout if stdout is not None else sys.stdout
    errors = stderr if stderr is not None else sys.stderr
    try:
        with redirect_stdout(output), redirect_stderr(errors):
            arguments = _parser().parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 2

    try:
        journal = arguments.journal or default_journal_path()
        store = SQLiteEventStore(journal)
        document: _Document
        if arguments.command == "recommend":
            document = _recommend(arguments, store)
            human = _render_recommendation(document)
        elif arguments.command == "review":
            document = _review(arguments, store)
            human = _render_review(document)
        else:
            document = _verify(store)
            human = _render_verification(document)
    except (JournalError, PolicyInputError, ValueError) as error:
        print(f"gworker: {_safe_error_message(error)}", file=errors)
        return 1

    if arguments.json:
        print(
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=output,
        )
    else:
        print(human, file=output)
    return 0


def main() -> None:
    """Console-script entry point."""

    raise SystemExit(run())


if __name__ == "__main__":
    main()
