"""Generate deterministic SVG evidence from the implemented GWorker APIs.

The eight visuals in this bundle are deliberately non-result evidence.  They
exercise the event reducer, durable policy lineage, policy guardrails,
publication state, and locked protocol inventory without entering the held-out
evaluation namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final
from uuid import NAMESPACE_URL, UUID, uuid5
from xml.sax.saxutils import escape

from gworker.domain import (
    AbandonReason,
    BreakCompleted,
    BreakStarted,
    DomainEvent,
    FocusCompleted,
    FocusStarted,
    InterruptionKind,
    InterruptionRecorded,
    SessionAbandoned,
    SessionPhase,
    SessionPlanned,
    SessionState,
    apply_event,
)
from gworker.evaluation import (
    DEFAULT_EXPERIMENT_CONFIG,
    LOCKED_EVALUATION_RUN_KEY,
    Strategy,
)
from gworker.evidence import (
    EvidenceCardinalities,
    expected_publication_cardinalities,
)
from gworker.policy import (
    DEFAULT_TEMPLATES,
    DurationFit,
    EnergyLevel,
    EvidenceBucket,
    FocusContext,
    HierarchicalSoftmaxUCB,
    Recommendation,
    ReviewedDecision,
    TaskKind,
)
from gworker.publication_state import PublicationStage
from gworker.storage import (
    SCHEMA_VERSION,
    FocusSessionLink,
    JournalVerification,
    PolicyJournalVerification,
    SQLiteEventStore,
)

ROOT: Final = Path(__file__).resolve().parents[2]
VISUAL_ROOT: Final = ROOT / "docs" / "visuals"
GENERATED_DIRECTORY_NAME: Final = "generated"
MANIFEST_NAME: Final = "manifest.json"
TOOL_NAME: Final = "gworker-visual-evidence"
TOOL_VERSION: Final = "4"
GENERATION_COMMAND: Final = "PYTHONPATH=src python3 scripts/visuals/generate.py"
VALIDATED_PYTHON_MINORS: Final = ("3.11", "3.12")
DURABLE_FIRST_DECISION_ID: Final = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e9101")
DURABLE_SECOND_DECISION_ID: Final = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e9102")
DURABLE_FIRST_SEED: Final = 20_260_725
DURABLE_SECOND_SEED: Final = 20_260_726
LINKAGE_DECISION_ID: Final = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e9201")
LINKAGE_SESSION_ID: Final = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e9202")
LINKAGE_PLANNED_EVENT_ID: Final = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e9203")
LINKAGE_STARTED_EVENT_ID: Final = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e9204")
LINKAGE_ABANDONED_EVENT_ID: Final = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e9205")
LINKAGE_SEED: Final = 20_260_726
LINKAGE_BASE_TIME: Final = datetime(2026, 7, 26, 10, 0, tzinfo=UTC)

# The Okabe-Ito palette remains distinguishable for common color-vision
# deficiencies. Every encoded value also has a direct text label.
BLACK: Final = "#1B1F23"
GRAY: Final = "#667085"
LIGHT_GRAY: Final = "#E4E7EC"
PANEL: Final = "#F8FAFC"
WHITE: Final = "#FFFFFF"
ORANGE: Final = "#E69F00"
SKY: Final = "#56B4E9"
GREEN: Final = "#009E73"
YELLOW: Final = "#F0E442"
BLUE: Final = "#0072B2"
VERMILION: Final = "#D55E00"
PURPLE: Final = "#CC79A7"

INPUT_FILES: Final = (
    "README.md",
    "docs/architecture.md",
    "docs/decision-lineage.md",
    "docs/evaluation-protocol.md",
    "docs/publication-evidence.md",
    "docs/session-linkage.md",
    "pyproject.toml",
    "scripts/demo_policy_journal.py",
    "scripts/visuals/generate.py",
    "src/gworker/__init__.py",
    "src/gworker/cli.py",
    "src/gworker/codec.py",
    "src/gworker/domain.py",
    "src/gworker/evaluation.py",
    "src/gworker/evidence.py",
    "src/gworker/policy.py",
    "src/gworker/publication_codec.py",
    "src/gworker/publication_runner.py",
    "src/gworker/publication_state.py",
    "src/gworker/reporting.py",
    "src/gworker/resource_preflight.py",
    "src/gworker/result_codec.py",
    "src/gworker/storage.py",
)


@dataclass(frozen=True, slots=True)
class RenderedVisual:
    """One complete deterministic SVG file."""

    filename: str
    title: str
    description: str
    content: bytes


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """One event and the projection produced by applying it."""

    sequence: int
    event_kind: str
    occurred_at: str
    phase: str
    detail: str


@dataclass(frozen=True, slots=True)
class PolicyScenario:
    """The fixed public-API policy walkthrough used by the visual."""

    reviewed_history: tuple[ReviewedDecision, ...]
    recommendation: Recommendation
    choices_minutes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DurableDecisionLineage:
    """One fixed recommend-review-reopen lineage from the public store."""

    first: Recommendation
    review: ReviewedDecision
    second: Recommendation
    verification: PolicyJournalVerification


@dataclass(frozen=True, slots=True)
class FocusSessionLinkageEvidence:
    """One fixed decision-plan-link-progress-review-reopen workflow."""

    recommendation: Recommendation
    planned: SessionPlanned
    link: FocusSessionLink
    before_review: PolicyJournalVerification
    review: ReviewedDecision
    reopened_link: FocusSessionLink
    state: SessionState
    journal_verification: JournalVerification
    policy_verification: PolicyJournalVerification
    schema_version: int
    link_columns: tuple[str, str]


@dataclass(frozen=True, slots=True)
class GuardrailRow:
    """One real feasible-template query and its directly labelled cells."""

    label: str
    context: FocusContext
    allowed_template_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    cells: tuple[str, ...]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _safe_text(value: object) -> str:
    return escape(str(value), {'"': "&quot;"})


def _short_uuid(value: UUID) -> str:
    canonical = str(value)
    return f"{canonical[:6]}…{canonical[-4:]}"


def _text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 16,
    weight: int = 400,
    fill: str = BLACK,
    anchor: str = "start",
    family: str | None = None,
) -> str:
    family_attribute = "" if family is None else f' font-family="{_safe_text(family)}"'
    return (
        f'<text x="{x:g}" y="{y:g}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"'
        f"{family_attribute}>{_safe_text(value)}</text>"
    )


def _multiline(
    x: float,
    y: float,
    lines: tuple[str, ...],
    *,
    size: int = 16,
    weight: int = 400,
    fill: str = BLACK,
    line_height: int = 22,
    anchor: str = "start",
) -> str:
    spans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else line_height
        spans.append(f'<tspan x="{x:g}" dy="{dy:g}">{_safe_text(line)}</tspan>')
    return (
        f'<text x="{x:g}" y="{y:g}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        + "".join(spans)
        + "</text>"
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = WHITE,
    stroke: str = LIGHT_GRAY,
    stroke_width: float = 1,
    radius: float = 12,
    dash: str | None = None,
) -> str:
    dash_attribute = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" '
        f'height="{height:g}" rx="{radius:g}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:g}"'
        f"{dash_attribute}/>"
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = GRAY,
    width: float = 2,
    arrow: bool = False,
    dash: str | None = None,
) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attribute = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'stroke="{stroke}" stroke-width="{width:g}"{marker}'
        f"{dash_attribute}/>"
    )


def _circle(
    cx: float,
    cy: float,
    radius: float,
    *,
    fill: str,
    stroke: str = WHITE,
    stroke_width: float = 3,
) -> str:
    return (
        f'<circle cx="{cx:g}" cy="{cy:g}" r="{radius:g}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:g}"/>'
    )


def _svg_document(
    *,
    stem: str,
    title: str,
    description: str,
    width: int,
    height: int,
    body: list[str],
) -> bytes:
    title_id = f"{stem}-title"
    description_id = f"{stem}-description"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="{title_id} {description_id}">'
        ),
        f'  <title id="{title_id}">{_safe_text(title)}</title>',
        (f'  <desc id="{description_id}">{_safe_text(description)}</desc>'),
        "  <defs>",
        (
            '    <marker id="arrow" markerWidth="10" markerHeight="10" '
            'refX="8" refY="3" orient="auto" markerUnits="strokeWidth">'
        ),
        f'      <path d="M0,0 L0,6 L9,3 z" fill="{GRAY}"/>',
        "    </marker>",
        "  </defs>",
        f'  <rect width="{width}" height="{height}" fill="{WHITE}"/>',
        *[f"  {item}" for item in body],
        "</svg>",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_event_replay() -> tuple[tuple[ReplayStep, ...], SessionState]:
    """Apply a fixed synthetic event stream through the public reducer."""

    session_id = uuid5(NAMESPACE_URL, "gworker-visual-replay-v1:session")
    base_time = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)
    minutes = (0, 1, 8, 17, 25, 26, 31)

    def metadata(sequence: int) -> dict[str, object]:
        return {
            "event_id": uuid5(
                NAMESPACE_URL,
                f"gworker-visual-replay-v1:event:{sequence}",
            ),
            "session_id": session_id,
            "sequence": sequence,
            "occurred_at": base_time + timedelta(minutes=minutes[sequence - 1]),
        }

    events: tuple[DomainEvent, ...] = (
        SessionPlanned(
            **metadata(1),
            objective="Reproduce the reducer walkthrough",
            target_focus_seconds=1_500,
            target_break_seconds=300,
            policy_id=HierarchicalSoftmaxUCB().policy_id,
        ),
        FocusStarted(**metadata(2)),
        InterruptionRecorded(
            **metadata(3),
            kind=InterruptionKind.CONTEXT_SWITCH,
            elapsed_seconds=45,
        ),
        InterruptionRecorded(
            **metadata(4),
            kind=InterruptionKind.ENVIRONMENT,
            elapsed_seconds=15,
        ),
        FocusCompleted(**metadata(5), elapsed_seconds=1_470),
        BreakStarted(**metadata(6)),
        BreakCompleted(**metadata(7), elapsed_seconds=280),
    )
    details = (
        "target 25 min + 5 min",
        "focus clock entered",
        "context switch · 45 s",
        "environment · 15 s",
        "1,470 / 1,500 s · 98%",
        "break clock entered",
        "280 / 300 s · complete",
    )
    state: SessionState | None = None
    steps: list[ReplayStep] = []
    for event, detail in zip(events, details, strict=True):
        state = apply_event(state, event)
        steps.append(
            ReplayStep(
                sequence=event.sequence,
                event_kind=event.KIND,
                occurred_at=event.occurred_at.strftime("%H:%M UTC"),
                phase=state.phase.value,
                detail=detail,
            )
        )
    if state is None or state.phase is not SessionPhase.COMPLETED:
        raise RuntimeError("event visual fixture did not replay to completion")
    if (
        state.revision != 7
        or state.interruption_count != 2
        or state.interruption_seconds != 60
        or state.actual_focus_seconds != 1_470
        or state.actual_break_seconds != 280
    ):
        raise RuntimeError("event visual fixture projection drifted")
    return tuple(steps), state


def build_policy_scenario() -> PolicyScenario:
    """Run the shared twelve-review scenario through recommend and review."""

    policy = HierarchicalSoftmaxUCB()
    history: list[ReviewedDecision] = []
    choices: list[int] = []
    previous_focus_seconds: int | None = None
    for sequence in range(1, 13):
        context = FocusContext(
            task_kind=TaskKind.DEEP_WORK,
            energy=EnergyLevel.MEDIUM,
            available_seconds=3_600,
            previous_focus_seconds=previous_focus_seconds,
        )
        recommendation = policy.recommend(
            context,
            history,
            decision_id=uuid5(
                NAMESPACE_URL,
                f"gworker-demo-policy-v1:{sequence}",
            ),
            decision_sequence=sequence,
            rng=random.Random(20_260_000 + sequence),
        )
        selected_minutes = recommendation.template.focus_seconds // 60
        if selected_minutes < 40:
            fit = DurationFit.TOO_SHORT
        elif selected_minutes == 40:
            fit = DurationFit.JUST_RIGHT
        else:
            fit = DurationFit.TOO_LONG
        history.append(
            recommendation.review(
                fit=fit,
                objective_completed=selected_minutes >= 40,
            )
        )
        choices.append(selected_minutes)
        previous_focus_seconds = recommendation.template.focus_seconds

    final_context = FocusContext(
        task_kind=TaskKind.DEEP_WORK,
        energy=EnergyLevel.MEDIUM,
        available_seconds=3_600,
        previous_focus_seconds=previous_focus_seconds,
    )
    final_recommendation = policy.recommend(
        final_context,
        history,
        decision_id=uuid5(
            NAMESPACE_URL,
            "gworker-demo-policy-v1:13",
        ),
        decision_sequence=13,
        rng=random.Random(20_260_013),
    )
    if choices != [25, 40, 40, 40, 50, 40, 40, 40, 40, 40, 40, 40]:
        raise RuntimeError("policy visual history drifted")
    if (
        final_recommendation.template.template_id != "focus-40"
        or final_recommendation.bucket is not EvidenceBucket.EXACT
        or final_recommendation.evidence_count != 12
    ):
        raise RuntimeError("policy visual recommendation drifted")
    probability_sum = math.fsum(
        arm.probability for arm in final_recommendation.arm_scores
    )
    if not math.isclose(probability_sum, 1.0, abs_tol=1e-15):
        raise RuntimeError("policy visual propensities do not sum to one")
    return PolicyScenario(
        reviewed_history=tuple(history),
        recommendation=final_recommendation,
        choices_minutes=tuple(choices),
    )


def build_durable_decision_lineage() -> DurableDecisionLineage:
    """Exercise fixed durable policy lineage through the public storage API."""

    policy = HierarchicalSoftmaxUCB()
    first_context = FocusContext(
        task_kind=TaskKind.DEEP_WORK,
        energy=EnergyLevel.MEDIUM,
        available_seconds=3_600,
    )
    with tempfile.TemporaryDirectory(
        prefix="gworker-visual-policy-lineage-"
    ) as temporary:
        database = Path(temporary) / "events.sqlite3"
        store = SQLiteEventStore(database)
        first = store.recommend(
            policy,
            first_context,
            decision_id=DURABLE_FIRST_DECISION_ID,
            rng_seed=DURABLE_FIRST_SEED,
        )
        review = store.record_review(
            policy,
            first.decision_id,
            fit=DurationFit.JUST_RIGHT,
            objective_completed=True,
        )

        reopened = SQLiteEventStore(database)
        second = reopened.recommend(
            policy,
            FocusContext(
                task_kind=TaskKind.DEEP_WORK,
                energy=EnergyLevel.MEDIUM,
                available_seconds=3_600,
                previous_focus_seconds=first.template.focus_seconds,
            ),
            decision_id=DURABLE_SECOND_DECISION_ID,
            rng_seed=DURABLE_SECOND_SEED,
        )

        verification_store = SQLiteEventStore(database)
        journal_verification = verification_store.verify()
        verification = verification_store.verify_policy_history(policy)

    if (
        first.decision_id != DURABLE_FIRST_DECISION_ID
        or first.decision_sequence != 1
        or first.template.template_id != "focus-40"
        or first.propensity.hex() != "0x1.0000000000000p-2"
    ):
        raise RuntimeError("first durable policy decision drifted")
    if (
        review.decision_id != first.decision_id
        or review.decision_sequence != first.decision_sequence
        or review.propensity.hex() != first.propensity.hex()
        or review.fit is not DurationFit.JUST_RIGHT
        or not review.objective_completed
    ):
        raise RuntimeError("durable policy review lost recommendation provenance")
    if (
        second.decision_id != DURABLE_SECOND_DECISION_ID
        or second.decision_sequence != 2
        or second.template.template_id != "focus-25"
        or second.propensity.hex() != "0x1.1e5ae1020930fp-2"
        or second.evidence_count != 1
    ):
        raise RuntimeError("second durable policy decision drifted")
    if (
        first.policy_id != policy.policy_id
        or review.policy_id != policy.policy_id
        or second.policy_id != policy.policy_id
        or verification.policy_id != policy.policy_id
    ):
        raise RuntimeError("durable policy fingerprint drifted")
    if (
        verification.decision_count,
        verification.review_count,
        verification.history_edge_count,
        verification.sqlite_check,
        journal_verification.sqlite_check,
    ) != (2, 1, 1, "ok", "ok"):
        raise RuntimeError("durable policy replay counts drifted")
    return DurableDecisionLineage(
        first=first,
        review=review,
        second=second,
        verification=verification,
    )


def build_focus_session_linkage() -> FocusSessionLinkageEvidence:
    """Exercise schema-v3 focus linkage through the public storage API."""

    policy = HierarchicalSoftmaxUCB()
    context = FocusContext(
        task_kind=TaskKind.DEEP_WORK,
        energy=EnergyLevel.HIGH,
        available_seconds=3_600,
    )
    with tempfile.TemporaryDirectory(
        prefix="gworker-visual-focus-linkage-"
    ) as temporary:
        database = Path(temporary) / "events.sqlite3"
        store = SQLiteEventStore(database)
        recommendation = store.recommend(
            policy,
            context,
            decision_id=LINKAGE_DECISION_ID,
            rng_seed=LINKAGE_SEED,
        )
        planned = SessionPlanned(
            event_id=LINKAGE_PLANNED_EVENT_ID,
            session_id=LINKAGE_SESSION_ID,
            sequence=1,
            occurred_at=LINKAGE_BASE_TIME,
            objective="Exercise explicit linkage with a synthetic session",
            target_focus_seconds=recommendation.template.focus_seconds,
            target_break_seconds=recommendation.template.break_seconds,
            policy_id=recommendation.policy_id,
        )
        planned_state = store.append(planned)
        link = store.link_focus_session(
            policy,
            recommendation.decision_id,
            session_id=planned.session_id,
        )
        store.append(
            FocusStarted(
                event_id=LINKAGE_STARTED_EVENT_ID,
                session_id=planned.session_id,
                sequence=2,
                occurred_at=LINKAGE_BASE_TIME + timedelta(minutes=1),
            )
        )
        state = store.append(
            SessionAbandoned(
                event_id=LINKAGE_ABANDONED_EVENT_ID,
                session_id=planned.session_id,
                sequence=3,
                occurred_at=LINKAGE_BASE_TIME + timedelta(minutes=13),
                reason=AbandonReason.PRIORITY_CHANGED,
            )
        )
        before_review = store.verify_policy_history(policy)
        if store.reviewed_decisions(policy):
            raise RuntimeError("session progress inferred policy feedback")
        review = store.record_review(
            policy,
            recommendation.decision_id,
            fit=DurationFit.TOO_SHORT,
            objective_completed=False,
        )

        reopened = SQLiteEventStore(database)
        reopened_link = reopened.focus_session_link(
            policy,
            session_id=planned.session_id,
        )
        replayed_state = reopened.replay(planned.session_id)
        journal_verification = reopened.verify()
        policy_verification = reopened.verify_policy_history(policy)
        reviewed = reopened.reviewed_decisions(policy)
        with sqlite3.connect(database) as connection:
            stored_schema_version = int(
                connection.execute(
                    """
                    SELECT value
                    FROM journal_metadata
                    WHERE key = 'schema_version'
                    """
                ).fetchone()[0]
            )
            stored_link_columns = tuple(
                row[1]
                for row in connection.execute("PRAGMA table_info(focus_session_links)")
            )
            stored_link_row = connection.execute(
                """
                SELECT decision_id, planned_event_id
                FROM focus_session_links
                """
            ).fetchone()

    if (
        planned_state.phase is not SessionPhase.PLANNED
        or planned_state.revision != 1
        or recommendation.decision_id != LINKAGE_DECISION_ID
        or recommendation.decision_sequence != 1
        or recommendation.template.template_id != "focus-15"
        or recommendation.propensity.hex() != "0x1.0000000000000p-2"
        or recommendation.evidence_count != 0
    ):
        raise RuntimeError("focus-linkage recommendation fixture drifted")
    if (
        link.decision_id != recommendation.decision_id
        or link.session_id != planned.session_id
        or link.planned_event_id != planned.event_id
        or link.policy_id != policy.policy_id
        or link.template_id != recommendation.template.template_id
        or reopened_link != link
    ):
        raise RuntimeError("focus-linkage provenance drifted")
    if (
        state != replayed_state
        or state.phase is not SessionPhase.ABANDONED
        or state.revision != 3
        or state.abandon_reason is not AbandonReason.PRIORITY_CHANGED
    ):
        raise RuntimeError("focus-linkage session replay drifted")
    if (
        before_review.decision_count,
        before_review.review_count,
        before_review.history_edge_count,
        before_review.sqlite_check,
    ) != (1, 0, 0, "ok"):
        raise RuntimeError("focus-linkage inferred feedback before explicit review")
    if (
        review.decision_id != recommendation.decision_id
        or review.fit is not DurationFit.TOO_SHORT
        or review.objective_completed
        or reviewed != (review,)
    ):
        raise RuntimeError("focus-linkage explicit review drifted")
    if (
        journal_verification.session_count,
        journal_verification.event_count,
        journal_verification.sqlite_check,
    ) != (1, 3, "ok"):
        raise RuntimeError("focus-linkage journal verification drifted")
    if (
        policy_verification.decision_count,
        policy_verification.review_count,
        policy_verification.history_edge_count,
        policy_verification.sqlite_check,
    ) != (1, 1, 0, "ok"):
        raise RuntimeError("focus-linkage policy verification drifted")
    if (
        SCHEMA_VERSION != 3
        or stored_schema_version != SCHEMA_VERSION
        or stored_link_columns != ("decision_id", "planned_event_id")
        or stored_link_row != (str(recommendation.decision_id), str(planned.event_id))
    ):
        raise RuntimeError("focus-linkage schema version drifted")
    if observed_locked_outcome_artifacts():
        raise RuntimeError("locked outcome artifacts now exist")
    return FocusSessionLinkageEvidence(
        recommendation=recommendation,
        planned=planned,
        link=link,
        before_review=before_review,
        review=review,
        reopened_link=reopened_link,
        state=state,
        journal_verification=journal_verification,
        policy_verification=policy_verification,
        schema_version=stored_schema_version,
        link_columns=stored_link_columns,
    )


def build_guardrail_rows() -> tuple[GuardrailRow, ...]:
    """Query the real availability and one-step guardrails."""

    policy = HierarchicalSoftmaxUCB()
    definitions = (
        ("18 min · no previous", 18, None),
        ("30 min · no previous", 30, None),
        ("60 min · previous 25", 60, 25),
        ("18 min · previous 50", 18, 50),
        ("60 min · previous 40", 60, 40),
        ("60 min · no previous", 60, None),
    )
    rows: list[GuardrailRow] = []
    index_by_duration = {
        template.focus_seconds: index for index, template in enumerate(policy.templates)
    }
    for label, available_minutes, previous_minutes in definitions:
        context = FocusContext(
            task_kind=TaskKind.DEEP_WORK,
            energy=EnergyLevel.MEDIUM,
            available_seconds=available_minutes * 60,
            previous_focus_seconds=(
                None if previous_minutes is None else previous_minutes * 60
            ),
        )
        feasible, reasons = policy.feasible_templates(context)
        allowed = tuple(template.template_id for template in feasible)
        previous_index = (
            None
            if context.previous_focus_seconds is None
            else index_by_duration[context.previous_focus_seconds]
        )
        cells: list[str] = []
        for template in policy.templates:
            if template.template_id in allowed:
                if "availability_overrode_step" in reasons:
                    cells.append("ALLOW · override")
                else:
                    cells.append("ALLOW")
            elif template.total_seconds > context.available_seconds:
                cells.append("BLOCK · budget")
            elif (
                previous_index is not None
                and abs(index_by_duration[template.focus_seconds] - previous_index) > 1
            ):
                cells.append("BLOCK · step")
            else:
                raise RuntimeError("guardrail visual could not explain a block")
        rows.append(
            GuardrailRow(
                label=label,
                context=context,
                allowed_template_ids=allowed,
                reason_codes=reasons,
                cells=tuple(cells),
            )
        )
    return tuple(rows)


def build_protocol_inventory() -> EvidenceCardinalities:
    """Return the exact zero-outcome protocol inventory."""

    return expected_publication_cardinalities(DEFAULT_EXPERIMENT_CONFIG)


def observed_locked_outcome_artifacts() -> tuple[str, ...]:
    """List committed outcome artifacts without opening or running evaluation."""

    run_directory = ROOT / ".gworker" / "publication" / LOCKED_EVALUATION_RUN_KEY
    if not run_directory.exists():
        return ()
    candidates = (
        "result.bin",
        "report.json",
        "evidence.json",
        "manifest.json",
    )
    observed = [name for name in candidates if (run_directory / name).is_file()]
    artifact_directory = run_directory / "artifacts"
    if artifact_directory.is_dir():
        observed.extend(
            f"artifacts/{path.name}"
            for path in sorted(artifact_directory.iterdir())
            if path.is_file()
        )
    return tuple(observed)


def _render_event_replay() -> RenderedVisual:
    steps, state = build_event_replay()
    title = "A real seven-event replay reaches a completed projection"
    description = (
        "Seven synthetic, privacy-safe domain events are applied through "
        "gworker.domain.apply_event. Direct labels show sequence, UTC time, "
        "event kind, resulting phase, and the final measured projection."
    )
    body = [
        _text(
            60,
            58,
            "EVENT-SOURCED SESSION · SYNTHETIC API FIXTURE",
            size=14,
            weight=700,
            fill=BLUE,
        ),
        _text(60, 96, title, size=28, weight=700),
        _text(
            60,
            126,
            "Every node is the projection returned by apply_event; no UI mockup.",
            size=16,
            fill=GRAY,
        ),
        _line(120, 214, 1320, 214, stroke=LIGHT_GRAY, width=6),
    ]
    colors = (BLUE, SKY, ORANGE, ORANGE, GREEN, PURPLE, BLUE)
    for index, (step, color) in enumerate(zip(steps, colors, strict=True)):
        x = 120 + index * 200
        if index < len(steps) - 1:
            body.append(_line(x + 20, 214, x + 176, 214, arrow=True))
        body.extend(
            [
                _circle(x, 214, 18, fill=color),
                _text(
                    x,
                    220,
                    step.sequence,
                    size=15,
                    weight=700,
                    fill=WHITE,
                    anchor="middle",
                ),
                _text(
                    x,
                    172,
                    step.occurred_at,
                    size=13,
                    weight=600,
                    fill=GRAY,
                    anchor="middle",
                ),
                _rect(x - 78, 252, 156, 148, fill=PANEL, stroke=color, stroke_width=2),
                _text(x, 280, step.event_kind, size=13, weight=700, anchor="middle"),
                _text(
                    x,
                    310,
                    f"→ {step.phase}",
                    size=14,
                    weight=700,
                    fill=color,
                    anchor="middle",
                ),
                _multiline(
                    x,
                    342,
                    tuple(step.detail.split(" · ")),
                    size=13,
                    fill=GRAY,
                    line_height=20,
                    anchor="middle",
                ),
            ]
        )
    body.extend(
        [
            _rect(60, 438, 1320, 100, fill="#F0F8FF", stroke=BLUE, stroke_width=2),
            _text(84, 470, "FINAL PROJECTION", size=14, weight=700, fill=BLUE),
            _text(
                84,
                504,
                (
                    f"phase={state.phase.value} · revision={state.revision} · "
                    f"interruptions={state.interruption_count} / "
                    f"{state.interruption_seconds}s"
                ),
                size=18,
                weight=650,
            ),
            _text(
                760,
                504,
                (
                    f"focus={state.actual_focus_seconds}s "
                    f"({(state.focus_completion_ratio or 0):.0%}) · "
                    f"break={state.actual_break_seconds}s"
                ),
                size=18,
                weight=650,
            ),
        ]
    )
    content = _svg_document(
        stem="event-replay",
        title=title,
        description=description,
        width=1440,
        height=580,
        body=body,
    )
    return RenderedVisual("event-replay.svg", title, description, content)


def _render_policy_scores() -> RenderedVisual:
    scenario = build_policy_scenario()
    recommendation = scenario.recommendation
    title = "Twelve explicit reviews produce an explainable focus-40 choice"
    description = (
        "A fixed sequence of twelve recommendations is reviewed through the "
        "public API. The thirteenth recommendation selects focus-40. Each "
        "feasible arm directly labels its posterior, ordinal adjustment, UCB "
        "bonus, total score, review count, and exact logged propensity."
    )
    body = [
        _text(
            56,
            52,
            "POLICY SCORE DECOMPOSITION · REAL PUBLIC API",
            size=14,
            weight=700,
            fill=BLUE,
        ),
        _text(56, 88, title, size=27, weight=700),
        _text(
            56,
            118,
            (
                f"bucket={recommendation.bucket.value} · "
                f"evidence={recommendation.evidence_count} reviews · "
                f"selected={recommendation.template.template_id} · "
                f"propensity={recommendation.propensity:.12f}"
            ),
            size=16,
            weight=600,
            fill=GRAY,
        ),
        _rect(56, 146, 1168, 54, fill=PANEL),
        _text(76, 178, "Score =", size=14, weight=700),
        _rect(150, 162, 20, 14, fill=BLUE, stroke=BLUE, radius=2),
        _text(178, 176, "posterior", size=13),
        _rect(278, 162, 20, 14, fill=ORANGE, stroke=ORANGE, radius=2),
        _text(306, 176, "ordinal", size=13),
        _rect(398, 162, 20, 14, fill=GREEN, stroke=GREEN, radius=2),
        _text(426, 176, "UCB bonus", size=13),
        _text(
            610,
            178,
            "Probability bars use exact softmax + 2% floor values.",
            size=13,
            fill=GRAY,
        ),
    ]
    max_score = max(arm.score for arm in recommendation.arm_scores)
    for index, arm in enumerate(recommendation.arm_scores):
        y = 232 + index * 124
        selected = arm.template == recommendation.template
        panel_fill = "#EEF8F5" if selected else WHITE
        panel_stroke = GREEN if selected else LIGHT_GRAY
        body.append(
            _rect(
                56,
                y - 22,
                1168,
                102,
                fill=panel_fill,
                stroke=panel_stroke,
                stroke_width=3 if selected else 1,
            )
        )
        body.append(
            _text(
                78,
                y + 8,
                arm.template.template_id,
                size=18,
                weight=700,
                fill=GREEN if selected else BLACK,
            )
        )
        body.append(
            _text(
                78,
                y + 34,
                f"{arm.review_count} reviewed decision"
                + ("" if arm.review_count == 1 else "s"),
                size=13,
                fill=GRAY,
            )
        )
        if selected:
            body.append(_text(250, y + 8, "SELECTED", size=12, weight=800, fill=GREEN))
        bar_x = 350
        bar_width = 480
        scale = bar_width / max_score
        components = (
            (arm.posterior_mean, BLUE),
            (arm.directional_adjustment, ORANGE),
            (arm.exploration_bonus, GREEN),
        )
        cursor = bar_x
        for value, color in components:
            width = value * scale
            body.append(
                _rect(
                    cursor,
                    y - 2,
                    width,
                    22,
                    fill=color,
                    stroke=color,
                    radius=2,
                )
            )
            cursor += width
        body.extend(
            [
                _text(
                    bar_x,
                    y + 45,
                    (
                        f"{arm.posterior_mean:.6f} + "
                        f"{arm.directional_adjustment:.6f} + "
                        f"{arm.exploration_bonus:.6f} = "
                        f"{arm.score:.6f}"
                    ),
                    size=13,
                    family="ui-monospace, SFMono-Regular, monospace",
                ),
                _rect(884, y - 2, 280, 22, fill="#EEF2F6", stroke="#EEF2F6", radius=4),
                _rect(
                    884,
                    y - 2,
                    280 * arm.probability,
                    22,
                    fill=PURPLE,
                    stroke=PURPLE,
                    radius=4,
                ),
                _text(
                    884,
                    y + 45,
                    f"p = {arm.probability:.12f}",
                    size=14,
                    weight=700,
                ),
            ]
        )
    choices = " → ".join(str(value) for value in scenario.choices_minutes)
    body.extend(
        [
            _rect(56, 598, 1168, 96, fill=PANEL),
            _text(
                76,
                628,
                "THE TWELVE REVIEWED CHOICES (minutes)",
                size=13,
                weight=700,
                fill=BLUE,
            ),
            _text(76, 660, choices, size=17, weight=650),
            _text(
                76,
                683,
                (
                    "Feedback rule: below 40 = too short; 40 = just right; "
                    "above 40 = too long. Completion is explicit at ≥40."
                ),
                size=13,
                fill=GRAY,
            ),
        ]
    )
    content = _svg_document(
        stem="policy-score",
        title=title,
        description=description,
        width=1280,
        height=740,
        body=body,
    )
    return RenderedVisual(
        "policy-score-decomposition.svg",
        title,
        description,
        content,
    )


def _render_guardrail_matrix() -> RenderedVisual:
    rows = build_guardrail_rows()
    title = "Availability and one-step movement are applied before scoring"
    description = (
        "Six fixed contexts are passed to feasible_templates. Matrix cells "
        "directly say ALLOW, BLOCK budget, BLOCK step, or ALLOW override for "
        "all four configured focus and break templates."
    )
    body = [
        _text(
            52,
            52,
            "GUARDRAIL MATRIX · REAL feasible_templates CALLS",
            size=14,
            weight=700,
            fill=BLUE,
        ),
        _text(52, 88, title, size=27, weight=700),
        _text(
            52,
            118,
            "A complete focus + break budget is checked before adjacency.",
            size=16,
            fill=GRAY,
        ),
    ]
    label_width = 270
    cell_width = 226.5
    table_x = 52
    table_y = 158
    row_height = 82
    body.append(_rect(table_x, table_y, 1176, 64, fill="#EAF2F8", radius=8))
    body.append(_text(table_x + 18, table_y + 38, "Context", size=15, weight=700))
    for index, template in enumerate(DEFAULT_TEMPLATES):
        x = table_x + label_width + index * cell_width
        body.extend(
            [
                _text(
                    x + cell_width / 2,
                    table_y + 28,
                    template.template_id,
                    size=15,
                    weight=700,
                    anchor="middle",
                ),
                _text(
                    x + cell_width / 2,
                    table_y + 50,
                    (
                        f"{template.focus_seconds // 60}+"
                        f"{template.break_seconds // 60}="
                        f"{template.total_seconds // 60} min"
                    ),
                    size=12,
                    fill=GRAY,
                    anchor="middle",
                ),
            ]
        )
    for row_index, row in enumerate(rows):
        y = table_y + 72 + row_index * row_height
        body.append(
            _rect(
                table_x,
                y,
                1176,
                row_height - 8,
                fill=PANEL if row_index % 2 else WHITE,
                radius=6,
            )
        )
        body.extend(
            [
                _text(table_x + 18, y + 29, row.label, size=15, weight=700),
                _text(
                    table_x + 18,
                    y + 54,
                    " · ".join(row.reason_codes) if row.reason_codes else "no filters",
                    size=12,
                    fill=GRAY,
                ),
            ]
        )
        for column, cell in enumerate(row.cells):
            x = table_x + label_width + column * cell_width
            if cell.startswith("ALLOW"):
                color = GREEN if "override" not in cell else ORANGE
                symbol = "✓"
            else:
                color = VERMILION if "budget" in cell else BLUE
                symbol = "X"
            body.extend(
                [
                    _circle(x + 34, y + 34, 14, fill=color),
                    _text(
                        x + 34,
                        y + 40,
                        symbol,
                        size=16,
                        weight=800,
                        fill=WHITE,
                        anchor="middle",
                    ),
                    _text(x + 56, y + 39, cell, size=13, weight=650, fill=color),
                ]
            )
    legend_y = table_y + 72 + len(rows) * row_height + 12
    body.extend(
        [
            _rect(52, legend_y, 1176, 72, fill=PANEL),
            _text(72, legend_y + 28, "DIRECT LABELS", size=12, weight=700, fill=GRAY),
            _text(
                72,
                legend_y + 52,
                (
                    "budget = full focus+break does not fit · step = more than "
                    "one adjacent template · override = no adjacent option fits"
                ),
                size=14,
            ),
        ]
    )
    content = _svg_document(
        stem="guardrail-matrix",
        title=title,
        description=description,
        width=1280,
        height=legend_y + 104,
        body=body,
    )
    return RenderedVisual("guardrail-matrix.svg", title, description, content)


def _architecture_box(
    body: list[str],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    lines: tuple[str, ...],
    color: str,
    dashed: bool = False,
) -> None:
    body.append(
        _rect(
            x,
            y,
            width,
            height,
            fill=WHITE,
            stroke=color,
            stroke_width=2,
            dash="8 6" if dashed else None,
        )
    )
    body.append(_text(x + 18, y + 30, title, size=16, weight=750, fill=color))
    body.append(
        _multiline(
            x + 18,
            y + 58,
            lines,
            size=13,
            fill=GRAY,
            line_height=20,
        )
    )


def _render_durable_decision_lineage() -> RenderedVisual:
    scenario = build_durable_decision_lineage()
    first = scenario.first
    review = scenario.review
    second = scenario.second
    verification = scenario.verification
    title = "Durable recommendations retain exact review and replay lineage"
    description = (
        "A source-derived fixed workflow uses the public SQLiteEventStore on a "
        "disposable private journal: recommend D1, record one explicit review, "
        "reopen for D2, then reopen and verify two decisions, one review, one "
        "history edge, the exact policy fingerprint, and SQLite integrity."
    )
    body = [
        _text(
            52,
            50,
            "DURABLE DECISION LINEAGE · REAL PUBLIC STORAGE API",
            size=14,
            weight=700,
            fill=PURPLE,
        ),
        _text(52, 86, title, size=27, weight=700),
        _text(
            52,
            116,
            "Fixed synthetic context · every value below is recomputed from source.",
            size=15,
            fill=GRAY,
        ),
        _rect(52, 140, 1296, 52, fill="#FBF7FC", stroke=PURPLE),
        _text(
            72,
            173,
            f"Exact policy fingerprint: {verification.policy_id}",
            size=15,
            weight=700,
            fill=PURPLE,
            family="monospace",
        ),
    ]
    _architecture_box(
        body,
        x=52,
        y=224,
        width=230,
        height=154,
        title="D1 · RECOMMEND",
        lines=(
            f"sequence {first.decision_sequence} · {first.template.template_id}",
            f"seed {DURABLE_FIRST_SEED:,}",
            f"p = {first.propensity.hex()}",
        ),
        color=BLUE,
    )
    _architecture_box(
        body,
        x=306,
        y=224,
        width=230,
        height=154,
        title="EXPLICIT REVIEW",
        lines=(
            f"fit = {review.fit.value}",
            "objective completed = true",
            "exact D1 propensity preserved",
        ),
        color=PURPLE,
    )
    _architecture_box(
        body,
        x=560,
        y=224,
        width=138,
        height=154,
        title="REOPEN",
        lines=("new store", "same file", "path omitted"),
        color=GREEN,
    )
    _architecture_box(
        body,
        x=722,
        y=224,
        width=230,
        height=154,
        title="D2 · RECOMMEND",
        lines=(
            f"sequence {second.decision_sequence} · {second.template.template_id}",
            f"history evidence = {second.evidence_count}",
            f"p = {second.propensity.hex()}",
        ),
        color=BLUE,
    )
    _architecture_box(
        body,
        x=976,
        y=224,
        width=372,
        height=154,
        title="REOPEN + VERIFY",
        lines=(
            "exact policy replay",
            (
                f"decisions / reviews = {verification.decision_count} / "
                f"{verification.review_count}"
            ),
            f"history edges = {verification.history_edge_count}",
            f"SQLite quick_check = {verification.sqlite_check}",
        ),
        color=GREEN,
    )
    body.extend(
        [
            _line(282, 301, 302, 301, arrow=True),
            _line(536, 301, 556, 301, arrow=True),
            _line(698, 301, 718, 301, arrow=True),
            _line(952, 301, 972, 301, arrow=True),
            _rect(52, 414, 1296, 252, fill=PANEL, radius=18),
            _text(
                72,
                446,
                "THREE APPEND-ONLY POLICY RELATIONS",
                size=13,
                weight=800,
                fill=GREEN,
            ),
        ]
    )
    _architecture_box(
        body,
        x=72,
        y=466,
        width=486,
        height=170,
        title="policy_decisions",
        lines=(
            (
                f"D1 · seq 1 · {first.template.template_id} · "
                f"seed {DURABLE_FIRST_SEED:,}"
            ),
            (
                f"D2 · seq 2 · {second.template.template_id} · "
                f"seed {DURABLE_SECOND_SEED:,}"
            ),
            "canonical hexadecimal propensity + history digest",
        ),
        color=BLUE,
    )
    _architecture_box(
        body,
        x=582,
        y=466,
        width=330,
        height=170,
        title="policy_reviews",
        lines=(
            "D1 · fit = just_right",
            "objective completed = true",
            "caller supplies no propensity",
        ),
        color=PURPLE,
    )
    _architecture_box(
        body,
        x=936,
        y=466,
        width=392,
        height=170,
        title="policy_decision_history",
        lines=(
            "D2 · position 0 → reviewed D1",
            "count + SHA-256 bind order",
            "no future or orphaned review",
        ),
        color=GREEN,
    )
    metrics = (
        ("2", "DECISIONS REPLAYED", BLUE),
        ("1", "REVIEW REPLAYED", PURPLE),
        ("1", "HISTORY EDGE VERIFIED", GREEN),
        ("OK", "SQLITE QUICK_CHECK", ORANGE),
    )
    for index, (value, label, color) in enumerate(metrics):
        x = 52 + index * 324
        body.extend(
            [
                _rect(x, 690, 300, 92, fill=WHITE, stroke=color, stroke_width=2),
                _text(x + 22, 734, value, size=30, weight=800, fill=color),
                _text(x + 82, 734, label, size=13, weight=750, fill=GRAY),
            ]
        )
    body.extend(
        [
            _rect(52, 806, 1296, 78, fill="#F7FBFF", stroke=SKY),
            _text(72, 836, "PRIVACY / CLAIM BOUNDARY", size=12, weight=800, fill=BLUE),
            _text(
                72,
                861,
                (
                    "Disposable private journal removed after verification · "
                    "no host path rendered · no locked evaluator or publication run"
                ),
                size=14,
                weight=600,
            ),
        ]
    )
    content = _svg_document(
        stem="durable-decision-lineage",
        title=title,
        description=description,
        width=1400,
        height=920,
        body=body,
    )
    return RenderedVisual(
        "durable-decision-lineage.svg",
        title,
        description,
        content,
    )


def _render_focus_session_linkage() -> RenderedVisual:
    scenario = build_focus_session_linkage()
    recommendation = scenario.recommendation
    planned = scenario.planned
    link = scenario.link
    state = scenario.state
    before = scenario.before_review
    after = scenario.policy_verification
    journal = scenario.journal_verification
    title = "Session progress stays provenance until an explicit review"
    description = (
        "A fixed synthetic workflow exercises the public SQLiteEventStore in a "
        "disposable private journal. It durably recommends, separately appends "
        "a matching plan, links before start, starts and abandons the session, "
        "proves zero inferred reviews, records one explicit review, reopens, "
        "derives the session identity through the planned event, and verifies "
        "the journal. Identifiers are stable and truncated; no objective or "
        "host path is rendered."
    )
    step_x = (42, 270, 498, 726, 954, 1182)
    step_width = 210
    step_y = 168
    step_height = 196
    body = [
        _text(
            42,
            48,
            "SCHEMA-v3 FOCUS LINKAGE · REAL PUBLIC STORAGE API",
            size=14,
            weight=700,
            fill=PURPLE,
        ),
        _text(42, 84, title, size=28, weight=700),
        _text(
            42,
            116,
            (
                "Fixed synthetic records · disposable private journal · "
                "replay and lookup verified after reopen"
            ),
            size=15,
            fill=GRAY,
        ),
        _rect(42, 132, 1350, 24, fill="#FBF7FC", stroke=PURPLE, radius=6),
        _text(
            717,
            149,
            (
                "Provenance link ≠ feedback · event codec remains v1 · "
                "locked outcome artifacts = 0"
            ),
            size=13,
            weight=700,
            fill=PURPLE,
            anchor="middle",
        ),
    ]
    steps = (
        (
            "1 · RECOMMEND",
            (
                f"decision {_short_uuid(recommendation.decision_id)}",
                (
                    f"seq {recommendation.decision_sequence} · "
                    f"{recommendation.template.template_id}"
                ),
                (
                    f"{recommendation.template.focus_seconds // 60}+"
                    f"{recommendation.template.break_seconds // 60} min"
                ),
                f"reviews seen = {recommendation.evidence_count}",
            ),
            BLUE,
        ),
        (
            "2 · APPEND PLAN",
            (
                f"event {_short_uuid(planned.event_id)}",
                f"session {_short_uuid(planned.session_id)}",
                "revision 1 · planned",
                "durations match choice",
            ),
            SKY,
        ),
        (
            "3 · LINK BEFORE START",
            (
                "link_focus_session()",
                "decision ↔ planned event",
                "one immutable row",
                "reviews remain 0",
            ),
            GREEN,
        ),
        (
            "4 · START → ABANDON",
            (
                "FocusStarted · seq 2",
                "SessionAbandoned · seq 3",
                f"phase = {state.phase.value}",
                f"reason = {state.abandon_reason.value}",
            ),
            ORANGE,
        ),
        (
            "5 · EXPLICIT REVIEW",
            (
                "record_review() separately",
                f"fit = {scenario.review.fit.value}",
                "objective completed = false",
                f"reviews {before.review_count} → {after.review_count}",
            ),
            PURPLE,
        ),
        (
            "6 · REOPEN + LOOKUP",
            (
                f"input session {_short_uuid(link.session_id)}",
                f"derived decision {_short_uuid(link.decision_id)}",
                f"derived plan {_short_uuid(link.planned_event_id)}",
                "same link = true",
            ),
            GREEN,
        ),
    )
    for index, (x, step) in enumerate(zip(step_x, steps, strict=True)):
        heading, lines, color = step
        if index < len(steps) - 1:
            body.append(
                _line(
                    x + step_width,
                    step_y + step_height / 2,
                    step_x[index + 1] - 4,
                    step_y + step_height / 2,
                    arrow=True,
                )
            )
        body.extend(
            [
                _rect(
                    x,
                    step_y,
                    step_width,
                    step_height,
                    fill=WHITE,
                    stroke=color,
                    stroke_width=2,
                ),
                _text(
                    x + step_width / 2,
                    step_y + 32,
                    heading,
                    size=13,
                    weight=800,
                    fill=color,
                    anchor="middle",
                ),
                _multiline(
                    x + 16,
                    step_y + 72,
                    lines,
                    size=12,
                    fill=GRAY,
                    line_height=28,
                ),
            ]
        )

    body.extend(
        [
            _rect(
                42,
                402,
                662,
                272,
                fill="#F6FBF8",
                stroke=GREEN,
                stroke_width=2,
                radius=18,
            ),
            _text(
                66,
                438,
                "PROVENANCE · focus_session_links",
                size=15,
                weight=800,
                fill=GREEN,
            ),
            _text(
                680,
                438,
                f"SQLite schema v{scenario.schema_version}",
                size=13,
                weight=700,
                fill=GRAY,
                anchor="end",
            ),
            _rect(66, 462, 614, 118, fill=WHITE, stroke=GREEN, stroke_width=2),
            _rect(66, 462, 614, 42, fill="#E7F5EF", stroke=GREEN, radius=10),
            _line(368, 462, 368, 580, stroke=GREEN, width=1),
            _text(
                217,
                489,
                scenario.link_columns[0],
                size=14,
                weight=800,
                fill=GREEN,
                anchor="middle",
            ),
            _text(
                524,
                489,
                scenario.link_columns[1],
                size=14,
                weight=800,
                fill=GREEN,
                anchor="middle",
            ),
            _text(
                217,
                548,
                _short_uuid(link.decision_id),
                size=16,
                weight=700,
                anchor="middle",
                family="monospace",
            ),
            _text(
                524,
                548,
                _short_uuid(link.planned_event_id),
                size=16,
                weight=700,
                anchor="middle",
                family="monospace",
            ),
            _text(
                66,
                610,
                "session_id is not stored in this table",
                size=15,
                weight=750,
                fill=GREEN,
            ),
            _text(
                66,
                640,
                (
                    "Lookup joins planned_event_id → events and derives "
                    f"session {_short_uuid(link.session_id)}."
                ),
                size=14,
                fill=GRAY,
            ),
            _rect(
                730,
                402,
                662,
                272,
                fill="#FBF7FC",
                stroke=PURPLE,
                stroke_width=2,
                radius=18,
            ),
            _text(
                754,
                438,
                "FEEDBACK · policy_reviews",
                size=15,
                weight=800,
                fill=PURPLE,
            ),
            _line(1056, 466, 1056, 642, stroke=LIGHT_GRAY, width=2),
            _text(
                778,
                478,
                "BEFORE EXPLICIT REVIEW",
                size=12,
                weight=800,
                fill=ORANGE,
            ),
            _text(
                778,
                526,
                str(before.review_count),
                size=46,
                weight=800,
                fill=ORANGE,
            ),
            _text(838, 516, "reviews", size=16, weight=700),
            _multiline(
                778,
                558,
                (
                    f"{journal.event_count} session events",
                    f"terminal phase = {state.phase.value}",
                    f"history edges = {before.history_edge_count}",
                ),
                size=14,
                fill=GRAY,
                line_height=26,
            ),
            _text(
                1080,
                478,
                "AFTER record_review()",
                size=12,
                weight=800,
                fill=PURPLE,
            ),
            _text(
                1080,
                526,
                str(after.review_count),
                size=46,
                weight=800,
                fill=PURPLE,
            ),
            _text(1140, 516, "review", size=16, weight=700),
            _multiline(
                1080,
                558,
                (
                    f"fit = {scenario.review.fit.value}",
                    "completion = false",
                    f"history edges = {after.history_edge_count}",
                ),
                size=14,
                fill=GRAY,
                line_height=26,
            ),
            _text(
                754,
                650,
                "Starting and abandoning the session never synthesize feedback.",
                size=14,
                weight=700,
                fill=PURPLE,
            ),
        ]
    )

    metrics = (
        (
            f"{journal.session_count} / {journal.event_count}",
            "SESSIONS / EVENTS",
            BLUE,
        ),
        (
            f"{before.decision_count} / {before.review_count}",
            "DECISIONS / REVIEWS · BEFORE",
            ORANGE,
        ),
        (
            f"{after.decision_count} / {after.review_count}",
            "DECISIONS / REVIEWS · AFTER",
            PURPLE,
        ),
        (str(after.history_edge_count), "HISTORY EDGES", GREEN),
        (journal.sqlite_check.upper(), "SQLITE QUICK_CHECK", GREEN),
    )
    for index, (value, label, color) in enumerate(metrics):
        x = 42 + index * 270
        body.extend(
            [
                _rect(x, 710, 246, 92, fill=WHITE, stroke=color, stroke_width=2),
                _text(x + 18, 750, value, size=26, weight=800, fill=color),
                _text(x + 18, 778, label, size=11, weight=750, fill=GRAY),
            ]
        )
    body.extend(
        [
            _rect(42, 838, 1350, 94, fill="#F7FBFF", stroke=SKY),
            _text(66, 870, "PRIVACY / CLAIM BOUNDARY", size=12, weight=800, fill=BLUE),
            _multiline(
                66,
                896,
                (
                    "Synthetic fixture only · objective and host path omitted · "
                    "temporary journal removed",
                    (
                        "No evaluator, publication run, result metric, inferred "
                        "feedback, or causal claim"
                    ),
                ),
                size=14,
                weight=600,
                line_height=22,
            ),
        ]
    )
    content = _svg_document(
        stem="focus-session-linkage",
        title=title,
        description=description,
        width=1434,
        height=970,
        body=body,
    )
    return RenderedVisual(
        "focus-session-linkage.svg",
        title,
        description,
        content,
    )


def _render_architecture() -> RenderedVisual:
    title = "Current architecture separates private journals from publication"
    description = (
        "A source-backed architecture map shows explicit caller inputs, the "
        "domain and policy cores, private canonical storage, and the locked "
        "synthetic publication path. No private journal data enters publication. "
        "Durable session linkage is implemented in the storage API; no linkage "
        "CLI exists. The dashed box marks rendering and sealing as next work."
    )
    body = [
        _text(
            52,
            50,
            "ARCHITECTURE + TRUST BOUNDARIES · CURRENT SOURCE",
            size=14,
            weight=700,
            fill=BLUE,
        ),
        _text(52, 86, title, size=27, weight=700),
        _text(
            52,
            116,
            "Solid = implemented · dashed = NEXT · arrows show data flow.",
            size=15,
            fill=GRAY,
        ),
        _rect(38, 146, 424, 554, fill="#F7FBFF", stroke=SKY, stroke_width=2, radius=18),
        _text(58, 176, "CALLER / LOCAL CORE", size=13, weight=800, fill=BLUE),
        _rect(
            486, 146, 424, 554, fill="#F6FBF8", stroke=GREEN, stroke_width=2, radius=18
        ),
        _text(506, 176, "PRIVATE DURABLE STATE", size=13, weight=800, fill=GREEN),
        _rect(
            934, 146, 428, 554, fill="#FFFAF2", stroke=ORANGE, stroke_width=2, radius=18
        ),
        _text(
            954, 176, "LOCKED SYNTHETIC PUBLICATION", size=13, weight=800, fill=ORANGE
        ),
    ]
    _architecture_box(
        body,
        x=62,
        y=204,
        width=176,
        height=132,
        title="Explicit events",
        lines=("objective + measured", "durations + bounded", "interruption kinds"),
        color=BLUE,
    )
    _architecture_box(
        body,
        x=262,
        y=204,
        width=176,
        height=132,
        title="Domain reducer",
        lines=("strict revisions", "phase transitions", "pure replay state"),
        color=BLUE,
    )
    body.append(_line(238, 270, 258, 270, arrow=True))
    _architecture_box(
        body,
        x=62,
        y=374,
        width=176,
        height=142,
        title="Policy context",
        lines=("task kind + energy", "available time", "previous template"),
        color=PURPLE,
    )
    _architecture_box(
        body,
        x=262,
        y=374,
        width=176,
        height=142,
        title="Policy kernel",
        lines=("guardrails", "posterior + ordinal", "UCB + propensity"),
        color=PURPLE,
    )
    body.append(_line(238, 445, 258, 445, arrow=True))
    _architecture_box(
        body,
        x=158,
        y=554,
        width=244,
        height=104,
        title="Journal storage + CLI",
        lines=("recommend + explicit review", "seeded replay + verify"),
        color=PURPLE,
    )
    body.extend(
        [
            _line(350, 516, 350, 550, arrow=True),
            (
                '<path d="M402 606 H470 V350 H798 V340" '
                f'fill="none" stroke="{GRAY}" stroke-width="2" '
                'marker-end="url(#arrow)"/>'
            ),
        ]
    )
    _architecture_box(
        body,
        x=510,
        y=204,
        width=176,
        height=132,
        title="Canonical codec",
        lines=("versioned JSON", "unknown fields fail", "byte-stable events"),
        color=GREEN,
    )
    _architecture_box(
        body,
        x=710,
        y=204,
        width=176,
        height=132,
        title="SQLite journal",
        lines=(
            "events + decisions",
            "reviews + provenance links",
            "transactional replay",
        ),
        color=GREEN,
    )
    body.append(_line(686, 270, 706, 270, arrow=True))
    _architecture_box(
        body,
        x=558,
        y=390,
        width=278,
        height=138,
        title="Local trust boundary",
        lines=(
            "0700 directories / 0600 files",
            "not encrypted by GWorker",
            "same-UID attacker is out of scope",
        ),
        color=GREEN,
    )
    _architecture_box(
        body,
        x=958,
        y=204,
        width=176,
        height=142,
        title="Double gate",
        lines=("clean Git + source", "read-only capacity", "clean Git again"),
        color=ORANGE,
    )
    _architecture_box(
        body,
        x=1158,
        y=204,
        width=176,
        height=142,
        title="Runner + state",
        lines=("single-use permit", "append-only stages", "crash burn/resume"),
        color=ORANGE,
    )
    body.append(_line(1134, 275, 1154, 275, arrow=True))
    _architecture_box(
        body,
        x=958,
        y=388,
        width=176,
        height=142,
        title="Locked evaluator",
        lines=("synthetic only", "pre-registered IDs", "no result produced yet"),
        color=VERMILION,
    )
    _architecture_box(
        body,
        x=1158,
        y=388,
        width=176,
        height=142,
        title="Canonical evidence",
        lines=("result.bin", "report + evidence", "exact inventories"),
        color=VERMILION,
    )
    body.append(_line(1134, 459, 1154, 459, arrow=True))
    _architecture_box(
        body,
        x=1058,
        y=566,
        width=244,
        height=92,
        title="Render + seal · NEXT",
        lines=("manifest + SVG outputs", "bind final checksums"),
        color=GRAY,
        dashed=True,
    )
    body.extend(
        [
            _line(438, 270, 506, 270, arrow=True),
            _line(1246, 346, 1046, 384, arrow=True),
            _line(1246, 530, 1180, 562, arrow=True, dash="7 5"),
            _rect(38, 730, 1324, 86, fill=PANEL),
            _text(
                58, 760, "PRIVACY / CLAIM BOUNDARIES", size=13, weight=800, fill=BLUE
            ),
            _text(
                58,
                790,
                (
                    "Policy omits objective text · private journal has no "
                    "publication path · synthetic visuals · no monitoring or "
                    "causal claim"
                ),
                size=15,
                weight=600,
            ),
        ]
    )
    content = _svg_document(
        stem="architecture",
        title=title,
        description=description,
        width=1400,
        height=850,
        body=body,
    )
    return RenderedVisual(
        "architecture-trust-boundaries.svg",
        title,
        description,
        content,
    )


def _render_publication_lifecycle() -> RenderedVisual:
    stages = tuple(PublicationStage)
    expected = (
        PublicationStage.PREPARED,
        PublicationStage.EVALUATING,
        PublicationStage.EVALUATED,
        PublicationStage.MATERIALIZED,
        PublicationStage.SEALED,
    )
    if stages != expected:
        raise RuntimeError("publication lifecycle stages drifted")
    title = "The runner stops at materialized; rendering and sealing are next"
    description = (
        "The implemented publication lifecycle is shown without claiming an "
        "executed run. Preflight precedes the append-only state chain. A crash "
        "in evaluating burns the run, while evaluated materialization can "
        "resume. Rendering and final sealing are directly marked NEXT."
    )
    body = [
        _text(
            52,
            52,
            "PUBLICATION CONTROL FLOW · NOT AN EXECUTED RESULT",
            size=14,
            weight=700,
            fill=BLUE,
        ),
        _text(52, 88, title, size=27, weight=700),
        _text(
            52,
            118,
            "Current locked outcome artifacts observed: 0.",
            size=16,
            weight=650,
            fill=VERMILION,
        ),
    ]
    nodes = (
        (
            "PREFLIGHT",
            ("clean source", "capacity read-only", "failure: no claim"),
            BLUE,
            False,
        ),
        (
            stages[0].value.upper(),
            ("provenance bound", "exclusive run", "may continue"),
            SKY,
            False,
        ),
        (
            stages[1].value.upper(),
            ("state fsynced", "then permit", "reopen = BURNED"),
            VERMILION,
            False,
        ),
        (
            stages[2].value.upper(),
            ("result.bin bound", "exact counts", "resume materialize"),
            GREEN,
            False,
        ),
        (
            stages[3].value.upper(),
            ("report + evidence", "verified bytes", "runner stops here"),
            ORANGE,
            False,
        ),
        (
            "RENDER · NEXT",
            ("SVG / tables", "manifest checksums", "not implemented"),
            GRAY,
            True,
        ),
        (
            f"{stages[4].value.upper()} · NEXT",
            ("state API exists", "runner path pending", "bind all outputs"),
            GRAY,
            True,
        ),
    )
    x_positions = (52, 242, 432, 622, 812, 1002, 1192)
    node_width = 166
    for index, (x, node) in enumerate(zip(x_positions, nodes, strict=True)):
        label, lines, color, dashed = node
        if index < len(nodes) - 1:
            body.append(
                _line(
                    x + node_width,
                    262,
                    x + 184,
                    262,
                    arrow=True,
                    dash="7 5" if index >= 4 else None,
                )
            )
        body.append(
            _rect(
                x,
                184,
                node_width,
                188,
                fill=WHITE if not dashed else PANEL,
                stroke=color,
                stroke_width=3 if index == 4 else 2,
                dash="8 6" if dashed else None,
            )
        )
        body.extend(
            [
                _text(
                    x + node_width / 2,
                    222,
                    label,
                    size=14,
                    weight=800,
                    fill=color,
                    anchor="middle",
                ),
                _multiline(
                    x + node_width / 2,
                    266,
                    lines,
                    size=13,
                    fill=GRAY,
                    line_height=28,
                    anchor="middle",
                ),
            ]
        )
    body.extend(
        [
            _line(515, 372, 515, 446, stroke=VERMILION, width=3, arrow=True),
            _rect(396, 452, 238, 82, fill="#FFF1ED", stroke=VERMILION, stroke_width=2),
            _text(
                515,
                482,
                "INTERRUPTED EVALUATING",
                size=13,
                weight=800,
                fill=VERMILION,
                anchor="middle",
            ),
            _text(515, 510, "burned · no retry/reset flag", size=13, anchor="middle"),
            _line(705, 372, 705, 446, stroke=GREEN, width=3, arrow=True),
            _rect(656, 452, 298, 82, fill="#EEF8F5", stroke=GREEN, stroke_width=2),
            _text(
                805,
                482,
                "POST-EVALUATION RESTART",
                size=13,
                weight=800,
                fill=GREEN,
                anchor="middle",
            ),
            _text(
                805,
                510,
                "reverify source + exact bytes · resume",
                size=13,
                anchor="middle",
            ),
            _rect(52, 584, 1306, 100, fill=PANEL),
            _text(74, 617, "IMPLEMENTED STOP", size=13, weight=800, fill=ORANGE),
            _text(
                74,
                650,
                (
                    "run_publication() can advance PREPARED → EVALUATED → "
                    "MATERIALIZED. It does not render or call seal()."
                ),
                size=16,
                weight=600,
            ),
        ]
    )
    content = _svg_document(
        stem="publication-lifecycle",
        title=title,
        description=description,
        width=1410,
        height=730,
        body=body,
    )
    return RenderedVisual(
        "publication-lifecycle.svg",
        title,
        description,
        content,
    )


def _format_count(value: int) -> str:
    return f"{value:,}"


def _render_protocol_inventory() -> RenderedVisual:
    inventory = build_protocol_inventory()
    observed = observed_locked_outcome_artifacts()
    if observed:
        raise RuntimeError(
            "locked outcome artifacts now exist; replace zero-outcome visual"
        )
    config = DEFAULT_EXPERIMENT_CONFIG
    title = "The locked protocol declares scale, but has produced zero outcomes"
    description = (
        "Counts are calculated by expected_publication_cardinalities from the "
        "locked configuration. The visual directly distinguishes declared "
        "population, implied computation, evidence rows, and the current zero "
        "outcome status. It contains no result metric."
    )
    body = [
        _text(
            52,
            50,
            "LOCKED PROTOCOL INVENTORY · EXPECTATIONS, NOT RESULTS",
            size=14,
            weight=700,
            fill=BLUE,
        ),
        _text(52, 86, title, size=27, weight=700),
        _text(
            52,
            116,
            "Computed from DEFAULT_EXPERIMENT_CONFIG without running evaluation.",
            size=15,
            fill=GRAY,
        ),
        _rect(52, 150, 1268, 104, fill="#FFF1ED", stroke=VERMILION, stroke_width=2),
        _text(
            78,
            187,
            "OBSERVED LOCKED OUTCOME ARTIFACTS",
            size=13,
            weight=800,
            fill=VERMILION,
        ),
        _text(78, 230, "0", size=42, weight=800, fill=VERMILION),
        _text(
            140,
            226,
            "No result.bin, report, evidence, manifest, or result plot.",
            size=18,
            weight=650,
        ),
    ]
    panels = (
        (
            52,
            286,
            "DECLARED POPULATION",
            (
                ("personas", len(config.personas)),
                ("availability modes", len(config.availability_modes)),
                ("environment seeds", len(config.environment_seeds)),
                ("strategies", len(Strategy)),
                ("adaptive replicas", config.policy_replicas),
                ("decisions / trajectory", config.horizon),
            ),
            BLUE,
        ),
        (
            372,
            286,
            "IMPLIED COMPUTATION",
            (
                ("cluster summaries", inventory.cluster_summaries),
                ("abrupt traces", inventory.abrupt_traces),
                ("raw trace points", inventory.raw_trace_points),
                ("trajectories", inventory.trajectories),
                ("decisions", inventory.decisions),
            ),
            GREEN,
        ),
        (
            692,
            286,
            "EVIDENCE TABLE ROWS",
            (
                ("scenario by strategy", inventory.scenario_strategy_rows),
                ("macro strategies", inventory.macro_strategy_rows),
                ("macro contrasts", inventory.macro_contrasts),
                ("scenario contrasts", inventory.scenario_contrasts),
                ("diagnostic scopes", inventory.adaptive_diagnostic_scopes),
                ("calibration rows", inventory.calibration_rows),
                ("template exposures", inventory.template_exposure_rows),
            ),
            ORANGE,
        ),
        (
            1012,
            286,
            "RECOVERY + RENDER INPUT",
            (
                ("recovery rows", inventory.recovery_rows),
                ("recovery contrasts", inventory.recovery_contrast_rows),
                ("trace series", inventory.trace_series),
                ("render trace points", inventory.trace_points),
            ),
            PURPLE,
        ),
    )
    panel_width = 296
    panel_height = 350
    for x, y, heading, values, color in panels:
        body.append(
            _rect(
                x,
                y,
                panel_width,
                panel_height,
                fill=PANEL,
                stroke=color,
                stroke_width=2,
            )
        )
        body.append(_text(x + 18, y + 34, heading, size=13, weight=800, fill=color))
        for row_index, (label, value) in enumerate(values):
            row_y = y + 76 + row_index * 40
            body.extend(
                [
                    _text(x + 18, row_y, label, size=13, fill=GRAY),
                    _text(
                        x + panel_width - 18,
                        row_y,
                        _format_count(value),
                        size=15,
                        weight=750,
                        anchor="end",
                    ),
                ]
            )
    body.extend(
        [
            _rect(52, 674, 1268, 90, fill="#F0F8FF", stroke=BLUE, stroke_width=2),
            _text(76, 706, "INTERPRETATION", size=13, weight=800, fill=BLUE),
            _text(
                76,
                738,
                (
                    "These values are cardinality invariants for a future "
                    "complete run—not measurements, effects, or claims."
                ),
                size=16,
                weight=600,
            ),
        ]
    )
    content = _svg_document(
        stem="protocol-inventory",
        title=title,
        description=description,
        width=1372,
        height=810,
        body=body,
    )
    return RenderedVisual(
        "locked-protocol-inventory.svg",
        title,
        description,
        content,
    )


def render_visuals() -> tuple[RenderedVisual, ...]:
    """Build every expected SVG in canonical filename order."""

    visuals = (
        _render_architecture(),
        _render_durable_decision_lineage(),
        _render_event_replay(),
        _render_focus_session_linkage(),
        _render_guardrail_matrix(),
        _render_protocol_inventory(),
        _render_policy_scores(),
        _render_publication_lifecycle(),
    )
    ordered = tuple(sorted(visuals, key=lambda visual: visual.filename))
    if len({visual.filename for visual in ordered}) != len(ordered):
        raise RuntimeError("visual filenames must be unique")
    return ordered


def build_manifest(visuals: tuple[RenderedVisual, ...]) -> bytes:
    """Create the canonical provenance manifest for the SVG bundle."""

    inputs = [
        {
            "path": relative_path,
            "sha256": _file_sha256(ROOT / relative_path),
        }
        for relative_path in sorted(INPUT_FILES)
    ]
    outputs = [
        {
            "byte_count": len(visual.content),
            "path": f"{GENERATED_DIRECTORY_NAME}/{visual.filename}",
            "sha256": _sha256(visual.content),
            "title": visual.title,
        }
        for visual in visuals
    ]
    payload = {
        "command": GENERATION_COMMAND,
        "inputs": inputs,
        "observations": {
            "locked_outcome_artifact_count": 0,
            "locked_result_visuals_generated": False,
        },
        "outputs": outputs,
        "python": {
            "requires": ">=3.11",
            "stdlib_only": True,
            "validated_minor_versions": list(VALIDATED_PYTHON_MINORS),
        },
        "schema_version": "gworker-visual-manifest-v1",
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
        },
    }
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def write_bundle(destination: Path) -> tuple[RenderedVisual, ...]:
    """Write a complete generated bundle beneath ``destination``."""

    visuals = render_visuals()
    generated = destination / GENERATED_DIRECTORY_NAME
    generated.mkdir(parents=True, exist_ok=True)
    expected_names = {visual.filename for visual in visuals}
    for path in generated.glob("*.svg"):
        if path.name not in expected_names:
            path.unlink()
    for visual in visuals:
        (generated / visual.filename).write_bytes(visual.content)
    (destination / MANIFEST_NAME).write_bytes(build_manifest(visuals))
    return visuals


def _bundle_differences(candidate: Path, committed: Path) -> tuple[str, ...]:
    expected_paths = (
        MANIFEST_NAME,
        *(
            f"{GENERATED_DIRECTORY_NAME}/{visual.filename}"
            for visual in render_visuals()
        ),
    )
    differences: list[str] = []
    for relative_path in expected_paths:
        candidate_path = candidate / relative_path
        committed_path = committed / relative_path
        if not committed_path.is_file():
            differences.append(f"missing: {relative_path}")
        elif candidate_path.read_bytes() != committed_path.read_bytes():
            differences.append(f"changed: {relative_path}")
    committed_generated = committed / GENERATED_DIRECTORY_NAME
    if committed_generated.is_dir():
        expected_svg_names = {
            Path(relative).name
            for relative in expected_paths
            if relative.endswith(".svg")
        }
        for path in sorted(committed_generated.glob("*.svg")):
            if path.name not in expected_svg_names:
                differences.append(
                    f"unexpected: {GENERATED_DIRECTORY_NAME}/{path.name}"
                )
    return tuple(differences)


def check_bundle() -> tuple[str, ...]:
    """Regenerate in a temporary directory and compare exact bytes."""

    VISUAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".visual-check-",
        dir=VISUAL_ROOT.parent,
    ) as temporary:
        candidate = Path(temporary) / "visuals"
        write_bundle(candidate)
        return _bundle_differences(candidate, VISUAL_ROOT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic non-result GWorker SVG evidence."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in a temporary directory and compare exact bytes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate or verify the committed SVG evidence bundle."""

    arguments = _parser().parse_args(argv)
    if arguments.check:
        differences = check_bundle()
        if differences:
            for difference in differences:
                sys.stderr.write(f"{difference}\n")
            return 1
        sys.stdout.write("visual evidence is reproducible\n")
        return 0
    visuals = write_bundle(VISUAL_ROOT)
    sys.stdout.write(f"generated {len(visuals)} SVG files and {MANIFEST_NAME}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
