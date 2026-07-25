"""Typed events and the pure GWorker session reducer.

Time measurement belongs to an application boundary. The domain receives
already-measured durations and UTC timestamps, which keeps replay deterministic
and makes the transition rules independently testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import ClassVar, TypeAlias
from uuid import UUID

MAX_OBJECTIVE_LENGTH = 240
MAX_DURATION_SECONDS = 24 * 60 * 60
MAX_POLICY_ID_LENGTH = 80


class InvalidEvent(ValueError):
    """Raised when an individual event is structurally invalid."""


class InvalidTransition(ValueError):
    """Raised when a valid event cannot follow the projected state."""


class SessionPhase(StrEnum):
    """Lifecycle states for one focus-and-break session."""

    PLANNED = "planned"
    FOCUSING = "focusing"
    FOCUS_COMPLETE = "focus_complete"
    BREAKING = "breaking"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class InterruptionKind(StrEnum):
    """Coarse categories that avoid collecting free-form private details."""

    NOTIFICATION = "notification"
    CONTEXT_SWITCH = "context_switch"
    ENVIRONMENT = "environment"
    INTERNAL = "internal"
    OTHER = "other"


class AbandonReason(StrEnum):
    """Bounded reasons suitable for aggregate policy evaluation."""

    PRIORITY_CHANGED = "priority_changed"
    INTERRUPTED = "interrupted"
    FATIGUE = "fatigue"
    TECHNICAL_FAILURE = "technical_failure"
    OTHER = "other"


def _validate_identifier(value: UUID, field_name: str) -> None:
    if not isinstance(value, UUID):
        raise InvalidEvent(f"{field_name} must be a UUID")
    if value.int == 0:
        raise InvalidEvent(f"{field_name} must not be the nil UUID")


def _validate_timestamp(value: datetime) -> None:
    if not isinstance(value, datetime):
        raise InvalidEvent("occurred_at must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise InvalidEvent("occurred_at must be timezone-aware UTC")


def _validate_duration(
    value: int, field_name: str, *, allow_zero: bool = False
) -> None:
    lower_bound = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidEvent(f"{field_name} must be an integer")
    if not lower_bound <= value <= MAX_DURATION_SECONDS:
        qualifier = "between 0" if allow_zero else "between 1"
        raise InvalidEvent(
            f"{field_name} must be {qualifier} and {MAX_DURATION_SECONDS} seconds"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EventMetadata:
    """Fields shared by every event in the v1 domain schema."""

    SCHEMA_VERSION: ClassVar[int] = 1

    event_id: UUID
    session_id: UUID
    sequence: int
    occurred_at: datetime

    def __post_init__(self) -> None:
        _validate_identifier(self.event_id, "event_id")
        _validate_identifier(self.session_id, "session_id")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise InvalidEvent("sequence must be an integer")
        if self.sequence < 1:
            raise InvalidEvent("sequence must be positive")
        _validate_timestamp(self.occurred_at)


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionPlanned(EventMetadata):
    """Declares one policy recommendation before timing begins."""

    KIND: ClassVar[str] = "session_planned"

    objective: str
    target_focus_seconds: int
    target_break_seconds: int
    policy_id: str

    def __post_init__(self) -> None:
        super(SessionPlanned, self).__post_init__()
        if not isinstance(self.objective, str):
            raise InvalidEvent("objective must be a string")
        if self.objective != self.objective.strip():
            raise InvalidEvent("objective must not have surrounding whitespace")
        if not 1 <= len(self.objective) <= MAX_OBJECTIVE_LENGTH:
            raise InvalidEvent(
                f"objective must contain 1 to {MAX_OBJECTIVE_LENGTH} characters"
            )
        if any(
            character.isspace() and character not in {" ", "\t"}
            for character in self.objective
        ):
            raise InvalidEvent("objective must be a single line")
        _validate_duration(self.target_focus_seconds, "target_focus_seconds")
        _validate_duration(self.target_break_seconds, "target_break_seconds")
        if not isinstance(self.policy_id, str):
            raise InvalidEvent("policy_id must be a string")
        if self.policy_id != self.policy_id.strip():
            raise InvalidEvent("policy_id must not have surrounding whitespace")
        if not 1 <= len(self.policy_id) <= MAX_POLICY_ID_LENGTH:
            raise InvalidEvent(
                f"policy_id must contain 1 to {MAX_POLICY_ID_LENGTH} characters"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class FocusStarted(EventMetadata):
    """Marks the point at which the planned focus block starts."""

    KIND: ClassVar[str] = "focus_started"


@dataclass(frozen=True, slots=True, kw_only=True)
class InterruptionRecorded(EventMetadata):
    """Records bounded aggregate interruption data without free-form content."""

    KIND: ClassVar[str] = "interruption_recorded"

    kind: InterruptionKind
    elapsed_seconds: int

    def __post_init__(self) -> None:
        super(InterruptionRecorded, self).__post_init__()
        if not isinstance(self.kind, InterruptionKind):
            raise InvalidEvent("kind must be an InterruptionKind")
        _validate_duration(self.elapsed_seconds, "elapsed_seconds", allow_zero=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class FocusCompleted(EventMetadata):
    """Closes a focus block with its measured active duration."""

    KIND: ClassVar[str] = "focus_completed"

    elapsed_seconds: int

    def __post_init__(self) -> None:
        super(FocusCompleted, self).__post_init__()
        _validate_duration(self.elapsed_seconds, "elapsed_seconds")


@dataclass(frozen=True, slots=True, kw_only=True)
class BreakStarted(EventMetadata):
    """Marks the transition into the planned recovery period."""

    KIND: ClassVar[str] = "break_started"


@dataclass(frozen=True, slots=True, kw_only=True)
class BreakCompleted(EventMetadata):
    """Closes a session with its measured break duration."""

    KIND: ClassVar[str] = "break_completed"

    elapsed_seconds: int

    def __post_init__(self) -> None:
        super(BreakCompleted, self).__post_init__()
        _validate_duration(self.elapsed_seconds, "elapsed_seconds")


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionAbandoned(EventMetadata):
    """Terminates a session without collecting a free-form explanation."""

    KIND: ClassVar[str] = "session_abandoned"

    reason: AbandonReason

    def __post_init__(self) -> None:
        super(SessionAbandoned, self).__post_init__()
        if not isinstance(self.reason, AbandonReason):
            raise InvalidEvent("reason must be an AbandonReason")


DomainEvent: TypeAlias = (
    SessionPlanned
    | FocusStarted
    | InterruptionRecorded
    | FocusCompleted
    | BreakStarted
    | BreakCompleted
    | SessionAbandoned
)


@dataclass(frozen=True, slots=True)
class SessionState:
    """Projection obtained solely by replaying a session's events."""

    session_id: UUID
    revision: int
    phase: SessionPhase
    objective: str
    policy_id: str
    target_focus_seconds: int
    target_break_seconds: int
    interruption_count: int
    interruption_seconds: int
    actual_focus_seconds: int | None
    actual_break_seconds: int | None
    abandon_reason: AbandonReason | None
    last_occurred_at: datetime

    @property
    def is_terminal(self) -> bool:
        """Return whether no further domain events may be applied."""

        return self.phase in {SessionPhase.COMPLETED, SessionPhase.ABANDONED}

    @property
    def focus_completion_ratio(self) -> float | None:
        """Return actual-to-target focus time once the block has closed."""

        if self.actual_focus_seconds is None:
            return None
        return self.actual_focus_seconds / self.target_focus_seconds


def _ensure_contiguous(state: SessionState, event: DomainEvent) -> None:
    if event.session_id != state.session_id:
        raise InvalidTransition("event belongs to a different session")
    expected_sequence = state.revision + 1
    if event.sequence != expected_sequence:
        raise InvalidTransition(
            f"expected sequence {expected_sequence}, received {event.sequence}"
        )
    if event.occurred_at < state.last_occurred_at:
        raise InvalidTransition("event timestamp precedes the current projection")
    if state.is_terminal:
        raise InvalidTransition(f"session is already {state.phase.value}")


def apply_event(
    state: SessionState | None,
    event: DomainEvent,
) -> SessionState:
    """Apply one event, rejecting invalid order or state transitions."""

    if state is None:
        if not isinstance(event, SessionPlanned):
            raise InvalidTransition("the first event must plan the session")
        if event.sequence != 1:
            raise InvalidTransition("the first event must have sequence 1")
        return SessionState(
            session_id=event.session_id,
            revision=event.sequence,
            phase=SessionPhase.PLANNED,
            objective=event.objective,
            policy_id=event.policy_id,
            target_focus_seconds=event.target_focus_seconds,
            target_break_seconds=event.target_break_seconds,
            interruption_count=0,
            interruption_seconds=0,
            actual_focus_seconds=None,
            actual_break_seconds=None,
            abandon_reason=None,
            last_occurred_at=event.occurred_at,
        )

    _ensure_contiguous(state, event)
    if isinstance(event, SessionAbandoned):
        return replace(
            state,
            revision=event.sequence,
            last_occurred_at=event.occurred_at,
            phase=SessionPhase.ABANDONED,
            abandon_reason=event.reason,
        )

    if isinstance(event, FocusStarted) and state.phase is SessionPhase.PLANNED:
        return replace(
            state,
            revision=event.sequence,
            last_occurred_at=event.occurred_at,
            phase=SessionPhase.FOCUSING,
        )

    if isinstance(event, InterruptionRecorded) and state.phase is SessionPhase.FOCUSING:
        return replace(
            state,
            revision=event.sequence,
            last_occurred_at=event.occurred_at,
            interruption_count=state.interruption_count + 1,
            interruption_seconds=state.interruption_seconds + event.elapsed_seconds,
        )

    if isinstance(event, FocusCompleted) and state.phase is SessionPhase.FOCUSING:
        return replace(
            state,
            revision=event.sequence,
            last_occurred_at=event.occurred_at,
            phase=SessionPhase.FOCUS_COMPLETE,
            actual_focus_seconds=event.elapsed_seconds,
        )

    if isinstance(event, BreakStarted) and state.phase is SessionPhase.FOCUS_COMPLETE:
        return replace(
            state,
            revision=event.sequence,
            last_occurred_at=event.occurred_at,
            phase=SessionPhase.BREAKING,
        )

    if isinstance(event, BreakCompleted) and state.phase is SessionPhase.BREAKING:
        return replace(
            state,
            revision=event.sequence,
            last_occurred_at=event.occurred_at,
            phase=SessionPhase.COMPLETED,
            actual_break_seconds=event.elapsed_seconds,
        )

    raise InvalidTransition(
        f"{type(event).__name__} cannot follow phase {state.phase.value}"
    )


def reduce_events(events: Iterable[DomainEvent]) -> SessionState:
    """Replay a non-empty event stream into its final projection."""

    state: SessionState | None = None
    for event in events:
        state = apply_event(state, event)
    if state is None:
        raise InvalidTransition("cannot reduce an empty event stream")
    return state
