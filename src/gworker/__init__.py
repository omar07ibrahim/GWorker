"""Local-first primitives for auditable focus experiments."""

from .domain import (
    AbandonReason,
    BreakCompleted,
    BreakStarted,
    DomainEvent,
    FocusCompleted,
    FocusStarted,
    InterruptionKind,
    InterruptionRecorded,
    InvalidEvent,
    InvalidTransition,
    SessionAbandoned,
    SessionPhase,
    SessionPlanned,
    SessionState,
    apply_event,
    reduce_events,
)

__all__ = [
    "AbandonReason",
    "BreakCompleted",
    "BreakStarted",
    "DomainEvent",
    "FocusCompleted",
    "FocusStarted",
    "InterruptionKind",
    "InterruptionRecorded",
    "InvalidEvent",
    "InvalidTransition",
    "SessionAbandoned",
    "SessionPhase",
    "SessionPlanned",
    "SessionState",
    "apply_event",
    "reduce_events",
]
