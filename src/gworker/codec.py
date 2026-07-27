"""Canonical wire format for GWorker domain events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import UUID

from .domain import (
    AbandonReason,
    BreakCompleted,
    BreakStarted,
    DomainEvent,
    EventMetadata,
    FocusCompleted,
    FocusStarted,
    InterruptionKind,
    InterruptionRecorded,
    InvalidEvent,
    SessionAbandoned,
    SessionPlanned,
)

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "event_type",
        "event_id",
        "session_id",
        "sequence",
        "occurred_at",
        "payload",
    }
)


class EventCodecError(ValueError):
    """Raised when an event document is malformed or unsupported."""


def _timestamp_text(value: datetime) -> str:
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _payload(event: DomainEvent) -> dict[str, object]:
    if isinstance(event, SessionPlanned):
        return {
            "objective": event.objective,
            "target_focus_seconds": event.target_focus_seconds,
            "target_break_seconds": event.target_break_seconds,
            "policy_id": event.policy_id,
        }
    if isinstance(event, InterruptionRecorded):
        return {
            "kind": event.kind.value,
            "elapsed_seconds": event.elapsed_seconds,
        }
    if isinstance(event, (FocusCompleted, BreakCompleted)):
        return {"elapsed_seconds": event.elapsed_seconds}
    if isinstance(event, SessionAbandoned):
        return {"reason": event.reason.value}
    if isinstance(event, (FocusStarted, BreakStarted)):
        return {}
    raise EventCodecError(f"unsupported event class: {type(event).__name__}")


def event_document(event: DomainEvent) -> dict[str, object]:
    """Return the canonical, JSON-compatible event document."""

    return {
        "schema_version": EventMetadata.SCHEMA_VERSION,
        "event_type": event.KIND,
        "event_id": str(event.event_id),
        "session_id": str(event.session_id),
        "sequence": event.sequence,
        "occurred_at": _timestamp_text(event.occurred_at),
        "payload": _payload(event),
    }


def encode_event(event: DomainEvent) -> str:
    """Encode an event as deterministic UTF-8-safe JSON text."""

    return json.dumps(
        event_document(event),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_json_constant(value: str) -> None:
    raise EventCodecError(f"invalid JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise EventCodecError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _object(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise EventCodecError(f"{location} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise EventCodecError(f"{location} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    location: str,
) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing={','.join(missing)}")
    if unknown:
        details.append(f"unknown={','.join(unknown)}")
    raise EventCodecError(f"{location} has invalid fields ({'; '.join(details)})")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise EventCodecError(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EventCodecError(f"{field} must be an integer")
    return value


def _uuid(value: object, field: str) -> UUID:
    text = _string(value, field)
    try:
        parsed = UUID(text)
    except ValueError as exc:
        raise EventCodecError(f"{field} must be a UUID") from exc
    if str(parsed) != text:
        raise EventCodecError(f"{field} must use canonical lowercase UUID text")
    return parsed


def _timestamp(value: object) -> datetime:
    text = _string(value, "occurred_at")
    if not text.endswith("Z"):
        raise EventCodecError("occurred_at must use canonical UTC text")
    try:
        parsed = datetime.fromisoformat(f"{text[:-1]}+00:00")
    except ValueError as exc:
        raise EventCodecError("occurred_at must be an ISO 8601 timestamp") from exc
    if _timestamp_text(parsed) != text:
        raise EventCodecError("occurred_at must include six fractional digits")
    return parsed


EnumValue = TypeVar("EnumValue", InterruptionKind, AbandonReason)


def _enum_value(
    enum_type: type[EnumValue],
    value: object,
    field: str,
) -> EnumValue:
    text = _string(value, field)
    try:
        return enum_type(text)
    except ValueError as exc:
        raise EventCodecError(f"{field} has unsupported value: {text}") from exc


def _metadata(document: Mapping[str, object]) -> dict[str, Any]:
    return {
        "event_id": _uuid(document["event_id"], "event_id"),
        "session_id": _uuid(document["session_id"], "session_id"),
        "sequence": _integer(document["sequence"], "sequence"),
        "occurred_at": _timestamp(document["occurred_at"]),
    }


def _decode_known_event(
    event_type: str,
    metadata: dict[str, Any],
    payload: Mapping[str, object],
) -> DomainEvent:
    if event_type == SessionPlanned.KIND:
        expected = frozenset(
            {
                "objective",
                "target_focus_seconds",
                "target_break_seconds",
                "policy_id",
            }
        )
        _exact_keys(payload, expected, "payload")
        return SessionPlanned(
            **metadata,
            objective=_string(payload["objective"], "payload.objective"),
            target_focus_seconds=_integer(
                payload["target_focus_seconds"],
                "payload.target_focus_seconds",
            ),
            target_break_seconds=_integer(
                payload["target_break_seconds"],
                "payload.target_break_seconds",
            ),
            policy_id=_string(payload["policy_id"], "payload.policy_id"),
        )

    if event_type == FocusStarted.KIND:
        _exact_keys(payload, frozenset(), "payload")
        return FocusStarted(**metadata)

    if event_type == InterruptionRecorded.KIND:
        _exact_keys(payload, frozenset({"kind", "elapsed_seconds"}), "payload")
        return InterruptionRecorded(
            **metadata,
            kind=_enum_value(
                InterruptionKind,
                payload["kind"],
                "payload.kind",
            ),
            elapsed_seconds=_integer(
                payload["elapsed_seconds"],
                "payload.elapsed_seconds",
            ),
        )

    if event_type == FocusCompleted.KIND:
        _exact_keys(payload, frozenset({"elapsed_seconds"}), "payload")
        return FocusCompleted(
            **metadata,
            elapsed_seconds=_integer(
                payload["elapsed_seconds"],
                "payload.elapsed_seconds",
            ),
        )

    if event_type == BreakStarted.KIND:
        _exact_keys(payload, frozenset(), "payload")
        return BreakStarted(**metadata)

    if event_type == BreakCompleted.KIND:
        _exact_keys(payload, frozenset({"elapsed_seconds"}), "payload")
        return BreakCompleted(
            **metadata,
            elapsed_seconds=_integer(
                payload["elapsed_seconds"],
                "payload.elapsed_seconds",
            ),
        )

    if event_type == SessionAbandoned.KIND:
        _exact_keys(payload, frozenset({"reason"}), "payload")
        return SessionAbandoned(
            **metadata,
            reason=_enum_value(
                AbandonReason,
                payload["reason"],
                "payload.reason",
            ),
        )

    raise EventCodecError(f"unsupported event_type: {event_type}")


def decode_event(value: str | bytes) -> DomainEvent:
    """Decode one exact v1 event document and run domain validation."""

    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
    except UnicodeDecodeError as exc:
        raise EventCodecError("event document must be UTF-8") from exc
    if not isinstance(text, str):
        raise EventCodecError("event document must be text or bytes")

    try:
        raw = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except EventCodecError:
        raise
    except json.JSONDecodeError as exc:
        raise EventCodecError("event document must be valid JSON") from exc
    except (RecursionError, ValueError) as exc:
        raise EventCodecError("event document exceeds JSON parser limits") from exc

    document = _object(raw, "event")
    _exact_keys(document, TOP_LEVEL_KEYS, "event")

    schema_version = _integer(document["schema_version"], "schema_version")
    if schema_version != EventMetadata.SCHEMA_VERSION:
        raise EventCodecError(f"unsupported schema_version: {schema_version}")

    event_type = _string(document["event_type"], "event_type")
    payload = _object(document["payload"], "payload")
    try:
        return _decode_known_event(
            event_type,
            _metadata(document),
            payload,
        )
    except InvalidEvent as exc:
        raise EventCodecError(f"invalid {event_type} event: {exc}") from exc
