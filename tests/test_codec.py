from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID

from gworker.codec import EventCodecError, decode_event, encode_event
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
    SessionPlanned,
)

SESSION_ID = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e1000")
BASE_TIME = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)


def metadata(sequence: int) -> dict[str, object]:
    return {
        "event_id": UUID(f"018f4f69-e7a2-7f84-8c2d-{sequence:012d}"),
        "session_id": SESSION_ID,
        "sequence": sequence,
        "occurred_at": BASE_TIME + timedelta(seconds=sequence),
    }


def all_events() -> list[DomainEvent]:
    return [
        SessionPlanned(
            **metadata(1),
            objective="Test the canonical codec",
            target_focus_seconds=1_500,
            target_break_seconds=300,
            policy_id="fixed-baseline-v1",
        ),
        FocusStarted(**metadata(2)),
        InterruptionRecorded(
            **metadata(3),
            kind=InterruptionKind.NOTIFICATION,
            elapsed_seconds=12,
        ),
        FocusCompleted(**metadata(4), elapsed_seconds=1_480),
        BreakStarted(**metadata(5)),
        BreakCompleted(**metadata(6), elapsed_seconds=290),
        SessionAbandoned(
            **metadata(7),
            reason=AbandonReason.PRIORITY_CHANGED,
        ),
    ]


class EventCodecTests(unittest.TestCase):
    def test_round_trips_every_v1_event(self) -> None:
        for event in all_events():
            with self.subTest(event=event.KIND):
                encoded = encode_event(event)
                self.assertEqual(decode_event(encoded), event)
                self.assertEqual(decode_event(encoded.encode()), event)

    def test_encoding_is_canonical_and_unicode_safe(self) -> None:
        event = SessionPlanned(
            **metadata(1),
            objective="Разобрать журнал",
            target_focus_seconds=900,
            target_break_seconds=180,
            policy_id="fixed-baseline-v1",
        )
        encoded = encode_event(event)
        self.assertIn("Разобрать журнал", encoded)
        self.assertNotIn('": ', encoded)
        self.assertNotIn(", ", encoded)
        self.assertEqual(encoded, encode_event(decode_event(encoded)))

    def test_rejects_invalid_json_and_utf8(self) -> None:
        for document in ("{", "NaN", b"\xff"):
            with self.subTest(document=document), self.assertRaises(EventCodecError):
                decode_event(document)

    def test_rejects_duplicate_json_fields(self) -> None:
        encoded = encode_event(all_events()[0])
        duplicated = encoded.replace(
            '"schema_version":1',
            '"schema_version":1,"schema_version":1',
        )
        with self.assertRaisesRegex(EventCodecError, "duplicate JSON field"):
            decode_event(duplicated)

    def test_wraps_oversized_integer_parser_failure(self) -> None:
        encoded = encode_event(all_events()[0])
        oversized = encoded.replace('"sequence":1', f'"sequence":{"9" * 5_000}')
        with self.assertRaisesRegex(EventCodecError, "JSON parser limits"):
            decode_event(oversized)

    def test_rejects_missing_and_unknown_top_level_fields(self) -> None:
        document = json.loads(encode_event(all_events()[0]))
        del document["event_id"]
        with self.assertRaisesRegex(EventCodecError, "missing=event_id"):
            decode_event(json.dumps(document))

        document = json.loads(encode_event(all_events()[0]))
        document["unexpected"] = True
        with self.assertRaisesRegex(EventCodecError, "unknown=unexpected"):
            decode_event(json.dumps(document))

    def test_rejects_unknown_schema_and_event_type(self) -> None:
        document = json.loads(encode_event(all_events()[0]))
        document["schema_version"] = 2
        with self.assertRaisesRegex(EventCodecError, "schema_version: 2"):
            decode_event(json.dumps(document))

        document = json.loads(encode_event(all_events()[0]))
        document["event_type"] = "future_event"
        with self.assertRaisesRegex(EventCodecError, "event_type: future_event"):
            decode_event(json.dumps(document))

    def test_rejects_noncanonical_uuid_and_timestamp(self) -> None:
        document = json.loads(encode_event(all_events()[0]))
        document["event_id"] = document["event_id"].upper()
        with self.assertRaisesRegex(EventCodecError, "canonical lowercase UUID"):
            decode_event(json.dumps(document))

        document = json.loads(encode_event(all_events()[0]))
        document["event_id"] = "not-a-uuid"
        with self.assertRaisesRegex(EventCodecError, "event_id must be a UUID"):
            decode_event(json.dumps(document))

        document = json.loads(encode_event(all_events()[0]))
        document["occurred_at"] = "2026-07-24T09:00:01Z"
        with self.assertRaisesRegex(EventCodecError, "six fractional digits"):
            decode_event(json.dumps(document))

        document["occurred_at"] = "2026-07-24T09:00:01.000000+00:00"
        with self.assertRaisesRegex(EventCodecError, "canonical UTC text"):
            decode_event(json.dumps(document))

        document["occurred_at"] = "not-a-timeZ"
        with self.assertRaisesRegex(EventCodecError, "ISO 8601 timestamp"):
            decode_event(json.dumps(document))

    def test_rejects_wrong_payload_shape_and_types(self) -> None:
        document = json.loads(encode_event(all_events()[0]))
        document["payload"]["extra"] = "surprise"
        with self.assertRaisesRegex(EventCodecError, "unknown=extra"):
            decode_event(json.dumps(document))

        document = json.loads(encode_event(all_events()[0]))
        document["sequence"] = True
        with self.assertRaisesRegex(EventCodecError, "sequence must be an integer"):
            decode_event(json.dumps(document))

        document = json.loads(encode_event(all_events()[0]))
        document["payload"] = []
        with self.assertRaisesRegex(EventCodecError, "payload must be a JSON object"):
            decode_event(json.dumps(document))

        document = json.loads(encode_event(all_events()[0]))
        document["event_type"] = 7
        with self.assertRaisesRegex(EventCodecError, "event_type must be a string"):
            decode_event(json.dumps(document))

    def test_rejects_unknown_enum_values_and_non_text_input(self) -> None:
        document = json.loads(encode_event(all_events()[2]))
        document["payload"]["kind"] = "screen_scrape"
        with self.assertRaisesRegex(EventCodecError, "unsupported value"):
            decode_event(json.dumps(document))

        with self.assertRaisesRegex(EventCodecError, "text or bytes"):
            decode_event(42)  # type: ignore[arg-type]

    def test_wraps_domain_validation_errors(self) -> None:
        document = json.loads(encode_event(all_events()[0]))
        document["payload"]["target_focus_seconds"] = 0
        with self.assertRaisesRegex(EventCodecError, "invalid session_planned"):
            decode_event(json.dumps(document))


if __name__ == "__main__":
    unittest.main()
