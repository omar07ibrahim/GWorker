from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

from gworker.domain import (
    MAX_DURATION_SECONDS,
    AbandonReason,
    BreakCompleted,
    BreakStarted,
    FocusCompleted,
    FocusStarted,
    InterruptionKind,
    InterruptionRecorded,
    InvalidEvent,
    InvalidTransition,
    SessionAbandoned,
    SessionPhase,
    SessionPlanned,
    apply_event,
    reduce_events,
)

SESSION_ID = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e1000")
BASE_TIME = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)


def event_id(sequence: int) -> UUID:
    return UUID(f"018f4f69-e7a2-7f84-8c2d-{sequence:012d}")


def planned(**overrides: object) -> SessionPlanned:
    values: dict[str, object] = {
        "event_id": event_id(1),
        "session_id": SESSION_ID,
        "sequence": 1,
        "occurred_at": BASE_TIME,
        "objective": "Review the event reducer",
        "target_focus_seconds": 1_500,
        "target_break_seconds": 300,
        "policy_id": "fixed-baseline-v1",
    }
    values.update(overrides)
    return SessionPlanned(**values)  # type: ignore[arg-type]


def metadata(
    sequence: int,
    *,
    minute: int | None = None,
    **overrides: object,
) -> dict[str, object]:
    values: dict[str, object] = {
        "event_id": event_id(sequence),
        "session_id": SESSION_ID,
        "sequence": sequence,
        "occurred_at": BASE_TIME + timedelta(minutes=minute or sequence - 1),
    }
    values.update(overrides)
    return values


def focus_started(sequence: int = 2) -> FocusStarted:
    return FocusStarted(**metadata(sequence))  # type: ignore[arg-type]


class EventValidationTests(unittest.TestCase):
    def test_event_requires_non_nil_uuid(self) -> None:
        with self.assertRaisesRegex(InvalidEvent, "event_id must not be the nil UUID"):
            planned(event_id=UUID(int=0))

    def test_event_requires_positive_integer_sequence(self) -> None:
        with self.assertRaisesRegex(InvalidEvent, "sequence must be positive"):
            planned(sequence=0)
        with self.assertRaisesRegex(InvalidEvent, "sequence must be an integer"):
            planned(sequence=True)

    def test_event_requires_utc_timestamp(self) -> None:
        with self.assertRaisesRegex(InvalidEvent, "timezone-aware UTC"):
            planned(occurred_at=BASE_TIME.replace(tzinfo=None))
        with self.assertRaisesRegex(InvalidEvent, "timezone-aware UTC"):
            planned(occurred_at=BASE_TIME.astimezone(timezone(timedelta(hours=2))))

    def test_plan_rejects_private_multiline_or_padded_objective(self) -> None:
        for objective in (
            "",
            " padded",
            "padded ",
            "first\nsecond",
            "nul\x00byte",
            "right\u202eto-left",
        ):
            with self.subTest(objective=objective), self.assertRaises(InvalidEvent):
                planned(objective=objective)

    def test_plan_rejects_invalid_durations(self) -> None:
        for duration in (0, -1, True, 1.5, MAX_DURATION_SECONDS + 1):
            with self.subTest(duration=duration), self.assertRaises(InvalidEvent):
                planned(target_focus_seconds=duration)

    def test_plan_rejects_empty_or_padded_policy_identifier(self) -> None:
        for policy_id in (
            "",
            " fixed",
            "fixed ",
            "fixed\nv1",
            "fixed\x00v1",
            "fixed\u2066v1",
            "UPPERCASE",
        ):
            with self.subTest(policy_id=policy_id), self.assertRaises(InvalidEvent):
                planned(policy_id=policy_id)

    def test_interruption_requires_enum_and_allows_zero_elapsed_time(self) -> None:
        event = InterruptionRecorded(
            **metadata(3),
            kind=InterruptionKind.NOTIFICATION,
            elapsed_seconds=0,
        )
        self.assertEqual(event.elapsed_seconds, 0)
        with self.assertRaisesRegex(InvalidEvent, "InterruptionKind"):
            InterruptionRecorded(
                **metadata(3),
                kind="notification",  # type: ignore[arg-type]
                elapsed_seconds=10,
            )

    def test_completion_requires_a_positive_bounded_duration(self) -> None:
        with self.assertRaisesRegex(InvalidEvent, "between 1"):
            FocusCompleted(**metadata(3), elapsed_seconds=0)
        with self.assertRaisesRegex(InvalidEvent, "between 1"):
            BreakCompleted(**metadata(5), elapsed_seconds=MAX_DURATION_SECONDS + 1)

    def test_abandonment_requires_bounded_reason(self) -> None:
        with self.assertRaisesRegex(InvalidEvent, "AbandonReason"):
            SessionAbandoned(
                **metadata(2),
                reason="other",  # type: ignore[arg-type]
            )


class ReducerTests(unittest.TestCase):
    def test_replays_a_completed_session(self) -> None:
        events = [
            planned(),
            focus_started(),
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
        ]

        state = reduce_events(events)

        self.assertEqual(state.phase, SessionPhase.COMPLETED)
        self.assertEqual(state.revision, 7)
        self.assertEqual(state.interruption_count, 2)
        self.assertEqual(state.interruption_seconds, 60)
        self.assertEqual(state.actual_focus_seconds, 1_470)
        self.assertEqual(state.actual_break_seconds, 280)
        self.assertAlmostEqual(state.focus_completion_ratio or 0, 0.98)
        self.assertTrue(state.is_terminal)

    def test_rejects_empty_stream(self) -> None:
        with self.assertRaisesRegex(InvalidTransition, "empty event stream"):
            reduce_events([])

    def test_first_event_must_be_plan_with_sequence_one(self) -> None:
        with self.assertRaisesRegex(InvalidTransition, "first event must plan"):
            apply_event(None, focus_started())
        with self.assertRaisesRegex(InvalidTransition, "sequence 1"):
            apply_event(None, planned(sequence=2))

    def test_rejects_sequence_gap_and_duplicate(self) -> None:
        state = apply_event(None, planned())
        with self.assertRaisesRegex(InvalidTransition, "expected sequence 2"):
            apply_event(state, FocusStarted(**metadata(3)))

        state = apply_event(state, focus_started())
        with self.assertRaisesRegex(InvalidTransition, "expected sequence 3"):
            apply_event(state, FocusStarted(**metadata(2)))

    def test_rejects_duplicate_event_id_at_a_new_revision(self) -> None:
        state = apply_event(None, planned())
        with self.assertRaisesRegex(InvalidTransition, "event_id has already"):
            apply_event(
                state,
                FocusStarted(
                    **metadata(2, event_id=planned().event_id),
                ),
            )

    def test_rejects_cross_session_event(self) -> None:
        state = apply_event(None, planned())
        other_session = UUID("018f4f69-e7a2-7f84-8c2d-9f531c4e2000")
        with self.assertRaisesRegex(InvalidTransition, "different session"):
            apply_event(
                state,
                FocusStarted(**{**metadata(2), "session_id": other_session}),
            )

    def test_rejects_timestamp_regression(self) -> None:
        state = apply_event(None, planned())
        with self.assertRaisesRegex(InvalidTransition, "timestamp precedes"):
            apply_event(
                state,
                FocusStarted(
                    **metadata(
                        2,
                        occurred_at=BASE_TIME - timedelta(seconds=1),
                    )
                ),
            )

    def test_rejects_invalid_phase_transition(self) -> None:
        state = apply_event(None, planned())
        with self.assertRaisesRegex(
            InvalidTransition,
            "BreakStarted cannot follow phase planned",
        ):
            apply_event(state, BreakStarted(**metadata(2)))

    def test_interruption_is_only_valid_while_focusing(self) -> None:
        state = reduce_events(
            [
                planned(),
                focus_started(),
                FocusCompleted(**metadata(3), elapsed_seconds=1_500),
            ]
        )
        with self.assertRaisesRegex(
            InvalidTransition,
            "InterruptionRecorded cannot follow phase focus_complete",
        ):
            apply_event(
                state,
                InterruptionRecorded(
                    **metadata(4),
                    kind=InterruptionKind.INTERNAL,
                    elapsed_seconds=2,
                ),
            )

    def test_abandonment_is_terminal_from_each_nonterminal_phase(self) -> None:
        event_streams = [
            [planned()],
            [planned(), focus_started()],
            [
                planned(),
                focus_started(),
                FocusCompleted(**metadata(3), elapsed_seconds=1_200),
            ],
            [
                planned(),
                focus_started(),
                FocusCompleted(**metadata(3), elapsed_seconds=1_200),
                BreakStarted(**metadata(4)),
            ],
        ]

        for events in event_streams:
            with self.subTest(phase=reduce_events(events).phase):
                state = reduce_events(events)
                abandonment = SessionAbandoned(
                    **metadata(state.revision + 1),
                    reason=AbandonReason.PRIORITY_CHANGED,
                )
                state = apply_event(state, abandonment)
                self.assertEqual(state.phase, SessionPhase.ABANDONED)
                self.assertEqual(
                    state.abandon_reason,
                    AbandonReason.PRIORITY_CHANGED,
                )
                self.assertTrue(state.is_terminal)
                with self.assertRaisesRegex(InvalidTransition, "already abandoned"):
                    apply_event(
                        state,
                        SessionAbandoned(
                            **metadata(state.revision + 1),
                            reason=AbandonReason.OTHER,
                        ),
                    )

    def test_equal_timestamps_are_valid_for_coarse_clocks(self) -> None:
        state = apply_event(None, planned())
        state = apply_event(
            state,
            FocusStarted(**metadata(2, occurred_at=BASE_TIME)),
        )
        self.assertEqual(state.phase, SessionPhase.FOCUSING)

    def test_ratio_is_unavailable_before_focus_completion(self) -> None:
        state = reduce_events([planned(), focus_started()])
        self.assertIsNone(state.focus_completion_ratio)
        self.assertFalse(state.is_terminal)

    def test_state_rejects_forged_invariants(self) -> None:
        focusing = reduce_events([planned(), focus_started()])
        invalid_changes = [
            {"revision": -1},
            {"event_ids": frozenset()},
            {"interruption_count": -1},
            {"interruption_seconds": 10},
            {"actual_break_seconds": 10},
            {"abandon_reason": AbandonReason.OTHER},
            {"policy_id": "INVALID"},
            {
                "phase": SessionPhase.PLANNED,
                "revision": 2,
            },
            {
                "phase": SessionPhase.PLANNED,
                "interruption_count": 1,
                "interruption_seconds": 10,
            },
        ]
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(InvalidEvent):
                replace(focusing, **changes)

        completed = reduce_events(
            [
                planned(),
                focus_started(),
                FocusCompleted(**metadata(3), elapsed_seconds=1_500),
                BreakStarted(**metadata(4)),
                BreakCompleted(**metadata(5), elapsed_seconds=300),
            ]
        )
        with self.assertRaisesRegex(InvalidEvent, "break duration is required"):
            replace(completed, actual_break_seconds=None)

        planned_state = reduce_events([planned()])
        with self.assertRaisesRegex(InvalidEvent, "revision is inconsistent"):
            replace(
                planned_state,
                phase=SessionPhase.FOCUS_COMPLETE,
                actual_focus_seconds=1,
            )

        abandoned_from_plan = reduce_events(
            [
                planned(),
                SessionAbandoned(
                    **metadata(2),
                    reason=AbandonReason.OTHER,
                ),
            ]
        )
        for impossible_revision in (4, 5):
            forged_ids = abandoned_from_plan.event_ids | {
                UUID(f"018f4f69-e7a2-7f84-8c2d-{suffix:012d}")
                for suffix in range(10, 10 + impossible_revision - 2)
            }
            with (
                self.subTest(revision=impossible_revision),
                self.assertRaisesRegex(InvalidEvent, "abandonment history"),
            ):
                replace(
                    abandoned_from_plan,
                    revision=impossible_revision,
                    event_ids=forged_ids,
                )


if __name__ == "__main__":
    unittest.main()
