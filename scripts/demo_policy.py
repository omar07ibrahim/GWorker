#!/usr/bin/env python3
"""Deterministic, synthetic demonstration of the production policy API."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from gworker.policy import (
    POLICY_FAMILY,
    DurationFit,
    EnergyLevel,
    FocusContext,
    HierarchicalSoftmaxUCB,
    Recommendation,
    ReviewedDecision,
    TaskKind,
)

SCHEMA_VERSION = "gworker-policy-demo-v1"
FIXTURE_NAME = "verified-12-review-synthetic-scenario"
REVIEW_COUNT = 12
NEXT_DECISION_SEQUENCE = REVIEW_COUNT + 1
RNG_SEED_BASE = 20_260_000
TARGET_FOCUS_SECONDS = 40 * 60
AVAILABLE_SECONDS = 60 * 60


def _decision_id(sequence: int) -> UUID:
    return uuid5(NAMESPACE_URL, f"gworker-demo-policy-v1:{sequence}")


def _feedback(recommendation: Recommendation) -> tuple[DurationFit, bool]:
    focus_seconds = recommendation.template.focus_seconds
    if focus_seconds < TARGET_FOCUS_SECONDS:
        fit = DurationFit.TOO_SHORT
    elif focus_seconds == TARGET_FOCUS_SECONDS:
        fit = DurationFit.JUST_RIGHT
    else:
        fit = DurationFit.TOO_LONG
    return fit, focus_seconds >= TARGET_FOCUS_SECONDS


def _context(previous_focus_seconds: int | None) -> FocusContext:
    return FocusContext(
        task_kind=TaskKind.DEEP_WORK,
        energy=EnergyLevel.MEDIUM,
        available_seconds=AVAILABLE_SECONDS,
        previous_focus_seconds=previous_focus_seconds,
    )


def _context_document(context: FocusContext) -> dict[str, object]:
    return {
        "available_seconds": context.available_seconds,
        "energy": context.energy.value,
        "previous_focus_seconds": context.previous_focus_seconds,
        "task_kind": context.task_kind.value,
    }


def _arm_documents(recommendation: Recommendation) -> list[dict[str, object]]:
    return [
        {
            "break_seconds": arm.template.break_seconds,
            "directional_adjustment": arm.directional_adjustment,
            "exploration_bonus": arm.exploration_bonus,
            "focus_seconds": arm.template.focus_seconds,
            "posterior_mean": arm.posterior_mean,
            "probability": arm.probability,
            "review_count": arm.review_count,
            "score": arm.score,
            "template_id": arm.template.template_id,
        }
        for arm in recommendation.arm_scores
    ]


def _recommendation_document(
    recommendation: Recommendation,
) -> dict[str, object]:
    return {
        "arm_scores": _arm_documents(recommendation),
        "break_seconds": recommendation.template.break_seconds,
        "evidence_bucket": recommendation.bucket.value,
        "evidence_bucket_label": recommendation.bucket_label,
        "evidence_count": recommendation.evidence_count,
        "focus_seconds": recommendation.template.focus_seconds,
        "propensity": recommendation.propensity,
        "reason_codes": list(recommendation.reason_codes),
        "template_id": recommendation.template.template_id,
        "total_seconds": recommendation.template.total_seconds,
    }


def _review_is_bound(
    recommendation: Recommendation,
    review: ReviewedDecision,
) -> bool:
    return (
        review.decision_id == recommendation.decision_id
        and review.decision_sequence == recommendation.decision_sequence
        and review.policy_id == recommendation.policy_id
        and review.context == recommendation.context
        and review.template_id == recommendation.template.template_id
        and review.propensity == recommendation.propensity
    )


def build_demo() -> dict[str, object]:
    """Run the fixed 12-review scenario through the real policy."""

    policy = HierarchicalSoftmaxUCB()
    history: list[ReviewedDecision] = []
    decisions: list[dict[str, object]] = []
    previous_focus_seconds: int | None = None
    provenance_checks: list[bool] = []

    for sequence in range(1, REVIEW_COUNT + 1):
        context = _context(previous_focus_seconds)
        rng_seed = RNG_SEED_BASE + sequence
        recommendation = policy.recommend(
            context,
            history,
            decision_id=_decision_id(sequence),
            decision_sequence=sequence,
            rng=random.Random(rng_seed),
        )
        fit, objective_completed = _feedback(recommendation)
        review = recommendation.review(
            fit=fit,
            objective_completed=objective_completed,
        )
        provenance_checks.append(_review_is_bound(recommendation, review))
        history.append(review)
        previous_focus_seconds = recommendation.template.focus_seconds
        decisions.append(
            {
                "context": _context_document(context),
                "decision_id": str(recommendation.decision_id),
                "decision_sequence": sequence,
                "recommendation": _recommendation_document(recommendation),
                "review": {
                    "fit": review.fit.value,
                    "objective_completed": review.objective_completed,
                    "reward": review.reward,
                },
                "rng_seed": rng_seed,
            }
        )

    next_context = _context(previous_focus_seconds)
    next_seed = RNG_SEED_BASE + NEXT_DECISION_SEQUENCE
    next_recommendation = policy.recommend(
        next_context,
        history,
        decision_id=_decision_id(NEXT_DECISION_SEQUENCE),
        decision_sequence=NEXT_DECISION_SEQUENCE,
        rng=random.Random(next_seed),
    )
    probability_sum = math.fsum(
        arm.probability for arm in next_recommendation.arm_scores
    )
    if not all(provenance_checks):
        raise RuntimeError("review provenance verification failed")
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("arm probabilities do not sum to one")

    selected_counts = Counter(
        str(decision["recommendation"]["template_id"])  # type: ignore[index]
        for decision in decisions
    )
    review_counts = Counter(
        str(decision["review"]["fit"])  # type: ignore[index]
        for decision in decisions
    )

    return {
        "decisions": decisions,
        "fixture_kind": "deterministic-synthetic-policy-demo",
        "fixture_name": FIXTURE_NAME,
        "fixture_notice": (
            "Synthetic API demonstration only; not locked-evaluation evidence."
        ),
        "next_recommendation": {
            "context": _context_document(next_context),
            "decision_id": str(next_recommendation.decision_id),
            "decision_sequence": NEXT_DECISION_SEQUENCE,
            "recommendation": _recommendation_document(next_recommendation),
            "rng_seed": next_seed,
        },
        "policy": {
            "family": POLICY_FAMILY,
            "policy_id": policy.policy_id,
            "templates": [
                {
                    "break_seconds": template.break_seconds,
                    "focus_seconds": template.focus_seconds,
                    "template_id": template.template_id,
                    "total_seconds": template.total_seconds,
                }
                for template in policy.templates
            ],
        },
        "review_summary": {
            "feedback_counts": dict(sorted(review_counts.items())),
            "review_count": len(history),
            "selected_template_counts": dict(sorted(selected_counts.items())),
            "target_focus_seconds": TARGET_FOCUS_SECONDS,
        },
        "schema_version": SCHEMA_VERSION,
        "verification": {
            "all_reviews_bound_to_recommendations": all(provenance_checks),
            "arm_probability_sum": probability_sum,
            "fixed_rng_seeds": True,
            "fixed_uuid_namespace": str(NAMESPACE_URL),
        },
    }


def canonical_json(document: dict[str, object], *, pretty: bool) -> str:
    """Encode the demo without non-finite or order-dependent JSON."""

    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
    )


def render_human(document: dict[str, object]) -> str:
    """Render a terminal-friendly view of the same structured result."""

    policy = document["policy"]
    summary = document["review_summary"]
    next_decision = document["next_recommendation"]
    assert isinstance(policy, dict)
    assert isinstance(summary, dict)
    assert isinstance(next_decision, dict)
    recommendation = next_decision["recommendation"]
    assert isinstance(recommendation, dict)
    arm_scores = recommendation["arm_scores"]
    assert isinstance(arm_scores, list)

    lines = [
        "GWorker policy demo | deterministic synthetic scenario",
        f"Policy: {policy['policy_id']}",
        (
            f"Reviewed decisions: {summary['review_count']} | "
            f"target: {int(summary['target_focus_seconds']) // 60} minutes"
        ),
        "",
        "Review history",
        " seq  selected   propensity  bucket   feedback     completed",
    ]
    decisions = document["decisions"]
    assert isinstance(decisions, list)
    for item in decisions:
        assert isinstance(item, dict)
        item_recommendation = item["recommendation"]
        item_review = item["review"]
        assert isinstance(item_recommendation, dict)
        assert isinstance(item_review, dict)
        lines.append(
            f" {int(item['decision_sequence']):>3}  "
            f"{item_recommendation['template_id']!s:<10} "
            f"{float(item_recommendation['propensity']):>10.6f}  "
            f"{item_recommendation['evidence_bucket']!s:<7}  "
            f"{item_review['fit']!s:<11}  "
            f"{str(item_review['objective_completed']).lower()}"
        )

    lines.extend(
        [
            "",
            (
                f"Decision {next_decision['decision_sequence']}: "
                f"{recommendation['template_id']} at "
                f"p={float(recommendation['propensity']):.6f}"
            ),
            (
                f"Evidence: {recommendation['evidence_bucket_label']} "
                f"({recommendation['evidence_count']} reviews)"
            ),
            "Reasons: " + ", ".join(recommendation["reason_codes"]),
            "",
            "Arm scores",
            (
                " arm        n  posterior  direction  exploration  "
                "score      probability"
            ),
        ]
    )
    for arm in arm_scores:
        assert isinstance(arm, dict)
        lines.append(
            f" {arm['template_id']!s:<10} "
            f"{int(arm['review_count']):>2}  "
            f"{float(arm['posterior_mean']):>9.6f}  "
            f"{float(arm['directional_adjustment']):>9.6f}  "
            f"{float(arm['exploration_bonus']):>11.6f}  "
            f"{float(arm['score']):>8.6f}  "
            f"{float(arm['probability']):>11.6f}"
        )
    verification = document["verification"]
    assert isinstance(verification, dict)
    lines.extend(
        [
            "",
            (
                "Checks: review provenance bound; probabilities sum to "
                f"{float(verification['arm_probability_sum']):.12f}"
            ),
            "Notice: synthetic API demo, not locked-evaluation evidence.",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show a deterministic 12-review GWorker policy scenario."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit canonical, pretty JSON instead of terminal text",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    document = build_demo()
    output = (
        canonical_json(document, pretty=True)
        if arguments.json
        else render_human(document)
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
