#!/usr/bin/env python3
"""Deterministic aggregate replay diagnostics over authored synthetic rows."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Final

import gworker.offline as offline

FIXTURE_ID: Final = "authored-synthetic-replay-v1"
FORBIDDEN_GWORKER_PREFIXES: Final = (
    "gworker.evaluation",
    "gworker.evidence",
    "gworker.publication",
    "gworker.reporting",
    "gworker.resource_preflight",
    "gworker.result",
)

_ArmSpec = tuple[str, float, float]
_RowSpec = tuple[
    str,
    offline.DurationFit,
    bool,
    tuple[_ArmSpec, ...],
]

_FIXED_ROW_SPECS: Final[tuple[_RowSpec, ...]] = (
    (
        "focus-50",
        offline.DurationFit.JUST_RIGHT,
        True,
        (
            ("focus-15", 0.10, 0.52),
            ("focus-25", 0.40, 0.25),
            ("focus-40", 0.80, 0.15),
            ("focus-50", 1.10, 0.08),
        ),
    ),
    (
        "focus-15",
        offline.DurationFit.TOO_SHORT,
        True,
        (
            ("focus-15", 0.80, 0.10),
            ("focus-25", 0.60, 0.20),
            ("focus-40", 0.40, 0.30),
            ("focus-50", 0.20, 0.40),
        ),
    ),
    (
        "focus-25",
        offline.DurationFit.JUST_RIGHT,
        False,
        (
            ("focus-15", 0.20, 0.20),
            ("focus-25", 0.70, 0.45),
            ("focus-40", 0.50, 0.25),
            ("focus-50", 0.10, 0.10),
        ),
    ),
    (
        "focus-40",
        offline.DurationFit.TOO_LONG,
        True,
        (
            ("focus-15", 0.10, 0.40),
            ("focus-25", 0.30, 0.30),
            ("focus-40", 0.60, 0.20),
            ("focus-50", 0.90, 0.10),
        ),
    ),
    (
        "focus-25",
        offline.DurationFit.JUST_RIGHT,
        True,
        (
            ("focus-15", 0.35, 0.25),
            ("focus-25", 0.65, 0.35),
            ("focus-40", 0.55, 0.25),
            ("focus-50", 0.25, 0.15),
        ),
    ),
    (
        "focus-40",
        offline.DurationFit.JUST_RIGHT,
        True,
        (
            ("focus-15", 0.15, 0.45),
            ("focus-25", 0.45, 0.30),
            ("focus-40", 0.85, 0.15),
            ("focus-50", 1.05, 0.10),
        ),
    ),
    (
        "focus-15",
        offline.DurationFit.TOO_SHORT,
        False,
        (
            ("focus-15", 0.75, 0.20),
            ("focus-25", 0.55, 0.30),
            ("focus-40", 0.25, 0.30),
            ("focus-50", 0.05, 0.20),
        ),
    ),
    (
        "focus-50",
        offline.DurationFit.TOO_LONG,
        False,
        (
            ("focus-15", 0.05, 0.35),
            ("focus-25", 0.25, 0.30),
            ("focus-40", 0.55, 0.25),
            ("focus-50", 0.95, 0.10),
        ),
    ),
    (
        "focus-40",
        offline.DurationFit.JUST_RIGHT,
        False,
        (
            ("focus-15", 0.25, 0.30),
            ("focus-25", 0.50, 0.30),
            ("focus-40", 0.80, 0.25),
            ("focus-50", 0.70, 0.15),
        ),
    ),
    (
        "focus-25",
        offline.DurationFit.JUST_RIGHT,
        True,
        (
            ("focus-15", 0.40, 0.25),
            ("focus-25", 0.75, 0.25),
            ("focus-40", 0.65, 0.30),
            ("focus-50", 0.30, 0.20),
        ),
    ),
    (
        "focus-50",
        offline.DurationFit.TOO_LONG,
        True,
        (
            ("focus-15", 0.10, 0.30),
            ("focus-25", 0.35, 0.30),
            ("focus-40", 0.70, 0.25),
            ("focus-50", 1.00, 0.15),
        ),
    ),
    (
        "focus-15",
        offline.DurationFit.TOO_SHORT,
        True,
        (
            ("focus-15", 0.90, 0.15),
            ("focus-25", 0.70, 0.25),
            ("focus-40", 0.45, 0.30),
            ("focus-50", 0.20, 0.30),
        ),
    ),
    (
        "focus-25",
        offline.DurationFit.JUST_RIGHT,
        False,
        (
            ("focus-15", 0.30, 0.30),
            ("focus-25", 0.85, 0.20),
            ("focus-40", 0.60, 0.30),
            ("focus-50", 0.15, 0.20),
        ),
    ),
    (
        "focus-40",
        offline.DurationFit.JUST_RIGHT,
        True,
        (
            ("focus-15", 0.10, 0.40),
            ("focus-25", 0.40, 0.30),
            ("focus-40", 0.90, 0.20),
            ("focus-50", 0.75, 0.10),
        ),
    ),
    (
        "focus-15",
        offline.DurationFit.TOO_SHORT,
        False,
        (
            ("focus-15", 0.70, 0.25),
            ("focus-25", 0.60, 0.30),
            ("focus-40", 0.35, 0.25),
            ("focus-50", 0.05, 0.20),
        ),
    ),
    (
        "focus-50",
        offline.DurationFit.TOO_LONG,
        True,
        (
            ("focus-15", 0.05, 0.45),
            ("focus-25", 0.25, 0.30),
            ("focus-40", 0.55, 0.15),
            ("focus-50", 1.05, 0.10),
        ),
    ),
)


class DemoImportBoundaryError(RuntimeError):
    """Raised if replay evidence crosses into a forbidden GWorker module."""


def _is_forbidden_module(module_name: str) -> bool:
    return any(
        module_name == prefix
        or module_name.startswith(f"{prefix}.")
        or module_name.startswith(f"{prefix}_")
        for prefix in FORBIDDEN_GWORKER_PREFIXES
    )


def assert_safe_import_boundary(
    module_names: Iterable[str] | None = None,
) -> None:
    """Fail closed if locked-result or publication modules are loaded."""

    observed = sys.modules if module_names is None else module_names
    forbidden = tuple(sorted(name for name in observed if _is_forbidden_module(name)))
    if forbidden:
        raise DemoImportBoundaryError(
            "offline replay demo loaded a forbidden GWorker module"
        )


def fixed_replay_rows() -> tuple[offline.ReplayRow, ...]:
    """Return the complete ordered, authored synthetic replay fixture."""

    return tuple(
        offline.ReplayRow(
            decision_sequence=sequence,
            selected_template_id=selected_template_id,
            fit=fit,
            objective_completed=objective_completed,
            arms=tuple(
                offline.ReplayArm(
                    template_id=template_id,
                    score=score,
                    behavior_probability=behavior_probability,
                )
                for template_id, score, behavior_probability in arm_specs
            ),
        )
        for sequence, (
            selected_template_id,
            fit,
            objective_completed,
            arm_specs,
        ) in enumerate(_FIXED_ROW_SPECS, start=1)
    )


def fixed_replay_target() -> offline.ReplayTarget:
    """Return the declared score-temperature sensitivity target."""

    return offline.ReplayTarget.score_temperature(
        temperature=0.75,
        probability_floor=0.02,
    )


def fixed_replay_config() -> offline.ReplayConfig:
    """Return the fixed raw-support and clipping thresholds."""

    return offline.ReplayConfig(
        minimum_reviews=12,
        minimum_raw_ess_ratio=0.25,
        maximum_raw_weight=10.0,
        clip_weight=3.0,
    )


def build_fixed_replay_report() -> offline.ReplayReport:
    """Run the production replay core over the complete synthetic fixture."""

    return offline.build_replay_report(
        fixed_replay_rows(),
        target=fixed_replay_target(),
        config=fixed_replay_config(),
    )


def _required(value: float | None, field: str) -> float:
    if value is None:
        raise RuntimeError(f"fixed replay report omitted {field}")
    return value


def render_fixed_replay_report(report: offline.ReplayReport) -> str:
    """Render one stable, aggregate-only, path-free terminal document."""

    if type(report) is not offline.ReplayReport:
        raise TypeError("report must be an exact ReplayReport")
    if report != build_fixed_replay_report():
        raise ValueError("report does not match the fixed replay fixture")
    candidate = report.candidate
    control = report.behavior_control
    config = report.config
    nonclaims = report.nonclaims
    control_weight = _required(control.mean_raw_weight, "control weight")
    control_ess = _required(control.raw_effective_sample_size, "control ESS")
    control_ips = _required(control.raw_inverse_propensity, "control IPS")
    raw_ess = _required(candidate.raw_effective_sample_size, "raw ESS")
    raw_ess_ratio = _required(
        candidate.raw_effective_sample_size_ratio,
        "raw ESS ratio",
    )
    target_temperature = _required(
        candidate.target.temperature,
        "target temperature",
    )
    target_floor = _required(
        candidate.target.probability_floor,
        "target probability floor",
    )
    maximum_weight = _required(candidate.maximum_raw_weight, "maximum weight")
    observed_mean = _required(
        candidate.observed_behavior_mean,
        "observed mean",
    )
    raw_ips = _required(candidate.raw_inverse_propensity, "raw IPS")
    raw_snips = _required(candidate.raw_self_normalized, "raw SNIPS")
    clipped_ips = _required(
        candidate.clipped_inverse_propensity,
        "clipped IPS",
    )
    clipped_snips = _required(
        candidate.clipped_self_normalized,
        "clipped SNIPS",
    )
    minimum_behavior = _required(
        candidate.minimum_selected_behavior_probability,
        "minimum behavior probability",
    )
    template_support = " | ".join(
        f"{item.template_id}={item.reviewed_count}/{item.feasible_count}"
        for item in candidate.templates
    )
    lines = (
        f"GWorker offline replay | fixture={FIXTURE_ID}",
        (f"behavior-control | status={control.readiness.value} | exact=true"),
        (
            f"  rows={control.reviewed_count}"
            f" | mean-weight={control_weight:.6f}"
            f" | raw-ess={control_ess:.6f}"
            f" | raw-ips={control_ips:.6f}"
        ),
        (
            "candidate"
            f" | target={candidate.target.kind.value}"
            f" | status={candidate.readiness.value}"
        ),
        (
            f"  rows={candidate.reviewed_count}"
            f" | temperature={target_temperature:.6f}"
            f" | probability-floor={target_floor:.6f}"
        ),
        (
            f"  raw-ess={raw_ess:.6f}"
            f" | raw-ess-ratio={raw_ess_ratio:.6f}"
            f" | max-weight={maximum_weight:.6f}"
        ),
        (
            f"  clipped-rows={candidate.clipped_row_count}"
            f" | clip={config.clip_weight:.6f}"
            f" | removed-weight-mass={candidate.removed_raw_weight_mass:.6f}"
        ),
        "descriptive",
        (
            f"  observed={observed_mean:.6f}"
            f" | raw-ips={raw_ips:.6f}"
            f" | raw-snips={raw_snips:.6f}"
        ),
        (f"  clipped-ips={clipped_ips:.6f} | clipped-snips={clipped_snips:.6f}"),
        (
            "support"
            f" | minimum-behavior={minimum_behavior:.6f}"
            f" | minimum-reviews={config.minimum_reviews}"
        ),
        (
            f"  minimum-raw-ess-ratio={config.minimum_raw_ess_ratio:.6f}"
            f" | maximum-raw-weight={config.maximum_raw_weight:.6f}"
        ),
        "template-support",
        f"  {template_support}",
        "nonclaims",
        (
            "  review-selection-corrected="
            f"{str(nonclaims.review_selection_corrected).lower()}"
            " | target-policy-value-estimated="
            f"{str(nonclaims.target_policy_value_estimated).lower()}"
        ),
        (
            "  sequential-policy-value-estimated="
            f"{str(nonclaims.sequential_policy_value_estimated).lower()}"
            " | causal-effect-estimated="
            f"{str(nonclaims.causal_effect_estimated).lower()}"
        ),
        (f"  locked-evaluation-used={str(nonclaims.locked_evaluation_used).lower()}"),
        ("boundary | authored synthetic rows; descriptive one-step support only"),
        "  no locked evaluation",
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the closed synthetic demo without loading forbidden modules."""

    try:
        assert_safe_import_boundary()
        document = render_fixed_replay_report(build_fixed_replay_report())
        assert_safe_import_boundary()
    except Exception:
        print("offline replay demo failed closed", file=sys.stderr)
        return 2
    sys.stdout.write(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
