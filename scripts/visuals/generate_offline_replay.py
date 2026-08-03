#!/usr/bin/env python3
"""Generate isolated, source-derived SVG evidence for offline replay.

This generator deliberately imports only the public ``gworker.offline``
surface and the fixed authored demo fixture. It never imports or invokes the
locked evaluator, publication workflow, result codecs, or a user journal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final
from xml.sax.saxutils import escape

import gworker.offline as offline
from scripts import demo_offline_replay as demo

ROOT: Final = Path(__file__).resolve().parents[2]
VISUAL_ROOT: Final = ROOT / "docs" / "offline"
GENERATED_DIRECTORY_NAME: Final = "generated"
MANIFEST_NAME: Final = "manifest.json"
TOOL_NAME: Final = "gworker-offline-replay-visuals"
TOOL_VERSION: Final = "1"
SCHEMA_VERSION: Final = "gworker-offline-visual-manifest-v1"
GENERATION_COMMAND: Final = (
    "PYTHONPATH=src python3 -m scripts.visuals.generate_offline_replay"
)
CHECK_COMMAND: Final = f"{GENERATION_COMMAND} --check"
VALIDATED_PYTHON_MINORS: Final = ("3.11", "3.12", "3.13")

INPUT_FILES: Final = (
    "README.md",
    "docs/architecture.md",
    "docs/offline-replay.md",
    "docs/visuals/terminal/manifest.json",
    "docs/visuals/terminal/offline-replay.svg",
    "docs/visuals/terminal/offline-replay.txt",
    "pyproject.toml",
    "scripts/demo_offline_replay.py",
    "scripts/visuals/__init__.py",
    "scripts/visuals/generate_offline_replay.py",
    "src/gworker/__init__.py",
    "src/gworker/codec.py",
    "src/gworker/domain.py",
    "src/gworker/offline.py",
    "src/gworker/policy.py",
    "src/gworker/storage.py",
)

# Two chromatic roots carry meaning. Everything else is neutral structure.
INK: Final = "#132238"
MUTED: Final = "#526277"
GRID: Final = "#D7E1EC"
PANEL: Final = "#F5F8FC"
WHITE: Final = "#FFFFFF"
CYAN: Final = "#087E8B"
CYAN_SOFT: Final = "#D9F3F5"
VIOLET: Final = "#6D4BD1"
VIOLET_SOFT: Final = "#EEE9FC"


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    """One plotted row obtained through the public replay API."""

    sequence: int
    template_id: str
    behavior_probability: float
    target_probability: float
    raw_weight: float
    clipped_weight: float
    reward: float


@dataclass(frozen=True, slots=True)
class OfflineReplayEvidence:
    """The complete fixed fixture and its production replay report."""

    fixture_id: str
    observations: tuple[ReplayObservation, ...]
    report: offline.ReplayReport


@dataclass(frozen=True, slots=True)
class RenderedVisual:
    """One deterministic, self-contained SVG."""

    filename: str
    title: str
    description: str
    content: bytes


def _required(value: float | None, field: str) -> float:
    if value is None:
        raise RuntimeError(f"offline replay evidence omitted {field}")
    return value


def build_offline_replay_evidence() -> OfflineReplayEvidence:
    """Recompute every plotted field through the public offline API."""

    rows = demo.fixed_replay_rows()
    report = demo.build_fixed_replay_report()
    target = demo.fixed_replay_target()
    config = report.config
    observations: list[ReplayObservation] = []
    for row in rows:
        one_row = offline.build_replay_report(
            (row,),
            target=target,
            config=config,
        ).candidate
        selected = next(
            item
            for item in one_row.templates
            if item.template_id == row.selected_template_id
        )
        observations.append(
            ReplayObservation(
                sequence=row.decision_sequence,
                template_id=row.selected_template_id,
                behavior_probability=row.selected_arm.behavior_probability,
                target_probability=selected.target_mass,
                raw_weight=selected.raw_weight_sum,
                clipped_weight=selected.clipped_weight_sum,
                reward=_required(
                    one_row.observed_behavior_mean,
                    "singleton observed behavior mean",
                ),
            )
        )

    evidence = OfflineReplayEvidence(
        fixture_id=demo.FIXTURE_ID,
        observations=tuple(observations),
        report=report,
    )
    _validate_evidence(evidence)
    return evidence


def _validate_evidence(evidence: OfflineReplayEvidence) -> None:
    candidate = evidence.report.candidate
    control = evidence.report.behavior_control
    observations = evidence.observations
    if evidence.fixture_id != "authored-synthetic-replay-v1":
        raise RuntimeError("offline replay fixture identity drifted")
    if tuple(item.sequence for item in observations) != tuple(range(1, 17)):
        raise RuntimeError("offline replay observation order drifted")
    if len(observations) != 16:
        raise RuntimeError("offline replay evidence must contain sixteen rows")
    template_counts = {
        template_id: sum(item.template_id == template_id for item in observations)
        for template_id in ("focus-15", "focus-25", "focus-40", "focus-50")
    }
    if template_counts != {
        "focus-15": 4,
        "focus-25": 4,
        "focus-40": 4,
        "focus-50": 4,
    }:
        raise RuntimeError("offline replay template support drifted")
    if candidate.readiness is not offline.ReplayReadiness.REPORTABLE:
        raise RuntimeError("offline replay candidate is not reportable")
    if control.readiness is not offline.ReplayReadiness.REPORTABLE:
        raise RuntimeError("offline replay control is not reportable")
    if (
        control.mean_raw_weight != 1.0
        or control.raw_effective_sample_size != 16.0
        or control.raw_inverse_propensity != 0.525
    ):
        raise RuntimeError("behavior replay negative control drifted")
    if (
        sum(
            item.raw_weight > evidence.report.config.clip_weight
            for item in observations
        )
        != candidate.clipped_row_count
    ):
        raise RuntimeError("offline replay clipped-row evidence drifted")
    if max(item.raw_weight for item in observations) != candidate.maximum_raw_weight:
        raise RuntimeError("offline replay maximum weight drifted")
    if (
        min(item.behavior_probability for item in observations)
        != candidate.minimum_selected_behavior_probability
    ):
        raise RuntimeError("offline replay behavior support drifted")
    nonclaims = tuple(
        getattr(evidence.report.nonclaims, field.name)
        for field in fields(evidence.report.nonclaims)
    )
    if nonclaims != (False, False, False, False, False):
        raise RuntimeError("offline replay interpretation boundary drifted")
    for item in observations:
        if not math.isclose(
            item.target_probability / item.behavior_probability,
            item.raw_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("offline replay public row evidence is inconsistent")
        if item.clipped_weight != min(
            item.raw_weight,
            evidence.report.config.clip_weight,
        ):
            raise RuntimeError("offline replay clipping evidence is inconsistent")


def _xml(value: object) -> str:
    return escape(str(value), {'"': "&quot;"})


def _text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 15,
    weight: int = 500,
    fill: str = INK,
    anchor: str = "start",
    family: str = "Inter,Segoe UI,Arial,sans-serif",
) -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" font-family="{family}" '
        f'text-anchor="{anchor}">{_xml(value)}</text>'
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str | None = None,
    stroke_width: float = 1,
    radius: float = 12,
) -> str:
    stroke_fragment = (
        "" if stroke is None else f' stroke="{stroke}" stroke-width="{stroke_width:g}"'
    )
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
        f'rx="{radius:g}" fill="{fill}"{stroke_fragment}/>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 1,
    dash: str | None = None,
) -> str:
    dash_fragment = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'stroke="{stroke}" stroke-width="{width:g}"{dash_fragment}/>'
    )


def _circle(
    x: float,
    y: float,
    radius: float,
    *,
    fill: str,
    stroke: str,
    width: float = 2,
) -> str:
    return (
        f'<circle cx="{x:g}" cy="{y:g}" r="{radius:g}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width:g}"/>'
    )


def _polyline(
    points: tuple[tuple[float, float], ...],
    *,
    stroke: str,
    width: float = 3,
    dash: str | None = None,
) -> str:
    coordinates = " ".join(f"{x:g},{y:g}" for x, y in points)
    dash_fragment = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<polyline points="{coordinates}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width:g}" stroke-linejoin="round" '
        f'stroke-linecap="round"{dash_fragment}/>'
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
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="{title_id} {description_id}" '
            f'data-evidence-fixture="{_xml(demo.FIXTURE_ID)}">'
        ),
        f'<title id="{title_id}">{_xml(title)}</title>',
        f'<desc id="{description_id}">{_xml(description)}</desc>',
        _rect(0, 0, width, height, fill=WHITE, radius=0),
        *body,
        "</svg>",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _header(
    title: str,
    subtitle: str,
    *,
    right_badge: str,
) -> list[str]:
    return [
        _text(60, 54, title, size=28, weight=760),
        _text(60, 84, subtitle, size=15, fill=MUTED),
        _rect(876, 28, 264, 42, fill=VIOLET_SOFT, stroke=VIOLET, radius=21),
        _text(1008, 55, right_badge, size=13, weight=760, fill=VIOLET, anchor="middle"),
        _line(60, 108, 1140, 108, stroke=GRID),
    ]


def _scale_x(index: int, *, left: float, right: float, count: int) -> float:
    if count < 2:
        return left
    return left + (right - left) * index / (count - 1)


def _scale_y(
    value: float,
    *,
    minimum: float,
    maximum: float,
    top: float,
    bottom: float,
) -> float:
    if maximum <= minimum:
        raise RuntimeError("visual scale must have a positive span")
    bounded = min(max(value, minimum), maximum)
    return bottom - (bounded - minimum) * (bottom - top) / (maximum - minimum)


def _render_ordered_propensity_and_weight(
    evidence: OfflineReplayEvidence,
) -> RenderedVisual:
    title = "Selected-action propensities and weights"
    description = (
        "Two aligned panels show behavior and score-temperature target "
        "propensities, then raw importance weights for all sixteen ordered "
        "authored synthetic decisions. A dashed line marks the declared clip "
        "at three. Connecting lines show order only, not a time trend or effect."
    )
    body = _header(
        title,
        "16 authored synthetic decisions · public replay API · no journal read",
        right_badge="DESCRIPTIVE SUPPORT ONLY",
    )
    observations = evidence.observations
    left, right = 118.0, 1100.0

    body.extend(
        [
            _rect(60, 132, 1080, 276, fill=PANEL, stroke=GRID),
            _text(88, 166, "SELECTED-ACTION PROPENSITY", size=14, weight=760),
            _text(
                1112, 166, "zero-based · probability", size=13, fill=MUTED, anchor="end"
            ),
        ]
    )
    probability_top, probability_bottom = 202.0, 350.0
    for tick in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        y = _scale_y(
            tick,
            minimum=0.0,
            maximum=0.55,
            top=probability_top,
            bottom=probability_bottom,
        )
        body.extend(
            [
                _line(left, y, right, y, stroke=GRID),
                _text(104, y + 5, f"{tick:.1f}", size=12, fill=MUTED, anchor="end"),
            ]
        )
    behavior_points = tuple(
        (
            _scale_x(index, left=left, right=right, count=len(observations)),
            _scale_y(
                item.behavior_probability,
                minimum=0.0,
                maximum=0.55,
                top=probability_top,
                bottom=probability_bottom,
            ),
        )
        for index, item in enumerate(observations)
    )
    target_points = tuple(
        (
            _scale_x(index, left=left, right=right, count=len(observations)),
            _scale_y(
                item.target_probability,
                minimum=0.0,
                maximum=0.55,
                top=probability_top,
                bottom=probability_bottom,
            ),
        )
        for index, item in enumerate(observations)
    )
    body.extend(
        [
            _polyline(behavior_points, stroke=CYAN),
            _polyline(target_points, stroke=VIOLET, dash="8 5"),
        ]
    )
    for x, y in behavior_points:
        body.append(_circle(x, y, 4.5, fill=CYAN, stroke=CYAN))
    for x, y in target_points:
        body.append(_circle(x, y, 5, fill=WHITE, stroke=VIOLET, width=2.5))
    body.extend(
        [
            _line(748, 188, 782, 188, stroke=CYAN, width=3),
            _circle(765, 188, 4, fill=CYAN, stroke=CYAN),
            _text(792, 193, "behavior bᵢ", size=13, weight=650, fill=CYAN),
            _line(930, 188, 964, 188, stroke=VIOLET, width=3, dash="8 5"),
            _circle(947, 188, 4.5, fill=WHITE, stroke=VIOLET, width=2.5),
            _text(974, 193, "target qᵢ", size=13, weight=650, fill=VIOLET),
        ]
    )

    body.extend(
        [
            _rect(60, 430, 1080, 278, fill=PANEL, stroke=GRID),
            _text(88, 464, "RAW IMPORTANCE WEIGHT · qᵢ / bᵢ", size=14, weight=760),
            _text(
                1112,
                464,
                "four rows exceed c = 3",
                size=13,
                fill=MUTED,
                anchor="end",
            ),
        ]
    )
    weight_top, weight_bottom = 498.0, 646.0
    for tick in range(0, 7):
        y = _scale_y(
            float(tick),
            minimum=0.0,
            maximum=6.0,
            top=weight_top,
            bottom=weight_bottom,
        )
        body.extend(
            [
                _line(left, y, right, y, stroke=GRID),
                _text(104, y + 5, str(tick), size=12, fill=MUTED, anchor="end"),
            ]
        )
    clip = evidence.report.config.clip_weight
    clip_y = _scale_y(
        clip,
        minimum=0.0,
        maximum=6.0,
        top=weight_top,
        bottom=weight_bottom,
    )
    body.extend(
        [
            _line(left, clip_y, right, clip_y, stroke=MUTED, width=2, dash="10 6"),
            _rect(954, clip_y - 15, 146, 24, fill=WHITE, stroke=GRID, radius=12),
            _text(
                1027,
                clip_y + 2,
                "clip c = 3.000000",
                size=12,
                fill=MUTED,
                anchor="middle",
            ),
        ]
    )
    raw_points = tuple(
        (
            _scale_x(index, left=left, right=right, count=len(observations)),
            _scale_y(
                item.raw_weight,
                minimum=0.0,
                maximum=6.0,
                top=weight_top,
                bottom=weight_bottom,
            ),
        )
        for index, item in enumerate(observations)
    )
    body.append(_polyline(raw_points, stroke=CYAN))
    for observation, (x, y) in zip(observations, raw_points, strict=True):
        body.append(_circle(x, y, 4.5, fill=CYAN, stroke=CYAN))
        if observation.raw_weight > clip:
            body.append(_circle(x, clip_y, 6, fill=WHITE, stroke=VIOLET, width=2.5))
    for index, observation in enumerate(observations):
        x = _scale_x(index, left=left, right=right, count=len(observations))
        body.append(
            _text(x, 674, observation.sequence, size=12, fill=MUTED, anchor="middle")
        )
    body.extend(
        [
            _text(
                60, 748, "ORDERED SYNTHETIC DECISION", size=12, weight=760, fill=MUTED
            ),
            _text(
                1140,
                748,
                "lines guide row order · not time, causality, or target-policy value",
                size=13,
                fill=MUTED,
                anchor="end",
            ),
        ]
    )
    content = _svg_document(
        stem="ordered-propensity-and-weight",
        title=title,
        description=description,
        width=1200,
        height=780,
        body=body,
    )
    return RenderedVisual(
        filename="ordered-propensity-and-weight.svg",
        title=title,
        description=description,
        content=content,
    )


def _metric_card(
    x: float,
    y: float,
    *,
    heading: str,
    rows: tuple[tuple[str, str], ...],
    accent: str,
) -> list[str]:
    body = [
        _rect(x, y, 500, 118, fill=WHITE, stroke=GRID),
        _rect(x, y, 8, 118, fill=accent, radius=4),
        _text(x + 24, y + 28, heading, size=13, weight=760, fill=accent),
    ]
    for index, (formula, value) in enumerate(rows):
        row_y = y + 58 + index * 30
        body.extend(
            [
                _text(x + 24, row_y, formula, size=13, fill=MUTED),
                _text(x + 476, row_y, value, size=14, weight=720, anchor="end"),
            ]
        )
    return body


def _render_estimator_decomposition(
    evidence: OfflineReplayEvidence,
) -> RenderedVisual:
    title = "Estimator decomposition from exact row evidence"
    description = (
        "All sixteen authored synthetic rows list the reward, raw importance "
        "weight, and clipped weight returned by public replay calls. Formula "
        "cards show the exact aggregate outputs for behavior, raw, and clipped "
        "descriptive summaries, plus explicit interpretation nonclaims."
    )
    report = evidence.report
    candidate = report.candidate
    observed_mean = _required(candidate.observed_behavior_mean, "observed mean")
    raw_ips = _required(candidate.raw_inverse_propensity, "raw IPS")
    raw_snips = _required(candidate.raw_self_normalized, "raw SNIPS")
    clipped_ips = _required(candidate.clipped_inverse_propensity, "clipped IPS")
    clipped_snips = _required(candidate.clipped_self_normalized, "clipped SNIPS")
    body = _header(
        title,
        "every displayed value is recomputed from the fixed public-API fixture",
        right_badge="16 ROWS · EXACT INPUTS",
    )
    body.extend(
        [
            _rect(60, 132, 532, 430, fill=PANEL, stroke=GRID),
            _text(84, 166, "ROW INPUTS RETURNED BY PUBLIC REPLAY", size=14, weight=760),
            _text(
                84,
                190,
                "decision · reward yᵢ · raw wᵢ · clipped w̄ᵢ",
                size=13,
                fill=MUTED,
            ),
        ]
    )
    for index, item in enumerate(evidence.observations):
        column = index // 8
        row = index % 8
        x = 84 + column * 252
        y = 226 + row * 38
        fill = VIOLET_SOFT if item.raw_weight > report.config.clip_weight else WHITE
        body.extend(
            [
                _rect(x, y - 24, 232, 34, fill=fill, stroke=GRID, radius=6),
                _text(
                    x + 10,
                    y - 8,
                    (
                        f"{item.sequence:02d} · y={item.reward:.1f} · "
                        f"raw w={item.raw_weight:.6f}"
                    ),
                    size=10,
                    family="ui-monospace,SFMono-Regular,Consolas,monospace",
                ),
                _text(
                    x + 10,
                    y + 6,
                    f"clipped w̄={item.clipped_weight:.6f}",
                    size=10,
                    family="ui-monospace,SFMono-Regular,Consolas,monospace",
                ),
            ]
        )
    body.extend(
        [
            _text(
                84,
                542,
                "violet rows were clipped at c = 3.000000",
                size=12,
                weight=650,
                fill=VIOLET,
            ),
            _rect(620, 132, 520, 430, fill=PANEL, stroke=GRID),
            _text(
                644, 166, "AGGREGATE FORMULAS + PRODUCTION OUTPUT", size=14, weight=760
            ),
        ]
    )
    body.extend(
        _metric_card(
            630,
            184,
            heading="BEHAVIOR OBSERVATION",
            rows=(
                (
                    "Σ yᵢ / n",
                    f"{observed_mean:.6f}",
                ),
                ("n", str(candidate.reviewed_count)),
            ),
            accent=CYAN,
        )
    )
    body.extend(
        _metric_card(
            630,
            314,
            heading="RAW REWEIGHTING · READINESS INPUT",
            rows=(
                (
                    "Σ wᵢyᵢ / n · IPS",
                    f"{raw_ips:.6f}",
                ),
                (
                    "Σ wᵢyᵢ / Σ wᵢ · SNIPS",
                    f"{raw_snips:.6f}",
                ),
            ),
            accent=VIOLET,
        )
    )
    body.extend(
        _metric_card(
            630,
            444,
            heading="CLIPPED SENSITIVITY · NEVER A READINESS GATE",
            rows=(
                (
                    "Σ w̄ᵢyᵢ / n · clipped IPS",
                    f"{clipped_ips:.6f}",
                ),
                (
                    "Σ w̄ᵢyᵢ / Σ w̄ᵢ · clipped SNIPS",
                    f"{clipped_snips:.6f}",
                ),
            ),
            accent=CYAN,
        )
    )

    raw_ess = _required(candidate.raw_effective_sample_size, "raw ESS")
    clipped_ess = _required(candidate.clipped_effective_sample_size, "clipped ESS")
    pipeline_y = 596
    body.extend(
        [
            _rect(60, pipeline_y, 1080, 172, fill=WHITE, stroke=GRID),
            _text(84, pipeline_y + 32, "TRACEABLE DECOMPOSITION", size=13, weight=760),
            _rect(84, pipeline_y + 52, 256, 76, fill=CYAN_SOFT, stroke=CYAN),
            _text(
                212,
                pipeline_y + 82,
                "ROW INGREDIENTS",
                size=13,
                weight=760,
                fill=CYAN,
                anchor="middle",
            ),
            _text(
                212, pipeline_y + 108, "bᵢ · qᵢ · yᵢ · c=3", size=14, anchor="middle"
            ),
            _line(340, pipeline_y + 90, 400, pipeline_y + 90, stroke=MUTED, width=2),
            _text(370, pipeline_y + 80, "→", size=22, fill=MUTED, anchor="middle"),
            _rect(410, pipeline_y + 52, 300, 76, fill=VIOLET_SOFT, stroke=VIOLET),
            _text(
                560,
                pipeline_y + 82,
                "DETERMINISTIC WEIGHTS",
                size=13,
                weight=760,
                fill=VIOLET,
                anchor="middle",
            ),
            _text(
                560,
                pipeline_y + 108,
                "wᵢ=qᵢ/bᵢ · w̄ᵢ=min(wᵢ,c)",
                size=14,
                anchor="middle",
            ),
            _line(710, pipeline_y + 90, 770, pipeline_y + 90, stroke=MUTED, width=2),
            _text(740, pipeline_y + 80, "→", size=22, fill=MUTED, anchor="middle"),
            _rect(780, pipeline_y + 52, 336, 76, fill=PANEL, stroke=INK),
            _text(
                948,
                pipeline_y + 79,
                "FINITE-SNAPSHOT OUTPUTS",
                size=13,
                weight=760,
                anchor="middle",
            ),
            _text(
                948,
                pipeline_y + 104,
                f"raw ESS {raw_ess:.6f} · clipped ESS {clipped_ess:.6f}",
                size=13,
                anchor="middle",
            ),
            _text(
                948,
                pipeline_y + 122,
                f"readiness {candidate.readiness.value}",
                size=12,
                weight=700,
                fill=VIOLET,
                anchor="middle",
            ),
            _rect(60, 790, 1080, 70, fill=VIOLET_SOFT, stroke=VIOLET),
            _text(84, 818, "EXPLICIT NONCLAIMS", size=13, weight=760, fill=VIOLET),
            _text(
                84,
                844,
                (
                    "review selection corrected=false · target-policy value=false · "
                    "sequential value=false · causal effect=false · "
                    "locked evaluation=false"
                ),
                size=13,
                weight=650,
            ),
        ]
    )
    content = _svg_document(
        stem="estimator-decomposition",
        title=title,
        description=description,
        width=1200,
        height=890,
        body=body,
    )
    return RenderedVisual(
        filename="estimator-decomposition.svg",
        title=title,
        description=description,
        content=content,
    )


def _bar(
    x: float,
    y: float,
    width: float,
    *,
    fraction: float,
    color: str,
    height: float = 18,
) -> list[str]:
    bounded = min(max(fraction, 0.0), 1.0)
    return [
        _rect(x, y, width, height, fill=WHITE, stroke=GRID, radius=height / 2),
        _rect(
            x,
            y,
            max(width * bounded, 1.0),
            height,
            fill=color,
            radius=height / 2,
        ),
    ]


def _render_support_and_template_coverage(
    evidence: OfflineReplayEvidence,
) -> RenderedVisual:
    title = "Raw support gates and template coverage"
    description = (
        "Zero-based panels compare the sixteen reviewed rows, raw effective "
        "sample-size ratio, and maximum raw weight with their declared support "
        "thresholds. Separate panels show reviewed count and target probability "
        "mass for each template. Clipping is labelled as sensitivity only."
    )
    report = evidence.report
    candidate = report.candidate
    config = report.config
    raw_ess_ratio = _required(
        candidate.raw_effective_sample_size_ratio,
        "raw ESS ratio",
    )
    body = _header(
        title,
        (
            "readiness is determined only by raw overlap · "
            "unlike units use separate scales"
        ),
        right_badge=f"STATUS · {candidate.readiness.value.upper()}",
    )
    panels = (
        (
            60.0,
            "REVIEWED ROWS",
            float(candidate.reviewed_count),
            float(candidate.reviewed_count),
            float(config.minimum_reviews),
            f"{candidate.reviewed_count} / minimum {config.minimum_reviews}",
            CYAN,
        ),
        (
            426.0,
            "RAW ESS RATIO",
            raw_ess_ratio,
            1.0,
            config.minimum_raw_ess_ratio,
            (f"{raw_ess_ratio:.6f} / minimum {config.minimum_raw_ess_ratio:.2f}"),
            VIOLET,
        ),
        (
            792.0,
            "MAXIMUM RAW WEIGHT",
            _required(candidate.maximum_raw_weight, "maximum raw weight"),
            config.maximum_raw_weight,
            config.maximum_raw_weight,
            (
                f"{_required(candidate.maximum_raw_weight, 'maximum raw weight'):.6f}"
                f" / maximum {config.maximum_raw_weight:.1f}"
            ),
            VIOLET,
        ),
    )
    for x, heading, value, maximum, threshold, label, color in panels:
        body.extend(
            [
                _rect(x, 136, 348, 190, fill=PANEL, stroke=GRID),
                _text(x + 22, 170, heading, size=13, weight=760),
                _text(x + 22, 199, label, size=14, weight=700, fill=color),
                *_bar(
                    x + 22,
                    226,
                    304,
                    fraction=value / maximum,
                    color=color,
                    height=22,
                ),
            ]
        )
        threshold_x = x + 22 + 304 * threshold / maximum
        body.extend(
            [
                _line(
                    threshold_x,
                    217,
                    threshold_x,
                    259,
                    stroke=INK,
                    width=2,
                    dash="4 3",
                ),
                _text(x + 22, 284, "0", size=12, fill=MUTED),
                _text(x + 326, 284, f"{maximum:g}", size=12, fill=MUTED, anchor="end"),
                _text(
                    x + 22,
                    306,
                    (
                        "threshold shown as dashed reference"
                        if heading != "MAXIMUM RAW WEIGHT"
                        else "upper raw-support gate · zero baseline"
                    ),
                    size=12,
                    fill=MUTED,
                ),
            ]
        )

    body.extend(
        [
            _rect(60, 350, 522, 378, fill=PANEL, stroke=GRID),
            _text(84, 386, "TEMPLATE REVIEWED COUNT", size=14, weight=760, fill=CYAN),
            _text(
                84,
                410,
                "separate zero-to-16 scale · denominator shown",
                size=12,
                fill=MUTED,
            ),
            _rect(618, 350, 522, 378, fill=PANEL, stroke=GRID),
            _text(642, 386, "TEMPLATE TARGET MASS", size=14, weight=760, fill=VIOLET),
            _text(
                642,
                410,
                "sum of target probability across 16 rows · separate scale",
                size=12,
                fill=MUTED,
            ),
        ]
    )
    for index, template in enumerate(candidate.templates):
        y = 454 + index * 64
        body.extend(
            [
                _text(84, y, template.template_id, size=13, weight=700),
                *_bar(
                    176,
                    y - 16,
                    300,
                    fraction=template.reviewed_count / candidate.reviewed_count,
                    color=CYAN,
                ),
                _text(
                    550,
                    y,
                    f"{template.reviewed_count} / {candidate.reviewed_count}",
                    size=13,
                    weight=700,
                    fill=CYAN,
                    anchor="end",
                ),
                _text(642, y, template.template_id, size=13, weight=700),
                *_bar(
                    734,
                    y - 16,
                    300,
                    fraction=template.target_mass / candidate.reviewed_count,
                    color=VIOLET,
                ),
                _text(
                    1110,
                    y,
                    f"{template.target_mass:.6f} / {candidate.reviewed_count}",
                    size=13,
                    weight=700,
                    fill=VIOLET,
                    anchor="end",
                ),
            ]
        )
    body.extend(
        [
            _line(84, 684, 558, 684, stroke=GRID),
            _line(642, 684, 1116, 684, stroke=GRID),
            _text(
                84,
                710,
                "all four templates: 4 reviewed selections",
                size=12,
                fill=MUTED,
            ),
            _text(
                642,
                710,
                "target mass uses feasible-arm probabilities",
                size=12,
                fill=MUTED,
            ),
            _rect(60, 754, 1080, 92, fill=WHITE, stroke=GRID),
            _text(
                84,
                786,
                "CLIPPING SENSITIVITY · NOT A READINESS GATE",
                size=13,
                weight=760,
            ),
            _text(
                84,
                819,
                (
                    f"clipped rows {candidate.clipped_row_count}"
                    f" / {candidate.reviewed_count}"
                    f" · clip c={config.clip_weight:.6f}"
                    " · removed raw weight mass "
                    f"{candidate.removed_raw_weight_mass:.6f}"
                ),
                size=14,
                weight=650,
                fill=VIOLET,
            ),
            _text(
                1116,
                819,
                "reportable ≠ effective",
                size=14,
                weight=760,
                fill=MUTED,
                anchor="end",
            ),
        ]
    )
    content = _svg_document(
        stem="support-and-template-coverage",
        title=title,
        description=description,
        width=1200,
        height=880,
        body=body,
    )
    return RenderedVisual(
        filename="support-and-template-coverage.svg",
        title=title,
        description=description,
        content=content,
    )


def render_visuals(
    evidence: OfflineReplayEvidence | None = None,
) -> tuple[RenderedVisual, ...]:
    """Render the closed, canonical offline evidence set."""

    source = build_offline_replay_evidence() if evidence is None else evidence
    _validate_evidence(source)
    visuals = (
        _render_estimator_decomposition(source),
        _render_ordered_propensity_and_weight(source),
        _render_support_and_template_coverage(source),
    )
    ordered = tuple(sorted(visuals, key=lambda item: item.filename))
    if len({item.filename for item in ordered}) != len(ordered):
        raise RuntimeError("offline replay visual filenames must be unique")
    return ordered


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _input_record(relative_path: str) -> dict[str, object]:
    content = (ROOT / relative_path).read_bytes()
    return {
        "byte_count": len(content),
        "path": relative_path,
        "sha256": _sha256(content),
    }


def build_manifest(
    visuals: tuple[RenderedVisual, ...],
    evidence: OfflineReplayEvidence,
) -> bytes:
    """Build a canonical manifest binding exact source and output bytes."""

    _validate_evidence(evidence)
    candidate = evidence.report.candidate
    config = evidence.report.config
    target = candidate.target
    nonclaims = evidence.report.nonclaims
    payload = {
        "command": GENERATION_COMMAND,
        "inputs": [
            _input_record(relative_path) for relative_path in sorted(INPUT_FILES)
        ],
        "observations": {
            "candidate_readiness": candidate.readiness.value,
            "clipped_row_count": candidate.clipped_row_count,
            "config": {
                "clip_weight": config.clip_weight,
                "maximum_raw_weight": config.maximum_raw_weight,
                "minimum_raw_ess_ratio": config.minimum_raw_ess_ratio,
                "minimum_reviews": config.minimum_reviews,
            },
            "fixture_id": evidence.fixture_id,
            "locked_outcome_artifact_count": 0,
            "nonclaims": {
                "causal_effect_estimated": nonclaims.causal_effect_estimated,
                "locked_evaluation_used": nonclaims.locked_evaluation_used,
                "review_selection_corrected": nonclaims.review_selection_corrected,
                "sequential_policy_value_estimated": (
                    nonclaims.sequential_policy_value_estimated
                ),
                "target_policy_value_estimated": (
                    nonclaims.target_policy_value_estimated
                ),
            },
            "row_count": len(evidence.observations),
            "synthetic_fixture": True,
            "target": {
                "kind": target.kind.value,
                "probability_floor": target.probability_floor,
                "temperature": target.temperature,
            },
        },
        "outputs": [
            {
                "byte_count": len(visual.content),
                "path": f"{GENERATED_DIRECTORY_NAME}/{visual.filename}",
                "sha256": _sha256(visual.content),
                "title": visual.title,
            }
            for visual in visuals
        ],
        "python": {
            "requires": ">=3.11,<3.14",
            "stdlib_only": True,
            "validated_minor_versions": list(VALIDATED_PYTHON_MINORS),
        },
        "schema_version": SCHEMA_VERSION,
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


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def write_bundle(destination: Path = VISUAL_ROOT) -> tuple[RenderedVisual, ...]:
    """Write SVGs first and publish their canonical manifest last."""

    evidence = build_offline_replay_evidence()
    visuals = render_visuals(evidence)
    generated = destination / GENERATED_DIRECTORY_NAME
    generated.mkdir(parents=True, exist_ok=True)
    expected = {visual.filename for visual in visuals}
    for path in generated.iterdir():
        if path.is_file() and path.name not in expected:
            path.unlink()
    for visual in visuals:
        _atomic_write(generated / visual.filename, visual.content)
    _atomic_write(destination / MANIFEST_NAME, build_manifest(visuals, evidence))
    return visuals


def check_bundle(root: Path = VISUAL_ROOT) -> tuple[str, ...]:
    """Compare committed evidence with a fresh in-memory reconstruction."""

    differences: list[str] = []
    evidence = build_offline_replay_evidence()
    visuals = render_visuals(evidence)
    expected_names = {visual.filename for visual in visuals}

    try:
        root_mode = os.lstat(root).st_mode
    except OSError:
        return ("offline replay visual root is unavailable",)
    if not stat.S_ISDIR(root_mode):
        return ("offline replay visual root is not a directory",)

    try:
        with os.scandir(root) as iterator:
            root_entries = {entry.name: entry for entry in iterator}
    except OSError:
        return ("offline replay visual root is unavailable",)

    expected_root_names = {GENERATED_DIRECTORY_NAME, MANIFEST_NAME}
    for name in sorted(set(root_entries) - expected_root_names):
        differences.append(f"unexpected bundle entry: {name}")
    for name in sorted(expected_root_names - set(root_entries)):
        differences.append(f"missing bundle entry: {name}")

    generated_entry = root_entries.get(GENERATED_DIRECTORY_NAME)
    generated_available = False
    if generated_entry is not None:
        try:
            generated_available = generated_entry.is_dir(follow_symlinks=False)
        except OSError:
            generated_available = False
        if not generated_available:
            differences.append("generated entry is not a directory")

    generated = root / GENERATED_DIRECTORY_NAME
    generated_entries: dict[str, os.DirEntry[str]] = {}
    if generated_available:
        try:
            with os.scandir(generated) as iterator:
                generated_entries = {entry.name: entry for entry in iterator}
        except OSError:
            differences.append("offline replay generated directory is unavailable")
            generated_available = False

    if generated_available:
        actual_names = set(generated_entries)
        for name in sorted(actual_names - expected_names):
            differences.append(f"unexpected generated entry: {name}")
        for name in sorted(expected_names - actual_names):
            differences.append(f"missing generated file: {name}")
        for visual in visuals:
            entry = generated_entries.get(visual.filename)
            if entry is None:
                continue
            try:
                regular_file = entry.is_file(follow_symlinks=False)
            except OSError:
                regular_file = False
            if not regular_file:
                differences.append(f"invalid generated entry: {visual.filename}")
                continue
            try:
                actual = (generated / visual.filename).read_bytes()
            except OSError:
                differences.append(f"unreadable generated file: {visual.filename}")
                continue
            if actual != visual.content:
                differences.append(f"changed: {visual.filename}")

    manifest_entry = root_entries.get(MANIFEST_NAME)
    manifest_available = False
    if manifest_entry is not None:
        try:
            manifest_available = manifest_entry.is_file(follow_symlinks=False)
        except OSError:
            manifest_available = False
        if not manifest_available:
            differences.append("manifest entry is not a regular file")

    expected_manifest = build_manifest(visuals, evidence)
    if manifest_available:
        try:
            actual_manifest = (root / MANIFEST_NAME).read_bytes()
        except OSError:
            differences.append("unreadable: manifest.json")
        else:
            if actual_manifest != expected_manifest:
                differences.append("changed: manifest.json")
    return tuple(differences)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify isolated offline replay SVG evidence."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare committed files without writing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        differences = check_bundle()
        if differences:
            for difference in differences:
                print(difference)
            return 1
        print("offline replay visual evidence is reproducible")
        return 0
    visuals = write_bundle()
    print(f"generated {len(visuals)} offline replay SVG files and {MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
