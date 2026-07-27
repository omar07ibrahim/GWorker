#!/usr/bin/env python3
"""Render the locked protocol inventory without running any experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from gworker.evaluation import (
    DEFAULT_EXPERIMENT_CONFIG,
    EVALUATOR_VERSION,
    LOCKED_DESIGN_ID,
    LOCKED_EVALUATION_RUN_KEY,
    LOCKED_EVALUATOR_ID,
    LOCKED_MINIMUM_PROPENSITY,
    LOCKED_POLICY_ID,
    LOCKED_POPULATION_ID,
    RESULT_SCHEMA_VERSION,
    Strategy,
    validate_experiment_config,
)
from gworker.evidence import (
    PUBLICATION_EVIDENCE_SCHEMA_VERSION,
    expected_publication_cardinalities,
)
from gworker.policy import DEFAULT_TEMPLATES
from gworker.reporting import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_VERSION,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    LOCKED_BOOTSTRAP_INDICES_SHA256,
    LOCKED_BOOTSTRAP_MODE_INDICES_SHA256,
    LOCKED_BOOTSTRAP_NAMESPACE_SHA256,
    REPORT_SCHEMA_VERSION,
)

SCHEMA_VERSION = "gworker-protocol-inventory-v1"


def build_inventory() -> dict[str, object]:
    """Call only the locked config validator and cardinality calculator."""

    config = DEFAULT_EXPERIMENT_CONFIG
    validate_experiment_config(config)
    cardinalities = expected_publication_cardinalities(config)

    return {
        "bootstrap": {
            "combined_indices_sha256": LOCKED_BOOTSTRAP_INDICES_SHA256,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "mode_indices_sha256": [
                {
                    "availability_mode": mode.value,
                    "sha256": digest,
                }
                for mode, digest in LOCKED_BOOTSTRAP_MODE_INDICES_SHA256
            ],
            "namespace_sha256": LOCKED_BOOTSTRAP_NAMESPACE_SHA256,
            "resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
            "version": BOOTSTRAP_VERSION,
        },
        "call_surface": [
            "validate_experiment_config",
            "expected_publication_cardinalities",
        ],
        "configuration": {
            "availability_modes": [mode.value for mode in config.availability_modes],
            "drift_decision": config.drift_decision,
            "environment_seeds": {
                "count": len(config.environment_seeds),
                "first": config.environment_seeds[0],
                "last": config.environment_seeds[-1],
            },
            "horizon": config.horizon,
            "personas": [
                {
                    "label": persona.label,
                    "persona_id": persona.persona_id,
                    "primary": persona.primary,
                    "selective_reviews": persona.selective_reviews,
                    "sigma_minutes": persona.sigma_minutes,
                }
                for persona in config.personas
            ],
            "policy_replicas": config.policy_replicas,
            "recovery_block_size": config.recovery_block_size,
            "recovery_blocks": config.recovery_blocks,
            "split": config.split,
            "strategies": [strategy.value for strategy in Strategy],
            "templates": [
                {
                    "break_seconds": template.break_seconds,
                    "focus_seconds": template.focus_seconds,
                    "template_id": template.template_id,
                    "total_seconds": template.total_seconds,
                }
                for template in DEFAULT_TEMPLATES
            ],
        },
        "contains_evaluation_results": False,
        "expected_cardinalities": asdict(cardinalities),
        "kind": "locked-protocol-inventory",
        "locked_identity": {
            "design_id": LOCKED_DESIGN_ID,
            "evaluator_id": LOCKED_EVALUATOR_ID,
            "evaluator_version": EVALUATOR_VERSION,
            "evidence_schema_version": PUBLICATION_EVIDENCE_SCHEMA_VERSION,
            "minimum_propensity": LOCKED_MINIMUM_PROPENSITY,
            "policy_id": LOCKED_POLICY_ID,
            "population_id": LOCKED_POPULATION_ID,
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "run_key": LOCKED_EVALUATION_RUN_KEY,
        },
        "notice": (
            "Inventory arithmetic only; no evaluator or publication runner was called."
        ),
        "schema_version": SCHEMA_VERSION,
    }


def canonical_json(document: dict[str, object], *, pretty: bool) -> str:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
    )


def render_human(document: dict[str, object]) -> str:
    identity = document["locked_identity"]
    config = document["configuration"]
    counts = document["expected_cardinalities"]
    bootstrap = document["bootstrap"]
    assert isinstance(identity, dict)
    assert isinstance(config, dict)
    assert isinstance(counts, dict)
    assert isinstance(bootstrap, dict)
    call_surface = document["call_surface"]
    assert isinstance(call_surface, list)
    assert all(isinstance(item, str) for item in call_surface)
    seeds = config["environment_seeds"]
    assert isinstance(seeds, dict)

    lines = [
        "GWorker locked protocol inventory | no evaluation executed",
        f"Evaluator:  {identity['evaluator_version']}",
        f"Policy:     {identity['policy_id']}",
        f"Run key:    {identity['run_key']}",
        f"Design ID:  {identity['design_id']}",
        f"Population: {identity['population_id']}",
        "",
        "Locked configuration",
        (
            f" split={config['split']}  horizon={config['horizon']}  "
            f"policy_replicas={config['policy_replicas']}"
        ),
        (
            f" seeds={seeds['count']} ({seeds['first']}..{seeds['last']})  "
            f"personas={len(config['personas'])}  "
            f"availability_modes={len(config['availability_modes'])}"
        ),
        (
            f" drift_decision={config['drift_decision']}  "
            f"recovery={config['recovery_blocks']} x "
            f"{config['recovery_block_size']}-decision blocks"
        ),
        "",
        "Expected publication inventory",
        " artifact                         rows",
    ]
    for key, value in counts.items():
        lines.append(f" {key:<32} {int(value):>10,}")
    lines.extend(
        [
            "",
            (
                f"Bootstrap: {bootstrap['version']}, "
                f"{int(bootstrap['resamples']):,} resamples, "
                f"{float(bootstrap['confidence']):.0%} confidence"
            ),
            "Call surface: " + ", ".join(call_surface),
            "Result data: none (inventory arithmetic only).",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect GWorker's locked config and expected inventories."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit canonical, pretty JSON instead of terminal text",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    document = build_inventory()
    output = (
        canonical_json(document, pretty=True)
        if arguments.json
        else render_human(document)
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
