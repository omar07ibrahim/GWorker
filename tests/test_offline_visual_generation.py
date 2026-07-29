from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from collections import Counter
from pathlib import Path

import gworker.offline as offline
from scripts import demo_offline_replay as demo
from scripts.visuals import generate_offline_replay as visual

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILENAMES = (
    "estimator-decomposition.svg",
    "ordered-propensity-and-weight.svg",
    "support-and-template-coverage.svg",
)
FROZEN_BUNDLE_SHA256 = {
    "generated/estimator-decomposition.svg": (
        "dcffa8974db043badd26a2f8287be0ae565ce7ae035576f1c1a22759ac326db4"
    ),
    "generated/ordered-propensity-and-weight.svg": (
        "46ca13dbaa26bea893f2a0471faf1bae84a2b48877b80b3852beb9d00e981359"
    ),
    "generated/support-and-template-coverage.svg": (
        "3ccf099500f7e230336b7f2eff9cf51cbb306f5f44213c8f81d695123b717520"
    ),
    "manifest.json": (
        "5cdbac0ade955b057f66edc1d92fb27b9a9c357208eb52adb235e2bd95f3d0a3"
    ),
}
FROZEN_BUNDLE_BYTE_COUNTS = {
    "generated/estimator-decomposition.svg": 15_727,
    "generated/ordered-propensity-and-weight.svg": 15_090,
    "generated/support-and-template-coverage.svg": 11_038,
    "manifest.json": 4_527,
}
EXPECTED_INPUT_PATHS = (
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
EXPECTED_SELECTED_FACTS = (
    (1, "focus-50", 0.08, 1.0),
    (2, "focus-15", 0.10, 0.2),
    (3, "focus-25", 0.45, 0.8),
    (4, "focus-40", 0.20, 0.2),
    (5, "focus-25", 0.35, 1.0),
    (6, "focus-40", 0.15, 1.0),
    (7, "focus-15", 0.20, 0.0),
    (8, "focus-50", 0.10, 0.0),
    (9, "focus-40", 0.25, 0.8),
    (10, "focus-25", 0.25, 1.0),
    (11, "focus-50", 0.15, 0.2),
    (12, "focus-15", 0.15, 0.2),
    (13, "focus-25", 0.20, 0.8),
    (14, "focus-40", 0.20, 1.0),
    (15, "focus-15", 0.25, 0.0),
    (16, "focus-50", 0.10, 0.2),
)
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
FORBIDDEN_IDENTITY_OR_HOST_METADATA = re.compile(
    r"(?:/home/|/tmp/|\\\\Users\\\\|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b|"
    r"\b\d{4}-\d{2}-\d{2}(?:T|\s)|"
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b|"
    r"@[A-Za-z0-9.-]+)",
    re.IGNORECASE,
)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def path_sha256(path: Path) -> str:
    return sha256(path.read_bytes())


def closed_json_document(content: bytes) -> dict[str, object]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError(f"duplicate JSON key: {key}")
            document[key] = value
        return document

    loaded = json.loads(content, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(loaded, dict):
        raise TypeError("manifest must be a JSON object")
    return loaded


def subprocess_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": "src",
    }


class OfflineReplayVisualEvidenceTests(unittest.TestCase):
    def test_observations_are_exactly_derived_from_public_replay_api(
        self,
    ) -> None:
        evidence = visual.build_offline_replay_evidence()
        rows = demo.fixed_replay_rows()
        report = demo.build_fixed_replay_report()
        expected = []

        for row in rows:
            singleton = offline.build_replay_report(
                (row,),
                target=report.candidate.target,
                config=report.config,
            )
            selected = tuple(
                summary
                for summary in singleton.candidate.templates
                if summary.template_id == row.selected_template_id
            )
            self.assertEqual(len(selected), 1)
            summary = selected[0]
            reward = singleton.candidate.observed_behavior_mean
            self.assertIsNotNone(reward)
            assert reward is not None
            self.assertEqual(reward, row.reward)
            expected.append(
                visual.ReplayObservation(
                    sequence=row.decision_sequence,
                    template_id=row.selected_template_id,
                    behavior_probability=row.selected_arm.behavior_probability,
                    target_probability=summary.target_mass,
                    raw_weight=summary.raw_weight_sum,
                    clipped_weight=summary.clipped_weight_sum,
                    reward=reward,
                )
            )

        self.assertIs(type(evidence), visual.OfflineReplayEvidence)
        self.assertIs(type(evidence.report), offline.ReplayReport)
        self.assertEqual(evidence.report, report)
        self.assertEqual(evidence.observations, tuple(expected))
        self.assertEqual(len(evidence.observations), 16)
        self.assertEqual(
            tuple(
                (
                    observation.sequence,
                    observation.template_id,
                    observation.behavior_probability,
                    observation.reward,
                )
                for observation in evidence.observations
            ),
            EXPECTED_SELECTED_FACTS,
        )
        self.assertEqual(
            Counter(observation.template_id for observation in evidence.observations),
            {
                "focus-15": 4,
                "focus-25": 4,
                "focus-40": 4,
                "focus-50": 4,
            },
        )
        self.assertEqual(
            sum(
                observation.raw_weight > report.config.clip_weight
                for observation in evidence.observations
            ),
            4,
        )
        for observation in evidence.observations:
            with self.subTest(sequence=observation.sequence):
                self.assertEqual(
                    observation.raw_weight,
                    observation.target_probability / observation.behavior_probability,
                )
                self.assertEqual(
                    observation.clipped_weight,
                    min(observation.raw_weight, report.config.clip_weight),
                )

    def test_generator_directly_imports_only_public_offline_surface(self) -> None:
        source = (
            ROOT / "scripts" / "visuals" / "generate_offline_replay.py"
        ).read_text("utf-8")
        tree = ast.parse(source)
        gworker_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                gworker_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name == "gworker" or alias.name.startswith("gworker.")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and (node.module == "gworker" or node.module.startswith("gworker."))
            ):
                gworker_imports.append(node.module)

        self.assertEqual(gworker_imports, ["gworker.offline"])

    def test_clean_process_loads_no_forbidden_gworker_module(self) -> None:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from scripts import demo_offline_replay as demo;"
                    "from scripts.visuals import generate_offline_replay as visual;"
                    "evidence=visual.build_offline_replay_evidence();"
                    "visual.render_visuals(evidence);"
                    "demo.assert_safe_import_boundary()"
                ),
            ],
            cwd=ROOT,
            env=subprocess_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )

        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(probe.stdout, "")
        self.assertEqual(probe.stderr, "")

    def test_filenames_and_rendered_bytes_are_deterministic(self) -> None:
        first = visual.render_visuals(visual.build_offline_replay_evidence())
        second = visual.render_visuals(visual.build_offline_replay_evidence())
        implicit = visual.render_visuals()

        self.assertEqual(first, second)
        self.assertEqual(first, implicit)
        self.assertEqual(
            tuple(rendered.filename for rendered in first),
            EXPECTED_FILENAMES,
        )
        self.assertEqual(
            tuple(rendered.filename for rendered in first),
            tuple(sorted(rendered.filename for rendered in first)),
        )
        for rendered in first:
            with self.subTest(filename=rendered.filename):
                self.assertIs(type(rendered), visual.RenderedVisual)
                self.assertEqual(Path(rendered.filename).name, rendered.filename)
                self.assertTrue(rendered.filename.endswith(".svg"))
                self.assertTrue(rendered.title)
                self.assertTrue(rendered.description)
                self.assertTrue(rendered.content.startswith(b"<?xml "))
                self.assertTrue(rendered.content.endswith(b"\n"))

    def test_svgs_are_accessible_self_contained_and_identity_free(self) -> None:
        for rendered in visual.render_visuals():
            with self.subTest(filename=rendered.filename):
                content = rendered.content.decode("utf-8")
                root = ElementTree.fromstring(content)
                elements = tuple(root.iter())
                ids = {
                    element.attrib["id"]: element
                    for element in elements
                    if "id" in element.attrib
                }
                labelled_by = root.attrib["aria-labelledby"].split()

                self.assertEqual(root.tag, f"{{{SVG_NAMESPACE}}}svg")
                self.assertEqual(root.attrib["role"], "img")
                self.assertEqual(len(labelled_by), 2)
                self.assertTrue(all(label in ids for label in labelled_by))
                self.assertEqual(
                    ids[labelled_by[0]].tag,
                    f"{{{SVG_NAMESPACE}}}title",
                )
                self.assertEqual(
                    ids[labelled_by[1]].tag,
                    f"{{{SVG_NAMESPACE}}}desc",
                )
                self.assertEqual(ids[labelled_by[0]].text, rendered.title)
                self.assertEqual(ids[labelled_by[1]].text, rendered.description)
                self.assertIn("viewBox", root.attrib)
                self.assertIn("width", root.attrib)
                self.assertIn("height", root.attrib)
                self.assertNotRegex(
                    content,
                    FORBIDDEN_IDENTITY_OR_HOST_METADATA,
                )
                self.assertNotIn("@import", content)

                local_tags = {
                    element.tag.rsplit("}", 1)[-1].lower() for element in elements
                }
                self.assertTrue(
                    local_tags.isdisjoint(
                        {"foreignobject", "iframe", "image", "script"}
                    )
                )
                for element in elements:
                    for attribute in element.attrib:
                        self.assertNotEqual(
                            attribute.rsplit("}", 1)[-1].lower(),
                            "href",
                        )
                for reference in re.findall(r"url\(([^)]+)\)", content):
                    target = reference.strip("\"'")
                    self.assertTrue(target.startswith("#"), target)
                    self.assertIn(target[1:], ids)


class OfflineReplayVisualBundleTests(unittest.TestCase):
    def test_committed_bundle_matches_frozen_hashes_and_byte_counts(self) -> None:
        paths = {
            path.relative_to(visual.VISUAL_ROOT).as_posix(): path
            for path in visual.VISUAL_ROOT.rglob("*")
            if path.is_file()
        }

        self.assertEqual(set(paths), set(FROZEN_BUNDLE_SHA256))
        self.assertEqual(
            {name: path_sha256(path) for name, path in paths.items()},
            FROZEN_BUNDLE_SHA256,
        )
        self.assertEqual(
            {name: path.stat().st_size for name, path in paths.items()},
            FROZEN_BUNDLE_BYTE_COUNTS,
        )
        self.assertTrue(all(not path.is_symlink() for path in paths.values()))
        self.assertEqual(visual.check_bundle(), ())

    def test_manifest_is_canonical_and_binds_sources_and_outputs(self) -> None:
        evidence = visual.build_offline_replay_evidence()
        visuals = visual.render_visuals(evidence)
        content = visual.build_manifest(visuals, evidence)
        payload = closed_json_document(content)

        self.assertEqual(
            content,
            (
                json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("ascii"),
        )
        self.assertEqual(
            payload["schema_version"],
            "gworker-offline-visual-manifest-v1",
        )
        observations = payload["observations"]
        self.assertIsInstance(observations, dict)
        assert isinstance(observations, dict)
        self.assertEqual(
            {
                "candidate_readiness": "reportable",
                "clipped_row_count": 4,
                "config": {
                    "clip_weight": 3.0,
                    "maximum_raw_weight": 10.0,
                    "minimum_raw_ess_ratio": 0.25,
                    "minimum_reviews": 12,
                },
                "fixture_id": demo.FIXTURE_ID,
                "locked_outcome_artifact_count": 0,
                "nonclaims": {
                    "causal_effect_estimated": False,
                    "locked_evaluation_used": False,
                    "review_selection_corrected": False,
                    "sequential_policy_value_estimated": False,
                    "target_policy_value_estimated": False,
                },
                "row_count": 16,
                "synthetic_fixture": True,
                "target": {
                    "kind": "score-temperature",
                    "probability_floor": 0.02,
                    "temperature": 0.75,
                },
            },
            observations,
        )

        inputs = payload["inputs"]
        self.assertIsInstance(inputs, list)
        assert isinstance(inputs, list)
        self.assertEqual(
            tuple(record["path"] for record in inputs),
            EXPECTED_INPUT_PATHS,
        )
        for record in inputs:
            path = visual.ROOT / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(record["byte_count"], path.stat().st_size)
            self.assertEqual(record["sha256"], path_sha256(path))

        outputs = payload["outputs"]
        self.assertIsInstance(outputs, list)
        assert isinstance(outputs, list)
        self.assertEqual(
            tuple(record["path"] for record in outputs),
            tuple(
                f"{visual.GENERATED_DIRECTORY_NAME}/{filename}"
                for filename in EXPECTED_FILENAMES
            ),
        )
        for record, rendered in zip(outputs, visuals, strict=True):
            self.assertEqual(record["byte_count"], len(rendered.content))
            self.assertEqual(record["sha256"], sha256(rendered.content))
            self.assertEqual(record["title"], rendered.title)

    def test_written_bundle_is_closed_and_check_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".offline-visual-test-",
            dir=ROOT,
        ) as temporary:
            destination = Path(temporary) / "offline"
            rendered = visual.write_bundle(destination=destination)
            expected_paths = {
                visual.MANIFEST_NAME,
                *(
                    f"{visual.GENERATED_DIRECTORY_NAME}/{item.filename}"
                    for item in rendered
                ),
            }
            actual_paths = {
                path.relative_to(destination).as_posix()
                for path in destination.rglob("*")
                if path.is_file()
            }

            self.assertEqual(actual_paths, expected_paths)
            self.assertEqual(visual.check_bundle(root=destination), ())

            victim = (
                destination / visual.GENERATED_DIRECTORY_NAME / EXPECTED_FILENAMES[0]
            )
            victim.write_bytes(victim.read_bytes() + b" ")
            changed = visual.check_bundle(root=destination)
            self.assertTrue(
                any(
                    "changed" in difference and EXPECTED_FILENAMES[0] in difference
                    for difference in changed
                ),
                changed,
            )

            visual.write_bundle(destination=destination)
            unexpected = (
                destination / visual.GENERATED_DIRECTORY_NAME / "unexpected.svg"
            )
            unexpected.write_text("<svg/>", encoding="ascii")
            closed_set = visual.check_bundle(root=destination)
            self.assertTrue(
                any(
                    "unexpected" in difference and "unexpected.svg" in difference
                    for difference in closed_set
                ),
                closed_set,
            )

            visual.write_bundle(destination=destination)
            manifest = destination / visual.MANIFEST_NAME
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            manifest_tamper = visual.check_bundle(root=destination)
            self.assertTrue(
                any(
                    "changed" in difference and visual.MANIFEST_NAME in difference
                    for difference in manifest_tamper
                ),
                manifest_tamper,
            )

    def test_check_rejects_root_nested_and_symlink_shape_changes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".offline-visual-shape-test-",
            dir=ROOT,
        ) as temporary:
            base = Path(temporary)

            root_file = base / "root-file"
            visual.write_bundle(destination=root_file)
            (root_file / "notes.txt").write_text("unexpected", encoding="ascii")
            self.assertIn(
                "unexpected bundle entry: notes.txt",
                visual.check_bundle(root=root_file),
            )

            nested = base / "nested"
            visual.write_bundle(destination=nested)
            nested_directory = nested / visual.GENERATED_DIRECTORY_NAME / "nested"
            nested_directory.mkdir()
            (nested_directory / "surprise.svg").write_text(
                "<svg/>",
                encoding="ascii",
            )
            self.assertIn(
                "unexpected generated entry: nested",
                visual.check_bundle(root=nested),
            )

            output_link = base / "output-link"
            visual.write_bundle(destination=output_link)
            linked_output = (
                output_link / visual.GENERATED_DIRECTORY_NAME / EXPECTED_FILENAMES[0]
            )
            linked_output.unlink()
            linked_output.symlink_to(EXPECTED_FILENAMES[1])
            self.assertIn(
                f"invalid generated entry: {EXPECTED_FILENAMES[0]}",
                visual.check_bundle(root=output_link),
            )

            manifest_link = base / "manifest-link"
            visual.write_bundle(destination=manifest_link)
            manifest = manifest_link / visual.MANIFEST_NAME
            manifest.unlink()
            manifest.symlink_to(
                Path(visual.GENERATED_DIRECTORY_NAME) / EXPECTED_FILENAMES[0]
            )
            self.assertIn(
                "manifest entry is not a regular file",
                visual.check_bundle(root=manifest_link),
            )

            generated_link = base / "generated-link"
            visual.write_bundle(destination=generated_link)
            generated = generated_link / visual.GENERATED_DIRECTORY_NAME
            backing = generated_link / "generated-backing"
            generated.rename(backing)
            generated.symlink_to(backing.name, target_is_directory=True)
            shape_drift = visual.check_bundle(root=generated_link)
            self.assertIn("generated entry is not a directory", shape_drift)
            self.assertIn(
                "unexpected bundle entry: generated-backing",
                shape_drift,
            )

    def test_cli_check_verifies_committed_bundle_without_mutation(self) -> None:
        self.assertEqual(visual.ROOT, ROOT)
        self.assertTrue(visual.VISUAL_ROOT.is_dir())
        before = {
            path.relative_to(visual.VISUAL_ROOT).as_posix(): path_sha256(path)
            for path in visual.VISUAL_ROOT.rglob("*")
            if path.is_file()
        }
        probe = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.visuals.generate_offline_replay",
                "--check",
            ],
            cwd=ROOT,
            env=subprocess_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        after = {
            path.relative_to(visual.VISUAL_ROOT).as_posix(): path_sha256(path)
            for path in visual.VISUAL_ROOT.rglob("*")
            if path.is_file()
        }

        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertEqual(
            probe.stdout,
            "offline replay visual evidence is reproducible\n",
        )
        self.assertEqual(probe.stderr, "")
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
