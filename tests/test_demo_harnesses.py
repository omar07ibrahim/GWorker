from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout, suppress
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(".gworker/visual-demo/journal-recovery")
FORBIDDEN_RUNTIME_NAMES = frozenset(
    {
        "_issue_locked_evaluation_permit",
        "generate_environment",
        "run_experiment",
        "run_publication",
        "simulate_trajectory",
    }
)
SCRIPT_NAMES = ("demo_journal.py", "demo_policy.py", "protocol_inventory.py")


def _assert_static_runtime_safety() -> None:
    for name in SCRIPT_NAMES:
        source_path = REPOSITORY_ROOT / "scripts" / name
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "gworker.publication_runner":
                    raise RuntimeError(f"{name} imports the publication runner")
                imported = {alias.name for alias in node.names}
                if imported & FORBIDDEN_RUNTIME_NAMES:
                    raise RuntimeError(f"{name} imports a forbidden runtime API")
            elif isinstance(node, ast.Call):
                called: str | None = None
                if isinstance(node.func, ast.Name):
                    called = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    called = node.func.attr
                if called in FORBIDDEN_RUNTIME_NAMES:
                    raise RuntimeError(f"{name} calls a forbidden runtime API")


_assert_static_runtime_safety()

storage = import_module("gworker.storage")
CorruptJournal = storage.CorruptJournal
SQLiteEventStore = storage.SQLiteEventStore
demo_journal = import_module("scripts.demo_journal")
demo_policy = import_module("scripts.demo_policy")
protocol_inventory = import_module("scripts.protocol_inventory")


def compact_digest(
    encoder: object,
    document: dict[str, object],
) -> str:
    encoded = encoder(document, pretty=False)  # type: ignore[operator]
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_demo_repo(parent: Path, name: str = "repo") -> Path:
    root = parent / name
    root.mkdir(mode=0o700)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "gworker-demo-test"\n',
        encoding="utf-8",
    )
    (root / "pyproject.toml").chmod(0o600)
    (root / "src" / "gworker").mkdir(parents=True, mode=0o700)
    return root


class ForbiddenRuntimeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        guards = ExitStack()
        for target in (
            "gworker.evaluation.run_experiment",
            "gworker.evaluation.generate_environment",
            "gworker.evaluation.simulate_trajectory",
            "gworker.evaluation._issue_locked_evaluation_permit",
            "gworker.publication_runner.run_publication",
        ):
            guards.enter_context(
                patch(
                    target,
                    side_effect=AssertionError(
                        f"forbidden demo runtime path called: {target}"
                    ),
                )
            )
        self.addCleanup(guards.close)


class PolicyDemoTests(ForbiddenRuntimeTestCase):
    def test_verified_twelve_review_vector_uses_real_recommendations(self) -> None:
        document = demo_policy.build_demo()
        decisions = document["decisions"]
        self.assertIsInstance(decisions, list)
        assert isinstance(decisions, list)

        self.assertEqual(
            [
                item["recommendation"]["template_id"]  # type: ignore[index]
                for item in decisions
            ],
            [
                "focus-25",
                "focus-40",
                "focus-40",
                "focus-40",
                "focus-50",
                "focus-40",
                "focus-40",
                "focus-40",
                "focus-40",
                "focus-40",
                "focus-40",
                "focus-40",
            ],
        )
        self.assertEqual(
            [item["rng_seed"] for item in decisions],  # type: ignore[index]
            list(range(20_260_001, 20_260_013)),
        )
        self.assertEqual(
            decisions[0]["decision_id"],  # type: ignore[index]
            "42ce16d2-a28f-5289-9f18-9e2bb6f41fb4",
        )
        self.assertEqual(
            decisions[-1]["decision_id"],  # type: ignore[index]
            "9f763ab0-e147-5228-b8e1-83bb63d74b11",
        )
        self.assertTrue(
            document["verification"]["all_reviews_bound_to_recommendations"]  # type: ignore[index]
        )

        next_decision = document["next_recommendation"]
        assert isinstance(next_decision, dict)
        recommendation = next_decision["recommendation"]
        assert isinstance(recommendation, dict)
        self.assertEqual(next_decision["decision_sequence"], 13)
        self.assertEqual(recommendation["template_id"], "focus-40")
        self.assertEqual(recommendation["propensity"], 0.7322645461484336)
        self.assertEqual(recommendation["evidence_bucket"], "exact")
        self.assertEqual(
            recommendation["evidence_bucket_label"],
            "exact:deep_work:medium",
        )
        self.assertEqual(recommendation["evidence_count"], 12)
        self.assertEqual(
            recommendation["reason_codes"],
            [
                "exact_evidence",
                "bounded_exploration",
                "one_step_guardrail",
            ],
        )
        self.assertEqual(
            [
                (
                    arm["template_id"],
                    arm["review_count"],
                    arm["posterior_mean"],
                    arm["directional_adjustment"],
                    arm["exploration_bonus"],
                    arm["score"],
                    arm["probability"],
                )
                for arm in recommendation["arm_scores"]  # type: ignore[union-attr]
            ],
            [
                (
                    "focus-25",
                    1,
                    0.3666666666666667,
                    0.010714285714285714,
                    0.33253378256202315,
                    0.7099147349429755,
                    0.11506356348247043,
                ),
                (
                    "focus-40",
                    10,
                    0.9249999999999999,
                    0.02142857142857143,
                    0.16626689128101158,
                    1.112695462709583,
                    0.7322645461484336,
                ),
                (
                    "focus-50",
                    1,
                    0.43333333333333335,
                    0.010714285714285714,
                    0.33253378256202315,
                    0.7765814016096422,
                    0.15267189036909593,
                ),
            ],
        )

    def test_policy_document_and_both_renderers_are_deterministic(self) -> None:
        first = demo_policy.build_demo()
        second = demo_policy.build_demo()
        self.assertEqual(
            demo_policy.canonical_json(first, pretty=False),
            demo_policy.canonical_json(second, pretty=False),
        )
        self.assertEqual(
            compact_digest(demo_policy.canonical_json, first),
            "d7a6e9f9790ffea4d13be6182f7e8dcbe10862203e66ab52016e837a3f5da2a2",
        )
        pretty = demo_policy.canonical_json(first, pretty=True)
        human = demo_policy.render_human(first)
        self.assertEqual(
            text_digest(pretty),
            "16524b609636953a2f06c322ca39fb8051e932fc6aa799caec351cd002f9027c",
        )
        self.assertEqual(
            text_digest(human),
            "d7c70a42c7cb6be5a9475f69e1b4da0608c2b41ecf02e5dd9e1ab967e0db416f",
        )

        json_output = io.StringIO()
        with redirect_stdout(json_output):
            self.assertEqual(demo_policy.main(["--json"]), 0)
        self.assertEqual(json.loads(json_output.getvalue()), first)
        self.assertEqual(json_output.getvalue(), f"{pretty}\n")

        human_output = io.StringIO()
        with redirect_stdout(human_output):
            self.assertEqual(demo_policy.main([]), 0)
        rendered = human_output.getvalue()
        self.assertEqual(rendered, f"{human}\n")
        self.assertIn("Decision 13: focus-40 at p=0.732265", rendered)
        self.assertIn("probabilities sum to 1.000000000000", rendered)
        self.assertIn("not locked-evaluation evidence", rendered)


class ProtocolInventoryTests(ForbiddenRuntimeTestCase):
    def test_inventory_is_exact_and_does_not_execute_runtime_paths(self) -> None:
        expected_counts = {
            "abrupt_traces": 3_584,
            "adaptive_diagnostic_scopes": 19,
            "calibration_rows": 152,
            "cluster_summaries": 16_128,
            "decisions": 6_635_520,
            "macro_contrasts": 6,
            "macro_strategy_rows": 7,
            "raw_trace_points": 1_032_192,
            "recovery_contrast_rows": 24,
            "recovery_rows": 28,
            "scenario_contrasts": 108,
            "scenario_strategy_rows": 126,
            "template_exposure_rows": 532,
            "trace_points": 8_064,
            "trace_series": 28,
            "trajectories": 23_040,
        }
        with (
            patch.object(
                protocol_inventory,
                "validate_experiment_config",
                wraps=protocol_inventory.validate_experiment_config,
            ) as validate,
            patch.object(
                protocol_inventory,
                "expected_publication_cardinalities",
                wraps=protocol_inventory.expected_publication_cardinalities,
            ) as cardinalities,
        ):
            document = protocol_inventory.build_inventory()

        validate.assert_called_once_with(protocol_inventory.DEFAULT_EXPERIMENT_CONFIG)
        cardinalities.assert_called_once_with(
            protocol_inventory.DEFAULT_EXPERIMENT_CONFIG
        )
        self.assertFalse(document["contains_evaluation_results"])
        self.assertEqual(document["expected_cardinalities"], expected_counts)
        self.assertEqual(
            document["call_surface"],
            [
                "validate_experiment_config",
                "expected_publication_cardinalities",
            ],
        )

    def test_protocol_source_has_a_closed_gworker_call_surface(self) -> None:
        source_path = REPOSITORY_ROOT / "scripts" / "protocol_inventory.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden = {
            "_issue_locked_evaluation_permit",
            "generate_environment",
            "run_experiment",
            "run_publication",
            "simulate_trajectory",
        }
        imported: set[str] = set()
        calls: set[str] = set()
        gworker_symbols: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "gworker.publication_runner")
                imported.update(alias.name for alias in node.names)
                if node.module is not None and node.module.startswith("gworker."):
                    gworker_symbols.update(
                        {alias.asname or alias.name: alias.name for alias in node.names}
                    )
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        self.assertFalse(imported & forbidden)
        self.assertFalse(calls & forbidden)
        called_gworker_symbols = {
            original for local, original in gworker_symbols.items() if local in calls
        }
        self.assertEqual(
            called_gworker_symbols,
            {
                "expected_publication_cardinalities",
                "validate_experiment_config",
            },
        )

    def test_inventory_document_and_both_renderers_are_deterministic(self) -> None:
        first = protocol_inventory.build_inventory()
        second = protocol_inventory.build_inventory()
        self.assertEqual(first, second)
        self.assertEqual(
            compact_digest(protocol_inventory.canonical_json, first),
            "3e4d053bae69698fe01d3228c01a5306b6fd6d09272063ff15d1313cdd871bb7",
        )
        pretty = protocol_inventory.canonical_json(first, pretty=True)
        human = protocol_inventory.render_human(first)
        self.assertEqual(
            text_digest(pretty),
            "4bb1d672ec623508c3ae1e044bd06786b56689296fb810a3d95c59ed98e3acb0",
        )
        self.assertEqual(
            text_digest(human),
            "659061d9f99131706a9bf7638c1d0e0029ce2683bf0e00efbbba372d6d6aef80",
        )

        json_output = io.StringIO()
        with redirect_stdout(json_output):
            self.assertEqual(protocol_inventory.main(["--json"]), 0)
        self.assertEqual(json.loads(json_output.getvalue()), first)
        self.assertEqual(json_output.getvalue(), f"{pretty}\n")

        human_output = io.StringIO()
        with redirect_stdout(human_output):
            self.assertEqual(protocol_inventory.main([]), 0)
        rendered = human_output.getvalue()
        self.assertEqual(rendered, f"{human}\n")
        self.assertIn("no evaluation executed", rendered)
        self.assertIn("decisions", rendered)
        self.assertIn("6,635,520", rendered)
        self.assertIn("Result data: none", rendered)


class JournalDemoTests(ForbiddenRuntimeTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.root = make_demo_repo(self.base)

    def test_real_store_reopens_replays_verifies_and_detects_copy_tamper(
        self,
    ) -> None:
        document = demo_journal.build_demo(
            self.root,
            WORKSPACE,
            reset=False,
        )
        self.assertEqual(document["workspace"], WORKSPACE.as_posix())
        journal = document["journal"]
        replay = document["replay"]
        tamper = document["tamper_copy"]
        assert isinstance(journal, dict)
        assert isinstance(replay, dict)
        assert isinstance(tamper, dict)
        self.assertEqual(journal["sqlite_quick_check"], "ok")
        self.assertEqual(journal["session_count"], 1)
        self.assertEqual(journal["event_count"], 6)
        self.assertTrue(journal["reopen_matches_written_events"])
        self.assertEqual(journal["database_permissions"]["mode"], "0600")  # type: ignore[index]
        self.assertEqual(journal["directory_permissions"]["mode"], "0700")  # type: ignore[index]
        self.assertEqual(replay["phase"], "completed")
        self.assertEqual(replay["revision"], 6)
        self.assertTrue(replay["is_terminal"])
        self.assertEqual(replay["interruption_count"], 1)
        self.assertEqual(
            tamper["sqlite_quick_check_before_domain_replay"],
            "ok",
        )
        self.assertTrue(tamper["separate_from_live_database"])
        self.assertTrue(tamper["detection"]["detected"])  # type: ignore[index]
        self.assertEqual(
            tamper["detection"]["error_type"],  # type: ignore[index]
            "CorruptJournal",
        )

        live_path = self.root / str(journal["database"])
        tamper_path = self.root / str(tamper["database"])
        self.assertNotEqual(os.stat(live_path).st_ino, os.stat(tamper_path).st_ino)
        reopened = SQLiteEventStore(live_path)
        self.assertEqual(reopened.verify().sqlite_check, "ok")
        with self.assertRaises(CorruptJournal):
            SQLiteEventStore(tamper_path).verify()

        created_files = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file() and path.name != "pyproject.toml"
        }
        self.assertTrue(created_files)
        self.assertTrue(
            all(path.startswith(f"{WORKSPACE.as_posix()}/") for path in created_files)
        )

    def test_journal_reset_and_renderers_are_deterministic(self) -> None:
        first = demo_journal.build_demo(self.root, WORKSPACE, reset=False)
        second = demo_journal.build_demo(self.root, WORKSPACE, reset=True)
        self.assertEqual(
            demo_journal.canonical_json(first, pretty=False),
            demo_journal.canonical_json(second, pretty=False),
        )
        self.assertEqual(
            compact_digest(demo_journal.canonical_json, first),
            "844250955a72a2f13ac89abe54ca9d64288188063d55fe8a5387b39978297812",
        )
        pretty = demo_journal.canonical_json(first, pretty=True)
        human = demo_journal.render_human(first)
        self.assertEqual(
            text_digest(pretty),
            "4147e2369a243fa1563eff4abb3856ff01645b573dd2023d5f1252bbc17f5164",
        )
        self.assertEqual(
            text_digest(human),
            "2c6284cc355ff0ad4a07dad7400d370ad3516597aaff99e78077e068d541a42c",
        )

        human_output = io.StringIO()
        with redirect_stdout(human_output):
            self.assertEqual(
                demo_journal.main(
                    [
                        "--repo-root",
                        str(self.root),
                        "--workspace",
                        str(WORKSPACE),
                        "--reset",
                    ]
                ),
                0,
            )
        rendered = human_output.getvalue()
        self.assertEqual(rendered, f"{human}\n")
        self.assertIn("real SQLite store", rendered)
        self.assertIn("PRAGMA quick_check=ok", rendered)
        self.assertIn("detected=true", rendered)

        json_output = io.StringIO()
        with redirect_stdout(json_output):
            self.assertEqual(
                demo_journal.main(
                    [
                        "--repo-root",
                        str(self.root),
                        "--workspace",
                        str(WORKSPACE),
                        "--reset",
                        "--json",
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(json_output.getvalue()), first)
        self.assertEqual(json_output.getvalue(), f"{pretty}\n")

    def test_nonempty_workspace_requires_explicit_reset(self) -> None:
        demo_journal.build_demo(self.root, WORKSPACE, reset=False)
        live_path = self.root / WORKSPACE / "live" / "events.sqlite3"
        before = live_path.read_bytes()
        with self.assertRaisesRegex(
            demo_journal.DemoPathError,
            "not empty",
        ):
            demo_journal.build_demo(self.root, WORKSPACE, reset=False)
        self.assertEqual(live_path.read_bytes(), before)

    def test_workspace_swap_after_preparation_cannot_create_outside(self) -> None:
        outside = self.base / "raced-workspace-target"
        outside.mkdir(mode=0o700)
        original = demo_journal.prepare_workspace

        def raced_prepare(
            repo_root: str | Path,
            workspace: str | Path,
            *,
            reset: bool,
        ) -> tuple[Path, Path]:
            root, location = original(repo_root, workspace, reset=reset)
            location.rmdir()
            location.symlink_to(outside, target_is_directory=True)
            return root, location

        with (
            patch.object(
                demo_journal,
                "prepare_workspace",
                side_effect=raced_prepare,
            ),
            self.assertRaises(demo_journal.DemoPathError),
        ):
            demo_journal.build_demo(self.root, WORKSPACE, reset=False)
        self.assertEqual(list(outside.iterdir()), [])

    def test_tamper_copy_uses_pinned_files_after_parent_swap(self) -> None:
        _, location = demo_journal.prepare_workspace(
            self.root,
            WORKSPACE,
            reset=False,
        )
        live_directory = location / demo_journal.LIVE_DIRECTORY_NAME
        tamper_directory = location / demo_journal.TAMPER_DIRECTORY_NAME
        live_directory.mkdir(mode=0o700)
        tamper_directory.mkdir(mode=0o700)
        live_path = live_directory / demo_journal.DATABASE_NAME
        tamper_path = tamper_directory / demo_journal.DATABASE_NAME
        live_store = SQLiteEventStore(live_path)
        for event in demo_journal._events():
            live_store.append(event)

        live_descriptor = os.open(
            live_directory,
            demo_journal._directory_flags(),
        )
        tamper_descriptor = os.open(
            tamper_directory,
            demo_journal._directory_flags(),
        )
        moved = location / "tamper-copy-moved"
        outside = self.base / "raced-tamper-target"
        outside.mkdir(mode=0o700)
        real_store = demo_journal.SQLiteEventStore

        def swapping_store(path: str | Path) -> SQLiteEventStore:
            store = real_store(path)
            if Path(path) == tamper_path:
                tamper_directory.rename(moved)
                tamper_directory.symlink_to(outside, target_is_directory=True)
            return store

        try:
            with patch.object(
                demo_journal,
                "SQLiteEventStore",
                side_effect=swapping_store,
            ):
                demo_journal._make_tamper_copy(
                    tamper_path,
                    source_directory=live_descriptor,
                    destination_directory=tamper_descriptor,
                )
        finally:
            os.close(tamper_descriptor)
            os.close(live_descriptor)

        self.assertFalse((outside / demo_journal.DATABASE_NAME).exists())
        moved_database = moved / demo_journal.DATABASE_NAME
        connection = sqlite3.connect(moved_database)
        try:
            stored = connection.execute(
                "SELECT event_json FROM events WHERE sequence = 3"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(stored, '{"schema_version":1}')

    def test_tamper_cleanup_attempts_every_fd_without_masking_primary(
        self,
    ) -> None:
        source_descriptor = os.open("/dev/null", os.O_RDONLY)
        destination_descriptor = os.open("/dev/null", os.O_RDONLY)
        real_close = os.close
        close_attempts: list[int] = []

        def flaky_close(descriptor: int) -> None:
            close_attempts.append(descriptor)
            if descriptor == destination_descriptor:
                raise OSError("synthetic destination close failure")
            real_close(descriptor)

        try:
            with (
                patch.object(
                    demo_journal,
                    "_open_database_at",
                    side_effect=(source_descriptor, destination_descriptor),
                ),
                patch.object(
                    demo_journal,
                    "SQLiteEventStore",
                    return_value=object(),
                ),
                patch.object(
                    demo_journal.sqlite3,
                    "connect",
                    side_effect=RuntimeError("primary connect failure"),
                ),
                patch.object(
                    demo_journal.os,
                    "close",
                    side_effect=flaky_close,
                ),
                self.assertRaisesRegex(RuntimeError, "primary connect failure"),
            ):
                demo_journal._make_tamper_copy(
                    Path("/unused/events.sqlite3"),
                    source_directory=10,
                    destination_directory=11,
                )
        finally:
            with suppress(OSError):
                real_close(destination_descriptor)

        self.assertIn(destination_descriptor, close_attempts)
        self.assertIn(source_descriptor, close_attempts)

    def test_build_cleanup_attempts_all_fds_without_masking_primary(self) -> None:
        workspace_descriptor = os.open("/dev/null", os.O_RDONLY)
        live_descriptor = os.open("/dev/null", os.O_RDONLY)
        tamper_descriptor = os.open("/dev/null", os.O_RDONLY)
        real_close = os.close
        close_attempts: list[int] = []
        root = Path("/synthetic/repo")
        location = root / WORKSPACE

        def flaky_close(descriptor: int) -> None:
            close_attempts.append(descriptor)
            if descriptor == tamper_descriptor:
                raise OSError("synthetic tamper close failure")
            real_close(descriptor)

        try:
            with (
                patch.object(
                    demo_journal,
                    "prepare_workspace",
                    return_value=(root, location),
                ),
                patch.object(
                    demo_journal,
                    "_open_relative_directory",
                    return_value=workspace_descriptor,
                ),
                patch.object(
                    demo_journal,
                    "_ensure_private_child_directory",
                    side_effect=(live_descriptor, tamper_descriptor),
                ),
                patch.object(
                    demo_journal,
                    "_verify_directory_identity",
                ),
                patch.object(
                    demo_journal,
                    "SQLiteEventStore",
                    side_effect=RuntimeError("primary store failure"),
                ),
                patch.object(
                    demo_journal.os,
                    "close",
                    side_effect=flaky_close,
                ),
                self.assertRaisesRegex(RuntimeError, "primary store failure"),
            ):
                demo_journal.build_demo(root, WORKSPACE, reset=False)
        finally:
            with suppress(OSError):
                real_close(tamper_descriptor)

        self.assertIn(tamper_descriptor, close_attempts)
        self.assertIn(live_descriptor, close_attempts)
        self.assertIn(workspace_descriptor, close_attempts)

    def test_escape_and_visual_demo_root_are_rejected_before_writes(self) -> None:
        outside = self.base / "outside"
        outside.mkdir(mode=0o700)
        sentinel = outside / "sentinel"
        sentinel.write_text("preserve", encoding="utf-8")
        cases = (
            Path("../outside"),
            outside / "journal",
            Path(".gworker/visual-demo"),
            Path(".gworker/visual-demo/bad name"),
        )
        for workspace in cases:
            with (
                self.subTest(workspace=workspace),
                self.assertRaises(demo_journal.DemoPathError),
            ):
                demo_journal.build_demo(self.root, workspace, reset=True)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertFalse((outside / "journal").exists())

    def test_symlinked_workspace_is_rejected_without_touching_target(self) -> None:
        base = self.root / ".gworker" / "visual-demo"
        base.mkdir(parents=True, mode=0o700)
        (self.root / ".gworker").chmod(0o700)
        base.chmod(0o700)
        outside = self.base / "symlink-target"
        outside.mkdir(mode=0o700)
        sentinel = outside / "sentinel"
        sentinel.write_text("preserve", encoding="utf-8")
        (base / "journal-recovery").symlink_to(
            outside,
            target_is_directory=True,
        )

        with self.assertRaisesRegex(
            demo_journal.DemoPathError,
            "symlinks",
        ):
            demo_journal.build_demo(self.root, WORKSPACE, reset=True)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_reset_prescans_nested_symlink_and_is_all_or_nothing(self) -> None:
        workspace = self.root / WORKSPACE
        workspace.mkdir(parents=True, mode=0o700)
        (self.root / ".gworker").chmod(0o700)
        (self.root / ".gworker" / "visual-demo").chmod(0o700)
        workspace.chmod(0o700)
        ordinary = workspace / "a-preserve.txt"
        ordinary.write_text("preserve", encoding="utf-8")
        ordinary.chmod(0o600)
        outside = self.base / "nested-target"
        outside.mkdir(mode=0o700)
        sentinel = outside / "sentinel"
        sentinel.write_text("outside", encoding="utf-8")
        (workspace / "z-link").symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(
            demo_journal.DemoPathError,
            "must not contain symlinks",
        ):
            demo_journal.build_demo(self.root, WORKSPACE, reset=True)
        self.assertEqual(ordinary.read_text(encoding="utf-8"), "preserve")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "outside")
        self.assertTrue((workspace / "z-link").is_symlink())

    def test_reset_rejects_hardlinks_without_removing_either_name(self) -> None:
        workspace = self.root / WORKSPACE
        workspace.mkdir(parents=True, mode=0o700)
        (self.root / ".gworker").chmod(0o700)
        (self.root / ".gworker" / "visual-demo").chmod(0o700)
        workspace.chmod(0o700)
        outside = self.base / "outside-hardlink"
        outside.write_text("preserve", encoding="utf-8")
        outside.chmod(0o600)
        alias = workspace / "linked-file"
        os.link(outside, alias)

        with self.assertRaisesRegex(
            demo_journal.DemoPathError,
            "hard links",
        ):
            demo_journal.build_demo(self.root, WORKSPACE, reset=True)
        self.assertEqual(outside.read_text(encoding="utf-8"), "preserve")
        self.assertEqual(alias.read_text(encoding="utf-8"), "preserve")
        self.assertEqual(os.stat(outside).st_nlink, 2)

    def test_symlinked_visual_demo_ancestor_is_rejected(self) -> None:
        (self.root / ".gworker").mkdir(mode=0o700)
        outside = self.base / "ancestor-target"
        outside.mkdir(mode=0o700)
        sentinel = outside / "sentinel"
        sentinel.write_text("outside", encoding="utf-8")
        (self.root / ".gworker" / "visual-demo").symlink_to(
            outside,
            target_is_directory=True,
        )

        with self.assertRaisesRegex(
            demo_journal.DemoPathError,
            "symlinks",
        ):
            demo_journal.build_demo(self.root, WORKSPACE, reset=True)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "outside")
        self.assertFalse((outside / "journal-recovery").exists())

    def test_cli_requires_an_explicit_workspace(self) -> None:
        stderr = io.StringIO()
        with (
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            demo_journal.main(["--repo-root", str(self.root)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--workspace", stderr.getvalue())
        self.assertFalse((self.root / ".gworker").exists())

    def test_script_contains_no_host_specific_absolute_path(self) -> None:
        source = (REPOSITORY_ROOT / "scripts" / "demo_journal.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("/home/", source)
        self.assertNotIn("/Users/", source)


if __name__ == "__main__":
    unittest.main()
