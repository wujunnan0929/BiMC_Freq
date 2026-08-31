"""Standard-library checks for paired runs, resumption and failure reporting."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock
import uuid

from tools import run_incremental_residual_experiments as experiments


def example_metrics(accuracy=70.0):
    return {
        "schema_version": 1,
        "status": "completed",
        "sessions": [{"session": 0, "accuracy": 80.0}, {"session": 1, "accuracy": accuracy}],
        "summary": {
            "final_accuracy": accuracy,
            "average_accuracy": (80.0 + accuracy) / 2,
            "base_accuracy": 75.0,
            "novel_accuracy": 60.0,
            "harmonic_accuracy": 200.0 / 3,
            "old_to_new_rate": 5.0,
            "forgetting": 5.0,
        },
    }


class ResidualExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.temp_root = experiments.REPO_ROOT / ".codex_tmp" / "residual_tool_tests"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        # Python 3.12 mkdtemp's owner-only ACL is incompatible with some Windows
        # sandbox identities. Use a normal workspace directory with a unique name.
        self.root = self.temp_root / ("residual experiment " + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.cleanup_temporary)
        self.data_cfg = self.root / "data.yaml"
        self.train_cfg = self.root / "train.yaml"
        self.data_cfg.write_text("DATASET:\n  NAME: test\n", encoding="utf-8")
        self.train_cfg.write_text("SEED: 1\n", encoding="utf-8")

    def cleanup_temporary(self):
        # Resolve and check the recursive cleanup target within our test root.
        self.root.resolve().relative_to(self.temp_root.resolve())
        shutil.rmtree(self.root)

    def args(self, *extra):
        return experiments.parse_args([
            "--data-cfg", str(self.data_cfg), "--train-cfg", str(self.train_cfg),
            "--output-root", str(self.root / "outputs"), "--seeds", "1", "2", *extra,
        ])

    def test_variants_pair_support_and_isolate_outputs(self):
        plan = experiments.build_plan(self.args("--suite", "extended"))
        self.assertEqual(len(plan), 14)
        self.assertEqual(len({run["output_dir"] for run in plan}), 14)
        for seed in (1, 2):
            group = [run for run in plan if run["seed"] == seed]
            self.assertEqual(len({run["support_manifest"] for run in group}), 1)
            for run in group:
                opts = run["command"][run["command"].index("--opts") + 1:]
                cfg = dict(zip(opts[::2], opts[1::2]))
                self.assertEqual(cfg["DATASET.SUPPORT_SEED"], str(seed))
                self.assertEqual(cfg["SEED"], str(seed))
                if run["variant"] == "zero_residual":
                    self.assertEqual(cfg[experiments.PREFIX + "GAIN"], "0.0")
                    self.assertEqual(cfg[experiments.PREFIX + "ENABLED"], "True")
                if run["variant"] == "full_rank":
                    self.assertEqual(cfg[experiments.PREFIX + "DICTIONARY"], "identity")
        self.assertEqual(len({run["support_manifest"] for run in plan}), 2)

    def test_rejects_seed_override_and_records_common_opts_in_fingerprint(self):
        with self.assertRaisesRegex(ValueError, "matrix-managed"):
            experiments.build_plan(self.args("--opts", "DATASET.SUPPORT_SEED", "9"))
        plain = experiments.build_plan(self.args())
        changed = experiments.build_plan(self.args("--opts", "TRAINER.BiMC.RESIDUAL.TRAIN_STEPS", "3"))
        self.assertNotEqual(plain[0]["fingerprint"], changed[0]["fingerprint"])
        self.assertEqual(plain[0]["support_manifest"], changed[0]["support_manifest"])
        changed_dataset = experiments.build_plan(self.args("--opts", "DATASET.NUM_INC_SHOT", "3"))
        self.assertNotEqual(plain[0]["support_manifest"], changed_dataset[0]["support_manifest"])

    def test_dry_run_never_writes_or_launches_training(self):
        args = self.args("--dry-run")
        with mock.patch.object(experiments, "parse_args", return_value=args), mock.patch.object(experiments.subprocess, "run") as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiments.main([]), 0)
        execute.assert_not_called()
        self.assertFalse(Path(args.output_root).exists())

    def test_incomplete_and_nonfinite_metrics_cannot_certify_success(self):
        path = self.root / "metrics.json"
        metrics = example_metrics()
        experiments.write_json(path, metrics)
        self.assertIsNotNone(experiments.load_completed_metrics(path))
        metrics["summary"]["final_accuracy"] = float("nan")
        path.write_text(json.dumps(metrics), encoding="utf-8")
        self.assertIsNone(experiments.load_completed_metrics(path))
        metrics = example_metrics()
        metrics["status"] = "running"
        experiments.write_json(path, metrics)
        self.assertIsNone(experiments.load_completed_metrics(path))
        metrics = example_metrics()
        del metrics["summary"]["old_to_new_rate"]
        experiments.write_json(path, metrics)
        self.assertIsNone(experiments.load_completed_metrics(path))

    def successful_process(self, command, **kwargs):
        self.assertIsInstance(command, list)
        self.assertFalse(kwargs.get("shell", False))
        opts = command[command.index("--opts") + 1:]
        cfg = dict(zip(opts[::2], opts[1::2]))
        experiments.write_json(Path(cfg["OUTPUT_DIR"]) / "metrics.json", example_metrics())
        experiments.write_json(Path(cfg["DATASET.SUPPORT_MANIFEST"]), {"seed": int(cfg["SEED"])})
        return subprocess.CompletedProcess(command, 0)

    def test_success_resumes_only_matching_completed_runs_with_same_manifest(self):
        args = self.args("--variants", "baseline", "residual_svd", "--execute")
        plan = experiments.build_plan(args)
        with mock.patch.object(experiments.subprocess, "run", side_effect=self.successful_process) as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiments.execute_plan(plan, args), 0)
            self.assertEqual(execute.call_count, 4)
            self.assertEqual(experiments.execute_plan(plan, args), 0)
            self.assertEqual(execute.call_count, 4)
        self.assertIsNotNone(experiments.completed_result(plan[0]))
        experiments.write_json(Path(plan[0]["support_manifest"]), {"seed": "tampered"})
        self.assertIsNone(experiments.completed_result(plan[0]))
        self.assertIsNone(experiments.completed_result(plan[1]))
        self.assertIsNotNone(experiments.completed_result(plan[2]))
        summary = experiments.read_json(Path(args.output_root) / "summary.json")
        self.assertEqual(summary["completed_runs"], 4)
        self.assertEqual(summary["paired_delta_vs_baseline"]["residual_svd"]["final_accuracy"]["mean"], 0.0)

    def test_failed_rerun_archives_old_metrics_and_does_not_leave_success(self):
        args = self.args("--variants", "baseline", "--execute")
        plan = experiments.build_plan(args)[:1]
        with mock.patch.object(experiments.subprocess, "run", side_effect=self.successful_process), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiments.execute_plan(plan, args), 0)
        args.rerun = True
        with mock.patch.object(experiments.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiments.execute_plan(plan, args), 1)
        self.assertIsNone(experiments.completed_result(plan[0]))
        output = Path(plan[0]["output_dir"])
        self.assertFalse((output / "metrics.json").exists())
        self.assertEqual(len(list(output.glob("metrics.previous.*.json"))), 1)
        summary = experiments.read_json(Path(args.output_root) / "summary.json")
        self.assertEqual(summary["failed_runs"], 1)

    def test_zero_exit_without_metrics_is_failure_and_stops(self):
        args = self.args("--execute")
        plan = experiments.build_plan(args)
        with mock.patch.object(experiments.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as execute, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiments.execute_plan(plan, args), 1)
        self.assertEqual(execute.call_count, 1)

    def test_aggregates_use_seed_pairs_and_do_not_fill_missing_values(self):
        rows = []
        for variant, seed, accuracy in (("baseline", 1, 70), ("baseline", 2, 75), ("meta", 1, 73), ("meta", 2, 76), ("meta", 3, 50)):
            summary = example_metrics(accuracy)["summary"]
            summary["forgetting"] = None
            rows.append({"variant": variant, "seed": seed, "status": "completed", "summary": summary})
        aggregates, paired = experiments.aggregate_results(rows)
        self.assertEqual(aggregates["meta"]["n_runs"], 3)
        self.assertEqual(paired["meta"]["final_accuracy"]["n"], 2)
        self.assertEqual(paired["meta"]["final_accuracy"]["mean"], 2)
        self.assertIsNone(paired["meta"]["forgetting"]["mean"])
        self.assertEqual(paired["meta"]["forgetting"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
