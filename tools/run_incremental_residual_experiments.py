"""Plan or sequentially run paired FSCIL residual ablations.

This launcher uses only the standard library. Planning is the default; pass
``--execute`` to train. Every variant of one seed reuses one support manifest.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFIX = "TRAINER.BiMC.RESIDUAL."
METRIC_NAMES = (
    "final_accuracy",
    "average_accuracy",
    "base_accuracy",
    "novel_accuracy",
    "harmonic_accuracy",
    "old_to_new_rate",
    "forgetting",
)
REQUIRED_NUMERIC = {"final_accuracy", "average_accuracy", "base_accuracy"}
MANAGED_KEYS = {
    "SEED", "OUTPUT_DIR", "DATASET.SUPPORT_SEED", "DATASET.SUPPORT_MANIFEST",
    *(PREFIX + key for key in (
        "ENABLED", "DICTIONARY", "RANK", "MAX_DELTA", "GAIN", "OLD_LOSS_WEIGHT",
    )),
}


def variants_for_suite(suite, rank=8, max_delta=0.2, gain=1.0, old_loss_weight=1.0):
    """Return controlled variants; identity uses the feature dimension as rank."""
    common = {
        PREFIX + "ENABLED": True,
        PREFIX + "DICTIONARY": "meta",
        PREFIX + "RANK": rank,
        PREFIX + "MAX_DELTA": max_delta,
        PREFIX + "GAIN": gain,
        PREFIX + "OLD_LOSS_WEIGHT": old_loss_weight,
    }
    definitions = [
        ("baseline", {"ENABLED": False}),
        ("zero_residual", {"GAIN": 0.0}),
        ("random", {"DICTIONARY": "random"}),
        ("residual_svd", {"DICTIONARY": "residual_svd"}),
        ("meta", {}),
    ]
    if suite == "extended":
        definitions.extend([
            ("meta_no_old_loss", {"OLD_LOSS_WEIGHT": 0.0}),
            ("full_rank", {"DICTIONARY": "identity"}),
        ])
    elif suite != "core":
        raise ValueError("unknown suite: " + suite)
    result = []
    for name, changes in definitions:
        overrides = dict(common)
        overrides.update({PREFIX + key: value for key, value in changes.items()})
        result.append((name, overrides))
    return result


def validate_extra_opts(opts):
    if len(opts) % 2:
        raise ValueError("--opts requires KEY VALUE pairs")
    keys = opts[::2]
    if len(set(keys)) != len(keys):
        raise ValueError("--opts contains duplicate keys")
    collision = set(keys) & MANAGED_KEYS
    if collision:
        raise ValueError("matrix-managed options cannot be overridden: " + ", ".join(sorted(collision)))


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def source_digest(repo_root):
    """Detect implementation changes before reusing an existing completed run."""
    paths = [repo_root / "main.py", Path(__file__).resolve()]
    for directory in ("engine", "models", "utils", "datasets", "dataloader", "clip"):
        folder = repo_root / directory
        if folder.is_dir():
            paths.extend(folder.rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=str):
        if path.is_file():
            try:
                name = path.relative_to(repo_root).as_posix()
            except ValueError:
                name = path.name
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def serialize_cfg_value(value):
    # yacs accepts the capitalized Python boolean representation.
    return str(value)


def build_plan(args, repo_root=REPO_ROOT):
    validate_extra_opts(args.opts)
    data_cfg = Path(args.data_cfg).resolve()
    train_cfg = Path(args.train_cfg).resolve()
    for path in (data_cfg, train_cfg):
        if not path.is_file():
            raise ValueError("configuration does not exist: " + str(path))
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain distinct integers")
    if min(args.seeds) < 0:
        raise ValueError("--seeds must be nonnegative for deterministic paired sampling")
    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    if not math.isfinite(args.max_delta) or args.max_delta <= 0:
        raise ValueError("--max-delta must be finite and positive")
    if not math.isfinite(args.gain) or not 0 < args.gain <= 1:
        raise ValueError("--gain must be in (0, 1]; zero is a separate ablation")
    if not math.isfinite(args.old_loss_weight) or args.old_loss_weight <= 0:
        raise ValueError("--old-loss-weight must be finite and positive; no-old-loss is a separate ablation")
    configs = {
        str(data_cfg): hashlib.sha256(data_cfg.read_bytes()).hexdigest(),
        str(train_cfg): hashlib.sha256(train_cfg.read_bytes()).hexdigest(),
    }
    dataset_opts = {
        key: value for key, value in zip(args.opts[::2], args.opts[1::2])
        if key.startswith("DATASET.")
    }
    support_id = sha256_json({"config": configs[str(data_cfg)], "opts": dataset_opts})[:12]
    dataset_id = data_cfg.stem + "_" + support_id
    output_root = Path(args.output_root).resolve()
    source_hash = source_digest(repo_root)
    matrix = variants_for_suite(args.suite, args.rank, args.max_delta, args.gain, args.old_loss_weight)
    selected = set(args.variants or [name for name, _ in matrix])
    unknown = selected - {name for name, _ in matrix}
    if unknown:
        raise ValueError("variants not in selected suite: " + ", ".join(sorted(unknown)))
    plan = []
    for seed in args.seeds:
        support_path = output_root / "support" / dataset_id / ("seed_" + str(seed) + ".json")
        for name, overrides in matrix:
            if name not in selected:
                continue
            output_dir = output_root / data_cfg.stem / name / ("seed_" + str(seed))
            command = [
                args.python, str(repo_root / "main.py"),
                "--data_cfg", str(data_cfg), "--train_cfg", str(train_cfg), "--opts",
                *args.opts,
                "SEED", str(seed), "DATASET.SUPPORT_SEED", str(seed),
                "DATASET.SUPPORT_MANIFEST", str(support_path),
                "OUTPUT_DIR", str(output_dir),
            ]
            for key, value in overrides.items():
                command.extend([key, serialize_cfg_value(value)])
            fingerprint = sha256_json({
                "command": command, "cwd": str(repo_root),
                "configs": configs, "source": source_hash,
            })
            plan.append({
                "variant": name, "seed": seed, "support_seed": seed,
                "support_manifest": str(support_path), "output_dir": str(output_dir),
                "command": command, "cwd": str(repo_root), "fingerprint": fingerprint,
                "config_hashes": configs, "source_hash": source_hash,
            })
    return plan


def load_completed_metrics(path):
    """Reject incomplete, malformed and nonfinite results instead of skipping them."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("status") != "completed":
            return None
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            return None
        for key in METRIC_NAMES:
            if key not in summary:
                return None
            value = summary[key]
            if value is None and key not in REQUIRED_NUMERIC:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return None
            if not (-100 <= value <= 100 if key == "forgetting" else 0 <= value <= 100):
                return None
        sessions = payload.get("sessions")
        if not isinstance(sessions, list) or not sessions:
            return None
        for row in sessions:
            value = row.get("accuracy") if isinstance(row, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                return None
        return payload
    except (OSError, ValueError, TypeError):
        return None


def read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def completed_result(run):
    output_dir = Path(run["output_dir"])
    state = read_json(output_dir / "run_state.json")
    spec = read_json(output_dir / "run_spec.json")
    if not isinstance(state, dict) or not isinstance(spec, dict):
        return None
    if state.get("status") != "completed" or state.get("fingerprint") != run["fingerprint"]:
        return None
    if spec.get("fingerprint") != run["fingerprint"]:
        return None
    try:
        support_hash = hashlib.sha256(Path(run["support_manifest"]).read_bytes()).hexdigest()
    except OSError:
        return None
    if state.get("support_manifest_sha256") != support_hash:
        return None
    return load_completed_metrics(output_dir / "metrics.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def describe_command(command):
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def metric_statistics(values):
    numbers = [value for value in values if value is not None]
    return {
        "n": len(numbers),
        "mean": statistics.mean(numbers) if numbers else None,
        "std": statistics.stdev(numbers) if len(numbers) > 1 else None,
    }


def aggregate_results(rows):
    successful = [row for row in rows if row["status"] in ("completed", "skipped")]
    variants = sorted({row["variant"] for row in rows})
    aggregate = {}
    paired = {}
    baseline = {row["seed"]: row for row in successful if row["variant"] == "baseline"}
    for name in variants:
        group = [row for row in successful if row["variant"] == name]
        aggregate[name] = {
            "n_runs": len(group),
            "metrics": {key: metric_statistics([row["summary"][key] for row in group]) for key in METRIC_NAMES},
        }
        if name == "baseline":
            continue
        paired[name] = {}
        for key in METRIC_NAMES:
            differences = []
            for row in group:
                reference = baseline.get(row["seed"])
                value = row["summary"][key]
                previous = reference["summary"][key] if reference else None
                if value is not None and previous is not None:
                    differences.append(value - previous)
            paired[name][key] = metric_statistics(differences)
    return aggregate, paired


def save_summary(path, args, rows, planned_runs):
    aggregates, paired = aggregate_results(rows)
    payload = {
        "schema_version": 1, "generated_at": utc_now(), "metrics_unit": "percent",
        "suite": args.suite, "seeds": args.seeds, "planned_runs": planned_runs,
        "completed_runs": sum(row["status"] in ("completed", "skipped") for row in rows),
        "failed_runs": sum(row["status"] == "failed" for row in rows),
        "runs": rows, "aggregates": aggregates, "paired_delta_vs_baseline": paired,
    }
    write_json(path, payload)


def execute_plan(plan, args):
    rows = []
    summary_path = Path(args.output_root).resolve() / "summary.json"
    failed = False
    for index, run in enumerate(plan, start=1):
        output_dir = Path(run["output_dir"])
        row = {key: run[key] for key in ("variant", "seed", "support_seed", "output_dir", "support_manifest", "fingerprint")}
        metrics = completed_result(run)
        if metrics is not None and not args.rerun:
            print("[{}/{}] skip completed {} seed={}".format(index, len(plan), run["variant"], run["seed"]), flush=True)
            row.update(status="skipped", summary=metrics["summary"])
            rows.append(row)
            save_summary(summary_path, args, rows, len(plan))
            continue
        old_spec = read_json(output_dir / "run_spec.json")
        if isinstance(old_spec, dict) and old_spec.get("fingerprint") != run["fingerprint"] and not args.rerun:
            row.update(status="failed", error="Existing run has a different command, config or source fingerprint. Use a new --output-root or --rerun.")
            print(row["error"] + " " + str(output_dir), file=sys.stderr)
            rows.append(row)
            save_summary(summary_path, args, rows, len(plan))
            failed = True
            if not args.keep_going:
                break
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(run["support_manifest"]).parent.mkdir(parents=True, exist_ok=True)
        # Keep any old result, but prevent it from certifying a new failed run.
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            archive_name = "metrics.previous." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
            metrics_path.replace(output_dir / archive_name)
        write_json(output_dir / "run_spec.json", run)
        state = {"status": "running", "fingerprint": run["fingerprint"], "started_at": utc_now()}
        write_json(output_dir / "run_state.json", state)
        print("[{}/{}] run {} seed={} -> {}".format(index, len(plan), run["variant"], run["seed"], output_dir / "run.log"), flush=True)
        try:
            with (output_dir / "run.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(run["command"], cwd=run["cwd"], stdout=log, stderr=subprocess.STDOUT, check=False)
            metrics = load_completed_metrics(metrics_path) if process.returncode == 0 else None
            if process.returncode != 0:
                error = "Training process exited with code " + str(process.returncode)
            elif metrics is None:
                error = "Training exited successfully but metrics.json is missing, incomplete or invalid"
            elif not Path(run["support_manifest"]).is_file():
                error = "Training did not save the shared support manifest"
            else:
                error = None
                state["support_manifest_sha256"] = hashlib.sha256(Path(run["support_manifest"]).read_bytes()).hexdigest()
        except OSError as exc:
            metrics = None
            error = str(exc)
        state["finished_at"] = utc_now()
        if error is None:
            row.update(status="completed", summary=metrics["summary"])
            state["status"] = "completed"
        else:
            row.update(status="failed", error=error)
            state.update(status="failed", error=error)
            print(error, file=sys.stderr, flush=True)
            failed = True
        write_json(output_dir / "run_state.json", state)
        rows.append(row)
        save_summary(summary_path, args, rows, len(plan))
        if error is not None and not args.keep_going:
            break
    print("Summary: " + str(summary_path), flush=True)
    return 1 if failed else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cfg", required=True, help="Dataset YAML; no dataset is downloaded by this tool")
    parser.add_argument("--train-cfg", default=str(REPO_ROOT / "configs/trainers/bimc_incremental_residual.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs/incremental_residual"))
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for main.py")
    parser.add_argument("--suite", choices=("core", "extended"), default="core")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--variants", nargs="+", help="Subset of variants from the chosen suite")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--old-loss-weight", type=float, default=1.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Execute the matrix sequentially; default is planning only")
    mode.add_argument("--dry-run", action="store_true", help="Print commands only; no files or training processes are created")
    parser.add_argument("--rerun", action="store_true", help="Rerun even completed/mismatching runs; preserve previous metrics in dated files")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failures, still exit nonzero if any run fails")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Additional yacs KEY VALUE pairs; place last")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except (OSError, ValueError) as exc:
        print("Invalid experiment plan: " + str(exc), file=sys.stderr)
        return 2
    if not args.execute:
        for index, run in enumerate(plan, start=1):
            print("[{}/{}] {} seed={}".format(index, len(plan), run["variant"], run["seed"]))
            print(describe_command(run["command"]))
        print("Dry run: {} commands; no training or output writes. Pass --execute to run.".format(len(plan)))
        return 0
    return execute_plan(plan, args)


if __name__ == "__main__":
    raise SystemExit(main())
