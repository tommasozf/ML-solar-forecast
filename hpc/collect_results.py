"""
Extract best hyperparameters from SLURM log files and write them into the
per-province hp_study_results.json files the notebooks already read.

Parses the "Best val_loss:" / "Best MAE:" summary block that each worker
prints at the end.  For a given study (province × model_type), the worker
that reports the lowest metric is authoritative (it saw the most trials).

Usage:
    python hpc/collect_results.py                        # all studies
    python hpc/collect_results.py --province caceres     # one province
    python hpc/collect_results.py --dry-run              # print only, no writes
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

PROVINCES = ["caceres", "cadiz", "zaragoza"]
TFT_TYPES = ["standard", "perfect_forecast", "nwp_forecast"]
XGB_TYPES = ["standard", "nwp"]

TFT_MODEL_DIR = {
    "standard": "standard",
    "perfect_forecast": "perfect_forecast",
    "nwp_forecast": "nwp_forecast",
}
XGB_MODEL_DIR = {
    "standard": "xgboost",
    "nwp": "xgboost_nwp",
}

# Maps (province, model_type) → SLURM job IDs for the successful second batch.
# First batch (1019688-1019709) all failed with FileNotFoundError.
TFT_JOB_IDS = {
    ("caceres", "standard"): [1019815],
    ("caceres", "perfect_forecast"): [1019816],
    ("caceres", "nwp_forecast"): [1019817],
    ("cadiz", "standard"): [1019818],
    ("cadiz", "perfect_forecast"): [1019819],
    ("cadiz", "nwp_forecast"): [1019820],
    ("zaragoza", "standard"): [1019821],
    ("zaragoza", "perfect_forecast"): [1019822],
    ("zaragoza", "nwp_forecast"): [1019823],
}
XGB_JOB_IDS = {
    ("caceres", "standard"): [1019824],
    ("caceres", "nwp"): [1019825],
    ("cadiz", "standard"): [1019826],
    ("cadiz", "nwp"): [1019827],
    ("zaragoza", "standard"): [1019828],
    ("zaragoza", "nwp"): [1019829],
}


@dataclass
class WorkerResult:
    file: str
    n_trials: int = 0
    best_trial: int = -1
    best_value: float = float("inf")
    best_params: dict = field(default_factory=dict)


def parse_log_file(path: str, metric_label: str) -> Optional[WorkerResult]:
    """Parse a SLURM .out file for the best-result summary block."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None

    # Look for "Best val_loss: 0.018459" or "Best MAE: 0.213855"
    m = re.search(rf"{re.escape(metric_label)}:\s+([\d.]+)", text)
    if not m:
        return None

    result = WorkerResult(file=os.path.basename(path))
    result.best_value = float(m.group(1))

    # Best trial number
    m_trial = re.search(r"Best trial.*?#(\d+)", text)
    if m_trial:
        result.best_trial = int(m_trial.group(1))

    # Total trials
    m_trials = re.search(r"Study has (\d+) total trials", text)
    if m_trials:
        result.n_trials = int(m_trials.group(1))

    # Extract param lines: "  key: value"
    params_section = text[m.start():]
    for pm in re.finditer(r"^\s{2}(\w+):\s+(.+)$", params_section, re.MULTILINE):
        key, val = pm.group(1), pm.group(2).strip()
        try:
            if "." in val or "e" in val.lower():
                result.best_params[key] = float(val)
            else:
                result.best_params[key] = int(val)
        except ValueError:
            result.best_params[key] = val

    return result


def collect_study(logs_dir: str, job_ids: list[int], prefix: str, metric_label: str) -> Optional[WorkerResult]:
    """Find the best result across all workers for a study."""
    best: Optional[WorkerResult] = None

    for jid in job_ids:
        pattern = os.path.join(logs_dir, f"{prefix}_{jid}_*.out")
        for path in sorted(glob(pattern)):
            r = parse_log_file(path, metric_label)
            if r and (best is None or r.best_value < best.best_value):
                best = r

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", choices=PROVINCES, default=None)
    parser.add_argument(
        "--logs-dir", default=None, dest="logs_dir",
        help="Directory containing .out files (default: hpc/logs/ next to this script)",
    )
    parser.add_argument(
        "--base-dir", default=None, dest="base_dir",
        help="Repo root (default: parent of this script's directory)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = args.base_dir or os.path.dirname(script_dir)
    logs_dir = args.logs_dir or os.path.join(script_dir, "logs")

    provinces = [args.province] if args.province else PROVINCES

    for prov in provinces:
        print(f"\n{'─'*60}")
        print(f"Province: {prov}")
        print(f"{'─'*60}")

        # ── TFT studies ──
        for mtype in TFT_TYPES:
            study_name = f"tft_{prov}_{mtype}"
            job_ids = TFT_JOB_IDS.get((prov, mtype), [])
            result = collect_study(logs_dir, job_ids, "tft_tft_hp", "Best val_loss")

            if result is None:
                print(f"  [{study_name}] — no results found, skipping")
                continue

            print(
                f"  [{study_name}] {result.n_trials} trials "
                f"| best val_loss={result.best_value:.6f} (trial #{result.best_trial})"
            )
            for k, v in result.best_params.items():
                print(f"    {k}: {v}")

            if not args.dry_run:
                model_dir = os.path.join(base_dir, "regional_analysis", prov, "models", TFT_MODEL_DIR[mtype])
                os.makedirs(model_dir, exist_ok=True)
                out_path = os.path.join(model_dir, "hp_study_results.json")
                existing = {}
                if os.path.exists(out_path):
                    with open(out_path) as f:
                        existing = json.load(f)
                existing.update({
                    "best_trial": result.best_trial,
                    "best_params": result.best_params,
                    "n_trials": result.n_trials,
                    "best_val_loss": result.best_value,
                    "source": "hpc_logs",
                })
                with open(out_path, "w") as f:
                    json.dump(existing, f, indent=2)
                print(f"    → written to {out_path}")

        # ── XGBoost studies ──
        for mtype in XGB_TYPES:
            study_name = f"xgb_{prov}_{mtype}"
            job_ids = XGB_JOB_IDS.get((prov, mtype), [])
            result = collect_study(logs_dir, job_ids, "xgb_xgb_hp", "Best MAE")

            if result is None:
                print(f"  [{study_name}] — no results found, skipping")
                continue

            print(
                f"  [{study_name}] {result.n_trials} trials "
                f"| best MAE={result.best_value:.6f} (trial #{result.best_trial})"
            )
            for k, v in result.best_params.items():
                print(f"    {k}: {v}")

            if not args.dry_run:
                model_dir = os.path.join(base_dir, "regional_analysis", prov, "models", XGB_MODEL_DIR.get(mtype, "xgboost"))
                os.makedirs(model_dir, exist_ok=True)
                out_path = os.path.join(model_dir, "hp_study_results.json")
                existing = {}
                if os.path.exists(out_path):
                    with open(out_path) as f:
                        existing = json.load(f)
                existing.update({
                    "best_trial": result.best_trial,
                    "best_params": result.best_params,
                    "n_trials": result.n_trials,
                    "best_val_loss": result.best_value,
                    "source": "hpc_logs",
                })
                with open(out_path, "w") as f:
                    json.dump(existing, f, indent=2)
                print(f"    → written to {out_path}")

    if args.dry_run:
        print("\n(dry-run — no files written)")


if __name__ == "__main__":
    main()
