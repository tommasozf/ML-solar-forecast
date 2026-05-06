"""
Extract best hyperparameters from finished Optuna studies and write them back
into the per-province hp_study_results.json files the notebooks already read.

Usage:
    python hpc/collect_results.py                        # all studies
    python hpc/collect_results.py --province caceres     # one province
    python hpc/collect_results.py --dry-run              # print only, no writes
"""

import argparse
import json
import os
import sys

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROVINCES = ["caceres", "cadiz", "zaragoza"]
TFT_TYPES = ["standard", "perfect_forecast", "nwp_forecast"]
XGB_TYPES = ["standard", "nwp"]

# Maps model_type → subdirectory under province/models/
TFT_MODEL_DIR = {
    "standard": "standard",
    "perfect_forecast": "perfect_forecast",
    "nwp_forecast": "nwp_forecast",
}
XGB_MODEL_DIR = {
    "standard": "xgboost",
    "nwp": "xgboost_nwp",
}


def load_study(db_path: str, study_name: str):
    if not os.path.exists(db_path):
        return None
    storage = f"sqlite:///{db_path}"
    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"  Could not load {db_path}: {e}")
        return None


def summarise_study(study):
    finished = [t for t in study.trials if t.value is not None and t.value < float("inf")]
    return {
        "n_trials_total": len(study.trials),
        "n_trials_finished": len(finished),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", choices=PROVINCES, default=None)
    parser.add_argument(
        "--studies-dir",
        default=None,
        dest="studies_dir",
        help="Directory containing .db files (default: hpc/studies/ next to this script)",
    )
    parser.add_argument(
        "--base-dir", default=None, dest="base_dir",
        help="Repo root (default: parent of this script's directory)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = args.base_dir or os.path.dirname(script_dir)
    studies_dir = args.studies_dir or os.path.join(script_dir, "studies")

    provinces = [args.province] if args.province else PROVINCES

    for prov in provinces:
        print(f"\n{'─'*50}")
        print(f"Province: {prov}")
        print(f"{'─'*50}")

        # ── TFT studies ───────────────────────────────────────────────────────
        for mtype in TFT_TYPES:
            study_name = f"tft_{prov}_{mtype}"
            db_path = os.path.join(studies_dir, f"{study_name}.db")
            study = load_study(db_path, study_name)
            if study is None:
                print(f"  [{study_name}] — not found, skipping")
                continue

            s = summarise_study(study)
            print(
                f"  [{study_name}] {s['n_trials_finished']}/{s['n_trials_total']} trials "
                f"| best val_loss={s['best_value']:.6f} (trial #{s['best_trial']})"
            )
            print(f"    params: {s['best_params']}")

            if not args.dry_run:
                model_dir = os.path.join(base_dir, "regional_analysis", prov, "models", TFT_MODEL_DIR[mtype])
                os.makedirs(model_dir, exist_ok=True)
                out_path = os.path.join(model_dir, "hp_study_results.json")
                existing = {}
                if os.path.exists(out_path):
                    with open(out_path) as f:
                        existing = json.load(f)
                existing.update({
                    "best_trial": s["best_trial"],
                    "best_params": s["best_params"],
                    "n_trials": s["n_trials_finished"],
                    "best_val_loss": s["best_value"],
                    "source": "hpc_optuna",
                })
                with open(out_path, "w") as f:
                    json.dump(existing, f, indent=2)
                print(f"    → written to {out_path}")

        # ── XGBoost studies ───────────────────────────────────────────────────
        for mtype in XGB_TYPES:
            study_name = f"xgb_{prov}_{mtype}"
            db_path = os.path.join(studies_dir, f"{study_name}.db")
            study = load_study(db_path, study_name)
            if study is None:
                print(f"  [{study_name}] — not found, skipping")
                continue

            s = summarise_study(study)
            print(
                f"  [{study_name}] {s['n_trials_finished']}/{s['n_trials_total']} trials "
                f"| best MAE={s['best_value']:.6f} (trial #{s['best_trial']})"
            )
            print(f"    params: {s['best_params']}")

            if not args.dry_run:
                model_dir = os.path.join(base_dir, "regional_analysis", prov, "models", XGB_MODEL_DIR.get(mtype, "xgboost"))
                os.makedirs(model_dir, exist_ok=True)
                out_path = os.path.join(model_dir, "hp_study_results.json")
                existing = {}
                if os.path.exists(out_path):
                    with open(out_path) as f:
                        existing = json.load(f)
                existing.update({
                    "best_trial": s["best_trial"],
                    "best_params": s["best_params"],
                    "n_trials": s["n_trials_finished"],
                    "best_val_loss": s["best_value"],
                    "source": "hpc_optuna",
                })
                with open(out_path, "w") as f:
                    json.dump(existing, f, indent=2)
                print(f"    → written to {out_path}")

    if args.dry_run:
        print("\n(dry-run — no files written)")


if __name__ == "__main__":
    main()
