"""
XGBoost hyperparameter search worker — runs on CPU, designed for SLURM job arrays.
Multiple workers share the same Optuna SQLite study.

Usage:
    python hpc/train_xgb_hp.py --province caceres --model-type standard \
        --storage sqlite:///hpc/studies/xgb_caceres_standard.db --n-trials 40
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")

TARGET = "pv_generation_mwh"

# Standard XGBoost uses ERA5 weather features + calendar features as columns.
# NWP XGBoost also includes NWP forecast columns.
# We simply drop non-feature columns and use everything else.
DROP_COLS = {"datetime_utc", "time_idx", "group_id", TARGET}


def load_xy(province: str, model_type: str, base_dir: str):
    data_dir = os.path.join(base_dir, province, "data")

    if model_type == "nwp":
        train_csv, val_csv = "nwp_train_processed.csv", "nwp_val_processed.csv"
        params_json = "nwp_preprocessing_params.json"
    else:
        train_csv, val_csv = "train_processed.csv", "val_processed.csv"
        params_json = "preprocessing_params.json"

    train_df = pd.read_csv(os.path.join(data_dir, train_csv))
    val_df = pd.read_csv(os.path.join(data_dir, val_csv))

    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET].values

    with open(os.path.join(data_dir, params_json)) as f:
        pp = json.load(f)

    return X_train, y_train, X_val, y_val, pp["target_mean"], pp["target_std"], feature_cols


def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            max_bin=trial.suggest_categorical("max_bin", [128, 256, 512]),
            tree_method="hist",
            device="cpu",
            random_state=42,
            verbosity=0,
        )
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    return objective


def main():
    parser = argparse.ArgumentParser(description="XGBoost Optuna HP worker")
    parser.add_argument(
        "--province", required=True, choices=["caceres", "cadiz", "zaragoza"]
    )
    parser.add_argument(
        "--model-type", required=True, choices=["standard", "nwp"], dest="model_type"
    )
    parser.add_argument("--n-trials", type=int, default=40, dest="n_trials")
    parser.add_argument("--storage", required=True)
    parser.add_argument("--base-dir", default=None, dest="base_dir")
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(
        f"XGB worker: province={args.province} model_type={args.model_type} "
        f"n_trials={args.n_trials}",
        flush=True,
    )

    X_train, y_train, X_val, y_val, t_mean, t_std, feat_cols = load_xy(
        args.province, args.model_type, args.base_dir
    )
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Features: {len(feat_cols)}", flush=True)

    study_name = f"xgb_{args.province}_{args.model_type}"
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        make_objective(X_train, y_train, X_val, y_val),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    print(f"\nWorker done. Study has {len(study.trials)} total trials.", flush=True)
    print(f"Best MAE: {study.best_value:.6f}", flush=True)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
