"""
TFT hyperparameter search worker — designed to run as one task in a SLURM job array.
Multiple workers share the same Optuna SQLite study and each contribute n_trials trials.

Usage (single machine / interactive testing):
    python hpc/train_tft_hp.py --province caceres --model-type standard \
        --storage sqlite:///hpc/studies/caceres_standard.db --n-trials 10

Usage on ADA (called from submit_tft_hp.sbatch):
    python hpc/train_tft_hp.py --province $PROVINCE --model-type $MODEL_TYPE \
        --storage $STORAGE --n-trials $N_TRIALS_PER_WORKER --max-epochs $MAX_EPOCHS
"""

import argparse
import functools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Restore pre-PyTorch-2.6 torch.load behavior — needed for pytorch-forecasting checkpoints.
if not getattr(torch, "_orig_load", None):
    torch._orig_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return torch._orig_load(*args, **kwargs)


torch.load = _patched_torch_load

# ── Feature definitions ─────────────────────────────────────────────────────

CALENDAR_SOLAR = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos", "solar_zenith", "solar_azimuth", "clearsky_ghi",
]
ERA5_WEATHER = [
    "dewpoint_2m_C", "temperature_2m_C", "surface_pressure_hPa",
    "total_precip_mm", "ssrd_wm2", "strd_wm2", "kt", "dewpoint_depression_C",
]
NWP_FORECAST = [
    "ssrd_wm2_forecast", "temperature_2m_C_forecast", "dewpoint_2m_C_forecast",
    "surface_pressure_hPa_forecast", "total_precip_mm_forecast",
    "kt_forecast", "dewpoint_depression_C_forecast",
]

MODEL_CONFIGS = {
    "standard": {
        "train_csv": "train_processed.csv",
        "val_csv": "val_processed.csv",
        "params_json": "preprocessing_params.json",
        "known_future": CALENDAR_SOLAR,
        "observed": ERA5_WEATHER,
    },
    "perfect_forecast": {
        "train_csv": "train_processed.csv",
        "val_csv": "val_processed.csv",
        "params_json": "preprocessing_params.json",
        "known_future": CALENDAR_SOLAR + ERA5_WEATHER,
        "observed": [],
    },
    "nwp_forecast": {
        "train_csv": "nwp_train_processed.csv",
        "val_csv": "nwp_val_processed.csv",
        "params_json": "nwp_preprocessing_params.json",
        "known_future": CALENDAR_SOLAR + NWP_FORECAST,
        "observed": ERA5_WEATHER,
    },
}

MAX_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 24
TARGET = "pv_generation_mwh"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(province: str, model_type: str, base_dir: str):
    cfg = MODEL_CONFIGS[model_type]
    data_dir = os.path.join(base_dir, province, "data")

    train_df = pd.read_csv(
        os.path.join(data_dir, cfg["train_csv"]),
        parse_dates=["datetime_utc"], index_col="datetime_utc",
    )
    val_df = pd.read_csv(
        os.path.join(data_dir, cfg["val_csv"]),
        parse_dates=["datetime_utc"], index_col="datetime_utc",
    )

    assert train_df.isna().sum().sum() == 0, "NaN in training data"
    assert val_df.isna().sum().sum() == 0, "NaN in validation data"

    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    train_df["time_idx"] = np.arange(len(train_df))
    val_df["time_idx"] = np.arange(len(train_df), len(train_df) + len(val_df))
    train_df["group_id"] = "0"
    val_df["group_id"] = "0"

    with open(os.path.join(data_dir, cfg["params_json"])) as f:
        pp = json.load(f)

    return train_df, val_df, pp["target_mean"], pp["target_std"]


def build_datasets(train_df, val_df, known_future, observed):
    dataset_kwargs = dict(
        time_idx="time_idx",
        target=TARGET,
        group_ids=["group_id"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        min_encoder_length=MAX_ENCODER_LENGTH,
        time_varying_known_reals=known_future,
        target_normalizer=None,
        scalers={col: None for col in known_future},
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    if observed:
        dataset_kwargs["time_varying_unknown_reals"] = observed
        dataset_kwargs["scalers"].update({col: None for col in observed})

    training_dataset = TimeSeriesDataSet(train_df, **dataset_kwargs)

    val_with_context = pd.concat(
        [train_df.iloc[-MAX_ENCODER_LENGTH:], val_df], ignore_index=True
    )
    val_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_with_context, stop_randomization=True
    )
    assert len(val_dataset) > 100, f"Val dataset too small: {len(val_dataset)}"
    return training_dataset, val_dataset


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(training_dataset, train_dataloader, val_dataloader, max_epochs):
    def objective(trial):
        # ── Expanded search space (9 parameters vs original 6) ──
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.05, 0.5)
        hidden_continuous_size = trial.suggest_categorical(
            "hidden_continuous_size", [8, 16, 32, 64]
        )
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 1.0, log=True)
        reduce_on_plateau_patience = trial.suggest_categorical(
            "reduce_on_plateau_patience", [2, 3, 5]
        )

        # Rebuild dataloaders if batch_size changes
        t_dl = training_dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=4, pin_memory=True
        )
        v_dl = training_dataset.to_dataloader(
            train=False, batch_size=batch_size * 2, num_workers=4, pin_memory=True
        )
        # Use the prebuilt loaders for the default batch_size (64), otherwise use above
        use_t_dl = t_dl if batch_size != 64 else train_dataloader
        use_v_dl = v_dl if batch_size != 64 else val_dataloader

        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            learning_rate=learning_rate,
            loss=QuantileLoss(),
            optimizer="adam",
            reduce_on_plateau_patience=reduce_on_plateau_patience,
            output_size=7,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            gradient_clip_val=gradient_clip_val,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", patience=3, mode="min", verbose=False
                )
            ],
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
        )

        try:
            trainer.fit(tft, train_dataloaders=use_t_dl, val_dataloaders=use_v_dl)
            val_loss = trainer.callback_metrics["val_loss"].item()
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}", flush=True)
            val_loss = float("inf")
        finally:
            del tft, trainer
            torch.cuda.empty_cache()

        return val_loss

    return objective


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TFT Optuna HP worker")
    parser.add_argument(
        "--province", required=True, choices=["caceres", "cadiz", "zaragoza"]
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        dest="model_type",
    )
    parser.add_argument("--n-trials", type=int, default=10, dest="n_trials")
    parser.add_argument("--max-epochs", type=int, default=15, dest="max_epochs")
    parser.add_argument(
        "--storage",
        required=True,
        help="Optuna storage URL, e.g. sqlite:////scistor/.../studies/study.db",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        dest="base_dir",
        help="Root of repo (default: parent of this script's directory)",
    )
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pl.seed_everything(42 + int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    print(
        f"Worker starting: province={args.province} model_type={args.model_type} "
        f"n_trials={args.n_trials} max_epochs={args.max_epochs}",
        flush=True,
    )
    print(f"CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    cfg = MODEL_CONFIGS[args.model_type]
    train_df, val_df, target_mean, target_std = load_data(
        args.province, args.model_type, args.base_dir
    )
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}", flush=True)

    training_dataset, val_dataset = build_datasets(
        train_df, val_df, cfg["known_future"], cfg["observed"]
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=64, num_workers=4, pin_memory=True
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False, batch_size=128, num_workers=4, pin_memory=True
    )

    study_name = f"tft_{args.province}_{args.model_type}"
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = make_objective(
        training_dataset, train_dataloader, val_dataloader, args.max_epochs
    )
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print(f"\nWorker done. Study has {len(study.trials)} total trials.", flush=True)
    print(f"Best trial so far: #{study.best_trial.number}", flush=True)
    print(f"Best val_loss: {study.best_value:.6f}", flush=True)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
