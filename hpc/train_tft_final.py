"""
Train a TFT model using the best hyperparameters from the HP search.
Reads HPs from the hp_study_results.json written by collect_results.py.

Usage:
    python hpc/train_tft_final.py --province caceres --model-type standard
    python hpc/train_tft_final.py --province cadiz --model-type nwp_forecast --max-epochs 100
    python hpc/train_tft_final.py --all                # train all 9 combinations
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

if not getattr(torch, "_orig_load", None):
    torch._orig_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return torch._orig_load(*args, **kwargs)


torch.load = _patched_torch_load

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
        "model_subdir": "standard",
    },
    "perfect_forecast": {
        "train_csv": "train_processed.csv",
        "val_csv": "val_processed.csv",
        "params_json": "preprocessing_params.json",
        "known_future": CALENDAR_SOLAR + ERA5_WEATHER,
        "observed": [],
        "model_subdir": "perfect_forecast",
    },
    "nwp_forecast": {
        "train_csv": "nwp_train_processed.csv",
        "val_csv": "nwp_val_processed.csv",
        "params_json": "nwp_preprocessing_params.json",
        "known_future": CALENDAR_SOLAR + NWP_FORECAST,
        "observed": ERA5_WEATHER,
        "model_subdir": "nwp_forecast",
    },
}

MAX_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 24
TARGET = "pv_generation_mwh"
PROVINCES = ["caceres", "cadiz", "zaragoza"]


def train_one(province: str, model_type: str, base_dir: str, max_epochs: int, seed: int):
    pl.seed_everything(seed)
    cfg = MODEL_CONFIGS[model_type]
    data_dir = os.path.join(base_dir, "regional_analysis", province, "data")
    model_dir = os.path.join(base_dir, "regional_analysis", province, "models", cfg["model_subdir"])
    os.makedirs(model_dir, exist_ok=True)

    hp_path = os.path.join(model_dir, "hp_study_results.json")
    with open(hp_path) as f:
        hp_data = json.load(f)
    hp = hp_data["best_params"]

    print(f"\n{'='*60}")
    print(f"Training: {province} / {model_type}")
    print(f"HPs from {hp_data.get('n_trials', '?')} trials (source: {hp_data.get('source', '?')})")
    for k, v in hp.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # ── Load data ──
    train_df = pd.read_csv(
        os.path.join(data_dir, cfg["train_csv"]),
        parse_dates=["datetime_utc"], index_col="datetime_utc",
    )
    val_df = pd.read_csv(
        os.path.join(data_dir, cfg["val_csv"]),
        parse_dates=["datetime_utc"], index_col="datetime_utc",
    )

    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    train_df["time_idx"] = np.arange(len(train_df))
    val_df["time_idx"] = np.arange(len(train_df), len(train_df) + len(val_df))
    train_df["group_id"] = "0"
    val_df["group_id"] = "0"

    print(f"Train: {len(train_df):,} rows  Val: {len(val_df):,} rows")

    # ── Build datasets ──
    dataset_kwargs = dict(
        time_idx="time_idx",
        target=TARGET,
        group_ids=["group_id"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        min_encoder_length=MAX_ENCODER_LENGTH,
        time_varying_known_reals=cfg["known_future"],
        target_normalizer=None,
        scalers={col: None for col in cfg["known_future"]},
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    if cfg["observed"]:
        dataset_kwargs["time_varying_unknown_reals"] = cfg["observed"]
        dataset_kwargs["scalers"].update({col: None for col in cfg["observed"]})

    training_dataset = TimeSeriesDataSet(train_df, **dataset_kwargs)

    val_with_context = pd.concat(
        [train_df.iloc[-MAX_ENCODER_LENGTH:], val_df], ignore_index=True
    )
    val_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_with_context, stop_randomization=True
    )

    batch_size = int(hp.get("batch_size", 64))
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=4, pin_memory=True
    )

    # ── Build model ──
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=int(hp["hidden_size"]),
        lstm_layers=int(hp["lstm_layers"]),
        attention_head_size=int(hp["attention_head_size"]),
        dropout=float(hp["dropout"]),
        hidden_continuous_size=int(hp["hidden_continuous_size"]),
        learning_rate=float(hp["learning_rate"]),
        loss=QuantileLoss(),
        optimizer="adam",
        reduce_on_plateau_patience=int(hp.get("reduce_on_plateau_patience", 5)),
        output_size=7,
    )

    n_params = sum(p.numel() for p in tft.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Train ──
    gradient_clip_val = float(hp.get("gradient_clip_val", 0.1))

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="tft_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=gradient_clip_val,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True),
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=50,
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss:   {checkpoint_callback.best_model_score:.6f}")

    # ── Update hp_study_results.json with final training info ──
    hp_data["training_config"] = {
        "max_epochs": max_epochs,
        "early_stopping_patience": 10,
        "gradient_clip_val": gradient_clip_val,
        "batch_size": batch_size,
        "optimizer": "adam",
        "loss": "QuantileLoss",
        "quantiles": [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
    }
    hp_data["sequence_config"] = {
        "max_encoder_length": MAX_ENCODER_LENGTH,
        "max_prediction_length": MAX_PREDICTION_LENGTH,
    }
    with open(hp_path, "w") as f:
        json.dump(hp_data, f, indent=2)

    del tft, trainer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Train TFT with best HPs from HP search")
    parser.add_argument("--province", choices=PROVINCES)
    parser.add_argument("--model-type", choices=list(MODEL_CONFIGS.keys()), dest="model_type")
    parser.add_argument("--all", action="store_true", help="Train all 9 province × model_type combos")
    parser.add_argument("--max-epochs", type=int, default=75, dest="max_epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-dir", default=None, dest="base_dir",
        help="Repo root (default: parent of this script's directory)",
    )
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.all:
        for prov in PROVINCES:
            for mtype in MODEL_CONFIGS:
                train_one(prov, mtype, args.base_dir, args.max_epochs, args.seed)
    elif args.province and args.model_type:
        train_one(args.province, args.model_type, args.base_dir, args.max_epochs, args.seed)
    else:
        parser.error("Provide --province and --model-type, or use --all")


if __name__ == "__main__":
    main()
