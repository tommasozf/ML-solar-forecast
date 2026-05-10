"""
Evaluate trained TFT models on the held-out test set.
Loads best checkpoint, generates predictions, computes metrics (MAE, RMSE, R², nRMSE),
and saves results + predictions CSV.

Usage:
    python hpc/evaluate_tft_test.py --province caceres --model-type standard
    python hpc/evaluate_tft_test.py --all
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

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
        "test_csv": "test_processed.csv",
        "params_json": "preprocessing_params.json",
        "known_future": CALENDAR_SOLAR,
        "observed": ERA5_WEATHER,
        "model_subdir": "standard",
    },
    "perfect_forecast": {
        "train_csv": "train_processed.csv",
        "val_csv": "val_processed.csv",
        "test_csv": "test_processed.csv",
        "params_json": "preprocessing_params.json",
        "known_future": CALENDAR_SOLAR + ERA5_WEATHER,
        "observed": [],
        "model_subdir": "perfect_forecast",
    },
    "nwp_forecast": {
        "train_csv": "nwp_train_processed.csv",
        "val_csv": "nwp_val_processed.csv",
        "test_csv": "nwp_test_processed.csv",
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


def find_best_checkpoint(model_dir: str) -> str:
    """Find the most recent tft_best*.ckpt in model_dir."""
    candidates = glob.glob(os.path.join(model_dir, "tft_best*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No tft_best*.ckpt found in {model_dir}")
    return max(candidates, key=os.path.getmtime)


def compute_metrics(actual_mwh: np.ndarray, predicted_mwh: np.ndarray, target_mean: float):
    residuals = actual_mwh - predicted_mwh
    mae = np.abs(residuals).mean()
    rmse = np.sqrt((residuals ** 2).mean())
    ss_res = (residuals ** 2).sum()
    ss_tot = ((actual_mwh - actual_mwh.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    nrmse = (rmse / target_mean) * 100
    return {
        "MAE_MWh": float(mae),
        "RMSE_MWh": float(rmse),
        "R2": float(r2),
        "nRMSE_pct": float(nrmse),
    }


def evaluate_one(province: str, model_type: str, base_dir: str, seed: int):
    pl.seed_everything(seed)
    cfg = MODEL_CONFIGS[model_type]
    data_dir = os.path.join(base_dir, "regional_analysis", province, "data")
    model_dir = os.path.join(base_dir, "regional_analysis", province, "models", cfg["model_subdir"])

    # Load preprocessing params for denormalization
    params_path = os.path.join(data_dir, cfg["params_json"])
    with open(params_path) as f:
        params = json.load(f)
    target_mean = params["target_mean"]
    target_std = params["target_std"]

    # Load data splits
    train_df = pd.read_csv(
        os.path.join(data_dir, cfg["train_csv"]),
        parse_dates=["datetime_utc"], index_col="datetime_utc",
    ).reset_index()
    val_df = pd.read_csv(
        os.path.join(data_dir, cfg["val_csv"]),
        parse_dates=["datetime_utc"], index_col="datetime_utc",
    ).reset_index()
    test_df = pd.read_csv(
        os.path.join(data_dir, cfg["test_csv"]),
        parse_dates=["datetime_utc"], index_col="datetime_utc",
    ).reset_index()

    # Assign continuous time_idx
    n_train = len(train_df)
    n_val = len(val_df)
    train_df["time_idx"] = np.arange(n_train)
    val_df["time_idx"] = np.arange(n_train, n_train + n_val)
    test_df["time_idx"] = np.arange(n_train + n_val, n_train + n_val + len(test_df))
    train_df["group_id"] = "0"
    val_df["group_id"] = "0"
    test_df["group_id"] = "0"

    print(f"\n{'='*60}")
    print(f"Evaluating: {province} / {model_type}")
    print(f"Test set: {len(test_df):,} rows")
    print(f"{'='*60}")

    # Build training dataset (needed to reconstruct dataset from_dataset)
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

    # Build test dataset with context from val tail
    test_with_context = pd.concat(
        [val_df.iloc[-MAX_ENCODER_LENGTH:], test_df], ignore_index=True
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, test_with_context, stop_randomization=True
    )
    test_dataloader = test_dataset.to_dataloader(
        train=False, batch_size=64, num_workers=4, pin_memory=True
    )

    # Load best checkpoint
    ckpt_path = find_best_checkpoint(model_dir)
    print(f"Checkpoint: {ckpt_path}")
    best_model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    best_model.eval()
    best_model.cuda() if torch.cuda.is_available() else None

    # Generate predictions
    predictions = best_model.predict(
        test_dataloader, mode="prediction", return_x=True, return_y=True,
        trainer_kwargs={"accelerator": "auto"},
    )
    pred_tensor = predictions.output
    if pred_tensor.ndim == 3:
        # Quantile output: take median (index 3 for 7 quantiles)
        pred_z = pred_tensor[:, :, 3].cpu().numpy()
    else:
        pred_z = pred_tensor.cpu().numpy()

    # Aggregate overlapping windows: each test hour may be predicted by multiple windows
    n_test = len(test_df)
    n_windows = pred_z.shape[0]
    agg_sum = np.zeros(n_test)
    agg_count = np.zeros(n_test)

    for i in range(n_windows):
        for h in range(MAX_PREDICTION_LENGTH):
            t = i + h
            if t < n_test:
                agg_sum[t] += pred_z[i, h]
                agg_count[t] += 1

    mask = agg_count > 0
    pred_z_agg = np.zeros(n_test)
    pred_z_agg[mask] = agg_sum[mask] / agg_count[mask]

    # Denormalize
    pred_mwh = np.clip(pred_z_agg * target_std + target_mean, 0, None)
    actual_z = test_df[TARGET].values
    actual_mwh = np.clip(actual_z * target_std + target_mean, 0, None)

    # Compute metrics
    metrics = compute_metrics(actual_mwh[mask], pred_mwh[mask], target_mean)

    # Persistence baseline (24h lag)
    persistence_mwh = np.clip(
        np.roll(actual_z, 24) * target_std + target_mean, 0, None
    )
    persistence_metrics = compute_metrics(
        actual_mwh[24:], persistence_mwh[24:], target_mean
    )

    # MASE (relative to persistence)
    mae_model = metrics["MAE_MWh"]
    mae_persistence = persistence_metrics["MAE_MWh"]
    mase = mae_model / mae_persistence if mae_persistence > 0 else float("inf")

    print(f"\nTest Metrics:")
    print(f"  MAE:   {metrics['MAE_MWh']:.2f} MWh")
    print(f"  RMSE:  {metrics['RMSE_MWh']:.2f} MWh")
    print(f"  R²:    {metrics['R2']:.4f}")
    print(f"  nRMSE: {metrics['nRMSE_pct']:.2f}%")
    print(f"  MASE:  {mase:.4f} (vs persistence-24h)")
    print(f"\nPersistence baseline:")
    print(f"  MAE:   {persistence_metrics['MAE_MWh']:.2f} MWh")
    print(f"  RMSE:  {persistence_metrics['RMSE_MWh']:.2f} MWh")
    print(f"  R²:    {persistence_metrics['R2']:.4f}")

    # Per-horizon metrics
    horizon_metrics = []
    for h in range(MAX_PREDICTION_LENGTH):
        h_preds_z = pred_z[:, h]
        h_preds_mwh = np.clip(h_preds_z * target_std + target_mean, 0, None)
        # Actual values at each horizon offset
        h_actuals_z = np.array([
            actual_z[i + h] for i in range(n_windows) if (i + h) < n_test
        ])
        h_actuals_mwh = np.clip(h_actuals_z * target_std + target_mean, 0, None)
        h_preds_mwh = h_preds_mwh[:len(h_actuals_mwh)]
        h_metrics = compute_metrics(h_actuals_mwh, h_preds_mwh, target_mean)
        h_metrics["horizon_h"] = h + 1
        horizon_metrics.append(h_metrics)

    # Save predictions CSV
    predictions_df = pd.DataFrame({
        "datetime_utc": test_df["datetime_utc"].values,
        "actual_mwh": actual_mwh,
        "predicted_mwh": pred_mwh,
        "residual_mwh": actual_mwh - pred_mwh,
    })
    pred_csv_path = os.path.join(model_dir, "test_predictions.csv")
    predictions_df.to_csv(pred_csv_path, index=False)
    print(f"\nPredictions saved: {pred_csv_path}")

    # Save results to hp_study_results.json
    hp_path = os.path.join(model_dir, "hp_study_results.json")
    with open(hp_path) as f:
        hp_data = json.load(f)

    hp_data["test_metrics"] = metrics
    hp_data["test_metrics"]["MASE_vs_persistence24h"] = mase
    hp_data["test_persistence_baseline"] = persistence_metrics
    hp_data["test_horizon_metrics"] = horizon_metrics
    hp_data["test_checkpoint"] = ckpt_path

    with open(hp_path, "w") as f:
        json.dump(hp_data, f, indent=2)
    print(f"Results saved: {hp_path}")

    del best_model
    torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate TFT on test set")
    parser.add_argument("--province", choices=PROVINCES)
    parser.add_argument("--model-type", choices=list(MODEL_CONFIGS.keys()), dest="model_type")
    parser.add_argument("--all", action="store_true", help="Evaluate all 9 combinations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dir", default=None, dest="base_dir")
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    all_results = {}

    if args.all:
        for prov in PROVINCES:
            for mtype in MODEL_CONFIGS:
                metrics = evaluate_one(prov, mtype, args.base_dir, args.seed)
                all_results[f"{prov}/{mtype}"] = metrics

        print(f"\n\n{'='*70}")
        print("SUMMARY - Test Set Results")
        print(f"{'='*70}")
        print(f"{'Province/Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'nRMSE%':>8}")
        print("-" * 70)
        for key, m in all_results.items():
            print(f"{key:<30} {m['MAE_MWh']:>8.2f} {m['RMSE_MWh']:>8.2f} {m['R2']:>8.4f} {m['nRMSE_pct']:>8.2f}")

    elif args.province and args.model_type:
        evaluate_one(args.province, args.model_type, args.base_dir, args.seed)
    else:
        parser.error("Provide --province and --model-type, or use --all")


if __name__ == "__main__":
    main()
