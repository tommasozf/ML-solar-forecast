"""
Bootstrap Confidence Intervals for Test-Set Metrics
====================================================
Resamples test residuals with replacement to compute 95% CIs
on R², RMSE, MAE, and nRMSE — no retraining needed.

Run after all eval notebooks (06, 07, 11, 12, 13) have been executed.

Usage:
    python bootstrap_confidence_intervals.py

Output:
    - ../figures/fig_bootstrap_ci.png  (R² with 95% CI error bars)
    - prints full CI table to stdout
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
SEED = 420

REGIONS = ['caceres', 'cadiz', 'zaragoza']
REGION_DISPLAY = {'caceres': 'Cáceres', 'cadiz': 'Cádiz', 'zaragoza': 'Zaragoza'}

MODEL_CONFIGS = {
    'TFT Standard': {
        'test_csv': 'models/standard/test_predictions.csv',
        'val_csv': 'models/standard/val_predictions.csv',
        'json_path': 'models/standard/hp_study_results.json',
        'params_json': 'data/preprocessing_params.json',
    },
    'TFT Perfect': {
        'test_csv': 'models/perfect_forecast/test_predictions.csv',
        'val_csv': 'models/perfect_forecast/val_predictions_pf.csv',
        'json_path': 'models/perfect_forecast/hp_study_results_pf.json',
        'params_json': 'data/preprocessing_params.json',
    },
    'TFT NWP': {
        'test_csv': 'models/nwp_forecast/nwp_test_predictions.csv',
        'val_csv': 'models/nwp_forecast/nwp_val_predictions.csv',
        'json_path': 'models/nwp_forecast/nwp_hp_study_results.json',
        'params_json': 'data/preprocessing_params.json',
    },
    'XGBoost Standard': {
        'json_path': 'models/xgboost/xgboost_hp_study_results.json',
        'params_json': 'data/preprocessing_params.json',
    },
    'XGBoost NWP': {
        'json_path': 'models/xgboost/xgboost_nwp_hp_study_results.json',
        'params_json': 'data/preprocessing_params.json',
    },
    'Stacking': {
        'test_csv': 'models/stacking_predictions_test.csv',
        'stacking': True,
        'json_path': 'models/stacking_results.json',
        'params_json': 'data/preprocessing_params.json',
    },
}


def bootstrap_metrics(actual, predicted, target_mean, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Resample (actual, predicted) pairs and compute metrics on each resample."""
    rng = np.random.default_rng(SEED)
    n = len(actual)

    r2_boot = np.zeros(n_boot)
    rmse_boot = np.zeros(n_boot)
    mae_boot = np.zeros(n_boot)
    nrmse_boot = np.zeros(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = actual[idx]
        p = predicted[idx]
        residuals = a - p

        rmse = np.sqrt((residuals ** 2).mean())
        mae = np.abs(residuals).mean()
        ss_res = (residuals ** 2).sum()
        ss_tot = ((a - a.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        nrmse = (rmse / target_mean) * 100

        r2_boot[b] = r2
        rmse_boot[b] = rmse
        mae_boot[b] = mae
        nrmse_boot[b] = nrmse

    alpha = (1 - ci) / 2
    results = {}
    for name, arr in [('R2', r2_boot), ('RMSE_MWh', rmse_boot),
                       ('MAE_MWh', mae_boot), ('nRMSE_pct', nrmse_boot)]:
        results[name] = {
            'mean': float(np.mean(arr)),
            'lo': float(np.percentile(arr, 100 * alpha)),
            'hi': float(np.percentile(arr, 100 * (1 - alpha))),
            'std': float(np.std(arr)),
        }
    return results


def load_predictions(region, model_name, config):
    """Try to load test predictions; fall back to val predictions."""
    params_path = os.path.join('../regional_analysis', region, config['params_json'])

    with open(params_path) as f:
        pp = json.load(f)
    target_mean = pp['target_mean']

    # Stacking: special format
    if config.get('stacking'):
        csv_path = os.path.join('../regional_analysis', region, config.get('test_csv', ''))
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df['actual_mwh'].values, df['ensemble_pred_mwh'].values, target_mean
        return None, None, target_mean

    # Try test CSV first, then fall back to val CSV
    for csv_key in ['test_csv', 'val_csv']:
        csv_path_rel = config.get(csv_key)
        if not csv_path_rel:
            continue
        csv_path = os.path.join('../regional_analysis', region, csv_path_rel)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'actual_mwh' in df.columns and 'predicted_mwh' in df.columns:
                actual = df['actual_mwh'].values
                predicted = df['predicted_mwh'].values

                # Handle flattened overlapping windows (val CSVs only)
                n_expected = 4416 if 'test' in csv_key else 4344
                if len(actual) > n_expected * 2:
                    n_windows = len(actual) // 24
                    pred_arr = predicted.reshape(n_windows, 24)
                    act_arr = actual.reshape(n_windows, 24)

                    agg_pred = np.zeros(n_expected)
                    agg_actual = np.zeros(n_expected)
                    agg_count = np.zeros(n_expected)
                    for i in range(n_windows):
                        for h in range(24):
                            t = i + h
                            if t < n_expected:
                                agg_pred[t] += pred_arr[i, h]
                                agg_actual[t] += act_arr[i, h]
                                agg_count[t] += 1
                    mask = agg_count > 0
                    agg_pred[mask] /= agg_count[mask]
                    agg_actual[mask] /= agg_count[mask]
                    actual = agg_actual[mask]
                    predicted = agg_pred[mask]

                source = 'TEST' if 'test' in csv_key else 'VAL'
                print(f'    [{source}] {csv_path} ({len(actual)} samples)')
                return actual, predicted, target_mean

    return None, None, target_mean


def main():
    os.makedirs('../figures', exist_ok=True)

    all_results = []
    print("=" * 90)
    print(f"{'Model':<22s} {'Province':<12s} {'R²':>8s}  {'95% CI':>16s}  {'nRMSE%':>8s}  {'95% CI':>16s}")
    print("=" * 90)

    for region in REGIONS:
        for model_name, config in MODEL_CONFIGS.items():
            actual, predicted, target_mean = load_predictions(region, model_name, config)

            if actual is None:
                # No per-sample predictions — skip bootstrap
                continue

            ci = bootstrap_metrics(actual, predicted, target_mean)

            row = {
                'region': region,
                'region_display': REGION_DISPLAY[region],
                'model': model_name,
                'R2_mean': ci['R2']['mean'],
                'R2_lo': ci['R2']['lo'],
                'R2_hi': ci['R2']['hi'],
                'R2_std': ci['R2']['std'],
                'RMSE_mean': ci['RMSE_MWh']['mean'],
                'RMSE_lo': ci['RMSE_MWh']['lo'],
                'RMSE_hi': ci['RMSE_MWh']['hi'],
                'MAE_mean': ci['MAE_MWh']['mean'],
                'MAE_lo': ci['MAE_MWh']['lo'],
                'MAE_hi': ci['MAE_MWh']['hi'],
                'nRMSE_mean': ci['nRMSE_pct']['mean'],
                'nRMSE_lo': ci['nRMSE_pct']['lo'],
                'nRMSE_hi': ci['nRMSE_pct']['hi'],
            }
            all_results.append(row)

            print(f"{model_name:<22s} {REGION_DISPLAY[region]:<12s} "
                  f"{ci['R2']['mean']:>8.4f}  [{ci['R2']['lo']:.4f}, {ci['R2']['hi']:.4f}]  "
                  f"{ci['nRMSE_pct']['mean']:>8.2f}  [{ci['nRMSE_pct']['lo']:.2f}, {ci['nRMSE_pct']['hi']:.2f}]")

    print("=" * 90)

    if not all_results:
        print("\nNo prediction CSVs found. Run eval notebooks first.")
        return

    df = pd.DataFrame(all_results)

    # ── Figure: R² with 95% CI error bars ──
    models_in_data = df['model'].unique()
    n_models = len(models_in_data)
    n_regions = len(REGIONS)

    fig, ax = plt.subplots(figsize=(12, 5))

    bar_width = 0.22
    x = np.arange(n_regions)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#795548']

    for i, model in enumerate(models_in_data):
        model_df = df[df['model'] == model]
        means = []
        errs_lo = []
        errs_hi = []
        for region in REGIONS:
            row = model_df[model_df['region'] == region]
            if len(row) == 0:
                means.append(0)
                errs_lo.append(0)
                errs_hi.append(0)
            else:
                r = row.iloc[0]
                means.append(r['R2_mean'])
                errs_lo.append(r['R2_mean'] - r['R2_lo'])
                errs_hi.append(r['R2_hi'] - r['R2_mean'])

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, label=model,
                      color=colors[i % len(colors)], alpha=0.8,
                      yerr=[errs_lo, errs_hi], capsize=3, error_kw={'linewidth': 1})

    ax.set_xticks(x)
    ax.set_xticklabels([REGION_DISPLAY[r] for r in REGIONS], fontsize=11)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Test/Validation R² with 95% Bootstrap Confidence Intervals', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig_bootstrap_ci.png', dpi=200, bbox_inches='tight')
    plt.savefig('../figures/fig_bootstrap_ci.pdf', bbox_inches='tight')
    print(f"\nFigure saved: ../figures/fig_bootstrap_ci.png")

    # Save results to JSON
    df.to_json('../figures/bootstrap_ci_results.json', orient='records', indent=2)
    print(f"Results saved: ../figures/bootstrap_ci_results.json")


if __name__ == '__main__':
    main()
