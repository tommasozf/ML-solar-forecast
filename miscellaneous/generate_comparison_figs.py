"""Generate nRMSE and delta-R² figures from cross_province_comparison data."""
import sys; sys.path.insert(0, '.')
from figure_style import *
apply_style()

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
PROVINCES = {
    'Cáceres':  {'dir': 'regional_analysis/caceres',  'capacity_mwp': 2463.01},
    'Cádiz':    {'dir': 'regional_analysis/cadiz',    'capacity_mwp': 714.60},
    'Zaragoza': {'dir': 'regional_analysis/zaragoza', 'capacity_mwp': 462.40},
}

MODELS_JSON = {
    'TFT Standard':         ['models/standard/hp_study_results.json'],
    'TFT NWP':              ['models/nwp_forecast/nwp_hp_study_results.json'],
    'TFT Perfect Forecast': ['models/perfect_forecast/hp_study_results_pf.json'],
    'XGBoost Standard':     ['models/xgboost/xgboost_hp_study_results.json'],
    'XGBoost NWP':          ['models/xgboost/xgboost_nwp_hp_study_results.json'],
    'Stacking Ensemble':    ['models/stacking_results.json'],
}

# Load all results
rows = []
for prov_name, prov_info in PROVINCES.items():
    csv_name = prov_name.lower().replace('á', 'a').replace('é', 'e') + '_assembled_dataset.csv'
    csv_path = os.path.join(PROJECT_ROOT, prov_info['dir'], 'data', csv_name)
    peak_gen = pd.read_csv(csv_path, usecols=['pv_generation_gwh'])['pv_generation_gwh'].max()

    for model_name, json_paths in MODELS_JSON.items():
        metrics = None
        for jp in json_paths:
            fp = os.path.join(PROJECT_ROOT, prov_info['dir'], jp)
            if os.path.exists(fp):
                with open(fp) as f:
                    data = json.load(f)
                metrics = data.get('test_metrics_mwh') or data.get('val_metrics_mwh')
                break
        if metrics:
            rows.append({
                'Province': prov_name, 'Model': model_name,
                'R2': metrics.get('R2'), 'RMSE_MWh': metrics.get('RMSE_MWh'),
                'Peak_Gen_MWh': peak_gen,
            })

results = pd.DataFrame(rows)
results['nRMSE_pct'] = (results['RMSE_MWh'] / results['Peak_Gen_MWh'] * 100).round(2)

prov_names = list(PROVINCES.keys())
x = np.arange(len(prov_names))

col_order = ['XGBoost Standard', 'XGBoost NWP', 'TFT Standard', 'TFT NWP',
             'TFT Perfect Forecast', 'Stacking Ensemble']
unified_colors = {
    'XGBoost Standard':     MODEL_COLORS['XGBoost Standard'],
    'XGBoost NWP':          MODEL_COLORS['XGBoost NWP'],
    'TFT Standard':         MODEL_COLORS['TFT Standard'],
    'TFT NWP':              MODEL_COLORS['TFT NWP'],
    'TFT Perfect Forecast': MODEL_COLORS['TFT Perfect'],
    'Stacking Ensemble':    MODEL_COLORS['Stacking'],
}
unified_short = {
    'XGBoost Standard': 'XGB Std.', 'XGBoost NWP': 'XGB NWP',
    'TFT Standard': 'TFT Std.', 'TFT NWP': 'TFT NWP',
    'TFT Perfect Forecast': 'TFT Perfect', 'Stacking Ensemble': 'Stacking',
}

available = results.dropna(subset=['R2'])
col_order = [c for c in col_order if c in available['Model'].values]
n_models = len(col_order)
bar_width = 0.8 / n_models

# ── Figure: nRMSE by Province ──
fig, ax = plt.subplots(figsize=(10, 5))
for i, model in enumerate(col_order):
    vals = []
    for prov in prov_names:
        row = available[(available['Province'] == prov) & (available['Model'] == model)]
        vals.append(row['nRMSE_pct'].values[0] if len(row) > 0 else 0)
    pos = x + i * bar_width
    ax.bar(pos, vals, bar_width * 0.9, label=unified_short[model], color=unified_colors[model])

ax.set_xticks(x + bar_width * (n_models - 1) / 2)
ax.set_xticklabels(prov_names)
ax.set_ylabel('nRMSE (% of peak generation)')
ax.set_title('Test-set nRMSE by Province and Model Variant')
ax.legend(loc='upper left', ncol=2, fontsize=8, framealpha=0.9)
save_fig(fig, 'fig_nrmse_by_province', folder='../figures')
plt.close()
print('nRMSE figure: DONE')

# ── Figure: Delta R² pair ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) R² improvement over baseline
bar_w = 0.15
delta_items = [
    ('TFT: NWP-Std',  'TFT Standard', 'TFT NWP',              MODEL_COLORS['TFT NWP']),
    ('TFT: PF-Std',   'TFT Standard', 'TFT Perfect Forecast',  MODEL_COLORS['TFT Perfect']),
    ('XGB: NWP-Std',  'XGBoost Standard', 'XGBoost NWP',       MODEL_COLORS['XGBoost NWP']),
]
if 'Stacking Ensemble' in available['Model'].values:
    delta_items.append(('Stack-Best NWP', 'XGBoost NWP', 'Stacking Ensemble', MODEL_COLORS['Stacking']))

n_deltas = len(delta_items)
for j, (label, base, improved, color) in enumerate(delta_items):
    deltas = []
    for prov in prov_names:
        base_row = available[(available['Province'] == prov) & (available['Model'] == base)]
        imp_row  = available[(available['Province'] == prov) & (available['Model'] == improved)]
        if len(base_row) and len(imp_row):
            deltas.append(imp_row['R2'].values[0] - base_row['R2'].values[0])
        else:
            deltas.append(0)
    offset = (j - (n_deltas - 1) / 2) * bar_w
    axes[0].bar(x + offset, deltas, bar_w * 0.9, label=label, color=color, alpha=0.85)

axes[0].set_xticks(x)
axes[0].set_xticklabels(prov_names)
axes[0].set_ylabel(r'$\Delta R^2$')
axes[0].set_title(r'(a) $R^2$ Improvement over Baseline')
axes[0].legend(fontsize=7, loc='lower left')
axes[0].axhline(0, color='black', lw=0.5)

# (b) Architecture advantage
delta_arch = [
    ('Standard features',      'XGBoost Standard', 'TFT Standard', MODEL_COLORS['TFT Standard']),
    ('NWP features',           'XGBoost NWP',      'TFT NWP',      MODEL_COLORS['TFT NWP']),
]
if 'Stacking Ensemble' in available['Model'].values:
    delta_arch.append(('Stacking vs XGB NWP', 'XGBoost NWP', 'Stacking Ensemble', MODEL_COLORS['Stacking']))

n_bars = len(delta_arch)
bar_w_d = 0.22
for j, (label, xgb_model, tft_model, color) in enumerate(delta_arch):
    deltas = []
    for prov in prov_names:
        xgb_row = available[(available['Province'] == prov) & (available['Model'] == xgb_model)]
        tft_row = available[(available['Province'] == prov) & (available['Model'] == tft_model)]
        if len(xgb_row) and len(tft_row):
            deltas.append(tft_row['R2'].values[0] - xgb_row['R2'].values[0])
        else:
            deltas.append(0)
    offset = (j - (n_bars - 1) / 2) * bar_w_d
    axes[1].bar(x + offset, deltas, bar_w_d * 0.9, label=label, color=color, alpha=0.85)

axes[1].set_xticks(x)
axes[1].set_xticklabels(prov_names)
axes[1].set_ylabel(r'$\Delta R^2$ (vs XGBoost)')
axes[1].set_title('(b) Architecture Advantage over XGBoost')
axes[1].legend(fontsize=8)
axes[1].axhline(0, color='black', lw=0.5)

fig.suptitle('Model Comparison: Feature Effect and Architecture Effect',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, 'fig_delta_r2_pair', folder='../figures')
plt.close()
print('Delta R² pair: DONE')
