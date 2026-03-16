# Stacking Ensemble — TFT + XGBoost Combiner

## Context
The project compares XGBoost vs TFT for solar generation forecasting across 3 Spanish regions (Caceres, Cadiz, Zaragoza). Each region already has trained TFT (standard/NWP/perfect-forecast) and XGBoost (standard/NWP) models. The goal is to create a 3rd model per region: a **stacking ensemble** that feeds TFT NWP + XGBoost NWP predictions into a Ridge Regression meta-learner, letting it learn which model to trust in which conditions.

## Step 0 — Branch Setup
- Switch to `origin/xgboost` branch and create a new branch off it (e.g., `stacking-ensemble`)
- This ensures all XGBoost notebooks, HP result JSONs, and TFT checkpoints are available

## Step 1 — Create `8_stacking_ensemble.ipynb` for Caceres (template)

### Notebook Structure

**Section 0: Imports & Config**
- Same imports as evaluation notebooks + `sklearn.linear_model.Ridge`
- Load `preprocessing_params.json` and `nwp_preprocessing_params.json` for denorm params
- Define `denormalize()`, `compute_metrics()` (reuse from existing notebooks)
- Paths to: `tft_nwp_best.ckpt`, `xgboost_nwp_hp_study_results.json`, processed CSVs

**Section 1: Generate TFT NWP Predictions (val + test)**
- Reconstruct `TimeSeriesDataSet` from `nwp_train_processed.csv` (needed for `from_dataset`)
- Build val/test datasets with 168h encoder context (same pattern as `5c_nwp_evaluation.ipynb`)
- Load `tft_nwp_best.ckpt`, run `model.predict()` → `pred_z` shape `(n_windows, 24)`
- **Aggregate overlapping windows**: for each window `i` and horizon `h`, timestep = `i + h`. Average all predictions per timestep. This gives one TFT prediction per timestep, aligned with XGBoost.
- Denormalize and clip to [0, inf)
- Key files: [5c_nwp_evaluation.ipynb](caceres_analysis/5c_nwp_evaluation.ipynb) for the pattern, [nwp_preprocessing_params.json](caceres_analysis/data/nwp_preprocessing_params.json) for NWP feature lists

**Section 2: Generate XGBoost NWP Predictions (val + test)**
- Load best HPs from `models/xgboost_nwp_hp_study_results.json`
- Retrain XGBoost with those HPs on train data (notebook 7 doesn't save the model object, only HPs)
- Predict val and test — already 1:1 with timesteps, no alignment needed
- Denormalize and clip
- Key file: [7_xgboost_baseline.ipynb](caceres_analysis/7_xgboost_baseline.ipynb) for pattern

**Section 3: Assemble Aligned Prediction DataFrame**
- Create `val_preds` and `test_preds` DataFrames indexed by `datetime_utc`:
  - `actual_mwh`, `tft_nwp`, `xgb_nwp` (predictions in MWh)
  - Context features: `hour_sin`, `hour_cos`, `solar_zenith`, `clearsky_ghi`
- Print correlation matrix as sanity check

**Section 4: Train Meta-Learner (Ridge Regression)**
- Two variants compared:
  1. **Predictions only**: features = `[tft_nwp, xgb_nwp]`
  2. **Predictions + context**: features = `[tft_nwp, xgb_nwp, hour_sin, hour_cos, solar_zenith, clearsky_ghi]`
- Train `Ridge(alpha=1.0)` on val predictions, evaluate on test
- The val set (4344 rows) is out-of-sample for both base models → proper stacking
- Print comparison table of both variants
- Inspect Ridge coefficients to show learned model weighting

**Section 5: Full Model Comparison Table**
- Load metrics from existing JSONs: TFT standard, TFT NWP, XGBoost standard, XGBoost NWP
- Add persistence and climatological baselines
- Add stacking ensemble (best variant)
- Print side-by-side: RMSE, MAE, R2

**Section 6: Visualizations**
- Bar chart: R2 across all models (including ensemble)
- Time series: selected test weeks — actual vs TFT NWP vs XGBoost NWP vs ensemble
- Scatter: ensemble predicted vs actual
- Residual analysis: ensemble vs best individual model, by hour-of-day
- Meta-learner weight bar chart

**Section 7: Save Results**
- Save `models/stacking_results.json` with: best config, meta weights, val/test metrics
- Save `models/stacking_predictions_test.csv` with: datetime_utc, actual_mwh, ensemble_pred_mwh

## Step 2 — Replicate to Cadiz and Zaragoza
- Copy notebook, update only:
  - Directory paths (`cadiz_analysis/`, `zaragoza_analysis/`)
  - Region name in titles/print statements
  - Verify NWP checkpoint filenames match (check each region's `models/` directory)

## Step 3 — Update Cross-Province Comparison
- Update [cross_province_comparison.ipynb](cross_province_comparison.ipynb) to load `stacking_results.json` from each region
- Add "Stacking Ensemble" row to the comparison table

## Verification
1. Run `8_stacking_ensemble.ipynb` for Caceres end-to-end
2. Confirm TFT and XGBoost predictions align (same number of val/test timesteps)
3. Verify ensemble R2 > best individual model R2 (expected but not guaranteed)
4. Check Ridge coefficients are reasonable (both positive, no extreme values)
5. Replicate for Cadiz and Zaragoza, verify consistent improvements
6. Run updated cross-province comparison

## Key Dependencies
- TFT NWP checkpoint: `{region}_analysis/models/nwp_forecast/tft_nwp_best.ckpt`
- XGBoost NWP HPs: `{region}_analysis/models/xgboost_nwp_hp_study_results.json`
- NWP processed data: `{region}_analysis/data/nwp_{train,val,test}_processed.csv`
- Preprocessing params: `{region}_analysis/data/nwp_preprocessing_params.json`
