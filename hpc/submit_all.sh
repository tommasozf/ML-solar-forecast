#!/bin/bash
# Submit all HP search jobs.
# Run from the repo root: bash hpc/submit_all.sh
# Optionally pass overrides: N_TRIALS_PER_WORKER=20 bash hpc/submit_all.sh

set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${REPO_DIR}/hpc/logs"

N_TRIALS_PER_WORKER=${N_TRIALS_PER_WORKER:-10}
MAX_EPOCHS=${MAX_EPOCHS:-15}

PROVINCES=(caceres cadiz zaragoza)
TFT_TYPES=(standard perfect_forecast nwp_forecast)
XGB_TYPES=(standard nwp)

echo "=== Submitting TFT jobs (${#PROVINCES[@]} × ${#TFT_TYPES[@]} studies) ==="
for prov in "${PROVINCES[@]}"; do
    for mtype in "${TFT_TYPES[@]}"; do
        JOB_ID=$(sbatch \
            --export=ALL,PROVINCE="${prov}",MODEL_TYPE="${mtype}",N_TRIALS_PER_WORKER="${N_TRIALS_PER_WORKER}",MAX_EPOCHS="${MAX_EPOCHS}" \
            "${REPO_DIR}/hpc/submit_tft_hp.sbatch" | awk '{print $NF}')
        echo "  tft ${prov} ${mtype} → job ${JOB_ID}"
    done
done

echo ""
echo "=== Submitting XGBoost jobs (${#PROVINCES[@]} × ${#XGB_TYPES[@]} studies) ==="
XGB_TRIALS=${XGB_TRIALS_PER_WORKER:-40}
for prov in "${PROVINCES[@]}"; do
    for mtype in "${XGB_TYPES[@]}"; do
        JOB_ID=$(sbatch \
            --export=ALL,PROVINCE="${prov}",MODEL_TYPE="${mtype}",N_TRIALS_PER_WORKER="${XGB_TRIALS}" \
            "${REPO_DIR}/hpc/submit_xgb_hp.sbatch" | awk '{print $NF}')
        echo "  xgb ${prov} ${mtype} → job ${JOB_ID}"
    done
done

echo ""
echo "Monitor with: squeue -u tza100"
echo "After jobs finish:"
echo "  source ~/solar-venv/bin/activate"
echo "  python hpc/collect_results.py"
