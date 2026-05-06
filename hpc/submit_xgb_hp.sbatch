#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# XGBoost hyperparameter search — CPU-only, parallel Optuna workers
#
# Submit:
#   sbatch --export=ALL,PROVINCE=caceres,MODEL_TYPE=standard hpc/submit_xgb_hp.sbatch
#   sbatch --export=ALL,PROVINCE=caceres,MODEL_TYPE=nwp hpc/submit_xgb_hp.sbatch
# Or: bash hpc/submit_all.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=xgb_hp
#SBATCH --array=0-4                  # 5 workers → 200 trials total
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=defq
#SBATCH --output=hpc/logs/xgb_%x_%A_%a.out
#SBATCH --error=hpc/logs/xgb_%x_%A_%a.err

PROVINCE=${PROVINCE:-caceres}
MODEL_TYPE=${MODEL_TYPE:-standard}
N_TRIALS_PER_WORKER=${N_TRIALS_PER_WORKER:-40}

module purge
module load 2025
module load GCCcore/13.2.0
module load Python/3.11.5-GCCcore-13.2.0   # same version used in setup_env_ada.sh

source "${HOME}/solar-venv/bin/activate"
echo "Python: $(python --version) | $(which python)"

REPO_DIR="${SLURM_SUBMIT_DIR}"
STUDIES_DIR="${REPO_DIR}/hpc/studies"
mkdir -p "${STUDIES_DIR}"

STORAGE="sqlite:///${STUDIES_DIR}/xgb_${PROVINCE}_${MODEL_TYPE}.db"

echo "=============================="
echo "Job:        ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Province:   ${PROVINCE}"
echo "Model type: ${MODEL_TYPE}"
echo "Storage:    ${STORAGE}"
echo "Trials/worker: ${N_TRIALS_PER_WORKER}"
echo "=============================="

python "${REPO_DIR}/hpc/train_xgb_hp.py" \
    --province "${PROVINCE}" \
    --model-type "${MODEL_TYPE}" \
    --storage "${STORAGE}" \
    --n-trials "${N_TRIALS_PER_WORKER}" \
    --base-dir "${REPO_DIR}"

echo "Worker ${SLURM_ARRAY_TASK_ID} finished."
