#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# TFT hyperparameter search — parallel Optuna workers sharing SQLite storage
#
# Each array task is one Optuna worker contributing N_TRIALS_PER_WORKER trials
# to the shared study. Total trials = N_WORKERS × N_TRIALS_PER_WORKER.
#
# Submit ONE province+model combination at a time:
#   sbatch --export=ALL,PROVINCE=caceres,MODEL_TYPE=standard hpc/submit_tft_hp.sbatch
#   sbatch --export=ALL,PROVINCE=caceres,MODEL_TYPE=nwp_forecast hpc/submit_tft_hp.sbatch
#   sbatch --export=ALL,PROVINCE=caceres,MODEL_TYPE=perfect_forecast hpc/submit_tft_hp.sbatch
#   (repeat for cadiz, zaragoza)
#
# Or use the helper: bash hpc/submit_all.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=tft_hp
#SBATCH --array=0-9                  # 10 workers → 100 trials total
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # 4 dataloader workers
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --partition=defq-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=hpc/logs/tft_%x_%A_%a.out
#SBATCH --error=hpc/logs/tft_%x_%A_%a.err

# ── Defaults (override via --export) ─────────────────────────────────────────
PROVINCE=${PROVINCE:-caceres}
MODEL_TYPE=${MODEL_TYPE:-standard}
N_TRIALS_PER_WORKER=${N_TRIALS_PER_WORKER:-10}
MAX_EPOCHS=${MAX_EPOCHS:-15}

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load 2025
module load GCCcore/13.2.0
module load Python/3.11.5-GCCcore-13.2.0   # same version used in setup_env_ada.sh

source "${HOME}/solar-venv/bin/activate"
echo "Python: $(python --version) | $(which python)"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="${SLURM_SUBMIT_DIR}"
STUDIES_DIR="${REPO_DIR}/hpc/studies"
mkdir -p "${STUDIES_DIR}"

STORAGE="sqlite:///${STUDIES_DIR}/tft_${PROVINCE}_${MODEL_TYPE}.db"

echo "=============================="
echo "Job:        ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Province:   ${PROVINCE}"
echo "Model type: ${MODEL_TYPE}"
echo "Storage:    ${STORAGE}"
echo "Trials/worker: ${N_TRIALS_PER_WORKER}"
echo "Max epochs:    ${MAX_EPOCHS}"
echo "=============================="

# ── Run ───────────────────────────────────────────────────────────────────────
python "${REPO_DIR}/hpc/train_tft_hp.py" \
    --province "${PROVINCE}" \
    --model-type "${MODEL_TYPE}" \
    --storage "${STORAGE}" \
    --n-trials "${N_TRIALS_PER_WORKER}" \
    --max-epochs "${MAX_EPOCHS}" \
    --base-dir "${REPO_DIR}"

echo "Worker ${SLURM_ARRAY_TASK_ID} finished."
