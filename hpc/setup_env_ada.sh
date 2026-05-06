#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build a Python virtualenv on ADA with all packages needed for the HP search.
# Run this ONCE from an interactive node (NOT a login node), then all batch
# jobs will just activate the venv.
#
# Request an interactive GPU session first:
#   srun --partition=defq-gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G \
#        --time=01:00:00 --pty bash
#
# Then run:
#   bash hpc/setup_env_ada.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

VENV_DIR="${HOME}/solar-venv"

module purge
module load 2025
module load GCCcore/13.2.0
module load Python/3.11.5-GCCcore-13.2.0   # check: module avail Python
module load CUDA/12.4.0                      # check: module avail CUDA

# System Python is 3.9 — must use the module version
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $(python3 --version) (from $(which python3))"
if [[ "${PY_VERSION}" < "3.10" ]]; then
    echo "ERROR: Python >= 3.10 required (got ${PY_VERSION})."
    echo "Run: module avail Python   — then edit the module load line above."
    exit 1
fi

echo "Creating venv at ${VENV_DIR} ..."

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel

echo "Installing PyTorch (CUDA 12.4) ..."
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

echo "Installing pytorch-forecasting + lightning ..."
pip install pytorch-forecasting lightning

echo "Installing optuna, xgboost, data stack ..."
pip install \
    optuna \
    xgboost \
    pandas \
    numpy \
    scikit-learn \
    scipy

echo ""
echo "Verifying CUDA ..."
python3 -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Done. Venv at ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo ""
echo "If Python/3.11.5-GCCcore-13.2.0 is not available, check with:"
echo "  module avail Python"
echo "and edit the module load line in this script and in the sbatch files."
