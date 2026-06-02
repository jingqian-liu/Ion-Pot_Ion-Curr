#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/server10/pinhao2/ML/Ion_Prediction/pbgnn}"
cd "${PROJECT_ROOT}"

# ---- overrides (export before running if needed) ----
GPU_ID="${GPU_ID:-0}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"
RUN_ID="${RUN_ID:-afdb_pot_$(date +%Y%m%d_%H%M%S)}"
DATASET_GLOB="${DATASET_GLOB:-/data/server10/pinhao2/ML/Ion_Prediction/alphafold_v6/06_apbs_out_npz/*.npz}"

# nntool envs (fixes KeyError: NNTOOL_OUTPUT_PATH_DATE)
NNTOOL_OUTPUT_PATH="${NNTOOL_OUTPUT_PATH:-${PROJECT_ROOT}/outputs}"
NNTOOL_OUTPUT_PATH_DATE="${NNTOOL_OUTPUT_PATH_DATE:-$(date +%m%d%Y/%H%M%S)}"

OUT_ROOT="${NNTOOL_OUTPUT_PATH}/${RUN_ID}"
mkdir -p "${OUT_ROOT}/"{logs,meta,train}

# sanity check dataset glob
if ! compgen -G "${DATASET_GLOB}" > /dev/null; then
  echo "[ERR] No files matched DATASET_GLOB=${DATASET_GLOB}"
  exit 1
fi

# reproducibility metadata
python -V > "${OUT_ROOT}/meta/python_version.txt"
pip freeze > "${OUT_ROOT}/meta/pip_freeze.txt"
git rev-parse HEAD > "${OUT_ROOT}/meta/git_commit.txt" 2>/dev/null || true
git diff > "${OUT_ROOT}/meta/git_diff.patch" 2>/dev/null || true
nvidia-smi > "${OUT_ROOT}/meta/nvidia_smi.txt" 2>/dev/null || true

{
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "GPU_ID=${GPU_ID}"
  echo "WANDB_DISABLED=${WANDB_DISABLED}"
  echo "RUN_ID=${RUN_ID}"
  echo "DATASET_GLOB=${DATASET_GLOB}"
  echo "NNTOOL_OUTPUT_PATH=${NNTOOL_OUTPUT_PATH}"
  echo "NNTOOL_OUTPUT_PATH_DATE=${NNTOOL_OUTPUT_PATH_DATE}"
} > "${OUT_ROOT}/meta/run_env.txt"

CMD_FILE="${OUT_ROOT}/meta/cmd.sh"
cat > "${CMD_FILE}" <<CMD
python -m scripts.3d.train_3d_energy_distributed \
  unet_psz32_ctx48_protein_complex_with_lset_fully_coverage_rotation_augmented_medium \
  --trainer.output-folder "${OUT_ROOT}/train" \
  --trainer.dataset-path "${DATASET_GLOB}" \
  --trainer.dataset-group-pattern ".npz" \
  --trainer.pkl-filter "" \
  --trainer.no-use-sparse-dataset \
  --trainer.no-use-full-coverage-sparse-dataset \
  --trainer.train-on-potential-map \
  --trainer.potential-map-loss smooth_l1 \
  --trainer.potential-map-smooth-l1-beta 0.001 \
  --trainer.no-potential-map-use-atom-mask \
  --trainer.patch-size 96 \
  --energy-model.patch-size 96 \
  --trainer.train-batch-size 2 \
  --trainer.eval-batch-size 2
CMD
chmod +x "${CMD_FILE}"

echo "Running on GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
WANDB_DISABLED="${WANDB_DISABLED}" \
NNTOOL_OUTPUT_PATH="${NNTOOL_OUTPUT_PATH}" \
NNTOOL_OUTPUT_PATH_DATE="${NNTOOL_OUTPUT_PATH_DATE}" \
bash "${CMD_FILE}" 2>&1 | tee "${OUT_ROOT}/logs/train.log"
