#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/server10/pinhao2/ML/Ion_Prediction/pbgnn}"
cd "${PROJECT_ROOT}"

# ---- run controls ----
GPU_ID="${GPU_ID:-0}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"
RUN_ID="${RUN_ID:-afdb_pot_$(date +%Y%m%d_%H%M%S)}"

# dataset mode: both | salt | water
DATASET_MODE="${DATASET_MODE:-both}"

# explicit dataset globs (can override)
SALT_DATASET_GLOB="${SALT_DATASET_GLOB:-/data/server10/pinhao2/ML/Ion_Prediction/alphafold_v5_with_ionic_conc/06_apbs_out_npz/*.npz}"
WATER_DATASET_GLOB="${WATER_DATASET_GLOB:-/data/server10/pinhao2/ML/Ion_Prediction/alphafold_v5/06_apbs_out_npz/*.npz}"

case "${DATASET_MODE}" in
  both)
    # Python glob has no brace expansion; this wildcard covers both folders.
    DEFAULT_DATASET_GLOB="/data/server10/pinhao2/ML/Ion_Prediction/alphafold_v5*/06_apbs_out_npz/*.npz"
    ;;
  salt)
    DEFAULT_DATASET_GLOB="${SALT_DATASET_GLOB}"
    ;;
  water)
    DEFAULT_DATASET_GLOB="${WATER_DATASET_GLOB}"
    ;;
  *)
    echo "[ERR] DATASET_MODE must be one of: both | salt | water (got: ${DATASET_MODE})"
    exit 1
    ;;
esac

# manual override still works
DATASET_GLOB="${DATASET_GLOB:-${DEFAULT_DATASET_GLOB}}"

# model/training knobs
EXPERIMENT="${EXPERIMENT:-unet_psz32_ctx48_protein_complex_with_lset_fully_coverage_rotation_augmented_medium}"
PATCH_SIZE="${PATCH_SIZE:-96}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
USE_IONIC_CONC="${USE_IONIC_CONC:-true}"

# nntool envs
NNTOOL_OUTPUT_PATH="${NNTOOL_OUTPUT_PATH:-${PROJECT_ROOT}/outputs}"
NNTOOL_OUTPUT_PATH_DATE="${NNTOOL_OUTPUT_PATH_DATE:-$(date +%m%d%Y/%H%M%S)}"

OUT_ROOT="${NNTOOL_OUTPUT_PATH}/${RUN_ID}"
mkdir -p "${OUT_ROOT}/"{logs,meta,train}

# sanity check dataset glob
if ! compgen -G "${DATASET_GLOB}" > /dev/null; then
  echo "[ERR] No files matched DATASET_GLOB=${DATASET_GLOB}"
  exit 1
fi
DATASET_COUNT="$(compgen -G "${DATASET_GLOB}" | wc -l | tr -d ' ')"

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
  echo "DATASET_MODE=${DATASET_MODE}"
  echo "DATASET_GLOB=${DATASET_GLOB}"
  echo "DATASET_COUNT=${DATASET_COUNT}"
  echo "USE_IONIC_CONC=${USE_IONIC_CONC}"
  echo "PATCH_SIZE=${PATCH_SIZE}"
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
  echo "NNTOOL_OUTPUT_PATH=${NNTOOL_OUTPUT_PATH}"
  echo "NNTOOL_OUTPUT_PATH_DATE=${NNTOOL_OUTPUT_PATH_DATE}"
} > "${OUT_ROOT}/meta/run_env.txt"

CMD=(
  python -m scripts.3d.train_3d_energy_distributed
  "${EXPERIMENT}"
  --trainer.output-folder "${OUT_ROOT}/train"
  --trainer.dataset-path "${DATASET_GLOB}"
  --trainer.dataset-group-pattern ".npz"
  --trainer.pkl-filter ""
  --trainer.no-use-sparse-dataset
  --trainer.no-use-full-coverage-sparse-dataset
  --trainer.train-on-potential-map
  --trainer.potential-map-loss smooth_l1
  --trainer.potential-map-smooth-l1-beta 0.001
  --trainer.no-potential-map-use-atom-mask
  --trainer.patch-size "${PATCH_SIZE}"
  --energy-model.patch-size "${PATCH_SIZE}"
  --trainer.train-batch-size "${TRAIN_BATCH_SIZE}"
  --trainer.eval-batch-size "${EVAL_BATCH_SIZE}"
)

if [[ "${USE_IONIC_CONC}" == "true" ]]; then
  CMD+=(--energy-model.use-ionic-conc)
fi

CMD_FILE="${OUT_ROOT}/meta/cmd.sh"
printf '%q ' "${CMD[@]}" > "${CMD_FILE}"
echo >> "${CMD_FILE}"
chmod +x "${CMD_FILE}"

echo "Running on GPU ${GPU_ID}"
echo "Dataset mode: ${DATASET_MODE}"
echo "Matched files: ${DATASET_COUNT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
WANDB_DISABLED="${WANDB_DISABLED}" \
NNTOOL_OUTPUT_PATH="${NNTOOL_OUTPUT_PATH}" \
NNTOOL_OUTPUT_PATH_DATE="${NNTOOL_OUTPUT_PATH_DATE}" \
bash "${CMD_FILE}" 2>&1 | tee "${OUT_ROOT}/logs/train.log"
