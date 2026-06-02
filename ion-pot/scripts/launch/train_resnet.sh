#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/server10/pinhao2/ML/Ion_Prediction/pbgnn}"
cd "$PROJECT_ROOT"

# Overrides:
# GPU_ID=0 MAPPER_VERSION=resnet WANDB_DISABLED=true DATASET_GLOB="/path/*.npz" bash train.sh
GPU_ID="${GPU_ID:-0}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"
DATASET_GLOB="${DATASET_GLOB:-/data/server10/pinhao2/ML/Ion_Prediction/alphafold_v6/06_apbs_out_npz/*.npz}"
NNTOOL_OUTPUT_PATH="${NNTOOL_OUTPUT_PATH:-$PROJECT_ROOT/outputs}"
NNTOOL_OUTPUT_PATH_DATE="${NNTOOL_OUTPUT_PATH_DATE:-$(date +%m%d%Y/%H%M%S)}"
MAPPER_VERSION="${MAPPER_VERSION:-resnet}"
PATCH_SIZE="${PATCH_SIZE:-96}"
RESNET_NGF="${RESNET_NGF:-64}"
RESNET_N_BLOCKS="${RESNET_N_BLOCKS:-6}"
RESNET_DROPOUT_P="${RESNET_DROPOUT_P:-0.1}"

RUN_ID="${RUN_ID:-afdb_pot_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="outputs/${RUN_ID}"
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/meta"

# Repro metadata
python -V > "${OUT_ROOT}/meta/python_version.txt"
pip freeze > "${OUT_ROOT}/meta/pip_freeze.txt"
git rev-parse HEAD > "${OUT_ROOT}/meta/git_commit.txt" 2>/dev/null || true
git diff > "${OUT_ROOT}/meta/git_diff.patch" 2>/dev/null || true
{
  echo "GPU_ID=${GPU_ID}"
  echo "WANDB_DISABLED=${WANDB_DISABLED}"
  echo "DATASET_GLOB=${DATASET_GLOB}"
  echo "RUN_ID=${RUN_ID}"
  echo "NNTOOL_OUTPUT_PATH=${NNTOOL_OUTPUT_PATH}"
  echo "NNTOOL_OUTPUT_PATH_DATE=${NNTOOL_OUTPUT_PATH_DATE}"
  echo "MAPPER_VERSION=${MAPPER_VERSION}"
  echo "PATCH_SIZE=${PATCH_SIZE}"
  echo "RESNET_NGF=${RESNET_NGF}"
  echo "RESNET_N_BLOCKS=${RESNET_N_BLOCKS}"
  echo "RESNET_DROPOUT_P=${RESNET_DROPOUT_P}"
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
  --trainer.patch-size "${PATCH_SIZE}" \
  --energy-model.patch-size "${PATCH_SIZE}" \
  --energy-model.reaction-field-mapping-version "${MAPPER_VERSION}" \
  --energy-model.resnet-ngf "${RESNET_NGF}" \
  --energy-model.resnet-n-blocks "${RESNET_N_BLOCKS}" \
  --energy-model.resnet-dropout-p "${RESNET_DROPOUT_P}" \
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
