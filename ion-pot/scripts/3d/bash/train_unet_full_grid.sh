#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# UNet training on the FULL 193³ grid (96 Å at 0.5 Å spacing)
#
# Runs three variants sequentially:
#   1) UNet + Add   (ionic conditioning, additive)
#   2) UNet + FiLM  (ionic conditioning, feature-wise linear modulation)
#   3) UNet w/o ion (no ionic conditioning)
#
# Memory note:
#   193³ ≈ 7.2M voxels vs 96³ ≈ 885K → ~8× more memory per sample.
#   We use batch_size=1 with gradient_accumulation=4 → effective batch = 4.
#
# Usage:
#   bash train_unet_full_grid.sh                         # all 3 variants
#   VARIANTS=add        bash train_unet_full_grid.sh     # only Add
#   VARIANTS=film       bash train_unet_full_grid.sh     # only FiLM
#   VARIANTS=no_ion     bash train_unet_full_grid.sh     # only w/o ion
#   VARIANTS=add,film   bash train_unet_full_grid.sh     # Add + FiLM
#   GPU_ID=2            bash train_unet_full_grid.sh     # specific GPU
# ============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-/data/server10/pinhao2/ML/Ion_Prediction/pbgnn}"
cd "${PROJECT_ROOT}"

# --------------- Hardware & logging ---------------
GPU_ID="${GPU_ID:-0}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# --------------- Dataset ---------------
DATASET_GLOB="${DATASET_GLOB:-/data/server10/pinhao2/ML/Ion_Prediction/alphafold_v7/06_apbs_out_npz/salt*/*.npz}"
DATASET_GROUP_PATTERN="${DATASET_GROUP_PATTERN:-.npz}"
DATA_SPLIT_JSON_PATH="${DATA_SPLIT_JSON_PATH:-}"
AUTO_REUSE_SPLIT="${AUTO_REUSE_SPLIT:-true}"

# --------------- Model ---------------
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unet_psz32_ctx48_protein_complex_with_lset_fully_coverage_rotation_augmented_medium}"
MAPPER_VERSION="${MAPPER_VERSION:-u-net}"

# --------------- Full-grid training ---------------
PATCH_SIZE="${PATCH_SIZE:-193}"          # full 193³ grid (was 96)
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"   # reduced from 2 (8× larger volume)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"              # effective batch = 1 × 4 = 4

# --------------- Training hyperparams ---------------
TRAIN_LR="${TRAIN_LR:-5e-4}"
FINAL_TRAIN_LR="${FINAL_TRAIN_LR:-1e-6}"
TRAIN_NUM_STEPS="${TRAIN_NUM_STEPS:-24000}"
SMOOTH_L1_BETA="${SMOOTH_L1_BETA:-0.001}"
SAVE_AND_EVAL_EVERY="${SAVE_AND_EVAL_EVERY:-400}"

# --------------- Variant selection ---------------
VARIANTS="${VARIANTS:-add,film,no_ion}"    # comma-separated: add, film, no_ion

# --------------- Output ---------------
NNTOOL_OUTPUT_PATH="${NNTOOL_OUTPUT_PATH:-${PROJECT_ROOT}/outputs}"
NNTOOL_OUTPUT_PATH_DATE="${NNTOOL_OUTPUT_PATH_DATE:-$(date +%m%d%Y/%H%M%S)}"
RUN_SET_ID="${RUN_SET_ID:-unet_full193_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${NNTOOL_OUTPUT_PATH}/${RUN_SET_ID}"
mkdir -p "${OUT_ROOT}/"{logs,meta,runs}

# --------------- Sanity checks ---------------
if ! compgen -G "${DATASET_GLOB}" > /dev/null; then
  echo "[ERR] No files matched DATASET_GLOB=${DATASET_GLOB}"
  exit 1
fi

if [[ -n "${DATA_SPLIT_JSON_PATH}" ]] && [[ ! -f "${DATA_SPLIT_JSON_PATH}" ]]; then
  echo "[ERR] DATA_SPLIT_JSON_PATH does not exist: ${DATA_SPLIT_JSON_PATH}"
  exit 1
fi

# --------------- Repro metadata ---------------
python -V > "${OUT_ROOT}/meta/python_version.txt"
pip freeze > "${OUT_ROOT}/meta/pip_freeze.txt"
git rev-parse HEAD > "${OUT_ROOT}/meta/git_commit.txt" 2>/dev/null || true
git diff > "${OUT_ROOT}/meta/git_diff.patch" 2>/dev/null || true
nvidia-smi > "${OUT_ROOT}/meta/nvidia_smi.txt" 2>/dev/null || true

{
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "GPU_ID=${GPU_ID}"
  echo "DATASET_GLOB=${DATASET_GLOB}"
  echo "PATCH_SIZE=${PATCH_SIZE}"
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
  echo "GRAD_ACCUM=${GRAD_ACCUM}"
  echo "TRAIN_NUM_STEPS=${TRAIN_NUM_STEPS}"
  echo "TRAIN_LR=${TRAIN_LR}"
  echo "FINAL_TRAIN_LR=${FINAL_TRAIN_LR}"
  echo "SMOOTH_L1_BETA=${SMOOTH_L1_BETA}"
  echo "MAPPER_VERSION=${MAPPER_VERSION}"
  echo "VARIANTS=${VARIANTS}"
  echo "RUN_SET_ID=${RUN_SET_ID}"
} > "${OUT_ROOT}/meta/run_env.txt"

# --------------- Shared split tracking ---------------
ACTIVE_SPLIT_JSON_PATH="${DATA_SPLIT_JSON_PATH}"

contains_variant() {
  local key="$1"
  if [[ "${VARIANTS}" == "all" ]]; then return 0; fi
  [[ ",${VARIANTS}," == *",${key},"* ]]
}

run_variant() {
  local run_name="$1"
  local use_ionic="$2"          # true | false
  local ionic_conditioning="$3" # add | film | (ignored if use_ionic=false)

  if ! contains_variant "${run_name}"; then
    return 0
  fi

  local run_dir="${OUT_ROOT}/runs/${run_name}"
  local run_log="${OUT_ROOT}/logs/${run_name}.log"
  local cmd_file="${run_dir}/cmd.sh"
  mkdir -p "${run_dir}"

  local -a cmd=(
    python -m scripts.3d.train_3d_energy_distributed
    "${EXPERIMENT_NAME}"
    --trainer.output-folder "${run_dir}/train"
    --trainer.dataset-path "${DATASET_GLOB}"
    --trainer.dataset-group-pattern "${DATASET_GROUP_PATTERN}"
    --trainer.pkl-filter ""
    --trainer.no-use-sparse-dataset
    --trainer.no-use-full-coverage-sparse-dataset
    --trainer.train-on-potential-map
    --trainer.potential-map-loss smooth_l1
    --trainer.potential-map-smooth-l1-beta "${SMOOTH_L1_BETA}"
    --trainer.no-potential-map-use-atom-mask
    --trainer.patch-size "${PATCH_SIZE}"
    --trainer.train-batch-size "${TRAIN_BATCH_SIZE}"
    --trainer.eval-batch-size "${EVAL_BATCH_SIZE}"
    --trainer.gradient-accumulation-steps "${GRAD_ACCUM}"
    --trainer.num-workers "${NUM_WORKERS}"
    --trainer.save-and-eval-every "${SAVE_AND_EVAL_EVERY}"
    --trainer.train-lr "${TRAIN_LR}"
    --trainer.final-train-lr "${FINAL_TRAIN_LR}"
    --trainer.train-num-steps "${TRAIN_NUM_STEPS}"
    --energy-model.patch-size "${PATCH_SIZE}"
    --energy-model.reaction-field-mapping-version "${MAPPER_VERSION}"
  )

  # Data split reuse
  if [[ -n "${ACTIVE_SPLIT_JSON_PATH}" ]]; then
    cmd+=(--trainer.data-split-json-path "${ACTIVE_SPLIT_JSON_PATH}")
  fi

  # Ionic conditioning
  if [[ "${use_ionic}" == "true" ]]; then
    cmd+=(--energy-model.use-ionic-conc)
    cmd+=(--energy-model.ionic-conditioning "${ionic_conditioning}")
  else
    cmd+=(--energy-model.no-use-ionic-conc)
  fi

  # Save command for reproducibility
  {
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "${cmd_file}"
  chmod +x "${cmd_file}"

  echo "=============================================="
  echo "Variant : ${run_name}"
  echo "Patch   : ${PATCH_SIZE}³  (full grid)"
  echo "Batch   : ${TRAIN_BATCH_SIZE} × ${GRAD_ACCUM} accum = effective $(( TRAIN_BATCH_SIZE * GRAD_ACCUM ))"
  echo "Steps   : ${TRAIN_NUM_STEPS}"
  echo "Ionic   : ${use_ionic} (${ionic_conditioning:-n/a})"
  echo "Log     : ${run_log}"
  echo "Cmd     : ${cmd_file}"
  echo "=============================================="

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  WANDB_DISABLED="${WANDB_DISABLED}" \
  NNTOOL_OUTPUT_PATH="${NNTOOL_OUTPUT_PATH}" \
  NNTOOL_OUTPUT_PATH_DATE="${NNTOOL_OUTPUT_PATH_DATE}" \
    "${cmd[@]}" 2>&1 | tee "${run_log}"

  # Reuse the first generated split for remaining runs
  if [[ -z "${ACTIVE_SPLIT_JSON_PATH}" ]] && [[ "${AUTO_REUSE_SPLIT}" == "true" ]]; then
    local generated_split="${run_dir}/train/data_split.json"
    if [[ -f "${generated_split}" ]]; then
      ACTIVE_SPLIT_JSON_PATH="${generated_split}"
      echo "[INFO] Reusing generated split: ${ACTIVE_SPLIT_JSON_PATH}"
    fi
  fi
}

# ============================================================================
# Run the three UNet variants
# ============================================================================
run_variant "add"    true  add
run_variant "film"   true  film
run_variant "no_ion" false ""

echo ""
echo "All UNet full-grid runs complete: ${OUT_ROOT}"
echo ""
echo "Evaluate with:"
echo "  python -m scripts.3d.visualize_test_potential_maps \\"
echo "    --mapper-version u-net \\"
echo "    --checkpoint-path ${OUT_ROOT}/runs/<variant>/train/best_model.pt \\"
echo "    --use-ionic-conc --ionic-conditioning <add|film>"