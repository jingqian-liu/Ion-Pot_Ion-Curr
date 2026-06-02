#!/usr/bin/env bash
# ============================================================================
# Evaluate all 9 conditioning-ablation models (3 backbones x 3 conditionings)
# on the test split and dump per-concentration metrics (including c = 0.00 M).
#
# Output layout:
#   ${EVAL_ROOT}/${BACKBONE}_${COND}/
#     metrics_summary.json           <- per_concentration["0.0000"] has the row
#     per_concentration_metrics.csv
#     sample_metrics.csv
#
# After this finishes, summarize the c=0.00 rows into LaTeX with:
#   python scripts/summarize_ablation_c0_table.py --eval-root ${EVAL_ROOT}
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/server10/pinhao2/ML/Ion_Prediction/pbgnn}"
cd "${PROJECT_ROOT}"

# ---- Common eval settings (match your original command) ---------------------
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unet_psz32_ctx48_protein_complex_with_lset_fully_coverage_rotation_augmented_medium}"
DEVICE="${DEVICE:-cuda:3}"
PATCH_SIZE="${PATCH_SIZE:-193}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
SPLIT="${SPLIT:-test}"
EVAL_ROOT="${EVAL_ROOT:-/data/server10/pinhao2/ML/Ion_Prediction/ablation_eval_c0}"
mkdir -p "${EVAL_ROOT}"

# ---- Fill in per-model checkpoints & data splits ---------------------------
# Each run_id below must point at the ablation training directory that produced
# that specific backbone+conditioning checkpoint. Leave RUN_DIR empty to skip.
#
# Each RUN_DIR is expected to contain:
#   train/models/model_final/model.safetensors
#   train/data_split.json
#
# Example (the one you already ran — FNO, ionic_conditioning=? — for reference):
#   /data/server10/pinhao2/ML/Ion_Prediction/pbgnn/outputs/afdb_pot_ablation_20260405_153252/runs/steps24k_batch4
# ----------------------------------------------------------------------------

declare -A RUN_DIR
RUN_DIR[unet_woion]=""
RUN_DIR[unet_film]=""
RUN_DIR[unet_add]=""
RUN_DIR[resnet_woion]=""
RUN_DIR[resnet_film]=""
RUN_DIR[resnet_add]=""
RUN_DIR[fno_woion]=""
RUN_DIR[fno_film]=""
RUN_DIR[fno_add]=""

# ---- Backbone / conditioning decode ----------------------------------------
# key format: <backbone>_<cond>
#   backbone: unet | resnet | fno -> --mapper-version u-net | resnet | fno
#   cond:     woion | film | add
#             woion -> --no-use-ionic-conc
#             film  -> --use-ionic-conc --ionic-conditioning film
#             add   -> --use-ionic-conc --ionic-conditioning add
# ----------------------------------------------------------------------------

mapper_for() {
  case "$1" in
    unet)   echo "u-net" ;;
    resnet) echo "resnet" ;;
    fno)    echo "fno" ;;
    *) echo "[ERR] unknown backbone: $1" >&2; exit 1 ;;
  esac
}

cond_flags_for() {
  case "$1" in
    woion) echo "--no-use-ionic-conc" ;;
    film)  echo "--use-ionic-conc --ionic-conditioning film" ;;
    add)   echo "--use-ionic-conc --ionic-conditioning add" ;;
    *) echo "[ERR] unknown conditioning: $1" >&2; exit 1 ;;
  esac
}

run_one_eval() {
  local key="$1"            # e.g. fno_add
  local backbone="${key%%_*}"
  local cond="${key#*_}"
  local run_dir="${RUN_DIR[$key]}"

  if [[ -z "${run_dir}" ]]; then
    echo "[SKIP] ${key}: RUN_DIR not set"
    return 0
  fi

  local ckpt="${run_dir}/train/models/model_final/model.safetensors"
  local split_json="${run_dir}/train/data_split.json"
  local out_dir="${EVAL_ROOT}/${key}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERR] checkpoint not found for ${key}: ${ckpt}"
    return 1
  fi
  if [[ ! -f "${split_json}" ]]; then
    echo "[ERR] data_split.json not found for ${key}: ${split_json}"
    return 1
  fi
  mkdir -p "${out_dir}"

  local mapper
  mapper="$(mapper_for "${backbone}")"
  # shellcheck disable=SC2206
  local cond_flags=( $(cond_flags_for "${cond}") )

  echo "=============================================="
  echo "Eval: ${key}   (backbone=${mapper}, cond=${cond})"
  echo "  ckpt:   ${ckpt}"
  echo "  split:  ${split_json}"
  echo "  out:    ${out_dir}"
  echo "=============================================="

  MPLBACKEND=Agg PYTHONPATH="${PROJECT_ROOT}" \
  python "${PROJECT_ROOT}/scripts/visualize_test_potential_maps_per_conc.py" \
      --experiment-name "${EXPERIMENT_NAME}" \
      --model-ckpt-path "${ckpt}" \
      --data-split-json-path "${split_json}" \
      --split "${SPLIT}" \
      --output-dir "${out_dir}" \
      --max-samples "${MAX_SAMPLES}" \
      --device "${DEVICE}" \
      --patch-size "${PATCH_SIZE}" \
      --mapper-version "${mapper}" \
      "${cond_flags[@]}" \
      --no-figures 2>&1 | tee "${out_dir}/eval.log"
}

# ---- Run all 9 (order groups by backbone so GPU memory behaves) ------------
KEYS=(
  unet_woion unet_film unet_add
  resnet_woion resnet_film resnet_add
  fno_woion fno_film fno_add
)

# Allow selecting a subset via env: EVAL_KEYS="fno_woion,fno_film,fno_add"
if [[ -n "${EVAL_KEYS:-}" ]]; then
  IFS=',' read -r -a KEYS <<< "${EVAL_KEYS}"
fi

for key in "${KEYS[@]}"; do
  run_one_eval "${key}"
done

echo
echo "All evals complete. Root: ${EVAL_ROOT}"
echo "Summarize LaTeX rows:"
echo "  python ${PROJECT_ROOT}/scripts/summarize_ablation_c0_table.py --eval-root ${EVAL_ROOT}"
