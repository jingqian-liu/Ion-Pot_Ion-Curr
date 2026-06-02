#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "${RUN_ROOT}" ]]; then
  echo "[ERR] Please set RUN_ROOT to one APBS concentration sweep folder."
  echo "      Example: RUN_ROOT=/data/.../06_apbs_out/conc_sweep/apbs_conc_20260223_131606"
  exit 1
fi

POT_GLOB="${POT_GLOB:-${RUN_ROOT}/pot/*_pot.dx}"
CHARGE_DIR="${CHARGE_DIR:-${RUN_ROOT}/charge}"
VDW_DIR="${VDW_DIR:-${RUN_ROOT}/vdw}"
OUT_NPZ_ROOT="${OUT_NPZ_ROOT:-${RUN_ROOT}/npz}"
OVERWRITE="${OVERWRITE:-false}"
STRICT_GRID_CHECK="${STRICT_GRID_CHECK:-false}"

mkdir -p "${OUT_NPZ_ROOT}"

cmd=(
  python scripts/apbs/convert_apbs_dx_to_npz.py
  --pot-glob "${POT_GLOB}"
  --charge-dir "${CHARGE_DIR}"
  --vdw-dir "${VDW_DIR}"
  --out-root "${OUT_NPZ_ROOT}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  cmd+=(--overwrite)
fi
if [[ "${STRICT_GRID_CHECK}" == "true" ]]; then
  cmd+=(--strict-grid-check)
fi

echo "Running DX->NPZ conversion"
echo "RUN_ROOT: ${RUN_ROOT}"
echo "OUT_NPZ_ROOT: ${OUT_NPZ_ROOT}"
echo "POT_GLOB: ${POT_GLOB}"

"${cmd[@]}"

echo "Done."
echo "NPZ files: ${OUT_NPZ_ROOT}/*.npz"
