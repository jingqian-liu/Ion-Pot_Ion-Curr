#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

APBS_BIN="${APBS_BIN:-apbs}"
PQR_FILE="${PQR_FILE:-04_pqr/A0A0B4J2F0_pH7.4_CHARMM.pqr}"
MOL_NAME="${MOL_NAME:-$(basename "${PQR_FILE}" .pqr)}"

# Space-separated list in mol/L, e.g. "0.00 0.05 0.10 0.15 0.30"
CONCENTRATIONS="${CONCENTRATIONS:-0.00 0.05 0.10 0.15 0.30}"

RUN_ID="${RUN_ID:-apbs_conc_$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-06_apbs_out/conc_sweep}"
RUN_ROOT="${OUT_BASE}/${RUN_ID}"
OVERWRITE="${OVERWRITE:-false}"
WRITE_EXTRA_MAPS="${WRITE_EXTRA_MAPS:-true}"
CONVERT_TO_NPZ="${CONVERT_TO_NPZ:-false}"
NPZ_OUT_ROOT="${NPZ_OUT_ROOT:-${RUN_ROOT}/npz}"

# APBS solver settings
NLEV="${NLEV:-4}"
DIME="${DIME:-193 193 193}"
FGLEN="${FGLEN:-96.000 96.000 96.000}"
CGLEN="${CGLEN:-96.000 96.000 96.000}"
PDIE="${PDIE:-2.000}"
SDIE="${SDIE:-78.540}"
TEMP="${TEMP:-298.0}"
BCFL="${BCFL:-mdh}"
CHGM="${CHGM:-spl0}"
SRFM="${SRFM:-smol}"
SRAD="${SRAD:-1.4}"
SDENS="${SDENS:-10.0}"

# Ion radii in Angstrom
ION_POS_RADIUS="${ION_POS_RADIUS:-1.76375}"
ION_NEG_RADIUS="${ION_NEG_RADIUS:-2.27000}"

if ! command -v "${APBS_BIN}" >/dev/null 2>&1; then
  echo "[ERR] APBS binary not found: ${APBS_BIN}"
  exit 1
fi

if [[ ! -f "${PQR_FILE}" ]]; then
  echo "[ERR] PQR file not found: ${PQR_FILE}"
  exit 1
fi

mkdir -p "${RUN_ROOT}/"{inputs,logs,pot,charge,vdw,ndens,meta}

{
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "APBS_BIN=${APBS_BIN}"
  echo "PQR_FILE=${PQR_FILE}"
  echo "MOL_NAME=${MOL_NAME}"
  echo "CONCENTRATIONS=${CONCENTRATIONS}"
  echo "RUN_ID=${RUN_ID}"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "OVERWRITE=${OVERWRITE}"
  echo "WRITE_EXTRA_MAPS=${WRITE_EXTRA_MAPS}"
  echo "CONVERT_TO_NPZ=${CONVERT_TO_NPZ}"
  echo "NPZ_OUT_ROOT=${NPZ_OUT_ROOT}"
  echo "NLEV=${NLEV}"
  echo "DIME=${DIME}"
  echo "FGLEN=${FGLEN}"
  echo "CGLEN=${CGLEN}"
  echo "PDIE=${PDIE}"
  echo "SDIE=${SDIE}"
  echo "TEMP=${TEMP}"
  echo "ION_POS_RADIUS=${ION_POS_RADIUS}"
  echo "ION_NEG_RADIUS=${ION_NEG_RADIUS}"
} > "${RUN_ROOT}/meta/run_env.txt"

python -V > "${RUN_ROOT}/meta/python_version.txt" 2>/dev/null || true

conc_to_tag() {
  local c="$1"
  echo "${c}" | tr '.' 'p'
}

for conc in ${CONCENTRATIONS}; do
  tag="$(conc_to_tag "${conc}")"
  in_file="${RUN_ROOT}/inputs/${MOL_NAME}_c${tag}.in"
  log_file="${RUN_ROOT}/logs/${MOL_NAME}_c${tag}.log"
  pot_out="${RUN_ROOT}/pot/${MOL_NAME}_c${tag}_pot"
  charge_out="${RUN_ROOT}/charge/${MOL_NAME}_c${tag}_charge"
  vdw_out="${RUN_ROOT}/vdw/${MOL_NAME}_c${tag}_vdw"
  ndens_out="${RUN_ROOT}/ndens/${MOL_NAME}_c${tag}_ndens"

  if [[ "${OVERWRITE}" != "true" && -f "${pot_out}.dx" ]]; then
    echo "[SKIP] Found existing output: ${pot_out}.dx"
    continue
  fi

  cat > "${in_file}" <<EOF
read
    mol pqr ${PQR_FILE}
end

elec
    mg-auto
    nlev ${NLEV}
    dime ${DIME}
    fglen ${FGLEN}
    fgcent mol 1
    cglen ${CGLEN}
    cgcent mol 1
    mol 1
    npbe
    bcfl ${BCFL}
    ion charge 1 radius ${ION_POS_RADIUS} conc ${conc}
    ion charge -1 radius ${ION_NEG_RADIUS} conc ${conc}
    pdie ${PDIE}
    sdie ${SDIE}
    chgm ${CHGM}
    srfm ${SRFM}
    srad ${SRAD}
    sdens ${SDENS}
    temp ${TEMP}
    calcenergy no
    calcforce no
    write pot dx ${pot_out}
EOF

  if [[ "${WRITE_EXTRA_MAPS}" == "true" ]]; then
    cat >> "${in_file}" <<EOF
    write charge dx ${charge_out}
    write vdw dx ${vdw_out}
    write ndens dx ${ndens_out}
EOF
  fi

  cat >> "${in_file}" <<EOF
end
EOF

  echo "================================================="
  echo "Running APBS for concentration ${conc} M"
  echo "Input: ${in_file}"
  echo "Log:   ${log_file}"
  echo "================================================="

  "${APBS_BIN}" "${in_file}" 2>&1 | tee "${log_file}"
done

echo "Done. Results in: ${RUN_ROOT}"
echo "Potential maps: ${RUN_ROOT}/pot/*.dx"

if [[ "${CONVERT_TO_NPZ}" == "true" ]]; then
  if [[ "${WRITE_EXTRA_MAPS}" != "true" ]]; then
    echo "[ERR] CONVERT_TO_NPZ=true requires WRITE_EXTRA_MAPS=true (need charge/vdw maps)."
    exit 1
  fi

  echo "Converting DX triplets to NPZ..."
  python scripts/apbs/convert_apbs_dx_to_npz.py \
    --pot-glob "${RUN_ROOT}/pot/*_pot.dx" \
    --charge-dir "${RUN_ROOT}/charge" \
    --vdw-dir "${RUN_ROOT}/vdw" \
    --out-root "${NPZ_OUT_ROOT}"

  echo "NPZ output: ${NPZ_OUT_ROOT}"
fi
