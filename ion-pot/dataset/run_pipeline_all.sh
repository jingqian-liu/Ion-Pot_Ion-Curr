#!/usr/bin/env bash
set -euo pipefail

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_NPROC="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  DEFAULT_NPROC="$(getconf _NPROCESSORS_ONLN)"
else
  DEFAULT_NPROC="4"
fi
NPROC="${NPROC:-$DEFAULT_NPROC}"
export NPROC
SPACING="${SPACING:-0.5}"
DIME="${DIME:-193}"
if [[ -n "${SALTS:-}" ]]; then
  SALTS="$SALTS"
elif [[ -n "${SALT:-}" ]]; then
  SALTS="$SALT"
else
  SALTS="0.050,0.150,0.450"
fi
export SPACING DIME
export SALTS

echo "[PIPELINE] Starting full pipeline"
echo "           nproc=$NPROC"
echo "           apbs_spacing=$SPACING dime=$DIME"
echo "           salts=$SALTS"

echo "[PIPELINE] step1_download"
bash step1_download.sh

echo "[PIPELINE] step2_foldseek"
bash step2_foldseek.sh

echo "[PIPELINE] step3_split"
bash step3_split.sh

echo "[PIPELINE] step4_make_pqr_parallel"
bash step4_make_pqr_parallel.sh

echo "[PIPELINE] step5 (APBS inputs; supports SALTS='0.05,0.15,0.45')"
SPACING="$SPACING" DIME="$DIME" SALTS="$SALTS" bash step5.sh

if [[ "${RUN_STEP3B:-0}" == "1" ]]; then
  echo "[PIPELINE] step3b_filter_splits_by_apbs_fit (optional)"
  bash step3b_filter_splits_by_apbs_fit.sh
fi

echo "[PIPELINE] step6 (APBS runs)"
bash step6.sh

echo "[PIPELINE] Done"
