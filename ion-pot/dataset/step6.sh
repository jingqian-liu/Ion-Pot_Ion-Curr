#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${IN_DIR:-05_apbs_in}"
RUN_ONE="${RUN_ONE:-run_apbs_one.sh}"

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_NPROC="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  DEFAULT_NPROC="$(getconf _NPROCESSORS_ONLN)"
else
  DEFAULT_NPROC="4"
fi
NPROC="${NPROC:-$DEFAULT_NPROC}"
JOBS="${JOBS:-$NPROC}"

if [[ ! -d "$IN_DIR" ]]; then
  echo "[ERR] missing input directory: $IN_DIR"
  echo "Run step5.sh first, or set IN_DIR=/path/to/apbs_inputs"
  exit 1
fi

if [[ ! -f "$RUN_ONE" ]]; then
  echo "[ERR] missing helper script: $RUN_ONE"
  exit 1
fi

mkdir -p 07_logs/apbs

mapfile -d '' IN_FILES < <(find "$IN_DIR" -type f -name "*.in" -print0 | sort -z)
if [[ "${#IN_FILES[@]}" -eq 0 ]]; then
  echo "[ERR] no .in files found in $IN_DIR"
  exit 1
fi

echo "[STEP6] Running APBS in parallel"
echo "        inputs=${#IN_FILES[@]} jobs=$JOBS (nproc=$NPROC) in_dir=$IN_DIR"

printf '%s\0' "${IN_FILES[@]}" | xargs -0 -n 1 -P "$JOBS" bash "$RUN_ONE"

echo "[DONE] step6"
