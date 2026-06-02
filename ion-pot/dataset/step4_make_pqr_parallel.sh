#!/usr/bin/env bash
set -euo pipefail

IDS="${IDS:-08_manifests/ids_fit97_from_pdb.txt}"
PDBDIR="01_structures"
OUTDIR="04_pqr"
LOGDIR="07_logs/pdb2pqr"
PH="7.4"
FF="CHARMM"
if command -v nproc >/dev/null 2>&1; then
  DEFAULT_NPROC="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  DEFAULT_NPROC="$(getconf _NPROCESSORS_ONLN)"
else
  DEFAULT_NPROC="4"
fi
NPROC="${NPROC:-$DEFAULT_NPROC}"
JOBS="${JOBS:-$NPROC}"   # override: NPROC=32 bash step4_make_pqr_parallel.sh

if [[ ! -s "$IDS" ]]; then
  echo "[ERR] missing or empty IDs file: $IDS"
  exit 1
fi

command -v parallel >/dev/null 2>&1 || {
  echo "[ERR] GNU parallel not found. Try: conda install -c conda-forge parallel"
  exit 1
}

mkdir -p "$OUTDIR" "$LOGDIR"
echo "[STEP4] pdb2pqr jobs=$JOBS (nproc=$NPROC)"

do_one() {
  local ID="$1"
  local PDB="${PDBDIR}/${ID}.pdb"
  local PQR="${OUTDIR}/${ID}_pH${PH}_${FF}.pqr"
  local LOG="${LOGDIR}/${ID}.log"

  [[ -n "$ID" ]] || exit 0

  if [[ ! -s "$PDB" ]]; then
    echo "[SKIP] $ID missing PDB"
    exit 0
  fi
  if [[ -s "$PQR" ]]; then
    echo "[SKIP] $ID PQR exists"
    exit 0
  fi

  echo "[PDB2PQR] $ID"
  pdb2pqr --ff="$FF" --with-ph="$PH" "$PDB" "$PQR" >"$LOG" 2>&1 || {
    echo "[FAIL] $ID (see $LOG)"
    exit 0
  }
}

export -f do_one
export PDBDIR OUTDIR LOGDIR PH FF

# -j: concurrent jobs; --halt never: keep going on failures; --line-buffer: nicer logs
parallel -j "$JOBS" --halt never --line-buffer do_one :::: "$IDS"
