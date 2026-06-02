#!/usr/bin/env bash
set -euo pipefail

# Override any variable when launching, e.g.
# IDS=03_splits/train.txt FGLEN=50 CGLEN=50 bash step5.sh
# SALTS="0.05,0.15,0.45" bash step5.sh
IDS="${IDS:-08_manifests/ids_fit97_from_pdb.txt}"
PQR_DIR="${PQR_DIR:-04_pqr}"
PQR_SUFFIX="${PQR_SUFFIX:-_pH7.4_CHARMM.pqr}"
OUT_DIR="${OUT_DIR:-05_apbs_in}"
OUT_ROOT="${OUT_ROOT:-06_apbs_out}"

FGLEN="${FGLEN:-96}"
CGLEN="${CGLEN:-96}"
MAX_PROTEIN_SIZE="${MAX_PROTEIN_SIZE:-$FGLEN}"
MODE="${MODE:-npbe}"
SPACING="${SPACING:-1.0}"
SALT="${SALT:-0.450}"
SALTS="${SALTS:-$SALT}"
DIME="${DIME:-97}"

KEPT_IDS_OUT="${KEPT_IDS_OUT:-08_manifests/ids_apbs_in_fit${FGLEN}.txt}"
SKIPPED_IDS_OUT="${SKIPPED_IDS_OUT:-08_manifests/ids_apbs_skipped_fit${FGLEN}.txt}"

if [[ ! -s "$IDS" ]]; then
  echo "[ERR] missing or empty IDs file: $IDS"
  echo "Set IDS=/path/to/ids.txt when running this script."
  exit 1
fi

mkdir -p "$OUT_DIR" 08_manifests

salts_raw="${SALTS//,/ }"
read -r -a SALT_LIST <<< "$salts_raw"
if [[ "${#SALT_LIST[@]}" -eq 0 ]]; then
  echo "[ERR] SALTS is empty after parsing: $SALTS"
  exit 1
fi

HAS_DIME_FLAG=0
if [[ -n "$DIME" ]]; then
  if python make_apbs_inputs.py --help 2>&1 | grep -q -- '--dime'; then
    HAS_DIME_FLAG=1
  else
    echo "[WARN] make_apbs_inputs.py does not support --dime; ignoring DIME=$DIME"
    echo "[WARN] With SPACING=1.0, set FGLEN=96 to effectively use dime=97."
  fi
fi

echo "[STEP5] Generating APBS input files"
echo "        ids=$IDS"
echo "        fglen=$FGLEN cglen=$CGLEN spacing=$SPACING mode=$MODE dime=${DIME:-auto}"
echo "        salts=${SALT_LIST[*]}"

first_kept=""
first_skipped=""
for salt in "${SALT_LIST[@]}"; do
  if [[ ! "$salt" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
    echo "[ERR] invalid salt concentration: '$salt'"
    exit 1
  fi

  tag="salt${salt//-/m}"
  tag="${tag//./p}"

  run_out_dir="$OUT_DIR"
  run_out_root="$OUT_ROOT"
  run_kept="$KEPT_IDS_OUT"
  run_skipped="$SKIPPED_IDS_OUT"
  if [[ "${#SALT_LIST[@]}" -gt 1 ]]; then
    run_out_dir="${OUT_DIR%/}/${tag}"
    run_out_root="${OUT_ROOT%/}/${tag}"
    run_kept="08_manifests/ids_apbs_in_fit${FGLEN}_${tag}.txt"
    run_skipped="08_manifests/ids_apbs_skipped_fit${FGLEN}_${tag}.txt"
  fi

  echo "        [salt=$salt] in_dir=$run_out_dir out_root=$run_out_root"
  cmd=(
    python make_apbs_inputs.py
    --ids "$IDS"
    --pqr-dir "$PQR_DIR"
    --pqr-suffix "$PQR_SUFFIX"
    --out-dir "$run_out_dir"
    --output-root "$run_out_root"
    --spacing "$SPACING"
    --fglen "$FGLEN"
    --cglen "$CGLEN"
    --max-protein-size "$MAX_PROTEIN_SIZE"
    --mode "$MODE"
    --salt "$salt"
    --kept-ids-out "$run_kept"
    --skipped-ids-out "$run_skipped"
  )

  if [[ "$HAS_DIME_FLAG" -eq 1 ]]; then
    cmd+=(--dime "$DIME")
  fi

  "${cmd[@]}"

  if [[ -z "$first_kept" ]]; then
    first_kept="$run_kept"
    first_skipped="$run_skipped"
  fi
done

if [[ "${#SALT_LIST[@]}" -gt 1 ]]; then
  if [[ "$first_kept" != "$KEPT_IDS_OUT" ]]; then
    cp "$first_kept" "$KEPT_IDS_OUT"
    echo "        [compat] copied first-salt kept IDs to $KEPT_IDS_OUT"
  fi
  if [[ "$first_skipped" != "$SKIPPED_IDS_OUT" ]]; then
    cp "$first_skipped" "$SKIPPED_IDS_OUT"
    echo "        [compat] copied first-salt skipped IDs to $SKIPPED_IDS_OUT"
  fi
fi

echo "[DONE] step5"
