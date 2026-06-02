#!/usr/bin/env bash
set -euo pipefail

# Override when needed, e.g.
# KEPT_IDS=08_manifests/ids_apbs_in_fit50.txt OUT_DIR=03_splits_fit50 bash step3b_filter_splits_by_apbs_fit.sh
SPLIT_DIR="${SPLIT_DIR:-03_splits}"
KEPT_IDS="${KEPT_IDS:-08_manifests/ids_apbs_in_fit96.txt}"
OUT_DIR="${OUT_DIR:-03_splits_apbs_fit}"

if [[ ! -s "$KEPT_IDS" ]]; then
  echo "[ERR] missing or empty kept IDs file: $KEPT_IDS"
  echo "Run step5.sh first, or override KEPT_IDS=/path/to/ids.txt"
  exit 1
fi

for split in train val test; do
  if [[ ! -f "$SPLIT_DIR/$split.txt" ]]; then
    echo "[ERR] missing split file: $SPLIT_DIR/$split.txt"
    echo "Run step3_split.sh first, or override SPLIT_DIR=/path/to/splits"
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

echo "[STEP3B] Filtering splits by APBS-fit IDs"
echo "         split_dir=$SPLIT_DIR"
echo "         kept_ids=$KEPT_IDS"
echo "         out_dir=$OUT_DIR"

for split in train val test; do
  in_file="$SPLIT_DIR/$split.txt"
  out_file="$OUT_DIR/$split.txt"

  grep -Fxf "$KEPT_IDS" "$in_file" > "$out_file" || true

  n_in=$(wc -l < "$in_file")
  n_out=$(wc -l < "$out_file")
  n_drop=$((n_in - n_out))
  echo "[$split] in=$n_in kept=$n_out dropped=$n_drop"
done

overlap=$(cat "$OUT_DIR"/train.txt "$OUT_DIR"/val.txt "$OUT_DIR"/test.txt | sort | uniq -d | wc -l | tr -d ' ')
if [[ "$overlap" != "0" ]]; then
  echo "[WARN] Found duplicated IDs across filtered splits: $overlap"
else
  echo "[OK] No duplicated IDs across filtered splits."
fi

echo "[DONE] step3b"
