#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="01_structures_fit97"
if command -v nproc >/dev/null 2>&1; then
  DEFAULT_NPROC="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  DEFAULT_NPROC="$(getconf _NPROCESSORS_ONLN)"
else
  DEFAULT_NPROC="4"
fi
NPROC="${NPROC:-$DEFAULT_NPROC}"
THREADS="${THREADS:-$NPROC}"

mkdir -p 02_foldseek
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[ERR] missing filtered structures dir: $INPUT_DIR"
  echo "Run step1_download.sh first to prepare size-filtered inputs."
  exit 1
fi

cd 02_foldseek
rm -f foldDB* foldDB_clu* clusters.tsv
rm -rf tmp
foldseek createdb "../$INPUT_DIR" foldDB
foldseek cluster foldDB foldDB_clu tmp --cov-mode 1 -c 0.5 --min-seq-id 0.0 -e 1e-3 --threads "$THREADS"
foldseek createtsv foldDB foldDB foldDB_clu clusters.tsv
