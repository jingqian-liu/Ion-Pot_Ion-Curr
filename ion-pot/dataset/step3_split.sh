#!/usr/bin/env bash
set -euo pipefail

python scripts/split_by_foldseek_clusters.py \
  --clusters-tsv 02_foldseek/clusters.tsv \
  --seed 0 \
  --out 03_splits