#!/usr/bin/env bash
set -euo pipefail

OUTDIR="01_structures"
QUALITY_FILTERED_OUTDIR="01_structures_qc"
FILTERED_OUTDIR="01_structures_fit97"
MANIFEST="08_manifests/download_manifest.tsv"

# pLDDT quality filter outputs
QUALITY_KEPT_IDS="08_manifests/ids_plddt_kept.txt"
QUALITY_SKIPPED_IDS="08_manifests/ids_plddt_skipped.txt"
QUALITY_REPORT="08_manifests/afdb_quality_report.tsv"
PLDDT_MEAN_MIN="${PLDDT_MEAN_MIN:-75}"
PLDDT_RES_MIN="${PLDDT_RES_MIN:-70}"
PLDDT_FRAC_MIN="${PLDDT_FRAC_MIN:-0.70}"
MIN_RESIDUES="${MIN_RESIDUES:-30}"

# APBS box-fit filter outputs
KEPT_IDS="08_manifests/ids_fit97_from_pdb.txt"
SKIPPED_IDS="08_manifests/ids_too_large_from_pdb.txt"
BOX_REPORT="08_manifests/pdb_box_report.tsv"
BOX_SIZE="97"

MAX_N="${MAX_N:-10000}"
ACCESSION_METHOD="${ACCESSION_METHOD:-offset}"
REVIEWED="${REVIEWED:-0}"      # set REVIEWED=1 for Swiss-Prot only
ORGANISM_ID="${ORGANISM_ID:-}" # e.g. 9606 for human
PAGE_SIZE="${PAGE_SIZE:-2000}"

mkdir -p "$OUTDIR" 08_manifests

# Download (add --reviewed if you want Swiss-Prot only)
cmd=(
python scripts/download.py
  --max-mass 50000 \
  --max-n "$MAX_N" \
  --format pdb \
  --out "$OUTDIR" \
  # --accession-method "$ACCESSION_METHOD" \
  # --page-size "$PAGE_SIZE" \
  --skip-existing
)

if [[ "$REVIEWED" == "1" ]]; then
  cmd+=(--reviewed)
fi

if [[ -n "$ORGANISM_ID" ]]; then
  cmd+=(--organism-id "$ORGANISM_ID")
fi

"${cmd[@]}"

# Create a simple manifest from local files (reproducibility baseline)
# (Better is to log UniProt metadata during download, but this is still useful)
{
  shopt -s nullglob
  pdb_files=("$OUTDIR"/*.pdb)
  echo -e "id\tpath\tsize_bytes"
  for f in "${pdb_files[@]}"; do
    b=$(basename "$f" .pdb)
    s=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
    echo -e "${b}\t${f}\t${s}"
  done
  shopt -u nullglob
} > "$MANIFEST"

echo "Wrote $MANIFEST"

# AFDB quality filter based on pLDDT statistics from PDB B-factor field.
python filter_ids_by_afdb_quality.py \
  --pdb-dir "$OUTDIR" \
  --mean-plddt-min "$PLDDT_MEAN_MIN" \
  --res-plddt-min "$PLDDT_RES_MIN" \
  --frac-plddt-min "$PLDDT_FRAC_MIN" \
  --min-residues "$MIN_RESIDUES" \
  --kept-ids-out "$QUALITY_KEPT_IDS" \
  --skipped-ids-out "$QUALITY_SKIPPED_IDS" \
  --report-tsv "$QUALITY_REPORT" \
  --filtered-dir "$QUALITY_FILTERED_OUTDIR" \
  --filtered-mode symlink

echo "Wrote pLDDT-kept IDs: $QUALITY_KEPT_IDS"
echo "Wrote pLDDT-skipped IDs: $QUALITY_SKIPPED_IDS"
echo "Wrote pLDDT report: $QUALITY_REPORT"
echo "Prepared pLDDT-filtered structures dir: $QUALITY_FILTERED_OUTDIR"

# Early geometric filter for APBS box; keeps only structures expected to fit.
python filter_ids_by_pdb_box.py \
  --pdb-dir "$QUALITY_FILTERED_OUTDIR" \
  --box-size "$BOX_SIZE" \
  --kept-ids-out "$KEPT_IDS" \
  --skipped-ids-out "$SKIPPED_IDS" \
  --report-tsv "$BOX_REPORT" \
  --filtered-dir "$FILTERED_OUTDIR" \
  --filtered-mode symlink

echo "Wrote kept IDs: $KEPT_IDS"
echo "Wrote skipped IDs: $SKIPPED_IDS"
echo "Wrote size report: $BOX_REPORT"
echo "Prepared filtered structures dir: $FILTERED_OUTDIR"
