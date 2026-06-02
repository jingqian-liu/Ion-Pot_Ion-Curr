#!/usr/bin/env bash
set -euo pipefail

f="$1"
id="$(basename "$f" .in)"
rel_in="${f#./}"
safe_rel="${rel_in//\//__}"

mkdir -p 07_logs/apbs

get_dx_out() {
  local kind="$1"
  local base
  base="$(awk -v k="$kind" '$1=="write" && $2==k && $3=="dx" {print $4; exit}' "$f")"
  if [[ -n "$base" ]]; then
    echo "${base}.dx"
  fi
}

pot="$(get_dx_out pot)"
chg="$(get_dx_out charge)"
vdw="$(get_dx_out vdw)"
nd="$(get_dx_out ndens || true)"
log="07_logs/apbs/${safe_rel%.in}.log"

if [[ -z "$pot" || -z "$chg" || -z "$vdw" ]]; then
  echo "[FAIL] $id missing write directives in $f"
  exit 2
fi

mkdir -p "$(dirname "$pot")" "$(dirname "$chg")" "$(dirname "$vdw")"
if [[ -n "$nd" ]]; then
  mkdir -p "$(dirname "$nd")"
fi

outputs=("$pot" "$chg" "$vdw")
if [[ -n "$nd" ]]; then
  outputs+=("$nd")
fi

# resumable skip
all_ok=1
for out in "${outputs[@]}"; do
  if [[ ! -s "$out" ]]; then
    all_ok=0
    break
  fi
done
if [[ "$all_ok" -eq 1 ]]; then
  echo "[SKIP] $id"
  exit 0
fi

# run
apbs "$f" > "$log" 2>&1 || true

# success/fail
all_ok=1
for out in "${outputs[@]}"; do
  if [[ ! -s "$out" ]]; then
    all_ok=0
    break
  fi
done
if [[ "$all_ok" -eq 1 ]]; then
  echo "[OK] $id"
  exit 0
else
  echo "[FAIL] $id  (see $log)"
  exit 2
fi
