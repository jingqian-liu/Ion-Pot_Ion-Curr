#!/usr/bin/env python
"""
Rewrite absolute paths inside a data_split.json produced on one machine so it
works on another (e.g. Delta -> tbgl). Preserves the trailing
`saltXpXX/<id>.npz` (or `_cXpXX` filename) structure that
`infer_ionic_conc_from_path` needs to bucket per-concentration metrics.

Examples
--------
# Simple prefix swap and verify every rewritten file exists:
python scripts/rewrite_data_split_paths.py \
    --input  /home/pinhao2/pbgnn_outputs/delta/fno_full193_20260417_110735/runs/add/train/data_split.json \
    --output /home/pinhao2/pbgnn_outputs/delta/fno_full193_20260417_110735/runs/add/train/data_split.tbgl.json \
    --old-prefix /work/nvme/lhi/gu1 \
    --new-prefix /data/server10/pinhao2/ML/Ion_Prediction/alphafold_v6/06_apbs_out_npz \
    --verify

# Don't know the old prefix? Auto-strip everything up to and including the
# first salt<d>p<dd>/ component, and re-root under --new-prefix:
python scripts/rewrite_data_split_paths.py \
    --input  .../data_split.json \
    --output .../data_split.tbgl.json \
    --new-prefix /data/.../06_apbs_out_npz \
    --mode keep-salt-tail \
    --verify
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List


SALT_RE = re.compile(r"/(salt\d+p\d+)/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="Path to original data_split.json (single-file mode).")
    p.add_argument("--output", type=Path, help="Where to write the rewritten JSON (single-file mode).")
    p.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help=(
            "Recursively find every data_split.json under this directory and rewrite each "
            "to a sibling file (see --batch-suffix). Use instead of --input/--output."
        ),
    )
    p.add_argument(
        "--batch-suffix",
        type=str,
        default=".tbgl.json",
        help="Suffix appended in place of '.json' for each batch output (default: .tbgl.json).",
    )
    p.add_argument(
        "--batch-overwrite",
        action="store_true",
        help="In batch mode, overwrite the output file if it already exists.",
    )
    p.add_argument(
        "--mode",
        choices=["prefix", "keep-salt-tail"],
        default="prefix",
        help=(
            "prefix: replace --old-prefix with --new-prefix. "
            "keep-salt-tail: drop everything before the first /saltXpXX/ and re-root under --new-prefix."
        ),
    )
    p.add_argument("--old-prefix", type=str, default=None, help="Prefix to strip (mode=prefix).")
    p.add_argument("--new-prefix", type=str, required=True, help="New root to prepend.")
    p.add_argument(
        "--verify",
        action="store_true",
        help="After rewriting, check every file exists; exit non-zero if any are missing.",
    )
    p.add_argument(
        "--allow-missing",
        type=int,
        default=0,
        help="When --verify is on, tolerate up to this many missing files (default: 0).",
    )
    return p.parse_args()


def rewrite_path(path: str, mode: str, old_prefix: str | None, new_prefix: str) -> str:
    new_prefix = new_prefix.rstrip("/")
    if mode == "prefix":
        if old_prefix is None:
            raise SystemExit("--old-prefix is required when --mode prefix.")
        op = old_prefix.rstrip("/")
        if path.startswith(op + "/"):
            return new_prefix + path[len(op):]
        # allow exact-match prefix without trailing slash
        if path.startswith(op):
            return new_prefix + path[len(op):]
        return path  # untouched if it doesn't match
    # keep-salt-tail
    m = SALT_RE.search(path)
    if not m:
        # Fall back: take just the basename and hope --new-prefix is the flat dir.
        return f"{new_prefix}/{Path(path).name}"
    tail = path[m.start() + 1:]  # drop the leading '/'
    return f"{new_prefix}/{tail}"


def rewrite_one(
    input_path: Path,
    output_path: Path,
    mode: str,
    old_prefix: str | None,
    new_prefix: str,
    verify: bool,
    allow_missing: int,
) -> bool:
    """Rewrite a single data_split.json. Returns True on success (within tolerance)."""
    with open(input_path, "r") as f:
        data = json.load(f)

    if "splits" not in data or not isinstance(data["splits"], dict):
        print(f"[SKIP] {input_path}: no top-level 'splits' dict.")
        return False

    rewritten = 0
    untouched = 0
    missing: List[str] = []
    for split_name, paths in data["splits"].items():
        if not isinstance(paths, list):
            continue
        new_paths = []
        for p in paths:
            new_p = rewrite_path(p, mode, old_prefix, new_prefix)
            if new_p != p:
                rewritten += 1
            else:
                untouched += 1
            new_paths.append(new_p)
            if verify and not Path(new_p).is_file():
                missing.append(new_p)
        data["splits"][split_name] = new_paths

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote: {output_path}")
    print(f"  rewritten paths: {rewritten}")
    print(f"  untouched paths: {untouched}")

    if verify:
        if len(missing) > allow_missing:
            print(f"  [ERR] {len(missing)} files missing (tolerance={allow_missing}).")
            for mp in missing[:10]:
                print(f"        - {mp}")
            if len(missing) > 10:
                print(f"        ... and {len(missing) - 10} more")
            return False
        if missing:
            print(f"  [WARN] {len(missing)} files missing (within tolerance={allow_missing}).")
        else:
            print("  verify: OK (all files exist).")
    return True


def main() -> None:
    args = parse_args()

    if args.batch_root is not None:
        if args.input is not None or args.output is not None:
            raise SystemExit("--batch-root is mutually exclusive with --input/--output.")
        if not args.batch_root.is_dir():
            raise SystemExit(f"--batch-root is not a directory: {args.batch_root}")
        found = sorted(args.batch_root.rglob("data_split.json"))
        if not found:
            raise SystemExit(f"No data_split.json files found under {args.batch_root}")
        print(f"Found {len(found)} data_split.json files under {args.batch_root}")

        suffix = args.batch_suffix
        if not suffix.endswith(".json"):
            suffix = suffix + ".json"

        n_ok = 0
        n_err = 0
        for ip in found:
            op = ip.with_name(ip.name.replace(".json", suffix))
            if op.exists() and not args.batch_overwrite:
                print(f"[SKIP] {op} already exists (use --batch-overwrite to force).")
                continue
            print(f"\n=== {ip}")
            ok = rewrite_one(
                input_path=ip,
                output_path=op,
                mode=args.mode,
                old_prefix=args.old_prefix,
                new_prefix=args.new_prefix,
                verify=args.verify,
                allow_missing=args.allow_missing,
            )
            if ok:
                n_ok += 1
            else:
                n_err += 1
        print(f"\nBatch done: {n_ok} ok, {n_err} failed.")
        if n_err:
            sys.exit(1)
        return

    # single-file mode
    if args.input is None or args.output is None:
        raise SystemExit("--input and --output are required unless --batch-root is set.")
    ok = rewrite_one(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
        verify=args.verify,
        allow_missing=args.allow_missing,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
