#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def read_dx(path: Path) -> np.ndarray:
    nx = ny = nz = None
    data = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("object 1 class gridpositions counts"):
                parts = line.split()
                nx, ny, nz = map(int, parts[-3:])
                continue
            if line.startswith(("attribute", "object", "component")):
                continue

            # numeric line(s)
            try:
                data.extend(map(float, line.split()))
            except ValueError:
                continue

    if nx is None:
        raise ValueError(f"Missing 'gridpositions counts' in {path}")
    n = nx * ny * nz
    if len(data) < n:
        raise ValueError(f"Not enough floats in {path}: {len(data)} < {n}")
    if len(data) > n:
        data = data[:n]

    return np.asarray(data, dtype=np.float32).reshape((nx, ny, nz), order="C")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--subs", nargs="+", default=["pot","charge","vdw"])
    ap.add_argument("--pattern", default="*.dx")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    for sub in args.subs:
        inp = in_root / sub
        outp = out_root / sub
        outp.mkdir(parents=True, exist_ok=True)

        files = sorted(inp.glob(args.pattern))
        if not files:
            print(f"[WARN] no files in {inp}")
            continue

        for dx in files:
            npy = outp / (dx.name + ".npy")  # e.g. P10620_ndens.dx.npy
            if npy.exists() and npy.stat().st_size > 0:
                continue
            try:
                arr = read_dx(dx)
                np.save(npy, arr)
                n_ok += 1
            except Exception as e:
                print(f"[FAIL] {dx}: {e}")
                n_fail += 1

    print(f"Done. wrote={n_ok}, failed={n_fail}. out={out_root}")

if __name__ == "__main__":
    main()
