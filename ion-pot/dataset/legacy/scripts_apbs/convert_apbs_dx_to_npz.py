#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


CONC_RE = re.compile(r"_c(\d+)p(\d+)(?:_|\.|$)")
SALT_RE = re.compile(r"(?:^|/)salt(\d+)p(\d+)(?:/|$)")


@dataclass
class DxMeta:
    counts: tuple[int, int, int]
    origin: Optional[tuple[float, float, float]]
    deltas: list[tuple[float, float, float]]
    grid_space: Optional[float]


def parse_conc_from_name(path: str) -> float:
    base = os.path.basename(path)
    m = CONC_RE.search(base)
    if m is None:
        normalized_path = path.replace("\\", "/").lower()
        m = SALT_RE.search(normalized_path)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return 0.0


def infer_grid_space_from_deltas(deltas: list[tuple[float, float, float]]) -> Optional[float]:
    candidates = []
    for dx, dy, dz in deltas:
        n = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if n > 0:
            candidates.append(n)
    if not candidates:
        return None
    return float(np.median(np.asarray(candidates, dtype=np.float64)))


def read_dx(path: str) -> tuple[np.ndarray, DxMeta]:
    counts = None
    origin = None
    deltas: list[tuple[float, float, float]] = []
    data_start = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        s = line.strip()
        if "counts" in s and "gridpositions" in s:
            m = re.search(r"counts\s+(\d+)\s+(\d+)\s+(\d+)", s)
            if m:
                counts = tuple(int(x) for x in m.groups())
        elif s.startswith("origin "):
            parts = s.split()
            if len(parts) >= 4:
                origin = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif s.startswith("delta "):
            parts = s.split()
            if len(parts) >= 4:
                deltas.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif "data follows" in s:
            data_start = i + 1
            break

    if counts is None:
        raise ValueError(f"Could not parse counts from DX file: {path}")
    if data_start is None:
        raise ValueError(f"Could not find data section in DX file: {path}")

    n_items = counts[0] * counts[1] * counts[2]
    vals = []
    for line in lines[data_start:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("attribute") or s.startswith("object"):
            break
        vals.extend(float(x) for x in s.split())
        if len(vals) >= n_items:
            break

    if len(vals) < n_items:
        raise ValueError(f"DX data length mismatch in {path}: expected {n_items}, got {len(vals)}")

    arr = np.asarray(vals[:n_items], dtype=np.float32).reshape(counts)
    meta = DxMeta(
        counts=counts,
        origin=origin,
        deltas=deltas,
        grid_space=infer_grid_space_from_deltas(deltas),
    )
    return arr, meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert APBS DX triplets (pot/charge/vdw) to NPZ files for PBGNN."
    )
    parser.add_argument("--pot-glob", required=True, help="Glob for pot DX files, e.g. /path/pot/*_pot.dx")
    parser.add_argument("--charge-dir", required=True, help="Directory containing *_charge.dx")
    parser.add_argument("--vdw-dir", required=True, help="Directory containing *_vdw.dx")
    parser.add_argument("--out-root", required=True, help="Output directory for NPZ files")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output NPZ files",
    )
    parser.add_argument(
        "--strict-grid-check",
        action="store_true",
        help="Fail if per-file grid_space from pot/charge/vdw are inconsistent",
    )
    parser.add_argument(
        "--manifest-csv",
        default="",
        help="Optional manifest CSV path. Defaults to <out-root>/manifest.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pot_paths = sorted(glob.glob(args.pot_glob))
    if not pot_paths:
        raise FileNotFoundError(f"No pot files matched: {args.pot_glob}")

    os.makedirs(args.out_root, exist_ok=True)
    manifest_path = args.manifest_csv or os.path.join(args.out_root, "manifest.csv")

    rows = []
    converted = 0
    skipped = 0

    for pot_path in pot_paths:
        base = os.path.basename(pot_path)
        if not base.endswith("_pot.dx"):
            continue
        root = base[: -len("_pot.dx")]
        charge_path = os.path.join(args.charge_dir, f"{root}_charge.dx")
        vdw_path = os.path.join(args.vdw_dir, f"{root}_vdw.dx")
        out_path = os.path.join(args.out_root, f"{root}.npz")

        if not os.path.exists(charge_path) or not os.path.exists(vdw_path):
            raise FileNotFoundError(
                f"Missing pair files for {pot_path}.\n"
                f"charge: {charge_path}\n"
                f"vdw: {vdw_path}\n"
                "Tip: rerun APBS sweep with WRITE_EXTRA_MAPS=true."
            )

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        pot, meta_p = read_dx(pot_path)
        charge, meta_c = read_dx(charge_path)
        vdw, meta_v = read_dx(vdw_path)

        if not (pot.shape == charge.shape == vdw.shape):
            raise ValueError(
                "Shape mismatch:\n"
                f"pot: {pot_path} -> {pot.shape}\n"
                f"charge: {charge_path} -> {charge.shape}\n"
                f"vdw: {vdw_path} -> {vdw.shape}"
            )

        gs_candidates = [x for x in (meta_p.grid_space, meta_c.grid_space, meta_v.grid_space) if x is not None]
        grid_space = float(gs_candidates[0]) if gs_candidates else 0.5
        if args.strict_grid_check and len(gs_candidates) >= 2:
            if max(gs_candidates) - min(gs_candidates) > 1e-6:
                raise ValueError(
                    f"Inconsistent grid_space in {root}: {gs_candidates}"
                )

        grid_origin = meta_p.origin if meta_p.origin is not None else (0.0, 0.0, 0.0)
        ionic_conc = parse_conc_from_name(pot_path)

        np.savez_compressed(
            out_path,
            pot=pot,
            charge=charge,
            vdw=vdw,
            grid_space=np.array(grid_space, dtype=np.float32),
            grid_dims=np.array(pot.shape, dtype=np.int32),
            grid_origin=np.array(grid_origin, dtype=np.float32),
            ionic_conc=np.array(ionic_conc, dtype=np.float32),
            conc=np.array(ionic_conc, dtype=np.float32),
        )

        rows.append(
            {
                "root": root,
                "pot_path": pot_path,
                "charge_path": charge_path,
                "vdw_path": vdw_path,
                "npz_path": out_path,
                "ionic_conc": ionic_conc,
                "grid_space": grid_space,
                "shape": str(tuple(int(x) for x in pot.shape)),
            }
        )
        converted += 1

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "root",
                "pot_path",
                "charge_path",
                "vdw_path",
                "npz_path",
                "ionic_conc",
                "grid_space",
                "shape",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    print(f"Manifest: {manifest_path}")
    print(f"NPZ root: {args.out_root}")


if __name__ == "__main__":
    main()
