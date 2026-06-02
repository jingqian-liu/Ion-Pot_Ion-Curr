#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from dx_to_npy import read_dx


SUBS_DEFAULT = ("pot", "charge", "vdw", "ndens")


def parse_ids(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0])
    return out


def infer_ionic_conc_from_path(path: Path) -> float | None:
    match = re.search(r"salt(\d+)p(\d+)", path.as_posix())
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def discover_ids(in_root: Path, key_sub: str) -> list[str]:
    base = in_root / key_sub
    suffix = f"_{key_sub}.dx"
    ids = []
    for p in sorted(base.glob(f"*{suffix}")):
        if p.name.endswith(suffix):
            ids.append(p.name[: -len(suffix)])
    return ids


def convert_one(
    pid: str,
    in_root: Path,
    out_root: Path,
    subs: tuple[str, ...],
    grid_space: float | None,
    ionic_conc: float | None,
) -> tuple[str, str, str]:
    arrays: dict[str, np.ndarray] = {}
    missing: list[str] = []

    for sub in subs:
        dx = in_root / sub / f"{pid}_{sub}.dx"
        if not (dx.is_file() and dx.stat().st_size > 0):
            missing.append(sub)
            continue
        arrays[sub] = read_dx(dx).astype(np.float32, copy=False)

    if missing:
        return ("skip", pid, f"missing subs: {','.join(missing)}")

    payload: dict[str, np.ndarray] = dict(arrays)
    if "charge" in arrays:
        atom_mask = (np.abs(arrays["charge"]) > 1e-12).astype(np.float32)
        payload["atom_mask"] = atom_mask
        payload["atom_type"] = atom_mask.copy()
    if grid_space is not None:
        payload["grid_space"] = np.array(grid_space, dtype=np.float32)
    if ionic_conc is not None:
        payload["ionic_conc"] = np.array(ionic_conc, dtype=np.float32)

    out_path = out_root / f"{pid}.npz"
    np.savez_compressed(out_path, **payload)
    return ("ok", pid, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default="06_apbs_out")
    ap.add_argument("--out-root", default="06_apbs_out_npz")
    ap.add_argument("--subs", nargs="+", default=list(SUBS_DEFAULT))
    ap.add_argument("--ids", default=None, help="optional file with one ID per line")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("NPROC", "8")),
        help="parallel workers, default from NPROC or 8",
    )
    ap.add_argument(
        "--grid-space",
        type=float,
        default=None,
        help="optional grid spacing metadata to store in npz",
    )
    ap.add_argument(
        "--ionic-conc",
        type=float,
        default=None,
        help="optional ionic concentration metadata to store in npz",
    )
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    subs = tuple(args.subs)
    if not subs:
        raise SystemExit("[ERR] --subs is empty")

    if args.ids is not None:
        ids = parse_ids(Path(args.ids))
    else:
        ids = discover_ids(in_root, subs[0])

    if not ids:
        raise SystemExit(
            f"[ERR] no IDs found from in_root={in_root} using key_sub={subs[0]}"
        )

    ionic_conc = args.ionic_conc
    if ionic_conc is None:
        ionic_conc = infer_ionic_conc_from_path(in_root)

    n_ok = n_skip = n_err = 0
    worker = partial(
        convert_one,
        in_root=in_root,
        out_root=out_root,
        subs=subs,
        grid_space=args.grid_space,
        ionic_conc=ionic_conc,
    )

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for status, pid, msg in ex.map(worker, ids, chunksize=1):
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
                print(f"[SKIP] {pid}: {msg}")
            else:
                n_err += 1
                print(f"[ERR] {pid}: {msg}")

    print(
        f"Done. workers={args.workers}, ids={len(ids)}, wrote={n_ok}, "
        f"skipped={n_skip}, errors={n_err}, out={out_root}"
    )


if __name__ == "__main__":
    main()
