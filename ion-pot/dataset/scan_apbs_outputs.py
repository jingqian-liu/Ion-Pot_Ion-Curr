#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np


SUBS_DEFAULT = ("pot", "charge", "vdw", "ndens")


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
            try:
                data.extend(map(float, line.split()))
            except ValueError:
                continue

    if nx is None:
        raise ValueError(f"Missing gridpositions counts in {path}")
    n = nx * ny * nz
    if len(data) < n:
        raise ValueError(f"Not enough values in {path}: {len(data)} < {n}")
    if len(data) > n:
        data = data[:n]

    return np.asarray(data, dtype=np.float32).reshape((nx, ny, nz), order="C")


def load_ids(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0])
    return out


def discover_ids(dx_root: Path, npy_root: Path, npz_root: Path, fmt: str) -> list[str]:
    ids = set()
    if fmt in ("dx", "auto"):
        for p in (dx_root / "ndens").glob("*_ndens.dx"):
            ids.add(p.name[: -len("_ndens.dx")])
    if fmt in ("npy", "auto"):
        for p in (npy_root / "ndens").glob("*_ndens.dx.npy"):
            ids.add(p.name[: -len("_ndens.dx.npy")])
    if fmt == "npz":
        for p in npz_root.glob("*.npz"):
            ids.add(p.stem)
    return sorted(ids)


def pick_input_path(dx_root: Path, npy_root: Path, pid: str, sub: str, fmt: str) -> tuple[Path, str]:
    npy = npy_root / sub / f"{pid}_{sub}.dx.npy"
    dx = dx_root / sub / f"{pid}_{sub}.dx"

    if fmt == "npy":
        if npy.is_file() and npy.stat().st_size > 0:
            return npy, "npy"
        raise FileNotFoundError(str(npy))

    if fmt == "dx":
        if dx.is_file() and dx.stat().st_size > 0:
            return dx, "dx"
        raise FileNotFoundError(str(dx))

    # auto: prefer npy
    if npy.is_file() and npy.stat().st_size > 0:
        return npy, "npy"
    if dx.is_file() and dx.stat().st_size > 0:
        return dx, "dx"
    raise FileNotFoundError(f"Missing both {npy} and {dx}")


def load_array(path: Path, source_fmt: str) -> np.ndarray:
    if source_fmt == "npy":
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return arr
    return read_dx(path)


def load_arrays_from_npz(
    npz_root: Path, pid: str, subs: list[str]
) -> tuple[dict[str, np.ndarray], list[str]]:
    arrays: dict[str, np.ndarray] = {}
    reasons: list[str] = []
    npz_path = npz_root / f"{pid}.npz"
    if not (npz_path.is_file() and npz_path.stat().st_size > 0):
        for sub in subs:
            reasons.append(f"missing_or_bad_{sub}:{npz_path}")
        return arrays, reasons

    try:
        with np.load(npz_path) as data:
            for sub in subs:
                if sub not in data:
                    reasons.append(f"missing_or_bad_{sub}:missing key '{sub}' in {npz_path}")
                    continue
                arr = np.asarray(data[sub])
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                arrays[sub] = arr
    except Exception as e:
        for sub in subs:
            reasons.append(f"missing_or_bad_{sub}:{e}")
    return arrays, reasons


def write_ids(path: Path | None, ids: list[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{x}\n" for x in ids))


def safe_stat(arr: np.ndarray, fn) -> float:
    return float(fn(arr))


def fmt_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return ""
    return "x".join(str(x) for x in shape)


def maybe_remove_bad(
    bad_ids: list[str],
    subs: list[str],
    dx_root: Path,
    npy_root: Path,
    npz_root: Path,
    remove_formats: str,
) -> int:
    removed = 0
    for pid in bad_ids:
        for sub in subs:
            if remove_formats in ("dx", "both"):
                p = dx_root / sub / f"{pid}_{sub}.dx"
                if p.exists():
                    p.unlink()
                    removed += 1
            if remove_formats in ("npy", "both"):
                p = npy_root / sub / f"{pid}_{sub}.dx.npy"
                if p.exists():
                    p.unlink()
                    removed += 1
        if remove_formats in ("npz", "both"):
            p = npz_root / f"{pid}.npz"
            if p.exists():
                p.unlink()
                removed += 1
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx-root", default="06_apbs_out")
    ap.add_argument("--npy-root", default="06_apbs_out_npy")
    ap.add_argument("--npz-root", default="06_apbs_out_npz")
    ap.add_argument("--format", choices=["auto", "dx", "npy", "npz"], default="auto")
    ap.add_argument("--ids", default=None, help="optional file with one ID per line")
    ap.add_argument("--subs", nargs="+", default=list(SUBS_DEFAULT))

    ap.add_argument("--ndens-max", type=float, default=5.0, help="fail if ndens max exceeds this")
    ap.add_argument("--ndens-p99-max", type=float, default=None, help="optional fail threshold for ndens p99")
    ap.add_argument("--ndens-min", type=float, default=-1e-4, help="fail if ndens min below this")
    ap.add_argument("--vdw-min", type=float, default=-1e-3)
    ap.add_argument("--vdw-max", type=float, default=1.001)

    ap.add_argument("--report-tsv", default="08_manifests/apbs_qc_report.tsv")
    ap.add_argument("--ok-ids-out", default="08_manifests/apbs_qc_ok_ids.txt")
    ap.add_argument("--bad-ids-out", default="08_manifests/apbs_qc_bad_ids.txt")

    ap.add_argument("--remove-bad", action="store_true", help="delete bad-case output files")
    ap.add_argument("--remove-formats", choices=["dx", "npy", "npz", "both"], default="both")
    args = ap.parse_args()

    dx_root = Path(args.dx_root)
    npy_root = Path(args.npy_root)
    npz_root = Path(args.npz_root)
    subs = list(args.subs)
    if not subs:
        raise SystemExit("[ERR] --subs is empty")

    if args.ids is not None:
        ids = load_ids(Path(args.ids))
    else:
        ids = discover_ids(dx_root, npy_root, npz_root, args.format)

    if not ids:
        raise SystemExit("[ERR] no IDs found to scan")

    report_path = Path(args.report_tsv)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "id", "status", "reasons", "shape",
        "pot_min", "pot_max",
        "charge_min", "charge_max",
        "vdw_min", "vdw_max",
        "ndens_min", "ndens_max", "ndens_p99",
    ]
    rows = []
    ok_ids, bad_ids = [], []

    for pid in ids:
        reasons = []
        arrays: dict[str, np.ndarray] = {}
        shape = None

        if args.format == "npz":
            arrays, npz_reasons = load_arrays_from_npz(npz_root, pid, subs)
            reasons.extend(npz_reasons)
        else:
            for sub in subs:
                try:
                    path, src = pick_input_path(dx_root, npy_root, pid, sub, args.format)
                    arr = load_array(path, src)
                    arrays[sub] = arr
                except Exception as e:
                    reasons.append(f"missing_or_bad_{sub}:{e}")

        if arrays:
            shapes = {k: v.shape for k, v in arrays.items()}
            uniq = set(shapes.values())
            if len(uniq) > 1:
                reasons.append("shape_mismatch")
            shape = next(iter(uniq)) if uniq else None

            for sub, arr in arrays.items():
                if not np.isfinite(arr).all():
                    reasons.append(f"non_finite_{sub}")

            if "vdw" in arrays:
                vmin = safe_stat(arrays["vdw"], np.min)
                vmax = safe_stat(arrays["vdw"], np.max)
                if vmin < args.vdw_min or vmax > args.vdw_max:
                    reasons.append("vdw_out_of_range")

            if "ndens" in arrays:
                nd = arrays["ndens"]
                nd_min = safe_stat(nd, np.min)
                nd_max = safe_stat(nd, np.max)
                nd_p99 = safe_stat(nd, lambda x: np.percentile(x, 99))
                if nd_min < args.ndens_min:
                    reasons.append("ndens_too_negative")
                if nd_max > args.ndens_max:
                    reasons.append("ndens_too_high")
                if args.ndens_p99_max is not None and nd_p99 > args.ndens_p99_max:
                    reasons.append("ndens_p99_too_high")

        status = "OK" if len(reasons) == 0 else "BAD"
        if status == "OK":
            ok_ids.append(pid)
        else:
            bad_ids.append(pid)

        row = {
            "id": pid,
            "status": status,
            "reasons": ";".join(reasons),
            "shape": fmt_shape(shape),
            "pot_min": "",
            "pot_max": "",
            "charge_min": "",
            "charge_max": "",
            "vdw_min": "",
            "vdw_max": "",
            "ndens_min": "",
            "ndens_max": "",
            "ndens_p99": "",
        }

        if "pot" in arrays:
            row["pot_min"] = safe_stat(arrays["pot"], np.min)
            row["pot_max"] = safe_stat(arrays["pot"], np.max)
        if "charge" in arrays:
            row["charge_min"] = safe_stat(arrays["charge"], np.min)
            row["charge_max"] = safe_stat(arrays["charge"], np.max)
        if "vdw" in arrays:
            row["vdw_min"] = safe_stat(arrays["vdw"], np.min)
            row["vdw_max"] = safe_stat(arrays["vdw"], np.max)
        if "ndens" in arrays:
            row["ndens_min"] = safe_stat(arrays["ndens"], np.min)
            row["ndens_max"] = safe_stat(arrays["ndens"], np.max)
            row["ndens_p99"] = safe_stat(arrays["ndens"], lambda x: np.percentile(x, 99))

        rows.append(row)

    with report_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    write_ids(Path(args.ok_ids_out) if args.ok_ids_out else None, ok_ids)
    write_ids(Path(args.bad_ids_out) if args.bad_ids_out else None, bad_ids)

    removed = 0
    if args.remove_bad and bad_ids:
        removed = maybe_remove_bad(
            bad_ids=bad_ids,
            subs=subs,
            dx_root=dx_root,
            npy_root=npy_root,
            npz_root=npz_root,
            remove_formats=args.remove_formats,
        )

    print(f"Scanned IDs: {len(ids)}")
    print(f"OK: {len(ok_ids)}")
    print(f"BAD: {len(bad_ids)}")
    print(f"Report: {report_path}")
    print(f"OK IDs: {args.ok_ids_out}")
    print(f"BAD IDs: {args.bad_ids_out}")
    if args.remove_bad:
        print(f"Removed files: {removed}")


if __name__ == "__main__":
    main()
