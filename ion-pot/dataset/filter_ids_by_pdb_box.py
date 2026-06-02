#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}
_DEFAULT_RADIUS = 1.70


def element_from_pdb_atom_line(line: str) -> str:
    if len(line) >= 78:
        elem = line[76:78].strip().upper()
        if elem:
            return elem

    name = line[12:16].strip()
    letters = "".join(ch for ch in name if ch.isalpha()).upper()
    if not letters:
        return "C"
    if len(letters) >= 2 and letters[:2] in _VDW_RADII:
        return letters[:2]
    return letters[0]


def xyz_from_pdb_atom_line(line: str) -> tuple[float, float, float]:
    return (
        float(line[30:38]),
        float(line[38:46]),
        float(line[46:54]),
    )


def pdb_extents_with_radii(pdb: Path) -> tuple[float, float, float]:
    xs0, xs1, ys0, ys1, zs0, zs1 = [], [], [], [], [], []
    with pdb.open() as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x, y, z = xyz_from_pdb_atom_line(line)
            except ValueError:
                continue

            elem = element_from_pdb_atom_line(line)
            r = _VDW_RADII.get(elem, _DEFAULT_RADIUS)
            xs0.append(x - r)
            xs1.append(x + r)
            ys0.append(y - r)
            ys1.append(y + r)
            zs0.append(z - r)
            zs1.append(z + r)

    if not xs0:
        raise ValueError(f"no atoms parsed from {pdb}")

    lx = max(xs1) - min(xs0)
    ly = max(ys1) - min(ys0)
    lz = max(zs1) - min(zs0)
    return lx, ly, lz


def write_ids(path: Path | None, ids: list[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{x}\n" for x in ids))


def stage_kept_structures(kept_files: list[Path], out_dir: Path, mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.pdb"):
        old.unlink()

    for src in kept_files:
        dst = out_dir / src.name
        if mode == "copy":
            shutil.copy2(src, dst)
            continue
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", required=True, help="directory containing *.pdb files")
    ap.add_argument("--box-size", type=float, required=True, help="max allowed protein extent in any axis (A)")
    ap.add_argument("--kept-ids-out", required=True, help="output file for IDs that fit")
    ap.add_argument("--skipped-ids-out", default=None, help="optional output file for IDs that do not fit")
    ap.add_argument("--report-tsv", default=None, help="optional per-protein size report")
    ap.add_argument("--filtered-dir", default=None, help="optional output directory for kept PDBs")
    ap.add_argument("--filtered-mode", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    files = sorted(pdb_dir.glob("*.pdb"))
    if not files:
        raise SystemExit(f"[ERR] no PDB files in {pdb_dir}")

    kept_ids, skipped_ids = [], []
    kept_files = []
    report_rows = []
    n_parse_fail = 0

    for pdb in files:
        pid = pdb.stem
        try:
            lx, ly, lz = pdb_extents_with_radii(pdb)
        except ValueError as e:
            skipped_ids.append(pid)
            n_parse_fail += 1
            report_rows.append((pid, pdb.as_posix(), "NA", "NA", "NA", "NA", "skip_parse_fail", str(e)))
            print(f"[SKIP] {pid}: {e}")
            continue

        max_extent = max(lx, ly, lz)
        if max_extent <= args.box_size:
            kept_ids.append(pid)
            kept_files.append(pdb)
            status = "keep"
            reason = "fits_box"
            print(f"[KEEP] {pid}: lx={lx:.2f} ly={ly:.2f} lz={lz:.2f} A")
        else:
            skipped_ids.append(pid)
            status = "skip_oversized"
            reason = "oversized"
            print(
                f"[SKIP] {pid}: lx={lx:.2f} ly={ly:.2f} lz={lz:.2f} A "
                f"(limit={args.box_size:.2f} A)"
            )

        report_rows.append(
            (
                pid,
                pdb.as_posix(),
                f"{lx:.3f}",
                f"{ly:.3f}",
                f"{lz:.3f}",
                f"{max_extent:.3f}",
                status,
                reason,
            )
        )

    write_ids(Path(args.kept_ids_out), kept_ids)
    if args.skipped_ids_out is not None:
        write_ids(Path(args.skipped_ids_out), skipped_ids)

    if args.report_tsv is not None:
        rpt = Path(args.report_tsv)
        rpt.parent.mkdir(parents=True, exist_ok=True)
        with rpt.open("w") as f:
            f.write("id\tpath\tlx_A\tly_A\tlz_A\tmax_extent_A\tstatus\treason\n")
            for row in report_rows:
                f.write("\t".join(row) + "\n")

    if args.filtered_dir is not None:
        stage_kept_structures(kept_files, Path(args.filtered_dir), args.filtered_mode)

    print(
        "Done. "
        f"total={len(files)}, "
        f"kept={len(kept_ids)}, "
        f"skipped={len(skipped_ids)}, "
        f"parse_fail={n_parse_fail}."
    )


if __name__ == "__main__":
    main()
