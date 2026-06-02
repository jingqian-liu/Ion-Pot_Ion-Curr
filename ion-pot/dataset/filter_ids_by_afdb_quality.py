#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import numpy as np


def parse_bfactor(line: str) -> float | None:
    if len(line) < 66:
        return None
    token = line[60:66].strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def residue_key_from_atom_line(line: str) -> tuple[str, str, str]:
    chain = line[21].strip() if len(line) > 21 else ""
    resseq = line[22:26].strip() if len(line) > 26 else ""
    icode = line[26].strip() if len(line) > 26 else ""
    return chain, resseq, icode


def residue_plddt_values_from_pdb(pdb: Path) -> list[float]:
    residues: dict[tuple[str, str, str], dict[str, object]] = {}
    with pdb.open() as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            b = parse_bfactor(line)
            if b is None:
                continue

            key = residue_key_from_atom_line(line)
            atom_name = line[12:16].strip() if len(line) >= 16 else ""
            info = residues.setdefault(key, {"ca": None, "all": []})
            info["all"].append(b)
            if atom_name == "CA" and info["ca"] is None:
                info["ca"] = b

    if not residues:
        raise ValueError(f"no parsable ATOM/HETATM records in {pdb}")

    values: list[float] = []
    for info in residues.values():
        if info["ca"] is not None:
            values.append(float(info["ca"]))
        else:
            values.append(float(np.mean(info["all"])))
    return values


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
    ap.add_argument(
        "--mean-plddt-min",
        type=float,
        default=75.0,
        help="minimum mean residue pLDDT",
    )
    ap.add_argument(
        "--res-plddt-min",
        type=float,
        default=70.0,
        help="residue-level pLDDT cutoff used for fraction statistic",
    )
    ap.add_argument(
        "--frac-plddt-min",
        type=float,
        default=0.70,
        help="minimum fraction of residues with pLDDT >= res-plddt-min",
    )
    ap.add_argument(
        "--min-residues",
        type=int,
        default=30,
        help="minimum residue count to keep",
    )
    ap.add_argument("--kept-ids-out", required=True, help="output file for kept IDs")
    ap.add_argument("--skipped-ids-out", default=None, help="optional output file for skipped IDs")
    ap.add_argument("--report-tsv", default=None, help="optional per-protein quality report")
    ap.add_argument("--filtered-dir", default=None, help="optional output directory for kept PDBs")
    ap.add_argument("--filtered-mode", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    files = sorted(pdb_dir.glob("*.pdb"))
    if not files:
        raise SystemExit(f"[ERR] no PDB files in {pdb_dir}")

    kept_ids: list[str] = []
    skipped_ids: list[str] = []
    kept_files: list[Path] = []
    report_rows: list[tuple[str, str, str, str, str, str]] = []

    for pdb in files:
        pid = pdb.stem
        try:
            residue_plddt = residue_plddt_values_from_pdb(pdb)
        except Exception as exc:
            skipped_ids.append(pid)
            report_rows.append((pid, "NA", "NA", "NA", "skip_parse_fail", str(exc)))
            print(f"[SKIP] {pid}: parse_fail ({exc})")
            continue

        n_res = len(residue_plddt)
        mean_plddt = float(np.mean(residue_plddt))
        frac_good = float(np.mean(np.array(residue_plddt) >= args.res_plddt_min))

        if n_res < args.min_residues:
            skipped_ids.append(pid)
            report_rows.append(
                (
                    pid,
                    str(n_res),
                    f"{mean_plddt:.3f}",
                    f"{frac_good:.3f}",
                    "skip_too_short",
                    f"n_res<{args.min_residues}",
                )
            )
            print(f"[SKIP] {pid}: too_short n_res={n_res}")
            continue

        if mean_plddt < args.mean_plddt_min:
            skipped_ids.append(pid)
            report_rows.append(
                (
                    pid,
                    str(n_res),
                    f"{mean_plddt:.3f}",
                    f"{frac_good:.3f}",
                    "skip_low_mean_plddt",
                    f"mean<{args.mean_plddt_min}",
                )
            )
            print(
                f"[SKIP] {pid}: low_mean_plddt mean={mean_plddt:.2f} "
                f"(min={args.mean_plddt_min:.2f})"
            )
            continue

        if frac_good < args.frac_plddt_min:
            skipped_ids.append(pid)
            report_rows.append(
                (
                    pid,
                    str(n_res),
                    f"{mean_plddt:.3f}",
                    f"{frac_good:.3f}",
                    "skip_low_plddt_fraction",
                    f"frac<{args.frac_plddt_min}",
                )
            )
            print(
                f"[SKIP] {pid}: low_plddt_fraction frac={frac_good:.3f} "
                f"(min={args.frac_plddt_min:.3f})"
            )
            continue

        kept_ids.append(pid)
        kept_files.append(pdb)
        report_rows.append(
            (
                pid,
                str(n_res),
                f"{mean_plddt:.3f}",
                f"{frac_good:.3f}",
                "keep",
                "passes_thresholds",
            )
        )
        print(
            f"[KEEP] {pid}: n_res={n_res} mean_plddt={mean_plddt:.2f} "
            f"frac_plddt>={args.res_plddt_min:.1f}={frac_good:.3f}"
        )

    write_ids(Path(args.kept_ids_out), kept_ids)
    if args.skipped_ids_out is not None:
        write_ids(Path(args.skipped_ids_out), skipped_ids)

    if args.report_tsv is not None:
        rpt = Path(args.report_tsv)
        rpt.parent.mkdir(parents=True, exist_ok=True)
        with rpt.open("w") as f:
            f.write("id\tn_res\tmean_plddt\tfrac_plddt_good\tstatus\treason\n")
            for row in report_rows:
                f.write("\t".join(row) + "\n")

    if args.filtered_dir is not None:
        stage_kept_structures(kept_files, Path(args.filtered_dir), args.filtered_mode)

    print(
        "Done. "
        f"total={len(files)}, kept={len(kept_ids)}, skipped={len(skipped_ids)}."
    )


if __name__ == "__main__":
    main()
