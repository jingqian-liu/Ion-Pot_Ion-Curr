#!/usr/bin/env python3
from __future__ import annotations
import argparse
import math
import re
from pathlib import Path

_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")
_DEFAULT_NLEV = 4


def apbs_compatible_dime(target_len: float, h: float = 1.0, nlev: int = _DEFAULT_NLEV) -> tuple[int, float]:
    """
    Build an APBS multigrid-compatible dime and matching box length for exact spacing.
    For nlev=4, APBS expects (dime - 1) to be a multiple of 8.
    """
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if h <= 0:
        raise ValueError("spacing must be positive")
    if nlev < 1:
        raise ValueError("nlev must be >= 1")

    mult = 2 ** (nlev - 1)
    target_intervals = int(math.ceil(target_len / h))
    intervals = int(math.ceil(target_intervals / mult) * mult)
    dime = intervals + 1
    used_len = intervals * h
    return dime, used_len


def validate_dime(dime: int, nlev: int = _DEFAULT_NLEV) -> None:
    if dime < 3:
        raise ValueError("dime must be >= 3")
    if dime % 2 == 0:
        raise ValueError("dime must be odd")
    mult = 2 ** (nlev - 1)
    if (dime - 1) % mult != 0:
        raise ValueError(
            f"dime={dime} is not APBS-compatible for nlev={nlev}; "
            f"(dime-1) must be divisible by {mult}"
        )


def pqr_extents_with_radii(pqr: Path) -> tuple[float, float, float]:
    xs0, xs1, ys0, ys1, zs0, zs1 = [], [], [], [], [], []
    with pqr.open() as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            nums = _FLOAT_RE.findall(line)
            if len(nums) < 5:
                continue
            x, y, z, _q, r = map(float, nums[-5:])
            xs0.append(x - r)
            xs1.append(x + r)
            ys0.append(y - r)
            ys1.append(y + r)
            zs0.append(z - r)
            zs1.append(z + r)

    if not xs0:
        raise ValueError(f"No atoms parsed from {pqr}")

    lx = max(xs1) - min(xs0)
    ly = max(ys1) - min(ys0)
    lz = max(zs1) - min(zs0)
    return lx, ly, lz


def write_ids(path: str | None, ids: list[str]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{x}\n" for x in ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="file with one ID per line")
    ap.add_argument("--pqr-dir", required=True)
    ap.add_argument("--pqr-suffix", default="_pH7.4_PARSE.pqr")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--output-root",
        default="06_apbs_out",
        help="root directory for APBS DX outputs referenced in generated .in files",
    )

    ap.add_argument("--spacing", type=float, default=1.0,
                    help="fine-grid spacing in Å (forced to 1.0)")
    ap.add_argument("--fglen", type=float, required=True, help="fixed cubic fine box length (Å)")
    ap.add_argument("--cglen", type=float, required=True, help="fixed cubic coarse box length (Å)")
    ap.add_argument("--dime", type=int, default=None,
                    help="optional cubic grid points override (uses dime dime dime).")
    ap.add_argument("--max-protein-size", type=float, default=None,
                    help="skip proteins with any axis extent larger than this (Å). Defaults to --fglen.")
    ap.add_argument("--kept-ids-out", default=None, help="optional output file for kept IDs")
    ap.add_argument("--skipped-ids-out", default=None, help="optional output file for skipped IDs")

    ap.add_argument("--pdie", type=float, default=2.0)
    ap.add_argument("--sdie", type=float, default=78.54)
    ap.add_argument("--temp", type=float, default=298.0)
    ap.add_argument("--mode", choices=["lpbe","npbe"], default="npbe")
    ap.add_argument("--salt", type=float, default=0.15)
    ap.add_argument("--ion-radius-plus", type=float, default=1.76375)
    ap.add_argument("--ion-radius-minus", type=float, default=2.27)
    args = ap.parse_args()

    if args.spacing <= 0:
        raise SystemExit("[ERR] spacing must be > 0")

    ids = [x.strip() for x in Path(args.ids).read_text().splitlines() if x.strip()]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)

    if args.dime is None:
        dime, fglen_used = apbs_compatible_dime(args.fglen, args.spacing, _DEFAULT_NLEV)
    else:
        try:
            validate_dime(args.dime, _DEFAULT_NLEV)
        except ValueError as e:
            raise SystemExit(f"[ERR] {e}")
        dime = args.dime
        fglen_used = args.spacing * (dime - 1)
        if abs(fglen_used - args.fglen) > 1e-9:
            print(
                f"[GRID] explicit dime used: fglen {args.fglen:.3f} -> {fglen_used:.3f} Å "
                f"to keep spacing={args.spacing:.3f} Å with dime={dime}"
            )

    cglen_used = max(args.cglen, fglen_used)
    max_protein_size = args.max_protein_size if args.max_protein_size is not None else fglen_used
    if max_protein_size > fglen_used:
        print(
            f"[GRID] max-protein-size reduced to match fine box: "
            f"{max_protein_size:.3f} -> {fglen_used:.3f} Å"
        )
        max_protein_size = fglen_used

    if args.dime is None and fglen_used != args.fglen:
        print(
            f"[GRID] APBS-compatible fine box adjusted: "
            f"fglen {args.fglen:.3f} -> {fglen_used:.3f} Å, dime={dime}, spacing={args.spacing:.3f} Å"
        )
    if cglen_used != args.cglen:
        print(
            f"[GRID] coarse box raised to keep cglen >= fglen: "
            f"cglen {args.cglen:.3f} -> {cglen_used:.3f} Å"
        )
    kept_ids, skipped_ids = [], []
    n_missing = n_parse_fail = n_oversized = n_written = 0

    for ID in ids:
        pqr = Path(args.pqr_dir) / f"{ID}{args.pqr_suffix}"
        if not pqr.exists():
            print(f"[SKIP] missing PQR: {pqr}")
            skipped_ids.append(ID)
            n_missing += 1
            continue

        try:
            lx, ly, lz = pqr_extents_with_radii(pqr)
        except ValueError as e:
            print(f"[SKIP] {ID}: {e}")
            skipped_ids.append(ID)
            n_parse_fail += 1
            continue

        protein_size = max(lx, ly, lz)
        if protein_size > max_protein_size:
            print(
                f"[SKIP] oversized {ID}: "
                f"lx={lx:.2f} ly={ly:.2f} lz={lz:.2f} Å (limit={max_protein_size:.2f} Å)"
            )
            skipped_ids.append(ID)
            n_oversized += 1
            continue

        pot_out = (output_root / "pot" / f"{ID}_pot").as_posix()
        ndens_out = (output_root / "ndens" / f"{ID}_ndens").as_posix()
        chg_out = (output_root / "charge" / f"{ID}_charge").as_posix()
        vdw_out = (output_root / "vdw" / f"{ID}_vdw").as_posix()

        in_text = f"""read
    mol pqr {pqr.as_posix()}
end

elec
    mg-auto
    nlev {_DEFAULT_NLEV}
    dime {dime} {dime} {dime}
    fglen {fglen_used:.3f} {fglen_used:.3f} {fglen_used:.3f}
    fgcent mol 1
    cglen {cglen_used:.3f} {cglen_used:.3f} {cglen_used:.3f}
    cgcent mol 1
    mol 1
    {args.mode}
    bcfl mdh
    ion charge 1 radius {args.ion_radius_plus:.5f} conc {args.salt:.5f}
    ion charge -1 radius {args.ion_radius_minus:.5f} conc {args.salt:.5f}
    pdie {args.pdie:.3f}
    sdie {args.sdie:.3f}
    chgm spl0
    srfm smol
    srad 1.4
    sdens 10.0
    temp {args.temp:.1f}
    calcenergy no
    calcforce no
    write pot dx {pot_out}
    write charge dx {chg_out}
    write vdw dx {vdw_out}
"""
        if args.mode == "npbe":
            in_text += f"    write ndens dx {ndens_out}\n"
        in_text += "end\n"

        out_path = out_dir / f"{ID}.in"
        out_path.write_text(in_text)
        print(f"[OK] wrote {out_path}")
        kept_ids.append(ID)
        n_written += 1

    write_ids(args.kept_ids_out, kept_ids)
    write_ids(args.skipped_ids_out, skipped_ids)
    print(
        "Done. "
        f"written={n_written}, "
        f"skipped_missing={n_missing}, "
        f"skipped_parse_fail={n_parse_fail}, "
        f"skipped_oversized={n_oversized}."
    )

if __name__ == "__main__":
    main()
