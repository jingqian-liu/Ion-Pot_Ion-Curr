#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


CONC_RE = re.compile(r"_c(\d+)p(\d+)(?:_|\.|$)")
SALT_RE = re.compile(r"(?:^|/)salt(\d+)p(\d+)(?:/|$)")


@dataclass(frozen=True)
class APBSTask:
    split: str
    source_path: str
    target_id: str
    ionic_conc: float
    conc_tag: str
    pqr_path: Path
    input_path: Path
    log_path: Path
    pot_dx: Path
    charge_dx: Path
    vdw_dx: Path
    ndens_dx: Path
    duplicate_count: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare APBS input files from a data_split.json, optionally run APBS, "
            "and optionally convert DX outputs to NPZ with organized manifests."
        )
    )
    parser.add_argument(
        "--data-split-json-path",
        type=str,
        required=True,
        help="Path to data_split.json with {'splits': {'train':[], 'eval':[], 'test':[]}}.",
    )
    parser.add_argument(
        "--pqr-dir",
        type=str,
        required=True,
        help="Directory containing source PQR files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root for APBS inputs/outputs/manifests.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,eval,test",
        help="Comma-separated split names to process.",
    )
    parser.add_argument(
        "--pqr-pattern",
        type=str,
        default="{id}_pH7.4_CHARMM.pqr",
        help="Filename pattern under --pqr-dir, e.g. '{id}_pH7.4_CHARMM.pqr'.",
    )
    parser.add_argument(
        "--pqr-fallback-glob",
        action="store_true",
        help="If exact pattern path is missing, search recursively for '*{id}*.pqr'.",
    )
    parser.add_argument(
        "--concentration-source",
        type=str,
        choices=["from_path", "fixed"],
        default="from_path",
        help="How to set ionic concentration per task.",
    )
    parser.add_argument(
        "--fixed-concentration",
        type=float,
        default=None,
        help="Used when --concentration-source=fixed.",
    )
    parser.add_argument(
        "--target-concentrations",
        type=str,
        default="",
        help=(
            "Comma-separated concentrations to generate for every selected sample, "
            "e.g. '0.70,1.00'. If set, this overrides --concentration-source."
        ),
    )
    parser.add_argument(
        "--exclude-train-concentrations",
        action="store_true",
        help=(
            "When using --target-concentrations, drop any value that appears in the "
            "training split concentration set."
        ),
    )
    parser.add_argument(
        "--train-split-name",
        type=str,
        default="train",
        help="Split name used to infer train concentrations for filtering.",
    )
    parser.add_argument(
        "--concentration-eps",
        type=float,
        default=1e-8,
        help="Tolerance for concentration equality checks.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lpbe", "npbe"],
        default="npbe",
    )
    parser.add_argument("--nlev", type=int, default=4)
    parser.add_argument("--dime", type=int, default=193)
    parser.add_argument(
        "--fglen",
        type=float,
        default=96.0,
        help="Cubic fine grid length in Angstrom.",
    )
    parser.add_argument(
        "--cglen",
        type=float,
        default=96.0,
        help="Cubic coarse grid length in Angstrom.",
    )
    parser.add_argument("--pdie", type=float, default=2.0)
    parser.add_argument("--sdie", type=float, default=78.54)
    parser.add_argument("--temp", type=float, default=298.0)
    parser.add_argument("--ion-radius-plus", type=float, default=1.76375)
    parser.add_argument("--ion-radius-minus", type=float, default=2.27)
    parser.add_argument(
        "--run-apbs",
        action="store_true",
        help="Run APBS after preparing input files.",
    )
    parser.add_argument(
        "--apbs-bin",
        type=str,
        default="apbs",
        help="APBS executable path/name.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Parallel APBS jobs when --run-apbs is enabled.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing APBS outputs.",
    )
    parser.add_argument(
        "--convert-to-npz",
        action="store_true",
        help="Convert produced DX files to NPZ per split/salt after APBS run.",
    )
    parser.add_argument(
        "--strict-grid-check",
        action="store_true",
        help="Pass --strict-grid-check to DX->NPZ converter.",
    )
    return parser.parse_args()


def parse_concentration_from_path(path: str) -> Optional[float]:
    base = os.path.basename(path)
    match = CONC_RE.search(base)
    if not match:
        match = SALT_RE.search(path.replace("\\", "/").lower())
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None


def parse_concentration_list(raw: str) -> list[float]:
    values: list[float] = []
    if raw is None:
        return values
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def format_conc_tag(conc: float) -> str:
    conc_str = f"{conc:.6f}".rstrip("0").rstrip(".")
    if conc_str == "":
        conc_str = "0"
    return f"salt{conc_str.replace('-', 'm').replace('.', 'p')}"


def extract_target_id(path: str) -> str:
    name = Path(path).name
    suffixes = [
        "_sparse.pkl.gz",
        "_raw.pkl.gz",
        "_full.npz",
        ".pkl.gz",
        ".npz",
        ".pkl",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"_c\d+p\d+$", "", name)
    return name


def resolve_pqr_path(
    pqr_dir: Path,
    target_id: str,
    pqr_pattern: str,
    allow_fallback_glob: bool,
) -> Optional[Path]:
    candidate = pqr_dir / pqr_pattern.format(id=target_id)
    if candidate.exists():
        return candidate

    if not allow_fallback_glob:
        return None

    matches = sorted(pqr_dir.rglob(f"*{target_id}*.pqr"))
    if len(matches) == 0:
        return None
    return matches[0]


def write_apbs_input(task: APBSTask, args: argparse.Namespace) -> None:
    task.input_path.parent.mkdir(parents=True, exist_ok=True)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    task.pot_dx.parent.mkdir(parents=True, exist_ok=True)
    task.charge_dx.parent.mkdir(parents=True, exist_ok=True)
    task.vdw_dx.parent.mkdir(parents=True, exist_ok=True)
    task.ndens_dx.parent.mkdir(parents=True, exist_ok=True)

    pot_base = task.pot_dx.with_suffix("")
    charge_base = task.charge_dx.with_suffix("")
    vdw_base = task.vdw_dx.with_suffix("")
    ndens_base = task.ndens_dx.with_suffix("")

    content = (
        "read\n"
        f"    mol pqr {task.pqr_path.as_posix()}\n"
        "end\n\n"
        "elec\n"
        "    mg-auto\n"
        f"    nlev {args.nlev}\n"
        f"    dime {args.dime} {args.dime} {args.dime}\n"
        f"    fglen {args.fglen:.3f} {args.fglen:.3f} {args.fglen:.3f}\n"
        "    fgcent mol 1\n"
        f"    cglen {args.cglen:.3f} {args.cglen:.3f} {args.cglen:.3f}\n"
        "    cgcent mol 1\n"
        "    mol 1\n"
        f"    {args.mode}\n"
        "    bcfl mdh\n"
        f"    ion charge 1 radius {args.ion_radius_plus:.5f} conc {task.ionic_conc:.5f}\n"
        f"    ion charge -1 radius {args.ion_radius_minus:.5f} conc {task.ionic_conc:.5f}\n"
        f"    pdie {args.pdie:.3f}\n"
        f"    sdie {args.sdie:.3f}\n"
        "    chgm spl0\n"
        "    srfm smol\n"
        "    srad 1.4\n"
        "    sdens 10.0\n"
        f"    temp {args.temp:.1f}\n"
        "    calcenergy no\n"
        "    calcforce no\n"
        f"    write pot dx {pot_base.as_posix()}\n"
        f"    write charge dx {charge_base.as_posix()}\n"
        f"    write vdw dx {vdw_base.as_posix()}\n"
    )
    if args.mode == "npbe":
        content += f"    write ndens dx {ndens_base.as_posix()}\n"
    content += "end\n"

    task.input_path.write_text(content)


def outputs_exist(task: APBSTask, mode: str) -> bool:
    required = [task.pot_dx, task.charge_dx, task.vdw_dx]
    if mode == "npbe":
        required.append(task.ndens_dx)
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def run_one_apbs(task: APBSTask, args: argparse.Namespace) -> tuple[str, str]:
    if outputs_exist(task, args.mode) and not args.overwrite:
        return "skip_exists", ""

    cmd = [args.apbs_bin, str(task.input_path)]
    with open(task.log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        return "fail", f"apbs return code {proc.returncode}"
    if not outputs_exist(task, args.mode):
        return "fail", "missing output dx files after APBS finished"
    return "ok", ""


def write_csv(path: Path, rows: list[dict], header: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_split_concentrations(
    split_dict: dict, split_name: str, eps: float
) -> list[float]:
    concs: list[float] = []
    for source_path in split_dict.get(split_name, []):
        parsed = parse_concentration_from_path(source_path)
        if parsed is None:
            continue
        exists = any(abs(parsed - x) <= eps for x in concs)
        if not exists:
            concs.append(float(parsed))
    return sorted(concs)


def build_tasks(args: argparse.Namespace) -> tuple[list[APBSTask], list[dict], list[float]]:
    split_json_path = Path(args.data_split_json_path)
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    split_dict = split_data.get("splits", {})

    split_names = [x.strip() for x in args.splits.split(",") if x.strip()]
    if len(split_names) == 0:
        raise ValueError("No split names provided.")

    pqr_dir = Path(args.pqr_dir)
    out_root = Path(args.output_root)
    requested_target_concs = parse_concentration_list(args.target_concentrations)
    train_concs = infer_split_concentrations(
        split_dict=split_dict,
        split_name=args.train_split_name,
        eps=args.concentration_eps,
    )

    tasks: list[APBSTask] = []
    issues: list[dict] = []
    dedupe_counts: dict[tuple[str, str, str], int] = {}

    for split in split_names:
        if split not in split_dict:
            issues.append(
                {
                    "split": split,
                    "source_path": "",
                    "target_id": "",
                    "ionic_conc": "",
                    "issue": "split_not_found_in_json",
                }
            )
            continue

        for source_path in split_dict[split]:
            target_id = extract_target_id(source_path)
            target_concs: list[float]
            if requested_target_concs:
                target_concs = [float(x) for x in requested_target_concs]
            elif args.concentration_source == "fixed":
                if args.fixed_concentration is None:
                    raise ValueError(
                        "--fixed-concentration is required when --concentration-source=fixed."
                    )
                target_concs = [float(args.fixed_concentration)]
            else:
                parsed = parse_concentration_from_path(source_path)
                if parsed is None:
                    issues.append(
                        {
                            "split": split,
                            "source_path": source_path,
                            "target_id": target_id,
                            "ionic_conc": "",
                            "issue": "cannot_infer_concentration_from_path",
                        }
                    )
                    continue
                target_concs = [float(parsed)]

            if args.exclude_train_concentrations and requested_target_concs:
                kept_concs = [
                    c
                    for c in target_concs
                    if not any(abs(c - tc) <= args.concentration_eps for tc in train_concs)
                ]
                if not kept_concs:
                    issues.append(
                        {
                            "split": split,
                            "source_path": source_path,
                            "target_id": target_id,
                            "ionic_conc": "",
                            "issue": "all_target_concentrations_filtered_by_train_split",
                        }
                    )
                    continue
                target_concs = kept_concs

            pqr_path = resolve_pqr_path(
                pqr_dir=pqr_dir,
                target_id=target_id,
                pqr_pattern=args.pqr_pattern,
                allow_fallback_glob=args.pqr_fallback_glob,
            )
            if pqr_path is None:
                issues.append(
                    {
                        "split": split,
                        "source_path": source_path,
                        "target_id": target_id,
                        "ionic_conc": ",".join(f"{x:g}" for x in target_concs),
                        "issue": "pqr_not_found",
                    }
                )
                continue

            for ionic_conc in target_concs:
                conc_tag = format_conc_tag(ionic_conc)
                dedupe_key = (split, target_id, conc_tag)
                dedupe_counts[dedupe_key] = dedupe_counts.get(dedupe_key, 0) + 1
                if dedupe_counts[dedupe_key] > 1:
                    continue

                base_dir = out_root / "outputs" / split / conc_tag
                input_path = out_root / "inputs" / split / conc_tag / f"{target_id}.in"
                log_path = out_root / "logs" / split / conc_tag / f"{target_id}.log"
                task = APBSTask(
                    split=split,
                    source_path=source_path,
                    target_id=target_id,
                    ionic_conc=float(ionic_conc),
                    conc_tag=conc_tag,
                    pqr_path=pqr_path,
                    input_path=input_path,
                    log_path=log_path,
                    pot_dx=base_dir / "pot" / f"{target_id}_pot.dx",
                    charge_dx=base_dir / "charge" / f"{target_id}_charge.dx",
                    vdw_dx=base_dir / "vdw" / f"{target_id}_vdw.dx",
                    ndens_dx=base_dir / "ndens" / f"{target_id}_ndens.dx",
                    duplicate_count=1,
                )
                tasks.append(task)

    # Fill duplicate counts in task objects for manifest transparency.
    tasks = [
        APBSTask(
            **{
                **task.__dict__,
                "duplicate_count": dedupe_counts[(task.split, task.target_id, task.conc_tag)],
            }
        )
        for task in tasks
    ]
    return tasks, issues, train_concs


def maybe_convert_dx_to_npz(args: argparse.Namespace, task_rows: list[dict]) -> None:
    if not args.convert_to_npz:
        return

    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for row in task_rows:
        if row["status"] not in {"ok", "skip_exists"}:
            continue
        key = (row["split"], row["conc_tag"])
        if key not in grouped:
            grouped[key] = {}

    for split, conc_tag in sorted(grouped.keys()):
        base = Path(args.output_root) / "outputs" / split / conc_tag
        pot_glob = str(base / "pot" / "*_pot.dx")
        charge_dir = str(base / "charge")
        vdw_dir = str(base / "vdw")
        out_root = Path(args.output_root) / "npz" / split / conc_tag
        out_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "scripts/apbs/convert_apbs_dx_to_npz.py",
            "--pot-glob",
            pot_glob,
            "--charge-dir",
            charge_dir,
            "--vdw-dir",
            vdw_dir,
            "--out-root",
            str(out_root),
        ]
        if args.strict_grid_check:
            cmd.append("--strict-grid-check")

        print(f"[NPZ] Converting split={split}, conc={conc_tag}")
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            print(
                f"[WARN] DX->NPZ conversion failed for split={split}, conc={conc_tag} "
                f"with return code {proc.returncode}"
            )


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    manifests_dir = out_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    (manifests_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True)
    )
    split_snapshot = manifests_dir / "data_split.snapshot.json"
    split_snapshot.write_text(Path(args.data_split_json_path).read_text())

    tasks, issues, train_concs = build_tasks(args)
    print(f"Prepared {len(tasks)} unique APBS tasks from split json.")

    for task in tasks:
        write_apbs_input(task, args)

    task_rows: list[dict] = []
    if args.run_apbs:
        print(f"Running APBS with jobs={args.jobs} ...")
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            future_map = {pool.submit(run_one_apbs, task, args): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                status, message = future.result()
                task_rows.append(
                    {
                        "split": task.split,
                        "source_path": task.source_path,
                        "target_id": task.target_id,
                        "ionic_conc": task.ionic_conc,
                        "conc_tag": task.conc_tag,
                        "duplicate_count": task.duplicate_count,
                        "pqr_path": str(task.pqr_path),
                        "input_path": str(task.input_path),
                        "log_path": str(task.log_path),
                        "pot_dx": str(task.pot_dx),
                        "charge_dx": str(task.charge_dx),
                        "vdw_dx": str(task.vdw_dx),
                        "ndens_dx": str(task.ndens_dx),
                        "status": status,
                        "message": message,
                    }
                )
                if status == "ok":
                    print(f"[OK] {task.split} {task.conc_tag} {task.target_id}")
                elif status == "skip_exists":
                    print(f"[SKIP] existing {task.split} {task.conc_tag} {task.target_id}")
                else:
                    print(f"[FAIL] {task.split} {task.conc_tag} {task.target_id}: {message}")
    else:
        for task in tasks:
            task_rows.append(
                {
                    "split": task.split,
                    "source_path": task.source_path,
                    "target_id": task.target_id,
                    "ionic_conc": task.ionic_conc,
                    "conc_tag": task.conc_tag,
                    "duplicate_count": task.duplicate_count,
                    "pqr_path": str(task.pqr_path),
                    "input_path": str(task.input_path),
                    "log_path": str(task.log_path),
                    "pot_dx": str(task.pot_dx),
                    "charge_dx": str(task.charge_dx),
                    "vdw_dx": str(task.vdw_dx),
                    "ndens_dx": str(task.ndens_dx),
                    "status": "prepared_only",
                    "message": "",
                }
            )

    issues_path = manifests_dir / "prepare_issues.csv"
    write_csv(
        issues_path,
        issues,
        header=["split", "source_path", "target_id", "ionic_conc", "issue"],
    )

    tasks_path = manifests_dir / "apbs_tasks.csv"
    write_csv(
        tasks_path,
        task_rows,
        header=[
            "split",
            "source_path",
            "target_id",
            "ionic_conc",
            "conc_tag",
            "duplicate_count",
            "pqr_path",
            "input_path",
            "log_path",
            "pot_dx",
            "charge_dx",
            "vdw_dx",
            "ndens_dx",
            "status",
            "message",
        ],
    )

    if args.run_apbs and args.convert_to_npz:
        maybe_convert_dx_to_npz(args, task_rows)

    status_counts: dict[str, int] = {}
    for row in task_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    summary = {
        "num_tasks": len(task_rows),
        "num_prepare_issues": len(issues),
        "train_concentrations_inferred": train_concs,
        "status_counts": status_counts,
        "manifests": {
            "tasks": str(tasks_path),
            "issues": str(issues_path),
            "config": str(manifests_dir / "run_config.json"),
            "data_split_snapshot": str(split_snapshot),
        },
    }
    (manifests_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nDone.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
