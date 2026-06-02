#!/usr/bin/env python
"""
Auto-discover every trained ablation checkpoint under --delta-root and launch
visualize_test_potential_maps_per_conc.py for each one, writing per-run metrics
into --eval-root/<backbone>_<cond>/.

Discovery rule
--------------
Walk:
    <delta-root>/<backbone>_full193_<timestamp>/runs/<cond>/train/models/model_final/model.safetensors

- <backbone>   parsed as the leading token of the run dir: unet | resnet | fno
- <cond>       parsed from the runs/<cond>/ subdir and normalized:
                 no_ion | noion | woion | wo_ion  -> woion
                 film                             -> film
                 add                              -> add
- <split-json> sibling 'data_split.tbgl.json' (falls back to 'data_split.json'
                 if you haven't run the rewrite yet).

When multiple (backbone, cond) checkpoints exist across different timestamps,
the one with the newest mtime wins. Override or skip via --pick / --skip.

Usage
-----
Dry-run (just print the commands it would execute):
    python scripts/eval_ablation_grid.py \
        --delta-root /home/pinhao2/pbgnn_outputs/delta \
        --eval-root  /home/pinhao2/pbgnn_outputs/eval_no_conc \
        --device cuda:7 --dry-run

Real run:
    python scripts/eval_ablation_grid.py \
        --delta-root /home/pinhao2/pbgnn_outputs/delta \
        --eval-root  /home/pinhao2/pbgnn_outputs/eval_no_conc \
        --device cuda:7
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BACKBONES = {"unet", "resnet", "fno"}
MAPPER_FOR = {"unet": "u-net", "resnet": "resnet", "fno": "fno"}
COND_ALIASES = {
    "add": "add",
    "film": "film",
    "woion": "woion",
    "wo_ion": "woion",
    "no_ion": "woion",
    "noion": "woion",
    "without_ion": "woion",
    "without-ion": "woion",
    "none": "woion",
}

RUN_DIR_RE = re.compile(r"^(unet|resnet|fno)(?:_|$)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--delta-root", required=True, type=Path)
    p.add_argument("--eval-root", required=True, type=Path)
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("/data/server10/pinhao2/ML/Ion_Prediction/pbgnn"),
    )
    p.add_argument(
        "--visualize-script",
        type=Path,
        default=None,
        help="Override path to visualize_test_potential_maps_per_conc.py. "
        "Defaults to <project-root>/scripts/visualize_test_potential_maps_per_conc.py "
        "(tries <project-root>/scripts/3d/... as fallback).",
    )
    p.add_argument(
        "--experiment-name",
        default="unet_psz32_ctx48_protein_complex_with_lset_fully_coverage_rotation_augmented_medium",
        help="Only changes trainer/energy-model defaults; backbone is overridden per-run.",
    )
    p.add_argument("--split", default="test", choices=["train", "eval", "test"])
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--patch-size", type=int, default=193)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--split-json-name",
        default="data_split.tbgl.json",
        help="File to use for --data-split-json-path (falls back to data_split.json if absent).",
    )
    p.add_argument(
        "--pick",
        action="append",
        default=[],
        metavar="BACKBONE_COND=PATH",
        help=(
            "Explicitly pin a (backbone, cond) pair to a specific run dir, e.g. "
            "--pick fno_add=/home/.../fno_full193_20260417_110735/runs/add. "
            "Can be repeated."
        ),
    )
    p.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="BACKBONE_COND",
        help="Skip this pair, e.g. --skip unet_woion. Can be repeated.",
    )
    p.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="BACKBONE_COND",
        help="Run only the listed pairs, e.g. --only fno_add --only fno_film.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if <eval-root>/<pair>/metrics_summary.json exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every resolved command and exit without running.",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the whole grid on the first non-zero exit code.",
    )
    return p.parse_args()


def derive_backbone(run_dir_name: str) -> Optional[str]:
    m = RUN_DIR_RE.match(run_dir_name)
    return m.group(1) if m else None


def derive_cond(cond_dir_name: str) -> Optional[str]:
    key = cond_dir_name.lower().replace("-", "_")
    return COND_ALIASES.get(key)


def discover(delta_root: Path) -> Dict[Tuple[str, str], Path]:
    """Return {(backbone, cond): checkpoint_path} keeping newest mtime per key."""
    found: Dict[Tuple[str, str], Path] = {}
    for ckpt in delta_root.glob("*_full193_*/runs/*/train/models/model_final/model.safetensors"):
        try:
            run_dir_name = ckpt.parents[5].name  # <backbone>_full193_<ts>
            cond_dir_name = ckpt.parents[3].name  # runs/<cond>
        except IndexError:
            continue
        backbone = derive_backbone(run_dir_name)
        cond = derive_cond(cond_dir_name)
        if backbone not in BACKBONES or cond is None:
            continue
        key = (backbone, cond)
        if key not in found or ckpt.stat().st_mtime > found[key].stat().st_mtime:
            found[key] = ckpt
    return found


def apply_overrides(
    found: Dict[Tuple[str, str], Path],
    picks: List[str],
) -> Dict[Tuple[str, str], Path]:
    for raw in picks:
        if "=" not in raw:
            raise SystemExit(f"--pick expects BACKBONE_COND=PATH, got: {raw}")
        key_raw, path_str = raw.split("=", 1)
        parts = key_raw.strip().split("_", 1)
        if len(parts) != 2:
            raise SystemExit(f"--pick key must be '<backbone>_<cond>', got: {key_raw}")
        backbone = parts[0]
        cond = COND_ALIASES.get(parts[1].lower())
        if backbone not in BACKBONES or cond is None:
            raise SystemExit(f"--pick key unrecognized: {key_raw}")
        run_dir = Path(path_str).expanduser()
        ckpt = run_dir / "train" / "models" / "model_final" / "model.safetensors"
        if not ckpt.is_file():
            raise SystemExit(f"--pick: checkpoint not found at {ckpt}")
        found[(backbone, cond)] = ckpt
    return found


def cond_flags(cond: str) -> List[str]:
    if cond == "woion":
        return ["--no-use-ionic-conc"]
    if cond == "film":
        return ["--use-ionic-conc", "--ionic-conditioning", "film"]
    if cond == "add":
        return ["--use-ionic-conc", "--ionic-conditioning", "add"]
    raise ValueError(cond)


def resolve_split_json(ckpt: Path, preferred_name: str) -> Optional[Path]:
    train_dir = ckpt.parents[2]  # .../train/models/model_final -> .../train
    preferred = train_dir / preferred_name
    if preferred.is_file():
        return preferred
    fallback = train_dir / "data_split.json"
    if fallback.is_file():
        return fallback
    return None


def resolve_script(project_root: Path, override: Optional[Path]) -> Path:
    if override is not None:
        if not override.is_file():
            raise SystemExit(f"--visualize-script not found: {override}")
        return override
    primary = project_root / "scripts" / "visualize_test_potential_maps_per_conc.py"
    if primary.is_file():
        return primary
    fallback = project_root / "scripts" / "3d" / "visualize_test_potential_maps_per_conc.py"
    if fallback.is_file():
        return fallback
    raise SystemExit(
        f"visualize_test_potential_maps_per_conc.py not found under {project_root}/scripts/ "
        f"or {project_root}/scripts/3d/. Pass --visualize-script explicitly."
    )


def main() -> None:
    args = parse_args()
    script_path = resolve_script(args.project_root, args.visualize_script)

    found = discover(args.delta_root)
    found = apply_overrides(found, args.pick)

    skip = {s.lower() for s in args.skip}
    only = {o.lower() for o in args.only}

    # Sort so output is deterministic: by backbone then conditioning.
    order = [
        (b, c) for b in ("unet", "resnet", "fno")
        for c in ("woion", "film", "add")
    ]
    pairs = [k for k in order if k in found]
    # Anything discovered that isn't in the canonical 9 still gets run at the end.
    extras = [k for k in found if k not in order]
    pairs.extend(extras)

    if not pairs:
        print(f"[ERR] No checkpoints discovered under {args.delta_root}")
        sys.exit(2)

    print(f"Discovered {len(pairs)} (backbone, cond) pairs:")
    for (b, c) in pairs:
        print(f"  {b}_{c} -> {found[(b, c)]}")
    print()

    exit_code = 0
    for (backbone, cond) in pairs:
        key = f"{backbone}_{cond}"
        if only and key not in only:
            print(f"[SKIP] {key}: not in --only set")
            continue
        if key in skip:
            print(f"[SKIP] {key}: --skip")
            continue

        ckpt = found[(backbone, cond)]
        split_json = resolve_split_json(ckpt, args.split_json_name)
        if split_json is None:
            print(f"[SKIP] {key}: no data_split.json next to {ckpt}")
            continue

        out_dir = args.eval_root / key
        metrics_path = out_dir / "metrics_summary.json"
        if metrics_path.is_file() and not args.overwrite:
            print(f"[SKIP] {key}: metrics_summary.json already exists (use --overwrite to force)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "eval.log"

        cmd = [
            "python", str(script_path),
            "--experiment-name", args.experiment_name,
            "--model-ckpt-path", str(ckpt),
            "--data-split-json-path", str(split_json),
            "--split", args.split,
            "--output-dir", str(out_dir),
            "--max-samples", str(args.max_samples),
            "--device", args.device,
            "--patch-size", str(args.patch_size),
            "--mapper-version", MAPPER_FOR[backbone],
            *cond_flags(cond),
            "--no-figures",
        ]

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = str(args.project_root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

        print("=" * 70)
        print(f"Run {key}")
        print("  ckpt:  ", ckpt)
        print("  split: ", split_json)
        print("  out:   ", out_dir)
        print("  cmd:   ", " ".join(shlex.quote(x) for x in cmd))
        print("=" * 70)

        if args.dry_run:
            continue

        with open(log_path, "w") as logf:
            logf.write("# " + " ".join(shlex.quote(x) for x in cmd) + "\n")
            logf.flush()
            rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)

        if rc != 0:
            print(f"[FAIL] {key} exited with code {rc}. Log: {log_path}")
            exit_code = rc
            if args.stop_on_error:
                sys.exit(rc)
        else:
            print(f"[OK]   {key} -> {metrics_path}")

    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
