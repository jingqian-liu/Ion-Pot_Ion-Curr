#!/usr/bin/env python3
"""
Run inference with a trained GlobalContextPredictor (CurrentCBAMNet13) checkpoint.

Input is either:
  - --input: a directory containing .npy occupancy volumes, or a single .npy file
  - --psf/--pdb: a PSF/PDB structure pair, from which the distance grid is
    computed on the fly (no precomputed .npy needed)

Example:
    python inference.py --input some_dir_of_npy/ --output preds.csv --voltage 0.4
    python inference.py --input some_dir_of_npy/ --output preds.csv --voltage 0.4 \
        --checkpoint checkpoint/model_best.pth
    python inference.py --psf sgG_3g.psf --pdb example/frame0000.pdb \
        --voltage 0.4 --output preds.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import MDAnalysis as mda
from scipy.spatial import cKDTree

from IonCurr import GlobalContextPredictor


def get_radius(atom_type, rules, default=1.5):
    """Return the radius for an atom type using prefix-matching rules
    (list of {"prefix": ..., "radius": ...} dicts)."""
    for rule in rules:
        if atom_type.startswith(rule["prefix"]):
            return rule["radius"]
    return default


def compute_distance_grid(psf_file, pdb_file, str_file="radius_mapping.json",
                           frame=0, exclude="resname TIP3 WAT CLA POT SOD",
                           cap=5.0):
    """Compute the atom-distance grid for a PSF/PDB structure in memory.

    Mirrors occ/pdb_to_distmap.py, minus the argparse/file-writing wrapper.
    """
    with open(str_file, "r") as f:
        radius_mapping = json.load(f)

    universe = mda.Universe(psf_file, pdb_file)
    universe.trajectory[frame]

    # Same fixed box/grid used by the SEM pipeline (1 A spacing).
    x_min, y_min, x_max, y_max = -29.5, -29.5, 29.5, 29.5
    z_min, z_max = -44.5, 44.5

    nx = int(np.ceil(x_max - x_min)) + 1
    ny = int(np.ceil(y_max - y_min)) + 1
    nz = int(np.ceil(z_max - z_min)) + 1

    x_coords = np.arange(x_min, x_min + nx, 1)
    y_coords = np.arange(y_min, y_min + ny, 1)
    z_coords = np.arange(z_min, z_min + nz, 1)

    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    sel = universe.select_atoms(f"not ({exclude})")
    positions = sel.positions

    tree = cKDTree(positions)
    distances, neighbor_indices = tree.query(grid_points)

    neighbor_types = [sel[i].type for i in neighbor_indices]
    radii = [get_radius(at, radius_mapping) for at in neighbor_types]

    distances = distances - radii
    distances = np.maximum(distances, 0.0)
    distances[distances > cap] = cap

    return distances.reshape((nx, ny, nz))


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    model = GlobalContextPredictor(input_nc=1).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def resolve_inputs(input_path: str):
    """Return a list of (name, npy_path) pairs."""
    if os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if f.endswith(".npy"))
        return [(f, os.path.join(input_path, f)) for f in files]

    if input_path.endswith(".npy"):
        return [(os.path.basename(input_path), input_path)]

    raise ValueError(
        f"Unrecognized --input: {input_path!r} "
        "(expected a directory of .npy files or a single .npy file)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",
                        help="A directory of .npy files, or a single .npy file. "
                             "Mutually exclusive with --psf/--pdb.")
    parser.add_argument("--psf", help="Path to a PSF file (used with --pdb).")
    parser.add_argument("--pdb", help="Path to a PDB file (used with --psf).")
    parser.add_argument("--str_file", default="param/radius_mapping.json",
                        help="Path to the JSON file for radius mapping (default: radius_mapping.json). "
                             "Only used with --psf/--pdb.")
    parser.add_argument("--frame", type=int, default=0,
                        help="Trajectory/model frame index to use (default: 0). "
                             "Only used with --psf/--pdb.")
    parser.add_argument("--exclude", default="resname TIP3 WAT CLA POT SOD",
                        help="Atom selection to exclude from the distance calculation "
                             "(default: 'resname TIP3 WAT CLA POT SOD'). Only used with --psf/--pdb.")
    parser.add_argument("--cap", type=float, default=5.0,
                        help="Maximum distance value; larger distances are capped (default: 5.0 A). "
                             "Only used with --psf/--pdb.")
    parser.add_argument("--output", required=True,
                        help="Path to write predictions CSV.")
    parser.add_argument("--checkpoint", default="checkpoint/model_best.pth",
                        help="Path to a model checkpoint (.pth). "
                             "Default: checkpoint/model_best.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--voltage", type=float, required=True,
                        help="Input voltage. Final output is scaled as "
                             "predicted_current / 0.2 * voltage.")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect).")
    args = parser.parse_args()

    if bool(args.psf) != bool(args.pdb):
        sys.exit("[ERROR] --psf and --pdb must be given together.")
    if bool(args.input) == bool(args.psf):
        sys.exit("[ERROR] Provide exactly one of --input or --psf/--pdb.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    print(f"[INFO] Loading model from {args.checkpoint} on {device}")
    model = load_model(args.checkpoint, device)

    if args.psf:
        print(f"[INFO] Computing distance grid from {args.psf} / {args.pdb}")
        grid = compute_distance_grid(
            args.psf, args.pdb, str_file=args.str_file, frame=args.frame,
            exclude=args.exclude, cap=args.cap,
        )
        pairs = [(os.path.basename(args.pdb), grid)]
    else:
        pairs = resolve_inputs(args.input)
        if not pairs:
            sys.exit(f"[ERROR] No .npy inputs found for --input {args.input}")
    print(f"[INFO] Found {len(pairs)} input sample(s)")

    names, preds = [], []
    missing = corrupted = 0

    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            tensors, batch_names = [], []

            for name, item in batch:
                if isinstance(item, np.ndarray):
                    occ = item
                else:
                    path = item
                    if not os.path.exists(path):
                        print(f"[WARN] Missing file, skipping: {path}")
                        missing += 1
                        continue
                    occ = np.load(path)
                if occ.ndim != 3:
                    print(f"[WARN] Wrong shape {occ.shape}, skipping: {name}")
                    corrupted += 1
                    continue
                tensors.append(torch.from_numpy(occ).unsqueeze(0).float())
                batch_names.append(name)

            if not tensors:
                continue

            try:
                x = torch.stack(tensors).to(device)
            except RuntimeError as e:
                sys.exit(f"[ERROR] Inputs in this batch have mismatched shapes "
                         f"(try --batch-size 1): {e}")

            out = model(x).view(-1).cpu().numpy()
            names.extend(batch_names)
            preds.extend(out.tolist())

    if missing or corrupted:
        print(f"[INFO] Skipped {missing} missing and {corrupted} corrupted file(s)")

    if not names:
        sys.exit("[ERROR] No predictions were produced (all inputs missing/corrupted).")

    preds = [p / 0.2 * args.voltage for p in preds]

    out_df = pd.DataFrame({"dist": names, "predicted_current": preds})
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[DONE] Wrote {len(out_df)} prediction(s) to {args.output}")


if __name__ == "__main__":
    main()
