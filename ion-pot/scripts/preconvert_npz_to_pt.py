#!/usr/bin/env python3
"""
Pre-convert .npz files to .pt for faster training data loading.

Instead of reading compressed .npz + parsing + padding + permuting at every
__getitem__, this script does it ONCE and saves ready-to-use torch tensors.

Speedup: ~5-10x faster data loading during training.

Usage:
    # Convert all files (default paths)
    python scripts/preconvert_npz_to_pt.py

    # Custom paths
    python scripts/preconvert_npz_to_pt.py \
        --input-glob "/data/server10/pinhao2/ML/Ion_Prediction/alphafold_v7/06_apbs_out_npz/salt*/*.npz" \
        --output-dir "/data/server10/pinhao2/ML/Ion_Prediction/alphafold_v7/06_apbs_out_pt" \
        --patch-size 193 \
        --num-workers 8

    # On Delta
    python scripts/preconvert_npz_to_pt.py \
        --input-glob "/work/nvme/lhi/gu1/salt*/*.npz" \
        --output-dir "/work/nvme/lhi/gu1/salt_pt" \
        --patch-size 193 \
        --num-workers 16

After conversion, point your training script to the new directory:
    DATASET_GLOB="/path/to/output_dir/**/*.pt" bash train_unet_full_grid.sh
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from ion_pot.data import load_dense_voxel_features


def pad_3d_array(arr: np.ndarray, anchor_D, anchor_H, anchor_W, patch_size):
    """Pad/crop a 3D array (D,H,W,C) to (patch_size, patch_size, patch_size, C)."""
    D, H, W, C = arr.shape
    result = np.zeros((patch_size, patch_size, patch_size, C), dtype=arr.dtype)

    # Source slice
    src_d_start = anchor_D
    src_h_start = anchor_H
    src_w_start = anchor_W
    src_d_end = min(anchor_D + patch_size, D)
    src_h_end = min(anchor_H + patch_size, H)
    src_w_end = min(anchor_W + patch_size, W)

    # Destination slice
    dst_d_len = src_d_end - src_d_start
    dst_h_len = src_h_end - src_h_start
    dst_w_len = src_w_end - src_w_start

    result[:dst_d_len, :dst_h_len, :dst_w_len, :] = arr[
        src_d_start:src_d_end, src_h_start:src_h_end, src_w_start:src_w_end, :
    ]
    return result


def convert_one_file(npz_path: str, output_dir: str, patch_size: int) -> str:
    """Convert a single .npz file to .pt with pre-processed tensors."""
    try:
        # Load and parse the npz file (same logic as the dataset)
        feat_dict, grid_space = load_dense_voxel_features(npz_path)
        ionic_conc = float(feat_dict.get("ionic_conc", 0.0))

        # Center crop (same as training with do_random_crop=False)
        D, H, W, _ = feat_dict["level_set"].shape
        anchor_D = max(D // 2 - patch_size // 2, 0)
        anchor_H = max(H // 2 - patch_size // 2, 0)
        anchor_W = max(W // 2 - patch_size // 2, 0)

        # Pad and convert to CDHW tensors
        dim_inds = (3, 0, 1, 2)
        tensors = {}
        for key in ("level_set", "atom_charge", "atom_type", "atom_mask", "atom_potential"):
            padded = pad_3d_array(feat_dict[key], anchor_D, anchor_H, anchor_W, patch_size)
            tensors[key] = torch.from_numpy(padded).permute(*dim_inds)

        # Grid info
        tensors["grid_info"] = torch.tensor([grid_space, ionic_conc], dtype=torch.float32)

        # Preserve directory structure: salt0p15/file.npz -> salt0p15/file.pt
        rel_path = os.path.basename(os.path.dirname(npz_path))
        filename = os.path.splitext(os.path.basename(npz_path))[0] + ".pt"
        out_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, filename)

        # Save as uncompressed .pt (fast to load)
        torch.save(tensors, out_path)
        return f"OK: {npz_path} -> {out_path}"

    except Exception as e:
        return f"FAIL: {npz_path} — {e}"


def main():
    parser = argparse.ArgumentParser(description="Pre-convert .npz to .pt for faster training")
    parser.add_argument(
        "--input-glob",
        default="/data/server10/pinhao2/ML/Ion_Prediction/alphafold_v7/06_apbs_out_npz/salt*/*.npz",
        help="Glob pattern for input .npz files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to sibling of input dir with '_pt' suffix",
    )
    parser.add_argument("--patch-size", type=int, default=193, help="Patch size for padding")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Just count files, don't convert")
    args = parser.parse_args()

    # Find input files
    npz_files = sorted(glob.glob(args.input_glob))
    if not npz_files:
        print(f"[ERR] No files matched: {args.input_glob}")
        sys.exit(1)

    print(f"Found {len(npz_files)} .npz files")

    # Determine output directory
    if args.output_dir is None:
        # Auto: /path/to/06_apbs_out_npz -> /path/to/06_apbs_out_pt
        first_file = npz_files[0]
        # Go up two levels (past salt*/file.npz)
        base_dir = os.path.dirname(os.path.dirname(first_file))
        dir_name = os.path.basename(base_dir)
        args.output_dir = os.path.join(
            os.path.dirname(base_dir),
            dir_name.replace("_npz", "_pt").replace("npz", "pt") if "npz" in dir_name else dir_name + "_pt",
        )

    print(f"Output directory: {args.output_dir}")
    print(f"Patch size: {args.patch_size}")
    print(f"Workers: {args.num_workers}")

    if args.dry_run:
        print("[DRY RUN] Would convert the above files. Exiting.")
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    # Convert in parallel
    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(convert_one_file, f, args.output_dir, args.patch_size): f
            for f in npz_files
        }
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("FAIL"):
                print(result)
                failed += 1
            else:
                done += 1
                if done % 100 == 0:
                    print(f"  Converted {done}/{len(npz_files)} files...")

    print(f"\nDone: {done} converted, {failed} failed out of {len(npz_files)} total")
    print(f"\nTo use in training:")
    print(f'  DATASET_GLOB="{args.output_dir}/**/*.pt" bash train_unet_full_grid.sh')


if __name__ == "__main__":
    main()