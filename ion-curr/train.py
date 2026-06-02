#!/usr/bin/env python3
"""
Ion-Curr: 5 fold CV
"""

import multiprocessing
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from typing import Tuple
from tqdm import tqdm
import psutil

from IonCurr import GlobalContextPredictor


class PreloadedDistCurrentDataset(Dataset):
    def __init__(self, npy_dir: str, csv_file: str, verbose: bool = True):
        self.npy_dir = npy_dir

        df = pd.read_csv(csv_file)
        if not {'dist', 'current', 'pore'}.issubset(df.columns):
            raise ValueError("CSV must have columns: 'dist', 'current', and 'pore'.")

        self.inputs = []
        self.targets = []
        self.groups = []

        missing = 0
        corrupted = 0

        if verbose:
            print("=" * 70)
            print(f"Preloading dataset: {len(df)} samples")
            print("=" * 70)

        mem_start = psutil.Process(os.getpid()).memory_info().rss / 1e9
        time_start = time.time()

        iterator = tqdm(enumerate(df.itertuples(index=False)),
                        total=len(df), desc="Loading", disable=not verbose)

        for i, row in iterator:
            name = str(row.dist)
            npy_path = name if (os.sep in name or
                                (name.endswith(".npy") and os.path.isabs(name))) \
                else os.path.join(npy_dir, name)

            if not os.path.exists(npy_path):
                if missing < 5 and verbose:
                    tqdm.write(f"[WARN] Missing: {npy_path}")
                missing += 1
                continue

            try:
                occ = np.load(npy_path)
                if occ.ndim != 3:
                    if verbose:
                        tqdm.write(f"[ERROR] Wrong shape {occ.shape} in {npy_path}")
                    corrupted += 1
                    continue

                x = torch.from_numpy(occ).unsqueeze(0).float()
                y = torch.tensor(float(row.current), dtype=torch.float32)

                self.inputs.append(x)
                self.targets.append(y)
                self.groups.append(row.pore)

            except Exception as e:
                if verbose:
                    tqdm.write(f"[ERROR] Failed to load {npy_path}: {e}")
                corrupted += 1
                continue

        time_end = time.time()
        mem_end = psutil.Process(os.getpid()).memory_info().rss / 1e9

        if verbose:
            print("\n" + "=" * 70)
            print("Loading complete!")
            print("=" * 70)
            print(f"Successfully loaded: {len(self.inputs)}/{len(df)} samples")
            if missing > 0:
                print(f"Missing files: {missing}")
            if corrupted > 0:
                print(f"Corrupted files: {corrupted}")
            print(f"\nPerformance stats:")
            print(f"  Loading time: {time_end - time_start:.2f} sec")
            print(f"  Memory usage: {mem_end - mem_start:.2f} GB")
            print(f"  Average speed: {len(self.inputs) / (time_end - time_start):.1f} samples/sec")
            if self.inputs:
                sample_shape = self.inputs[0].shape
                sample_size_mb = self.inputs[0].numel() * self.inputs[0].element_size() / 1e6
                print(f"\nSample info:")
                print(f"  Shape: {sample_shape}")
                print(f"  Size per sample: {sample_size_mb:.2f} MB")
                print(f"  Current range: [{min(t.item() for t in self.targets):.3f}, "
                      f"{max(t.item() for t in self.targets):.3f}] nA")
            unique_groups = sorted(set(self.groups))
            print(f"\nGroup info:")
            print(f"  {len(unique_groups)} pore types: {unique_groups}")
            print("=" * 70 + "\n")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_prefix: str,
    optimizer: optim.Optimizer,
    scheduler: LambdaLR,
    criterion: nn.Module,
    epochs: int = 120,
    device: str = "cuda",
) -> Tuple[float, float]:
    output_prefix = os.path.abspath(output_prefix)
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    print(f"[INFO] Saving to: {os.path.dirname(output_prefix)}")

    scaler = GradScaler()
    best_val = float('inf')

    log_path = output_prefix + "_loss.txt"

    with open(log_path, "w") as log_file:
        log_file.write("epoch\ttrain_loss\tval_loss\tlr\n")

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0

            train_bar = tqdm(train_loader,
                             desc=f'Epoch {epoch:3d}/{epochs} [Train]',
                             leave=False)

            for x, y in train_bar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).view(-1, 1)

                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    out = model(x)
                    if out.ndim == 1:
                        out = out.view(-1, 1)

                loss = criterion(out, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running += loss.item()
                train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss = running / max(1, len(train_loader))

            model.eval()
            v_running = 0.0

            val_bar = tqdm(val_loader,
                           desc=f'Epoch {epoch:3d}/{epochs} [Val]  ',
                           leave=False)

            with torch.no_grad(), autocast():
                for x, y in val_bar:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True).view(-1, 1)
                    out = model(x)
                    if out.ndim == 1:
                        out = out.view(-1, 1)
                    v_loss = criterion(out, y).item()
                    v_running += v_loss
                    val_bar.set_postfix({'loss': f'{v_loss:.4f}'})

            val_loss = v_running / max(1, len(val_loader))

            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            log_file.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t{current_lr:.2e}\n")
            if epoch % 10 == 0:
                log_file.flush()

            print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = f"{output_prefix}_e{epoch:03d}.pth"
                torch.save(model.state_dict(), ckpt_path)
                torch.save(model.state_dict(), f"{output_prefix}_best.pth")
                print(f"  --> [CKPT] val loss improved to {best_val:.6f}, "
                      f"saved {os.path.basename(ckpt_path)}")

        torch.save(model.state_dict(), f"{output_prefix}_last.pth")

    return float(best_val), float(val_loss)


def make_model(seed: int, in_channels: int = 1, device: str = "cuda") -> nn.Module:
    torch.manual_seed(seed)
    model = GlobalContextPredictor(in_channels).to(device)
    if torch.cuda.device_count() > 1 and device == "cuda":
        model = nn.DataParallel(model)
    return model


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def compute_test_metrics(model: nn.Module, test_loader: DataLoader,
                         device: str) -> dict:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).view(-1)
            out = model(x)
            if out.ndim == 2:
                out = out.view(-1)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    nonzero_mask = np.abs(targets) > 1e-8
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((preds[nonzero_mask] - targets[nonzero_mask])
                                    / targets[nonzero_mask])) * 100.0)
    else:
        mape = float('nan')

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def get_or_create_test_split(out_dir: str, unique_pores: np.ndarray,
                              groups: np.ndarray, test_fraction: float,
                              seed: int) -> tuple:
    test_idx_path   = os.path.join(out_dir, "test_idx.txt")
    test_pores_path = os.path.join(out_dir, "test_pores.txt")

    if os.path.exists(test_idx_path) and os.path.exists(test_pores_path):
        print(f"[TEST SPLIT] Loading existing test split from {out_dir}/")
        test_idx = np.loadtxt(test_idx_path, dtype=int)
        with open(test_pores_path) as f:
            test_pores = [line.strip() for line in f if line.strip()]
        print(f"[TEST SPLIT] Loaded: {len(test_idx)} test samples "
              f"({len(test_pores)} pores): {sorted(test_pores)}")
    else:
        rng = np.random.default_rng(seed)
        n_test_pores = max(1, int(round(len(unique_pores) * test_fraction)))
        test_pores_arr = rng.choice(unique_pores, size=n_test_pores, replace=False)
        test_pores = sorted(test_pores_arr.tolist())

        mask = np.isin(groups, test_pores)
        test_idx = np.where(mask)[0]

        np.savetxt(test_idx_path, test_idx, fmt='%d')
        with open(test_pores_path, "w") as f:
            f.write("\n".join(test_pores) + "\n")

        print(f"[TEST SPLIT] Created: {len(test_idx)} test samples "
              f"({len(test_pores)} pores): {test_pores}")

    test_pore_set = set(test_pores)
    all_idx = np.arange(len(groups))
    trainval_idx = all_idx[~np.isin(groups, list(test_pore_set))]

    return test_idx, trainval_idx


def main():
    print("\n" + "=" * 70)
    print("Ablation Study: No Attention (CurrentCBAMNet13) — Fold 1")
    print("=" * 70 + "\n")

    # ========== Config ==========
    npy_dir       = "IonCurr_dataset"
    csv_file      = "dist_current_labeled.csv"
    out_dir       = "CV_results"
    folds         = 5
    start_fold    = 1
    end_fold      = 1
    test_fraction = 0.1

    epochs       = 90
    lr           = 1e-3
    batch_size   = 64
    seed         = 1777
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    debug_mode   = False

    print(f"Configuration:")
    print(f"  Model          : IonCurr")
    print(f"  Data directory : {npy_dir}")
    print(f"  CSV file       : {csv_file}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Epochs         : {epochs}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Learning rate  : {lr}")
    print(f"  Device         : {device}")
    print(f"  Folds          : {folds}")
    print(f"  Running folds  : {start_fold} to {end_fold}")
    print(f"  Test fraction  : {test_fraction:.0%} of pore labels")
    print(f"  Seed           : {seed}\n")

    os.makedirs(out_dir, exist_ok=True)

    if debug_mode:
        print("Loading dataset (DEBUG: sampling 5000 rows)...")
        _df = pd.read_csv(csv_file)
        _df = _df.sample(n=min(5000, len(_df)), random_state=seed).reset_index(drop=True)
        _debug_csv = "_debug_tmp.csv"
        _df.to_csv(_debug_csv, index=False)
        full_ds = PreloadedDistCurrentDataset(npy_dir, _debug_csv, verbose=True)
    else:
        print("Loading dataset...")
        full_ds = PreloadedDistCurrentDataset(npy_dir, csv_file, verbose=True)

    N = len(full_ds)
    groups       = np.array(full_ds.groups)
    unique_pores = np.unique(groups)

    print(f"Dataset loaded: {N} samples, {len(unique_pores)} pore types\n")

    test_idx, trainval_idx = get_or_create_test_split(
        out_dir, unique_pores, groups, test_fraction, seed
    )

    trainval_pores = np.unique(groups[trainval_idx])
    test_pores_set = set(groups[test_idx])

    assert len(set(trainval_pores) & test_pores_set) == 0, \
        "Test/trainval pore leak detected!"

    print(f"\n[SPLIT SUMMARY]")
    print(f"  Total samples   : {N}")
    print(f"  Test samples    : {len(test_idx)} ({len(test_pores_set)} pores)")
    print(f"  Train+val pool  : {len(trainval_idx)} ({len(trainval_pores)} pores)\n")

    if len(trainval_pores) < folds:
        print(f"[WARNING] Train+val pore types ({len(trainval_pores)}) < folds ({folds}), "
              f"adjusting folds.")
        folds = len(trainval_pores)

    gkf = GroupKFold(n_splits=folds)
    trainval_groups = groups[trainval_idx]

    results = []

    print(f"Starting {folds}-fold cross-validation (ablation: no attention)")
    print("=" * 70 + "\n")

    num_cpus = multiprocessing.cpu_count()
    num_workers = min(4, num_cpus - 1)

    test_loader = DataLoader(
        Subset(full_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    for fold_id, (rel_train_idx, rel_val_idx) in enumerate(
        gkf.split(trainval_idx, groups=trainval_groups), start=1
    ):
        if fold_id < start_fold or fold_id > end_fold:
            print(f"Skipping fold {fold_id}")
            continue

        train_idx = trainval_idx[rel_train_idx]
        val_idx   = trainval_idx[rel_val_idx]

        train_pores = np.unique(groups[train_idx])
        val_pores   = np.unique(groups[val_idx])

        assert len(np.intersect1d(train_pores, val_pores)) == 0, \
            "Train/val pore leak detected!"
        assert len(np.intersect1d(train_pores, list(test_pores_set))) == 0, \
            "Train/test pore leak detected!"
        assert len(np.intersect1d(val_pores, list(test_pores_set))) == 0, \
            "Val/test pore leak detected!"

        print(f"\n{'=' * 70}")
        print(f"Fold {fold_id}/{folds}")
        print(f"{'=' * 70}")
        print(f"[SPLIT] Train: {len(train_idx)} samples ({len(train_pores)} pores)")
        print(f"        Val:   {len(val_idx)} samples ({len(val_pores)} pores)")
        print(f"        Test:  {len(test_idx)} samples ({len(test_pores_set)} pores) [shared]")
        print(f"  Val pores: {sorted(val_pores)}\n")

        fold_dir = os.path.join(out_dir, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        np.savetxt(os.path.join(fold_dir, "train_idx.txt"), train_idx, fmt='%d')
        np.savetxt(os.path.join(fold_dir, "val_idx.txt"),   val_idx,   fmt='%d')
        with open(os.path.join(fold_dir, "val_pores.txt"), "w") as f:
            f.write("\n".join(sorted(val_pores)) + "\n")

        train_loader = DataLoader(
            Subset(full_ds, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            Subset(full_ds, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        print(f"DataLoader: num_workers={num_workers}\n")

        model = make_model(seed, in_channels=1, device=device)
        print(f"[MODEL] {model.__class__.__name__} (no attention) "
              f"with {count_params(model):,} trainable params\n")

        lr_end = 1e-4
        optimizer  = optim.Adam(model.parameters(), lr=lr)
        scheduler  = LambdaLR(
            optimizer,
            lr_lambda=lambda ep: 1.0 - (1.0 - lr_end / lr) * ep / 99
        )
        criterion  = nn.L1Loss()

        prefix = os.path.join(fold_dir, "model")

        best_val, last_val = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_prefix=prefix,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epochs=epochs,
            device=device,
        )

        best_ckpt = f"{prefix}_best.pth"
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

        test_metrics = compute_test_metrics(model, test_loader, device)

        test_metrics_path = os.path.join(fold_dir, "test_metrics.csv")
        pd.DataFrame([{
            "fold": fold_id,
            "MAE":  test_metrics["MAE"],
            "RMSE": test_metrics["RMSE"],
            "MAPE": test_metrics["MAPE"],
            "R2":   test_metrics["R2"],
        }]).to_csv(test_metrics_path, index=False)

        results.append({
            "fold":         fold_id,
            "best_val_L1":  best_val,
            "last_val_L1":  last_val,
            "test_MAE":     test_metrics["MAE"],
            "test_RMSE":    test_metrics["RMSE"],
            "test_MAPE":    test_metrics["MAPE"],
            "test_R2":      test_metrics["R2"],
            "val_pores":    ",".join(sorted(val_pores)),
        })

        print(f"\n[FOLD {fold_id}] best_val_L1={best_val:.6f} | last_val_L1={last_val:.6f}")
        print(f"[FOLD {fold_id}] Test metrics on held-out test set:")
        print(f"             MAE  = {test_metrics['MAE']:.6f}")
        print(f"             RMSE = {test_metrics['RMSE']:.6f}")
        print(f"             MAPE = {test_metrics['MAPE']:.4f}%")
        print(f"             R²   = {test_metrics['R2']:.6f}")

    print("\n" + "=" * 70)
    print("Cross-validation summary")
    print("=" * 70)

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)

    if len(df_res) > 0:
        print(df_res[["fold", "best_val_L1", "last_val_L1",
                       "test_MAE", "test_RMSE", "test_MAPE", "test_R2"]].to_string(index=False))

        best_mean = df_res["best_val_L1"].mean()
        best_std  = df_res["best_val_L1"].std(ddof=1) if len(df_res) > 1 else 0.0
        last_mean = df_res["last_val_L1"].mean()
        last_std  = df_res["last_val_L1"].std(ddof=1) if len(df_res) > 1 else 0.0

        mean_mae  = df_res["test_MAE"].mean()
        std_mae   = df_res["test_MAE"].std(ddof=1) if len(df_res) > 1 else 0.0
        mean_rmse = df_res["test_RMSE"].mean()
        std_rmse  = df_res["test_RMSE"].std(ddof=1) if len(df_res) > 1 else 0.0
        mean_mape = df_res["test_MAPE"].mean()
        std_mape  = df_res["test_MAPE"].std(ddof=1) if len(df_res) > 1 else 0.0
        mean_r2   = df_res["test_R2"].mean()
        std_r2    = df_res["test_R2"].std(ddof=1) if len(df_res) > 1 else 0.0

        print(f"\nBest Val L1   (mean ± std): {best_mean:.6f} ± {best_std:.6f}")
        print(f"Last Val L1   (mean ± std): {last_mean:.6f} ± {last_std:.6f}")
        print(f"Test MAE      (mean ± std): {mean_mae:.6f} ± {std_mae:.6f}")
        print(f"Test RMSE     (mean ± std): {mean_rmse:.6f} ± {std_rmse:.6f}")
        print(f"Test MAPE     (mean ± std): {mean_mape:.4f}% ± {std_mape:.4f}%")
        print(f"Test R²       (mean ± std): {mean_r2:.6f} ± {std_r2:.6f}")
    else:
        print("No folds were run in this invocation.")

    print(f"\n[DONE] Results saved to: {out_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
