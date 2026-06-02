#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters-tsv", required=True, type=str)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="split_files")
    args = ap.parse_args()

    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-6:
        raise SystemExit("Fractions must sum to 1.0")

    # Read: col1 = cluster rep/id, col2 = member id (filename or UniProt-like id)
    clusters = defaultdict(list)
    with open(args.clusters_tsv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            cluster_id, member = parts[0], parts[1]
            clusters[cluster_id].append(member)

    cluster_ids = list(clusters.keys())
    rng = random.Random(args.seed)
    rng.shuffle(cluster_ids)

    total_members = sum(len(v) for v in clusters.values())
    tgt_train = args.train_frac * total_members
    tgt_val = args.val_frac * total_members

    train, val, test = [], [], []
    n_train = n_val = 0

    for cid in cluster_ids:
        members = clusters[cid]
        # Greedy packing by member count
        if n_train + len(members) <= tgt_train or not train:
            train.extend(members); n_train += len(members)
        elif n_val + len(members) <= tgt_val or not val:
            val.extend(members); n_val += len(members)
        else:
            test.extend(members)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, items):
        with open(out / f"{name}.txt", "w") as f:
            for x in sorted(items):
                f.write(x + "\n")

    write("train", train)
    write("val", val)
    write("test", test)

    print("Done.")
    print(f"Clusters: {len(cluster_ids)}")
    print(f"Members total: {total_members}")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"Wrote: {out.resolve()}/train.txt, val.txt, test.txt")


if __name__ == "__main__":
    main()

