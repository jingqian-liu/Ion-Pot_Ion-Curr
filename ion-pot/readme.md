# Ion-Pot

**Voxel-based CNN surrogates for ion-dependent biomolecular electrostatic potential prediction**

Official code for the Ion-Pot benchmark introduced in
*Ion-Pot and Ion-Curr: Datasets and Benchmarks for Electrostatic Field
and Observable Prediction in Complex Geometries*
(Liu, Gu, Aksimentiev — ICML 2026 AI for Physics workshop,
under submission).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)
[![Paper: ICML 2026 AI4Phys (under submission)](https://img.shields.io/badge/paper-ICML%202026%20AI4Phys%20(under%20submission)-b31b1b.svg)](#citation)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Ion-Pot** is a voxel-based learning benchmark and set of CNN surrogates for predicting the electrostatic potential field around biomolecules under varying ionic conditions. Given a protein's voxelized Van der Waals / charge representation and a scalar salt concentration, the models output the full 3D electrostatic potential that would otherwise be computed by a nonlinear Poisson-Boltzmann solver such as APBS. This repository provides the full pipeline: dataset construction from AlphaFoldDB (3,875 proteins × 9 ionic conditions), voxel data loaders, three backbone architectures (U-Net, ResNet, FNO) with additive or FiLM ion-concentration conditioning, distributed training, and benchmark scripts against APBS.

> Portions of the codebase are adapted from the upstream
> [PBGNN](https://github.com/yxwu21/PBGNN) project by Yongxian Wu
> (MIT license). See [Acknowledgments](#acknowledgments).

## Table of contents

- [Ion-Pot](#ion-pot)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Datasets](#datasets)
    - [Option A — Build from AlphaFoldDB (recommended, fully reproducible)](#option-a--build-from-alphafolddb-recommended-fully-reproducible)
    - [Option B — Download a prebuilt release](#option-b--download-a-prebuilt-release)
  - [Building the Ion-Pot dataset from AlphaFoldDB](#building-the-ion-pot-dataset-from-alphafolddb)
    - [Pipeline stages](#pipeline-stages)
    - [Common knobs](#common-knobs)
  - [Configuration](#configuration)
  - [Quick start](#quick-start)
  - [Training on full-grid `.npz` data](#training-on-full-grid-npz-data)
  - [Repository layout](#repository-layout)
  - [Benchmarks](#benchmarks)
  - [Reproducibility](#reproducibility)
  - [Contributing](#contributing)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

## Installation

We recommend [miniconda](https://docs.conda.io/en/latest/miniconda.html)
to manage Python environments.

```bash
# 1. Clone the repository
git clone https://github.com/pinhaogu/ion-pot.git
cd ion-pot

# 2. Create and activate an environment
conda create -n ion-pot python=3.9 -y
conda activate ion-pot

# 3. Install the package (editable) with dev extras
pip install -e ".[dev]"
```

PBGNN pins `torch==2.2.2`. If you need a different CUDA build, install
PyTorch first following the instructions at
[pytorch.org](https://pytorch.org/get-started/locally/) and then run
`pip install -e ".[dev]"` to pull the remaining dependencies.

## Datasets

The **Ion-Pot** dataset is derived from
[AlphaFoldDB](https://alphafold.ebi.ac.uk/) (v4) and contains
**3,875 monomeric proteins** (molecular mass ≤ 50 kDa, pLDDT ≥ 75)
evaluated at **nine 1:1 monovalent salt concentrations**
(training: 0.00, 0.05, 0.15, 0.45, 0.50 M; interpolation: 0.10, 0.30 M;
extrapolation: 0.70, 1.00 M) using the APBS nonlinear Poisson–Boltzmann
solver on a 96³ voxel grid at 0.5 Å spacing. Structures are clustered
with Foldseek at 50% coverage and split 80/10/10 by cluster
(2,712 / 387 / 776 proteins). See the paper (Appendix A.1) and the
[`dataset/`](dataset) pipeline for full construction details.

There are two ways to get the data:

### Option A — Build from AlphaFoldDB (recommended, fully reproducible)

The primary data source is AlphaFoldDB. The
[`dataset/`](dataset) folder contains a one-shot pipeline that
downloads structures from AlphaFoldDB, applies pLDDT / box-fit quality
filters, clusters with Foldseek, splits by cluster, assigns charges
with PDB2PQR, runs APBS at every requested salt concentration, and
packs the resulting `.dx` volumes into training-ready `.npz` files:

```bash
cd dataset
# Defaults reproduce the paper configuration:
#   SPACING=0.5 Å, DIME=193, FGLEN/CGLEN=96,
#   SALTS="0.000,0.050,0.150,0.450,0.500"
NPROC=16 bash run_pipeline_all.sh
```

See [Building the Ion-Pot dataset from AlphaFoldDB](#building-the-ion-pot-dataset-from-alphafolddb)
below for a full description of each stage and all configurable knobs.

### Option B — Download a prebuilt release

Once the official Ion-Pot release is published, a prebuilt archive of
the `.npz` volumes will be available so the pipeline does not have to
be re-run from scratch:

```bash
mkdir -p datasets
cd datasets
# Replace <RELEASE_URL> with the Ion-Pot release link once published.
wget "<RELEASE_URL>" -O ion_pot.zip
unzip ion_pot.zip
cd ..
```

The `datasets/` directory is git-ignored, so large files stay local.

## Building the Ion-Pot dataset from AlphaFoldDB

The full dataset-construction pipeline lives in [`dataset/`](dataset)
and takes raw AlphaFoldDB structures through pLDDT quality filtering,
box-fit filtering, Foldseek clustering, train/val/test splitting,
PQR generation with PDB2PQR, APBS input generation at multiple salt
concentrations, APBS runs, and finally conversion of APBS `.dx`
outputs to compact `.npz` volumes ready for training.

To run the whole pipeline end-to-end:

```bash
cd dataset
# Override knobs via env vars; defaults are in each step script.
NPROC=16 SPACING=0.5 DIME=193 SALTS="0.000,0.050,0.150,0.450,0.500" \
  bash run_pipeline_all.sh
```

### Pipeline stages

Each stage writes to a numbered directory so outputs are easy to
inspect and intermediate steps can be resumed individually.

| Step | Script | Inputs | Outputs | What it does |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `step1_download.sh` | UniProt / AFDB | `01_structures/`, `01_structures_qc/`, `01_structures_fit97/`, manifests in `08_manifests/` | Download AlphaFoldDB PDBs (`--max-mass 50000`), filter by mean pLDDT (≥ 75 by default), per-residue pLDDT coverage, and APBS box fit. |
| 2 | `step2_foldseek.sh` | `01_structures_fit97/` | `02_foldseek/clusters.tsv` | Build a Foldseek structural database and cluster at 50% target coverage. |
| 3 | `step3_split.sh` | `02_foldseek/clusters.tsv` | `03_splits/{train,val,test}.txt` | Greedy 80/10/10 split by cluster so train/val/test share no structural cluster. |
| 3b | `step3b_filter_splits_by_apbs_fit.sh` *(optional, `RUN_STEP3B=1`)* | splits + APBS geometry | filtered `03_splits/` | Optional extra filter to drop IDs that do not fit the APBS grid. |
| 4 | `step4_make_pqr_parallel.sh` | `01_structures/` + splits | `04_pqr/<id>_pH7.4_CHARMM.pqr` | Assign charges and protonation with PDB2PQR (CHARMM force field, pH 7.4) in parallel via GNU `parallel`. |
| 5 | `step5.sh` | `04_pqr/`, ID lists | `05_apbs_in/{salt...}/...in`, `06_apbs_out/{salt...}/` | Generate APBS `.in` files for each requested salt concentration (`SALTS="0.05,0.15,0.45"`), with fine/coarse grid lengths `FGLEN`/`CGLEN` and grid dimension `DIME`. |
| 6 | `step6.sh` | `05_apbs_in/` | `06_apbs_out/.../{pot,charge,vdw,ndens}/*.dx`, logs in `07_logs/` | Run APBS on every generated input in parallel via `run_apbs_one.sh`. |
| 7 | `dx_to_npz.py` | `06_apbs_out/` | `06_apbs_out_npz/*.npz` | Collapse the four per-ID `.dx` volumes (`pot`, `charge`, `vdw`, `ndens`) into one compressed `.npz` per protein, with `grid_space` and `ionic_conc` metadata inferred from the path. |

After step 7, each `.npz` file contains the full-grid tensors expected
by the voxel data loader (`level_set`, `atom_charge`, `atom_type`,
`atom_mask`, `atom_potential` — see
[Training on full-grid `.npz` data](#training-on-full-grid-npz-data)).
Point the training launcher at
`06_apbs_out_npz/*.npz` (or a symlinked subfolder under `datasets/`) to
train on the freshly built Ion-Pot dataset.

### Common knobs

- `SALTS` — comma-separated list of NaCl concentrations in M, e.g.
  `"0.000,0.050,0.150,0.450,0.500"` to reproduce the paper's five
  training concentrations, or include `0.100,0.300` for interpolation
  and `0.700,1.000` for extrapolation eval.
- `SPACING`, `DIME`, `FGLEN`, `CGLEN` — APBS grid parameters. The
  paper uses a 96³ fine grid at 0.5 Å spacing.
- `PLDDT_MEAN_MIN`, `PLDDT_RES_MIN`, `PLDDT_FRAC_MIN`, `MIN_RESIDUES` —
  AFDB quality gates used in step 1.
- `NPROC` / `JOBS` / `THREADS` — parallelism for download, Foldseek,
  PDB2PQR, and APBS.

Every stage reads and writes text manifests under
`dataset/08_manifests/`, so partial pipelines (e.g. a single salt
concentration or a single split) can be driven by swapping the
`IDS=` environment variable for the ID list you want to process.

## Configuration

Two runtime inputs must be configured before training:

1. **`env.toml`** — your local paths and credentials. Copy the template:

   ```bash
   cp env.example.toml env.toml
   ```

   then fill in the `[project].path`, `[wandb]`, and `[slurm]` sections.
   `env.toml` is git-ignored.

2. **Weights & Biases** — create a project at
   [wandb.ai](https://wandb.ai/) and put the API key / entity / project
   into `env.toml`. Set `WANDB_DISABLED=true` to opt out.

## Quick start

Once you have built the Ion-Pot dataset from AlphaFoldDB (see
[Building the Ion-Pot dataset from AlphaFoldDB](#building-the-ion-pot-dataset-from-alphafolddb))
or downloaded a prebuilt release into `datasets/`, you can evaluate and
train using the bash wrappers in `scripts/launch/`:

```bash
mkdir -p outputs
sh scripts/launch/test.sh       # evaluate with an existing checkpoint
sh scripts/launch/train.sh      # train from scratch on your own split
```

Legacy development checkpoints from the upstream PBGNN project are
kept under `checkpoints/legacy_amber_pbsa/` and
`checkpoints/legacy_pbsmall/` for regression testing; the Ion-Pot paper
results are produced on the AlphaFoldDB-derived dataset built by
`dataset/run_pipeline_all.sh`.

All CLI arguments of a predefined configuration can be inspected with:

```bash
python -m scripts.train_3d_energy_distributed amber_pbsa -h
```

and overridden on the command line:

```bash
python -m scripts.train_3d_energy_distributed amber_pbsa \
  --trainer.train-dataset-extra-config.neighbor-list-cutoff 15 \
  --trainer.eval-dataset-extra-config.neighbor-list-cutoff 15 \
  --trainer.train-num-steps 32000
```

## Training on full-grid `.npz` data

The voxel data loader now supports dense full-grid `.npz` files.

**Required keys per `.npz`:**

- `level_set` — `D,H,W` or `D,H,W,1`
- `atom_charge` — `D,H,W` or `D,H,W,1`
- `atom_type` — `D,H,W` or `D,H,W,1`
- `atom_mask` — `D,H,W` or `D,H,W,1`
- `atom_potential` — `D,H,W` or `D,H,W,1`

**Optional keys:**

- `grid_space` *(recommended; otherwise inferred from path naming)*
- `grid_dims`, `grid_origin`

If you already have sparse files (`*_sparse.pkl.gz`), convert them to
dense grids:

```bash
python -m scripts.convert_sparse_to_full_npz \
  --input-pattern "datasets/processed/new_full_3d_energy_sparse_v2/*/*/*/*_sparse.pkl.gz" \
  --output-root "datasets/processed/new_full_3d_energy_npz" \
  --output-suffix "_full.npz"
```

Then train with a voxel experiment, pointing at the dense files:

```bash
python -m scripts.train_3d_energy_distributed \
  3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_medium \
  --trainer.dataset-path "datasets/processed/new_full_3d_energy_npz/*/*/*/*_full.npz" \
  --trainer.use-sparse-dataset False \
  --trainer.use-full-coverage-sparse-dataset False \
  --trainer.dataset-group-pattern "_full.npz"
```

## Repository layout

```
ion-pot/
├── ion_pot/              # Python package (library code)
│   ├── __init__.py       # exposes __version__
│   ├── data.py, data_aug.py, trainer.py, model.py, utils.py ...
│   ├── nn/               # Neural network backbones (pbgnn, unet, fno, ...)
│   ├── preprocess/       # Preprocessing helpers
│   └── configs/          # Experiment configurations (epb_3d, preprocess)
├── dataset/              # AlphaFoldDB -> .npz construction pipeline
│   ├── run_pipeline_all.sh        # one-shot driver for steps 1-6
│   ├── step1_download.sh          # AFDB download + pLDDT / box filters
│   ├── step2_foldseek.sh          # Foldseek structural clustering
│   ├── step3_split.sh             # 80/10/10 split by cluster
│   ├── step4_make_pqr_parallel.sh # pdb2pqr (CHARMM, pH 7.4)
│   ├── step5.sh                   # APBS input generation per salt
│   ├── step6.sh                   # APBS runs (parallel)
│   ├── dx_to_npz.py               # pack .dx outputs into .npz per ID
│   ├── helpers/, *.py             # helpers (download, filters, splits)
│   └── legacy/                    # archived upstream-PBGNN APBS scripts
├── scripts/              # CLI entry points
│   ├── train_3d_energy_distributed.py
│   ├── test_3d_energy_distributed.py
│   ├── benchmark_apbs.py
│   ├── benchmark_nanopot_models.py
│   ├── launch/                   # bash wrappers (train_*.sh, test.sh, ...)
│   └── viz/                      # plotting + VMD render scripts
│       └── vmd/                  # render_apbs_*.tcl
├── tests/                # Smoke + unit tests
├── benchmark/            # Benchmark result JSON
├── checkpoints/          # Data splits (and, where released, weights)
│   ├── legacy_amber_pbsa/   # upstream-PBGNN dev split (regression only)
│   └── legacy_pbsmall/      # upstream-PBGNN dev split (regression only)
├── notebooks/            # Walkthrough and scratch notebooks
├── assets/               # Figures used by README / paper
├── .github/              # CI workflows, issue + PR templates
├── env.example.toml      # Template for local env.toml
├── pyproject.toml        # Package metadata and lint config
├── reorganize.sh         # One-shot reorg driver (see CONTRIBUTING.md)
└── readme.md             # You are here
```

## Benchmarks

APBS benchmark results are stored in `benchmark/apbs_benchmark.json`.
To reproduce Ion-Pot vs APBS timing and accuracy comparisons on the
AlphaFoldDB-derived dataset:

```bash
python -m scripts.benchmark_apbs
python -m scripts.benchmark_nanopot_models
```

## Reproducibility

- All training launchers under `scripts/3d/bash/` capture `python`
  version, `pip freeze`, the current git commit, any working-tree diff,
  and `nvidia-smi` output into the run's `meta/` directory.
- Dataset splits used in the released checkpoints are saved as
  `data_split.json` alongside the weights under `checkpoints/`.
- Set `WANDB_DISABLED=true` to run without Weights & Biases.

## Contributing

Contributions are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) for the dev environment, style, and
PR process.

## Citation

If you use Ion-Pot in your research, please cite the paper:

```bibtex
@inproceedings{liu2026ionpot,
  title     = {Ion-Pot and Ion-Curr: Datasets and Benchmarks for
               Electrostatic Field and Observable Prediction in
               Complex Geometries},
  author    = {Liu, Jingqian and Gu, Pinhao and Aksimentiev, Aleksei},
  booktitle = {ICML 2026 Workshop on AI for Physics (under submission)},
  year      = {2026},
  note      = {Jingqian Liu and Pinhao Gu contributed equally.}
}
```

and optionally the software release:

```bibtex
@software{ionpot_code_2026,
  author  = {Liu, Jingqian and Gu, Pinhao and Aksimentiev, Aleksei},
  title   = {Ion-Pot: Voxel-based CNN surrogates for ion-dependent
             biomolecular electrostatic potential prediction},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/pinhaogu/ion-pot},
  license = {MIT}
}
```

A machine-readable citation is also provided in [CITATION.cff](CITATION.cff).

## Acknowledgments

Portions of this codebase are adapted from the upstream
[PBGNN](https://github.com/yxwu21/PBGNN) project by Yongxian Wu, released
under the MIT License. We thank the original authors for making their
work openly available. We also acknowledge the APBS and PDB2PQR
communities for the underlying Poisson–Boltzmann solver used to
generate ground-truth potentials.

## License

PBGNN is released under the [MIT License](LICENSE).
