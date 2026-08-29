# Ion-Curr

**Ion-Curr** is a benchmark for ionic current prediction in nanopore-based biomolecular systems. It is part of a paired testbed for ion-aware biomolecular surrogate modeling introduced alongside Ion-Pot.

## Overview

Ion channels and biological nanopores are nano-scale pores, often formed by proteins, that allow ions and molecules to pass through a confined channel. The measurable current signal in these systems is directly determined by ionic rearrangement and flux, making ionic current a key observable linking biomolecular geometry to electrostatic function.

Ion-Curr contains approximately 80K samples mapping complex nanopore-based biomolecular geometries to ionic current under a reference ionic condition and applied electric bias. Current labels are generated using a FEniCS-based solver built on the Steric Exclusion Method (SEM), whose predictions closely match ionic transport estimated from molecular dynamics simulations.

## Model

We use a lightweight 3D CNN with ~39K parameters (`IonCurr.py`) that takes voxelized nanopore geometry as input and predicts ionic current. Voxel representations focus computation on geometry most relevant to transport — pore shape, constrictions, and analyte configurations — rather than atoms buried inside the biomolecule or surrounding membrane.

**Performance:**
- R² > 0.99
- ~5700× speedup over the FEniCS numerical solver

A key finding is that lightweight models are sufficient for high accuracy on this task, suggesting that for structured observable-level prediction under physically constrained regimes, high model complexity is not necessary.

## Usage

Install dependencies with `pip install -r requirements.txt` (see the file for which packages are needed for inference vs. training only).

**Training (5-fold cross-validation):**
```bash
python train.py
```

`train.py` expects:
- A directory of `.npy` voxel files (one per sample)
- A CSV file with columns `dist`, `current`, and `pore`

**Inference:**
```bash
python inference.py --input some_dir_of_npy/ --output preds.csv --voltage 0.4
python inference.py --input some_dir_of_npy/ --output preds.csv --voltage 0.4 \
    --checkpoint checkpoint/model_best.pth
python inference.py --psf sgG_3g.psf --pdb example/frame0000.pdb \
    --voltage 0.4 --output preds.csv
```

By default, `inference.py` loads the checkpoint from `checkpoint/model_best.pth` and auto-detects CUDA/CPU. See `python inference.py --help` for all options.

## Docker

A `Dockerfile` is provided for running inference in a container. The image bundles the code (`inference.py`, `IonCurr.py`, `param/`).

```bash
# Build (from this directory)
docker build -t ion-curr-inference .

# Run against a PSF/PDB pair
docker run --rm \
  -v "$(pwd)/checkpoint:/app/checkpoint:ro" \
  -v "$(pwd)/example:/app/example:ro" \
  -v /path/to/output_dir:/app/output \
  ion-curr-inference \
  --psf /app/example/CsgG_3g.psf --pdb /app/example/frame0000.pdb \
  --voltage 0.4 --output /app/output/preds.csv \
  --checkpoint /app/checkpoint/model_best.pth

# Or against a directory of .npy files
docker run --rm \
  -v "$(pwd)/checkpoint:/app/checkpoint:ro" \
  -v /path/to/npy_dir:/app/input:ro \
  -v /path/to/output_dir:/app/output \
  ion-curr-inference \
  --input /app/input --voltage 0.4 --output /app/output/preds.csv \
  --checkpoint /app/checkpoint/model_best.pth
```

The image installs CPU-only torch (the model is small enough that CPU inference is fast); swap in a CUDA-enabled base image/torch build and add `--gpus all` if GPU inference is needed instead. If you're on a system with `podman` instead of Docker, substitute `podman` for `docker` in the commands above.

## Dataset

The Ion-Curr dataset is available at: https://uofi.box.com/s/60vw1i7hd7x41djg71eul0vnvvkv3m7t 

## Citation

If you use Ion-Curr, please cite the corresponding paper.
