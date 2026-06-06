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

**Training (5-fold cross-validation):**
```bash
python train.py
```

`train.py` expects:
- A directory of `.npy` voxel files (one per sample)
- A CSV file with columns `dist`, `current`, and `pore`

## Dataset

The Ion-Curr dataset is available at: https://uofi.box.com/s/60vw1i7hd7x41djg71eul0vnvvkv3m7t 

## Citation

If you use Ion-Curr, please cite the corresponding paper.
