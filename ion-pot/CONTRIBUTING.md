# Contributing to Ion-Pot

Thanks for your interest in contributing! This document covers the
workflow we use to keep the codebase healthy.

> **One-time repo reorganization.** If you are looking at a checkout
> that still has `src/`, `configs/`, `scripts/3d/`, top-level
> `train_*.sh`, and `render_*.tcl` at the root, run
> `bash reorganize.sh` once. It renames `src/` → `ion_pot/`, folds
> `configs/` into the package, flattens `scripts/3d/` into `scripts/`,
> moves the top-level shell/TCL launchers into `scripts/launch/` and
> `scripts/viz/vmd/`, archives `scripts/apbs/` under `dataset/legacy/`,
> renames the two legacy checkpoint folders, and sed-sweeps every
> import. It is idempotent, so re-running it after a rebase is safe.

## Development setup

1. Fork and clone the repository.
2. Create a fresh environment (we recommend [miniconda](https://docs.conda.io/en/latest/miniconda.html)):

   ```bash
   conda create -n pbgnn python=3.9
   conda activate pbgnn
   ```

3. Install the package with the development extras:

   ```bash
   pip install -e ".[dev]"
   ```

4. Copy the environment template and fill it in. `env.toml` is git-ignored.

   ```bash
   cp env.example.toml env.toml
   ```

## Code style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Line length is 100. Target Python version is 3.9.
- Run locally before pushing:

  ```bash
  ruff check .
  ruff format .
  ```

## Tests

- Run the test suite with:

  ```bash
  pytest
  ```

- Please add tests for new functionality where practical. Heavy
  integration tests that require GPUs or large datasets should be
  marked and skipped by default.

## Commits and pull requests

- Keep commits focused and with descriptive messages.
- Open a pull request against `main`. Describe what changed, why, and
  how you verified it.
- Link any related issues in the PR description.
- Ensure CI (lint + tests) passes before requesting review.

## Reporting issues

When filing a bug, please include:

- Your OS, Python version, CUDA / PyTorch versions.
- A minimal reproduction (command line, config, dataset pointer).
- The full stack trace or error message.

## Code of conduct

Be kind and constructive. Harassment or disrespectful behavior will
not be tolerated.
