#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# reorganize.sh
# ---------------------------------------------------------------------------
# One-shot reorganization of the ion-pot repository.
#
# Runs six groups of operations:
#   1) Remove tracked junk (__pycache__, env.toml, outputs_*, etc.) from git
#      without deleting the files on disk.
#   2) Move top-level train_*.sh and render_*.tcl into scripts/launch/ and
#      scripts/viz/vmd/.
#   3) Archive scripts/apbs/ under dataset/legacy/.
#   4) Rename checkpoints/amber_pbsa -> checkpoints/legacy_amber_pbsa and
#      checkpoints/pbsmall -> checkpoints/legacy_pbsmall.
#   5) Add tests/ with a smoke test (file created separately by the
#      accompanying Python refactor — here we just ensure the directory
#      is picked up by CI).
#   6) Rename src/ -> ion_pot/, move configs/ into the package, flatten
#      scripts/3d/ -> scripts/, and sed-sweep every 'from src.' / 'import
#      src' / 'from configs.' / 'import configs' import + every
#      'scripts.3d.' reference across Python, shell, notebook, and README
#      files.
#
# The script is idempotent: it skips any step whose result already exists,
# so you can re-run it safely.
#
# After running:
#     git status        # review the renames/removals
#     git diff --stat   # see sed edits
#     python -c "import ion_pot; print(ion_pot.__version__)"
#     pytest -q
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
echo "[reorganize] root = $ROOT"

# Portable in-place sed (GNU vs BSD/macOS).
sed_inplace() {
  if sed --version >/dev/null 2>&1; then
    sed -i "$@"
  else
    sed -i '' "$@"
  fi
}

have_git() { git rev-parse --is-inside-work-tree >/dev/null 2>&1; }

git_mv() {
  local src="$1" dst="$2"
  if [[ ! -e "$src" ]]; then
    echo "  [skip] $src does not exist"
    return 0
  fi
  if [[ -e "$dst" ]]; then
    echo "  [skip] $dst already exists"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  if have_git && git ls-files --error-unmatch "$src" >/dev/null 2>&1; then
    git mv "$src" "$dst"
  else
    mv "$src" "$dst"
  fi
  echo "  [mv]   $src -> $dst"
}

git_rm_cached() {
  local target="$1"
  if have_git && git ls-files --error-unmatch "$target" >/dev/null 2>&1; then
    git rm -r --cached --quiet "$target" || true
    echo "  [untrack] $target"
  fi
}

# ---------------------------------------------------------------------------
# Step 1: cleanup pass — stop tracking junk; leave files on disk.
# ---------------------------------------------------------------------------
echo
echo "[1/6] Cleanup pass (git rm --cached)"
if have_git; then
  git_rm_cached env.toml
  # __pycache__ anywhere in the tree
  while IFS= read -r -d '' d; do
    git_rm_cached "$d"
  done < <(find . -type d -name '__pycache__' -not -path './.git/*' -print0)
  # stale experiment / output dirs
  for d in outputs outputs_wrong_domain outputs_* wandb lightning_logs tb_logs mlruns; do
    if [[ -e "$d" ]]; then git_rm_cached "$d"; fi
  done
  # figure binaries that the gitignore already excludes
  for f in figures/*.png figures/*.pdf figures/*.txt; do
    [[ -e "$f" ]] && git_rm_cached "$f"
  done
else
  echo "  [warn] not a git repo — skipping git-rm pass"
fi

# ---------------------------------------------------------------------------
# Step 2: move top-level launchers and VMD render scripts.
# ---------------------------------------------------------------------------
echo
echo "[2/6] Move top-level train_*.sh and render_*.tcl"
mkdir -p scripts/launch scripts/viz/vmd
for f in train.sh train_both.sh train_resnet.sh train_resnet_full_grid.sh \
         train_unet_full_grid.sh train_fno_full_grid.sh; do
  [[ -f "$f" ]] && git_mv "$f" "scripts/launch/$f"
done
for f in render_apbs_hq.tcl render_apbs_surface.tcl render_same_scale.tcl; do
  [[ -f "$f" ]] && git_mv "$f" "scripts/viz/vmd/$f"
done

# ---------------------------------------------------------------------------
# Step 3: archive scripts/apbs/ under dataset/legacy/.
# ---------------------------------------------------------------------------
echo
echo "[3/6] Archive scripts/apbs/ under dataset/legacy/"
if [[ -d scripts/apbs ]]; then
  mkdir -p dataset/legacy
  git_mv scripts/apbs dataset/legacy/scripts_apbs
fi

# ---------------------------------------------------------------------------
# Step 4: rename checkpoints/amber_pbsa and pbsmall to legacy_*.
# ---------------------------------------------------------------------------
echo
echo "[4/6] Rename legacy checkpoint folders"
[[ -d checkpoints/amber_pbsa ]] && git_mv checkpoints/amber_pbsa checkpoints/legacy_amber_pbsa
[[ -d checkpoints/pbsmall    ]] && git_mv checkpoints/pbsmall    checkpoints/legacy_pbsmall

# ---------------------------------------------------------------------------
# Step 5: ensure tests/ directory is present.
# ---------------------------------------------------------------------------
echo
echo "[5/6] Ensure tests/ directory"
mkdir -p tests
if [[ ! -f tests/__init__.py ]]; then
  : > tests/__init__.py
  echo "  [create] tests/__init__.py"
fi
if [[ ! -f tests/test_smoke.py ]]; then
  cat > tests/test_smoke.py <<'PY'
"""Smoke tests: make sure the package imports cleanly after the rename.

These intentionally stay light (no torch, no matplotlib) so that they
can run in a minimal CI job without the full scientific stack.
"""


def test_import_package():
    import ion_pot  # noqa: F401


def test_version_string():
    from ion_pot.version import VERSION

    assert isinstance(VERSION, str) and len(VERSION) > 0


def test_package_exposes_version():
    import ion_pot

    assert hasattr(ion_pot, "__version__")
    assert ion_pot.__version__ == ion_pot.version.VERSION
PY
  echo "  [create] tests/test_smoke.py"
fi

if [[ ! -f tests/test_submodules.py ]]; then
  cat > tests/test_submodules.py <<'PY'
"""Heavier submodule import tests — skipped if torch is not installed."""

import pytest

torch = pytest.importorskip("torch")


def test_data_imports():
    from ion_pot import data  # noqa: F401


def test_trainer_imports():
    from ion_pot import trainer  # noqa: F401


def test_utils_imports():
    from ion_pot import utils  # noqa: F401


def test_nn_backbones_import():
    from ion_pot.nn import pbgnn, unet  # noqa: F401
PY
  echo "  [create] tests/test_submodules.py"
fi

# ---------------------------------------------------------------------------
# Step 6: rename src/ -> ion_pot/, flatten scripts/3d/, move configs into
#          the package, and sweep every import.
# ---------------------------------------------------------------------------
echo
echo "[6/6] Rename src/ -> ion_pot/ and sweep imports"

# 6a. src/ -> ion_pot/
if [[ -d src && ! -d ion_pot ]]; then
  git_mv src ion_pot
fi

# 6b. configs/ -> ion_pot/configs/
if [[ -d configs && ! -d ion_pot/configs ]]; then
  git_mv configs ion_pot/configs
fi

# 6c. flatten scripts/3d/
if [[ -d scripts/3d ]]; then
  # Move every .py up one level.
  while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    git_mv "$f" "scripts/$base"
  done < <(find scripts/3d -maxdepth 1 -type f -name '*.py' -print0)
  # Move any bash wrappers into scripts/launch/
  if [[ -d scripts/3d/bash ]]; then
    while IFS= read -r -d '' f; do
      base="$(basename "$f")"
      [[ -e "scripts/launch/$base" ]] && { echo "  [skip] scripts/launch/$base exists"; continue; }
      git_mv "$f" "scripts/launch/$base"
    done < <(find scripts/3d/bash -maxdepth 1 -type f -print0)
  fi
  # Clean up empty directories.
  rmdir scripts/3d/bash 2>/dev/null || true
  rmdir scripts/3d      2>/dev/null || true
fi

# 6d. Sweep imports and path references.
#     Target: Python, shell, notebook, config, markdown, TOML.
echo "  [sweep] updating imports and paths"
mapfile -d '' FILES < <(git ls-files -z 2>/dev/null || find . -type f -not -path './.git/*' -print0)

for f in "${FILES[@]}"; do
  # Skip the reorganize script itself and binary-ish files.
  case "$f" in
    ./reorganize.sh|reorganize.sh) continue ;;
    *.png|*.pdf|*.jpg|*.jpeg|*.npz|*.npy|*.pkl|*.gz|*.zip|*.pt|*.pth|*.bin|*.dx|*.pqr|*.pdb|*.cif) continue ;;
  esac
  [[ -f "$f" ]] || continue

  # Python / shell / notebook edits — all done via a single multi-expression sed.
  #
  # IMPORTANT: patterns intentionally avoid \b (GNU-only word boundary).
  # BSD sed on macOS silently treats \b as a literal, which would skip
  # every Python import. Plain substring rewrites are safe here because
  # "from src." / "from configs." only appear in real import statements.
  sed_inplace \
    -e 's|from src\.|from ion_pot.|g' \
    -e 's|^import src$|import ion_pot|g' \
    -e 's|^import src\.|import ion_pot.|g' \
    -e 's| import src$| import ion_pot|g' \
    -e 's|from configs\.|from ion_pot.configs.|g' \
    -e 's|^import configs$|import ion_pot.configs|g' \
    -e 's|^import configs\.|import ion_pot.configs.|g' \
    -e 's|scripts\.3d\.|scripts.|g' \
    -e 's|scripts/3d/bash/|scripts/launch/|g' \
    -e 's|scripts/3d/|scripts/|g' \
    -e 's|checkpoints/amber_pbsa/|checkpoints/legacy_amber_pbsa/|g' \
    -e 's|checkpoints/pbsmall/|checkpoints/legacy_pbsmall/|g' \
    "$f"
done

echo
echo "[done] Reorganization complete."
echo
echo "Next steps:"
echo "  git status                     # review moves + untrackings"
echo "  git diff --stat                # inspect sed-driven import rewrites"
echo "  pip install -e '.[dev]'        # reinstall with the new package name"
echo "  python -c 'import ion_pot; print(ion_pot.__version__)'"
echo "  pytest -q"
echo "  ruff check ion_pot scripts tests"
