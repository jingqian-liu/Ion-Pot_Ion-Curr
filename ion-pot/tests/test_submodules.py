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
