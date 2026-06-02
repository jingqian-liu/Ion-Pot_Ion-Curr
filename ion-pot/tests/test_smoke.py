"""Smoke tests: make sure the package imports cleanly after the rename.

These intentionally stay light (no torch, no matplotlib) so that they
can run in a minimal CI job without the full scientific stack.
Submodule import tests that need torch live in ``test_submodules.py``
and are skipped if torch is not installed.
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
