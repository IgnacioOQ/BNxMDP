import pytest


def pytest_collection_modifyitems(config, items):
    """Skip compat tests if the required libraries aren't installed."""
    skip_compat = pytest.mark.skip(reason="compat extras not installed")
    for item in items:
        if "compat" in item.nodeid:
            try:
                import pgmpy  # noqa: F401
                import pymdptoolbox  # noqa: F401
            except ImportError:
                item.add_marker(skip_compat)
