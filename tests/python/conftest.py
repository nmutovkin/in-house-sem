import sys
from pathlib import Path

# Make the analytic reference module importable from the tests.
sys.path.insert(0, str(Path(__file__).parent))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked slow")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running verification (run with --runslow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    import pytest

    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
