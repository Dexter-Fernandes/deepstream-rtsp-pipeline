import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="run GPU-dependent tests (requires NGC DeepStream container with --gpus all)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: requires a physical NVIDIA GPU — run with pytest --gpu inside the NGC container",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip = pytest.mark.skip(reason="pass --gpu to run GPU-dependent tests")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
