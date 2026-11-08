"""Pytest configuration cho Vietnamese AI Framework tests."""

import warnings

import pytest


@pytest.fixture(autouse=True)
def _reset_torch_default_device():
    """Đảm bảo không rò rỉ meta tensor hoặc default device giữa các test."""
    try:
        import torch

        if hasattr(torch, "set_default_device"):
            try:
                if str(torch.get_default_device()) == "meta":
                    torch.set_default_device(None)
            except Exception:
                pass
    except Exception:
        pass
    yield
    try:
        import torch

        if hasattr(torch, "set_default_device"):
            try:
                if str(torch.get_default_device()) == "meta":
                    torch.set_default_device(None)
            except Exception:
                pass
    except Exception:
        pass


def pytest_configure(config):
    """Suppress pickle security warnings globally."""
    warnings.filterwarnings(
        "ignore",
        message="Đang tải mô hình từ pickle",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Đang tải pipeline từ pickle",
        category=UserWarning,
    )
