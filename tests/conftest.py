"""Pytest configuration cho Vietnamese AI Framework tests."""

import warnings


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
