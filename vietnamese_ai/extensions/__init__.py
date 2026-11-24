"""Extensions - Không gian tên tách biệt cho các phân hệ mở rộng tùy chọn (Lazy PEP 562)."""

from __future__ import annotations

import importlib
from typing import Any, List

from vietnamese_ai.utils.feature_status import BAN_DO_MODULAR

DANH_SACH_EXTENSIONS = BAN_DO_MODULAR["extensions"]

_SUBPACKAGES = tuple(DANH_SACH_EXTENSIONS.keys())

__all__ = ["DANH_SACH_EXTENSIONS", *_SUBPACKAGES]


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"vietnamese_ai.extensions.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"Module 'vietnamese_ai.extensions' không có thuộc tính '{name}'")


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
