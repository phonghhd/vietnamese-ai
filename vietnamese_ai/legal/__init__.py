"""Namespace Pháp Lý & Hành Chính Việt Nam (`vietnamese_ai.legal`, Lazy PEP 562 — đồng bộ 100% với `vietnamese_ai.rag.legal`)."""

from __future__ import annotations

from vietnamese_ai.rag.legal import _LAZY as _RAG_LEGAL_LAZY
from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = dict(_RAG_LEGAL_LAZY)

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
