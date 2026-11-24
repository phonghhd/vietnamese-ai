"""Namespace RAG Pháp Lý (`vietnamese_ai.legal.rag`, Lazy PEP 562)."""

from __future__ import annotations

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "BoToiUuNguCanhHanhChinh": ("vietnamese_ai.rag.vietnamese_context", "BoToiUuNguCanhHanhChinh"),
    "VietnameseAdministrativeRAG": ("vietnamese_ai.rag.vietnamese_context", "VietnameseAdministrativeRAG"),
    "BoDocTaiLieuPhapLy": ("vietnamese_ai.rag.legal.document_parser", "BoDocTaiLieuPhapLy"),
    "LegalDocumentParser": ("vietnamese_ai.rag.legal.document_parser", "LegalDocumentParser"),
    "BoMauPromptPhapLy": ("vietnamese_ai.rag.legal.prompts", "BoMauPromptPhapLy"),
    "LegalPromptLibrary": ("vietnamese_ai.rag.legal.prompts", "LegalPromptLibrary"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
