"""Gói RAG Hành Chính & Pháp Lý Việt Nam (`vietnamese_ai.rag.legal`, Lazy PEP 562)."""

from __future__ import annotations

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "BoToiUuNguCanhHanhChinh": ("vietnamese_ai.rag.vietnamese_context", "BoToiUuNguCanhHanhChinh"),
    "VietnameseAdministrativeRAG": ("vietnamese_ai.rag.vietnamese_context", "VietnameseAdministrativeRAG"),
    "BoDocTaiLieuPhapLy": ("vietnamese_ai.rag.legal.document_parser", "BoDocTaiLieuPhapLy"),
    "LegalDocumentParser": ("vietnamese_ai.rag.legal.document_parser", "LegalDocumentParser"),
    "BoMauPromptPhapLy": ("vietnamese_ai.rag.legal.prompts", "BoMauPromptPhapLy"),
    "LegalPromptLibrary": ("vietnamese_ai.rag.legal.prompts", "LegalPromptLibrary"),
    "HoSoHieuLucVanBan": ("vietnamese_ai.rag.legal.validity_tracker", "HoSoHieuLucVanBan"),
    "LegalValidityRecord": ("vietnamese_ai.rag.legal.validity_tracker", "LegalValidityRecord"),
    "CO_SO_HIEU_LUC_PHAP_LUAT_VN": ("vietnamese_ai.rag.legal.validity_tracker", "CO_SO_HIEU_LUC_PHAP_LUAT_VN"),
    "BoTheoDoiHieuLucPhapLy": ("vietnamese_ai.rag.legal.validity_tracker", "BoTheoDoiHieuLucPhapLy"),
    "LegalValidityTracker": ("vietnamese_ai.rag.legal.validity_tracker", "LegalValidityTracker"),
    "tai_bo_phap_ly_mo": ("vietnamese_ai.rag.legal.open_corpus", "tai_bo_phap_ly_mo"),
    "tai_bo_hoi_dap_phap_ly_mo": ("vietnamese_ai.rag.legal.open_corpus", "tai_bo_hoi_dap_phap_ly_mo"),
    "QuanHePhapLy": ("vietnamese_ai.rag.legal.citation_graph", "QuanHePhapLy"),
    "LegalRelationEdge": ("vietnamese_ai.rag.legal.citation_graph", "LegalRelationEdge"),
    "DoThiThamChieuPhapLy": ("vietnamese_ai.rag.legal.citation_graph", "DoThiThamChieuPhapLy"),
    "LegalCitationGraph": ("vietnamese_ai.rag.legal.citation_graph", "LegalCitationGraph"),
    "CanhBaoRuiRoHopDong": ("vietnamese_ai.rag.legal.contract_review", "CanhBaoRuiRoHopDong"),
    "KhacBietDieuKhoan": ("vietnamese_ai.rag.legal.contract_review", "KhacBietDieuKhoan"),
    "BoSoSanhRaSoatHopDong": ("vietnamese_ai.rag.legal.contract_review", "BoSoSanhRaSoatHopDong"),
    "ContractReviewEngine": ("vietnamese_ai.rag.legal.contract_review", "ContractReviewEngine"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
