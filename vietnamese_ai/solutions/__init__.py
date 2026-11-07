"""Solutions Module - Bộ giải pháp thực chiến 1-Click tiếng Việt (Lazy PEP 562)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

__all__ = [
    "TroLyPhapLy",
    "LegalAssistant",
    "TrichXuatThongMinh",
    "SmartExtractor",
    "TongDaiCSKH",
    "CustomerSupportBot",
]

__getattr__ = tao_lazy_getattr(
    __name__,
    {
        "TroLyPhapLy": ("vietnamese_ai.solutions.phap_ly", "TroLyPhapLy"),
        "LegalAssistant": ("vietnamese_ai.solutions.phap_ly", "LegalAssistant"),
        "TrichXuatThongMinh": ("vietnamese_ai.solutions.trich_xuat", "TrichXuatThongMinh"),
        "SmartExtractor": ("vietnamese_ai.solutions.trich_xuat", "SmartExtractor"),
        "TongDaiCSKH": ("vietnamese_ai.solutions.cskh", "TongDaiCSKH"),
        "CustomerSupportBot": ("vietnamese_ai.solutions.cskh", "CustomerSupportBot"),
    },
    globals(),
)
