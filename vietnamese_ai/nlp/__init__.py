"""NLP tiếng Việt - NER, QA, tóm tắt, dịch, chính tả, cảm xúc, PhoBERT (Lazy-loaded < 5ms)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "NhanDienThucThe": ("vietnamese_ai.nlp.ner", "NhanDienThucThe"),
    "HoiDapTiengViet": ("vietnamese_ai.nlp.qa", "HoiDapTiengViet"),
    "TomTatVanBan": ("vietnamese_ai.nlp.summarization", "TomTatVanBan"),
    "DichThuat": ("vietnamese_ai.nlp.translation", "DichThuat"),
    "KiemTraChinhTa": ("vietnamese_ai.nlp.spelling", "KiemTraChinhTa"),
    "PhanTichCamXuc": ("vietnamese_ai.nlp.sentiment", "PhanTichCamXuc"),
    "PhoBERTWrapper": ("vietnamese_ai.nlp.pretrained.phobert", "PhoBERTWrapper"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
