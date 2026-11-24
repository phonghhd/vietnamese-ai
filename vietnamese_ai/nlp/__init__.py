"""Gói Xử Lý Ngôn Ngữ Tự Nhiên Tiếng Việt Chuyên Sâu (`vietnamese_ai.nlp`, Lazy-loaded < 5ms).

Tập hợp lõi NLP tiếng Việt (`BoTachTuTiengVietSieuToc`, `BoNhungVanBanSieuToc`, `XuLyVanBan`,
`GanNhanTuLoai`, `NhanDienThucThe`, `PhoBERTWrapper`, QA, tóm tắt, dịch thuật, chính tả, cảm xúc).
Phụ thuộc nâng cao tùy chọn `underthesea` được quản lý qua extra `pip install vietnamese-ai[vietnamese-nlp]`.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "XuLyVanBan": ("vietnamese_ai.preprocessing.text", "XuLyVanBan"),
    "TangCuongVanBan": ("vietnamese_ai.preprocessing.text_augmentation", "TangCuongVanBan"),
    "BoTachTuTiengVietSieuToc": ("vietnamese_ai.nlp.native_tokenizer", "BoTachTuTiengVietSieuToc"),
    "FastVietnameseTokenizer": ("vietnamese_ai.nlp.native_tokenizer", "FastVietnameseTokenizer"),
    "NativeVietnameseTokenizer": ("vietnamese_ai.nlp.native_tokenizer", "NativeVietnameseTokenizer"),
    "BoNhungVanBanSieuToc": ("vietnamese_ai.nlp.native_tokenizer", "BoNhungVanBanSieuToc"),
    "FastVietnameseEmbedder": ("vietnamese_ai.nlp.native_tokenizer", "FastVietnameseEmbedder"),
    "co_ho_tro_rust": ("vietnamese_ai.nlp.native_tokenizer", "co_ho_tro_rust"),
    "GanNhanTuLoai": ("vietnamese_ai.nlp.pos_tagger", "GanNhanTuLoai"),
    "VietnamesePOSTagger": ("vietnamese_ai.nlp.pos_tagger", "VietnamesePOSTagger"),
    "NhanDienThucThe": ("vietnamese_ai.nlp.ner", "NhanDienThucThe"),
    "HoiDapTiengViet": ("vietnamese_ai.nlp.qa", "HoiDapTiengViet"),
    "TomTatVanBan": ("vietnamese_ai.nlp.summarization", "TomTatVanBan"),
    "DichThuat": ("vietnamese_ai.nlp.translation", "DichThuat"),
    "KiemTraChinhTa": ("vietnamese_ai.nlp.spelling", "KiemTraChinhTa"),
    "PhanTichCamXuc": ("vietnamese_ai.nlp.sentiment", "PhanTichCamXuc"),
    "PhoBERTWrapper": ("vietnamese_ai.nlp.pretrained.phobert", "PhoBERTWrapper"),
}


def kiem_tra_phu_thuoc_nlp() -> Dict[str, Any]:
    """Kiểm tra trạng thái các thư viện NLP tùy chọn (`underthesea`, `transformers`, `torch`, Rust core)."""
    co_underthesea = importlib.util.find_spec("underthesea") is not None
    co_transformers = importlib.util.find_spec("transformers") is not None
    co_torch = importlib.util.find_spec("torch") is not None
    co_rust = importlib.util.find_spec("vietnamese_ai_rust") is not None
    return {
        "underthesea": co_underthesea,
        "transformers": co_transformers,
        "torch": co_torch,
        "rust_core": co_rust,
        "lenh_cai_dat_goi_y": "pip install vietnamese-ai[vietnamese-nlp]",
    }


__all__ = list(_LAZY.keys()) + ["kiem_tra_phu_thuoc_nlp"]
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
