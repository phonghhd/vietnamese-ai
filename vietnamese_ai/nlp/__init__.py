"""NLP tiếng Việt - nhận diện thực thể, hỏi đáp, tóm tắt, dịch, chính tả."""

from vietnamese_ai.nlp.ner import NhanDienThucThe
from vietnamese_ai.nlp.qa import HoiDapTiengViet
from vietnamese_ai.nlp.spelling import KiemTraChinhTa
from vietnamese_ai.nlp.summarization import TomTatVanBan
from vietnamese_ai.nlp.translation import DichThuat

__all__ = [
    "NhanDienThucThe",
    "HoiDapTiengViet",
    "TomTatVanBan",
    "DichThuat",
    "KiemTraChinhTa",
]
