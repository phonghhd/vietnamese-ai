"""GanNhanTuLoai - Bộ Gán Nhãn Từ Loại Tiếng Việt (Vietnamese POS Tagger chuẩn nhãn VLSP).

Hỗ trợ hai chế độ linh hoạt:
- Chế độ `underthesea` (nếu cài gói mở rộng `vietnamese-ai[vietnamese-nlp]`): dùng trực tiếp `underthesea.pos_tag`.
- Chế độ lõi tiếng Việt không phụ thuộc nặng: kết hợp `BoTachTuTiengVietSieuToc`, từ điển từ loại chuẩn VLSP
  và luật phân giải ngữ cảnh tiếng Việt (phó từ -> động từ/tính từ, loại từ -> danh từ, danh từ riêng, số từ).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc

MO_TA_NHAN_VLSP: Dict[str, str] = {
    "N": "Danh từ",
    "Np": "Danh từ riêng",
    "Nc": "Danh từ chỉ loại",
    "Nu": "Danh từ đơn vị",
    "V": "Động từ",
    "A": "Tính từ",
    "R": "Phó từ",
    "P": "Đại từ",
    "L": "Định từ",
    "M": "Số từ",
    "E": "Giới từ",
    "C": "Liên từ",
    "T": "Trợ từ / Tiểu từ",
    "CH": "Dấu câu",
}


class GanNhanTuLoai:
    """Bộ gán nhãn từ loại (POS Tagging) tiếng Việt theo chuẩn nhãn VLSP."""

    TU_DIEN_LOAI_TU: Set[str] = {
        "con",
        "cái",
        "chiếc",
        "người",
        "cuốn",
        "quyển",
        "bản",
        "tờ",
        "bức",
        "điều",
        "khoản",
        "mục",
        "chương",
    }

    TU_DIEN_DON_VI: Set[str] = {
        "đồng",
        "vnd",
        "vnđ",
        "triệu",
        "tỷ",
        "nghìn",
        "ngàn",
        "km",
        "mét",
        "kg",
        "giờ",
        "phút",
        "giây",
        "ngày",
        "tháng",
        "năm",
        "phần trăm",
    }

    TU_DIEN_DAI_TU: Set[str] = {
        "tôi",
        "ta",
        "chúng tôi",
        "chúng ta",
        "bạn",
        "anh",
        "chị",
        "ông",
        "bà",
        "họ",
        "nó",
        "mình",
        "này",
        "đó",
        "kia",
        "ấy",
        "gì",
        "ai",
        "đâu",
        "nào",
    }

    TU_DIEN_DINH_TU: Set[str] = {
        "các",
        "những",
        "mọi",
        "mỗi",
        "từng",
        "toàn bộ",
        "tất cả",
        "nhiều",
        "ít",
        "vài",
    }

    TU_DIEN_PHO_TU: Set[str] = {
        "đã",
        "đang",
        "sẽ",
        "vừa",
        "mới",
        "từng",
        "chưa",
        "không",
        "chẳng",
        "rất",
        "quá",
        "lắm",
        "khá",
        "hơi",
        "cũng",
        "vẫn",
        "còn",
        "lại",
        "hãy",
        "đừng",
        "chớ",
        "luôn",
        "thường",
    }

    TU_DIEN_GIOI_TU: Set[str] = {
        "của",
        "tại",
        "ở",
        "trong",
        "ngoài",
        "trên",
        "dưới",
        "về",
        "cho",
        "từ",
        "đến",
        "tới",
        "theo",
        "bởi",
        "bằng",
        "với",
        "qua",
        "sau",
        "trước",
        "giữa",
    }

    TU_DIEN_LIEN_TU: Set[str] = {
        "và",
        "hoặc",
        "hay",
        "nhưng",
        "song",
        "tuy",
        "tuy nhiên",
        "nếu",
        "thì",
        "vì",
        "bởi vì",
        "do",
        "nên",
        "mà",
        "để",
        "khi",
        "như",
    }

    TU_DIEN_TRO_TU: Set[str] = {"là", "được", "bị", "chính", "ngay", "cả", "nhỉ", "nhé", "ạ", "ơi"}

    TU_DIEN_DONG_TU: Set[str] = {
        "ban hành",
        "quy định",
        "thực hiện",
        "thi hành",
        "áp dụng",
        "xử lý",
        "phân tích",
        "nghiên cứu",
        "phát triển",
        "triển khai",
        "xây dựng",
        "cung cấp",
        "hỗ trợ",
        "bảo vệ",
        "kiểm tra",
        "xác thực",
        "đánh giá",
        "ký",
        "gửi",
        "nhận",
        "làm",
        "học",
        "chạy",
        "tạo",
        "sử dụng",
        "tích hợp",
        "tối ưu",
        "huấn luyện",
        "dự đoán",
        "suy luận",
        "truy hồi",
        "hoàn thành",
        "thông qua",
        "có",
        "đi",
        "sống",
        "đọc",
        "viết",
    }

    TU_DIEN_TINH_TU: Set[str] = {
        "tốt",
        "xấu",
        "hay",
        "đẹp",
        "nhanh",
        "chậm",
        "cao",
        "thấp",
        "lớn",
        "nhỏ",
        "mới",
        "cũ",
        "chính xác",
        "an toàn",
        "bảo mật",
        "hiệu quả",
        "ổn định",
        "hợp lệ",
        "hợp pháp",
        "bền vững",
        "quan trọng",
        "chuyên sâu",
        "hiện đại",
        "độc lập",
        "phức tạp",
        "đơn giản",
        "tự nhiên",
        "nghiêm ngặt",
        "minh bạch",
    }

    TU_DIEN_DANH_TU_RIENG: Set[str] = {
        "việt nam",
        "hà nội",
        "hồ chí minh",
        "đà nẵng",
        "hải phòng",
        "cần thơ",
        "huế",
        "phobert",
    }

    _PHO_TU_DONG_TU: Set[str] = {"đã", "đang", "sẽ", "vừa", "hãy", "đừng", "chớ", "chưa"}
    _PHO_TU_TINH_TU: Set[str] = {"rất", "quá", "khá", "hơi", "vô cùng", "cực kỳ"}
    _RE_SO = re.compile(r"^\d+(?:[.,:/-]\d+)*%?$")
    _RE_DAU_CAU = re.compile(r"^[.,!?;:\"'()\[\]{}–—…/]+$")

    def __init__(
        self,
        su_dung_underthesea: bool = True,
        tokenizer: Optional[BoTachTuTiengVietSieuToc] = None,
        tu_dien_bo_sung: Optional[Dict[str, str]] = None,
    ) -> None:
        if tokenizer is None:
            tu_ghep_pos: Set[str] = set()
            for nhom in (
                self.TU_DIEN_DON_VI,
                self.TU_DIEN_DAI_TU,
                self.TU_DIEN_DINH_TU,
                self.TU_DIEN_LIEN_TU,
                self.TU_DIEN_DONG_TU,
                self.TU_DIEN_TINH_TU,
                self.TU_DIEN_DANH_TU_RIENG,
            ):
                for tu in nhom:
                    if " " in tu:
                        tu_ghep_pos.add(tu)
            if tu_dien_bo_sung:
                for k in tu_dien_bo_sung:
                    if " " in k.strip():
                        tu_ghep_pos.add(k.strip().lower())
            self.tokenizer = BoTachTuTiengVietSieuToc(tu_dien_bo_sung=tu_ghep_pos)
        else:
            self.tokenizer = tokenizer
        self._tu_dien_tuy_chinh: Dict[str, str] = {
            k.strip().lower(): v for k, v in (tu_dien_bo_sung or {}).items()
        }
        self._underthesea_pos_tag = None
        if su_dung_underthesea:
            try:
                from underthesea import pos_tag

                self._underthesea_pos_tag = pos_tag
            except ImportError:
                self._underthesea_pos_tag = None

    @property
    def co_underthesea(self) -> bool:
        """Cho biết hệ thống có đang sử dụng `underthesea.pos_tag` hay không."""
        return self._underthesea_pos_tag is not None

    def _gan_nhan_mot_token(self, token: str, vi_tri: int, nhan_truoc: Optional[str], tu_truoc: str) -> str:
        """Phân loại nhãn VLSP cho từng token kết hợp từ điển và ngữ cảnh lân cận."""
        tu_sach = token.replace("_", " ").strip()
        tu_thuong = tu_sach.lower()

        if not tu_sach:
            return "CH"
        if self._RE_DAU_CAU.match(tu_sach):
            return "CH"
        if tu_thuong in self._tu_dien_tuy_chinh:
            return self._tu_dien_tuy_chinh[tu_thuong]
        if self._RE_SO.match(tu_sach):
            return "M"
        if tu_thuong in self.TU_DIEN_DANH_TU_RIENG:
            return "Np"
        if tu_thuong in self.TU_DIEN_GIOI_TU:
            return "E"
        if tu_thuong in self.TU_DIEN_LIEN_TU:
            return "C"
        if tu_thuong in self.TU_DIEN_PHO_TU:
            return "R"
        if tu_thuong in self.TU_DIEN_DINH_TU:
            return "L"
        if tu_thuong in self.TU_DIEN_DAI_TU:
            return "P"
        if tu_thuong in self.TU_DIEN_TRO_TU:
            return "T"
        if tu_thuong in self.TU_DIEN_LOAI_TU:
            return "Nc"
        if tu_thuong in self.TU_DIEN_DON_VI:
            return "Nu"
        if tu_thuong in self.TU_DIEN_DONG_TU:
            return "V"
        if tu_thuong in self.TU_DIEN_TINH_TU:
            return "A"

        if tu_truoc in self._PHO_TU_TINH_TU:
            return "A"
        if tu_truoc in self._PHO_TU_DONG_TU:
            return "V"
        if nhan_truoc in ("Nc", "L", "M"):
            return "N"

        cac_tieng = tu_sach.split()
        if vi_tri > 0 and all(t[0].isupper() for t in cac_tieng if t and t[0].isalpha()):
            return "Np"
        if len(cac_tieng) >= 2 and all(t[0].isupper() for t in cac_tieng if t and t[0].isalpha()):
            return "Np"

        return "N"

    def gan_nhan(self, van_ban: str) -> List[Tuple[str, str]]:
        """Gán nhãn từ loại VLSP cho chuỗi văn bản tiếng Việt, trả về danh sách `(từ, nhãn)`."""
        if not van_ban or not van_ban.strip():
            return []

        if self._underthesea_pos_tag is not None:
            try:
                ket_qua_ut = self._underthesea_pos_tag(van_ban)
                return [(str(tu), str(nhan)) for tu, nhan in ket_qua_ut]
            except Exception:
                pass

        tokens = self.tokenizer.tach_tu(van_ban, noi_tu_ghep=False)
        ket_qua: List[Tuple[str, str]] = []
        nhan_truoc: Optional[str] = None
        tu_truoc = ""
        for idx, tok in enumerate(tokens):
            nhan = self._gan_nhan_mot_token(tok, idx, nhan_truoc, tu_truoc)
            ket_qua.append((tok, nhan))
            nhan_truoc = nhan
            tu_truoc = tok.lower()
        return ket_qua

    def gan_nhan_chi_tiet(self, van_ban: str) -> List[Dict[str, Any]]:
        """Trả về kết quả gán nhãn chi tiết kèm giải thích tiếng Việt cho từng nhãn VLSP."""
        return [
            {
                "tu": tu,
                "nhan": nhan,
                "mo_ta": MO_TA_NHAN_VLSP.get(nhan, "Khác"),
            }
            for tu, nhan in self.gan_nhan(van_ban)
        ]

    def gan_nhan_lo(self, ds_van_ban: List[str]) -> List[List[Tuple[str, str]]]:
        """Gán nhãn từ loại cho một lô nhiều văn bản tiếng Việt."""
        return [self.gan_nhan(vb) for vb in ds_van_ban]

    def thong_ke(self) -> Dict[str, Any]:
        """Trả về thông tin cấu hình và backend hiện tại của bộ gán nhãn từ loại."""
        return {
            "backend": "underthesea" if self.co_underthesea else "native_vlsp_rules",
            "co_underthesea": self.co_underthesea,
            "so_nhan_vlsp": len(MO_TA_NHAN_VLSP),
            "so_tu_tuy_chinh": len(self._tu_dien_tuy_chinh),
        }


VietnamesePOSTagger = GanNhanTuLoai

__all__ = [
    "GanNhanTuLoai",
    "VietnamesePOSTagger",
    "MO_TA_NHAN_VLSP",
]
