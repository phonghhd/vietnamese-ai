"""NhanDienThucThe - Named Entity Recognition cho tiếng Việt."""

import re
from typing import Any, Dict, List, Optional

from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.logger import Logger


class NhanDienThucThe:
    """
    Nhận diện thực thể có tên (NER) cho tiếng Việt.

    Hỗ trợ:
    - Regex-based NER (ngày tháng, số điện thoại, email, URL)
    - Dictionary-based NER (địa danh, tổ chức, người)
    - underthesea NER (nếu có)
    - Custom entity types

    Sử dụng:
        >>> ner = NhanDienThucThe()
        >>> ket_qua = ner.nhan_dien("Nguyễn Văn A sống tại Hà Nội từ 01/01/2024")
        >>> # [{"van_ban": "Nguyễn Văn A", "loai": "PERSON"}, ...]
    """

    MAU_MAC_DINH = {
        "NGAY_THANG": [
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
            r"\d{1,2}\s*tháng\s*\d{1,2}\s*(năm)?\s*\d{4}",
            r"ngày\s+\d{1,2}\s*tháng\s*\d{1,2}\s*năm\s*\d{4}",
        ],
        "SO_DIEN_THOAI": [
            r"(?:\+84|0)[1-9]\d{8}",
            r"(?:\+84|0)\s?\d{2}\s?\d{3}\s?\d{4}",
        ],
        "EMAIL": [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        ],
        "URL": [
            r"https?://[^\s]+",
            r"www\.[^\s]+",
        ],
        "TIEN_TE": [
            r"\d+(?:\.\d{3})*(?:\s*đồng|\s*VND|\s*VNĐ)",
            r"\d+(?:\.\d{3})*\s*(?:triệu|tỷ|nghìn)\s*(?:đồng)?",
        ],
        "DIA_CHI": [
            r"\d+\s+[A-ZÀ-Ỵ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỵ][a-zà-ỹ]+)*(?:\s+đường|\s+phố)",
            r"(?:quận|huyện|thành phố|tỉnh)\s+[A-ZÀ-Ỵ][a-zà-ỹ]+",
        ],
    }

    DIA_DANH = {
        "Hà Nội",
        "TP.HCM",
        "Hồ Chí Minh",
        "Đà Nẵng",
        "Hải Phòng",
        "Cần Thơ",
        "Nha Trang",
        "Huế",
        "Vũng Tàu",
        "Đà Lạt",
        "Việt Nam",
        "Hàn Quốc",
        "Nhật Bản",
        "Mỹ",
        "Trung Quốc",
        "Anh",
        "Pháp",
        "Đức",
        "Singapore",
        "Thái Lan",
    }

    CHUC_DANH = {
        "Giám đốc",
        "Tổng giám đốc",
        "Chủ tịch",
        "Phó giám đốc",
        "Trưởng phòng",
        "Phó phòng",
        "Giáo viên",
        "Bác sĩ",
        "Kỹ sư",
        "Luật sư",
        "Thủ tướng",
        "Chủ tịch nước",
    }

    def __init__(
        self,
        su_dung_underthesea: bool = True,
        mau_tuy_chinh: Optional[Dict[str, List[str]]] = None,
        tu_dien_tuy_chinh: Optional[Dict[str, set]] = None,
    ):
        self.logger = Logger("NhanDienThucThe")
        self.su_dung_underthesea = su_dung_underthesea
        self._xl = XuLyVanBan()
        self._underthesea = None

        if su_dung_underthesea:
            try:
                import underthesea

                self._underthesea = underthesea
            except ImportError:
                self.logger.info("underthesea không khả dụng, dùng regex + dictionary")

        # Compile regex patterns
        self._mau: Dict[str, List[re.Pattern]] = {}
        all_patterns = {**self.MAU_MAC_DINH, **(mau_tuy_chinh or {})}
        for loai, patterns in all_patterns.items():
            self._mau[loai] = [re.compile(p) for p in patterns]

        # Dictionaries
        self._dia_danh = set(self.DIA_DANH)
        self._chuc_danh = set(self.CHUC_DANH)
        if tu_dien_tuy_chinh:
            for loai, words in tu_dien_tuy_chinh.items():
                if loai == "dia_danh":
                    self._dia_danh.update(words)
                elif loai == "chuc_danh":
                    self._chuc_danh.update(words)

    def nhan_dien(
        self,
        van_ban: str,
        loai_loc: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Nhận diện thực thể trong văn bản.

        Args:
            van_ban: Văn bản đầu vào
            loai_loc: Chỉ trả về các loại thực thể này

        Returns:
            [{van_ban, loai, vi_tri_bat_dau, vi_tri_ket_thuc}, ...]
        """
        ket_qua: List[Dict[str, Any]] = []

        # Regex-based NER
        for loai, patterns in self._mau.items():
            if loai_loc and loai not in loai_loc:
                continue
            for mau in patterns:
                for match in mau.finditer(van_ban):
                    ket_qua.append(
                        {
                            "van_ban": match.group(0),
                            "loai": loai,
                            "vi_tri_bat_dau": match.start(),
                            "vi_tri_ket_thuc": match.end(),
                        }
                    )

        # Dictionary-based NER
        if not loai_loc or "DIA_DANH" in loai_loc:
            for dia in self._dia_danh:
                start = 0
                while True:
                    idx = van_ban.find(dia, start)
                    if idx == -1:
                        break
                    ket_qua.append(
                        {
                            "van_ban": dia,
                            "loai": "DIA_DANH",
                            "vi_tri_bat_dau": idx,
                            "vi_tri_ket_thuc": idx + len(dia),
                        }
                    )
                    start = idx + len(dia)

        if not loai_loc or "CHUC_DANH" in loai_loc:
            for chuc in self._chuc_danh:
                start = 0
                while True:
                    idx = van_ban.find(chuc, start)
                    if idx == -1:
                        break
                    ket_qua.append(
                        {
                            "van_ban": chuc,
                            "loai": "CHUC_DANH",
                            "vi_tri_bat_dau": idx,
                            "vi_tri_ket_thuc": idx + len(chuc),
                        }
                    )
                    start = idx + len(chuc)

        # underthesea NER
        if self._underthesea and (not loai_loc or "PERSON" in loai_loc):
            try:
                ner_results = self._underthesea.ner(van_ban)
                for item in ner_results:
                    if len(item) >= 4 and item[3] in ("B-PER", "I-PER"):
                        ket_qua.append(
                            {
                                "van_ban": item[0],
                                "loai": "PERSON",
                                "vi_tri_bat_dau": -1,
                                "vi_tri_ket_thuc": -1,
                            }
                        )
            except Exception:
                pass

        # Sắp xếp theo vị trí
        ket_qua.sort(key=lambda x: x["vi_tri_bat_dau"])

        # Loại bỏ trùng lặp
        return self._loai_trung_lap(ket_qua)

    def them_dia_danh(self, *ten: str) -> None:
        """Thêm địa danh vào từ điển."""
        self._dia_danh.update(ten)

    def them_chuc_danh(self, *ten: str) -> None:
        """Thêm chức danh vào từ điển."""
        self._chuc_danh.update(ten)

    def them_mau(self, loai: str, mau: str) -> None:
        """Thêm pattern regex mới."""
        if loai not in self._mau:
            self._mau[loai] = []
        self._mau[loai].append(re.compile(mau))

    def _loai_trung_lap(self, ket_qua: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Loại bỏ thực thể trùng lặp."""
        if not ket_qua:
            return []

        da_thay = set()
        ket_qua_moi = []

        for item in ket_qua:
            key = (item["van_ban"], item["loai"])
            if key not in da_thay:
                da_thay.add(key)
                ket_qua_moi.append(item)

        return ket_qua_moi

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "so_loai_mau": len(self._mau),
            "so_dia_danh": len(self._dia_danh),
            "so_chuc_danh": len(self._chuc_danh),
            "co_underthesea": self._underthesea is not None,
        }

    def __repr__(self) -> str:
        return f"NhanDienThucThe(so_loai={len(self._mau)})"
