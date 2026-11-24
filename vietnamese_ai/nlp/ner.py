"""NhanDienThucThe - Named Entity Recognition chuyên sâu cho tiếng Việt."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc
from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.logger import Logger


class NhanDienThucThe:
    """
    Nhận diện thực thể có tên (NER) chuyên sâu cho tiếng Việt.

    Hỗ trợ:
    - Regex-based NER: ngày tháng, số điện thoại, email, URL, tiền tệ, địa chỉ, CCCD, mã số thuế,
      số hiệu văn bản pháp luật, tên tổ chức/cơ quan nhà nước, tên người tiếng Việt.
    - Dictionary-based NER: địa danh, chức danh, tổ chức/cơ quan phổ biến tại Việt Nam.
    - Tích hợp chuẩn hóa Unicode/dấu thanh qua `BoTachTuTiengVietSieuToc` và `underthesea` NER (tùy chọn).
    - Khử trùng lặp thông minh theo cặp `(van_ban, loai)` và loại bỏ khoảng chồng lấp cùng loại.
    """

    _HO_VIET_NAM = (
        "Nguyễn|Trần|Lê|Phạm|Hoàng|Huỳnh|Phan|Vũ|Võ|Đặng|Bùi|Đỗ|Hồ|Ngô|Dương|Lý|"
        "Đào|Đoàn|Vương|Trịnh|Đinh|Lâm|Phùng|Mai|Tô|Hà|Tạ|Châu|Lương|Trương"
    )

    MAU_MAC_DINH: Dict[str, List[str]] = {
        "NGAY_THANG": [
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
            r"ngày\s+\d{1,2}\s*tháng\s*\d{1,2}\s*năm\s*\d{4}",
            r"\d{1,2}\s*tháng\s*\d{1,2}\s*(?:năm\s*)?\d{4}",
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
            r"(?:quận|huyện|thành phố|tỉnh)\s+[A-ZÀ-Ỵ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỵ][a-zà-ỹ]+)*",
        ],
        "CCCD": [
            r"(?:CCCD|Căn cước công dân|CMND)\s*(?:số|:)?\s*\d{9,12}",
            r"\b0\d{11}\b",
        ],
        "MA_SO_THUE": [
            r"(?:MST|Mã số thuế)\s*(?:số|:)?\s*\d{10}(?:-\d{3})?",
        ],
        "VAN_BAN_PHAP_LUAT": [
            r"(?:Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Chỉ thị|Nghị quyết|Công văn)\s+"
            r"(?:số\s+)?\d+/\d{4}/[A-ZĐ0-9-]+",
        ],
        "TO_CHUC": [
            r"(?:Công ty(?:\s+(?:TNHH|Cổ phần|CP|Trách nhiệm hữu hạn))?|"
            r"Tập đoàn|Ngân hàng(?:\s+(?:TMCP|Thương mại cổ phần|Nhà nước))?|"
            r"Trường\s+Đại học|Viện\s+Nghiên cứu)"
            r"(?:\s+[A-ZÀ-Ỵ0-9][A-Za-zÀ-ỹ0-9&-]*){1,6}",
            r"(?:Ủy ban nhân dân|Hội đồng nhân dân|Sở|Bộ|Tòa án nhân dân|Viện kiểm sát nhân dân)"
            r"(?:\s+(?:tỉnh|thành phố|quận|huyện|phường|xã|TP\.))?"
            r"(?:\s+[A-ZÀ-Ỵ][a-zà-ỹ]+){1,4}",
        ],
        "NGUOI": [
            rf"\b(?:{_HO_VIET_NAM})(?:\s+[A-ZÀ-Ỵ][a-zà-ỹ]+){{1,3}}\b",
        ],
    }

    DIA_DANH: Set[str] = {
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

    CHUC_DANH: Set[str] = {
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
        "Bộ trưởng",
        "Thứ trưởng",
    }

    TO_CHUC_MAC_DINH: Set[str] = {
        "Chính phủ",
        "Quốc hội",
        "Bộ Tư pháp",
        "Bộ Tài chính",
        "Bộ Công an",
        "Bộ Thông tin và Truyền thông",
        "Bộ Giáo dục và Đào tạo",
        "Bộ Y tế",
        "Ngân hàng Nhà nước Việt Nam",
    }

    def __init__(
        self,
        su_dung_underthesea: bool = True,
        mau_tuy_chinh: Optional[Dict[str, List[str]]] = None,
        tu_dien_tuy_chinh: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        self.logger = Logger("NhanDienThucThe")
        self.su_dung_underthesea = su_dung_underthesea
        self._xl = XuLyVanBan()
        self._tokenizer = BoTachTuTiengVietSieuToc()
        self._underthesea = None

        if su_dung_underthesea:
            try:
                import underthesea

                self._underthesea = underthesea
            except ImportError:
                self.logger.info("underthesea không khả dụng, dùng regex + dictionary tiếng Việt")

        self._mau: Dict[str, List[re.Pattern[str]]] = {}
        all_patterns = {**self.MAU_MAC_DINH, **(mau_tuy_chinh or {})}
        for loai, patterns in all_patterns.items():
            self._mau[loai] = [re.compile(p) for p in patterns]

        self._dia_danh: Set[str] = set(self.DIA_DANH)
        self._chuc_danh: Set[str] = set(self.CHUC_DANH)
        self._to_chuc: Set[str] = set(self.TO_CHUC_MAC_DINH)
        if tu_dien_tuy_chinh:
            for loai, words in tu_dien_tuy_chinh.items():
                if loai == "dia_danh":
                    self._dia_danh.update(words)
                elif loai == "chuc_danh":
                    self._chuc_danh.update(words)
                elif loai == "to_chuc":
                    self._to_chuc.update(words)

    def _tim_tu_dien(self, van_ban: str, tap_tu: Set[str], loai: str) -> List[Dict[str, Any]]:
        """Tìm kiếm các cụm từ trong từ điển theo độ dài giảm dần."""
        ket_qua: List[Dict[str, Any]] = []
        for tu in sorted(tap_tu, key=len, reverse=True):
            start = 0
            while True:
                idx = van_ban.find(tu, start)
                if idx == -1:
                    break
                ket_qua.append(
                    {
                        "van_ban": tu,
                        "loai": loai,
                        "vi_tri_bat_dau": idx,
                        "vi_tri_ket_thuc": idx + len(tu),
                    }
                )
                start = idx + len(tu)
        return ket_qua

    def nhan_dien(
        self,
        van_ban: str,
        loai_loc: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Nhận diện thực thể trong văn bản tiếng Việt.

        Args:
            van_ban: Văn bản đầu vào.
            loai_loc: Chỉ trả về các loại thực thể nằm trong danh sách này.

        Returns:
            Danh sách các dict `{van_ban, loai, vi_tri_bat_dau, vi_tri_ket_thuc}`.
        """
        if not van_ban:
            return []
        van_ban_chuan = self._tokenizer.chuan_hoa_dau_thanh(van_ban)
        ket_qua: List[Dict[str, Any]] = []

        for loai, patterns in self._mau.items():
            if loai_loc and loai not in loai_loc:
                continue
            for mau in patterns:
                for match in mau.finditer(van_ban_chuan):
                    ket_qua.append(
                        {
                            "van_ban": match.group(0).strip(),
                            "loai": loai,
                            "vi_tri_bat_dau": match.start(),
                            "vi_tri_ket_thuc": match.end(),
                        }
                    )

        if not loai_loc or "DIA_DANH" in loai_loc:
            ket_qua.extend(self._tim_tu_dien(van_ban_chuan, self._dia_danh, "DIA_DANH"))

        if not loai_loc or "CHUC_DANH" in loai_loc:
            ket_qua.extend(self._tim_tu_dien(van_ban_chuan, self._chuc_danh, "CHUC_DANH"))

        if not loai_loc or "TO_CHUC" in loai_loc:
            ket_qua.extend(self._tim_tu_dien(van_ban_chuan, self._to_chuc, "TO_CHUC"))

        if self._underthesea and (not loai_loc or "PERSON" in loai_loc):
            try:
                ner_results = self._underthesea.ner(van_ban_chuan)
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

        ket_qua.sort(
            key=lambda x: (
                x["vi_tri_bat_dau"],
                -(x["vi_tri_ket_thuc"] - x["vi_tri_bat_dau"]),
            )
        )
        return self._loai_trung_lap(ket_qua)

    def them_dia_danh(self, *ten: str) -> None:
        """Thêm địa danh vào từ điển."""
        self._dia_danh.update(ten)

    def them_chuc_danh(self, *ten: str) -> None:
        """Thêm chức danh vào từ điển."""
        self._chuc_danh.update(ten)

    def them_to_chuc(self, *ten: str) -> None:
        """Thêm tên tổ chức hoặc cơ quan vào từ điển."""
        self._to_chuc.update(ten)

    def them_mau(self, loai: str, mau: str) -> None:
        """Thêm pattern regex mới."""
        if loai not in self._mau:
            self._mau[loai] = []
        self._mau[loai].append(re.compile(mau))

    def _loai_trung_lap(self, ket_qua: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Loại bỏ thực thể trùng lặp và khoảng chồng lấp con cùng loại (ưu tiên span dài hơn)."""
        if not ket_qua:
            return []

        da_thay: Set[tuple[str, str]] = set()
        khoang_theo_loai: Dict[str, List[tuple[int, int]]] = {}
        ket_qua_moi: List[Dict[str, Any]] = []

        for item in ket_qua:
            key = (item["van_ban"], item["loai"])
            if key in da_thay:
                continue
            bat_dau = item["vi_tri_bat_dau"]
            ket_thuc = item["vi_tri_ket_thuc"]
            loai = item["loai"]
            if bat_dau >= 0 and ket_thuc > bat_dau:
                cac_khoang = khoang_theo_loai.setdefault(loai, [])
                bi_chong_lap = any(not (ket_thuc <= s or bat_dau >= e) for s, e in cac_khoang)
                if bi_chong_lap:
                    continue
                cac_khoang.append((bat_dau, ket_thuc))
            da_thay.add(key)
            ket_qua_moi.append(item)

        return ket_qua_moi

    def thong_ke(self) -> Dict[str, Any]:
        """Trả về thống kê số lượng mẫu và từ điển NER hiện có."""
        return {
            "so_loai_mau": len(self._mau),
            "so_dia_danh": len(self._dia_danh),
            "so_chuc_danh": len(self._chuc_danh),
            "so_to_chuc": len(self._to_chuc),
            "co_underthesea": self._underthesea is not None,
        }

    def __repr__(self) -> str:
        return f"NhanDienThucThe(so_loai={len(self._mau)})"
