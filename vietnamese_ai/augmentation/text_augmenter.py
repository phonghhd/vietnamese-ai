"""TangCuongVanBan - Tăng cường dữ liệu văn bản tiếng Việt."""

import random
from typing import List, Optional

from vietnamese_ai.preprocessing.text import XuLyVanBan


class TangCuongVanBan:
    """
    Tăng cường dữ liệu văn bản tiếng Việt.

    Kỹ thuật:
    - Đổi từ đồng nghĩa (từ điển)
    - Xóa từ ngẫu nhiên
    - Hoán đổi vị trí từ
    - Thêm nhiễu (gõ sai dấu, thiếu dấu)
    - Lặp lại từ

    Sử dụng:
        >>> tc = TangCuongVanBan(seed=42)
        >>> van_ban_moi = tc.tang_cuong("Sản phẩm rất tốt", so_mau=5)
    """

    TU_DONG_NGHIA = {
        "tốt": ["hay", "giỏi", "đẹp", "tuyệt"],
        "xấu": ["kém", "dở", "tệ", "tồi"],
        "lớn": ["to", "khổng_lồ", "vĩ_đại"],
        "nhỏ": ["bé", "tí", "nhỏ_xíu"],
        "nhanh": ["lẹ", "mau", "tốc_độ"],
        "chậm": ["lâu", "ừ_ừ", "rề_rà"],
        "đẹp": ["xinh", "tuyệt_vời", "lung_linh"],
        "vui": ["hạnh_phúc", "sung_sướng", "thú_vị"],
        "buồn": ["chán", "ảm_đạm", "thảm"],
        "ghét": ["không_thích", "chán", "anti"],
        "thích": ["yêu", "mến", "ưa"],
        "hay": ["thú_vị", "hấp_dẫn", "lôi_cuốn"],
        "dở": ["chán", "nhạt", "kém"],
        "mới": ["tân", "fresh", "mới_mẻ"],
        "cũ": ["lâu", "cổ", "xưa"],
        "rất": ["quá", "lắm", "siêu"],
        "sản_phẩm": ["món_hàng", "item"],
        "tuyệt_vời": ["tuyệt_hảo", "hoàn_hảo", "xuất_sắc"],
    }

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._xl = XuLyVanBan()
        if seed is not None:
            random.seed(seed)

    def _thay_tu_dong_nghia(self, text: str) -> str:
        """Thay thế từ bằng từ đồng nghĩa."""
        cac_tu = self._xl.tach_tu(text)
        ket_qua = []
        thay_doi = False

        for tu in cac_tu:
            if tu in self.TU_DONG_NGHIA and not thay_doi:
                tu_moi = random.choice(self.TU_DONG_NGHIA[tu])
                ket_qua.append(tu_moi)
                thay_doi = True
            else:
                ket_qua.append(tu)

        return " ".join(ket_qua)

    def _xoa_tu_ngau_nhien(self, text: str, ty_le: float = 0.1) -> str:
        """Xóa một số từ ngẫu nhiên."""
        cac_tu = self._xl.tach_tu(text)
        so_tu_xoa = max(1, int(len(cac_tu) * ty_le))

        for _ in range(so_tu_xoa):
            if len(cac_tu) > 2:
                idx = random.randint(0, len(cac_tu) - 1)
                cac_tu.pop(idx)

        return " ".join(cac_tu)

    def _hoan_vi_tu(self, text: str) -> str:
        """Hoán đổi vị trí 2 từ ngẫu nhiên."""
        cac_tu = self._xl.tach_tu(text)
        if len(cac_tu) < 2:
            return text

        i, j = random.sample(range(len(cac_tu)), 2)
        cac_tu[i], cac_tu[j] = cac_tu[j], cac_tu[i]
        return " ".join(cac_tu)

    def _them_tu_lap(self, text: str) -> str:
        """Lặp lại một từ ngẫu nhiên (mô phỏng lỗi gõ)."""
        cac_tu = self._xl.tach_tu(text)
        if not cac_tu:
            return text

        idx = random.randint(0, len(cac_tu) - 1)
        cac_tu.insert(idx, cac_tu[idx])
        return " ".join(cac_tu)

    def _thieu_dau(self, text: str) -> str:
        """Mô phỏng thiếu dấu tiếng Việt."""
        from vietnamese_ai.preprocessing.text import XuLyVanBan
        xl = XuLyVanBan()
        return xl._loai_dau(text)

    def tang_cuong(
        self,
        text: str,
        so_mau: int = 5,
        cac_ky_thuat: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Tăng cường văn bản.

        Args:
            text: Văn bản gốc
            so_mau: Số mẫu muốn tạo
            cac_ky_thuat: Danh sách kỹ thuật ('dong_nghia', 'xoa_tu', 'hoan_vi', 'lap_tu', 'thieu_dau')

        Returns:
            Danh sách văn bản mới (bao gồm cả bản gốc)
        """
        if cac_ky_thuat is None:
            cac_ky_thuat = ["dong_nghia", "xoa_tu", "hoan_vi", "lap_tu", "thieu_dau"]

        ky_thuat_map = {
            "dong_nghia": self._thay_tu_dong_nghia,
            "xoa_tu": self._xoa_tu_ngau_nhien,
            "hoan_vi": self._hoan_vi_tu,
            "lap_tu": self._them_tu_lap,
            "thieu_dau": self._thieu_dau,
        }

        ket_qua = [text]  # Luôn giữ bản gốc

        for _ in range(so_mau - 1):
            ky_thuat = random.choice(cac_ky_thuat)
            if ky_thuat in ky_thuat_map:
                van_ban_moi = ky_thuat_map[ky_thuat](text)
                if van_ban_moi != text:
                    ket_qua.append(van_ban_moi)

        return ket_qua

    def tang_cuong_tap_du_lieu(
        self,
        cac_van_ban: List[str],
        cac_nhan: List,
        so_mau_moi_mau: int = 3,
    ) -> tuple:
        """
        Tăng cường toàn bộ tập dữ liệu.

        Args:
            cac_van_ban: Danh sách văn bản
            cac_nhan: Danh sách nhãn
            so_mau_moi_mau: Số mẫu mới cho mỗi mẫu gốc

        Returns:
            (van_ban_moi, nhan_moi)
        """
        van_ban_moi = list(cac_van_ban)
        nhan_moi = list(cac_nhan)

        for vb, nhan in zip(cac_van_ban, cac_nhan):
            mau_moi = self.tang_cuong(vb, so_mau=so_mau_moi_mau)
            for vb_m in mau_moi[1:]:  # Bỏ bản gốc (đã có)
                van_ban_moi.append(vb_m)
                nhan_moi.append(nhan)

        return van_ban_moi, nhan_moi
