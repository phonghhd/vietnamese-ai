"""HoiDapTiengViet - Question Answering cho tiếng Việt."""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.logger import Logger


class HoiDapTiengViet:
    """
    Hệ thống hỏi đáp (Question Answering) cho tiếng Việt.

    Hỗ trợ:
    - Extractive QA (trích xuất câu trả lời từ context)
    - Keyword-based QA
    - TF-IDF based answer ranking

    Sử dụng:
        >>> qa = HoiDapTiengViet()
        >>> qa.them_tai_lieu("doc1", van_ban)
        >>> ket_qua = qa.hoi("Câu hỏi?")
    """

    def __init__(
        self,
        so_cau_toi_da: int = 5,
        toi_thieu_diem: float = 0.1,
    ):
        self.logger = Logger("HoiDapTiengViet")
        self.so_cau_toi_da = so_cau_toi_da
        self.toi_thieu_diem = toi_thieu_diem
        self._xl = XuLyVanBan()

        self._tai_lieu: Dict[str, List[str]] = {}
        self._cau: List[Dict[str, Any]] = []
        self._tu_dien: Dict[str, int] = {}
        self._df: Dict[str, int] = {}
        self._tong_cau = 0

    def them_tai_lieu(self, ma: str, van_ban: str) -> int:
        """Thêm tài liệu vào knowledge base."""
        cac_cau = self._tach_cau(van_ban)
        self._tai_lieu[ma] = cac_cau

        for i, cau in enumerate(cac_cau):
            tu_set = set(self._xl.tach_tu(cau.lower()))

            self._cau.append({
                "ma": ma,
                "cau": cau,
                "vi_tri": i,
                "tu_set": tu_set,
            })

            for tu in tu_set:
                self._df[tu] = self._df.get(tu, 0) + 1

            self._tong_cau += 1

        return len(cac_cau)

    def hoi(
        self,
        cau_hoi: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Hỏi và tìm câu trả lời.

        Args:
            cau_hoi: Câu hỏi
            top_k: Số kết quả

        Returns:
            {cau_hoi, tra_loi, nguon, diem}
        """
        top_k = top_k or self.so_cau_toi_da

        tu_cau_hoi = set(self._xl.tach_tu(cau_hoi.lower()))

        diem_list = []
        for cau_info in self._cau:
            diem = self._tinh_diem(tu_cau_hoi, cau_info["tu_set"])
            if diem >= self.toi_thieu_diem:
                diem_list.append((diem, cau_info))

        diem_list.sort(key=lambda x: x[0], reverse=True)
        top_results = diem_list[:top_k]

        if not top_results:
            return {
                "cau_hoi": cau_hoi,
                "tra_loi": "Không tìm thấy câu trả lời trong tài liệu.",
                "nguon": [],
                "diem": 0.0,
            }

        tra_loi = top_results[0][1]["cau"]
        nguon = [
            {
                "cau": r[1]["cau"],
                "tai_lieu": r[1]["ma"],
                "diem": r[0],
            }
            for r in top_results
        ]

        return {
            "cau_hoi": cau_hoi,
            "tra_loi": tra_loi,
            "nguon": nguon,
            "diem": top_results[0][0],
        }

    def _tinh_diem(self, tu_cau_hoi: set, tu_cau: set) -> float:
        """Tính điểm liên quan giữa câu hỏi và câu trả lời."""
        if not tu_cau_hoi or not tu_cau:
            return 0.0

        hop = tu_cau_hoi & tu_cau
        if not hop:
            return 0.0

        # TF-IDF-like scoring
        diem = 0.0
        for tu in hop:
            df = self._df.get(tu, 1)
            idf = np.log(self._tong_cau / df + 1)
            diem += idf

        # Normalize
        max_possible = sum(
            np.log(self._tong_cau / self._df.get(t, 1) + 1)
            for t in tu_cau_hoi
        )

        return diem / max(max_possible, 1e-10)

    def _tach_cau(self, van_ban: str) -> List[str]:
        """Tách văn bản thành câu."""
        cau = re.split(r'[.!?\n]+', van_ban)
        return [c.strip() for c in cau if c.strip() and len(c.strip()) > 5]

    def xoa_tai_lieu(self, ma: str) -> bool:
        """Xóa tài liệu."""
        if ma not in self._tai_lieu:
            return False

        del self._tai_lieu[ma]
        self._cau = [c for c in self._cau if c["ma"] != ma]
        self._tong_cau = len(self._cau)
        return True

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "so_tai_lieu": len(self._tai_lieu),
            "so_cau": self._tong_cau,
            "kich_thuoc_tu_dien": len(self._df),
        }

    def __repr__(self) -> str:
        return (
            f"HoiDapTiengViet(so_tai_lieu={len(self._tai_lieu)}, "
            f"so_cau={self._tong_cau})"
        )
