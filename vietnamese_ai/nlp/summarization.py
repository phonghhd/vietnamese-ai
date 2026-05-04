"""TomTatVanBan - Text Summarization cho tiếng Việt."""

import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.preprocessing.text import XuLyVanBan
from vietnamese_ai.utils.logger import Logger


class TomTatVanBan:
    """
    Tóm tắt văn bản tiếng Việt.

    Hỗ trợ:
    - Extractive summarization (trích câu quan trọng)
    - TF-IDF based sentence scoring
    - Position-based scoring
    - LLM-based summarization (nếu có generator)

    Sử dụng:
        >>> tom_tat = TomTatVanBan()
        >>> ket_qua = tom_tat.tom_tat(van_ban_dai, so_cau=3)
        >>> print(ket_qua["tom_tat"])
    """

    def __init__(
        self,
        che_do: str = "extractive",
        ham_sinh: Optional[Callable[[str], str]] = None,
        trong_so_vi_tri: float = 0.3,
        trong_so_tfidf: float = 0.5,
        trong_so_do_dai: float = 0.2,
    ):
        if che_do not in ("extractive", "abstractive"):
            raise ValueError("che_do phải là: extractive hoặc abstractive")

        self.che_do = che_do
        self.ham_sinh = ham_sinh
        self.trong_so_vi_tri = trong_so_vi_tri
        self.trong_so_tfidf = trong_so_tfidf
        self.trong_so_do_dai = trong_so_do_dai
        self.logger = Logger("TomTatVanBan")
        self._xl = XuLyVanBan()

    def tom_tat(
        self,
        van_ban: str,
        so_cau: int = 3,
        toi_da_tu: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Tóm tắt văn bản.

        Args:
            van_ban: Văn bản cần tóm tắt
            so_cau: Số câu tóm tắt (cho extractive)
            toi_da_tu: Giới hạn số từ

        Returns:
            {tom_tat, cac_cau_chon, ty_le_nen, goc, che_do}
        """
        if not van_ban or not van_ban.strip():
            return {
                "tom_tat": "",
                "cac_cau_chon": [],
                "ty_le_nen": 0.0,
                "goc": van_ban,
                "che_do": self.che_do,
            }

        if self.che_do == "abstractive" and self.ham_sinh:
            return self._tom_tat_abstractive(van_ban, toi_da_tu)

        return self._tom_tat_extractive(van_ban, so_cau, toi_da_tu)

    def _tom_tat_extractive(
        self,
        van_ban: str,
        so_cau: int,
        toi_da_tu: Optional[int],
    ) -> Dict[str, Any]:
        """Tóm tắt extractive."""
        cac_cau = self._tach_cau(van_ban)
        if not cac_cau:
            return {
                "tom_tat": van_ban,
                "cac_cau_chon": [],
                "ty_le_nen": 1.0,
                "goc": van_ban,
                "che_do": self.che_do,
            }

        if len(cac_cau) <= so_cau:
            return {
                "tom_tat": van_ban,
                "cac_cau_chon": list(range(len(cac_cau))),
                "ty_le_nen": 1.0,
                "goc": van_ban,
                "che_do": self.che_do,
            }

        # Tính điểm cho mỗi câu
        diem_cau = self._tinh_diem_cau(cac_cau, van_ban)

        # Chọn câu có điểm cao nhất, giữ thứ tự gốc
        chi_so_chon = np.argsort(diem_cau)[::-1][:so_cau]
        chi_so_chon = sorted(chi_so_chon)

        # Giới hạn số từ
        if toi_da_tu:
            tong_tu = 0
            chi_so_cuoi = []
            for idx in chi_so_chon:
                so_tu = len(cac_cau[idx].split())
                if tong_tu + so_tu > toi_da_tu:
                    break
                tong_tu += so_tu
                chi_so_cuoi.append(idx)
            chi_so_chon = chi_so_cuoi

        tom_tat = " ".join(cac_cau[i] for i in chi_so_chon)

        return {
            "tom_tat": tom_tat,
            "cac_cau_chon": list(chi_so_chon),
            "ty_le_nen": len(tom_tat) / max(len(van_ban), 1),
            "goc": van_ban,
            "che_do": self.che_do,
        }

    def _tom_tat_abstractive(
        self,
        van_ban: str,
        toi_da_tu: Optional[int],
    ) -> Dict[str, Any]:
        """Tóm tắt abstractive (dùng LLM)."""
        yeu_cau = "Tóm tắt văn bản sau một cách ngắn gọn và súc tích."
        if toi_da_tu:
            yeu_cau += f" Tối đa {toi_da_tu} từ."

        prompt = f"{yeu_cau}\n\nVăn bản:\n{van_ban}\n\nTóm tắt:"

        try:
            tom_tat = self.ham_sinh(prompt)
        except Exception as e:
            self.logger.info(f"Lỗi abstractive, fallback extractive: {e}")
            return self._tom_tat_extractive(van_ban, 3, toi_da_tu)

        return {
            "tom_tat": tom_tat,
            "cac_cau_chon": [],
            "ty_le_nen": len(tom_tat) / max(len(van_ban), 1),
            "goc": van_ban,
            "che_do": "abstractive",
        }

    def _tinh_diem_cau(
        self,
        cac_cau: List[str],
        van_ban: str,
    ) -> np.ndarray:
        """Tính điểm cho mỗi câu."""
        n = len(cac_cau)
        diem = np.zeros(n)

        # TF-IDF scoring
        tu_van_ban = {}
        for cau in cac_cau:
            for tu in self._xl.tach_tu(cau.lower()):
                tu_van_ban[tu] = tu_van_ban.get(tu, 0) + 1

        for i, cau in enumerate(cac_cau):
            tu_cau = self._xl.tach_tu(cau.lower())
            tfidf = sum(
                np.log(tu_van_ban.get(t, 1) + 1)
                for t in tu_cau
                if t in tu_van_ban
            )
            diem[i] += self.trong_so_tfidf * tfidf / max(len(tu_cau), 1)

        # Position scoring (câu đầu và câu giữa thường quan trọng)
        for i in range(n):
            if i == 0:
                vi_tri_diem = 1.0
            elif i < n * 0.2:
                vi_tri_diem = 0.8
            elif i < n * 0.4:
                vi_tri_diem = 0.6
            else:
                vi_tri_diem = 0.3
            diem[i] += self.trong_so_vi_tri * vi_tri_diem

        # Length scoring (câu vừa phải thường tốt hơn)
        do_dai_tb = np.mean([len(c.split()) for c in cac_cau])
        for i, cau in enumerate(cac_cau):
            do_dai = len(cau.split())
            if do_dai_tb > 0:
                ratio = min(do_dai, do_dai_tb) / max(do_dai, do_dai_tb)
                diem[i] += self.trong_so_do_dai * ratio

        return diem

    def _tach_cau(self, van_ban: str) -> List[str]:
        """Tách văn bản thành câu."""
        cau = re.split(r'[.!?\n]+', van_ban)
        return [c.strip() for c in cau if c.strip() and len(c.strip()) > 10]

    def tom_tat_nhieu(
        self,
        van_ban_list: List[str],
        so_cau: int = 3,
    ) -> List[Dict[str, Any]]:
        """Tóm tắt nhiều văn bản."""
        return [self.tom_tat(vb, so_cau) for vb in van_ban_list]

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "che_do": self.che_do,
            "co_llm": self.ham_sinh is not None,
        }

    def __repr__(self) -> str:
        return f"TomTatVanBan(che_do='{self.che_do}')"
