"""KiemTraChinhTa - Spell checking cho tiếng Việt."""

from collections import Counter
from typing import Any, Dict, List, Optional, Set

from vietnamese_ai.utils.logger import Logger


class KiemTraChinhTa:
    """
    Kiểm tra và sửa lỗi chính tả tiếng Việt.

    Hỗ trợ:
    - Dictionary-based spell checking
    - Edit distance suggestions
    - Tone mark correction
    - Common Vietnamese spelling errors

    Sử dụng:
        >>> kt = KiemTraChinhTa()
        >>> kt.them_tu_dien({"xin chào", "thế giới", "học máy"})
        >>> ket_qua = kt.kiem_tra("xin chao the gioi")
        >>> print(ket_qua["loi"])
    """

    PHIEN_BAN_THAY_THE = {
        "aa": "â", "aw": "ă", "dd": "đ", "ee": "ê", "oo": "ô", "ow": "ơ",
        "uu": "ư",
    }

    LOI_PHO_BIEN = {
        "khong": "không",
        "duoc": "được",
        "cua": "của",
        "nguoi": "người",
        "noi": "nói",
        "lam": "làm",
        "hoc": "học",
        "nha": "nhà",
        "doi": "đời",
        "phai": "phải",
        "muon": "muốn",
        "nhung": "nhưng",
        "dau": "đâu",
        "day": "đây",
        "kia": "kia",
        "nhu": "như",
        "vi": "vì",
        "nen": "nên",
        "dung": "dùng",
        "tai": "tại",
        "hay": "hay",
    }

    def __init__(
        self,
        tu_dien: Optional[Set[str]] = None,
        su_dung_underthesea: bool = True,
        toi_da_sua: int = 5,
        nguong_khoang_cach: int = 2,
    ):
        self.logger = Logger("KiemTraChinhTa")
        self.toi_da_sua = toi_da_sua
        self.nguong_khoang_cach = nguong_khoang_cach

        self._tu_dien: Set[str] = tu_dien or set()
        self._tu_dien.update(self.LOI_PHO_BIEN.values())
        self._dem_tu: Counter = Counter()

        self._underthesea = None
        if su_dung_underthesea:
            try:
                import underthesea
                self._underthesea = underthesea
            except ImportError:
                pass

    def kiem_tra(
        self,
        van_ban: str,
        sua_tu_dong: bool = False,
    ) -> Dict[str, Any]:
        """
        Kiểm tra chính tả văn bản.

        Args:
            van_ban: Văn bản cần kiểm tra
            sua_tu_dong: Tự động sửa lỗi

        Returns:
            {van_ban, da_sua, loi, so_loi, ty_le_loi}
        """
        cac_tu = self._tach_tu(van_ban)
        loi = []
        da_sua = van_ban

        for i, tu in enumerate(cac_tu):
            tu_lower = tu.lower()

            if len(tu_lower) < 2:
                continue

            if not self._la_tu_hop_le(tu_lower):
                goi_y = self._goi_y_sua(tu_lower)
                loi.append({
                    "tu": tu,
                    "vi_tri": i,
                    "goi_y": goi_y,
                })

                if sua_tu_dong and goi_y:
                    # Tìm và thay thế trong van_ban
                    best = goi_y[0]
                    if tu[0].isupper():
                        best = best[0].upper() + best[1:]
                    da_sua = da_sua.replace(tu, best, 1)

        return {
            "van_ban": van_ban,
            "da_sua": da_sua,
            "loi": loi,
            "so_loi": len(loi),
            "ty_le_loi": len(loi) / max(len(cac_tu), 1),
        }

    def sua(self, van_ban: str) -> str:
        """Sửa chính tả tự động."""
        return self.kiem_tra(van_ban, sua_tu_dong=True)["da_sua"]

    def them_tu_dien(self, tu: Set[str]) -> None:
        """Thêm từ vào từ điển."""
        self._tu_dien.update(t.lower() for t in tu)

    def them_tu(self, *tu: str) -> None:
        """Thêm một hoặc nhiều từ."""
        self._tu_dien.update(t.lower() for t in tu)

    def _la_tu_hop_le(self, tu: str) -> bool:
        """Kiểm tra từ có trong từ điển không."""
        if tu in self._tu_dien:
            return True
        if tu in self.LOI_PHO_BIEN:
            return False

        # Kiểm tra với underthesea
        if self._underthesea:
            try:
                result = self._underthesea.word_tokenize(tu)
                if result:
                    return True
            except Exception:
                pass

        # Nếu từ điển trống, chấp nhận mọi từ
        if len(self._tu_dien) < 10:
            return True

        return False

    def _goi_y_sua(self, tu: str) -> List[str]:
        """Gợi ý sửa lỗi."""
        goi_y = []

        # Kiểm tra lỗi phổ biến
        if tu in self.LOI_PHO_BIEN:
            goi_y.append(self.LOI_PHO_BIEN[tu])

        # Edit distance suggestions
        for tu_dien in self._tu_dien:
            kc = self._khoang_cach_chinh_sua(tu, tu_dien)
            if kc <= self.nguong_khoang_cach:
                goi_y.append(tu_dien)

        # Sắp xếp theo khoảng cách
        goi_y.sort(key=lambda x: self._khoang_cach_chinh_sua(tu, x))
        return goi_y[:self.toi_da_sua]

    def _khoang_cach_chinh_sua(self, s1: str, s2: str) -> int:
        """Tính Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._khoang_cach_chinh_sua(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    def _tach_tu(self, van_ban: str) -> List[str]:
        """Tách từ."""
        if self._underthesea:
            try:
                return self._underthesea.word_tokenize(van_ban)
            except Exception:
                pass
        return van_ban.split()

    def huan_luyen_tu_corpus(self, van_ban_list: List[str]) -> None:
        """Học từ điển từ corpus."""
        for vb in van_ban_list:
            cac_tu = self._tach_tu(vb)
            for tu in cac_tu:
                tu_lower = tu.lower()
                if len(tu_lower) >= 2:
                    self._dem_tu[tu_lower] += 1
                    # Thêm từ xuất hiện >= 2 lần vào từ điển
                    if self._dem_tu[tu_lower] >= 2:
                        self._tu_dien.add(tu_lower)

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "kich_thuoc_tu_dien": len(self._tu_dien),
            "so_loi_pho_bien": len(self.LOI_PHO_BIEN),
            "so_tu_da_hoc": len(self._dem_tu),
            "co_underthesea": self._underthesea is not None,
        }

    def __repr__(self) -> str:
        return f"KiemTraChinhTa(kich_thuoc_tu_dien={len(self._tu_dien)})"
