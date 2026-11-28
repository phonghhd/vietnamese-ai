"""KiemTraChinhTa - Spell checking cho tiếng Việt với từ điển tiếng Việt nén ~38.000+ từ."""

import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from vietnamese_ai.nlp.native_tokenizer import tai_tu_dien_tieng_viet
from vietnamese_ai.utils.logger import Logger


def _bo_dau_chuan(tu: str) -> str:
    """Chuyển âm tiết tiếng Việt về chữ thường không dấu ASCII để tra ứng viên sửa chính tả nhanh O(1)."""
    thay = tu.lower().replace("đ", "d")
    nfd = unicodedata.normalize("NFD", thay)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


class KiemTraChinhTa:
    """
    Kiểm tra và sửa lỗi chính tả tiếng Việt.

    Hỗ trợ:
    - Từ điển tiếng Việt nén ~38.000+ từ & âm tiết đóng gói sẵn trong wheel
    - Tra cứu ứng viên không dấu O(1) kết hợp Levenshtein lọc theo độ dài
    - Sửa lỗi gõ phổ biến và học bổ sung từ corpus
    """

    PHIEN_BAN_THAY_THE = {
        "aa": "â",
        "aw": "ă",
        "dd": "đ",
        "ee": "ê",
        "oo": "ô",
        "ow": "ơ",
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
        nap_tu_dien_mac_dinh: bool = True,
    ):
        self.logger = Logger("KiemTraChinhTa")
        self.toi_da_sua = toi_da_sua
        self.nguong_khoang_cach = nguong_khoang_cach

        self._tu_dien: Set[str] = set()
        if nap_tu_dien_mac_dinh:
            self._tu_dien.update(tai_tu_dien_tieng_viet(chi_tu_ghep=False))
        if tu_dien:
            self._tu_dien.update(t.lower() for t in tu_dien)
        self._tu_dien.update(self.LOI_PHO_BIEN.values())
        self._dem_tu: Counter[str] = Counter()
        self._tu_theo_do_dai: Dict[int, List[str]] = {}
        self._khong_dau_map: Dict[str, List[str]] = {}
        self._chi_muc_ban = True

        self._underthesea: Any = None
        if su_dung_underthesea:
            try:
                import underthesea

                self._underthesea = underthesea
            except ImportError:
                pass

    def _dam_bao_chi_muc(self) -> None:
        """Xây dựng lười chỉ mục theo độ dài và dạng không dấu giúp gợi ý sửa lỗi nhanh ngay cả với 38.000+ từ."""
        if not self._chi_muc_ban:
            return
        theo_do_dai: Dict[int, List[str]] = {}
        khong_dau: Dict[str, List[str]] = {}
        for tu in self._tu_dien:
            if " " in tu:
                continue
            theo_do_dai.setdefault(len(tu), []).append(tu)
            kd = _bo_dau_chuan(tu)
            khong_dau.setdefault(kd, []).append(tu)
        self._tu_theo_do_dai = theo_do_dai
        self._khong_dau_map = khong_dau
        self._chi_muc_ban = False

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
                loi.append(
                    {
                        "tu": tu,
                        "vi_tri": i,
                        "goi_y": goi_y,
                    }
                )

                if sua_tu_dong and goi_y:
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
        self._chi_muc_ban = True

    def them_tu(self, *tu: str) -> None:
        """Thêm một hoặc nhiều từ."""
        self._tu_dien.update(t.lower() for t in tu)
        self._chi_muc_ban = True

    def _la_tu_hop_le(self, tu: str) -> bool:
        """Kiểm tra từ có trong từ điển không."""
        if tu in self.LOI_PHO_BIEN and self.LOI_PHO_BIEN[tu] != tu:
            return False
        if tu in self._tu_dien:
            return True

        if self._underthesea:
            try:
                result = self._underthesea.word_tokenize(tu)
                if result:
                    return True
            except Exception:
                pass

        if len(self._tu_dien) < 10:
            return True

        return False

    def _goi_y_sua(self, tu: str) -> List[str]:
        """Gợi ý sửa lỗi nhanh bằng bảng lỗi phổ biến, tra không dấu O(1) và Levenshtein lọc độ dài."""
        goi_y: List[str] = []
        da_co: Set[str] = set()

        if tu in self.LOI_PHO_BIEN:
            chuan = self.LOI_PHO_BIEN[tu]
            goi_y.append(chuan)
            da_co.add(chuan)

        self._dam_bao_chi_muc()
        kd_tu = _bo_dau_chuan(tu)
        for cand in self._khong_dau_map.get(kd_tu, ()):
            if cand not in da_co:
                goi_y.append(cand)
                da_co.add(cand)

        if len(goi_y) < self.toi_da_sua:
            n = len(tu)
            ky_tu_dau = kd_tu[0] if kd_tu else ""
            ung_vien_lev: List[tuple[int, str]] = []
            for do_dai in range(max(2, n - self.nguong_khoang_cach), n + self.nguong_khoang_cach + 1):
                for tu_dien in self._tu_theo_do_dai.get(do_dai, ()):
                    if tu_dien in da_co:
                        continue
                    if ky_tu_dau and _bo_dau_chuan(tu_dien[0]) != ky_tu_dau and abs(do_dai - n) > 1:
                        continue
                    kc = self._khoang_cach_chinh_sua(tu, tu_dien)
                    if kc <= self.nguong_khoang_cach:
                        ung_vien_lev.append((kc, tu_dien))
            ung_vien_lev.sort(key=lambda item: (item[0], item[1]))
            for _, cand in ung_vien_lev:
                goi_y.append(cand)
                da_co.add(cand)
                if len(goi_y) >= self.toi_da_sua * 2:
                    break

        goi_y.sort(key=lambda x: (0 if self.LOI_PHO_BIEN.get(tu) == x else 1, self._khoang_cach_chinh_sua(tu, x)))
        return goi_y[: self.toi_da_sua]

    def _khoang_cach_chinh_sua(self, s1: str, s2: str) -> int:
        """Tính Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._khoang_cach_chinh_sua(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row: List[int] = list(range(len(s2) + 1))
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
                    if self._dem_tu[tu_lower] >= 2:
                        self._tu_dien.add(tu_lower)
                        self._chi_muc_ban = True

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "kich_thuoc_tu_dien": len(self._tu_dien),
            "so_loi_pho_bien": len(self.LOI_PHO_BIEN),
            "so_tu_da_hoc": len(self._dem_tu),
            "co_underthesea": self._underthesea is not None,
        }

    def __repr__(self) -> str:
        return f"KiemTraChinhTa(kich_thuoc_tu_dien={len(self._tu_dien)})"
