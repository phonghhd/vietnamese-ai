"""CatVanBan - chia văn bản thành các đoạn (chunks) cho RAG."""

from typing import Any, Dict, List, Optional


class CatVanBan:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để đưa vào vector store.

    Hỗ trợ nhiều chiến lược chia:
    - theo câu (cau)
    - theo đoạn (doan)
    - theo số từ (tu)
    - sliding window với overlap

    Sử dụng:
        >>> cat = CatVanBan(kich_thuoc=200, chong_chong=50, chien_luoc="tu")
        >>> cac_doan = cat.chia("Văn bản dài cần chia nhỏ...")
    """

    def __init__(
        self,
        kich_thuoc: int = 200,
        chong_chong: int = 50,
        chien_luoc: str = "tu",
        toi_thieu_kich_thuoc: int = 20,
        bo_qua_khoang_trang: bool = True,
    ):
        if chien_luoc not in ("cau", "doan", "tu", "ky_tu"):
            raise ValueError("chien_luoc phải là: cau, doan, tu, ky_tu")
        if chong_chong >= kich_thuoc:
            raise ValueError("chong_chong phải nhỏ hơn kich_thuoc")

        self.kich_thuoc = kich_thuoc
        self.chong_chong = chong_chong
        self.chien_luoc = chien_luoc
        self.toi_thieu_kich_thuoc = toi_thieu_kich_thuoc
        self.bo_qua_khoang_trang = bo_qua_khoang_trang

    def chia(
        self,
        van_ban: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chia văn bản thành các đoạn.

        Args:
            van_ban: Văn bản đầu vào
            metadata: Metadata chung cho tất cả chunks

        Returns:
            Danh sách [{noi_dung, vi_tri_bat_dau, vi_tri_ket_thuc, metadata}, ...]
        """
        if not van_ban or not van_ban.strip():
            return []

        metadata = metadata or {}

        if self.chien_luoc == "cau":
            return self._chia_theo_cau(van_ban, metadata)
        elif self.chien_luoc == "doan":
            return self._chia_theo_doan(van_ban, metadata)
        elif self.chien_luoc == "tu":
            return self._chia_theo_tu(van_ban, metadata)
        else:
            return self._chia_theo_ky_tu(van_ban, metadata)

    def _chia_theo_cau(
        self, van_ban: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chia theo câu."""
        ket_qua = []
        cac_cau = self._tach_cau(van_ban)
        vi_tri = 0

        for cau in cac_cau:
            cau = cau.strip()
            if not cau:
                vi_tri += len(cau) + 1
                continue

            bat_dau = van_ban.find(cau, vi_tri)
            if bat_dau == -1:
                bat_dau = vi_tri

            ket_qua.append({
                "noi_dung": cau,
                "vi_tri_bat_dau": bat_dau,
                "vi_tri_ket_thuc": bat_dau + len(cau),
                "metadata": {**metadata, "loai": "cau"},
            })
            vi_tri = bat_dau + len(cau)

        return self._gop_chunks_nho(ket_qua)

    def _chia_theo_doan(
        self, van_ban: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chia theo đoạn (double newline)."""
        ket_qua = []
        cac_doan = van_ban.split("\n\n")
        vi_tri = 0

        for doan in cac_doan:
            doan = doan.strip()
            if not doan:
                vi_tri += len(doan) + 2
                continue

            bat_dau = van_ban.find(doan, vi_tri)
            if bat_dau == -1:
                bat_dau = vi_tri

            if len(doan.split()) > self.kich_thuoc:
                sub_chunks = self._chia_theo_tu(doan, metadata, bat_dau_offset=bat_dau)
                ket_qua.extend(sub_chunks)
            else:
                ket_qua.append({
                    "noi_dung": doan,
                    "vi_tri_bat_dau": bat_dau,
                    "vi_tri_ket_thuc": bat_dau + len(doan),
                    "metadata": {**metadata, "loai": "doan"},
                })
            vi_tri = bat_dau + len(doan)

        return ket_qua

    def _chia_theo_tu(
        self,
        van_ban: str,
        metadata: Dict[str, Any],
        bat_dau_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Chia theo số từ với sliding window."""
        ket_qua = []
        cac_tu = van_ban.split()
        if not cac_tu:
            return []

        buoc = self.kich_thuoc - self.chong_chong
        i = 0

        while i < len(cac_tu):
            ket_thuc = min(i + self.kich_thuoc, len(cac_tu))
            chunk_tu = cac_tu[i:ket_thuc]
            noi_dung = " ".join(chunk_tu)

            if len(chunk_tu) >= self.toi_thieu_kich_thuoc:
                # Tính vị trí trong văn bản gốc
                truoc = " ".join(cac_tu[:i])
                bat_dau = bat_dau_offset + len(truoc) + (1 if truoc else 0)

                ket_qua.append({
                    "noi_dung": noi_dung,
                    "vi_tri_bat_dau": bat_dau,
                    "vi_tri_ket_thuc": bat_dau + len(noi_dung),
                    "metadata": {**metadata, "loai": "tu", "so_tu": len(chunk_tu)},
                })

            if ket_thuc >= len(cac_tu):
                break
            i += buoc

        return ket_qua

    def _chia_theo_ky_tu(
        self, van_ban: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chia theo số ký tự."""
        ket_qua = []
        buoc = self.kich_thuoc - self.chong_chong
        i = 0

        while i < len(van_ban):
            ket_thuc = min(i + self.kich_thuoc, len(van_ban))
            noi_dung = van_ban[i:ket_thuc]

            if self.bo_qua_khoang_trang:
                noi_dung = noi_dung.strip()

            if len(noi_dung) >= self.toi_thieu_kich_thuoc:
                ket_qua.append({
                    "noi_dung": noi_dung,
                    "vi_tri_bat_dau": i,
                    "vi_tri_ket_thuc": ket_thuc,
                    "metadata": {**metadata, "loai": "ky_tu", "so_ky_tu": len(noi_dung)},
                })

            if ket_thuc >= len(van_ban):
                break
            i += buoc

        return ket_qua

    def _tach_cau(self, van_ban: str) -> List[str]:
        """Tách văn bản thành các câu."""
        ket_qua = []
        hien_tai = []

        for ky_tu in van_ban:
            hien_tai.append(ky_tu)
            if ky_tu in ".!?\n":
                text = "".join(hien_tai).strip()
                if text:
                    ket_qua.append(text)
                hien_tai = []

        if hien_tai:
            text = "".join(hien_tai).strip()
            if text:
                ket_qua.append(text)

        return ket_qua

    def _gop_chunks_nho(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Gộp các chunk quá nhỏ lại."""
        if not chunks:
            return []

        ket_qua = []
        gop = []

        for chunk in chunks:
            gop.append(chunk)
            tong = sum(len(c["noi_dung"]) for c in gop)

            if tong >= self.kich_thuoc or chunk == chunks[-1]:
                noi_dung = " ".join(c["noi_dung"] for c in gop)
                if len(noi_dung.split()) >= self.toi_thieu_kich_thuoc:
                    ket_qua.append({
                        "noi_dung": noi_dung,
                        "vi_tri_bat_dau": gop[0]["vi_tri_bat_dau"],
                        "vi_tri_ket_thuc": gop[-1]["vi_tri_ket_thuc"],
                        "metadata": {**gop[0]["metadata"], "loai": "gop"},
                    })
                gop = []

        return ket_qua

    def chia_nhieu(
        self,
        van_ban_list: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Chia nhiều văn bản."""
        if metadata_list is None:
            metadata_list = [{}] * len(van_ban_list)

        ket_qua = []
        for vb, meta in zip(van_ban_list, metadata_list):
            ket_qua.extend(self.chia(vb, meta))
        return ket_qua

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê cấu hình."""
        return {
            "kich_thuoc": self.kich_thuoc,
            "chong_chong": self.chong_chong,
            "chien_luoc": self.chien_luoc,
            "toi_thieu_kich_thuoc": self.toi_thieu_kich_thuoc,
        }

    def __repr__(self) -> str:
        return (
            f"CatVanBan(kich_thuoc={self.kich_thuoc}, "
            f"chong_chong={self.chong_chong}, "
            f"chien_luoc='{self.chien_luoc}')"
        )
