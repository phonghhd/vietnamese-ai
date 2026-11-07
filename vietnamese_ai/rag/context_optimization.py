"""NenNguCanh (PromptCompressor) & BoNhoDemNguNghia (SemanticCache) tối ưu hóa ngữ cảnh dài và tốc độ truy vấn RAG."""

import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.rag.embedder import BoNhungVanBan


class NenNguCanh:
    """
    Bộ nén ngữ cảnh dài (Long-Context Prompt Compressor - phong cách LLMLingua cho tiếng Việt).

    Tự động chấm điểm từng câu và cụm từ dựa trên:
    - Độ tương đồng ngữ nghĩa với câu hỏi truy vấn (`cau_hoi`)
    - Mật độ thông tin thực thể (chữ số, ngày tháng, tên riêng viết hoa, thuật ngữ chuyên ngành)
    - Loại bỏ các từ đệm / câu trùng lặp để đạt tỷ lệ nén mục tiêu (`ty_le_nen`),
      giúp tiết kiệm 40%–70% lượng token đưa vào LLM mà vẫn bảo toàn thông tin cốt lõi.
    """

    TU_DEM_TIENG_VIET = {
        "thì",
        "là",
        "mà",
        "những",
        "các",
        "một",
        "này",
        "đó",
        "ấy",
        "vậy",
        "nhé",
        "nha",
        "ạ",
        "với",
        "cùng",
        "như",
        "cũng",
        "rất",
        "khá",
        "hơi",
    }

    def __init__(self, ty_le_giu: float = 0.5, kich_thuoc_embed: int = 128):
        self.ty_le_giu = ty_le_giu
        self.embedder = BoNhungVanBan(kich_thuoc=kich_thuoc_embed)

    def _diem_thong_tin_cau(self, cau: str, vec_cau_hoi: Optional[np.ndarray] = None) -> float:
        tu_list = cau.split()
        if not tu_list:
            return 0.0

        # Thưởng cho câu chứa số liệu, phần trăm, năm hoặc chữ viết hoa (tên riêng)
        co_so = len(re.findall(r"\d+", cau))
        co_viet_hoa = sum(1 for tu in tu_list[1:] if tu[:1].isupper())
        diem_thuc_the = min(1.0, (co_so * 0.25) + (co_viet_hoa * 0.15))

        # Mật độ từ không phải từ đệm
        ty_le_thuc_tu = sum(1 for tu in tu_list if tu.lower() not in self.TU_DEM_TIENG_VIET) / len(
            tu_list
        )

        diem_ngu_nghia = 0.5
        if vec_cau_hoi is not None:
            vec_cau = self.embedder.nhung(cau)
            diem_ngu_nghia = float(np.clip((np.dot(vec_cau, vec_cau_hoi) + 1.0) / 2.0, 0.0, 1.0))

        return 0.50 * diem_ngu_nghia + 0.30 * diem_thuc_the + 0.20 * ty_le_thuc_tu

    def nen(
        self,
        ngu_canh: str,
        cau_hoi: str = "",
        ty_le_nen: Optional[float] = None,
        ty_le_giu: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Nén văn bản ngữ cảnh dài theo `ty_le_nen` $\\in (0.1, 1.0]$.
        """
        actual_ratio = (
            ty_le_nen
            if ty_le_nen is not None
            else (ty_le_giu if ty_le_giu is not None else self.ty_le_giu)
        )
        ty_le_nen = float(np.clip(actual_ratio, 0.1, 1.0))
        tu_goc = ngu_canh.split()
        if not tu_goc:
            return {
                "van_ban_nen": "",
                "so_tu_goc": 0,
                "so_tu_nen": 0,
                "ty_le_tiet_kiem": 0.0,
            }

        muc_tieu_so_tu = max(5, int(round(len(tu_goc) * ty_le_nen)))
        cac_cau = [c.strip() for c in re.split(r"(?<=[.!?])\s+|\n+", ngu_canh) if c.strip()]
        vec_hoi = self.embedder.nhung(cau_hoi) if cau_hoi.strip() else None

        # Xếp hạng câu theo điểm thông tin nhưng giữ nguyên thứ tự xuất hiện ban đầu khi ghép lại
        diem_cau = [
            (idx, cau, self._diem_thong_tin_cau(cau, vec_hoi)) for idx, cau in enumerate(cac_cau)
        ]
        diem_cau_sorted = sorted(diem_cau, key=lambda x: x[2], reverse=True)

        chon_idx = set()
        dem_tu = 0
        for idx, cau, _ in diem_cau_sorted:
            so_tu_cau = len(cau.split())
            if dem_tu + so_tu_cau <= muc_tieu_so_tu or not chon_idx:
                chon_idx.add(idx)
                dem_tu += so_tu_cau
            if dem_tu >= muc_tieu_so_tu:
                break

        cau_da_chon = [cac_cau[i] for i in sorted(chon_idx)]
        van_ban_so_bo = " ".join(cau_da_chon)

        # Nếu vẫn dài hơn mục tiêu, lọc tiếp các từ đệm không mang thông tin số/thực thể
        tu_so_bo = van_ban_so_bo.split()
        if len(tu_so_bo) > muc_tieu_so_tu:
            tu_giu_lai = []
            for tu in tu_so_bo:
                clean_tu = tu.lower().strip(".,;:!?")
                if clean_tu in self.TU_DEM_TIENG_VIET and len(tu_giu_lai) + (len(tu_so_bo) - len(tu_giu_lai)) > muc_tieu_so_tu:
                    continue
                tu_giu_lai.append(tu)
            van_ban_nen = " ".join(tu_giu_lai[:muc_tieu_so_tu])
        else:
            van_ban_nen = van_ban_so_bo

        so_tu_nen = len(van_ban_nen.split())
        tiet_kiem = max(0.0, 1.0 - (so_tu_nen / max(1, len(tu_goc))))
        return {
            "van_ban_nen": van_ban_nen,
            "so_tu_goc": len(tu_goc),
            "so_tu_nen": so_tu_nen,
            "ty_le_tiet_kiem": round(tiet_kiem, 4),
        }


class BoNhoDemNguNghia:
    """
    Bộ nhớ đệm Ngữ nghĩa (Semantic Cache) cho `RAGPipeline` và LLM.

    Cho phép phát hiện các câu hỏi mới có ý nghĩa tương đương với câu hỏi đã trả lời trước đó
    (dựa trên độ tương đồng Cosine $\\ge$ `nguong_tuong_dong`) để trả về kết quả tức thì.
    """

    def __init__(
        self,
        nguong_tuong_dong: float = 0.85,
        kich_thuoc_toi_da: int = 512,
        kich_thuoc_embed: int = 128,
        ham_embed: Optional[Any] = None,
    ):
        self.nguong_tuong_dong = nguong_tuong_dong
        self.kich_thuoc_toi_da = kich_thuoc_toi_da
        self.embedder = ham_embed if ham_embed is not None else BoNhungVanBan(kich_thuoc=kich_thuoc_embed)
        self._muc_luu: List[Dict[str, Any]] = []
        self.so_lan_trung: int = 0
        self.so_lan_truot: int = 0

    def _nhung_cau(self, cau_hoi: str) -> np.ndarray:
        if hasattr(self.embedder, "nhung"):
            v = self.embedder.nhung(cau_hoi)
        else:
            v = self.embedder(cau_hoi)
        arr = np.asarray(v, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0 else arr

    def tra_cuu(self, cau_hoi: str) -> Optional[Dict[str, Any]]:
        """Tìm câu trả lời trong bộ nhớ đệm ngữ nghĩa nếu độ tương đồng $\\ge$ `nguong_tuong_dong`."""
        if not self._muc_luu or not cau_hoi.strip():
            self.so_lan_truot += 1
            return None

        vec_q = self._nhung_cau(cau_hoi)
        best_sim = -1.0
        best_entry: Optional[Dict[str, Any]] = None

        for entry in self._muc_luu:
            sim = float(np.dot(vec_q, entry["vector"]))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.nguong_tuong_dong:
            self.so_lan_trung += 1
            ket_qua = dict(best_entry["ket_qua"])
            ket_qua["semantic_cache_hit"] = True
            ket_qua["do_tuong_dong_cache"] = round(best_sim, 4)
            return ket_qua

        self.so_lan_truot += 1
        return None

    def luu(self, cau_hoi: str, ket_qua: Dict[str, Any]) -> None:
        """Lưu cặp `(cau_hoi, ket_qua)` vào bộ nhớ đệm ngữ nghĩa."""
        if not cau_hoi.strip():
            return
        vec_q = self._nhung_cau(cau_hoi)
        if len(self._muc_luu) >= self.kich_thuoc_toi_da:
            self._muc_luu.pop(0)
        self._muc_luu.append(
            {
                "cau_hoi": cau_hoi,
                "vector": vec_q,
                "ket_qua": dict(ket_qua),
                "thoi_gian": time.time(),
            }
        )

    def xoa(self) -> None:
        """Xóa toàn bộ bộ nhớ đệm."""
        self._muc_luu.clear()
        self.so_lan_trung = 0
        self.so_lan_truot = 0

    def thong_ke(self) -> Dict[str, Any]:
        tong = self.so_lan_trung + self.so_lan_truot
        return {
            "so_muc_da_luu": len(self._muc_luu),
            "so_lan_trung": self.so_lan_trung,
            "so_lan_truot": self.so_lan_truot,
            "hits": self.so_lan_trung,
            "misses": self.so_lan_truot,
            "ty_le_trung": round(self.so_lan_trung / max(1, tong), 4),
        }


PromptCompressor = NenNguCanh
SemanticCache = BoNhoDemNguNghia
