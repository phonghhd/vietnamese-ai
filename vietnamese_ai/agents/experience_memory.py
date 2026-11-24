"""SoTayKinhNghiem - Bộ Nhớ Kinh Nghiệm Tác Tử Dùng Truy Hồi Vector Cosine Thực Tế."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.nlp.native_tokenizer import BoNhungVanBanSieuToc


class SoTayKinhNghiem:
    """
    Sổ tay kinh nghiệm (Experience Memory / Agent Memory 2.0):
    Lưu trữ lỗi và bài học của Tác tử vào VectorDB hoặc chỉ mục vector cosine nội bộ `BoNhungVanBanSieuToc`.
    """

    def __init__(self, vector_store: Any = None, so_chieu: int = 256) -> None:
        self.vector_store = vector_store
        self._embedder = BoNhungVanBanSieuToc(so_chieu=so_chieu)
        self._bo_nho_vector: List[Dict[str, Any]] = []

        if self.vector_store is None:
            try:
                from vietnamese_ai.rag.vector_store import CSDLVector

                self.vector_store = CSDLVector(kich_thuoc=so_chieu)
            except ImportError:
                pass

    def ghi_nhan_bai_hoc(self, ngu_canh: str, sai_lam: str, cach_khac_phuc: str) -> str:
        """Lưu một bài học kinh nghiệm mới kèm vector nhúng chuẩn hóa L2."""
        noi_dung = f"[SAI LẦM]: {sai_lam} -> [BÀI HỌC]: {cach_khac_phuc}"
        doc_id = f"exp_{uuid.uuid4().hex[:8]}"
        vec = self._embedder.nhung(ngu_canh)

        if self.vector_store and hasattr(self.vector_store, "them_tai_lieu"):
            self.vector_store.them_tai_lieu(
                id_tai_lieu=doc_id, noi_dung=ngu_canh, metadata={"bai_hoc": noi_dung}
            )
        elif self.vector_store and hasattr(self.vector_store, "chen"):
            self.vector_store.chen(
                ma=doc_id,
                vector=vec.astype(np.float32),
                metadata={"ngu_canh": ngu_canh, "bai_hoc": noi_dung},
            )

        self._bo_nho_vector.append(
            {
                "id": doc_id,
                "ngu_canh": ngu_canh,
                "bai_hoc": noi_dung,
                "vector": vec,
            }
        )
        return doc_id

    def truy_xuat_kinh_nghiem(
        self,
        truy_van: str,
        top_k: int = 1,
        nguong_tuong_dong: float = 0.05,
    ) -> Optional[str]:
        """Tìm kiếm bài học liên quan nhất bằng độ tương đồng cosine trên không gian vector thực tế."""
        bai_hoc_tim_thay: List[str] = []

        if self.vector_store and hasattr(self.vector_store, "tim_kiem"):
            try:
                if hasattr(self.vector_store, "them_tai_lieu"):
                    ket_qua = self.vector_store.tim_kiem(truy_van, top_k=top_k)
                else:
                    q_vec = self._embedder.nhung(truy_van).astype(np.float32)
                    ket_qua = self.vector_store.tim_kiem(q_vec, top_k=top_k)
                for item in ket_qua:
                    meta = item.get("metadata", {}) if isinstance(item, dict) else {}
                    if "bai_hoc" in meta:
                        bai_hoc_tim_thay.append(str(meta["bai_hoc"]))
            except Exception:
                pass

        if not bai_hoc_tim_thay and self._bo_nho_vector:
            q_vec = self._embedder.nhung(truy_van)
            ma_tran = np.vstack([item["vector"] for item in self._bo_nho_vector])
            diem = self._embedder.tim_kiem_tuong_dong_hang_loat(q_vec, ma_tran)
            thu_tu = np.argsort(diem)[::-1]
            for idx in thu_tu[:top_k]:
                if float(diem[int(idx)]) >= nguong_tuong_dong:
                    bai_hoc_tim_thay.append(str(self._bo_nho_vector[int(idx)]["bai_hoc"]))

        if bai_hoc_tim_thay:
            return "\n".join(bai_hoc_tim_thay)
        return None
