"""RAGPipeline - pipeline RAG hoàn chỉnh cho tiếng Việt."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.rag.chunker import CatVanBan
from vietnamese_ai.rag.reranker import SapXepLai
from vietnamese_ai.rag.retriever import TrichXuat
from vietnamese_ai.rag.vector_store import CSDLVector


class RAGPipeline:
    """
    Pipeline RAG (Retrieval-Augmented Generation) hoàn chỉnh.

    Kết hợp: chunking → vector store → retrieval → reranking → generation.

    Sử dụng:
        >>> rag = RAGPipeline(ham_embed=embed_fn, ham_sinh=generate_fn)
        >>> rag.them_tai_lieu("doc1.txt", van_ban)
        >>> ket_qua = rag.hoi("Câu hỏi về tài liệu?")
    """

    def __init__(
        self,
        ham_embed: Optional[Callable[[str], np.ndarray]] = None,
        ham_sinh: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
        kich_thuoc_vector: int = 128,
        kich_thuoc_chunk: int = 200,
        chong_chong_chunk: int = 50,
        chien_luoc_chunk: str = "tu",
        che_do_tim_kiem: str = "hybrid",
        che_do_rerank: str = "mmr",
        trong_so_semantic: float = 0.7,
        trong_so_da_dang: float = 0.5,
        top_k: int = 5,
        nguong_diem: Optional[float] = None,
        toi_da_tu_vung: int = 3000,
    ):
        self.ham_embed = ham_embed
        self.ham_sinh = ham_sinh
        self.top_k = top_k
        self.nguong_diem = nguong_diem
        self.toi_da_tu_vung = toi_da_tu_vung

        self.cat_van_ban = CatVanBan(
            kich_thuoc=kich_thuoc_chunk,
            chong_chong=chong_chong_chunk,
            chien_luoc=chien_luoc_chunk,
        )

        self.csdl_vector = CSDLVector(
            kich_thuoc=kich_thuoc_vector,
            khoang_cach="cosine",
        )

        self.trich_xuat = TrichXuat(
            csdl_vector=self.csdl_vector,
            ham_embed=self.ham_embed,
            cat_van_ban=self.cat_van_ban,
            che_do=che_do_tim_kiem,
            trong_so_semantic=trong_so_semantic,
        )

        self.reranker = SapXepLai(
            che_do=che_do_rerank,
            trong_so_da_dang=trong_so_da_dang,
        )

        self._lich_su: List[Dict[str, Any]] = []

    def them_tai_lieu(
        self,
        ma: str,
        van_ban: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Thêm tài liệu vào RAG pipeline.

        Args:
            ma: ID tài liệu
            van_ban: Nội dung
            metadata: Metadata bổ sung

        Returns:
            Số chunks được tạo
        """
        return self.trich_xuat.them_tai_lieu(ma, van_ban, metadata)

    def them_nhieu_tai_lieu(
        self,
        tai_lieu: List[Dict[str, Any]],
    ) -> int:
        """Thêm nhiều tài liệu."""
        return self.trich_xuat.them_nhieu_tai_lieu(tai_lieu)

    def hoi(
        self,
        cau_hoi: str,
        top_k: Optional[int] = None,
        rerank: bool = True,
        co_kem_nguon: bool = True,
    ) -> Dict[str, Any]:
        """
        Hỏi và trả lời dựa trên tài liệu.

        Args:
            cau_hoi: Câu hỏi
            top_k: Số kết quả (ghi đè default)
            rerank: Có rerank không
            co_kem_nguon: Có kèm nguồn tham khảo không

        Returns:
            {cau_hoi, tra_loi, nguon, so_luong_nguon}
        """
        top_k = top_k or self.top_k

        # Bước 1: Retrieval
        ket_qua = self.trich_xuat.tim_kiem(cau_hoi, top_k=top_k * 2, nguong=self.nguong_diem)

        # Bước 2: Reranking
        if rerank and len(ket_qua) > 1:
            ket_qua = self.reranker.sap_xep_lai(cau_hoi, ket_qua, top_k)
        else:
            ket_qua = ket_qua[:top_k]

        # Bước 3: Generation
        if self.ham_sinh and ket_qua:
            tra_loi = self.ham_sinh(cau_hoi, ket_qua)
        else:
            tra_loi = self._sinh_tra_loi_don_gian(cau_hoi, ket_qua)

        ket_qua_cuoi = {
            "cau_hoi": cau_hoi,
            "tra_loi": tra_loi,
            "nguon": ket_qua if co_kem_nguon else [],
            "so_luong_nguon": len(ket_qua),
        }

        self._lich_su.append(ket_qua_cuoi)
        return ket_qua_cuoi

    def tim_kiem(
        self,
        cau_hoi: str,
        top_k: Optional[int] = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Chỉ tìm kiếm, không sinh câu trả lời."""
        top_k = top_k or self.top_k

        ket_qua = self.trich_xuat.tim_kiem(cau_hoi, top_k=top_k * 2, nguong=self.nguong_diem)

        if rerank and len(ket_qua) > 1:
            ket_qua = self.reranker.sap_xep_lai(cau_hoi, ket_qua, top_k)

        return ket_qua[:top_k]

    def _sinh_tra_loi_don_gian(
        self,
        cau_hoi: str,
        ket_qua: List[Dict[str, Any]],
    ) -> str:
        """Sinh câu trả lời đơn giản khi không có generator."""
        if not ket_qua:
            return "Không tìm thấy thông tin liên quan trong tài liệu."

        noi_dung_list = []
        for kq in ket_qua:
            nd = kq.get("metadata", {}).get("noi_dung", "")
            if not nd:
                nd = kq.get("noi_dung", "")
            if nd:
                noi_dung_list.append(nd)

        if not noi_dung_list:
            return "Không tìm thấy thông tin liên quan."

        tra_loi = "Dựa trên tài liệu, đây là thông tin liên quan:\n\n"
        for i, nd in enumerate(noi_dung_list, 1):
            tra_loi += f"{i}. {nd}\n\n"

        return tra_loi.strip()

    def xoa_tai_lieu(self, ma: str) -> int:
        """Xóa tài liệu."""
        return self.trich_xuat.xoa_tai_lieu(ma)

    def xoa_tat_ca(self) -> None:
        """Xóa toàn bộ tài liệu."""
        self.csdl_vector.xoa_tat_ca()
        self.trich_xuat._van_ban_goc.clear()
        self.trich_xuat._tu_dien.clear()
        self.trich_xuat._df.clear()
        self.trich_xuat._tong_tai_lieu = 0

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử hỏi đáp."""
        return self._lich_su.copy()

    def xoa_lich_su(self) -> None:
        """Xóa lịch sử hỏi đáp."""
        self._lich_su.clear()

    def luu(self, duong_dan: str) -> None:
        """Lưu RAG pipeline (vector store + config)."""
        self.csdl_vector.luu(duong_dan)

    def tai(self, duong_dan: str) -> None:
        """Tải RAG pipeline."""
        self.csdl_vector = CSDLVector.tai(duong_dan)
        self.trich_xuat.csdl_vector = self.csdl_vector

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê RAG pipeline."""
        return {
            "so_tai_lieu": self.trich_xuat.so_tai_lieu(),
            "so_chunks": self.csdl_vector.so_luong(),
            "top_k": self.top_k,
            "nguong_diem": self.nguong_diem,
            "trich_xuat": self.trich_xuat.thong_ke(),
            "reranker": self.reranker.thong_ke(),
            "lich_su": len(self._lich_su),
        }

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(so_tai_lieu={self.trich_xuat.so_tai_lieu()}, "
            f"so_chunks={self.csdl_vector.so_luong()}, "
            f"top_k={self.top_k})"
        )
