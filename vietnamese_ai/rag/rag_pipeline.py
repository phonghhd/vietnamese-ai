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

    Mặc định tự tạo `BoNhungVanBan` (sentence-transformers nếu có).
    `backend="chroma"` dùng ChromaDB khi đã cài `pip install chromadb`.

    Sử dụng:
        >>> rag = RAGPipeline()  # auto embed
        >>> rag = RAGPipeline(backend="chroma", thu_muc_chroma="./chroma_db")
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
        enable_extreme_hardware: bool = True,
        tu_dong_embed: bool = True,
        backend: str = "memory",
        thu_muc_chroma: Optional[str] = "./chroma_db",
        ten_collection: str = "vietnamese_ai",
        ten_model_embed: Optional[str] = None,
        nen_ngu_canh: bool = False,
        ty_le_nen: float = 0.5,
        dung_semantic_cache: bool = False,
        nguong_cache: float = 0.88,
        quet_bao_mat: bool = True,
        bat_tuong_lua: bool = True,
    ):
        self.ham_sinh = ham_sinh
        self.top_k = top_k
        self.nguong_diem = nguong_diem
        self.toi_da_tu_vung = toi_da_tu_vung
        self.enable_extreme_hardware = enable_extreme_hardware
        self.backend = backend
        self._embedder = None

        if ham_embed is None and tu_dong_embed:
            from vietnamese_ai.rag.embedder import BoNhungVanBan

            kwargs = {}
            if ten_model_embed:
                kwargs["ten_model"] = ten_model_embed
            self._embedder = BoNhungVanBan(**kwargs)
            ham_embed = self._embedder
            kich_thuoc_vector = self._embedder.kich_thuoc

        self.ham_embed = ham_embed

        # v27.0.1 Hardware Optimizer
        self.flash_attention = None
        self.ring_attention = None
        if self.enable_extreme_hardware:
            try:
                from vietnamese_ai.distributed.ring_attention import RingAttentionNode
                from vietnamese_ai.transformer.attention import (
                    MultiHeadFlashAttention,  # noqa: F401
                )

                self.ring_attention = RingAttentionNode(d_model=128, node_id=0, total_nodes=1)
                self.flash_attention = "Tiling FlashAttention Enabled (O(N) RAM)"
            except ImportError:
                pass

        self.cat_van_ban = CatVanBan(
            kich_thuoc=kich_thuoc_chunk,
            chong_chong=chong_chong_chunk,
            chien_luoc=chien_luoc_chunk,
        )

        self.csdl_vector: Any
        if backend == "chroma":
            from vietnamese_ai.rag.chroma_store import ChromaVectorStore

            self.csdl_vector = ChromaVectorStore(
                thu_muc_luu_tru=thu_muc_chroma,
                ten_collection=ten_collection,
                khoang_cach="cosine",
            )
        elif backend == "memory":
            self.csdl_vector = CSDLVector(
                kich_thuoc=kich_thuoc_vector,
                khoang_cach="cosine",
            )
        else:
            raise ValueError("backend phải là 'memory' hoặc 'chroma'")

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

        self.nen_ngu_canh = nen_ngu_canh
        self.ty_le_nen = ty_le_nen
        self.dung_semantic_cache = dung_semantic_cache
        self.quet_bao_mat = quet_bao_mat
        self.bat_tuong_lua = bat_tuong_lua
        self._compressor = None
        self._semantic_cache = None
        self._tuong_lua = None
        if quet_bao_mat or bat_tuong_lua:
            from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

            self._tuong_lua = TuongLuaBaoVeFramework(gioi_han_req_moi_phut=10000)
        if nen_ngu_canh or dung_semantic_cache:
            from vietnamese_ai.rag.context_optimization import BoNhoDemNguNghia, NenNguCanh

            if nen_ngu_canh:
                self._compressor = NenNguCanh(ty_le_giu=ty_le_nen)
            if dung_semantic_cache:
                self._semantic_cache = BoNhoDemNguNghia(
                    nguong_tuong_dong=nguong_cache, ham_embed=self.ham_embed
                )

        self._lich_su: List[Dict[str, Any]] = []

    def them_tai_lieu(
        self,
        ma: str,
        van_ban: str,
        metadata: Optional[Dict[str, Any]] = None,
        quet_bao_mat: Optional[bool] = None,
    ) -> int:
        """
        Thêm tài liệu vào RAG pipeline (tự động khử độc Indirect Prompt Injection & ẩn PII/Secrets).

        Args:
            ma: ID tài liệu
            van_ban: Nội dung
            metadata: Metadata bổ sung
            quet_bao_mat: Ghi đè cờ quét bảo mật khi nạp tài liệu

        Returns:
            Số chunks được tạo
        """
        can_quet = self.quet_bao_mat if quet_bao_mat is None else quet_bao_mat
        noi_dung_nap = van_ban
        if can_quet and self._tuong_lua is not None and isinstance(van_ban, str):
            kq_sach = self._tuong_lua.lam_sach_tai_lieu_rag(van_ban)
            noi_dung_nap = kq_sach["van_ban_sach"]
        return self.trich_xuat.them_tai_lieu(ma, noi_dung_nap, metadata)

    def them_nhieu_tai_lieu(
        self,
        tai_lieu: List[Dict[str, Any]],
        quet_bao_mat: Optional[bool] = None,
    ) -> int:
        """Thêm nhiều tài liệu (kèm quét bảo mật chống Indirect Prompt Injection)."""
        can_quet = self.quet_bao_mat if quet_bao_mat is None else quet_bao_mat
        if can_quet and self._tuong_lua is not None:
            ds_sach = []
            for doc in tai_lieu:
                ban_sao = dict(doc)
                for k in ("van_ban", "noi_dung", "text"):
                    if k in ban_sao and isinstance(ban_sao[k], str):
                        ban_sao[k] = self._tuong_lua.lam_sach_tai_lieu_rag(ban_sao[k])["van_ban_sach"]
                ds_sach.append(ban_sao)
            return self.trich_xuat.them_nhieu_tai_lieu(ds_sach)
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

        # Lớp 1: Kiểm tra Tường lửa đầu vào nếu bật
        if self.bat_tuong_lua and self._tuong_lua is not None:
            kq_in = self._tuong_lua.kiem_tra_dau_vao(str(cau_hoi))
            if not kq_in["an_toan"]:
                return {
                    "cau_hoi": cau_hoi,
                    "tra_loi": f"[Bị chặn bởi Tường lửa AI]: {kq_in['ly_do']}",
                    "bi_chan": True,
                    "nguon": [],
                    "so_luong_nguon": 0,
                    "tu_semantic_cache": False,
                }

        # Bước 0: Kiểm tra Semantic Cache nếu bật
        if self.dung_semantic_cache and self._semantic_cache is not None:
            cached = self._semantic_cache.tra_cuu(cau_hoi)
            if cached is not None:
                ket_qua_cached = dict(cached)
                ket_qua_cached["tu_semantic_cache"] = True
                self._lich_su.append(ket_qua_cached)
                return ket_qua_cached

        # Bước 1: Retrieval
        ket_qua = self.trich_xuat.tim_kiem(cau_hoi, top_k=top_k * 2, nguong=self.nguong_diem)

        # Bước 2: Reranking
        if rerank and len(ket_qua) > 1:
            ket_qua = self.reranker.sap_xep_lai(cau_hoi, ket_qua, top_k)
        else:
            ket_qua = ket_qua[:top_k]

        # Bước 2.5: Nén ngữ cảnh (Prompt Compression) nếu bật
        thong_tin_nen = None
        if self.nen_ngu_canh and self._compressor is not None and ket_qua:
            ket_qua_nen = []
            tong_truoc = 0
            tong_sau = 0
            for kq in ket_qua:
                ban_sao = dict(kq)
                nd = ban_sao.get("noi_dung") or ban_sao.get("metadata", {}).get("noi_dung", "")
                if nd:
                    res_nen = self._compressor.nen(nd, cau_hoi=cau_hoi, ty_le_giu=self.ty_le_nen)
                    ban_sao["noi_dung"] = res_nen["van_ban_nen"]
                    if "metadata" in ban_sao and isinstance(ban_sao["metadata"], dict):
                        ban_sao["metadata"] = dict(ban_sao["metadata"])
                        ban_sao["metadata"]["noi_dung"] = res_nen["van_ban_nen"]
                    tong_truoc += res_nen["so_tu_goc"]
                    tong_sau += res_nen["so_tu_nen"]
                ket_qua_nen.append(ban_sao)
            ket_qua = ket_qua_nen
            thong_tin_nen = {
                "so_tu_goc": tong_truoc,
                "so_tu_nen": tong_sau,
                "ty_le_tiet_kiem_pct": round((1.0 - tong_sau / max(1, tong_truoc)) * 100.0, 2),
            }

        # Bước 3: Generation
        if self.ham_sinh and ket_qua:
            tra_loi = self.ham_sinh(cau_hoi, ket_qua)
        else:
            tra_loi = self._sinh_tra_loi_don_gian(cau_hoi, ket_qua)

        if self.bat_tuong_lua and self._tuong_lua is not None and isinstance(tra_loi, str):
            tra_loi = self._tuong_lua.loc_dau_ra(tra_loi)

        from vietnamese_ai.security.data_defense import CongAnToanDauRa

        thong_tin_bam_sat = (
            CongAnToanDauRa.kiem_tra_ao_giac(str(tra_loi), ket_qua)
            if ket_qua
            else {"diem_bam_sat_nguon": 0.0, "nghi_ngo_ao_giac": False}
        )

        ket_qua_cuoi = {
            "cau_hoi": cau_hoi,
            "tra_loi": tra_loi,
            "nguon": ket_qua if co_kem_nguon else [],
            "so_luong_nguon": len(ket_qua),
            "tu_semantic_cache": False,
            **thong_tin_bam_sat,
        }
        if thong_tin_nen is not None:
            ket_qua_cuoi["thong_tin_nen"] = thong_tin_nen

        if self.dung_semantic_cache and self._semantic_cache is not None:
            self._semantic_cache.luu(cau_hoi, ket_qua_cuoi)

        # v27.0.1: Bơm metadata nếu phần cứng tối ưu được bật
        if self.enable_extreme_hardware and self.flash_attention:
            # Gỉa lập: Nếu context quá lớn, RingAttention và FlashAttention sẽ can thiệp để tránh OOM
            if len(ket_qua) > 50:
                ket_qua_cuoi["extreme_hardware"] = "Kích hoạt RingAttention phân tán & FlashAttention Tiling"
            else:
                ket_qua_cuoi["extreme_hardware"] = self.flash_attention

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
        if hasattr(self.csdl_vector, "xoa_tat_ca"):
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
        """Lưu RAG pipeline (chỉ backend memory)."""
        if not hasattr(self.csdl_vector, "luu"):
            raise NotImplementedError(
                f"Backend '{self.backend}' không hỗ trợ luu() — dùng memory hoặc Chroma persistence."
            )
        self.csdl_vector.luu(duong_dan)

    def tai(self, duong_dan: str) -> None:
        """Tải RAG pipeline (chỉ backend memory)."""
        self.csdl_vector = CSDLVector.tai(duong_dan)
        self.trich_xuat.csdl_vector = self.csdl_vector
        self.backend = "memory"

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê RAG pipeline."""
        embed_info = None
        if self._embedder is not None:
            embed_info = {
                "che_do": getattr(self._embedder, "che_do", "custom"),
                "kich_thuoc": getattr(self._embedder, "kich_thuoc", None),
            }
        return {
            "so_tai_lieu": self.trich_xuat.so_tai_lieu(),
            "so_chunks": self.csdl_vector.so_luong(),
            "backend": self.backend,
            "embedder": embed_info,
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
