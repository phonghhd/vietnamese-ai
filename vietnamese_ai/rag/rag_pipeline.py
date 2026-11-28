"""RAGPipeline - pipeline RAG hoàn chỉnh cho tiếng Việt."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

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
        enable_extreme_hardware: bool = False,
        tu_dong_embed: bool = True,
        backend: str = "memory",
        thu_muc_chroma: Optional[str] = "./chroma_db",
        duong_dan_vaidb: Optional[Union[str, Path]] = None,
        ten_collection: str = "vietnamese_ai",
        ten_model_embed: Optional[str] = None,
        che_do_nhung: str = "auto",
        duong_dan_onnx_embed: Optional[Union[str, Path]] = None,
        duong_dan_onnx_rerank: Optional[Union[str, Path]] = None,
        nen_ngu_canh: bool = False,
        ty_le_nen: float = 0.5,
        dung_semantic_cache: bool = False,
        nguong_cache: float = 0.88,
        quet_bao_mat: bool = True,
        bat_tuong_lua: bool = True,
        che_do_hanh_chinh: bool = False,
        bo_toi_uu_hanh_chinh: Optional[Any] = None,
    ):
        self.ham_sinh = ham_sinh
        self.top_k = top_k
        self.nguong_diem = nguong_diem
        self.toi_da_tu_vung = toi_da_tu_vung
        self.enable_extreme_hardware = enable_extreme_hardware
        self.backend = backend
        self.duong_dan_vaidb = str(duong_dan_vaidb) if duong_dan_vaidb is not None else None
        self.che_do_hanh_chinh = che_do_hanh_chinh
        self.bo_toi_uu_hanh_chinh = bo_toi_uu_hanh_chinh
        self._embedder: Any = None
        self._bo_theo_doi_hieu_luc: Optional[Any] = None
        self._do_thi_tham_chieu: Optional[Any] = None

        if ham_embed is None and tu_dong_embed:
            if che_do_nhung in ("onnx_viet", "vietnamese_semantic") or duong_dan_onnx_embed is not None:
                from vietnamese_ai.rag.onnx_semantic import BoNhungONNXTiengViet

                self._embedder = BoNhungONNXTiengViet(
                    duong_dan_onnx=duong_dan_onnx_embed,
                    kich_thuoc=384 if kich_thuoc_vector == 128 else kich_thuoc_vector,
                )
            else:
                from vietnamese_ai.rag.embedder import BoNhungVanBan

                self._embedder = (
                    BoNhungVanBan(ten_model=ten_model_embed)
                    if ten_model_embed
                    else BoNhungVanBan(phuong_phap="hash" if che_do_nhung == "hash" else None)
                )
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
        elif backend == "vaidb":
            from vietnamese_ai.rag.vaidb_store import KhoTriThucVAIDB

            db_target = duong_dan_vaidb if duong_dan_vaidb is not None else "vietnamese_ai_knowledge.vaidb"
            self.duong_dan_vaidb = str(db_target)
            self.csdl_vector = KhoTriThucVAIDB(
                duong_dan=db_target,
                kich_thuoc=kich_thuoc_vector,
                khoang_cach="cosine",
            )
        elif backend == "memory":
            self.csdl_vector = CSDLVector(
                kich_thuoc=kich_thuoc_vector,
                khoang_cach="cosine",
            )
        else:
            raise ValueError("backend phải là 'memory', 'vaidb' hoặc 'chroma'")

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
            duong_dan_onnx=str(duong_dan_onnx_rerank) if duong_dan_onnx_rerank else None,
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

        if self.che_do_hanh_chinh and self.bo_toi_uu_hanh_chinh is None:
            from vietnamese_ai.rag.vietnamese_context import BoToiUuNguCanhHanhChinh

            self.bo_toi_uu_hanh_chinh = BoToiUuNguCanhHanhChinh(bat_tuong_lua=self.bat_tuong_lua)

        self._lich_su: List[Dict[str, Any]] = []

    def them_tai_lieu(
        self,
        ma: str,
        van_ban: str,
        metadata: Optional[Dict[str, Any]] = None,
        quet_bao_mat: Optional[bool] = None,
        che_do_hanh_chinh: Optional[bool] = None,
    ) -> int:
        """
        Thêm tài liệu vào RAG pipeline (tự động khử độc Indirect Prompt Injection & ẩn PII/Secrets).
        Hỗ trợ chế độ bảo toàn cấu trúc văn bản hành chính & quy phạm pháp luật Việt Nam.

        Args:
            ma: ID tài liệu
            van_ban: Nội dung
            metadata: Metadata bổ sung
            quet_bao_mat: Ghi đè cờ quét bảo mật khi nạp tài liệu
            che_do_hanh_chinh: Bật/tắt phân đoạn bảo toàn Điều, Khoản, Căn cứ pháp lý

        Returns:
            Số chunks được tạo
        """
        can_quet = self.quet_bao_mat if quet_bao_mat is None else quet_bao_mat
        noi_dung_nap = van_ban
        if can_quet and self._tuong_lua is not None and isinstance(van_ban, str):
            kq_sach = self._tuong_lua.lam_sach_tai_lieu_rag(van_ban)
            noi_dung_nap = kq_sach["van_ban_sach"]

        if self.backend == "vaidb" and hasattr(self.csdl_vector, "xoa_tai_lieu"):
            self.csdl_vector.xoa_tai_lieu(ma)

        ap_dung_hc = self.che_do_hanh_chinh if che_do_hanh_chinh is None else che_do_hanh_chinh
        if ap_dung_hc:
            if self.bo_toi_uu_hanh_chinh is None:
                from vietnamese_ai.rag.vietnamese_context import BoToiUuNguCanhHanhChinh

                self.bo_toi_uu_hanh_chinh = BoToiUuNguCanhHanhChinh(bat_tuong_lua=self.bat_tuong_lua)

            chunks_hc = self.bo_toi_uu_hanh_chinh.phan_doan_giu_ngu_canh(
                van_ban=noi_dung_nap,
                ten_tai_lieu=ma,
                max_tokens=self.cat_van_ban.kich_thuoc,
            )
            trich_xuat_meta = self.bo_toi_uu_hanh_chinh.trich_xuat_can_cu_va_so_hieu(noi_dung_nap)
            meta_goc = dict(metadata or {})
            for c in chunks_hc:
                c["metadata"] = dict(c.get("metadata", {}))
                c["metadata"].update(meta_goc)
                if trich_xuat_meta.get("so_hieu_van_ban"):
                    c["metadata"]["so_hieu"] = trich_xuat_meta["so_hieu_van_ban"]
                if trich_xuat_meta.get("danh_sach_can_cu"):
                    c["metadata"]["can_cu"] = trich_xuat_meta["danh_sach_can_cu"]

            so_chunks = self.trich_xuat.them_chunks(ma, chunks_hc, van_ban_goc=noi_dung_nap)
            sh_chinh = meta_goc.get("so_hieu")
            if isinstance(sh_chinh, str) and sh_chinh.strip():
                self.lay_do_thi_tham_chieu().trich_xuat_va_them_tu_van_ban(sh_chinh, noi_dung_nap)
        else:
            so_chunks = self.trich_xuat.them_tai_lieu(ma, noi_dung_nap, metadata)

        if self.backend == "vaidb" and hasattr(self.csdl_vector, "ghi_nhan_tai_lieu"):
            import hashlib

            sha = hashlib.sha256(str(noi_dung_nap).encode("utf-8")).hexdigest()
            meta_doc = dict(metadata or {})
            self.csdl_vector.ghi_nhan_tai_lieu(
                ma_tai_lieu=ma,
                sha256_noi_dung=sha,
                so_chunks=so_chunks,
                duong_dan_nguon=meta_doc.get("duong_dan_tep"),
                kich_thuoc_bytes=len(str(noi_dung_nap).encode("utf-8")),
                metadata=meta_doc,
            )

        return so_chunks

    def them_nhieu_tai_lieu(
        self,
        tai_lieu: List[Dict[str, Any]],
        quet_bao_mat: Optional[bool] = None,
        che_do_hanh_chinh: Optional[bool] = None,
    ) -> int:
        """Thêm nhiều tài liệu (kèm quét bảo mật chống Indirect Prompt Injection và chế độ hành chính)."""
        tong = 0
        for doc in tai_lieu:
            ma = doc.get("ma") or doc.get("id") or f"doc_{tong}"
            nd = doc.get("van_ban") or doc.get("noi_dung") or doc.get("text") or ""
            meta = doc.get("metadata")
            tong += self.them_tai_lieu(
                ma=ma,
                van_ban=nd,
                metadata=meta,
                quet_bao_mat=quet_bao_mat,
                che_do_hanh_chinh=che_do_hanh_chinh,
            )
        return tong

    def them_tep(
        self,
        duong_dan: Union[str, Path],
        ma: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        quet_bao_mat: Optional[bool] = None,
        che_do_hanh_chinh: Optional[bool] = None,
        bat_bao_ve_ocr: bool = True,
    ) -> int:
        """
        Đọc tệp đa định dạng (`.txt`, `.md`, `.docx`, `.pdf` kèm bảng biểu, ảnh scan OCR, `.json`/`.jsonl`)
        bằng `BoDocTaiLieuPhapLy` và nạp trực tiếp vào RAG pipeline.
        """
        from vietnamese_ai.rag.legal.document_parser import BoDocTaiLieuPhapLy

        path = Path(duong_dan)
        parser = BoDocTaiLieuPhapLy(bat_bao_ve_ocr=bat_bao_ve_ocr)
        kq_doc = parser.doc_tep(path)
        meta_hop = dict(kq_doc.get("metadata", {}))
        meta_hop["dinh_dang"] = kq_doc.get("dinh_dang", "")
        meta_hop["so_trang"] = kq_doc.get("so_trang", 1)
        if metadata:
            meta_hop.update(metadata)
        return self.them_tai_lieu(
            ma=ma or path.name,
            van_ban=str(kq_doc.get("van_ban", "")),
            metadata=meta_hop,
            quet_bao_mat=quet_bao_mat,
            che_do_hanh_chinh=che_do_hanh_chinh,
        )

    def them_thu_muc(
        self,
        thu_muc: Union[str, Path],
        de_quy: bool = False,
        duoi_tep: Optional[List[str]] = None,
        quet_bao_mat: Optional[bool] = None,
        che_do_hanh_chinh: Optional[bool] = None,
        bat_bao_ve_ocr: bool = True,
    ) -> Dict[str, Any]:
        """Nạp toàn bộ tài liệu hợp lệ trong một thư mục vào RAG pipeline qua `BoDocTaiLieuPhapLy`."""
        from vietnamese_ai.rag.legal.document_parser import BoDocTaiLieuPhapLy

        dir_path = Path(thu_muc)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục tài liệu: {dir_path}")

        duoi_hop_le = (
            {d.lower() if d.startswith(".") else f".{d.lower()}" for d in duoi_tep}
            if duoi_tep
            else (
                BoDocTaiLieuPhapLy.DINH_DANG_VAN_BAN
                | BoDocTaiLieuPhapLy.DINH_DANG_DOCX
                | BoDocTaiLieuPhapLy.DINH_DANG_PDF
                | BoDocTaiLieuPhapLy.DINH_DANG_ANH
                | BoDocTaiLieuPhapLy.DINH_DANG_JSON
            )
        )

        duyet = dir_path.rglob("*") if de_quy else dir_path.iterdir()
        so_tep_thanh_cong = 0
        tong_chunks = 0
        cac_tep_loi: List[Dict[str, str]] = []
        for tep in sorted(duyet):
            if not tep.is_file() or tep.suffix.lower() not in duoi_hop_le:
                continue
            try:
                so_chunks = self.them_tep(
                    tep,
                    ma=str(tep.relative_to(dir_path)),
                    quet_bao_mat=quet_bao_mat,
                    che_do_hanh_chinh=che_do_hanh_chinh,
                    bat_bao_ve_ocr=bat_bao_ve_ocr,
                )
                so_tep_thanh_cong += 1
                tong_chunks += so_chunks
            except Exception as exc:
                cac_tep_loi.append({"tep": str(tep), "loi": str(exc)})

        return {
            "thu_muc": str(dir_path),
            "so_tep_thanh_cong": so_tep_thanh_cong,
            "tong_chunks": tong_chunks,
            "cac_tep_loi": cac_tep_loi,
        }

    def dong_bo_thu_muc(
        self,
        thu_muc: Union[str, Path],
        de_quy: bool = True,
        duoi_tep: Optional[List[str]] = None,
        xoa_tep_da_mat: bool = True,
        quet_bao_mat: Optional[bool] = None,
        che_do_hanh_chinh: Optional[bool] = None,
        bat_bao_ve_ocr: bool = True,
    ) -> Dict[str, Any]:
        """
        Đồng bộ tăng dần một thư mục tài liệu vào kho tri thức `.vaidb`:
        - Bỏ qua tức thì các tệp không thay đổi nội dung (khớp mã băm SHA-256).
        - Chỉ lập chỉ mục lại các tệp mới thêm hoặc vừa sửa đổi.
        - Tự dọn các tài liệu thuộc thư mục này đã bị xóa khỏi đĩa nếu `xoa_tep_da_mat=True`.
        """
        import hashlib

        from vietnamese_ai.rag.legal.document_parser import BoDocTaiLieuPhapLy

        dir_path = Path(thu_muc)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục tài liệu: {dir_path}")

        duoi_hop_le = (
            {d.lower() if d.startswith(".") else f".{d.lower()}" for d in duoi_tep}
            if duoi_tep
            else (
                BoDocTaiLieuPhapLy.DINH_DANG_VAN_BAN
                | BoDocTaiLieuPhapLy.DINH_DANG_DOCX
                | BoDocTaiLieuPhapLy.DINH_DANG_PDF
                | BoDocTaiLieuPhapLy.DINH_DANG_ANH
                | BoDocTaiLieuPhapLy.DINH_DANG_JSON
            )
        )

        duyet = dir_path.rglob("*") if de_quy else dir_path.iterdir()
        cac_ma_hien_con: Set[str] = set()
        so_tep_cap_nhat = 0
        so_tep_bo_qua = 0
        so_tep_da_xoa = 0
        tong_chunks_moi = 0
        cac_tep_loi: List[Dict[str, str]] = []

        for tep in sorted(duyet):
            if not tep.is_file() or tep.suffix.lower() not in duoi_hop_le:
                continue
            ma_rel = str(tep.relative_to(dir_path))
            cac_ma_hien_con.add(ma_rel)
            try:
                sha_tep = hashlib.sha256(tep.read_bytes()).hexdigest()
                if (
                    self.backend == "vaidb"
                    and hasattr(self.csdl_vector, "da_ton_tai_tai_lieu")
                    and self.csdl_vector.da_ton_tai_tai_lieu(ma_rel, sha_tep)
                ):
                    so_tep_bo_qua += 1
                    continue

                so_chunks = self.them_tep(
                    tep,
                    ma=ma_rel,
                    metadata={"duong_dan_tep": str(tep.resolve()), "thu_muc_dong_bo": str(dir_path.resolve())},
                    quet_bao_mat=quet_bao_mat,
                    che_do_hanh_chinh=che_do_hanh_chinh,
                    bat_bao_ve_ocr=bat_bao_ve_ocr,
                )
                if self.backend == "vaidb" and hasattr(self.csdl_vector, "ghi_nhan_tai_lieu"):
                    self.csdl_vector.ghi_nhan_tai_lieu(
                        ma_tai_lieu=ma_rel,
                        sha256_noi_dung=sha_tep,
                        so_chunks=so_chunks,
                        duong_dan_nguon=str(tep.resolve()),
                        kich_thuoc_bytes=tep.stat().st_size,
                        metadata={"thu_muc_dong_bo": str(dir_path.resolve())},
                    )
                so_tep_cap_nhat += 1
                tong_chunks_moi += so_chunks
            except Exception as exc:
                cac_tep_loi.append({"tep": str(tep), "loi": str(exc)})

        if xoa_tep_da_mat and self.backend == "vaidb" and hasattr(self.csdl_vector, "danh_sach_tai_lieu"):
            thu_muc_res = str(dir_path.resolve())
            for doc in self.csdl_vector.danh_sach_tai_lieu():
                meta_d = doc.get("metadata") or {}
                if meta_d.get("thu_muc_dong_bo") == thu_muc_res and doc["ma_tai_lieu"] not in cac_ma_hien_con:
                    self.xoa_tai_lieu(doc["ma_tai_lieu"])
                    so_tep_da_xoa += 1

        return {
            "thu_muc": str(dir_path),
            "so_tep_cap_nhat": so_tep_cap_nhat,
            "so_tep_bo_qua_khong_doi": so_tep_bo_qua,
            "so_tep_da_xoa": so_tep_da_xoa,
            "tong_chunks_moi": tong_chunks_moi,
            "cac_tep_loi": cac_tep_loi,
        }

    def lay_do_thi_tham_chieu(self) -> Any:
        """Lấy (hoặc khởi tạo lười) đối tượng `DoThiThamChieuPhapLy` dùng chung với `RAGPipeline`."""
        if self._do_thi_tham_chieu is None:
            from vietnamese_ai.rag.legal.citation_graph import DoThiThamChieuPhapLy
            from vietnamese_ai.rag.legal.validity_tracker import BoTheoDoiHieuLucPhapLy

            if self._bo_theo_doi_hieu_luc is None:
                self._bo_theo_doi_hieu_luc = BoTheoDoiHieuLucPhapLy()
            self._do_thi_tham_chieu = DoThiThamChieuPhapLy(bo_theo_doi=self._bo_theo_doi_hieu_luc)
        return self._do_thi_tham_chieu

    def dong(self) -> None:
        """Đóng tài nguyên lưu trữ bền vững (`.vaidb`) an toàn nếu đang mở."""
        if hasattr(self.csdl_vector, "dong"):
            self.csdl_vector.dong()

    def toi_uu_chi_muc_vector(
        self,
        so_cum: int = 16,
        so_vong_lap: int = 8,
        nprobe: int = 3,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Xây dựng chỉ mục gần đúng IVF-Flat cho `CSDLVector` để tăng tốc truy hồi ở quy mô lớn."""
        if hasattr(self.csdl_vector, "xay_dung_chi_muc_ivf"):
            return self.csdl_vector.xay_dung_chi_muc_ivf(
                so_cum=so_cum,
                so_vong_lap=so_vong_lap,
                nprobe=nprobe,
                seed=seed,
            )
        return {
            "da_xay_dung": False,
            "ly_do": f"Backend '{self.backend}' tự quản lý chỉ mục nội bộ.",
        }

    def xay_dung_chi_muc_ivf(
        self,
        so_cum: int = 16,
        so_vong_lap: int = 8,
        nprobe: int = 3,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Alias cho `toi_uu_chi_muc_vector`."""
        return self.toi_uu_chi_muc_vector(
            so_cum=so_cum,
            so_vong_lap=so_vong_lap,
            nprobe=nprobe,
            seed=seed,
        )

    def nap_plugin(
        self,
        plugin: Any,
        quet_bao_mat: Optional[bool] = None,
        che_do_hanh_chinh: Optional[bool] = None,
    ) -> int:
        """
        Nạp tri thức, tài liệu và từ điển nghiệp vụ từ Plugin (PluginTriThuc hoặc QuanLyPlugin).

        Args:
            plugin: Đối tượng PluginTriThuc hoặc QuanLyPlugin.
            quet_bao_mat: Bật/tắt quét bảo mật Indirect Prompt Injection.
            che_do_hanh_chinh: Kích hoạt bảo tồn ngữ cảnh số hiệu điều khoản.

        Returns:
            Số lượng tài liệu/đoạn tri thức mới được nạp vào RAG Pipeline.
        """
        from vietnamese_ai.plugin.plugin_base import PluginTriThuc
        from vietnamese_ai.plugin.plugin_manager import QuanLyPlugin

        dem = 0
        if isinstance(plugin, PluginTriThuc):
            tai_lieu_list = plugin.nap_nguon_tri_thuc()
            for doc in tai_lieu_list:
                ma = doc.get("ma") or doc.get("id") or f"{plugin.ten}_{dem}"
                nd = doc.get("noi_dung") or doc.get("van_ban") or doc.get("text") or ""
                meta = doc.get("metadata") or {}
                if "nguon" in doc and "nguon" not in meta:
                    meta["nguon"] = doc["nguon"]
                if "tieu_de" in doc and "tieu_de" not in meta:
                    meta["tieu_de"] = doc["tieu_de"]
                meta["plugin"] = plugin.ten
                dem += self.them_tai_lieu(
                    ma=ma,
                    van_ban=nd,
                    metadata=meta,
                    quet_bao_mat=quet_bao_mat,
                    che_do_hanh_chinh=che_do_hanh_chinh,
                )
        elif isinstance(plugin, QuanLyPlugin):
            for p_item in plugin.cac_plugin_tri_thuc.values():
                dem += self.nap_plugin(
                    p_item,
                    quet_bao_mat=quet_bao_mat,
                    che_do_hanh_chinh=che_do_hanh_chinh,
                )
        return dem

    def nap_plugin_tri_thuc(
        self,
        plugin: Any,
        quet_bao_mat: Optional[bool] = None,
        che_do_hanh_chinh: Optional[bool] = None,
    ) -> int:
        """Alias cho nap_plugin."""
        return self.nap_plugin(
            plugin,
            quet_bao_mat=quet_bao_mat,
            che_do_hanh_chinh=che_do_hanh_chinh,
        )

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

        # Bước 2.6: Trích xuất trích dẫn pháp lý và số hiệu nếu ở chế độ hành chính
        trich_dan_phap_ly = None
        if self.che_do_hanh_chinh and self.bo_toi_uu_hanh_chinh is not None:
            can_cu_set: Set[str] = set()
            so_hieu_set: Set[str] = set()
            dieu_khoan_set: Set[str] = set()

            for kq in ket_qua:
                m = kq.get("metadata", {})
                if m.get("dieu"):
                    tl = m.get("tai_lieu") or m.get("tai_lieu_goc") or m.get("van_ban") or ""
                    dieu_khoan_set.add(f"{m['dieu']}{(' - ' + tl) if tl else ''}")
                if m.get("so_hieu"):
                    sh = m["so_hieu"]
                    if isinstance(sh, list):
                        so_hieu_set.update(sh)
                    elif isinstance(sh, str):
                        so_hieu_set.add(sh)
                if m.get("can_cu"):
                    cc = m["can_cu"]
                    if isinstance(cc, list):
                        can_cu_set.update(cc)
                    elif isinstance(cc, str):
                        can_cu_set.add(cc)

                nd_kq = kq.get("noi_dung") or kq.get("metadata", {}).get("noi_dung", "")
                if nd_kq:
                    info_ext = self.bo_toi_uu_hanh_chinh.trich_xuat_can_cu_va_so_hieu(nd_kq)
                    so_hieu_set.update(info_ext.get("so_hieu_van_ban", []))
                    can_cu_set.update(info_ext.get("danh_sach_can_cu", []))

            info_q = self.bo_toi_uu_hanh_chinh.trich_xuat_can_cu_va_so_hieu(cau_hoi)
            so_hieu_set.update(info_q.get("so_hieu_van_ban", []))

            do_thi = self.lay_do_thi_tham_chieu()
            bo_hl = do_thi.bo_theo_doi
            ket_qua, canh_bao_hl = bo_hl.bo_sung_hieu_luc_cho_rag(ket_qua)
            hieu_luc_chi_tiet: List[Dict[str, Any]] = []
            for sh_item in sorted(list(so_hieu_set)):
                tt_hl = bo_hl.kiem_tra_hieu_luc(sh_item)
                if tt_hl["tim_thay"]:
                    hieu_luc_chi_tiet.append(tt_hl)
                    if tt_hl.get("canh_bao") and tt_hl["canh_bao"] not in canh_bao_hl:
                        canh_bao_hl.append(str(tt_hl["canh_bao"]))

            do_thi_mo_rong = do_thi.mo_rong_ngu_canh_rag(sorted(list(so_hieu_set))) if so_hieu_set else {
                "so_hieu_dau_vao": [],
                "van_ban_huong_dan_con_hieu_luc": [],
                "van_ban_thay_the_hien_hanh": [],
                "van_ban_sua_doi_bo_sung": [],
            }

            trich_dan_phap_ly = {
                "dieu_khoan": sorted(list(dieu_khoan_set)),
                "so_hieu_van_ban": sorted(list(so_hieu_set)),
                "danh_sach_can_cu": sorted(list(can_cu_set)),
                "hieu_luc_van_ban": hieu_luc_chi_tiet,
                "canh_bao_hieu_luc": canh_bao_hl,
                "do_thi_tham_chieu": do_thi_mo_rong,
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
        if trich_dan_phap_ly is not None:
            ket_qua_cuoi["trich_dan_phap_ly"] = trich_dan_phap_ly
        if thong_tin_nen is not None:
            ket_qua_cuoi["thong_tin_nen"] = thong_tin_nen

        if self.dung_semantic_cache and self._semantic_cache is not None:
            self._semantic_cache.luu(cau_hoi, ket_qua_cuoi)

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

        if self.che_do_hanh_chinh:
            tra_loi = "Căn cứ theo các quy định và tài liệu hành chính liên quan:\n\n"
        else:
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
        """Lưu RAG pipeline (hỗ trợ cả định dạng JSON memory lẫn xuất thẳng ra tệp bền vững `.vaidb`)."""
        p_dich = Path(duong_dan)
        if p_dich.suffix.lower() == ".vaidb":
            if self.backend == "vaidb" and hasattr(self.csdl_vector, "_conn"):
                p_dich.parent.mkdir(parents=True, exist_ok=True)
                src_path = Path(self.duong_dan_vaidb).resolve() if self.duong_dan_vaidb else None
                if src_path is not None and src_path == p_dich.resolve():
                    self.csdl_vector._conn.commit()
                    return
                import sqlite3

                self.csdl_vector._conn.commit()
                dst_conn = sqlite3.connect(str(p_dich))
                try:
                    self.csdl_vector._conn.backup(dst_conn)
                finally:
                    dst_conn.close()
                return

            from vietnamese_ai.rag.vaidb_store import KhoTriThucVAIDB

            kich_thuoc = int(getattr(self.csdl_vector, "kich_thuoc", 384))
            store = KhoTriThucVAIDB(p_dich, kich_thuoc=kich_thuoc)
            try:
                ids = list(getattr(self.csdl_vector, "_ids", []))
                vecs = getattr(self.csdl_vector, "_vectors", None)
                metas = list(getattr(self.csdl_vector, "_metadata", []))
                if ids and vecs is not None and len(ids) > 0:
                    store.chen_batch(ids, vecs, metas)
            finally:
                store.dong()
            return

        if not hasattr(self.csdl_vector, "luu"):
            raise NotImplementedError(
                f"Backend '{self.backend}' không hỗ trợ luu() với định dạng '{p_dich.suffix}' — hãy dùng đuôi '.vaidb' hoặc backend 'memory'."
            )
        self.csdl_vector.luu(duong_dan)

    def tai(self, duong_dan: str) -> None:
        """Tải RAG pipeline (tự nhận diện tệp `.vaidb` SQLite/FTS5 hoặc định dạng JSON memory)."""
        p_nguon = Path(duong_dan)
        if p_nguon.suffix.lower() == ".vaidb":
            from vietnamese_ai.rag.vaidb_store import KhoTriThucVAIDB

            self.dong()
            store: Any = KhoTriThucVAIDB(p_nguon)
            self.csdl_vector = store
            self.trich_xuat.csdl_vector = store
            self.backend = "vaidb"
            self.duong_dan_vaidb = str(p_nguon)
            return

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
            "chi_muc_ivf": (
                getattr(self.csdl_vector, "_ivf_centroids", None) is not None
                and not getattr(self.csdl_vector, "_ivf_dirty", True)
            ),
            "che_do_hanh_chinh": self.che_do_hanh_chinh,
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
