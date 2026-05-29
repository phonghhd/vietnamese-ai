"""TrichXuat - trích xuất thông tin từ vector store cho RAG."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.rag.chunker import CatVanBan
from vietnamese_ai.rag.vector_store import CSDLVector


class TrichXuat:
    """
    Trích xuất thông tin (retriever) kết hợp vector store và text chunker.

    Hỗ trợ:
    - Semantic search (vector similarity)
    - Keyword search (BM25-like)
    - Hybrid search (kết hợp cả hai)

    Sử dụng:
        >>> csdl = CSDLVector(kich_thuoc=128)
        >>> trich_xuat = TrichXuat(csdl, ham_embed=my_embed_fn)
        >>> trich_xuat.them_tai_lieu("doc.txt", van_ban)
        >>> ket_qua = trich_xuat.tim_kiem("câu hỏi", top_k=5)
    """

    def __init__(
        self,
        csdl_vector: Optional[CSDLVector] = None,
        ham_embed: Optional[Callable[[str], np.ndarray]] = None,
        cat_van_ban: Optional[CatVanBan] = None,
        che_do: str = "hybrid",
        trong_so_semantic: float = 0.7,
    ):
        if che_do not in ("semantic", "keyword", "hybrid"):
            raise ValueError("che_do phải là: semantic, keyword, hybrid")

        self.csdl_vector = csdl_vector or CSDLVector(kich_thuoc=128)
        self.ham_embed = ham_embed
        self.cat_van_ban = cat_van_ban or CatVanBan()
        self.che_do = che_do
        self.trong_so_semantic = trong_so_semantic

        self._van_ban_goc: Dict[str, str] = {}
        self._tu_dien: Dict[str, Dict[str, int]] = {}
        self._df: Dict[str, int] = {}
        self._tong_tai_lieu = 0

    def them_tai_lieu(
        self,
        ma: str,
        van_ban: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Thêm tài liệu vào retriever.

        Args:
            ma: ID tài liệu
            van_ban: Nội dung tài liệu
            metadata: Metadata bổ sung

        Returns:
            Số chunks được tạo
        """
        self._van_ban_goc[ma] = van_ban
        metadata = metadata or {}
        metadata["tai_lieu_goc"] = ma

        chunks = self.cat_van_ban.chia(van_ban, metadata)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{ma}_chunk_{i}"

            # BM25 indexing
            tu_list = chunk["noi_dung"].lower().split()
            self._tu_dien[chunk_id] = {}
            for tu in tu_list:
                self._tu_dien[chunk_id][tu] = self._tu_dien[chunk_id].get(tu, 0) + 1

            for tu in set(tu_list):
                self._df[tu] = self._df.get(tu, 0) + 1

            self._tong_tai_lieu += 1

            # Vector embedding
            if self.ham_embed is not None:
                vector = self.ham_embed(chunk["noi_dung"])
                self.csdl_vector.chen(chunk_id, vector, chunk["metadata"])
            else:
                # Fallback: TF-IDF-like vector
                vector = self._tao_vector_don_gian(chunk["noi_dung"])
                self.csdl_vector.chen(chunk_id, vector, chunk["metadata"])

        return len(chunks)

    def them_nhieu_tai_lieu(
        self,
        tai_lieu: List[Dict[str, Any]],
    ) -> int:
        """
        Thêm nhiều tài liệu cùng lúc.

        Args:
            tai_lieu: [{ma, van_ban, metadata}, ...]

        Returns:
            Tổng số chunks
        """
        tong = 0
        for tl in tai_lieu:
            tong += self.them_tai_lieu(
                tl["ma"], tl["van_ban"], tl.get("metadata")
            )
        return tong

    def tim_kiem(
        self,
        cau_hoi: str,
        top_k: int = 5,
        nguong: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm thông tin liên quan đến câu hỏi.

        Args:
            cau_hoi: Câu hỏi truy vấn
            top_k: Số kết quả
            nguong: Ngưỡng điểm tối thiểu

        Returns:
            [{noi_dung, diem, metadata, tai_lieu_goc}, ...]
        """
        if self.che_do == "semantic":
            return self._tim_semantic(cau_hoi, top_k, nguong)
        elif self.che_do == "keyword":
            return self._tim_keyword(cau_hoi, top_k, nguong)
        else:
            return self._tim_hybrid(cau_hoi, top_k, nguong)

    def _tim_semantic(
        self, cau_hoi: str, top_k: int, nguong: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm semantic."""
        if self.ham_embed:
            query_vec = self.ham_embed(cau_hoi)
        else:
            query_vec = self._tao_vector_don_gian(cau_hoi)

        ket_qua = self.csdl_vector.tim_kiem(query_vec, top_k=top_k, nguong=nguong)

        for kq in ket_qua:
            kq["loai"] = "semantic"

        return ket_qua

    def _tim_keyword(
        self, cau_hoi: str, top_k: int, nguong: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm keyword (BM25-like)."""
        tu_query = cau_hoi.lower().split()
        diem_map: Dict[str, float] = {}

        k1 = 1.5
        b = 0.75
        avgdl = self._tinh_avgdl()

        for chunk_id, term_freq in self._tu_dien.items():
            dl = sum(term_freq.values())
            diem = 0.0

            for tu in tu_query:
                if tu not in term_freq:
                    continue

                tf = term_freq[tu]
                df = self._df.get(tu, 0)
                idf = max(
                    0,
                    np.log((self._tong_tai_lieu - df + 0.5) / (df + 0.5) + 1),
                )

                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                diem += idf * tf_norm

            if diem > 0:
                diem_map[chunk_id] = diem

        # Normalize scores
        if diem_map:
            max_diem = max(diem_map.values())
            if max_diem > 0:
                diem_map = {k: v / max_diem for k, v in diem_map.items()}

        # Sắp xếp
        sorted_items = sorted(diem_map.items(), key=lambda x: x[1], reverse=True)

        ket_qua = []
        for chunk_id, diem in sorted_items[:top_k]:
            if nguong is not None and diem < nguong:
                break
            meta = self.csdl_vector.lay_metadata(chunk_id) or {}
            ket_qua.append({
                "ma": chunk_id,
                "diem": diem,
                "metadata": meta,
                "loai": "keyword",
            })

        return ket_qua

    def _tim_hybrid(
        self, cau_hoi: str, top_k: int, nguong: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm hybrid (semantic + keyword)."""
        ket_qua_semantic = self._tim_semantic(cau_hoi, top_k * 2, None)
        ket_qua_keyword = self._tim_keyword(cau_hoi, top_k * 2, None)

        # Kết hợp điểm
        diem_map: Dict[str, Dict[str, Any]] = {}

        for kq in ket_qua_semantic:
            ma = kq["ma"]
            diem_map[ma] = {
                "ma": ma,
                "diem_semantic": kq["diem"],
                "diem_keyword": 0.0,
                "metadata": kq["metadata"],
            }

        for kq in ket_qua_keyword:
            ma = kq["ma"]
            if ma in diem_map:
                diem_map[ma]["diem_keyword"] = kq["diem"]
            else:
                diem_map[ma] = {
                    "ma": ma,
                    "diem_semantic": 0.0,
                    "diem_keyword": kq["diem"],
                    "metadata": kq["metadata"],
                }

        # Tính điểm hybrid
        ket_qua = []
        for ma, info in diem_map.items():
            diem = (
                self.trong_so_semantic * info["diem_semantic"]
                + (1 - self.trong_so_semantic) * info["diem_keyword"]
            )
            if nguong is not None and diem < nguong:
                continue
            ket_qua.append({
                "ma": ma,
                "diem": diem,
                "metadata": info["metadata"],
                "loai": "hybrid",
            })

        ket_qua.sort(key=lambda x: x["diem"], reverse=True)
        return ket_qua[:top_k]

    def _tao_vector_don_gian(self, van_ban: str) -> np.ndarray:
        """Tạo vector đơn giản (TF-IDF-like) khi không có embedding model."""
        kich_thuoc = self.csdl_vector.kich_thuoc
        tu_list = van_ban.lower().split()

        vector = np.zeros(kich_thuoc, dtype=np.float32)
        for tu in tu_list:
            idx = hash(tu) % kich_thuoc
            vector[idx] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector

    def _tinh_avgdl(self) -> float:
        """Tính độ dài trung bình của documents."""
        if not self._tu_dien:
            return 1.0
        tong = sum(sum(tf.values()) for tf in self._tu_dien.values())
        return tong / max(len(self._tu_dien), 1)

    def xoa_tai_lieu(self, ma: str) -> int:
        """Xóa tài liệu và tất cả chunks liên quan."""
        if ma not in self._van_ban_goc:
            return 0

        del self._van_ban_goc[ma]
        da_xoa = 0

        keys_to_remove = [
            k for k in self._tu_dien.keys()
            if k.startswith(f"{ma}_chunk_")
        ]
        for key in keys_to_remove:
            del self._tu_dien[key]
            self.csdl_vector.xoa(key)
            da_xoa += 1

        self._tong_tai_lieu -= da_xoa
        return da_xoa

    def so_tai_lieu(self) -> int:
        """Số tài liệu trong retriever."""
        return len(self._van_ban_goc)

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê retriever."""
        return {
            "so_tai_lieu": len(self._van_ban_goc),
            "so_chunks": self._tong_tai_lieu,
            "che_do": self.che_do,
            "trong_so_semantic": self.trong_so_semantic,
            "csdl_vector": self.csdl_vector.thong_ke(),
        }

    def __repr__(self) -> str:
        return (
            f"TrichXuat(so_tai_lieu={len(self._van_ban_goc)}, "
            f"so_chunks={self._tong_tai_lieu}, "
            f"che_do='{self.che_do}')"
        )

class IdentityAwareRetriever(TrichXuat):
    """
    Retriever hỗ trợ phân quyền (RBAC) cho dữ liệu RAG.
    Đảm bảo LLM chỉ tìm kiếm trên các dữ liệu mà người dùng được phép truy cập.
    """

    def tim_kiem(
        self,
        cau_hoi: str,
        top_k: int = 5,
        nguong: Optional[float] = None,
        required_roles: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm với bộ lọc quyền truy cập.

        Args:
            cau_hoi: Câu hỏi truy vấn.
            top_k: Số kết quả.
            nguong: Ngưỡng điểm.
            required_roles: Danh sách vai trò của người dùng hiện tại (vd: ['admin']).
                            Nếu tài liệu có metadata 'allowed_roles', người dùng phải có
                            ít nhất 1 role khớp để được đọc.
        """
        ket_qua_tho = super().tim_kiem(cau_hoi, top_k * 3, nguong)

        # Nếu không yêu cầu kiểm tra quyền, trả về top_k kết quả
        if required_roles is None:
            return ket_qua_tho[:top_k]

        ket_qua_da_loc = []
        for kq in ket_qua_tho:
            meta = kq.get("metadata", {})
            allowed = meta.get("allowed_roles")

            # Nếu tài liệu không giới hạn quyền, hoặc người dùng có quyền phù hợp
            if allowed is None or any(role in allowed for role in required_roles):
                ket_qua_da_loc.append(kq)

            if len(ket_qua_da_loc) >= top_k:
                break

        return ket_qua_da_loc
