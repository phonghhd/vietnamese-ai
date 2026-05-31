"""CSDL Vector - lưu trữ và tìm kiếm vector embeddings."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class CSDLVector:
    """
    Cơ sở dữ liệu vector in-memory cho RAG.

    Hỗ trợ cosine similarity, L2 distance và inner product.
    Có thể lưu/tải từ file JSON hoặc pickle.

    Sử dụng:
        >>> csdl = CSDLVector(kich_thuoc=128, khoang_cach="cosine")
        >>> csdl.chen("doc_1", vector_1, {"noi_dung": "Hello world"})
        >>> csdl.chen("doc_2", vector_2, {"noi_dung": "Xin chao"})
        >>> ket_qua = csdl.tim_kiem(query_vector, top_k=5)
    """

    def __init__(
        self,
        kich_thuoc: int = 128,
        khoang_cach: str = "cosine",
        suc_chua_toi_da: int = 100000,
    ):
        if khoang_cach not in ("cosine", "l2", "inner_product"):
            raise ValueError("khoang_cach phải là: cosine, l2, inner_product")

        self.kich_thuoc = kich_thuoc
        self.khoang_cach = khoang_cach
        self.suc_chua_toi_da = suc_chua_toi_da

        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}

    def chen(
        self,
        ma: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Chen một vector vào CSDL."""
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if vector.shape[0] != self.kich_thuoc:
            raise ValueError(f"Vector phải có {self.kich_thuoc} chiều, nhận được {vector.shape[0]}")

        if len(self._ids) >= self.suc_chua_toi_da:
            raise RuntimeError(
                f"Đã đạt giới hạn {self.suc_chua_toi_da} vectors. "
                "Tăng suc_chua_toi_da hoặc xóa bớt."
            )

        if ma in self._id_to_idx:
            idx = self._id_to_idx[ma]
            self._vectors[idx] = vector
            self._metadata[idx] = metadata or {}
            return

        idx = len(self._ids)
        self._ids.append(ma)
        self._metadata.append(metadata or {})
        self._id_to_idx[ma] = idx

        if self._vectors is None:
            self._vectors = vector.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])

    def chen_batch(
        self,
        ma_list: List[str],
        vectors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Chen nhiều vector cùng lúc."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.kich_thuoc:
            raise ValueError(
                f"Vector phải có {self.kich_thuoc} chiều, nhận được {vectors.shape[1]}"
            )
        if metadata_list is None:
            metadata_list = [{}] * len(ma_list)

        for ma, vec, meta in zip(ma_list, vectors, metadata_list):
            self.chen(ma, vec, meta)

    def xoa(self, ma: str) -> bool:
        """Xóa một vector theo ID."""
        if ma not in self._id_to_idx:
            return False

        idx = self._id_to_idx[ma]
        del self._id_to_idx[ma]
        self._ids.pop(idx)
        self._metadata.pop(idx)

        if self._vectors is not None:
            self._vectors = np.delete(self._vectors, idx, axis=0)
            if len(self._ids) == 0:
                self._vectors = None

        self._id_to_idx = {ma: i for i, ma in enumerate(self._ids)}
        return True

    def tim_kiem(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        nguong: Optional[float] = None,
        bo_loc: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm vector gần nhất.

        Args:
            query_vector: Vector truy vấn
            top_k: Số kết quả trả về
            nguong: Ngưỡng điểm tối thiểu
            bo_loc: Metadata filter (key-value match)

        Returns:
            Danh sách kết quả [{ma, diem, metadata}, ...]
        """
        if self._vectors is None or len(self._ids) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).flatten()
        if query.shape[0] != self.kich_thuoc:
            raise ValueError(f"Query vector phải có {self.kich_thuoc} chiều")

        if self.khoang_cach == "cosine":
            scores = self._cosine_scores(query)
        elif self.khoang_cach == "l2":
            scores = self._l2_scores(query)
        else:
            scores = self._vectors @ query

        # Sắp xếp giảm dần
        chi_so = np.argsort(scores)[::-1]

        ket_qua = []
        for idx in chi_so:
            if len(ket_qua) >= top_k:
                break
            idx = int(idx)
            diem = float(scores[idx])

            if nguong is not None and diem < nguong:
                break

            if bo_loc is not None:
                meta = self._metadata[idx]
                if not all(meta.get(k) == v for k, v in bo_loc.items()):
                    continue

            ket_qua.append(
                {
                    "ma": self._ids[idx],
                    "diem": diem,
                    "metadata": self._metadata[idx],
                }
            )

        return ket_qua

    def lay_vector(self, ma: str) -> Optional[np.ndarray]:
        """Lấy vector theo ID."""
        if ma not in self._id_to_idx:
            return None
        return self._vectors[self._id_to_idx[ma]].copy()

    def lay_metadata(self, ma: str) -> Optional[Dict[str, Any]]:
        """Lấy metadata theo ID."""
        if ma not in self._id_to_idx:
            return None
        return self._metadata[self._id_to_idx[ma]].copy()

    def so_luong(self) -> int:
        """Số lượng vector trong CSDL."""
        return len(self._ids)

    def danh_sach_ma(self) -> List[str]:
        """Danh sách tất cả ID."""
        return self._ids.copy()

    def _cosine_scores(self, query: np.ndarray) -> np.ndarray:
        """Tính cosine similarity scores."""
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return np.zeros(len(self._ids))

        vec_norms = np.linalg.norm(self._vectors, axis=1)
        vec_norms = np.maximum(vec_norms, 1e-10)

        return (self._vectors @ query) / (vec_norms * query_norm)

    def _l2_scores(self, query: np.ndarray) -> np.ndarray:
        """Tính L2 distance scores (chuyển thành similarity)."""
        diffs = self._vectors - query.reshape(1, -1)
        distances = np.linalg.norm(diffs, axis=1)
        return 1.0 / (1.0 + distances)

    def xoa_tat_ca(self) -> None:
        """Xóa toàn bộ vector."""
        self._ids.clear()
        self._vectors = None
        self._metadata.clear()
        self._id_to_idx.clear()

    def luu(self, duong_dan: str) -> None:
        """Lưu CSDL vector ra file pickle."""
        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "kich_thuoc": self.kich_thuoc,
            "khoang_cach": self.khoang_cach,
            "ids": self._ids,
            "vectors": self._vectors,
            "metadata": self._metadata,
        }
        with open(duong_dan, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def tai(cls, duong_dan: str) -> "CSDLVector":
        """Tải CSDL vector từ file."""
        try:
            with open(duong_dan, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Lỗi đọc file pickle: {e}")

        csdl = cls(
            kich_thuoc=data["kich_thuoc"],
            khoang_cach=data["khoang_cach"],
        )
        csdl._ids = data["ids"]
        csdl._vectors = data["vectors"]
        csdl._metadata = data["metadata"]
        csdl._id_to_idx = {ma: i for i, ma in enumerate(csdl._ids)}
        return csdl

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê CSDL vector."""
        stats = {
            "so_luong": len(self._ids),
            "kich_thuoc": self.kich_thuoc,
            "khoang_cach": self.khoang_cach,
            "suc_chua_toi_da": self.suc_chua_toi_da,
        }
        if self._vectors is not None and len(self._ids) > 0:
            norms = np.linalg.norm(self._vectors, axis=1)
            stats["vector_norm_tb"] = float(np.mean(norms))
            stats["vector_norm_min"] = float(np.min(norms))
            stats["vector_norm_max"] = float(np.max(norms))
        return stats

    def __repr__(self) -> str:
        return (
            f"CSDLVector(so_luong={len(self._ids)}, "
            f"kich_thuoc={self.kich_thuoc}, "
            f"khoang_cach='{self.khoang_cach}')"
        )
