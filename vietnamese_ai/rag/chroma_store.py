"""ChromaVectorStore - Lưu trữ và tìm kiếm vector bằng ChromaDB."""

from typing import Any, Dict, List, Optional

import numpy as np


class ChromaVectorStore:
    """
    Cơ sở dữ liệu vector sử dụng ChromaDB.
    Thích hợp cho môi trường dev và production vừa/nhỏ.

    Yêu cầu: pip install chromadb
    """

    def __init__(
        self,
        thu_muc_luu_tru: Optional[str] = "./chroma_db",
        ten_collection: str = "vietnamese_ai",
        khoang_cach: str = "cosine",
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Vui lòng cài đặt chromadb: pip install chromadb")

        # Map distance metric string to ChromaDB expected string
        # Chroma supports 'l2', 'ip' (inner product), 'cosine'
        space_map = {"cosine": "cosine", "l2": "l2", "inner_product": "ip"}
        self.space = space_map.get(khoang_cach, "cosine")

        if thu_muc_luu_tru:
            self.client = chromadb.PersistentClient(path=thu_muc_luu_tru)
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=ten_collection, metadata={"hnsw:space": self.space}
        )

    def chen(
        self,
        ma: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Chen một vector vào CSDL."""
        vec = np.asarray(vector, dtype=float).flatten().tolist()
        meta = metadata or {}
        # Ensure metadata values are str, int, float, or bool
        meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}

        self.collection.add(
            ids=[ma],
            embeddings=[vec],
            metadatas=[meta] if meta else None,  # type: ignore
            documents=[""],  # Chroma expects documents or embeddings, we just use embeddings
        )

    def chen_batch(
        self,
        ma_list: List[str],
        vectors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Chen nhiều vector cùng lúc."""
        vecs = np.asarray(vectors, dtype=float).tolist()

        metas = None
        if metadata_list:
            metas = []
            for m in metadata_list:
                meta = {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
                metas.append(meta)

        self.collection.add(
            ids=ma_list,
            embeddings=vecs,
            metadatas=metas,  # type: ignore
            documents=[""] * len(ma_list),
        )

    def xoa(self, ma: str) -> bool:
        """Xóa một vector theo ID."""
        try:
            self.collection.delete(ids=[ma])
            return True
        except Exception:
            return False

    def tim_kiem(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        nguong: Optional[float] = None,
        bo_loc: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm vector gần nhất."""
        vec = np.asarray(query_vector, dtype=float).flatten().tolist()

        # Build Chroma where filter
        where_filter = None
        if bo_loc:
            # Simple equal filter support for demonstration
            where_filter = bo_loc

        results = self.collection.query(
            query_embeddings=[vec],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        ket_qua = []
        if not results["ids"] or not results["ids"][0]:
            return ket_qua

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

        for i in range(len(ids)):
            diem = distances[i]
            # Convert distance to similarity if needed
            if self.space == "cosine" or self.space == "ip":
                # Chroma returns cosine distance (1 - cosine_sim).
                # Convert back to similarity for consistency with CSDLVector
                diem = 1.0 - diem

            if nguong is not None and diem < nguong:
                continue

            ket_qua.append({"ma": ids[i], "diem": float(diem), "metadata": metadatas[i] or {}})

        return ket_qua

    def so_luong(self) -> int:
        """Số lượng vector trong CSDL."""
        return self.collection.count()
