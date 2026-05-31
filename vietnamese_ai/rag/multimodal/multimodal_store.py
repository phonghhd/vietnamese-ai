from typing import Any, Dict, List

from .image_embedder import ImageEmbedder


class MultimodalStore:
    """
    Store chuyên dụng cho Multi-modal RAG (Text + Image).
    Quản lý đồng thời Vector Text (qua Chroma/Qdrant) và Vector Image.
    """

    def __init__(self, text_store: Any, image_embedder: ImageEmbedder):
        self.text_store = text_store
        self.image_embedder = image_embedder

        # CSDL vector ảnh cục bộ đơn giản (dict map ID -> vector)
        self.image_vectors: Dict[str, List[float]] = {}
        self.image_metadata: Dict[str, Dict[str, Any]] = {}

    def them_hinh_anh(self, image_paths: List[str], metadatas: List[Dict[str, Any]] = None):
        """Nhúng và lưu trữ ảnh."""
        vectors = self.image_embedder.nhung_hinh_anh(image_paths)

        for i, (path, vec) in enumerate(zip(image_paths, vectors)):
            doc_id = f"img_{len(self.image_vectors)}"
            self.image_vectors[doc_id] = vec

            meta = metadatas[i] if metadatas else {}
            meta["source_path"] = path
            self.image_metadata[doc_id] = meta

    def them_van_ban(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Lưu trữ văn bản (ủy quyền cho text_store)."""
        self.text_store.add_documents(texts, metadatas=metadatas)

    def tim_kiem_anh_tu_anh(self, image_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Truy xuất ảnh tương đồng bằng ảnh đầu vào (Image-to-Image)."""
        query_vec = self.image_embedder.nhung_hinh_anh([image_path])[0]
        return self._tim_kiem_cosine(query_vec, top_k)

    def tim_kiem_anh_tu_van_ban(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Truy xuất ảnh dựa trên mô tả văn bản (Text-to-Image)."""
        if not hasattr(self.image_embedder, "nhung_van_ban"):
            raise NotImplementedError("ImageEmbedder không hỗ trợ nhung_van_ban.")
        query_vec = self.image_embedder.nhung_van_ban([text])[0]
        return self._tim_kiem_cosine(query_vec, top_k)

    def tim_kiem_van_ban_tu_anh(self, image_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Truy xuất văn bản liên quan dựa trên ảnh (Image-to-Text)."""
        if not hasattr(self.text_store, "tim_kiem"):
            raise NotImplementedError("TextStore không hỗ trợ tim_kiem.")

        # Lấy vector của ảnh
        query_vec = self.image_embedder.nhung_hinh_anh([image_path])[0]

        # Tìm kiếm trên Text Store bằng vector (giả định text_store.tim_kiem hỗ trợ nhận vector)
        # Nếu text_store là CSDLVector:
        return self.text_store.tim_kiem(query_vec, top_k=top_k)

    def _tim_kiem_cosine(self, query_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Tìm kiếm cosine similarity đơn giản trên dictionary."""
        import numpy as np

        if not self.image_vectors:
            return []

        q_arr = np.array(query_vec)
        results = []
        for doc_id, vec in self.image_vectors.items():
            v_arr = np.array(vec)
            # Cosine similarity
            sim = np.dot(q_arr, v_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(v_arr))
            results.append((sim, doc_id))

        results.sort(key=lambda x: x[0], reverse=True)

        top_results = []
        for sim, doc_id in results[:top_k]:
            meta = self.image_metadata[doc_id].copy()
            meta["score"] = float(sim)
            top_results.append(meta)

        return top_results
