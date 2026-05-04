"""SapXepLai - reranking kết quả tìm kiếm cho RAG."""

from typing import Any, Callable, Dict, List, Optional


class SapXepLai:
    """
    Reranking kết quả tìm kiếm để cải thiện chất lượng RAG.

    Hỗ trợ:
    - Cross-encoder scoring (nếu có model)
    - MMR (Maximal Marginal Relevance) - đa dạng hóa kết quả
    - Keyword overlap scoring
    - Position-aware reranking

    Sử dụng:
        >>> reranker = SapXepLai(che_do="mmr")
        >>> ket_qua_moi = reranker.sap_xep_lai(cau_hoi, ket_qua_cu)
    """

    def __init__(
        self,
        che_do: str = "mmr",
        ham_score: Optional[Callable[[str, str], float]] = None,
        trong_so_da_dang: float = 0.5,
    ):
        if che_do not in ("mmr", "cross_encoder", "keyword", "position"):
            raise ValueError("che_do phải là: mmr, cross_encoder, keyword, position")

        self.che_do = che_do
        self.ham_score = ham_score
        self.trong_so_da_dang = trong_so_da_dang

    def sap_xep_lai(
        self,
        cau_hoi: str,
        ket_qua: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Sắp xếp lại kết quả tìm kiếm.

        Args:
            cau_hoi: Câu hỏi gốc
            ket_qua: Kết quả từ retriever
            top_k: Số kết quả trả về

        Returns:
            Danh sách kết quả đã sắp xếp lại
        """
        if not ket_qua:
            return []

        if self.che_do == "mmr":
            return self._mmr(cau_hoi, ket_qua, top_k)
        elif self.che_do == "cross_encoder":
            return self._cross_encoder(cau_hoi, ket_qua, top_k)
        elif self.che_do == "keyword":
            return self._keyword_rerank(cau_hoi, ket_qua, top_k)
        else:
            return self._position_rerank(ket_qua, top_k)

    def _mmr(
        self,
        cau_hoi: str,
        ket_qua: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Maximal Marginal Relevance - cân bằng relevance và diversity."""
        if len(ket_qua) <= top_k:
            return ket_qua

        selected: List[Dict[str, Any]] = []
        candidates = list(ket_qua)

        # Chọn kết quả đầu tiên (điểm cao nhất)
        candidates.sort(key=lambda x: x.get("diem", 0), reverse=True)
        selected.append(candidates.pop(0))

        while len(selected) < top_k and candidates:
            best_idx = -1
            best_score = -float("inf")

            for i, cand in enumerate(candidates):
                relevance = cand.get("diem", 0)

                # Tính max similarity với các kết quả đã chọn
                max_sim = 0.0
                cand_words = set(cand.get("metadata", {}).get("noi_dung", "").lower().split())
                if not cand_words:
                    cand_words = set(cand.get("noi_dung", "").lower().split())

                for sel in selected:
                    sel_words = set(sel.get("metadata", {}).get("noi_dung", "").lower().split())
                    if not sel_words:
                        sel_words = set(sel.get("noi_dung", "").lower().split())

                    if cand_words and sel_words:
                        sim = len(cand_words & sel_words) / max(
                            len(cand_words | sel_words), 1
                        )
                        max_sim = max(max_sim, sim)

                mmr_score = (
                    self.trong_so_da_dang * relevance
                    - (1 - self.trong_so_da_dang) * max_sim
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(candidates.pop(best_idx))
            else:
                break

        return selected

    def _cross_encoder(
        self,
        cau_hoi: str,
        ket_qua: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Cross-encoder scoring (cần custom ham_score)."""
        if self.ham_score is None:
            # Fallback to keyword scoring
            return self._keyword_rerank(cau_hoi, ket_qua, top_k)

        for kq in ket_qua:
            noi_dung = kq.get("metadata", {}).get("noi_dung", "")
            if not noi_dung:
                noi_dung = kq.get("noi_dung", "")
            kq["diem_rerank"] = self.ham_score(cau_hoi, noi_dung)

        ket_qua.sort(key=lambda x: x.get("diem_rerank", 0), reverse=True)
        return ket_qua[:top_k]

    def _keyword_rerank(
        self,
        cau_hoi: str,
        ket_qua: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Reranking dựa trên keyword overlap + original score."""
        query_words = set(cau_hoi.lower().split())

        for kq in ket_qua:
            noi_dung = kq.get("metadata", {}).get("noi_dung", "")
            if not noi_dung:
                noi_dung = kq.get("noi_dung", "")

            doc_words = set(noi_dung.lower().split())
            if doc_words and query_words:
                overlap = len(query_words & doc_words) / len(query_words)
            else:
                overlap = 0.0

            original_score = kq.get("diem", 0)
            kq["diem_rerank"] = 0.6 * original_score + 0.4 * overlap

        ket_qua.sort(key=lambda x: x.get("diem_rerank", 0), reverse=True)
        return ket_qua[:top_k]

    def _position_rerank(
        self,
        ket_qua: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Reranking theo vị trí (ưu tiên kết quả xuất hiện sớm trong tài liệu)."""
        for i, kq in enumerate(ket_qua):
            vi_tri = kq.get("metadata", {}).get("vi_tri_bat_dau", i * 1000)
            position_penalty = 1.0 / (1.0 + vi_tri / 10000.0)
            original_score = kq.get("diem", 0)
            kq["diem_rerank"] = 0.7 * original_score + 0.3 * position_penalty

        ket_qua.sort(key=lambda x: x.get("diem_rerank", 0), reverse=True)
        return ket_qua[:top_k]

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê cấu hình reranker."""
        return {
            "che_do": self.che_do,
            "trong_so_da_dang": self.trong_so_da_dang,
            "co_custom_score": self.ham_score is not None,
        }

    def __repr__(self) -> str:
        return f"SapXepLai(che_do='{self.che_do}')"
