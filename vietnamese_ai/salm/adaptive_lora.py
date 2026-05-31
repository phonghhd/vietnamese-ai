"""AdaptiveLoRA - tự chọn và kết hợp LoRA adapters theo task."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class AdaptiveLoRA:
    """
    Adaptive LoRA: tự chọn adapter phù hợp dựa trên input task.

    Thay vì dùng 1 LoRA cho mọi task, hệ thống:
    1. Phân loại task từ input
    2. Chọn hoặc kết hợp nhiều LoRA adapters
    3. Dynamic routing dựa trên input features

    Dựa trên concept "Mixture of LoRAs" và "AdaLoRA".

    Sử dụng:
        >>> adaptive = AdaptiveLoRA()
        >>> adaptive.dang_ky_adapter("math", lora_math, ["tính", "cộng", "trừ"])
        >>> adaptive.dang_ky_adapter("code", lora_code, ["code", "python", "function"])
        >>> adapter = adaptive.chon_adapter("Tính tổng 2 số")
    """

    def __init__(
        self,
        che_do: str = "keyword",
        trong_so_mac_dinh: float = 1.0,
    ):
        if che_do not in ("keyword", "embedding", "hybrid"):
            raise ValueError("che_do phải là: keyword, embedding, hybrid")

        self.che_do = che_do
        self.trong_so_mac_dinh = trong_so_mac_dinh
        self.logger = Logger("AdaptiveLoRA")

        self._adapters: Dict[str, Dict[str, Any]] = {}
        self._keywords: Dict[str, List[str]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._usage_stats: Dict[str, int] = {}
        self._ham_embed: Optional[Callable[[str], np.ndarray]] = None

    def dang_ky_adapter(
        self,
        ten: str,
        adapter: Any,
        keywords: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None,
        trong_so: float = 1.0,
    ) -> None:
        """
        Đăng ký một LoRA adapter.

        Args:
            ten: Tên adapter
            adapter: LoRA adapter object
            keywords: Từ khóa liên quan đến task
            embedding: Vector biểu diễn task
            trong_so: Trọng số ưu tiên
        """
        self._adapters[ten] = {
            "adapter": adapter,
            "trong_so": trong_so,
        }
        self._keywords[ten] = keywords or []
        self._usage_stats[ten] = 0

        if embedding is not None:
            self._embeddings[ten] = np.asarray(embedding, dtype=np.float32)

    def dang_ky_ham_embed(self, ham: Callable[[str], np.ndarray]) -> None:
        """Đăng ký hàm embedding cho semantic routing."""
        self._ham_embed = ham

    def chon_adapter(
        self,
        input_text: str,
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Chọn adapter phù hợp nhất cho input.

        Args:
            input_text: Văn bản đầu vào
            top_k: Số adapter trả về

        Returns:
            [{ten, adapter, diem, trong_so}, ...]
        """
        if not self._adapters:
            return []

        diem_map: Dict[str, float] = {}

        if self.che_do == "keyword":
            diem_map = self._tinh_diem_keyword(input_text)
        elif self.che_do == "embedding":
            diem_map = self._tinh_diem_embedding(input_text)
        else:
            kw_scores = self._tinh_diem_keyword(input_text)
            emb_scores = self._tinh_diem_embedding(input_text)
            for ten in self._adapters:
                diem_map[ten] = 0.6 * kw_scores.get(ten, 0) + 0.4 * emb_scores.get(ten, 0)

        # Áp dụng trọng số
        for ten in diem_map:
            diem_map[ten] *= self._adapters[ten]["trong_so"]

        # Sắp xếp từ cao xuống thấp
        sorted_adapters = sorted(diem_map.items(), key=lambda x: x[1], reverse=True)

        # [VÁ BUG TẠI ĐÂY] 🛡️ Lọc bỏ các chuyên gia bị 0 điểm
        danh_sach_hop_le = [(ten, diem) for ten, diem in sorted_adapters if diem > 0]

        # Nếu không có ai qua bài test (hoặc người dùng hỏi vu vơ) -> Dùng Base Model
        if not danh_sach_hop_le:
            return []

        ket_qua = []
        # Chỉ lấy top_k từ danh sách ĐÃ HỢP LỆ
        for ten, diem in danh_sach_hop_le[:top_k]:
            self._usage_stats[ten] += 1
            ket_qua.append(
                {
                    "ten": ten,
                    "adapter": self._adapters[ten]["adapter"],
                    "diem": round(diem, 4),
                    "trong_so": self._adapters[ten]["trong_so"],
                }
            )

        return ket_qua

    def ket_hop_adapters(
        self,
        input_text: str,
        trong_so_combine: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Tính trọng số kết hợp cho tất cả adapters.

        Args:
            input_text: Văn bản đầu vào
            trong_so_combine: Trọng số tùy chỉnh

        Returns:
            {ten_adapter: trong_so, ...}
        """
        chon = self.chon_adapter(input_text, top_k=len(self._adapters))

        if trong_so_combine:
            return trong_so_combine

        tong_diem = sum(c["diem"] for c in chon)
        if tong_diem == 0:
            return {c["ten"]: 1.0 / len(chon) for c in chon}

        return {c["ten"]: c["diem"] / tong_diem for c in chon}

    def _tinh_diem_keyword(self, text: str) -> Dict[str, float]:
        """Tính điểm dựa trên keyword matching."""
        text_lower = text.lower()
        diem = {}

        for ten, keywords in self._keywords.items():
            if not keywords:
                diem[ten] = 0.1  # Base score
                continue

            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            diem[ten] = matches / max(len(keywords), 1)

        return diem

    def _tinh_diem_embedding(self, text: str) -> Dict[str, float]:
        """Tính điểm dựa trên embedding similarity."""
        if not self._embeddings:
            return {ten: 0.1 for ten in self._adapters}

        if self._ham_embed:
            text_vec = self._ham_embed(text)
        else:
            # Fallback: hash-based vector
            text_vec = np.zeros(128, dtype=np.float32)
            for tu in text.lower().split():
                text_vec[hash(tu) % 128] += 1.0
            norm = np.linalg.norm(text_vec)
            if norm > 0:
                text_vec /= norm

        diem = {}
        for ten, emb in self._embeddings.items():
            if len(text_vec) != len(emb):
                diem[ten] = 0.1
                continue
            sim = float(
                np.dot(text_vec, emb) / (np.linalg.norm(text_vec) * np.linalg.norm(emb) + 1e-10)
            )
            diem[ten] = max(sim, 0)

        # Adapter không có embedding
        for ten in self._adapters:
            if ten not in diem:
                diem[ten] = 0.05

        return diem

    def xoa_adapter(self, ten: str) -> bool:
        """Xóa adapter."""
        if ten in self._adapters:
            del self._adapters[ten]
            self._keywords.pop(ten, None)
            self._embeddings.pop(ten, None)
            self._usage_stats.pop(ten, None)
            return True
        return False

    def danh_sach_adapters(self) -> List[str]:
        """Danh sách adapters."""
        return list(self._adapters.keys())

    def thong_ke_su_dung(self) -> Dict[str, int]:
        """Thống kê tần suất sử dụng adapters."""
        return self._usage_stats.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "so_adapters": len(self._adapters),
            "che_do": self.che_do,
            "co_embedding": bool(self._embeddings),
            "co_ham_embed": self._ham_embed is not None,
            "su_dung": self._usage_stats,
        }

    def __repr__(self) -> str:
        return f"AdaptiveLoRA(so_adapters={len(self._adapters)}, che_do='{self.che_do}')"
