"""Marketplace - Nơi chia sẻ mô hình, datasets, pipelines."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from vietnamese_ai.utils.logger import Logger


class Marketplace:
    """
    Vietnamese AI Marketplace - Nơi chia sẻ và khám phá resources.

    Tính năng:
    - Đăng ký mô hình, datasets, pipelines
    - Tìm kiếm theo category, tags, rating
    - Đánh giá và review
    - Version management
    - Download tracking

    Sử dụng:
        >>> mp = Marketplace()
        >>> mp.dang_ky(ten="sentiment_vi", loai="model", tac_gia="EvoNet",
        ...            tags=["sentiment", "nlp"], mo_ta="Mô hình phân tích cảm xúc")
        >>> ds = mp.tim_kiem(category="model", tags=["sentiment"])
    """

    CATEGORIES = ["model", "dataset", "pipeline", "embedding", "tool"]

    def __init__(self, duong_dan: str = "marketplace"):
        self.duong_dan = Path(duong_dan)
        self.duong_dan.mkdir(parents=True, exist_ok=True)
        self.logger = Logger("Marketplace")
        self._catalog_file = self.duong_dan / "marketplace.json"
        self._catalog = self._tai_catalog()

    def _tai_catalog(self) -> Dict:
        if self._catalog_file.exists():
            with open(self._catalog_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"items": {}}

    def _luu_catalog(self) -> None:
        with open(self._catalog_file, "w", encoding="utf-8") as f:
            json.dump(self._catalog, f, ensure_ascii=False, indent=2)

    def dang_ky(
        self,
        ten: str,
        loai: str,
        tac_gia: str = "anonymous",
        mo_ta: str = "",
        tags: Optional[List[str]] = None,
        version: str = "1.0.0",
        gia: float = 0.0,
    ) -> str:
        """
        Đăng ký resource lên marketplace.

        Args:
            ten: Tên resource
            loai: Loại (model, dataset, pipeline, embedding, tool)
            tac_gia: Tác giả
            mo_ta: Mô tả
            tags: Thẻ gắn nhãn
            version: Phiên bản
            gia: Giá (0 = miễn phí)

        Returns:
            Item ID
        """
        if loai not in self.CATEGORIES:
            raise ValueError(f"Loại không hợp lệ. Chọn: {self.CATEGORIES}")

        item_id = f"{tac_gia}/{ten}"
        self._catalog["items"][item_id] = {
            "id": item_id,
            "ten": ten,
            "loai": loai,
            "tac_gia": tac_gia,
            "mo_ta": mo_ta,
            "tags": tags or [],
            "version": version,
            "gia": gia,
            "luot_tai": 0,
            "danh_gia_sao": 0,
            "so_danh_gia": 0,
            "reviews": [],
        }
        self._luu_catalog()
        self.logger.info(f"Đã đăng ký: {item_id} ({loai})")
        return item_id

    def tim_kiem(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tac_gia: Optional[str] = None,
        sort_by: str = "popularity",
    ) -> List[Dict]:
        """Tìm kiếm resources."""
        ket_qua = []

        for item_id, info in self._catalog["items"].items():
            if category and info["loai"] != category:
                continue
            if tac_gia and info["tac_gia"] != tac_gia:
                continue
            if tags and not any(t in info["tags"] for t in tags):
                continue
            if query:
                q = query.lower()
                if q not in info["ten"].lower() and q not in info["mo_ta"].lower():
                    continue

            ket_qua.append(info.copy())

        if sort_by == "popularity":
            ket_qua.sort(key=lambda x: x["luot_tai"], reverse=True)
        elif sort_by == "rating":
            ket_qua.sort(key=lambda x: x["danh_gia_sao"], reverse=True)
        elif sort_by == "newest":
            ket_qua.reverse()

        return ket_qua

    def danh_gia(self, item_id: str, sao: int, binh_luan: str = "", nguoi_danh_gia: str = "anonymous") -> None:
        """Đánh giá resource."""
        if item_id not in self._catalog["items"]:
            raise KeyError(f"Không tìm thấy: {item_id}")

        item = self._catalog["items"][item_id]
        item["reviews"].append({
            "sao": sao,
            "binh_luan": binh_luan,
            "nguoi_danh_gia": nguoi_danh_gia,
        })

        tong_sao = sum(r["sao"] for r in item["reviews"])
        item["so_danh_gia"] = len(item["reviews"])
        item["danh_gia_sao"] = round(tong_sao / item["so_danh_gia"], 1)
        self._luu_catalog()

    def thong_ke(self) -> Dict:
        """Thống kê marketplace."""
        items = self._catalog["items"]
        theo_loai = {}
        for info in items.values():
            loai = info["loai"]
            theo_loai[loai] = theo_loai.get(loai, 0) + 1

        return {
            "tong_items": len(items),
            "theo_loai": theo_loai,
            "tong_luot_tai": sum(i["luot_tai"] for i in items.values()),
        }

    def danh_sach(self, loai: Optional[str] = None) -> List[Dict]:
        """Liệt kê tất cả resources."""
        return self.tim_kiem(category=loai)

    def xoa(self, item_id: str) -> None:
        """Xóa resource."""
        if item_id in self._catalog["items"]:
            del self._catalog["items"][item_id]
            self._luu_catalog()
