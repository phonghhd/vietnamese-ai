"""ModelHub - Nơi chia sẻ mô hình cộng đồng."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.utils.logger import Logger


class ModelHub:
    """
    Vietnamese AI Model Hub - Nơi chia sẻ và tải mô hình cộng đồng.

    Tính năng:
    - Đăng ký mô hình lên hub (local registry)
    - Tìm kiếm mô hình theo tên, tag, task
    - Tải mô hình từ hub
    - Đánh giá và đánh giá sao

    Sử dụng:
        >>> hub = ModelHub()
        >>> hub.dang_ky(mo_hinh, ten="sentiment_vi", tac_gia="EvoNet",
        ...            tags=["sentiment", "vietnamese"], mo_ta="Mô hình phân tích cảm xúc")
        >>> ds = hub.tim_kiem(tags=["sentiment"])
        >>> mo_hinh = hub.tai("sentiment_vi")
    """

    def __init__(self, duong_dan: str = "model_hub"):
        self.duong_dan = Path(duong_dan)
        self.duong_dan.mkdir(parents=True, exist_ok=True)
        self.logger = Logger("ModelHub")
        self._catalog_file = self.duong_dan / "catalog.json"
        self._catalog = self._tai_catalog()

    def _tai_catalog(self) -> Dict:
        if self._catalog_file.exists():
            with open(self._catalog_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"models": {}}

    def _luu_catalog(self) -> None:
        with open(self._catalog_file, "w", encoding="utf-8") as f:
            json.dump(self._catalog, f, ensure_ascii=False, indent=2)

    def dang_ky(
        self,
        mo_hinh: Any,
        ten: str,
        tac_gia: str = "anonymous",
        mo_ta: str = "",
        tags: Optional[List[str]] = None,
        version: str = "1.0.0",
    ) -> str:
        """
        Đăng ký mô hình lên hub.

        Args:
            mo_hinh: Đối tượng mô hình
            ten: Tên mô hình
            tac_gia: Tác giả
            mo_ta: Mô tả
            tags: Thẻ gắn nhãn
            version: Phiên bản

        Returns:
            ID mô hình
        """
        model_id = f"{tac_gia}/{ten}"
        duong_dan_model = self.duong_dan / model_id.replace("/", "_") / version
        duong_dan_model.mkdir(parents=True, exist_ok=True)

        mo_hinh.luu(str(duong_dan_model / "model.pkl"))

        ban_ghi = {
            "id": model_id,
            "ten": ten,
            "tac_gia": tac_gia,
            "mo_ta": mo_ta,
            "tags": tags or [],
            "version": version,
            "duong_dan": str(duong_dan_model / "model.pkl"),
            "luot_tai": 0,
            "danh_gia_sao": 0,
            "so_danh_gia": 0,
        }

        if model_id not in self._catalog["models"]:
            self._catalog["models"][model_id] = {}
        self._catalog["models"][model_id][version] = ban_ghi
        self._luu_catalog()

        self.logger.info(f"Đã đăng ký: {model_id} v{version}")
        return model_id

    def tai(self, model_id: str, version: Optional[str] = None) -> Any:
        """Tải mô hình từ hub."""
        if model_id not in self._catalog["models"]:
            raise KeyError(f"Không tìm thấy mô hình: {model_id}")

        versions = self._catalog["models"][model_id]
        if version is None:
            version = list(versions.keys())[-1]

        if version not in versions:
            raise KeyError(f"Không tìm thấy version {version}")

        ban_ghi = versions[version]
        ban_ghi["luot_tai"] += 1
        self._luu_catalog()

        from vietnamese_ai.models.base import BaseModel

        return BaseModel.tai(ban_ghi["duong_dan"])

    def tim_kiem(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tac_gia: Optional[str] = None,
    ) -> List[Dict]:
        """
        Tìm kiếm mô hình.

        Args:
            query: Từ khóa tìm kiếm
            tags: Lọc theo tags
            tac_gia: Lọc theo tác giả

        Returns:
            Danh sách mô hình phù hợp
        """
        ket_qua = []

        for model_id, versions in self._catalog["models"].items():
            latest_version = list(versions.values())[-1]

            if tac_gia and latest_version["tac_gia"] != tac_gia:
                continue
            if tags and not any(t in latest_version["tags"] for t in tags):
                continue
            if (
                query
                and query.lower() not in latest_version["ten"].lower()
                and query.lower() not in latest_version["mo_ta"].lower()
            ):
                continue

            ket_qua.append(
                {
                    "id": model_id,
                    "ten": latest_version["ten"],
                    "tac_gia": latest_version["tac_gia"],
                    "mo_ta": latest_version["mo_ta"],
                    "tags": latest_version["tags"],
                    "version": latest_version["version"],
                    "luot_tai": latest_version["luot_tai"],
                    "danh_gia_sao": latest_version["danh_gia_sao"],
                }
            )

        return sorted(ket_qua, key=lambda x: x["luot_tai"], reverse=True)

    def danh_gia_sao(self, model_id: str, sao: int) -> None:
        """Đánh giá mô hình (1-5 sao)."""
        if model_id not in self._catalog["models"]:
            raise KeyError(f"Không tìm thấy mô hình: {model_id}")

        versions = self._catalog["models"][model_id]
        latest = list(versions.values())[-1]

        tong_sao = latest["danh_gia_sao"] * latest["so_danh_gia"] + sao
        latest["so_danh_gia"] += 1
        latest["danh_gia_sao"] = round(tong_sao / latest["so_danh_gia"], 1)
        self._luu_catalog()

    def danh_sach(self) -> List[Dict]:
        """Liệt kê tất cả mô hình trong hub."""
        return self.tim_kiem()

    def xoa(self, model_id: str) -> None:
        """Xóa mô hình khỏi hub."""
        if model_id in self._catalog["models"]:
            del self._catalog["models"][model_id]
            self._luu_catalog()
            self.logger.info(f"Đã xóa: {model_id}")
