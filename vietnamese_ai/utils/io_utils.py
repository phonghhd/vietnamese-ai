"""IO Utils - Lưu và tải dữ liệu, mô hình."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class LuuTai:
    """
    Công cụ lưu/tải dữ liệu và mô hình.

    Sử dụng:
        >>> LuuTai.luu_mo_hinh(mo_hinh, "model.pkl")
        >>> mo_hinh = LuuTai.tai_mo_hinh("model.pkl")
        >>> LuuTai.luu_numpy(arr, "data.npz", key=arr)
        >>> data = LuuTai.tai_numpy("data.npz")
    """

    @staticmethod
    def luu_mo_hinh(mo_hinh: Any, duong_dan: str) -> None:
        """Lưu mô hình ra file pickle."""
        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan, "wb") as f:
            pickle.dump(mo_hinh, f)

    @staticmethod
    def tai_mo_hinh(duong_dan: str) -> Any:
        """Tải mô hình từ file pickle."""
        with open(duong_dan, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def luu_numpy(duong_dan: str, **matran: np.ndarray) -> None:
        """Lưu nhiều mảng numpy vào một file .npz."""
        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(duong_dan, **matran)

    @staticmethod
    def tai_numpy(duong_dan: str) -> Dict[str, np.ndarray]:
        """Tải mảng numpy từ file .npz."""
        data = np.load(duong_dan)
        return dict(data)

    @staticmethod
    def luu_json(data: Dict[str, Any], duong_dan: str) -> None:
        """Lưu dữ liệu ra file JSON."""
        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def tai_json(duong_dan: str) -> Dict[str, Any]:
        """Tải dữ liệu từ file JSON."""
        with open(duong_dan, "r", encoding="utf-8") as f:
            return json.load(f)
