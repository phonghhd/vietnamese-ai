"""Validator - Trình kiểm tra dữ liệu."""

from typing import Any, Optional, Tuple

import numpy as np


class Validator:
    """
    Trình kiểm tra tính hợp lệ của dữ liệu.

    Sử dụng:
        >>> Validator.kiem_tra_kich_thuoc(X, (100, 5))
        >>> Validator.kiem_tra_gia_tri_thieu(X)
    """

    @staticmethod
    def kiem_tra_kich_thuoc(
        data: np.ndarray, kich_thuoc_mong_duoi: Tuple[int, ...]
    ) -> bool:
        """Kiểm tra kích thước dữ liệu có khớp mong đợi không."""
        data = np.asarray(data)
        return data.shape == kich_thuoc_mong_duoi

    @staticmethod
    def kiem_tra_gia_tri_thieu(data: np.ndarray) -> bool:
        """Kiểm tra dữ liệu có chứa giá trị NaN không."""
        data = np.asarray(data, dtype=float)
        return bool(np.isnan(data).any())

    @staticmethod
    def kiem_tra_loai_du_lieu(data: Any, loai_mong_duoi: type) -> bool:
        """Kiểm tra loại dữ liệu."""
        return isinstance(data, loai_mong_duoi)

    @staticmethod
    def kiem_tra_du_lieu_hop_le(
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        cho_phep_nan: bool = False,
        cho_phep_am: bool = True,
    ) -> Tuple[bool, str]:
        """
        Kiểm tra tính hợp lệ tổng thể của dữ liệu.

        Args:
            X: Dữ liệu đầu vào
            y: Nhãn (tùy chọn)
            cho_phep_nan: Có cho phép giá trị NaN không
            cho_phep_am: Có cho phép giá trị âm không

        Returns:
            (hop_le, thong_bao)
        """
        X = np.asarray(X, dtype=float)

        if X.ndim < 2:
            return False, "X phải là mảng 2D (n_samples, n_features)"

        if not cho_phep_nan and np.isnan(X).any():
            return False, "Dữ liệu chứa giá trị NaN"

        if not cho_phep_am and (X < 0).any():
            return False, "Dữ liệu chứa giá trị âm"

        if y is not None:
            y = np.asarray(y)
            if len(X) != len(y):
                return False, f"Số mẫu X ({len(X)}) != số nhãn y ({len(y)})"

        return True, "Dữ liệu hợp lệ"

    @staticmethod
    def kiem_tra_nhiem_vu(y: np.ndarray) -> str:
        """Tự động phát hiện nhiệm vụ: phân loại hay hồi quy."""
        y = np.asarray(y)
        if y.dtype.kind in ("i", "u", "b", "S", "U"):
            return "phan_loai"
        if y.dtype.kind == "f" and len(np.unique(y)) < 20 and np.all(y == y.astype(int)):
            return "phan_loai"
        return "hoi_quy"
