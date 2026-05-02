"""Xử lý dữ liệu số học."""

from typing import Dict, Optional, Tuple

import numpy as np


class XuLySo:
    """
    Bộ xử lý dữ liệu số học.

    Tính năng:
    - Chuẩn hóa Min-Max
    - Chuẩn hóa Z-Score (Standardization)
    - Xử lý giá trị thiếu
    - Mã hóa one-hot
    - Mã hóa nhãn

    Sử dụng:
        >>> xl = XuLySo()
        >>> du_lieu_chuan = xl.chuan_hoa_minmax(du_lieu)
        >>> du_lieu_chuan = xl.chuan_hoa_zscore(du_lieu)
    """

    def __init__(self, phuong_phap: str = "minmax"):
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._da_fit = False
        self._phuong_phap = phuong_phap

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit và transform dữ liệu (tương thích Pipeline)."""
        return self.chuan_hoa_minmax(data) if self._phuong_phap == "minmax" else self.chuan_hoa_zscore(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform dữ liệu đã fit (tương thích Pipeline)."""
        return self.chuan_hoa_minmax(data, fit=False) if self._phuong_phap == "minmax" else self.chuan_hoa_zscore(data, fit=False)

    def fit(self, data: np.ndarray) -> "XuLySo":
        """Tính toán tham số từ dữ liệu."""
        data = np.asarray(data, dtype=float)
        self._min = np.nanmin(data, axis=0)
        self._max = np.nanmax(data, axis=0)
        self._mean = np.nanmean(data, axis=0)
        self._std = np.nanstd(data, axis=0)
        self._da_fit = True
        return self

    def chuan_hoa_minmax(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Chuẩn hóa Min-Max: đưa dữ liệu về khoảng [0, 1].

        Args:
            data: Dữ liệu đầu vào
            fit: True nếu chưa fit, False nếu đã fit trước đó
        """
        data = np.asarray(data, dtype=float)
        if fit:
            self.fit(data)
        if not self._da_fit:
            raise RuntimeError("Cần gọi fit() trước.")

        pham_vi = self._max - self._min
        pham_vi = np.where(pham_vi == 0, 1, pham_vi)
        return (data - self._min) / pham_vi

    def chuan_hoa_zscore(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Chuẩn hóa Z-Score: đưa dữ liệu về phân phối chuẩn (mean=0, std=1).

        Args:
            data: Dữ liệu đầu vào
            fit: True nếu chưa fit, False nếu đã fit trước đó
        """
        data = np.asarray(data, dtype=float)
        if fit:
            self.fit(data)
        if not self._da_fit:
            raise RuntimeError("Cần gọi fit() trước.")

        std_safe = np.where(self._std == 0, 1, self._std)
        return (data - self._mean) / std_safe

    @staticmethod
    def xu_ly_gia_tri_thieu(
        data: np.ndarray, phuong_phap: str = "trung_vi"
    ) -> np.ndarray:
        """
        Xử lý giá trị thiếu (NaN).

        Args:
            data: Dữ liệu đầu vào
            phuong_phap: 'trung_vi', 'trung_binh', 'mode', 'xoa'
        """
        data = np.asarray(data, dtype=float).copy()
        mask = np.isnan(data)

        if not mask.any():
            return data

        if phuong_phap == "trung_vi":
            gia_tri = np.nanmedian(data, axis=0)
        elif phuong_phap == "trung_binh":
            gia_tri = np.nanmean(data, axis=0)
        elif phuong_phap == "xoa":
            return data[~mask.any(axis=1)]
        else:
            raise ValueError(f"Phương pháp '{phuong_phap}' không hỗ trợ.")

        for j in range(data.shape[1]):
            col_mask = mask[:, j]
            if col_mask.any():
                data[col_mask, j] = gia_tri[j]

        return data

    @staticmethod
    def ma_hoa_nhan(nhan: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Mã hóa nhãn phân loại thành số nguyên.

        Returns:
            (nhan_so, tu_dien): Nhãn đã mã hóa và ánh xạ ngược
        """
        nhan = np.asarray(nhan)
        gia_tri_duy_nhat = np.unique(nhan)
        tu_dien = {val: idx for idx, val in enumerate(gia_tri_duy_nhat)}
        nhan_so = np.array([tu_dien[v] for v in nhan])
        return nhan_so, tu_dien

    @staticmethod
    def ma_hoa_onehot(nhan: np.ndarray, so_lop: Optional[int] = None) -> np.ndarray:
        """Mã hóa nhãn thành one-hot vector."""
        nhan = np.asarray(nhan, dtype=int)
        if so_lop is None:
            so_lop = int(nhan.max()) + 1
        onehot = np.zeros((len(nhan), so_lop))
        onehot[np.arange(len(nhan)), nhan] = 1
        return onehot

    @staticmethod
    def chia_du_lieu(
        X: np.ndarray,
        y: np.ndarray,
        ty_le_test: float = 0.2,
        ngau_nhien: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Chia dữ liệu thành tập huấn luyện và kiểm tra.

        Args:
            X: Đặc trưng
            y: Nhãn
            ty_le_test: Tỷ lệ dữ liệu kiểm tra (0-1)
            ngau_nhien: Seed ngẫu nhiên

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X, y = np.asarray(X), np.asarray(y)
        np.random.seed(ngau_nhien)
        n = len(X)
        indices = np.random.permutation(n)
        test_size = int(n * ty_le_test)

        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
