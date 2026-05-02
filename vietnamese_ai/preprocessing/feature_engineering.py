"""Tạo đặc trưng - Feature Engineering."""

from typing import List, Optional

import numpy as np


class TaoDacTrung:
    """
    Bộ tạo và biến đổi đặc trưng.

    Tính năng:
    - Tạo đặc trưng bậc 2 (polynomial features)
    - Tạo đặc trưng tương tác
    - Giảm chiều PCA cơ bản
    - Lựa chọn đặc trưng theo phương sai

    Sử dụng:
        >>> tao = TaoDacTrung()
        >>> X_moi = tao.dac_trung_da_thuc(X, bac=2)
        >>> X_giam = tao.giam_chieu_pca(X, so_chieu=2)
    """

    @staticmethod
    def dac_trung_da_thuc(X: np.ndarray, bac: int = 2) -> np.ndarray:
        """
        Tạo đặc trưng bậc n (polynomial features).

        Args:
            X: Ma trận đầu vào (n_samples, n_features)
            bac: Bậc đa thức

        Returns:
            Ma trận đặc trưng mới với các bậc đa thức
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        ket_qua = [X]
        for b in range(2, bac + 1):
            ket_qua.append(X ** b)

        return np.hstack(ket_qua)

    @staticmethod
    def dac_trung_tuong_tac(X: np.ndarray) -> np.ndarray:
        """
        Tạo đặc trưng tương tác (mỗi cặp đặc trưng nhân với nhau).

        Args:
            X: Ma trận đầu vào (n_samples, n_features)

        Returns:
            Ma trận với các đặc trưng tương tác được thêm vào
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        tuong_tac = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                tuong_tac.append((X[:, i] * X[:, j]).reshape(-1, 1))

        if tuong_tac:
            return np.hstack([X] + tuong_tac)
        return X

    @staticmethod
    def giam_chieu_pca(X: np.ndarray, so_chieu: int = 2) -> np.ndarray:
        """
        Giảm chiều dữ liệu bằng PCA cơ bản (không phụ thuộc sklearn).

        Args:
            X: Ma trận đầu vào (n_samples, n_features)
            so_chieu: Số chiều mục tiêu

        Returns:
            Ma trận đã giảm chiều
        """
        X = np.asarray(X, dtype=float)
        X_centered = X - np.mean(X, axis=0)

        ma_tran_hop = X_centered.T @ X_centered / (len(X) - 1)
        gia_tri_rieng, vector_rieng = np.linalg.eigh(ma_tran_hop)

        idx = np.argsort(gia_tri_rieng)[::-1]
        vector_rieng = vector_rieng[:, idx[:so_chieu]]

        return X_centered @ vector_rieng

    @staticmethod
    def chon_dac_trung_phuong_sai(
        X: np.ndarray, nguong: float = 0.0
    ) -> tuple:
        """
        Lựa chọn đặc trưng theo phương sai.

        Args:
            X: Ma trận đầu vào
            nguong: Ngưỡng phương sai tối thiểu

        Returns:
            (X_da_chon, chi_so_da_chon)
        """
        X = np.asarray(X, dtype=float)
        phuong_sai = np.var(X, axis=0)
        chi_so = np.where(phuong_sai > nguong)[0]
        return X[:, chi_so], chi_so
