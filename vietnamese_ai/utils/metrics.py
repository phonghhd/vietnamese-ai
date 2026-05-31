"""Metrics - Các chỉ số đánh giá mô hình."""

from typing import Dict

import numpy as np


class Metrics:
    """
    Tập hợp các chỉ số đánh giá mô hình học máy.

    Sử dụng:
        >>> m = Metrics()
        >>> m.do_chinh_xac(y_thuc, y_du_doan)
        >>> m.mse(y_thuc, y_du_doan)
        >>> m.bao_cao_phan_loai(y_thuc, y_du_doan)
    """

    @staticmethod
    def do_chinh_xac(y_thuc: np.ndarray, y_du_doan: np.ndarray) -> float:
        """Tính độ chính xác (accuracy)."""
        y_thuc, y_du_doan = np.asarray(y_thuc), np.asarray(y_du_doan)
        return float(np.mean(y_thuc == y_du_doan))

    @staticmethod
    def mse(y_thuc: np.ndarray, y_du_doan: np.ndarray) -> float:
        """Tính Mean Squared Error."""
        y_thuc, y_du_doan = np.asarray(y_thuc), np.asarray(y_du_doan)
        return float(np.mean((y_thuc - y_du_doan) ** 2))

    @staticmethod
    def rmse(y_thuc: np.ndarray, y_du_doan: np.ndarray) -> float:
        """Tính Root Mean Squared Error."""
        return float(np.sqrt(Metrics.mse(y_thuc, y_du_doan)))

    @staticmethod
    def mae(y_thuc: np.ndarray, y_du_doan: np.ndarray) -> float:
        """Tính Mean Absolute Error."""
        y_thuc, y_du_doan = np.asarray(y_thuc), np.asarray(y_du_doan)
        return float(np.mean(np.abs(y_thuc - y_du_doan)))

    @staticmethod
    def r2_score(y_thuc: np.ndarray, y_du_doan: np.ndarray) -> float:
        """Tính R-squared (hệ số xác định)."""
        y_thuc, y_du_doan = np.asarray(y_thuc), np.asarray(y_du_doan)
        ss_res = np.sum((y_thuc - y_du_doan) ** 2)
        ss_tot = np.sum((y_thuc - np.mean(y_thuc)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def precision_recall_f1(
        y_thuc: np.ndarray,
        y_du_doan: np.ndarray,
        lop_pos: int = 1,
    ) -> Dict[str, float]:
        """
        Tính Precision, Recall, F1-score cho phân loại nhị phân.

        Args:
            y_thuc: Nhãn thực tế
            y_du_doan: Nhãn dự đoán
            lop_pos: Lớp dương (positive class)

        Returns:
            Dict chứa precision, recall, f1
        """
        y_thuc, y_du_doan = np.asarray(y_thuc), np.asarray(y_du_doan)

        tp = np.sum((y_thuc == lop_pos) & (y_du_doan == lop_pos))
        fp = np.sum((y_thuc != lop_pos) & (y_du_doan == lop_pos))
        fn = np.sum((y_thuc == lop_pos) & (y_du_doan != lop_pos))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    @staticmethod
    def bao_cao_phan_loai(y_thuc: np.ndarray, y_du_doan: np.ndarray) -> Dict[str, float]:
        """Báo cáo đầy đủ cho bài toán phân loại."""
        y_thuc, y_du_doan = np.asarray(y_thuc), np.asarray(y_du_doan)
        ket_qua = {"do_chinh_xac": Metrics.do_chinh_xac(y_thuc, y_du_doan)}

        cac_lop = np.unique(y_thuc)
        if len(cac_lop) == 2:
            ket_qua.update(Metrics.precision_recall_f1(y_thuc, y_du_doan, lop_pos=cac_lop[1]))
        else:
            tong_f1 = 0.0
            for lop in cac_lop:
                pr = Metrics.precision_recall_f1(y_thuc, y_du_doan, lop_pos=lop)
                tong_f1 += pr["f1"]
            ket_qua["f1_macro"] = float(tong_f1 / len(cac_lop))

        return ket_qua

    @staticmethod
    def bao_cao_hoi_quy(y_thuc: np.ndarray, y_du_doan: np.ndarray) -> Dict[str, float]:
        """Báo cáo đầy đủ cho bài toán hồi quy."""
        return {
            "mse": Metrics.mse(y_thuc, y_du_doan),
            "rmse": Metrics.rmse(y_thuc, y_du_doan),
            "mae": Metrics.mae(y_thuc, y_du_doan),
            "r2": Metrics.r2_score(y_thuc, y_du_doan),
        }
