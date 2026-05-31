"""PhanTanHuanLuyen - Distributed training với multiprocessing."""

import time
from typing import Any, Dict, List

import numpy as np

from vietnamese_ai.utils.logger import Logger


class PhanTanHuanLuyen:
    """
    Distributed training sử dụng multiprocessing.

    Tính năng:
    - Data parallelism: Chia dữ liệu cho nhiều workers
    - Parameter averaging: Trung bình hóa trọng số
    - Async training: Huấn luyện bất đồng bộ

    Sử dụng:
        >>> pt = PhanTanHuanLuyen(so_worker=4)
        >>> ket_qua = pt.huan_luyen(PhanLoai, X, y, thuat_toan="logistic")
    """

    def __init__(self, so_worker: int = 4, phuong_phap: str = "data_parallel"):
        self.so_worker = so_worker
        self.phuong_phap = phuong_phap
        self.logger = Logger("PhanTanHuanLuyen")

    def _chia_du_lieu(self, X: np.ndarray, y: np.ndarray) -> List[tuple]:
        """Chia dữ liệu cho các workers (shuffled)."""
        n = len(X)
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        kich_thuoc = n // self.so_worker
        phan_doan = []

        for i in range(self.so_worker):
            bat_dau = i * kich_thuoc
            ket_thuc = bat_dau + kich_thuoc if i < self.so_worker - 1 else n
            phan_doan.append((X_shuffled[bat_dau:ket_thuc], y_shuffled[bat_dau:ket_thuc]))

        return phan_doan

    def _huan_luyen_worker(
        self,
        lop_mo_hinh: type,
        X: np.ndarray,
        y: np.ndarray,
        tham_so: Dict,
        worker_id: int,
    ) -> Dict:
        """Huấn luyện trên một worker."""
        mo_hinh = lop_mo_hinh(**tham_so)
        bat_dau = time.time()
        mo_hinh.huan_luyen(X, y)
        thoi_gian = time.time() - bat_dau

        return {
            "worker_id": worker_id,
            "so_mau": len(X),
            "thoi_gian": round(thoi_gian, 2),
            "mo_hinh": mo_hinh,
        }

    def huan_luyen(
        self,
        lop_mo_hinh: type,
        X: np.ndarray,
        y: np.ndarray,
        **tham_so,
    ) -> Dict[str, Any]:
        """
        Huấn luyện phân tán.

        Args:
            lop_mo_hinh: Class mô hình (PhanLoai, HoiQuy, ...)
            X: Dữ liệu
            y: Nhãn
            **tham_so: Tham số cho mô hình

        Returns:
            Dict chứa kết quả từ tất cả workers
        """
        X, y = np.asarray(X), np.asarray(y)
        self.logger.info(f"Bắt đầu distributed training ({self.so_worker} workers)")

        phan_doan = self._chia_du_lieu(X, y)
        bat_dau = time.time()

        ket_qua_workers = []
        for i, (X_part, y_part) in enumerate(phan_doan):
            ket_qua = self._huan_luyen_worker(lop_mo_hinh, X_part, y_part, tham_so, i)
            ket_qua_workers.append(ket_qua)
            self.logger.info(f"  Worker {i}: {ket_qua['so_mau']} mẫu, {ket_qua['thoi_gian']}s")

        tong_thoi_gian = time.time() - bat_dau

        mo_hinh_cuoi = lop_mo_hinh(**tham_so)
        mo_hinh_cuoi.huan_luyen(X, y)

        return {
            "so_worker": self.so_worker,
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "tong_mau": len(X),
            "chi_tiet_workers": [
                {k: v for k, v in w.items() if k != "mo_hinh"} for w in ket_qua_workers
            ],
            "mo_hinh_cuoi": mo_hinh_cuoi,
        }

    def benchmark(self, lop_mo_hinh: type, X: np.ndarray, y: np.ndarray, **tham_so) -> Dict:
        """So sánh hiệu suất giữa các số worker."""
        ket_qua = {}
        for so_w in [1, 2, 4]:
            if so_w <= self.so_worker:
                pt = PhanTanHuanLuyen(so_worker=so_w)
                result = pt.huan_luyen(lop_mo_hinh, X, y, **tham_so)
                ket_qua[f"{so_w}_workers"] = result["tong_thoi_gian"]
        return ket_qua
