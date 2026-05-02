"""KiemDinhCheo - Cross-validation cho mô hình học máy."""

from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class KiemDinhCheo:
    """
    Kiểm định chéo (Cross-Validation) để đánh giá mô hình ổn định.

    Hỗ trợ:
    - K-Fold Cross-Validation
    - Stratified K-Fold (giữ tỷ lệ lớp)
    - Repeated K-Fold

    Sử dụng:
        >>> kdc = KiemDinhCheo(so_fold=5)
        >>> ket_qua = kdc.chay(mo_hinh, X, y)
        >>> print(ket_qua['diem_trung_binh'])
    """

    def __init__(self, so_fold: int = 5, lap_lai: int = 1, seed: int = 42):
        if so_fold < 2:
            raise ValueError("so_fold phải >= 2")
        self.so_fold = so_fold
        self.lap_lai = lap_lai
        self.seed = seed
        self.logger = Logger("KiemDinhCheo")

    def _chia_fold(
        self, X: np.ndarray, y: np.ndarray, stratified: bool = True
    ) -> List[tuple]:
        """Chia dữ liệu thành K fold."""
        n = len(X)
        indices = np.arange(n)

        if stratified:
            return self._chia_fold_stratified(indices, y)

        np.random.seed(self.seed)
        np.random.shuffle(indices)
        fold_sizes = np.full(self.so_fold, n // self.so_fold)
        fold_sizes[: n % self.so_fold] += 1

        folds = []
        current = 0
        for size in fold_sizes:
            test_idx = indices[current : current + size]
            train_idx = np.concatenate([indices[:current], indices[current + size :]])
            folds.append((train_idx, test_idx))
            current += size

        return folds

    def _chia_fold_stratified(
        self, indices: np.ndarray, y: np.ndarray
    ) -> List[tuple]:
        """Chia fold giữ tỷ lệ lớp (stratified)."""
        np.random.seed(self.seed)
        folds_test = [[] for _ in range(self.so_fold)]

        for lop in np.unique(y):
            lop_idx = indices[y[indices] == lop]
            np.random.shuffle(lop_idx)
            for i, idx in enumerate(lop_idx):
                folds_test[i % self.so_fold].append(idx)

        folds = []
        for i in range(self.so_fold):
            test_idx = np.array(folds_test[i])
            train_idx = np.array([idx for j, f in enumerate(folds_test) if j != i for idx in f])
            folds.append((train_idx, test_idx))

        return folds

    def chay(
        self,
        mo_hinh: Any,
        X: np.ndarray,
        y: np.ndarray,
        chi_so: str = "do_chinh_xac",
        stratified: bool = True,
    ) -> Dict[str, Any]:
        """
        Chạy K-Fold Cross-Validation.

        Args:
            mo_hinh: Mô hình có phương thức huan_luyen() và danh_gia()
            X: Dữ liệu đầu vào
            y: Nhãn
            chi_so: Tên chỉ số đánh giá ('do_chinh_xac', 'mse', 'f1')
            stratified: Giữ tỷ lệ lớp

        Returns:
            Dict chứa: diem_trung_binh, do_lech_chuan, cac_diem, chi_so
        """
        X, y = np.asarray(X), np.asarray(y)
        self.logger.info(
            f"Bắt đầu {self.so_fold}-Fold CV ({self.lap_lai} lần lặp)"
        )

        tat_ca_diem = []

        for lan in range(self.lap_lai):
            folds = self._chia_fold(X, y, stratified)

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Clone mô hình (tạo instance mới)
                mo_hinh_fold = type(mo_hinh)(**mo_hinh.lay_tham_so())
                mo_hinh_fold.huan_luyen(X_train, y_train)
                diem = mo_hinh_fold.danh_gia(X_test, y_test)
                tat_ca_diem.append(diem)

                self.logger.debug(
                    f"  Lần {lan+1}, Fold {fold_idx+1}: {chi_so}={diem:.4f}"
                )

        tat_ca_diem = np.array(tat_ca_diem)
        ket_qua = {
            "chi_so": chi_so,
            "cac_diem": tat_ca_diem.tolist(),
            "diem_trung_binh": float(np.mean(tat_ca_diem)),
            "do_lech_chuan": float(np.std(tat_ca_diem)),
            "diem_toi_thieu": float(np.min(tat_ca_diem)),
            "diem_toi_da": float(np.max(tat_ca_diem)),
            "so_fold": self.so_fold,
            "so_lan_lap": self.lap_lai,
        }

        self.logger.info(
            f"Kết quả CV: {chi_so}={ket_qua['diem_trung_binh']:.4f} "
            f"(+/- {ket_qua['do_lech_chuan']:.4f})"
        )

        return ket_qua
