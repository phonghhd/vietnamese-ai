"""GiaiThichMoHinh - Giải thích kết quả mô hình (Feature Importance, Permutation, LIME cơ bản)."""

from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class GiaiThichMoHinh:
    """
    Bộ công cụ giải thích mô hình học máy.

    Hỗ trợ:
    - Feature Importance (từ mô hình cây)
    - Permutation Importance (mọi mô hình)
    - LIME cơ bản (local explanation)

    Sử dụng:
        >>> gt = GiaiThichMoHinh(mo_hinh, X_train, y_train)
        >>> tam_quan_trong = gt.feature_importance()
        >>> gt.permutation_importance(X_test, y_test)
        >>> gt.giai_thich_mau(X_test[0])  # LIME cho 1 mẫu
    """

    def __init__(
        self,
        mo_hinh: Any,
        X: np.ndarray,
        y: np.ndarray,
        ten_dac_trung: Optional[List[str]] = None,
    ):
        self.mo_hinh = mo_hinh
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.logger = Logger("GiaiThich")

        if ten_dac_trung is None:
            self.ten_dac_trung = [f"dac_trung_{i}" for i in range(self.X.shape[1])]
        else:
            self.ten_dac_trung = ten_dac_trung

    def feature_importance(self) -> Dict[str, float]:
        """
        Lấy mức độ quan trọng của đặc trưng từ mô hình.

        Hỗ trợ: RandomForest, DecisionTree, GradientBoosting, LogisticRegression
        """
        mo_hinh = self.mo_hinh
        if hasattr(mo_hinh, "_mo_hinh"):
            mo_hinh = mo_hinh._mo_hinh

        if hasattr(mo_hinh, "feature_importances_"):
            importance = mo_hinh.feature_importances_
        elif hasattr(mo_hinh, "coef_"):
            importance = np.abs(mo_hinh.coef_).flatten()
            if len(importance) != len(self.ten_dac_trung):
                importance = np.mean(np.abs(mo_hinh.coef_), axis=0)
        else:
            self.logger.warning(
                "Mô hình không hỗ trợ feature_importance trực tiếp. "
                "Sử dụng permutation_importance() thay thế."
            )
            return {}

        # Sắp xếp giảm dần
        chi_so_sap_xep = np.argsort(importance)[::-1]
        ket_qua = {}
        for idx in chi_so_sap_xep:
            ten = self.ten_dac_trung[idx] if idx < len(self.ten_dac_trung) else f"f{idx}"
            ket_qua[ten] = float(importance[idx])

        return ket_qua

    def permutation_importance(
        self,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        so_lan: int = 10,
    ) -> Dict[str, float]:
        """
        Tính Permutation Importance - mức giảm hiệu suất khi xáo trộn từng đặc trưng.

        Args:
            X_test: Dữ liệu test (mặc định: self.X)
            y_test: Nhãn test (mặc định: self.y)
            so_lan: Số lần lặp trung bình

        Returns:
            Dict ánh xạ ten_dac_trung -> importance_score
        """
        X = np.asarray(X_test if X_test is not None else self.X)
        y = np.asarray(y_test if y_test is not None else self.y)

        diem_goc = self.mo_hinh.danh_gia(X, y)
        self.logger.info(f"Điểm gốc: {diem_goc:.4f}")

        ket_qua = {}
        for j in range(X.shape[1]):
            diem_trung_binh = 0.0
            for _ in range(so_lan):
                X_perm = X.copy()
                np.random.shuffle(X_perm[:, j])
                diem_trung_binh += self.mo_hinh.danh_gia(X_perm, y)
            diem_trung_binh /= so_lan

            suy_giam = diem_goc - diem_trung_binh
            ten = self.ten_dac_trung[j]
            ket_qua[ten] = float(suy_giam)

        # Sắp xếp theo mức suy giảm
        ket_qua = dict(sorted(ket_qua.items(), key=lambda x: x[1], reverse=True))
        return ket_qua

    def giai_thich_mau(
        self,
        mau: np.ndarray,
        so_lang_gieng: int = 50,
    ) -> Dict[str, float]:
        """
        Giải thích một mẫu cụ thể (LIME cơ bản).

        Tạo mẫu lân cận xung quanh mẫu cần giải thích,
        huấn luyện mô hình tuyến tính cục bộ.

        Args:
            mau: Mẫu cần giải thích (1, n_features)
            so_lang_gieng: Số mẫu lân cận

        Returns:
            Dict ánh xạ ten_dac_trung -> trọng số (âm/âm = giảm/tăng kết quả)
        """
        mau = np.asarray(mau).flatten()
        n_features = len(mau)

        # Tạo mẫu lân cận
        np.random.seed(42)
        mau_lang_gieng = np.random.normal(0, 1, (so_lang_gieng, n_features))
        for j in range(n_features):
            mau_lang_gieng[:, j] = mau_lang_gieng[:, j] * 0.1 + mau[j]

        # Dự đoán cho mẫu lân cận
        try:
            du_doan = self.mo_hinh.du_doan(mau_lang_gieng)
        except Exception:
            return {}

        # Tính khoảng cách (kernel)
        khoang_cach = np.sqrt(np.sum((mau_lang_gieng - mau) ** 2, axis=1))
        trong_so = np.exp(-(khoang_cach**2) / (2 * (0.75**2)))

        # Huấn luyện mô hình tuyến tính cục bộ (weighted OLS)
        try:
            X_w = mau_lang_gieng * np.sqrt(trong_so[:, np.newaxis])
            y_w = du_doan * np.sqrt(trong_so)

            # Giải bằng least squares
            from numpy.linalg import lstsq

            beta, _, _, _ = lstsq(X_w, y_w, rcond=None)

            ket_qua = {}
            for j in range(n_features):
                ten = self.ten_dac_trung[j]
                ket_qua[ten] = float(beta[j])

            # Sắp xếp theo giá trị tuyệt đối
            ket_qua = dict(sorted(ket_qua.items(), key=lambda x: abs(x[1]), reverse=True))
            return ket_qua

        except Exception as e:
            self.logger.warning(f"LIME thất bại: {e}")
            return {}

    def bao_cao(self, X_test=None, y_test=None) -> str:
        """Tạo báo cáo giải thích dạng text."""
        lines = ["=== BÁO CÁO GIẢI THÍCH MÔ HÌNH ===\n"]

        # Feature Importance
        fi = self.feature_importance()
        if fi:
            lines.append("1. Feature Importance (từ mô hình):")
            for i, (ten, diem) in enumerate(fi.items()):
                bar = "█" * int(diem * 50 / max(fi.values())) if max(fi.values()) > 0 else ""
                lines.append(f"   {i + 1}. {ten}: {diem:.4f} {bar}")
            lines.append("")

        # Permutation Importance
        pi = self.permutation_importance(X_test, y_test)
        if pi:
            lines.append("2. Permutation Importance:")
            for i, (ten, diem) in enumerate(pi.items()):
                bar = "█" * max(0, int(diem * 50 / max(abs(v) for v in pi.values()))) if pi else ""
                lines.append(f"   {i + 1}. {ten}: {diem:.4f} {bar}")

        return "\n".join(lines)
