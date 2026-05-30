"""DuLieuMau - Các bộ dữ liệu mẫu để thử nghiệm."""

from typing import Tuple

import numpy as np


class DuLieuMau:
    """
    Các bộ dữ liệu mẫu để thử nghiệm framework.

    Sử dụng:
        >>> X, y = DuLieuMau.phan_loai_don_gian()
        >>> X, y = DuLieuMau.hoi_quy_don_gian()
        >>> X = DuLieuMau.phan_cum_don_gian()
    """

    @staticmethod
    def phan_loai_don_gian(
        so_mau: int = 200, so_dac_trung: int = 2, so_lop: int = 2, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo dữ liệu phân loại đơn giản."""
        np.random.seed(seed)
        X_list, y_list = [], []
        mau_moi_lop = so_mau // so_lop

        for lop in range(so_lop):
            tam = np.random.randn(so_dac_trung) * 3
            X_lop = np.random.randn(mau_moi_lop, so_dac_trung) + tam
            X_list.append(X_lop)
            y_list.append(np.full(mau_moi_lop, lop))

        return np.vstack(X_list), np.concatenate(y_list)

    @staticmethod
    def hoi_quy_don_gian(
        so_mau: int = 100, so_dac_trung: int = 1, nhieu: float = 0.5, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo dữ liệu hồi quy đơn giản."""
        np.random.seed(seed)
        X = np.random.randn(so_mau, so_dac_trung)
        w = np.random.randn(so_dac_trung)
        y = X @ w + np.random.randn(so_mau) * nhieu
        return X, y

    @staticmethod
    def phan_cum_don_gian(
        so_mau: int = 300, so_dac_trung: int = 2, so_cum: int = 3, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo dữ liệu phân cụm đơn giản."""
        np.random.seed(seed)
        X_list, y_list = [], []
        mau_moi_cum = so_mau // so_cum

        for cum in range(so_cum):
            tam = np.random.randn(so_dac_trung) * 2
            X_cum = np.random.randn(mau_moi_cum, so_dac_trung) * 0.5 + tam
            X_list.append(X_cum)
            y_list.append(np.full(mau_moi_cum, cum))

        return np.vstack(X_list), np.concatenate(y_list)

    @staticmethod
    def van_ban_tieng_viet() -> list:
        """Trả về danh sách mẫu văn bản tiếng Việt."""
        return [
            "Học máy là một nhánh của trí tuệ nhân tạo",
            "Mạng nơ-ron nhân tạo mô phỏng não người",
            "Xử lý ngôn ngữ tự nhiên giúp máy hiểu tiếng Việt",
            "Python là ngôn ngữ lập trình phổ biến cho AI",
            "TensorFlow và PyTorch là hai framework học máy hàng đầu",
            "Dữ liệu lớn là nền tảng cho học máy hiện đại",
            "Học sâu đã đạt được nhiều thành tựu trong nhận dạng hình ảnh",
            "Phân loại văn bản là bài toán phổ biến trong NLP",
            "Hồi quy tuyến tính là thuật toán học máy cơ bản nhất",
            "K-means là thuật toán phân cụm không giám sát phổ biến",
            "Random Forest kết hợp nhiều cây quyết định",
            "Gradient Boosting cải thiện hiệu suất từng bước",
            "Cross-validation giúp đánh giá mô hình chính xác hơn",
            "Overfitting xảy ra khi mô hình học quá kỹ dữ liệu huấn luyện",
            "Feature engineering là bước quan trọng trong pipeline học máy",
        ]

    @staticmethod
    def du_lieu_thuc_te_don_gian(
        so_mau: int = 500, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Tạo dữ liệu mô phỏng bài toán dự đoán giá nhà.

        Returns:
            (X, y, ten_dac_trung)
        """
        np.random.seed(seed)

        dien_tich = np.random.uniform(30, 200, so_mau)
        so_phong = np.random.randint(1, 6, so_mau)
        khoang_cach_trung_tam = np.random.uniform(0.5, 20, so_mau)
        tuoi_nha = np.random.uniform(0, 50, so_mau)

        gia = (
            dien_tich * 30
            + so_phong * 200
            - khoang_cach_trung_tam * 50
            - tuoi_nha * 10
            + np.random.randn(so_mau) * 500
        )

        X = np.column_stack([dien_tich, so_phong, khoang_cach_trung_tam, tuoi_nha])
        ten_dac_trung = ["dien_tich", "so_phong", "khoang_cach_trung_tam", "tuoi_nha"]

        return X, gia, ten_dac_trung
