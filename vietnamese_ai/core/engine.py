"""Engine - Cỗ máy chính điều phối toàn bộ quy trình học máy."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class Engine:
    """
    Cỗ máy chính điều phối toàn bộ quy trình học máy.

    Chịu trách nhiệm:
    - Quản lý vòng đời mô hình (khởi tạo -> huấn luyện -> đánh giá -> dự đoán)
    - Lưu trữ lịch sử huấn luyện
    - Cung cấp giao diện thống nhất cho tất cả mô hình

    Sử dụng:
        >>> engine = Engine()
        >>> engine.cap_nhat_moi(moi_truong)  # optional
        >>> engine.huan_luyen(mo_hinh, X, y)
        >>> du_doan = engine.du_doan(mo_hinh, X_moi)
    """

    def __init__(self, ten: str = "VietnameseAI", log_level: str = "INFO"):
        self.ten = ten
        self.logger = Logger(ten, level=log_level)
        self.lich_su_huan_luyen: List[Dict[str, Any]] = []
        self.mo_hinh_dang_su_dung: Dict[str, Any] = {}
        self.logger.info(f"Đã khởi tạo Engine: {ten}")

    def huan_luyen(
        self,
        mo_hinh: Any,
        X: np.ndarray,
        y: np.ndarray,
        ten_mo_hinh: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện một mô hình với dữ liệu cho trước.

        Args:
            mo_hinh: Đối tượng mô hình cần huấn luyện
            X: Dữ liệu đầu vào (ma trận đặc trưng)
            y: Nhãn mục tiêu
            ten_mo_hinh: Tên định danh mô hình (tùy chọn)

        Returns:
            Dict chứa thông tin huấn luyện (thời gian, trạng thái)
        """
        ten = ten_mo_hinh or getattr(mo_hinh, "ten", mo_hinh.__class__.__name__)
        self.logger.info(f"Bắt đầu huấn luyện mô hình: {ten}")

        bat_dau = time.time()
        try:
            mo_hinh.huan_luyen(X, y)
            thoi_gian = time.time() - bat_dau

            ban_ghi = {
                "ten_mo_hinh": ten,
                "thoi_gian_huan_luyen": round(thoi_gian, 4),
                "so_mau": len(X),
                "so_dac_trung": X.shape[1] if X.ndim > 1 else 1,
                "trang_thai": "thanh_cong",
            }
            self.lich_su_huan_luyen.append(ban_ghi)
            self.mo_hinh_dang_su_dung[ten] = mo_hinh

            self.logger.info(f"Huấn luyện hoàn tất: {ten} ({len(X)} mẫu, {thoi_gian:.2f}s)")
            return ban_ghi

        except Exception as loi:
            thoi_gian = time.time() - bat_dau
            ban_ghi = {
                "ten_mo_hinh": ten,
                "thoi_gian_huan_luyen": round(thoi_gian, 4),
                "trang_thai": "that_bai",
                "loi": str(loi),
            }
            self.lich_su_huan_luyen.append(ban_ghi)
            self.logger.error(f"Huấn luyện thất bại: {ten} - {loi}")
            raise

    def du_doan(self, mo_hinh: Any, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán kết quả với mô hình đã huấn luyện.

        Args:
            mo_hinh: Mô hình đã được huấn luyện
            X: Dữ liệu đầu vào cần dự đoán

        Returns:
            Mảng kết quả dự đoán
        """
        self.logger.info(f"Bắt đầu dự đoán với {len(X)} mẫu")
        ket_qua = mo_hinh.du_doan(X)
        self.logger.info(f"Dự đoán hoàn tất: {len(ket_qua)} kết quả")
        return ket_qua

    def danh_gia(self, mo_hinh: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Đánh giá hiệu suất mô hình.

        Args:
            mo_hinh: Mô hình đã được huấn luyện
            X: Dữ liệu kiểm tra
            y: Nhãn thực tế

        Returns:
            Dict chứa các chỉ số đánh giá
        """
        diem = mo_hinh.danh_gia(X, y)
        self.logger.info(f"Đánh giá mô hình: điểm = {diem:.4f}")
        return {"diem": diem}

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Trả về toàn bộ lịch sử huấn luyện."""
        return self.lich_su_huan_luyen.copy()

    def lay_mo_hinh(self, ten: str) -> Any:
        """Lấy mô hình theo tên."""
        if ten not in self.mo_hinh_dang_su_dung:
            raise KeyError(f"Không tìm thấy mô hình: {ten}")
        return self.mo_hinh_dang_su_dung[ten]

    def danh_sach_mo_hinh(self) -> List[str]:
        """Trả về danh sách tên tất cả mô hình đã đăng ký."""
        return list(self.mo_hinh_dang_su_dung.keys())

    def __repr__(self) -> str:
        return f"Engine(ten='{self.ten}', so_mo_hinh={len(self.mo_hinh_dang_su_dung)})"
