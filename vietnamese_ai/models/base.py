"""Base Model - Lớp trừu tượng cơ sở cho tất cả mô hình."""

import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np


class BaseModel(ABC):
    """
    Lớp trừu tượng cơ sở cho tất cả mô hình trong framework.

    Mọi mô hình phải triển khai các phương thức:
    - huan_luyen(X, y): Huấn luyện mô hình
    - du_doan(X): Dự đoán kết quả
    - danh_gia(X, y): Đánh giá mô hình

    Sử dụng:
        >>> class MoHinhCuaToi(BaseModel):
        ...     def huan_luyen(self, X, y): ...
        ...     def du_doan(self, X): ...
        ...     def danh_gia(self, X, y): ...
    """

    def __init__(self, ten: str = "BaseModel"):
        self.ten = ten
        self.da_huan_luyen = False
        self.tham_so: Dict[str, Any] = {}

    @abstractmethod
    def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
        """Huấn luyện mô hình với dữ liệu X và nhãn y."""
        pass

    @abstractmethod
    def du_doan(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán kết quả cho dữ liệu X."""
        pass

    @abstractmethod
    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        """Đánh giá mô hình, trả về điểm số (0-1 cho phân loại, MSE cho hồi quy)."""
        pass

    def lay_tham_so(self) -> Dict[str, Any]:
        """Trả về các tham số của mô hình."""
        return self.tham_so.copy()

    def kiem_dinh_du_lieu(
        self, X: Any, y: Any = None, X_test: Any = None, loai_bo_mau_doc: bool = False
    ) -> Dict[str, Any]:
        """
        Kiểm định chất lượng đầu vào (0–100) và quét đầu độc dữ liệu (Spectral SVD & k-NN Label Flip).
        """
        from vietnamese_ai.security.data_defense import CongChatLuongDauVao, PhongThuDuLieu

        bc_chat_luong = CongChatLuongDauVao.kiem_tra_mang_so(X, y=y, X_test=X_test)
        du_lieu_sach, bc_phong_thu = PhongThuDuLieu.quet_va_lam_sach_dataset(
            X, y=y, loai_bo_mau_doc=loai_bo_mau_doc
        )
        return {
            **bc_chat_luong,
            "bao_cao_phong_thu": bc_phong_thu,
            "du_lieu_sach": du_lieu_sach if loai_bo_mau_doc else None,
        }

    def luu(self, duong_dan: str) -> None:
        """Lưu mô hình ra file (pickle format).

        Cảnh báo: File pickle có thể chứa mã độc. Chỉ tải mô hình từ nguồn tin cậy.
        """
        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def tai(cls, duong_dan: str) -> "BaseModel":
        """Tải mô hình từ file pickle.

        Cảnh báo: File pickle có thể chứa mã độc. Chỉ tải mô hình từ nguồn tin cậy.
        Sử dụng LuuTai.tai_mo_hinh_an_toan() cho dữ liệu cơ bản.
        """
        warnings.warn(
            "Đang tải mô hình từ pickle. Chỉ tải file từ nguồn tin cậy "
            "để tránh mã độc (remote code execution). Sử dụng "
            "LuuTai.tai_mo_hinh_an_toan() cho dữ liệu cơ bản.",
            stacklevel=2,
        )
        try:
            with open(duong_dan, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Lỗi đọc file pickle: {e}")

    def __repr__(self) -> str:
        trang_thai = "đã huấn luyện" if self.da_huan_luyen else "chưa huấn luyện"
        return f"{self.ten}({trang_thai})"
