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

    def __init__(
        self,
        ten: str = "VietnameseAI",
        log_level: str = "INFO",
        theo_doi_thi_nghiem: Optional[Any] = None,
        quan_ly_mo_hinh: Optional[Any] = None,
    ):
        self.ten = ten
        self.logger = Logger(ten, level=log_level)
        self.lich_su_huan_luyen: List[Dict[str, Any]] = []
        self.mo_hinh_dang_su_dung: Dict[str, Any] = {}
        self.theo_doi_thi_nghiem = theo_doi_thi_nghiem
        self.quan_ly_mo_hinh = quan_ly_mo_hinh
        self.logger.info(f"Đã khởi tạo Engine: {ten}")

    def huan_luyen(
        self,
        mo_hinh: Any,
        X: Any,
        y: Optional[Any] = None,
        ten_mo_hinh: Optional[str] = None,
        dang_ky_registry: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Huấn luyện một mô hình với dữ liệu cho trước (hỗ trợ cả học máy cổ điển,
        GPTModel, HuggingFaceWrapper, BoTinhChinhMoHinh và Pipeline).

        Args:
            mo_hinh: Đối tượng mô hình cần huấn luyện
            X: Dữ liệu đầu vào (ma trận đặc trưng, danh sách văn bản, hoặc dataset fine-tune)
            y: Nhãn mục tiêu (tùy chọn đối với SFT/DPO/Unsupervised)
            ten_mo_hinh: Tên định danh mô hình (tùy chọn)
            dang_ky_registry: Tự động đăng ký vào QuanLyMoHinh nếu có

        Returns:
            Dict chứa thông tin huấn luyện (thời gian, trạng thái, kết quả)
        """
        ten = ten_mo_hinh or getattr(mo_hinh, "ten", mo_hinh.__class__.__name__)
        self.logger.info(f"Bắt đầu huấn luyện mô hình: {ten}")

        bat_dau = time.time()
        try:
            if y is not None:
                try:
                    ket_qua_raw = mo_hinh.huan_luyen(X, y, **kwargs)
                except TypeError:
                    ket_qua_raw = mo_hinh.huan_luyen((X, y), **kwargs)
            else:
                ket_qua_raw = mo_hinh.huan_luyen(X, **kwargs)

            thoi_gian = time.time() - bat_dau
            so_mau = len(X) if hasattr(X, "__len__") else 1
            so_dac_trung = (
                X.shape[1]
                if isinstance(X, np.ndarray) and X.ndim > 1
                else 1
            )

            ban_ghi: Dict[str, Any] = {
                "ten_mo_hinh": ten,
                "thoi_gian_huan_luyen": round(thoi_gian, 4),
                "so_mau": so_mau,
                "so_dac_trung": so_dac_trung,
                "trang_thai": "thanh_cong",
            }
            if isinstance(ket_qua_raw, dict):
                ban_ghi["chi_tiet"] = ket_qua_raw

            self.lich_su_huan_luyen.append(ban_ghi)
            self.mo_hinh_dang_su_dung[ten] = mo_hinh

            if self.theo_doi_thi_nghiem is not None and hasattr(
                self.theo_doi_thi_nghiem, "bat_dau"
            ):
                try:
                    self.theo_doi_thi_nghiem.bat_dau(ten_chay=ten)
                    self.theo_doi_thi_nghiem.log_param("ten_mo_hinh", ten)
                    self.theo_doi_thi_nghiem.log_param("so_mau", so_mau)
                    self.theo_doi_thi_nghiem.log_metric("thoi_gian_s", round(thoi_gian, 4))
                    if isinstance(ket_qua_raw, dict) and "train_loss_min" in ket_qua_raw:
                        self.theo_doi_thi_nghiem.log_metric(
                            "train_loss_min", float(ket_qua_raw["train_loss_min"])
                        )
                    self.theo_doi_thi_nghiem.ket_thuc()
                except Exception:
                    pass

            if (dang_ky_registry or self.quan_ly_mo_hinh is not None) and self.quan_ly_mo_hinh is not None:
                try:
                    ver = self.quan_ly_mo_hinh.dang_ky(
                        mo_hinh,
                        ten=ten,
                        metrics={"thoi_gian_s": round(thoi_gian, 4)},
                    )
                    ban_ghi["version_registry"] = ver
                except Exception:
                    pass

            self.logger.info(f"Huấn luyện hoàn tất: {ten} ({so_mau} mẫu, {thoi_gian:.2f}s)")
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

    def _giai_quyet_mo_hinh(self, mo_hinh: Any) -> Any:
        if isinstance(mo_hinh, str):
            if mo_hinh in self.mo_hinh_dang_su_dung:
                return self.mo_hinh_dang_su_dung[mo_hinh]
            if self.quan_ly_mo_hinh is not None:
                return self.quan_ly_mo_hinh.tai(mo_hinh)
            raise KeyError(f"Không tìm thấy mô hình: {mo_hinh}")
        return mo_hinh

    def du_doan(self, mo_hinh: Any, X: Any, **kwargs: Any) -> Any:
        """
        Dự đoán kết quả với mô hình đã huấn luyện (chấp nhận cả đối tượng mô hình hoặc tên mô hình).
        """
        target = self._giai_quyet_mo_hinh(mo_hinh)
        so_mau = len(X) if hasattr(X, "__len__") and not isinstance(X, str) else 1
        self.logger.info(f"Bắt đầu dự đoán với {so_mau} mẫu")
        if hasattr(target, "du_doan"):
            ket_qua = target.du_doan(X, **kwargs)
        elif hasattr(target, "predict"):
            ket_qua = target.predict(X)
        elif hasattr(target, "sinh_van_ban") and isinstance(X, str):
            ket_qua = target.sinh_van_ban(X, **kwargs)
        else:
            raise AttributeError(f"Mô hình {target} không hỗ trợ dự đoán.")
        return ket_qua

    def danh_gia(self, mo_hinh: Any, X: Any, y: Optional[Any] = None) -> Dict[str, Any]:
        """
        Đánh giá hiệu suất mô hình.
        """
        target = self._giai_quyet_mo_hinh(mo_hinh)
        if y is not None:
            diem = target.danh_gia(X, y)
        else:
            diem = target.danh_gia(X)
        if isinstance(diem, dict):
            return diem
        self.logger.info(f"Đánh giá mô hình: điểm = {diem:.4f}")
        return {"diem": float(diem)}

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
