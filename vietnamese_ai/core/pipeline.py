"""Pipeline - Quy trình học máy tự động."""

import pickle
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from vietnamese_ai.utils.logger import Logger


class Pipeline:
    """
    Quy trình học máy tự động (Pipeline).

    Kết hợp nhiều bước tiền xử lý và mô hình thành một chuỗi thống nhất.
    Hỗ trợ fit -> predict -> evaluate -> save/load trong một luồng duy nhất.

    Sử dụng:
        >>> pipe = Pipeline()
        >>> pipe.them_buoc("chuan_hoa", normalizer)
        >>> pipe.them_buoc("mo_hinh", classifier)
        >>> pipe.fit(X_train, y_train)
        >>> du_doan = pipe.predict(X_test)
        >>> pipe.luu("pipeline.pkl")
        >>> pipe2 = Pipeline.tai("pipeline.pkl")
    """

    def __init__(self, ten: str = "Pipeline"):
        self.ten = ten
        self.cac_buoc: List[Tuple[str, Any]] = []
        self.da_fit = False
        self.logger = Logger(ten)

    def them_buoc(self, ten: str, bo_xu_ly: Any) -> "Pipeline":
        """
        Thêm một bước vào pipeline.

        Args:
            ten: Tên của bước
            bo_xu_ly: Đối tượng xử lý (phương thức fit_transform/transform
                      cho bước tiền xử lý, hoặc huan_luyen/du_doan cho mô hình)

        Returns:
            self (để hỗ trợ chaining)
        """
        self.cac_buoc.append((ten, bo_xu_ly))
        self.logger.info(f"Đã thêm bước: {ten}")
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Pipeline":
        """
        Huấn luyện toàn bộ pipeline.

        Với mỗi bước tiền xử lý: gọi fit_transform()
        Với mô hình cuối cùng: gọi huan_luyen()

        Args:
            X: Dữ liệu đầu vào
            y: Nhãn mục tiêu (bắt buộc cho bước mô hình)

        Returns:
            self
        """
        self.logger.info(f"Bắt đầu fit pipeline ({len(self.cac_buoc)} bước)")

        du_lieu_hien_tai = X
        for i, (ten, buoc) in enumerate(self.cac_buoc):
            la_buoc_cuoi = i == len(self.cac_buoc) - 1

            if la_buoc_cuoi and hasattr(buoc, "huan_luyen"):
                buoc.huan_luyen(du_lieu_hien_tai, y)
                self.logger.info(f"Bước [{ten}]: Đã huấn luyện mô hình")
            elif hasattr(buoc, "fit_transform"):
                du_lieu_hien_tai = buoc.fit_transform(du_lieu_hien_tai)
                self.logger.info(f"Bước [{ten}]: fit_transform hoàn tất")
            elif hasattr(buoc, "transform"):
                du_lieu_hien_tai = buoc.transform(du_lieu_hien_tai)
                self.logger.info(f"Bước [{ten}]: transform hoàn tất")
            else:
                self.logger.warning(f"Bước [{ten}]: Không có phương thức phù hợp")

        self.da_fit = True
        self.logger.info("Pipeline fit hoàn tất")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán với toàn bộ pipeline.

        Args:
            X: Dữ liệu đầu vào

        Returns:
            Mảng kết quả dự đoán
        """
        if not self.da_fit:
            raise RuntimeError("Pipeline chưa được fit. Gọi fit() trước.")

        du_lieu_hien_tai = X
        for i, (ten, buoc) in enumerate(self.cac_buoc):
            la_buoc_cuoi = i == len(self.cac_buoc) - 1

            if la_buoc_cuoi and hasattr(buoc, "du_doan"):
                return buoc.du_doan(du_lieu_hien_tai)
            elif hasattr(buoc, "transform"):
                du_lieu_hien_tai = buoc.transform(du_lieu_hien_tai)

        return du_lieu_hien_tai

    def danh_sach_buoc(self) -> List[str]:
        """Trả về danh sách tên các bước trong pipeline."""
        return [ten for ten, _ in self.cac_buoc]

    def lay_buoc(self, ten: str) -> Any:
        """Lấy một bước theo tên."""
        for ten_buoc, buoc in self.cac_buoc:
            if ten_buoc == ten:
                return buoc
        raise KeyError(f"Không tìm thấy bước: {ten}")

    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        """Đánh giá pipeline (gọi danh_gia() của bước cuối)."""
        if not self.da_fit:
            raise RuntimeError("Pipeline chưa được fit.")

        du_lieu_hien_tai = X
        for ten, buoc in self.cac_buoc:
            if hasattr(buoc, "du_doan"):
                du_lieu_hien_tai = buoc.du_doan(du_lieu_hien_tai)
            elif hasattr(buoc, "transform"):
                du_lieu_hien_tai = buoc.transform(du_lieu_hien_tai)

        buoc_cuoi = self.cac_buoc[-1][1]
        if hasattr(buoc_cuoi, "danh_gia"):
            return buoc_cuoi.danh_gia(X, y)
        return 0.0

    def luu(self, duong_dan: str) -> None:
        """Lưu pipeline ra file pickle.

        Cảnh báo: File pickle có thể chứa mã độc. Chỉ tải pipeline từ nguồn tin cậy.
        """
        duong_dan = Path(duong_dan)
        duong_dan.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan, "wb") as f:
            pickle.dump(self, f)
        self.logger.info(f"Đã lưu pipeline tại: {duong_dan}")

    @classmethod
    def tai(cls, duong_dan: str) -> "Pipeline":
        """Tải pipeline từ file pickle.

        Cảnh báo: File pickle có thể chứa mã độc. Chỉ tải file từ nguồn tin cậy.
        """
        warnings.warn(
            "Đang tải pipeline từ pickle. Chỉ tải file từ nguồn tin cậy "
            "để tránh mã độc (remote code execution).",
            stacklevel=2,
        )
        try:
            with open(duong_dan, "rb") as f:
                pipe = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Lỗi đọc file pickle: {e}")
        pipe.logger.info(f"Đã tải pipeline từ: {duong_dan}")
        return pipe

    def __repr__(self) -> str:
        buoc_str = " -> ".join(self.danh_sach_buoc())
        trang_thai = "đã fit" if self.da_fit else "chưa fit"
        return f"Pipeline('{self.ten}': {buoc_str} [{trang_thai}])"
