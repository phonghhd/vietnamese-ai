"""DuDoanChuoiThoiGian - Dự đoán chuỗi thời gian."""

from typing import Optional

import numpy as np

from vietnamese_ai.models.base import BaseModel
from vietnamese_ai.utils.logger import Logger


class DuDoanChuoiThoiGian(BaseModel):
    """
    Dự đoán chuỗi thời gian.

    Hỗ trợ:
    - Moving Average (Trung bình trượt)
    - Exponential Smoothing (Làm mượt mũ)
    - Linear Trend (Xu hướng tuyến tính)
    - LSTM-like (sử dụng window features + regression)

    Sử dụng:
        >>> dstg = DuDoanChuoiThoiGian(phuong_phap="exponential")
        >>> dstg.huan_luyen(y_train)
        >>> du_doan = dstg.du_doan(30)  # Dự đoán 30 bước tiếp
    """

    PHUONG_PHAP = ["moving_average", "exponential", "linear_trend", "window_regression"]

    def __init__(
        self,
        phuong_phap: str = "exponential",
        cua_so: int = 10,
        alpha: float = 0.3,
        ten: Optional[str] = None,
    ):
        super().__init__(ten or f"DuDoanChuoiThoiGian({phuong_phap})")

        if phuong_phap not in self.PHUONG_PHAP:
            raise ValueError(f"Phương pháp '{phuong_phap}' không hỗ trợ. Chọn: {self.PHUONG_PHAP}")

        self.phuong_phap = phuong_phap
        self.cua_so = cua_so
        self.alpha = alpha
        self.logger = Logger("DuDoanChuoiThoiGian")

        self._du_lieu: Optional[np.ndarray] = None
        self._trung_binh_truot: Optional[np.ndarray] = None
        self._he_so_tuyen_tinh: Optional[np.ndarray] = None
        self._mo_hinh_regression = None

    def huan_luyen(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Huấn luyện mô hình trên chuỗi thời gian.

        Args:
            X: Chuỗi thời gian 1D, hoặc (n_samples, n_features)
            y: Không sử dụng (để tương thích BaseModel)
        """
        if y is not None and X.ndim == 2:
            du_lieu = y.astype(float)
        elif X.ndim == 2:
            du_lieu = X[:, 0].astype(float)
        else:
            du_lieu = X.astype(float)

        self._du_lieu = du_lieu
        self.logger.info(f"Huấn luyện trên {len(du_lieu)} điểm dữ liệu")

        if self.phuong_phap == "moving_average":
            self._huan_luyen_moving_average()
        elif self.phuong_phap == "exponential":
            self._huan_luyen_exponential()
        elif self.phuong_phap == "linear_trend":
            self._huan_luyen_linear_trend()
        elif self.phuong_phap == "window_regression":
            self._huan_luyen_window_regression()

        self.da_huan_luyen = True
        self.logger.info("Huấn luyện hoàn tất")

    def _huan_luyen_moving_average(self) -> None:
        cumsum = np.cumsum(self._du_lieu)
        cumsum[self.cua_so :] = cumsum[self.cua_so :] - cumsum[: -self.cua_so]
        self._trung_binh_truot = cumsum[self.cua_so - 1 :] / self.cua_so

    def _huan_luyen_exponential(self) -> None:
        ket_qua = np.zeros(len(self._du_lieu))
        ket_qua[0] = self._du_lieu[0]
        for i in range(1, len(self._du_lieu)):
            ket_qua[i] = self.alpha * self._du_lieu[i] + (1 - self.alpha) * ket_qua[i - 1]
        self._trung_binh_truot = ket_qua

    def _huan_luyen_linear_trend(self) -> None:
        n = len(self._du_lieu)
        x = np.arange(n)
        A = np.vstack([x, np.ones(n)]).T
        self._he_so_tuyen_tinh, _ = np.linalg.lstsq(A, self._du_lieu, rcond=None)[:2]

    def _huan_luyen_window_regression(self) -> None:
        from vietnamese_ai.models.regression import HoiQuy

        X_win, y_win = [], []
        for i in range(self.cua_so, len(self._du_lieu)):
            X_win.append(self._du_lieu[i - self.cua_so : i])
            y_win.append(self._du_lieu[i])

        X_win, y_win = np.array(X_win), np.array(y_win)
        self._mo_hinh_regression = HoiQuy(thuat_toan="ridge")
        self._mo_hinh_regression.huan_luyen(X_win, y_win)

    def du_doan(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán các bước tiếp theo.

        Args:
            X: Số bước cần dự đoán (int) hoặc dữ liệu đầu vào

        Returns:
            Mảng giá trị dự đoán
        """
        if not self.da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện.")

        if isinstance(X, (int, np.integer)):
            so_buoc = int(X)
        else:
            so_buoc = len(X)

        if self.phuong_phap == "moving_average":
            return self._du_doan_moving_average(so_buoc)
        elif self.phuong_phap == "exponential":
            return self._du_doan_exponential(so_buoc)
        elif self.phuong_phap == "linear_trend":
            return self._du_doan_linear_trend(so_buoc)
        elif self.phuong_phap == "window_regression":
            return self._du_doan_window_regression(so_buoc)

        return np.zeros(so_buoc)

    def _du_doan_moving_average(self, so_buoc: int) -> np.ndarray:
        gia_tri = self._trung_binh_truot[-1]
        return np.full(so_buoc, gia_tri)

    def _du_doan_exponential(self, so_buoc: int) -> np.ndarray:
        gia_tri_cuoi = self._trung_binh_truot[-1]
        return np.full(so_buoc, gia_tri_cuoi)

    def _du_doan_linear_trend(self, so_buoc: int) -> np.ndarray:
        n = len(self._du_lieu)
        x_moi = np.arange(n, n + so_buoc)
        return x_moi * self._he_so_tuyen_tinh[0] + self._he_so_tuyen_tinh[1]

    def _du_doan_window_regression(self, so_buoc: int) -> np.ndarray:
        ket_qua = []
        cua_so_hien_tai = self._du_lieu[-self.cua_so :].copy()

        for _ in range(so_buoc):
            du_doan = self._mo_hinh_regression.du_doan(cua_so_hien_tai.reshape(1, -1))[0]
            ket_qua.append(du_doan)
            cua_so_hien_tai = np.append(cua_so_hien_tai[1:], du_doan)

        return np.array(ket_qua)

    def danh_gia(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """Đánh giá bằng MAE trên tập test."""
        if y is None:
            return 0.0
        du_doan = self.du_doan(len(y))
        return float(np.mean(np.abs(np.asarray(y) - du_doan)))

    def lay_du_lieu_goc(self) -> np.ndarray:
        """Trả về dữ liệu gốc."""
        return self._du_lieu.copy()
