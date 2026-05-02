"""Hồi quy - Các thuật toán hồi quy học máy."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from vietnamese_ai.models.base import BaseModel


class HoiQuy(BaseModel):
    """
    Bộ hồi quy đa thuật toán.

    Hỗ trợ các thuật toán:
    - 'tuyen_tinh': Hồi quy tuyến tính (mặc định)
    - 'ridge': Ridge Regression
    - 'lasso': Lasso Regression
    - 'elastic_net': Elastic Net
    - 'svm': Support Vector Regression
    - 'cay_quyet_dinh': Cây quyết định hồi quy
    - 'rung_ngau_nhien': Rừng ngẫu nhiên hồi quy
    - 'gradient_boosting': Gradient Boosting hồi quy

    Sử dụng:
        >>> hoi_quy = HoiQuy(thuat_toan='tuyen_tinh')
        >>> hoi_quy.huan_luyen(X_train, y_train)
        >>> du_doan = hoi_quy.du_doan(X_test)
        >>> mse = hoi_quy.danh_gia(X_test, y_test)
    """

    THUAT_TOAN = {
        "tuyen_tinh": SKLinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        "svm": SVR,
        "cay_quyet_dinh": DecisionTreeRegressor,
        "rung_ngau_nhien": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
    }

    def __init__(
        self,
        thuat_toan: str = "tuyen_tinh",
        ten: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(ten or f"HoiQuy({thuat_toan})")

        if thuat_toan not in self.THUAT_TOAN:
            hop_le = ", ".join(self.THUAT_TOAN.keys())
            raise ValueError(
                f"Thuật toán '{thuat_toan}' không hợp lệ. "
                f"Chọn một trong: {hop_le}"
            )

        self.thuat_toan = thuat_toan
        self.tham_so = kwargs
        self._mo_hinh = self.THUAT_TOAN[thuat_toan](**kwargs)

    def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        self._mo_hinh.fit(X, y)
        self.da_huan_luyen = True

    def du_doan(self, X: np.ndarray) -> np.ndarray:
        if not self.da_huan_luyen:
            raise RuntimeError("Mô hình chưa được huấn luyện. Gọi huan_luyen() trước.")
        return self._mo_hinh.predict(np.asarray(X))

    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        """Đánh giá bằng MSE (Mean Squared Error)."""
        du_doan = self.du_doan(X)
        return float(mean_squared_error(np.asarray(y), du_doan))

    def bao_cao(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Báo cáo chi tiết các chỉ số hồi quy."""
        y = np.asarray(y)
        du_doan = self.du_doan(X)
        return {
            "mse": float(mean_squared_error(y, du_doan)),
            "rmse": float(np.sqrt(mean_squared_error(y, du_doan))),
            "mae": float(mean_absolute_error(y, du_doan)),
            "r2": float(r2_score(y, du_doan)),
        }
