"""Mạng nơ-ron - Neural Network cơ bản."""

from typing import List, Optional, Tuple

import numpy as np

from vietnamese_ai.models.base import BaseModel


class MangNron(BaseModel):
    """
    Mạng nơ-ron nhân tạo cơ bản (tự cài đặt, không phụ thuộc PyTorch/TensorFlow).

    Hỗ trợ:
    - Tùy chỉnh số lớp ẩn và số nơ-ron mỗi lớp
    - Các hàm kích hoạt: relu, sigmoid, tanh
    - Lan truyền ngược (backpropagation)
    - Mini-batch gradient descent

    Sử dụng:
        >>> mang = MangNron(lop_an=[64, 32], ham_kich_hoat='relu')
        >>> mang.huan_luyen(X_train, y_train, so_vong=100)
        >>> du_doan = mang.du_doan(X_test)
    """

    def __init__(
        self,
        lop_an: Optional[List[int]] = None,
        ham_kich_hoat: str = "relu",
        toc_do_hoc: float = 0.01,
        so_vong: int = 100,
        kich_thuoc_batch: int = 32,
        ten: Optional[str] = None,
    ):
        super().__init__(ten or "MangNron")

        self.lop_an = lop_an or [64, 32]
        self.ham_kich_hoat = ham_kich_hoat
        self.toc_do_hoc = toc_do_hoc
        self.so_vong = so_vong
        self.kich_thuoc_batch = kich_thuoc_batch

        self._trong_so: List[np.ndarray] = []
        self._he_so: List[np.ndarray] = []
        self._loss_history: List[float] = []

    @staticmethod
    def _kich_hoat_relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def _dao_ham_relu(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def _kich_hoat_sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _kich_hoat_tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def _dao_ham_tanh(z: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(z) ** 2

    def _lay_ham_kich_hoat(self, z: np.ndarray) -> np.ndarray:
        if self.ham_kich_hoat == "relu":
            return self._kich_hoat_relu(z)
        elif self.ham_kich_hoat == "sigmoid":
            return self._kich_hoat_sigmoid(z)
        elif self.ham_kich_hoat == "tanh":
            return self._kich_hoat_tanh(z)
        raise ValueError(f"Hàm kích hoạt '{self.ham_kich_hoat}' không hỗ trợ")

    def _lay_dao_ham_kich_hoat(self, z: np.ndarray) -> np.ndarray:
        if self.ham_kich_hoat == "relu":
            return self._dao_ham_relu(z)
        elif self.ham_kich_hoat == "sigmoid":
            s = self._kich_hoat_sigmoid(z)
            return s * (1 - s)
        elif self.ham_kich_hoat == "tanh":
            return self._dao_ham_tanh(z)
        raise ValueError(f"Hàm kích hoạt '{self.ham_kich_hoat}' không hỗ trợ")

    def _khoi_tao_trong_so(self, kich_thuoc_dau_vao: int, kich_thuoc_dau_ra: int) -> None:
        cac_lop = [kich_thuoc_dau_vao] + self.lop_an + [kich_thuoc_dau_ra]
        self._trong_so = []
        self._he_so = []

        np.random.seed(42)
        for i in range(len(cac_lop) - 1):
            w = np.random.randn(cac_lop[i], cac_lop[i + 1]) * np.sqrt(2.0 / cac_lop[i])
            b = np.zeros((1, cac_lop[i + 1]))
            self._trong_so.append(w)
            self._he_so.append(b)

    def _lan_truyen_thuan(self, X: np.ndarray) -> Tuple[List, List]:
        cac_z = []
        cac_a = [X]

        hien_tai = X
        for i in range(len(self._trong_so)):
            z = hien_tai @ self._trong_so[i] + self._he_so[i]
            cac_z.append(z)

            if i < len(self._trong_so) - 1:
                a = self._lay_ham_kich_hoat(z)
            else:
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                a = exp_z / np.sum(exp_z, axis=1, keepdims=True)

            cac_a.append(a)
            hien_tai = a

        return cac_z, cac_a

    def _lan_truyen_nguoc(self, X: np.ndarray, y: np.ndarray, cac_z: List, cac_a: List) -> None:
        so_mau = X.shape[0]
        so_lop = self._trong_so[-1].shape[1]

        # One-hot encoding
        y_onehot = np.zeros((so_mau, so_lop))
        y_onehot[np.arange(so_mau), y.astype(int)] = 1

        # Gradient lớp đầu ra
        dz = cac_a[-1] - y_onehot

        for i in range(len(self._trong_so) - 1, -1, -1):
            dw = cac_a[i].T @ dz / so_mau
            db = np.sum(dz, axis=0, keepdims=True) / so_mau

            self._trong_so[i] -= self.toc_do_hoc * dw
            self._he_so[i] -= self.toc_do_hoc * db

            if i > 0:
                dz = (dz @ self._trong_so[i].T) * self._lay_dao_ham_kich_hoat(cac_z[i - 1])

    def _tinh_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        so_mau = len(y_true)
        y_onehot = np.zeros_like(y_pred)
        y_onehot[np.arange(so_mau), y_true.astype(int)] = 1
        y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.sum(y_onehot * np.log(y_pred_clip)) / so_mau)

    def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        so_lop = len(np.unique(y))
        self._khoi_tao_trong_so(X.shape[1], so_lop)
        self._loss_history = []

        for vong in range(self.so_vong):
            indices = np.random.permutation(len(X))

            for bat_dau in range(0, len(X), self.kich_thuoc_batch):
                batch_idx = indices[bat_dau:bat_dau + self.kich_thuoc_batch]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                cac_z, cac_a = self._lan_truyen_thuan(X_batch)
                self._lan_truyen_nguoc(X_batch, y_batch, cac_z, cac_a)

            _, cac_a_full = self._lan_truyen_thuan(X)
            loss = self._tinh_loss(cac_a_full[-1], y)
            self._loss_history.append(loss)

        self.da_huan_luyen = True

    def du_doan(self, X: np.ndarray) -> np.ndarray:
        if not self.da_huan_luyen:
            raise RuntimeError("Mô hình chưa được huấn luyện.")
        _, cac_a = self._lan_truyen_thuan(np.asarray(X, dtype=float))
        return np.argmax(cac_a[-1], axis=1)

    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        du_doan = self.du_doan(X)
        return float(np.mean(du_doan == np.asarray(y)))

    def lay_lich_su_loss(self) -> List[float]:
        return self._loss_history.copy()
