"""PhanLoaiHinhAnh - Phân loại hình ảnh (CNN với PyTorch hoặc feature extraction)."""

from typing import Optional, Tuple

import numpy as np

from vietnamese_ai.models.base import BaseModel
from vietnamese_ai.utils.logger import Logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _CO_PYTORCH = True

    class _CNNModel(nn.Module):
        """Mạng CNN đơn giản cho phân loại hình ảnh."""

        def __init__(self, kenh_dau_vao: int, so_lop: int, kich_thuoc: Tuple[int, int] = (32, 32)):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(kenh_dau_vao, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, so_lop),
            )

        def forward(self, x):
            return self.fc(self.conv(x))

except ImportError:
    _CO_PYTORCH = False
    _CNNModel = None


class PhanLoaiHinhAnh(BaseModel):
    """
    Phân loại hình ảnh với CNN.

    Hỗ trợ 2 chế độ:
    - PyTorch CNN: Nếu đã cài PyTorch
    - Feature extraction + ML: Flatten + PhanLoai (fallback)

    Sử dụng:
        >>> plha = PhanLoaiHinhAnh(so_lop=10, so_vong=20)
        >>> plha.huan_luyen(X_train, y_train)  # X shape: (N, C, H, W) hoặc (N, H, W)
        >>> du_doan = plha.du_doan(X_test)
    """

    def __init__(
        self,
        so_lop: int = 10,
        kenh_dau_vao: int = 1,
        kich_thuoc: Tuple[int, int] = (32, 32),
        so_vong: int = 20,
        toc_do_hoc: float = 0.001,
        kich_thuoc_batch: int = 32,
        ten: Optional[str] = None,
    ):
        super().__init__(ten or f"PhanLoaiHinhAnh({so_lop}lop)")
        self.so_lop = so_lop
        self.kenh_dau_vao = kenh_dau_vao
        self.kich_thuoc = kich_thuoc
        self.so_vong = so_vong
        self.toc_do_hoc = toc_do_hoc
        self.kich_thuoc_batch = kich_thuoc_batch
        self.logger = Logger("PhanLoaiHinhAnh")

        self._backend = "pytorch" if _CO_PYTORCH else "numpy"
        self._model = None
        self._ml_model = None

    def _chuan_hoa_du_lieu(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.max() > 1.0:
            X = X / 255.0
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        elif X.ndim == 2:
            batch = X.shape[0]
            side = int(np.sqrt(X.shape[1]))
            X = X.reshape(batch, 1, side, side)
        return X

    def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self._chuan_hoa_du_lieu(X)
        y = np.asarray(y, dtype=int)
        self.logger.info(f"Backend: {self._backend} | Dữ liệu: {X.shape}")

        if self._backend == "pytorch":
            self._huan_luyen_pytorch(X, y)
        else:
            self._huan_luyen_numpy(X, y)

        self.da_huan_luyen = True
        self.logger.info("Huấn luyện hoàn tất")

    def _huan_luyen_pytorch(self, X: np.ndarray, y: np.ndarray) -> None:
        thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = _CNNModel(X.shape[1], self.so_lop).to(thiet_bi)

        X_t = torch.FloatTensor(X).to(thiet_bi)
        y_t = torch.LongTensor(y).to(thiet_bi)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.kich_thuoc_batch, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self.toc_do_hoc)

        self._model.train()
        for vong in range(self.so_vong):
            tong_loss = 0.0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(X_b), y_b)
                loss.backward()
                optimizer.step()
                tong_loss += loss.item()
            if (vong + 1) % 5 == 0:
                self.logger.info(
                    f"Vòng {vong + 1}/{self.so_vong}: loss={tong_loss / len(loader):.4f}"
                )

    def _huan_luyen_numpy(self, X: np.ndarray, y: np.ndarray) -> None:
        self.logger.warning("PyTorch chưa cài, dùng feature extraction + Random Forest")
        from vietnamese_ai.models.classifier import PhanLoai

        X_flat = X.reshape(X.shape[0], -1)
        self._ml_model = PhanLoai(thuat_toan="rung_ngau_nhien", n_estimators=100)
        self._ml_model.huan_luyen(X_flat, y)

    def du_doan(self, X: np.ndarray) -> np.ndarray:
        if not self.da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện.")
        X = self._chuan_hoa_du_lieu(X)

        if self._backend == "pytorch" and self._model is not None:
            thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(thiet_bi)
                return self._model(X_t).argmax(dim=1).cpu().numpy()
        else:
            X_flat = X.reshape(X.shape[0], -1)
            return self._ml_model.du_doan(X_flat)

    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        du_doan = self.du_doan(X)
        return float(np.mean(du_doan == np.asarray(y, dtype=int)))
