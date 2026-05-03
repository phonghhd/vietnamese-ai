"""MangSau - Mạng nơ-ron sâu với PyTorch backend (optional)."""

from typing import List, Optional

import numpy as np

from vietnamese_ai.models.base import BaseModel
from vietnamese_ai.utils.logger import Logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _CO_PYTORCH = True
except ImportError:
    _CO_PYTORCH = False


class MangSau(BaseModel):
    """
    Mạng nơ-ron sâu (Deep Learning).

    Tự động chọn backend:
    - PyTorch: Nếu đã cài (hỗ trợ GPU)
    - NumPy: Fallback, chạy CPU

    Sử dụng:
        >>> mang = MangSau(lop_an=[128, 64, 32], so_vong=50)
        >>> mang.huan_luyen(X_train, y_train)
        >>> du_doan = mang.du_doan(X_test)

        >>> # Với PyTorch GPU
        >>> mang = MangSau(lop_an=[128, 64], thiet_bi="cuda")
    """

    def __init__(
        self,
        lop_an: Optional[List[int]] = None,
        ham_kich_hoat: str = "relu",
        toc_do_hoc: float = 0.001,
        so_vong: int = 100,
        kich_thuoc_batch: int = 32,
        dropout: float = 0.0,
        thiet_bi: Optional[str] = None,
        ten: Optional[str] = None,
    ):
        super().__init__(ten or "MangSau")
        self.lop_an = lop_an or [128, 64, 32]
        self.ham_kich_hoat = ham_kich_hoat
        self.toc_do_hoc = toc_do_hoc
        self.so_vong = so_vong
        self.kich_thuoc_batch = kich_thuoc_batch
        self.dropout = dropout
        self.logger = Logger("MangSau")

        if thiet_bi is None:
            self.thiet_bi = "cuda" if _CO_PYTORCH and torch.cuda.is_available() else "cpu"
        else:
            self.thiet_bi = thiet_bi

        self._backend = "pytorch" if _CO_PYTORCH else "numpy"
        self._model = None
        self._loss_history: List[float] = []

    def _tao_model_pytorch(self, kich_thuoc_dau_vao: int, so_lop: int):
        """Tạo mô hình PyTorch."""
        layers = []
        prev_size = kich_thuoc_dau_vao

        for size in self.lop_an:
            layers.append(nn.Linear(prev_size, size))
            if self.ham_kich_hoat == "relu":
                layers.append(nn.ReLU())
            elif self.ham_kich_hoat == "tanh":
                layers.append(nn.Tanh())
            elif self.ham_kich_hoat == "sigmoid":
                layers.append(nn.Sigmoid())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.BatchNorm1d(size))
            prev_size = size

        layers.append(nn.Linear(prev_size, so_lop))
        return nn.Sequential(*layers).to(self.thiet_bi)

    def _huan_luyen_pytorch(self, X: np.ndarray, y: np.ndarray) -> None:
        """Huấn luyện với PyTorch backend."""
        so_lop = int(y.max()) + 1
        self._model = self._tao_model_pytorch(X.shape[1], so_lop)

        X_tensor = torch.FloatTensor(X).to(self.thiet_bi)
        y_tensor = torch.LongTensor(y.astype(int)).to(self.thiet_bi)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.kich_thuoc_batch, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self.toc_do_hoc)

        self._model.train()
        self._loss_history = []

        for vong in range(self.so_vong):
            tong_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self._model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                tong_loss += loss.item()

            avg_loss = tong_loss / len(loader)
            self._loss_history.append(avg_loss)

            if (vong + 1) % 10 == 0:
                self.logger.info(f"Vòng {vong+1}/{self.so_vong}: loss={avg_loss:.4f}")

    def _du_doan_pytorch(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán với PyTorch."""
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.thiet_bi)
            output = self._model(X_tensor)
            return output.argmax(dim=1).cpu().numpy()

    def _huan_luyen_numpy(self, X: np.ndarray, y: np.ndarray) -> None:
        """Huấn luyện với NumPy fallback (sử dụng MangNron)."""
        from vietnamese_ai.models.neural_net import MangNron

        self.logger.warning("PyTorch chưa cài, sử dụng NumPy backend")
        mang_np = MangNron(
            lop_an=self.lop_an,
            ham_kich_hoat=self.ham_kich_hoat,
            toc_do_hoc=self.toc_do_hoc,
            so_vong=self.so_vong,
            kich_thuoc_batch=self.kich_thuoc_batch,
        )
        mang_np.huan_luyen(X, y)
        self._model = mang_np
        self._loss_history = mang_np.lay_lich_su_loss()

    def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.logger.info(f"Backend: {self._backend} | Thiết bị: {self.thiet_bi}")
        self.logger.info(f"Cấu trúc: {X.shape[1]} -> {' -> '.join(map(str, self.lop_an))} -> {int(y.max())+1}")

        if self._backend == "pytorch":
            self._huan_luyen_pytorch(X, y)
        else:
            self._huan_luyen_numpy(X, y)

        self.da_huan_luyen = True
        self.logger.info("Huấn luyện hoàn tất")

    def du_doan(self, X: np.ndarray) -> np.ndarray:
        if not self.da_huan_luyen:
            raise RuntimeError("Chưa huấn luyện.")
        X = np.asarray(X, dtype=float)
        if self._backend == "pytorch":
            return self._du_doan_pytorch(X)
        return self._model.du_doan(X)

    def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
        du_doan = self.du_doan(X)
        return float(np.mean(du_doan == np.asarray(y)))

    def lay_lich_su_loss(self) -> List[float]:
        return self._loss_history.copy()

    @property
    def co_pytorch(self) -> bool:
        return _CO_PYTORCH
