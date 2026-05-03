"""HuanLuyenPyTorch - PyTorch Trainer với GPU, Mixed Precision, Checkpoint."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from vietnamese_ai.utils.logger import Logger

if TYPE_CHECKING:
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _CO_PYTORCH = True
    _no_grad = torch.no_grad
except ImportError:
    _CO_PYTORCH = False

    @contextmanager
    def _no_grad():
        yield


class HuanLuyenPyTorch:
    """
    PyTorch Trainer nâng cao.

    Tính năng:
    - GPU auto-detect (CUDA, MPS, CPU)
    - Mixed Precision Training (FP16/BF16)
    - Gradient Accumulation
    - Learning Rate Scheduler (cosine, linear, step)
    - Early Stopping
    - Checkpoint save/resume
    - Gradient clipping
    - Training history logging

    Sử dụng:
        >>> trainer = HuanLuyenPyTorch(
        ...     thiet_bi="auto",
        ...     hon_lep=True,
        ...     gradient_accumulation=4,
        ... )
        >>> trainer.huan_luyen(model, X_train, y_train, X_val, y_val)
        >>> trainer.luu_checkpoint("checkpoint.pt")
    """

    def __init__(
        self,
        thiet_bi: str = "auto",
        toc_do_hoc: float = 1e-3,
        so_vong: int = 100,
        kich_thuoc_batch: int = 32,
        hon_lep: bool = False,
        gradient_accumulation: int = 1,
        gradient_clip: float = 1.0,
        early_stopping: int = 0,
        scheduler: str = "none",
        seed: int = 42,
    ):
        if not _CO_PYTORCH:
            raise ImportError(
                "Cần cài đặt PyTorch: pip install torch "
                "hoặc pip install vietnamese-ai[torch]"
            )

        if thiet_bi == "auto":
            if torch.cuda.is_available():
                self.thiet_bi = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.thiet_bi = "mps"
            else:
                self.thiet_bi = "cpu"
        else:
            self.thiet_bi = thiet_bi

        self.toc_do_hoc = toc_do_hoc
        self.so_vong = so_vong
        self.kich_thuoc_batch = kich_thuoc_batch
        self.hon_lep = hon_lep
        self.gradient_accumulation = gradient_accumulation
        self.gradient_clip = gradient_clip
        self.early_stopping = early_stopping
        self.scheduler_type = scheduler
        self.seed = seed
        self.logger = Logger("HuanLuyenPyTorch")

        self._model: Optional[nn.Module] = None
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }
        self._best_val_loss = float("inf")
        self._best_model_state = None
        self._patience_counter = 0

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def co_pytorch(self) -> bool:
        return _CO_PYTORCH

    @property
    def co_gpu(self) -> bool:
        return self.thiet_bi != "cpu"

    def _tao_optimizer(self, model: nn.Module) -> None:
        """Tạo optimizer AdamW."""
        self._optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.toc_do_hoc, weight_decay=0.01
        )

    def _tao_scheduler(self, tong_steps: int) -> None:
        """Tạo learning rate scheduler."""
        if self.scheduler_type == "cosine":
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=tong_steps
            )
        elif self.scheduler_type == "linear":
            self._scheduler = torch.optim.lr_scheduler.LinearLR(
                self._optimizer, start_factor=1.0, end_factor=0.1, total_iters=tong_steps
            )
        elif self.scheduler_type == "step":
            self._scheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer, step_size=tong_steps // 3, gamma=0.5
            )

    def _tao_scaler(self) -> None:
        """Tạo GradScaler cho mixed precision."""
        if self.hon_lep and self.thiet_bi == "cuda":
            self._scaler = torch.amp.GradScaler("cuda")

    def _chuan_bi_du_lieu(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> DataLoader:
        """Chuyển dữ liệu sang DataLoader."""
        X_tensor = torch.FloatTensor(np.asarray(X, dtype=np.float32))
        if y is not None:
            y_tensor = torch.LongTensor(np.asarray(y, dtype=np.int64))
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        return DataLoader(
            dataset, batch_size=self.kich_thuoc_batch, shuffle=(y is not None)
        )

    def huan_luyen(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        ham_loss: Optional[Any] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện mô hình PyTorch.

        Args:
            model: Mô hình PyTorch (nn.Module)
            X_train, y_train: Dữ liệu huấn luyện
            X_val, y_val: Dữ liệu validation (tùy chọn)
            ham_loss: Hàm loss (mặc định: CrossEntropyLoss)
            callback: Callback sau mỗi epoch

        Returns:
            Dict chứa history và best metrics
        """
        self._model = model.to(self.thiet_bi)
        self._tao_optimizer(model)
        self._tao_scaler()

        train_loader = self._chuan_bi_du_lieu(X_train, y_train)
        val_loader = self._chuan_bi_du_lieu(X_val, y_val) if X_val is not None else None

        tong_steps = self.so_vong * len(train_loader) // self.gradient_accumulation
        self._tao_scheduler(tong_steps)

        criterion = ham_loss or nn.CrossEntropyLoss()
        self._history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

        self.logger.info(f"Bắt đầu huấn luyện trên {self.thiet_bi}")
        self.logger.info(
            f"  Epochs={self.so_vong}, Batch={self.kich_thuoc_batch}, "
            f"Mixed Precision={self.hon_lep}, Grad Accum={self.gradient_accumulation}"
        )

        bat_dau = time.time()

        for epoch in range(self.so_vong):
            self._model.train()
            tong_loss = 0.0
            dung = 0
            tong = 0
            self._optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                X_batch = batch[0].to(self.thiet_bi)
                y_batch = batch[1].to(self.thiet_bi)

                if self.hon_lep and self.thiet_bi == "cuda":
                    with torch.amp.autocast("cuda"):
                        output = self._model(X_batch)
                        loss = criterion(output, y_batch) / self.gradient_accumulation
                    self._scaler.scale(loss).backward()
                else:
                    output = self._model(X_batch)
                    loss = criterion(output, y_batch) / self.gradient_accumulation
                    loss.backward()

                tong_loss += loss.item() * self.gradient_accumulation

                if (step + 1) % self.gradient_accumulation == 0:
                    if self.gradient_clip > 0:
                        if self._scaler:
                            self._scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), self.gradient_clip
                        )

                    if self._scaler:
                        self._scaler.step(self._optimizer)
                        self._scaler.update()
                    else:
                        self._optimizer.step()

                    self._optimizer.zero_grad()
                    if self._scheduler:
                        self._scheduler.step()

                preds = output.argmax(dim=1)
                dung += (preds == y_batch).sum().item()
                tong += len(y_batch)

            avg_loss = tong_loss / len(train_loader)
            acc = dung / tong
            self._history["train_loss"].append(avg_loss)
            self._history["train_acc"].append(acc)
            self._history["lr"].append(self._optimizer.param_groups[0]["lr"])

            val_loss = 0.0
            val_acc = 0.0
            if val_loader:
                val_loss, val_acc = self._danh_gia_val(val_loader, criterion)
                self._history["val_loss"].append(val_loss)
                self._history["val_acc"].append(val_acc)

                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._best_model_state = {
                        k: v.cpu().clone() for k, v in self._model.state_dict().items()
                    }
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1

                if self.early_stopping > 0 and self._patience_counter >= self.early_stopping:
                    self.logger.info(f"Early stopping tại epoch {epoch + 1}")
                    break

            if (epoch + 1) % max(1, self.so_vong // 10) == 0:
                msg = f"Epoch {epoch+1}/{self.so_vong}: loss={avg_loss:.4f}, acc={acc:.4f}"
                if val_loader:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                self.logger.info(msg)

            if callback:
                callback(epoch + 1, avg_loss, acc)

        tong_thoi_gian = time.time() - bat_dau

        if self._best_model_state is not None:
            self._model.load_state_dict(self._best_model_state)
            self.logger.info("Đã khôi phục model tốt nhất")

        self.logger.info(f"Huấn luyện hoàn tất ({tong_thoi_gian:.1f}s)")

        return {
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": len(self._history["train_loss"]),
            "train_loss_min": min(self._history["train_loss"]),
            "val_loss_min": self._best_val_loss if val_loader else None,
            "train_acc_max": max(self._history["train_acc"]),
            "val_acc_max": max(self._history["val_acc"]) if val_loader else None,
            "history": self._history,
        }

    @_no_grad()
    def _danh_gia_val(self, val_loader: DataLoader, criterion: Any) -> Tuple[float, float]:
        """Đánh giá trên validation set."""
        self._model.eval()
        tong_loss = 0.0
        dung = 0
        tong = 0

        for batch in val_loader:
            X_batch = batch[0].to(self.thiet_bi)
            y_batch = batch[1].to(self.thiet_bi)
            output = self._model(X_batch)
            tong_loss += criterion(output, y_batch).item()
            preds = output.argmax(dim=1)
            dung += (preds == y_batch).sum().item()
            tong += len(y_batch)

        return tong_loss / len(val_loader), dung / tong

    @_no_grad()
    def du_doan(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán."""
        if self._model is None:
            raise RuntimeError("Chưa huấn luyện.")

        self._model.eval()
        X_tensor = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(self.thiet_bi)
        output = self._model(X_tensor)
        return output.argmax(dim=1).cpu().numpy()

    @_no_grad()
    def du_doan_xac_suat(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất."""
        if self._model is None:
            raise RuntimeError("Chưa huấn luyện.")

        self._model.eval()
        X_tensor = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(self.thiet_bi)
        output = self._model(X_tensor)
        return torch.softmax(output, dim=1).cpu().numpy()

    def lay_lich_su(self) -> Dict[str, List[float]]:
        """Lấy lịch sử huấn luyện."""
        return self._history.copy()

    def luu_checkpoint(self, duong_dan: str) -> str:
        """Lưu checkpoint."""
        if self._model is None:
            raise RuntimeError("Chưa huấn luyện.")

        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "history": self._history,
            "thiet_bi": self.thiet_bi,
            "best_val_loss": self._best_val_loss,
        }
        if self._scheduler:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()

        torch.save(checkpoint, duong_dan)
        self.logger.info(f"Đã lưu checkpoint: {duong_dan}")
        return str(duong_dan_path)

    def tai_checkpoint(self, duong_dan: str, model: nn.Module) -> None:
        """Tải checkpoint."""
        checkpoint = torch.load(duong_dan, map_location=self.thiet_bi, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        self._model = model.to(self.thiet_bi)
        self._history = checkpoint.get("history", self._history)
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.logger.info(f"Đã tải checkpoint: {duong_dan}")

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê trainer."""
        return {
            "thiet_bi": self.thiet_bi,
            "co_gpu": self.co_gpu,
            "hon_lep": self.hon_lep,
            "so_vong": self.so_vong,
            "kich_thuoc_batch": self.kich_thuoc_batch,
            "gradient_accumulation": self.gradient_accumulation,
            "gradient_clip": self.gradient_clip,
            "scheduler": self.scheduler_type,
            "so_epoch_da_train": len(self._history["train_loss"]),
        }
