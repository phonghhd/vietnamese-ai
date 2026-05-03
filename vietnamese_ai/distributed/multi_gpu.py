"""MultiGPUTrainer - Hỗ trợ huấn luyện đa GPU."""

from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class MultiGPUTrainer:
    """
    Hỗ trợ huấn luyện trên nhiều GPU.

    Tính năng:
    - Data Parallelism: Chia dữ liệu cho nhiều GPU
    - Tự động phát hiện GPU có sẵn
    - Gradient accumulation cho batch lớn
    - Mixed precision training (FP16)

    Sử dụng:
        >>> trainer = MultiGPUTrainer()
        >>> trainer.huan_luyen(mo_hinh, X, y, so_gpu=2)

    CLI:
        vai train --data data.csv --model logistic --gpu 2
    """

    def __init__(self):
        self.logger = Logger("MultiGPU")
        self._gpu_list = self._phat_hien_gpu()

    def _phat_hien_gpu(self) -> List[Dict]:
        """Phát hiện GPU có sẵn."""
        gpu_list = []

        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_list.append({
                        "id": i,
                        "ten": torch.cuda.get_device_name(i),
                        "bo_nho": torch.cuda.get_device_properties(i).total_mem,
                    })
                self.logger.info(f"Phát hiện {len(gpu_list)} GPU")
            else:
                self.logger.info("Không tìm thấy GPU, sử dụng CPU")
        except ImportError:
            self.logger.info("PyTorch chưa cài, sử dụng CPU")

        return gpu_list

    @property
    def so_gpu(self) -> int:
        return len(self._gpu_list)

    @property
    def co_gpu(self) -> bool:
        return self.so_gpu > 0

    def huan_luyen(
        self,
        mo_hinh: Any,
        X: np.ndarray,
        y: np.ndarray,
        so_gpu: Optional[int] = None,
        gradient_accumulation: int = 1,
        mixed_precision: bool = False,
    ) -> Dict[str, Any]:
        """
        Huấn luyện mô hình với multi-GPU support.

        Args:
            mo_hinh: Mô hình PyTorch hoặc wrapper
            X: Dữ liệu đầu vào
            y: Nhãn
            so_gpu: Số GPU sử dụng (None = tất cả)
            gradient_accumulation: Số bước tích lũy gradient
            mixed_precision: Sử dụng FP16 mixed precision

        Returns:
            Dict chứa thông tin huấn luyện
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y)

        if not self.co_gpu:
            self.logger.info("Không có GPU, huấn luyện trên CPU")
            mo_hinh.huan_luyen(X, y)
            return {"thiet_bi": "cpu", "so_gpu": 0}

        so_gpu = so_gpu or self.so_gpu
        so_gpu = min(so_gpu, self.so_gpu)

        try:
            return self._huan_luyen_pytorch(mo_hinh, X, y, so_gpu, gradient_accumulation, mixed_precision)
        except Exception as e:
            self.logger.warning(f"Multi-GPU thất bại, fallback CPU: {e}")
            mo_hinh.huan_luyen(X, y)
            return {"thiet_bi": "cpu_fallback", "so_gpu": 0, "loi": str(e)}

    def _huan_luyen_pytorch(
        self, mo_hinh, X, y, so_gpu, gradient_accumulation, mixed_precision
    ) -> Dict:
        """Huấn luyện với PyTorch DataParallel."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        thiet_bi = "cuda:0"
        model = mo_hinh._model if hasattr(mo_hinh, "_model") else mo_hinh

        if so_gpu > 1 and isinstance(model, nn.Module):
            model = nn.DataParallel(model, device_ids=list(range(so_gpu)))
            self.logger.info(f"DataParallel trên {so_gpu} GPU")

        model = model.to(thiet_bi)

        X_t = torch.FloatTensor(X).to(thiet_bi)
        y_t = torch.LongTensor(y.astype(int)).to(thiet_bi)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scaler = None
        if mixed_precision and torch.cuda.is_available():
            scaler = torch.amp.GradScaler("cuda")
            self.logger.info("Mixed precision (FP16) enabled")

        model.train()
        for vong in range(getattr(mo_hinh, "so_vong", 10)):
            tong_loss = 0.0
            for i, (X_b, y_b) in enumerate(loader):
                optimizer.zero_grad()

                if scaler:
                    with torch.amp.autocast("cuda"):
                        output = model(X_b)
                        loss = criterion(output, y_b)
                    scaler.scale(loss).backward()
                    if (i + 1) % gradient_accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    output = model(X_b)
                    loss = criterion(output, y_b)
                    loss.backward()
                    if (i + 1) % gradient_accumulation == 0:
                        optimizer.step()

                tong_loss += loss.item()

            if (vong + 1) % 5 == 0:
                self.logger.info(f"Vòng {vong+1}: loss={tong_loss/len(loader):.4f}")

        if hasattr(mo_hinh, "_model"):
            if isinstance(model, nn.DataParallel):
                mo_hinh._model = model.module
            else:
                mo_hinh._model = model
        mo_hinh.da_huan_luyen = True

        return {
            "thiet_bi": thiet_bi,
            "so_gpu": so_gpu,
            "mixed_precision": mixed_precision,
            "gradient_accumulation": gradient_accumulation,
        }

    def thong_tin_gpu(self) -> List[Dict]:
        """Trả về thông tin GPU."""
        return self._gpu_list.copy()
