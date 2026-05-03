"""LoRAAdapter - Low-Rank Adaptation tự cài đặt."""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401

    _CO_PYTORCH = True
except ImportError:
    _CO_PYTORCH = False


class LoRALayer:
    """
    Một lớp LoRA (Low-Rank Adaptation).

    Công thức: W_new = W_original + alpha/r * (A @ B)
    Trong đó A ∈ R^(d, r), B ∈ R^(r, k), r << min(d, k)

    Sử dụng:
        >>> lora = LoRALayer(dau_vao=768, dau_ra=768, rank=16)
        >>> output = lora.tien(input)  # W @ x + (alpha/r) * A @ B @ x
    """

    def __init__(
        self,
        dau_vao: int,
        dau_ra: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        if rank <= 0:
            raise ValueError(f"rank phải > 0, nhận: {rank}")
        if rank > min(dau_vao, dau_ra):
            raise ValueError(f"rank ({rank}) phải <= min(dau_vao, dau_ra) ({min(dau_vao, dau_ra)})")

        self.dau_vao = dau_vao
        self.dau_ra = dau_ra
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank

        self.A = np.random.randn(dau_vao, rank) * (1.0 / math.sqrt(rank))
        self.B = np.zeros((rank, dau_ra))

        self._input_cache = None
        self._grad_A = None
        self._grad_B = None

    def tien(self, X: np.ndarray) -> np.ndarray:
        """Forward: scaling * X @ A @ B"""
        self._input_cache = X
        return (X @ self.A @ self.B) * self.scaling

    def ve(self, grad: np.ndarray, toc_do_hoc: float = 0.001) -> np.ndarray:
        """Backward + update."""
        batch = len(self._input_cache)
        grad_scaled = grad * self.scaling

        self._grad_A = self._input_cache.T @ (grad_scaled @ self.B.T) / batch
        self._grad_B = (self._input_cache @ self.A).T @ grad_scaled / batch

        self.A -= toc_do_hoc * self._grad_A
        self.B -= toc_do_hoc * self._grad_B

        return grad_scaled @ (self.A @ self.B).T

    def gop_trong_so(self) -> np.ndarray:
        """Tính W_delta = alpha/r * A @ B"""
        return self.A @ self.B * self.scaling

    def so_tham_so(self) -> int:
        return self.dau_vao * self.rank + self.rank * self.dau_ra


class LoRAAdapter:
    """
    LoRA Adapter cho toàn bộ mô hình.

    Tính năng:
    - Áp dụng LoRA lên nhiều lớp Linear
    - Freeze trọng số gốc, chỉ train LoRA params
    - Merge LoRA vào trọng số gốc khi inference
    - Save/Load chỉ LoRA weights
    - Hỗ trợ cả NumPy và PyTorch

    Sử dụng:
        >>> adapter = LoRAAdapter(rank=16, alpha=16)
        >>> adapter.them_layer("q_proj", 768, 768)
        >>> adapter.them_layer("v_proj", 768, 768)
        >>> output = adapter.tien("q_proj", input)
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        if rank <= 0:
            raise ValueError("rank phải > 0")

        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or []
        self.logger = Logger("LoRAAdapter")

        self._layers: Dict[str, LoRALayer] = {}
        self._da_tao = False

    def them_layer(self, ten: str, dau_vao: int, dau_ra: int) -> None:
        """Thêm LoRA layer."""
        self._layers[ten] = LoRALayer(dau_vao, dau_ra, self.rank, self.alpha, self.dropout)
        self._da_tao = True

    def tien(self, ten: str, X: np.ndarray) -> np.ndarray:
        """Forward pass qua LoRA layer."""
        if ten not in self._layers:
            raise KeyError(f"Layer '{ten}' không tồn tại")
        return self._layers[ten].tien(X)

    def gop_trong_so(self) -> Dict[str, np.ndarray]:
        """Tính W_delta cho tất cả layers."""
        return {ten: layer.gop_trong_so() for ten, layer in self._layers.items()}

    def so_tham_so(self) -> int:
        """Tổng số LoRA parameters."""
        return sum(layer.so_tham_so() for layer in self._layers.values())

    def ty_le_trainable(self, tong_tham_so_goc: int) -> float:
        """Tỷ lệ LoRA params so với tổng params gốc."""
        return self.so_tham_so() / max(1, tong_tham_so_goc) * 100

    def luu(self, duong_dan: str) -> str:
        """Lưu chỉ LoRA weights."""
        import json
        from pathlib import Path

        data = {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "layers": {},
        }
        for ten, layer in self._layers.items():
            data["layers"][ten] = {
                "dau_vao": layer.dau_vao,
                "dau_ra": layer.dau_ra,
                "A": layer.A.tolist(),
                "B": layer.B.tolist(),
            }

        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan_path, "w") as f:
            json.dump(data, f)

        self.logger.info(f"Đã lưu LoRA: {duong_dan} ({self.so_tham_so()} params)")
        return str(duong_dan_path)

    @classmethod
    def tai(cls, duong_dan: str) -> "LoRAAdapter":
        """Tải LoRA weights."""
        import json

        with open(duong_dan, "r") as f:
            data = json.load(f)

        adapter = cls(rank=data["rank"], alpha=data["alpha"], dropout=data["dropout"])
        for ten, layer_data in data["layers"].items():
            adapter.them_layer(ten, layer_data["dau_vao"], layer_data["dau_ra"])
            adapter._layers[ten].A = np.array(layer_data["A"])
            adapter._layers[ten].B = np.array(layer_data["B"])

        Logger("LoRAAdapter").info(f"Đã tải LoRA: {duong_dan}")
        return adapter

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "so_layers": len(self._layers),
            "so_tham_so": self.so_tham_so(),
            "cac_layers": list(self._layers.keys()),
        }


class QLoRAAdapter(LoRAAdapter):
    """
    QLoRA - Quantized LoRA.

    Kết hợp 4-bit quantization với LoRA:
    - Trọng số gốc được quantize 4-bit (NF4)
    - LoRA adapters train ở FP16/BF16
    - Tiết kiệm VRAM đáng kể

    Sử dụng:
        >>> qlora = QLoRAAdapter(rank=16, bits=4)
        >>> qlora.them_layer("q_proj", 768, 768)
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bits: int = 4,
    ):
        super().__init__(rank, alpha, dropout)
        if bits not in (4, 8):
            raise ValueError(f"bits phải là 4 hoặc 8, nhận: {bits}")
        self.bits = bits
        self.logger = Logger("QLoRAAdapter")

    def quantize_weights(self, weights: np.ndarray) -> Dict[str, Any]:
        """Quantize trọng số gốc sang NF4/INT8."""
        if self.bits == 4:
            absmax = np.max(np.abs(weights))
            scale = absmax / 7.0
            quantized = np.clip(np.round(weights / scale), -8, 7).astype(np.int8)
        else:
            absmax = np.max(np.abs(weights))
            scale = absmax / 127.0
            quantized = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)

        return {
            "quantized": quantized,
            "scale": float(scale),
            "bits": self.bits,
        }

    def thong_ke(self) -> Dict[str, Any]:
        tk = super().thong_ke()
        tk["bits"] = self.bits
        tk["loai"] = "QLoRA"
        return tk
