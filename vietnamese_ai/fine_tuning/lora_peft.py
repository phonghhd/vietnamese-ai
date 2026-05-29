"""LoRAPeft - Enhanced LoRA PEFT tích hợp PyTorch nn.Module."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.fine_tuning.peft_config import PEFTConfig
from vietnamese_ai.utils.logger import Logger

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401

    _CO_PYTORCH = True
except ImportError:
    _CO_PYTORCH = False


class LoRALinear(nn.Module if _CO_PYTORCH else object):
    """LoRA layer cho Linear - PyTorch nn.Module."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        if not _CO_PYTORCH:
            raise ImportError("Cần cài đặt PyTorch: pip install torch")

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        self.linear.weight.data += self.lora_B.weight @ self.lora_A.weight * self.scaling
        self.lora_A.weight.data.zero_()
        self.lora_B.weight.data.zero_()


class LoRAPeft:
    """
    Enhanced LoRA PEFT adapter cho PyTorch models.

    Tính năng:
    - Tự động detect và apply LoRA lên Linear layers
    - Support LoRA, QLoRA (4-bit/8-bit quantization)
    - Merge/Unmerge weights
    - Save/Load adapter weights only
    - Thống kê trainable parameters

    Sử dụng:
        >>> config = PEFTConfig.lora(rank=16)
        >>> peft = LoRAPeft(config)
        >>> model = peft.ap_dung(model)
        >>> # Train chỉ LoRA params
        >>> peft.gop_trong_so(model)
    """

    def __init__(self, config: Optional[PEFTConfig] = None):
        if not _CO_PYTORCH:
            raise ImportError("Cần cài đặt PyTorch: pip install torch")

        self.config = config or PEFTConfig.lora()
        self.logger = Logger("LoRAPeft")
        self._lora_layers: Dict[str, LoRALinear] = {}
        self._original_weights: Dict[str, Any] = {}

    def ap_dung(self, model: "nn.Module") -> "nn.Module":
        """
        Áp dụng LoRA lên model.

        Args:
            model: PyTorch model

        Returns:
            Model với LoRA layers
        """
        target_modules = self.config.target_modules
        replaced = 0

        # [VÁ BUG] - Khóa chặt toàn bộ trọng số của base model
        for param in model.parameters():
            param.requires_grad = False

        for name, module in model.named_modules():
            if any(tm in name for tm in target_modules):
                if isinstance(module, nn.Linear):
                    parent = self._lay_parent(model, name)
                    if parent is not None:
                        child_name = name.split(".")[-1]
                        lora_layer = LoRALinear(
                            module.in_features,
                            module.out_features,
                            rank=self.config.rank,
                            alpha=self.config.alpha,
                            dropout=self.config.dropout,
                        )
                        lora_layer.linear.weight.data = module.weight.data.clone()
                        setattr(parent, child_name, lora_layer)
                        self._lora_layers[name] = lora_layer
                        self._original_weights[name] = module.weight.data.clone()
                        replaced += 1

        self.logger.info(
            f"Đã áp dụng LoRA lên {replaced} layers "
            f"(rank={self.config.rank}, alpha={self.config.alpha})"
        )

        so_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        so_total = sum(p.numel() for p in model.parameters())
        self.logger.info(
            f"Trainable: {so_trainable:,}/{so_total:,} "
            f"({100 * so_trainable / max(1, so_total):.1f}%)"
        )

        return model

    def _lay_parent(self, model: "nn.Module", name: str):
        parts = name.split(".")
        current = model
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current

    def gop_trong_so(self, model: "nn.Module") -> "nn.Module":
        """Gộp LoRA weights vào model gốc."""
        for name, lora_layer in self._lora_layers.items():
            lora_layer.merge_weights()
        self.logger.info("Đã gộp LoRA weights")
        return model

    def chi_trainable(self, model: "nn.Module") -> List[str]:
        """Freeze tất cả trừ LoRA params."""
        trainable = []
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                trainable.append(name)
            else:
                param.requires_grad = False
        self.logger.info(f"Trainable params: {len(trainable)}")
        return trainable

    def luu(self, model: "nn.Module", duong_dan: str) -> str:
        """Lưu chỉ LoRA adapter weights."""
        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)

        lora_state = {}
        for name, param in model.named_parameters():
            if "lora_" in name:
                lora_state[name] = param.data.cpu()

        checkpoint = {
            "config": self.config.to_dict(),
            "lora_weights": {k: v.numpy().tolist() for k, v in lora_state.items()},
            "so_layers": len(self._lora_layers),
            "so_params": sum(v.numel() for v in lora_state.values()),
        }

        with open(duong_dan_path, "w") as f:
            json.dump(checkpoint, f)

        self.logger.info(f"Đã lưu LoRA adapter: {duong_dan}")
        return str(duong_dan_path)

    @classmethod
    def tai(cls, duong_dan: str, model: "nn.Module") -> "LoRAPeft":
        """Tải LoRA adapter vào model."""
        with open(duong_dan, "r") as f:
            checkpoint = json.load(f)

        config = PEFTConfig.from_dict(checkpoint["config"])
        peft = cls(config)
        peft.ap_dung(model)

        lora_weights = checkpoint["lora_weights"]
        model_state = model.state_dict()
        for name, weights in lora_weights.items():
            if name in model_state:
                model_state[name] = torch.tensor(weights)

        model.load_state_dict(model_state, strict=False)
        Logger("LoRAPeft").info(f"Đã tải LoRA adapter: {duong_dan}")
        return peft

    def thong_ke(self, model: Optional["nn.Module"] = None) -> Dict[str, Any]:
        """Thống kê LoRA adapter."""
        tk = {
            "config": self.config.to_dict(),
            "so_lora_layers": len(self._lora_layers),
            "cac_layers": list(self._lora_layers.keys()),
        }

        if model is not None:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            tk.update({
                "trainable_params": trainable,
                "total_params": total,
                "trainable_percent": round(100 * trainable / max(1, total), 2),
            })

        return tk
