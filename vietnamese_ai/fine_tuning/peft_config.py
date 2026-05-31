"""PEFTConfig - Cấu hình Parameter-Efficient Fine-Tuning."""

from typing import Any, Dict, List, Optional


class PEFTConfig:
    """
    Cấu hình cho các phương pháp PEFT.

    Hỗ trợ:
    - LoRA: Low-Rank Adaptation
    - QLoRA: Quantized LoRA (4-bit/8-bit)
    - Prefix Tuning
    - Prompt Tuning

    Sử dụng:
        >>> config = PEFTConfig(
        ...     phuong_phap="lora",
        ...     rank=16,
        ...     alpha=16,
        ...     target_modules=["q_proj", "v_proj"],
        ... )
    """

    PHUONG_PHAP_HO_TRO = ["lora", "qlora", "prefix_tuning", "prompt_tuning"]

    def __init__(
        self,
        phuong_phap: str = "lora",
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bits: int = 4,
        bias: str = "none",
        task_type: str = "causal_lm",
        fan_in_fan_out: bool = False,
        init_weights: str = "gaussian",
    ):
        if phuong_phap not in self.PHUONG_PHAP_HO_TRO:
            raise ValueError(
                f"phuong_phap phải là một trong {self.PHUONG_PHAP_HO_TRO}, nhận: '{phuong_phap}'"
            )
        if rank <= 0:
            raise ValueError(f"rank phải > 0, nhận: {rank}")
        if bits not in (4, 8, 16):
            raise ValueError(f"bits phải là 4, 8 hoặc 16, nhận: {bits}")
        if bias not in ("none", "all", "lora_only"):
            raise ValueError(f"bias phải là 'none', 'all' hoặc 'lora_only', nhận: '{bias}'")

        self.phuong_phap = phuong_phap
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.bits = bits
        self.bias = bias
        self.task_type = task_type
        self.fan_in_fan_out = fan_in_fan_out
        self.init_weights = init_weights

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phuong_phap": self.phuong_phap,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bits": self.bits,
            "bias": self.bias,
            "task_type": self.task_type,
            "scaling": self.scaling,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PEFTConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__init__.__code__.co_varnames})

    @classmethod
    def lora(cls, rank: int = 16, alpha: float = 16.0, **kwargs) -> "PEFTConfig":
        return cls(phuong_phap="lora", rank=rank, alpha=alpha, **kwargs)

    @classmethod
    def qlora(cls, rank: int = 16, alpha: float = 16.0, bits: int = 4, **kwargs) -> "PEFTConfig":
        return cls(phuong_phap="qlora", rank=rank, alpha=alpha, bits=bits, **kwargs)

    def __repr__(self) -> str:
        return (
            f"PEFTConfig(phuong_phap='{self.phuong_phap}', rank={self.rank}, "
            f"alpha={self.alpha}, target_modules={self.target_modules})"
        )
