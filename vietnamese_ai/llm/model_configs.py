"""ModelConfigs - Cấu hình Vietnamese LLM từ 125M đến 7B."""

from typing import Any, Dict


class ModelConfig:
    """
    Cấu hình cho Vietnamese LLM.

    Pre-defined configs:
    - vnlm-tiny: 10M params (testing)
    - vnlm-small: 125M params
    - vnlm-medium: 350M params
    - vnlm-large: 1.3B params
    - vnlm-xl: 2.7B params
    - vnlm-7b: 6.8B params

    Sử dụng:
        >>> config = ModelConfig.from_preset("vnlm-small")
        >>> model = GPTModel(**config.to_dict())
    """

    PRESETS = {
        "vnlm-tiny": {
            "d_model": 64,
            "so_dau": 2,
            "d_ff": 256,
            "so_block": 2,
            "so_tu_vung": 1000,
            "do_dai_toi_da": 256,
            "dropout": 0.1,
            "mo_ta": "Tiny model cho testing (~10M params)",
        },
        "vnlm-small": {
            "d_model": 768,
            "so_dau": 12,
            "d_ff": 3072,
            "so_block": 12,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 2048,
            "dropout": 0.1,
            "mo_ta": "Small model 125M params (GPT-2 small)",
        },
        "vnlm-medium": {
            "d_model": 1024,
            "so_dau": 16,
            "d_ff": 4096,
            "so_block": 24,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 2048,
            "dropout": 0.1,
            "mo_ta": "Medium model 350M params",
        },
        "vnlm-large": {
            "d_model": 2048,
            "so_dau": 32,
            "d_ff": 8192,
            "so_block": 24,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 4096,
            "dropout": 0.1,
            "mo_ta": "Large model 1.3B params",
        },
        "vnlm-xl": {
            "d_model": 2560,
            "so_dau": 32,
            "d_ff": 10240,
            "so_block": 32,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 4096,
            "dropout": 0.1,
            "mo_ta": "XL model 2.7B params",
        },
        "vnlm-7b": {
            "d_model": 4096,
            "so_dau": 32,
            "d_ff": 11008,
            "so_block": 32,
            "so_tu_vung": 32000,
            "do_dai_toi_da": 4096,
            "dropout": 0.0,
            "mo_ta": "7B params (LLaMA-2 style)",
        },
    }

    def __init__(
        self,
        d_model: int = 768,
        so_dau: int = 12,
        d_ff: int = 3072,
        so_block: int = 12,
        so_tu_vung: int = 32000,
        do_dai_toi_da: int = 2048,
        dropout: float = 0.1,
        ten: str = "custom",
        mo_ta: str = "",
    ):
        if d_model % so_dau != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho so_dau ({so_dau})")

        self.d_model = d_model
        self.so_dau = so_dau
        self.d_ff = d_ff
        self.so_block = so_block
        self.so_tu_vung = so_tu_vung
        self.do_dai_toi_da = do_dai_toi_da
        self.dropout = dropout
        self.ten = ten
        self.mo_ta = mo_ta

    @classmethod
    def from_preset(cls, preset: str) -> "ModelConfig":
        if preset not in cls.PRESETS:
            raise ValueError(
                f"Preset '{preset}' không tồn tại. "
                f"Chọn: {list(cls.PRESETS.keys())}"
            )
        cfg = cls.PRESETS[preset]
        return cls(ten=preset, **cfg)

    @property
    def so_tham_so(self) -> int:
        return (
            self.so_tu_vung * self.d_model
            + self.so_block * (
                self.d_model * self.d_model * 4
                + self.d_model * self.d_ff * 2
                + self.d_model * 4
            )
            + self.d_model * 2
        )

    @property
    def so_tham_so_str(self) -> str:
        params = self.so_tham_so
        if params >= 1e9:
            return f"{params/1e9:.1f}B"
        elif params >= 1e6:
            return f"{params/1e6:.0f}M"
        return f"{params/1e3:.0f}K"

    @classmethod
    def danh_sach_presets(cls) -> Dict[str, str]:
        return {k: v["mo_ta"] for k, v in cls.PRESETS.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "so_dau": self.so_dau,
            "d_ff": self.d_ff,
            "so_block": self.so_block,
            "so_tu_vung": self.so_tu_vung,
            "do_dai_toi_da": self.do_dai_toi_da,
            "dropout": self.dropout,
        }

    def __repr__(self) -> str:
        return (
            f"ModelConfig(ten='{self.ten}', "
            f"d_model={self.d_model}, so_block={self.so_block}, "
            f"params={self.so_tham_so_str})"
        )
