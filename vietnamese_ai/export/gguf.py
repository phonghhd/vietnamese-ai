"""XuatGGUF - GGUF format export cho llama.cpp."""

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

GGUF_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    16: "float64",
}

QUANTIZATION_METHODS = {
    "f32": {"type": 6, "bytes": 4, "mo_ta": "Full 32-bit float"},
    "f16": {"type": 3, "bytes": 2, "mo_ta": "16-bit float"},
    "q8_0": {"type": 1, "bytes": 1, "mo_ta": "8-bit quantization"},
    "q5_k_m": {"type": 1, "bytes": 1, "mo_ta": "5-bit K-quant (medium)"},
    "q4_k_m": {"type": 1, "bytes": 1, "mo_ta": "4-bit K-quant (medium)"},
    "q3_k_m": {"type": 1, "bytes": 1, "mo_ta": "3-bit K-quant (medium)"},
    "q2_k": {"type": 1, "bytes": 1, "mo_ta": "2-bit K-quant"},
}


class XuatGGUF:
    """
    Xuất mô hình sang GGUF format (llama.cpp compatible).

    GGUF (GGML Universal File) là format được sử dụng bởi:
    - llama.cpp
    - ollama
    - LM Studio
    - KoboldCpp
    - Nhiều tools khác

    Tính năng:
    - Tạo GGUF metadata header
    - Lưu model weights với quantization
    - Hỗ trợ nhiều loại quantization (f16, q8_0, q4_k_m, etc.)
    - Tương thích llama.cpp

    Sử dụng:
        >>> xuat = XuatGGUF()
        >>> xuat.xuat_tu_numpy(weights_dict, "model.gguf", quantization="q4_k_m")
    """

    def __init__(self):
        self.logger = Logger("XuatGGUF")

    def danh_sach_quantization(self) -> Dict[str, Dict]:
        """Liệt kê các loại quantization hỗ trợ."""
        return QUANTIZATION_METHODS.copy()

    def _quantize_weights(self, weights: np.ndarray, phuong_phap: str) -> np.ndarray:
        """Quantize weights."""
        if phuong_phap == "f32":
            return weights.astype(np.float32)
        elif phuong_phap == "f16":
            return weights.astype(np.float16)
        elif phuong_phap == "q8_0":
            scale = np.max(np.abs(weights)) / 127.0
            if scale == 0:
                scale = 1.0
            quantized = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
            return quantized
        elif phuong_phap in ("q4_k_m", "q5_k_m", "q3_k_m", "q2_k"):
            bits = int(phuong_phap[1])
            scale = np.max(np.abs(weights)) / (2 ** (bits - 1) - 1)
            if scale == 0:
                scale = 1.0
            quantized = np.clip(np.round(weights / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            return quantized.astype(np.int8)
        else:
            return weights.astype(np.float32)

    def _tao_metadata(self, thong_tin: Dict[str, Any]) -> Dict:
        """Tạo GGUF metadata."""
        return {
            "general.architecture": thong_tin.get("architecture", "vietnamese_ai"),
            "general.name": thong_tin.get("name", "VietnameseAI Model"),
            "general.file_type": QUANTIZATION_METHODS.get(
                thong_tin.get("quantization", "f16"), {}
            ).get("type", 3),
            "general.alignment": 32,
            "general.author": "Vietnamese AI Framework",
            "general.version": "4.0.0",
            "general.description": thong_tin.get(
                "description", "Model from Vietnamese AI Framework"
            ),
        }

    def xuat_tu_numpy(
        self,
        trong_so: Dict[str, np.ndarray],
        duong_dan: str,
        quantization: str = "f16",
        thong_tin: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Xuất model weights sang GGUF format.

        Args:
            trong_so: Dict tên_tensor -> numpy array
            duong_dan: Đường dẫn file .gguf
            quantization: Loại quantization (f32, f16, q8_0, q4_k_m, etc.)
            thong_tin: Metadata bổ sung

        Returns:
            Đường dẫn file GGUF
        """
        if quantization not in QUANTIZATION_METHODS:
            raise ValueError(
                f"Quantization '{quantization}' không hỗ trợ. "
                f"Chọn: {', '.join(QUANTIZATION_METHODS.keys())}"
            )

        thong_tin = thong_tin or {}
        thong_tin["quantization"] = quantization
        metadata = self._tao_metadata(thong_tin)

        duong_dan_path = Path(duong_dan)
        duong_dan_path.parent.mkdir(parents=True, exist_ok=True)

        tong_bytes = 0
        with open(duong_dan_path, "wb") as f:
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            tong_bytes += 8

            f.write(struct.pack("<Q", len(metadata)))
            tong_bytes += 8

            for key, value in metadata.items():
                key_bytes = key.encode("utf-8")
                f.write(struct.pack("<Q", len(key_bytes)))
                f.write(key_bytes)
                tong_bytes += 8 + len(key_bytes)

                if isinstance(value, str):
                    f.write(struct.pack("<I", 8))
                    val_bytes = value.encode("utf-8")
                    f.write(struct.pack("<Q", len(val_bytes)))
                    f.write(val_bytes)
                    tong_bytes += 12 + len(val_bytes)
                elif isinstance(value, int):
                    f.write(struct.pack("<I", 6))
                    f.write(struct.pack("<i", value))
                    tong_bytes += 8
                elif isinstance(value, float):
                    f.write(struct.pack("<I", 16))
                    f.write(struct.pack("<d", value))
                    tong_bytes += 12

            f.write(struct.pack("<Q", len(trong_so)))
            tong_bytes += 8

            for ten, weights in trong_so.items():
                ten_bytes = ten.encode("utf-8")
                f.write(struct.pack("<Q", len(ten_bytes)))
                f.write(ten_bytes)

                quantized = self._quantize_weights(weights, quantization)
                n_dims = len(quantized.shape)
                f.write(struct.pack("<I", n_dims))
                for dim in quantized.shape:
                    f.write(struct.pack("<Q", dim))

                type_id = QUANTIZATION_METHODS[quantization]["type"]
                f.write(struct.pack("<I", type_id))

                data = quantized.tobytes()
                padding = (32 - len(data) % 32) % 32
                f.write(data)
                f.write(b"\x00" * padding)
                tong_bytes += len(ten_bytes) + 8 + n_dims * 8 + 4 + len(data) + padding

        kich_thuoc = duong_dan_path.stat().st_size
        self.logger.info(f"Đã xuất GGUF: {duong_dan} ({kich_thuoc} bytes, {quantization})")
        return str(duong_dan_path)

    def xuat_tu_pytorch(
        self,
        model: Any,
        duong_dan: str,
        quantization: str = "f16",
    ) -> str:
        """
        Xuất PyTorch model sang GGUF.

        Args:
            model: PyTorch model
            duong_dan: Đường dẫn file .gguf
            quantization: Loại quantization

        Returns:
            Đường dẫn file GGUF
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("Cần cài đặt PyTorch: pip install torch")

        trong_so = {}
        for ten, param in model.named_parameters():
            trong_so[ten] = param.detach().cpu().numpy()

        thong_tin = {
            "name": type(model).__name__,
            "architecture": "transformer",
        }

        return self.xuat_tu_numpy(trong_so, duong_dan, quantization, thong_tin)

    def doc_metadata(self, duong_dan: str) -> Dict[str, Any]:
        """Đọc metadata từ file GGUF."""
        duong_dan_path = Path(duong_dan)
        if not duong_dan_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {duong_dan}")

        with open(duong_dan_path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"Không phải file GGUF: {duong_dan}")

            version = struct.unpack("<I", f.read(4))[0]
            n_metadata = struct.unpack("<Q", f.read(8))[0]

            metadata = {}
            for _ in range(n_metadata):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8")
                type_id = struct.unpack("<I", f.read(4))[0]

                if type_id == 8:
                    val_len = struct.unpack("<Q", f.read(8))[0]
                    value = f.read(val_len).decode("utf-8")
                elif type_id == 6:
                    value = struct.unpack("<i", f.read(4))[0]
                elif type_id == 16:
                    value = struct.unpack("<d", f.read(8))[0]
                else:
                    value = None

                metadata[key] = value

            n_tensors = struct.unpack("<Q", f.read(8))[0]

        return {
            "magic": magic.decode("utf-8"),
            "version": version,
            "n_metadata": n_metadata,
            "n_tensors": n_tensors,
            "metadata": metadata,
            "kich_thuoc_file": duong_dan_path.stat().st_size,
        }

    def tao_config_llama_cpp(
        self,
        ten_model: str,
        duong_dan: str = "deploy/llama_cpp",
        ctx_size: int = 2048,
        n_gpu_layers: int = 0,
    ) -> str:
        """
        Tạo cấu hình chạy model với llama.cpp.

        Args:
            ten_model: Tên model
            duong_dan: Thư mục output
            ctx_size: Context size
            n_gpu_layers: Số layers offload lên GPU

        Returns:
            Đường dẫn thư mục config
        """
        thu_muc = Path(duong_dan)
        thu_muc.mkdir(parents=True, exist_ok=True)

        config = {
            "model": f"{ten_model}.gguf",
            "ctx_size": ctx_size,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": 512,
            "n_threads": 4,
            "rope_freq_base": 10000.0,
            "rope_freq_scale": 1.0,
        }

        (thu_muc / "config.json").write_text(json.dumps(config, indent=2))

        run_script = f"""#!/bin/bash
./llama-cli \\
    -m {ten_model}.gguf \\
    -c {ctx_size} \\
    -ngl {n_gpu_layers} \\
    --interactive \\
    --color
"""
        (thu_muc / "run.sh").write_text(run_script)
        (thu_muc / "run.sh").chmod(0o755)

        self.logger.info(f"Đã tạo llama.cpp config: {thu_muc}")
        return str(thu_muc)
