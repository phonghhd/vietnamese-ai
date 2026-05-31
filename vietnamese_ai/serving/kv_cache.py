"""Quantized KV Cache - Tiết kiệm RAM qua ép kiểu FP8/INT8."""

from typing import Tuple

import numpy as np


class QuantizedKVCache:
    """
    Bộ nhớ KV Cache siêu tiết kiệm.
    Thay vì lưu trữ dữ liệu dạng float32 (4 bytes) hoặc float16 (2 bytes),
    module này ép kiểu (Quantize) ma trận xuống int8 (1 byte).
    Giúp giảm 50%-75% lượng RAM cần thiết cho context dài.
    """

    def __init__(self, max_seq_len: int, d_model: int):
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Lưu trữ mảng kiểu int8
        self.k_cache = np.zeros((max_seq_len, d_model), dtype=np.int8)
        self.v_cache = np.zeros((max_seq_len, d_model), dtype=np.int8)

        # Scaling factor để giải nén
        self.k_scales = np.zeros((max_seq_len, 1), dtype=np.float32)
        self.v_scales = np.zeros((max_seq_len, 1), dtype=np.float32)

        self.current_len = 0

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Nén Float32 -> Int8.
        - Tìm phần tử có trị tuyệt đối lớn nhất làm max_val.
        - Tỷ lệ: scale = max_val / 127
        - Int8 Tensor = round(tensor / scale)
        """
        max_vals = np.max(np.abs(tensor), axis=-1, keepdims=True)
        # Chống chia cho 0
        max_vals = np.maximum(max_vals, 1e-9)

        scales = max_vals / 127.0
        quantized = np.round(tensor / scales).astype(np.int8)

        return quantized, scales

    def dequantize(self, quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Giải nén Int8 -> Float32.
        - Float32 Tensor = quantized * scale
        """
        return quantized.astype(np.float32) * scales

    def append(self, k_new: np.ndarray, v_new: np.ndarray):
        """
        Lưu thêm Key/Value mới vào Cache.
        k_new, v_new shape: (seq_len, d_model)
        """
        seq_len = k_new.shape[0]
        if self.current_len + seq_len > self.max_seq_len:
            raise ValueError("Vượt quá độ dài KV Cache tối đa (Context Window).")

        k_q, k_scale = self.quantize(k_new)
        v_q, v_scale = self.quantize(v_new)

        start = self.current_len
        end = self.current_len + seq_len

        self.k_cache[start:end] = k_q
        self.k_scales[start:end] = k_scale

        self.v_cache[start:end] = v_q
        self.v_scales[start:end] = v_scale

        self.current_len += seq_len

    def get_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy toàn bộ Cache ra để tính toán (Dequantize on-the-fly).
        """
        if self.current_len == 0:
            return None, None

        k_out = self.dequantize(self.k_cache[: self.current_len], self.k_scales[: self.current_len])
        v_out = self.dequantize(self.v_cache[: self.current_len], self.v_scales[: self.current_len])

        return k_out, v_out
