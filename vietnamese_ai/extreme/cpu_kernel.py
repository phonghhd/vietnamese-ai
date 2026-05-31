"""EvoKernelCPU - Nhân ma trận 1.58-bit (BitNet) tối ưu cho CPU."""

import logging
from typing import Any

try:
    import torch  # noqa: F401

    _CO_TORCH = True
except ImportError:
    _CO_TORCH = False

logger = logging.getLogger("EvoKernelCPU")


class EvoKernelCPU:
    """
    Nhân ma trận lượng tử hoá 1.58-bit (Trọng số chỉ là -1, 0, 1).
    Bằng cách thay thế phép nhân (Multiplication) bằng phép cộng/trừ (Addition/Subtraction),
    CPU có thể thực thi suy luận cực kỳ tiết kiệm năng lượng và RAM.
    """

    @staticmethod
    def add_only_matmul(x: Any, w_quantized: Any) -> Any:
        """
        Nhân ma trận Add-only cho kiến trúc 1.58-bit chạy trên trình biên dịch C++ JIT.
        Thay vì dùng vòng lặp Python siêu chậm (bị kẹt bởi GIL), hàm này gọi mã máy C++ thuần.

        Args:
            x: Tensor đầu vào (numpy.ndarray), kích thước (B, In).
            w_quantized: Trọng số đã lượng tử hoá (-1, 0, 1) (numpy.ndarray), kích thước (In, Out).
        Returns:
            numpy.ndarray kết quả.
        """
        import ctypes

        import numpy as np

        from vietnamese_ai.extreme.jit_engine import EvoJITCompiler

        # Đảm bảo dữ liệu là numpy array
        if hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        if hasattr(w_quantized, "cpu"):
            w_quantized = w_quantized.detach().cpu().numpy()

        x = np.ascontiguousarray(x, dtype=np.float32)
        w_quantized = np.ascontiguousarray(w_quantized, dtype=np.int8)

        batch_size, in_features = x.shape
        _, out_features = w_quantized.shape

        out = np.zeros((batch_size, out_features), dtype=np.float32)

        # 1. Mã nguồn C++ đa luồng
        cpp_code = """
        #include <stdint.h>

        extern "C" {
            // Hàm tính toán ma trận Add-Only C++
            void bitnet_matmul_cpu(
                const float* x,
                const int8_t* w,
                float* out,
                int batch_size,
                int in_features,
                int out_features
            ) {
                // Tận dụng OpenMP nếu có cờ biên dịch (bỏ qua nếu không hỗ trợ)
                #pragma omp parallel for
                for (int b = 0; b < batch_size; b++) {
                    for (int j = 0; j < out_features; j++) {
                        float sum = 0.0f;
                        for (int i = 0; i < in_features; i++) {
                            int8_t weight = w[i * out_features + j];
                            // Chỉ dùng cộng trừ, không dùng nhân
                            if (weight == 1) {
                                sum += x[b * in_features + i];
                            } else if (weight == -1) {
                                sum -= x[b * in_features + i];
                            }
                        }
                        out[b * out_features + j] = sum;
                    }
                }
            }
        }
        """

        # 2. Sinh mã JIT & Biên dịch on-the-fly
        compiler = EvoJITCompiler(use_openmp=False)  # Bật True nếu muốn dùng OMP
        c_func = compiler.compile_and_load(
            name="bitnet_core",
            code=cpp_code,
            func_name="bitnet_matmul_cpu",
            arg_types=[
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ],
            restype=None,
        )

        # 3. Ép kiểu con trỏ Ctypes (Không tốn RAM copy)
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = w_quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # 4. GỌI HÀM C++ VÀ PHÁ KHÓA GIL !!!
        c_func(x_ptr, w_ptr, out_ptr, batch_size, in_features, out_features)

        return out
