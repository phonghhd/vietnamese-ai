"""Kernel Fusion - Dung hợp Toán tử tối thiểu hoá Băng thông RAM."""

import ctypes

import numpy as np

from vietnamese_ai.extreme.jit_engine import EvoJITCompiler


class FusedDenseNormReLU:
    """
    Dung hợp 4 Toán tử: Nhân ma trận (Dense) + Bias + LayerNorm + ReLU.
    Thay vì Python gọi 4 vòng lặp NumPy rời rạc (tốn 4 lần đọc/ghi RAM),
    C++ JIT sẽ chạy 1 vòng lặp duy nhất trên thanh ghi (Registers) / L1 Cache.
    Giảm 75% độ trễ băng thông.
    """

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)

        self.gamma = np.ones(out_features, dtype=np.float32)
        self.beta = np.zeros(out_features, dtype=np.float32)

        self.compiler = EvoJITCompiler(use_openmp=True)
        self._compile_kernel()

    def _compile_kernel(self):
        """Khởi tạo mã C++ Fused Kernel."""
        cpp_code = """
        #include <math.h>

        extern "C" {
            void fused_dense_norm_relu(
                const float* x,
                const float* w,
                const float* b,
                const float* gamma,
                const float* beta,
                float* out,
                int batch,
                int in_f,
                int out_f
            ) {
                #pragma omp parallel for
                for (int i = 0; i < batch; i++) {
                    // Bước 1: Nhân ma trận + Bias (MatMul + Bias)
                    // L1 Cache locality
                    for (int j = 0; j < out_f; j++) {
                        float sum = b[j];
                        for (int k = 0; k < in_f; k++) {
                            sum += x[i * in_f + k] * w[k * out_f + j];
                        }
                        out[i * out_f + j] = sum;
                    }

                    // Bước 2: LayerNorm (Tính Mean & Variance cục bộ)
                    float mean = 0.0f;
                    for (int j = 0; j < out_f; j++) {
                        mean += out[i * out_f + j];
                    }
                    mean /= out_f;

                    float var = 0.0f;
                    for (int j = 0; j < out_f; j++) {
                        float diff = out[i * out_f + j] - mean;
                        var += diff * diff;
                    }
                    var /= out_f;
                    float std = sqrt(var + 1e-5f);

                    // Bước 3: Áp dụng Norm + ReLU + Ghi xuống RAM
                    for (int j = 0; j < out_f; j++) {
                        float norm_val = (out[i * out_f + j] - mean) / std;
                        float final_val = norm_val * gamma[j] + beta[j];

                        // ReLU Fusion
                        if (final_val < 0) final_val = 0.0f;

                        out[i * out_f + j] = final_val;
                    }
                }
            }
        }
        """

        self.c_func = self.compiler.compile_and_load(
            name="fused_kernel",
            code=cpp_code,
            func_name="fused_dense_norm_relu",
            arg_types=[
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ],
            restype=None,
        )

    def tien(self, X: np.ndarray) -> np.ndarray:
        """Thực thi Fused Kernel qua C++."""
        batch_size = X.shape[0]
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        out = np.zeros((batch_size, self.out_features), dtype=np.float32)

        x_ptr = X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = self.W.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = self.b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        gamma_ptr = self.gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        beta_ptr = self.beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.c_func(
            x_ptr,
            w_ptr,
            b_ptr,
            gamma_ptr,
            beta_ptr,
            out_ptr,
            batch_size,
            self.in_features,
            self.out_features,
        )

        return out
