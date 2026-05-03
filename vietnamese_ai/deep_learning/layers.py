"""Custom layers cho Deep Learning."""

from typing import Optional

import numpy as np


class LopDense:
    """Lớp Dense (Fully Connected) tự cài đặt."""

    def __init__(self, dau_vao: int, dau_ra: int, ham_kich_hoat: Optional[str] = None):
        self.dau_vao = dau_vao
        self.dau_ra = dau_ra
        self.ham_kich_hoat = ham_kich_hoat
        limit = np.sqrt(6.0 / (dau_vao + dau_ra))
        self.trong_so = np.random.uniform(-limit, limit, (dau_vao, dau_ra))
        self.bias = np.zeros((1, dau_ra))
        self._input_cache = None
        self._grad_w = None
        self._grad_b = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.tien(X)

    def tien(self, X: np.ndarray) -> np.ndarray:
        self._input_cache = X
        z = X @ self.trong_so + self.bias
        if self.ham_kich_hoat == "relu":
            return np.maximum(0, z)
        elif self.ham_kich_hoat == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        elif self.ham_kich_hoat == "tanh":
            return np.tanh(z)
        elif self.ham_kich_hoat == "softmax":
            e = np.exp(z - np.max(z, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)
        return z

    def ve(self, grad: np.ndarray, toc_do_hoc: float) -> np.ndarray:
        if self.ham_kich_hoat == "relu":
            grad = grad * (self._input_cache @ self.trong_so + self.bias > 0).astype(float)
        elif self.ham_kich_hoat == "sigmoid":
            s = self.tien(self._input_cache)
            grad = grad * s * (1 - s)
        elif self.ham_kich_hoat == "tanh":
            t = np.tanh(self._input_cache @ self.trong_so + self.bias)
            grad = grad * (1 - t ** 2)

        self._grad_w = self._input_cache.T @ grad / len(self._input_cache)
        self._grad_b = np.sum(grad, axis=0, keepdims=True) / len(self._input_cache)
        self.trong_so -= toc_do_hoc * self._grad_w
        self.bias -= toc_do_hoc * self._grad_b
        return grad @ self.trong_so.T


class LopDropout:
    """Lớp Dropout."""

    def __init__(self, ty_le: float = 0.5):
        self.ty_le = ty_le
        self._mask = None

    def __call__(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.ty_le > 0:
            self._mask = (np.random.rand(*X.shape) > self.ty_le).astype(float)
            return X * self._mask / (1 - self.ty_le)
        return X

    def ve(self, grad: np.ndarray) -> np.ndarray:
        if self._mask is not None:
            return grad * self._mask / (1 - self.ty_le)
        return grad


class LopBatchNorm:
    """Lớp Batch Normalization."""

    def __init__(self, so_dac_trung: int, momentum: float = 0.1):
        self.so_dac_trung = so_dac_trung
        self.momentum = momentum
        self.gamma = np.ones((1, so_dac_trung))
        self.beta = np.zeros((1, so_dac_trung))
        self.running_mean = np.zeros((1, so_dac_trung))
        self.running_var = np.ones((1, so_dac_trung))

    def __call__(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        X_norm = (X - mean) / np.sqrt(var + 1e-8)
        return self.gamma * X_norm + self.beta
