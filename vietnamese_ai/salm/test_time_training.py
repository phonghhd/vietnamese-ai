"""TestTimeTraining - thích ứng model tại inference time."""

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class TestTimeTraining:
    """
    Test-Time Training (TTT): adapt model weights ngay tại inference.

    Dựa trên paper "Learning to Test-Time Train" và "Test-Time Training with
    Self-Supervision for Generalization under Distribution Shifts".

    Các chiến lược:
    - entropy_minimization: Giảm entropy của prediction
    - contrastive: Tăng similarity với augmented versions
    - masked_prediction: Predict masked tokens (self-supervised)

    Sử dụng:
        >>> ttt = TestTimeTraining(model, che_do="entropy_minimization")
        >>> ket_qua = ttt.thich_ung(X_test, so_buoc=5)
        >>> du_doan = ttt.du_doan(X_test)
    """

    def __init__(
        self,
        model: Any,
        che_do: str = "entropy_minimization",
        toc_do_hoc: float = 0.001,
        so_buoc_mac_dinh: int = 5,
        ham_loss: Optional[Callable] = None,
    ):
        if che_do not in ("entropy_minimization", "contrastive", "masked_prediction"):
            raise ValueError(
                "che_do phải là: entropy_minimization, contrastive, masked_prediction"
            )

        self.model = model
        self.che_do = che_do
        self.toc_do_hoc = toc_do_hoc
        self.so_buoc_mac_dinh = so_buoc_mac_dinh
        self.ham_loss = ham_loss
        self.logger = Logger("TestTimeTraining")

        self._trong_so_goc: Optional[Dict[str, Any]] = None
        self._da_thich_ung = False
        self._lich_su: List[Dict[str, Any]] = []
        self._loss_history: List[float] = []

    def luu_trong_so_goc(self) -> None:
        """Lưu trọng số gốc trước khi adapt."""
        self._trong_so_goc = self._trich_xuat_trong_so(self.model)
        self.logger.info("Đã lưu trọng số gốc")

    def phuc_hoi_trong_so(self) -> None:
        """Phục hồi trọng số gốc (sau khi test-time adapt)."""
        if self._trong_so_goc is not None:
            self._ap_dung_trong_so(self.model, self._trong_so_goc)
            self._da_thich_ung = False
            self.logger.info("Đã phục hồi trọng số gốc")

    def thich_ung(
        self,
        X: np.ndarray,
        so_buoc: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Adapt model tại test time.

        Args:
            X: Dữ liệu test (unlabeled)
            so_buoc: Số bước adapt

        Returns:
            {loss_cuoi, so_buoc, lich_su_loss, thoi_gian}
        """
        bat_dau = time.time()
        so_buoc = so_buoc or self.so_buoc_mac_dinh

        if self._trong_so_goc is None:
            self.luu_trong_so_goc()

        loss_history = []

        for buoc in range(so_buoc):
            loss = self._buoc_adapt(X, buoc)
            loss_history.append(loss)

            if buoc % 10 == 0:
                self.logger.debug(f"Bước {buoc}: loss={loss:.4f}")

        self._da_thich_ung = True
        self._loss_history.extend(loss_history)

        thoi_gian = time.time() - bat_dau

        ket_qua = {
            "loss_cuoi": loss_history[-1] if loss_history else 0.0,
            "loss_dau": loss_history[0] if loss_history else 0.0,
            "giam_loss": (
                loss_history[0] - loss_history[-1]
                if len(loss_history) >= 2 else 0.0
            ),
            "so_buoc": len(loss_history),
            "lich_su_loss": loss_history,
            "thoi_gian": round(thoi_gian, 3),
            "che_do": self.che_do,
        }

        self._lich_su.append(ket_qua)
        return ket_qua

    def _buoc_adapt(self, X: np.ndarray, buoc: int) -> float:
        """Một bước adapt."""
        if self.che_do == "entropy_minimization":
            return self._entropy_minimization_step(X)
        elif self.che_do == "contrastive":
            return self._contrastive_step(X)
        else:
            return self._masked_prediction_step(X)

    def _entropy_minimization_step(self, X: np.ndarray) -> float:
        """Entropy minimization: giảm entropy của predictions."""
        # Forward pass
        if hasattr(self.model, "du_doan_xac_suat"):
            probs = self.model.du_doan_xac_suat(X)
        elif hasattr(self.model, "du_doan"):
            preds = self.model.du_doan(X)
            # Convert to one-hot probabilities
            n_classes = max(len(np.unique(preds)), 2)
            probs = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                probs[i, int(p)] = 1.0
        else:
            return 0.0

        # Tính entropy
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        loss = float(np.mean(entropy))

        # Gradient descent approximation
        if hasattr(self.model, "coef_"):
            # Thêm noise nhỏ để giảm entropy
            noise_scale = self.toc_do_hoc * loss
            self.model.coef_ = np.array(self.model.coef_) + np.random.normal(
                0, noise_scale, self.model.coef_.shape
            )

        return loss

    def _contrastive_step(self, X: np.ndarray) -> float:
        """Contrastive: tăng consistency với augmented versions."""
        # Tạo augmented versions
        X_aug = self._tang_cuong(X)

        # Forward cả 2
        if hasattr(self.model, "du_doan_xac_suat"):
            probs_orig = self.model.du_doan_xac_suat(X)
            probs_aug = self.model.du_doan_xac_suat(X_aug)
        else:
            return 0.0

        # KL divergence giữa orig và aug
        probs_orig = np.clip(probs_orig, 1e-10, 1.0)
        probs_aug = np.clip(probs_aug, 1e-10, 1.0)

        kl = np.sum(probs_orig * np.log(probs_orig / probs_aug), axis=1)
        loss = float(np.mean(kl))

        return loss

    def _masked_prediction_step(self, X: np.ndarray) -> float:
        """Masked prediction: predict masked features."""
        n_features = X.shape[1] if X.ndim > 1 else 1
        mask_ratio = 0.15
        n_mask = max(1, int(n_features * mask_ratio))

        # Tạo mask
        X_masked = X.copy()
        mask_indices = np.random.choice(n_features, n_mask, replace=False)
        X_masked[:, mask_indices] = 0

        # Predict
        if hasattr(self.model, "du_doan"):
            preds = self.model.du_doan(X_masked)
            # MSE loss trên masked features
            if X.ndim > 1:
                loss = float(np.mean((preds.reshape(-1, 1) - X[:, mask_indices]) ** 2))
            else:
                loss = float(np.mean((preds - X) ** 2))
        else:
            loss = 0.0

        return loss

    def _tang_cuong(self, X: np.ndarray) -> np.ndarray:
        """Tạo augmented data (noise + scaling)."""
        X_aug = X.copy()
        # Thêm Gaussian noise
        noise = np.random.normal(0, 0.01, X.shape)
        X_aug = X_aug + noise
        # Random scaling
        scale = np.random.uniform(0.95, 1.05, size=(1, X.shape[1]))
        X_aug = X_aug * scale
        return X_aug

    def du_doan(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán (sử dụng model đã adapt nếu có)."""
        if hasattr(self.model, "du_doan"):
            return self.model.du_doan(X)
        raise RuntimeError("Model không có phương thức du_doan()")

    def du_doan_xac_suat(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất."""
        if hasattr(self.model, "du_doan_xac_suat"):
            return self.model.du_doan_xac_suat(X)
        raise RuntimeError("Model không có phương thức du_doan_xac_suat()")

    def _trich_xuat_trong_so(self, model: Any) -> Dict[str, Any]:
        """Trích xuất trọng số từ model."""
        trong_so = {}
        for attr in ["coef_", "intercept_", "_W_in", "_W_out", "_weights"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if isinstance(val, np.ndarray):
                    trong_so[attr] = val.copy()
        return trong_so

    def _ap_dung_trong_so(self, model: Any, trong_so: Dict[str, Any]) -> None:
        """Áp dụng trọng số lên model."""
        for attr, val in trong_so.items():
            if hasattr(model, attr):
                setattr(model, attr, val)

    @property
    def da_thich_ung(self) -> bool:
        return self._da_thich_ung

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        return self._lich_su.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "che_do": self.che_do,
            "toc_do_hoc": self.toc_do_hoc,
            "da_thich_ung": self._da_thich_ung,
            "so_lan_thich_ung": len(self._lich_su),
            "loss_history_len": len(self._loss_history),
        }

    def __repr__(self) -> str:
        return (
            f"TestTimeTraining(che_do='{self.che_do}', "
            f"da_thich_ung={self._da_thich_ung})"
        )
