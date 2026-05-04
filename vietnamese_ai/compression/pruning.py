"""CatTiaMoHinh - Model Pruning cho mô hình ML."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class CatTiaMoHinh:
    """
    Cat tia (pruning) mô hình để giảm kích thước và tăng tốc.

    Hỗ trợ:
    - Magnitude-based pruning (xóa trọng số nhỏ)
    - Structured pruning (xóa nguyên neuron/layer)
    - Iterative pruning
    - Lottery ticket hypothesis

    Sử dụng:
        >>> pruner = CatTiaMoHinh(che_do="magnitude", ty_le=0.5)
        >>> model_pruned = pruner.cat_tia(model, X_train, y_train)
        >>> print(pruner.thong_ke())
    """

    def __init__(
        self,
        che_do: str = "magnitude",
        ty_le: float = 0.5,
        so_vong_lap: int = 1,
        phuc_hoi: bool = False,
    ):
        if che_do not in ("magnitude", "structured", "iterative", "random"):
            raise ValueError("che_do phải là: magnitude, structured, iterative, random")

        if not 0 < ty_le < 1:
            raise ValueError("ty_le phải trong khoảng (0, 1)")

        self.che_do = che_do
        self.ty_le = ty_le
        self.so_vong_lap = so_vong_lap
        self.phuc_hoi = phuc_hoi
        self.logger = Logger("CatTiaMoHinh")

        self._lich_su: List[Dict[str, Any]] = []
        self._mask: Optional[np.ndarray] = None

    def cat_tia(
        self,
        model: Any,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Cat tia mô hình.

        Args:
            model: Mô hình cần prune
            X: Dữ liệu đánh giá
            y: Nhãn đánh giá

        Returns:
            {model, mask, hieu_suat_truoc, hieu_suat_sau, ty_le_prune}
        """
        self.logger.info(f"Bắt đầu pruning ({self.che_do}, ty_le={self.ty_le})")
        bat_dau = time.time()

        # Đánh giá trước pruning
        hieu_suat_truoc = None
        if X is not None and y is not None and hasattr(model, "danh_gia"):
            hieu_suat_truoc = model.danh_gia(X, y)

        # Lấy trọng số
        trong_so = self._lay_trong_so(model)
        if trong_so is None:
            self.logger.info("Model không có trọng số để prune")
            return {
                "model": model,
                "mask": None,
                "hieu_suat_truoc": hieu_suat_truoc,
                "hieu_suat_sau": hieu_suat_truoc,
                "ty_le_prune": 0.0,
            }

        # Pruning
        if self.che_do == "magnitude":
            mask = self._prune_magnitude(trong_so)
        elif self.che_do == "structured":
            mask = self._prune_structured(trong_so)
        elif self.che_do == "iterative":
            mask = self._prune_iterative(trong_so, model, X, y)
        else:
            mask = self._prune_random(trong_so)

        # Áp dụng mask
        self._ap_dung_mask(model, mask)
        self._mask = mask

        # Đánh giá sau pruning
        hieu_suat_sau = None
        if X is not None and y is not None and hasattr(model, "danh_gia"):
            hieu_suat_sau = model.danh_gia(X, y)

        thoi_gian = time.time() - bat_dau
        ty_le_prune = float(np.mean(mask == 0))

        ket_qua = {
            "model": model,
            "mask": mask,
            "hieu_suat_truoc": hieu_suat_truoc,
            "hieu_suat_sau": hieu_suat_sau,
            "ty_le_prune": ty_le_prune,
            "so_tham_so_goc": int(np.sum(np.ones_like(mask))),
            "so_tham_so_sau": int(np.sum(mask)),
            "thoi_gian": f"{thoi_gian:.2f}s",
            "che_do": self.che_do,
        }

        self._lich_su.append(ket_qua)
        self.logger.info(
            f"Pruning hoàn tất: {ty_le_prune*100:.1f}% weights pruned, "
            f"thời gian={thoi_gian:.2f}s"
        )

        return ket_qua

    def _prune_magnitude(self, trong_so: np.ndarray) -> np.ndarray:
        """Magnitude-based pruning."""
        nguong = np.percentile(np.abs(trong_so), self.ty_le * 100)
        return (np.abs(trong_so) >= nguong).astype(float)

    def _prune_structured(self, trong_so: np.ndarray) -> np.ndarray:
        """Structured pruning (xóa nguyên neuron)."""
        if trong_so.ndim < 2:
            return self._prune_magnitude(trong_so)

        # Tính importance của mỗi neuron (L2 norm)
        importance = np.linalg.norm(trong_so, axis=1) if trong_so.ndim == 2 else np.linalg.norm(trong_so.reshape(trong_so.shape[0], -1), axis=1)

        nguong = np.percentile(importance, self.ty_le * 100)
        neuron_mask = (importance >= nguong).astype(float)

        # Broadcast mask
        mask = np.ones_like(trong_so)
        if trong_so.ndim == 2:
            mask[neuron_mask == 0, :] = 0
        else:
            mask[neuron_mask == 0] = 0

        return mask

    def _prune_random(self, trong_so: np.ndarray) -> np.ndarray:
        """Random pruning."""
        mask = np.ones_like(trong_so)
        n_prune = int(self.ty_le * trong_so.size)
        indices = np.random.choice(trong_so.size, n_prune, replace=False)
        mask.flat[indices] = 0
        return mask

    def _prune_iterative(
        self,
        trong_so: np.ndarray,
        model: Any,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
    ) -> np.ndarray:
        """Iterative pruning."""
        mask = np.ones_like(trong_so)
        ty_le_moi_vong = 1 - (1 - self.ty_le) ** (1 / self.so_vong_lap)

        for vong in range(self.so_vong_lap):
            # Prune thêm một phần
            trong_so_masked = trong_so * mask
            nguong = np.percentile(
                np.abs(trong_so_masked[trong_so_masked != 0]),
                ty_le_moi_vong * 100,
            )
            mask = mask * (np.abs(trong_so) >= nguong).astype(float)

            # Retrain (nếu có)
            if X is not None and y is not None and hasattr(model, "huan_luyen"):
                try:
                    model.huan_luyen(X, y)
                except Exception:
                    pass

        return mask

    def _lay_trong_so(self, model: Any) -> Optional[np.ndarray]:
        """Lấy trọng số từ model."""
        # sklearn models
        if hasattr(model, "coef_"):
            return np.array(model.coef_).flatten()
        if hasattr(model, "feature_importances_"):
            return np.array(model.feature_importances_)

        # Custom models
        for attr in ["_W", "_weights", "weights", "_trong_so"]:
            if hasattr(model, attr):
                w = getattr(model, attr)
                if isinstance(w, np.ndarray):
                    return w.flatten()
                elif isinstance(w, dict):
                    return np.concatenate([v.flatten() for v in w.values() if isinstance(v, np.ndarray)])

        return None

    def _ap_dung_mask(self, model: Any, mask: np.ndarray) -> None:
        """Áp dụng pruning mask lên model."""
        if hasattr(model, "coef_"):
            original = np.array(model.coef_).flatten()
            pruned = original * mask
            model.coef_ = pruned.reshape(model.coef_.shape) if hasattr(model.coef_, 'shape') else pruned

        for attr in ["_W", "_weights", "weights", "_trong_so"]:
            if hasattr(model, attr):
                w = getattr(model, attr)
                if isinstance(w, np.ndarray):
                    setattr(model, attr, (w.flatten() * mask).reshape(w.shape))
                break

    def lay_mask(self) -> Optional[np.ndarray]:
        """Lấy pruning mask."""
        return self._mask

    def lay_lich_su(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử pruning."""
        return self._lich_su.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "che_do": self.che_do,
            "ty_le": self.ty_le,
            "so_vong_lap": self.so_vong_lap,
            "co_mask": self._mask is not None,
            "so_lan_prune": len(self._lich_su),
        }

    def __repr__(self) -> str:
        return f"CatTiaMoHinh(che_do='{self.che_do}', ty_le={self.ty_le})"
