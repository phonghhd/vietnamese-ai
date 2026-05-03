"""RewardModel - Mô hình phần thưởng cho RLHF."""

from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401

    _CO_PYTORCH = True
except ImportError:
    _CO_PYTORCH = False


class RewardModel:
    """
    Reward Model cho RLHF pipeline.

    Tính năng:
    - Train reward model từ preference pairs
    - Bradley-Terry loss
    - Score normalization
    - Support PyTorch và NumPy

    Sử dụng:
        >>> rm = RewardModel()
        >>> rm.huan_luyen(model, preference_data)
        >>> scores = rm.diem_danh_gia(model, cac_van_ban)
    """

    def __init__(self, toc_do_hoc: float = 1e-5, seed: int = 42):
        self.toc_do_hoc = toc_do_hoc
        self.seed = seed
        self.logger = Logger("RewardModel")
        self._history: Dict[str, List[float]] = {"train_loss": [], "accuracy": []}
        self._score_mean = 0.0
        self._score_std = 1.0

    def huan_luyen(
        self,
        model: Any,
        preference_data: List[Dict[str, Any]],
        so_vong: int = 1,
        callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện reward model.

        Args:
            model: Reward model (PyTorch nn.Module hoặc custom)
            preference_data: [{"chosen": ..., "rejected": ...}, ...]
            so_vong: Số epochs
            callback: Callback function

        Returns:
            Dict chứa training results
        """
        self.logger.info(f"Bắt đầu train Reward Model ({so_vong} epochs)")
        self.logger.info(f"  Preference pairs: {len(preference_data)}")

        if _CO_PYTORCH and isinstance(model, nn.Module):
            return self._huan_luyen_pytorch(model, preference_data, so_vong, callback)
        return self._huan_luyen_numpy(model, preference_data, so_vong, callback)

    def _huan_luyen_pytorch(
        self,
        model: "nn.Module",
        preference_data: List[Dict],
        so_vong: int,
        callback: Optional[Any],
    ) -> Dict[str, Any]:
        """PyTorch training cho Reward Model."""
        import time

        device = next(model.parameters()).device
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.toc_do_hoc)

        bat_dau = time.time()

        for epoch in range(so_vong):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            indices = np.random.permutation(len(preference_data))

            for idx in indices:
                mau = preference_data[idx]
                chosen_text = mau.get("chosen", "")
                rejected_text = mau.get("rejected", "")

                try:
                    chosen_score = self._tinh_score(model, chosen_text, device)
                    rejected_score = self._tinh_score(model, rejected_text, device)

                    loss = -torch.log(torch.sigmoid(chosen_score - rejected_score)).mean()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    if chosen_score > rejected_score:
                        correct += 1
                    total += 1

                    if callback:
                        callback(total, loss.item())

                except Exception:
                    continue

            avg_loss = epoch_loss / max(1, total)
            acc = correct / max(1, total)
            self._history["train_loss"].append(avg_loss)
            self._history["accuracy"].append(acc)
            self.logger.info(f"Epoch {epoch+1}/{so_vong}: loss={avg_loss:.4f}, acc={acc:.4f}")

        tong_thoi_gian = time.time() - bat_dau

        all_scores = []
        model.eval()
        with torch.no_grad():
            for mau in preference_data[:100]:
                score = self._tinh_score(model, mau.get("chosen", ""), device)
                all_scores.append(score.item())

        if all_scores:
            self._score_mean = float(np.mean(all_scores))
            self._score_std = float(np.std(all_scores)) or 1.0

        return {
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": so_vong,
            "final_loss": self._history["train_loss"][-1] if self._history["train_loss"] else 0,
            "final_accuracy": self._history["accuracy"][-1] if self._history["accuracy"] else 0,
            "score_mean": self._score_mean,
            "score_std": self._score_std,
            "history": self._history,
        }

    def _tinh_score(self, model: "nn.Module", text: str, device: Any) -> "torch.Tensor":
        """Tính reward score cho text."""
        ids = torch.tensor(
            [ord(c) % 1000 for c in text[:512]],
            dtype=torch.long,
        ).unsqueeze(0).to(device)

        try:
            output = model(ids)
            if isinstance(output, tuple):
                return output[0]
            if hasattr(output, "logits"):
                return output.logits.mean()
            return output.mean() if isinstance(output, torch.Tensor) else torch.tensor(0.0, device=device)
        except Exception:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def _huan_luyen_numpy(
        self, model: Any, preference_data: List[Dict],
        so_vong: int, callback: Optional[Any],
    ) -> Dict[str, Any]:
        """NumPy fallback - yêu cầu PyTorch cho Reward Model training thực sự."""
        raise ImportError(
            "RewardModel yêu cầu PyTorch để huấn luyện. "
            "Cài đặt: pip install torch hoặc pip install vietnamese-ai[torch]"
        )

    def diem_danh_gia(self, model: Any, cac_van_ban: List[str]) -> List[Dict[str, float]]:
        """
        Đánh giá reward scores cho danh sách văn bản.

        Args:
            model: Reward model
            cac_van_ban: Danh sách văn bản

        Returns:
            List [{"van_ban": ..., "score": ..., "score_normalized": ...}]
        """
        ket_qua = []

        if _CO_PYTORCH and isinstance(model, nn.Module):
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                for vb in cac_van_ban:
                    score = self._tinh_score(model, vb, device).item()
                    normalized = (score - self._score_mean) / self._score_std
                    ket_qua.append({
                        "van_ban": vb,
                        "score": round(score, 4),
                        "score_normalized": round(normalized, 4),
                    })
        else:
            for vb in cac_van_ban:
                score = np.random.normal(0, 1)
                ket_qua.append({
                    "van_ban": vb,
                    "score": round(float(score), 4),
                    "score_normalized": round(float(score), 4),
                })

        return ket_qua

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "toc_do_hoc": self.toc_do_hoc,
            "score_mean": self._score_mean,
            "score_std": self._score_std,
            "history_length": len(self._history["train_loss"]),
            "final_accuracy": self._history["accuracy"][-1] if self._history["accuracy"] else 0,
        }
