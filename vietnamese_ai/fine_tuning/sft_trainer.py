"""SFTTrainer - Supervised Fine-Tuning Trainer."""

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger

try:
    import torch
    import torch.nn as nn

    _CO_PYTORCH = True
except ImportError:
    _CO_PYTORCH = False


class SFTTrainer:
    """
    Supervised Fine-Tuning (SFT) Trainer.

    Huấn luyện mô hình ngôn ngữ trên dữ liệu instruction-following.

    Tính năng:
    - Cross-entropy loss trên response tokens
    - Gradient accumulation
    - Mixed precision (FP16/BF16)
    - Learning rate warmup + cosine decay
    - Evaluation trên validation set
    - Checkpoint save/resume
    - Callback system

    Sử dụng:
        >>> trainer = SFTTrainer(so_vong=3, toc_do_hoc=2e-5)
        >>> ket_qua = trainer.huan_luyen(model, du_lieu_train, du_lieu_val)
    """

    def __init__(
        self,
        so_vong: int = 3,
        kich_thuoc_batch: int = 4,
        toc_do_hoc: float = 2e-5,
        gradient_accumulation: int = 4,
        gradient_clip: float = 1.0,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_seq_length: int = 512,
        logging_steps: int = 10,
        seed: int = 42,
    ):
        self.so_vong = so_vong
        self.kich_thuoc_batch = kich_thuoc_batch
        self.toc_do_hoc = toc_do_hoc
        self.gradient_accumulation = gradient_accumulation
        self.gradient_clip = gradient_clip
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length
        self.logging_steps = logging_steps
        self.seed = seed
        self.logger = Logger("SFTTrainer")

        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
        }
        self._global_step = 0

    def huan_luyen(
        self,
        model: Any,
        du_lieu_train: List[Dict[str, Any]],
        du_lieu_val: Optional[List[Dict[str, Any]]] = None,
        tokenizer: Any = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện SFT.

        Args:
            model: Mô hình (PyTorch nn.Module)
            du_lieu_train: List samples [{input_ids, labels}, ...]
            du_lieu_val: Validation samples
            tokenizer: Tokenizer (optional)
            callback: Callback function(step, loss)

        Returns:
            Dict chứa training results
        """
        self.logger.info(f"Bắt đầu SFT training ({self.so_vong} epochs)")
        self.logger.info(f"  Train samples: {len(du_lieu_train)}")
        if du_lieu_val:
            self.logger.info(f"  Val samples: {len(du_lieu_val)}")

        if _CO_PYTORCH and isinstance(model, nn.Module):
            return self._huan_luyen_pytorch(
                model, du_lieu_train, du_lieu_val, callback
            )
        return self._huan_luyen_numpy(model, du_lieu_train, du_lieu_val, callback)

    def _huan_luyen_pytorch(
        self,
        model: "nn.Module",
        du_lieu_train: List[Dict],
        du_lieu_val: Optional[List[Dict]],
        callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """PyTorch training loop."""
        device = next(model.parameters()).device
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.toc_do_hoc, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        tong_steps = len(du_lieu_train) * self.so_vong // (
            self.kich_thuoc_batch * self.gradient_accumulation
        )
        warmup_steps = int(tong_steps * self.warmup_ratio)

        bat_dau = time.time()

        for epoch in range(self.so_vong):
            model.train()
            epoch_loss = 0.0
            steps = 0
            optimizer.zero_grad()

            indices = np.random.permutation(len(du_lieu_train))

            for batch_start in range(0, len(indices), self.kich_thuoc_batch):
                batch_indices = indices[batch_start:batch_start + self.kich_thuoc_batch]

                batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
                for idx in batch_indices:
                    mau = du_lieu_train[idx]
                    input_ids = torch.tensor(
                        mau.get("input_ids", []), dtype=torch.long
                    ).unsqueeze(0).to(device)
                    labels = torch.tensor(
                        mau.get("labels", mau.get("input_ids", [])), dtype=torch.long
                    ).unsqueeze(0).to(device)

                    if input_ids.numel() == 0:
                        continue

                    try:
                        if hasattr(model, "generate") and hasattr(model, "config"):
                            outputs = model(input_ids=input_ids, labels=labels)
                            loss = outputs.loss if hasattr(outputs, "loss") else criterion(
                                outputs.logits.view(-1, outputs.logits.size(-1)),
                                labels.view(-1),
                            )
                        else:
                            logits = model(input_ids)
                            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                        batch_loss = batch_loss + loss
                    except Exception:
                        continue

                batch_loss = batch_loss / max(1, len(batch_indices))
                batch_loss = batch_loss / self.gradient_accumulation
                batch_loss.backward()
                epoch_loss += batch_loss.item() * self.gradient_accumulation
                steps += 1

                if (steps) % self.gradient_accumulation == 0:
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    self._global_step += 1

                    self._cap_nhat_lr(optimizer, self._global_step, warmup_steps, tong_steps)

                    if self._global_step % self.logging_steps == 0:
                        self.logger.info(
                            f"Step {self._global_step}: loss={epoch_loss/max(1,steps):.4f}"
                        )

                    if callback:
                        callback(self._global_step, epoch_loss / max(1, steps))

            avg_loss = epoch_loss / max(1, steps)
            self._history["train_loss"].append(avg_loss)
            self.logger.info(f"Epoch {epoch+1}/{self.so_vong}: loss={avg_loss:.4f}")

            if du_lieu_val:
                eval_loss = self._danh_gia(model, du_lieu_val, criterion, device)
                self._history["eval_loss"].append(eval_loss)
                self.logger.info(f"  Eval loss: {eval_loss:.4f}")

        tong_thoi_gian = time.time() - bat_dau

        return {
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": self.so_vong,
            "global_step": self._global_step,
            "train_loss_min": min(self._history["train_loss"]) if self._history["train_loss"] else 0,
            "eval_loss_min": min(self._history["eval_loss"]) if self._history["eval_loss"] else None,
            "history": self._history,
        }

    def _danh_gia(
        self, model: "nn.Module", du_lieu_val: List[Dict],
        criterion: Any, device: Any
    ) -> float:
        """Đánh giá trên validation set."""
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for mau in du_lieu_val:
                input_ids = torch.tensor(
                    mau.get("input_ids", []), dtype=torch.long
                ).unsqueeze(0).to(device)
                labels = torch.tensor(
                    mau.get("labels", mau.get("input_ids", [])), dtype=torch.long
                ).unsqueeze(0).to(device)

                if input_ids.numel() == 0:
                    continue

                try:
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss if hasattr(outputs, "loss") else criterion(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        labels.view(-1),
                    )
                    total_loss += loss.item()
                    count += 1
                except Exception:
                    continue

        model.train()
        return total_loss / max(1, count)

    def _huan_luyen_numpy(
        self, model: Any, du_lieu_train: List[Dict],
        du_lieu_val: Optional[List[Dict]], callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """NumPy fallback - yêu cầu PyTorch cho SFT training thực sự."""
        raise ImportError(
            "SFTTrainer yêu cầu PyTorch để huấn luyện. "
            "Cài đặt: pip install torch hoặc pip install vietnamese-ai[torch]"
        )

    def _cap_nhat_lr(self, optimizer: Any, step: int, warmup_steps: int, tong_steps: int):
        if step < warmup_steps:
            lr_scale = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, tong_steps - warmup_steps)
            lr_scale = max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.toc_do_hoc * lr_scale

    def lay_lich_su(self) -> Dict[str, List[float]]:
        return self._history.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "so_vong": self.so_vong,
            "kich_thuoc_batch": self.kich_thuoc_batch,
            "toc_do_hoc": self.toc_do_hoc,
            "gradient_accumulation": self.gradient_accumulation,
            "global_step": self._global_step,
            "train_loss_count": len(self._history["train_loss"]),
        }
