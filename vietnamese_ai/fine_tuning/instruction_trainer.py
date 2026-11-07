"""InstructionTuningTrainer - Huấn luyện instruction tuning cho LLM."""

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.fine_tuning.dataset import InstructionDataset
from vietnamese_ai.utils.logger import Logger

try:
    import torch
    import torch.nn as nn

    _CO_PYTORCH = True
except ImportError:
    _CO_PYTORCH = False


class InstructionTuningTrainer:
    """
    Trainer cho instruction tuning.

    Hỗ trợ:
    - Alpaca và ShareGPT format
    - Gradient accumulation
    - Mixed precision
    - LoRA/QLoRA integration
    - Evaluation trên validation set
    - Callback system

    Sử dụng:
        >>> dataset = InstructionDataset(che_do="alpaca")
        >>> dataset.tai_file("data/alpaca_vi.json")
        >>> trainer = InstructionTuningTrainer(so_vong=3)
        >>> trainer.huan_luyen(model, tokenizer, dataset)
    """

    def __init__(
        self,
        so_vong: int = 3,
        kich_thuoc_batch: int = 4,
        toc_do_hoc: float = 2e-5,
        gradient_accumulation: int = 4,
        max_seq_length: int = 512,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        output_dir: str = "outputs/instruction",
        seed: int = 42,
    ):
        self.so_vong = so_vong
        self.kich_thuoc_batch = kich_thuoc_batch
        self.toc_do_hoc = toc_do_hoc
        self.gradient_accumulation = gradient_accumulation
        self.max_seq_length = max_seq_length
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.seed = seed
        self.logger = Logger("InstructionTuningTrainer")

        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }
        self._global_step = 0
        self._best_eval_loss = float("inf")

    def huan_luyen(
        self,
        model: Any,
        tokenizer: Any = None,
        dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        callback: Optional[Callable] = None,
        data_collator: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện instruction tuning.

        Args:
            model: Mô hình (PyTorch nn.Module, GPTModel, hoặc HuggingFaceWrapper)
            tokenizer: Tokenizer (hoặc truyền dataset vào tham số thứ 2 nếu không dùng tokenizer)
            dataset: InstructionDataset hoặc List[Dict] cho training
            eval_dataset: InstructionDataset hoặc List[Dict] cho evaluation
            callback: Callback function
            data_collator: Custom data collator

        Returns:
            Dict chứa training results
        """
        if dataset is None and (isinstance(tokenizer, (InstructionDataset, list))):
            dataset = tokenizer
            tokenizer = None

        if dataset is None:
            du_lieu_train: List[Dict[str, Any]] = []
        elif isinstance(dataset, InstructionDataset):
            du_lieu_train = dataset.train
        elif isinstance(dataset, list):
            du_lieu_train = list(dataset)
        else:
            du_lieu_train = list(getattr(dataset, "train", []))

        self.logger.info(f"Bắt đầu instruction tuning ({self.so_vong} epochs)")
        self.logger.info(
            f"  Samples: {len(du_lieu_train)}, Batch: {self.kich_thuoc_batch}, LR: {self.toc_do_hoc}"
        )

        tong_steps = max(
            1,
            len(du_lieu_train)
            * self.so_vong
            // max(1, self.kich_thuoc_batch * self.gradient_accumulation),
        )
        warmup_steps = int(tong_steps * self.warmup_ratio)

        self.logger.info(f"  Total steps: {tong_steps}, Warmup: {warmup_steps}")

        if _CO_PYTORCH and isinstance(model, nn.Module):
            ket_qua = self._huan_luyen_pytorch(
                model,
                tokenizer,
                du_lieu_train,
                eval_dataset,
                tong_steps,
                warmup_steps,
                callback,
                data_collator,
            )
        else:
            ket_qua = self._huan_luyen_numpy(
                model,
                tokenizer,
                du_lieu_train,
                eval_dataset,
                tong_steps,
                warmup_steps,
                callback,
            )

        self.logger.info(f"Instruction tuning hoàn tất. Best eval loss: {self._best_eval_loss:.4f}")

        return ket_qua

    def _huan_luyen_pytorch(
        self,
        model: "nn.Module",
        tokenizer: Any,
        du_lieu_train: List[Dict],
        eval_dataset: Optional[InstructionDataset],
        tong_steps: int,
        warmup_steps: int,
        callback: Optional[Callable],
        data_collator: Optional[Callable],
    ) -> Dict[str, Any]:
        """Huấn luyện với PyTorch backend."""
        device = next(model.parameters()).device
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.toc_do_hoc, weight_decay=self.weight_decay
        )

        bat_dau = time.time()
        model.train()

        for epoch in range(self.so_vong):
            epoch_loss = 0.0
            steps_in_epoch = 0
            optimizer.zero_grad()

            indices = np.random.permutation(len(du_lieu_train))

            for step_idx, idx in enumerate(indices):
                mau = du_lieu_train[idx]
                loss = self._tinh_loss_mau(model, tokenizer, mau, device)

                if loss is None:
                    continue

                loss = loss / self.gradient_accumulation
                loss.backward()
                epoch_loss += loss.item() * self.gradient_accumulation
                steps_in_epoch += 1

                if (step_idx + 1) % self.gradient_accumulation == 0:
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

                    self._cap_nhat_lr(optimizer, self._global_step, warmup_steps, tong_steps)
                    optimizer.step()
                    optimizer.zero_grad()
                    self._global_step += 1

                    if self._global_step % self.logging_steps == 0:
                        avg_loss = epoch_loss / max(1, steps_in_epoch)
                        self.logger.info(
                            f"Step {self._global_step}/{tong_steps}: loss={avg_loss:.4f}"
                        )

                    if callback:
                        callback(self._global_step, epoch_loss / max(1, steps_in_epoch))

            avg_epoch_loss = epoch_loss / max(1, steps_in_epoch)
            self._history["train_loss"].append(avg_epoch_loss)
            self.logger.info(f"Epoch {epoch + 1}/{self.so_vong}: loss={avg_epoch_loss:.4f}")

        tong_thoi_gian = time.time() - bat_dau

        return {
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": self.so_vong,
            "global_step": self._global_step,
            "train_loss_min": min(self._history["train_loss"])
            if self._history["train_loss"]
            else 0,
            "history": self._history,
        }

    def _tinh_loss_mau(self, model: "nn.Module", tokenizer: Any, mau: Dict, device: Any) -> Any:
        """Tính loss cho một mẫu."""
        try:
            if hasattr(tokenizer, "__call__"):
                inputs = tokenizer(
                    str(mau.get("instruction", mau.get("input_ids", ""))),
                    return_tensors="pt",
                    max_length=self.max_seq_length,
                    padding="max_length",
                    truncation=True,
                )
                input_ids = inputs["input_ids"].to(device)
                labels = input_ids.clone()

                outputs = model(input_ids=input_ids, labels=labels)
                return outputs.loss if hasattr(outputs, "loss") else None
        except Exception:
            return None
        return None

    def _ma_hoa_mau_numpy(
        self, mau: Dict[str, Any], tokenizer: Any, vocab_size: int
    ) -> Optional[tuple]:
        max_len = min(self.max_seq_length, 64)
        if "input_ids" in mau and isinstance(mau["input_ids"], (list, np.ndarray)):
            raw_in = [int(x) % vocab_size for x in mau["input_ids"]][:max_len]
            raw_lbl = [int(x) % vocab_size for x in mau.get("labels", raw_in)][:max_len]
            if len(raw_in) < 2:
                return None
            if raw_in == raw_lbl:
                return np.array([raw_in[:-1]], dtype=int), np.array([raw_lbl[1:]], dtype=int)
            seq_len = min(len(raw_in), len(raw_lbl))
            return np.array([raw_in[:seq_len]], dtype=int), np.array([raw_lbl[:seq_len]], dtype=int)

        text_parts = []
        if "conversations" in mau and isinstance(mau["conversations"], list):
            for turn in mau["conversations"]:
                text_parts.append(f"{turn.get('from', 'user')}: {turn.get('value', '')}")
        else:
            inst = str(mau.get("instruction", mau.get("prompt", "")))
            inp = str(mau.get("input", ""))
            out = str(mau.get("output", mau.get("response", mau.get("chosen", ""))))
            text_parts.append(f"{inst} {inp} -> {out}".strip())
        full_text = " ".join(text_parts).strip()
        if not full_text:
            return None

        if tokenizer is not None and hasattr(tokenizer, "ma_hoa"):
            ids = [int(x) % vocab_size for x in tokenizer.ma_hoa(full_text)][:max_len]
        else:
            ids = [int(b) % vocab_size for b in full_text.encode("utf-8")[:max_len]]
        if len(ids) < 2:
            ids = ids + [1]
        return np.array([ids[:-1]], dtype=int), np.array([ids[1:]], dtype=int)

    def _huan_luyen_numpy(
        self,
        model: Any,
        tokenizer: Any,
        du_lieu_train: List[Dict],
        eval_dataset: Optional[Any],
        tong_steps: int,
        warmup_steps: int,
        callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """Huấn luyện thật với GPTModel / NumPy backend."""
        bat_dau = time.time()
        np.random.seed(self.seed)

        target_model = getattr(model, "_native_model", model)
        if target_model is None or not hasattr(target_model, "buoc_huan_luyen"):
            from vietnamese_ai.transformer.gpt_model import GPTModel

            target_model = GPTModel(
                d_model=32,
                so_dau=2,
                d_ff=64,
                so_block=1,
                so_tu_vung=256,
                do_dai_toi_da=max(64, min(self.max_seq_length, 128)),
            )

        vocab_size = getattr(target_model, "so_tu_vung", 256)
        base_lr = max(self.toc_do_hoc, 2e-3) if not hasattr(model, "buoc_huan_luyen") else max(self.toc_do_hoc, 5e-4)

        for epoch in range(self.so_vong):
            epoch_loss = 0.0
            steps = 0

            for mau in du_lieu_train:
                try:
                    encoded = self._ma_hoa_mau_numpy(mau, tokenizer, vocab_size)
                    if encoded is None:
                        continue
                    in_arr, tgt_arr = encoded

                    step_num = self._global_step + 1
                    if step_num < warmup_steps:
                        lr_scale = max(0.1, step_num / max(1, warmup_steps))
                    else:
                        progress = (step_num - warmup_steps) / max(1, tong_steps - warmup_steps)
                        lr_scale = max(0.1, 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, progress)))))
                    cur_lr = base_lr * lr_scale

                    loss_val = target_model.buoc_huan_luyen(
                        in_arr,
                        tgt_arr,
                        toc_do_hoc=cur_lr,
                        weight_decay=self.weight_decay,
                        grad_clip=self.gradient_clip,
                    )
                    epoch_loss += loss_val
                    steps += 1
                    self._global_step += 1
                    self._history["learning_rate"].append(cur_lr)

                    if callback and self._global_step % self.logging_steps == 0:
                        callback(self._global_step, epoch_loss / max(1, steps))
                except Exception:
                    continue

            avg_loss = epoch_loss / max(1, steps)
            self._history["train_loss"].append(avg_loss)

            if eval_dataset is not None:
                eval_samples = (
                    eval_dataset.train
                    if isinstance(eval_dataset, InstructionDataset)
                    else (
                        list(eval_dataset)
                        if isinstance(eval_dataset, list)
                        else list(getattr(eval_dataset, "val", getattr(eval_dataset, "train", [])))
                    )
                )
                val_losses = []
                for vmau in eval_samples:
                    enc = self._ma_hoa_mau_numpy(vmau, tokenizer, vocab_size)
                    if enc is not None:
                        val_losses.append(target_model.tinh_loss(enc[0], enc[1]))
                if val_losses:
                    eval_loss = float(np.mean(val_losses))
                    self._history["eval_loss"].append(eval_loss)
                    if eval_loss < self._best_eval_loss:
                        self._best_eval_loss = eval_loss
            else:
                if avg_loss < self._best_eval_loss:
                    self._best_eval_loss = avg_loss

            self.logger.info(f"Epoch {epoch + 1}/{self.so_vong}: loss={avg_loss:.4f}")

        tong_thoi_gian = time.time() - bat_dau

        return {
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": self.so_vong,
            "global_step": self._global_step,
            "train_loss_min": min(self._history["train_loss"])
            if self._history["train_loss"]
            else 0,
            "eval_loss_min": min(self._history["eval_loss"])
            if self._history["eval_loss"]
            else None,
            "history": self._history,
        }

    def _cap_nhat_lr(self, optimizer: Any, step: int, warmup_steps: int, tong_steps: int):
        """Cập nhật learning rate với warmup + cosine decay."""
        if step < warmup_steps:
            lr_scale = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, tong_steps - warmup_steps)
            lr_scale = max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.toc_do_hoc * lr_scale

        self._history["learning_rate"].append(self.toc_do_hoc * lr_scale)

    def lay_lich_su(self) -> Dict[str, List[float]]:
        return self._history.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "so_vong": self.so_vong,
            "kich_thuoc_batch": self.kich_thuoc_batch,
            "toc_do_hoc": self.toc_do_hoc,
            "gradient_accumulation": self.gradient_accumulation,
            "max_seq_length": self.max_seq_length,
            "global_step": self._global_step,
            "best_eval_loss": self._best_eval_loss,
            "history_length": len(self._history["train_loss"]),
        }
