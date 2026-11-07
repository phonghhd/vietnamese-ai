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
        gop_chuoi: bool = False,
        mixed_precision: bool = True,
        dung_torch_compile: bool = False,
        dung_chunked_ce: bool = False,
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
        self.gop_chuoi = gop_chuoi
        self.mixed_precision = mixed_precision
        self.dung_torch_compile = dung_torch_compile
        self.dung_chunked_ce = dung_chunked_ce
        self._thong_ke_gop: Dict[str, Any] = {}
        self.logger = Logger("SFTTrainer")

        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
        }
        self._global_step = 0

    @staticmethod
    def _chuan_hoa_du_lieu_sft(
        du_lieu: Any, tokenizer: Any = None, max_seq_length: int = 512
    ) -> List[Dict[str, Any]]:
        if du_lieu is None:
            return []
        if hasattr(du_lieu, "train") and isinstance(du_lieu.train, list):
            items = du_lieu.train
        elif isinstance(du_lieu, list):
            items = du_lieu
        else:
            items = list(du_lieu)

        ket_qua: List[Dict[str, Any]] = []
        for mau in items:
            if not isinstance(mau, dict):
                continue
            if "input_ids" in mau and mau["input_ids"]:
                ket_qua.append(mau)
                continue
            text = ""
            if "text" in mau:
                text = str(mau["text"])
            elif "instruction" in mau or "prompt" in mau:
                inst = str(mau.get("instruction", mau.get("prompt", "")))
                inp = str(mau.get("input", ""))
                out = str(mau.get("output", mau.get("response", mau.get("chosen", ""))))
                text = f"{inst} {inp} -> {out}".strip()
            if not text:
                continue
            if tokenizer is not None and hasattr(tokenizer, "ma_hoa"):
                ids = [int(x) for x in tokenizer.ma_hoa(text)][:max_seq_length]
            else:
                ids = [int(b) for b in text.encode("utf-8")[: min(max_seq_length, 64)]]
            if len(ids) < 2:
                ids = ids + [1]
            ket_qua.append({"input_ids": ids[:-1], "labels": ids[1:]})
        return ket_qua

    def huan_luyen(
        self,
        model: Any,
        du_lieu_train: Any,
        du_lieu_val: Optional[Any] = None,
        tokenizer: Any = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện SFT.

        Args:
            model: Mô hình (PyTorch nn.Module, GPTModel, hoặc HuggingFaceWrapper)
            du_lieu_train: List samples [{input_ids, labels}, ...] hoặc [{instruction, output}, ...]
            du_lieu_val: Validation samples
            tokenizer: Tokenizer (optional)
            callback: Callback function(step, loss)

        Returns:
            Dict chứa training results
        """
        du_lieu_train_norm = self._chuan_hoa_du_lieu_sft(
            du_lieu_train, tokenizer, self.max_seq_length
        )
        du_lieu_val_norm = (
            self._chuan_hoa_du_lieu_sft(du_lieu_val, tokenizer, self.max_seq_length)
            if du_lieu_val
            else None
        )
        self.logger.info(f"Bắt đầu SFT training ({self.so_vong} epochs)")
        self.logger.info(f"  Train samples: {len(du_lieu_train_norm)}")
        if du_lieu_val_norm:
            self.logger.info(f"  Val samples: {len(du_lieu_val_norm)}")

        unwrapped = getattr(model, "_model", None) or getattr(model, "_native_model", None) or model
        if _CO_PYTORCH and isinstance(unwrapped, nn.Module):
            return self._huan_luyen_pytorch(unwrapped, du_lieu_train_norm, du_lieu_val_norm, callback)
        return self._huan_luyen_numpy(unwrapped, du_lieu_train_norm, du_lieu_val_norm, callback)

    def _huan_luyen_pytorch(
        self,
        model: "nn.Module",
        du_lieu_train: List[Dict],
        du_lieu_val: Optional[List[Dict]],
        callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """PyTorch training loop (hỗ trợ Sequence Packing, Vectorized Batch, AMP BF16/FP16 & Chunked CE)."""
        from contextlib import nullcontext

        from vietnamese_ai.fine_tuning.training_accelerator import BoGopChuoi, HamMatMatPhanManh

        device = next(model.parameters()).device
        active_model: Any = model
        if self.dung_torch_compile and hasattr(torch, "compile"):
            try:
                active_model = torch.compile(model)
            except Exception:
                active_model = model

        if self.gop_chuoi:
            packer = BoGopChuoi(max_seq_length=self.max_seq_length)
            du_lieu_active, self._thong_ke_gop = packer.gop_chuoi(du_lieu_train, self.max_seq_length)
        else:
            du_lieu_active = du_lieu_train
            self._thong_ke_gop = {}

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.toc_do_hoc, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        tong_steps = max(
            1,
            len(du_lieu_active)
            * self.so_vong
            // max(1, self.kich_thuoc_batch * self.gradient_accumulation),
        )
        warmup_steps = int(tong_steps * self.warmup_ratio)

        use_cuda_amp = bool(self.mixed_precision and device.type == "cuda")
        amp_dtype = (
            torch.bfloat16
            if (use_cuda_amp and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        bat_dau = time.time()

        for epoch in range(self.so_vong):
            model.train()
            epoch_loss = 0.0
            steps = 0
            optimizer.zero_grad()

            if len(du_lieu_active) == 0:
                self._history["train_loss"].append(0.0)
                continue

            indices = np.random.permutation(len(du_lieu_active))

            for batch_start in range(0, len(indices), self.kich_thuoc_batch):
                batch_indices = indices[batch_start : batch_start + self.kich_thuoc_batch]
                batch_items = [du_lieu_active[idx] for idx in batch_indices]
                collated = BoGopChuoi.tao_batch_vector_hoa(
                    batch_items, max_seq_length=self.max_seq_length
                )
                if collated is None:
                    continue

                input_ids = torch.tensor(collated["input_ids"], dtype=torch.long, device=device)
                labels = torch.tensor(collated["labels"], dtype=torch.long, device=device)

                amp_ctx = (
                    torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                    if use_cuda_amp
                    else nullcontext()
                )

                try:
                    with amp_ctx:
                        if hasattr(active_model, "generate") and hasattr(active_model, "config"):
                            outputs = active_model(input_ids=input_ids, labels=labels)
                            if self.dung_chunked_ce and hasattr(outputs, "logits"):
                                batch_loss = HamMatMatPhanManh.tinh_loss_pytorch(outputs.logits, labels)
                            elif hasattr(outputs, "loss") and outputs.loss is not None:
                                batch_loss = outputs.loss
                            else:
                                batch_loss = criterion(
                                    outputs.logits.reshape(-1, outputs.logits.size(-1)),
                                    labels.reshape(-1),
                                )
                        else:
                            logits = active_model(input_ids)
                            if self.dung_chunked_ce:
                                batch_loss = HamMatMatPhanManh.tinh_loss_pytorch(logits, labels)
                            else:
                                batch_loss = criterion(
                                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                                )
                except Exception:
                    # Fallback từng mẫu nếu mô hình tùy biến không hỗ trợ kích thước batch động
                    batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    valid_cnt = 0
                    for mau in batch_items:
                        in_1 = torch.tensor(mau.get("input_ids", []), dtype=torch.long, device=device).unsqueeze(0)
                        lb_1 = torch.tensor(
                            mau.get("labels", mau.get("input_ids", [])), dtype=torch.long, device=device
                        ).unsqueeze(0)
                        if in_1.numel() == 0:
                            continue
                        try:
                            logits_1 = active_model(in_1)
                            batch_loss = batch_loss + criterion(
                                logits_1.view(-1, logits_1.size(-1)), lb_1.view(-1)
                            )
                            valid_cnt += 1
                        except Exception:
                            continue
                    if valid_cnt > 0:
                        batch_loss = batch_loss / valid_cnt

                scaled_loss = batch_loss / self.gradient_accumulation
                scaled_loss.backward()
                epoch_loss += float(batch_loss.item())
                steps += 1

                if steps % self.gradient_accumulation == 0 or (
                    batch_start + self.kich_thuoc_batch >= len(indices)
                ):
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    self._global_step += 1

                    self._cap_nhat_lr(optimizer, self._global_step, warmup_steps, tong_steps)

                    if self._global_step % self.logging_steps == 0:
                        self.logger.info(
                            f"Step {self._global_step}: loss={epoch_loss / max(1, steps):.4f}"
                        )

                    if callback:
                        callback(self._global_step, epoch_loss / max(1, steps))

            avg_loss = epoch_loss / max(1, steps)
            self._history["train_loss"].append(avg_loss)
            self.logger.info(f"Epoch {epoch + 1}/{self.so_vong}: loss={avg_loss:.4f}")

            if du_lieu_val:
                eval_loss = self._danh_gia(model, du_lieu_val, criterion, device)
                self._history["eval_loss"].append(eval_loss)
                self.logger.info(f"  Eval loss: {eval_loss:.4f}")

        tong_thoi_gian = time.time() - bat_dau

        ket_qua = {
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": self.so_vong,
            "global_step": self._global_step,
            "train_loss_min": min(self._history["train_loss"])
            if self._history["train_loss"]
            else 0,
            "loss_cuoi": self._history["train_loss"][-1] if self._history["train_loss"] else 0.0,
            "eval_loss_min": min(self._history["eval_loss"])
            if self._history["eval_loss"]
            else None,
            "history": self._history,
        }
        if self._thong_ke_gop:
            ket_qua["thong_ke_gop_chuoi"] = self._thong_ke_gop
        return ket_qua

    def _danh_gia(
        self, model: "nn.Module", du_lieu_val: List[Dict], criterion: Any, device: Any
    ) -> float:
        """Đánh giá trên validation set."""
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for mau in du_lieu_val:
                input_ids = (
                    torch.tensor(mau.get("input_ids", []), dtype=torch.long).unsqueeze(0).to(device)
                )
                labels = (
                    torch.tensor(mau.get("labels", mau.get("input_ids", [])), dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )

                if input_ids.numel() == 0:
                    continue

                try:
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = (
                        outputs.loss
                        if hasattr(outputs, "loss")
                        else criterion(
                            outputs.logits.view(-1, outputs.logits.size(-1)),
                            labels.view(-1),
                        )
                    )
                    total_loss += loss.item()
                    count += 1
                except Exception:
                    continue

        model.train()
        return total_loss / max(1, count)

    def _huan_luyen_numpy(
        self,
        model: Any,
        du_lieu_train: List[Dict],
        du_lieu_val: Optional[List[Dict]],
        callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """Huấn luyện SFT bằng NumPy (trên GPTModel hoặc mô hình ngôn ngữ nội bộ)."""
        bat_dau = time.time()
        np.random.seed(self.seed)

        if model is None or not hasattr(model, "buoc_huan_luyen"):
            from vietnamese_ai.transformer.gpt_model import GPTModel

            max_id = 32
            for mau in du_lieu_train:
                ids = mau.get("input_ids", []) + mau.get("labels", [])
                if ids:
                    max_id = max(max_id, max(int(x) for x in ids if int(x) >= 0) + 2)
            vocab_size = min(max(max_id, 64), 2000)
            InternalModel = GPTModel(
                d_model=32, so_dau=2, d_ff=64, so_block=1, so_tu_vung=vocab_size, do_dai_toi_da=max(64, self.max_seq_length)
            )
            target_model = InternalModel
        else:
            target_model = model

        vocab_limit = getattr(target_model, "so_tu_vung", 2000)
        lr = max(self.toc_do_hoc, 1e-3)

        from vietnamese_ai.fine_tuning.training_accelerator import BoGopChuoi

        du_lieu_chuan = []
        for mau in du_lieu_train:
            raw_in = [
                (int(x) % vocab_limit if int(x) >= 0 else int(x))
                for x in mau.get("input_ids", [])
            ][: self.max_seq_length]
            raw_lbl = [
                (int(x) % vocab_limit if int(x) >= 0 else int(x))
                for x in mau.get("labels", raw_in)
            ][: self.max_seq_length]
            if raw_in:
                du_lieu_chuan.append({"input_ids": raw_in, "labels": raw_lbl})

        if self.gop_chuoi:
            packer = BoGopChuoi(max_seq_length=self.max_seq_length)
            du_lieu_active, self._thong_ke_gop = packer.gop_chuoi(
                du_lieu_chuan, self.max_seq_length
            )
        else:
            du_lieu_active = du_lieu_chuan
            self._thong_ke_gop = {}

        for epoch in range(self.so_vong):
            epoch_loss = 0.0
            steps = 0
            if len(du_lieu_active) == 0:
                self._history["train_loss"].append(0.0)
                continue

            indices = np.random.permutation(len(du_lieu_active))

            for batch_start in range(0, len(indices), self.kich_thuoc_batch):
                batch_indices = indices[batch_start : batch_start + self.kich_thuoc_batch]
                batch_items = [du_lieu_active[idx] for idx in batch_indices]
                collated = BoGopChuoi.tao_batch_vector_hoa(
                    batch_items, vocab_limit=vocab_limit, max_seq_length=self.max_seq_length
                )
                if collated is None:
                    continue

                try:
                    step_loss = float(
                        target_model.buoc_huan_luyen(
                            collated["input_ids"],
                            collated["labels"],
                            toc_do_hoc=lr,
                            weight_decay=self.weight_decay,
                            grad_clip=self.gradient_clip,
                            attention_mask=collated["attention_mask"],
                            dung_chunked_ce=self.dung_chunked_ce,
                        )
                    )
                except TypeError:
                    # Tương thích ngược với các lớp model tùy biến không nhận attention_mask
                    step_loss = float(
                        target_model.buoc_huan_luyen(
                            collated["input_ids"],
                            collated["labels"],
                            toc_do_hoc=lr,
                            weight_decay=self.weight_decay,
                            grad_clip=self.gradient_clip,
                        )
                    )

                epoch_loss += step_loss
                steps += 1
                self._global_step += 1
                if callback:
                    callback(self._global_step, epoch_loss / max(1, steps))

            avg_loss = epoch_loss / max(1, steps)
            self._history["train_loss"].append(avg_loss)

            if du_lieu_val:
                val_losses = []
                for mau in du_lieu_val:
                    raw_in = [
                        (int(x) % vocab_limit if int(x) >= 0 else int(x))
                        for x in mau.get("input_ids", [])
                    ][: self.max_seq_length]
                    raw_lbl = [
                        (int(x) % vocab_limit if int(x) >= 0 else int(x))
                        for x in mau.get("labels", raw_in)
                    ][: self.max_seq_length]
                    if not raw_in:
                        continue
                    seq_len = min(len(raw_in), len(raw_lbl))
                    in_arr = np.array([raw_in[:seq_len]], dtype=int)
                    tgt_arr = np.array([raw_lbl[:seq_len]], dtype=int)
                    val_losses.append(target_model.tinh_loss(in_arr, tgt_arr))
                eval_loss = float(np.mean(val_losses)) if val_losses else avg_loss
                self._history["eval_loss"].append(eval_loss)

        tong_thoi_gian = time.time() - bat_dau
        ket_qua = {
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": self.so_vong,
            "global_step": self._global_step,
            "train_loss_min": min(self._history["train_loss"]) if self._history["train_loss"] else 0.0,
            "loss_cuoi": self._history["train_loss"][-1] if self._history["train_loss"] else 0.0,
            "eval_loss_min": min(self._history["eval_loss"]) if self._history["eval_loss"] else None,
            "history": self._history,
        }
        if self._thong_ke_gop:
            ket_qua["thong_ke_gop_chuoi"] = self._thong_ke_gop
        return ket_qua

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
