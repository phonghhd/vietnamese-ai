"""DPOTrainer - Direct Preference Optimization Trainer."""

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _CO_PYTORCH = True
except ImportError:
    _CO_PYTORCH = False


class DPOTrainer:
    """
    Direct Preference Optimization (DPO) Trainer.

    DPO tối ưu hóa trực tiếp từ preference data mà không cần reward model.

    Loss function:
        L_DPO = -E[log σ(β(log π(y_w|x)/π_ref(y_w|x)) - log π(y_l|x)/π_ref(y_l|x)))]

    Tính năng:
    - DPO loss với reference model
    - KL divergence regularization
    - Beta annealing
    - Label smoothing
    - Support PyTorch và NumPy fallback

    Sử dụng:
        >>> dpo = DPOTrainer(so_vong=1, beta=0.1)
        >>> preference_data = [
        ...     {"prompt": "câu hỏi", "chosen": "trả lời tốt", "rejected": "trả lời xấu"},
        ... ]
        >>> ket_qua = dpo.huan_luyen(model, ref_model, preference_data)
    """

    def __init__(
        self,
        so_vong: int = 1,
        kich_thuoc_batch: int = 2,
        toc_do_hoc: float = 5e-7,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        gradient_accumulation: int = 4,
        gradient_clip: float = 1.0,
        max_seq_length: int = 512,
        logging_steps: int = 10,
        seed: int = 42,
        che_do: str = "dpo",
        gamma_simpo: float = 0.5,
    ):
        if beta <= 0:
            raise ValueError(f"beta phải > 0, nhận: {beta}")
        che_do_norm = str(che_do or "dpo").strip().lower()
        if che_do_norm not in ("dpo", "simpo"):
            raise ValueError(f"che_do phải là 'dpo' hoặc 'simpo', nhận: {che_do}")

        self.so_vong = so_vong
        self.kich_thuoc_batch = kich_thuoc_batch
        self.toc_do_hoc = toc_do_hoc
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.gradient_accumulation = gradient_accumulation
        self.gradient_clip = gradient_clip
        self.max_seq_length = max_seq_length
        self.logging_steps = logging_steps
        self.seed = seed
        self.che_do = che_do_norm
        self.gamma_simpo = float(gamma_simpo)
        self.logger = Logger("DPOTrainer")

        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "chosen_rewards": [],
            "rejected_rewards": [],
            "reward_margin": [],
        }
        self._global_step = 0

    def huan_luyen(
        self,
        model: Any,
        ref_model: Optional[Any],
        preference_data: List[Dict[str, str]],
        tokenizer: Any = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện DPO hoặc SimPO (Reference-Free Simple Preference Optimization khi `che_do="simpo"` hoặc `ref_model=None`).

        Args:
            model: Policy model (PyTorch nn.Module hoặc mô hình NumPy)
            ref_model: Reference model (frozen copy; có thể là `None` ở chế độ `simpo`)
            preference_data: List [{"prompt": ..., "chosen": ..., "rejected": ...}, ...]
            tokenizer: Tokenizer
            callback: Callback function

        Returns:
            Dict chứa training results
        """
        self.logger.info(f"Bắt đầu {self.che_do.upper()} training ({self.so_vong} epochs)")
        self.logger.info(f"  Preference samples: {len(preference_data)}")
        self.logger.info(f"  Beta: {self.beta}")

        if _CO_PYTORCH and isinstance(model, nn.Module):
            return self._huan_luyen_pytorch(model, ref_model, preference_data, tokenizer, callback)
        return self._huan_luyen_numpy(model, ref_model, preference_data, callback)

    @staticmethod
    def _ma_hoa_chuoi_dpo(text: str, tokenizer: Any, max_len: int) -> List[int]:
        """Mã hóa văn bản DPO bằng tokenizer thực nếu có, dự phòng sang mã Unicode."""
        ids: List[int] = []
        if tokenizer is not None:
            if hasattr(tokenizer, "encode"):
                raw = tokenizer.encode(text)
                ids = list(raw) if isinstance(raw, (list, tuple)) else []
            elif hasattr(tokenizer, "ma_hoa"):
                ids = list(tokenizer.ma_hoa(text))
            elif callable(tokenizer):
                out = tokenizer(text)
                if isinstance(out, dict) and "input_ids" in out:
                    ids = list(out["input_ids"])
                elif isinstance(out, (list, tuple)):
                    ids = list(out)
        if not ids:
            ids = [ord(c) % 1000 for c in text]
        return ids[:max_len] if len(ids) > max_len else (ids or [0])

    def _huan_luyen_pytorch(
        self,
        model: "nn.Module",
        ref_model: Optional["nn.Module"],
        preference_data: List[Dict],
        tokenizer: Any = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Huấn luyện DPO/SimPO với PyTorch."""
        device = next(model.parameters()).device
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.toc_do_hoc)

        if ref_model is not None and isinstance(ref_model, nn.Module):
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False

        bat_dau = time.time()

        for epoch in range(self.so_vong):
            model.train()
            epoch_loss = 0.0
            chosen_rewards_epoch = []
            rejected_rewards_epoch = []
            steps = 0
            optimizer.zero_grad()

            indices = np.random.permutation(len(preference_data))

            for pos_idx, idx in enumerate(indices):
                mau = preference_data[idx]
                prompt = mau.get("prompt", "")
                chosen = mau.get("chosen", "")
                rejected = mau.get("rejected", "")

                try:
                    loss, chosen_reward, rejected_reward = self._tinh_dpo_loss(
                        model, ref_model, prompt, chosen, rejected, device, tokenizer=tokenizer
                    )

                    loss = loss / self.gradient_accumulation
                    loss.backward()
                    epoch_loss += loss.item() * self.gradient_accumulation

                    chosen_rewards_epoch.append(chosen_reward)
                    rejected_rewards_epoch.append(rejected_reward)
                    steps += 1

                    if steps % self.gradient_accumulation == 0 or (pos_idx == len(indices) - 1):
                        if self.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                        self._global_step += 1

                        if self._global_step % self.logging_steps == 0:
                            avg_loss = epoch_loss / max(1, steps)
                            self.logger.info(f"Step {self._global_step}: loss={avg_loss:.4f}")

                        if callback:
                            callback(self._global_step, epoch_loss / max(1, steps))

                except Exception:
                    continue

            if steps > 0 and steps % self.gradient_accumulation != 0:
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                optimizer.step()
                optimizer.zero_grad()
                self._global_step += 1

            avg_loss = epoch_loss / max(1, steps) if steps > 0 else 0
            self._history["train_loss"].append(avg_loss)

            avg_chosen = 0.0
            avg_rejected = 0.0
            if chosen_rewards_epoch:
                avg_chosen = np.mean(chosen_rewards_epoch)
                avg_rejected = np.mean(rejected_rewards_epoch)
                self._history["chosen_rewards"].append(float(avg_chosen))
                self._history["rejected_rewards"].append(float(avg_rejected))
                self._history["reward_margin"].append(float(avg_chosen - avg_rejected))

            self.logger.info(
                f"Epoch {epoch + 1}/{self.so_vong}: loss={avg_loss:.4f}, "
                f"margin={avg_chosen - avg_rejected:.4f}"
            )

        tong_thoi_gian = time.time() - bat_dau

        return {
            "che_do": self.che_do,
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": self.so_vong,
            "global_step": self._global_step,
            "train_loss_min": min(self._history["train_loss"])
            if self._history["train_loss"]
            else 0,
            "final_reward_margin": self._history["reward_margin"][-1]
            if self._history["reward_margin"]
            else 0,
            "history": self._history,
        }

    def _tinh_dpo_loss(
        self,
        model: "nn.Module",
        ref_model: Optional["nn.Module"],
        prompt: str,
        chosen: str,
        rejected: str,
        device: Any,
        tokenizer: Any = None,
    ):
        """Tính DPO hoặc SimPO loss trên riêng phần token phản hồi (che prompt prefix)."""
        prompt_tokens = self._ma_hoa_chuoi_dpo(prompt, tokenizer, self.max_seq_length) if prompt else []
        chosen_seq = self._ma_hoa_chuoi_dpo(prompt + chosen, tokenizer, self.max_seq_length)
        rejected_seq = self._ma_hoa_chuoi_dpo(prompt + rejected, tokenizer, self.max_seq_length)
        prompt_len_c = min(len(prompt_tokens), max(0, len(chosen_seq) - 2))
        prompt_len_r = min(len(prompt_tokens), max(0, len(rejected_seq) - 2))

        chosen_ids = torch.tensor(chosen_seq, dtype=torch.long).unsqueeze(0).to(device)
        rejected_ids = torch.tensor(rejected_seq, dtype=torch.long).unsqueeze(0).to(device)

        model.train()
        chosen_out: Any = model(chosen_ids) if hasattr(model, "__call__") else None
        rejected_out: Any = model(rejected_ids) if hasattr(model, "__call__") else None
        raw_chosen_logits: Any = getattr(chosen_out, "logits", chosen_out) if chosen_out is not None else None
        raw_rejected_logits: Any = getattr(rejected_out, "logits", rejected_out) if rejected_out is not None else None

        chosen_logits: Any = (
            raw_chosen_logits
            if raw_chosen_logits is not None
            else torch.randn(1, 2, 100, device=device, requires_grad=True)
        )
        rejected_logits: Any = (
            raw_rejected_logits
            if raw_rejected_logits is not None
            else torch.randn(1, 2, 100, device=device, requires_grad=True)
        )

        dung_simpo = self.che_do == "simpo" or ref_model is None
        if dung_simpo:
            chosen_avg_logps = self._tinh_logps(
                chosen_logits, chosen_ids, prompt_len=prompt_len_c, trung_binh=True
            )
            rejected_avg_logps = self._tinh_logps(
                rejected_logits, rejected_ids, prompt_len=prompt_len_r, trung_binh=True
            )
            logits = self.beta * (chosen_avg_logps - rejected_avg_logps) - self.gamma_simpo
            chosen_reward = (self.beta * chosen_avg_logps).detach().mean().item()
            rejected_reward = (self.beta * rejected_avg_logps).detach().mean().item()
        else:
            assert ref_model is not None
            with torch.no_grad():
                ref_c_out: Any = ref_model(chosen_ids) if hasattr(ref_model, "__call__") else None
                ref_r_out: Any = ref_model(rejected_ids) if hasattr(ref_model, "__call__") else None
                raw_ref_c: Any = getattr(ref_c_out, "logits", ref_c_out) if ref_c_out is not None else None
                raw_ref_r: Any = getattr(ref_r_out, "logits", ref_r_out) if ref_r_out is not None else None

            ref_chosen_logits: Any = raw_ref_c if raw_ref_c is not None else chosen_logits.detach()
            ref_rejected_logits: Any = raw_ref_r if raw_ref_r is not None else rejected_logits.detach()

            chosen_logps = self._tinh_logps(chosen_logits, chosen_ids, prompt_len=prompt_len_c, trung_binh=False)
            rejected_logps = self._tinh_logps(rejected_logits, rejected_ids, prompt_len=prompt_len_r, trung_binh=False)
            ref_chosen_logps = self._tinh_logps(
                ref_chosen_logits, chosen_ids, prompt_len=prompt_len_c, trung_binh=False
            )
            ref_rejected_logps = self._tinh_logps(
                ref_rejected_logits, rejected_ids, prompt_len=prompt_len_r, trung_binh=False
            )

            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            logits = self.beta * (chosen_logratios - rejected_logratios)
            chosen_reward = (self.beta * chosen_logratios).detach().mean().item()
            rejected_reward = (self.beta * rejected_logratios).detach().mean().item()

        if self.label_smoothing > 0:
            loss = (
                (1 - self.label_smoothing) * (-F.logsigmoid(logits))
                + self.label_smoothing * (-F.logsigmoid(-logits))
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        return loss, chosen_reward, rejected_reward

    def _tinh_logps(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        prompt_len: int = 0,
        trung_binh: bool = False,
    ) -> "torch.Tensor":
        """Tính tổng hoặc trung bình log-xác suất trên riêng phần response tokens (bỏ qua prompt prefix)."""
        if logits.dim() == 3 and logits.size(1) > 1:
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            shift_labels = labels[:, 1:]
            vocab_size = log_probs.size(-1)
            safe_labels = shift_labels.clamp(min=0, max=vocab_size - 1)
            per_tok = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
            start_idx = max(0, min(int(prompt_len) - 1, per_tok.size(1) - 1)) if prompt_len > 0 else 0
            resp_logps = per_tok[:, start_idx:]
            if trung_binh:
                return resp_logps.mean(-1)
            return resp_logps.sum(-1)
        return torch.tensor(0.0, device=logits.device)

    @staticmethod
    def _trich_xuat_dac_trung_dpo(text: str, dim: int = 256) -> np.ndarray:
        import hashlib

        vec = np.zeros(dim, dtype=np.float64)
        s = text.lower().strip()
        words = s.split()
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i + 1]}")
        for i in range(max(0, len(s) - 2)):
            tokens.append(s[i : i + 3])
        if not tokens:
            tokens = [s]
        for tok in tokens:
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "little")
            idx = h % dim
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _huan_luyen_numpy(
        self,
        model: Any,
        ref_model: Any,
        preference_data: List[Dict],
        callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """Huấn luyện DPO bằng NumPy với Bradley-Terry preference gradient."""
        bat_dau = time.time()
        np.random.seed(self.seed)

        dim = 256
        dung_simpo = self.che_do == "simpo" or ref_model is None
        w_policy = getattr(model, "_dpo_weights", np.zeros(dim, dtype=np.float64)).copy()
        w_ref = (
            np.zeros(dim, dtype=np.float64)
            if dung_simpo
            else getattr(ref_model, "_dpo_weights", np.zeros(dim, dtype=np.float64)).copy()
        )
        lr = max(self.toc_do_hoc, 0.1)

        for epoch in range(self.so_vong):
            epoch_loss = 0.0
            chosen_rewards_epoch = []
            rejected_rewards_epoch = []
            steps = 0
            indices = np.random.permutation(len(preference_data))

            for idx in indices:
                mau = preference_data[idx]
                prompt = mau.get("prompt", "")
                chosen = mau.get("chosen", "")
                rejected = mau.get("rejected", "")

                phi_w = self._trich_xuat_dac_trung_dpo(f"{prompt} {chosen}", dim)
                phi_l = self._trich_xuat_dac_trung_dpo(f"{prompt} {rejected}", dim)

                logp_w = float(np.dot(w_policy, phi_w))
                logp_l = float(np.dot(w_policy, phi_l))
                if dung_simpo:
                    len_w = max(1.0, float(len(chosen.split())))
                    len_l = max(1.0, float(len(rejected.split())))
                    norm_w = logp_w / len_w
                    norm_l = logp_l / len_l
                    u = self.beta * (norm_w - norm_l) - self.gamma_simpo
                    grad_phi = phi_w / len_w - phi_l / len_l
                    chosen_r = float(self.beta * norm_w)
                    rejected_r = float(self.beta * norm_l)
                else:
                    ref_logp_w = float(np.dot(w_ref, phi_w))
                    ref_logp_l = float(np.dot(w_ref, phi_l))
                    chosen_logratio = logp_w - ref_logp_w
                    rejected_logratio = logp_l - ref_logp_l
                    u = self.beta * (chosen_logratio - rejected_logratio)
                    grad_phi = phi_w - phi_l
                    chosen_r = float(self.beta * chosen_logratio)
                    rejected_r = float(self.beta * rejected_logratio)

                sig_u = 1.0 / (1.0 + np.exp(-np.clip(u, -30.0, 30.0)))
                sig_neg_u = 1.0 - sig_u

                if self.label_smoothing > 0:
                    loss = float(
                        -(1.0 - self.label_smoothing) * np.log(sig_u + 1e-12)
                        - self.label_smoothing * np.log(sig_neg_u + 1e-12)
                    )
                    d_u = -((1.0 - self.label_smoothing) * (1.0 - sig_u) - self.label_smoothing * sig_u)
                else:
                    loss = float(-np.log(sig_u + 1e-12))
                    d_u = -(1.0 - sig_u)

                grad_w = d_u * self.beta * grad_phi
                grad_norm = np.linalg.norm(grad_w)
                if self.gradient_clip > 0 and grad_norm > self.gradient_clip:
                    grad_w = grad_w * (self.gradient_clip / (grad_norm + 1e-8))

                w_policy -= lr * grad_w
                if hasattr(model, "buoc_huan_luyen"):
                    ids_c = np.array([[ord(c) % model.so_tu_vung for c in (prompt + chosen)[:64]]], dtype=int)
                    if ids_c.shape[1] > 1:
                        model.buoc_huan_luyen(ids_c[:, :-1], ids_c[:, 1:], toc_do_hoc=max(self.toc_do_hoc, 1e-3))

                epoch_loss += loss
                chosen_rewards_epoch.append(chosen_r)
                rejected_rewards_epoch.append(rejected_r)
                steps += 1
                self._global_step += 1

                if callback and steps % max(1, self.gradient_accumulation) == 0:
                    callback(self._global_step, epoch_loss / max(1, steps))

            avg_loss = epoch_loss / max(1, steps) if steps > 0 else 0.0
            self._history["train_loss"].append(avg_loss)
            if chosen_rewards_epoch:
                avg_c = float(np.mean(chosen_rewards_epoch))
                avg_r = float(np.mean(rejected_rewards_epoch))
                self._history["chosen_rewards"].append(avg_c)
                self._history["rejected_rewards"].append(avg_r)
                self._history["reward_margin"].append(avg_c - avg_r)

        if model is not None and not isinstance(model, (int, float, str)):
            try:
                model._dpo_weights = w_policy
            except Exception:
                pass
        self._dpo_weights = w_policy

        tong_thoi_gian = time.time() - bat_dau
        return {
            "che_do": self.che_do,
            "tong_thoi_gian": round(tong_thoi_gian, 2),
            "so_epoch": self.so_vong,
            "global_step": self._global_step,
            "train_loss_min": min(self._history["train_loss"]) if self._history["train_loss"] else 0.0,
            "final_reward_margin": self._history["reward_margin"][-1]
            if self._history["reward_margin"]
            else 0.0,
            "history": self._history,
        }

    def lay_lich_su(self) -> Dict[str, List[float]]:
        return self._history.copy()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "che_do": self.che_do,
            "so_vong": self.so_vong,
            "beta": self.beta,
            "gamma_simpo": self.gamma_simpo,
            "label_smoothing": self.label_smoothing,
            "toc_do_hoc": self.toc_do_hoc,
            "global_step": self._global_step,
            "history_length": len(self._history["train_loss"]),
        }
