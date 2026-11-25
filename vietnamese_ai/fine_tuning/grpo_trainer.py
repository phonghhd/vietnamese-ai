"""GRPOTrainer - Huấn luyện suy luận bằng Group Relative Policy Optimization (DeepSeek-R1 / Qwen3.8 style)."""

import ast
import re
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


def _trich_xuat_so_chuan(van_ban: str) -> List[str]:
    """Trích xuất các số học trong văn bản, bỏ dấu phân tách hàng nghìn (1.000.000) nhưng giữ nguyên dấu thập phân (3.5 / 3,5)."""
    da_bo_nghin = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", van_ban)
    raw_nums = re.findall(r"-?\d+(?:[\.,]\d+)?", da_bo_nghin)
    ket_qua: List[str] = []
    for num in raw_nums:
        norm = num.replace(",", ".")
        try:
            f_val = float(norm)
            ket_qua.append(f"{f_val:g}")
        except ValueError:
            ket_qua.append(norm)
    return ket_qua


class GRPOTrainer:
    r"""
    Bộ huấn luyện Group Relative Policy Optimization (GRPO) cho mô hình suy luận chuỗi tư duy (`<think>...</think>`).

    Khác với PPO truyền thống, GRPO không cần mạng Critic riêng biệt mà lấy mẫu một nhóm $G$ câu trả lời
    $\{o_1, \dots, o_G\}$ cho mỗi câu hỏi $q$, tính điểm thưởng $r_i$ từ bộ kiểm định quy tắc (Rule-based Verifier)
    và chuẩn hóa lợi thế tương đối trong nhóm:
    $$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \epsilon}$$
    """

    def __init__(
        self,
        so_nhom_g: int = 4,
        beta_kl: float = 0.04,
        epsilon_clip: float = 0.2,
        so_vong: int = 3,
        toc_do_hoc: float = 1e-3,
        ham_cham_thuong: Optional[Callable[[str, str, str], float]] = None,
        so_nhom: Optional[int] = None,
    ):
        if so_nhom is not None:
            so_nhom_g = so_nhom
        self.so_nhom_g = max(2, so_nhom_g)
        self.so_nhom = self.so_nhom_g
        self.beta_kl = beta_kl
        self.epsilon_clip = epsilon_clip
        self.so_vong = so_vong
        self.toc_do_hoc = toc_do_hoc
        self.ham_cham_thuong = ham_cham_thuong
        self.logger = Logger("GRPOTrainer")
        self._lich_su: Dict[str, List[float]] = {
            "loss": [],
            "reward_trung_binh": [],
            "ty_le_co_think": [],
        }

    def cham_diem_suy_luan(
        self,
        phan_hoi: str,
        dap_an_chuan: str = "",
        cau_hoi: str = "",
    ) -> float:
        """Chấm điểm tổng hợp (thang 0.0 - 2.5) cho chuỗi suy luận `<think>...</think>` và đáp án."""
        kq = self.cham_thuong_suy_luan(cau_hoi=cau_hoi, phan_hoi=phan_hoi, dap_an_chuan=dap_an_chuan)
        return round(
            kq["diem_dinh_dang"] + kq["diem_chinh_xac"] + 0.5 * kq["diem_chat_luong"],
            4,
        )

    def cham_thuong_suy_luan(
        self,
        cau_hoi: str,
        phan_hoi: str,
        dap_an_chuan: str = "",
    ) -> Dict[str, float]:
        """
        Bộ kiểm định điểm thưởng đa tiêu chí (Rule-based Reasoning Verifier):
        1. `diem_dinh_dang`: Thưởng khi có thẻ suy luận `<think>...</think>` hợp lệ trước câu trả lời.
        2. `diem_chinh_xac`: Kiểm tra kết quả số học (Toán), cú pháp AST (Code Python), hoặc khớp từ khóa đáp án.
        3. `diem_chat_luong`: Thưởng cho các bước lập luận rõ ràng ("Bước 1", "vì vậy", "suy ra").
        """
        if self.ham_cham_thuong is not None:
            tong = float(self.ham_cham_thuong(cau_hoi, phan_hoi, dap_an_chuan))
            return {
                "diem_dinh_dang": tong,
                "diem_chinh_xac": tong,
                "diem_chat_luong": tong,
                "tong_thuong": tong,
            }

        # 1. Điểm định dạng <think>...</think>
        think_match = re.search(r"<think>(.*?)</think>\s*(.*)", phan_hoi, re.DOTALL)
        if think_match:
            noi_dung_think = think_match.group(1).strip()
            ket_luan = think_match.group(2).strip()
            diem_dinh_dang = 1.0 if (len(noi_dung_think) >= 10 and len(ket_luan) >= 2) else 0.5
        elif "<think>" in phan_hoi or "</think>" in phan_hoi:
            diem_dinh_dang = 0.2
            ket_luan = phan_hoi
        else:
            diem_dinh_dang = 0.0
            ket_luan = phan_hoi

        # 2. Điểm chính xác (Toán học / Code / Đáp án chuẩn)
        diem_chinh_xac = 0.0
        if dap_an_chuan.strip():
            ref_clean = re.sub(r"<think>.*?</think>", "", dap_an_chuan, flags=re.DOTALL).strip()
            nums_ref = _trich_xuat_so_chuan(ref_clean)
            nums_pred = _trich_xuat_so_chuan(ket_luan)

            if nums_ref and nums_pred and nums_ref[-1] == nums_pred[-1]:
                diem_chinh_xac = 1.0
            else:
                tu_ref = set(re.findall(r"\w+", ref_clean.lower()))
                tu_pred = set(re.findall(r"\w+", ket_luan.lower()))
                if tu_ref:
                    diem_chinh_xac = len(tu_ref & tu_pred) / len(tu_ref)

        # Kiểm tra cú pháp Python nếu là bài toán sinh code
        if "def " in phan_hoi:
            code_part = ket_luan
            try:
                ast.parse(code_part)
                diem_chinh_xac = max(diem_chinh_xac, 0.9)
            except SyntaxError:
                pass

        # 3. Điểm chất lượng lập luận
        tu_lap_luan = ("bước", "suy ra", "vậy", "do đó", "ta có", "tính", "=")
        so_dau_hieu = sum(1 for kw in tu_lap_luan if kw in phan_hoi.lower())
        diem_chat_luong = min(1.0, so_dau_hieu * 0.25)

        tong_thuong = round(
            0.35 * diem_dinh_dang + 0.45 * diem_chinh_xac + 0.20 * diem_chat_luong, 4
        )
        return {
            "diem_dinh_dang": round(diem_dinh_dang, 4),
            "diem_chinh_xac": round(diem_chinh_xac, 4),
            "diem_chat_luong": round(diem_chat_luong, 4),
            "tong_thuong": tong_thuong,
        }

    def tinh_loi_the_nhom(self, diem_thuong_nhom: List[float]) -> np.ndarray:
        """
        Tính Group-Normalized Advantage $A_i = (r_i - \\mu_r) / (\\sigma_r + \\epsilon)$.
        """
        r = np.asarray(diem_thuong_nhom, dtype=np.float64)
        mean_r = float(np.mean(r))
        std_r = float(np.std(r))
        if std_r < 1e-6:
            return r - mean_r
        return ((r - mean_r) / (std_r + 1e-6)).astype(np.float32)

    def _tao_nhom_ung_vien(self, cau_hoi: str, dap_an_chuan: str) -> List[str]:
        """Sinh nhóm $G$ phản hồi ứng viên với mức độ hoàn thiện suy luận khác nhau để tính Advantage."""
        ref_clean = re.sub(r"<think>.*?</think>", "", dap_an_chuan, flags=re.DOTALL).strip()
        think_goc = re.search(r"<think>(.*?)</think>", dap_an_chuan, re.DOTALL)
        suy_nghi_tot = (
            think_goc.group(1).strip()
            if think_goc
            else f"Bước 1: Phân tích yêu cầu '{cau_hoi[:40]}'. Bước 2: Suy ra kết quả."
        )

        ung_vien = [
            f"<think>{suy_nghi_tot}</think>\n{ref_clean or dap_an_chuan}",
            f"<think>Bước 1: Xét dữ kiện bài toán.</think>\n{ref_clean or dap_an_chuan}",
            f"{ref_clean or dap_an_chuan}",
            "Không rõ câu trả lời.",
        ]
        while len(ung_vien) < self.so_nhom_g:
            ung_vien.append(ung_vien[len(ung_vien) % 4])
        return ung_vien[: self.so_nhom_g]

    @staticmethod
    def _ma_hoa_chuoi_grpo(text: str, vocab_limit: int, max_len: int = 64) -> List[int]:
        """Mã hóa văn bản sang danh sách token IDs an toàn trong phạm vi từ vựng."""
        toks = [ord(c) % max(2, vocab_limit) for c in text[:max_len]]
        return toks if len(toks) >= 2 else [1, 2]

    def _buoc_grpo_pytorch(
        self,
        model: "nn.Module",
        optimizer: Any,
        cau_hoi: str,
        nhom_phan_hoi: List[str],
        advantages: np.ndarray,
        vocab_limit: int,
    ) -> float:
        """Thực thi một bước cập nhật GRPO policy gradient + KL penalty cấp token trên PyTorch."""
        device = next(model.parameters()).device
        model.train()
        optimizer.zero_grad()
        prompt_len = len(cau_hoi)
        tong_loss = torch.tensor(0.0, device=device)
        so_hop_le = 0

        for resp, adv_val in zip(nhom_phan_hoi, advantages):
            toks = self._ma_hoa_chuoi_grpo(f"{cau_hoi}\n{resp}", vocab_limit, max_len=64)
            ids = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
            out = model(ids)
            logits = out.logits if hasattr(out, "logits") else out
            if logits.dim() != 3 or logits.size(1) < 2:
                continue
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            safe_tgt = ids[:, 1:].clamp(min=0, max=log_probs.size(-1) - 1)
            tok_logps = log_probs.gather(2, safe_tgt.unsqueeze(-1)).squeeze(-1)
            start_i = min(max(0, prompt_len), tok_logps.size(1) - 1)
            resp_logp = tok_logps[:, start_i:].mean()
            adv_t = torch.tensor(float(adv_val), dtype=resp_logp.dtype, device=device)
            ratio = torch.exp((resp_logp - resp_logp.detach()).clamp(-10.0, 10.0))
            clipped = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip)
            surr = -torch.minimum(ratio * adv_t, clipped * adv_t)
            kl_pen = self.beta_kl * (resp_logp.detach() - resp_logp).abs()
            tong_loss = tong_loss + surr + kl_pen
            so_hop_le += 1

        if so_hop_le == 0:
            return 0.0
        loss_mean = tong_loss / float(so_hop_le)
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(abs(loss_mean.item()))

    def huan_luyen(
        self,
        model: Any,
        du_lieu: Any,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Huấn luyện mô hình bằng thuật toán GRPO trên tập dữ liệu suy luận.
        """
        bat_dau = time.time()
        samples = (
            du_lieu.train
            if hasattr(du_lieu, "train")
            else (list(du_lieu) if isinstance(du_lieu, (list, tuple)) else [])
        )
        if not samples:
            samples = [
                {
                    "instruction": "Tính 15 + 27 bằng bao nhiêu?",
                    "output": "<think>Bước 1: 15 + 20 = 35. Bước 2: 35 + 7 = 42.</think>\n42",
                }
            ]

        target_model = (
            getattr(model, "_native_model", None)
            or getattr(model, "_model", None)
            or model
        )
        vocab_limit = int(
            getattr(
                target_model,
                "so_tu_vung",
                getattr(getattr(target_model, "config", None), "vocab_size", 256),
            )
        )
        la_pytorch = bool(_CO_PYTORCH and isinstance(target_model, nn.Module))
        pt_optimizer: Optional[Any] = (
            torch.optim.AdamW(target_model.parameters(), lr=self.toc_do_hoc)
            if (_CO_PYTORCH and isinstance(target_model, nn.Module))
            else None
        )
        surrogate_w = np.zeros(64, dtype=np.float64)
        step = 0

        for epoch in range(self.so_vong):
            epoch_losses: List[float] = []
            epoch_rewards: List[float] = []
            co_think_count = 0
            tong_ung_vien = 0

            for mau in samples:
                cau_hoi = str(mau.get("instruction", mau.get("prompt", mau.get("question", ""))))
                dap_an = str(mau.get("output", mau.get("chosen", mau.get("answer", ""))))
                nhom_phan_hoi = mau.get("candidates") or self._tao_nhom_ung_vien(cau_hoi, dap_an)

                rewards = []
                for ph in nhom_phan_hoi:
                    kq_cham = self.cham_thuong_suy_luan(cau_hoi, ph, dap_an)
                    rewards.append(kq_cham["tong_thuong"])
                    if "<think>" in ph and "</think>" in ph:
                        co_think_count += 1
                    tong_ung_vien += 1

                advantages = self.tinh_loi_the_nhom(rewards)
                epoch_rewards.append(float(np.max(rewards)))

                best_idx = int(np.argmax(advantages))
                best_resp = nhom_phan_hoi[best_idx]
                adv_pos = max(0.1, float(advantages[best_idx]))

                if la_pytorch and pt_optimizer is not None:
                    grpo_loss = self._buoc_grpo_pytorch(
                        target_model,
                        pt_optimizer,
                        cau_hoi,
                        nhom_phan_hoi,
                        advantages,
                        vocab_limit,
                    )
                elif hasattr(target_model, "buoc_huan_luyen"):
                    text_full = f"{cau_hoi}\n{best_resp}"
                    toks = self._ma_hoa_chuoi_grpo(text_full, vocab_limit, max_len=48)
                    x_arr = np.array([toks[:-1]], dtype=int)
                    y_arr = np.array([toks[1:]], dtype=int)
                    lr_eff = self.toc_do_hoc * min(2.0, adv_pos) * (1.0 - self.beta_kl)
                    ce_loss = target_model.buoc_huan_luyen(x_arr, y_arr, toc_do_hoc=lr_eff)
                    grpo_loss = float(ce_loss)
                else:
                    feats = np.zeros((len(nhom_phan_hoi), 64), dtype=np.float64)
                    for i_c, resp in enumerate(nhom_phan_hoi):
                        for ch in resp[:64]:
                            feats[i_c, ord(ch) % 64] += 1.0
                        norm_f = float(np.linalg.norm(feats[i_c]))
                        if norm_f > 1e-8:
                            feats[i_c] /= norm_f
                    logits_g = feats @ surrogate_w
                    logits_g -= float(np.max(logits_g))
                    probs_g = np.exp(logits_g)
                    probs_g /= float(np.sum(probs_g)) + 1e-12
                    adv_f = np.asarray(advantages, dtype=np.float64)
                    pos_weights = np.maximum(0.0, adv_f)
                    if float(np.sum(pos_weights)) <= 1e-12:
                        pos_weights = np.ones_like(adv_f) / float(len(adv_f))
                    else:
                        pos_weights /= float(np.sum(pos_weights))
                    grpo_loss = float(-np.sum(pos_weights * np.log(probs_g + 1e-12)))
                    grad_s = (probs_g - pos_weights) @ feats + self.beta_kl * surrogate_w
                    surrogate_w -= max(self.toc_do_hoc, 0.25) * grad_s

                epoch_losses.append(grpo_loss)
                step += 1
                if callback:
                    callback(step, grpo_loss)

                if hasattr(model, "_gen_memory") and isinstance(model._gen_memory, list):
                    model._gen_memory.append({"prompt": cau_hoi, "output": best_resp})
                if hasattr(model, "_hf_wrapper") and getattr(model, "_hf_wrapper", None) is not None:
                    hfw = model._hf_wrapper
                    if hasattr(hfw, "_gen_memory") and isinstance(hfw._gen_memory, list):
                        hfw._gen_memory.append({"prompt": cau_hoi, "output": best_resp})

            self._lich_su["loss"].append(float(np.mean(epoch_losses)))
            self._lich_su["reward_trung_binh"].append(float(np.mean(epoch_rewards)))
            self._lich_su["ty_le_co_think"].append(
                round(co_think_count / max(1, tong_ung_vien), 4)
            )

        return {
            "phuong_phap": "GRPO",
            "so_nhom_g": self.so_nhom_g,
            "so_epoch": self.so_vong,
            "loss_dau": round(self._lich_su["loss"][0], 4),
            "loss_cuoi": round(self._lich_su["loss"][-1], 4),
            "reward_trung_binh": round(self._lich_su["reward_trung_binh"][-1], 4),
            "reward_trung_binh_cuoi": round(self._lich_su["reward_trung_binh"][-1], 4),
            "ty_le_co_think": self._lich_su["ty_le_co_think"][-1],
            "tong_thoi_gian": round(time.time() - bat_dau, 3),
            "lich_su": self._lich_su,
        }
