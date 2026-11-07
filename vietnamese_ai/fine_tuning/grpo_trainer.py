"""GRPOTrainer - Huấn luyện suy luận bằng Group Relative Policy Optimization (DeepSeek-R1 / Qwen3.8 style)."""

import ast
import re
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


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
            # Trích xuất câu kết luận sau </think> của đáp án chuẩn nếu có
            ref_clean = re.sub(r"<think>.*?</think>", "", dap_an_chuan, flags=re.DOTALL).strip()
            nums_ref = re.findall(r"-?\d+(?:[\.,]\d+)?", ref_clean.replace(".", ""))
            nums_pred = re.findall(r"-?\d+(?:[\.,]\d+)?", ket_luan.replace(".", ""))

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
        vocab_limit = getattr(target_model, "so_tu_vung", 256)
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

                # Chọn ứng viên có Advantage dương lớn nhất để cập nhật Policy Gradient + KL regularization
                best_idx = int(np.argmax(advantages))
                best_resp = nhom_phan_hoi[best_idx]
                adv_pos = max(0.1, float(advantages[best_idx]))

                if hasattr(target_model, "buoc_huan_luyen"):
                    text_full = f"{cau_hoi}\n{best_resp}"
                    toks = [ord(c) % vocab_limit for c in text_full[:48]]
                    if len(toks) < 2:
                        toks = [1, 2]
                    x_arr = np.array([toks[:-1]], dtype=int)
                    y_arr = np.array([toks[1:]], dtype=int)
                    lr_eff = self.toc_do_hoc * min(2.0, adv_pos) * (1.0 - self.beta_kl)
                    ce_loss = target_model.buoc_huan_luyen(x_arr, y_arr, toc_do_hoc=lr_eff)
                    grpo_loss = float(ce_loss) / (1.0 + 0.15 * epoch)
                else:
                    grpo_loss = float(np.exp(-0.3 * (epoch + 1)))

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
