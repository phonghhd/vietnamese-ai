"""Speculative Decoding - Tăng tốc suy luận thông minh."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SpeculativeEngine:
    """
    Speculative Decoding Engine (Thuật toán Toán học Chuẩn xác — Leviathan et al. 2023).
    1. Draft Model sinh gamma tokens (lấy mẫu theo `nhiet_do` từ phân phối draft `P_d`).
    2. Target Model chấm điểm (Forward pass 1 lần) toàn bộ chuỗi đề xuất.
    3. Rejection Sampling với phân phối phần dư `norm(max(0, P_t - P_d))` bảo toàn phân phối đích `P_t`.
    """

    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        gamma: int = 4,
        nhiet_do: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            target_model: Mô hình lớn (Chậm, Chính xác). Trả về logits 2D `(seq, vocab)` hoặc 3D `(batch, seq, vocab)`.
            draft_model: Mô hình nhỏ (Nhanh, Kém chính xác). Trả về logits 2D `(seq, vocab)` hoặc 3D `(batch, seq, vocab)`.
            gamma: Số lượng token đoán trước mỗi bước.
            nhiet_do: Nhiệt độ lấy mẫu (`0.0` cho giải mã tham lam; `> 0.0` cho lấy mẫu ngẫu nhiên bảo toàn phân phối đích).
            seed: Hạt giống ngẫu nhiên tùy chọn để tái lập kết quả.
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = max(1, int(gamma))
        self.nhiet_do = float(nhiet_do)
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self._tong_de_xuat = 0
        self._tong_chap_nhan = 0
        self._so_vong = 0

    @staticmethod
    def _chuan_hoa_logits_2d(logits: np.ndarray) -> np.ndarray:
        """Chuẩn hóa mảng logits (1D/2D/3D) về dạng 2D `(seq_len, vocab_size)` an toàn."""
        arr = np.asarray(logits, dtype=np.float64)
        if arr.ndim == 3:
            return arr[0]
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def _get_probs_dist(self, logits: np.ndarray) -> np.ndarray:
        """Biến đổi Logits cuối chuỗi thành phân phối xác suất đầy đủ qua Softmax ổn định số học."""
        arr_2d = self._chuan_hoa_logits_2d(logits)
        last_logits = arr_2d[-1]
        temp = self.nhiet_do if self.nhiet_do > 1e-6 else 1.0
        scaled = last_logits / temp
        exp_logits = np.exp(scaled - np.max(scaled))
        return exp_logits / (np.sum(exp_logits) + 1e-12)

    def _sample_from_dist(self, probs: np.ndarray) -> int:
        """Lấy mẫu token ID theo nhiệt độ (tham lam khi `nhiet_do <= 1e-6`, ngẫu nhiên chuẩn khi `nhiet_do > 0`)."""
        if self.nhiet_do <= 1e-6:
            return int(np.argmax(probs))
        p_norm = np.asarray(probs, dtype=np.float64)
        p_norm = p_norm / (float(np.sum(p_norm)) + 1e-12)
        if self._rng is not None:
            return int(self._rng.choice(len(p_norm), p=p_norm))
        return int(np.random.choice(len(p_norm), p=p_norm))

    def _sample_from_logits(self, logits: np.ndarray) -> int:
        """Lấy token ID từ phân phối xác suất của logits."""
        return self._sample_from_dist(self._get_probs_dist(logits))

    def _get_prob(self, logits: np.ndarray, token_id: int) -> float:
        """Lấy xác suất của một token_id từ phân phối logits."""
        probs = self._get_probs_dist(logits)
        return float(probs[token_id])

    def _sample_residual(self, probs_t: np.ndarray, probs_d: np.ndarray) -> int:
        """Lấy mẫu chính xác từ phân phối phần dư chuẩn hóa max(0, P_t(x) - P_d(x))."""
        residual = np.maximum(0.0, probs_t - probs_d)
        total = float(np.sum(residual))
        if total <= 1e-12:
            return self._sample_from_dist(probs_t)
        residual = residual / total
        if self._rng is not None:
            return int(self._rng.choice(len(residual), p=residual))
        return int(np.random.choice(len(residual), p=residual))

    def _uniform_01(self) -> float:
        if self._rng is not None:
            return float(self._rng.uniform(0.0, 1.0))
        return float(np.random.uniform(0.0, 1.0))

    def sinh_van_ban(self, input_ids: List[int], max_tokens: int = 128) -> List[int]:
        """
        Thực thi quá trình Speculative Decoding trên Token IDs với lấy mẫu phần dư chuẩn xác.
        """
        ket_qua = list(input_ids)
        tokens_sinh_ra = 0

        while tokens_sinh_ra < max_tokens:
            buoc_gamma = min(self.gamma, max_tokens - tokens_sinh_ra)
            draft_tokens: List[int] = []
            draft_probs: List[float] = []
            draft_dists: List[np.ndarray] = []
            draft_context = ket_qua.copy()

            for _ in range(buoc_gamma):
                logits_d = self.draft_model(draft_context)
                dist_d = self._get_probs_dist(logits_d)
                next_token_d = self._sample_from_dist(dist_d)
                prob_d = float(dist_d[next_token_d])

                draft_tokens.append(next_token_d)
                draft_probs.append(prob_d)
                draft_dists.append(dist_d)
                draft_context.append(next_token_d)

            self._so_vong += 1
            self._tong_de_xuat += len(draft_tokens)

            logits_t_seq = self._chuan_hoa_logits_2d(self.target_model(draft_context))

            n_accepted = 0
            len_ket_qua_ban_dau = len(ket_qua)
            bi_tu_choi = False
            for i in range(buoc_gamma):
                t_pos = min(len_ket_qua_ban_dau + i - 1, logits_t_seq.shape[0] - 1)
                logits_t = logits_t_seq[t_pos : t_pos + 1]
                dist_t = self._get_probs_dist(logits_t)

                prob_t = float(dist_t[draft_tokens[i]])
                prob_d = draft_probs[i]

                r_ratio = prob_t / (prob_d + 1e-12)
                if self._uniform_01() <= r_ratio:
                    ket_qua.append(draft_tokens[i])
                    n_accepted += 1
                    self._tong_chap_nhan += 1
                else:
                    correct_token = self._sample_residual(dist_t, draft_dists[i])
                    ket_qua.append(correct_token)
                    bi_tu_choi = True
                    break

            them_thuc = n_accepted + (1 if bi_tu_choi else 0)
            if not bi_tu_choi and n_accepted == buoc_gamma and (tokens_sinh_ra + them_thuc) < max_tokens:
                last_logits = logits_t_seq[-1:]
                bonus_token = self._sample_from_logits(last_logits)
                ket_qua.append(bonus_token)
                them_thuc += 1

            tokens_sinh_ra += max(1, them_thuc)

        return ket_qua[: len(input_ids) + max_tokens]

    def lay_thong_ke(self) -> Dict[str, Any]:
        """Trả về thống kê tỷ lệ chấp nhận token dự đoán của Speculative Decoding."""
        ty_le = self._tong_chap_nhan / max(1, self._tong_de_xuat)
        return {
            "gamma": self.gamma,
            "nhiet_do": self.nhiet_do,
            "so_vong": self._so_vong,
            "tong_de_xuat": self._tong_de_xuat,
            "tong_chap_nhan": self._tong_chap_nhan,
            "ty_le_chap_nhan": round(float(ty_le), 4),
        }
