"""Speculative Decoding - Tăng tốc suy luận thông minh."""

from typing import Any, List

import numpy as np


class SpeculativeEngine:
    """
    Speculative Decoding Engine (Thuật toán Toán học Chuẩn xác).
    1. Draft Model sinh gamma tokens.
    2. Target Model chấm điểm (Forward pass 1 lần) các tokens đó.
    3. Rejection Sampling để chấp nhận/từ chối token.
    """

    def __init__(self, target_model: Any, draft_model: Any, gamma: int = 4):
        """
        Args:
            target_model: Mô hình lớn (Chậm, Chính xác). Trả về (logits, next_token).
            draft_model: Mô hình nhỏ (Nhanh, Kém chính xác). Trả về (logits, next_token).
            gamma: Số lượng token đoán trước mỗi bước.
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma

    def _sample_from_logits(self, logits: np.ndarray) -> int:
        """Lấy token ID từ phân phối xác suất (Argmax đơn giản hoá)."""
        # Trong thực tế có thể dùng Top-P, Temperature. Ở đây ta dùng Argmax.
        return int(np.argmax(logits[-1]))

    def _get_probs_dist(self, logits: np.ndarray) -> np.ndarray:
        """Biến đổi Logits thành phân phối xác suất đầy đủ qua Softmax ổn định số học."""
        last_logits = np.asarray(logits[-1], dtype=np.float64)
        exp_logits = np.exp(last_logits - np.max(last_logits))
        return exp_logits / (np.sum(exp_logits) + 1e-12)

    def _get_prob(self, logits: np.ndarray, token_id: int) -> float:
        """Lấy xác suất của một token_id từ phân phối logits."""
        probs = self._get_probs_dist(logits)
        return float(probs[token_id])

    def _sample_residual(self, probs_t: np.ndarray, probs_d: np.ndarray) -> int:
        """Lấy mẫu chính xác từ phân phối phần dư chuẩn hóa max(0, P_t(x) - P_d(x))."""
        residual = np.maximum(0.0, probs_t - probs_d)
        total = float(np.sum(residual))
        if total <= 1e-12:
            return int(np.argmax(probs_t))
        residual = residual / total
        return int(np.random.choice(len(residual), p=residual))

    def sinh_van_ban(self, input_ids: List[int], max_tokens: int = 128) -> List[int]:
        """
        Thực thi quá trình Speculative Decoding trên Token IDs với lấy mẫu phần dư chuẩn xác.
        """
        ket_qua = input_ids.copy()
        tokens_sinh_ra = 0

        while tokens_sinh_ra < max_tokens:
            draft_tokens = []
            draft_probs = []
            draft_dists = []
            draft_context = ket_qua.copy()

            for _ in range(self.gamma):
                logits_d = self.draft_model(draft_context)
                dist_d = self._get_probs_dist(logits_d)
                next_token_d = self._sample_from_logits(logits_d)
                prob_d = float(dist_d[next_token_d])

                draft_tokens.append(next_token_d)
                draft_probs.append(prob_d)
                draft_dists.append(dist_d)
                draft_context.append(next_token_d)

            logits_t_seq = self.target_model(draft_context)

            n_accepted = 0
            len_ket_qua_ban_dau = len(ket_qua)
            for i in range(self.gamma):
                t_token_idx = len_ket_qua_ban_dau + i
                logits_t = logits_t_seq[t_token_idx - 1 : t_token_idx]
                dist_t = self._get_probs_dist(logits_t)

                prob_t = float(dist_t[draft_tokens[i]])
                prob_d = draft_probs[i]

                r_ratio = prob_t / (prob_d + 1e-9)
                r_random = np.random.uniform(0, 1)

                if r_random <= r_ratio:
                    ket_qua.append(draft_tokens[i])
                    n_accepted += 1
                else:
                    correct_token = self._sample_residual(dist_t, draft_dists[i])
                    ket_qua.append(correct_token)
                    break

            if n_accepted == self.gamma:
                last_logits = logits_t_seq[-1:]
                bonus_token = self._sample_from_logits(last_logits)
                ket_qua.append(bonus_token)
                n_accepted += 1

            tokens_sinh_ra += n_accepted if n_accepted > 0 else 1

        return ket_qua
