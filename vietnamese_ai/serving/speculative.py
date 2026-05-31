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

    def _get_prob(self, logits: np.ndarray, token_id: int) -> float:
        """Lấy xác suất của một token_id từ phân phối logits."""
        # Biến đổi Logits thành Xác suất qua Softmax
        exp_logits = np.exp(logits[-1] - np.max(logits[-1]))
        probs = exp_logits / np.sum(exp_logits)
        return probs[token_id]

    def sinh_van_ban(self, input_ids: List[int], max_tokens: int = 128) -> List[int]:
        """
        Thực thi quá trình Speculative Decoding trên Token IDs.
        """
        ket_qua = input_ids.copy()
        tokens_sinh_ra = 0

        while tokens_sinh_ra < max_tokens:
            # BƯỚC 1: Vòng lặp Nháp (Drafting)
            draft_tokens = []
            draft_probs = []
            draft_context = ket_qua.copy()

            for _ in range(self.gamma):
                logits_d = self.draft_model(draft_context)
                next_token_d = self._sample_from_logits(logits_d)
                prob_d = self._get_prob(logits_d, next_token_d)

                draft_tokens.append(next_token_d)
                draft_probs.append(prob_d)
                draft_context.append(next_token_d)

            # BƯỚC 2: Kiểm duyệt bằng Target Model (Verification)
            # Truyền toàn bộ ngữ cảnh (Kể cả token nháp) vào mô hình lớn
            # Trong thực tế, hàm này nhận danh sách và trả về Logits cho từng bước
            logits_t_seq = self.target_model(draft_context)

            # BƯỚC 3: Rejection Sampling (Duyệt qua từng token nháp)
            n_accepted = 0
            for i in range(self.gamma):
                t_token_idx = len(ket_qua) + i
                # Logits do Target dự đoán cho vị trí này
                logits_t = logits_t_seq[t_token_idx - 1 : t_token_idx]

                prob_t = self._get_prob(logits_t, draft_tokens[i])
                prob_d = draft_probs[i]

                # Tính tỷ lệ chấp nhận
                # Nếu P_target >= P_draft -> R_ratio >= 1 -> Luôn chấp nhận
                r_ratio = prob_t / (prob_d + 1e-9)
                r_random = np.random.uniform(0, 1)

                if r_random <= r_ratio:
                    # Gật đầu (Accept)
                    ket_qua.append(draft_tokens[i])
                    n_accepted += 1
                else:
                    # Lắc đầu (Reject) -> Sửa sai bằng Target Model
                    # Token sửa sai sẽ được sample lại từ phân phối Max(0, P_t - P_d)
                    # Ở đây đơn giản hoá bằng cách lấy Argmax của Target Model
                    correct_token = self._sample_from_logits(logits_t)
                    ket_qua.append(correct_token)
                    break  # Từ chối token này thì vứt bỏ mọi token nháp phía sau

            # Nếu tất cả token nháp đều được chấp nhận, Target model được thưởng sinh thêm 1 token
            if n_accepted == self.gamma:
                last_logits = logits_t_seq[-1:]
                bonus_token = self._sample_from_logits(last_logits)
                ket_qua.append(bonus_token)
                n_accepted += 1

            tokens_sinh_ra += n_accepted if n_accepted > 0 else 1  # Ít nhất 1 token được sinh ra

        return ket_qua
