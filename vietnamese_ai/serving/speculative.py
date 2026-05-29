from typing import Any


class SpeculativeEngine:
    """
    Công cụ tăng tốc suy luận (Inference) sử dụng kỹ thuật Speculative Decoding.
    Cần hai mô hình:
    1. Draft Model (Mô hình nháp - nhỏ, siêu nhanh)
    2. Target Model (Mô hình chính - lớn, chính xác)
    """
    def __init__(self, target_model: Any, draft_model: Any, gamma: int = 4):
        """
        Args:
            target_model: Mô hình LLM chính (ví dụ 70B parameters)
            draft_model: Mô hình LLM nháp (ví dụ 1B parameters)
            gamma: Số token mà Draft model sẽ đoán trước trong mỗi bước.
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma

    def _goi_mo_hinh(self, model: Any, prompt: str, num_tokens: int) -> str:
        """Hàm bọc gọi mô hình (Mô phỏng)."""
        if hasattr(model, "sinh_van_ban"):
            # Chú ý: Ở hệ thống thực tế, cần API trả về list of tokens, không phải string text.
            # Ở đây mô phỏng việc sinh văn bản độ dài ngắn
            return model.sinh_van_ban(prompt, do_dai=num_tokens)
        elif callable(model):
            return str(model(prompt))
        return f" [Tạo bởi {model.__class__.__name__}]"

    def sinh_van_ban(self, prompt: str, max_tokens: int = 128) -> str:
        """
        Thực thi quá trình Speculative Decoding (Mô phỏng logic).
        """
        ket_qua = []
        tokens_sinh_ra = 0
        prompt_hien_tai = prompt

        while tokens_sinh_ra < max_tokens:
            # BƯỚC 1: Draft Model đoán trước `gamma` tokens
            draft_guess = self._goi_mo_hinh(self.draft_model, prompt_hien_tai, num_tokens=self.gamma)

            # Trong thực tế, Target Model sẽ chấm điểm (forward pass 1 lần) các token đoán này
            # Ở bản mô phỏng này, ta giả định Target Model "chấp nhận" 80% độ dài của draft_guess
            # và tự sinh thêm 1 token mới (như thuật toán thực tế).

            # Giả lập token được chấp nhận
            accepted_text = draft_guess.strip() + " "

            # Target model sinh ra 1 token cuối cùng
            target_correction = self._goi_mo_hinh(self.target_model, prompt_hien_tai + accepted_text, num_tokens=1)

            buoc_nay = accepted_text + target_correction.strip()
            ket_qua.append(buoc_nay)
            prompt_hien_tai += buoc_nay + " "

            # Cập nhật số token (mô phỏng 1 word = 1 token)
            tokens_sinh_ra += len(buoc_nay.split())

            # Thêm logic ngắt nếu gặp token kết thúc (dấu chấm)
            if '.' in buoc_nay:
                break

        return "".join(ket_qua).strip()
