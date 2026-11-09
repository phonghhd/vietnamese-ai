import numpy as np

from vietnamese_ai.serving.speculative import SpeculativeEngine


class MockModel:
    """Mock Model để test Speculative Decoding."""

    def __init__(self, vocab_size: int, is_target: bool):
        self.vocab_size = vocab_size
        self.is_target = is_target
        self.call_count = 0

    def __call__(self, input_ids: list) -> np.ndarray:
        self.call_count += 1
        seq_len = len(input_ids)
        # Giả lập trả về Logits cho toàn bộ chuỗi dựa trên token ngữ cảnh
        logits = np.zeros((seq_len, self.vocab_size))
        for i, tid in enumerate(input_ids):
            next_tid = (tid + 1) % self.vocab_size
            logits[i, next_tid] = 5.0 if self.is_target else 4.0

        return logits


def test_speculative_decoding_speedup():
    """Kiểm tra Speculative Decoding sinh token nhanh hơn Auto-regressive thông thường."""
    np.random.seed(42)
    vocab_size = 100
    gamma = 4
    max_tokens = 20

    target = MockModel(vocab_size=vocab_size, is_target=True)
    draft = MockModel(vocab_size=vocab_size, is_target=False)

    engine = SpeculativeEngine(target_model=target, draft_model=draft, gamma=gamma)

    input_prompt = [1, 5, 10]

    # Chạy thuật toán
    output_ids = engine.sinh_van_ban(input_ids=input_prompt, max_tokens=max_tokens)

    # Sinh ra đủ token
    tokens_generated = len(output_ids) - len(input_prompt)
    assert tokens_generated >= max_tokens

    # Kiểm tra HIỆU SUẤT TĂNG TỐC (Cốt lõi của Speculative Decoding)
    # Target Model chỉ nên được gọi một số lần nhỏ hơn số lượng token sinh ra
    # Nếu chạy Auto-regressive thường, target.call_count == tokens_generated (Ví dụ 20)
    # Với Speculative, target.call_count phải < 20 (Thường khoảng 5-10)

    assert target.call_count < tokens_generated
    print(f"Tokens Sinh Ra: {tokens_generated}")
    print(f"Số lần Target Model tính toán (Forward Pass): {target.call_count}")
    print(f"Tốc độ tăng tốc (Speedup Ratio): {tokens_generated / target.call_count:.2f}x")
