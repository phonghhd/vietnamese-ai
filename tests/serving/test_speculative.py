from vietnamese_ai.serving.speculative import SpeculativeEngine


import numpy as np

from vietnamese_ai.serving.speculative import SpeculativeEngine


def mock_draft(ctx):
    # Trả về numpy array mô phỏng logits: (1, vocab_size=10)
    # Token 1 luôn có xác suất cao nhất
    logits = np.zeros((1, 10))
    logits[0, 1] = 10.0
    return logits


def mock_target(ctx):
    # Trả về logits sequence: (len(ctx), vocab_size=10)
    # Token 2 luôn có xác suất cao nhất
    logits = np.zeros((len(ctx), 10))
    for i in range(len(ctx)):
        logits[i, 2] = 10.0
    return logits


def test_speculative_decoding():
    engine = SpeculativeEngine(target_model=mock_target, draft_model=mock_draft, gamma=2)

    # Chạy speculative decoding với input là List[int]
    kq = engine.sinh_van_ban([0], max_tokens=5)

    # Kết quả sẽ có ít nhất 5 tokens, cộng thêm token ban đầu
    assert len(kq) >= 5
    # Token do target sửa (lắc đầu) sẽ là 2
    assert 2 in kq
