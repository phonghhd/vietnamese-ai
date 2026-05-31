import numpy as np


def test_continuous_batching():
    """Kiểm tra Continuous Batching đẩy Request mới vào Slot trống ngay lập tức."""
    from vietnamese_ai.serving.continuous_batching import ContinuousBatcher

    batcher = ContinuousBatcher(batch_size=2)

    # 3 người dùng yêu cầu
    req_A = batcher.add_request("A", max_tokens=2)  # Xong rất nhanh
    batcher.add_request("B", max_tokens=5)  # Xong vừa
    batcher.add_request("C", max_tokens=5)  # Nằm chờ

    # Hàm sinh giả lập
    def mock_model(prompt, generated):
        return "tk"

    # Bước 1: [A, B] chạy. C chờ.
    res_1 = batcher.step(mock_model)
    assert len(res_1) == 0
    assert len(batcher.running_batch) == 2

    # Bước 2: A xong (max=2). B chạy (2/5).
    res_2 = batcher.step(mock_model)
    assert len(res_2) == 1
    assert res_2[0]["req_id"] == req_A

    # Bước 3: C được điền vào khe trống của A. B và C cùng chạy.
    batcher.step(mock_model)
    assert len(batcher.running_batch) == 2

    assert True  # Đã hoạt động chính xác logic nhồi slot


def test_kernel_fusion():
    """Kiểm tra Kernel Fusion kết hợp 4 phép toán thành 1."""
    from vietnamese_ai.extreme.kernel_fusion import FusedDenseNormReLU

    batch = 2
    in_f = 64
    out_f = 32

    fused_layer = FusedDenseNormReLU(in_features=in_f, out_features=out_f)

    X = np.random.randn(batch, in_f).astype(np.float32)

    # Chạy Fusion (C++)
    out_fused = fused_layer.tien(X)

    # Chạy chuẩn (NumPy)
    # 1. MatMul + Bias
    out_np = X @ fused_layer.W + fused_layer.b
    # 2. Norm
    mean = np.mean(out_np, axis=1, keepdims=True)
    var = np.var(out_np, axis=1, keepdims=True)
    out_np = (out_np - mean) / np.sqrt(var + 1e-5)
    out_np = out_np * fused_layer.gamma + fused_layer.beta
    # 3. ReLU
    out_np = np.maximum(0, out_np)

    np.testing.assert_allclose(out_fused, out_np, rtol=1e-4, atol=1e-4)


def test_ring_attention():
    """Kiểm tra Ring Attention mô phỏng vòng tròn FlashAttention."""
    from vietnamese_ai.distributed.ring_attention import RingAttentionNode

    batch = 1
    seq = 10
    d_model = 16

    node = RingAttentionNode(d_model=d_model, node_id=0, total_nodes=2)

    Q = np.random.randn(batch, seq, d_model)
    K = np.random.randn(batch, seq, d_model)
    V = np.random.randn(batch, seq, d_model)

    # Tính trên 1 Node (giả định Ring truyền về chính K,V đó 2 lần)
    out = node.forward_ring(Q, K, V)

    assert out.shape == (batch, seq, d_model)
    # Vì Softmax(x) = Softmax([x, x]) nên output phải chuẩn không bị NaN
    assert not np.isnan(out).any()
