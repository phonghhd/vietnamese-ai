import numpy as np
from vietnamese_ai.transformer.attention import MultiHeadLatentAttention
from vietnamese_ai.models.deepseek_moe import DeepSeekMoE
from vietnamese_ai.serving.kv_cache import QuantizedKVCache

def test_multi_head_latent_attention():
    """Kiểm tra MLA nén và giải nén thành công không lỗi."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    so_dau = 4
    latent_dim = 16 # Ép nén 75%
    
    mla = MultiHeadLatentAttention(d_model=d_model, so_dau=so_dau, latent_dim=latent_dim)
    
    # Input tensor
    X = np.random.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output = mla.tien(X)
    
    assert output.shape == (batch_size, seq_len, d_model)
    
def test_deepseek_moe_routing():
    """Kiểm tra Router của DeepSeekMoE có hoạt động không."""
    batch_size = 3
    d_model = 32
    top_k = 2
    num_routed = 8
    
    moe = DeepSeekMoE(d_model=d_model, num_shared_experts=1, num_routed_experts=num_routed, top_k=top_k)
    
    X = np.random.randn(batch_size, d_model)
    
    output = moe.tien(X)
    
    assert output.shape == (batch_size, d_model)
    
def test_quantized_kv_cache():
    """Kiểm tra Quantization KV Cache giảm dung lượng và khôi phục hợp lệ."""
    seq_len = 5
    d_model = 16
    max_len = 20
    
    cache = QuantizedKVCache(max_seq_len=max_len, d_model=d_model)
    
    # Giả lập Key/Value từ Transformer
    K = np.random.randn(seq_len, d_model).astype(np.float32)
    V = np.random.randn(seq_len, d_model).astype(np.float32)
    
    cache.append(K, V)
    
    assert cache.current_len == 5
    assert cache.k_cache.dtype == np.int8 # Đã ép xuống INT8
    
    # Lấy ra
    K_out, V_out = cache.get_cache()
    
    assert K_out.shape == (5, d_model)
    assert K_out.dtype == np.float32 # Giải nén thành công
    
    # Kiểm tra sai số (Lossy compression)
    # Vì ép kiểu sẽ có sai số nhỏ, ta chỉ kiểm tra độ tương quan
    mae = np.mean(np.abs(K - K_out))
    assert mae < 0.1 # Sai số trung bình phải nhỏ

def test_flash_attention_output():
    """Kiểm tra FlashAttention tạo ra output (gần) giống Attention thường."""
    from vietnamese_ai.transformer.attention import MultiHeadAttention, MultiHeadFlashAttention
    
    batch_size = 2
    seq_len = 32
    d_model = 64
    so_dau = 4
    
    # Khởi tạo 2 module với cùng weight
    attn_normal = MultiHeadAttention(d_model=d_model, so_dau=so_dau)
    attn_flash = MultiHeadFlashAttention(d_model=d_model, so_dau=so_dau, block_size=8)
    
    # Ép weights giống nhau để test
    attn_flash.W_q = attn_normal.W_q
    attn_flash.W_k = attn_normal.W_k
    attn_flash.W_v = attn_normal.W_v
    attn_flash.W_o = attn_normal.W_o
    
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    out_normal = attn_normal.tien(Q, K, V)
    out_flash = attn_flash.tien(Q, K, V)
    
    # So sánh output (Flash Attention có thể lệch 1 chút do sai số thập phân phân rã)
    np.testing.assert_allclose(out_normal, out_flash, rtol=1e-5, atol=1e-5)

def test_jit_bitnet():
    """Kiểm tra C++ JIT Compiler cho thuật toán BitNet 1.58-bit."""
    from vietnamese_ai.extreme.cpu_kernel import EvoKernelCPU
    import time
    
    batch_size = 4
    in_features = 256
    out_features = 128
    
    X = np.random.randn(batch_size, in_features).astype(np.float32)
    # Lượng tử hóa W thành -1, 0, 1
    W = np.random.randint(-1, 2, size=(in_features, out_features)).astype(np.int8)
    
    # Tính bằng Python/NumPy chuẩn để đối chiếu
    # Out = X * W
    out_expected = X @ W.astype(np.float32)
    
    # Tính bằng C++ JIT (Bypass GIL, Add-Only)
    start_time = time.time()
    out_jit = EvoKernelCPU.add_only_matmul(X, W)
    jit_time = time.time() - start_time
    
    print(f"JIT Compilation & Execution Time: {jit_time:.4f}s")
    
    # Sai số phải bằng 0 (vì phép cộng hoàn toàn tương đương phép nhân int)
    np.testing.assert_allclose(out_expected, out_jit, rtol=1e-5, atol=1e-5)
