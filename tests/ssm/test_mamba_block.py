import torch

from vietnamese_ai.ssm.mamba_block import MambaBlock


def test_mamba_block_forward():
    batch_size = 2
    # Thử nghiệm với Context Window dài để kiểm tra tính ổn định OOM
    seqlen = 1024
    d_model = 128

    mamba = MambaBlock(d_model=d_model, d_state=16)

    x = torch.randn(batch_size, seqlen, d_model)
    out = mamba(x)

    # Kích thước đầu ra phải giống đầu vào
    assert out.shape == (batch_size, seqlen, d_model)

    # Cần có requires_grad để dùng cho huấn luyện
    assert out.requires_grad
