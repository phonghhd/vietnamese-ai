import torch
import pytest
from vietnamese_ai.ssm.jamba_block import JambaBlock

def test_jamba_block_forward():
    batch_size = 2
    seqlen = 64
    d_model = 128
    
    # Test nhánh Mamba (layer_idx % 8 != 0)
    jamba_mamba = JambaBlock(d_model=d_model, layer_idx=1)
    x = torch.randn(batch_size, seqlen, d_model)
    out_mamba = jamba_mamba(x)
    
    assert out_mamba.shape == (batch_size, seqlen, d_model)
    assert out_mamba.requires_grad == True
    
    # Test nhánh Attention + MoE (layer_idx % 8 == 0)
    jamba_attn = JambaBlock(d_model=d_model, layer_idx=8)
    out_attn = jamba_attn(x)
    
    assert out_attn.shape == (batch_size, seqlen, d_model)
    assert out_attn.requires_grad == True
