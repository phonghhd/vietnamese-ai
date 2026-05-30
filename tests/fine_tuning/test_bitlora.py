import torch
import pytest
from vietnamese_ai.compression.extreme import BitLinear
from vietnamese_ai.fine_tuning.bitlora import BitLoRALinear

def test_bitlora_freezes_base():
    in_features = 128
    out_features = 64
    base = BitLinear(in_features, out_features)
    
    # Gói vào BitLoRA
    lora = BitLoRALinear(base, r=8)
    
    # Kiểm tra xem Base có bị khóa gradient không
    assert lora.base_layer.weight.requires_grad == False
    if lora.base_layer.bias is not None:
        assert lora.base_layer.bias.requires_grad == False
        
    # Lora A và Lora B phải có gradient (sẵn sàng train)
    assert lora.lora_A.weight.requires_grad == True
    assert lora.lora_B.weight.requires_grad == True

def test_bitlora_forward_pass():
    batch_size = 4
    in_features = 128
    out_features = 64
    
    base = BitLinear(in_features, out_features)
    lora = BitLoRALinear(base, r=8)
    
    x = torch.randn(batch_size, in_features)
    
    # Forward pass kết hợp base 1.58-bit và lora FP32
    out = lora(x)
    
    # Đảm bảo output có kích thước chính xác
    assert out.shape == (batch_size, out_features)
