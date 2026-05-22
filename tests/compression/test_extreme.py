import torch
from vietnamese_ai.compression.extreme import BitLinear

def test_bitlinear_quantization():
    """Kiểm tra lớp BitLinear có trả về đúng trọng số [-1, 0, 1] hay không."""
    in_features = 10
    out_features = 5
    layer = BitLinear(in_features, out_features)
    
    # Ép trọng số ban đầu
    with torch.no_grad():
        layer.weight.data = torch.randn(out_features, in_features) * 2.0
        
    x = torch.randn(2, in_features)
    
    # Chạy forward pass
    out = layer(x)
    assert out.shape == (2, out_features)
    
    # Lấy trọng số đã được lượng tử hoá (thông qua hàm _quantize_weight_1_58_bit)
    quantized_w = layer.weight_quantizer(layer.weight)
    
    # Kiểm tra tất cả các giá trị đều thuộc tập {-1, 0, 1}
    unique_vals = torch.unique(quantized_w)
    for val in unique_vals:
        assert val.item() in [-1.0, 0.0, 1.0]
