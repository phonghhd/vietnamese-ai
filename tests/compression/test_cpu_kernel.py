import torch

from vietnamese_ai.compression.cpu_kernel import EvoKernelCPU
from vietnamese_ai.distributed.offload import OffloadEngine


def test_cpu_kernel_accuracy():
    # Tạo tensor mô phỏng 1.58-bit
    batch_size = 4
    in_features = 256
    out_features = 128

    x = torch.randn(batch_size, in_features)

    # Sinh trọng số -1, 0, 1 ngẫu nhiên
    w_quantized = torch.randint(-1, 2, (in_features, out_features)).to(torch.int8)

    # 1. Chạy với PyTorch gốc (Casting to float để nhân)
    w_float = w_quantized.to(torch.float32)
    out_torch = torch.matmul(x, w_float)

    # 2. Chạy với EvoKernelCPU (Thuật toán Add-only)
    out_evo = EvoKernelCPU.add_only_matmul(x, w_quantized)

    # So sánh kết quả (phải giống hệt nhau về mặt toán học, cho phép sai số float)
    diff = torch.max(torch.abs(out_torch - out_evo))
    assert diff < 1e-4, f"Lỗi sai số quá lớn: {diff}"


def test_offload_engine():
    # Mô phỏng tính toán cấu hình Offload
    # TH1: Mô hình 10GB, GPU có 8GB, RAM 16GB
    config = OffloadEngine.tinh_toan_device_map(
        model_size_gb=10.0, max_gpu_vram_gb=8.0, max_cpu_ram_gb=16.0
    )

    # Nếu hệ thống test không có GPU, hàm sẽ trả về cpu
    if config.get("device_map") == {"": "cpu"}:
        assert True
    else:
        assert config["device_map"] == "auto"
        assert "max_memory" in config
        assert "cpu" in config["max_memory"]
        assert config["max_memory"]["cpu"] == "16GiB"
