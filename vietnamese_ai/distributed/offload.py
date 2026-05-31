"""Module hỗ trợ Offloading (San sẻ bộ nhớ) giữa GPU và CPU."""

import logging
from typing import Any, Dict, Optional

try:
    import torch

    _CO_TORCH = True
except ImportError:
    _CO_TORCH = False

logger = logging.getLogger("OffloadEngine")


class OffloadEngine:
    """
    Công cụ tính toán chiến lược chia sẻ mô hình giữa GPU và CPU.
    Sử dụng thuật toán ước tính dung lượng dựa trên kích thước VRAM và RAM.
    """

    def __init__(self):
        if not _CO_TORCH:
            raise ImportError("OffloadEngine yêu cầu cài đặt PyTorch.")

    @staticmethod
    def tinh_toan_device_map(
        model_size_gb: float,
        max_gpu_vram_gb: Optional[float] = None,
        max_cpu_ram_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Tính toán Device Map để tự động chia model.

        Args:
            model_size_gb: Kích thước mô hình ước tính (Gigabytes).
            max_gpu_vram_gb: Giới hạn VRAM GPU tối đa cho phép sử dụng.
            max_cpu_ram_gb: Giới hạn RAM CPU tối đa cho phép sử dụng.

        Returns:
            Dict chứa cấu hình device_map (thích hợp cho Hugging Face `accelerate`).
        """
        if not torch.cuda.is_available():
            logger.warning("Không tìm thấy GPU. Toàn bộ mô hình sẽ chạy trên CPU.")
            return {"device_map": {"": "cpu"}}

        # Lấy thông số hệ thống thực tế nếu không cung cấp
        gpu_vram_thuc = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_cho_phep = (
            max_gpu_vram_gb if max_gpu_vram_gb else (gpu_vram_thuc * 0.9)
        )  # Dùng tối đa 90%

        if model_size_gb <= vram_cho_phep:
            logger.info(
                f"Mô hình ({model_size_gb:.1f}GB) vừa với GPU VRAM ({vram_cho_phep:.1f}GB). Không cần Offload."
            )
            return {"device_map": {"": 0}}  # Chạy toàn bộ trên GPU 0

        logger.warning(
            f"Mô hình ({model_size_gb:.1f}GB) vượt quá GPU VRAM ({vram_cho_phep:.1f}GB). Kích hoạt CPU Offloading!"
        )

        # Cấu hình chia sẻ bộ nhớ (Dành cho Hugging Face Accelerate)
        max_memory_config = {
            0: f"{int(vram_cho_phep)}GiB",
            "cpu": f"{int(max_cpu_ram_gb if max_cpu_ram_gb else 32)}GiB",
        }

        return {
            "device_map": "auto",
            "max_memory": max_memory_config,
            "offload_folder": "offload_weights",  # Thư mục tạm nếu RAM cũng đầy (Disk Offloading)
            "offload_state_dict": True,
        }
