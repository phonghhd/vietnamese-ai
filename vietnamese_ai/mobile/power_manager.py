"""Quản lý Năng lượng Thông minh cho Mobile/Edge."""

import logging
from typing import Any, Dict, Tuple

try:
    import psutil

    _CO_PSUTIL = True
except ImportError:
    _CO_PSUTIL = False

logger = logging.getLogger("PowerManager")


class PowerManager:
    """
    Theo dõi pin và cấu hình thiết bị để tự động điều tiết độ chính xác của AI.
    """

    @staticmethod
    def get_battery_status() -> Tuple[float, bool]:
        """
        Lấy % pin và trạng thái sạc.

        Returns:
            Tuple: (Phần trăm pin, Có đang cắm sạc không)
        """
        if not _CO_PSUTIL:
            logger.warning(
                "Cần cài đặt psutil (pip install psutil) để đọc pin. Mặc định trả về 100% sạc."
            )
            return 100.0, True

        battery = psutil.sensors_battery()
        if battery is None:
            # Không phải thiết bị dùng pin (Desktop/Server)
            return 100.0, True

        return battery.percent, battery.power_plugged

    @classmethod
    def dieu_tiet_do_chinh_xac(cls) -> Dict[str, Any]:
        """
        Thuật toán Dynamic Precision Scaling.
        Quyết định độ phân giải của mô hình dựa trên pin.

        Returns:
            Dict chứa độ chính xác khuyên dùng và cấu hình.
        """
        percent, plugged = cls.get_battery_status()

        if plugged or percent > 60:
            logger.info(f"Năng lượng dồi dào (Pin {percent}%). Chọn 4-bit (Chất lượng cao).")
            return {
                "precision": "4-bit",
                "load_in_4bit": True,
                "load_in_8bit": False,
                "use_bitnet": False,
            }

        if 20 <= percent <= 60:
            logger.info(f"Pin trung bình (Pin {percent}%). Chọn 8-bit (Cân bằng).")
            return {
                "precision": "8-bit",
                "load_in_4bit": False,
                "load_in_8bit": True,
                "use_bitnet": False,
            }

        # Dưới 20%, không cắm sạc
        logger.warning(
            f"Pin yếu (Pin {percent}%). KÍCH HOẠT CHẾ ĐỘ SINH TỒN: 1.58-bit (Tiết kiệm tối đa)."
        )
        return {
            "precision": "1.58-bit",
            "load_in_4bit": False,
            "load_in_8bit": False,
            "use_bitnet": True,
        }
