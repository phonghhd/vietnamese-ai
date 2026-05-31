"""
TheoDoiGPU (GPU Monitor) - Giao tiếp với NVML qua ctypes để theo dõi thông số GPU
không cần cài đặt thêm thư viện pynvml.
"""

import ctypes
import logging
import os
import sys
from typing import Any, Dict, List

logger = logging.getLogger("V-Monitor")


# Định nghĩa Struct cho Memory Info theo chuẩn NVML
class nvmlMemory_t(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


# Hằng số
NVML_TEMPERATURE_GPU = 0
NVML_SUCCESS = 0


class TheoDoiGPU:
    """
    Theo dõi thông số GPU (VRAM, Nhiệt độ, Công suất) trực tiếp qua thư viện NVML.
    Tương thích Linux/Windows. Zero-dependency.
    """

    def __init__(self):
        self._nvml_lib = None
        self._da_khoi_tao = False
        self._tai_thu_vien_nvml()

    def _tai_thu_vien_nvml(self):
        """Tải thư viện NVML dựa trên OS."""
        try:
            if sys.platform.startswith("win"):
                nvml_path = os.path.join(
                    os.environ.get("ProgramW6432", "C:\\Program Files"),
                    "NVIDIA Corporation",
                    "NVSMI",
                    "nvml.dll",
                )
                if not os.path.exists(nvml_path):
                    nvml_path = "nvml.dll"  # Thử tìm trong PATH
                self._nvml_lib = ctypes.CDLL(nvml_path)
            else:
                self._nvml_lib = ctypes.CDLL("libnvidia-ml.so.1")

            # Khởi tạo NVML
            ret = self._nvml_lib.nvmlInit_v2()
            if ret == NVML_SUCCESS:
                self._da_khoi_tao = True
                logger.info("[TheoDoiGPU] NVML đã khởi tạo thành công.")
            else:
                logger.warning(f"[TheoDoiGPU] Không thể khởi tạo NVML (Mã lỗi: {ret}).")

        except Exception as e:
            logger.warning(
                f"[TheoDoiGPU] Không tìm thấy thư viện NVML hoặc máy không có GPU NVIDIA. Lỗi: {e}"
            )

    def lay_thong_tin_tat_ca_gpu(self) -> List[Dict[str, Any]]:
        """Lấy thông tin VRAM, Nhiệt độ, Công suất của toàn bộ GPU."""
        if not self._da_khoi_tao or self._nvml_lib is None:
            return []  # Trả về rỗng nếu không có GPU

        danh_sach_gpu = []
        count = ctypes.c_uint()
        ret = self._nvml_lib.nvmlDeviceGetCount_v2(ctypes.byref(count))

        if ret != NVML_SUCCESS:
            return []

        for i in range(count.value):
            thong_tin = self._lay_thong_tin_mot_gpu(i)
            if thong_tin:
                danh_sach_gpu.append(thong_tin)

        return danh_sach_gpu

    def _lay_thong_tin_mot_gpu(self, index: int) -> Dict[str, Any]:
        """Đọc chỉ số của một GPU cụ thể qua index."""
        handle = ctypes.c_void_p()
        ret = self._nvml_lib.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(index), ctypes.byref(handle)
        )
        if ret != NVML_SUCCESS:
            return {}

        # 1. Lấy tên
        name_buffer = ctypes.create_string_buffer(96)
        self._nvml_lib.nvmlDeviceGetName(handle, name_buffer, ctypes.c_uint(96))
        ten_gpu = name_buffer.value.decode("utf-8")

        # 2. Lấy bộ nhớ
        mem_info = nvmlMemory_t()
        self._nvml_lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem_info))

        # 3. Lấy nhiệt độ
        nhiet_do = ctypes.c_uint()
        self._nvml_lib.nvmlDeviceGetTemperature(
            handle, ctypes.c_uint(NVML_TEMPERATURE_GPU), ctypes.byref(nhiet_do)
        )

        # 4. Lấy công suất (milliWatts -> Watts)
        cong_suat_mw = ctypes.c_uint()
        ret_power = self._nvml_lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(cong_suat_mw))
        cong_suat_w = cong_suat_mw.value / 1000.0 if ret_power == NVML_SUCCESS else 0.0

        return {
            "id": index,
            "ten": ten_gpu,
            "vram_tong_mb": mem_info.total // (1024 * 1024),
            "vram_su_dung_mb": mem_info.used // (1024 * 1024),
            "nhiet_do_c": nhiet_do.value,
            "cong_suat_w": cong_suat_w,
        }

    def tat(self):
        """Đóng kết nối NVML."""
        if self._da_khoi_tao and self._nvml_lib:
            self._nvml_lib.nvmlShutdown()
            self._da_khoi_tao = False

    def __del__(self):
        self.tat()
