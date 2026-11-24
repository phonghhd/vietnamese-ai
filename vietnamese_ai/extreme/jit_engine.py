"""EvoJITCompiler - Trình biên dịch C++ Just-In-Time phá vỡ GIL."""

import ctypes
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional

from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

logger = logging.getLogger("EvoJITCompiler")


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.EXTERNAL_DEPENDENCY,
    mo_ta=(
        "Trình biên dịch C++ JIT thử nghiệm. KHÔNG phải Zero-Dependency: "
        "bắt buộc cài đặt trình biên dịch g++ (GCC/Clang) trên hệ điều hành."
    ),
    phu_thuoc_ngoai=["g++ (GNU C++ Compiler / build-essential)"],
)
class EvoJITCompiler:
    """
    Trình biên dịch C++ JIT cực đoan (Thử nghiệm — Yêu cầu công cụ ngoài `g++`).

    Nhận mã nguồn C++, biên dịch thành thư viện động `.so` qua `g++` và nạp bằng `ctypes`
    để chạy tách biệt khỏi GIL của Python.

    Yêu cầu bắt buộc: Máy chạy phải cài đặt sẵn `g++` trong PATH
    (ví dụ Ubuntu/Debian: `sudo apt install g++`). Tính năng này không hoạt động ở chế độ
    Zero-Dependency thuần Python.
    """

    def __init__(self, use_openmp: bool = False):
        self.use_openmp = use_openmp
        self.temp_dir = tempfile.gettempdir()
        self.loaded_libs: Dict[str, ctypes.CDLL] = {}

    def _write_cpp(self, name: str, code: str) -> str:
        """Lưu mã nguồn C++ vào ổ cứng."""
        file_path = os.path.join(self.temp_dir, f"{name}.cpp")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        return file_path

    def _compile(self, cpp_path: str, so_path: str) -> bool:
        """Gọi g++ để biên dịch với tối ưu hóa cao nhất (-O3)."""
        if shutil.which("g++") is None:
            logger.error(
                "Thiếu trình biên dịch C++ 'g++' trên hệ thống. "
                "EvoJITCompiler yêu cầu công cụ ngoài 'g++' (ví dụ: sudo apt install g++)."
            )
            return False
        cmd = ["g++", "-O3", "-shared", "-fPIC", cpp_path, "-o", so_path]
        if self.use_openmp:
            cmd.extend(["-fopenmp"])

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi biên dịch JIT:\n{e.stderr.decode('utf-8')}")
            return False

    def compile_and_load(
        self,
        name: str,
        code: str,
        func_name: str,
        arg_types: List[Any],
        restype: Optional[Any],
    ) -> Callable[..., Any]:
        """
        Toàn bộ luồng JIT: Sinh mã -> Compile -> Load Ctypes.
        """
        if name in self.loaded_libs:
            # Đã biên dịch trước đó, tải trực tiếp hàm ra
            func = getattr(self.loaded_libs[name], func_name)
            func.argtypes = arg_types
            func.restype = restype
            return func

        cpp_path = self._write_cpp(name, code)
        so_path = os.path.join(self.temp_dir, f"{name}.so")

        # Biên dịch
        if not self._compile(cpp_path, so_path):
            raise RuntimeError("Không thể biên dịch mã C++ JIT.")

        # Nạp thư viện
        try:
            lib = ctypes.cdll.LoadLibrary(so_path)
            self.loaded_libs[name] = lib

            func = getattr(lib, func_name)
            func.argtypes = arg_types
            func.restype = restype
            return func
        except Exception as e:
            logger.error(f"Lỗi nạp ctypes: {e}")
            raise
