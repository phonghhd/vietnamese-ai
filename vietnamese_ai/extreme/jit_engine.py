"""EvoJITCompiler - Trình biên dịch C++ Just-In-Time phá vỡ GIL."""

import ctypes
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger("EvoJITCompiler")


class EvoJITCompiler:
    """
    Trình biên dịch C++ JIT cực đoan.
    Nhận mã nguồn C++, biên dịch nó thành file .so qua g++, và nạp vào bằng ctypes.
    Đảm bảo hàm C++ chạy hoàn toàn tách biệt khỏi khóa GIL của Python.
    """

    def __init__(self, use_openmp: bool = False):
        self.use_openmp = use_openmp
        self.temp_dir = tempfile.gettempdir()
        self.loaded_libs = {}

    def _write_cpp(self, name: str, code: str) -> str:
        """Lưu mã nguồn C++ vào ổ cứng."""
        file_path = os.path.join(self.temp_dir, f"{name}.cpp")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        return file_path

    def _compile(self, cpp_path: str, so_path: str) -> bool:
        """Gọi g++ để biên dịch với tối ưu hóa cao nhất (-O3)."""
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
        self, name: str, code: str, func_name: str, arg_types: list, restype: type
    ) -> callable:
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
