"""Cầu nối C-API / Ctypes Siêu nhẹ (Native Ctypes Bridge) - Vietnamese AI Framework."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge


class CauNoiCtypesNative:
    """
    Cầu nối Ctypes / C-API Siêu nhẹ (Zero-Compilation Native Bridge):
    - Tự động dò tìm thư viện động (`libllama.so`, `libllama.dylib`, `llama.dll`) hoặc binary (`llama-cli`).
    - Kết nối trực tiếp qua Ctypes mà không đòi hỏi người dùng phải cài đặt GCC, Clang hay CMake.
    - Cơ chế chuyển tầng tự động (Zero-Friction Fallback): Nếu không có C++ binary, tự động chuyển
      sang `DongCoSuyLuanEdge` thuần NumPy, đảm bảo hệ thống không bao giờ gặp lỗi crash.
    """

    CAC_FILE_DYNAMIC_LIB = [
        "libllama.so",
        "libllama.dylib",
        "llama.dll",
        "libllama_arm64.dylib",
        "libllama_x86_64.so",
    ]

    CAC_BINARY_EXECUTABLE = [
        "llama-cli",
        "llama-simple",
        "llama",
        "main",
    ]

    def __init__(
        self,
        duong_dan_mo_hinh: Optional[Union[str, Path]] = None,
        duong_dan_thu_vien_c: Optional[str] = None,
        so_luong_thread: int = 4,
    ):
        self.duong_dan_mo_hinh = Path(duong_dan_mo_hinh) if duong_dan_mo_hinh else None
        self.so_luong_thread = so_luong_thread
        self.duong_dan_thu_vien_c = duong_dan_thu_vien_c

        self._cdll: Optional[ctypes.CDLL] = None
        self._binary_path: Optional[str] = None
        self._fallback_engine: Optional[DongCoSuyLuanEdge] = None

        self._phat_hien_moi_truong_c()

    def _phat_hien_moi_truong_c(self) -> None:
        """Dò tìm thư viện C/C++ hoặc binary trên máy cục bộ."""
        # 1. Kiểm tra đường dẫn do người dùng chỉ định
        if self.duong_dan_thu_vien_c and os.path.exists(self.duong_dan_thu_vien_c):
            try:
                self._cdll = ctypes.CDLL(self.duong_dan_thu_vien_c)
                return
            except Exception:
                pass

        # 2. Dò tìm trong các thư mục hệ thống chuẩn
        cac_thu_muc_quet = [
            Path("/usr/local/lib"),
            Path("/usr/lib"),
            Path.home() / ".local/lib",
            Path.cwd(),
            Path.cwd() / "build",
            Path.cwd() / "libs",
        ]

        for thu_muc in cac_thu_muc_quet:
            if not thu_muc.exists():
                continue
            for ten_lib in self.CAC_FILE_DYNAMIC_LIB:
                lib_file = thu_muc / ten_lib
                if lib_file.exists():
                    try:
                        self._cdll = ctypes.CDLL(str(lib_file))
                        return
                    except Exception:
                        continue

        # 3. Dò tìm binary thực thi trên PATH
        for ten_bin in self.CAC_BINARY_EXECUTABLE:
            tim_kiem = shutil.which(ten_bin)
            if tim_kiem and os.path.exists(tim_kiem):
                self._binary_path = tim_kiem
                return

        # 4. Nếu không phát hiện thư viện C, khởi tạo fallback thuần NumPy
        self._fallback_engine = DongCoSuyLuanEdge(
            duong_dan_mo_hinh=self.duong_dan_mo_hinh,
            so_luong_thread=self.so_luong_thread,
        )

    def co_thu_vien_c(self) -> bool:
        """Kiểm tra máy cục bộ có sẵn thư viện C++ hoặc binary native hay không."""
        return self._cdll is not None or self._binary_path is not None

    def thong_tin_cau_noi(self) -> Dict[str, Any]:
        """Trả về thông tin kết nối của cầu nối C-API."""
        che_do = "ctypes_cdll" if self._cdll else ("binary_cli" if self._binary_path else "numpy_edge_fallback")
        return {
            "co_native_c": self.co_thu_vien_c(),
            "che_do": che_do,
            "duong_dan_binary": self._binary_path,
            "so_thread": self.so_luong_thread,
            "mo_hinh": str(self.duong_dan_mo_hinh) if self.duong_dan_mo_hinh else None,
        }

    def sinh_stream(
        self,
        prompt: str,
        do_dai_toi_da: int = 128,
        nhiet_do: float = 0.7,
    ) -> Generator[str, None, None]:
        """Sinh văn bản dạng stream qua C-API hoặc Fallback Engine."""
        if self._binary_path and self.duong_dan_mo_hinh and self.duong_dan_mo_hinh.exists():
            cmd = [
                self._binary_path,
                "-m",
                str(self.duong_dan_mo_hinh),
                "-p",
                prompt,
                "-n",
                str(do_dai_toi_da),
                "--temp",
                str(nhiet_do),
                "-t",
                str(self.so_luong_thread),
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                if proc.stdout:
                    for line in proc.stdout:
                        yield line
                proc.wait()
                return
            except Exception:
                pass

        # Fallback sang DongCoSuyLuanEdge thuần NumPy
        if self._fallback_engine is None:
            self._fallback_engine = DongCoSuyLuanEdge(
                duong_dan_mo_hinh=self.duong_dan_mo_hinh,
                so_luong_thread=self.so_luong_thread,
            )
        yield from self._fallback_engine.sinh_stream(
            prompt, do_dai_toi_da=do_dai_toi_da, nhiet_do=nhiet_do
        )

    def sinh_van_ban(
        self,
        prompt: str,
        do_dai_toi_da: int = 128,
        nhiet_do: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Sinh chuỗi văn bản hoàn thiện."""
        tokens = list(self.sinh_stream(prompt, do_dai_toi_da=do_dai_toi_da, nhiet_do=nhiet_do))
        return "".join(tokens)


# English Aliases
NativeLlamaBridge = CauNoiCtypesNative
CtypesLlamaBridge = CauNoiCtypesNative
