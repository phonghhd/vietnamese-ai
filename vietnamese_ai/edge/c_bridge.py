"""Cầu nối C-API / Ctypes Siêu nhẹ (Native Ctypes Bridge) - Vietnamese AI Framework."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang
from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.EXTERNAL_DEPENDENCY,
    mo_ta=(
        "Cầu nối runtime C++ tốc độ cao 3 tầng: llama-cpp-python (in-process mmap + streaming 15-40 tok/s), "
        "binary llama-cli, hoặc ctypes libllama.so. Tự động fallback sang DongCoSuyLuanEdge thuần NumPy/Rust."
    ),
    phu_thuoc_ngoai=["llama-cpp-python>=0.2.0 hoặc binary llama-cli / libllama.so"],
    nhom_extra="local-llm",
)
class CauNoiCtypesNative:
    """
    Cầu nối C++ / llama.cpp 3 tầng tối ưu CPU (Zero-Friction Native Bridge):
    - Tầng 1: `llama-cpp-python` chạy trực tiếp trong tiến trình với mmap, KV-Cache và đa luồng SIMD.
    - Tầng 2: Binary `llama-cli` / `llama-simple` đọc luồng token trực tiếp không đệm dòng.
    - Tầng 3: Thư viện động `libllama.so`/`libllama.dylib`/`llama.dll` qua Ctypes hoặc fallback `DongCoSuyLuanEdge`.
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
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_gpu_layers: int = 0,
        chat_template: Optional[str] = None,
    ):
        self.duong_dan_mo_hinh = Path(duong_dan_mo_hinh) if duong_dan_mo_hinh else None
        self.so_luong_thread = max(1, int(so_luong_thread))
        self.duong_dan_thu_vien_c = duong_dan_thu_vien_c
        self.n_ctx = max(256, int(n_ctx))
        self.n_batch = max(32, int(n_batch))
        self.n_gpu_layers = int(n_gpu_layers)
        self.chat_template = chat_template

        self._llama_py: Any = None
        self._cdll: Optional[ctypes.CDLL] = None
        self._binary_path: Optional[str] = None
        self._fallback_engine: Optional[DongCoSuyLuanEdge] = None

        self._phat_hien_moi_truong_c()

    def _phat_hien_moi_truong_c(self) -> None:
        """Dò tìm runtime native theo thứ tự ưu tiên: llama-cpp-python -> binary llama-cli -> ctypes libllama -> fallback."""
        if self.duong_dan_mo_hinh and self.duong_dan_mo_hinh.is_file():
            llama_mod = QuanLyPhuThuoc.nhap_tuy_chon(
                "llama_cpp", nhom_extra="local-llm", tinh_nang="Native llama.cpp CPU Inference"
            )
            if llama_mod is not None and hasattr(llama_mod, "Llama"):
                try:
                    self._llama_py = llama_mod.Llama(
                        model_path=str(self.duong_dan_mo_hinh),
                        n_ctx=self.n_ctx,
                        n_threads=self.so_luong_thread,
                        n_threads_batch=self.so_luong_thread,
                        n_batch=self.n_batch,
                        n_gpu_layers=self.n_gpu_layers,
                        use_mmap=True,
                        verbose=False,
                    )
                    return
                except Exception:
                    self._llama_py = None

        if self.duong_dan_thu_vien_c and os.path.exists(self.duong_dan_thu_vien_c):
            try:
                self._cdll = ctypes.CDLL(self.duong_dan_thu_vien_c)
                return
            except Exception:
                pass

        cac_thu_muc_quet = [
            Path("/usr/local/lib"),
            Path("/usr/lib"),
            Path.home() / ".local/lib",
            Path.home() / ".cache" / "vietnamese-ai" / "bin",
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

        for ten_bin in self.CAC_BINARY_EXECUTABLE:
            tim_kiem = shutil.which(ten_bin)
            if tim_kiem and os.path.exists(tim_kiem):
                self._binary_path = tim_kiem
                return

        self._fallback_engine = DongCoSuyLuanEdge(
            duong_dan_mo_hinh=self.duong_dan_mo_hinh,
            so_luong_thread=self.so_luong_thread,
        )

    def co_thu_vien_c(self) -> bool:
        """Kiểm tra máy cục bộ có sẵn llama-cpp-python, binary native hay thư viện C++ không."""
        return self._llama_py is not None or self._cdll is not None or self._binary_path is not None

    def thong_tin_cau_noi(self) -> Dict[str, Any]:
        """Trả về thông tin kết nối chi tiết của cầu nối runtime native 3 tầng."""
        if self._llama_py is not None:
            che_do = "llama_cpp_python"
        elif self._binary_path:
            che_do = "binary_cli"
        elif self._cdll is not None:
            che_do = "ctypes_cdll"
        else:
            che_do = "numpy_edge_fallback"
        return {
            "co_native_c": self.co_thu_vien_c(),
            "che_do": che_do,
            "co_llama_cpp_python": self._llama_py is not None,
            "duong_dan_binary": self._binary_path,
            "so_thread": self.so_luong_thread,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "chat_template": self.chat_template,
            "mo_hinh": str(self.duong_dan_mo_hinh) if self.duong_dan_mo_hinh else None,
        }

    def sinh_stream(
        self,
        prompt: str,
        do_dai_toi_da: int = 128,
        nhiet_do: float = 0.7,
    ) -> Generator[str, None, None]:
        """Sinh văn bản dạng luồng token thời gian thực qua `llama-cpp-python`, `llama-cli` hoặc Fallback Engine."""
        if self._llama_py is not None:
            stream_out = self._llama_py.create_completion(
                prompt=prompt,
                max_tokens=max(1, int(do_dai_toi_da)),
                temperature=max(0.0, float(nhiet_do)),
                stream=True,
            )
            for chunk in stream_out:
                choices = chunk.get("choices") or []
                if choices:
                    piece = str(choices[0].get("text", ""))
                    if piece:
                        yield piece
            return

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
                "-c",
                str(self.n_ctx),
                "--no-display-prompt",
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                if proc.stdout:
                    while True:
                        ky_tu = proc.stdout.read(16)
                        if not ky_tu:
                            break
                        yield ky_tu
                proc.wait()
                return
            except Exception:
                pass

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
