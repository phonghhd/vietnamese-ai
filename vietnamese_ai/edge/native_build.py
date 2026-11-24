"""Quản Lý Kiến Trúc CPU, SIMD & Cấu Hình Build C/C++ Native (`vietnamese_ai.edge.native_build`).

Phát hiện kiến trúc phần cứng (x86_64, ARM64/Apple Silicon), tập lệnh vector SIMD (AVX2, AVX-512, FMA, NEON),
trình biên dịch C/C++ (`g++`, `clang++`, `cmake`) và sinh cấu hình `setuptools` chuẩn hóa cho các thành phần
suy luận Edge native.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


def _chuan_hoa_kien_truc(machine: str) -> str:
    """Chuẩn hóa tên kiến trúc CPU về dạng thống nhất (`x86_64`, `arm64`, `riscv64`, ...)."""
    m = machine.strip().lower()
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if m in ("aarch64", "arm64", "armv8l", "armv8b"):
        return "arm64"
    if m in ("i386", "i686", "x86"):
        return "x86"
    return m or "unknown"


def _doc_co_simd_linux() -> List[str]:
    """Đọc tập lệnh SIMD CPU từ `/proc/cpuinfo` trên Linux."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return []
    try:
        noi_dung = cpuinfo.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return []
    flags: set[str] = set()
    for dong in noi_dung.splitlines():
        if dong.startswith("flags") or dong.startswith("features"):
            phan = dong.split(":", 1)
            if len(phan) == 2:
                flags.update(phan[1].split())
    tap_quan_tam = ("sse4_2", "avx", "avx2", "avx512f", "fma", "f16c", "neon", "asimd", "sve", "dotprod")
    return sorted(f for f in tap_quan_tam if f in flags)


def thong_tin_kien_truc_cpu() -> Dict[str, Any]:
    """Thu thập thông tin kiến trúc CPU, số lõi, tập lệnh SIMD và cờ biên dịch C/C++ khuyến nghị."""
    arch = _chuan_hoa_kien_truc(platform.machine())
    so_loi = os.cpu_count() or 1
    simd_flags: List[str] = []

    if sys.platform.startswith("linux"):
        simd_flags = _doc_co_simd_linux()
    elif sys.platform == "darwin" and arch == "arm64":
        simd_flags = ["neon", "asimd", "dotprod"]
    elif arch == "x86_64":
        simd_flags = ["sse4_2"]

    co_bien_dich: List[str] = ["-O3"]
    if arch == "x86_64":
        if "avx512f" in simd_flags:
            co_bien_dich.extend(["-mavx512f", "-mfma"])
        elif "avx2" in simd_flags:
            co_bien_dich.extend(["-mavx2", "-mfma"])
        else:
            co_bien_dich.append("-msse4.2")
    elif arch == "arm64":
        co_bien_dich.append("-mcpu=native" if sys.platform.startswith("linux") else "-march=armv8-a+simd")

    return {
        "he_dieu_hanh": sys.platform,
        "kien_truc": arch,
        "kien_truc_goc": platform.machine(),
        "so_loi_logic": so_loi,
        "tap_lenh_simd": simd_flags,
        "ho_tro_avx2": "avx2" in simd_flags,
        "ho_tro_avx512": "avx512f" in simd_flags,
        "ho_tro_neon": ("neon" in simd_flags) or ("asimd" in simd_flags),
        "co_bien_dich_khuyen_nghi": co_bien_dich,
    }


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.EXTERNAL_DEPENDENCY,
    mo_ta="Quản lý toolchain C/C++ (g++/clang++/cmake) và cấu hình biên dịch setuptools cho kernel Edge native.",
    phu_thuoc_ngoai=["g++ hoặc clang++", "cmake (tùy chọn cho libllama)"],
    nhom_extra="core",
)
class KiemTraMoiTruongNative:
    """Kiểm tra toolchain biên dịch C/C++/Rust và sinh cấu hình setuptools cho `vietnamese_ai.edge`."""

    @staticmethod
    def kiem_tra_toolchain() -> Dict[str, Any]:
        """Quét hệ thống để xác định các trình biên dịch C/C++/Rust và thư viện native sẵn có."""
        cpu = thong_tin_kien_truc_cpu()
        trinh_bien_dich = {
            "g++": shutil.which("g++"),
            "gcc": shutil.which("gcc"),
            "clang++": shutil.which("clang++"),
            "cmake": shutil.which("cmake"),
            "cargo": shutil.which("cargo"),
            "llama_cli": shutil.which("llama-cli") or shutil.which("llama-server"),
        }
        co_cpp = bool(trinh_bien_dich["g++"] or trinh_bien_dich["clang++"])
        return {
            "cpu": cpu,
            "trinh_bien_dich": trinh_bien_dich,
            "san_sang_bien_dich_cpp": co_cpp,
            "san_sang_rust": bool(trinh_bien_dich["cargo"]),
            "san_sang_llama_cpp": bool(trinh_bien_dich["llama_cli"]),
        }

    @staticmethod
    def cau_hinh_setuptools_extension(
        ten_extension: str,
        cac_tep_nguon: List[str],
        bat_openmp: bool = True,
        duong_dan_include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Sinh từ điển tham số chuẩn để khởi tạo `setuptools.Extension` tối ưu theo kiến trúc CPU hiện tại."""
        cpu = thong_tin_kien_truc_cpu()
        extra_compile_args = list(cpu["co_bien_dich_khuyen_nghi"])
        extra_link_args: List[str] = []

        if sys.platform != "win32":
            extra_compile_args.append("-std=c++17")
            if bat_openmp and sys.platform != "darwin":
                extra_compile_args.append("-fopenmp")
                extra_link_args.append("-fopenmp")
        else:
            extra_compile_args = ["/O2", "/std:c++17"]
            if bat_openmp:
                extra_compile_args.append("/openmp")

        return {
            "name": ten_extension,
            "sources": list(cac_tep_nguon),
            "include_dirs": list(duong_dan_include or []),
            "extra_compile_args": extra_compile_args,
            "extra_link_args": extra_link_args,
            "language": "c++",
            "kien_truc_muc_tieu": cpu["kien_truc"],
        }


NativeBuildManager = KiemTraMoiTruongNative

__all__ = [
    "thong_tin_kien_truc_cpu",
    "KiemTraMoiTruongNative",
    "NativeBuildManager",
]
