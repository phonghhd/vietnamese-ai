"""
ThucThiDocLap (Isolated Executor) - Lớp khiên phòng thủ động.
Đưa mã nguồn vào một Process riêng biệt với không gian biến toàn cục (globals)
trống rỗng và đặt giới hạn thời gian chạy (Timeout).
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import resource as _resource

    _CO_RESOURCE = True
except ImportError:
    _resource = None  # type: ignore[assignment]
    _CO_RESOURCE = False

from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

from .ast_analyzer import LoiAnNinh, PhanTichAST

logger = logging.getLogger("V-Sandbox")


def _ap_dung_gioi_han_tai_nguyen_os(timeout_giay: int, gioi_han_ram_mb: int) -> None:
    """Áp đặt giới hạn tài nguyên mức hạt nhân POSIX (CPU, bộ nhớ ảo, kích thước tệp, số file descriptor)."""
    if not _CO_RESOURCE or _resource is None:
        return
    try:
        cpu_soft = max(1, int(timeout_giay) + 1)
        _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1))
    except (ValueError, OSError):
        pass
    try:
        _resource.setrlimit(_resource.RLIMIT_FSIZE, (0, 0))
    except (ValueError, OSError):
        pass
    try:
        cur_soft, cur_hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
        muc_fd = min(cur_soft, 64) if cur_soft > 0 else 64
        muc_fd = max(muc_fd, 32)
        if cur_hard == _resource.RLIM_INFINITY or muc_fd <= cur_hard:
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (muc_fd, cur_hard))
    except (ValueError, OSError):
        pass
    if gioi_han_ram_mb > 0 and hasattr(_resource, "RLIMIT_AS"):
        try:
            vm_hien_tai = 0
            try:
                with open("/proc/self/statm", encoding="utf-8") as f:
                    pages = int(f.read().split()[0])
                vm_hien_tai = pages * os.sysconf("SC_PAGE_SIZE")
            except Exception:
                vm_hien_tai = 512 * 1024 * 1024
            gioi_han_bytes = vm_hien_tai + max(32, int(gioi_han_ram_mb)) * 1024 * 1024
            _, cur_hard_as = _resource.getrlimit(_resource.RLIMIT_AS)
            if cur_hard_as != _resource.RLIM_INFINITY:
                gioi_han_bytes = min(gioi_han_bytes, cur_hard_as)
            _resource.setrlimit(_resource.RLIMIT_AS, (gioi_han_bytes, cur_hard_as))
        except (ValueError, OSError):
            pass


_ALLOWED_SANDBOX_MODULES = {
    "math",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "decimal",
    "fractions",
    "re",
    "json",
    "datetime",
}


def _safe_sandbox_import(
    name: str,
    globals_: Any = None,
    locals_: Any = None,
    fromlist: Tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Chỉ cho phép nhập các module chuẩn toán học/xử lý chuỗi thuần túy trong danh sách trắng."""
    root = name.split(".")[0]
    if root not in _ALLOWED_SANDBOX_MODULES:
        raise ImportError(f"Module không được phép trong Sandbox: '{name}'")
    return __import__(name, globals_, locals_, fromlist, level)


def _tien_trinh_thuc_thi(
    ma_nguon: str,
    ket_qua_queue: multiprocessing.Queue,
    timeout_giay: int = 5,
    gioi_han_ram_mb: int = 256,
):
    """
    Hàm được chạy trong một Process riêng kèm giới hạn tài nguyên POSIX (`setrlimit`).
    Không gian tên ở đây hoàn toàn trống rỗng để tránh mã độc leo thang.
    """
    _ap_dung_gioi_han_tai_nguyen_os(timeout_giay=timeout_giay, gioi_han_ram_mb=gioi_han_ram_mb)

    buf_stdout = io.StringIO()

    safe_builtins = {
        "print": print,
        "range": range,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "frozenset": frozenset,
        "tuple": tuple,
        "sum": sum,
        "max": max,
        "min": min,
        "abs": abs,
        "round": round,
        "pow": pow,
        "all": all,
        "any": any,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "reversed": reversed,
        "isinstance": isinstance,
        "callable": callable,
        "slice": slice,
        "map": map,
        "filter": filter,
        "__import__": _safe_sandbox_import,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "ZeroDivisionError": ZeroDivisionError,
        "AssertionError": AssertionError,
        "MemoryError": MemoryError,
    }

    safe_globals = {"__builtins__": safe_builtins}

    try:
        with contextlib.redirect_stdout(buf_stdout):
            exec(ma_nguon, safe_globals)

        ket_qua = {}
        for k, v in safe_globals.items():
            if k != "__builtins__" and not callable(v):
                if isinstance(v, (int, float, str, bool, list, dict, set, tuple)):
                    ket_qua[k] = v

        ket_qua_queue.put(("THANH_CONG", ket_qua, buf_stdout.getvalue()))

    except MemoryError:
        ket_qua_queue.put(
            ("LOI", f"Vượt giới hạn bộ nhớ Sandbox ({gioi_han_ram_mb} MB)", buf_stdout.getvalue())
        )
    except Exception as e:
        ket_qua_queue.put(("LOI", str(e), buf_stdout.getvalue()))


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.EXPERIMENTAL,
    mo_ta=(
        "V-Sandbox tách tiến trình (PhanTichAST + multiprocessing spawn + POSIX resource.setrlimit CPU/RAM/FSIZE/NOFILE "
        "+ tùy chọn bubblewrap --unshare-net trên Linux)."
    ),
)
class ThucThiDocLap:
    """
    Môi trường hộp cát (V-Sandbox) thực thi mã Python tách tiến trình kèm giới hạn tài nguyên cấp HĐH.

    Hỗ trợ giới hạn `RLIMIT_CPU`, `RLIMIT_AS` (RAM), `RLIMIT_FSIZE=0` (chặn ghi tệp), `RLIMIT_NOFILE`
    trên hệ thống POSIX và tùy chọn cô lập namespace mạng qua `bubblewrap` (`bwrap --unshare-net`).
    """

    def __init__(
        self,
        timeout_giay: int = 5,
        gioi_han_ram_mb: int = 256,
        su_dung_bwrap: bool = False,
    ):
        """
        Khởi tạo Sandbox.

        Args:
            timeout_giay: Thời gian sống tối đa của tiến trình thực thi (mặc định 5s).
            gioi_han_ram_mb: Hạn mức RAM bổ sung tối đa cho tiến trình con (MB, mặc định 256MB).
            su_dung_bwrap: Nếu True và máy chủ có cài `bubblewrap`, cách ly thêm namespace mạng `--unshare-net`.
        """
        self.timeout_giay = timeout_giay
        self.gioi_han_ram_mb = max(32, int(gioi_han_ram_mb))
        self.su_dung_bwrap = bool(su_dung_bwrap)

    @staticmethod
    def co_bubblewrap() -> bool:
        """Kiểm tra hệ thống có sẵn công cụ cách ly Linux `bubblewrap` (`bwrap`) hay không."""
        return shutil.which("bwrap") is not None

    def thong_ke_cach_ly(self) -> Dict[str, Any]:
        """Trả về thông số các lớp phòng thủ HĐH đang được kích hoạt trong Sandbox."""
        return {
            "timeout_giay": self.timeout_giay,
            "gioi_han_ram_mb": self.gioi_han_ram_mb,
            "co_rlimit_posix": _CO_RESOURCE,
            "co_bubblewrap": self.co_bubblewrap(),
            "su_dung_bwrap": self.su_dung_bwrap and self.co_bubblewrap(),
        }

    def _chay_qua_bwrap(self, ma_nguon: str) -> Tuple[bool, Any]:
        """Chạy mã Python trong container nhẹ `bubblewrap` không mạng (`--unshare-net`) và hệ thống tệp chỉ đọc."""
        runner = (
            "import json, sys\n"
            "safe_b = {'print': print, 'range': range, 'len': len, 'int': int, 'float': float, "
            "'str': str, 'bool': bool, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple, "
            "'sum': sum, 'max': max, 'min': min, 'abs': abs, 'round': round, 'map': map, 'filter': filter, "
            "'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError, 'MemoryError': MemoryError}\n"
            "g = {'__builtins__': safe_b}\n"
            "code = sys.stdin.read()\n"
            "try:\n"
            "    exec(code, g)\n"
            "    out = {k: v for k, v in g.items() if k != '__builtins__' and not callable(v) "
            "and isinstance(v, (int, float, str, bool, list, dict))}\n"
            "    print(json.dumps({'ok': True, 'data': out}, ensure_ascii=False))\n"
            "except Exception as exc:\n"
            "    print(json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False))\n"
        )
        cmd = [
            "bwrap",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--die-with-parent",
            sys.executable,
            "-I",
            "-c",
            runner,
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=ma_nguon,
                capture_output=True,
                text=True,
                timeout=self.timeout_giay,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "Lỗi Timeout: Mã chạy quá thời gian cho phép hoặc dính vòng lặp vô tận."
        except OSError as exc:
            return False, f"Lỗi khởi chạy bubblewrap: {exc}"

        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            return False, f"Tiến trình bubblewrap bị lỗi (Mã thoát: {proc.returncode})"
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            return False, f"Phản hồi bubblewrap không hợp lệ: {lines[-1][:160]}"
        if payload.get("ok"):
            return True, payload.get("data", {})
        return False, f"Lỗi Thực Thi: {payload.get('error', 'Không xác định')}"

    def chay_kem_stdout(
        self,
        ma_nguon: str,
        module_cam_bo_sung: Optional[Iterable[str]] = None,
        ham_cam_bo_sung: Optional[Iterable[str]] = None,
    ) -> Tuple[bool, Any, str]:
        """Kiểm tra AST tĩnh và thực thi mã trong tiến trình tách biệt kèm thu nhận stdout."""
        try:
            PhanTichAST.kiem_tra(
                ma_nguon,
                module_cam_bo_sung=module_cam_bo_sung,
                ham_cam_bo_sung=ham_cam_bo_sung,
            )
        except LoiAnNinh as e:
            logger.warning(f"[Sandbox] Đã chặn mã độc: {e}")
            return False, f"Lỗi An Ninh: {e}", ""
        except SyntaxError as e:
            return False, f"Lỗi Cú Pháp: {e}", ""

        if self.su_dung_bwrap and self.co_bubblewrap():
            ok, data = self._chay_qua_bwrap(ma_nguon)
            return ok, data, ""

        ctx = multiprocessing.get_context("spawn")
        hang_doi = ctx.Queue()

        tien_trinh = ctx.Process(
            target=_tien_trinh_thuc_thi,
            args=(ma_nguon, hang_doi, self.timeout_giay, self.gioi_han_ram_mb),
        )
        tien_trinh.daemon = True

        tien_trinh.start()
        tien_trinh.join(timeout=self.timeout_giay)

        if tien_trinh.is_alive():
            logger.error(f"[Sandbox] Đã tiêu diệt Process (Vượt quá {self.timeout_giay}s)")
            tien_trinh.terminate()
            tien_trinh.join()
            return False, "Lỗi Timeout: Mã chạy quá thời gian cho phép hoặc dính vòng lặp vô tận.", ""

        try:
            trang_thai, du_lieu, stdout_text = hang_doi.get_nowait()
            if trang_thai == "THANH_CONG":
                return True, du_lieu, stdout_text
            return False, f"Lỗi Thực Thi: {du_lieu}", stdout_text
        except queue.Empty:
            if tien_trinh.exitcode != 0:
                return False, f"Tiến trình bị lỗi (Mã thoát: {tien_trinh.exitcode})", ""
            return False, "Không nhận được kết quả (Tiến trình bị ngắt đột ngột).", ""

    def chay(self, ma_nguon: str) -> Tuple[bool, Any]:
        """
        Kiểm tra AST tĩnh và sau đó thực thi động trong tiến trình bị giới hạn tài nguyên HĐH.

        Args:
            ma_nguon: Mã Python.

        Returns:
            Tuple (Trạng_thái_thành_công_boolean, Kết_quả_chạy_hoặc_Lỗi_chuỗi).
        """
        ok, du_lieu, _ = self.chay_kem_stdout(ma_nguon)
        return ok, du_lieu
