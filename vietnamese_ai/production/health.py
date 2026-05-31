"""KiemTraSucKhoe - Health check system cho production."""

import os
import time
from typing import Any, Callable, Dict, List

import numpy as np

from vietnamese_ai.utils.logger import Logger


class KiemTraSucKhoe:
    """
    Hệ thống health check cho production deployment.

    Hỗ trợ:
    - Model health check
    - System resource check (CPU, Memory, Disk)
    - Dependency check
    - Custom health checks
    - Readiness & Liveness probes

    Sử dụng:
        >>> health = KiemTraSucKhoe()
        >>> health.dang_ky_check("model", lambda: model is not None)
        >>> ket_qua = health.kiem_tra()
        >>> print(ket_qua["trang_thai"])  # "healthy" | "degraded" | "unhealthy"
    """

    def __init__(
        self,
        timeout: float = 5.0,
        mau_lich_su: int = 100,
    ):
        self.timeout = timeout
        self.mau_lich_su = mau_lich_su
        self.logger = Logger("KiemTraSucKhoe")

        self._checks: Dict[str, Dict[str, Any]] = {}
        self._lich_su: List[Dict[str, Any]] = []
        self._bat_dau = time.time()

    def dang_ky_check(
        self,
        ten: str,
        ham: Callable[[], bool],
        mo_ta: str = "",
        quan_trong: bool = True,
    ) -> None:
        """
        Đăng ký một health check.

        Args:
            ten: Tên check
            ham: Hàm check (trả về True nếu OK)
            mo_ta: Mô tả
            quan_trong: Nếu True, check này fail = unhealthy
        """
        self._checks[ten] = {
            "ham": ham,
            "mo_ta": mo_ta,
            "quan_trong": quan_trong,
        }

    def kiem_tra(self) -> Dict[str, Any]:
        """
        Chạy tất cả health checks.

        Returns:
            {trang_thai, chi_tiet, thoi_gian, uptime}
        """
        bat_dau = time.time()
        chi_tiet = {}
        tat_ca_ok = True
        co_loi_quan_trong = False

        for ten, check in self._checks.items():
            try:
                result = check["ham"]()
                chi_tiet[ten] = {
                    "trang_thai": "ok" if result else "fail",
                    "quan_trong": check["quan_trong"],
                    "mo_ta": check["mo_ta"],
                }
                if not result:
                    tat_ca_ok = False
                    if check["quan_trong"]:
                        co_loi_quan_trong = True
            except Exception as e:
                chi_tiet[ten] = {
                    "trang_thai": "error",
                    "loi": str(e),
                    "quan_trong": check["quan_trong"],
                    "mo_ta": check["mo_ta"],
                }
                tat_ca_ok = False
                if check["quan_trong"]:
                    co_loi_quan_trong = True

        # System checks
        system_info = self._kiem_tra_he_thong()

        # Determine overall status
        if co_loi_quan_trong:
            trang_thai = "unhealthy"
        elif not tat_ca_ok:
            trang_thai = "degraded"
        else:
            trang_thai = "healthy"

        thoi_gian = time.time() - bat_dau
        uptime = time.time() - self._bat_dau

        ket_qua = {
            "trang_thai": trang_thai,
            "chi_tiet": chi_tiet,
            "he_thong": system_info,
            "thoi_gian_ms": round(thoi_gian * 1000, 2),
            "uptime_s": round(uptime, 1),
            "timestamp": time.time(),
        }

        # Lưu lịch sử
        self._lich_su.append(
            {
                "trang_thai": trang_thai,
                "thoi_gian": thoi_gian,
                "timestamp": time.time(),
            }
        )
        if len(self._lich_su) > self.mau_lich_su:
            self._lich_su = self._lich_su[-self.mau_lich_su :]

        return ket_qua

    def ready(self) -> bool:
        """Readiness probe - service sẵn sàng nhận traffic chưa?"""
        ket_qua = self.kiem_tra()
        return ket_qua["trang_thai"] in ("healthy", "degraded")

    def live(self) -> bool:
        """Liveness probe - service còn sống không?"""
        try:
            return True
        except Exception:
            return False

    def _kiem_tra_he_thong(self) -> Dict[str, Any]:
        """Kiểm tra tài nguyên hệ thống."""
        info = {}

        # CPU usage
        try:
            load_avg = os.getloadavg()
            info["cpu_load_1m"] = load_avg[0]
            info["cpu_load_5m"] = load_avg[1]
            info["cpu_load_15m"] = load_avg[2]
        except (OSError, AttributeError):
            info["cpu_load"] = "N/A"

        # Memory
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        meminfo[key] = int(val)

                total = meminfo.get("MemTotal", 0)
                available = meminfo.get("MemAvailable", 0)
                if total > 0:
                    info["memory_total_mb"] = total // 1024
                    info["memory_available_mb"] = available // 1024
                    info["memory_usage_pct"] = round((total - available) / total * 100, 1)
        except (FileNotFoundError, PermissionError):
            info["memory"] = "N/A"

        # Disk
        try:
            st = os.statvfs("/")
            total = st.f_blocks * st.f_frsize
            free = st.f_bavail * st.f_frsize
            if total > 0:
                info["disk_total_gb"] = round(total / (1024**3), 1)
                info["disk_free_gb"] = round(free / (1024**3), 1)
                info["disk_usage_pct"] = round((total - free) / total * 100, 1)
        except (OSError, AttributeError):
            info["disk"] = "N/A"

        # Python info
        info["python_version"] = (
            f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        )
        info["pid"] = os.getpid()

        return info

    def dang_ky_checks_mac_dinh(self, model: Any = None) -> None:
        """Đăng ký các health checks mặc định."""
        if model is not None:
            self.dang_ky_check(
                "model_loaded",
                lambda: model is not None,
                "Kiểm tra model đã được load",
                quan_trong=True,
            )

            if hasattr(model, "du_doan"):

                def check_model():
                    try:
                        test_input = np.zeros((1, 1))
                        model.du_doan(test_input)
                        return True
                    except Exception:
                        return False

                self.dang_ky_check(
                    "model_inference",
                    check_model,
                    "Kiểm tra model có thể dự đoán",
                    quan_trong=True,
                )

        # Disk space check
        def check_disk():
            try:
                st = os.statvfs("/")
                free_pct = st.f_bavail / st.f_blocks
                return free_pct > 0.1  # > 10% free
            except Exception:
                return True

        self.dang_ky_check(
            "disk_space",
            check_disk,
            "Kiểm tra dung lượng ổ đĩa (>10% free)",
            quan_trong=False,
        )

    def lich_su(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử health checks."""
        return self._lich_su.copy()

    def ty_le_healthy(self) -> float:
        """Tỷ lệ healthy trong lịch sử."""
        if not self._lich_su:
            return 1.0
        healthy = sum(1 for h in self._lich_su if h["trang_thai"] == "healthy")
        return healthy / len(self._lich_su)

    def __repr__(self) -> str:
        return f"KiemTraSucKhoe(so_checks={len(self._checks)})"
