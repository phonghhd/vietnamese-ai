"""MachCat - Circuit Breaker pattern cho production."""

import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TrangThaiMach(Enum):
    """Trạng thái của circuit breaker."""
    DONG = "dong"          # Closed - bình thường
    MO = "mo"              # Open - chặn request
    NUA_MO = "nua_mo"      # Half-open - thử nghiệm


class MachCat:
    """
    Circuit Breaker pattern - ngăn cascade failures.

    Hỗ trợ:
    - 3 trạng thái: Đóng (bình thường), Mở (chặn), Nửa mở (thử nghiệm)
    - Configurable failure threshold
    - Auto-recovery timeout
    - Fallback function
    - Failure tracking per endpoint

    Sử dụng:
        >>> circuit = MachCat(so_loi_toi_da=5, timeout_phuc_hoi=30)
        >>> with circuit:
        ...     ket_qua = goi_api()
    """

    def __init__(
        self,
        so_loi_toi_da: int = 5,
        timeout_phuc_hoi: float = 30.0,
        cua_so_thoi_gian: float = 60.0,
        ham_fallback: Optional[Callable] = None,
        ten: str = "default",
    ):
        self.so_loi_toi_da = so_loi_toi_da
        self.timeout_phuc_hoi = timeout_phuc_hoi
        self.cua_so_thoi_gian = cua_so_thoi_gian
        self.ham_fallback = ham_fallback
        self.ten = ten

        self._trang_thai = TrangThaiMach.DONG
        self._so_loi = 0
        self._so_thanh_cong = 0
        self._lan_loi_cuoi = 0.0
        self._lan_trang_thai_cuoi = time.time()
        self._lock = threading.Lock()

        self._thong_ke = {
            "tong_request": 0,
            "thanh_cong": 0,
            "that_bai": 0,
            "bi_chan": 0,
            "fallback": 0,
            "chuyen_trang_thai": 0,
        }
        self._lich_su_trang_thai: List[Dict[str, Any]] = []

    def __enter__(self) -> "MachCat":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.ghi_nhan_loi()
        else:
            self.ghi_nhan_thanh_cong()

    def cho_phep(self) -> bool:
        """Kiểm tra request có được phép không."""
        with self._lock:
            self._thong_ke["tong_request"] += 1

            if self._trang_thai == TrangThaiMach.DONG:
                return True

            if self._trang_thai == TrangThaiMach.MO:
                # Kiểm tra timeout
                if time.time() - self._lan_loi_cuoi >= self.timeout_phuc_hoi:
                    self._chuyen_trang_thai(TrangThaiMach.NUA_MO)
                    return True
                self._thong_ke["bi_chan"] += 1
                return False

            if self._trang_thai == TrangThaiMach.NUA_MO:
                return True

            return False

    def ghi_nhan_thanh_cong(self) -> None:
        """Ghi nhận request thành công."""
        with self._lock:
            self._so_thanh_cong += 1
            self._thong_ke["thanh_cong"] += 1

            if self._trang_thai == TrangThaiMach.NUA_MO:
                self._chuyen_trang_thai(TrangThaiMach.DONG)
                self._so_loi = 0

    def ghi_nhan_loi(self) -> None:
        """Ghi nhận request thất bại."""
        with self._lock:
            self._so_loi += 1
            self._thong_ke["that_bai"] += 1
            self._lan_loi_cuoi = time.time()

            if self._trang_thai == TrangThaiMach.DONG:
                if self._so_loi >= self.so_loi_toi_da:
                    self._chuyen_trang_thai(TrangThaiMach.MO)
            elif self._trang_thai == TrangThaiMach.NUA_MO:
                self._chuyen_trang_thai(TrangThaiMach.MO)

    def thuc_hien(self, ham: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Thực hiện hàm với circuit breaker protection.

        Args:
            ham: Hàm cần thực hiện
            *args, **kwargs: Tham số

        Returns:
            Kết quả của hàm hoặc fallback
        """
        if not self.cho_phep():
            if self.ham_fallback:
                self._thong_ke["fallback"] += 1
                return self.ham_fallback(*args, **kwargs)
            raise RuntimeError(
                f"Circuit breaker '{self.ten}' đang mở. "
                f"Thử lại sau {self.timeout_phuc_hoi}s."
            )

        try:
            ket_qua = ham(*args, **kwargs)
            self.ghi_nhan_thanh_cong()
            return ket_qua
        except Exception:
            self.ghi_nhan_loi()
            if self.ham_fallback:
                self._thong_ke["fallback"] += 1
                return self.ham_fallback(*args, **kwargs)
            raise

    def _chuyen_trang_thai(self, trang_thai_moi: TrangThaiMach) -> None:
        """Chuyển trạng thái."""
        cu = self._trang_thai
        self._trang_thai = trang_thai_moi
        self._lan_trang_thai_cuoi = time.time()
        self._thong_ke["chuyen_trang_thai"] += 1

        self._lich_su_trang_thai.append({
            "tu": cu.value,
            "sang": trang_thai_moi.value,
            "thoi_gian": time.time(),
        })

    def reset(self) -> None:
        """Reset circuit breaker về trạng thái đóng."""
        with self._lock:
            self._chuyen_trang_thai(TrangThaiMach.DONG)
            self._so_loi = 0
            self._so_thanh_cong = 0

    @property
    def trang_thai(self) -> str:
        """Trạng thái hiện tại."""
        return self._trang_thai.value

    def lay_thong_ke(self) -> Dict[str, Any]:
        """Lấy thống kê circuit breaker."""
        with self._lock:
            stats = self._thong_ke.copy()
            stats["trang_thai"] = self._trang_thai.value
            stats["so_loi_hien_tai"] = self._so_loi
            stats["so_loi_toi_da"] = self.so_loi_toi_da
            stats["timeout_phuc_hoi"] = self.timeout_phuc_hoi

            if stats["tong_request"] > 0:
                stats["ty_le_thanh_cong"] = stats["thanh_cong"] / stats["tong_request"]
                stats["ty_le_that_bai"] = stats["that_bai"] / stats["tong_request"]
            else:
                stats["ty_le_thanh_cong"] = 0.0
                stats["ty_le_that_bai"] = 0.0

            return stats

    def __repr__(self) -> str:
        return (
            f"MachCat(ten='{self.ten}', "
            f"trang_thai='{self._trang_thai.value}', "
            f"so_loi={self._so_loi}/{self.so_loi_toi_da})"
        )
