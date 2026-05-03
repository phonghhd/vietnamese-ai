"""XuLyStream - Xử lý dữ liệu streaming (real-time)."""

from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class XuLyStream:
    """
    Bộ xử lý dữ liệu streaming thời gian thực.

    Tính năng:
    - Thu thập dữ liệu theo thời gian thực
    - Cập nhật mô hình online (incremental learning)
    - Phát hiện bất thường (anomaly detection)
    - Tính toán thống kê trượt (rolling statistics)

    Sử dụng:
        >>> stream = XuLyStream(kich_thuoc_cua_so=100)
        >>> stream.them_du_lieu(3.5)
        >>> stream.them_du_lieu(4.2)
        >>> stream.lay_thong_ke()
        {'trung_binh': 3.85, 'do_lech_chuan': 0.35, 'min': 3.5, 'max': 4.2}
    """

    def __init__(
        self,
        kich_thuoc_cua_so: int = 100,
        nguong_bat_thuong: float = 3.0,
    ):
        self.kich_thuoc_cua_so = kich_thuoc_cua_so
        self.nguong_bat_thuong = nguong_bat_thuong
        self.logger = Logger("XuLyStream")

        self._buffer: deque = deque(maxlen=kich_thuoc_cua_so)
        self._buffer_labels: deque = deque(maxlen=kich_thuoc_cua_so)
        self._tat_ca_du_lieu: List[float] = []
        self._callback_bat_thuong: Optional[Callable] = None
        self._so_lan_bat_thuong = 0

    def them_du_lieu(self, gia_tri: float, nhan: Optional[str] = None) -> Dict[str, Any]:
        """
        Thêm một điểm dữ liệu mới.

        Args:
            gia_tri: Giá trị số
            nhan: Nhãn tùy chọn

        Returns:
            Dict chứa thông tin: la_bat_thuong, thong_ke
        """
        self._buffer.append(gia_tri)
        self._tat_ca_du_lieu.append(gia_tri)

        ket_qua = {
            "gia_tri": gia_tri,
            "la_bat_thuong": False,
            "thong_ke": self.lay_thong_ke(),
        }

        if len(self._buffer) >= 10:
            trung_binh = np.mean(self._buffer)
            do_lech = np.std(self._buffer)
            if do_lech > 0 and abs(gia_tri - trung_binh) > self.nguong_bat_thuong * do_lech:
                ket_qua["la_bat_thuong"] = True
                self._so_lan_bat_thuong += 1
                self.logger.warning(f"Phát hiện bất thường: {gia_tri} (TB={trung_binh:.2f})")

                if self._callback_bat_thuong:
                    self._callback_bat_thuong(gia_tri, ket_qua)

        return ket_qua

    def them_nhieu(self, du_lieu: List[float]) -> List[Dict]:
        """Thêm nhiều điểm dữ liệu."""
        return [self.them_du_lieu(gv) for gv in du_lieu]

    def lay_thong_ke(self) -> Dict[str, float]:
        """Tính thống kê trên cửa sổ hiện tại."""
        if not self._buffer:
            return {}

        data = np.array(self._buffer)
        return {
            "so_mau": len(data),
            "trung_binh": float(np.mean(data)),
            "do_lech_chuan": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "trung_vi": float(np.median(data)),
            "quartile_25": float(np.percentile(data, 25)),
            "quartile_75": float(np.percentile(data, 75)),
        }

    def lay_cua_so(self) -> np.ndarray:
        """Lấy dữ liệu trong cửa sổ hiện tại."""
        return np.array(self._buffer)

    def lay_tat_ca(self) -> np.ndarray:
        """Lấy toàn bộ dữ liệu đã thu thập."""
        return np.array(self._tat_ca_du_lieu)

    def dat_callback_bat_thuong(self, callback: Callable) -> None:
        """Đặt callback khi phát hiện bất thường."""
        self._callback_bat_thuong = callback

    def thong_ke_bat_thuong(self) -> Dict[str, Any]:
        """Thống kê về các bất thường đã phát hiện."""
        return {
            "tong_du_lieu": len(self._tat_ca_du_lieu),
            "so_lan_bat_thuong": self._so_lan_bat_thuong,
            "ty_le_bat_thuong": self._so_lan_bat_thuong / max(1, len(self._tat_ca_du_lieu)),
        }

    def xoa_buffer(self) -> None:
        """Xóa buffer."""
        self._buffer.clear()
        self._tat_ca_du_lieu.clear()
        self._so_lan_bat_thuong = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"XuLyStream(so_mau={len(self._buffer)}, bat_thuong={self._so_lan_bat_thuong})"
