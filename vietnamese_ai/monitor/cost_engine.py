"""
MayTinhTien (Cost Engine) - Tính toán chi phí tiêu thụ điện hoặc phí thuê GPU theo thời gian.
"""

import time
from typing import Dict, Any


class MayTinhTien:
    """
    Công cụ tính toán chi phí (Cost Engine) cho V-Monitor.
    Tính tiền điện dựa trên số Watt tiêu thụ, hoặc tính tiền thuê GPU theo giờ.
    """

    def __init__(self, gia_dien_kwh: float = 3000.0, gia_thue_gpu_gio: float = 0.0):
        """
        Khởi tạo máy tính tiền.

        Args:
            gia_dien_kwh: Giá tiền điện cho 1 kWh (Mặc định: 3000 VNĐ).
            gia_thue_gpu_gio: Giá thuê GPU trong 1 giờ (Mặc định: 0 VNĐ - nếu tự host).
        """
        self.gia_dien_kwh = gia_dien_kwh
        self.gia_thue_gpu_gio = gia_thue_gpu_gio
        
        self._tong_dien_nang_w_s = 0.0 # Tổng công suất tích lũy (Watt-giây)
        self._thoi_gian_bat_dau = time.time()
        self._thoi_gian_cap_nhat_cuoi = time.time()

    def cap_nhat_muc_tieu_thu(self, cong_suat_w: float):
        """
        Cập nhật mức tiêu thụ điện năng hiện tại (Watt).
        Hàm này nên được gọi định kỳ (ví dụ: mỗi giây).
        """
        thoi_gian_hien_tai = time.time()
        khoang_thoi_gian = thoi_gian_hien_tai - self._thoi_gian_cap_nhat_cuoi
        
        # Công suất (W) * thời gian (s) = W.s (Joules)
        self._tong_dien_nang_w_s += cong_suat_w * khoang_thoi_gian
        self._thoi_gian_cap_nhat_cuoi = thoi_gian_hien_tai

    def lay_bao_cao_chi_phi(self) -> Dict[str, Any]:
        """
        Tính toán và trả về báo cáo chi phí tổng thể.
        
        Returns:
            Dict chứa chi tiết về tiền điện, tiền thuê GPU và tổng tiền.
        """
        # Cập nhật thời gian lần cuối cho chính xác
        thoi_gian_hien_tai = time.time()
        thoi_gian_chay_giay = thoi_gian_hien_tai - self._thoi_gian_bat_dau
        
        # Tính tiền điện
        # 1 kWh = 1000 W * 3600 s = 3,600,000 W.s
        tong_kwh = self._tong_dien_nang_w_s / 3600000.0
        tien_dien = tong_kwh * self.gia_dien_kwh
        
        # Tính tiền thuê GPU
        thoi_gian_chay_gio = thoi_gian_chay_giay / 3600.0
        tien_thue = thoi_gian_chay_gio * self.gia_thue_gpu_gio
        
        return {
            "thoi_gian_chay_giay": round(thoi_gian_chay_giay, 2),
            "tong_kwh": round(tong_kwh, 4),
            "chi_phi_dien_vnd": round(tien_dien, 2),
            "chi_phi_thue_vnd": round(tien_thue, 2),
            "tong_chi_phi_vnd": round(tien_dien + tien_thue, 2)
        }

    def reset(self):
        """Đặt lại bộ đếm về 0."""
        self._tong_dien_nang_w_s = 0.0
        self._thoi_gian_bat_dau = time.time()
        self._thoi_gian_cap_nhat_cuoi = time.time()
