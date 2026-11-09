"""
Bộ phát hiện tiếng nói (Voice Activity Detection - VAD) cho tiếng Việt.
Hoạt động thuần NumPy và thư viện chuẩn, tuân thủ Triết lý Zero-Dependency.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TrangThaiVAD(enum.Enum):
    """Trạng thái phân loại âm thanh trong dòng phát hiện giọng nói."""

    IM_LANG = "im_lang"
    BAT_DAU_GIONG_NOI = "bat_dau_giong_noi"
    DANG_NOI = "dang_noi"
    KET_THUC_GIONG_NOI = "ket_thuc_giong_noi"


class PhatHienGiongNoiVAD:
    """
    Bộ phát hiện hoạt động giọng nói (VAD) tối ưu hóa thời gian thực.

    Sử dụng kết hợp:
    1. Năng lượng ngắn hạn (Short-Time Energy - STE) có ngưỡng tương thích (Adaptive Threshold).
    2. Tỷ lệ đảo dấu (Zero Crossing Rate - ZCR) để nhận diện phụ âm và thanh điệu tiếng Việt.
    3. Bộ máy trạng thái (State Machine) lọc nhiễu nháy để phát hiện chính xác điểm bắt đầu
       và điểm kết thúc phát ngôn của người nói.
    """

    def __init__(
        self,
        tan_so_lay_mau: int = 16000,
        kich_thuoc_khung_ms: float = 20.0,
        nguong_nang_luong_khoi_tao: float = 0.015,
        nguong_zcr_toi_thieu: float = 0.02,
        nguong_zcr_toi_da: float = 0.65,
        thoi_gian_im_lang_ngat_cau_ms: float = 600.0,
        thoi_gian_noi_toi_thieu_ms: float = 120.0,
        tu_dong_thich_nghi_tieng_on: bool = True,
    ) -> None:
        """
        Khởi tạo bộ phát hiện tiếng nói VAD.

        Args:
            tan_so_lay_mau: Tần số lấy mẫu âm thanh (mặc định 16000 Hz).
            kich_thuoc_khung_ms: Kích thước mỗi khung tính toán theo mili-giây (mặc định 20ms).
            nguong_nang_luong_khoi_tao: Ngưỡng năng lượng ngắn hạn ban đầu.
            nguong_zcr_toi_thieu: Ngưỡng tỷ lệ qua điểm 0 tối thiểu.
            nguong_zcr_toi_da: Ngưỡng tỷ lệ qua điểm 0 tối đa.
            thoi_gian_im_lang_ngat_cau_ms: Thời gian im lặng tối thiểu để chốt câu (ms).
            thoi_gian_noi_toi_thieu_ms: Thời gian phát ngôn tối thiểu để không bị coi là tiếng lách cách (ms).
            tu_dong_thich_nghi_tieng_on: Tự động cập nhật mức ồn nền từ các khung im lặng.
        """
        self.tan_so_lay_mau = tan_so_lay_mau
        self.kich_thuoc_khung_ms = kich_thuoc_khung_ms
        self.so_mau_khung = max(16, int(round(tan_so_lay_mau * (kich_thuoc_khung_ms / 1000.0))))
        self.nguong_nang_luong = float(nguong_nang_luong_khoi_tao)
        self.nguong_zcr_min = float(nguong_zcr_toi_thieu)
        self.nguong_zcr_max = float(nguong_zcr_toi_da)
        self.so_khung_im_lang_ngat_cau = max(
            1, int(round(thoi_gian_im_lang_ngat_cau_ms / kich_thuoc_khung_ms))
        )
        self.so_khung_noi_toi_thieu = max(
            1, int(round(thoi_gian_noi_toi_thieu_ms / kich_thuoc_khung_ms))
        )
        self.tu_dong_thich_nghi = tu_dong_thich_nghi_tieng_on

        # Trạng thái nội bộ
        self._trang_thai_hien_tai = TrangThaiVAD.IM_LANG
        self._dem_khung_noi = 0
        self._dem_khung_im_lang = 0
        self._muc_on_nen = float(nguong_nang_luong_khoi_tao) * 0.5
        self._he_so_hoc_on = 0.05

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái nội bộ về ban đầu."""
        self._trang_thai_hien_tai = TrangThaiVAD.IM_LANG
        self._dem_khung_noi = 0
        self._dem_khung_im_lang = 0

    @staticmethod
    def tinh_nang_luong(khung: np.ndarray) -> float:
        """Tính năng lượng hiệu dụng (RMS energy) của khung âm thanh."""
        if len(khung) == 0:
            return 0.0
        return float(np.sqrt(np.mean(khung.astype(np.float32) ** 2)))

    @staticmethod
    def tinh_zcr(khung: np.ndarray) -> float:
        """Tính tỷ lệ đảo dấu (Zero Crossing Rate) của khung âm thanh."""
        if len(khung) < 2:
            return 0.0
        dau = np.sign(khung)
        dau[dau == 0] = 1
        return float(np.mean(np.abs(dau[1:] - dau[:-1]) > 0))

    def xu_ly_khung(self, khung_tin_hieu: np.ndarray) -> Dict[str, Any]:
        """
        Xử lý một khung âm thanh đơn lẻ và cập nhật trạng thái giọng nói.

        Args:
            khung_tin_hieu: Mảng NumPy chứa các mẫu âm thanh của một khung (thường 20ms - 320 mẫu).

        Returns:
            Dict chứa trạng thái `trang_thai`, cờ `co_giong_noi`, `nang_luong`, và `zcr`.
        """
        tin_hieu = np.asarray(khung_tin_hieu, dtype=np.float32).flatten()
        nang_luong = self.tinh_nang_luong(tin_hieu)
        zcr = self.tinh_zcr(tin_hieu)

        # Điều kiện phát hiện khung có giọng nói:
        # Năng lượng vượt ngưỡng thích nghi VÀ ZCR nằm trong dải âm vị tự nhiên
        khung_la_giong_noi = (
            nang_luong > max(self.nguong_nang_luong, self._muc_on_nen * 1.8)
        ) and (self.nguong_zcr_min <= zcr <= self.nguong_zcr_max)

        trang_thai_truoc = self._trang_thai_hien_tai
        su_kien = TrangThaiVAD.IM_LANG

        if khung_la_giong_noi:
            self._dem_khung_noi += 1
            self._dem_khung_im_lang = 0

            if self._trang_thai_hien_tai == TrangThaiVAD.IM_LANG:
                if self._dem_khung_noi >= self.so_khung_noi_toi_thieu:
                    self._trang_thai_hien_tai = TrangThaiVAD.DANG_NOI
                    su_kien = TrangThaiVAD.BAT_DAU_GIONG_NOI
                else:
                    su_kien = TrangThaiVAD.IM_LANG
            else:
                self._trang_thai_hien_tai = TrangThaiVAD.DANG_NOI
                su_kien = TrangThaiVAD.DANG_NOI
        else:
            self._dem_khung_im_lang += 1
            if self.tu_dong_thich_nghi and self._trang_thai_hien_tai == TrangThaiVAD.IM_LANG:
                # Cập nhật mức ồn nền khi đang yên tĩnh
                self._muc_on_nen = (
                    (1.0 - self._he_so_hoc_on) * self._muc_on_nen + self._he_so_hoc_on * nang_luong
                )

            if self._trang_thai_hien_tai == TrangThaiVAD.DANG_NOI:
                if self._dem_khung_im_lang >= self.so_khung_im_lang_ngat_cau:
                    self._trang_thai_hien_tai = TrangThaiVAD.IM_LANG
                    self._dem_khung_noi = 0
                    su_kien = TrangThaiVAD.KET_THUC_GIONG_NOI
                else:
                    su_kien = TrangThaiVAD.DANG_NOI
            else:
                self._trang_thai_hien_tai = TrangThaiVAD.IM_LANG
                su_kien = TrangThaiVAD.IM_LANG

        return {
            "trang_thai": su_kien.value,
            "co_giong_noi": self._trang_thai_hien_tai == TrangThaiVAD.DANG_NOI,
            "nang_luong": round(nang_luong, 6),
            "zcr": round(zcr, 4),
            "muc_on_nen": round(self._muc_on_nen, 6),
            "thay_doi_trang_thai": trang_thai_truoc != self._trang_thai_hien_tai,
        }

    def phan_doan_am_thanh(
        self,
        tin_hieu: np.ndarray,
        tan_so: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Phân đoạn một bản ghi âm thanh hoàn chỉnh thành các đoạn có giọng nói (start_sample, end_sample).

        Args:
            tin_hieu: Mảng âm thanh toàn phần (1D array).
            tan_so: Tần số lấy mẫu tương ứng.

        Returns:
            Danh sách các tuple `(chi_so_dau, chi_so_cuoi)` tính theo mẫu âm thanh.
        """
        self.reset()
        audio = np.asarray(tin_hieu, dtype=np.float32).flatten()
        do_dai = len(audio)
        if do_dai == 0:
            return []

        khung_size = self.so_mau_khung
        danh_sach_doan: List[Tuple[int, int]] = []
        diem_bat_dau: Optional[int] = None

        for pos in range(0, do_dai, khung_size):
            khung = audio[pos : min(pos + khung_size, do_dai)]
            if len(khung) < khung_size:
                khung = np.pad(khung, (0, khung_size - len(khung)))

            ket_qua = self.xu_ly_khung(khung)
            tt = ket_qua["trang_thai"]

            if tt == TrangThaiVAD.BAT_DAU_GIONG_NOI.value and diem_bat_dau is None:
                # Lùi lại vài khung để không làm mất âm đầu câu
                diem_bat_dau = max(0, pos - khung_size * self.so_khung_noi_toi_thieu)
            elif tt == TrangThaiVAD.KET_THUC_GIONG_NOI.value and diem_bat_dau is not None:
                danh_sach_doan.append((diem_bat_dau, min(do_dai, pos + khung_size)))
                diem_bat_dau = None

        if diem_bat_dau is not None:
            danh_sach_doan.append((diem_bat_dau, do_dai))

        return danh_sach_doan
