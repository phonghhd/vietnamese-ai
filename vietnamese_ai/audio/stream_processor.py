"""
Bộ xử lý luồng âm thanh thời gian thực (Real-time Audio Stream Processor).
Tiếp nhận dòng dữ liệu âm thanh liên tục (streaming chunks), phát hiện ngắt câu VAD,
và tự động chuyển giao cho bộ nhận diện giọng nói ASR.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from vietnamese_ai.audio.speech import NhanDienGiongNoi
from vietnamese_ai.audio.vad import PhatHienGiongNoiVAD, TrangThaiVAD


class LuongAmThanhThoiGianThuc:
    """
    Bộ quản lý và xử lý dòng âm thanh thời gian thực từ microphone hoặc WebRTC.

    Khả năng:
    - Tiếp nhận các mẩu nhị phân PCM 16-bit hoặc mảng số thực float32 từ trình duyệt web.
    - Bộ đệm xoay vòng (Ring Buffer) tránh rò rỉ bộ nhớ khi chạy liên tục.
    - Tự động hạ mẫu (Resample) âm thanh từ các tần số phổ biến (44.1kHz, 48kHz) về 16kHz chuẩn.
    - Tích hợp `PhatHienGiongNoiVAD` để phân đoạn phát ngôn và tự động chuyển tiếp tới `NhanDienGiongNoi`.
    """

    def __init__(
        self,
        tan_so_lay_mau_dich: int = 16000,
        tan_so_lay_mau_nguon: int = 16000,
        bo_nhan_dien: Optional[NhanDienGiongNoi] = None,
        bo_vad: Optional[PhatHienGiongNoiVAD] = None,
        dung_luong_buffer_giay: float = 30.0,
        callback_van_ban: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Khởi tạo luồng âm thanh thời gian thực.

        Args:
            tan_so_lay_mau_dich: Tần số lấy mẫu xử lý chuẩn (mặc định 16000 Hz).
            tan_so_lay_mau_nguon: Tần số lấy mẫu của luồng đầu vào từ thiết bị ghi âm (16000, 44100, 48000 Hz).
            bo_nhan_dien: Thể hiện của `NhanDienGiongNoi` để chuyển âm thanh thành văn bản.
            bo_vad: Thể hiện của `PhatHienGiongNoiVAD` (tự khởi tạo nếu None).
            dung_luong_buffer_giay: Dung lượng lưu đệm tối đa tính theo giây.
            callback_van_ban: Hàm gọi lại khi một câu nói hoàn tất được nhận diện.
        """
        self.tan_so_dich = tan_so_lay_mau_dich
        self.tan_so_nguon = tan_so_lay_mau_nguon
        self.asr = bo_nhan_dien or NhanDienGiongNoi(tan_so_lay_mau=self.tan_so_dich)
        self.vad = bo_vad or PhatHienGiongNoiVAD(tan_so_lay_mau=self.tan_so_dich)
        self.callback_van_ban = callback_van_ban

        self.kich_thuoc_buffer_toi_da = int(round(self.tan_so_dich * dung_luong_buffer_giay))
        self._buffer: List[float] = []
        self._mau_dang_noi: List[float] = []
        self._khung_du_thua: List[float] = []
        self._dang_trong_phat_ngon = False

    def reset(self) -> None:
        """Xóa bộ đệm và đặt lại trạng thái luồng."""
        self._buffer.clear()
        self._mau_dang_noi.clear()
        self._khung_du_thua.clear()
        self._dang_trong_phat_ngon = False
        self.vad.reset()

    @staticmethod
    def pcm16_sang_float32(du_lieu_bytes: bytes) -> np.ndarray:
        """Chuyển đổi chuỗi byte PCM 16-bit little-endian sang mảng NumPy float32 `[-1.0, 1.0]`."""
        if not du_lieu_bytes:
            return np.array([], dtype=np.float32)
        pcm = np.frombuffer(du_lieu_bytes, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    @staticmethod
    def float32_sang_pcm16(tin_hieu: np.ndarray) -> bytes:
        """Chuyển đổi mảng NumPy float32 `[-1.0, 1.0]` sang chuỗi byte PCM 16-bit little-endian."""
        clipped = np.clip(tin_hieu, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        return pcm.tobytes()

    def ha_mau(self, tin_hieu: np.ndarray) -> np.ndarray:
        """Hạ mẫu tín hiệu từ `tan_so_nguon` về `tan_so_dich` bằng phép nội suy tuyến tính nhanh."""
        if self.tan_so_nguon == self.tan_so_dich or len(tin_hieu) == 0:
            return tin_hieu

        so_mau_moi = int(round(len(tin_hieu) * (self.tan_so_dich / self.tan_so_nguon)))
        if so_mau_moi <= 0:
            return np.array([], dtype=np.float32)

        indices_goc = np.linspace(0, len(tin_hieu) - 1, num=so_mau_moi)
        return np.interp(indices_goc, np.arange(len(tin_hieu)), tin_hieu).astype(np.float32)

    def nap_du_lieu_raw(self, du_lieu_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Nạp chuỗi byte âm thanh thô (PCM 16-bit) từ socket / stream vào luồng xử lý.

        Returns:
            Danh sách các sự kiện phát sinh (ví dụ: phát ngôn bắt đầu, kết thúc câu có kết quả ASR).
        """
        samples = self.pcm16_sang_float32(du_lieu_bytes)
        return self.nap_tin_hieu(samples)

    def nap_tin_hieu(self, tin_hieu: Union[np.ndarray, List[float]]) -> List[Dict[str, Any]]:
        """
        Nạp mảng mẫu âm thanh float32 vào đường ống nhận diện thời gian thực.

        Returns:
            Danh sách các sự kiện: `{"loai": "bat_dau_noi"}` hoặc `{"loai": "hoan_thanh_cau", ...}`.
        """
        arr = np.asarray(tin_hieu, dtype=np.float32).flatten()
        if len(arr) == 0:
            return []

        # Hạ mẫu nếu cần
        if self.tan_so_nguon != self.tan_so_dich:
            arr = self.ha_mau(arr)

        danh_sach_su_kien: List[Dict[str, Any]] = []
        samples_moi = self._khung_du_thua + arr.tolist()
        self._khung_du_thua.clear()

        so_mau_khung = self.vad.so_mau_khung
        idx = 0
        tong_mau = len(samples_moi)

        while idx + so_mau_khung <= tong_mau:
            khung = np.array(samples_moi[idx : idx + so_mau_khung], dtype=np.float32)
            idx += so_mau_khung

            # Lưu vào bộ đệm tổng thể
            self._buffer.extend(khung.tolist())
            if len(self._buffer) > self.kich_thuoc_buffer_toi_da:
                self._buffer = self._buffer[-self.kich_thuoc_buffer_toi_da :]

            ket_qua_vad = self.vad.xu_ly_khung(khung)
            trang_thai = ket_qua_vad["trang_thai"]

            if trang_thai == TrangThaiVAD.BAT_DAU_GIONG_NOI.value:
                self._dang_trong_phat_ngon = True
                self._mau_dang_noi.extend(khung.tolist())
                danh_sach_su_kien.append(
                    {
                        "loai": "bat_dau_noi",
                        "nang_luong": ket_qua_vad["nang_luong"],
                        "zcr": ket_qua_vad["zcr"],
                    }
                )

            elif trang_thai == TrangThaiVAD.DANG_NOI.value:
                self._mau_dang_noi.extend(khung.tolist())

            elif trang_thai == TrangThaiVAD.KET_THUC_GIONG_NOI.value:
                self._dang_trong_phat_ngon = False
                self._mau_dang_noi.extend(khung.tolist())

                # Chốt phát ngôn và chạy ASR
                am_thanh_cau = np.array(self._mau_dang_noi, dtype=np.float32)
                self._mau_dang_noi.clear()

                ket_qua_asr = self.asr.nhan_dien(am_thanh_cau)
                su_kien = {
                    "loai": "hoan_thanh_cau",
                    "van_ban": ket_qua_asr.get("van_ban", ""),
                    "do_tin_cay": ket_qua_asr.get("do_tin_cay", 0.0),
                    "thoi_luong_giay": round(len(am_thanh_cau) / self.tan_so_dich, 3),
                    "backend": ket_qua_asr.get("backend", "native_mfcc"),
                    "tin_hieu_audio": am_thanh_cau,
                }
                danh_sach_su_kien.append(su_kien)

                if self.callback_van_ban is not None:
                    self.callback_van_ban(su_kien)

        if idx < tong_mau:
            self._khung_du_thua = samples_moi[idx:]

        return danh_sach_su_kien

    def lay_buffer(self) -> np.ndarray:
        """Lấy toàn bộ nội dung trong bộ đệm trượt hiện tại."""
        return np.array(self._buffer, dtype=np.float32)
