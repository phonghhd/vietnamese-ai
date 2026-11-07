"""Tổng đài Chăm sóc Khách hàng Đa phương thức (CSKH) - Vietnamese AI Framework."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from vietnamese_ai.audio.speech import NhanDienGiongNoi, TongHopGiongNoi
from vietnamese_ai.security.data_defense import CongAnToanDauRa
from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework


class TongDaiCSKH:
    """
    Hệ thống Tổng đài Chăm sóc Khách hàng Tự động (Omnichannel Voicebot / Chatbot):
    - Hỗ trợ tiếp nhận cả Văn bản (Chat) và Âm thanh (Voicebot).
    - Tự động nhận diện ý định (Intent), tra cứu đơn hàng, chính sách đổi trả, bảo hành.
    - Bảo vệ đa tầng: Tường lửa AI chặn prompt injection đầu vào + Cổng an toàn ẩn PII & chống lặp đầu ra.
    - Tổng hợp giọng nói tiếng Việt 6 thanh điệu mượt mà phản hồi khách hàng.
    """

    CS_DOI_TRA = "Chính sách đổi trả: Quý khách được đổi trả miễn phí trong vòng 7 ngày kể từ khi nhận hàng đối với sản phẩm còn nguyên tem mác và hóa đơn."
    CS_BAO_HANH = "Chính sách bảo hành: Sản phẩm được bảo hành chính hãng 12 tháng tại tất cả các trung tâm bảo hành trên toàn quốc."
    THOI_GIAN_LAM_VIEC = "Thời gian hoạt động của tổng đài: Từ 8h00 đến 21h00 hàng ngày, kể cả Thứ 7 và Chủ Nhật."

    CO_SO_DU_LIEU_DON_HANG: Dict[str, Dict[str, str]] = {
        "DH1001": {
            "trang_thai": "Đang giao hàng",
            "ngay_du_kien": "10/11/2026",
            "san_pham": "Áo sơ mi công sở",
            "tong_tien": "350.000 VNĐ",
        },
        "DH1002": {
            "trang_thai": "Đã giao thành công",
            "ngay_du_kien": "05/11/2026",
            "san_pham": "Giày thể thao",
            "tong_tien": "850.000 VNĐ",
        },
        "DH1003": {
            "trang_thai": "Đang đóng gói tại kho",
            "ngay_du_kien": "12/11/2026",
            "san_pham": "Balo laptop",
            "tong_tien": "520.000 VNĐ",
        },
    }

    def __init__(
        self,
        ten_thuong_hieu: str = "Vietnamese AI Shop",
        asr: Optional[NhanDienGiongNoi] = None,
        tts: Optional[TongHopGiongNoi] = None,
    ):
        self.ten_thuong_hieu = ten_thuong_hieu
        self.tuong_lua = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
        self.asr = asr or NhanDienGiongNoi()
        self.tts = tts or TongHopGiongNoi()

    def _chuan_hoa_dau_vao(self, yeu_cau: Union[str, Path, np.ndarray]) -> Tuple[str, bool]:
        """Chuẩn hóa yêu cầu đầu vào từ text hoặc giọng nói."""
        if isinstance(yeu_cau, (str, Path)) and Path(str(yeu_cau)).is_file():
            p = Path(str(yeu_cau))
            if p.suffix.lower() == ".wav":
                res_asr = self.asr.nhan_dien(p)
                return res_asr.get("van_ban", ""), True
        if isinstance(yeu_cau, np.ndarray):
            res_asr = self.asr.nhan_dien(yeu_cau)
            return res_asr.get("van_ban", ""), True
        return str(yeu_cau or "").strip(), False

    def tra_cuu_don_hang(self, ma_don: str) -> Dict[str, Any]:
        """Công cụ nghiệp vụ tra cứu trạng thái đơn hàng (Tool Calling)."""
        ma_chuan = ma_don.strip().upper()
        if ma_chuan in self.CO_SO_DU_LIEU_DON_HANG:
            return {"ton_tai": True, "ma_don": ma_chuan, **self.CO_SO_DU_LIEU_DON_HANG[ma_chuan]}
        return {"ton_tai": False, "ma_don": ma_chuan, "thong_bao": f"Không tìm thấy thông tin cho mã đơn hàng {ma_chuan}."}

    def xu_ly_yeu_cau(
        self,
        yeu_cau: Union[str, Path, np.ndarray],
        co_tra_ve_am_thanh: bool = False,
        duong_dan_luu_am_thanh: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý yêu cầu của khách hàng, tự động tra cứu nghiệp vụ và trả lời.
        """
        van_ban_khach, la_giong_noi = self._chuan_hoa_dau_vao(yeu_cau)

        # 1. Quét bảo mật Tường lửa AI
        kt_dau_vao = self.tuong_lua.kiem_tra_dau_vao(van_ban_khach)
        if not kt_dau_vao["an_toan"]:
            tra_loi_chan = "Dạ hệ thống xin phép từ chối yêu cầu do phát hiện nội dung chưa phù hợp tiêu chuẩn an toàn."
            return {
                "cau_hoi": van_ban_khach,
                "tra_loi": tra_loi_chan,
                "y_dinh": "BI_CHAN",
                "la_giong_noi": la_giong_noi,
                "an_toan": False,
            }

        text_lower = van_ban_khach.lower()
        y_dinh = "HOI_DAP_CHUNG"
        thong_tin_bo_sung = {}

        # 2. Nhận diện ý định & Gọi công cụ nghiệp vụ
        m_don = re.search(r"\b(DH\d{3,5})\b", van_ban_khach, flags=re.IGNORECASE)
        if m_don or "đơn hàng" in text_lower or "tra cứu đơn" in text_lower:
            y_dinh = "TRA_CUU_DON_HANG"
            if m_don:
                ma_dh = m_don.group(1).upper()
                info = self.tra_cuu_don_hang(ma_dh)
                thong_tin_bo_sung = info
                if info["ton_tai"]:
                    tra_loi = (
                        f"Dạ đơn hàng {ma_dh} ({info['san_pham']}) hiện đang ở trạng thái: {info['trang_thai']}. "
                        f"Thời gian giao hàng dự kiến là {info['ngay_du_kien']} với tổng giá trị {info['tong_tien']} ạ."
                    )
                else:
                    tra_loi = f"Dạ hệ thống không tìm thấy đơn hàng mã {ma_dh}. Quý khách vui lòng kiểm tra lại mã đơn ạ."
            else:
                tra_loi = "Dạ để tra cứu đơn hàng, quý khách vui lòng cung cấp mã đơn hàng (ví dụ: DH1001) giúp em ạ."

        elif "đổi trả" in text_lower or "trả hàng" in text_lower or "đổi hàng" in text_lower:
            y_dinh = "CHINH_SACH_DOI_TRA"
            tra_loi = f"Dạ tại {self.ten_thuong_hieu}, {self.CS_DOI_TRA}"

        elif "bảo hành" in text_lower or "sửa chữa" in text_lower:
            y_dinh = "CHINH_SACH_BAO_HANH"
            tra_loi = f"Dạ tại {self.ten_thuong_hieu}, {self.CS_BAO_HANH}"

        elif "mấy giờ" in text_lower or "thời gian" in text_lower or "mở cửa" in text_lower:
            y_dinh = "THOI_GIAN_LAM_VIEC"
            tra_loi = f"Dạ, {self.THOI_GIAN_LAM_VIEC}"

        elif "chào" in text_lower or "hello" in text_lower or "hi" in text_lower:
            y_dinh = "CHAO_HOI"
            tra_loi = f"Dạ {self.ten_thuong_hieu} xin chào quý khách! Em có thể hỗ trợ gì cho mình hôm nay ạ?"

        else:
            tra_loi = (
                f"Dạ {self.ten_thuong_hieu} đã ghi nhận yêu cầu của quý khách: '{van_ban_khach}'. "
                "Em sẽ chuyển thông tin đến tổng đài viên chuyên trách để hỗ trợ quý khách chu đáo nhất ạ."
            )

        # 3. Kiểm định an toàn đầu ra
        kt_dau_ra = CongAnToanDauRa.kiem_tra_va_lam_sach(tra_loi)
        tra_loi_sach = kt_dau_ra["van_ban_sach"]

        # 4. Tổng hợp tiếng nói nếu yêu cầu
        tep_am_thanh = None
        if co_tra_ve_am_thanh:
            res_tts = self.tts.tong_hop(tra_loi_sach, duong_dan_luu=duong_dan_luu_am_thanh)
            tep_am_thanh = res_tts.get("duong_dan") or duong_dan_luu_am_thanh

        return {
            "cau_hoi": van_ban_khach,
            "tra_loi": tra_loi_sach,
            "y_dinh": y_dinh,
            "la_giong_noi": la_giong_noi,
            "thong_tin_nghiep_vu": thong_tin_bo_sung,
            "tep_am_thanh": str(tep_am_thanh) if tep_am_thanh else None,
            "an_toan": kt_dau_ra["an_toan"],
        }


# English alias
CustomerSupportBot = TongDaiCSKH
