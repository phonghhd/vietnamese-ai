"""
Plugin Tài Chính - Chứng Khoán (VN Stocks) & Crypto Real-time.
Tích hợp:
1. Thu thập dữ liệu giá Real-time & Nến OHLCV (Độ trễ thấp, Zero heavy dependency).
2. Động cơ tính toán chỉ báo kỹ thuật định lượng thuần NumPy (RSI, MACD, Bollinger Bands, EMA).
3. Quản trị rủi ro & định cỡ vị thế (Position sizing 2% rule).
4. Phân tích tâm lý thị trường chứng khoán bản địa (F319, CafeF, Telegram).
5. Bộ tri thức chuẩn quy chế giao dịch HOSE/HNX/UPCoM và pháp lý UBCKNN.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from vietnamese_ai.plugin.plugin_base import PluginCongCu, PluginTriThuc
from vietnamese_ai.plugins.finance.market_feed import BoDieuPhoiThiTruong
from vietnamese_ai.plugins.finance.market_predictor import BoDuDoanXuHuongNganHan
from vietnamese_ai.plugins.finance.quant_indicators import (
    phan_tich_tin_hieu_tong_hop,
    tinh_quan_tri_rui_ro,
)


class PluginChungKhoanCrypto(PluginCongCu, PluginTriThuc):
    """
    Plugin Tài chính - Chứng khoán Việt Nam & Crypto Real-time.
    Hỗ trợ Tác tử (TacTu) thực hiện phân tích kỹ thuật và tra cứu nghiệp vụ thị trường.
    """

    ten: str = "Chứng Khoán & Crypto Real-time"
    phien_ban: str = "1.0.0"
    mo_ta: str = (
        "Phân tích kỹ thuật chứng khoán VN (HOSE/HNX/UPCoM) & Crypto, "
        "lấy giá realtime, quản trị rủi ro và tra cứu quy chế giao dịch"
    )
    loai: str = "tai_chinh_chung_khoan_crypto"

    CANH_BAO_PHAP_LY: str = (
        "\n\n---\n"
        "⚠️ **LƯU Ý MIỄN TRỪ TRÁCH NHIỆM PHÁP LÝ (UBCKNN & V-AI GUARDRAIL):**\n"
        "Mọi dữ liệu giá, chỉ báo kỹ thuật và phân tích trên được cung cấp nhằm mục đích "
        "nghiên cứu học thuật và tham khảo thông tin, hoàn toàn KHÔNG cấu thành lời khuyên đầu tư "
        "hoặc khuyến nghị mua bán tài chính. Nhà đầu tư tự chịu trách nhiệm về các quyết định của mình."
    )

    TU_DIEN_QUY_CHE_VN = {
        "T+2.5": (
            "Chu kỳ thanh toán T+2.5 tại Việt Nam: Nhà đầu tư mua cổ phiếu vào ngày T "
            "sẽ nhận cổ phiếu về tài khoản vào khoảng 11h30 - 13h00 ngày T+2 và được phép "
            "đặt lệnh bán ngay trong phiên giao dịch buổi chiều ngày T+2."
        ),
        "ATO": (
            "Phiên khớp lệnh định kỳ mở cửa (At The Opening): Diễn ra từ 09h00 đến 09h15 trên sàn HOSE. "
            "Xác định giá mở cửa của ngày. Trong phiên này lệnh ATO được ưu tiên khớp trước lệnh giới hạn (LO), "
            "không được phép hủy hoặc sửa lệnh."
        ),
        "ATC": (
            "Phiên khớp lệnh định kỳ đóng cửa (At The Close): Diễn ra từ 14h30 đến 14h45 trên sàn HOSE và HNX. "
            "Xác định giá đóng cửa của ngày. Lệnh ATC được ưu tiên khớp trước lệnh LO, không được phép hủy hoặc sửa lệnh."
        ),
        "BIEN_DO_TRAN_SAN": (
            "Biên độ dao động giá trong ngày theo quy chế UBCKNN: "
            "Sàn HOSE: ±7% so với giá tham chiếu; "
            "Sàn HNX: ±10% so với giá tham chiếu; "
            "Sàn UPCoM: ±15% so với giá bình quân của phiên giao dịch liền trước."
        ),
        "LO_GIAO_DICH": (
            "Quy định lô giao dịch: Lô chẵn là 100 cổ phiếu hoặc chứng chỉ quỹ. "
            "Lô lẻ là từ 1 đến 99 cổ phiếu, được giao dịch riêng trên bảng lô lẻ hoặc bán cho công ty chứng khoán."
        ),
        "MARGIN": (
            "Giao dịch ký quỹ (Margin): Tỷ lệ ký quỹ ban đầu tối thiểu thường là 50%. "
            "Khi tỷ lệ tài khoản giảm dưới tỷ lệ duy trì (thường 35%-40%), công ty chứng khoán sẽ phát lệnh "
            "Margin Call yêu cầu nộp thêm tiền; nếu không bổ sung kịp sẽ bị bán giải chấp (Force Sell)."
        ),
        "NGHI_TRUA": (
            "Giờ nghỉ trưa thị trường chứng khoán Việt Nam: Từ 11h30 đến 13h00 hàng ngày, "
            "thị trường tạm ngừng giao dịch khớp lệnh."
        ),
    }

    TU_DIEN_CRYPTO = {
        "HALVING": (
            "Bitcoin Halving: Sự kiện diễn ra khoảng 4 năm một lần (mỗi 210.000 block), "
            "trong đó phần thưởng khối cho thợ đào Bitcoin bị cắt giảm 50%, làm giảm tốc độ phát hành BTC mới."
        ),
        "FUNDING_RATE": (
            "Funding Rate trong hợp đồng tương lai vĩnh cửu (Perpetual Futures): "
            "Khoản phí thanh toán định kỳ giữa phe Long và phe Short để giữ giá phái sinh bám sát giá giao ngay (Spot)."
        ),
        "LIQUIDATION": (
            "Thanh lý vị thế (Liquidation): Quá trình sàn giao dịch tự động đóng lệnh đòn bẩy của người dùng "
            "khi mức lỗ chạm ngưỡng ký quỹ duy trì, dẫn tới cháy tài khoản."
        ),
        "SLIPPAGE": (
            "Độ trượt giá (Slippage): Chênh lệch giữa mức giá kỳ vọng khi đặt lệnh và mức giá thực tế được khớp, "
            "thường xảy ra ở các thị trường biến động mạnh hoặc bể thanh khoản thấp trên DEX."
        ),
        "DEFI": (
            "Decentralized Finance (Tài chính phi tập trung): Hệ sinh thái các ứng dụng tài chính "
            "vận hành trên Smart Contract của Blockchain (Ethereum, Solana, BNB Chain) không thông qua bên trung gian."
        ),
    }

    def __init__(self, dung_gia_lap_khi_loi: bool = True):
        super().__init__()
        self.market_feed = BoDieuPhoiThiTruong(dung_gia_lap_khi_loi=dung_gia_lap_khi_loi)
        self.predictor = BoDuDoanXuHuongNganHan()

    # ==================== PHẦN CÔNG CỤ (TOOLS) ====================

    def dang_ky_cong_cu(self) -> List[Dict[str, Any]]:
        """Đăng ký các công cụ phân tích tài chính cho Tác tử (TacTu)."""
        return [
            {
                "ten": "lay_gia_realtime",
                "mo_ta": "Lấy giá realtime, khối lượng và biến động 24h của cổ phiếu VN (HOSE/HNX/UPCoM) hoặc tiền số Crypto (Binance)",
                "ham": self.lay_gia_realtime,
                "tham_so": {
                    "ma": {"kieu": "str", "bat_buoc": True, "mo_ta": "Mã cổ phiếu (vd: HPG, SSI) hoặc crypto (vd: BTCUSDT)"},
                    "loai_tai_san": {"kieu": "str", "bat_buoc": False, "mo_ta": "Tùy chọn: 'chung_khoan_vn' hoặc 'crypto'"},
                },
            },
            {
                "ten": "phan_tich_ky_thuat",
                "mo_ta": "Tính toán và phân tích các chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands, EMA, Pivot) cho mã tài sản",
                "ham": self.phan_tich_ky_thuat,
                "tham_so": {
                    "ma": {"kieu": "str", "bat_buoc": True, "mo_ta": "Mã chứng khoán hoặc crypto"},
                    "khung_thoi_gian": {"kieu": "str", "bat_buoc": False, "mo_ta": "Khung thời gian (mặc định '1D')"},
                    "so_nen": {"kieu": "int", "bat_buoc": False, "mo_ta": "Số lượng nến phân tích (mặc định 60)"},
                },
            },
            {
                "ten": "du_doan_xu_huong_5p",
                "mo_ta": "Dự đoán xác suất xu hướng giá (Tăng / Giảm / Đi ngang) và biên độ biến động trong 5-10 phút tới cho Crypto hoặc Phái sinh VN30",
                "ham": self.du_doan_xu_huong_5p,
                "tham_so": {
                    "ma": {"kieu": "str", "bat_buoc": True, "mo_ta": "Mã tài sản (vd: BTCUSDT, ETHUSDT, VN30F1M, HPG)"},
                    "khung_du_doan": {"kieu": "str", "bat_buoc": False, "mo_ta": "Khung dự đoán: '5p' hoặc '10p' (mặc định '5p')"},
                },
            },
            {
                "ten": "tinh_quan_tri_rui_ro",
                "mo_ta": "Tính toán khối lượng vào lệnh an toàn theo quy tắc quản trị rủi ro 2% tài khoản và các mốc chốt lời",
                "ham": self.tinh_quan_tri_rui_ro_lenh,
                "tham_so": {
                    "tong_von": {"kieu": "float", "bat_buoc": True, "mo_ta": "Tổng vốn tài khoản (VND hoặc USDT)"},
                    "gia_vao": {"kieu": "float", "bat_buoc": True, "mo_ta": "Mức giá dự kiến mua"},
                    "gia_cat_lo": {"kieu": "float", "bat_buoc": True, "mo_ta": "Mức giá cắt lỗ kỹ thuật"},
                    "ty_le_rui_ro": {"kieu": "float", "bat_buoc": False, "mo_ta": "Tỷ lệ rủi ro tối đa (mặc định 0.02 tức 2%)"},
                },
            },
            {
                "ten": "tra_cuu_luat_giao_dich",
                "mo_ta": "Tra cứu các quy chế giao dịch chứng khoán (T+2.5, ATO, ATC, biên độ, Margin) và thuật ngữ Crypto",
                "ham": self.tra_cuu_luat_giao_dich,
                "tham_so": {
                    "tu_khoa": {"kieu": "str", "bat_buoc": True, "mo_ta": "Từ khóa cần tra cứu (vd: T+2.5, ATO, Margin, Halving)"},
                },
            },
            {
                "ten": "phan_tich_tam_ly_thi_truong",
                "mo_ta": "Phân tích sắc thái tâm lý (Lạc quan / Bi quan / Trung lập, Fear & Greed) từ tin tức và diễn đàn chứng khoán",
                "ham": self.phan_tich_tam_ly_thi_truong,
                "tham_so": {
                    "van_ban": {"kieu": "str", "bat_buoc": True, "mo_ta": "Nội dung bài viết diễn đàn (F319/Telegram/CafeF) cần phân tích"},
                },
            },
        ]

    def lay_gia_realtime(self, ma: str, loai_tai_san: str = "tu_dong") -> Dict[str, Any]:
        """Lấy giá cập nhật mới nhất từ thị trường."""
        loai = None if loai_tai_san == "tu_dong" else loai_tai_san
        ket_qua = self.market_feed.lay_gia_hien_tai(ma, loai_tai_san=loai)

        don_vi = "USDT" if ket_qua.get("loai") == "crypto" else "nghìn VNĐ"
        tom_tat = (
            f"Mã: {ket_qua.get('ma')} | Giá hiện tại: {ket_qua.get('gia_hien_tai')} {don_vi} | "
            f"Biến động: {ket_qua.get('thay_doi_pct'):+.2f}% ({ket_qua.get('thay_doi'):+.2f}) | "
            f"Khối lượng: {ket_qua.get('khoi_luong'):,} | Nguồn: {ket_qua.get('nguon')}"
        )

        return {
            "trang_thai": "thanh_cong",
            "du_lieu": ket_qua,
            "tom_tat": tom_tat,
            "canh_bao": self.CANH_BAO_PHAP_LY,
        }

    def phan_tich_ky_thuat(
        self,
        ma: str,
        khung_thoi_gian: str = "1D",
        so_nen: int = 60,
    ) -> Dict[str, Any]:
        """Lấy chuỗi nến và phân tích kỹ thuật toàn diện."""
        nen = self.market_feed.lay_du_lieu_nen(
            ma, khung_thoi_gian=khung_thoi_gian, so_nen=so_nen
        )
        if not nen:
            return {
                "trang_thai": "loi",
                "thong_bao": f"Không thể lấy dữ liệu nến cho mã '{ma}'.",
            }

        pt = phan_tich_tin_hieu_tong_hop(nen)
        return {
            "trang_thai": "thanh_cong",
            "ma": ma.upper(),
            "khung_thoi_gian": khung_thoi_gian,
            "so_nen_phan_tich": len(nen),
            "ket_qua": pt,
            "canh_bao": self.CANH_BAO_PHAP_LY,
        }

    def du_doan_xu_huong_5p(
        self,
        ma: str,
        khung_du_doan: str = "5p",
    ) -> Dict[str, Any]:
        """
        Dự đoán xác suất xu hướng giá (Tăng / Giảm / Đi ngang) trong 5 hoặc 10 phút tới.
        Sử dụng mạng nơ-ron vi mô phân tích 24 đặc trưng vi cấu trúc từ nến 1 phút.
        """
        # Thu thập 35 nến 1 phút gần nhất
        nen_1m = self.market_feed.lay_du_lieu_nen(
            ma, khung_thoi_gian="1m", so_nen=35
        )
        if not nen_1m:
            return {
                "trang_thai": "loi",
                "thong_bao": f"Không thể lấy dữ liệu nến 1 phút cho mã '{ma}'.",
            }

        gia_ht_info = self.market_feed.lay_gia_hien_tai(ma)
        gia_ht = gia_ht_info.get("gia_hien_tai") or float(nen_1m[-1].get("dong_cua", 0.0))

        kq = self.predictor.du_doan_tu_nen(
            danh_sach_nen_1m=nen_1m,
            khung_du_doan=khung_du_doan,
            gia_tham_chieu=gia_ht,
        )

        return {
            "trang_thai": "thanh_cong",
            "ma": ma.upper(),
            "loai": gia_ht_info.get("loai", "chua_ro"),
            "du_doan": kq,
            "canh_bao": self.CANH_BAO_PHAP_LY,
        }

    def tinh_quan_tri_rui_ro_lenh(
        self,
        tong_von: float,
        gia_vao: float,
        gia_cat_lo: float,
        ty_le_rui_ro: float = 0.02,
    ) -> Dict[str, Any]:
        """Tính toán kích thước vị thế vào lệnh an toàn."""
        kq = tinh_quan_tri_rui_ro(
            tong_von=tong_von,
            gia_vao=gia_vao,
            gia_cat_lo=gia_cat_lo,
            ty_le_rui_ro=ty_le_rui_ro,
        )
        return {
            "trang_thai": "thanh_cong",
            "ket_qua": kq,
            "canh_bao": self.CANH_BAO_PHAP_LY,
        }

    def tra_cuu_luat_giao_dich(self, tu_khoa: str) -> Dict[str, Any]:
        """Tra cứu nhanh quy chế thị trường chứng khoán Việt Nam hoặc Crypto."""
        tk = tu_khoa.upper().strip()
        ket_qua = []

        # Tra cứu từ điển chứng khoán Việt Nam
        for k, v in self.TU_DIEN_QUY_CHE_VN.items():
            if tk in k or k in tk or any(w in v.upper() for w in tk.split()):
                ket_qua.append({"loai": "Quy chế TTCK Việt Nam", "muc": k, "noi_dung": v})

        # Tra cứu từ điển Crypto
        for k, v in self.TU_DIEN_CRYPTO.items():
            if tk in k or k in tk or any(w in v.upper() for w in tk.split()):
                ket_qua.append({"loai": "Thuật ngữ Crypto", "muc": k, "noi_dung": v})

        if not ket_qua:
            return {
                "trang_thai": "khong_tim_thay",
                "thong_bao": f"Không tìm thấy quy chế hoặc thuật ngữ phù hợp với từ khóa '{tu_khoa}'.",
            }

        return {
            "trang_thai": "thanh_cong",
            "so_luong": len(ket_qua),
            "ket_qua": ket_qua,
            "canh_bao": self.CANH_BAO_PHAP_LY,
        }

    def phan_tich_tam_ly_thi_truong(self, van_ban: str) -> Dict[str, Any]:
        """
        Phân tích tâm lý văn bản tài chính / diễn đàn chứng khoán (F319, CafeF, Telegram).
        Đánh giá chỉ số Sợ hãi & Tham lam (Fear & Greed Index).
        """
        text = van_ban.lower()

        TU_KHOA_TICH_CUC = [
            "vượt đỉnh",
            "tím",
            "tăng trần",
            "bùng nổ",
            "gom hàng",
            "dòng tiền vào",
            "lãi đột biến",
            "bullish",
            "moon",
            "up trend",
            "khối ngoại mua ròng",
            "kỳ vọng",
            "tích cực",
            "tăng trưởng",
            "mua mạnh",
        ]

        TU_KHOA_TIEU_CUC = [
            "sập",
            "bán tháo",
            "lau sàn",
            "giải chấp",
            "force sell",
            "thủng đáy",
            "cắt lỗ",
            "úp bô",
            "bearish",
            "scam",
            "chia đôi",
            "down trend",
            "khối ngoại bán ròng",
            "tiêu cực",
            "thua lỗ",
            "bán mạnh",
        ]

        diem_tich_cuc = sum(1 for w in TU_KHOA_TICH_CUC if re.search(r"\b" + re.escape(w) + r"\b", text) or w in text)
        diem_tieu_cuc = sum(1 for w in TU_KHOA_TIEU_CUC if re.search(r"\b" + re.escape(w) + r"\b", text) or w in text)

        tong = diem_tich_cuc + diem_tieu_cuc
        if tong == 0:
            tam_ly = "TRUNG_LAP"
            chi_so_f_and_g = 50.0
            diem_so = 0.0
        else:
            ty_le_tich_cuc = diem_tich_cuc / tong
            diem_so = (diem_tich_cuc - diem_tieu_cuc) / tong * 100.0
            chi_so_f_and_g = round(ty_le_tich_cuc * 100.0, 1)

            if diem_so >= 30:
                tam_ly = "LAC_QUAN"
            elif diem_so <= -30:
                tam_ly = "BI_QUAN"
            else:
                tam_ly = "TRUNG_LAP"

        return {
            "trang_thai": "thanh_cong",
            "tam_ly": tam_ly,
            "diem_so_tam_ly": round(diem_so, 1),
            "chi_so_so_hai_tham_lam": chi_so_f_and_g,
            "chi_tiet": {
                "so_tu_khoa_tich_cuc": diem_tich_cuc,
                "so_tu_khoa_tieu_cuc": diem_tieu_cuc,
            },
            "canh_bao": self.CANH_BAO_PHAP_LY,
        }

    # ==================== PHẦN TRI THỨC (KNOWLEDGE / RAG) ====================

    def nap_nguon_tri_thuc(self) -> List[Dict[str, Any]]:
        """Cung cấp cơ sở tri thức nghiệp vụ cho RAG Pipeline."""
        tri_thuc = []

        for k, v in self.TU_DIEN_QUY_CHE_VN.items():
            tri_thuc.append(
                {
                    "tieu_de": f"Quy chế TTCK Việt Nam: {k}",
                    "noi_dung": f"Quy chế giao dịch thị trường chứng khoán Việt Nam (HOSE/HNX/UPCoM) về {k}: {v}",
                    "metadata": {
                        "chu_de": "chung_khoan_viet_nam",
                        "nguon": "UBCKNN",
                        "muc": k,
                    },
                }
            )

        for k, v in self.TU_DIEN_CRYPTO.items():
            tri_thuc.append(
                {
                    "tieu_de": f"Thuật ngữ Crypto: {k}",
                    "noi_dung": f"Cơ chế và giải thích thuật ngữ thị trường tài sản số Crypto về {k}: {v}",
                    "metadata": {
                        "chu_de": "crypto_blockchain",
                        "nguon": "Binance Academy",
                        "muc": k,
                    },
                }
            )

        tri_thuc.append(
            {
                "tieu_de": "Nguyên tắc Quản trị Rủi ro 2% trong Đầu tư",
                "noi_dung": (
                    "Quy tắc quản trị vốn 2% quy định rằng trong bất kỳ một giao dịch nào, "
                    "nhà đầu tư không bao giờ để rủi ro mất mát vượt quá 2% tổng giá trị tài khoản. "
                    "Công thức tính khối lượng mua: Số lượng = (Tổng vốn * 2%) / (Giá vào - Giá cắt lỗ)."
                ),
                "metadata": {"chu_de": "quan_tri_rui_ro", "nguon": "Alexander Elder"},
            }
        )

        tri_thuc.append(
            {
                "tieu_de": "Phương pháp Dự đoán Vi cấu trúc Khung 5 - 10 Phút",
                "noi_dung": (
                    "Dự đoán vi cấu trúc 5 - 10 phút sử dụng mạng nơ-ron đa tầng phân tích 24 đặc trưng nến 1 phút, "
                    "bao gồm Log-return, EMA9/21, RSI-7, MACD Histogram và Volume Surge. "
                    "Đầu ra là phân phối xác suất Softmax: P(Tăng), P(Đi ngang), P(Giảm) kèm biên độ biến động kỳ vọng ATR. "
                    "Chỉ nên vào lệnh khi xác suất một hướng vượt trên 60% và có sự đồng thuận từ chỉ báo kỹ thuật khung lớn."
                ),
                "metadata": {"chu_de": "du_doan_vi_cau_truc", "nguon": "Vietnamese AI Quant"},
            }
        )

        return tri_thuc

    cung_cap_tri_thuc = nap_nguon_tri_thuc
