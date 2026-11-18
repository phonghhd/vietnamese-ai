"""
Plugin Tài Chính - Kế Toán & Thuế (FinTech & Accounting Plugin cho Vietnamese AI Framework).
Cung cấp công cụ tính thuế TNCN/TNDN, bóc tách hóa đơn điện tử, tính khấu hao TSCĐ
và bộ tri thức tra cứu chuẩn mực kế toán Việt Nam (VAS/VFRS).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from vietnamese_ai.plugin.plugin_base import PluginCongCu, PluginTriThuc


class PluginTaiChinh(PluginCongCu, PluginTriThuc):
    """Plugin Tài chính, Kế toán và Thuế Việt Nam."""

    ten: str = "Tài Chính & Kế Toán"
    phien_ban: str = "1.0.0"
    mo_ta: str = "Tính thuế TNCN/TNDN, bóc tách hóa đơn VAT, khấu hao TSCĐ và tri thức VAS"
    loai: str = "tai_chinh"

    # Mức giảm trừ gia cảnh (áp dụng theo Nghị quyết 954/2020/UBTVQH14)
    GIAM_TRU_BAN_THAN = 11_000_000.0  # 11 triệu VNĐ/tháng
    GIAM_TRU_NGUOI_PHU_THUOC = 4_400_000.0  # 4.4 triệu VNĐ/tháng/người

    # Biểu thuế lũy tiến từng phần thuế TNCN từ tiền lương, tiền công (7 bậc)
    BAC_THUE_TNCN = [
        (5_000_000.0, 0.05, 0.0),
        (10_000_000.0, 0.10, 250_000.0),
        (18_000_000.0, 0.15, 750_000.0),
        (32_000_000.0, 0.20, 1_650_000.0),
        (52_000_000.0, 0.25, 3_250_000.0),
        (80_000_000.0, 0.30, 5_850_000.0),
        (float("inf"), 0.35, 9_850_000.0),
    ]

    TU_DIEN_VAS = {
        "TSCĐ": "Tài sản cố định - Tư liệu lao động có giá trị từ 30 triệu đồng trở lên và thời gian sử dụng trên 1 năm.",
        "GTGT": "Thuế Giá trị gia tăng (VAT) - Thuế gián thu tính trên giá trị tăng thêm của hàng hóa, dịch vụ.",
        "TNCN": "Thuế Thu nhập cá nhân - Thuế trực thu tính trên thu nhập chịu thuế của cá nhân cư trú hoặc không cư trú.",
        "TNDN": "Thuế Thu nhập doanh nghiệp - Thuế trực thu tính trên thu nhập chịu thuế của doanh nghiệp (mức chuẩn: 20%).",
        "CQT": "Cơ quan thuế (Cục Thuế, Chi cục Thuế quản lý trực tiếp doanh nghiệp).",
        "HĐĐT": "Hóa đơn điện tử - Hóa đơn thể hiện dưới dạng dữ liệu điện tử có mã hoặc không có mã của CQT.",
        "MST": "Mã số thuế - Dãy số gồm 10 hoặc 13 chữ số do cơ quan thuế cấp cho người nộp thuế.",
        "BHXH": "Bảo hiểm xã hội bắt buộc (tỷ lệ người lao động đóng thường là 8% mức lương đóng bảo hiểm).",
        "BHYT": "Bảo hiểm y tế (người lao động đóng 1.5%).",
        "BHTN": "Bảo hiểm thất nghiệp (người lao động đóng 1%).",
    }

    # ==================== PHẦN CÔNG CỤ (TOOLS) ====================

    def dang_ky_cong_cu(self) -> List[Dict[str, Any]]:
        """Đăng ký các công cụ tính toán tài chính cho Tác tử (TacTu)."""
        return [
            {
                "ten": "tinh_thue_tncn",
                "mo_ta": "Tính thuế thu nhập cá nhân (TNCN) theo biểu thuế lũy tiến từng phần 7 bậc của Việt Nam",
                "ham": self.tinh_thue_tncn,
                "tham_so": {
                    "thu_nhap_thang": {"kieu": "float", "bat_buoc": True},
                    "so_nguoi_phu_thuoc": {"kieu": "int", "bat_buoc": False},
                    "bao_hiem_bat_buoc": {"kieu": "float", "bat_buoc": False},
                },
            },
            {
                "ten": "tinh_thue_tndn",
                "mo_ta": "Tính thuế thu nhập doanh nghiệp (TNDN) dựa trên doanh thu và chi phí hợp lý",
                "ham": self.tinh_thue_tndn,
                "tham_so": {
                    "doanh_thu": {"kieu": "float", "bat_buoc": True},
                    "chi_phi_hop_ly": {"kieu": "float", "bat_buoc": True},
                    "thue_suat": {"kieu": "float", "bat_buoc": False},
                },
            },
            {
                "ten": "boc_tach_hoa_don",
                "mo_ta": "Trích xuất thông tin trọng yếu từ hóa đơn điện tử (Số HĐ, Ngày, MST bên bán/mua, Tiền hàng, VAT)",
                "ham": self.boc_tach_hoa_don,
                "tham_so": {
                    "noi_dung_hoa_don": {"kieu": "str", "bat_buoc": True},
                },
            },
            {
                "ten": "tinh_khau_hao_tscd",
                "mo_ta": "Tính khấu hao tài sản cố định theo phương pháp đường thẳng (Thông tư 45/2013/TT-BTC)",
                "ham": self.tinh_khau_hao_tscd,
                "tham_so": {
                    "nguyen_gia": {"kieu": "float", "bat_buoc": True},
                    "so_nam_su_dung": {"kieu": "float", "bat_buoc": True},
                    "gia_tri_thanh_ly": {"kieu": "float", "bat_buoc": False},
                },
            },
        ]

    def tinh_thue_tncn(
        self,
        thu_nhap_thang: float,
        so_nguoi_phu_thuoc: int = 0,
        bao_hiem_bat_buoc: float = 0.0,
    ) -> Dict[str, Any]:
        """Tính thuế thu nhập cá nhân theo biểu lũy tiến từng phần 7 bậc."""
        tong_giam_tru = (
            self.GIAM_TRU_BAN_THAN
            + (so_nguoi_phu_thuoc * self.GIAM_TRU_NGUOI_PHU_THUOC)
            + bao_hiem_bat_buoc
        )
        thu_nhap_tinh_thue = max(0.0, thu_nhap_thang - tong_giam_tru)

        thue_phai_nop = 0.0
        bac_ap_dung = 0
        if thu_nhap_tinh_thue > 0:
            if thu_nhap_tinh_thue <= 5_000_000:
                thue_phai_nop = thu_nhap_tinh_thue * 0.05
                bac_ap_dung = 1
            elif thu_nhap_tinh_thue <= 10_000_000:
                thue_phai_nop = thu_nhap_tinh_thue * 0.10 - 250_000
                bac_ap_dung = 2
            elif thu_nhap_tinh_thue <= 18_000_000:
                thue_phai_nop = thu_nhap_tinh_thue * 0.15 - 750_000
                bac_ap_dung = 3
            elif thu_nhap_tinh_thue <= 32_000_000:
                thue_phai_nop = thu_nhap_tinh_thue * 0.20 - 1_650_000
                bac_ap_dung = 4
            elif thu_nhap_tinh_thue <= 52_000_000:
                thue_phai_nop = thu_nhap_tinh_thue * 0.25 - 3_250_000
                bac_ap_dung = 5
            elif thu_nhap_tinh_thue <= 80_000_000:
                thue_phai_nop = thu_nhap_tinh_thue * 0.30 - 5_850_000
                bac_ap_dung = 6
            else:
                thue_phai_nop = thu_nhap_tinh_thue * 0.35 - 9_850_000
                bac_ap_dung = 7

        thue_phai_nop = max(0.0, thue_phai_nop)
        thu_nhap_thuc_nhan = thu_nhap_thang - bao_hiem_bat_buoc - thue_phai_nop

        return {
            "thu_nhap_thang": thu_nhap_thang,
            "tong_giam_tru": tong_giam_tru,
            "thu_nhap_tinh_thue": thu_nhap_tinh_thue,
            "bac_thue_cao_nhat": bac_ap_dung,
            "thue_tncn_phai_nop": round(thue_phai_nop, 0),
            "thu_nhap_thuc_nhan": round(thu_nhap_thuc_nhan, 0),
        }

    def tinh_thue_tndn(
        self,
        doanh_thu: float,
        chi_phi_hop_ly: float,
        thue_suat: float = 0.20,
    ) -> Dict[str, Any]:
        """Tính thuế thu nhập doanh nghiệp."""
        thu_nhap_tinh_thue = max(0.0, doanh_thu - chi_phi_hop_ly)
        thue_phai_nop = thu_nhap_tinh_thue * thue_suat
        loi_nhuan_sau_thue = (doanh_thu - chi_phi_hop_ly) - thue_phai_nop

        return {
            "doanh_thu": doanh_thu,
            "chi_phi_hop_ly": chi_phi_hop_ly,
            "thu_nhap_tinh_thue": thu_nhap_tinh_thue,
            "thue_suat": thue_suat,
            "thue_tndn_phai_nop": round(thue_phai_nop, 0),
            "loi_nhuan_sau_thue": round(loi_nhuan_sau_thue, 0),
        }

    def boc_tach_hoa_don(self, noi_dung_hoa_don: str) -> Dict[str, Any]:
        """Trích xuất các trường dữ liệu quan trọng của hóa đơn điện tử."""
        text = noi_dung_hoa_don.strip()

        # Số hóa đơn (VD: 0001234, HĐ00129)
        so_hd_m = re.search(r"(?:Số(?:\s*hóa\s*đơn|\s*HĐ)?\s*[:#]?\s*)([A-Z0-9_-]+)", text, re.IGNORECASE)
        so_hd = so_hd_m.group(1) if so_hd_m else "KHÔNG_XÁC_ĐỊNH"

        # Mã số thuế bên bán
        mst_m = re.findall(r"(?:MST|Mã\s*số\s*thuế)\s*[:#]?\s*(\d{10}(?:-\d{3})?)", text, re.IGNORECASE)
        mst_ban = mst_m[0] if mst_m else "KHÔNG_CÓ"
        mst_mua = mst_m[1] if len(mst_m) > 1 else "KHÔNG_CÓ"

        # Tổng tiền thanh toán
        tien_m = re.findall(r"([\d.,]+)\s*(?:VND|VNĐ|đồng)", text, re.IGNORECASE)
        tien_gia_tri = []
        for raw_val in tien_m:
            clean = raw_val.replace(".", "").replace(",", "")
            if clean.isdigit():
                tien_gia_tri.append(float(clean))

        tong_tien = max(tien_gia_tri) if tien_gia_tri else 0.0

        return {
            "so_hoa_don": so_hd,
            "mst_nguoi_ban": mst_ban,
            "mst_nguoi_mua": mst_mua,
            "tong_tien_thanh_toan": tong_tien,
            "hop_le": so_hd != "KHÔNG_XÁC_ĐỊNH" and mst_ban != "KHÔNG_CÓ",
        }

    def tinh_khau_hao_tscd(
        self,
        nguyen_gia: float,
        so_nam_su_dung: float,
        gia_tri_thanh_ly: float = 0.0,
    ) -> Dict[str, Any]:
        """Tính mức trích khấu hao hàng năm và hàng tháng theo Thông tư 45/2013."""
        if so_nam_su_dung <= 0:
            raise ValueError("Số năm sử dụng phải lớn hơn 0")

        gia_tri_tinh_khau_hao = max(0.0, nguyen_gia - gia_tri_thanh_ly)
        khau_hao_nam = gia_tri_tinh_khau_hao / so_nam_su_dung
        khau_hao_thang = khau_hao_nam / 12.0

        return {
            "nguyen_gia": nguyen_gia,
            "so_nam_su_dung": so_nam_su_dung,
            "gia_tri_tinh_khau_hao": gia_tri_tinh_khau_hao,
            "muc_khau_hao_nam": round(khau_hao_nam, 0),
            "muc_khau_hao_thang": round(khau_hao_thang, 0),
        }

    # ==================== PHẦN TRI THỨC (KNOWLEDGE / RAG) ====================

    def nap_nguon_tri_thuc(self) -> List[Dict[str, Any]]:
        """Nạp các văn bản quy phạm pháp luật và sổ tay nghiệp vụ kế toán thuế."""
        return [
            {
                "tieu_de": "Quy định về biểu thuế lũy tiến thuế Thu nhập cá nhân",
                "noi_dung": (
                    "Theo Luật Thuế thu nhập cá nhân và Nghị quyết số 954/2020/UBTVQH14, "
                    "mức giảm trừ gia cảnh cho bản thân người nộp thuế là 11 triệu đồng/tháng (132 triệu đồng/năm); "
                    "mức giảm trừ cho mỗi người phụ thuộc là 4,4 triệu đồng/tháng. "
                    "Thuế suất lũy tiến áp dụng từ bậc 1 (5% cho phần thu nhập tính thuế đến 5 triệu) "
                    "đến bậc 7 (35% cho phần thu nhập tính thuế trên 80 triệu đồng/tháng)."
                ),
                "nguon": "Luật Thuế TNCN & Nghị quyết 954/2020/UBTVQH14",
                "metadata": {"chu_de": "thue_tncn", "co_quan": "Quốc hội"},
            },
            {
                "tieu_de": "Khung thời gian trích khấu hao các loại tài sản cố định",
                "noi_dung": (
                    "Theo Thông tư 45/2013/TT-BTC, tài sản cố định phải đáp ứng đồng thời 3 tiêu chuẩn: "
                    "chắc chắn thu được lợi ích kinh tế trong tương lai; có thời gian sử dụng trên 1 năm; "
                    "nguyên giá từ 30 triệu đồng trở lên. "
                    "Phương pháp trích khấu hao thông dụng là phương pháp khấu hao đường thẳng: "
                    "Mức trích khấu hao trung bình hàng năm bằng Nguyên giá chia cho Thời gian trích khấu hao."
                ),
                "nguon": "Thông tư số 45/2013/TT-BTC",
                "metadata": {"chu_de": "khau_hao_tscd", "co_quan": "Bộ Tài chính"},
            },
            {
                "tieu_de": "Quy định về tính hợp pháp của Hóa đơn điện tử",
                "noi_dung": (
                    "Theo Nghị định 123/2020/NĐ-CP và Thông tư 78/2021/TT-BTC, hóa đơn điện tử hợp pháp phải "
                    "có đầy đủ các nội dung: Tên hóa đơn, ký hiệu mẫu số hóa đơn, ký hiệu hóa đơn, số hóa đơn; "
                    "Tên, địa chỉ, mã số thuế của người bán; Tên, địa chỉ, mã số thuế của người mua (nếu có); "
                    "Tên, đơn vị tính, số lượng, đơn giá hàng hóa, dịch vụ; Chữ ký số của người bán và người mua."
                ),
                "nguon": "Nghị định 123/2020/NĐ-CP & Thông tư 78/2021/TT-BTC",
                "metadata": {"chu_de": "hoa_don_dien_tu", "co_quan": "Chính phủ"},
            },
        ]

    def lay_tu_dien_nghiep_vu(self) -> Dict[str, str]:
        """Cung cấp bảng từ điển thuật ngữ kế toán VAS chuẩn."""
        return self.TU_DIEN_VAS
