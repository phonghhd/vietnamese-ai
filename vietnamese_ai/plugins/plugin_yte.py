"""
Plugin Y Tế - Dược Khoa & Sức Khỏe (HealthTech & Medical Plugin cho Vietnamese AI Framework).
Cung cấp công cụ tra cứu Dược thư Quốc gia Việt Nam, tra cứu danh mục mã bệnh ICD-10 tiếng Việt,
sàng lọc phân loại sơ bộ triệu chứng và tri thức y tế cộng đồng.
"""

from __future__ import annotations

from typing import Any, Dict, List

from vietnamese_ai.plugin.plugin_base import PluginCongCu, PluginTriThuc


class PluginYTe(PluginCongCu, PluginTriThuc):
    """Plugin Y tế, Dược khoa và Sức khỏe cộng đồng."""

    ten: str = "Y Tế & Dược Khoa"
    phien_ban: str = "1.0.0"
    mo_ta: str = "Tra cứu Dược thư Quốc gia, mã bệnh ICD-10 và sàng lọc sơ bộ triệu chứng"
    loai: str = "y_te"

    # Cơ sở dữ liệu Dược thư Quốc gia tóm tắt (tham khảo Dược thư Việt Nam)
    CO_SO_DU_LIEU_THUOC: Dict[str, Dict[str, Any]] = {
        "paracetamol": {
            "hoat_chat": "Paracetamol (Acetaminophen)",
            "nhom_duoc_ly": "Thuốc giảm đau, hạ sốt không opioid",
            "chi_dinh": "Hạ sốt, giảm các cơn đau nhẹ đến vừa (đau đầu, đau răng, đau cơ)",
            "lieu_dung_chuan": "Người lớn: 500mg - 1000mg/lần, cách nhau 4-6 giờ; tối đa không quá 4g/ngày.",
            "chong_chi_dinh": "Người mẫn cảm với paracetamol, người suy gan nặng.",
            "luu_y_dac_biet": "Không uống rượu bia khi dùng thuốc vì tăng độc tính trên gan.",
        },
        "amoxicillin": {
            "hoat_chat": "Amoxicillin",
            "nhom_duoc_ly": "Kháng sinh nhóm Beta-lactam (Penicillin)",
            "chi_dinh": "Nhiễm khuẩn đường hô hấp trên và dưới, nhiễm khuẩn tiết niệu, da.",
            "lieu_dung_chuan": "Theo chỉ định của bác sĩ. Liều thường dùng người lớn: 500mg/lần x 3 lần/ngày.",
            "chong_chi_dinh": "Dị ứng với bất kỳ penicillin hoặc cephalosporin nào.",
            "luu_y_dac_biet": "Cần dùng đủ liệu trình kháng sinh (thường 5-7 ngày) để tránh kháng thuốc.",
        },
        "ibuprofen": {
            "hoat_chat": "Ibuprofen",
            "nhom_duoc_ly": "Thuốc chống viêm không steroid (NSAID)",
            "chi_dinh": "Giảm đau, kháng viêm trong viêm khớp, đau răng, đau bụng kinh.",
            "lieu_dung_chuan": "Người lớn: 200mg - 400mg/lần, sau ăn no.",
            "chong_chi_dinh": "Loét dạ dày tá tràng tiến triển, suy gan thận nặng, phụ nữ mang thai 3 tháng cuối.",
            "luu_y_dac_biet": "Uống sau khi ăn no để hạn chế kích ứng niêm mạc dạ dày.",
        },
        "berberin": {
            "hoat_chat": "Berberin clorid",
            "nhom_duoc_ly": "Thuốc điều trị tiêu chảy có nguồn gốc thảo dược",
            "chi_dinh": "Hội chứng lỵ, viêm ruột, tiêu chảy do nhiễm khuẩn đường tiêu hóa.",
            "lieu_dung_chuan": "Người lớn: 2 - 4 viên (50mg) x 2 lần/ngày.",
            "chong_chi_dinh": "Phụ nữ có thai (do nguy cơ co bóp tử cung gây sảy thai).",
            "luu_y_dac_biet": "Nên bổ sung oresol bù nước điện giải kèm theo.",
        },
    }

    # Danh mục mã bệnh ICD-10 tiếng Việt phổ biến
    DANH_MUC_ICD10: List[Dict[str, str]] = [
        {"ma": "J00", "ten": "Viêm mũi họng cấp (Cảm thường)"},
        {"ma": "J02", "ten": "Viêm họng cấp"},
        {"ma": "J18", "ten": "Viêm phổi, tác nhân không xác định"},
        {"ma": "J45", "ten": "Hen phế quản (Suyễn)"},
        {"ma": "E11", "ten": "Đái tháo đường không phụ thuộc insulin (Type 2)"},
        {"ma": "I10", "ten": "Tăng huyết áp vô căn (nguyên phát)"},
        {"ma": "K21", "ten": "Bệnh trào ngược dạ dày - thực quản (GERD)"},
        {"ma": "K29", "ten": "Viêm và loét dạ dày - tá tràng"},
        {"ma": "A09", "ten": "Tiêu chảy và viêm dạ dày ruột do nhiễm trùng"},
        {"ma": "M54.5", "ten": "Đau lưng vùng thấp (Đau thắt lưng)"},
    ]

    TU_DIEN_Y_TE = {
        "HA": "Huyết áp (Huyết áp tâm thu / Huyết áp tâm trương).",
        "ĐTĐ": "Đái tháo đường (bệnh tiểu đường).",
        "THA": "Tăng huyết áp.",
        "NMCT": "Nhồi máu cơ tim - Tình trạng cấp cứu khẩn cấp do tắc mạch vành.",
        "COPD": "Bệnh phổi tắc nghẽn mạn tính.",
        "ICD-10": "Phân loại thống kê quốc tế về các bệnh tật và vấn đề sức khỏe liên quan (ấn bản 10).",
        "NSAID": "Nhóm thuốc chống viêm không steroid (như Ibuprofen, Aspirin).",
        "OTC": "Thuốc không cần kê đơn (Over-The-Counter).",
    }

    # ==================== PHẦN CÔNG CỤ (TOOLS) ====================

    def dang_ky_cong_cu(self) -> List[Dict[str, Any]]:
        """Đăng ký các công cụ tra cứu y tế cho Tác tử (TacTu)."""
        return [
            {
                "ten": "tra_cuu_thuoc",
                "mo_ta": "Tra cứu thông tin hoạt chất, chỉ định, chống chỉ định và liều dùng trong Dược thư Quốc gia",
                "ham": self.tra_cuu_thuoc,
                "tham_so": {"ten_thuoc": {"kieu": "str", "bat_buoc": True}},
            },
            {
                "ten": "tra_ma_icd10",
                "mo_ta": "Tra cứu mã bệnh hoặc tên bệnh lý theo danh mục ICD-10 tiếng Việt của Bộ Y tế",
                "ham": self.tra_ma_icd10,
                "tham_so": {"truy_van": {"kieu": "str", "bat_buoc": True}},
            },
            {
                "ten": "sang_loc_trieu_chung",
                "mo_ta": "Phân loại sơ bộ mức độ nghiêm trọng của triệu chứng và cảnh báo dấu hiệu khẩn cấp",
                "ham": self.sang_loc_trieu_chung,
                "tham_so": {"trieu_chung": {"kieu": "str", "bat_buoc": True}},
            },
        ]

    def tra_cuu_thuoc(self, ten_thuoc: str) -> Dict[str, Any]:
        """Tra cứu thuốc trong Dược thư Quốc gia."""
        key = ten_thuoc.strip().lower()
        # Tìm chính xác hoặc tương đối
        for k, thong_tin in self.CO_SO_DU_LIEU_THUOC.items():
            if k in key or key in k:
                return {
                    "tim_thay": True,
                    "ten_tra_cuu": ten_thuoc,
                    **thong_tin,
                }

        return {
            "tim_thay": False,
            "ten_tra_cuu": ten_thuoc,
            "thong_bao": (
                f"Chưa tìm thấy thông tin chi tiết cho '{ten_thuoc}' trong Dược thư tóm tắt. "
                "Khuyến cáo tham vấn ý kiến dược sĩ hoặc bác sĩ chuyên khoa."
            ),
        }

    def tra_ma_icd10(self, truy_van: str) -> List[Dict[str, str]]:
        """Tìm kiếm mã bệnh hoặc tên bệnh theo chuẩn ICD-10 tiếng Việt."""
        q = truy_van.strip().lower()
        ket_qua = []
        for item in self.DANH_MUC_ICD10:
            if q in item["ma"].lower() or q in item["ten"].lower():
                ket_qua.append(item)
        return ket_qua

    def sang_loc_trieu_chung(self, trieu_chung: str) -> Dict[str, Any]:
        """
        Sàng lọc dấu hiệu cảnh báo đỏ (Red Flags) và hướng dẫn phân luồng y tế ban đầu.
        """
        text = trieu_chung.lower()

        # Dấu hiệu cấp cứu khẩn cấp (Red flags)
        co_do = [
            "khó thở",
            "tức ngực",
            "đau thắt ngực",
            "co giật",
            "hôn mê",
            "bất tỉnh",
            "nôn ra máu",
            "liệt",
            "méo miệng",
            "sốt cao liên tục không hạ",
        ]

        phat_hien_nguy_hiem = [cd for cd in co_do if cd in text]

        if phat_hien_nguy_hiem:
            muc_do = "KHẨN_CẤP"
            khuyen_nghi = (
                f"CẢNH BÁO: Phát hiện triệu chứng nguy hiểm: {', '.join(phat_hien_nguy_hiem)}. "
                "Cần gọi ngay cấp cứu 115 hoặc đến cơ sở y tế gần nhất lập tức!"
            )
        elif any(k in text for k in ["sốt", "đau đầu", "ho kéo dài", "đau bụng dữ dội"]):
            muc_do = "CẦN_ĐI_KHÁM"
            khuyen_nghi = (
                "Triệu chứng cần được bác sĩ thăm khám trực tiếp để chẩn đoán nguyên nhân và kê đơn thuốc."
            )
        else:
            muc_do = "THEO_DÕI_TẠI_NHÀ"
            khuyen_nghi = (
                "Theo dõi triệu chứng tại nhà, nghỉ ngơi, uống đủ nước và đi khám nếu tình trạng xấu đi sau 48 giờ."
            )

        return {
            "trieu_chung_nhan": trieu_chung,
            "muc_do_nghiem_trong": muc_do,
            "dau_hieu_nguy_hiem": phat_hien_nguy_hiem,
            "khuyen_nghi_chuyen_mon": khuyen_nghi,
            "luu_y_phap_ly": "Đây là gợi ý sàng lọc ban đầu của trợ lý AI, không thay thế cho chẩn đoán y khoa chính thức.",
        }

    # ==================== PHẦN TRI THỨC (KNOWLEDGE / RAG) ====================

    def nap_nguon_tri_thuc(self) -> List[Dict[str, Any]]:
        """Cung cấp các hướng dẫn y tế và khuyến cáo sử dụng thuốc an toàn của Bộ Y tế."""
        return [
            {
                "tieu_de": "Khuyến cáo an toàn khi sử dụng thuốc hạ sốt giảm đau Paracetamol",
                "noi_dung": (
                    "Paracetamol là thuốc hạ sốt giảm đau phổ biến nhưng có thể gây hoại tử tế bào gan nếu quá liều. "
                    "Khoảng cách giữa hai lần uống tối thiểu là 4 đến 6 giờ. Liều tối đa mỗi ngày không quá 4g đối với "
                    "người lớn có chức năng gan bình thường. Tuyệt đối không dùng đồng thời nhiều chế phẩm có cùng hoạt chất "
                    "paracetamol (như vừa uống sủi, vừa uống viên nén hoặc thuốc cảm cúm hỗn hợp) để tránh ngộ độc tích lũy."
                ),
                "nguon": "Hướng dẫn sử dụng thuốc an toàn - Bộ Y tế",
                "metadata": {"chu_de": "duoc_thu", "loai_thuoc": "ha_sot"},
            },
            {
                "tieu_de": "Nguyên tắc sử dụng kháng sinh hợp lý và chống kháng thuốc",
                "noi_dung": (
                    "Theo Quyết định số 4068/QĐ-BYT của Bộ Y tế: Kháng sinh chỉ được sử dụng khi có bằng chứng hoặc "
                    "nghi ngờ cao về nhiễm khuẩn, không dùng kháng sinh để điều trị cảm lạnh thông thường do virus gây ra. "
                    "Khi được kê đơn kháng sinh, bệnh nhân cần tuân thủ đủ liều lượng và thời gian (thường 5-7 ngày), "
                    "không tự ý ngừng thuốc khi triệu chứng vừa thuyên giảm nhằm ngăn chặn vi khuẩn kháng thuốc."
                ),
                "nguon": "Quyết định 4068/QĐ-BYT - Hướng dẫn sử dụng kháng sinh",
                "metadata": {"chu_de": "khang_sinh", "co_quan": "Bộ Y tế"},
            },
            {
                "tieu_de": "Hướng dẫn xử trí ban đầu khi trẻ bị sốt cao co giật",
                "noi_dung": (
                    "Khi trẻ nhỏ bị sốt cao co giật, cha mẹ cần bình tĩnh đặt trẻ nằm nghiêng sang một bên trên mặt phẳng an toàn; "
                    "nới lỏng quần áo để trẻ thông thoáng đường thở. Tuyệt đối không nhét bất kỳ vật cứng, ngón tay hay khăn vào miệng trẻ. "
                    "Dùng khăn ấm lau các vùng nách, bẹn để hạ nhiệt độ; sau đó nhanh chóng đưa trẻ đến cơ sở y tế gần nhất."
                ),
                "nguon": "Hướng dẫn Chăm sóc trẻ em - Bệnh viện Nhi Trung ương",
                "metadata": {"chu_de": "so_cuu", "doi_tuong": "tre_em"},
            },
        ]

    def lay_tu_dien_nghiep_vu(self) -> Dict[str, str]:
        """Cung cấp từ điển thuật ngữ y tế phổ biến."""
        return self.TU_DIEN_Y_TE
