"""BoMauPromptPhapLy - Thư Viện Mẫu Prompt Chuyên Biệt Cho Lĩnh Vực Pháp Lý & Hành Chính Việt Nam."""

from __future__ import annotations

from typing import Any, Dict, List

from vietnamese_ai.prompts.templates import MauPrompt


class BoMauPromptPhapLy:
    """
    Thư viện các mẫu prompt (Prompt Templates) chuyên sâu cho văn bản quy phạm pháp luật,
    thủ tục hành chính nhà nước và rà soát hợp đồng tại Việt Nam.
    """

    def __init__(self) -> None:
        self._mau: Dict[str, MauPrompt] = self._khoi_tao_mau_mac_dinh()

    @staticmethod
    def _khoi_tao_mau_mac_dinh() -> Dict[str, MauPrompt]:
        """Khởi tạo bộ 6 mẫu prompt chuyên biệt lĩnh vực pháp lý/hành chính Việt Nam."""
        return {
            "tra_cuu_dieu_khoan": MauPrompt(
                ten="tra_cuu_dieu_khoan",
                mo_ta="Trả lời câu hỏi pháp lý có viện dẫn chính xác Điều, Khoản, Điểm và số hiệu văn bản.",
                mau=(
                    "Bạn là trợ lý pháp lý Việt Nam. Chỉ được sử dụng thông tin trong phần Căn cứ pháp lý dưới đây "
                    "để trả lời, không tự suy diễn ngoài văn bản.\n\n"
                    "Căn cứ pháp lý truy hồi được:\n{{ngu_canh_phap_ly}}\n\n"
                    "Câu hỏi của người dùng:\n{{cau_hoi}}\n\n"
                    "Yêu cầu trình bày:\n"
                    "1. Kết luận trực tiếp cho câu hỏi.\n"
                    "2. Viện dẫn rõ Điều, Khoản, Điểm và số hiệu văn bản liên quan.\n"
                    "3. Nếu ngữ cảnh không đủ căn cứ, nêu rõ 'Chưa đủ căn cứ pháp lý trong tài liệu được cung cấp'.\n"
                    "4. {{yeu_cau_bo_sung}}"
                ),
                bien_mac_dinh={
                    "yeu_cau_bo_sung": "Sử dụng văn phong hành chính - pháp lý rõ ràng, khách quan."
                },
            ),
            "ra_soat_hop_dong": MauPrompt(
                ten="ra_soat_hop_dong",
                mo_ta="Rà soát rủi ro pháp lý, điều khoản bất lợi và thiếu sót trong hợp đồng thương mại/lao động.",
                mau=(
                    "Hãy rà soát dự thảo {{loai_hop_dong}} dưới góc độ pháp luật Việt Nam.\n\n"
                    "Trọng tâm rà soát: {{trong_tam_ra_soat}}\n\n"
                    "Nội dung hợp đồng:\n{{noi_dung_hop_dong}}\n\n"
                    "Hãy phân tích theo cấu trúc:\n"
                    "1. Các điều khoản rủi ro cao hoặc mâu thuẫn pháp luật.\n"
                    "2. Nghĩa vụ, chế tài phạt vi phạm và giới hạn bồi thường thiệt hại.\n"
                    "3. Điều khoản cần bổ sung hoặc đề xuất chỉnh sửa câu chữ cụ thể."
                ),
                bien_mac_dinh={
                    "loai_hop_dong": "hợp đồng kinh tế / thương mại",
                    "trong_tam_ra_soat": "quyền và nghĩa vụ các bên, thanh toán, bảo mật, chấm dứt và giải quyết tranh chấp",
                },
            ),
            "tom_tat_van_ban_phap_ly": MauPrompt(
                ten="tom_tat_van_ban_phap_ly",
                mo_ta="Tóm tắt văn bản quy phạm pháp luật hoặc văn bản hành chính theo khung chuẩn.",
                mau=(
                    "Hãy tóm tắt văn bản hành chính/pháp lý sau đây: {{ten_van_ban}}\n\n"
                    "Nội dung văn bản:\n{{noi_dung_van_ban}}\n\n"
                    "Xuất kết quả gồm các mục:\n"
                    "- Số hiệu & Cơ quan ban hành\n"
                    "- Đối tượng áp dụng & Phạm vi điều chỉnh\n"
                    "- Các quy định trọng tâm (liệt kê theo Điều/Khoản)\n"
                    "- Hiệu lực thi hành & lưu ý thực thi"
                ),
                bien_mac_dinh={"ten_van_ban": "Văn bản pháp lý Việt Nam"},
            ),
            "so_sanh_quy_dinh": MauPrompt(
                ten="so_sanh_quy_dinh",
                mo_ta="So sánh điểm mới giữa hai phiên bản quy định pháp luật hoặc hai văn bản liên quan.",
                mau=(
                    "So sánh quy định pháp lý về chủ đề: {{chu_de}}\n\n"
                    "Quy định cũ / đối chiếu:\n{{quy_dinh_cu}}\n\n"
                    "Quy định mới:\n{{quy_dinh_moi}}\n\n"
                    "Hãy lập bảng so sánh gồm: Nội dung thay đổi | Quy định cũ | Quy định mới | Tác động tuân thủ."
                ),
                bien_mac_dinh={"chu_de": "Quy định pháp luật hiện hành"},
            ),
            "soan_thao_cong_van": MauPrompt(
                ten="soan_thao_cong_van",
                mo_ta="Soạn thảo công văn, tờ trình hoặc thông báo đúng thể thức Nghị định 30/2020/NĐ-CP.",
                mau=(
                    "Soạn thảo {{loai_van_ban}} tuân thủ thể thức văn bản hành chính Việt Nam (Nghị định 30/2020/NĐ-CP).\n\n"
                    "- Cơ quan ban hành: {{co_quan_ban_hanh}}\n"
                    "- Kính gửi / Nơi nhận: {{noi_nhan}}\n"
                    "- Trích yếu nội dung: {{trich_yeu}}\n"
                    "- Ý chính cần truyền đạt:\n{{noi_dung_chinh}}\n\n"
                    "Trình bày đầy đủ quốc hiệu, tiêu ngữ, trích yếu, nội dung và nơi nhận."
                ),
                bien_mac_dinh={"loai_van_ban": "Công văn hành chính"},
            ),
            "kiem_tra_tuan_thu_du_lieu": MauPrompt(
                ten="kiem_tra_tuan_thu_du_lieu",
                mo_ta="Đánh giá mức độ tuân thủ bảo vệ dữ liệu cá nhân (Nghị định 13/2023/NĐ-CP) cho hệ thống AI.",
                mau=(
                    "Đánh giá tuân thủ bảo vệ dữ liệu cá nhân theo pháp luật Việt Nam.\n\n"
                    "Căn cứ quy định liên quan:\n{{ngu_canh_phap_ly}}\n\n"
                    "Mô tả quy trình / hệ thống xử lý dữ liệu:\n{{mo_ta_he_thong}}\n\n"
                    "Phân tích rõ:\n"
                    "1. Loại dữ liệu cá nhân (cơ bản / nhạy cảm).\n"
                    "2. Yêu cầu về sự đồng ý của chủ thể dữ liệu và hồ sơ đánh giá tác động.\n"
                    "3. Các khoảng trống tuân thủ và biện pháp khắc phục khuyến nghị."
                ),
            ),
        }

    def danh_sach_mau(self) -> List[Dict[str, Any]]:
        """Liệt kê tất cả mẫu prompt pháp lý kèm mô tả và danh sách biến yêu cầu."""
        return [
            {
                "ten": ten,
                "mo_ta": mau.mo_ta,
                "bien": sorted(mau.danh_sach_bien()),
                "bien_mac_dinh": dict(mau.bien_mac_dinh),
            }
            for ten, mau in self._mau.items()
        ]

    def lay_mau(self, ten_mau: str) -> MauPrompt:
        """Lấy đối tượng `MauPrompt` theo tên mẫu."""
        if ten_mau not in self._mau:
            ds = ", ".join(sorted(self._mau.keys()))
            raise KeyError(f"Không tìm thấy mẫu prompt pháp lý '{ten_mau}'. Các mẫu khả dụng: {ds}")
        return self._mau[ten_mau]

    def them_mau(self, ten_mau: str, mau_prompt: MauPrompt) -> None:
        """Đăng ký thêm hoặc ghi đè một mẫu prompt pháp lý tùy chỉnh."""
        self._mau[ten_mau] = mau_prompt

    def tao_prompt(self, ten_mau: str, **kwargs: Any) -> str:
        """Render prompt hoàn chỉnh từ tên mẫu và các tham số biến."""
        return self.lay_mau(ten_mau).render(**kwargs)

    def tao_prompt_tu_truy_hoi(
        self,
        cau_hoi: str,
        ket_qua_truy_hoi: List[Dict[str, Any]],
        ten_mau: str = "tra_cuu_dieu_khoan",
        **kwargs: Any,
    ) -> str:
        """Đóng gói danh sách đoạn truy hồi từ `BoToiUuNguCanhHanhChinh.truy_hoi_phap_ly` thành prompt có trích dẫn."""
        cac_doan: List[str] = []
        for idx, muc in enumerate(ket_qua_truy_hoi, start=1):
            trich_dan = muc.get("trich_dan_phap_ly", f"Nguồn {idx}")
            ngu_canh = muc.get("ngu_canh") or muc.get("noi_dung", "")
            diem = muc.get("diem_tuong_dong")
            diem_str = f" [độ liên quan: {diem}]" if diem is not None else ""
            cac_doan.append(f"[{idx}] {trich_dan}{diem_str}\n{ngu_canh}")

        ngu_canh_gop = "\n\n---\n\n".join(cac_doan) if cac_doan else "(Không tìm thấy điều khoản phù hợp)"
        tham_so: Dict[str, Any] = {
            "cau_hoi": cau_hoi,
            "ngu_canh_phap_ly": ngu_canh_gop,
            **kwargs,
        }
        return self.tao_prompt(ten_mau, **tham_so)


LegalPromptLibrary = BoMauPromptPhapLy

__all__ = [
    "BoMauPromptPhapLy",
    "LegalPromptLibrary",
]
