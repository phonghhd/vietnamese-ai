"""Trợ lý Pháp lý & Doanh nghiệp - Vietnamese AI Framework."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vietnamese_ai.rag.rag_pipeline import RAGPipeline
from vietnamese_ai.security.data_defense import CongAnToanDauRa
from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework


class TroLyPhapLy:
    """
    Trợ lý Pháp lý & Doanh nghiệp thông minh cho hệ thống luật Việt Nam:
    - Tra cứu chính xác các điều khoản trong Bộ luật Lao động, Bộ luật Dân sự, Luật Doanh nghiệp, Luật Thuế.
    - Tự động trích dẫn số hiệu điều, khoản và tên văn bản luật làm căn cứ pháp lý.
    - Tích hợp Tường lửa AI kiểm duyệt câu hỏi và Cổng an toàn chấm điểm bám sát nguồn (chống ảo giác).
    """

    DU_LIEU_LUAT_MAC_DINH: List[Dict[str, str]] = [
        {
            "ma": "bllđ_2019_d105",
            "van_ban": "Bộ luật Lao động 2019",
            "dieu": "Điều 105",
            "tieu_de": "Thời giờ làm việc bình thường",
            "noi_dung": (
                "1. Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.\n"
                "2. Người sử dụng lao động có quyền quy định thời giờ làm việc theo ngày hoặc tuần nhưng phải thông báo cho người lao động biết; "
                "trường hợp theo tuần thì thời giờ làm việc bình thường không quá 10 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.\n"
                "3. Nhà nước khuyến khích người sử dụng lao động thực hiện tuần làm việc 40 giờ đối với người lao động."
            ),
        },
        {
            "ma": "bllđ_2019_d107",
            "van_ban": "Bộ luật Lao động 2019",
            "dieu": "Điều 107",
            "tieu_de": "Làm thêm giờ",
            "noi_dung": (
                "Người sử dụng lao động được sử dụng người lao động làm thêm giờ khi đáp ứng đầy đủ các điều kiện: "
                "Phải được sự đồng ý của người lao động; Bảo đảm số giờ làm thêm không quá 50% số giờ làm việc bình thường trong 01 ngày; "
                "không quá 40 giờ trong 01 tháng và không quá 200 giờ trong 01 năm (trừ trường hợp đặc biệt không quá 300 giờ)."
            ),
        },
        {
            "ma": "blds_2015_d119",
            "van_ban": "Bộ luật Dân sự 2015",
            "dieu": "Điều 119",
            "tieu_de": "Hình thức giao dịch dân sự",
            "noi_dung": (
                "1. Giao dịch dân sự được thể hiện bằng lời nói, bằng văn bản hoặc bằng hành vi cụ thể.\n"
                "2. Giao dịch dân sự thông qua phương tiện điện tử dưới hình thức thông điệp dữ liệu theo quy định của pháp luật về giao dịch điện tử "
                "được coi là giao dịch bằng văn bản.\n"
                "3. Trường hợp luật quy định giao dịch dân sự phải được thể hiện bằng văn bản có công chứng, chứng thực, đăng ký thì phải tuân theo quy định đó."
            ),
        },
        {
            "ma": "blds_2015_d468",
            "van_ban": "Bộ luật Dân sự 2015",
            "dieu": "Điều 468",
            "tieu_de": "Lãi suất vay",
            "noi_dung": (
                "Lãi suất vay do các bên thỏa thuận nhưng không được vượt quá 20%/năm của khoản tiền vay, "
                "trừ trường hợp luật khác có liên quan quy định khác. Trường hợp lãi suất thỏa thuận vượt quá lãi suất giới hạn quy định thì mức lãi suất vượt quá không có hiệu lực."
            ),
        },
        {
            "ma": "ldn_2020_d34",
            "van_ban": "Luật Doanh nghiệp 2020",
            "dieu": "Điều 34",
            "tieu_de": "Tài sản góp vốn",
            "noi_dung": (
                "Tài sản góp vốn là Đồng Việt Nam, ngoại tệ tự do chuyển đổi, vàng, quyền sử dụng đất, "
                "quyền sở hữu trí tuệ, công nghệ, bí quyết kỹ thuật, tài sản khác có thể định giá được bằng Đồng Việt Nam. "
                "Chỉ cá nhân, tổ chức là chủ sở hữu hợp pháp hoặc có quyền sử dụng hợp pháp đối với tài sản mới có quyền góp vốn."
            ),
        },
        {
            "ma": "ldn_2020_d74",
            "van_ban": "Luật Doanh nghiệp 2020",
            "dieu": "Điều 74",
            "tieu_de": "Công ty trách nhiệm hữu hạn một thành viên",
            "noi_dung": (
                "Công ty trách nhiệm hữu hạn một thành viên là doanh nghiệp do một tổ chức hoặc một cá nhân làm chủ sở hữu. "
                "Chủ sở hữu công ty chịu trách nhiệm về các khoản nợ và nghĩa vụ tài sản khác của công ty trong phạm vi số vốn điều lệ của công ty."
            ),
        },
        {
            "ma": "ltndn_2020_d10",
            "van_ban": "Luật Thuế Thu nhập Doanh nghiệp",
            "dieu": "Điều 10",
            "tieu_de": "Thuế suất thuế thu nhập doanh nghiệp",
            "noi_dung": (
                "Thuế suất thuế thu nhập doanh nghiệp phổ thông là 20%, trừ trường hợp doanh nghiệp thuộc đối tượng áp dụng thuế suất ưu đãi "
                "từ 10% đến 17% hoặc ngành nghề tìm kiếm, thăm dò, khai thác dầu khí và tài nguyên quý hiếm có thuế suất từ 32% đến 50%."
            ),
        },
    ]

    def __init__(self, bo_du_lieu_luat: Optional[List[Dict[str, str]]] = None):
        self.tuong_lua = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
        self.rag = RAGPipeline(nguong_diem=0.05, top_k=3)
        self.kho_tai_lieu = list(bo_du_lieu_luat or self.DU_LIEU_LUAT_MAC_DINH)
        self._nap_du_lieu_luat()

    def _nap_du_lieu_luat(self) -> None:
        """Nạp các tài liệu luật vào RAG pipeline."""
        for item in self.kho_tai_lieu:
            ma = item.get("ma", f"doc_{len(self.kho_tai_lieu)}")
            tieu_de = item.get("tieu_de", "")
            dieu = item.get("dieu", "")
            van_ban = item.get("van_ban", "")
            noi_dung = item.get("noi_dung", "")
            noi_dung_day_du = f"[{van_ban}] {dieu}: {tieu_de}\n{noi_dung}"
            self.rag.them_tai_lieu(
                ma,
                noi_dung_day_du,
                metadata={
                    "van_ban": van_ban,
                    "dieu": dieu,
                    "tieu_de": tieu_de,
                },
            )

    def them_dieu_khoan(self, ma: str, van_ban: str, dieu: str, tieu_de: str, noi_dung: str) -> None:
        """Thêm văn bản luật mới vào cơ sở tri thức."""
        item = {"ma": ma, "van_ban": van_ban, "dieu": dieu, "tieu_de": tieu_de, "noi_dung": noi_dung}
        self.kho_tai_lieu.append(item)
        full_text = f"[{van_ban}] {dieu}: {tieu_de}\n{noi_dung}"
        self.rag.them_tai_lieu(ma, full_text, metadata={"van_ban": van_ban, "dieu": dieu, "tieu_de": tieu_de})

    def tra_cuu(self, cau_hoi: str, trich_dan_chi_tiet: bool = True) -> Dict[str, Any]:
        """
        Tra cứu quy định pháp luật và sinh câu trả lời kèm trích dẫn văn bản luật chuẩn xác.
        """
        cau_hoi_sach = str(cau_hoi or "").strip()
        kt_dau_vao = self.tuong_lua.kiem_tra_dau_vao(cau_hoi_sach)
        if not kt_dau_vao["an_toan"]:
            return {
                "cau_hoi": cau_hoi_sach,
                "tra_loi": f"[Tường lửa AI chặn]: {kt_dau_vao['ly_do']}",
                "trich_dan": [],
                "can_cu_phap_ly": [],
                "an_toan": False,
                "diem_bam_sat_nguon": 0.0,
            }

        ket_qua_rag = self.rag.hoi(cau_hoi_sach, top_k=3, co_kem_nguon=True)
        nguon_list = ket_qua_rag.get("nguon", [])

        trich_dan: List[Dict[str, str]] = []
        can_cu: List[str] = []

        for ng in nguon_list:
            meta = ng.get("metadata", {}) if isinstance(ng, dict) else {}
            van_ban = meta.get("van_ban") or "Pháp luật Việt Nam"
            dieu = meta.get("dieu") or ""
            tieu_de = meta.get("tieu_de") or ""
            noi_dung = ng.get("noi_dung") or ng.get("van_ban") or ""

            if dieu:
                nhan_can_cu = f"{dieu} - {van_ban}"
                if nhan_can_cu not in can_cu:
                    can_cu.append(nhan_can_cu)

            trich_dan.append(
                {
                    "van_ban": van_ban,
                    "dieu": dieu,
                    "tieu_de": tieu_de,
                    "trich_doan": noi_dung[:200] + "..." if len(noi_dung) > 200 else noi_dung,
                }
            )

        tra_loi_goc = ket_qua_rag.get("tra_loi", "")
        # Nếu chưa có câu trả lời rõ ràng từ RAG đơn giản, định dạng văn phong chuẩn pháp lý
        if trich_dan and ("Không tìm thấy" in tra_loi_goc or len(tra_loi_goc.strip()) < 20):
            nguon_chinh = trich_dan[0]
            tra_loi = (
                f"Căn cứ theo quy định tại {nguon_chinh['dieu']} ({nguon_chinh['tieu_de']}) thuộc {nguon_chinh['van_ban']}:\n"
                f"{nguon_chinh['trich_doan']}"
            )
        else:
            tra_loi = tra_loi_goc

        if trich_dan_chi_tiet and can_cu:
            tra_loi += f"\n\nCăn cứ pháp lý tham chiếu: {'; '.join(can_cu)}."

        # Kiểm định đầu ra bằng CongAnToanDauRa
        kt_dau_ra = CongAnToanDauRa.kiem_tra_va_lam_sach(tra_loi, nguon_ngu_canh=nguon_list)
        tra_loi_sach = kt_dau_ra["van_ban_sach"]

        return {
            "cau_hoi": cau_hoi_sach,
            "tra_loi": tra_loi_sach,
            "can_cu_phap_ly": can_cu,
            "trich_dan": trich_dan,
            "diem_bam_sat_nguon": kt_dau_ra.get("diem_bam_sat_nguon", 1.0),
            "nghi_ngo_ao_giac": kt_dau_ra.get("nghi_ngo_ao_giac", False),
            "an_toan": kt_dau_ra["an_toan"],
        }


# English alias
LegalAssistant = TroLyPhapLy
