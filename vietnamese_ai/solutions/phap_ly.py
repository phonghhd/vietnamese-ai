"""Trợ lý Pháp lý & Doanh nghiệp - Vietnamese AI Framework."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

    def __init__(
        self,
        bo_du_lieu_luat: Optional[List[Dict[str, str]]] = None,
        nap_kho_mo: bool = False,
        duong_dan_vaidb: Optional[Union[str, Path]] = None,
        dung_onnx_semantic: bool = False,
    ):
        from vietnamese_ai.rag.legal.citation_graph import DoThiThamChieuPhapLy
        from vietnamese_ai.rag.legal.validity_tracker import BoTheoDoiHieuLucPhapLy
        from vietnamese_ai.rag.vietnamese_context import BoToiUuNguCanhHanhChinh

        self.tuong_lua = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
        self.bo_toi_uu = BoToiUuNguCanhHanhChinh(bat_tuong_lua=False)
        self.bo_theo_doi_hieu_luc = BoTheoDoiHieuLucPhapLy()
        self.do_thi_tham_chieu = DoThiThamChieuPhapLy(bo_theo_doi=self.bo_theo_doi_hieu_luc)
        self.rag = RAGPipeline(
            nguong_diem=0.05,
            top_k=3,
            backend="vaidb" if duong_dan_vaidb is not None else "memory",
            duong_dan_vaidb=duong_dan_vaidb,
            che_do_nhung="onnx_viet" if dung_onnx_semantic else "auto",
            che_do_rerank="onnx_viet" if dung_onnx_semantic else "mmr",
            che_do_hanh_chinh=True,
            bo_toi_uu_hanh_chinh=self.bo_toi_uu,
        )
        self.rag._bo_theo_doi_hieu_luc = self.bo_theo_doi_hieu_luc
        self.rag._do_thi_tham_chieu = self.do_thi_tham_chieu
        if duong_dan_vaidb is not None:
            from vietnamese_ai.rag.legal.auto_ingest import BoNapVanBanPhapQuyTuDong

            BoNapVanBanPhapQuyTuDong.nap_sieu_du_lieu_phap_ly_tu_vaidb(
                duong_dan_vaidb,
                self.bo_theo_doi_hieu_luc,
                self.do_thi_tham_chieu,
            )
        self.kho_tai_lieu = list(bo_du_lieu_luat or self.DU_LIEU_LUAT_MAC_DINH)
        if duong_dan_vaidb is None or self.rag.csdl_vector.so_luong() == 0:
            self._nap_du_lieu_luat()
            if nap_kho_mo:
                self.nap_kho_phap_ly_mo()
        elif nap_kho_mo and self.rag.csdl_vector.so_tai_lieu() < 50:
            self.nap_kho_phap_ly_mo()

    def tra_cuu_van_ban_lien_quan(
        self,
        so_hieu: str,
        ngay_tra_cuu: Optional[str] = None,
        chi_lay_con_hieu_luc: bool = False,
    ) -> Dict[str, Any]:
        """Truy vấn Nghị định hướng dẫn, Thông tư, sửa đổi và văn bản thay thế hiện hành của `so_hieu`."""
        return self.do_thi_tham_chieu.truy_van_van_ban_lien_quan(
            so_hieu,
            ngay_tra_cuu=ngay_tra_cuu,
            chi_lay_con_hieu_luc=chi_lay_con_hieu_luc,
        )

    def nap_kho_phap_ly_mo(self) -> int:
        """Nạp toàn bộ kho dữ liệu pháp lý mở Việt Nam (100 điều khoản) vào RAG nội bộ."""
        from vietnamese_ai.rag.legal.open_corpus import tai_bo_phap_ly_mo

        ban_ghi = tai_bo_phap_ly_mo()
        da_nap = 0
        for item in ban_ghi:
            ma = str(item.get("id", f"mo_{da_nap}"))
            so_hieu = str(item.get("so_hieu", ""))
            van_ban = str(item.get("ten_van_ban", ""))
            dieu = str(item.get("dieu", ""))
            noi_dung = str(item.get("noi_dung", ""))
            full = f"[{van_ban} - {so_hieu}]\n{dieu}. {noi_dung}"
            self.rag.them_tai_lieu(
                ma,
                full,
                metadata={"van_ban": van_ban, "dieu": dieu, "so_hieu": so_hieu},
                che_do_hanh_chinh=True,
            )
            da_nap += 1
        return da_nap

    def kiem_tra_hieu_luc(
        self,
        so_hieu_hoac_van_ban: str,
        ngay_tra_cuu: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Kiểm tra hiệu lực một số hiệu văn bản hoặc quét toàn bộ trích dẫn pháp lý trong văn bản."""
        text = str(so_hieu_hoac_van_ban or "").strip()
        if " " not in text and "/" in text:
            return self.bo_theo_doi_hieu_luc.kiem_tra_hieu_luc(text, ngay_tra_cuu=ngay_tra_cuu)
        return self.bo_theo_doi_hieu_luc.kiem_tra_hieu_luc_trong_van_ban(text, ngay_tra_cuu=ngay_tra_cuu)

    def tra_cuu_dieu_khoan_theo_ngay(
        self,
        so_hieu: str,
        dieu: str,
        ngay_tra_cuu: Optional[str] = None,
        khoan: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tra cứu phiên bản Điều/Khoản có hiệu lực tại mốc thời gian `ngay_tra_cuu` (Point-in-Time Legal Lookup)."""
        return self.bo_theo_doi_hieu_luc.tra_cuu_dieu_khoan_theo_ngay(
            so_hieu=so_hieu,
            dieu=dieu,
            ngay_tra_cuu=ngay_tra_cuu,
            khoan=khoan,
        )

    def xay_dung_chi_muc_hnsw(
        self,
        m: int = 12,
        ef_construction: int = 64,
        ef_search: int = 48,
    ) -> Dict[str, Any]:
        """Xây dựng chỉ mục đồ thị HNSW bền vững nếu kho tri thức đang sử dụng `.vaidb`."""
        store = self.rag.csdl_vector
        if hasattr(store, "xay_dung_chi_muc_hnsw"):
            return dict(
                store.xay_dung_chi_muc_hnsw(
                    m=m,
                    ef_construction=ef_construction,
                    ef_search=ef_search,
                )
            )
        return {
            "da_xay_dung": False,
            "ly_do": "Kho tri thức hiện tại dùng bộ nhớ RAM, cần khởi tạo với `duong_dan_vaidb` để lưu HNSW bền vững.",
        }

    def _nap_du_lieu_luat(self) -> None:
        """Nạp các tài liệu luật vào RAG pipeline với cấu trúc Điều/Khoản bảo toàn."""
        for item in self.kho_tai_lieu:
            ma = item.get("ma", f"doc_{len(self.kho_tai_lieu)}")
            tieu_de = item.get("tieu_de", "")
            dieu = item.get("dieu", "")
            van_ban = item.get("van_ban", "")
            noi_dung = item.get("noi_dung", "")
            noi_dung_day_du = f"[{van_ban}]\n{dieu}: {tieu_de}\n{noi_dung}"
            self.rag.them_tai_lieu(
                ma,
                noi_dung_day_du,
                metadata={
                    "van_ban": van_ban,
                    "dieu": dieu,
                    "tieu_de": tieu_de,
                },
                che_do_hanh_chinh=True,
            )

    def them_dieu_khoan(self, ma: str, van_ban: str, dieu: str, tieu_de: str, noi_dung: str) -> None:
        """Thêm văn bản luật mới vào cơ sở tri thức."""
        item = {"ma": ma, "van_ban": van_ban, "dieu": dieu, "tieu_de": tieu_de, "noi_dung": noi_dung}
        self.kho_tai_lieu.append(item)
        full_text = f"[{van_ban}]\n{dieu}: {tieu_de}\n{noi_dung}"
        self.rag.them_tai_lieu(
            ma,
            full_text,
            metadata={"van_ban": van_ban, "dieu": dieu, "tieu_de": tieu_de},
            che_do_hanh_chinh=True,
        )

    def tra_cuu(
        self,
        cau_hoi: str,
        trich_dan_chi_tiet: bool = True,
        ngay_tra_cuu: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tra cứu quy định pháp luật và sinh câu trả lời kèm trích dẫn văn bản luật chuẩn xác
        (hỗ trợ mốc thời gian hiệu lực `ngay_tra_cuu` và tra cứu phiên bản Điều/Khoản Point-in-Time).
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

        phien_ban_dieu: Optional[Dict[str, Any]] = None
        ds_sh_hoi = self.bo_theo_doi_hieu_luc.trich_xuat_so_hieu(cau_hoi_sach)
        m_dieu = re.search(r"điều\s+(\d+[a-zA-ZđĐ]?)", cau_hoi_sach, re.IGNORECASE | re.UNICODE)
        if ds_sh_hoi and m_dieu:
            kq_pb = self.tra_cuu_dieu_khoan_theo_ngay(
                so_hieu=ds_sh_hoi[0],
                dieu=f"Điều {m_dieu.group(1)}",
                ngay_tra_cuu=ngay_tra_cuu,
            )
            if kq_pb.get("tim_thay"):
                phien_ban_dieu = kq_pb

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
                    "noi_dung": noi_dung,
                    "trich_doan": noi_dung[:200] + "..." if len(noi_dung) > 200 else noi_dung,
                }
            )

        trich_dan_phap_ly = ket_qua_rag.get("trich_dan_phap_ly")
        if trich_dan_phap_ly:
            for dk in trich_dan_phap_ly.get("dieu_khoan", []):
                if dk and dk not in can_cu:
                    can_cu.append(dk)

        tra_loi_goc = ket_qua_rag.get("tra_loi", "")
        if phien_ban_dieu is not None and phien_ban_dieu.get("phien_ban_hieu_luc"):
            pb_hl = phien_ban_dieu["phien_ban_hieu_luc"]
            nhan_pb = f"{pb_hl['dieu']} - {pb_hl['so_hieu']}"
            if nhan_pb not in can_cu:
                can_cu.insert(0, nhan_pb)
            tra_loi = (
                f"Tại ngày {phien_ban_dieu['ngay_tra_cuu']}, theo {pb_hl['dieu']} ({pb_hl['so_hieu']} "
                f"- hiệu lực từ {pb_hl['hieu_luc_tu']}):\n{pb_hl['noi_dung']}"
            )
            if pb_hl.get("ghi_chu"):
                tra_loi += f"\nGhi chú hiệu lực: {pb_hl['ghi_chu']}"
            trich_dan.insert(
                0,
                {
                    "van_ban": str(pb_hl["so_hieu"]),
                    "dieu": str(pb_hl["dieu"]),
                    "tieu_de": str(pb_hl.get("khoan") or "Phiên bản hiệu lực theo ngày"),
                    "noi_dung": str(pb_hl["noi_dung"]),
                    "trich_doan": str(pb_hl["noi_dung"])[:200],
                },
            )
            nguon_list = [{"noi_dung": str(pb_hl["noi_dung"]), "metadata": pb_hl}] + list(nguon_list)
        elif trich_dan and ("Không tìm thấy" in tra_loi_goc or len(tra_loi_goc.strip()) < 20):
            nguon_chinh = trich_dan[0]
            tra_loi = (
                f"Căn cứ theo quy định tại {nguon_chinh['dieu']} ({nguon_chinh['tieu_de']}) thuộc {nguon_chinh['van_ban']}:\n"
                f"{nguon_chinh['trich_doan']}"
            )
        else:
            tra_loi = tra_loi_goc

        if trich_dan_chi_tiet and can_cu:
            tra_loi += f"\n\nCăn cứ pháp lý tham chiếu: {'; '.join(can_cu)}."

        kt_dau_ra = CongAnToanDauRa.kiem_tra_va_lam_sach(tra_loi, nguon_ngu_canh=nguon_list)
        tra_loi_sach = kt_dau_ra["van_ban_sach"]

        ket_qua: Dict[str, Any] = {
            "cau_hoi": cau_hoi_sach,
            "tra_loi": tra_loi_sach,
            "can_cu_phap_ly": can_cu,
            "trich_dan": trich_dan,
            "trich_dan_phap_ly": trich_dan_phap_ly,
            "diem_bam_sat_nguon": kt_dau_ra.get("diem_bam_sat_nguon", 1.0),
            "nghi_ngo_ao_giac": kt_dau_ra.get("nghi_ngo_ao_giac", False),
            "an_toan": kt_dau_ra["an_toan"],
        }
        if ngay_tra_cuu is not None:
            ket_qua["ngay_tra_cuu"] = ngay_tra_cuu
        if phien_ban_dieu is not None:
            ket_qua["phien_ban_dieu_khoan"] = phien_ban_dieu
        return ket_qua


# English alias
LegalAssistant = TroLyPhapLy
