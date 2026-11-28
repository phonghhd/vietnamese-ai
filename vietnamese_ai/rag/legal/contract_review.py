"""BoSoSanhRaSoatHopDong (ContractReviewEngine) - Rà Soát Rủi Ro, So Sánh Điều Khoản & Xuất Báo Cáo Redline `.docx` / Markdown."""

from __future__ import annotations

import difflib
import html
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from vietnamese_ai.rag.legal.citation_graph import DoThiThamChieuPhapLy
from vietnamese_ai.rag.legal.document_parser import BoDocTaiLieuPhapLy
from vietnamese_ai.rag.legal.validity_tracker import BoTheoDoiHieuLucPhapLy
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


@dataclass
class CanhBaoRuiRoHopDong:
    """Một cảnh báo rủi ro pháp lý hoặc thương mại phát hiện trong hợp đồng."""

    ma_quy_tac: str
    muc_do: str
    dieu_khoan: str
    tieu_de: str
    chi_tiet: str
    can_cu_phap_ly: str
    khuyen_nghi: str

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển cảnh báo rủi ro sang dict thuần JSON."""
        return asdict(self)


@dataclass
class KhacBietDieuKhoan:
    """Kết quả so sánh từng điều khoản giữa Hợp đồng mẫu chuẩn và Hợp đồng đối tác."""

    dieu: str
    tieu_de: str
    trang_thai: str
    muc_do_thay_doi: float
    noi_dung_chuan: str
    noi_dung_doi_tac: str
    diff_dong: List[str] = field(default_factory=list)
    nhan_xet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển kết quả so sánh điều khoản sang dict."""
        return asdict(self)


_RE_DIEU_HEADER = re.compile(
    r"^\s*(Điều\s+\d+[\.\:]?\s*[^\n]*)$",
    re.IGNORECASE | re.UNICODE,
)
_RE_PHAN_TRAM = re.compile(r"(\d+(?:[\.,]\d+)?)\s*%")
_RE_LAI_SUAT_NAM = re.compile(r"(\d+(?:[\.,]\d+)?)\s*%\s*/\s*năm", re.IGNORECASE | re.UNICODE)

_DIEU_KHOAN_BAT_BUOC: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("doi_tuong", "Đối tượng & Phạm vi hợp đồng", ("đối tượng", "phạm vi công việc", "hàng hóa", "dịch vụ")),
    ("thanh_toan", "Giá trị & Phương thức thanh toán", ("thanh toán", "giá trị hợp đồng", "đơn giá", "tiền lương")),
    ("bao_mat", "Bảo mật thông tin & Dữ liệu cá nhân", ("bảo mật", "dữ liệu cá nhân", "bí mật kinh doanh", "13/2023")),
    ("bat_kha_khang", "Sự kiện bất khả kháng", ("bất khả kháng", "thiên tai", "dịch bệnh")),
    ("tranh_chap", "Giải quyết tranh chấp & Luật áp dụng", ("tranh chấp", "tòa án", "trọng tài")),
]


def _tach_cac_dieu_hop_dong(van_ban: str) -> List[Dict[str, str]]:
    """Tách văn bản hợp đồng thành danh sách các điều khoản có tiêu đề và nội dung chuẩn hóa."""
    if not van_ban or not van_ban.strip():
        return []

    cac_dieu: List[Dict[str, str]] = []
    dieu_hien_tai: Optional[str] = None
    tieu_de_hien_tai = ""
    noi_dung_dong: List[str] = []

    def _chot() -> None:
        if dieu_hien_tai is not None:
            cac_dieu.append(
                {
                    "dieu": dieu_hien_tai,
                    "tieu_de": tieu_de_hien_tai,
                    "noi_dung": "\n".join(noi_dung_dong).strip(),
                }
            )

    for dong in van_ban.splitlines():
        m = _RE_DIEU_HEADER.match(dong)
        if m:
            _chot()
            dong_sach = m.group(1).strip()
            m_so = re.match(r"^(Điều\s+\d+)[\.\:\s]*(.*)$", dong_sach, re.IGNORECASE | re.UNICODE)
            if m_so:
                dieu_hien_tai = m_so.group(1).strip().capitalize()
                tieu_de_hien_tai = m_so.group(2).strip()
            else:
                dieu_hien_tai = dong_sach
                tieu_de_hien_tai = ""
            noi_dung_dong = [dong_sach]
        else:
            if dieu_hien_tai is None and dong.strip():
                dieu_hien_tai = "Phần mở đầu"
                tieu_de_hien_tai = "Thông tin chung & Căn cứ ký kết"
            if dieu_hien_tai is not None:
                noi_dung_dong.append(dong.rstrip())

    _chot()
    return cac_dieu


def _doc_noi_dung_hop_dong(nguon: Union[str, Path], cho_phep_doc_tep: bool = True) -> Tuple[str, str]:
    """Đọc nội dung hợp đồng từ đường dẫn tệp (`.docx`/`.pdf`/`.txt`/`.md`) hoặc chuỗi trực tiếp."""
    if isinstance(nguon, Path):
        info = BoDocTaiLieuPhapLy().doc_tep(nguon)
        return str(info.get("van_ban", "")), nguon.name
    text = str(nguon or "")
    if cho_phep_doc_tep and "\n" not in text and len(text) < 260:
        p = Path(text)
        if p.is_file():
            info = BoDocTaiLieuPhapLy().doc_tep(p)
            return str(info.get("van_ban", "")), p.name
    return text, "hop_dong_van_ban"


def _tao_xml_doan_docx(van_ban: str, in_dam: bool = False, mau_hex: Optional[str] = None, co_chu: int = 22) -> str:
    """Sinh thẻ XML `<w:p>` chuẩn OOXML cho một đoạn văn bản có màu sắc & định dạng."""
    rpr_parts = [f'<w:sz w:val="{co_chu}"/>']
    if in_dam:
        rpr_parts.append("<w:b/>")
    if mau_hex:
        rpr_parts.append(f'<w:color w:val="{mau_hex}"/>')
    rpr = f"<w:rPr>{''.join(rpr_parts)}</w:rPr>"
    escaped = html.escape(van_ban)
    return f'<w:p><w:r>{rpr}<w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>'


def _tao_doan_track_changes_xml(
    noi_dung_goc: str,
    noi_dung_moi: str,
    tac_gia: str,
    ngay_iso: str,
    bo_dem_id: List[int],
    comment_ids: Optional[List[int]] = None,
) -> str:
    """Sinh thẻ `<w:p>` chứa Track Changes cấp từ thực (`<w:del>` / `<w:ins>`) và neo chú thích lề Word."""
    tokens_a = re.findall(r"\S+\s*", noi_dung_goc or "")
    tokens_b = re.findall(r"\S+\s*", noi_dung_moi or "")
    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b)
    tac_gia_esc = html.escape(tac_gia, quote=True)
    ngay_esc = html.escape(ngay_iso, quote=True)

    parts: List[str] = []
    if comment_ids:
        for cid in comment_ids:
            parts.append(f'<w:commentRangeStart w:id="{cid}"/>')

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            text = html.escape("".join(tokens_a[i1:i2]))
            if text:
                parts.append(f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>')
        elif tag in ("delete", "replace"):
            del_text = html.escape("".join(tokens_a[i1:i2]))
            if del_text:
                rev_id = bo_dem_id[0]
                bo_dem_id[0] += 1
                parts.append(
                    f'<w:del w:id="{rev_id}" w:author="{tac_gia_esc}" w:date="{ngay_esc}">'
                    f'<w:r><w:rPr><w:color w:val="B91C1C"/></w:rPr>'
                    f'<w:delText xml:space="preserve">{del_text}</w:delText></w:r></w:del>'
                )
            if tag == "replace":
                ins_text = html.escape("".join(tokens_b[j1:j2]))
                if ins_text:
                    rev_id = bo_dem_id[0]
                    bo_dem_id[0] += 1
                    parts.append(
                        f'<w:ins w:id="{rev_id}" w:author="{tac_gia_esc}" w:date="{ngay_esc}">'
                        f'<w:r><w:rPr><w:color w:val="15803D"/><w:u w:val="single"/></w:rPr>'
                        f'<w:t xml:space="preserve">{ins_text}</w:t></w:r></w:ins>'
                    )
        elif tag == "insert":
            ins_text = html.escape("".join(tokens_b[j1:j2]))
            if ins_text:
                rev_id = bo_dem_id[0]
                bo_dem_id[0] += 1
                parts.append(
                    f'<w:ins w:id="{rev_id}" w:author="{tac_gia_esc}" w:date="{ngay_esc}">'
                    f'<w:r><w:rPr><w:color w:val="15803D"/><w:u w:val="single"/></w:rPr>'
                    f'<w:t xml:space="preserve">{ins_text}</w:t></w:r></w:ins>'
                )

    if comment_ids:
        for cid in comment_ids:
            parts.append(f'<w:commentRangeEnd w:id="{cid}"/>')
            parts.append(f'<w:r><w:commentReference w:id="{cid}"/></w:r>')

    return f"<w:p>{''.join(parts)}</w:p>"


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Công cụ rà soát rủi ro pháp lý, so sánh hợp đồng mẫu chuẩn vs hợp đồng đối tác (Contract Diff & Redline), "
        "kiểm toán văn bản luật hết hiệu lực và xuất trực tiếp báo cáo `.docx` / Markdown."
    ),
    nhom_extra="core",
)
class BoSoSanhRaSoatHopDong:
    """
    Hệ thống Rà Soát & So Sánh Hợp Đồng Tiếng Việt (Contract Diff & Redline Engine).

    Tính năng:
    - Tự động đọc hợp đồng từ chuỗi văn bản hoặc tệp `.docx`, `.pdf`, `.txt`, `.md`.
    - So sánh từng Điều khoản giữa Hợp đồng mẫu chuẩn và Hợp đồng đối tác gửi, sinh diff chi tiết.
    - Phát hiện điều khoản bất lợi/trái luật (phạt vi phạm > 8% theo Luật Thương mại, lãi suất > 20%/năm
      theo BLDS 2015, miễn trừ trách nhiệm một chiều, chấm dứt không báo trước, thiếu điều khoản bắt buộc).
    - Tự động kiểm toán mọi số hiệu luật viện dẫn qua `BoTheoDoiHieuLucPhapLy` & `DoThiThamChieuPhapLy`.
    - Xuất báo cáo rà soát chuẩn định dạng Markdown và tệp Word `.docx` (OOXML) sẵn gửi đối tác.
    """

    def __init__(
        self,
        bo_theo_doi: Optional[BoTheoDoiHieuLucPhapLy] = None,
        do_thi_tham_chieu: Optional[DoThiThamChieuPhapLy] = None,
    ) -> None:
        self.bo_theo_doi = bo_theo_doi or BoTheoDoiHieuLucPhapLy()
        self.do_thi_tham_chieu = do_thi_tham_chieu or DoThiThamChieuPhapLy(bo_theo_doi=self.bo_theo_doi)

    def quet_rui_ro_don_le(
        self,
        van_ban_hop_dong: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]:
        """Quét rủi ro pháp lý, điều khoản bất lợi, điều khoản thiếu và văn bản luật hết hiệu lực trên một hợp đồng."""
        cac_dieu = _tach_cac_dieu_hop_dong(van_ban_hop_dong)
        canh_bao: List[CanhBaoRuiRoHopDong] = []

        for dieu_item in cac_dieu:
            ten_dieu = dieu_item["dieu"]
            nd = dieu_item["noi_dung"]
            nd_low = nd.lower()

            tieu_de_low = dieu_item.get("tieu_de", "").lower()
            la_dieu_phat = "phạt vi phạm" in tieu_de_low or "mức phạt" in tieu_de_low
            if "phạt vi phạm" in nd_low or "mức phạt" in nd_low or la_dieu_phat:
                cac_ve = re.split(r"[.\n;:]+", nd)
                gia_tri_vuot: Optional[float] = None
                for ve in cac_ve:
                    ve_low = ve.lower()
                    co_tu_khoa_phat = any(
                        tk in ve_low for tk in ("phạt vi phạm", "mức phạt", "tiền phạt", "bị phạt", "phạt chậm")
                    )
                    la_ve_khong_lien_quan = any(
                        tk in ve_low
                        for tk in (
                            "tạm ứng",
                            "đặt cọc",
                            "thuế",
                            "vat",
                            "giới hạn trách nhiệm",
                            "trần trách nhiệm",
                            "tổng giá trị thanh toán",
                        )
                    )
                    if la_ve_khong_lien_quan and not co_tu_khoa_phat:
                        continue
                    if not co_tu_khoa_phat and not la_dieu_phat:
                        continue
                    for m_pt in _RE_PHAN_TRAM.findall(ve):
                        gia_tri = float(m_pt.replace(",", "."))
                        if gia_tri > 8.0:
                            gia_tri_vuot = gia_tri
                            break
                    if gia_tri_vuot is not None:
                        break
                if gia_tri_vuot is not None:
                    canh_bao.append(
                        CanhBaoRuiRoHopDong(
                            ma_quy_tac="phat_vi_pham_vuot_tran_8pct",
                            muc_do="cao",
                            dieu_khoan=ten_dieu,
                            tieu_de=f"Mức phạt vi phạm {gia_tri_vuot:g}% vượt trần 8% của Luật Thương mại",
                            chi_tiet=f"Phát hiện mức phạt {gia_tri_vuot:g}% trong {ten_dieu}.",
                            can_cu_phap_ly="Điều 301 Luật Thương mại 36/2005/QH11",
                            khuyen_nghi="Đề nghị điều chỉnh mức phạt vi phạm xuống tối đa 8% giá trị phần nghĩa vụ bị vi phạm.",
                        )
                    )

            for m_ls in _RE_LAI_SUAT_NAM.findall(nd):
                lai_suat = float(m_ls.replace(",", "."))
                if lai_suat > 20.0:
                    canh_bao.append(
                        CanhBaoRuiRoHopDong(
                            ma_quy_tac="lai_suat_vuot_tran_20pct",
                            muc_do="cao",
                            dieu_khoan=ten_dieu,
                            tieu_de=f"Lãi suất {lai_suat:g}%/năm vượt trần 20%/năm theo Bộ luật Dân sự",
                            chi_tiet=f"Phát hiện lãi suất thỏa thuận {lai_suat:g}%/năm trong {ten_dieu}.",
                            can_cu_phap_ly="Điều 468 Bộ luật Dân sự 91/2015/QH13",
                            khuyen_nghi="Phần lãi suất vượt quá 20%/năm không có hiệu lực pháp lý; cần điều chỉnh không vượt quá 20%/năm.",
                        )
                    )
                    break

            if any(
                cum in nd_low
                for cum in (
                    "miễn trừ toàn bộ trách nhiệm",
                    "không chịu bất kỳ trách nhiệm nào",
                    "bồi thường mọi thiệt hại không giới hạn",
                )
            ):
                canh_bao.append(
                    CanhBaoRuiRoHopDong(
                        ma_quy_tac="mien_tru_hoac_boi_thuong_khong_gioi_han",
                        muc_do="cao",
                        dieu_khoan=ten_dieu,
                        tieu_de="Điều khoản miễn trừ trách nhiệm một chiều hoặc bồi thường không giới hạn",
                        chi_tiet=f"Phát hiện ngôn ngữ phân bổ rủi ro mất cân bằng tại {ten_dieu}.",
                        can_cu_phap_ly="Điều 3 Bộ luật Dân sự 91/2015/QH13 (Nguyên tắc bình đẳng, thiện chí)",
                        khuyen_nghi="Bổ sung trần giới hạn trách nhiệm (Liability Cap) tối đa 100% giá trị hợp đồng và áp dụng hai chiều.",
                    )
                )

            la_bat_kha_khang = any(
                bk in nd_low for bk in ("bất khả kháng", "thiên tai", "dịch bệnh")
            )
            co_cham_dut_dot_ngot = any(
                cum in nd_low
                for cum in (
                    "chấm dứt ngay lập tức không cần thông báo trước",
                    "đơn phương chấm dứt bất kỳ lúc nào mà không cần lý do",
                )
            )
            co_cham_dut_khong_boi_thuong = (
                "không phải bồi thường" in nd_low
                and not la_bat_kha_khang
                and any(tk in nd_low for tk in ("đơn phương", "chấm dứt", "hủy hợp đồng", "hủy bỏ hợp đồng"))
            )
            if co_cham_dut_dot_ngot or co_cham_dut_khong_boi_thuong:
                canh_bao.append(
                    CanhBaoRuiRoHopDong(
                        ma_quy_tac="cham_dut_mot_chieu_khong_bao_truoc",
                        muc_do="trung_binh",
                        dieu_khoan=ten_dieu,
                        tieu_de="Quyền đơn phương chấm dứt không cần báo trước",
                        chi_tiet=f"Điều khoản tại {ten_dieu} cho phép chấm dứt đột ngột gây rủi ro vận hành.",
                        can_cu_phap_ly="Điều 428 Bộ luật Dân sự 91/2015/QH13 / Điều 36 BLLĐ 45/2019/QH14",
                        khuyen_nghi="Yêu cầu thời hạn thông báo trước tối thiểu 15–30 ngày và quy định rõ điều kiện chấm dứt.",
                    )
                )

            if "tự động gia hạn" in nd_low and "không gia hạn" not in nd_low and "thông báo hủy" not in nd_low:
                canh_bao.append(
                    CanhBaoRuiRoHopDong(
                        ma_quy_tac="tu_dong_gia_han_an",
                        muc_do="thap",
                        dieu_khoan=ten_dieu,
                        tieu_de="Điều khoản tự động gia hạn thiếu cơ chế từ chối gia hạn",
                        chi_tiet=f"Phát hiện 'tự động gia hạn' tại {ten_dieu} nhưng chưa nêu rõ quyền hủy gia hạn.",
                        can_cu_phap_ly="Thông lệ kiểm soát rủi ro hợp đồng doanh nghiệp",
                        khuyen_nghi="Bổ sung quyền gửi thông báo không gia hạn trước 30 ngày khi hết kỳ hạn.",
                    )
                )

        toan_van_low = van_ban_hop_dong.lower()
        dieu_khoan_thieu: List[str] = []
        for ma_bb, ten_bb, tu_khoa in _DIEU_KHOAN_BAT_BUOC:
            if not any(tk in toan_van_low for tk in tu_khoa):
                dieu_khoan_thieu.append(ten_bb)
                canh_bao.append(
                    CanhBaoRuiRoHopDong(
                        ma_quy_tac=f"thieu_dieu_khoan_{ma_bb}",
                        muc_do="trung_binh",
                        dieu_khoan="Toàn hợp đồng",
                        tieu_de=f"Thiếu điều khoản quan trọng: {ten_bb}",
                        chi_tiet=f"Không tìm thấy quy định rõ ràng về '{ten_bb}' trong dự thảo hợp đồng.",
                        can_cu_phap_ly="Điều 398 Bộ luật Dân sự 91/2015/QH13",
                        khuyen_nghi=f"Đề nghị bổ sung điều khoản riêng về '{ten_bb}'.",
                    )
                )

        kiem_toan_hl = self.bo_theo_doi.kiem_tra_hieu_luc_trong_van_ban(van_ban_hop_dong, ngay_tra_cuu=ngay_tra_cuu)
        for vb_het in kiem_toan_hl.get("van_ban_het_hieu_luc", []):
            sh = str(vb_het["so_hieu"])
            thay_the = ", ".join(vb_het.get("bi_thay_the_boi") or []) or "văn bản mới hơn"
            canh_bao.append(
                CanhBaoRuiRoHopDong(
                    ma_quy_tac="vien_dan_luat_het_hieu_luc",
                    muc_do="cao",
                    dieu_khoan="Căn cứ pháp lý",
                    tieu_de=f"Viện dẫn văn bản đã hết hiệu lực: {sh}",
                    chi_tiet=str(vb_het.get("canh_bao", f"{sh} đã hết hiệu lực.")),
                    can_cu_phap_ly=f"Thay thế bởi: {thay_the}",
                    khuyen_nghi=f"Cập nhật căn cứ pháp lý từ {sh} sang {thay_the}.",
                )
            )

        for vb_sua in kiem_toan_hl.get("van_ban_da_sua_doi", []):
            sh = str(vb_sua["so_hieu"])
            sua_boi = ", ".join(vb_sua.get("sua_doi_bo_sung_boi") or [])
            canh_bao.append(
                CanhBaoRuiRoHopDong(
                    ma_quy_tac="vien_dan_luat_da_sua_doi",
                    muc_do="thap",
                    dieu_khoan="Căn cứ pháp lý",
                    tieu_de=f"Văn bản viện dẫn đã được sửa đổi/bổ sung: {sh}",
                    chi_tiet=str(vb_sua.get("canh_bao", "")),
                    can_cu_phap_ly=f"Sửa đổi bởi: {sua_boi}",
                    khuyen_nghi=f"Bổ sung trích dẫn kèm theo văn bản sửa đổi ({sua_boi}).",
                )
            )

        diem_tru = sum(22 if c.muc_do == "cao" else (10 if c.muc_do == "trung_binh" else 4) for c in canh_bao)
        diem_an_toan = max(0, 100 - diem_tru)
        so_hieu_phat_hien = [str(x["so_hieu"]) for x in kiem_toan_hl.get("chi_tiet", [])]
        do_thi_lien_quan = self.do_thi_tham_chieu.mo_rong_ngu_canh_rag(so_hieu_phat_hien, ngay_tra_cuu=ngay_tra_cuu)

        return {
            "so_dieu_khoan": len(cac_dieu),
            "diem_an_toan_phap_ly": diem_an_toan,
            "xep_loai_rui_ro": "CAO" if diem_an_toan < 50 else ("TRUNG BÌNH" if diem_an_toan < 80 else "THẤP"),
            "so_canh_bao": len(canh_bao),
            "dieu_khoan_thieu": dieu_khoan_thieu,
            "danh_sach_canh_bao": [c.to_dict() for c in canh_bao],
            "kiem_toan_hieu_luc": kiem_toan_hl,
            "do_thi_tham_chieu": do_thi_lien_quan,
        }

    def so_sanh_va_ra_soat(
        self,
        hop_dong_doi_tac: Union[str, Path],
        hop_dong_chuan: Optional[Union[str, Path]] = None,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        cho_phep_doc_tep: bool = True,
    ) -> Dict[str, Any]:
        """So sánh từng Điều khoản giữa `hop_dong_chuan` và `hop_dong_doi_tac` kết hợp quét rủi ro pháp lý toàn diện."""
        nd_doi_tac, ten_doi_tac = _doc_noi_dung_hop_dong(
            hop_dong_doi_tac, cho_phep_doc_tep=cho_phep_doc_tep
        )
        rui_ro = self.quet_rui_ro_don_le(nd_doi_tac, ngay_tra_cuu=ngay_tra_cuu)

        khac_biet_list: List[KhacBietDieuKhoan] = []
        cau_truc_tach_gop: List[Dict[str, Any]] = []
        ten_chuan = ""
        if hop_dong_chuan is not None:
            nd_chuan, ten_chuan = _doc_noi_dung_hop_dong(
                hop_dong_chuan, cho_phep_doc_tep=cho_phep_doc_tep
            )
            dieu_chuan = _tach_cac_dieu_hop_dong(nd_chuan)
            dieu_dt = _tach_cac_dieu_hop_dong(nd_doi_tac)

            def _bo_tien_to_so_dieu(noi_dung: str) -> str:
                return re.sub(r"^\s*Điều\s+\d+[\.\:\s]*", "", noi_dung, count=1, flags=re.IGNORECASE | re.UNICODE).strip()

            def _tap_tu(noi_dung: str) -> set[str]:
                return {w for w in re.findall(r"[0-9a-zà-ỹđ]+", _bo_tien_to_so_dieu(noi_dung).lower()) if len(w) >= 2}

            def _tuong_dong_dieu(c_it: Dict[str, str], dt_it: Dict[str, str]) -> float:
                t1 = c_it["tieu_de"].lower().strip()
                t2 = dt_it["tieu_de"].lower().strip()
                b1 = _bo_tien_to_so_dieu(c_it["noi_dung"])
                b2 = _bo_tien_to_so_dieu(dt_it["noi_dung"])
                body_sim = difflib.SequenceMatcher(None, b1.lower(), b2.lower()).ratio()
                if t1 and t2:
                    title_sim = difflib.SequenceMatcher(None, t1, t2).ratio()
                    return 0.45 * title_sim + 0.55 * body_sim
                return body_sim

            matched_c: Dict[int, int] = {}
            matched_dt: set[int] = set()
            idx_dt_theo_so: Dict[str, List[int]] = {}
            for j, d in enumerate(dieu_dt):
                idx_dt_theo_so.setdefault(d["dieu"].lower(), []).append(j)

            for i, c_it in enumerate(dieu_chuan):
                khoa = c_it["dieu"].lower()
                ung_vien = [j for j in idx_dt_theo_so.get(khoa, []) if j not in matched_dt]
                if not ung_vien:
                    continue
                j_same = ung_vien[0]
                sim_same = _tuong_dong_dieu(c_it, dieu_dt[j_same])
                sim_best_other = max(
                    (_tuong_dong_dieu(c_it, dieu_dt[j]) for j in range(len(dieu_dt)) if j != j_same and j not in matched_dt),
                    default=0.0,
                )
                if sim_same >= 0.25 and not (sim_same < 0.45 and sim_best_other >= 0.70):
                    matched_c[i] = j_same
                    matched_dt.add(j_same)

            ung_vien_le: List[Tuple[float, int, int]] = []
            for i, c_it in enumerate(dieu_chuan):
                if i in matched_c:
                    continue
                for j, dt_it in enumerate(dieu_dt):
                    if j in matched_dt:
                        continue
                    sim = _tuong_dong_dieu(c_it, dt_it)
                    if sim >= 0.45:
                        ung_vien_le.append((sim, i, j))
            ung_vien_le.sort(key=lambda x: x[0], reverse=True)
            for _, i, j in ung_vien_le:
                if i not in matched_c and j not in matched_dt:
                    matched_c[i] = j
                    matched_dt.add(j)

            tach_tu_chuan: Dict[int, str] = {}
            for j, dt_it in enumerate(dieu_dt):
                if j in matched_dt:
                    continue
                toks_dt = _tap_tu(dt_it["noi_dung"])
                if len(toks_dt) < 3:
                    continue
                best_i: Optional[int] = None
                best_cov = 0.0
                for i, c_it in enumerate(dieu_chuan):
                    toks_c = _tap_tu(c_it["noi_dung"])
                    if not toks_c:
                        continue
                    cov = len(toks_dt & toks_c) / float(len(toks_dt))
                    if cov > best_cov:
                        best_cov = cov
                        best_i = i
                if best_i is not None and best_cov >= 0.50:
                    dieu_goc = dieu_chuan[best_i]["dieu"]
                    tach_tu_chuan[j] = dieu_goc
                    cau_truc_tach_gop.append(
                        {
                            "loai": "tach_dieu_khoan",
                            "dieu_chuan": dieu_goc,
                            "dieu_doi_tac": dt_it["dieu"],
                            "ty_le_bao_phu": round(best_cov, 4),
                        }
                    )

            gop_vao_dt: Dict[int, str] = {}
            for i, c_it in enumerate(dieu_chuan):
                if i in matched_c:
                    continue
                toks_c = _tap_tu(c_it["noi_dung"])
                if len(toks_c) < 3:
                    continue
                best_j: Optional[int] = None
                best_cov = 0.0
                for j, dt_it in enumerate(dieu_dt):
                    toks_dt = _tap_tu(dt_it["noi_dung"])
                    if not toks_dt:
                        continue
                    cov = len(toks_c & toks_dt) / float(len(toks_c))
                    if cov > best_cov:
                        best_cov = cov
                        best_j = j
                if best_j is not None and best_cov >= 0.50:
                    dieu_dich = dieu_dt[best_j]["dieu"]
                    gop_vao_dt[i] = dieu_dich
                    cau_truc_tach_gop.append(
                        {
                            "loai": "gop_dieu_khoan",
                            "dieu_chuan": c_it["dieu"],
                            "dieu_doi_tac": dieu_dich,
                            "ty_le_bao_phu": round(best_cov, 4),
                        }
                    )

            cap_sap_xep: List[Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]], Optional[str], Optional[str]]] = []
            for i, c_it in enumerate(dieu_chuan):
                if i in matched_c:
                    cap_sap_xep.append((c_it, dieu_dt[matched_c[i]], None, None))
                else:
                    cap_sap_xep.append((c_it, None, None, gop_vao_dt.get(i)))
            for j, dt_it in enumerate(dieu_dt):
                if j not in matched_dt:
                    cap_sap_xep.append((None, dt_it, tach_tu_chuan.get(j), None))

            for c_item, dt_item, nguon_tach, dich_gop in cap_sap_xep:
                if c_item is not None and dt_item is None:
                    ghi_chu_gop = (
                        f" Nội dung nhiều khả năng đã được đối tác gộp vào {dich_gop}."
                        if dich_gop
                        else ""
                    )
                    khac_biet_list.append(
                        KhacBietDieuKhoan(
                            dieu=c_item["dieu"],
                            tieu_de=c_item["tieu_de"],
                            trang_thai="thieu_dieu_khoan",
                            muc_do_thay_doi=1.0,
                            noi_dung_chuan=c_item["noi_dung"],
                            noi_dung_doi_tac="",
                            diff_dong=[f"- {line}" for line in c_item["noi_dung"].splitlines()],
                            nhan_xet=f"Đối tác đã lược bỏ điều khoản riêng này so với mẫu chuẩn.{ghi_chu_gop}",
                        )
                    )
                elif c_item is None and dt_item is not None:
                    ghi_chu_tach = (
                        f" Nội dung được phát hiện tách ra từ {nguon_tach} của mẫu chuẩn."
                        if nguon_tach
                        else ""
                    )
                    khac_biet_list.append(
                        KhacBietDieuKhoan(
                            dieu=dt_item["dieu"],
                            tieu_de=dt_item["tieu_de"],
                            trang_thai="them_moi",
                            muc_do_thay_doi=1.0,
                            noi_dung_chuan="",
                            noi_dung_doi_tac=dt_item["noi_dung"],
                            diff_dong=[f"+ {line}" for line in dt_item["noi_dung"].splitlines()],
                            nhan_xet=f"Điều khoản mới được đối tác bổ sung, cần rà soát kỹ nghĩa vụ phát sinh.{ghi_chu_tach}",
                        )
                    )
                elif c_item is not None and dt_item is not None:
                    doi_so_thu_tu = c_item["dieu"].lower() != dt_item["dieu"].lower()
                    so_sanh_chuan = _bo_tien_to_so_dieu(c_item["noi_dung"]) if doi_so_thu_tu else c_item["noi_dung"]
                    so_sanh_dt = _bo_tien_to_so_dieu(dt_item["noi_dung"]) if doi_so_thu_tu else dt_item["noi_dung"]
                    ratio = difflib.SequenceMatcher(None, so_sanh_chuan, so_sanh_dt).ratio()
                    diff_pct = round(1.0 - ratio, 4)
                    ghi_chu_doi_so = (
                        f" (Đổi số thứ tự từ {c_item['dieu']} sang {dt_item['dieu']})" if doi_so_thu_tu else ""
                    )
                    if diff_pct < 0.01:
                        khac_biet_list.append(
                            KhacBietDieuKhoan(
                                dieu=dt_item["dieu"],
                                tieu_de=dt_item["tieu_de"] or c_item["tieu_de"],
                                trang_thai="khong_doi",
                                muc_do_thay_doi=0.0,
                                noi_dung_chuan=c_item["noi_dung"],
                                noi_dung_doi_tac=dt_item["noi_dung"],
                                diff_dong=[],
                                nhan_xet=f"Giữ nguyên theo mẫu chuẩn.{ghi_chu_doi_so}",
                            )
                        )
                    else:
                        raw_diff = list(
                            difflib.unified_diff(
                                c_item["noi_dung"].splitlines(),
                                dt_item["noi_dung"].splitlines(),
                                lineterm="",
                                n=1,
                            )
                        )
                        diff_sach = [d for d in raw_diff if not d.startswith(("---", "+++", "@@"))]
                        khac_biet_list.append(
                            KhacBietDieuKhoan(
                                dieu=dt_item["dieu"],
                                tieu_de=dt_item["tieu_de"] or c_item["tieu_de"],
                                trang_thai="sua_doi",
                                muc_do_thay_doi=diff_pct,
                                noi_dung_chuan=c_item["noi_dung"],
                                noi_dung_doi_tac=dt_item["noi_dung"],
                                diff_dong=diff_sach,
                                nhan_xet=(
                                    f"Đối tác đã chỉnh sửa khoảng {round(diff_pct * 100, 1)}% nội dung điều khoản."
                                    f"{ghi_chu_doi_so}"
                                ),
                            )
                        )

        thong_ke_diff = {
            "tong_dieu_so_sanh": len(khac_biet_list),
            "khong_doi": sum(1 for x in khac_biet_list if x.trang_thai == "khong_doi"),
            "sua_doi": sum(1 for x in khac_biet_list if x.trang_thai == "sua_doi"),
            "thieu_dieu_khoan": sum(1 for x in khac_biet_list if x.trang_thai == "thieu_dieu_khoan"),
            "them_moi": sum(1 for x in khac_biet_list if x.trang_thai == "them_moi"),
            "tach_dieu_khoan": sum(1 for x in cau_truc_tach_gop if x["loai"] == "tach_dieu_khoan"),
            "gop_dieu_khoan": sum(1 for x in cau_truc_tach_gop if x["loai"] == "gop_dieu_khoan"),
        }

        return {
            "ten_hop_dong_doi_tac": ten_doi_tac,
            "ten_hop_dong_chuan": ten_chuan or None,
            **rui_ro,
            "thong_ke_so_sanh": thong_ke_diff,
            "cau_truc_tach_gop": cau_truc_tach_gop,
            "chi_tiet_so_sanh": [x.to_dict() for x in khac_biet_list],
        }

    def xuat_bao_cao_markdown(self, ket_qua: Dict[str, Any]) -> str:
        """Xuất báo cáo rà soát & so sánh hợp đồng (Redline Report) định dạng Markdown."""
        lines: List[str] = [
            "# BÁO CÁO RÀ SOÁT & SO SÁNH HỢP ĐỒNG (REDLINE LEGAL AUDIT)",
            "",
            f"- **Hợp đồng đối tác**: `{ket_qua.get('ten_hop_dong_doi_tac', 'N/A')}`",
        ]
        if ket_qua.get("ten_hop_dong_chuan"):
            lines.append(f"- **Hợp đồng mẫu chuẩn**: `{ket_qua['ten_hop_dong_chuan']}`")
        lines.extend(
            [
                f"- **Điểm an toàn pháp lý**: **{ket_qua.get('diem_an_toan_phap_ly', 0)}/100** (Mức rủi ro: **{ket_qua.get('xep_loai_rui_ro', 'N/A')}**)",
                f"- **Tổng cảnh báo phát hiện**: {ket_qua.get('so_canh_bao', 0)}",
                "",
                "## 1. Danh Sách Cảnh Báo Rủi Ro & Văn Bản Hết Hiệu Lực",
                "",
            ]
        )

        ds_cb = ket_qua.get("danh_sach_canh_bao") or []
        if not ds_cb:
            lines.append("Không phát hiện rủi ro pháp lý nghiêm trọng nào.")
        else:
            for idx, cb in enumerate(ds_cb, 1):
                lines.append(
                    f"{idx}. **[{str(cb['muc_do']).upper()}] {cb['tieu_de']}** ({cb['dieu_khoan']})\n"
                    f"   - Chi tiết: {cb['chi_tiet']}\n"
                    f"   - Căn cứ: {cb['can_cu_phap_ly']}\n"
                    f"   - Khuyến nghị: {cb['khuyen_nghi']}"
                )

        chi_tiet = ket_qua.get("chi_tiet_so_sanh") or []
        if chi_tiet:
            lines.extend(["", "## 2. Bảng So Sánh Chi Tiết Từng Điều Khoản (Redline Diff)", ""])
            for item in chi_tiet:
                lines.append(f"### {item['dieu']} — {item['tieu_de']} (`{item['trang_thai']}`)")
                lines.append(f"_{item['nhan_xet']}_")
                if item.get("diff_dong"):
                    lines.append("```diff")
                    lines.extend(item["diff_dong"])
                    lines.append("```")
                lines.append("")

        return "\n".join(lines).strip() + "\n"

    def xuat_bao_cao_docx(
        self,
        ket_qua: Dict[str, Any],
        duong_dan_xuat: Union[str, Path],
    ) -> Path:
        """Xuất báo cáo Redline hợp đồng thành tệp Word `.docx` (OOXML hợp lệ, không bắt buộc `python-docx`)."""
        dich = Path(duong_dan_xuat)
        dich.parent.mkdir(parents=True, exist_ok=True)

        doan_xml: List[str] = [
            _tao_xml_doan_docx("BÁO CÁO RÀ SOÁT & SO SÁNH HỢP ĐỒNG (VIETNAMESE AI)", in_dam=True, mau_hex="1E3A8A", co_chu=30),
            _tao_xml_doan_docx(
                f"Hợp đồng đối tác: {ket_qua.get('ten_hop_dong_doi_tac', 'N/A')} | "
                f"Điểm an toàn: {ket_qua.get('diem_an_toan_phap_ly', 0)}/100 ({ket_qua.get('xep_loai_rui_ro', 'N/A')})",
                in_dam=True,
                mau_hex="334155",
                co_chu=22,
            ),
            _tao_xml_doan_docx("1. CẢNH BÁO RỦI RO PHÁP LÝ & VĂN BẢN HẾT HIỆU LỰC", in_dam=True, mau_hex="B91C1C", co_chu=24),
        ]

        for idx, cb in enumerate(ket_qua.get("danh_sach_canh_bao") or [], 1):
            mau = "DC2626" if cb["muc_do"] == "cao" else ("D97706" if cb["muc_do"] == "trung_binh" else "2563EB")
            doan_xml.append(
                _tao_xml_doan_docx(
                    f"{idx}. [{str(cb['muc_do']).upper()}] {cb['tieu_de']} ({cb['dieu_khoan']})",
                    in_dam=True,
                    mau_hex=mau,
                )
            )
            doan_xml.append(_tao_xml_doan_docx(f"   Căn cứ: {cb['can_cu_phap_ly']} | Khuyến nghị: {cb['khuyen_nghi']}"))

        chi_tiet = ket_qua.get("chi_tiet_so_sanh") or []
        if chi_tiet:
            doan_xml.append(
                _tao_xml_doan_docx("2. SO SÁNH ĐIỀU KHOẢN (REDLINE DIFF)", in_dam=True, mau_hex="1E3A8A", co_chu=24)
            )
            for item in chi_tiet:
                doan_xml.append(
                    _tao_xml_doan_docx(
                        f"{item['dieu']} ({item['trang_thai']}): {item['nhan_xet']}",
                        in_dam=True,
                        mau_hex="0F172A",
                    )
                )
                for diff_line in item.get("diff_dong") or []:
                    mau_d = "15803D" if diff_line.startswith("+") else ("B91C1C" if diff_line.startswith("-") else "475569")
                    doan_xml.append(_tao_xml_doan_docx(diff_line, mau_hex=mau_d, co_chu=20))

        body_xml = "".join(doan_xml)
        with zipfile.ZipFile(dich, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>",
            )
            zf.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/>'
                "</Relationships>",
            )
            zf.writestr(
                "word/document.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body>{body_xml}</w:body></w:document>",
            )
        return dich

    def xuat_bao_cao_track_changes_docx(
        self,
        ket_qua: Dict[str, Any],
        duong_dan_xuat: Union[str, Path],
        tac_gia: str = "Vietnamese-AI Legal Auditor",
    ) -> Path:
        """
        Xuất tệp Word `.docx` chứa Track Changes chuẩn OOXML (`<w:ins>` / `<w:del>`) ở mức từ
        và chú thích lề Word (`word/comments.xml`) gắn trực tiếp vào từng điều khoản rủi ro.
        """
        dich = Path(duong_dan_xuat)
        dich.parent.mkdir(parents=True, exist_ok=True)
        ngay_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tac_gia_esc = html.escape(tac_gia, quote=True)
        bo_dem_id = [1]

        ds_cb = list(ket_qua.get("danh_sach_canh_bao") or [])
        comment_xml_items: List[str] = []
        cb_theo_dieu: Dict[str, List[int]] = {}
        cb_toan_van: List[int] = []

        for idx_cb, cb in enumerate(ds_cb):
            cid = idx_cb
            dieu_key = str(cb.get("dieu_khoan") or "").strip().lower()
            if dieu_key.startswith("điều"):
                cb_theo_dieu.setdefault(dieu_key, []).append(cid)
            else:
                cb_toan_van.append(cid)
            noi_dung_cmt = (
                f"[{str(cb.get('muc_do', 'trung_binh')).upper()}] {cb.get('tieu_de', '')}\n"
                f"Chi tiết: {cb.get('chi_tiet', '')}\n"
                f"Căn cứ: {cb.get('can_cu_phap_ly', '')}\n"
                f"Khuyến nghị: {cb.get('khuyen_nghi', '')}"
            )
            p_lines = "".join(
                f'<w:p><w:r><w:t xml:space="preserve">{html.escape(line)}</w:t></w:r></w:p>'
                for line in noi_dung_cmt.splitlines()
            )
            comment_xml_items.append(
                f'<w:comment w:id="{cid}" w:author="{tac_gia_esc}" w:date="{ngay_iso}" w:initials="VAI">'
                f"{p_lines}</w:comment>"
            )

        doan_xml: List[str] = [
            _tao_xml_doan_docx(
                "HỢP ĐỒNG RÀ SOÁT TRACK CHANGES & CHÚ THÍCH PHÁP LÝ (VIETNAMESE AI)",
                in_dam=True,
                mau_hex="1E3A8A",
                co_chu=28,
            ),
        ]

        tong_quan_txt = (
            f"Hợp đồng đối tác: {ket_qua.get('ten_hop_dong_doi_tac', 'N/A')} | "
            f"Điểm an toàn: {ket_qua.get('diem_an_toan_phap_ly', 0)}/100 ({ket_qua.get('xep_loai_rui_ro', 'N/A')}) | "
            f"Cảnh báo: {ket_qua.get('so_canh_bao', 0)}"
        )
        doan_xml.append(
            _tao_doan_track_changes_xml(
                tong_quan_txt,
                tong_quan_txt,
                tac_gia=tac_gia,
                ngay_iso=ngay_iso,
                bo_dem_id=bo_dem_id,
                comment_ids=cb_toan_van if cb_toan_van else None,
            )
        )

        chi_tiet = list(ket_qua.get("chi_tiet_so_sanh") or [])
        dieu_da_gan_cmt: set[str] = set()
        for item in chi_tiet:
            ten_dieu = str(item.get("dieu", ""))
            dieu_key = ten_dieu.strip().lower()
            doan_xml.append(
                _tao_xml_doan_docx(
                    f"{ten_dieu} — {item.get('tieu_de', '')} [{item.get('trang_thai', '')}]",
                    in_dam=True,
                    mau_hex="0F172A",
                    co_chu=24,
                )
            )
            cids = cb_theo_dieu.get(dieu_key, [])
            if cids:
                dieu_da_gan_cmt.add(dieu_key)
            doan_xml.append(
                _tao_doan_track_changes_xml(
                    str(item.get("noi_dung_chuan", "")),
                    str(item.get("noi_dung_doi_tac", "")),
                    tac_gia=tac_gia,
                    ngay_iso=ngay_iso,
                    bo_dem_id=bo_dem_id,
                    comment_ids=cids if cids else None,
                )
            )

        for dieu_key, cids in cb_theo_dieu.items():
            if dieu_key in dieu_da_gan_cmt:
                continue
            doan_xml.append(
                _tao_doan_track_changes_xml(
                    f"Cảnh báo tại {dieu_key}",
                    f"Cảnh báo tại {dieu_key}",
                    tac_gia=tac_gia,
                    ngay_iso=ngay_iso,
                    bo_dem_id=bo_dem_id,
                    comment_ids=cids,
                )
            )

        body_xml = "".join(doan_xml)
        comments_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"{''.join(comment_xml_items)}</w:comments>"
        )

        with zipfile.ZipFile(dich, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                '<Override PartName="/word/comments.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>'
                "</Types>",
            )
            zf.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/>'
                "</Relationships>",
            )
            zf.writestr(
                "word/_rels/document.xml.rels",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rIdComments" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" '
                'Target="comments.xml"/>'
                "</Relationships>",
            )
            zf.writestr(
                "word/document.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body>{body_xml}</w:body></w:document>",
            )
            zf.writestr("word/comments.xml", comments_xml)
        return dich


ContractReviewEngine = BoSoSanhRaSoatHopDong
ContractRiskAlert = CanhBaoRuiRoHopDong
ContractClauseDiff = KhacBietDieuKhoan

__all__ = [
    "CanhBaoRuiRoHopDong",
    "ContractRiskAlert",
    "KhacBietDieuKhoan",
    "ContractClauseDiff",
    "BoSoSanhRaSoatHopDong",
    "ContractReviewEngine",
]
