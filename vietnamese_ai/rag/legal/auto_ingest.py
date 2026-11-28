"""Pipeline Nạp Tự Động Văn Bản Pháp Quy & Văn Bản Hợp Nhất Vào Kho `.vaidb` HNSW + Point-in-Time (`vietnamese_ai.rag.legal.auto_ingest`)."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from vietnamese_ai.rag.legal.citation_graph import DoThiThamChieuPhapLy, QuanHePhapLy
from vietnamese_ai.rag.legal.document_parser import BoDocTaiLieuPhapLy
from vietnamese_ai.rag.legal.validity_tracker import (
    BoTheoDoiHieuLucPhapLy,
    HoSoHieuLucVanBan,
    PhienBanDieuKhoan,
)
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang

_RE_SO_HIEU_DOC = re.compile(
    r"\b(\d{1,4}/\d{4}/(?:QH\d{1,2}|UBTVQH\d{1,2}|NĐ-CP|ND-CP|TT-[A-ZĐ]{2,12}|TTLT-[A-ZĐ-]{3,16}|QĐ-TTg|QD-TTg|VBHN-[A-ZĐ]{2,12}|NQ-CP|NQ-QH\d{1,2}))\b",
    re.IGNORECASE | re.UNICODE,
)
_RE_NGAY_THANG_NAM = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE | re.UNICODE,
)
_RE_NGAY_GAU = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_RE_DIEU_HEADER = re.compile(
    r"(?m)^\s*(Điều\s+(\d+[a-zA-ZđĐ]?))(?:\s*[\.\:\-]\s*|\s+)(.*)$",
    re.UNICODE,
)
_RE_CHUONG_HEADER = re.compile(
    r"(?m)^\s*(Chương\s+[IVXLCDM\d]+)(?:\s*[\.\:\-]\s*|\s+)?(.*)$",
    re.IGNORECASE | re.UNICODE,
)
_RE_KHOAN_HEADER = re.compile(r"(?m)^\s*(\d{1,2})\.\s+(.+)$", re.UNICODE)
_RE_CUOC_CHU_DEF = re.compile(
    r"(?m)^\s*\[(\d{1,3})\]\s+(.+?)(?=\n\s*\[\d{1,3}\]\s+|\Z)",
    re.DOTALL | re.UNICODE,
)
_RE_CUOC_CHU_REF = re.compile(r"\[(\d{1,3})\]")
_RE_SUA_DOI_INLINE = re.compile(
    r"\(([^()]{10,260}(?:sửa\s+đổi|bổ\s+sung|bãi\s+bỏ|thay\s+thế)[^()]{0,200})\)",
    re.IGNORECASE | re.UNICODE,
)
_RE_THAY_THE = re.compile(
    r"(?:thay\s+thế|bãi\s+bỏ)\s+(?:toàn\s+bộ\s+)?(?:Luật|Nghị\s+định|Thông\s+tư|Bộ\s+luật|văn\s+bản)[^.\n;]{0,120}",
    re.IGNORECASE | re.UNICODE,
)
_RE_SUA_DOI_VB = re.compile(
    r"sửa\s+đổi,?\s+bổ\s+sung[^.\n;]{0,140}",
    re.IGNORECASE | re.UNICODE,
)
_RE_HUONG_DAN_VB = re.compile(
    r"(?:quy\s+định\s+chi\s+tiết|hướng\s+dẫn\s+thi\s+hành)[^.\n;]{0,140}",
    re.IGNORECASE | re.UNICODE,
)

META_KEY_VALIDITY = "legal_auto_validity_json"
META_KEY_CLAUSES = "legal_auto_clauses_json"
META_KEY_RELATIONS = "legal_auto_relations_json"


def _chuan_hoa_ngay_iso(text: str) -> Optional[str]:
    """Trích xuất ngày đầu tiên trong chuỗi sang định dạng chuẩn ISO `YYYY-MM-DD`."""
    m_vn = _RE_NGAY_THANG_NAM.search(text)
    if m_vn:
        d_i, m_i, y_i = int(m_vn.group(1)), int(m_vn.group(2)), int(m_vn.group(3))
        if 1 <= m_i <= 12 and 1 <= d_i <= 31:
            return f"{y_i:04d}-{m_i:02d}-{d_i:02d}"
    m_sl = _RE_NGAY_GAU.search(text)
    if m_sl:
        d_i, m_i, y_i = int(m_sl.group(1)), int(m_sl.group(2)), int(m_sl.group(3))
        if 1 <= m_i <= 12 and 1 <= d_i <= 31:
            return f"{y_i:04d}-{m_i:02d}-{d_i:02d}"
    return None


def _trich_so_hieu_chuan(text: str) -> List[str]:
    """Trích xuất danh sách số hiệu văn bản pháp luật chuẩn hóa chữ hoa, bảo toàn thứ tự xuất hiện."""
    seen: List[str] = []
    for m in _RE_SO_HIEU_DOC.findall(text):
        sh = re.sub(r"\s+", "", str(m).strip()).upper()
        if sh and sh not in seen:
            seen.append(sh)
    return seen


def _doan_linh_vuc(van_ban: str) -> str:
    """Suy luận lĩnh vực pháp lý dựa trên từ khóa trong tiêu đề và phần đầu văn bản."""
    mau = van_ban[:1500].lower()
    bang = (
        ("Thuế - Hóa đơn", ("hóa đơn", "thuế", "chứng từ", "gtgt")),
        ("Lao động - BHXH", ("lao động", "bảo hiểm xã hội", "lương hưu", "tiền lương")),
        ("Đất đai - Bất động sản", ("đất đai", "nhà ở", "bất động sản", "giấy chứng nhận")),
        ("Dữ liệu - Điện tử", ("dữ liệu cá nhân", "giao dịch điện tử", "an ninh mạng", "chữ ký số")),
        ("Doanh nghiệp - Thương mại", ("doanh nghiệp", "thương mại", "đầu tư", "đấu thầu", "hợp đồng")),
        ("Dân sự - Hình sự", ("dân sự", "hình sự", "tố tụng", "xử lý vi phạm hành chính")),
    )
    for ten_lv, tu_khoa in bang:
        if any(tk in mau for tk in tu_khoa):
            return ten_lv
    return "Pháp luật Việt Nam"


@dataclass
class KetQuaPhanTichVanBanPhapQuy:
    """Kết quả bóc tách cấu trúc văn bản pháp quy / văn bản hợp nhất tự động."""

    nguon: str
    so_hieu_chinh: str
    ten_van_ban: str
    linh_vuc: str
    ngay_ban_hanh: str
    ngay_hieu_luc: str
    ho_so_hieu_luc: HoSoHieuLucVanBan
    cac_dieu_khoan: List[Dict[str, Any]] = field(default_factory=list)
    phien_ban_dieu_khoan: List[PhienBanDieuKhoan] = field(default_factory=list)
    quan_he_do_thi: List[QuanHePhapLy] = field(default_factory=list)
    la_van_ban_hop_nhat: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển toàn bộ kết quả phân tích sang dict thuần JSON."""
        return {
            "nguon": self.nguon,
            "so_hieu_chinh": self.so_hieu_chinh,
            "ten_van_ban": self.ten_van_ban,
            "linh_vuc": self.linh_vuc,
            "ngay_ban_hanh": self.ngay_ban_hanh,
            "ngay_hieu_luc": self.ngay_hieu_luc,
            "la_van_ban_hop_nhat": self.la_van_ban_hop_nhat,
            "so_dieu_khoan": len(self.cac_dieu_khoan),
            "so_phien_ban_dieu_khoan": len(self.phien_ban_dieu_khoan),
            "so_quan_he_do_thi": len(self.quan_he_do_thi),
            "ho_so_hieu_luc": self.ho_so_hieu_luc.to_dict(),
            "cac_dieu_khoan": self.cac_dieu_khoan,
            "phien_ban_dieu_khoan": [pb.to_dict() for pb in self.phien_ban_dieu_khoan],
            "quan_he_do_thi": [qh.to_dict() for qh in self.quan_he_do_thi],
        }


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Pipeline nạp tự động văn bản pháp quy & văn bản hợp nhất (.docx/.pdf/.txt/.md) "
        "thành kho .vaidb đã build sẵn HNSW, kèm tự sinh chuỗi PhienBanDieuKhoan Point-in-Time và đồ thị tham chiếu."
    ),
)
class BoNapVanBanPhapQuyTuDong:
    """
    Pipeline nạp tự động kho tri thức pháp quy Việt Nam (Vertical Moat):

    1. Đọc thẳng tệp `.docx`, `.pdf`, `.txt`, `.md` hoặc thư mục văn bản qua `BoDocTaiLieuPhapLy`.
    2. Bóc tách tự động số hiệu văn bản, ngày ban hành, ngày có hiệu lực, quan hệ thay thế/sửa đổi/hướng dẫn.
    3. Nhận diện cấu trúc Chương > Điều > Khoản và cước chú Văn Bản Hợp Nhất (`[N]` hoặc ngoặc sửa đổi)
       để tự sinh chuỗi `PhienBanDieuKhoan` Point-in-Time và cạnh `QuanHePhapLy`.
    4. Đóng gói toàn bộ vào 1 tệp `.vaidb` duy nhất đã dựng sẵn chỉ mục HNSW bền vững và lưu kèm
       siêu dữ liệu hiệu lực/phiên bản/đồ thị ngay trong SQLite để tự phục hồi khi nạp lại.
    """

    DUOI_HO_TRO = (".txt", ".md", ".docx", ".pdf")

    def __init__(
        self,
        che_do_nhung: str = "onnx_viet",
        duong_dan_onnx_embed: Optional[Union[str, Path]] = None,
    ) -> None:
        self.che_do_nhung = che_do_nhung
        self.duong_dan_onnx_embed = str(duong_dan_onnx_embed) if duong_dan_onnx_embed else None
        self._bo_doc = BoDocTaiLieuPhapLy()

    def _tach_cuoc_chu_hop_nhat(self, van_ban: str) -> Tuple[str, Dict[str, str]]:
        """Tách phần thân văn bản và từ điển cước chú `[N] -> nội dung` trong văn bản hợp nhất."""
        cuoc_chu: Dict[str, str] = {}
        vi_tri_cat: Optional[int] = None
        for m in _RE_CUOC_CHU_DEF.finditer(van_ban):
            noi_dung_cc = re.sub(r"\s+", " ", m.group(2)).strip()
            if any(
                kw in noi_dung_cc.lower()
                for kw in ("sửa đổi", "bổ sung", "bãi bỏ", "thay thế", "hiệu lực", "nghị định", "luật", "thông tư")
            ):
                if vi_tri_cat is None:
                    vi_tri_cat = m.start()
                cuoc_chu[m.group(1)] = noi_dung_cc
        than = van_ban[:vi_tri_cat].strip() if vi_tri_cat is not None else van_ban
        return than, cuoc_chu

    def _xac_dinh_so_hieu_va_tieu_de(self, van_ban: str, ten_nguon: str) -> Tuple[str, str, bool]:
        """Xác định số hiệu chính, tên văn bản và cờ nhận diện văn bản hợp nhất."""
        cac_dong = [d.strip() for d in van_ban.splitlines() if d.strip()][:20]
        phan_dau = "\n".join(cac_dong)
        la_vbhn = bool(
            re.search(r"văn\s+bản\s+hợp\s+nhất|vbhn", phan_dau, re.IGNORECASE | re.UNICODE)
        )

        ds_sh_dau = _trich_so_hieu_chuan(phan_dau)
        so_hieu_chinh = ""
        if la_vbhn and len(ds_sh_dau) >= 2:
            for sh in ds_sh_dau:
                if "VBHN" not in sh:
                    so_hieu_chinh = sh
                    break
        if not so_hieu_chinh and ds_sh_dau:
            so_hieu_chinh = ds_sh_dau[0]
        if not so_hieu_chinh:
            ds_toan_van = _trich_so_hieu_chuan(van_ban)
            so_hieu_chinh = ds_toan_van[0] if ds_toan_van else f"VB/{Path(ten_nguon).stem.upper()}"

        ten_vb = ""
        for dong in cac_dong:
            if re.match(
                r"^(LUẬT|BỘ\s+LUẬT|NGHỊ\s+ĐỊNH|THÔNG\s+TƯ|QUYẾT\s+ĐỊNH|VĂN\s+BẢN\s+HỢP\s+NHẤT)\b",
                dong,
                re.IGNORECASE | re.UNICODE,
            ):
                ten_vb = dong[:180]
                break
        if not ten_vb:
            ten_vb = f"Văn bản pháp quy {so_hieu_chinh}"
        return so_hieu_chinh, ten_vb, la_vbhn

    def _trich_ngay_hieu_luc(self, van_ban: str) -> Tuple[str, str]:
        """Trích xuất ngày ban hành (phần đầu) và ngày có hiệu lực thi hành."""
        phan_dau = "\n".join(van_ban.splitlines()[:25])
        ngay_bh = _chuan_hoa_ngay_iso(phan_dau) or "2024-01-01"

        m_hl = re.search(
            r"(?:có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:kể\s+)?từ\s+|áp\s+dụng\s+từ\s+)([^.\n;]{5,60})",
            van_ban,
            re.IGNORECASE | re.UNICODE,
        )
        ngay_hl = _chuan_hoa_ngay_iso(m_hl.group(1)) if m_hl else None
        return ngay_bh, (ngay_hl or ngay_bh)

    def phan_tich_van_ban(
        self,
        nguon: Union[str, Path],
        noi_dung_truc_tiep: Optional[str] = None,
        tach_khoan_dieu_dai: bool = False,
        nguong_tach_khoan_tu: int = 180,
    ) -> KetQuaPhanTichVanBanPhapQuy:
        """Phân tích một văn bản pháp quy hoặc tệp `.docx`/`.pdf`/`.txt`/`.md` thành cấu trúc đầy đủ."""
        if noi_dung_truc_tiep is not None:
            van_ban = str(noi_dung_truc_tiep)
            ten_nguon = str(nguon)
        else:
            p = Path(nguon)
            if p.is_file():
                doc = self._bo_doc.doc_tep(p)
                van_ban = str(doc.get("van_ban", ""))
                ten_nguon = p.name
            else:
                van_ban = str(nguon)
                ten_nguon = "van_ban_truc_tiep.txt"

        than_vb, cuoc_chu = self._tach_cuoc_chu_hop_nhat(van_ban)
        so_hieu_chinh, ten_vb, la_vbhn = self._xac_dinh_so_hieu_va_tieu_de(van_ban, ten_nguon)
        ngay_bh, ngay_hl = self._trich_ngay_hieu_luc(van_ban)
        linh_vuc = _doan_linh_vuc(van_ban)

        thay_the_cho: List[str] = []
        quan_he: List[QuanHePhapLy] = []
        da_co_canh: set[Tuple[str, str, str]] = set()

        def _them_canh(src: str, dst: str, loai: str, dieu_lq: str = "", mo_ta: str = "") -> None:
            if not src or not dst or src == dst:
                return
            key = (src, dst, loai)
            if key in da_co_canh:
                return
            da_co_canh.add(key)
            quan_he.append(
                QuanHePhapLy(
                    nguon=src,
                    dich=dst,
                    loai_quan_he=loai,
                    dieu_lien_quan=dieu_lq,
                    mo_ta=mo_ta[:200],
                )
            )

        for m_tt in _RE_THAY_THE.finditer(than_vb):
            doan = m_tt.group(0)
            for sh_cu in _trich_so_hieu_chuan(doan):
                if sh_cu != so_hieu_chinh and sh_cu not in thay_the_cho:
                    thay_the_cho.append(sh_cu)
                    _them_canh(so_hieu_chinh, sh_cu, "thay_the", mo_ta=doan.strip())

        for m_hd in _RE_HUONG_DAN_VB.finditer(than_vb[:2500]):
            doan = m_hd.group(0)
            for sh_goc in _trich_so_hieu_chuan(doan):
                if sh_goc != so_hieu_chinh:
                    _them_canh(so_hieu_chinh, sh_goc, "huong_dan_thi_hanh", mo_ta=doan.strip())

        sua_doi_bo_sung_boi: List[str] = []
        for cc_text in cuoc_chu.values():
            for sh_sd in _trich_so_hieu_chuan(cc_text):
                if sh_sd != so_hieu_chinh and sh_sd not in sua_doi_bo_sung_boi:
                    sua_doi_bo_sung_boi.append(sh_sd)
                    _them_canh(sh_sd, so_hieu_chinh, "sua_doi_bo_sung", mo_ta=cc_text)

        if not la_vbhn:
            for m_sd in _RE_SUA_DOI_VB.finditer(than_vb[:2500]):
                doan = m_sd.group(0)
                for sh_bi_sd in _trich_so_hieu_chuan(doan):
                    if sh_bi_sd != so_hieu_chinh:
                        _them_canh(so_hieu_chinh, sh_bi_sd, "sua_doi_bo_sung", mo_ta=doan.strip())

        cac_dieu: List[Dict[str, Any]] = []
        phien_ban_list: List[PhienBanDieuKhoan] = []

        chuong_hien_tai = ""
        chuong_marks = [(m.start(), f"{m.group(1)} {m.group(2)}".strip()) for m in _RE_CHUONG_HEADER.finditer(than_vb)]
        dieu_matches = list(_RE_DIEU_HEADER.finditer(than_vb))

        for i_d, m_d in enumerate(dieu_matches):
            start_pos = m_d.start()
            end_pos = dieu_matches[i_d + 1].start() if i_d + 1 < len(dieu_matches) else len(than_vb)
            for c_pos, c_label in chuong_marks:
                if c_pos < start_pos:
                    chuong_hien_tai = c_label
            ten_dieu = f"Điều {m_d.group(2)}"
            tieu_de_dieu = re.sub(r"\[\d{1,3}\]", "", m_d.group(3)).strip()
            noi_dung_tho = than_vb[start_pos:end_pos].strip()
            refs = _RE_CUOC_CHU_REF.findall(noi_dung_tho)
            noi_dung_sach = re.sub(r"\[\d{1,3}\]", "", noi_dung_tho).strip()

            dieu_goc = {
                "ma": f"{so_hieu_chinh.lower().replace('/', '_')}_dieu_{m_d.group(2).lower()}",
                "van_ban": ten_vb,
                "so_hieu": so_hieu_chinh,
                "chuong": chuong_hien_tai,
                "dieu": ten_dieu,
                "khoan": "",
                "tieu_de": tieu_de_dieu,
                "noi_dung": noi_dung_sach,
            }
            if tach_khoan_dieu_dai:
                cac_dieu.extend(
                    self.tach_khoan_dieu_dai(dieu_goc, nguong_tu=nguong_tach_khoan_tu)
                )
            else:
                cac_dieu.append(dieu_goc)

            ghi_chu_sd = ""
            sh_sua_dieu: Optional[str] = None
            ngay_sd_dieu: Optional[str] = None
            for ref_id in refs:
                if ref_id in cuoc_chu:
                    cc_val = cuoc_chu[ref_id]
                    ghi_chu_sd = cc_val
                    sh_list = _trich_so_hieu_chuan(cc_val)
                    sh_sua_dieu = next((x for x in sh_list if x != so_hieu_chinh), None)
                    ngay_sd_dieu = _chuan_hoa_ngay_iso(cc_val)
                    if sh_sua_dieu:
                        _them_canh(sh_sua_dieu, so_hieu_chinh, "sua_doi_bo_sung", dieu_lq=ten_dieu, mo_ta=cc_val)
                    break

            if sh_sua_dieu is None:
                m_in = _RE_SUA_DOI_INLINE.search(noi_dung_sach)
                if m_in:
                    in_txt = m_in.group(1)
                    sh_list = _trich_so_hieu_chuan(in_txt)
                    sh_sua_dieu = next((x for x in sh_list if x != so_hieu_chinh), None)
                    ngay_sd_dieu = _chuan_hoa_ngay_iso(in_txt)
                    ghi_chu_sd = in_txt
                    if sh_sua_dieu:
                        if sh_sua_dieu not in sua_doi_bo_sung_boi:
                            sua_doi_bo_sung_boi.append(sh_sua_dieu)
                        _them_canh(sh_sua_dieu, so_hieu_chinh, "sua_doi_bo_sung", dieu_lq=ten_dieu, mo_ta=in_txt)

            if sh_sua_dieu is not None and ngay_sd_dieu is not None:
                trang_thai_dieu = "bi_bai_bo" if "bãi bỏ" in ghi_chu_sd.lower() else "da_sua_doi_bo_sung"
                phien_ban_list.append(
                    PhienBanDieuKhoan(
                        so_hieu=so_hieu_chinh,
                        dieu=ten_dieu,
                        noi_dung=noi_dung_sach,
                        hieu_luc_tu=ngay_sd_dieu,
                        het_hieu_luc_tu=None,
                        sua_doi_boi=sh_sua_dieu,
                        trang_thai=trang_thai_dieu,
                        ghi_chu=ghi_chu_sd[:240],
                    )
                )
            else:
                phien_ban_list.append(
                    PhienBanDieuKhoan(
                        so_hieu=so_hieu_chinh,
                        dieu=ten_dieu,
                        noi_dung=noi_dung_sach,
                        hieu_luc_tu=ngay_hl,
                        het_hieu_luc_tu=None,
                        sua_doi_boi=None,
                        trang_thai="hieu_luc",
                        ghi_chu=f"Trích xuất tự động từ {ten_nguon}",
                    )
                )

        if not cac_dieu and than_vb.strip():
            cac_dieu.append(
                {
                    "ma": f"{so_hieu_chinh.lower().replace('/', '_')}_toan_van",
                    "van_ban": ten_vb,
                    "so_hieu": so_hieu_chinh,
                    "chuong": "",
                    "dieu": "Toàn văn",
                    "tieu_de": ten_vb,
                    "noi_dung": than_vb.strip(),
                }
            )

        ho_so = HoSoHieuLucVanBan(
            so_hieu=so_hieu_chinh,
            ten_van_ban=ten_vb,
            linh_vuc=linh_vuc,
            ngay_ban_hanh=ngay_bh,
            ngay_hieu_luc=ngay_hl,
            thay_the_cho=thay_the_cho,
            sua_doi_bo_sung_boi=sua_doi_bo_sung_boi,
            ghi_chu=f"Nạp tự động từ {ten_nguon}" + (" (Văn bản hợp nhất)" if la_vbhn else ""),
        )

        return KetQuaPhanTichVanBanPhapQuy(
            nguon=ten_nguon,
            so_hieu_chinh=so_hieu_chinh,
            ten_van_ban=ten_vb,
            linh_vuc=linh_vuc,
            ngay_ban_hanh=ngay_bh,
            ngay_hieu_luc=ngay_hl,
            ho_so_hieu_luc=ho_so,
            cac_dieu_khoan=cac_dieu,
            phien_ban_dieu_khoan=phien_ban_list,
            quan_he_do_thi=quan_he,
            la_van_ban_hop_nhat=la_vbhn,
        )

    @staticmethod
    def tach_khoan_dieu_dai(
        dieu_item: Dict[str, Any],
        nguong_tu: int = 180,
    ) -> List[Dict[str, Any]]:
        """Tách các Điều dài có từ 2 Khoản đánh số trở lên thành từng đoạn Khoản kèm ngữ cảnh cha."""
        noi_dung = str(dieu_item.get("noi_dung", "")).strip()
        if len(noi_dung.split()) < max(40, int(nguong_tu)):
            return [dieu_item]

        khoan_matches = list(_RE_KHOAN_HEADER.finditer(noi_dung))
        if len(khoan_matches) < 2:
            return [dieu_item]

        phan_dau_dieu = noi_dung[: khoan_matches[0].start()].strip()
        ket_qua: List[Dict[str, Any]] = []
        for idx_k, m_k in enumerate(khoan_matches):
            bat_dau = m_k.start()
            ket_thuc = (
                khoan_matches[idx_k + 1].start()
                if idx_k + 1 < len(khoan_matches)
                else len(noi_dung)
            )
            so_khoan = m_k.group(1)
            doan_khoan = noi_dung[bat_dau:ket_thuc].strip()
            noi_dung_kem_cha = (
                f"{phan_dau_dieu}\n{doan_khoan}".strip() if phan_dau_dieu else doan_khoan
            )
            ban_sao = dict(dieu_item)
            ban_sao["ma"] = f"{dieu_item['ma']}_khoan_{so_khoan}"
            ban_sao["khoan"] = f"Khoản {so_khoan}"
            ban_sao["noi_dung"] = noi_dung_kem_cha
            ket_qua.append(ban_sao)
        return ket_qua

    @staticmethod
    def kiem_tra_chat_luong_bo_van_ban(
        ket_qua_list: Sequence[KetQuaPhanTichVanBanPhapQuy],
    ) -> Dict[str, Any]:
        """Kiểm định tự động chất lượng bộ văn bản pháp quy trước khi đóng gói `.vaidb`."""
        canh_bao: List[str] = []
        loi: List[str] = []
        so_hieu_da_nap = {kq.so_hieu_chinh for kq in ket_qua_list if kq.so_hieu_chinh}
        tong_dieu = 0

        for kq in ket_qua_list:
            tong_dieu += len(kq.cac_dieu_khoan)
            if not kq.cac_dieu_khoan:
                loi.append(f"{kq.so_hieu_chinh}: không trích xuất được điều khoản nào.")
            if not kq.ngay_hieu_luc:
                canh_bao.append(f"{kq.so_hieu_chinh}: thiếu ngày có hiệu lực.")

            so_dieu_thu_tu: List[int] = []
            dieu_da_thay: set[Tuple[str, str]] = set()
            for item in kq.cac_dieu_khoan:
                dieu_lbl = str(item.get("dieu", ""))
                khoan_lbl = str(item.get("khoan", ""))
                khoa_trung = (dieu_lbl, khoan_lbl)
                if khoa_trung in dieu_da_thay:
                    canh_bao.append(
                        f"{kq.so_hieu_chinh}: trùng lặp {dieu_lbl}{' ' + khoan_lbl if khoan_lbl else ''}."
                    )
                dieu_da_thay.add(khoa_trung)

                nd = str(item.get("noi_dung", "")).strip()
                if len(nd.split()) < 8:
                    canh_bao.append(f"{kq.so_hieu_chinh} {dieu_lbl}: nội dung quá ngắn ({len(nd.split())} từ).")

                m_num = re.search(r"Điều\s+(\d+)", dieu_lbl)
                if m_num:
                    n_val = int(m_num.group(1))
                    if not so_dieu_thu_tu or so_dieu_thu_tu[-1] != n_val:
                        so_dieu_thu_tu.append(n_val)

            if len(so_dieu_thu_tu) >= 2:
                sap_xep = sorted(set(so_dieu_thu_tu))
                for truoc, sau in zip(sap_xep, sap_xep[1:]):
                    if sau - truoc > 1:
                        canh_bao.append(
                            f"{kq.so_hieu_chinh}: số Điều nhảy cóc từ Điều {truoc} sang Điều {sau}."
                        )

            for qh in kq.quan_he_do_thi:
                if qh.dich and qh.dich not in so_hieu_da_nap:
                    canh_bao.append(
                        f"{kq.so_hieu_chinh}: quan hệ '{qh.loai_quan_he}' trỏ tới '{qh.dich}' chưa có trong tập đầu vào."
                    )

        return {
            "dat_chat_luong": len(loi) == 0,
            "so_van_ban": len(ket_qua_list),
            "tong_dieu": tong_dieu,
            "so_loi": len(loi),
            "so_canh_bao": len(canh_bao),
            "loi": loi,
            "canh_bao": canh_bao,
        }

    def dong_goi_kho_vaidb(
        self,
        nguon_dau_vao: Union[str, Path, Sequence[Union[str, Path]]],
        duong_dan_vaidb: Union[str, Path],
        xay_dung_hnsw: bool = True,
        hnsw_m: int = 12,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 48,
        de_quy: bool = True,
        tach_khoan_dieu_dai: bool = False,
        nguong_tach_khoan_tu: int = 180,
        xuat_manifest: bool = False,
    ) -> Dict[str, Any]:
        """Nạp 1 tệp, nhiều tệp hoặc cả thư mục pháp quy và đóng gói thành 1 tệp `.vaidb` đã dựng sẵn HNSW + Point-in-Time + Đồ thị."""
        from vietnamese_ai.rag.rag_pipeline import RAGPipeline

        danh_sach_tep: List[Path] = []
        nguon_items: Sequence[Union[str, Path]] = (
            [nguon_dau_vao] if isinstance(nguon_dau_vao, (str, Path)) else nguon_dau_vao
        )
        for item in nguon_items:
            p = Path(item)
            if p.is_dir():
                pattern = "**/*" if de_quy else "*"
                for child in sorted(p.glob(pattern)):
                    if child.is_file() and child.suffix.lower() in self.DUOI_HO_TRO:
                        danh_sach_tep.append(child)
            elif p.is_file():
                danh_sach_tep.append(p)
            else:
                raise FileNotFoundError(f"Không tìm thấy tệp hoặc thư mục văn bản pháp quy: {item}")

        if not danh_sach_tep:
            raise ValueError("Không tìm thấy tệp văn bản pháp quy hợp lệ (.txt/.md/.docx/.pdf).")

        out_db = Path(duong_dan_vaidb)
        out_db.parent.mkdir(parents=True, exist_ok=True)

        rag = RAGPipeline(
            backend="vaidb",
            duong_dan_vaidb=str(out_db),
            che_do_nhung=self.che_do_nhung,
            duong_dan_onnx_embed=self.duong_dan_onnx_embed,
            che_do_rerank="onnx_viet",
            che_do_hanh_chinh=True,
        )

        ho_so_list: List[Dict[str, Any]] = []
        phien_ban_list: List[Dict[str, Any]] = []
        quan_he_list: List[Dict[str, Any]] = []
        ket_qua_van_ban: List[Dict[str, Any]] = []
        ket_qua_phan_tich_list: List[KetQuaPhanTichVanBanPhapQuy] = []
        tong_dieu_nap = 0

        try:
            for tep in danh_sach_tep:
                kq = self.phan_tich_van_ban(
                    tep,
                    tach_khoan_dieu_dai=tach_khoan_dieu_dai,
                    nguong_tach_khoan_tu=nguong_tach_khoan_tu,
                )
                ket_qua_phan_tich_list.append(kq)
                ho_so_list.append(kq.ho_so_hieu_luc.to_dict())
                phien_ban_list.extend(pb.to_dict() for pb in kq.phien_ban_dieu_khoan)
                quan_he_list.extend(qh.to_dict() for qh in kq.quan_he_do_thi)
                ket_qua_van_ban.append(
                    {
                        "tep": tep.name,
                        "so_hieu": kq.so_hieu_chinh,
                        "ten_van_ban": kq.ten_van_ban,
                        "so_dieu": len(kq.cac_dieu_khoan),
                        "so_phien_ban": len(kq.phien_ban_dieu_khoan),
                        "so_quan_he": len(kq.quan_he_do_thi),
                        "la_van_ban_hop_nhat": kq.la_van_ban_hop_nhat,
                    }
                )
                for dieu_item in kq.cac_dieu_khoan:
                    khoan_tag = f" | {dieu_item['khoan']}" if dieu_item.get("khoan") else ""
                    noi_dung_day_du = (
                        f"[{dieu_item['van_ban']} - {dieu_item['so_hieu']}{khoan_tag}]\n"
                        f"{dieu_item['dieu']}: {dieu_item['tieu_de']}\n"
                        f"{dieu_item['noi_dung']}"
                    )
                    ma_doc = f"{dieu_item['ma']}_{hashlib.sha256(noi_dung_day_du.encode('utf-8')).hexdigest()[:8]}"
                    rag.them_tai_lieu(
                        ma_doc,
                        noi_dung_day_du,
                        metadata={
                            "van_ban": dieu_item["van_ban"],
                            "so_hieu": dieu_item["so_hieu"],
                            "chuong": dieu_item["chuong"],
                            "dieu": dieu_item["dieu"],
                            "khoan": dieu_item.get("khoan", ""),
                            "tieu_de": dieu_item["tieu_de"],
                            "tep_nguon": tep.name,
                        },
                        che_do_hanh_chinh=True,
                    )
                    tong_dieu_nap += 1

            hnsw_info: Optional[Dict[str, Any]] = None
            if xay_dung_hnsw and hasattr(rag.csdl_vector, "xay_dung_chi_muc_hnsw"):
                hnsw_info = rag.csdl_vector.xay_dung_chi_muc_hnsw(
                    m=hnsw_m,
                    ef_construction=hnsw_ef_construction,
                    ef_search=hnsw_ef_search,
                    luu_xuong_dia=True,
                )

            conn = getattr(rag.csdl_vector, "_conn", None)
            if conn is not None:
                self._ghi_sieu_du_lieu_phap_ly_sql(conn, ho_so_list, phien_ban_list, quan_he_list)

            thong_ke_db = (
                rag.csdl_vector.thong_ke()
                if hasattr(rag.csdl_vector, "thong_ke")
                else rag.thong_ke()
            )
        finally:
            rag.dong()

        bao_cao_chat_luong = self.kiem_tra_chat_luong_bo_van_ban(ket_qua_phan_tich_list)
        sha256_db = hashlib.sha256(out_db.read_bytes()).hexdigest() if out_db.is_file() else ""
        duong_dan_manifest: Optional[str] = None
        if xuat_manifest and out_db.is_file():
            from datetime import datetime, timezone

            manifest_path = Path(f"{out_db}.manifest.json")
            manifest_data = {
                "ten_tep": out_db.name,
                "sha256": sha256_db,
                "kich_thuoc_bytes": out_db.stat().st_size,
                "ngay_tao_utc": datetime.now(timezone.utc).isoformat(),
                "so_tep_da_nap": len(danh_sach_tep),
                "tong_dieu_khoan": tong_dieu_nap,
                "tach_khoan_dieu_dai": bool(tach_khoan_dieu_dai),
                "chat_luong": bao_cao_chat_luong,
            }
            manifest_path.write_text(
                json.dumps(manifest_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            duong_dan_manifest = str(manifest_path)

        return {
            "duong_dan_vaidb": str(out_db),
            "duong_dan_manifest": duong_dan_manifest,
            "sha256": sha256_db,
            "so_tep_da_nap": len(danh_sach_tep),
            "tong_dieu_khoan": tong_dieu_nap,
            "so_ho_so_hieu_luc": len(ho_so_list),
            "so_phien_ban_point_in_time": len(phien_ban_list),
            "so_quan_he_do_thi": len(quan_he_list),
            "hnsw": hnsw_info,
            "chat_luong": bao_cao_chat_luong,
            "thong_ke_vaidb": thong_ke_db,
            "chi_tiet_van_ban": ket_qua_van_ban,
        }

    @staticmethod
    def _ghi_sieu_du_lieu_phap_ly_sql(
        conn: sqlite3.Connection,
        ho_so_list: List[Dict[str, Any]],
        phien_ban_list: List[Dict[str, Any]],
        quan_he_list: List[Dict[str, Any]],
    ) -> None:
        """Gộp và lưu bền vững siêu dữ liệu pháp lý (hiệu lực, Point-in-Time, đồ thị) vào bảng `vai_meta` của `.vaidb`."""

        def _doc_cu(khoa: str) -> List[Dict[str, Any]]:
            row = conn.execute("SELECT gia_tri FROM vai_meta WHERE khoa = ?", (khoa,)).fetchone()
            if not row:
                return []
            try:
                parsed = json.loads(str(row[0]))
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []

        merged_hs = _doc_cu(META_KEY_VALIDITY) + ho_so_list
        merged_pb = _doc_cu(META_KEY_CLAUSES) + phien_ban_list
        merged_qh = _doc_cu(META_KEY_RELATIONS) + quan_he_list

        conn.executemany(
            "INSERT OR REPLACE INTO vai_meta(khoa, gia_tri) VALUES (?, ?)",
            [
                (META_KEY_VALIDITY, json.dumps(merged_hs, ensure_ascii=False)),
                (META_KEY_CLAUSES, json.dumps(merged_pb, ensure_ascii=False)),
                (META_KEY_RELATIONS, json.dumps(merged_qh, ensure_ascii=False)),
            ],
        )
        conn.commit()

    @staticmethod
    def nap_sieu_du_lieu_phap_ly_tu_vaidb(
        duong_dan_vaidb: Union[str, Path],
        bo_theo_doi: BoTheoDoiHieuLucPhapLy,
        do_thi: Optional[DoThiThamChieuPhapLy] = None,
    ) -> Dict[str, int]:
        """Tự động phục hồi hồ sơ hiệu lực, phiên bản Điều/Khoản Point-in-Time và đồ thị tham chiếu từ tệp `.vaidb`."""
        p = Path(duong_dan_vaidb)
        if not p.is_file():
            return {"ho_so": 0, "phien_ban": 0, "quan_he": 0}

        so_hs = 0
        so_pb = 0
        so_qh = 0
        conn = sqlite3.connect(str(p))
        try:
            rows = conn.execute(
                "SELECT khoa, gia_tri FROM vai_meta WHERE khoa IN (?, ?, ?)",
                (META_KEY_VALIDITY, META_KEY_CLAUSES, META_KEY_RELATIONS),
            ).fetchall()
        except sqlite3.OperationalError:
            return {"ho_so": 0, "phien_ban": 0, "quan_he": 0}
        finally:
            conn.close()

        meta_map = {str(k): str(v) for k, v in rows}

        if META_KEY_VALIDITY in meta_map:
            for item in json.loads(meta_map[META_KEY_VALIDITY]):
                if isinstance(item, dict) and item.get("so_hieu"):
                    bo_theo_doi.dang_ky_van_ban(
                        HoSoHieuLucVanBan(
                            so_hieu=str(item["so_hieu"]),
                            ten_van_ban=str(item.get("ten_van_ban", item["so_hieu"])),
                            linh_vuc=str(item.get("linh_vuc", "Pháp luật Việt Nam")),
                            ngay_ban_hanh=str(item.get("ngay_ban_hanh", "2024-01-01")),
                            ngay_hieu_luc=str(item.get("ngay_hieu_luc", "2024-01-01")),
                            ngay_het_hieu_luc=item.get("ngay_het_hieu_luc"),
                            thay_the_cho=list(item.get("thay_the_cho") or []),
                            bi_thay_the_boi=list(item.get("bi_thay_the_boi") or []),
                            sua_doi_bo_sung_boi=list(item.get("sua_doi_bo_sung_boi") or []),
                            ghi_chu=str(item.get("ghi_chu", "")),
                        )
                    )
                    so_hs += 1

        if META_KEY_CLAUSES in meta_map:
            for item in json.loads(meta_map[META_KEY_CLAUSES]):
                if isinstance(item, dict) and item.get("so_hieu") and item.get("dieu"):
                    bo_theo_doi.dang_ky_phien_ban_dieu_khoan(
                        PhienBanDieuKhoan(
                            so_hieu=str(item["so_hieu"]),
                            dieu=str(item["dieu"]),
                            noi_dung=str(item.get("noi_dung", "")),
                            hieu_luc_tu=str(item.get("hieu_luc_tu", "2024-01-01")),
                            het_hieu_luc_tu=item.get("het_hieu_luc_tu"),
                            khoan=item.get("khoan"),
                            sua_doi_boi=item.get("sua_doi_boi"),
                            trang_thai=str(item.get("trang_thai", "hieu_luc")),
                            ghi_chu=str(item.get("ghi_chu", "")),
                        )
                    )
                    so_pb += 1

        if do_thi is not None and META_KEY_RELATIONS in meta_map:
            for item in json.loads(meta_map[META_KEY_RELATIONS]):
                if isinstance(item, dict) and item.get("nguon") and item.get("dich"):
                    if do_thi.them_quan_he(
                        QuanHePhapLy(
                            nguon=str(item["nguon"]),
                            dich=str(item["dich"]),
                            loai_quan_he=str(item.get("loai_quan_he", "lien_quan")),
                            dieu_lien_quan=str(item.get("dieu_lien_quan", "")),
                            mo_ta=str(item.get("mo_ta", "")),
                        )
                    ):
                        so_qh += 1

        return {"ho_so": so_hs, "phien_ban": so_pb, "quan_he": so_qh}


AutoLegalCorpusPipeline = BoNapVanBanPhapQuyTuDong

__all__ = [
    "KetQuaPhanTichVanBanPhapQuy",
    "BoNapVanBanPhapQuyTuDong",
    "AutoLegalCorpusPipeline",
    "META_KEY_VALIDITY",
    "META_KEY_CLAUSES",
    "META_KEY_RELATIONS",
]
