"""BoKiemChungTrichDanPhapLy (LegalCitationVerifier) - Kiểm Chứng Trích Dẫn & Hiệu Lực Thời Gian 3 Tầng.

Kiểm chứng đầu ra sinh văn bản (từ mẫu quy tắc, Ollama, Gemini hoặc OpenAI) trước khi trả về người dùng:
1. Tầng 1 — Bám sát trích dẫn (Citation Grounding): phát hiện số hiệu văn bản/Điều khoản bịa đặt không có trong nguồn.
2. Tầng 2 — Nhất quán thời gian (Temporal Consistency): phát hiện xung đột khi câu trả lời viện dẫn điều luật đã hết hiệu lực
   hoặc chưa có hiệu lực tại mốc thời gian truy vấn `t` mà không có cảnh báo thay thế.
3. Tầng 3 — Nhất quán định lượng (Numeric & Deadline Consistency): kiểm tra tỷ lệ %, thời hạn ngày/tháng/năm/giờ và hạn mức tiền
   so với điều khoản gốc để chống sai lệch con số pháp lý.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from vietnamese_ai.rag.legal.temporal_query import BoPhanTichThoiGianPhapLy
from vietnamese_ai.rag.legal.validity_tracker import (
    BoTheoDoiHieuLucPhapLy,
    _chuan_hoa_dieu,
    _chuan_hoa_so_hieu,
    _phan_tich_ngay,
)
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


@dataclass
class KetQuaKiemChungTrichDan:
    """Báo cáo kiểm chứng 3 tầng cho một câu trả lời pháp lý."""

    hop_le: bool
    ngay_tra_cuu: str
    diem_trung_thuc: float
    diem_nhat_quan_thoi_gian: float
    diem_nhat_quan_so_lieu: float
    co_xung_dot_thoi_gian: bool
    co_trich_dan_ao: bool
    trich_dan_hop_le: List[str] = field(default_factory=list)
    trich_dan_khong_co_trong_nguon: List[str] = field(default_factory=list)
    trich_dan_het_hieu_luc: List[Dict[str, Any]] = field(default_factory=list)
    so_lieu_lech_nguon: List[str] = field(default_factory=list)
    canh_bao: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển báo cáo kiểm chứng trích dẫn sang từ điển JSON chuẩn."""
        return asdict(self)


def _chuan_hoa_so_thuc(gia_tri: str) -> str:
    """Chuẩn hóa chuỗi số liệu pháp lý (ví dụ `03` -> `3`, `500.000.000` -> `500000000`)."""
    s = gia_tri.strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
        s = s.replace(".", "")
    else:
        s = s.replace(",", ".")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    s = s.lstrip("0") or "0"
    return s


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ kiểm chứng trích dẫn pháp lý 3 tầng (Citation Grounding, Temporal Consistency, "
        "Numeric/Deadline Consistency) chống ảo giác điều luật và xung đột hiệu lực theo thời gian."
    ),
    nhom_extra="core",
)
class BoKiemChungTrichDanPhapLy:
    """Bộ kiểm chứng trích dẫn & hiệu lực pháp lý tất định (Deterministic 3-Tier Legal Verifier)."""

    _RE_SO_HIEU = re.compile(r"\b(\d{1,4}/\d{4}/[A-ZĐa-zđ0-9\-]+)\b", re.UNICODE)
    _RE_DIEU = re.compile(r"\bđiều\s+(\d+[a-zA-ZđĐ]?)\b", re.IGNORECASE | re.UNICODE)
    _RE_NGAY_TRONG_CAU = re.compile(
        r"(?:ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+(?:19|20)\d{2}|"
        r"\b\d{1,2}[./\-]\d{1,2}[./\-](?:19|20)\d{2}\b|"
        r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|"
        r"tháng\s+\d{1,2}\s*(?:[/\-]|năm\s+)(?:19|20)\d{2}|"
        r"(?:luật|bộ\s+luật|nghị\s+định|thông\s+tư)[^\n,.;?!]{0,40}?\b(?:19|20)\d{2}\b|"
        r"\bnăm\s+(?:19|20)\d{2}\b)",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_SO_LIEU = re.compile(
        r"\b(\d+(?:[.,]\d+)*)\s*(%|phần\s+trăm|ngày\s+làm\s+việc|ngày|tháng|năm|giờ|tuần|tỷ\s+đồng|triệu\s+đồng|đồng|vnđ|vnd)\b",
        re.IGNORECASE | re.UNICODE,
    )
    _TU_KHOA_CANH_BAO_HET_HL = (
        "hết hiệu lực",
        "đã bị bãi bỏ",
        "bãi bỏ",
        "thay thế bởi",
        "được thay thế",
        "không còn hiệu lực",
        "không còn áp dụng",
        "chưa có hiệu lực",
        "cảnh báo",
        "ngừng áp dụng",
        "trước khi sửa đổi",
        "bản cũ",
        "không tồn tại",
        "từ chối trả lời",
        "thiếu căn cứ",
        "ngoài phạm vi",
    )

    def __init__(
        self,
        bo_theo_doi: Optional[BoTheoDoiHieuLucPhapLy] = None,
        bo_phan_tich_thoi_gian: Optional[BoPhanTichThoiGianPhapLy] = None,
    ) -> None:
        self.bo_theo_doi = bo_theo_doi or BoTheoDoiHieuLucPhapLy()
        self.bo_phan_tich_thoi_gian = bo_phan_tich_thoi_gian or BoPhanTichThoiGianPhapLy()

    @classmethod
    def _trich_cap_so_lieu(cls, van_ban: str) -> List[Tuple[str, str, str]]:
        """Trích các cặp (giá_trị_chuẩn, đơn_vị_chuẩn, cụm_gốc) và bỏ qua ngày tháng/năm ban hành luật."""
        sach = cls._RE_SO_HIEU.sub(" ", van_ban)
        sach = cls._RE_NGAY_TRONG_CAU.sub(" ", sach)
        ket_qua: List[Tuple[str, str, str]] = []
        da_thay: Set[Tuple[str, str]] = set()
        for m in cls._RE_SO_LIEU.finditer(sach):
            val = _chuan_hoa_so_thuc(m.group(1))
            unit = re.sub(r"\s+", " ", m.group(2).strip().lower())
            if unit == "phần trăm":
                unit = "%"
            elif unit in ("vnđ", "vnd"):
                unit = "đồng"
            if unit == "năm" and len(val) == 4 and val.startswith(("19", "20")):
                continue
            khoa = (val, unit)
            if khoa not in da_thay:
                da_thay.add(khoa)
                ket_qua.append((val, unit, m.group(0).strip()))
        return ket_qua

    @classmethod
    def _co_canh_bao_xung_quanh(cls, van_ban: str, so_hieu: str) -> bool:
        """Kiểm tra xem câu chứa `so_hieu` có phải là lời cảnh báo hết hiệu lực/thay thế hay không."""
        low_sh = so_hieu.lower()
        for cau in re.split(r"[\n.!?;]+", van_ban):
            cau_low = cau.lower()
            pos_sh = cau_low.find(low_sh)
            if pos_sh != -1:
                if any(kw in cau_low for kw in cls._TU_KHOA_CANH_BAO_HET_HL):
                    return True
                pos_tt = cau_low.find("thay thế")
                if pos_tt != -1 and pos_tt < pos_sh:
                    return True
        return False

    def kiem_chung(
        self,
        cau_tra_loi: str,
        nguon_truy_hoi: List[Dict[str, Any]],
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        cau_hoi: Optional[str] = None,
    ) -> KetQuaKiemChungTrichDan:
        """Kiểm chứng 3 tầng cho `cau_tra_loi` dựa trên `nguon_truy_hoi` và mốc thời gian `ngay_tra_cuu`."""
        tra_loi = str(cau_tra_loi or "").strip()
        if ngay_tra_cuu is not None:
            moc = _phan_tich_ngay(ngay_tra_cuu)
        elif cau_hoi:
            moc = _phan_tich_ngay(self.bo_phan_tich_thoi_gian.trich_xuat_ngay(cau_hoi))
        else:
            moc = _phan_tich_ngay(None)

        so_hieu_nguon: Set[str] = set()
        dieu_nguon: Set[str] = set()
        van_ban_nguon_gop: List[str] = []
        nguon_theo_sh: Dict[str, List[Dict[str, Any]]] = {}

        for ng in nguon_truy_hoi:
            if not isinstance(ng, dict):
                continue
            meta = ng.get("metadata", {}) if isinstance(ng.get("metadata"), dict) else {}
            noi_dung = " ".join(
                str(x)
                for x in (
                    ng.get("noi_dung", ""),
                    ng.get("van_ban", ""),
                    ng.get("trich_doan", ""),
                    ng.get("tieu_de", ""),
                    meta.get("van_ban", ""),
                    meta.get("ten_van_ban", ""),
                    meta.get("dieu", ""),
                    meta.get("so_hieu", ""),
                    ng.get("so_hieu", ""),
                    ng.get("dieu", ""),
                )
                if x
            )
            van_ban_nguon_gop.append(noi_dung)

            for sh_raw in (meta.get("so_hieu"), ng.get("so_hieu")):
                if isinstance(sh_raw, str) and sh_raw.strip():
                    sh_n = _chuan_hoa_so_hieu(sh_raw)
                    so_hieu_nguon.add(sh_n)
                    nguon_theo_sh.setdefault(sh_n, []).append(ng)
            for sh_m in self._RE_SO_HIEU.findall(noi_dung):
                sh_n = _chuan_hoa_so_hieu(sh_m)
                so_hieu_nguon.add(sh_n)
                nguon_theo_sh.setdefault(sh_n, []).append(ng)

            for d_raw in (meta.get("dieu"), ng.get("dieu")):
                if isinstance(d_raw, str) and d_raw.strip():
                    dieu_nguon.add(_chuan_hoa_dieu(d_raw))
            for d_m in self._RE_DIEU.findall(noi_dung):
                dieu_nguon.add(_chuan_hoa_dieu(d_m))

        so_hieu_trong_hoi: Set[str] = set()
        if cau_hoi:
            for sh_q in self._RE_SO_HIEU.findall(cau_hoi):
                so_hieu_trong_hoi.add(_chuan_hoa_so_hieu(sh_q))

        ds_sh_tra_loi: List[str] = []
        for sh_m in self._RE_SO_HIEU.findall(tra_loi):
            sh_n = _chuan_hoa_so_hieu(sh_m)
            if sh_n not in ds_sh_tra_loi:
                ds_sh_tra_loi.append(sh_n)

        ds_dieu_tra_loi: List[str] = []
        for d_m in self._RE_DIEU.findall(tra_loi):
            d_n = _chuan_hoa_dieu(d_m)
            if d_n not in ds_dieu_tra_loi:
                ds_dieu_tra_loi.append(d_n)

        trich_dan_hop_le: List[str] = []
        trich_dan_ao: List[str] = []
        canh_bao: List[str] = []

        for sh in ds_sh_tra_loi:
            if sh in so_hieu_nguon:
                trich_dan_hop_le.append(sh)
            elif sh in so_hieu_trong_hoi and self._co_canh_bao_xung_quanh(tra_loi, sh):
                trich_dan_hop_le.append(sh)
            else:
                trich_dan_ao.append(sh)
                canh_bao.append(f"Trích dẫn văn bản '{sh}' không xuất hiện trong nguồn truy hồi.")

        for d in ds_dieu_tra_loi:
            if d in dieu_nguon:
                trich_dan_hop_le.append(d)
            else:
                trich_dan_ao.append(d)
                canh_bao.append(f"Trích dẫn '{d}' không xuất hiện trong các điều khoản nguồn.")

        tong_td = len(trich_dan_hop_le) + len(trich_dan_ao)
        diem_trung_thuc = round(len(trich_dan_hop_le) / tong_td, 4) if tong_td > 0 else 1.0
        co_trich_dan_ao = len(trich_dan_ao) > 0

        trich_dan_het_hieu_luc: List[Dict[str, Any]] = []
        so_kiem_tra_tg = 0
        for sh in ds_sh_tra_loi:
            dang_canh_bao = self._co_canh_bao_xung_quanh(tra_loi, sh)
            hl = self.bo_theo_doi.kiem_tra_hieu_luc(sh, ngay_tra_cuu=moc)
            co_xung_dot_sh = False

            if hl.get("tim_thay") and hl.get("con_hieu_luc") is False:
                so_kiem_tra_tg += 1
                if not dang_canh_bao:
                    co_xung_dot_sh = True
                    trich_dan_het_hieu_luc.append(
                        {
                            "so_hieu": sh,
                            "trang_thai": hl["trang_thai"],
                            "ngay_tra_cuu": moc.isoformat(),
                            "thay_the_boi": hl.get("bi_thay_the_boi", []),
                        }
                    )
                    canh_bao.append(
                        f"XUNG ĐỘT THỜI GIAN: Câu trả lời viện dẫn '{sh}' ({hl['trang_thai']}) "
                        f"tại mốc {moc.isoformat()} mà không cảnh báo hiệu lực."
                    )
            else:
                nguon_sh = nguon_theo_sh.get(sh, [])
                if nguon_sh:
                    so_kiem_tra_tg += 1
                    tat_ca_het_hl = True
                    co_meta_tg = False
                    for item_ng in nguon_sh:
                        meta_ng = item_ng.get("metadata", {}) if isinstance(item_ng.get("metadata"), dict) else {}
                        tu_s = meta_ng.get("hieu_luc_tu") or item_ng.get("hieu_luc_tu")
                        den_s = meta_ng.get("het_hieu_luc_tu") or item_ng.get("het_hieu_luc_tu")
                        if tu_s or den_s:
                            co_meta_tg = True
                            tu_d = _phan_tich_ngay(tu_s) if tu_s else date(1900, 1, 1)
                            den_d = _phan_tich_ngay(den_s) if den_s else None
                            if moc >= tu_d and (den_d is None or moc < den_d):
                                tat_ca_het_hl = False
                    if co_meta_tg and tat_ca_het_hl and not dang_canh_bao and not co_xung_dot_sh:
                        trich_dan_het_hieu_luc.append(
                            {
                                "so_hieu": sh,
                                "trang_thai": "het_hieu_luc_phien_ban_dieu_khoan",
                                "ngay_tra_cuu": moc.isoformat(),
                                "thay_the_boi": [],
                            }
                        )
                        canh_bao.append(
                            f"XUNG ĐỘT THỜI GIAN: Phiên bản điều khoản của '{sh}' trong nguồn không còn hiệu lực tại {moc.isoformat()}."
                        )

        co_xung_dot_thoi_gian = len(trich_dan_het_hieu_luc) > 0
        if so_kiem_tra_tg > 0:
            diem_nhat_quan_tg = round(max(0.0, 1.0 - len(trich_dan_het_hieu_luc) / so_kiem_tra_tg), 4)
        else:
            diem_nhat_quan_tg = 1.0

        nguon_text = "\n".join(van_ban_nguon_gop)
        so_lieu_nguon = {(val, unit) for val, unit, _ in self._trich_cap_so_lieu(nguon_text)}
        so_lieu_tra_loi = self._trich_cap_so_lieu(tra_loi)
        so_lieu_lech: List[str] = []
        for val, unit, cum_goc in so_lieu_tra_loi:
            if (val, unit) not in so_lieu_nguon:
                if unit == "ngày" and (val, "ngày làm việc") in so_lieu_nguon:
                    continue
                so_lieu_lech.append(cum_goc)
                canh_bao.append(f"Số liệu/thời hạn '{cum_goc}' không khớp trực tiếp với điều khoản nguồn.")

        if so_lieu_tra_loi:
            diem_so_lieu = round(max(0.0, 1.0 - len(so_lieu_lech) / len(so_lieu_tra_loi)), 4)
        else:
            diem_so_lieu = 1.0

        hop_le = (not co_trich_dan_ao) and (not co_xung_dot_thoi_gian) and (diem_so_lieu >= 0.8)

        return KetQuaKiemChungTrichDan(
            hop_le=hop_le,
            ngay_tra_cuu=moc.isoformat(),
            diem_trung_thuc=diem_trung_thuc,
            diem_nhat_quan_thoi_gian=diem_nhat_quan_tg,
            diem_nhat_quan_so_lieu=diem_so_lieu,
            co_xung_dot_thoi_gian=co_xung_dot_thoi_gian,
            co_trich_dan_ao=co_trich_dan_ao,
            trich_dan_hop_le=trich_dan_hop_le,
            trich_dan_khong_co_trong_nguon=trich_dan_ao,
            trich_dan_het_hieu_luc=trich_dan_het_hieu_luc,
            so_lieu_lech_nguon=so_lieu_lech,
            canh_bao=canh_bao,
        )


LegalCitationVerificationResult = KetQuaKiemChungTrichDan
LegalCitationVerifier = BoKiemChungTrichDanPhapLy

__all__ = [
    "KetQuaKiemChungTrichDan",
    "LegalCitationVerificationResult",
    "BoKiemChungTrichDanPhapLy",
    "LegalCitationVerifier",
]
