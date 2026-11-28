"""BoPhanTichThoiGianPhapLy (TemporalQueryParser) - Trích Xuất Mốc Thời Gian Hiệu Lực Từ Câu Hỏi Pháp Lý Tiếng Việt.

Hỗ trợ nhận diện chính xác mốc thời gian áp dụng pháp luật từ ngôn ngữ tự nhiên tiếng Việt
(ngày/tháng/năm, trước/từ/sau mốc sửa đổi, quý, đầu/cuối năm, cụm từ 'hiện hành'),
đồng thời tự động loại trừ nhiễu từ số hiệu văn bản (`123/2020/NĐ-CP`) và tên đạo luật (`Luật Đất đai 2024`).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional, Union

from vietnamese_ai.rag.legal.validity_tracker import _phan_tich_ngay
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


@dataclass
class KetQuaPhanTichThoiGian:
    """Kết quả phân tích mốc thời gian hiệu lực từ câu hỏi pháp lý tiếng Việt."""

    ngay_tra_cuu: str
    co_moc_tuong_minh: bool
    bieu_thuc_goc: str
    kieu_quan_he_thoi_gian: str
    do_tin_cay: float

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển kết quả phân tích thời gian sang từ điển JSON chuẩn."""
        return asdict(self)


def _tao_ngay_an_toan(nam: int, thang: int, ngay: int) -> Optional[date]:
    """Tạo đối tượng `date` hợp lệ (kẹp ngày theo số ngày thực tế của tháng)."""
    if not (1950 <= nam <= 2100 and 1 <= thang <= 12):
        return None
    so_ngay_thang = calendar.monthrange(nam, thang)[1]
    ngay_hop_le = max(1, min(ngay, so_ngay_thang))
    return date(nam, thang, ngay_hop_le)


def _chuan_hoa_quan_he(tien_to: str) -> str:
    """Ánh xạ cụm tiền tố thời gian tiếng Việt sang mã quan hệ chuẩn hóa."""
    low = tien_to.strip().lower()
    if low == "trước":
        return "truoc_ngay"
    if low == "sau":
        return "sau_ngay"
    if low in ("từ", "kể từ"):
        return "tu_ngay"
    if low == "đến":
        return "den_ngay"
    return "tai_ngay"


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ phân tích mốc thời gian pháp lý tự động từ câu hỏi tiếng Việt, "
        "tách biệt số hiệu/năm ban hành đạo luật với mốc thời gian áp dụng thực tế."
    ),
    nhom_extra="core",
)
class BoPhanTichThoiGianPhapLy:
    """Trình bóc tách mốc thời gian áp dụng pháp luật (Point-in-Time Query Parser) cho tiếng Việt."""

    _RE_SO_HIEU = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐa-zđ0-9\-]+\b", re.UNICODE)
    _RE_TEN_VAN_BAN_CO_NAM = re.compile(
        r"\b(?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|quyết\s+định|nghị\s+quyết|pháp\s+lệnh)"
        r"(?:\s+[^\n,.;?!()]{1,55}?)?\s+(?:năm\s+)?(?:19|20)\d{2}\b",
        re.IGNORECASE | re.UNICODE,
    )

    _RE_NGAY_ISO = re.compile(
        r"(?:(trước|kể\s+từ|từ|sau|vào|tại|đến)\s+(?:ngày\s+)?)?\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_NGAY_VN_CHU = re.compile(
        r"(?:(trước|kể\s+từ|từ|sau|vào|tại|đến)\s+)?ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+((?:19|20)\d{2})\b",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_NGAY_VN_SO = re.compile(
        r"(?:(trước|kể\s+từ|từ|sau|vào|tại|đến)\s+)?(?:ngày\s+)?(\d{1,2})[./\-](\d{1,2})[./\-]((?:19|20)\d{2})\b",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_THANG_NAM = re.compile(
        r"(?:(trước|kể\s+từ|từ|sau|vào|tại|trong)\s+)?tháng\s+(\d{1,2})\s*(?:[/\-]|năm\s+)((?:19|20)\d{2})\b",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_QUY_NAM = re.compile(
        r"(?:(trước|kể\s+từ|từ|sau|vào|tại|trong)\s+)?quý\s+(IV|III|II|I|[1-4])\s*(?:[/\-]|năm\s+)((?:19|20)\d{2})\b",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_NAM = re.compile(
        r"(?:(đầu|giữa|cuối|trước|kể\s+từ|từ|sau|vào|tại|trong|đến|sang)\s+)?năm\s+((?:19|20)\d{2})\b",
        re.IGNORECASE | re.UNICODE,
    )
    _RE_HIEN_HANH = re.compile(
        r"\b(hiện\s+hành|hiện\s+nay|hiện\s+tại|mới\s+nhất|đang\s+áp\s+dụng|đang\s+có\s+hiệu\s+lực)\b",
        re.IGNORECASE | re.UNICODE,
    )

    _MAP_QUY: Dict[str, int] = {
        "1": 2,
        "I": 2,
        "2": 5,
        "II": 5,
        "3": 8,
        "III": 8,
        "4": 11,
        "IV": 11,
    }

    @classmethod
    def _nam_luat_lon_nhat(cls, van_ban: str) -> int:
        """Tìm năm ban hành lớn nhất của văn bản luật được nhắc tường minh trong câu hỏi."""
        nam_max = 0
        for m in cls._RE_SO_HIEU.finditer(van_ban):
            m_nam = re.search(r"/((?:19|20)\d{2})/", m.group(0))
            if m_nam:
                nam_max = max(nam_max, int(m_nam.group(1)))
        for m in cls._RE_TEN_VAN_BAN_CO_NAM.finditer(van_ban):
            m_nam = re.search(r"\b((?:19|20)\d{2})\b", m.group(0))
            if m_nam:
                nam_max = max(nam_max, int(m_nam.group(1)))
        return nam_max

    @classmethod
    def _che_ten_luat_va_so_hieu(cls, van_ban: str) -> str:
        """Che số hiệu và năm nằm trong tên văn bản luật để tránh nhận nhầm làm mốc thời gian hỏi."""
        sach = cls._RE_SO_HIEU.sub(" __SO_HIEU__ ", van_ban)
        sach = cls._RE_TEN_VAN_BAN_CO_NAM.sub(" __TEN_VAN_BAN__ ", sach)
        return sach

    def phan_tich(
        self,
        cau_hoi: str,
        ngay_mac_dinh: Optional[Union[str, date]] = None,
    ) -> KetQuaPhanTichThoiGian:
        """Phân tích câu hỏi pháp lý tiếng Việt và trả về mốc thời gian hiệu lực chuẩn hóa ISO `YYYY-MM-DD`."""
        moc_mac_dinh = _phan_tich_ngay(ngay_mac_dinh)
        text = str(cau_hoi or "").strip()
        if not text:
            return KetQuaPhanTichThoiGian(
                ngay_tra_cuu=moc_mac_dinh.isoformat(),
                co_moc_tuong_minh=False,
                bieu_thuc_goc="",
                kieu_quan_he_thoi_gian="mac_dinh",
                do_tin_cay=0.5,
            )

        nam_luat_min = self._nam_luat_lon_nhat(text)
        sach = self._che_ten_luat_va_so_hieu(text)
        hoi_trang_thai_hl = "hiệu lực" in text.lower()

        for m_iso in self._RE_NGAY_ISO.finditer(sach):
            tien_to = m_iso.group(1) or ""
            dt = _tao_ngay_an_toan(int(m_iso.group(2)), int(m_iso.group(3)), int(m_iso.group(4)))
            if dt is not None:
                quan_he = _chuan_hoa_quan_he(tien_to)
                if quan_he == "truoc_ngay":
                    dt = dt - timedelta(days=1)
                elif quan_he == "sau_ngay":
                    dt = dt + timedelta(days=1)
                la_ngay_dieu_kien_cu = (
                    nam_luat_min > 0
                    and dt.year < nam_luat_min
                    and m_iso.start() > 6
                    and not hoi_trang_thai_hl
                )
                if la_ngay_dieu_kien_cu:
                    continue
                return KetQuaPhanTichThoiGian(
                    ngay_tra_cuu=dt.isoformat(),
                    co_moc_tuong_minh=True,
                    bieu_thuc_goc=m_iso.group(0).strip(),
                    kieu_quan_he_thoi_gian=quan_he,
                    do_tin_cay=0.98,
                )

        for regex in (self._RE_NGAY_VN_CHU, self._RE_NGAY_VN_SO):
            for m_vn in regex.finditer(sach):
                tien_to = m_vn.group(1) or ""
                dt = _tao_ngay_an_toan(int(m_vn.group(4)), int(m_vn.group(3)), int(m_vn.group(2)))
                if dt is not None:
                    quan_he = _chuan_hoa_quan_he(tien_to)
                    if quan_he == "truoc_ngay":
                        dt = dt - timedelta(days=1)
                    elif quan_he == "sau_ngay":
                        dt = dt + timedelta(days=1)
                    la_ngay_dieu_kien_cu = (
                        nam_luat_min > 0
                        and dt.year < nam_luat_min
                        and m_vn.start() > 6
                        and not hoi_trang_thai_hl
                    )
                    if la_ngay_dieu_kien_cu:
                        continue
                    return KetQuaPhanTichThoiGian(
                        ngay_tra_cuu=dt.isoformat(),
                        co_moc_tuong_minh=True,
                        bieu_thuc_goc=m_vn.group(0).strip(),
                        kieu_quan_he_thoi_gian=quan_he,
                        do_tin_cay=0.96,
                    )

        m_thang = self._RE_THANG_NAM.search(sach)
        if m_thang:
            tien_to = (m_thang.group(1) or "").strip().lower()
            thang = int(m_thang.group(2))
            nam = int(m_thang.group(3))
            if tien_to == "trước":
                dt = _tao_ngay_an_toan(nam, thang, 1)
                if dt is not None:
                    dt = dt - timedelta(days=1)
                    return KetQuaPhanTichThoiGian(
                        ngay_tra_cuu=dt.isoformat(),
                        co_moc_tuong_minh=True,
                        bieu_thuc_goc=m_thang.group(0).strip(),
                        kieu_quan_he_thoi_gian="truoc_ngay",
                        do_tin_cay=0.92,
                    )
            elif tien_to in ("từ", "kể từ"):
                dt = _tao_ngay_an_toan(nam, thang, 1)
                if dt is not None:
                    return KetQuaPhanTichThoiGian(
                        ngay_tra_cuu=dt.isoformat(),
                        co_moc_tuong_minh=True,
                        bieu_thuc_goc=m_thang.group(0).strip(),
                        kieu_quan_he_thoi_gian="tu_ngay",
                        do_tin_cay=0.92,
                    )
            elif tien_to == "sau":
                thang_sau = thang + 1
                nam_sau = nam + (1 if thang_sau > 12 else 0)
                thang_sau = 1 if thang_sau > 12 else thang_sau
                dt = _tao_ngay_an_toan(nam_sau, thang_sau, 1)
                if dt is not None:
                    return KetQuaPhanTichThoiGian(
                        ngay_tra_cuu=dt.isoformat(),
                        co_moc_tuong_minh=True,
                        bieu_thuc_goc=m_thang.group(0).strip(),
                        kieu_quan_he_thoi_gian="sau_ngay",
                        do_tin_cay=0.90,
                    )
            else:
                dt = _tao_ngay_an_toan(nam, thang, 15)
                if dt is not None:
                    return KetQuaPhanTichThoiGian(
                        ngay_tra_cuu=dt.isoformat(),
                        co_moc_tuong_minh=True,
                        bieu_thuc_goc=m_thang.group(0).strip(),
                        kieu_quan_he_thoi_gian="trong_thang",
                        do_tin_cay=0.90,
                    )

        m_quy = self._RE_QUY_NAM.search(sach)
        if m_quy:
            tien_to = (m_quy.group(1) or "").strip().lower()
            quy_key = m_quy.group(2).upper()
            nam = int(m_quy.group(3))
            thang_giua = self._MAP_QUY.get(quy_key, 6)
            thang_dau_quy = max(1, thang_giua - 1)
            thang_cuoi_quy = min(12, thang_giua + 1)
            if tien_to == "trước":
                dt = _tao_ngay_an_toan(nam, thang_dau_quy, 1)
                if dt is not None:
                    dt = dt - timedelta(days=1)
                quan_he = "truoc_ngay"
            elif tien_to in ("từ", "kể từ"):
                dt = _tao_ngay_an_toan(nam, thang_dau_quy, 1)
                quan_he = "tu_ngay"
            elif tien_to == "sau":
                thang_sau = thang_cuoi_quy + 1
                nam_sau = nam + (1 if thang_sau > 12 else 0)
                thang_sau = 1 if thang_sau > 12 else thang_sau
                dt = _tao_ngay_an_toan(nam_sau, thang_sau, 1)
                quan_he = "sau_ngay"
            else:
                dt = _tao_ngay_an_toan(nam, thang_giua, 15)
                quan_he = "trong_thang"
            if dt is not None:
                return KetQuaPhanTichThoiGian(
                    ngay_tra_cuu=dt.isoformat(),
                    co_moc_tuong_minh=True,
                    bieu_thuc_goc=m_quy.group(0).strip(),
                    kieu_quan_he_thoi_gian=quan_he,
                    do_tin_cay=0.86,
                )

        ds_nam = list(self._RE_NAM.finditer(sach))
        if ds_nam:
            m_nam = ds_nam[-1] if len(ds_nam) >= 2 else ds_nam[0]
            tien_to = (m_nam.group(1) or "").strip().lower()
            nam = int(m_nam.group(2))
            if tien_to == "trước":
                dt = _tao_ngay_an_toan(nam - 1, 12, 31)
                quan_he = "truoc_ngay"
            elif tien_to == "đầu":
                dt = _tao_ngay_an_toan(nam, 1, 15)
                quan_he = "trong_nam"
            elif tien_to == "cuối":
                dt = _tao_ngay_an_toan(nam, 12, 15)
                quan_he = "trong_nam"
            elif tien_to in ("từ", "kể từ"):
                dt = _tao_ngay_an_toan(nam, 1, 1)
                quan_he = "tu_ngay"
            elif tien_to == "sau":
                dt = _tao_ngay_an_toan(nam + 1, 1, 1)
                quan_he = "sau_ngay"
            else:
                dt = _tao_ngay_an_toan(nam, 7, 1)
                quan_he = "trong_nam"
            if dt is not None:
                return KetQuaPhanTichThoiGian(
                    ngay_tra_cuu=dt.isoformat(),
                    co_moc_tuong_minh=True,
                    bieu_thuc_goc=m_nam.group(0).strip(),
                    kieu_quan_he_thoi_gian=quan_he,
                    do_tin_cay=0.84,
                )

        m_hh = self._RE_HIEN_HANH.search(text)
        if m_hh:
            return KetQuaPhanTichThoiGian(
                ngay_tra_cuu=moc_mac_dinh.isoformat(),
                co_moc_tuong_minh=True,
                bieu_thuc_goc=m_hh.group(0).strip(),
                kieu_quan_he_thoi_gian="hien_hanh",
                do_tin_cay=0.90,
            )

        return KetQuaPhanTichThoiGian(
            ngay_tra_cuu=moc_mac_dinh.isoformat(),
            co_moc_tuong_minh=False,
            bieu_thuc_goc="",
            kieu_quan_he_thoi_gian="mac_dinh",
            do_tin_cay=0.5,
        )

    def trich_xuat_ngay(
        self,
        cau_hoi: str,
        ngay_mac_dinh: Optional[Union[str, date]] = None,
    ) -> str:
        """Trích xuất nhanh chuỗi ngày tra cứu chuẩn ISO `YYYY-MM-DD` từ câu hỏi pháp lý."""
        return self.phan_tich(cau_hoi, ngay_mac_dinh=ngay_mac_dinh).ngay_tra_cuu


TemporalQueryResult = KetQuaPhanTichThoiGian
TemporalQueryParser = BoPhanTichThoiGianPhapLy

__all__ = [
    "KetQuaPhanTichThoiGian",
    "TemporalQueryResult",
    "BoPhanTichThoiGianPhapLy",
    "TemporalQueryParser",
]
