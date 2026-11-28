"""BoTheoDoiHieuLucPhapLy (LegalValidityTracker) - Theo Dõi Hiệu Lực, Sửa Đổi & Thay Thế Văn Bản Pháp Luật Việt Nam.

Hỗ trợ:
- Tra cứu trạng thái hiệu lực theo mốc thời gian (`con_hieu_luc`, `con_hieu_luc_da_sua_doi`, `chua_co_hieu_luc`, `het_hieu_luc`).
- Theo dõi chuỗi thay thế (`thay_the_cho` / `bi_thay_the_boi`) và sửa đổi, bổ sung (`sua_doi_bo_sung_boi`).
- Tự động quét mọi số hiệu văn bản trích dẫn trong hợp đồng, công văn hoặc kết quả RAG để cảnh báo văn bản hết hiệu lực.
- Bổ sung trọng số ưu tiên văn bản còn hiệu lực cho truy hồi RAG Pháp lý Việt Nam.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


@dataclass
class HoSoHieuLucVanBan:
    """Hồ sơ hiệu lực của một văn bản quy phạm pháp luật Việt Nam."""

    so_hieu: str
    ten_van_ban: str
    linh_vuc: str
    ngay_ban_hanh: str
    ngay_hieu_luc: str
    ngay_het_hieu_luc: Optional[str] = None
    thay_the_cho: List[str] = field(default_factory=list)
    bi_thay_the_boi: List[str] = field(default_factory=list)
    sua_doi_bo_sung_boi: List[str] = field(default_factory=list)
    ghi_chu: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển hồ sơ hiệu lực sang dict thuần JSON."""
        return asdict(self)


@dataclass
class PhienBanDieuKhoan:
    """Phiên bản hiệu lực theo mốc thời gian (Point-in-Time) của một Điều/Khoản cụ thể."""

    so_hieu: str
    dieu: str
    noi_dung: str
    hieu_luc_tu: str
    het_hieu_luc_tu: Optional[str] = None
    khoan: Optional[str] = None
    diem: Optional[str] = None
    sua_doi_boi: Optional[str] = None
    trang_thai: str = "hieu_luc"
    ghi_chu: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển phiên bản điều khoản sang dict thuần JSON."""
        return asdict(self)


def _chuan_hoa_dieu(dieu: str) -> str:
    """Chuẩn hóa mã Điều luật (ví dụ: `dieu 9`, `9`, `Điều 09` -> `Điều 9`)."""
    s = str(dieu or "").strip()
    m = re.search(r"(\d+[a-zA-ZđĐ]?)", s)
    if m:
        num = m.group(1).lstrip("0") or "0"
        return f"Điều {num}"
    return s


CO_SO_PHIEN_BAN_DIEU_KHOAN_VN: List[PhienBanDieuKhoan] = [
    PhienBanDieuKhoan(
        so_hieu="123/2020/NĐ-CP",
        dieu="Điều 9",
        khoan="Khoản 1",
        noi_dung=(
            "Thời điểm lập hóa đơn đối với bán hàng hóa là thời điểm chuyển giao quyền sở hữu hoặc quyền sử dụng "
            "hàng hóa cho người mua, không phân biệt đã thu được tiền hay chưa thu được tiền."
        ),
        hieu_luc_tu="2022-07-01",
        het_hieu_luc_tu="2025-06-01",
        sua_doi_boi="70/2025/NĐ-CP",
        trang_thai="bi_sua_doi",
        ghi_chu="Bản gốc Nghị định 123/2020/NĐ-CP trước khi Nghị định 70/2025/NĐ-CP có hiệu lực.",
    ),
    PhienBanDieuKhoan(
        so_hieu="123/2020/NĐ-CP",
        dieu="Điều 9",
        khoan="Khoản 1",
        noi_dung=(
            "Thời điểm lập hóa đơn đối với bán hàng hóa (bao gồm cả bán, chuyển nhượng tài sản công và bán hàng "
            "dự trữ quốc gia) là thời điểm chuyển giao quyền sở hữu hoặc quyền sử dụng; đối với xuất khẩu hàng hóa "
            "do người bán tự xác định nhưng chậm nhất không quá ngày làm việc tiếp theo kể từ ngày hàng hóa được thông quan."
        ),
        hieu_luc_tu="2025-06-01",
        het_hieu_luc_tu=None,
        sua_doi_boi="70/2025/NĐ-CP",
        trang_thai="da_sua_doi_bo_sung",
        ghi_chu="Đã sửa đổi, bổ sung bởi Khoản 6 Điều 1 Nghị định 70/2025/NĐ-CP từ 01/06/2025.",
    ),
    PhienBanDieuKhoan(
        so_hieu="123/2020/NĐ-CP",
        dieu="Điều 19",
        noi_dung=(
            "Xử lý hóa đơn có sai sót: cho phép hủy hóa đơn điện tử đã cấp mã chưa gửi người mua hoặc lập "
            "hóa đơn điều chỉnh/thay thế đối với sai sót về mã số thuế, số tiền, thuế suất."
        ),
        hieu_luc_tu="2022-07-01",
        het_hieu_luc_tu="2025-06-01",
        sua_doi_boi="70/2025/NĐ-CP",
        trang_thai="bi_sua_doi",
        ghi_chu="Bản gốc Nghị định 123/2020/NĐ-CP trước 01/06/2025.",
    ),
    PhienBanDieuKhoan(
        so_hieu="123/2020/NĐ-CP",
        dieu="Điều 19",
        noi_dung=(
            "Bãi bỏ quy định hủy hóa đơn đã lập sai; người nộp thuế thực hiện thông báo sai sót (mẫu 04/SS-HĐĐT) "
            "hoặc lập hóa đơn điều chỉnh/thay thế kèm biên bản thỏa thuận giữa người bán và người mua."
        ),
        hieu_luc_tu="2025-06-01",
        het_hieu_luc_tu=None,
        sua_doi_boi="70/2025/NĐ-CP",
        trang_thai="da_sua_doi_bo_sung",
        ghi_chu="Sửa đổi theo Khoản 13 Điều 1 Nghị định 70/2025/NĐ-CP có hiệu lực từ 01/06/2025.",
    ),
    PhienBanDieuKhoan(
        so_hieu="58/2014/QH13",
        dieu="Điều 54",
        noi_dung=(
            "Người lao động khi nghỉ việc có đủ 20 năm đóng bảo hiểm xã hội trở lên và đủ tuổi nghỉ hưu "
            "thì được hưởng lương hưu hằng tháng."
        ),
        hieu_luc_tu="2016-01-01",
        het_hieu_luc_tu="2025-07-01",
        sua_doi_boi="41/2024/QH15",
        trang_thai="bi_thay_the",
        ghi_chu="Quy định 20 năm đóng BHXH của Luật BHXH 2014 trước khi Luật BHXH 2024 có hiệu lực.",
    ),
    PhienBanDieuKhoan(
        so_hieu="41/2024/QH15",
        dieu="Điều 64",
        noi_dung=(
            "Người lao động khi nghỉ việc có thời gian đóng bảo hiểm xã hội bắt buộc từ đủ 15 năm trở lên "
            "và đủ tuổi nghỉ hưu theo quy định của Bộ luật Lao động thì được hưởng lương hưu."
        ),
        hieu_luc_tu="2025-07-01",
        het_hieu_luc_tu=None,
        sua_doi_boi=None,
        trang_thai="hieu_luc",
        ghi_chu="Giảm điều kiện số năm đóng BHXH tối thiểu từ 20 năm xuống 15 năm kể từ 01/07/2025.",
    ),
    PhienBanDieuKhoan(
        so_hieu="13/2008/QH12",
        dieu="Điều 12",
        noi_dung=(
            "Điều kiện khấu trừ thuế GTGT đầu vào: có chứng từ thanh toán không dùng tiền mặt đối với "
            "hàng hóa, dịch vụ mua vào từ 20 triệu đồng trở lên."
        ),
        hieu_luc_tu="2009-01-01",
        het_hieu_luc_tu="2025-07-01",
        sua_doi_boi="48/2024/QH15",
        trang_thai="bi_thay_the",
        ghi_chu="Ngưỡng 20 triệu đồng theo Luật Thuế GTGT 2008 (hết hiệu lực từ 01/07/2025).",
    ),
    PhienBanDieuKhoan(
        so_hieu="48/2024/QH15",
        dieu="Điều 14",
        noi_dung=(
            "Điều kiện khấu trừ thuế GTGT đầu vào: phải có chứng từ thanh toán không dùng tiền mặt đối với "
            "hàng hóa, dịch vụ mua vào, trừ một số trường hợp đặc thù theo quy định của Chính phủ."
        ),
        hieu_luc_tu="2025-07-01",
        het_hieu_luc_tu=None,
        sua_doi_boi=None,
        trang_thai="hieu_luc",
        ghi_chu="Siết chặt điều kiện thanh toán không dùng tiền mặt kể từ 01/07/2025 theo Luật Thuế GTGT 2024.",
    ),
    PhienBanDieuKhoan(
        so_hieu="45/2013/QH13",
        dieu="Điều 114",
        noi_dung=(
            "Căn cứ nguyên tắc, phương pháp định giá đất và khung giá đất, UBND cấp tỉnh xây dựng "
            "và trình HĐND cùng cấp thông qua bảng giá đất định kỳ 05 năm một lần."
        ),
        hieu_luc_tu="2014-07-01",
        het_hieu_luc_tu="2024-08-01",
        sua_doi_boi="31/2024/QH15",
        trang_thai="bi_thay_the",
        ghi_chu="Cơ chế khung giá đất và bảng giá đất 5 năm theo Luật Đất đai 2013 (hết hiệu lực từ 01/08/2024).",
    ),
    PhienBanDieuKhoan(
        so_hieu="31/2024/QH15",
        dieu="Điều 159",
        noi_dung=(
            "Bãi bỏ khung giá đất; UBND cấp tỉnh xây dựng, trình HĐND cùng cấp quyết định bảng giá đất "
            "lần đầu để công bố và áp dụng từ ngày 01/01/2026 và điều chỉnh, sửa đổi, bổ sung hằng năm."
        ),
        hieu_luc_tu="2024-08-01",
        het_hieu_luc_tu=None,
        sua_doi_boi=None,
        trang_thai="hieu_luc",
        ghi_chu="Luật Đất đai 2024 bỏ khung giá đất và chuyển sang cập nhật bảng giá đất hằng năm.",
    ),
    PhienBanDieuKhoan(
        so_hieu="51/2005/QH11",
        dieu="Điều 21",
        noi_dung=(
            "Chữ ký điện tử được tạo lập dưới dạng từ, chữ, số, ký hiệu, âm thanh hoặc các hình thức khác "
            "bằng phương tiện điện tử, gắn liền hoặc kết hợp một cách lô gíc với thông điệp dữ liệu."
        ),
        hieu_luc_tu="2006-03-01",
        het_hieu_luc_tu="2024-07-01",
        sua_doi_boi="20/2023/QH15",
        trang_thai="bi_thay_the",
        ghi_chu="Quy định chữ ký điện tử theo Luật Giao dịch điện tử 2005 (hết hiệu lực từ 01/07/2024).",
    ),
    PhienBanDieuKhoan(
        so_hieu="20/2023/QH15",
        dieu="Điều 22",
        noi_dung=(
            "Phân loại chữ ký điện tử thành 03 nhóm: chữ ký điện tử chuyên dùng, chữ ký số công cộng "
            "và chữ ký số chuyên dùng công vụ; bảo đảm tính xác thực và toàn vẹn dữ liệu."
        ),
        hieu_luc_tu="2024-07-01",
        het_hieu_luc_tu=None,
        sua_doi_boi=None,
        trang_thai="hieu_luc",
        ghi_chu="Luật Giao dịch điện tử 2023 có hiệu lực từ 01/07/2024.",
    ),
    PhienBanDieuKhoan(
        so_hieu="65/2014/QH13",
        dieu="Điều 118",
        noi_dung=(
            "Giao dịch mua bán, cho thuê mua, tặng cho, đổi, thế chấp, góp vốn bằng nhà ở phải có "
            "Giấy chứng nhận theo quy định của pháp luật, trừ các trường hợp ngoại lệ luật định."
        ),
        hieu_luc_tu="2015-07-01",
        het_hieu_luc_tu="2024-08-01",
        sua_doi_boi="27/2023/QH15",
        trang_thai="bi_thay_the",
        ghi_chu="Điều kiện giao dịch nhà ở theo Luật Nhà ở 2014 (thay thế bởi Luật Nhà ở 2023 từ 01/08/2024).",
    ),
    PhienBanDieuKhoan(
        so_hieu="27/2023/QH15",
        dieu="Điều 160",
        noi_dung=(
            "Nhà ở tham gia giao dịch phải có Giấy chứng nhận, không thuộc diện tranh chấp, khiếu nại, "
            "khiếu kiện về quyền sở hữu, không bị kê biên thi hành án và còn thời hạn sở hữu nhà ở."
        ),
        hieu_luc_tu="2024-08-01",
        het_hieu_luc_tu=None,
        sua_doi_boi=None,
        trang_thai="hieu_luc",
        ghi_chu="Luật Nhà ở 2023 có hiệu lực sớm từ 01/08/2024.",
    ),
]


def _chuan_hoa_so_hieu(so_hieu: str) -> str:
    """Chuẩn hóa số hiệu văn bản pháp luật (ví dụ: `13/2023/nđ-cp` -> `13/2023/NĐ-CP`)."""
    sach = re.sub(r"\s+", "", str(so_hieu or "").strip())
    return sach.upper()


def _phan_tich_ngay(ngay: Optional[Union[str, date, datetime]]) -> date:
    """Chuyển chuỗi ISO `YYYY-MM-DD`, `DD/MM/YYYY` hoặc `datetime` sang đối tượng `datetime.date`."""
    if ngay is None:
        return date.today()
    if isinstance(ngay, datetime):
        return ngay.date()
    if isinstance(ngay, date):
        return ngay
    s = str(ngay).strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 3:
            d_i, m_i, y_i = int(parts[0]), int(parts[1]), int(parts[2])
            return date(y_i, m_i, d_i)
    return date.fromisoformat(s)


CO_SO_HIEU_LUC_PHAP_LUAT_VN: List[HoSoHieuLucVanBan] = [
    HoSoHieuLucVanBan(
        so_hieu="45/2019/QH14",
        ten_van_ban="Bộ luật Lao động 2019",
        linh_vuc="Lao động",
        ngay_ban_hanh="2019-11-20",
        ngay_hieu_luc="2021-01-01",
        thay_the_cho=["10/2012/QH13"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="10/2012/QH13",
        ten_van_ban="Bộ luật Lao động 2012",
        linh_vuc="Lao động",
        ngay_ban_hanh="2012-06-18",
        ngay_hieu_luc="2013-05-01",
        ngay_het_hieu_luc="2021-01-01",
        bi_thay_the_boi=["45/2019/QH14"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="91/2015/QH13",
        ten_van_ban="Bộ luật Dân sự 2015",
        linh_vuc="Dân sự",
        ngay_ban_hanh="2015-11-24",
        ngay_hieu_luc="2017-01-01",
        thay_the_cho=["33/2005/QH11"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="33/2005/QH11",
        ten_van_ban="Bộ luật Dân sự 2005",
        linh_vuc="Dân sự",
        ngay_ban_hanh="2005-06-14",
        ngay_hieu_luc="2006-01-01",
        ngay_het_hieu_luc="2017-01-01",
        bi_thay_the_boi=["91/2015/QH13"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="59/2020/QH14",
        ten_van_ban="Luật Doanh nghiệp 2020",
        linh_vuc="Doanh nghiệp",
        ngay_ban_hanh="2020-06-17",
        ngay_hieu_luc="2021-01-01",
        thay_the_cho=["68/2014/QH13"],
        sua_doi_bo_sung_boi=["03/2022/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="68/2014/QH13",
        ten_van_ban="Luật Doanh nghiệp 2014",
        linh_vuc="Doanh nghiệp",
        ngay_ban_hanh="2014-11-26",
        ngay_hieu_luc="2015-07-01",
        ngay_het_hieu_luc="2021-01-01",
        bi_thay_the_boi=["59/2020/QH14"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="31/2024/QH15",
        ten_van_ban="Luật Đất đai 2024",
        linh_vuc="Đất đai",
        ngay_ban_hanh="2024-01-18",
        ngay_hieu_luc="2024-08-01",
        thay_the_cho=["45/2013/QH13"],
        ghi_chu="Có hiệu lực sớm từ 01/08/2024 theo Luật 43/2024/QH15.",
    ),
    HoSoHieuLucVanBan(
        so_hieu="45/2013/QH13",
        ten_van_ban="Luật Đất đai 2013",
        linh_vuc="Đất đai",
        ngay_ban_hanh="2013-11-29",
        ngay_hieu_luc="2014-07-01",
        ngay_het_hieu_luc="2024-08-01",
        bi_thay_the_boi=["31/2024/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="20/2023/QH15",
        ten_van_ban="Luật Giao dịch điện tử 2023",
        linh_vuc="Giao dịch điện tử",
        ngay_ban_hanh="2023-06-22",
        ngay_hieu_luc="2024-07-01",
        thay_the_cho=["51/2005/QH11"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="51/2005/QH11",
        ten_van_ban="Luật Giao dịch điện tử 2005",
        linh_vuc="Giao dịch điện tử",
        ngay_ban_hanh="2005-11-29",
        ngay_hieu_luc="2006-03-01",
        ngay_het_hieu_luc="2024-07-01",
        bi_thay_the_boi=["20/2023/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="13/2023/NĐ-CP",
        ten_van_ban="Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân",
        linh_vuc="Dữ liệu cá nhân",
        ngay_ban_hanh="2023-04-17",
        ngay_hieu_luc="2023-07-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="30/2020/NĐ-CP",
        ten_van_ban="Nghị định 30/2020/NĐ-CP về công tác văn thư",
        linh_vuc="Hành chính & Văn thư",
        ngay_ban_hanh="2020-03-05",
        ngay_hieu_luc="2020-03-05",
    ),
    HoSoHieuLucVanBan(
        so_hieu="24/2018/QH14",
        ten_van_ban="Luật An ninh mạng 2018",
        linh_vuc="An ninh mạng",
        ngay_ban_hanh="2018-06-12",
        ngay_hieu_luc="2019-01-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="19/2023/QH15",
        ten_van_ban="Luật Bảo vệ quyền lợi người tiêu dùng 2023",
        linh_vuc="Thương mại & Tiêu dùng",
        ngay_ban_hanh="2023-06-20",
        ngay_hieu_luc="2024-07-01",
        thay_the_cho=["59/2010/QH12"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="59/2010/QH12",
        ten_van_ban="Luật Bảo vệ quyền lợi người tiêu dùng 2010",
        linh_vuc="Thương mại & Tiêu dùng",
        ngay_ban_hanh="2010-11-17",
        ngay_hieu_luc="2011-07-01",
        ngay_het_hieu_luc="2024-07-01",
        bi_thay_the_boi=["19/2023/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="36/2005/QH11",
        ten_van_ban="Luật Thương mại 2005",
        linh_vuc="Thương mại",
        ngay_ban_hanh="2005-06-14",
        ngay_hieu_luc="2006-01-01",
        sua_doi_bo_sung_boi=["05/2017/QH14", "44/2019/QH14"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="04/2007/QH12",
        ten_van_ban="Luật Thuế thu nhập cá nhân 2007",
        linh_vuc="Thuế",
        ngay_ban_hanh="2007-11-21",
        ngay_hieu_luc="2009-01-01",
        sua_doi_bo_sung_boi=["26/2012/QH13", "71/2014/QH13"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="123/2020/NĐ-CP",
        ten_van_ban="Nghị định 123/2020/NĐ-CP về hóa đơn, chứng từ",
        linh_vuc="Thuế & Hóa đơn",
        ngay_ban_hanh="2020-10-19",
        ngay_hieu_luc="2022-07-01",
        sua_doi_bo_sung_boi=["70/2025/NĐ-CP"],
        ghi_chu="Được sửa đổi, bổ sung một số điều bởi Nghị định 70/2025/NĐ-CP từ 01/06/2025.",
    ),
    HoSoHieuLucVanBan(
        so_hieu="70/2025/NĐ-CP",
        ten_van_ban="Nghị định 70/2025/NĐ-CP sửa đổi, bổ sung Nghị định 123/2020/NĐ-CP",
        linh_vuc="Thuế & Hóa đơn",
        ngay_ban_hanh="2025-03-20",
        ngay_hieu_luc="2025-06-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="41/2024/QH15",
        ten_van_ban="Luật Bảo hiểm xã hội 2024",
        linh_vuc="Bảo hiểm xã hội",
        ngay_ban_hanh="2024-06-29",
        ngay_hieu_luc="2025-07-01",
        thay_the_cho=["58/2014/QH13"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="58/2014/QH13",
        ten_van_ban="Luật Bảo hiểm xã hội 2014",
        linh_vuc="Bảo hiểm xã hội",
        ngay_ban_hanh="2014-11-20",
        ngay_hieu_luc="2016-01-01",
        ngay_het_hieu_luc="2025-07-01",
        bi_thay_the_boi=["41/2024/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="48/2024/QH15",
        ten_van_ban="Luật Thuế giá trị gia tăng 2024",
        linh_vuc="Thuế",
        ngay_ban_hanh="2024-11-26",
        ngay_hieu_luc="2025-07-01",
        thay_the_cho=["13/2008/QH12"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="13/2008/QH12",
        ten_van_ban="Luật Thuế giá trị gia tăng 2008",
        linh_vuc="Thuế",
        ngay_ban_hanh="2008-06-03",
        ngay_hieu_luc="2009-01-01",
        ngay_het_hieu_luc="2025-07-01",
        bi_thay_the_boi=["48/2024/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="38/2019/QH14",
        ten_van_ban="Luật Quản lý thuế 2019",
        linh_vuc="Thuế",
        ngay_ban_hanh="2019-06-13",
        ngay_hieu_luc="2020-07-01",
        sua_doi_bo_sung_boi=["56/2024/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="50/2005/QH11",
        ten_van_ban="Luật Sở hữu trí tuệ (đã hợp nhất sửa đổi 2022)",
        linh_vuc="Sở hữu trí tuệ",
        ngay_ban_hanh="2005-11-29",
        ngay_hieu_luc="2006-07-01",
        sua_doi_bo_sung_boi=["36/2009/QH12", "42/2019/QH14", "07/2022/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="22/2023/QH15",
        ten_van_ban="Luật Đấu thầu 2023",
        linh_vuc="Đấu thầu",
        ngay_ban_hanh="2023-06-23",
        ngay_hieu_luc="2024-01-01",
        thay_the_cho=["43/2013/QH13"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="43/2013/QH13",
        ten_van_ban="Luật Đấu thầu 2013",
        linh_vuc="Đấu thầu",
        ngay_ban_hanh="2013-11-26",
        ngay_hieu_luc="2014-07-01",
        ngay_het_hieu_luc="2024-01-01",
        bi_thay_the_boi=["22/2023/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="27/2023/QH15",
        ten_van_ban="Luật Nhà ở 2023",
        linh_vuc="Nhà ở & Bất động sản",
        ngay_ban_hanh="2023-11-27",
        ngay_hieu_luc="2024-08-01",
        thay_the_cho=["65/2014/QH13"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="65/2014/QH13",
        ten_van_ban="Luật Nhà ở 2014",
        linh_vuc="Nhà ở & Bất động sản",
        ngay_ban_hanh="2014-11-25",
        ngay_hieu_luc="2015-07-01",
        ngay_het_hieu_luc="2024-08-01",
        bi_thay_the_boi=["27/2023/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="29/2023/QH15",
        ten_van_ban="Luật Kinh doanh bất động sản 2023",
        linh_vuc="Nhà ở & Bất động sản",
        ngay_ban_hanh="2023-11-28",
        ngay_hieu_luc="2024-08-01",
        thay_the_cho=["66/2014/QH13"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="66/2014/QH13",
        ten_van_ban="Luật Kinh doanh bất động sản 2014",
        linh_vuc="Nhà ở & Bất động sản",
        ngay_ban_hanh="2014-11-25",
        ngay_hieu_luc="2015-07-01",
        ngay_het_hieu_luc="2024-08-01",
        bi_thay_the_boi=["29/2023/QH15"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="15/2012/QH13",
        ten_van_ban="Luật Xử lý vi phạm hành chính 2012",
        linh_vuc="Hành chính",
        ngay_ban_hanh="2012-06-20",
        ngay_hieu_luc="2013-07-01",
        sua_doi_bo_sung_boi=["67/2020/QH14"],
    ),
]


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ theo dõi hiệu lực, lịch sử sửa đổi/bổ sung và thay thế văn bản quy phạm pháp luật Việt Nam "
        "theo mốc thời gian thực, tích hợp trực tiếp vào Legal RAG và kiểm toán trích dẫn hợp đồng."
    ),
    nhom_extra="core",
)
class BoTheoDoiHieuLucPhapLy:
    """Bộ theo dõi vòng đời hiệu lực (còn hiệu lực / sửa đổi / chưa hiệu lực / hết hiệu lực) cho văn bản pháp luật Việt Nam."""

    RE_SO_HIEU = re.compile(
        r"\b(\d{1,4}/\d{4}/[A-ZĐa-zđ0-9\-]+)\b",
        re.UNICODE,
    )

    def __init__(
        self,
        du_lieu_bo_sung: Optional[List[HoSoHieuLucVanBan]] = None,
        phien_ban_dieu_khoan_bo_sung: Optional[List[PhienBanDieuKhoan]] = None,
    ) -> None:
        self._kho: Dict[str, HoSoHieuLucVanBan] = {}
        self._kho_dieu_khoan: Dict[Tuple[str, str], List[PhienBanDieuKhoan]] = {}
        for item in CO_SO_HIEU_LUC_PHAP_LUAT_VN:
            self.dang_ky_van_ban(item)
        if du_lieu_bo_sung:
            for item in du_lieu_bo_sung:
                self.dang_ky_van_ban(item)
        for pb in CO_SO_PHIEN_BAN_DIEU_KHOAN_VN:
            self.dang_ky_phien_ban_dieu_khoan(pb)
        if phien_ban_dieu_khoan_bo_sung:
            for pb in phien_ban_dieu_khoan_bo_sung:
                self.dang_ky_phien_ban_dieu_khoan(pb)

    def dang_ky_phien_ban_dieu_khoan(self, phien_ban: PhienBanDieuKhoan) -> None:
        """Đăng ký một phiên bản hiệu lực theo mốc thời gian cho Điều/Khoản cụ thể."""
        sh = _chuan_hoa_so_hieu(phien_ban.so_hieu)
        dieu_norm = _chuan_hoa_dieu(phien_ban.dieu)
        ban_sao = PhienBanDieuKhoan(
            so_hieu=sh,
            dieu=dieu_norm,
            noi_dung=phien_ban.noi_dung,
            hieu_luc_tu=phien_ban.hieu_luc_tu,
            het_hieu_luc_tu=phien_ban.het_hieu_luc_tu,
            khoan=phien_ban.khoan,
            diem=phien_ban.diem,
            sua_doi_boi=_chuan_hoa_so_hieu(phien_ban.sua_doi_boi) if phien_ban.sua_doi_boi else None,
            trang_thai=phien_ban.trang_thai,
            ghi_chu=phien_ban.ghi_chu,
        )
        ds = self._kho_dieu_khoan.setdefault((sh, dieu_norm), [])
        khoa_moi = (ban_sao.hieu_luc_tu, ban_sao.khoan, ban_sao.diem)
        da_cap_nhat = False
        for i, item_cu in enumerate(ds):
            if (item_cu.hieu_luc_tu, item_cu.khoan, item_cu.diem) == khoa_moi:
                ds[i] = ban_sao
                da_cap_nhat = True
                break
        if not da_cap_nhat:
            ds.append(ban_sao)
        ds.sort(key=lambda item: _phan_tich_ngay(item.hieu_luc_tu))

    def tra_cuu_dieu_khoan_theo_ngay(
        self,
        so_hieu: str,
        dieu: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        khoan: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Truy vấn chính xác phiên bản hiệu lực của một Điều/Khoản tại mốc thời gian `ngay_tra_cuu` (Point-in-Time Legal Lookup)."""
        sh = _chuan_hoa_so_hieu(so_hieu)
        dieu_norm = _chuan_hoa_dieu(dieu)
        moc = _phan_tich_ngay(ngay_tra_cuu)
        ds = self._kho_dieu_khoan.get((sh, dieu_norm), [])
        if khoan is not None:
            k_low = khoan.strip().lower()
            ds_loc = [item for item in ds if item.khoan and k_low in item.khoan.lower()]
            if ds_loc:
                ds = ds_loc

        if not ds:
            hl_vb = self.kiem_tra_hieu_luc(sh, ngay_tra_cuu=moc)
            return {
                "so_hieu": sh,
                "dieu": dieu_norm,
                "khoan": khoan,
                "ngay_tra_cuu": moc.isoformat(),
                "tim_thay": False,
                "trang_thai": hl_vb.get("trang_thai", "chua_co_du_lieu"),
                "phien_ban_hieu_luc": None,
                "lich_su_sua_doi": [],
            }

        hieu_luc_hien_hanh: Optional[PhienBanDieuKhoan] = None
        for item in ds:
            tu = _phan_tich_ngay(item.hieu_luc_tu)
            den = _phan_tich_ngay(item.het_hieu_luc_tu) if item.het_hieu_luc_tu else None
            if moc >= tu and (den is None or moc < den):
                hieu_luc_hien_hanh = item

        if hieu_luc_hien_hanh is not None:
            trang_thai = hieu_luc_hien_hanh.trang_thai
            if trang_thai == "bi_sua_doi":
                trang_thai = "con_hieu_luc_ban_goc"
            elif trang_thai == "bi_thay_the":
                trang_thai = "con_hieu_luc_truoc_thay_the"
            return {
                "so_hieu": sh,
                "dieu": dieu_norm,
                "khoan": hieu_luc_hien_hanh.khoan,
                "ngay_tra_cuu": moc.isoformat(),
                "tim_thay": True,
                "con_hieu_luc": True,
                "trang_thai": trang_thai,
                "phien_ban_hieu_luc": hieu_luc_hien_hanh.to_dict(),
                "lich_su_sua_doi": [x.to_dict() for x in ds],
            }

        dau_tien = ds[0]
        if moc < _phan_tich_ngay(dau_tien.hieu_luc_tu):
            trang_thai = "chua_co_hieu_luc"
        else:
            trang_thai = "het_hieu_luc"
        return {
            "so_hieu": sh,
            "dieu": dieu_norm,
            "khoan": khoan,
            "ngay_tra_cuu": moc.isoformat(),
            "tim_thay": True,
            "con_hieu_luc": False,
            "trang_thai": trang_thai,
            "phien_ban_hieu_luc": None,
            "lich_su_sua_doi": [x.to_dict() for x in ds],
        }

    def dang_ky_van_ban(self, ho_so: HoSoHieuLucVanBan) -> None:
        """Đăng ký hoặc cập nhật hồ sơ hiệu lực văn bản và tự liên kết hai chiều các quan hệ thay thế."""
        key = _chuan_hoa_so_hieu(ho_so.so_hieu)
        ban_sao = HoSoHieuLucVanBan(
            so_hieu=key,
            ten_van_ban=ho_so.ten_van_ban,
            linh_vuc=ho_so.linh_vuc,
            ngay_ban_hanh=ho_so.ngay_ban_hanh,
            ngay_hieu_luc=ho_so.ngay_hieu_luc,
            ngay_het_hieu_luc=ho_so.ngay_het_hieu_luc,
            thay_the_cho=[_chuan_hoa_so_hieu(x) for x in ho_so.thay_the_cho],
            bi_thay_the_boi=[_chuan_hoa_so_hieu(x) for x in ho_so.bi_thay_the_boi],
            sua_doi_bo_sung_boi=[_chuan_hoa_so_hieu(x) for x in ho_so.sua_doi_bo_sung_boi],
            ghi_chu=ho_so.ghi_chu,
        )
        self._kho[key] = ban_sao
        for ma_cu in ban_sao.thay_the_cho:
            cu = self._kho.get(ma_cu)
            if cu is not None:
                if key not in cu.bi_thay_the_boi:
                    cu.bi_thay_the_boi.append(key)
                if cu.ngay_het_hieu_luc is None:
                    cu.ngay_het_hieu_luc = ban_sao.ngay_hieu_luc
        for ma_moi, hs_moi in self._kho.items():
            if ma_moi != key and key in hs_moi.thay_the_cho:
                if ma_moi not in ban_sao.bi_thay_the_boi:
                    ban_sao.bi_thay_the_boi.append(ma_moi)
                if ban_sao.ngay_het_hieu_luc is None:
                    ban_sao.ngay_het_hieu_luc = hs_moi.ngay_hieu_luc

    def danh_sach_van_ban(self, linh_vuc: Optional[str] = None) -> List[Dict[str, Any]]:
        """Liệt kê tất cả văn bản pháp luật đang theo dõi trong cơ sở dữ liệu hiệu lực."""
        kq: List[Dict[str, Any]] = []
        for item in self._kho.values():
            if linh_vuc and linh_vuc.lower() not in item.linh_vuc.lower():
                continue
            kq.append(item.to_dict())
        return sorted(kq, key=lambda x: x["ngay_hieu_luc"], reverse=True)

    def kiem_tra_hieu_luc(
        self,
        so_hieu: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]:
        """Kiểm tra trạng thái hiệu lực chính xác của một số hiệu văn bản tại ngày `ngay_tra_cuu`."""
        key = _chuan_hoa_so_hieu(so_hieu)
        moc = _phan_tich_ngay(ngay_tra_cuu)
        ho_so = self._kho.get(key)
        if ho_so is None:
            return {
                "so_hieu": key,
                "tim_thay": False,
                "trang_thai": "chua_co_du_lieu",
                "con_hieu_luc": None,
                "ngay_tra_cuu": moc.isoformat(),
                "canh_bao": None,
            }

        ngay_hl = _phan_tich_ngay(ho_so.ngay_hieu_luc)
        ngay_het = _phan_tich_ngay(ho_so.ngay_het_hieu_luc) if ho_so.ngay_het_hieu_luc else None

        cac_sua_doi_hieu_luc: List[str] = []
        for ma_sd in ho_so.sua_doi_bo_sung_boi:
            hs_sd = self._kho.get(ma_sd)
            if hs_sd is None or moc >= _phan_tich_ngay(hs_sd.ngay_hieu_luc):
                cac_sua_doi_hieu_luc.append(ma_sd)

        canh_bao: Optional[str] = None
        if moc < ngay_hl:
            trang_thai = "chua_co_hieu_luc"
            con_hieu_luc = False
            canh_bao = f"Văn bản {key} ({ho_so.ten_van_ban}) chưa có hiệu lực tại ngày {moc.isoformat()} (bắt đầu từ {ho_so.ngay_hieu_luc})."
        elif ngay_het is not None and moc >= ngay_het:
            trang_thai = "het_hieu_luc"
            con_hieu_luc = False
            thay_the_txt = ", ".join(ho_so.bi_thay_the_boi) if ho_so.bi_thay_the_boi else "văn bản mới hơn"
            canh_bao = (
                f"CẢNH BÁO HẾT HIỆU LỰC: {key} ({ho_so.ten_van_ban}) đã hết hiệu lực từ {ho_so.ngay_het_hieu_luc}; "
                f"được thay thế bởi: {thay_the_txt}."
            )
        elif cac_sua_doi_hieu_luc:
            trang_thai = "con_hieu_luc_da_sua_doi"
            con_hieu_luc = True
            canh_bao = (
                f"LƯU Ý SỬA ĐỔI: {key} ({ho_so.ten_van_ban}) còn hiệu lực nhưng đã được sửa đổi/bổ sung bởi: "
                f"{', '.join(cac_sua_doi_hieu_luc)}."
            )
        else:
            trang_thai = "con_hieu_luc"
            con_hieu_luc = True

        return {
            "so_hieu": key,
            "tim_thay": True,
            "ten_van_ban": ho_so.ten_van_ban,
            "linh_vuc": ho_so.linh_vuc,
            "ngay_ban_hanh": ho_so.ngay_ban_hanh,
            "ngay_hieu_luc": ho_so.ngay_hieu_luc,
            "ngay_het_hieu_luc": ho_so.ngay_het_hieu_luc,
            "ngay_tra_cuu": moc.isoformat(),
            "trang_thai": trang_thai,
            "con_hieu_luc": con_hieu_luc,
            "thay_the_cho": list(ho_so.thay_the_cho),
            "bi_thay_the_boi": list(ho_so.bi_thay_the_boi),
            "sua_doi_bo_sung_boi": list(ho_so.sua_doi_bo_sung_boi),
            "ghi_chu": ho_so.ghi_chu,
            "canh_bao": canh_bao,
        }

    def trich_xuat_so_hieu(self, van_ban: str) -> List[str]:
        """Trích xuất danh sách duy nhất các số hiệu văn bản quy phạm pháp luật trong chuỗi văn bản."""
        if not van_ban:
            return []
        da_thay: List[str] = []
        for match in self.RE_SO_HIEU.findall(van_ban):
            norm = _chuan_hoa_so_hieu(match)
            if norm not in da_thay:
                da_thay.append(norm)
        return da_thay

    def kiem_tra_hieu_luc_trong_van_ban(
        self,
        van_ban: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]:
        """Quét toàn bộ trích dẫn pháp lý trong một văn bản/hợp đồng và đánh dấu mọi căn cứ đã hết hiệu lực hoặc sửa đổi."""
        moc = _phan_tich_ngay(ngay_tra_cuu)
        ds_so_hieu = self.trich_xuat_so_hieu(van_ban)
        chi_tiet: List[Dict[str, Any]] = []
        het_hieu_luc: List[Dict[str, Any]] = []
        da_sua_doi: List[Dict[str, Any]] = []
        canh_bao_list: List[str] = []

        for sh in ds_so_hieu:
            kq = self.kiem_tra_hieu_luc(sh, ngay_tra_cuu=moc)
            chi_tiet.append(kq)
            if kq["trang_thai"] == "het_hieu_luc":
                het_hieu_luc.append(kq)
            elif kq["trang_thai"] == "con_hieu_luc_da_sua_doi":
                da_sua_doi.append(kq)
            if kq.get("canh_bao"):
                canh_bao_list.append(str(kq["canh_bao"]))

        return {
            "ngay_tra_cuu": moc.isoformat(),
            "tong_so_hieu_phat_hien": len(ds_so_hieu),
            "so_van_ban_het_hieu_luc": len(het_hieu_luc),
            "so_van_ban_da_sua_doi": len(da_sua_doi),
            "hop_le_hieu_luc": len(het_hieu_luc) == 0,
            "chi_tiet": chi_tiet,
            "van_ban_het_hieu_luc": het_hieu_luc,
            "van_ban_da_sua_doi": da_sua_doi,
            "canh_bao": canh_bao_list,
        }

    def bo_sung_hieu_luc_cho_rag(
        self,
        ket_qua_rag: List[Dict[str, Any]],
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        loai_bo_het_hieu_luc: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Gắn siêu dữ liệu hiệu lực pháp lý vào từng đoạn trích xuất RAG, giảm điểm văn bản hết hiệu lực
        (hoặc lọc bỏ nếu `loai_bo_het_hieu_luc=True`) và trả kèm danh sách cảnh báo pháp lý.
        """
        moc = _phan_tich_ngay(ngay_tra_cuu)
        ket_qua_moi: List[Dict[str, Any]] = []
        canh_bao_tong: List[str] = []

        for item in ket_qua_rag:
            ban_sao = dict(item)
            meta = dict(ban_sao.get("metadata", {}))
            nd = str(ban_sao.get("noi_dung") or meta.get("noi_dung") or "")
            sh_list: List[str] = []
            sh_meta = meta.get("so_hieu")
            if isinstance(sh_meta, str) and sh_meta:
                sh_list.append(sh_meta)
            elif isinstance(sh_meta, list):
                sh_list.extend(str(x) for x in sh_meta if x)
            for sh_nd in self.trich_xuat_so_hieu(nd):
                if sh_nd not in sh_list:
                    sh_list.append(sh_nd)

            co_het_hl = False
            trang_thai_list: List[Dict[str, Any]] = []
            for sh in sh_list:
                info = self.kiem_tra_hieu_luc(sh, ngay_tra_cuu=moc)
                if info["tim_thay"]:
                    trang_thai_list.append(info)
                    if info["trang_thai"] == "het_hieu_luc":
                        co_het_hl = True
                    if info.get("canh_bao") and info["canh_bao"] not in canh_bao_tong:
                        canh_bao_tong.append(str(info["canh_bao"]))

            if loai_bo_het_hieu_luc and co_het_hl:
                continue

            meta["hieu_luc_phap_ly"] = trang_thai_list
            ban_sao["metadata"] = meta
            if co_het_hl and isinstance(ban_sao.get("diem"), (int, float)):
                ban_sao["diem"] = round(float(ban_sao["diem"]) * 0.65, 6)
            ket_qua_moi.append(ban_sao)

        return ket_qua_moi, canh_bao_tong


LegalValidityRecord = HoSoHieuLucVanBan
LegalClauseVersion = PhienBanDieuKhoan
LegalValidityTracker = BoTheoDoiHieuLucPhapLy

__all__ = [
    "HoSoHieuLucVanBan",
    "LegalValidityRecord",
    "PhienBanDieuKhoan",
    "LegalClauseVersion",
    "CO_SO_HIEU_LUC_PHAP_LUAT_VN",
    "CO_SO_PHIEN_BAN_DIEU_KHOAN_VN",
    "BoTheoDoiHieuLucPhapLy",
    "LegalValidityTracker",
]
