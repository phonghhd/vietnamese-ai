"""DoThiThamChieuPhapLy (LegalCitationGraph) - Đồ Thị Tham Chiếu Luật ↔ Nghị Định ↔ Thông Tư Việt Nam."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from vietnamese_ai.rag.legal.validity_tracker import (
    BoTheoDoiHieuLucPhapLy,
    HoSoHieuLucVanBan,
    _chuan_hoa_dieu,
    _chuan_hoa_so_hieu,
    _phan_tich_ngay,
)
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


@dataclass(frozen=True)
class QuanHePhapLy:
    """Một cạnh có hướng biểu diễn quan hệ pháp lý giữa hai văn bản quy phạm pháp luật."""

    nguon: str
    dich: str
    loai_quan_he: str
    dieu_lien_quan: str = ""
    mo_ta: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển cạnh quan hệ pháp lý sang dict thuần JSON."""
        return asdict(self)


@dataclass
class NutDieuKhoanPhapLy:
    """Một nút Điều/Khoản có phiên bản thời gian trong Đồ thị Pháp lý cấp Điều khoản."""

    ma_nut: str
    so_hieu: str
    dieu: str
    noi_dung: str
    ten_van_ban: str = ""
    khoan: Optional[str] = None
    tieu_de: str = ""
    linh_vuc: str = ""
    hieu_luc_tu: Optional[str] = None
    het_hieu_luc_tu: Optional[str] = None
    sua_doi_boi: Optional[str] = None
    trang_thai: str = "hieu_luc"

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển nút điều khoản pháp lý sang từ điển JSON chuẩn."""
        return asdict(self)


@dataclass(frozen=True)
class QuanHeDieuKhoan:
    """Một cạnh có hướng gắn khoảng thời gian hiệu lực giữa hai nút Điều/Khoản cụ thể."""

    nguon_nut: str
    dich_nut: str
    loai_quan_he: str
    hieu_luc_tu: Optional[str] = None
    het_hieu_luc_tu: Optional[str] = None
    trong_so: float = 1.0
    mo_ta: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển cạnh quan hệ điều khoản sang từ điển JSON chuẩn."""
        return asdict(self)


VAN_BAN_HUONG_DAN_MAC_DINH: List[HoSoHieuLucVanBan] = [
    HoSoHieuLucVanBan(
        so_hieu="145/2020/NĐ-CP",
        ten_van_ban="Nghị định 145/2020/NĐ-CP quy định chi tiết và hướng dẫn thi hành Bộ luật Lao động 2019",
        linh_vuc="Lao động",
        ngay_ban_hanh="2020-12-14",
        ngay_hieu_luc="2021-02-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="102/2024/NĐ-CP",
        ten_van_ban="Nghị định 102/2024/NĐ-CP quy định chi tiết thi hành một số điều của Luật Đất đai 2024",
        linh_vuc="Đất đai",
        ngay_ban_hanh="2024-07-30",
        ngay_hieu_luc="2024-08-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="101/2024/NĐ-CP",
        ten_van_ban="Nghị định 101/2024/NĐ-CP về điều tra cơ bản đất đai và cấp Giấy chứng nhận quyền sử dụng đất",
        linh_vuc="Đất đai",
        ngay_ban_hanh="2024-07-29",
        ngay_hieu_luc="2024-08-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="43/2014/NĐ-CP",
        ten_van_ban="Nghị định 43/2014/NĐ-CP hướng dẫn Luật Đất đai 2013",
        linh_vuc="Đất đai",
        ngay_ban_hanh="2014-05-15",
        ngay_hieu_luc="2014-07-01",
        ngay_het_hieu_luc="2024-08-01",
        bi_thay_the_boi=["102/2024/NĐ-CP"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="01/2021/NĐ-CP",
        ten_van_ban="Nghị định 01/2021/NĐ-CP về đăng ký doanh nghiệp",
        linh_vuc="Doanh nghiệp",
        ngay_ban_hanh="2021-01-04",
        ngay_hieu_luc="2021-01-04",
    ),
    HoSoHieuLucVanBan(
        so_hieu="78/2021/TT-BTC",
        ten_van_ban="Thông tư 78/2021/TT-BTC hướng dẫn Luật Quản lý thuế và Nghị định 123/2020/NĐ-CP về hóa đơn",
        linh_vuc="Thuế & Hóa đơn",
        ngay_ban_hanh="2021-09-17",
        ngay_hieu_luc="2022-07-01",
        ngay_het_hieu_luc="2025-06-01",
        bi_thay_the_boi=["32/2025/TT-BTC"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="32/2025/TT-BTC",
        ten_van_ban="Thông tư 32/2025/TT-BTC hướng dẫn thực hiện Nghị định 123/2020/NĐ-CP và Nghị định 70/2025/NĐ-CP",
        linh_vuc="Thuế & Hóa đơn",
        ngay_ban_hanh="2025-05-31",
        ngay_hieu_luc="2025-06-01",
        thay_the_cho=["78/2021/TT-BTC"],
    ),
    HoSoHieuLucVanBan(
        so_hieu="53/2022/NĐ-CP",
        ten_van_ban="Nghị định 53/2022/NĐ-CP quy định chi tiết một số điều của Luật An ninh mạng",
        linh_vuc="An ninh mạng",
        ngay_ban_hanh="2022-08-15",
        ngay_hieu_luc="2022-10-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="137/2024/NĐ-CP",
        ten_van_ban="Nghị định 137/2024/NĐ-CP quy định về giao dịch điện tử của cơ quan nhà nước và hệ thống thông tin",
        linh_vuc="Giao dịch điện tử",
        ngay_ban_hanh="2024-10-23",
        ngay_hieu_luc="2024-10-23",
    ),
    HoSoHieuLucVanBan(
        so_hieu="55/2024/NĐ-CP",
        ten_van_ban="Nghị định 55/2024/NĐ-CP quy định chi tiết một số điều của Luật Bảo vệ quyền lợi người tiêu dùng 2023",
        linh_vuc="Thương mại & Tiêu dùng",
        ngay_ban_hanh="2024-05-16",
        ngay_hieu_luc="2024-07-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="24/2024/NĐ-CP",
        ten_van_ban="Nghị định 24/2024/NĐ-CP quy định chi tiết thi hành Luật Đấu thầu 2023 về lựa chọn nhà thầu",
        linh_vuc="Đấu thầu",
        ngay_ban_hanh="2024-02-27",
        ngay_hieu_luc="2024-02-27",
    ),
    HoSoHieuLucVanBan(
        so_hieu="95/2024/NĐ-CP",
        ten_van_ban="Nghị định 95/2024/NĐ-CP quy định chi tiết một số điều của Luật Nhà ở 2023",
        linh_vuc="Nhà ở & Bất động sản",
        ngay_ban_hanh="2024-07-24",
        ngay_hieu_luc="2024-08-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="96/2024/NĐ-CP",
        ten_van_ban="Nghị định 96/2024/NĐ-CP quy định chi tiết một số điều của Luật Kinh doanh bất động sản 2023",
        linh_vuc="Nhà ở & Bất động sản",
        ngay_ban_hanh="2024-07-24",
        ngay_hieu_luc="2024-08-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="118/2021/NĐ-CP",
        ten_van_ban="Nghị định 118/2021/NĐ-CP quy định chi tiết thi hành Luật Xử lý vi phạm hành chính",
        linh_vuc="Hành chính",
        ngay_ban_hanh="2021-12-23",
        ngay_hieu_luc="2022-01-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="17/2023/NĐ-CP",
        ten_van_ban="Nghị định 17/2023/NĐ-CP hướng dẫn Luật Sở hữu trí tuệ về quyền tác giả, quyền liên quan",
        linh_vuc="Sở hữu trí tuệ",
        ngay_ban_hanh="2023-04-26",
        ngay_hieu_luc="2023-04-26",
    ),
    HoSoHieuLucVanBan(
        so_hieu="158/2025/NĐ-CP",
        ten_van_ban="Nghị định 158/2025/NĐ-CP quy định chi tiết thi hành một số điều của Luật Bảo hiểm xã hội 2024",
        linh_vuc="Bảo hiểm xã hội",
        ngay_ban_hanh="2025-06-25",
        ngay_hieu_luc="2025-07-01",
    ),
    HoSoHieuLucVanBan(
        so_hieu="181/2025/NĐ-CP",
        ten_van_ban="Nghị định 181/2025/NĐ-CP quy định chi tiết thi hành một số điều của Luật Thuế GTGT 2024",
        linh_vuc="Thuế",
        ngay_ban_hanh="2025-07-01",
        ngay_hieu_luc="2025-07-01",
    ),
]

QUAN_HE_PHAP_LY_MAC_DINH: List[QuanHePhapLy] = [
    QuanHePhapLy("145/2020/NĐ-CP", "45/2019/QH14", "huong_dan_thi_hanh", "Điều 105, Điều 107", "Hướng dẫn điều kiện lao động, thời giờ làm việc và chấm dứt HĐLĐ"),
    QuanHePhapLy("102/2024/NĐ-CP", "31/2024/QH15", "huong_dan_thi_hanh", "", "Quy định chi tiết thi hành Luật Đất đai 2024"),
    QuanHePhapLy("101/2024/NĐ-CP", "31/2024/QH15", "huong_dan_thi_hanh", "", "Hướng dẫn đăng ký đất đai và cấp Giấy chứng nhận QSDĐ"),
    QuanHePhapLy("43/2014/NĐ-CP", "45/2013/QH13", "huong_dan_thi_hanh", "", "Hướng dẫn thi hành Luật Đất đai 2013 (đã hết hiệu lực)"),
    QuanHePhapLy("01/2021/NĐ-CP", "59/2020/QH14", "huong_dan_thi_hanh", "Điều 27", "Hướng dẫn hồ sơ và thủ tục đăng ký doanh nghiệp"),
    QuanHePhapLy("123/2020/NĐ-CP", "38/2019/QH14", "huong_dan_thi_hanh", "", "Hướng dẫn Luật Quản lý thuế về hóa đơn, chứng từ điện tử"),
    QuanHePhapLy("70/2025/NĐ-CP", "123/2020/NĐ-CP", "sua_doi_bo_sung", "", "Sửa đổi, bổ sung nhiều quy định về hóa đơn điện tử từ 01/06/2025"),
    QuanHePhapLy("78/2021/TT-BTC", "123/2020/NĐ-CP", "huong_dan_thi_hanh", "", "Thông tư hướng dẫn hóa đơn điện tử (đã được thay thế)"),
    QuanHePhapLy("32/2025/TT-BTC", "123/2020/NĐ-CP", "huong_dan_thi_hanh", "", "Thông tư hướng dẫn hóa đơn điện tử cập nhật theo NĐ 70/2025"),
    QuanHePhapLy("32/2025/TT-BTC", "70/2025/NĐ-CP", "huong_dan_thi_hanh", "", "Hướng dẫn thực hiện Nghị định 70/2025/NĐ-CP"),
    QuanHePhapLy("53/2022/NĐ-CP", "24/2018/QH14", "huong_dan_thi_hanh", "Điều 26", "Hướng dẫn lưu trữ dữ liệu và bảo vệ an ninh mạng"),
    QuanHePhapLy("137/2024/NĐ-CP", "20/2023/QH15", "huong_dan_thi_hanh", "", "Hướng dẫn thi hành Luật Giao dịch điện tử 2023"),
    QuanHePhapLy("55/2024/NĐ-CP", "19/2023/QH15", "huong_dan_thi_hanh", "", "Hướng dẫn bảo vệ quyền lợi người tiêu dùng và hợp đồng theo mẫu"),
    QuanHePhapLy("24/2024/NĐ-CP", "22/2023/QH15", "huong_dan_thi_hanh", "", "Hướng dẫn chi tiết lựa chọn nhà thầu theo Luật Đấu thầu 2023"),
    QuanHePhapLy("95/2024/NĐ-CP", "27/2023/QH15", "huong_dan_thi_hanh", "", "Quy định chi tiết thi hành Luật Nhà ở 2023"),
    QuanHePhapLy("96/2024/NĐ-CP", "29/2023/QH15", "huong_dan_thi_hanh", "", "Quy định chi tiết thi hành Luật Kinh doanh bất động sản 2023"),
    QuanHePhapLy("118/2021/NĐ-CP", "15/2012/QH13", "huong_dan_thi_hanh", "", "Hướng dẫn thi hành Luật Xử lý vi phạm hành chính"),
    QuanHePhapLy("17/2023/NĐ-CP", "50/2005/QH11", "huong_dan_thi_hanh", "", "Hướng dẫn quyền tác giả và quyền liên quan theo Luật SHTT"),
    QuanHePhapLy("158/2025/NĐ-CP", "41/2024/QH15", "huong_dan_thi_hanh", "", "Hướng dẫn thi hành Luật Bảo hiểm xã hội 2024"),
    QuanHePhapLy("181/2025/NĐ-CP", "48/2024/QH15", "huong_dan_thi_hanh", "", "Hướng dẫn thi hành Luật Thuế giá trị gia tăng 2024"),
    QuanHePhapLy("13/2023/NĐ-CP", "24/2018/QH14", "can_cu_phap_ly", "", "Ban hành trên cơ sở Luật An ninh mạng 2018 và Bộ luật Dân sự 2015"),
    QuanHePhapLy("13/2023/NĐ-CP", "91/2015/QH13", "can_cu_phap_ly", "Điều 38", "Cụ thể hóa quyền bảo vệ bí mật đời tư và dữ liệu cá nhân"),
]


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Đồ thị tham chiếu pháp lý Việt Nam tự động liên kết Luật ↔ Nghị định hướng dẫn ↔ Thông tư "
        "↔ Văn bản sửa đổi/thay thế, kết hợp `BoTheoDoiHieuLucPhapLy` để kéo kèm văn bản thi hành còn hiệu lực."
    ),
    nhom_extra="core",
)
class DoThiThamChieuPhapLy:
    """
    Đồ thị tham chiếu quy phạm pháp luật Việt Nam (`Luật` ↔ `Nghị định` ↔ `Thông tư` ↔ `Sửa đổi/Thay thế`).

    Hỗ trợ:
    - Tự động truy vết văn bản thay thế hiện hành khi người dùng hỏi một đạo luật đã hết hiệu lực.
    - Kéo theo toàn bộ Nghị định hướng dẫn thi hành, Thông tư và văn bản sửa đổi bổ sung đang còn hiệu lực.
    - Tự động bóc tách cạnh tham chiếu mới từ nội dung văn bản hành chính/hợp đồng (`.docx`/`.pdf`).
    - Xuất sơ đồ Mermaid `flowchart LR` trực quan cho Web Studio hoặc báo cáo kiểm toán.
    """

    RE_SO_HIEU = re.compile(r"\b(\d{1,4}/\d{4}/[A-ZĐa-zđ0-9\-]+)\b", re.UNICODE)

    def __init__(
        self,
        bo_theo_doi: Optional[BoTheoDoiHieuLucPhapLy] = None,
    ) -> None:
        self.bo_theo_doi = bo_theo_doi or BoTheoDoiHieuLucPhapLy()
        for hs in VAN_BAN_HUONG_DAN_MAC_DINH:
            self.bo_theo_doi.dang_ky_van_ban(hs)

        self._canh: List[QuanHePhapLy] = []
        self._canh_set: Set[Tuple[str, str, str]] = set()
        self._ra: Dict[str, List[QuanHePhapLy]] = {}
        self._vao: Dict[str, List[QuanHePhapLy]] = {}

        self._nut_dieu_khoan: Dict[str, NutDieuKhoanPhapLy] = {}
        self._canh_dieu_khoan: List[QuanHeDieuKhoan] = []
        self._canh_dieu_khoan_set: Set[Tuple[str, str, str]] = set()
        self._ra_dieu_khoan: Dict[str, List[QuanHeDieuKhoan]] = {}
        self._vao_dieu_khoan: Dict[str, List[QuanHeDieuKhoan]] = {}
        self._chi_muc_so_hieu_nut: Dict[str, List[str]] = {}

        self._nap_mac_dinh()

    def _nap_mac_dinh(self) -> None:
        """Đồng bộ toàn bộ quan hệ thay thế/sửa đổi từ `BoTheoDoiHieuLucPhapLy` và các Nghị định/Thông tư hướng dẫn."""
        for vb in self.bo_theo_doi.danh_sach_van_ban():
            sh = _chuan_hoa_so_hieu(str(vb["so_hieu"]))
            for cu in vb.get("thay_the_cho", []):
                self.them_quan_he(
                    QuanHePhapLy(
                        nguon=sh,
                        dich=_chuan_hoa_so_hieu(cu),
                        loai_quan_he="thay_the",
                        mo_ta=f"{sh} thay thế cho {cu}",
                    )
                )
            for sua in vb.get("sua_doi_bo_sung_boi", []):
                self.them_quan_he(
                    QuanHePhapLy(
                        nguon=_chuan_hoa_so_hieu(sua),
                        dich=sh,
                        loai_quan_he="sua_doi_bo_sung",
                        mo_ta=f"{sua} sửa đổi, bổ sung {sh}",
                    )
                )

        for qh in QUAN_HE_PHAP_LY_MAC_DINH:
            self.them_quan_he(qh)

    def them_van_ban(self, ho_so: HoSoHieuLucVanBan) -> None:
        """Đăng ký thêm một nút văn bản vào cả bộ theo dõi hiệu lực và đồ thị tham chiếu."""
        self.bo_theo_doi.dang_ky_van_ban(ho_so)
        sh = _chuan_hoa_so_hieu(ho_so.so_hieu)
        for cu in ho_so.thay_the_cho:
            self.them_quan_he(QuanHePhapLy(sh, _chuan_hoa_so_hieu(cu), "thay_the"))
        for sua in ho_so.sua_doi_bo_sung_boi:
            self.them_quan_he(QuanHePhapLy(_chuan_hoa_so_hieu(sua), sh, "sua_doi_bo_sung"))

    def them_quan_he(self, quan_he: QuanHePhapLy) -> bool:
        """Thêm một cạnh quan hệ pháp lý có hướng (chống trùng lặp tự động)."""
        nguon = _chuan_hoa_so_hieu(quan_he.nguon)
        dich = _chuan_hoa_so_hieu(quan_he.dich)
        if not nguon or not dich or nguon == dich:
            return False
        khoa = (nguon, dich, quan_he.loai_quan_he)
        if khoa in self._canh_set:
            return False
        canh_chuan = QuanHePhapLy(
            nguon=nguon,
            dich=dich,
            loai_quan_he=quan_he.loai_quan_he,
            dieu_lien_quan=quan_he.dieu_lien_quan,
            mo_ta=quan_he.mo_ta,
        )
        self._canh_set.add(khoa)
        self._canh.append(canh_chuan)
        self._ra.setdefault(nguon, []).append(canh_chuan)
        self._vao.setdefault(dich, []).append(canh_chuan)
        return True

    def trich_xuat_va_them_tu_van_ban(
        self,
        so_hieu_nguon: str,
        noi_dung: str,
    ) -> List[QuanHePhapLy]:
        """Tự động trích xuất quan hệ `hướng dẫn`, `sửa đổi`, `thay thế`, `căn cứ` từ văn bản và thêm vào đồ thị."""
        nguon = _chuan_hoa_so_hieu(so_hieu_nguon)
        if not nguon or not noi_dung:
            return []

        da_them: List[QuanHePhapLy] = []
        for dong in noi_dung.splitlines():
            dong_clean = dong.strip()
            if not dong_clean:
                continue
            ds_sh = [_chuan_hoa_so_hieu(m) for m in self.RE_SO_HIEU.findall(dong_clean)]
            if not ds_sh:
                continue
            low = dong_clean.lower()
            if "thay thế" in low or "bãi bỏ" in low:
                loai = "thay_the"
            elif "sửa đổi" in low or "bổ sung" in low:
                loai = "sua_doi_bo_sung"
            elif "hướng dẫn" in low or "quy định chi tiết" in low:
                loai = "huong_dan_thi_hanh"
            else:
                loai = "can_cu_phap_ly"

            for dich in ds_sh:
                if dich == nguon:
                    continue
                qh = QuanHePhapLy(nguon=nguon, dich=dich, loai_quan_he=loai, mo_ta=dong_clean[:140])
                if self.them_quan_he(qh):
                    da_them.append(qh)
        return da_them

    def _truy_vet_van_ban_thay_the_moi_nhat(
        self,
        so_hieu: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> List[str]:
        """Truy vết chuỗi thay thế để tìm các văn bản kế nhiệm đang còn hiệu lực."""
        hien_tai = _chuan_hoa_so_hieu(so_hieu)
        ket_qua: List[str] = []
        hang_doi = [hien_tai]
        da_tham: Set[str] = {hien_tai}
        while hang_doi:
            nut = hang_doi.pop(0)
            for canh in self._vao.get(nut, []):
                if canh.loai_quan_he == "thay_the" and canh.nguon not in da_tham:
                    da_tham.add(canh.nguon)
                    tt = self.bo_theo_doi.kiem_tra_hieu_luc(canh.nguon, ngay_tra_cuu=ngay_tra_cuu)
                    if tt.get("con_hieu_luc") is True:
                        ket_qua.append(canh.nguon)
                    hang_doi.append(canh.nguon)
        return ket_qua

    def _thong_tin_nut(
        self,
        so_hieu: str,
        loai_quan_he: str,
        mo_ta: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]:
        """Đóng gói thông tin một văn bản liên quan kèm trạng thái hiệu lực theo ngày tra cứu."""
        tt = self.bo_theo_doi.kiem_tra_hieu_luc(so_hieu, ngay_tra_cuu=ngay_tra_cuu)
        return {
            "so_hieu": tt["so_hieu"],
            "ten_van_ban": tt.get("ten_van_ban", so_hieu),
            "loai_quan_he": loai_quan_he,
            "trang_thai": tt["trang_thai"],
            "con_hieu_luc": tt["con_hieu_luc"],
            "ngay_hieu_luc": tt.get("ngay_hieu_luc"),
            "ngay_het_hieu_luc": tt.get("ngay_het_hieu_luc"),
            "mo_ta": mo_ta,
        }

    def truy_van_van_ban_lien_quan(
        self,
        so_hieu: str,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        chi_lay_con_hieu_luc: bool = False,
        do_sau: int = 2,
    ) -> Dict[str, Any]:
        """
        Truy vấn toàn bộ hệ sinh thái văn bản liên quan của `so_hieu`:
        - Nếu `so_hieu` đã hết hiệu lực, tự động truy vết văn bản thay thế hiện hành và nghị định hướng dẫn mới.
        - Trả về đầy đủ `van_ban_huong_dan`, `van_ban_sua_doi_bo_sung`, `van_ban_thay_the_hien_hanh`, `van_ban_goc`.
        """
        goc = _chuan_hoa_so_hieu(so_hieu)
        trang_thai_goc = self.bo_theo_doi.kiem_tra_hieu_luc(goc, ngay_tra_cuu=ngay_tra_cuu)
        ke_nhiem = self._truy_vet_van_ban_thay_the_moi_nhat(goc, ngay_tra_cuu=ngay_tra_cuu)

        cac_nut_muc_tieu: List[str] = [goc]
        if trang_thai_goc["trang_thai"] == "het_hieu_luc":
            for kn in ke_nhiem:
                if kn not in cac_nut_muc_tieu:
                    cac_nut_muc_tieu.append(kn)

        huong_dan: List[Dict[str, Any]] = []
        sua_doi: List[Dict[str, Any]] = []
        van_ban_goc: List[Dict[str, Any]] = []
        can_cu: List[Dict[str, Any]] = []
        da_thay_hd: Set[str] = set()
        da_thay_sd: Set[str] = set()

        hang_doi: List[Tuple[str, int]] = [(n, 0) for n in cac_nut_muc_tieu]
        da_duyet: Set[str] = set()
        while hang_doi:
            nut, depth = hang_doi.pop(0)
            if nut in da_duyet or depth >= max(1, int(do_sau)):
                continue
            da_duyet.add(nut)

            for canh in self._vao.get(nut, []):
                info = self._thong_tin_nut(canh.nguon, canh.loai_quan_he, canh.mo_ta, ngay_tra_cuu)
                if chi_lay_con_hieu_luc and info["con_hieu_luc"] is False:
                    continue
                if canh.loai_quan_he == "huong_dan_thi_hanh" and canh.nguon not in da_thay_hd:
                    da_thay_hd.add(canh.nguon)
                    huong_dan.append(info)
                    hang_doi.append((canh.nguon, depth + 1))
                elif canh.loai_quan_he == "sua_doi_bo_sung" and canh.nguon not in da_thay_sd:
                    da_thay_sd.add(canh.nguon)
                    sua_doi.append(info)
                    hang_doi.append((canh.nguon, depth + 1))

            if nut == goc:
                for canh in self._ra.get(nut, []):
                    info = self._thong_tin_nut(canh.dich, canh.loai_quan_he, canh.mo_ta, ngay_tra_cuu)
                    if chi_lay_con_hieu_luc and info["con_hieu_luc"] is False:
                        continue
                    if canh.loai_quan_he in ("huong_dan_thi_hanh", "sua_doi_bo_sung"):
                        van_ban_goc.append(info)
                    elif canh.loai_quan_he == "can_cu_phap_ly":
                        can_cu.append(info)

        canh_bao: List[str] = []
        if trang_thai_goc.get("canh_bao"):
            canh_bao.append(str(trang_thai_goc["canh_bao"]))
        if ke_nhiem:
            canh_bao.append(
                f"Đã tự động mở rộng đồ thị sang văn bản thay thế hiện hành: {', '.join(ke_nhiem)}."
            )

        return {
            "so_hieu": goc,
            "hieu_luc_goc": trang_thai_goc,
            "van_ban_thay_the_hien_hanh": [
                self._thong_tin_nut(kn, "thay_the", f"Thay thế cho {goc}", ngay_tra_cuu)
                for kn in ke_nhiem
            ],
            "van_ban_huong_dan": huong_dan,
            "van_ban_sua_doi_bo_sung": sua_doi,
            "van_ban_goc": van_ban_goc,
            "van_ban_can_cu": can_cu,
            "canh_bao_tham_chieu": canh_bao,
        }

    def mo_rong_ngu_canh_rag(
        self,
        so_hieu_list: List[str],
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> Dict[str, Any]:
        """Mở rộng danh sách số hiệu trích xuất từ RAG sang các Nghị định/Thông tư hướng dẫn và văn bản thay thế."""
        van_ban_huong_dan_tong: List[Dict[str, Any]] = []
        van_ban_thay_the_tong: List[Dict[str, Any]] = []
        van_ban_sua_doi_tong: List[Dict[str, Any]] = []
        da_thay_hd: Set[str] = set()
        da_thay_tt: Set[str] = set()
        da_thay_sd: Set[str] = set()

        for sh in so_hieu_list:
            kq = self.truy_van_van_ban_lien_quan(sh, ngay_tra_cuu=ngay_tra_cuu, chi_lay_con_hieu_luc=True)
            for item in kq["van_ban_huong_dan"]:
                if item["so_hieu"] not in da_thay_hd:
                    da_thay_hd.add(item["so_hieu"])
                    van_ban_huong_dan_tong.append(item)
            for item in kq["van_ban_thay_the_hien_hanh"]:
                if item["so_hieu"] not in da_thay_tt:
                    da_thay_tt.add(item["so_hieu"])
                    van_ban_thay_the_tong.append(item)
            for item in kq["van_ban_sua_doi_bo_sung"]:
                if item["so_hieu"] not in da_thay_sd:
                    da_thay_sd.add(item["so_hieu"])
                    van_ban_sua_doi_tong.append(item)

        return {
            "so_hieu_dau_vao": [_chuan_hoa_so_hieu(x) for x in so_hieu_list],
            "van_ban_huong_dan_con_hieu_luc": van_ban_huong_dan_tong,
            "van_ban_thay_the_hien_hanh": van_ban_thay_the_tong,
            "van_ban_sua_doi_bo_sung": van_ban_sua_doi_tong,
        }

    def xuat_do_thi_mermaid(
        self,
        so_hieu_trung_tam: Optional[str] = None,
        do_sau: int = 2,
    ) -> str:
        """Xuất sơ đồ đồ thị tham chiếu pháp lý định dạng Mermaid `flowchart LR`."""
        nhan_quan_he = {
            "huong_dan_thi_hanh": "hướng dẫn thi hành",
            "sua_doi_bo_sung": "sửa đổi, bổ sung",
            "thay_the": "thay thế",
            "can_cu_phap_ly": "căn cứ",
        }
        if so_hieu_trung_tam:
            goc = _chuan_hoa_so_hieu(so_hieu_trung_tam)
            tap_nut: Set[str] = {goc}
            hang_doi: List[Tuple[str, int]] = [(goc, 0)]
            while hang_doi:
                nut, depth = hang_doi.pop(0)
                if depth >= max(1, int(do_sau)):
                    continue
                for canh in self._ra.get(nut, []) + self._vao.get(nut, []):
                    for ke in (canh.nguon, canh.dich):
                        if ke not in tap_nut:
                            tap_nut.add(ke)
                            hang_doi.append((ke, depth + 1))
            canh_hien_thi = [c for c in self._canh if c.nguon in tap_nut and c.dich in tap_nut]
        else:
            canh_hien_thi = list(self._canh)

        def _id_mermaid(sh: str) -> str:
            return "VB_" + re.sub(r"[^0-9A-Za-z_]", "_", sh)

        lines = ["flowchart LR"]
        nut_da_khai_bao: Set[str] = set()
        for c in canh_hien_thi:
            for sh in (c.nguon, c.dich):
                if sh not in nut_da_khai_bao:
                    nut_da_khai_bao.add(sh)
                    lines.append(f'    {_id_mermaid(sh)}["{sh}"]')
            nhan = nhan_quan_he.get(c.loai_quan_he, c.loai_quan_he)
            lines.append(f'    {_id_mermaid(c.nguon)} -->|"{nhan}"| {_id_mermaid(c.dich)}')
        return "\n".join(lines)

    _TRONG_SO_CANH_DIEU_KHOAN: Dict[str, float] = {
        "sua_doi_dieu": 1.0,
        "thay_the_dieu": 1.0,
        "phien_ban_ke_nhiem": 1.0,
        "huong_dan_dieu": 0.94,
        "vien_dan_cheo": 0.85,
        "bai_bo_dieu": 0.50,
    }

    def them_nut_dieu_khoan(self, nut: NutDieuKhoanPhapLy) -> None:
        """Đăng ký một nút Điều/Khoản có khoảng hiệu lực vào đồ thị cấp điều khoản."""
        sh = _chuan_hoa_so_hieu(nut.so_hieu)
        dieu_norm = _chuan_hoa_dieu(nut.dieu) if nut.dieu else ""
        ban_sao = NutDieuKhoanPhapLy(
            ma_nut=str(nut.ma_nut).strip(),
            so_hieu=sh,
            dieu=dieu_norm,
            noi_dung=nut.noi_dung,
            ten_van_ban=nut.ten_van_ban,
            khoan=nut.khoan,
            tieu_de=nut.tieu_de,
            linh_vuc=nut.linh_vuc,
            hieu_luc_tu=nut.hieu_luc_tu,
            het_hieu_luc_tu=nut.het_hieu_luc_tu,
            sua_doi_boi=_chuan_hoa_so_hieu(nut.sua_doi_boi) if nut.sua_doi_boi else None,
            trang_thai=nut.trang_thai,
        )
        self._nut_dieu_khoan[ban_sao.ma_nut] = ban_sao
        ds_sh = self._chi_muc_so_hieu_nut.setdefault(sh, [])
        if ban_sao.ma_nut not in ds_sh:
            ds_sh.append(ban_sao.ma_nut)

    def lay_nut_dieu_khoan(self, ma_nut: str) -> Optional[NutDieuKhoanPhapLy]:
        """Lấy thông tin một nút điều khoản theo mã nút."""
        return self._nut_dieu_khoan.get(str(ma_nut).strip())

    def them_quan_he_dieu_khoan(self, canh: QuanHeDieuKhoan) -> bool:
        """Thêm một cạnh có hướng gắn thời gian giữa hai nút Điều/Khoản (chống trùng lặp)."""
        nguon = str(canh.nguon_nut).strip()
        dich = str(canh.dich_nut).strip()
        if not nguon or not dich or nguon == dich:
            return False
        khoa = (nguon, dich, canh.loai_quan_he)
        if khoa in self._canh_dieu_khoan_set:
            return False
        canh_chuan = QuanHeDieuKhoan(
            nguon_nut=nguon,
            dich_nut=dich,
            loai_quan_he=canh.loai_quan_he,
            hieu_luc_tu=canh.hieu_luc_tu,
            het_hieu_luc_tu=canh.het_hieu_luc_tu,
            trong_so=float(canh.trong_so),
            mo_ta=canh.mo_ta,
        )
        self._canh_dieu_khoan_set.add(khoa)
        self._canh_dieu_khoan.append(canh_chuan)
        self._ra_dieu_khoan.setdefault(nguon, []).append(canh_chuan)
        self._vao_dieu_khoan.setdefault(dich, []).append(canh_chuan)
        return True

    def dieu_khoan_con_hieu_luc(
        self,
        nut: NutDieuKhoanPhapLy,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> bool:
        """Kiểm tra một nút Điều/Khoản có đang còn hiệu lực tại mốc `ngay_tra_cuu` hay không."""
        moc = _phan_tich_ngay(ngay_tra_cuu)
        if nut.hieu_luc_tu or nut.het_hieu_luc_tu:
            tu = _phan_tich_ngay(nut.hieu_luc_tu) if nut.hieu_luc_tu else date(1900, 1, 1)
            den = _phan_tich_ngay(nut.het_hieu_luc_tu) if nut.het_hieu_luc_tu else None
            return moc >= tu and (den is None or moc < den)
        hl_vb = self.bo_theo_doi.kiem_tra_hieu_luc(nut.so_hieu, ngay_tra_cuu=moc)
        if hl_vb.get("tim_thay"):
            return bool(hl_vb.get("con_hieu_luc"))
        return True

    def canh_dieu_khoan_con_hieu_luc(
        self,
        canh: QuanHeDieuKhoan,
        ngay_tra_cuu: Optional[Union[str, date]] = None,
    ) -> bool:
        """Kiểm tra một cạnh quan hệ điều khoản có hiệu lực tại mốc `ngay_tra_cuu` hay không."""
        if not canh.hieu_luc_tu and not canh.het_hieu_luc_tu:
            return True
        moc = _phan_tich_ngay(ngay_tra_cuu)
        tu = _phan_tich_ngay(canh.hieu_luc_tu) if canh.hieu_luc_tu else date(1900, 1, 1)
        den = _phan_tich_ngay(canh.het_hieu_luc_tu) if canh.het_hieu_luc_tu else None
        return moc >= tu and (den is None or moc < den)

    def mo_rong_da_buoc_theo_thoi_gian(
        self,
        ma_nut_khoi_dau: List[str],
        ngay_tra_cuu: Optional[Union[str, date]] = None,
        so_buoc: int = 2,
        he_so_suy_giam: float = 0.75,
        chi_lay_con_hieu_luc: bool = True,
        diem_khoi_dau: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Lan truyền đồ thị điều khoản đa bước có điều kiện thời gian `t` (Time-Conditioned Multi-Hop Traversal)."""
        moc = _phan_tich_ngay(ngay_tra_cuu)
        max_hop = max(1, int(so_buoc))
        diem_tich_luy: Dict[str, float] = {}
        duong_dan_tot_nhat: Dict[str, List[str]] = {}
        buoc_ngan_nhat: Dict[str, int] = {}
        quan_he_den: Dict[str, str] = {}

        tap_hat_giong: Set[str] = set()
        hang_doi: List[Tuple[str, float, int, List[str]]] = []
        for idx, ma in enumerate(ma_nut_khoi_dau):
            key = str(ma).strip()
            if not key or key not in self._nut_dieu_khoan:
                continue
            tap_hat_giong.add(key)
            diem_ban_dau = (
                float(diem_khoi_dau[key])
                if diem_khoi_dau is not None and key in diem_khoi_dau
                else 1.0 / (1.0 + 0.15 * idx)
            )
            hang_doi.append((key, diem_ban_dau, 0, [key]))

        da_lan_truyen: Set[Tuple[str, int]] = set()
        while hang_doi:
            nut_hien_tai, diem_ht, hop, path = hang_doi.pop(0)
            if hop >= max_hop or (nut_hien_tai, hop) in da_lan_truyen:
                continue
            da_lan_truyen.add((nut_hien_tai, hop))

            ung_vien_canh: List[Tuple[str, QuanHeDieuKhoan, float]] = []
            for canh in self._vao_dieu_khoan.get(nut_hien_tai, []):
                if not self.canh_dieu_khoan_con_hieu_luc(canh, ngay_tra_cuu=moc):
                    continue
                if canh.loai_quan_he in (
                    "huong_dan_dieu",
                    "sua_doi_dieu",
                    "thay_the_dieu",
                    "phien_ban_ke_nhiem",
                ):
                    ung_vien_canh.append((canh.nguon_nut, canh, 1.0))
                elif canh.loai_quan_he == "vien_dan_cheo":
                    ung_vien_canh.append((canh.nguon_nut, canh, 0.85))

            for canh in self._ra_dieu_khoan.get(nut_hien_tai, []):
                if not self.canh_dieu_khoan_con_hieu_luc(canh, ngay_tra_cuu=moc):
                    continue
                if canh.loai_quan_he in ("vien_dan_cheo", "huong_dan_dieu"):
                    ung_vien_canh.append((canh.dich_nut, canh, 0.92))
                elif canh.loai_quan_he in ("sua_doi_dieu", "thay_the_dieu", "phien_ban_ke_nhiem"):
                    ung_vien_canh.append((canh.dich_nut, canh, 0.70))

            for ke, canh, he_so_huong in ung_vien_canh:
                if ke in path or ke not in self._nut_dieu_khoan:
                    continue
                nut_ke = self._nut_dieu_khoan[ke]
                con_hl = self.dieu_khoan_con_hieu_luc(nut_ke, ngay_tra_cuu=moc)
                w_loai = self._TRONG_SO_CANH_DIEU_KHOAN.get(canh.loai_quan_he, 0.75)
                diem_moi = diem_ht * he_so_suy_giam * w_loai * he_so_huong * max(0.1, float(canh.trong_so))
                if not con_hl and chi_lay_con_hieu_luc:
                    if hop + 1 < max_hop:
                        hang_doi.append((ke, diem_moi * 0.9, hop + 1, path + [ke]))
                    continue
                if ke not in tap_hat_giong:
                    diem_tich_luy[ke] = diem_tich_luy.get(ke, 0.0) + diem_moi
                    if ke not in buoc_ngan_nhat or (hop + 1) < buoc_ngan_nhat[ke]:
                        buoc_ngan_nhat[ke] = hop + 1
                        duong_dan_tot_nhat[ke] = path + [ke]
                        quan_he_den[ke] = canh.loai_quan_he
                if hop + 1 < max_hop:
                    hang_doi.append((ke, diem_moi, hop + 1, path + [ke]))

        ket_qua: List[Dict[str, Any]] = []
        for ma_nut, diem in diem_tich_luy.items():
            nut_obj = self._nut_dieu_khoan[ma_nut]
            con_hl = self.dieu_khoan_con_hieu_luc(nut_obj, ngay_tra_cuu=moc)
            if chi_lay_con_hieu_luc and not con_hl:
                continue
            ket_qua.append(
                {
                    "ma_nut": ma_nut,
                    "diem_do_thi": round(float(diem), 6),
                    "so_buoc": buoc_ngan_nhat.get(ma_nut, 1),
                    "duong_dan": duong_dan_tot_nhat.get(ma_nut, [ma_nut]),
                    "loai_quan_he": quan_he_den.get(ma_nut, "lien_quan"),
                    "con_hieu_luc": con_hl,
                    "nut": nut_obj.to_dict(),
                }
            )
        ket_qua.sort(key=lambda x: (-float(x["diem_do_thi"]), int(x["so_buoc"]), str(x["ma_nut"])))
        return ket_qua

    def thong_ke_dieu_khoan(self) -> Dict[str, Any]:
        """Trả về thống kê tổng số nút và cạnh trong đồ thị cấp Điều/Khoản."""
        dem_loai: Dict[str, int] = {}
        for c in self._canh_dieu_khoan:
            dem_loai[c.loai_quan_he] = dem_loai.get(c.loai_quan_he, 0) + 1
        return {
            "so_nut_dieu_khoan": len(self._nut_dieu_khoan),
            "tong_canh_dieu_khoan": len(self._canh_dieu_khoan),
            "phan_loai_canh_dieu_khoan": dem_loai,
        }

    def thong_ke(self) -> Dict[str, Any]:
        """Trả về thống kê tổng số văn bản và quan hệ trong đồ thị tham chiếu pháp lý."""
        tap_nut: Set[str] = set(self._ra.keys()) | set(self._vao.keys())
        dem_loai: Dict[str, int] = {}
        for c in self._canh:
            dem_loai[c.loai_quan_he] = dem_loai.get(c.loai_quan_he, 0) + 1
        return {
            "so_van_ban_trong_do_thi": len(tap_nut),
            "tong_quan_he": len(self._canh),
            "phan_loai_quan_he": dem_loai,
            "so_nut_dieu_khoan": len(self._nut_dieu_khoan),
            "tong_canh_dieu_khoan": len(self._canh_dieu_khoan),
        }


LegalRelationEdge = QuanHePhapLy
LegalClauseNode = NutDieuKhoanPhapLy
LegalClauseEdge = QuanHeDieuKhoan
LegalCitationGraph = DoThiThamChieuPhapLy

__all__ = [
    "QuanHePhapLy",
    "LegalRelationEdge",
    "NutDieuKhoanPhapLy",
    "LegalClauseNode",
    "QuanHeDieuKhoan",
    "LegalClauseEdge",
    "VAN_BAN_HUONG_DAN_MAC_DINH",
    "QUAN_HE_PHAP_LY_MAC_DINH",
    "DoThiThamChieuPhapLy",
    "LegalCitationGraph",
]
