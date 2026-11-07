"""Trích xuất Thông minh Hóa đơn & CCCD - Vietnamese AI Framework."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from vietnamese_ai.security.data_defense import KhienBaoVeHinhAnh
from vietnamese_ai.vision.ocr import TrichXuatVanBanAnh


class TrichXuatThongMinh:
    """
    Hệ thống trích xuất thông tin có cấu trúc thông minh cho các loại tài liệu chuẩn Việt Nam:
    - Căn cước công dân (CCCD 12 số & CMND 9 số).
    - Hóa đơn Giá trị gia tăng (VAT Invoice điện tử).
    - Hợp đồng kinh tế / Hợp đồng lao động.
    Tự động kết hợp Khử độc ảnh CDR, OCR thích nghi Otsu và bộ phân tích cú pháp thực thể.
    """

    def __init__(self, khien_anh: Optional[KhienBaoVeHinhAnh] = None, ocr: Optional[TrichXuatVanBanAnh] = None):
        self.khien_anh = khien_anh or KhienBaoVeHinhAnh()
        self.ocr = ocr or TrichXuatVanBanAnh()

    def _chuan_bi_van_ban(self, dau_vao: Union[str, Path, np.ndarray, Any]) -> str:
        """Chuẩn bị chuỗi văn bản từ chuỗi thuần túy hoặc trích xuất từ tệp ảnh."""
        if isinstance(dau_vao, str):
            p = Path(dau_vao)
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                dau_vao = p
            else:
                return dau_vao.strip()

        if isinstance(dau_vao, (Path, np.ndarray)) or hasattr(dau_vao, "convert"):
            # Quét và khử độc CDR trước khi đọc OCR
            arr_sach, bc = self.khien_anh.khu_doc_anh_cdr(dau_vao)
            res_ocr = self.ocr.doc_van_ban(arr_sach)
            return res_ocr.get("van_ban", "")

        return str(dau_vao or "").strip()

    def trich_xuat_cccd(self, dau_vao: Union[str, Path, np.ndarray, Any]) -> Dict[str, Any]:
        """
        Trích xuất thông tin Căn cước công dân (CCCD / CMND).
        Trả về dict các trường thông tin chuẩn kèm điểm độ tin cậy.
        """
        text = self._chuan_bi_van_ban(dau_vao)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        so_cccd = ""
        ho_ten = ""
        ngay_sinh = ""
        gioi_tinh = ""
        que_quan = ""
        thuong_tru = ""
        ngay_het_han = ""

        # 1. Tìm số CCCD (12 chữ số) hoặc CMND (9 chữ số)
        match_id12 = re.search(r"\b(\d{12})\b", text)
        if match_id12:
            so_cccd = match_id12.group(1)
        else:
            match_id9 = re.search(r"\b(\d{9})\b", text)
            if match_id9:
                so_cccd = match_id9.group(1)

        # 2. Tìm ngày sinh (dd/mm/yyyy)
        match_ns = re.search(
            r"(?:ngày sinh|sinh ngày|date of birth)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
            text,
            flags=re.IGNORECASE,
        )
        if match_ns:
            ngay_sinh = match_ns.group(1).replace("-", "/")
        else:
            all_dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
            if all_dates:
                ngay_sinh = all_dates[0]

        # 3. Tìm giới tính
        match_gt = re.search(r"(?:giới tính|sex)[\s:]*(Nam|Nữ|Male|Female)", text, flags=re.IGNORECASE)
        if match_gt:
            gt_raw = match_gt.group(1).lower()
            gioi_tinh = "Nam" if gt_raw in ("nam", "male") else "Nữ"

        # 4. Tìm họ và tên (thường viết HOA tiếng Việt có dấu hoặc không dấu)
        match_ten = re.search(
            r"(?:họ và tên|họ tên|full name)[\s:]*([A-ZÀ-Ỹ\s]{3,40})",
            text,
            flags=re.IGNORECASE,
        )
        if match_ten:
            ho_ten = match_ten.group(1).strip()
        else:
            # Tìm dòng chữ in hoa độc lập
            for line in lines:
                if (
                    re.match(r"^[A-ZÀ-Ỹ\s]{4,40}$", line)
                    and "CỘNG HÒA" not in line
                    and "VIỆT NAM" not in line
                    and "CĂN CƯỚC" not in line
                ):
                    ho_ten = line.strip()
                    break

        # 5. Quê quán và Thường trú
        match_qq = re.search(
            r"(?:quê quán|nguyên quán|place of origin)[\s:]*([^\n;]+)",
            text,
            flags=re.IGNORECASE,
        )
        if match_qq:
            que_quan = match_qq.group(1).strip()

        match_tt = re.search(
            r"(?:nơi thường trú|thường trú|place of residence)[\s:]*([^\n;]+)",
            text,
            flags=re.IGNORECASE,
        )
        if match_tt:
            thuong_tru = match_tt.group(1).strip()

        # 6. Ngày hết hạn
        match_hh = re.search(
            r"(?:có giá trị đến|hết hạn|date of expiry)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{4}|không thời hạn)",
            text,
            flags=re.IGNORECASE,
        )
        if match_hh:
            ngay_het_han = match_hh.group(1).strip()

        # Tính điểm độ tin cậy dựa trên các trường quan trọng
        cac_truong = [so_cccd, ho_ten, ngay_sinh, gioi_tinh, que_quan or thuong_tru]
        so_truong_co = sum(1 for t in cac_truong if t)
        do_tin_cay = round(so_truong_co / len(cac_truong), 2)

        return {
            "loai_giay_to": "CCCD_VIET_NAM",
            "so_cccd": so_cccd,
            "ho_va_ten": ho_ten,
            "ngay_sinh": ngay_sinh,
            "gioi_tinh": gioi_tinh,
            "que_quan": que_quan,
            "noi_thuong_tru": thuong_tru,
            "ngay_het_han": ngay_het_han,
            "do_tin_cay": do_tin_cay,
            "van_ban_goc": text,
        }

    def trich_xuat_hoa_don(self, dau_vao: Union[str, Path, np.ndarray, Any]) -> Dict[str, Any]:
        """
        Trích xuất thông tin Hóa đơn Giá trị gia tăng (VAT Invoice).
        """
        text = self._chuan_bi_van_ban(dau_vao)

        so_hoa_don = ""
        ky_hieu = ""
        ngay_lap = ""
        mst_ban = ""
        ten_don_vi_ban = ""
        mst_mua = ""
        ten_don_vi_mua = ""
        tong_tien_truoc_thue = 0.0
        tien_thue = 0.0
        tong_thanh_toan = 0.0

        # 1. Số hóa đơn (7-8 chữ số)
        m_so_hd = re.search(r"(?:số|hóa đơn số|no\.)[\s:]*(\d{6,8})\b", text, flags=re.IGNORECASE)
        if m_so_hd:
            so_hoa_don = m_so_hd.group(1)

        # 2. Ký hiệu hóa đơn (VD: 1C24TAA, C23T)
        m_kh = re.search(r"(?:ký hiệu|serial)[\s:]*([12]?[A-Z]\d{2}[A-Z]{2,3})", text, flags=re.IGNORECASE)
        if m_kh:
            ky_hieu = m_kh.group(1)

        # 3. Mã số thuế (10 hoặc 13 chữ số)
        mst_list = re.findall(r"\b(\d{10}(?:-\d{3})?)\b", text)
        if mst_list:
            mst_ban = mst_list[0]
            if len(mst_list) > 1:
                mst_mua = mst_list[1]

        # 4. Ngày lập hóa đơn
        m_date = re.search(r"(?:ngày|date)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})", text, flags=re.IGNORECASE)
        if m_date:
            ngay_lap = m_date.group(1).replace("-", "/")
        else:
            dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
            if dates:
                ngay_lap = dates[0]

        # 5. Tên đơn vị bán & mua
        m_ban = re.search(
            r"(?:đơn vị bán|người bán|seller)[\s:]*([^\n;]+)",
            text,
            flags=re.IGNORECASE,
        )
        if m_ban:
            ten_don_vi_ban = m_ban.group(1).strip()

        m_mua = re.search(
            r"(?:đơn vị mua|người mua|buyer)[\s:]*([^\n;]+)",
            text,
            flags=re.IGNORECASE,
        )
        if m_mua:
            ten_don_vi_mua = m_mua.group(1).strip()

        # 6. Các khoản tiền (tiền hàng, thuế, tổng cộng)
        so_tien_list = re.findall(r"\b(\d{1,3}(?:[.,]\d{3})+(?:\s*(?:đồng|đ|vnd|vnđ))?)\b", text, flags=re.IGNORECASE)
        gia_tri_so: List[float] = []
        for st in so_tien_list:
            clean_num = re.sub(r"[^\d]", "", st)
            if clean_num:
                try:
                    val = float(clean_num)
                    if val > 1000:
                        gia_tri_so.append(val)
                except ValueError:
                    pass

        if gia_tri_so:
            gia_tri_so.sort()
            tong_thanh_toan = gia_tri_so[-1]
            if len(gia_tri_so) >= 2:
                tong_tien_truoc_thue = gia_tri_so[-2]
                tien_thue = round(tong_thanh_toan - tong_tien_truoc_thue, 2)
            else:
                tong_tien_truoc_thue = round(tong_thanh_toan / 1.1, 2)
                tien_thue = round(tong_thanh_toan - tong_tien_truoc_thue, 2)

        cac_truong = [so_hoa_don, mst_ban, ngay_lap, tong_thanh_toan > 0]
        do_tin_cay = round(sum(1 for c in cac_truong if c) / len(cac_truong), 2)

        return {
            "loai_tai_lieu": "HOA_DON_GTGT",
            "so_hoa_don": so_hoa_don,
            "ky_hieu": ky_hieu,
            "ngay_lap": ngay_lap,
            "don_vi_ban": ten_don_vi_ban,
            "mst_ban": mst_ban,
            "don_vi_mua": ten_don_vi_mua,
            "mst_mua": mst_mua,
            "tong_tien_truoc_thue": tong_tien_truoc_thue,
            "tien_thue": tien_thue,
            "tong_thanh_toan": tong_thanh_toan,
            "do_tin_cay": do_tin_cay,
            "van_ban_goc": text,
        }

    def trich_xuat_theo_schema(
        self,
        dau_vao: Union[str, Path, np.ndarray, Any],
        schema: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Trích xuất văn bản linh hoạt theo schema người dùng định nghĩa `{"field_name": "mô tả hoặc regex"}`.
        """
        text = self._chuan_bi_van_ban(dau_vao)
        ket_qua: Dict[str, Any] = {}
        for truong, mau in schema.items():
            try:
                m = re.search(mau, text, flags=re.IGNORECASE)
                if m:
                    ket_qua[truong] = m.group(1) if m.groups() else m.group(0)
                else:
                    ket_qua[truong] = ""
            except Exception:
                ket_qua[truong] = ""
        return {"ket_qua": ket_qua, "van_ban_goc": text}


# English alias
SmartExtractor = TrichXuatThongMinh
