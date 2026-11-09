"""
Hệ thống Phòng thủ Dữ liệu Đa tầng & Kiểm định Chất lượng Đầu vào – Đầu ra (Data Defense & I/O Gate).

Bao gồm 4 lá chắn chuyên sâu chạy 100% trên nền NumPy + thư viện chuẩn (giữ nguyên Micro-Core siêu nhẹ):
1. `KhienBaoVeHinhAnh` (`ImageSecurityShield`): Quét Polyglot/Appended Malware, chống Image Decompression Bomb,
   phát hiện giấu tin LSB Steganography, phát hiện chữ tàng hình tương phản thấp & khử độc ảnh CDR.
2. `PhongThuDuLieu` (`DatasetDefenseShield`) & `BoGiamSatTichTuDuLieu`: Phát hiện "Bom hẹn giờ" Sleeper Backdoor
   khi tập dữ liệu lớn vượt ngưỡng tích tụ, quét mã độc cắt mảnh qua nhiều dòng (Split-Payload),
   phân tích chữ ký phổ SVD (`Spectral Signatures`) và phát hiện đảo nhãn `k-NN`.
3. `CongChatLuongDauVao` (`InputQualityGate`): Chấm điểm sức khỏe dữ liệu (0–100), chặn Unicode Trojan Bidi,
   Null-byte, ngoại lai MAD Z-Score và rò rỉ trùng lặp Train–Test.
4. `CongAnToanDauRa` (`OutputSafetyQualityGate`): Chặn rò rỉ dữ liệu qua Markdown/URL Beacon,
   chặn mã lệnh phá hoại, sửa lỗi lặp suy thoái (Degeneration Loop) và đo độ bám sát ngữ cảnh RAG.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from vietnamese_ai.security.data_sanitizer import DataSanitizer

# =====================================================================
# 1. KHIÊN BẢO VỆ HÌNH ẢNH & KHỬ ĐỘC CDR (IMAGE SECURITY SHIELD)
# =====================================================================


class KhienBaoVeHinhAnh:
    """
    Lá chắn bảo vệ hình ảnh đa tầng chống mã độc ẩn giấu & tấn công thị giác:
    - Kiểm tra Magic Bytes thực tế & phát hiện Polyglot / dữ liệu độc nối đuôi sau `IEND` (PNG) / `FF D9` (JPEG).
    - Chặn **Image Decompression Bomb** ngay từ phần header nhị phân trước khi cấp phát RAM giải mã ảnh.
    - Phát hiện giấu tin **LSB Steganography** trên vùng phẳng và phân tích thống kê mặt phẳng bit thấp.
    - Phát hiện **Chữ tàng hình độ tương phản cực thấp** (Visual / Typographic Prompt Injection).
    - Khử độc **CDR (Content Disarm & Reconstruction)**: Tái tạo mảng pixel sạch, loại bỏ 100% metadata,
      cắt bỏ phần đuôi ẩn và làm phẳng kênh giấu tin LSB.
    """

    MAU_BYTE_NGUY_HIEM: Tuple[Tuple[bytes, str], ...] = (
        (b"PK\x03\x04", "Kho lưu trữ ZIP/JAR ẩn nối đuôi ảnh (Polyglot Archive)"),
        (b"\x7fELF", "Mã thực thi nhị phân Linux ELF ẩn trong ảnh"),
        (b"MZ\x90\x00", "Mã thực thi Windows PE/EXE ẩn trong ảnh"),
        (b"#!/bin/", "Shell script thực thi ẩn trong ảnh"),
        (b"<script", "Mã kịch bản JavaScript/XSS nhúng trong metadata/SVG"),
        (b"javascript:", "URI thực thi JavaScript độc hại trong ảnh"),
        (b"<!ENTITY", "Tấn công thực thể XML bên ngoài (XXE) trong SVG/XMP"),
        (b"\x80\x04\x95", "Payload tuần tự hóa Python Pickle v4 ẩn trong ảnh"),
        (b"os.system(", "Lệnh thực thi hệ thống Python ẩn trong metadata ảnh"),
        (b"subprocess.Popen", "Lệnh gọi tiến trình con ẩn trong metadata ảnh"),
        (b"eval(base64", "Payload mã hóa thực thi động ẩn trong ảnh"),
    )

    def __init__(
        self,
        gioi_han_pixel: int = 25_000_000,
        gioi_han_kich_thuoc_mb: float = 50.0,
        gioi_han_duoi_thua_bytes: int = 64,
    ):
        self.gioi_han_pixel = max(1024, int(gioi_han_pixel))
        self.gioi_han_kich_thuoc_bytes = int(max(0.1, float(gioi_han_kich_thuoc_mb)) * 1024 * 1024)
        self.gioi_han_duoi_thua_bytes = max(0, int(gioi_han_duoi_thua_bytes))

    @staticmethod
    def _nhan_dien_dinh_dang_va_kich_thuoc(raw: bytes) -> Tuple[str, int, int]:
        """Đọc trực tiếp Magic Bytes và kích thước `(width, height)` từ header mà không giải nén ảnh."""
        if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
            w, h = struct.unpack(">II", raw[16:24])
            return "png", int(w), int(h)

        if raw.startswith(b"\xff\xd8\xff"):
            # Duyệt các marker JPEG tìm SOF0..SOF3 để lấy kích thước thật trước khi giải nén
            idx = 2
            n = len(raw)
            while idx + 9 < n:
                if raw[idx] != 0xFF:
                    idx += 1
                    continue
                marker = raw[idx + 1]
                while marker == 0xFF and idx + 2 < n:
                    idx += 1
                    marker = raw[idx + 1]
                if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
                    idx += 2
                    continue
                if idx + 4 > n:
                    break
                seg_len = struct.unpack(">H", raw[idx + 2 : idx + 4])[0]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3) and idx + 9 <= n:
                    h, w = struct.unpack(">HH", raw[idx + 5 : idx + 9])
                    return "jpeg", int(w), int(h)
                if seg_len < 2:
                    break
                idx += 2 + seg_len
            return "jpeg", 0, 0

        if raw[:6] in (b"GIF87a", b"GIF89a") and len(raw) >= 10:
            w, h = struct.unpack("<HH", raw[6:10])
            return "gif", int(w), int(h)

        if raw.startswith(b"BM") and len(raw) >= 26:
            w, h = struct.unpack("<ii", raw[18:26])
            return "bmp", abs(int(w)), abs(int(h))

        if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            return "webp", 0, 0

        dau_sach = raw[:512].lstrip().lower()
        if dau_sach.startswith(b"<svg") or (dau_sach.startswith(b"<?xml") and b"<svg" in dau_sach):
            return "svg", 0, 0

        return "khong_xac_dinh", 0, 0

    def _kiem_tra_duoi_polyglot(self, raw: bytes, dinh_dang: str) -> Tuple[int, List[str]]:
        """Phát hiện dữ liệu ẩn nối sau điểm kết thúc hợp lệ của tệp ảnh."""
        canh_bao: List[str] = []
        duoi_thua = b""

        if dinh_dang == "png":
            iend_marker = b"IEND\xaeB`\x82"
            pos = raw.find(iend_marker)
            if pos != -1:
                duoi_thua = raw[pos + len(iend_marker) :]
        elif dinh_dang == "jpeg":
            pos = raw.rfind(b"\xff\xd9")
            # Nếu có dữ liệu lớn nằm giữa EOI đầu tiên và cuối file, kiểm tra EOI hợp lệ
            first_eoi = raw.find(b"\xff\xd9")
            if first_eoi != -1 and len(raw) - (first_eoi + 2) > self.gioi_han_duoi_thua_bytes:
                # Kiểm tra xem phần sau first_eoi có chứa chữ ký mã độc không
                phan_sau = raw[first_eoi + 2 :]
                for sig, mo_ta in self.MAU_BYTE_NGUY_HIEM:
                    if sig in phan_sau:
                        duoi_thua = phan_sau
                        break
            if not duoi_thua and pos != -1:
                duoi_thua = raw[pos + 2 :]

        so_byte_thua = len(duoi_thua.strip(b"\x00\r\n\t "))
        if so_byte_thua > self.gioi_han_duoi_thua_bytes:
            canh_bao.append(
                f"Phát hiện {len(duoi_thua)} bytes dữ liệu lạ nối đuôi sau điểm kết thúc ảnh ({dinh_dang.upper()})."
            )

        for sig, mo_ta in self.MAU_BYTE_NGUY_HIEM:
            if sig.lower() in raw.lower():
                canh_bao.append(mo_ta)

        # Quét văn bản ASCII khả nghi nhúng trong metadata/comment của ảnh
        chuoi_ascii = "".join(chr(b) if 32 <= b <= 126 else " " for b in raw[:65536] + duoi_thua[:65536])
        from vietnamese_ai.security.llm_firewall import TuongLuaAI

        tuong_lua = TuongLuaAI(ngat_ket_noi_khi_phat_hien=False)
        hop_le_prompt, ly_do_prompt = tuong_lua.kiem_tra_prompt(chuoi_ascii)
        if not hop_le_prompt:
            canh_bao.append(f"Chỉ thị Prompt Injection ẩn trong nhị phân/EXIF ảnh: {ly_do_prompt}")

        return len(duoi_thua), canh_bao

    @staticmethod
    def phan_tich_stego_va_chu_an(arr_anh: np.ndarray) -> Dict[str, Any]:
        """
        Phân tích mặt phẳng bit thấp (LSB Steganography) và văn bản tàng hình độ tương phản thấp.
        """
        arr = np.asarray(arr_anh)
        if arr.size == 0:
            return {
                "nghi_ngo_stego_lsb": False,
                "nghi_ngo_chu_tang_hinh": False,
                "entropy_lsb_vung_phang": 0.0,
                "ty_le_vi_tuong_phan": 0.0,
            }

        # Chuẩn hóa về uint8 [0, 255]
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.max(arr)) if arr.size > 0 else 1.0
            arr_u8 = np.clip(arr * 255.0 if max_val <= 1.0 else arr, 0, 255).astype(np.uint8)
        else:
            arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)

        kenh_dau = arr_u8[..., 0] if arr_u8.ndim == 3 else arr_u8
        h, w = kenh_dau.shape[:2]

        # 1. Kiểm tra LSB Steganography trên các vùng nền mượt (7 bit cao đồng nhất nhưng bit LSB nhiễu ngẫu nhiên)
        upper7 = kenh_dau >> 1
        lsb = kenh_dau & 1
        entropy_lsb_phang = 0.0
        nghi_ngo_stego = False

        if h >= 8 and w >= 8:
            # Xét độ chênh 7-bit cao theo láng giềng ngang
            dong_nhat_7bit = upper7[:, :-1] == upper7[:, 1:]
            so_cap_phang = int(np.sum(dong_nhat_7bit))
            if so_cap_phang >= 64:
                lsb_trai = lsb[:, :-1][dong_nhat_7bit]
                lsb_phai = lsb[:, 1:][dong_nhat_7bit]
                ty_le_doi_bit = float(np.mean(lsb_trai != lsb_phai))
                p1 = float(np.mean(lsb_trai))
                if 1e-6 < p1 < 1.0 - 1e-6:
                    entropy_lsb_phang = -(p1 * math.log2(p1) + (1.0 - p1) * math.log2(1.0 - p1))
                # Dữ liệu mã hóa/nén nhúng vào LSB trên vùng nền phẳng tạo xác suất đảo bit ~0.5 và entropy ~1.0
                if entropy_lsb_phang > 0.985 and 0.42 <= ty_le_doi_bit <= 0.58:
                    nghi_ngo_stego = True

        # 2. Phát hiện chữ tàng hình tương phản cực thấp (chênh 1..5 mức xám so với nền đồng nhất)
        ty_le_vi_tuong_phan = 0.0
        nghi_ngo_chu_an = False
        if h >= 12 and w >= 24:
            nen_uoc_luong = float(np.median(kenh_dau))
            do_lech = np.abs(kenh_dau.astype(np.float32) - nen_uoc_luong)
            # Nền tổng thể trông gần như đồng nhất với mắt người (phần lớn pixel lệch <= 5)
            ty_le_nen_dong_nhat = float(np.mean(do_lech <= 5.0))
            vi_cham = (do_lech >= 1.0) & (do_lech <= 5.0)
            ty_le_vi_tuong_phan = float(np.mean(vi_cham))
            if ty_le_nen_dong_nhat >= 0.95 and 0.01 <= ty_le_vi_tuong_phan <= 0.45 and not nghi_ngo_stego:
                # Kiểm tra xem các điểm vi tương phản có xếp thành dải dòng ngang như văn bản ẩn không
                mat_do_dong = np.mean(vi_cham, axis=1)
                so_dong_co_net = int(np.sum(mat_do_dong > 0.05))
                so_dong_trang = int(np.sum(mat_do_dong < 0.005))
                if so_dong_co_net >= 3 and so_dong_trang >= 3:
                    nghi_ngo_chu_an = True

        return {
            "nghi_ngo_stego_lsb": nghi_ngo_stego,
            "nghi_ngo_chu_tang_hinh": nghi_ngo_chu_an,
            "entropy_lsb_vung_phang": round(entropy_lsb_phang, 4),
            "ty_le_vi_tuong_phan": round(ty_le_vi_tuong_phan, 4),
        }

    def quet_anh(
        self,
        anh: Union[str, Path, bytes, np.ndarray, Any],
        chan_khi_nguy_hiem: bool = False,
    ) -> Dict[str, Any]:
        """
        Quét toàn diện tệp ảnh hoặc mảng ảnh để phát hiện virus ẩn, Polyglot, Image Bomb,
        LSB Steganography và chữ ẩn tương phản thấp.
        """
        moi_de_doa: List[str] = []
        dinh_dang = "ndarray"
        width, height = 0, 0
        so_byte_duoi_thua = 0
        raw_bytes: Optional[bytes] = None

        if isinstance(anh, (str, Path)):
            p = Path(anh)
            if p.is_file():
                kich_thuoc_file = p.stat().st_size
                if kich_thuoc_file > self.gioi_han_kich_thuoc_bytes:
                    moi_de_doa.append(
                        f"Kích thước tệp ảnh ({kich_thuoc_file} bytes) vượt ngưỡng an toàn ({self.gioi_han_kich_thuoc_bytes} bytes)."
                    )
                with open(p, "rb") as f:
                    raw_bytes = f.read(min(kich_thuoc_file, self.gioi_han_kich_thuoc_bytes + 1024))
        elif isinstance(anh, (bytes, bytearray)):
            raw_bytes = bytes(anh)
            if len(raw_bytes) > self.gioi_han_kich_thuoc_bytes:
                moi_de_doa.append("Dữ liệu byte ảnh vượt ngưỡng kích thước an toàn.")

        if raw_bytes is not None:
            dinh_dang, width, height = self._nhan_dien_dinh_dang_va_kich_thuoc(raw_bytes)
            if dinh_dang == "khong_xac_dinh":
                moi_de_doa.append("Magic Bytes không khớp định dạng ảnh hợp lệ (nghi ngờ giả mạo đuôi file).")
            if width > 0 and height > 0 and (width * height) > self.gioi_han_pixel:
                moi_de_doa.append(
                    f"Chặn Image Decompression Bomb: Kích thước khai báo {width}x{height} "
                    f"({width * height:,} pixels > giới hạn {self.gioi_han_pixel:,} pixels)."
                )
            so_byte_duoi_thua, canh_bao_poly = self._kiem_tra_duoi_polyglot(raw_bytes, dinh_dang)
            moi_de_doa.extend(canh_bao_poly)

        # Chỉ giải mã mảng pixel nếu không phải Image Bomb
        stego_info: Dict[str, Any] = {
            "nghi_ngo_stego_lsb": False,
            "nghi_ngo_chu_tang_hinh": False,
            "entropy_lsb_vung_phang": 0.0,
            "ty_le_vi_tuong_phan": 0.0,
        }
        co_bomb = any("Decompression Bomb" in m for m in moi_de_doa)
        if not co_bomb:
            arr_pixel = self._tai_mang_pixel_an_toan(anh)
            if arr_pixel is not None and arr_pixel.size > 0:
                height, width = int(arr_pixel.shape[0]), int(arr_pixel.shape[1])
                if width * height > self.gioi_han_pixel:
                    moi_de_doa.append(
                        f"Chặn Image Decompression Bomb: Mảng ảnh {width}x{height} vượt giới hạn an toàn."
                    )
                else:
                    stego_info = self.phan_tich_stego_va_chu_an(arr_pixel)
                    if stego_info["nghi_ngo_stego_lsb"]:
                        moi_de_doa.append(
                            f"Phát hiện dấu hiệu giấu tin LSB Steganography (entropy vùng phẳng={stego_info['entropy_lsb_vung_phang']})."
                        )
                    if stego_info["nghi_ngo_chu_tang_hinh"]:
                        moi_de_doa.append(
                            "Phát hiện dải văn bản tàng hình độ tương phản cực thấp (nguy cơ Visual Prompt Injection)."
                        )

        an_toan = len(moi_de_doa) == 0
        if chan_khi_nguy_hiem and not an_toan:
            raise ValueError(f"Bảo mật Hình ảnh chặn đầu vào: {'; '.join(moi_de_doa)}")

        return {
            "an_toan": an_toan,
            "dinh_dang": dinh_dang,
            "kich_thuoc": (width, height),
            "so_byte_duoi_thua": so_byte_duoi_thua,
            "moi_de_doa": moi_de_doa,
            **stego_info,
        }

    def _tai_mang_pixel_an_toan(self, anh: Any) -> Optional[np.ndarray]:
        """Tải mảng pixel có kiểm soát giới hạn kích thước."""
        if isinstance(anh, np.ndarray):
            if anh.shape[0] * (anh.shape[1] if anh.ndim >= 2 else 1) > self.gioi_han_pixel:
                return None
            return anh
        if hasattr(anh, "convert") and hasattr(anh, "size"):
            w, h = anh.size
            if w * h > self.gioi_han_pixel:
                return None
            return np.asarray(anh)
        if isinstance(anh, (str, Path)) and Path(anh).is_file():
            try:
                from PIL import Image

                Image.MAX_IMAGE_PIXELS = self.gioi_han_pixel
                with Image.open(str(anh)) as img:
                    w, h = img.size
                    if w * h > self.gioi_han_pixel:
                        return None
                    return np.asarray(img.convert("RGB"))
            except Exception:
                return None
        return None

    def khu_doc_anh_cdr(
        self,
        anh: Union[str, Path, np.ndarray, Any],
        duong_dan_xuat: Optional[Union[str, Path]] = None,
        xoa_lsb_so_bit: int = 2,
        xoa_chu_tang_hinh: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Content Disarm & Reconstruction (CDR) cho hình ảnh:
        1. Chặn ngay lập tức nếu là Image Decompression Bomb.
        2. Tái tạo mảng pixel thuần khiết (loại bỏ 100% EXIF, metadata script và mã độc Polyglot nối đuôi).
        3. Xóa sạch mặt phẳng bit thấp LSB (`xoa_lsb_so_bit` bit cuối) để vô hiệu hóa hoàn toàn payload Steganography.
        4. Làm phẳng các nét chữ tàng hình vi tương phản (`<= 5/255`) trên nền đồng nhất trước khi đưa vào OCR/VLM.
        """
        bao_cao = self.quet_anh(anh, chan_khi_nguy_hiem=False)
        if any("Decompression Bomb" in m for m in bao_cao["moi_de_doa"]):
            raise ValueError(f"CDR từ chối xử lý Image Bomb: {'; '.join(bao_cao['moi_de_doa'])}")

        arr = self._tai_mang_pixel_an_toan(anh)
        if arr is None:
            raise ValueError("Không thể giải mã mảng pixel hợp lệ để khử độc CDR.")

        if np.issubdtype(arr.dtype, np.floating):
            max_v = float(np.max(arr)) if arr.size > 0 else 1.0
            arr_u8 = np.clip(arr * 255.0 if max_v <= 1.0 else arr, 0, 255).astype(np.uint8)
        else:
            arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)

        # Vô hiệu hóa hoàn toàn dữ liệu ẩn trong các bit LSB
        if xoa_lsb_so_bit > 0:
            so_bit = min(4, max(1, int(xoa_lsb_so_bit)))
            mask = (0xFF << so_bit) & 0xFF
            arr_sach = (arr_u8 & mask).astype(np.uint8)
        else:
            arr_sach = arr_u8.copy()

        # Triệt tiêu chữ tàng hình vi tương phản nếu phát hiện trên nền đồng nhất
        if xoa_chu_tang_hinh and bao_cao.get("nghi_ngo_chu_tang_hinh"):
            nen = int(round(float(np.median(arr_sach))))
            khoang_cach = np.abs(arr_sach.astype(np.int16) - nen)
            arr_sach = np.where(khoang_cach <= 6, nen, arr_sach).astype(np.uint8)

        if duong_dan_xuat is not None:
            from PIL import Image

            out_img = Image.fromarray(arr_sach)
            # Lưu lại từ mảng pixel thuần khiết -> tự động loại bỏ 100% EXIF và đuôi Polyglot
            out_img.save(str(duong_dan_xuat))

        bao_cao["da_khu_doc_cdr"] = True
        return arr_sach, bao_cao


# =====================================================================
# 2. PHÒNG THỦ TẬP DỮ LIỆU LỚN & CHỐNG "BOM HẸN GIỜ" SLEEPER BACKDOOR
# =====================================================================


class BoGiamSatTichTuDuLieu:
    """
    Bộ giám sát tích tụ theo luồng (Streaming Critical-Mass & Sleeper Backdoor Monitor).

    Giải quyết bài toán: Kẻ tấn công rải các mẫu đầu độc trông hoàn toàn lành tính ở từng dòng đơn lẻ,
    hoặc cắt nhỏ mã độc qua nhiều dòng liên tiếp, chờ khi dữ liệu tích tụ đủ lớn mới kích nổ.
    """

    def __init__(
        self,
        nguong_kich_hoat_mau: int = 20,
        so_lan_lap_toi_thieu: int = 4,
        ty_le_ep_nhan_toi_thieu: float = 0.90,
        ty_le_hiem_toi_da: float = 0.35,
    ):
        self.nguong_kich_hoat_mau = max(8, int(nguong_kich_hoat_mau))
        self.so_lan_lap_toi_thieu = max(3, int(so_lan_lap_toi_thieu))
        self.ty_le_ep_nhan_toi_thieu = float(ty_le_ep_nhan_toi_thieu)
        self.ty_le_hiem_toi_da = float(ty_le_hiem_toi_da)

        self.tong_so_mau: int = 0
        self._dem_nhan: Counter[str] = Counter()
        self._dem_cum_tu: Counter[str] = Counter()
        self._dem_cum_tu_theo_nhan: Dict[str, Counter[str]] = {}
        self._vi_tri_mau_theo_cum_tu: Dict[str, List[int]] = {}
        self._cua_so_dau_vao: List[Tuple[int, str]] = []
        self._cua_so_muc_tieu: List[Tuple[int, str]] = []
        self.vi_tri_split_payload: set[int] = set()
        self._mean_nen: Optional[np.ndarray] = None
        self.canh_bao_troi_dat: List[str] = []

    @staticmethod
    def _khop_mau_injection(tuong_lua: Any, van_ban: str) -> bool:
        """Kiểm tra khớp trực tiếp mẫu Regex độc hại sau chuẩn hóa NFKC/Zero-Width/Base64."""
        if not van_ban:
            return False
        chuan = tuong_lua.chuan_hoa_prompt(van_ban).lower()
        return any(mau.search(chuan) for mau in tuong_lua.mau_regex)

    def _quet_cua_so_split_payload(
        self,
        tuong_lua: Any,
        cua_so: List[Tuple[int, str]],
        idx_moi: int,
        van_ban_moi: str,
    ) -> None:
        """Phát hiện mã độc cắt mảnh qua ranh giới 2 hoặc 3 bản ghi liên tiếp (Cross-Row Split-Payload)."""
        txt = van_ban_moi.strip()
        if not txt:
            return
        cua_so.append((idx_moi, txt))
        if len(cua_so) > 3:
            cua_so.pop(0)

        # Kiểm tra đường nối (boundary seam) giữa 2 bản ghi kề nhau
        if len(cua_so) >= 2:
            idx_a, txt_a = cua_so[-2]
            idx_b, txt_b = cua_so[-1]
            tu_a = txt_a.split()
            tu_b = txt_b.split()
            duoi_a = " ".join(tu_a[-6:]) if tu_a else txt_a
            for do_dai_dau_b in (3, 6, len(tu_b)):
                dau_b = " ".join(tu_b[: max(1, do_dai_dau_b)]) if tu_b else txt_b
                if not self._khop_mau_injection(tuong_lua, duoi_a) and not self._khop_mau_injection(
                    tuong_lua, dau_b
                ):
                    if self._khop_mau_injection(tuong_lua, f"{duoi_a} {dau_b}") or self._khop_mau_injection(
                        tuong_lua, f"{duoi_a}{dau_b}"
                    ):
                        self.vi_tri_split_payload.update((idx_a, idx_b))
                        return

        # Kiểm tra mã độc cắt mảnh qua 3 bản ghi liên tiếp
        if len(cua_so) == 3:
            (i1, t1), (i2, t2), (i3, t3) = cua_so
            if (
                not self._khop_mau_injection(tuong_lua, f"{t1} {t2}")
                and not self._khop_mau_injection(tuong_lua, f"{t2} {t3}")
                and self._khop_mau_injection(tuong_lua, f"{t1} {t2} {t3}")
            ):
                self.vi_tri_split_payload.update((i1, i2, i3))

    @staticmethod
    def _tach_dac_trung_cum_tu(van_ban: str) -> set[str]:
        """Trích xuất các từ và cụm 2-gram có ý nghĩa để theo dõi tín hiệu kích hoạt ngầm (Trigger)."""
        words = re.findall(r"[a-zA-Z0-9_À-ỹ\[\]\-#@]{3,}", van_ban.lower())
        ket_qua: set[str] = set(words[:64])
        for i in range(min(len(words) - 1, 48)):
            ket_qua.add(f"{words[i]} {words[i + 1]}")
        return ket_qua

    def cap_nhat_lo(self, lo_du_lieu: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Cập nhật một lô bản ghi mới và kiểm tra ngay nguy cơ tích tụ chạm ngưỡng bùng nổ."""
        from vietnamese_ai.security.llm_firewall import TuongLuaAI

        tuong_lua = TuongLuaAI(ngat_ket_noi_khi_phat_hien=False)

        for mau in lo_du_lieu:
            if not isinstance(mau, dict):
                continue
            idx_toan_cuc = self.tong_so_mau
            self.tong_so_mau += 1

            dau_vao = " ".join(
                str(mau[k])
                for k in ("instruction", "input", "prompt", "question", "text", "van_ban", "context")
                if k in mau and isinstance(mau[k], str)
            ).strip()
            muc_tieu = str(
                mau.get(
                    "label",
                    mau.get(
                        "nhan",
                        mau.get("output", mau.get("chosen", mau.get("answer", mau.get("response", "")))),
                    ),
                )
            ).strip()
            # Chuẩn hóa mục tiêu để nhận diện hành vi ép đầu ra giống hệt nhau
            khoa_muc_tieu = muc_tieu[:120].lower() if muc_tieu else "_khong_nhan_"
            self._dem_nhan[khoa_muc_tieu] += 1

            # 1. Theo dõi mã độc cắt mảnh ghép nối qua 2–3 bản ghi liên tiếp (Split-Payload Poisoning)
            self._quet_cua_so_split_payload(tuong_lua, self._cua_so_dau_vao, idx_toan_cuc, dau_vao)
            self._quet_cua_so_split_payload(tuong_lua, self._cua_so_muc_tieu, idx_toan_cuc, muc_tieu)

            # 2. Theo dõi tương quan cụm từ hiếm -> nhãn mục tiêu (Sleeper Backdoor Trigger)
            if dau_vao and khoa_muc_tieu != "_khong_nhan_":
                for tok in self._tach_dac_trung_cum_tu(dau_vao):
                    self._dem_cum_tu[tok] += 1
                    if tok not in self._dem_cum_tu_theo_nhan:
                        self._dem_cum_tu_theo_nhan[tok] = Counter()
                    self._dem_cum_tu_theo_nhan[tok][khoa_muc_tieu] += 1
                    ds_vt = self._vi_tri_mau_theo_cum_tu.setdefault(tok, [])
                    if len(ds_vt) < 500:
                        ds_vt.append(idx_toan_cuc)

        return self.kiem_tra_bom_tich_tu()

    def kiem_tra_bom_tich_tu(self) -> Dict[str, Any]:
        """Đánh giá xem tập dữ liệu đã xuất hiện Sleeper Trigger hoặc Split-Payload chạm ngưỡng hay chưa."""
        triggers_phat_hien: List[Dict[str, Any]] = []
        vi_tri_nghi_ngo: set[int] = set(self.vi_tri_split_payload)

        if self.tong_so_mau >= self.nguong_kich_hoat_mau:
            for tok, so_lan in self._dem_cum_tu.items():
                if so_lan < self.so_lan_lap_toi_thieu:
                    continue
                ty_le_xuat_hien = so_lan / self.tong_so_mau
                if ty_le_xuat_hien > self.ty_le_hiem_toi_da:
                    continue

                nhan_pho_bien, so_lan_ep = self._dem_cum_tu_theo_nhan[tok].most_common(1)[0]
                xac_suat_ep = so_lan_ep / so_lan
                ty_le_goc_nhan = self._dem_nhan[nhan_pho_bien] / self.tong_so_mau
                he_so_lift = xac_suat_ep / max(1e-6, ty_le_goc_nhan)

                if (
                    xac_suat_ep >= self.ty_le_ep_nhan_toi_thieu
                    and ty_le_goc_nhan <= 0.65
                    and he_so_lift >= 1.5
                ):
                    cac_vt = self._vi_tri_mau_theo_cum_tu.get(tok, [])
                    vi_tri_nghi_ngo.update(cac_vt)
                    triggers_phat_hien.append(
                        {
                            "trigger": tok,
                            "muc_tieu_bi_ep": nhan_pho_bien[:60],
                            "so_mau_nhiem": so_lan,
                            "xac_suat_ep_nhan": round(xac_suat_ep, 4),
                            "he_so_lift": round(he_so_lift, 2),
                        }
                    )

        triggers_phat_hien.sort(
            key=lambda d: (-d["he_so_lift"], -d["so_mau_nhiem"], -len(d["trigger"]), d["trigger"])
        )
        co_bom = len(triggers_phat_hien) > 0 or len(self.vi_tri_split_payload) > 0
        return {
            "tong_so_mau_da_duyet": self.tong_so_mau,
            "phat_hien_bom_hen_gio": co_bom,
            "so_trigger_ngam": len(triggers_phat_hien),
            "danh_sach_trigger_nghi_ngo": triggers_phat_hien[:20],
            "so_mau_split_payload": len(self.vi_tri_split_payload),
            "chi_so_mau_bi_cach_ly": sorted(vi_tri_nghi_ngo),
        }


class PhongThuDuLieu:
    """
    Lá chắn phòng thủ tập dữ liệu toàn cục (Dataset Defense Shield):
    - Phân tích **Chữ ký Phổ SVD (`Spectral Signatures`)** tìm cụm đầu độc ẩn trong không gian vector.
    - Phát hiện **Đảo Nhãn (`k-NN Label Consensus`)** cho bài toán phân loại.
    - Quét **Sleeper Backdoor**, **Split-Payload** và **Unicode Trojan Bidi** trên tập dữ liệu lớn.
    """

    @staticmethod
    def _nhung_van_ban_nhanh(danh_sach_van_ban: List[str], chieu: int = 64) -> np.ndarray:
        """Nhúng văn bản siêu tốc bằng Feature Hashing xác định phục vụ phân tích phổ SVD."""
        mat = np.zeros((len(danh_sach_van_ban), chieu), dtype=np.float64)
        for i, txt in enumerate(danh_sach_van_ban):
            for w in re.findall(r"\w+", txt.lower()):
                h = int.from_bytes(hashlib.blake2b(w.encode("utf-8"), digest_size=4).digest(), "little")
                idx = h % chieu
                sign = 1.0 if (h & 1) == 0 else -1.0
                mat[i, idx] += sign
            norm = np.linalg.norm(mat[i])
            if norm > 1e-8:
                mat[i] /= norm
        return mat

    @classmethod
    def quet_chu_ky_pho_svd(
        cls,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        nguong_mad: float = 4.5,
        ty_le_cat_toi_da: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Thuật toán Spectral Signatures (dựa trên SVD của ma trận hiệp phương sai tâm hóa):
        Các mẫu bị cấy Backdoor tạo ra một hướng phương sai phụ mạnh dọc theo vector kỳ dị phải đầu tiên `v_1`.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n_samples = len(X_arr)
        if n_samples < 8:
            return {"chi_so_nghi_ngo": [], "diem_pho": np.zeros(n_samples).tolist()}

        diem_pho = np.zeros(n_samples, dtype=np.float64)
        chi_so_nghi_ngo: List[int] = []

        nhom_nhan = [np.arange(n_samples)]
        if y is not None:
            y_arr = np.asarray(y)
            nhom_nhan = [np.where(y_arr == c)[0] for c in np.unique(y_arr) if np.sum(y_arr == c) >= 6]

        for idx_nhom in nhom_nhan:
            if len(idx_nhom) < 6:
                continue
            R = X_arr[idx_nhom] - np.mean(X_arr[idx_nhom], axis=0, keepdims=True)
            try:
                _, _, vt = np.linalg.svd(R, full_matrices=False)
                v1 = vt[0]
                scores = (R @ v1) ** 2
            except np.linalg.LinAlgError:
                scores = np.sum(R**2, axis=1)

            diem_pho[idx_nhom] = scores
            med = float(np.median(scores))
            mad = float(np.median(np.abs(scores - med))) + 1e-9
            robust_z = (scores - med) / (1.4826 * mad)

            ung_vien = idx_nhom[robust_z > nguong_mad]
            gioi_han = max(1, int(len(idx_nhom) * ty_le_cat_toi_da))
            if len(ung_vien) > gioi_han:
                thu_tu = np.argsort(-scores)
                ung_vien = idx_nhom[thu_tu[:gioi_han]]
            chi_so_nghi_ngo.extend(int(i) for i in ung_vien)

        chi_so_nghi_ngo = sorted(set(chi_so_nghi_ngo))
        return {
            "chi_so_nghi_ngo": chi_so_nghi_ngo,
            "so_mau_nghi_ngo": len(chi_so_nghi_ngo),
            "diem_pho_toi_da": round(float(np.max(diem_pho)) if n_samples > 0 else 0.0, 4),
        }

    @staticmethod
    def phat_hien_dao_nhan_knn(
        X: np.ndarray,
        y: np.ndarray,
        k: int = 5,
        nguong_dong_thuan: float = 0.85,
    ) -> List[int]:
        """Phát hiện các điểm bị kẻ tấn công đảo nhãn ngược hoàn toàn với `k` láng giềng gần nhất."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        n = len(X_arr)
        if n <= k + 2 or len(np.unique(y_arr)) < 2:
            return []

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # Giới hạn tính khoảng cách chính xác cho N <= 2500 để luôn siêu nhanh
        if n > 2500:
            return []

        dists = np.sum((X_arr[:, None, :] - X_arr[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(dists, np.inf)
        nn_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]

        bi_dao_nhan: List[int] = []
        for i in range(n):
            nhan_lang_gieng = y_arr[nn_idx[i]]
            ty_le_khac = float(np.mean(nhan_lang_gieng != y_arr[i]))
            if ty_le_khac >= nguong_dong_thuan:
                # Kiểm tra các láng giềng có đồng thuận cùng một nhãn khác không
                vals, counts = np.unique(nhan_lang_gieng, return_counts=True)
                if np.max(counts) / k >= nguong_dong_thuan and vals[int(np.argmax(counts))] != y_arr[i]:
                    bi_dao_nhan.append(i)
        return bi_dao_nhan

    @staticmethod
    def do_lech_phan_phoi_psi(mang_goc: np.ndarray, mang_moi: np.ndarray, so_bin: int = 10) -> float:
        """Tính chỉ số Population Stability Index (PSI) để phát hiện đợt bơm dữ liệu đầu độc bất thường."""
        a = np.asarray(mang_goc, dtype=np.float64).ravel()
        b = np.asarray(mang_moi, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            return 0.0
        bins = np.percentile(a, np.linspace(0, 100, so_bin + 1))
        bins[0] -= 1e-6
        bins[-1] += 1e-6
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        pa, _ = np.histogram(a, bins=bins)
        pb, _ = np.histogram(b, bins=bins)
        pa_pct = np.clip(pa / max(1, len(a)), 1e-4, 1.0)
        pb_pct = np.clip(pb / max(1, len(b)), 1e-4, 1.0)
        return float(np.sum((pb_pct - pa_pct) * np.log(pb_pct / pa_pct)))

    @classmethod
    def quet_va_lam_sach_dataset(
        cls,
        du_lieu: Any,
        y: Optional[np.ndarray] = None,
        an_pii: bool = True,
        loai_bo_mau_doc: bool = True,
        nguong_kich_hoat_bom: int = 20,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Quét & cách ly toàn diện cho cả Dataset Văn bản (`List[Dict]` / `InstructionDataset`)
        lẫn Ma trận Đặc trưng Số `(X, y)`:
        1. Quét từng dòng chống Prompt Injection, Unicode Trojan Bidi & PII (`DataSanitizer`).
        2. Giám sát tích tụ chống **Sleeper Backdoor Trigger** & **Split-Payload** qua nhiều dòng.
        3. Phân tích **Chữ ký Phổ SVD** và **k-NN Label Flipping** trên ma trận số `(X, y)`.
        """
        # Trường hợp dữ liệu bảng (X, y)
        if isinstance(du_lieu, np.ndarray) or (isinstance(du_lieu, list) and du_lieu and not isinstance(du_lieu[0], dict)):
            X_arr = np.asarray(du_lieu, dtype=np.float64)
            kq_svd = cls.quet_chu_ky_pho_svd(X_arr, y)
            dao_nhan = cls.phat_hien_dao_nhan_knn(X_arr, y) if y is not None else []
            chi_so_loai = sorted(set(kq_svd["chi_so_nghi_ngo"]) | set(dao_nhan))
            bao_cao_so = {
                "loai_du_lieu": "bang_so",
                "tong_mau_kiem_tra": int(len(X_arr)),
                "so_mau_nghi_ngo_svd": kq_svd["so_mau_nghi_ngo"],
                "so_mau_dao_nhan_knn": len(dao_nhan),
                "chi_so_bi_cach_ly": chi_so_loai,
                "an_toan": len(chi_so_loai) == 0,
            }
            if loai_bo_mau_doc and chi_so_loai and len(chi_so_loai) < len(X_arr):
                giu = np.ones(len(X_arr), dtype=bool)
                giu[chi_so_loai] = False
                if y is not None:
                    return (X_arr[giu], np.asarray(y)[giu]), bao_cao_so
                return X_arr[giu], bao_cao_so
            return (X_arr, y) if y is not None else X_arr, bao_cao_so

        # Trường hợp Dataset văn bản / Fine-Tuning
        la_dataset_obj = hasattr(du_lieu, "train") and isinstance(du_lieu.train, list)
        items = du_lieu.train if la_dataset_obj else (du_lieu if isinstance(du_lieu, list) else None)
        if items is None or not items:
            return du_lieu, {
                "tong_mau_kiem_tra": 0,
                "so_mau_dau_doc_bi_loai": 0,
                "so_mau_da_an_pii": 0,
                "phat_hien_bom_hen_gio": False,
                "an_toan": True,
            }

        # Bước 1: Giám sát tích tụ Sleeper Trigger & Split-Payload trên toàn bộ tập
        giam_sat = BoGiamSatTichTuDuLieu(nguong_kich_hoat_mau=nguong_kich_hoat_bom)
        bc_bom = giam_sat.cap_nhat_lo(items)
        tap_cach_ly_bom = set(bc_bom["chi_so_mau_bi_cach_ly"])

        # Bước 2: Lọc Unicode Trojan Bidi + giữ các mẫu không thuộc ổ dịch Sleeper/Split-Payload
        items_loc_bom: List[Dict[str, Any]] = []
        so_unicode_trojan = 0
        for idx, mau in enumerate(items):
            if loai_bo_mau_doc and idx in tap_cach_ly_bom:
                continue
            if isinstance(mau, dict):
                mau_sach_bidi = {}
                for k, v in mau.items():
                    if isinstance(v, str):
                        v_clean, co_bidi = CongChatLuongDauVao.loai_bo_unicode_trojan(v)
                        if co_bidi:
                            so_unicode_trojan += 1
                        mau_sach_bidi[k] = v_clean
                    else:
                        mau_sach_bidi[k] = v
                items_loc_bom.append(mau_sach_bidi)
            else:
                items_loc_bom.append(mau)

        # Bước 3: Chạy tiếp bộ lọc từng dòng & ẩn PII/Secrets của DataSanitizer
        items_cuoi, bc_dong = DataSanitizer.quet_dau_doc_du_lieu(
            items_loc_bom, an_pii=an_pii, loai_bo_mau_doc=loai_bo_mau_doc
        )

        tong_bi_loai = len(tap_cach_ly_bom) + int(bc_dong.get("so_mau_dau_doc_bi_loai", 0))
        bao_cao_gop = {
            **bc_dong,
            "tong_mau_kiem_tra": len(items),
            "so_mau_dau_doc_bi_loai": tong_bi_loai,
            "so_unicode_trojan_da_khu": so_unicode_trojan,
            "phat_hien_bom_hen_gio": bc_bom["phat_hien_bom_hen_gio"],
            "danh_sach_trigger_nghi_ngo": bc_bom["danh_sach_trigger_nghi_ngo"],
            "so_mau_split_payload": bc_bom["so_mau_split_payload"],
            "an_toan": tong_bi_loai == 0 and so_unicode_trojan == 0 and not bc_bom["phat_hien_bom_hen_gio"],
        }

        if not items_cuoi and items:
            raise ValueError(
                "Bảo mật Dữ liệu: Toàn bộ tập dữ liệu bị cô lập do chứa mã độc hoặc bom dữ liệu!"
            )

        if la_dataset_obj:
            du_lieu.train = items_cuoi
            return du_lieu, bao_cao_gop
        return items_cuoi, bao_cao_gop


# =====================================================================
# 3. CỔNG KIỂM ĐỊNH CHẤT LƯỢNG & AN TOÀN ĐẦU VÀO (INPUT QUALITY GATE)
# =====================================================================


class CongChatLuongDauVao:
    """
    Cổng kiểm định chất lượng & an toàn đầu vào toàn diện (Input Quality & Integrity Gate):
    - Loại bỏ ký tự điều khiển độc hại: Null-byte (`\\x00`) và **Trojan Source Bidi Override** (`U+202A`..`U+202E`, `U+2066`..`U+2069`).
    - Đánh giá ma trận số `(X, y)`: phát hiện `NaN`/`Inf`, cột hằng vô dụng, ngoại lai cực đoan MAD Z-Score,
      và phát hiện **Rò rỉ Dữ liệu Trùng lặp (Data Leakage)** giữa `X_train` và `X_test`.
    - Chấm **Điểm Sức Khỏe Dữ Liệu (`0.0` – `100.0`)**.
    """

    KY_TU_BIDI_TROJAN = frozenset(
        "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\x00"
    )

    @classmethod
    def loai_bo_unicode_trojan(cls, van_ban: str) -> Tuple[str, bool]:
        """Tước bỏ ký tự điều khiển hướng chữ Bidi Trojan (CVE-2021-42574) và Null-byte."""
        if not van_ban:
            return van_ban, False
        co_trojan = any(ch in cls.KY_TU_BIDI_TROJAN for ch in van_ban)
        if not co_trojan:
            return van_ban, False
        sach = "".join(ch for ch in van_ban if ch not in cls.KY_TU_BIDI_TROJAN)
        return sach, True

    @classmethod
    def kiem_tra_mang_so(
        cls,
        X: Any,
        y: Optional[Any] = None,
        X_test: Optional[Any] = None,
        tu_dong_sua: bool = False,
        nguong_z_ngoai_lai: float = 8.0,
    ) -> Dict[str, Any]:
        """Kiểm định chất lượng ma trận đặc trưng số và phát hiện Data Leakage."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples, n_features = X_arr.shape
        canh_bao: List[str] = []
        diem = 100.0

        so_nan_inf = int(np.sum(~np.isfinite(X_arr)))
        if so_nan_inf > 0:
            ty_le_loi = so_nan_inf / max(1, X_arr.size)
            diem -= min(50.0, ty_le_loi * 200.0 + 15.0)
            canh_bao.append(f"Phát hiện {so_nan_inf} giá trị NaN/Inf trong dữ liệu đầu vào.")

        # Thay tạm NaN/Inf bằng trung vị bền vững của cột để không bóp méo phương sai & ngoại lai
        X_finite_only = np.where(np.isfinite(X_arr), X_arr, np.nan)
        with np.errstate(invalid="ignore"):
            col_med_tam = np.nanmedian(X_finite_only, axis=0)
            stds_finite = np.nanstd(X_finite_only, axis=0)
        col_med_tam = np.where(np.isfinite(col_med_tam), col_med_tam, 0.0)
        X_huu_han = X_arr.copy()
        vi_tri_loi = ~np.isfinite(X_huu_han)
        if np.any(vi_tri_loi):
            r_idx, c_idx = np.where(vi_tri_loi)
            X_huu_han[r_idx, c_idx] = np.take(col_med_tam, c_idx)

        # Cột có phương sai bằng 0 trên các phần tử hữu hạn (không mang thông tin)
        so_cot_chet = int(np.sum(~np.isfinite(stds_finite) | (stds_finite < 1e-12))) if n_samples > 1 else 0
        if so_cot_chet > 0:
            diem -= min(15.0, (so_cot_chet / max(1, n_features)) * 20.0)
            canh_bao.append(f"Có {so_cot_chet}/{n_features} cột đặc trưng không đổi (phương sai = 0).")

        # Điểm ngoại lai cực đoan (Adversarial / Corrupted Feature Spike)
        med = np.median(X_huu_han, axis=0, keepdims=True)
        mad = np.median(np.abs(X_huu_han - med), axis=0, keepdims=True) + 1e-9
        rz = np.abs(X_huu_han - med) / (1.4826 * mad)
        so_dong_ngoai_lai = int(np.sum(np.any(rz > nguong_z_ngoai_lai, axis=1))) if n_samples >= 6 else 0
        if so_dong_ngoai_lai > 0:
            diem -= min(20.0, (so_dong_ngoai_lai / max(1, n_samples)) * 60.0)
            canh_bao.append(f"Phát hiện {so_dong_ngoai_lai} mẫu chứa giá trị ngoại lai cực đoan (Robust-Z > {nguong_z_ngoai_lai}).")

        # Kiểm tra rò rỉ dữ liệu trùng lặp giữa X_train và X_test
        so_dong_ro_ri = 0
        if X_test is not None:
            Xt_arr = np.asarray(X_test, dtype=np.float64)
            if Xt_arr.ndim == 1:
                Xt_arr = Xt_arr.reshape(-1, 1)
            tap_bam_train = {
                hashlib.blake2b(np.round(row, 6).tobytes(), digest_size=8).digest()
                for row in X_huu_han
            }
            for row in np.nan_to_num(Xt_arr, nan=0.0, posinf=0.0, neginf=0.0):
                if hashlib.blake2b(np.round(row, 6).tobytes(), digest_size=8).digest() in tap_bam_train:
                    so_dong_ro_ri += 1
            if so_dong_ro_ri > 0:
                diem -= min(25.0, (so_dong_ro_ri / max(1, len(Xt_arr))) * 50.0 + 5.0)
                canh_bao.append(
                    f"Cảnh báo Data Leakage: {so_dong_ro_ri}/{len(Xt_arr)} mẫu tập Test trùng hệt tập Train!"
                )

        if y is not None and len(np.asarray(y)) != n_samples:
            diem = 0.0
            canh_bao.append(f"Lệch kích thước nghiêm trọng: X có {n_samples} mẫu nhưng y có {len(np.asarray(y))} mẫu.")

        ket_qua: Dict[str, Any] = {
            "hop_le": so_nan_inf == 0 and (y is None or len(np.asarray(y)) == n_samples),
            "diem_chat_luong": round(max(0.0, diem), 2),
            "so_mau": n_samples,
            "so_dac_trung": n_features,
            "so_nan_inf": so_nan_inf,
            "so_cot_phuong_sai_khong": so_cot_chet,
            "so_mau_ngoai_lai": so_dong_ngoai_lai,
            "so_mau_ro_ri_train_test": so_dong_ro_ri,
            "canh_bao": canh_bao,
        }

        if tu_dong_sua:
            X_sua = X_arr.copy()
            # Thay NaN/Inf bằng trung vị cột bền vững
            col_medians = np.nanmedian(np.where(np.isfinite(X_sua), X_sua, np.nan), axis=0)
            col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
            bad_mask = ~np.isfinite(X_sua)
            if np.any(bad_mask):
                inds = np.where(bad_mask)
                X_sua[inds] = np.take(col_medians, inds[1])
            ket_qua["X_da_lam_sach"] = X_sua

        return ket_qua


# =====================================================================
# 4. CỔNG KIỂM ĐỊNH AN TOÀN & CHẤT LƯỢNG ĐẦU RA (OUTPUT SAFETY & QUALITY GATE)
# =====================================================================


class CongAnToanDauRa:
    """
    Cổng bảo vệ an toàn & nâng cao chất lượng đầu ra của LLM / VLM / RAG / Agent:
    - Chặn tấn công **Markdown Image / Link Exfiltration Beacon** (`![...](https://attacker.com/leak?q=...)`).
    - Cảnh báo / chặn mã lệnh hệ thống phá hoại sinh ra ở đầu ra (`rm -rf /`, Fork Bomb, Reverse Shell, `DROP DATABASE`).
    - Tự động phát hiện và cắt gọt **Vòng lặp Suy thoái (Degeneration Repetition Loop)**.
    - Chấm điểm **Độ Bám sát Ngữ cảnh (`Groundedness Score`)** chống ảo giác cho `RAGPipeline`.
    """

    MAU_MARKDOWN_BEACON = re.compile(
        r"!\[[^\]]*\]\(\s*https?://[^)\s]+\s*\)", flags=re.IGNORECASE
    )
    MAU_URL_RO_RI = re.compile(
        r"https?://[^\s)\"']+[?&](?:api_key|secret|token|password|passwd|exfil|steal|leak)=[^\s)\"']+",
        flags=re.IGNORECASE,
    )
    MAU_LENH_PHA_HOAI: Tuple[Tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(r"\brm\s+-[rRfF]{1,4}\s+(?:/[^\s;|&\"'`)]*|/|\*|~)(?=\s|$|[;|&\"'`)])"),
            "Lệnh xóa hủy diệt hệ thống tệp (rm -rf /)",
        ),
        (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "Mã độc Fork Bomb làm treo hệ điều hành"),
        (re.compile(r"/dev/tcp/[0-9a-zA-Z._-]+/\d+"), "Kết nối Reverse Shell trái phép qua /dev/tcp"),
        (re.compile(r"\b(?:DROP\s+DATABASE|TRUNCATE\s+TABLE)\b", re.IGNORECASE), "Câu lệnh SQL hủy diệt dữ liệu"),
        (re.compile(r"\bmkfs\.[a-z0-9]+\s+/dev/[^\s;|&\"'`)]*", re.IGNORECASE), "Lệnh định dạng phá hủy ổ đĩa (mkfs)"),
    )

    @classmethod
    def khu_lap_suy_thoai(cls, van_ban: str, so_lan_lap_toi_da: int = 3) -> Tuple[str, bool]:
        """
        Phát hiện và gọt bỏ hiện tượng LLM bị kẹt vòng lặp lặp lại cùng một cụm câu/cụm từ nhiều lần liên tiếp.
        """
        if not van_ban or len(van_ban) < 24:
            return van_ban, False

        words = van_ban.split()
        if len(words) < 12:
            return van_ban, False

        da_cat = False
        # Kiểm tra cụm n-gram (n từ 2 đến 8) lặp liên tiếp quá `so_lan_lap_toi_da` lần
        for n in range(6, 1, -1):
            i = 0
            ket_qua_tu: List[str] = []
            while i < len(words):
                cum = words[i : i + n]
                if len(cum) < n:
                    ket_qua_tu.extend(cum)
                    break
                so_lap = 1
                while i + (so_lap + 1) * n <= len(words) and words[i + so_lap * n : i + (so_lap + 1) * n] == cum:
                    so_lap += 1
                if so_lap > so_lan_lap_toi_da:
                    da_cat = True
                    ket_qua_tu.extend(cum)
                    i += so_lap * n
                else:
                    ket_qua_tu.append(words[i])
                    i += 1
            words = ket_qua_tu

        return (" ".join(words) if da_cat else van_ban), da_cat

    @classmethod
    def kiem_tra_va_lam_sach(
        cls,
        van_ban: str,
        chan_lenh_pha_hoai: bool = True,
        an_pii: bool = True,
        nguon_ngu_canh: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Kiểm định toàn diện đầu ra: tước bỏ beacon rò rỉ dữ liệu, chặn lệnh phá hoại,
        ẩn PII/Secrets, khử lặp suy thoái và chấm điểm chất lượng + độ bám sát nguồn.
        """
        if not isinstance(van_ban, str):
            van_ban = str(van_ban or "")

        vi_pham: List[str] = []
        ket_qua = van_ban

        # 1. Chặn Markdown Image Beacon & URL Exfiltration
        if cls.MAU_MARKDOWN_BEACON.search(ket_qua):
            vi_pham.append("Chặn Markdown Image Exfiltration Beacon (nguy cơ rò rỉ dữ liệu ra máy chủ ngoài).")
            ket_qua = cls.MAU_MARKDOWN_BEACON.sub("[BEACON HÌNH ẢNH RÒ RỈ ĐÃ CHẶN]", ket_qua)

        if cls.MAU_URL_RO_RI.search(ket_qua):
            vi_pham.append("Chặn URL chứa tham số rò rỉ khóa bí mật ra bên ngoài.")
            ket_qua = cls.MAU_URL_RO_RI.sub("[URL RÒ RỈ ĐÃ CHẶN]", ket_qua)

        # 2. Chặn lệnh hệ thống / SQL phá hoại
        if chan_lenh_pha_hoai:
            for pattern, mo_ta in cls.MAU_LENH_PHA_HOAI:
                if pattern.search(ket_qua):
                    vi_pham.append(mo_ta)
                    ket_qua = pattern.sub("[LỆNH NGUY HIỂM ĐÃ CHẶN]", ket_qua)

        # 3. Ẩn thông tin nhạy cảm PII & Khóa bí mật
        if an_pii:
            ket_qua = DataSanitizer.lam_sach(ket_qua, loc_bi_mat=True)

        # 4. Sửa lỗi lặp từ suy thoái (Degeneration Loop)
        ket_qua, bi_lap = cls.khu_lap_suy_thoai(ket_qua)

        # 5. Đánh giá độ bám sát ngữ cảnh RAG (nếu có truyền nguồn)
        thong_tin_bam_sat: Optional[Dict[str, Any]] = None
        if nguon_ngu_canh is not None:
            thong_tin_bam_sat = cls.kiem_tra_ao_giac(ket_qua, nguon_ngu_canh)

        diem_chat_luong = 100.0
        if not ket_qua.strip():
            diem_chat_luong = 0.0
        else:
            diem_chat_luong -= 25.0 * len(vi_pham)
            if bi_lap:
                diem_chat_luong -= 15.0
            if thong_tin_bam_sat and thong_tin_bam_sat["nghi_ngo_ao_giac"]:
                diem_chat_luong -= 20.0

        out: Dict[str, Any] = {
            "an_toan": len(vi_pham) == 0,
            "van_ban_sach": ket_qua,
            "vi_pham": vi_pham,
            "da_khu_lap_suy_thoai": bi_lap,
            "diem_chat_luong_dau_ra": round(max(0.0, diem_chat_luong), 2),
        }
        if thong_tin_bam_sat is not None:
            out.update(thong_tin_bam_sat)
        return out

    @staticmethod
    def kiem_tra_ao_giac(tra_loi: str, nguon_ngu_canh: List[Any]) -> Dict[str, Any]:
        """
        Đo độ bám sát ngữ cảnh (Groundedness Score) giữa câu trả lời và các đoạn tài liệu nguồn RAG
        để cảnh báo nguy cơ ảo giác (Hallucination).
        """
        tu_bo_qua = {
            "là", "và", "của", "có", "cho", "trong", "với", "được", "các", "những",
            "một", "này", "đó", "khi", "theo", "từ", "trên", "về", "để", "như",
            "the", "and", "for", "with", "that", "this", "from", "are", "was",
        }
        tu_tra_loi = [
            w for w in re.findall(r"[a-zA-Z0-9À-ỹ]{2,}", tra_loi.lower()) if w not in tu_bo_qua
        ]
        if not tu_tra_loi:
            return {"diem_bam_sat_nguon": 1.0, "nghi_ngo_ao_giac": False}

        van_ban_nguon_gop = []
        for ng in nguon_ngu_canh:
            if isinstance(ng, str):
                van_ban_nguon_gop.append(ng)
            elif isinstance(ng, dict):
                nd = ng.get("noi_dung") or ng.get("van_ban") or ng.get("text") or ng.get("metadata", {}).get("noi_dung", "")
                van_ban_nguon_gop.append(str(nd))

        tap_tu_nguon = set(re.findall(r"[a-zA-Z0-9À-ỹ]{2,}", " ".join(van_ban_nguon_gop).lower()))
        if not tap_tu_nguon:
            return {"diem_bam_sat_nguon": 0.0, "nghi_ngo_ao_giac": True}

        so_tu_khop = sum(1 for w in tu_tra_loi if w in tap_tu_nguon)
        diem_bam_sat = round(so_tu_khop / len(tu_tra_loi), 4)
        return {
            "diem_bam_sat_nguon": diem_bam_sat,
            "nghi_ngo_ao_giac": diem_bam_sat < 0.25 and len(tu_tra_loi) >= 6,
        }


# English Aliases đồng bộ hệ sinh thái
ImageSecurityShield = KhienBaoVeHinhAnh
DatasetDefenseShield = PhongThuDuLieu
StreamingPoisoningMonitor = BoGiamSatTichTuDuLieu
InputQualityGate = CongChatLuongDauVao
OutputSafetyQualityGate = CongAnToanDauRa
