"""BoDocTaiLieuPhapLy - Trình Trích Xuất & Làm Sạch Tài Liệu Hành Chính/Pháp Lý Đa Định Dạng.

Hỗ trợ đọc và chuẩn hóa bố cục văn bản tiếng Việt từ:
- Tệp văn bản thuần (`.txt`, `.md`)
- Tệp PDF (`.pdf` qua `PyPDFLoader` / `pypdf`)
- Ảnh scan tài liệu hành chính (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` qua `TrichXuatVanBanAnh`)
- Dữ liệu văn bản có cấu trúc (`.json`, `.jsonl`)
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc
from vietnamese_ai.rag.document_loaders import PyPDFLoader, TextLoader
from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc


class BoDocTaiLieuPhapLy:
    """Bộ đọc tài liệu hành chính/pháp lý đa định dạng (`.txt`, `.md`, `.docx`, `.pdf` kèm bảng biểu, ảnh OCR, `.json`/`.jsonl`)."""

    DINH_DANG_VAN_BAN = {".txt", ".md", ".rst"}
    DINH_DANG_DOCX = {".docx"}
    DINH_DANG_PDF = {".pdf"}
    DINH_DANG_ANH = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    DINH_DANG_JSON = {".json", ".jsonl"}

    _RE_SO_TRANG = re.compile(
        r"^\s*(?:Trang\s+\d+(?:\s*/\s*\d+)?|[-–—]\s*\d+\s*[-–—]|\d+\s*/\s*\d+)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    _RE_NHIEU_DONG_TRONG = re.compile(r"\n{3,}")

    def __init__(self, bat_bao_ve_ocr: bool = True, bat_ocr_pdf_scan: bool = True) -> None:
        self.bat_bao_ve_ocr = bat_bao_ve_ocr
        self.bat_ocr_pdf_scan = bat_ocr_pdf_scan
        self._tokenizer = BoTachTuTiengVietSieuToc()

    @staticmethod
    def loc_dau_do_anh_numpy(img_rgb: np.ndarray) -> np.ndarray:
        """Khử mực dấu tròn đỏ hành chính Việt Nam trên ảnh RGB bằng ngưỡng chênh kênh màu thuần NumPy để tăng độ chính xác OCR."""
        arr = np.asarray(img_rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return arr.copy()
        ket_qua = arr.copy()
        r = ket_qua[:, :, 0].astype(np.int16)
        g = ket_qua[:, :, 1].astype(np.int16)
        b = ket_qua[:, :, 2].astype(np.int16)
        mat_na_dau_do = (r >= 135) & ((r - g) >= 45) & ((r - b) >= 45)
        ket_qua[mat_na_dau_do, 0] = 255
        ket_qua[mat_na_dau_do, 1] = 255
        ket_qua[mat_na_dau_do, 2] = 255
        return ket_qua

    def lam_sach_van_ban_phap_ly(self, van_ban: str) -> str:
        """Loại bỏ số trang dư thừa, ký tự ngắt trang và chuẩn hóa dấu thanh Unicode NFC giữ cấu trúc dòng."""
        if not van_ban:
            return ""
        text = van_ban.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
        text = self._RE_SO_TRANG.sub("", text)
        dong_sach: List[str] = []
        for dong in text.split("\n"):
            dong_clean = self._tokenizer.chuan_hoa_dau_thanh(dong)
            if dong_clean:
                dong_sach.append(dong_clean)
        ket_qua = "\n".join(dong_sach)
        return self._RE_NHIEU_DONG_TRONG.sub("\n\n", ket_qua).strip()

    def _doc_json(self, duong_dan: Path) -> str:
        """Trích xuất văn bản từ tệp `.json` hoặc `.jsonl` chứa trường nội dung pháp lý."""
        noi_dung_goc = duong_dan.read_text(encoding="utf-8")
        cac_doan: List[str] = []

        def _lay_chuoi_tu_ban_ghi(obj: Any) -> None:
            if isinstance(obj, str):
                if obj.strip():
                    cac_doan.append(obj.strip())
            elif isinstance(obj, dict):
                for khoa in ("tieu_de", "title", "noi_dung", "content", "text", "van_ban"):
                    val = obj.get(khoa)
                    if isinstance(val, str) and val.strip():
                        cac_doan.append(val.strip())
                if not any(k in obj for k in ("tieu_de", "title", "noi_dung", "content", "text", "van_ban")):
                    for val in obj.values():
                        if isinstance(val, str) and val.strip():
                            cac_doan.append(val.strip())
            elif isinstance(obj, list):
                for item in obj:
                    _lay_chuoi_tu_ban_ghi(item)

        if duong_dan.suffix.lower() == ".jsonl":
            for dong in noi_dung_goc.splitlines():
                dong_s = dong.strip()
                if dong_s:
                    _lay_chuoi_tu_ban_ghi(json.loads(dong_s))
        else:
            _lay_chuoi_tu_ban_ghi(json.loads(noi_dung_goc))

        return "\n".join(cac_doan)

    @staticmethod
    def _dinh_dang_bang_markdown(bang: List[List[str]]) -> str:
        """Chuyển ma trận ô bảng hành chính thành bảng Markdown bảo toàn cấu trúc cột/hàng."""
        hang_sach = [
            [str(cell or "").replace("\n", " ").strip() for cell in row]
            for row in bang
            if any(str(cell or "").strip() for cell in row)
        ]
        if not hang_sach:
            return ""
        so_cot = max(len(r) for r in hang_sach)
        dong_md: List[str] = []
        for idx, row in enumerate(hang_sach):
            padded = row + [""] * (so_cot - len(row))
            dong_md.append("| " + " | ".join(padded) + " |")
            if idx == 0:
                dong_md.append("| " + " | ".join(["---"] * so_cot) + " |")
        return "\n".join(dong_md)

    def _doc_docx(self, duong_dan: Path) -> Tuple[str, int]:
        """Đọc đoạn văn và bảng biểu từ tệp `.docx` theo đúng thứ tự xuất hiện kèm kiểm tra chống Zip Bomb."""
        max_xml_bytes = 50 * 1024 * 1024
        max_total_bytes = 100 * 1024 * 1024
        with zipfile.ZipFile(str(duong_dan), "r") as zf_check:
            if "word/document.xml" not in zf_check.namelist():
                raise ValueError(f"Tệp DOCX không hợp lệ (thiếu word/document.xml): {duong_dan}")
            doc_info = zf_check.getinfo("word/document.xml")
            tong_giai_nen = sum(info.file_size for info in zf_check.infolist())
            if doc_info.file_size > max_xml_bytes or tong_giai_nen > max_total_bytes:
                raise ValueError(f"Tệp DOCX vượt giới hạn giải nén an toàn (nghi ngờ Zip Bomb): {duong_dan}")

        so_bang = 0
        docx_mod = QuanLyPhuThuoc.nhap_tuy_chon("docx", nhom_extra="legal-rag", tinh_nang="Đọc tài liệu DOCX")
        if docx_mod is not None:
            try:
                from docx.table import Table
                from docx.text.paragraph import Paragraph

                tai_lieu = docx_mod.Document(str(duong_dan))
                phan_doan: List[str] = []
                for child in tai_lieu.element.body.iterchildren():
                    tag = str(child.tag)
                    if tag.endswith("}p"):
                        p = Paragraph(child, tai_lieu)
                        if p.text and p.text.strip():
                            phan_doan.append(p.text.strip())
                    elif tag.endswith("}tbl"):
                        tbl = Table(child, tai_lieu)
                        ma_tran = [[c.text.strip() for c in row.cells] for row in tbl.rows]
                        md = self._dinh_dang_bang_markdown(ma_tran)
                        if md:
                            phan_doan.append(md)
                            so_bang += 1
                return "\n".join(phan_doan), so_bang
            except Exception:
                so_bang = 0

        return self._doc_docx_ooxml(duong_dan)

    def _doc_docx_ooxml(self, duong_dan: Path) -> Tuple[str, int]:
        """Phân tích trực tiếp `word/document.xml` chuẩn OOXML hỗ trợ gộp ô ngang (`w:gridSpan`) và gộp ô dọc (`w:vMerge`)."""
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        phan_doan_xml: List[str] = []
        so_bang = 0
        with zipfile.ZipFile(str(duong_dan), "r") as zf:
            xml_data = zf.read("word/document.xml")
        root = ET.fromstring(xml_data)
        body = root.find(f"{ns}body")
        if body is None:
            return "", 0

        for child in list(body):
            if child.tag == f"{ns}p":
                text_p = "".join(t.text or "" for t in child.iter(f"{ns}t")).strip()
                if text_p:
                    phan_doan_xml.append(text_p)
            elif child.tag == f"{ns}tbl":
                ma_tran_xml: List[List[str]] = []
                for tr in child.findall(f"{ns}tr"):
                    hang: List[str] = []
                    cot_idx = 0
                    for tc in tr.findall(f"{ns}tc"):
                        tc_pr = tc.find(f"{ns}tcPr")
                        grid_span = 1
                        vmerge_continue = False
                        if tc_pr is not None:
                            gs_el = tc_pr.find(f"{ns}gridSpan")
                            if gs_el is not None:
                                raw_span = gs_el.attrib.get(f"{ns}val") or gs_el.attrib.get("val") or "1"
                                try:
                                    grid_span = max(1, int(raw_span))
                                except ValueError:
                                    grid_span = 1
                            vm_el = tc_pr.find(f"{ns}vMerge")
                            if vm_el is not None:
                                vm_val = (vm_el.attrib.get(f"{ns}val") or vm_el.attrib.get("val") or "").strip().lower()
                                if vm_val != "restart":
                                    vmerge_continue = True
                        cell_txt = " ".join(
                            "".join(t.text or "" for t in p.iter(f"{ns}t")).strip()
                            for p in tc.findall(f"{ns}p")
                        ).strip()
                        if vmerge_continue and not cell_txt and ma_tran_xml:
                            prev_row = ma_tran_xml[-1]
                            if cot_idx < len(prev_row):
                                cell_txt = prev_row[cot_idx]
                        hang.append(cell_txt)
                        for _ in range(grid_span - 1):
                            hang.append("")
                        cot_idx += grid_span
                    ma_tran_xml.append(hang)
                md = self._dinh_dang_bang_markdown(ma_tran_xml)
                if md:
                    phan_doan_xml.append(md)
                    so_bang += 1
        return "\n".join(phan_doan_xml), so_bang

    def _doc_pdf_scan_ocr(self, duong_dan: Path, toi_da_trang: int = 20) -> Tuple[str, int]:
        """Chuyển trang PDF scan sang ảnh, khử dấu đỏ hành chính và chạy OCR tiếng Việt an toàn."""
        import tempfile

        from vietnamese_ai.vision.ocr import TrichXuatVanBanAnh

        pil_mod = QuanLyPhuThuoc.nhap_tuy_chon("PIL", nhom_extra="vision", tinh_nang="OCR PDF scan")
        if pil_mod is None:
            return "", 0

        anh_trang: List[Any] = []
        pdfium = QuanLyPhuThuoc.nhap_tuy_chon("pypdfium2", nhom_extra="legal-rag", tinh_nang="Render trang PDF scan")
        if pdfium is not None:
            try:
                pdf_doc = pdfium.PdfDocument(str(duong_dan))
                so_luong = min(len(pdf_doc), max(1, int(toi_da_trang)))
                for idx in range(so_luong):
                    page = pdf_doc[idx]
                    bitmap = page.render(scale=2.0)
                    anh_trang.append(bitmap.to_pil().convert("RGB"))
            except Exception:
                anh_trang = []

        if not anh_trang:
            pdf2img = QuanLyPhuThuoc.nhap_tuy_chon("pdf2image", nhom_extra="legal-rag", tinh_nang="Render trang PDF scan")
            if pdf2img is not None:
                try:
                    anh_trang = [
                        im.convert("RGB")
                        for im in pdf2img.convert_from_path(str(duong_dan), first_page=1, last_page=toi_da_trang)
                    ]
                except Exception:
                    anh_trang = []

        if not anh_trang:
            return "", 0

        from PIL import Image

        ocr = TrichXuatVanBanAnh(bat_bao_ve_anh=self.bat_bao_ve_ocr)
        cac_doan: List[str] = []
        with tempfile.TemporaryDirectory(prefix="vai_pdf_ocr_") as tmp_dir:
            for idx, pil_img in enumerate(anh_trang):
                arr_sach = self.loc_dau_do_anh_numpy(np.asarray(pil_img, dtype=np.uint8))
                img_path = Path(tmp_dir) / f"page_{idx + 1}.png"
                Image.fromarray(arr_sach).save(img_path, format="PNG")
                kq = ocr.doc_van_ban(img_path)
                txt = str(kq.get("van_ban", "")).strip()
                if txt:
                    cac_doan.append(txt)
        return "\n\n".join(cac_doan), len(anh_trang)

    def _doc_pdf_co_bang(self, duong_dan: Path) -> Tuple[str, int, int]:
        """Đọc văn bản và bảng biểu từ PDF bằng `pdfplumber` nếu cài, dự phòng `PyPDFLoader` (`pypdf`)."""
        pdfplumber_mod = QuanLyPhuThuoc.nhap_tuy_chon(
            "pdfplumber", nhom_extra="legal-rag", tinh_nang="Trích xuất bảng PDF"
        )
        if pdfplumber_mod is not None:
            try:
                phan_trang: List[str] = []
                so_bang = 0
                with pdfplumber_mod.open(str(duong_dan)) as pdf:
                    so_trang = len(pdf.pages)
                    for page in pdf.pages:
                        txt = (page.extract_text() or "").strip()
                        if txt:
                            phan_trang.append(txt)
                        for tbl in page.extract_tables() or []:
                            md = self._dinh_dang_bang_markdown(tbl)
                            if md:
                                phan_trang.append(md)
                                so_bang += 1
                return "\n".join(phan_trang), max(1, so_trang), so_bang
            except Exception:
                pass
        docs = PyPDFLoader(str(duong_dan)).load()
        return "\n".join(d.page_content for d in docs), max(1, len(docs)), 0

    def doc_tep(self, duong_dan: Union[str, Path]) -> Dict[str, Any]:
        """
        Đọc tài liệu pháp lý/hành chính từ tệp cục bộ theo đúng định dạng.

        Tham số:
            duong_dan: Đường dẫn tới tệp `.txt`, `.md`, `.docx`, `.pdf`, ảnh scan hoặc `.json`/`.jsonl`.

        Trả về:
            Dict gồm `van_ban`, `dinh_dang`, `duong_dan`, `so_trang` và `metadata`.
        """
        path = Path(duong_dan)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp tài liệu pháp lý: {path}")

        ext = path.suffix.lower()
        so_trang = 1
        metadata: Dict[str, Any] = {"ten_tep": path.name, "duong_dan": str(path)}

        if ext in self.DINH_DANG_VAN_BAN:
            docs = TextLoader(str(path)).load()
            van_ban_tho = "\n".join(d.page_content for d in docs)
            dinh_dang = "van_ban"
        elif ext in self.DINH_DANG_DOCX:
            van_ban_tho, so_bang = self._doc_docx(path)
            dinh_dang = "docx"
            metadata["so_bang"] = so_bang
        elif ext in self.DINH_DANG_PDF:
            van_ban_tho, so_trang, so_bang = self._doc_pdf_co_bang(path)
            dinh_dang = "pdf"
            metadata["so_bang"] = so_bang
            la_pdf_scan = (len(van_ban_tho.strip()) / max(1, so_trang)) < 25.0
            metadata["la_pdf_scan"] = la_pdf_scan
            metadata["da_chay_ocr_pdf"] = False
            if la_pdf_scan and self.bat_ocr_pdf_scan:
                ocr_txt, so_trang_ocr = self._doc_pdf_scan_ocr(path)
                if ocr_txt.strip():
                    van_ban_tho = ocr_txt
                    so_trang = max(so_trang, so_trang_ocr)
                    dinh_dang = "pdf_scan_ocr"
                    metadata["da_chay_ocr_pdf"] = True
        elif ext in self.DINH_DANG_ANH:
            from vietnamese_ai.vision.ocr import TrichXuatVanBanAnh

            ocr = TrichXuatVanBanAnh(bat_bao_ve_anh=self.bat_bao_ve_ocr)
            kq_ocr = ocr.doc_van_ban(path)
            van_ban_tho = str(kq_ocr.get("van_ban", ""))
            dinh_dang = "anh_ocr"
            metadata["so_dong_ocr"] = kq_ocr.get("so_dong", 0)
            metadata["nguong_otsu"] = kq_ocr.get("nguong_otsu")
        elif ext in self.DINH_DANG_JSON:
            van_ban_tho = self._doc_json(path)
            dinh_dang = "json"
        else:
            raise ValueError(
                f"Định dạng tệp '{ext}' chưa được hỗ trợ trong BoDocTaiLieuPhapLy. "
                "Hỗ trợ: .txt, .md, .docx, .pdf, .png, .jpg, .jpeg, .bmp, .tif, .tiff, .json, .jsonl."
            )

        van_ban_sach = self.lam_sach_van_ban_phap_ly(van_ban_tho)
        return {
            "van_ban": van_ban_sach,
            "dinh_dang": dinh_dang,
            "duong_dan": str(path),
            "so_trang": so_trang,
            "metadata": metadata,
        }


LegalDocumentParser = BoDocTaiLieuPhapLy

__all__ = [
    "BoDocTaiLieuPhapLy",
    "LegalDocumentParser",
]
