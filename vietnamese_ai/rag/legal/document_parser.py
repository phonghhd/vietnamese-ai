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

    def __init__(self, bat_bao_ve_ocr: bool = True) -> None:
        self.bat_bao_ve_ocr = bat_bao_ve_ocr
        self._tokenizer = BoTachTuTiengVietSieuToc()

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
        """Đọc đoạn văn và bảng biểu từ tệp `.docx` (ưu tiên `python-docx`, dự phòng giải nén OOXML thuần Python)."""
        so_bang = 0
        docx_mod = QuanLyPhuThuoc.nhap_tuy_chon("docx", nhom_extra="legal-rag", tinh_nang="Đọc tài liệu DOCX")
        if docx_mod is not None:
            try:
                tai_lieu = docx_mod.Document(str(duong_dan))
                phan_doan: List[str] = []
                for p in tai_lieu.paragraphs:
                    if p.text and p.text.strip():
                        phan_doan.append(p.text.strip())
                for tbl in tai_lieu.tables:
                    ma_tran = [[c.text.strip() for c in row.cells] for row in tbl.rows]
                    md = self._dinh_dang_bang_markdown(ma_tran)
                    if md:
                        phan_doan.append(md)
                        so_bang += 1
                return "\n".join(phan_doan), so_bang
            except Exception:
                pass

        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        phan_doan_xml: List[str] = []
        with zipfile.ZipFile(str(duong_dan), "r") as zf:
            if "word/document.xml" not in zf.namelist():
                raise ValueError(f"Tệp DOCX không hợp lệ (thiếu word/document.xml): {duong_dan}")
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
                    for tc in tr.findall(f"{ns}tc"):
                        cell_txt = " ".join(
                            "".join(t.text or "" for t in p.iter(f"{ns}t")).strip()
                            for p in tc.findall(f"{ns}p")
                        ).strip()
                        hang.append(cell_txt)
                    ma_tran_xml.append(hang)
                md = self._dinh_dang_bang_markdown(ma_tran_xml)
                if md:
                    phan_doan_xml.append(md)
                    so_bang += 1
        return "\n".join(phan_doan_xml), so_bang

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
