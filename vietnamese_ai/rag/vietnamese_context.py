"""Bộ Tối Ưu Ngữ Cảnh Hành Chính & Pháp Lý Việt Nam (Vietnamese Administrative & Legal RAG) - Vietnamese AI Framework.

Tối ưu hóa chuyên sâu cho tài liệu hành chính, thông tư, nghị định, luật pháp, công văn, tờ trình và hợp đồng Việt Nam.
Giải quyết triệt để lỗi phân mảnh câu (chunk truncation) của LangChain/LlamaIndex trên văn bản tiếng Việt.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from vietnamese_ai.nlp.native_tokenizer import BoNhungVanBanSieuToc, BoTachTuTiengVietSieuToc


class BoToiUuNguCanhHanhChinh:
    """
    Bộ Tối Ưu Ngữ Cảnh Hành Chính & Pháp Lý Việt Nam (Vietnamese Administrative & Legal RAG Optimizer):
    - Nhận diện nhiều loại văn bản: Hiến pháp, Bộ luật, Luật, Nghị định, Thông tư, Quyết định, Chỉ thị,
      Nghị quyết, Công văn, Tờ trình, Quy chế, Hợp đồng, Biên bản.
    - Tự động bóc tách cấu trúc phân cấp: Chương -> Mục -> Điều -> Khoản -> Điểm, Phụ lục và đề mục hành chính.
    - Kỹ thuật Hierarchy-Preserving Chunking: Giữ toàn vẹn văn cảnh tài liệu mẹ kèm trích dẫn số hiệu.
    - Hỗ trợ nạp trực tiếp từ tệp phức tạp (`.txt`, `.md`, `.pdf`, ảnh scan OCR, `.json`) qua `BoDocTaiLieuPhapLy`
      và sinh prompt pháp lý có trích dẫn qua `BoMauPromptPhapLy`.
    """

    RE_CHUONG = re.compile(r"^\s*(Chương\s+[IVXLCDM\d]+[.:\s]*[^\n]*)", re.MULTILINE | re.IGNORECASE)
    RE_MUC = re.compile(r"^\s*(Mục\s+[IVXLCDM\d]+[.:\s]*[^\n]*)", re.MULTILINE | re.IGNORECASE)
    RE_DIEU = re.compile(r"^\s*(Điều\s+\d+[a-zđ]?[.:\s]*[^\n]*)", re.MULTILINE | re.IGNORECASE)
    RE_PHU_LUC = re.compile(r"^\s*(Phụ\s+lục\s+[IVXLCDM\dA-Z]*[.:\s]*[^\n]*)", re.MULTILINE | re.IGNORECASE)
    RE_MUC_LA_MA = re.compile(r"^\s*([IVXLCDM]+\.\s+[^\n]+)", re.MULTILINE)
    RE_KHOAN = re.compile(r"^\s*(\d+\.\s+[^\n]*)", re.MULTILINE)
    RE_DIEM = re.compile(r"^\s*([a-zđ]\)\s+[^\n]*)", re.MULTILINE | re.IGNORECASE)
    RE_CAN_CU = re.compile(r"^\s*(Căn cứ\s+[^\n;.]+)", re.MULTILINE | re.IGNORECASE)
    RE_SO_HIEU = re.compile(
        r"((?:Hiến\s+pháp|Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Chỉ\s+thị|Nghị\s+quyết|Công\s+văn|Tờ\s+trình)"
        r"(?:\s+[^\n,;.]+?)?\s+số\s+[\d/]+[A-Za-z0-9Đđ\-/]+|"
        r"(?:Số\s*:\s*[\d/]+[A-Za-z0-9Đđ\-/]+)|"
        r"[\d]+/\d{4}/[A-Za-z0-9Đđ\-/]+|"
        r"[\d]+/[A-ZĐ0-9]+-[A-ZĐ0-9-]+)",
        re.IGNORECASE,
    )
    RE_NGAY_BAN_HANH = re.compile(
        r"(?:ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})",
        re.IGNORECASE,
    )
    RE_TRICH_YEU = re.compile(r"^\s*(?:V/v|Về việc)\s*:?\s*([^\n]+)", re.MULTILINE | re.IGNORECASE)
    RE_CO_QUAN = re.compile(
        r"^\s*((?:CHÍNH PHỦ|QUỐC HỘI|THỦ TƯỚNG CHÍNH PHỦ|ỦY BAN NHÂN DÂN|HỘI ĐỒNG NHÂN DÂN|"
        r"BỘ\s+[A-ZÀ-ỴĐ\s]+|SỞ\s+[A-ZÀ-ỴĐ\s]+|TÒA ÁN NHÂN DÂN[^\n]*|VIỆN KIỂM SÁT[^\n]*))\s*$",
        re.MULTILINE,
    )

    LOAI_VAN_BAN_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
        ("Hiến pháp", re.compile(r"\bHiến\s+pháp\b", re.IGNORECASE)),
        ("Bộ luật", re.compile(r"\bBộ\s+luật\b", re.IGNORECASE)),
        ("Luật", re.compile(r"\bLuật\b", re.IGNORECASE)),
        ("Nghị định", re.compile(r"\bNghị\s+định\b", re.IGNORECASE)),
        ("Thông tư", re.compile(r"\bThông\s+tư\b", re.IGNORECASE)),
        ("Quyết định", re.compile(r"\bQuyết\s+định\b", re.IGNORECASE)),
        ("Chỉ thị", re.compile(r"\bChỉ\s+thị\b", re.IGNORECASE)),
        ("Nghị quyết", re.compile(r"\bNghị\s+quyết\b", re.IGNORECASE)),
        ("Công văn", re.compile(r"\b(?:Công\s+văn|V/v\b|Kính\s+gửi\s*:)", re.IGNORECASE)),
        ("Tờ trình", re.compile(r"\bTờ\s+trình\b", re.IGNORECASE)),
        ("Quy chế", re.compile(r"\bQuy\s+chế\b", re.IGNORECASE)),
        ("Hợp đồng", re.compile(r"\bHợp\s+đồng\b", re.IGNORECASE)),
        ("Biên bản", re.compile(r"\bBiên\s+bản\b", re.IGNORECASE)),
    ]

    def __init__(
        self,
        embedder: Optional[BoNhungVanBanSieuToc] = None,
        bat_tuong_lua: bool = True,
    ) -> None:
        self.tokenizer = BoTachTuTiengVietSieuToc()
        self.embedder = embedder or BoNhungVanBanSieuToc(tokenizer=self.tokenizer)
        self.bat_tuong_lua = bat_tuong_lua
        if bat_tuong_lua:
            from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

            self.tuong_lua: Optional[TuongLuaBaoVeFramework] = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
        else:
            self.tuong_lua = None

    def phan_loai_van_ban(self, van_ban: str) -> str:
        """Nhận diện loại văn bản hành chính/pháp lý Việt Nam ưu tiên tiêu đề đầu văn bản và bỏ qua dòng Căn cứ."""
        cac_dong = [d.strip() for d in van_ban.splitlines() if d.strip()]
        dong_tieu_de = [d for d in cac_dong[:15] if not self.RE_CAN_CU.match(d)]

        tieu_de_patterns: List[Tuple[str, re.Pattern[str]]] = [
            ("Công văn", re.compile(r"^(?:CÔNG\s+VĂN\b|V/v\b|Về\s+việc\b|Kính\s+gửi\s*:)", re.IGNORECASE)),
            ("Hiến pháp", re.compile(r"^Hiến\s+pháp\b", re.IGNORECASE)),
            ("Bộ luật", re.compile(r"^Bộ\s+luật\b", re.IGNORECASE)),
            ("Luật", re.compile(r"^Luật\b", re.IGNORECASE)),
            ("Nghị định", re.compile(r"^Nghị\s+định\b", re.IGNORECASE)),
            ("Thông tư", re.compile(r"^Thông\s+tư\b", re.IGNORECASE)),
            ("Quyết định", re.compile(r"^Quyết\s+định\b", re.IGNORECASE)),
            ("Chỉ thị", re.compile(r"^Chỉ\s+thị\b", re.IGNORECASE)),
            ("Nghị quyết", re.compile(r"^Nghị\s+quyết\b", re.IGNORECASE)),
            ("Tờ trình", re.compile(r"^Tờ\s+trình\b", re.IGNORECASE)),
            ("Quy chế", re.compile(r"^Quy\s+chế\b", re.IGNORECASE)),
            ("Hợp đồng", re.compile(r"^Hợp\s+đồng\b", re.IGNORECASE)),
            ("Biên bản", re.compile(r"^Biên\s+bản\b", re.IGNORECASE)),
        ]
        for dong in dong_tieu_de:
            for loai, pattern in tieu_de_patterns:
                if pattern.search(dong):
                    return loai

        doan_dau = "\n".join(dong_tieu_de)
        for loai, pattern in self.LOAI_VAN_BAN_PATTERNS:
            if pattern.search(doan_dau):
                return loai
        for loai, pattern in self.LOAI_VAN_BAN_PATTERNS:
            if pattern.search(van_ban):
                return loai
        return "Văn bản hành chính"

    def trich_xuat_can_cu_va_so_hieu(self, van_ban: str) -> Dict[str, List[str]]:
        """Trích xuất tất cả các căn cứ pháp lý và số hiệu văn bản có trong nội dung."""
        can_cu = [m.group(0).strip() for m in self.RE_CAN_CU.finditer(van_ban)]
        so_hieu = list(dict.fromkeys(m.group(0).strip() for m in self.RE_SO_HIEU.finditer(van_ban)))
        return {
            "danh_sach_can_cu": can_cu,
            "so_hieu_van_ban": so_hieu,
        }

    def trich_xuat_metadata_hanh_chinh(self, van_ban: str) -> Dict[str, Any]:
        """Trích xuất siêu dữ liệu chuẩn của văn bản hành chính/pháp lý (loại, số hiệu, cơ quan, ngày, trích yếu)."""
        can_cu_so_hieu = self.trich_xuat_can_cu_va_so_hieu(van_ban)
        match_ngay = self.RE_NGAY_BAN_HANH.search(van_ban)
        match_trich_yeu = self.RE_TRICH_YEU.search(van_ban)
        match_co_quan = self.RE_CO_QUAN.search(van_ban)
        return {
            "loai_van_ban": self.phan_loai_van_ban(van_ban),
            "so_hieu_van_ban": can_cu_so_hieu["so_hieu_van_ban"],
            "danh_sach_can_cu": can_cu_so_hieu["danh_sach_can_cu"],
            "co_quan_ban_hanh": match_co_quan.group(1).strip() if match_co_quan else "",
            "ngay_ban_hanh": match_ngay.group(0).strip() if match_ngay else "",
            "trich_yeu": match_trich_yeu.group(1).strip() if match_trich_yeu else "",
        }

    def phan_tich_cau_truc(self, van_ban: str) -> List[Dict[str, Any]]:
        """
        Phân tích văn bản hành chính/pháp lý thành cấu trúc phân cấp:
        - Văn bản quy phạm/hợp đồng có `Điều`: Chương -> Mục -> Điều -> Khoản/Điểm + Phụ lục.
        - Công văn/Tờ trình/Biên bản không có `Điều`: tự động phân tách theo đề mục La Mã (`I.`, `II.`) hoặc Phụ lục.
        """
        dong_list = van_ban.split("\n")
        co_dieu = bool(self.RE_DIEU.search(van_ban))
        chuong_hien_tai = "Phần chung"
        muc_hien_tai = ""
        dieu_hien_tai = ""
        loai_don_vi = "dieu"
        noi_dung_dieu: List[str] = []
        danh_sach_dieu: List[Dict[str, Any]] = []

        def _luu_don_vi_hien_tai() -> None:
            nonlocal dieu_hien_tai, noi_dung_dieu
            if dieu_hien_tai:
                noi_dung = "\n".join(noi_dung_dieu) if noi_dung_dieu else dieu_hien_tai
                danh_sach_dieu.append(
                    {
                        "chuong": chuong_hien_tai,
                        "muc": muc_hien_tai,
                        "tieu_de_dieu": dieu_hien_tai,
                        "loai_don_vi": loai_don_vi,
                        "noi_dung": noi_dung,
                    }
                )
                dieu_hien_tai = ""
                noi_dung_dieu = []

        for dong in dong_list:
            dong_clean = dong.strip()
            if not dong_clean:
                continue

            if self.RE_CHUONG.match(dong_clean):
                _luu_don_vi_hien_tai()
                chuong_hien_tai = dong_clean
                muc_hien_tai = ""
                continue

            if self.RE_MUC.match(dong_clean):
                _luu_don_vi_hien_tai()
                muc_hien_tai = dong_clean
                continue

            if self.RE_PHU_LUC.match(dong_clean):
                _luu_don_vi_hien_tai()
                dieu_hien_tai = dong_clean
                loai_don_vi = "phu_luc"
                noi_dung_dieu = []
                continue

            if self.RE_DIEU.match(dong_clean):
                _luu_don_vi_hien_tai()
                dieu_hien_tai = dong_clean
                loai_don_vi = "dieu"
                noi_dung_dieu = []
                continue

            if not co_dieu and self.RE_MUC_LA_MA.match(dong_clean):
                _luu_don_vi_hien_tai()
                dieu_hien_tai = dong_clean
                loai_don_vi = "muc_hanh_chinh"
                noi_dung_dieu = []
                continue

            noi_dung_dieu.append(dong_clean)

        if dieu_hien_tai:
            _luu_don_vi_hien_tai()
        elif noi_dung_dieu and not danh_sach_dieu:
            danh_sach_dieu.append(
                {
                    "chuong": chuong_hien_tai,
                    "muc": muc_hien_tai,
                    "tieu_de_dieu": "Văn bản chung",
                    "loai_don_vi": "chung",
                    "noi_dung": "\n".join(noi_dung_dieu),
                }
            )

        return danh_sach_dieu

    def phan_doan_giu_ngu_canh(
        self,
        van_ban: str,
        ten_tai_lieu: str = "Tài liệu",
        max_tokens: int = 250,
    ) -> List[Dict[str, Any]]:
        """
        Chia nhỏ văn bản (Hierarchy-Preserving Chunking) luôn bảo toàn tên tài liệu, Chương, Mục, Điều và metadata.
        """
        meta_doc = self.trich_xuat_metadata_hanh_chinh(van_ban)
        ds_dieu = self.phan_tich_cau_truc(van_ban)
        chunks: List[Dict[str, Any]] = []

        def _tao_prefix(chuong: str, muc: str, dieu: str) -> str:
            thanh_phan = [ten_tai_lieu, chuong]
            if muc:
                thanh_phan.append(muc)
            thanh_phan.append(dieu)
            return "[" + " | ".join(thanh_phan) + "]\n"

        for idx_dieu, item in enumerate(ds_dieu):
            chuong = item["chuong"]
            muc = item.get("muc", "")
            tieu_de_dieu = item["tieu_de_dieu"]
            loai_don_vi = item.get("loai_don_vi", "dieu")
            noi_dung = item["noi_dung"]

            khoan_list = [k.strip() for k in noi_dung.split("\n") if k.strip()]
            chunk_tam: List[str] = []
            so_tu_tam = 0

            for k in khoan_list:
                so_tu_k = len(k.split())
                if so_tu_tam + so_tu_k > max_tokens and chunk_tam:
                    chunk_text = "\n".join(chunk_tam)
                    prefix = _tao_prefix(chuong, muc, tieu_de_dieu)
                    chunks.append(
                        {
                            "id": f"{ten_tai_lieu}_dieu_{idx_dieu}_{len(chunks)}",
                            "noi_dung": chunk_text,
                            "ngu_canh_day_du": prefix + chunk_text,
                            "metadata": {
                                "tai_lieu": ten_tai_lieu,
                                "loai_van_ban": meta_doc["loai_van_ban"],
                                "chuong": chuong,
                                "muc": muc,
                                "dieu": tieu_de_dieu,
                                "loai_don_vi": loai_don_vi,
                                "co_quan_ban_hanh": meta_doc["co_quan_ban_hanh"],
                                "ngay_ban_hanh": meta_doc["ngay_ban_hanh"],
                            },
                            "vector": self.embedder.nhung(prefix + chunk_text),
                        }
                    )
                    chunk_tam = [k]
                    so_tu_tam = so_tu_k
                else:
                    chunk_tam.append(k)
                    so_tu_tam += so_tu_k

            if chunk_tam:
                chunk_text = "\n".join(chunk_tam)
                prefix = _tao_prefix(chuong, muc, tieu_de_dieu)
                chunks.append(
                    {
                        "id": f"{ten_tai_lieu}_dieu_{idx_dieu}_{len(chunks)}",
                        "noi_dung": chunk_text,
                        "ngu_canh_day_du": prefix + chunk_text,
                        "metadata": {
                            "tai_lieu": ten_tai_lieu,
                            "loai_van_ban": meta_doc["loai_van_ban"],
                            "chuong": chuong,
                            "muc": muc,
                            "dieu": tieu_de_dieu,
                            "loai_don_vi": loai_don_vi,
                            "co_quan_ban_hanh": meta_doc["co_quan_ban_hanh"],
                            "ngay_ban_hanh": meta_doc["ngay_ban_hanh"],
                        },
                        "vector": self.embedder.nhung(prefix + chunk_text),
                    }
                )

        return chunks

    def nap_tu_tep(
        self,
        duong_dan: Union[str, Path],
        ten_tai_lieu: Optional[str] = None,
        max_tokens: int = 250,
    ) -> Dict[str, Any]:
        """Đọc tài liệu từ tệp đa định dạng (.txt, .md, .pdf, ảnh OCR, .json) và tạo danh sách chunks pháp lý."""
        from vietnamese_ai.rag.legal.document_parser import BoDocTaiLieuPhapLy

        bo_doc = BoDocTaiLieuPhapLy()
        kq_doc = bo_doc.doc_tep(duong_dan)
        ten_mac_dinh = ten_tai_lieu or Path(duong_dan).stem
        chunks = self.phan_doan_giu_ngu_canh(
            kq_doc["van_ban"],
            ten_tai_lieu=ten_mac_dinh,
            max_tokens=max_tokens,
        )
        return {
            "ten_tai_lieu": ten_mac_dinh,
            "dinh_dang": kq_doc["dinh_dang"],
            "so_trang": kq_doc["so_trang"],
            "metadata_hanh_chinh": self.trich_xuat_metadata_hanh_chinh(kq_doc["van_ban"]),
            "chunks": chunks,
        }

    def truy_hoi_phap_ly(
        self,
        cau_hoi: str,
        cac_chunks: List[Dict[str, Any]],
        top_k: int = 3,
        trong_so_hybrid: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các đoạn văn bản pháp lý/hành chính có độ liên quan cao nhất kèm điểm tin cậy.

        Sử dụng cơ chế Hybrid Retrieval: Kết hợp Dense Semantic Vector + Lexical BM25 Matching
        giúp nhận diện chuẩn xác cả ngữ cảnh ngữ nghĩa lẫn từ khóa căn cứ (Điều, Khoản, thuật ngữ).
        """
        if not cac_chunks:
            return []

        if self.tuong_lua is not None:
            kt = self.tuong_lua.kiem_tra_dau_vao(str(cau_hoi or ""))
            if not kt.get("an_toan", True):
                ly_do = kt.get("ly_do", "Vi phạm chính sách an toàn thông tin")
                return [
                    {
                        "id": "firewall_blocked",
                        "diem_tuong_dong": 0.0,
                        "trich_dan_phap_ly": "Tường lửa AI (Cảnh báo an ninh)",
                        "noi_dung": f"[Bị chặn bởi Tường lửa AI]: {ly_do}",
                        "ngu_canh": "Truy vấn vi phạm chính sách an toàn thông tin.",
                    }
                ]

        q_vec = self.embedder.nhung(cau_hoi)
        q_tokens = [t.lower() for t in self.embedder.tokenizer.tach_tu(cau_hoi, noi_tu_ghep=True)]
        scored: List[Tuple[float, Dict[str, Any]]] = []

        k1 = 1.5
        b = 0.75
        avgdl = 50.0

        for c in cac_chunks:
            vec = c.get("vector")
            if vec is None:
                vec = self.embedder.nhung(c["ngu_canh_day_du"])
            sem_score = float(np.dot(q_vec, vec))

            if trong_so_hybrid > 0:
                c_tokens = c.get("_tokens")
                if c_tokens is None:
                    c_text = c.get("ngu_canh_day_du") or c.get("noi_dung", "")
                    c_tokens = [t.lower() for t in self.embedder.tokenizer.tach_tu(c_text, noi_tu_ghep=True)]
                    c["_tokens"] = c_tokens

                doc_len = len(c_tokens)
                doc_counts: Dict[str, int] = {}
                for t in c_tokens:
                    doc_counts[t] = doc_counts.get(t, 0) + 1

                bm25 = 0.0
                for qt in q_tokens:
                    if qt in doc_counts:
                        tf = doc_counts[qt]
                        bm25 += (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avgdl)))
                bm25_norm = bm25 / (bm25 + 5.0)
                score = (1.0 - trong_so_hybrid) * sem_score + trong_so_hybrid * bm25_norm
            else:
                score = sem_score

            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        ket_qua: List[Dict[str, Any]] = []
        for score, c in scored[:top_k]:
            ket_qua.append(
                {
                    "id": c.get("id"),
                    "diem_tuong_dong": round(score, 4),
                    "trich_dan_phap_ly": f"{c['metadata']['dieu']} ({c['metadata']['tai_lieu']})",
                    "noi_dung": c["noi_dung"],
                    "ngu_canh": c["ngu_canh_day_du"],
                }
            )

        return ket_qua

    def tao_prompt_phap_ly(
        self,
        cau_hoi: str,
        ket_qua_truy_hoi: List[Dict[str, Any]],
        ten_mau: str = "tra_cuu_dieu_khoan",
        **kwargs: Any,
    ) -> str:
        """Sinh prompt pháp lý chuẩn hóa gắn trích dẫn từ kết quả `truy_hoi_phap_ly`."""
        from vietnamese_ai.rag.legal.prompts import BoMauPromptPhapLy

        thu_vien = BoMauPromptPhapLy()
        return thu_vien.tao_prompt_tu_truy_hoi(
            cau_hoi=cau_hoi,
            ket_qua_truy_hoi=ket_qua_truy_hoi,
            ten_mau=ten_mau,
            **kwargs,
        )


VietnameseAdministrativeRAG = BoToiUuNguCanhHanhChinh
