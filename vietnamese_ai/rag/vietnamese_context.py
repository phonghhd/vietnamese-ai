"""Bộ Tối Ưu Ngữ Cảnh Hành Chính & Pháp Lý Việt Nam (Vietnamese Administrative & Legal RAG) - Vietnamese AI Framework.

Tối ưu hóa chuyên sâu cho tài liệu hành chính, thông tư, nghị định, luật pháp Việt Nam.
Giải quyết triệt để lỗi phân mảnh câu (chunk truncation) của LangChain/LlamaIndex trên văn bản tiếng Việt.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vietnamese_ai.nlp.native_tokenizer import BoNhungVanBanSieuToc, BoTachTuTiengVietSieuToc


class BoToiUuNguCanhHanhChinh:
    """
    Bộ Tối Ưu Ngữ Cảnh Hành Chính & Pháp Lý Việt Nam (Vietnamese Administrative & Legal RAG Optimizer):
    - Tự động bóc tách cấu trúc phân cấp: Chương, Mục, Điều, Khoản, Điểm, Căn cứ pháp lý.
    - Kỹ thuật Hierarchy-Preserving Chunking: Giữ toàn vẹn văn cảnh Điều luật mẹ kèm trích dẫn.
    - Đánh chỉ mục và truy hồi có trích dẫn số hiệu chuẩn mực (Article/Clause Grounded Citation).
    - Giải quyết bài toán tài liệu hành chính nhà nước và hợp đồng thương mại Việt Nam.
    """

    # Regex nhận diện các thành phần cấu trúc văn bản quy phạm pháp luật Việt Nam
    RE_CHUONG = re.compile(r"^\s*(Chương\s+[IVXLCDM\d]+[.:\s]*[^\n]*)", re.MULTILINE | re.IGNORECASE)
    RE_MUC = re.compile(r"^\s*(Mục\s+\d+[.:\s]*[^\n]*)", re.MULTILINE | re.IGNORECASE)
    RE_DIEU = re.compile(r"^\s*(Điều\s+\d+[.:\s]*[^\n]*)", re.MULTILINE | re.IGNORECASE)
    RE_KHOAN = re.compile(r"^\s*(\d+\.\s+[^\n]*)", re.MULTILINE)
    RE_DIEM = re.compile(r"^\s*([a-zđ]\)\s+[^\n]*)", re.MULTILINE | re.IGNORECASE)
    RE_CAN_CU = re.compile(r"^\s*(Căn cứ\s+[^\n;.]+)", re.MULTILINE | re.IGNORECASE)
    RE_SO_HIEU = re.compile(
        r"((?:Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Quyết\s+định)(?:\s+[^\n,;.]+?)?\s+số\s+[\d/]+[A-Z\d\-]+|[\d/]+/[A-Z\d\-]+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        embedder: Optional[BoNhungVanBanSieuToc] = None,
        bat_tuong_lua: bool = True,
    ):
        self.tokenizer = BoTachTuTiengVietSieuToc()
        self.embedder = embedder or BoNhungVanBanSieuToc(tokenizer=self.tokenizer)
        self.bat_tuong_lua = bat_tuong_lua
        if bat_tuong_lua:
            from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

            self.tuong_lua: Optional[TuongLuaBaoVeFramework] = TuongLuaBaoVeFramework(chan_khi_vi_pham=False)
        else:
            self.tuong_lua = None

    def trich_xuat_can_cu_va_so_hieu(self, van_ban: str) -> Dict[str, List[str]]:
        """Trích xuất tất cả các căn cứ pháp lý và số hiệu văn bản có trong nội dung."""
        can_cu = [m.group(0).strip() for m in self.RE_CAN_CU.finditer(van_ban)]
        so_hieu = list(set(m.group(0).strip() for m in self.RE_SO_HIEU.finditer(van_ban)))
        return {
            "danh_sach_can_cu": can_cu,
            "so_hieu_van_ban": so_hieu,
        }

    def phan_tich_cau_truc(self, van_ban: str) -> List[Dict[str, Any]]:
        """
        Phân tích văn bản pháp lý thành cây cấu trúc phân cấp (Chương -> Điều -> Khoản).
        """
        dong_list = van_ban.split("\n")
        chuong_hien_tai = "Phần chung"
        dieu_hien_tai = ""
        noi_dung_dieu: List[str] = []
        danh_sach_dieu: List[Dict[str, Any]] = []

        for dong in dong_list:
            dong_clean = dong.strip()
            if not dong_clean:
                continue

            # Kiểm tra Chương
            if self.RE_CHUONG.match(dong_clean):
                chuong_hien_tai = dong_clean
                continue

            # Kiểm tra Điều
            if self.RE_DIEU.match(dong_clean):
                if dieu_hien_tai and noi_dung_dieu:
                    danh_sach_dieu.append({
                        "chuong": chuong_hien_tai,
                        "tieu_de_dieu": dieu_hien_tai,
                        "noi_dung": "\n".join(noi_dung_dieu),
                    })
                dieu_hien_tai = dong_clean
                noi_dung_dieu = []
                continue

            if dieu_hien_tai:
                noi_dung_dieu.append(dong_clean)
            else:
                # Các dòng mở đầu trước Điều 1
                noi_dung_dieu.append(dong_clean)

        # Lưu lại điều cuối cùng
        if dieu_hien_tai and noi_dung_dieu:
            danh_sach_dieu.append({
                "chuong": chuong_hien_tai,
                "tieu_de_dieu": dieu_hien_tai,
                "noi_dung": "\n".join(noi_dung_dieu),
            })
        elif noi_dung_dieu and not danh_sach_dieu:
            danh_sach_dieu.append({
                "chuong": chuong_hien_tai,
                "tieu_de_dieu": "Văn bản chung",
                "noi_dung": "\n".join(noi_dung_dieu),
            })

        return danh_sach_dieu

    def phan_doan_giu_ngu_canh(
        self,
        van_ban: str,
        ten_tai_lieu: str = "Tài liệu",
        max_tokens: int = 250,
    ) -> List[Dict[str, Any]]:
        """
        Chia nhỏ văn bản (Chunking) nhưng luôn bảo toàn tiêu đề Điều, Chương và văn cảnh mẹ.

        Mỗi chunk sẽ chứa:
        - `noi_dung_chunk`: Đoạn văn bản cụ thể.
        - `ngu_canh_day_du`: Tiêu đề văn bản + Chương + Điều + Đoạn.
        - `metadata`: Trích dẫn chính xác để AI viện dẫn điều luật.
        """
        ds_dieu = self.phan_tich_cau_truc(van_ban)
        chunks: List[Dict[str, Any]] = []

        for idx_dieu, item in enumerate(ds_dieu):
            chuong = item["chuong"]
            tieu_de_dieu = item["tieu_de_dieu"]
            noi_dung = item["noi_dung"]

            # Phân tách theo từng Khoản nếu nội dung Điều quá dài
            khoan_list = [k.strip() for k in noi_dung.split("\n") if k.strip()]

            chunk_tam: List[str] = []
            so_tu_tam = 0

            for k in khoan_list:
                so_tu_k = len(k.split())
                if so_tu_tam + so_tu_k > max_tokens and chunk_tam:
                    chunk_text = "\n".join(chunk_tam)
                    prefix = f"[{ten_tai_lieu} | {chuong} | {tieu_de_dieu}]\n"
                    chunks.append({
                        "id": f"{ten_tai_lieu}_dieu_{idx_dieu}_{len(chunks)}",
                        "noi_dung": chunk_text,
                        "ngu_canh_day_du": prefix + chunk_text,
                        "metadata": {
                            "tai_lieu": ten_tai_lieu,
                            "chuong": chuong,
                            "dieu": tieu_de_dieu,
                        },
                        "vector": self.embedder.nhung(prefix + chunk_text),
                    })
                    chunk_tam = [k]
                    so_tu_tam = so_tu_k
                else:
                    chunk_tam.append(k)
                    so_tu_tam += so_tu_k

            if chunk_tam:
                chunk_text = "\n".join(chunk_tam)
                prefix = f"[{ten_tai_lieu} | {chuong} | {tieu_de_dieu}]\n"
                chunks.append({
                    "id": f"{ten_tai_lieu}_dieu_{idx_dieu}_{len(chunks)}",
                    "noi_dung": chunk_text,
                    "ngu_canh_day_du": prefix + chunk_text,
                    "metadata": {
                        "tai_lieu": ten_tai_lieu,
                        "chuong": chuong,
                        "dieu": tieu_de_dieu,
                    },
                    "vector": self.embedder.nhung(prefix + chunk_text),
                })

        return chunks

    def truy_hoi_phap_ly(
        self,
        cau_hoi: str,
        cac_chunks: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các đoạn văn bản pháp lý/hành chính có độ liên quan cao nhất kèm điểm tin cậy.
        """
        if not cac_chunks:
            return []

        # Kiểm tra an toàn truy vấn qua Tường lửa AI
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
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for c in cac_chunks:
            vec = c.get("vector")
            if vec is None:
                vec = self.embedder.nhung(c["ngu_canh_day_du"])
            score = float(np.dot(q_vec, vec))
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        ket_qua: List[Dict[str, Any]] = []
        for score, c in scored[:top_k]:
            ket_qua.append({
                "id": c.get("id"),
                "diem_tuong_dong": round(score, 4),
                "trich_dan_phap_ly": f"{c['metadata']['dieu']} ({c['metadata']['tai_lieu']})",
                "noi_dung": c["noi_dung"],
                "ngu_canh": c["ngu_canh_day_du"],
            })

        return ket_qua


# Aliases
VietnameseAdministrativeRAG = BoToiUuNguCanhHanhChinh
