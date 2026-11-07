"""DanhGiaRAG - Bộ đo lường chất lượng hệ thống RAG theo chuẩn RAGAS Tiếng Việt."""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from vietnamese_ai.rag.embedder import BoNhungVanBan


class DanhGiaRAG:
    """
    Bộ đánh giá hệ thống RAG (Retrieval-Augmented Generation) theo chuẩn RAGAS cho tiếng Việt.

    Đo lường 4 chỉ số cốt lõi trong đoạn `[0.0, 1.0]`:
    1. `do_trung_thuc` (Faithfulness): Mức độ câu trả lời bám sát ngữ cảnh truy xuất (chống ảo giác).
    2. `do_lien_quan_tra_loi` (Answer Relevancy): Mức độ câu trả lời giải quyết đúng trọng tâm câu hỏi.
    3. `do_chinh_xac_ngu_canh` (Context Precision): Tỷ lệ các đoạn ngữ cảnh xếp hạng cao thực sự hữu ích.
    4. `do_phu_ngu_canh` (Context Recall): Mức độ ngữ cảnh truy xuất bao phủ được đáp án chuẩn.
    """

    TU_DUNG_CO_BAN = {
        "là",
        "của",
        "và",
        "các",
        "có",
        "trong",
        "được",
        "cho",
        "với",
        "một",
        "những",
        "này",
        "đó",
        "thì",
        "mà",
        "ở",
        "tại",
        "về",
        "gì",
        "nào",
        "đâu",
        "như",
        "thế",
        "dựa",
        "trên",
        "tài",
        "liệu",
        "đây",
        "thông",
        "tin",
        "liên",
        "quan",
    }

    def __init__(self, kich_thuoc_embed: int = 128):
        self.embedder = BoNhungVanBan(kich_thuoc=kich_thuoc_embed)

    def _tach_tu_noi_dung(self, van_ban: str) -> List[str]:
        tokens = re.findall(r"\w+", van_ban.lower())
        filtered = [t for t in tokens if t not in self.TU_DUNG_CO_BAN and len(t) > 1]
        return filtered if filtered else tokens

    def _do_tuong_dong_ngu_nghia(self, vb1: str, vb2: str) -> float:
        if not vb1.strip() or not vb2.strip():
            return 0.0
        vecs = self.embedder.nhung([vb1, vb2])
        sim = float(np.dot(vecs[0], vecs[1]))
        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))

    def tinh_do_trung_thuc(self, tra_loi: str, ngu_canh: List[str]) -> float:
        """
        Tính Độ trung thực (Faithfulness): Tỷ lệ thông tin trong `tra_loi` có căn cứ từ `ngu_canh`.
        """
        if not tra_loi.strip() or not ngu_canh:
            return 0.0
        gop_ngu_canh = " ".join(ngu_canh)
        tu_tra_loi = self._tach_tu_noi_dung(tra_loi)
        tu_ngu_canh = set(self._tach_tu_noi_dung(gop_ngu_canh))
        if not tu_tra_loi:
            return 0.0

        ty_le_tu = sum(1 for t in tu_tra_loi if t in tu_ngu_canh) / len(tu_tra_loi)
        sim = self._do_tuong_dong_ngu_nghia(tra_loi, gop_ngu_canh)
        return round(float(0.65 * ty_le_tu + 0.35 * sim), 4)

    def tinh_do_lien_quan_tra_loi(self, cau_hoi: str, tra_loi: str) -> float:
        """
        Tính Độ liên quan câu trả lời (Answer Relevancy) giữa `cau_hoi` và `tra_loi`.
        """
        if not cau_hoi.strip() or not tra_loi.strip():
            return 0.0
        if "không tìm thấy thông tin" in tra_loi.lower():
            return 0.05

        tu_hoi = set(self._tach_tu_noi_dung(cau_hoi))
        tu_tl = set(self._tach_tu_noi_dung(tra_loi))
        overlap = len(tu_hoi & tu_tl) / max(1, len(tu_hoi)) if tu_hoi else 0.5
        sim = self._do_tuong_dong_ngu_nghia(cau_hoi, tra_loi)
        return round(float(np.clip(0.5 * overlap + 0.5 * sim, 0.0, 1.0)), 4)

    def tinh_do_chinh_xac_ngu_canh(
        self, cau_hoi: str, ngu_canh: List[str], dap_an_chuan: Optional[str] = None
    ) -> float:
        """
        Tính Độ chính xác ngữ cảnh (Context Precision - Average Precision@K).
        """
        if not ngu_canh:
            return 0.0
        muc_tieu = f"{cau_hoi} {dap_an_chuan or ''}".strip()
        tu_muc_tieu = set(self._tach_tu_noi_dung(muc_tieu))

        so_dung = 0
        tong_ap = 0.0
        for idx, ctx in enumerate(ngu_canh, start=1):
            tu_ctx = set(self._tach_tu_noi_dung(ctx))
            ty_le = len(tu_muc_tieu & tu_ctx) / max(1, len(tu_muc_tieu))
            sim = self._do_tuong_dong_ngu_nghia(muc_tieu, ctx)
            lien_quan = (0.6 * ty_le + 0.4 * sim) >= 0.25
            if lien_quan:
                so_dung += 1
                tong_ap += so_dung / idx

        if so_dung == 0:
            return 0.0
        return round(float(tong_ap / so_dung), 4)

    def tinh_do_phu_ngu_canh(
        self, ngu_canh: List[str], dap_an_chuan: Optional[str] = None, cau_hoi: str = ""
    ) -> float:
        """
        Tính Độ phủ ngữ cảnh (Context Recall) so với `dap_an_chuan`.
        """
        if not ngu_canh:
            return 0.0
        tham_chieu = dap_an_chuan if (dap_an_chuan and dap_an_chuan.strip()) else cau_hoi
        if not tham_chieu.strip():
            return 0.0

        gop_ngu_canh = " ".join(ngu_canh)
        tu_ref = self._tach_tu_noi_dung(tham_chieu)
        tu_ctx = set(self._tach_tu_noi_dung(gop_ngu_canh))
        if not tu_ref:
            return 0.0

        recall_lex = sum(1 for t in tu_ref if t in tu_ctx) / len(tu_ref)
        sim = self._do_tuong_dong_ngu_nghia(tham_chieu, gop_ngu_canh)
        return round(float(0.7 * recall_lex + 0.3 * sim), 4)

    def cham_diem_mau(
        self,
        cau_hoi: str,
        tra_loi: str,
        ngu_canh: List[str],
        dap_an_chuan: Optional[str] = None,
    ) -> Dict[str, float]:
        """Chấm điểm toàn diện 4 chỉ số RAGAS cho một mẫu truy vấn RAG."""
        faithfulness = self.tinh_do_trung_thuc(tra_loi, ngu_canh)
        relevancy = self.tinh_do_lien_quan_tra_loi(cau_hoi, tra_loi)
        precision = self.tinh_do_chinh_xac_ngu_canh(cau_hoi, ngu_canh, dap_an_chuan)
        recall = self.tinh_do_phu_ngu_canh(ngu_canh, dap_an_chuan, cau_hoi)
        tong_hop = round((faithfulness + relevancy + precision + recall) / 4.0, 4)
        return {
            "do_trung_thuc": faithfulness,
            "do_lien_quan_tra_loi": relevancy,
            "do_chinh_xac_ngu_canh": precision,
            "do_phu_ngu_canh": recall,
            "diem_ragas": tong_hop,
        }

    def danh_gia_pipeline(
        self,
        rag_pipeline: Any,
        tap_kiem_thu: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Đánh giá tự động một `RAGPipeline` hoặc `QuyTrinhDauCuoi` trên tập kiểm thử
        `[{"question": ..., "ground_truth": ...}, ...]`.
        """
        chi_tiet: List[Dict[str, Any]] = []
        for mau in tap_kiem_thu:
            cau_hoi = str(mau.get("cau_hoi", mau.get("question", mau.get("instruction", ""))))
            dap_an = str(mau.get("dap_an_chuan", mau.get("ground_truth", mau.get("output", ""))))

            kq = rag_pipeline.hoi(cau_hoi)
            tra_loi = str(kq.get("tra_loi", ""))
            nguon_raw = kq.get("nguon", [])
            ngu_canh = [
                str(n.get("metadata", {}).get("noi_dung", n.get("noi_dung", "")))
                for n in nguon_raw
            ]
            ngu_canh = [c for c in ngu_canh if c]

            diem = self.cham_diem_mau(cau_hoi, tra_loi, ngu_canh, dap_an)
            chi_tiet.append({"cau_hoi": cau_hoi, "tra_loi": tra_loi, **diem})

        if not chi_tiet:
            return {"so_mau": 0, "diem_ragas": 0.0, "chi_tiet": []}

        keys = [
            "do_trung_thuc",
            "do_lien_quan_tra_loi",
            "do_chinh_xac_ngu_canh",
            "do_phu_ngu_canh",
            "diem_ragas",
        ]
        trung_binh = {
            k: round(float(np.mean([m[k] for m in chi_tiet])), 4) for k in keys
        }
        return {
            "so_mau": len(chi_tiet),
            **trung_binh,
            "chi_tiet": chi_tiet,
        }
