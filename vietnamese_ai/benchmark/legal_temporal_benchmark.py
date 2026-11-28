"""BoDanhGiaLegalTemporalRAG (ViLegalTemporalBenchmark) - Bộ Đo Lường & Ablation Chuẩn Học Thuật Cho Legal Temporal RAG.

Đánh giá định lượng 5 cấu hình đối chứng (Ablation Study) trên bộ dữ liệu mở `ViLegal-TempBench`:
1. `bm25`: Okapi BM25 thuần túy trên token tiếng Việt.
2. `hybrid`: Kết hợp BM25 + Biểu diễn ngữ nghĩa vector (CPU/ONNX) qua Reciprocal Rank Fusion.
3. `temporal`: Hybrid + Bóc tách mốc thời gian tự động + Lọc hiệu lực Point-in-Time & Chuyển hướng văn bản thay thế.
4. `temporal_graph`: Temporal + Lan truyền đồ thị đa bước cấp Điều/Khoản có điều kiện thời gian `t`.
5. `full`: Temporal-Graph + Xếp hạng lại chéo + Cổng từ chối trả lời (Abstention Gate) + Kiểm chứng trích dẫn 3 tầng.

Hệ chỉ số học thuật: Recall@1/3/5, MRR@10, nDCG@10, TVA (Temporal Validity Accuracy),
TCA (Temporal Contrast Accuracy), Multi-Hop Bridge Recall, Trap Correction Rate,
Citation Faithfulness, Temporal Conflict Rate, Abstention F1 và Paired Bootstrap p-value.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vietnamese_ai.benchmark.datasets_vn import (
    tai_bo_hoi_dap_phap_ly_thoi_gian,
    tai_bo_phap_ly_mo,
    tai_bo_phap_ly_thoi_gian,
    tai_canh_do_thi_phap_ly_thoi_gian,
)
from vietnamese_ai.rag.legal.temporal_rag import BoTruyHoiPhapLyThoiGian
from vietnamese_ai.rag.legal.validity_tracker import _phan_tich_ngay
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


def tinh_ndcg_at_k(danh_sach_id: Sequence[str], tap_dung: Sequence[str], k: int = 10) -> float:
    """Tính chuẩn hóa nDCG@k với độ liên quan nhị phân cho một truy vấn."""
    gold = set(tap_dung)
    if not gold:
        return 0.0
    dcg = 0.0
    for idx, doc_id in enumerate(danh_sach_id[:k], start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(idx + 1.0)
    so_luong_ly_tuong = min(len(gold), k)
    idcg = sum(1.0 / math.log2(idx + 1.0) for idx in range(1, so_luong_ly_tuong + 1))
    return (dcg / idcg) if idcg > 0.0 else 0.0


def kiem_dinh_bootstrap_cap(
    diem_moi: Sequence[float],
    diem_co_so: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Kiểm định ý nghĩa thống kê Paired Bootstrap hai phía giữa hai chuỗi điểm theo từng truy vấn."""
    arr_moi = np.asarray(diem_moi, dtype=np.float64)
    arr_base = np.asarray(diem_co_so, dtype=np.float64)
    n = min(arr_moi.shape[0], arr_base.shape[0])
    if n == 0:
        return {
            "so_mau": 0,
            "delta_trung_binh": 0.0,
            "khoang_tin_cay_95": [0.0, 0.0],
            "p_value": 1.0,
            "co_y_nghia_p05": False,
        }

    hieu = arr_moi[:n] - arr_base[:n]
    delta_quan_sat = float(np.mean(hieu))
    rng = np.random.default_rng(int(seed))
    chi_so = rng.integers(0, n, size=(max(100, int(n_bootstrap)), n))
    cac_delta = np.mean(hieu[chi_so], axis=1)
    ci_low, ci_high = np.percentile(cac_delta, [2.5, 97.5])

    if abs(delta_quan_sat) < 1e-12:
        p_val = 1.0
    else:
        hieu_tam = hieu - delta_quan_sat
        cac_delta_null = np.mean(hieu_tam[chi_so], axis=1)
        p_val = float(np.mean(np.abs(cac_delta_null) >= abs(delta_quan_sat) - 1e-12))

    return {
        "so_mau": int(n),
        "delta_trung_binh": round(delta_quan_sat, 4),
        "khoang_tin_cay_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "p_value": round(p_val, 4),
        "co_y_nghia_p05": bool(p_val < 0.05),
    }


def tao_ham_sinh_llm(
    nha_cung_cap: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    nhiet_do: float = 0.0,
    timeout: float = 45.0,
    do_tre_giua_goi_giay: float = 0.0,
    so_lan_thu_lai: int = 2,
) -> Callable[[str], str]:
    """Tạo hàm sinh văn bản chuẩn HTTPS không phụ thuộc SDK ngoài cho Google Gemini, OpenAI-compatible hoặc Ollama."""
    provider = str(nha_cung_cap or "").strip().lower()
    if provider not in ("gemini", "openai", "ollama"):
        raise ValueError(
            f"Nhà cung cấp LLM không hỗ trợ: {nha_cung_cap}. Chọn một trong: gemini, openai, ollama."
        )

    if provider == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        if not key.strip():
            raise ValueError(
                "Thiếu GEMINI_API_KEY (hoặc GOOGLE_API_KEY) để chạy thực nghiệm Google Gemini API."
            )
        model_name = model or "gemini-2.5-flash"
        base = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        url = f"{base}/v1beta/models/{model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": key.strip(),
        }
    elif provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        if not key.strip():
            raise ValueError("Thiếu OPENAI_API_KEY để chạy thực nghiệm OpenAI-compatible API.")
        model_name = model or "gpt-4o-mini"
        base = (base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key.strip()}",
        }
    else:
        model_name = model or "qwen2.5:3b"
        base = (base_url or "http://127.0.0.1:11434").rstrip("/")
        url = f"{base}/api/generate"
        headers = {"Content-Type": "application/json"}

    trang_thai: Dict[str, float] = {"lan_goi_cuoi": 0.0}

    def _goi_api(prompt: str) -> str:
        if do_tre_giua_goi_giay > 0.0 and trang_thai["lan_goi_cuoi"] > 0.0:
            da_qua = time.monotonic() - trang_thai["lan_goi_cuoi"]
            if da_qua < do_tre_giua_goi_giay:
                time.sleep(do_tre_giua_goi_giay - da_qua)

        if provider == "gemini":
            payload: Dict[str, Any] = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": float(nhiet_do),
                    "maxOutputTokens": 512,
                },
            }
        elif provider == "openai":
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(nhiet_do),
                "max_tokens": 512,
            }
        else:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": float(nhiet_do)},
            }

        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        loi_cuoi: Optional[Exception] = None
        for attempt in range(max(1, int(so_lan_thu_lai) + 1)):
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                    raw = resp.read().decode("utf-8")
                trang_thai["lan_goi_cuoi"] = time.monotonic()
                parsed = json.loads(raw)
                if provider == "gemini":
                    cands = parsed.get("candidates") or []
                    if cands:
                        parts = cands[0].get("content", {}).get("parts") or []
                        return "".join(str(p.get("text", "")) for p in parts).strip()
                    return ""
                if provider == "openai":
                    choices = parsed.get("choices") or []
                    if choices:
                        return str(choices[0].get("message", {}).get("content") or "").strip()
                    return ""
                return str(parsed.get("response") or "").strip()
            except urllib.error.HTTPError as exc:
                loi_cuoi = exc
                if exc.code in (429, 500, 502, 503, 504) and attempt < so_lan_thu_lai:
                    time.sleep(min(8.0, 1.5 * (2**attempt)))
                    continue
                raise RuntimeError(f"Lỗi gọi {provider} API (HTTP {exc.code}): {exc.reason}") from exc
            except Exception as exc:
                loi_cuoi = exc
                if attempt < so_lan_thu_lai:
                    time.sleep(1.0)
                    continue
                raise RuntimeError(f"Lỗi kết nối {provider} API: {exc}") from exc
        raise RuntimeError(f"Không thể hoàn tất gọi {provider} API: {loi_cuoi}")

    return _goi_api


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ đo lường và thí nghiệm đối chứng (Ablation Study) chuẩn học thuật cho Temporal & Graph-Grounded Legal RAG "
        "trên bộ dữ liệu mở ViLegal-TempBench, hỗ trợ xuất bảng Markdown/LaTeX và kiểm định Paired Bootstrap."
    ),
    nhom_extra="core",
)
class BoDanhGiaLegalTemporalRAG:
    """Bộ đánh giá khoa học tái lập trên `ViLegal-TempBench v2` (T1, T2 Contrast, T3 Multi-Hop, T4 Traps/Abstention, T5 Colloquial)."""

    TEN_CAU_HINH: Dict[str, str] = {
        "bm25": "BM25-only (Vietnamese Tokenizer)",
        "hybrid": "Hybrid (BM25 + Dense RRF)",
        "naive_date_filter": "Hybrid + Naive Date Filter (Start-Date Only)",
        "temporal": "Hybrid + Point-in-Time Temporal",
        "temporal_graph": "Temporal + Clause Graph Multi-Hop",
        "full": "Full ViLegal-TemporalRAG (+Rerank & Gate)",
    }

    def __init__(
        self,
        gop_kho_phap_ly_mo: bool = True,
        ngay_mac_dinh: str = "2026-01-15",
        ham_nhung: Optional[Callable[[str], np.ndarray]] = None,
        bo_xep_hang_cheo: Optional[Any] = None,
        ham_sinh_llm: Optional[Callable[[str], str]] = None,
        ten_mo_hinh_llm: str = "extractive-grounded",
        duong_dan_onnx_embed: Optional[str] = None,
        duong_dan_onnx_rerank: Optional[str] = None,
    ) -> None:
        self.gop_kho_phap_ly_mo = gop_kho_phap_ly_mo
        self.ngay_mac_dinh = ngay_mac_dinh
        self.ten_bo_nhung = "cpu_lexicon_hash"
        self.ten_bo_rerank = "cross_interaction_cpu"

        if ham_nhung is None and duong_dan_onnx_embed:
            from vietnamese_ai.rag.onnx_semantic import BoNhungONNXTiengViet

            emb_onnx = BoNhungONNXTiengViet(duong_dan_onnx=duong_dan_onnx_embed)
            ham_nhung = emb_onnx.nhung
            self.ten_bo_nhung = f"{emb_onnx.che_do} ({duong_dan_onnx_embed})"
        elif ham_nhung is not None:
            self.ten_bo_nhung = "custom_callable"

        if bo_xep_hang_cheo is None and duong_dan_onnx_rerank:
            from vietnamese_ai.rag.onnx_semantic import BoSapXepLaiONNXTiengViet

            rr_onnx = BoSapXepLaiONNXTiengViet(duong_dan_onnx=duong_dan_onnx_rerank)
            bo_xep_hang_cheo = rr_onnx
            self.ten_bo_rerank = f"{rr_onnx.che_do} ({duong_dan_onnx_rerank})"
        elif bo_xep_hang_cheo is not None:
            self.ten_bo_rerank = type(bo_xep_hang_cheo).__name__

        self.ham_nhung = ham_nhung
        self.bo_xep_hang_cheo = bo_xep_hang_cheo
        self.ham_sinh_llm = ham_sinh_llm
        self.ten_mo_hinh_llm = ten_mo_hinh_llm

        self.du_lieu_dieu_khoan: List[Dict[str, Any]] = []
        self.du_lieu_canh: List[Dict[str, Any]] = []
        self.du_lieu_qa: List[Dict[str, Any]] = []
        self.dong_co: BoTruyHoiPhapLyThoiGian = self._khoi_tao_dong_co()

    def _khoi_tao_dong_co(self) -> BoTruyHoiPhapLyThoiGian:
        """Tải `ViLegal-TempBench`, gộp kho nhiễu 100 điều luật mở và lập chỉ mục truy hồi."""
        corpus_tb = tai_bo_phap_ly_thoi_gian()
        edges_tb = tai_canh_do_thi_phap_ly_thoi_gian()
        qa_tb = tai_bo_hoi_dap_phap_ly_thoi_gian()

        gop_dieu_khoan: List[Dict[str, Any]] = list(corpus_tb)
        da_co_id = {str(x.get("id")) for x in gop_dieu_khoan}
        if self.gop_kho_phap_ly_mo:
            for item in tai_bo_phap_ly_mo():
                ma = str(item.get("id", ""))
                if not ma or ma in da_co_id:
                    ma = f"mo_{ma}"
                da_co_id.add(ma)
                ban_sao = dict(item)
                ban_sao["id"] = ma
                gop_dieu_khoan.append(ban_sao)

        self.du_lieu_dieu_khoan = gop_dieu_khoan
        self.du_lieu_canh = list(edges_tb)
        self.du_lieu_qa = list(qa_tb)

        engine = BoTruyHoiPhapLyThoiGian(
            ham_nhung=self.ham_nhung,
            bo_xep_hang_cheo=self.bo_xep_hang_cheo,
        )
        engine.nap_tu_danh_sach(self.du_lieu_dieu_khoan, self.du_lieu_canh)
        return engine

    def danh_gia_boc_tach_thoi_gian(self, tap_qa: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Đo độ chính xác bóc tách mốc thời gian tự động (Temporal Query Parsing Accuracy) không rò rỉ nhãn."""
        parser = self.dong_co.bo_phan_tich_thoi_gian
        tong = 0
        dung_regime = 0
        chi_tiet_sai: List[Dict[str, Any]] = []

        for q in tap_qa:
            if not q.get("ngay_tra_cuu"):
                continue
            tong += 1
            kq = parser.phan_tich(str(q["cau_hoi"]), ngay_mac_dinh=self.ngay_mac_dinh)
            moc_du_doan = _phan_tich_ngay(kq.ngay_tra_cuu)
            moc_chuan = _phan_tich_ngay(str(q["ngay_tra_cuu"]))

            hop_regime = True
            for tid in q.get("tap_id_ky_vong", []):
                nut = self.dong_co.do_thi.lay_nut_dieu_khoan(str(tid))
                if nut is not None:
                    v_pred = self.dong_co.do_thi.dieu_khoan_con_hieu_luc(nut, ngay_tra_cuu=moc_du_doan)
                    v_gold = self.dong_co.do_thi.dieu_khoan_con_hieu_luc(nut, ngay_tra_cuu=moc_chuan)
                    if v_pred != v_gold:
                        hop_regime = False
                        break
            if hop_regime and abs((moc_du_doan - moc_chuan).days) <= 185:
                dung_regime += 1
            else:
                chi_tiet_sai.append(
                    {
                        "id": q.get("id"),
                        "ngay_chuan": moc_chuan.isoformat(),
                        "ngay_du_doan": moc_du_doan.isoformat(),
                        "bieu_thuc_goc": kq.bieu_thuc_goc,
                    }
                )

        return {
            "tong_cau_hoi": tong,
            "so_cau_dung_vung_hieu_luc": dung_regime,
            "do_chinh_xac_tqpa": round(dung_regime / max(1, tong), 4),
            "cac_truong_hop_lech": chi_tiet_sai,
        }

    def danh_gia_mot_cau_hinh(
        self,
        che_do: str,
        tap_qa: Sequence[Dict[str, Any]],
        top_k: int = 10,
    ) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
        """Chạy đầy đủ bộ câu hỏi cho 1 cấu hình Ablation và trả về tổng hợp chỉ số + vector điểm từng truy vấn."""
        r1_list: List[float] = []
        r3_list: List[float] = []
        r5_list: List[float] = []
        mrr_list: List[float] = []
        ndcg_list: List[float] = []
        tva_list: List[float] = []
        faith_list: List[float] = []
        numeric_list: List[float] = []
        conflict_list: List[float] = []
        latency_list: List[float] = []
        mh_list: List[float] = []
        trap_list: List[float] = []

        theo_nhom_r3: Dict[str, List[float]] = {"T1": [], "T2": [], "T3": [], "T4": [], "T5": []}
        theo_nhom_mrr: Dict[str, List[float]] = {"T1": [], "T2": [], "T3": [], "T4": [], "T5": []}
        theo_nhom_tva: Dict[str, List[float]] = {"T1": [], "T2": [], "T3": [], "T4": [], "T5": []}
        phan_tich_loi: List[Dict[str, Any]] = []

        nhom_tuong_phan: Dict[str, List[bool]] = {}
        tp_abs = 0
        fp_abs = 0
        fn_abs = 0

        for q in tap_qa:
            loai = str(q.get("loai", "T1"))
            gold_ids = [str(x) for x in q.get("tap_id_ky_vong", [])]
            ky_vong_tu_choi = bool(q.get("ky_vong_tu_choi", False))
            ngay_chuan = str(q.get("ngay_tra_cuu") or self.ngay_mac_dinh)

            kq = self.dong_co.tra_loi_phap_ly(
                cau_hoi=str(q["cau_hoi"]),
                ngay_tra_cuu=None,
                top_k=top_k,
                che_do=che_do,
                ham_sinh_llm=self.ham_sinh_llm,
                ngay_mac_dinh=self.ngay_mac_dinh,
            )
            latency_list.append(float(kq.get("thoi_gian_ms", 0.0)))
            du_doan_tu_choi = bool(kq.get("tu_choi_tra_loi", False))

            if ky_vong_tu_choi and du_doan_tu_choi:
                tp_abs += 1
            elif (not ky_vong_tu_choi) and du_doan_tu_choi:
                fp_abs += 1
                phan_tich_loi.append(
                    {
                        "id": str(q.get("id", "")),
                        "loai": loai,
                        "nguyen_nhan": "tu_choi_nham_false_abstention",
                        "ky_vong": gold_ids,
                        "top3_du_doan": [],
                    }
                )
            elif ky_vong_tu_choi and (not du_doan_tu_choi):
                fn_abs += 1
                phan_tich_loi.append(
                    {
                        "id": str(q.get("id", "")),
                        "loai": loai,
                        "nguyen_nhan": "khong_tu_choi_cau_ngoai_pham_vi",
                        "ky_vong": [],
                        "top3_du_doan": [str(item["id"]) for item in list(kq.get("ket_qua", []))[:3]],
                    }
                )

            kc = kq.get("kiem_chung_trich_dan", {})
            faith_list.append(float(kc.get("diem_trung_thuc", 1.0)))
            numeric_list.append(float(kc.get("diem_nhat_quan_so_lieu", 1.0)))
            conflict_list.append(1.0 if kc.get("co_xung_dot_thoi_gian") else 0.0)

            if ky_vong_tu_choi or not gold_ids:
                continue

            retrieved = list(kq.get("ket_qua", []))
            pred_ids = [str(item["id"]) for item in retrieved]
            gold_set = set(gold_ids)

            rec1 = len(set(pred_ids[:1]) & gold_set) / len(gold_set)
            rec3 = len(set(pred_ids[:3]) & gold_set) / len(gold_set)
            rec5 = len(set(pred_ids[:5]) & gold_set) / len(gold_set)

            mrr = 0.0
            for rank_i, pid in enumerate(pred_ids[:10], start=1):
                if pid in gold_set:
                    mrr = 1.0 / rank_i
                    break
            ndcg = tinh_ndcg_at_k(pred_ids, gold_ids, k=10)

            top3_items = retrieved[:3]
            if top3_items:
                so_hop_le = 0
                for item in top3_items:
                    nut_obj = self.dong_co.do_thi.lay_nut_dieu_khoan(str(item["id"]))
                    if nut_obj is not None and self.dong_co.do_thi.dieu_khoan_con_hieu_luc(nut_obj, ngay_tra_cuu=ngay_chuan):
                        so_hop_le += 1
                tva_q = so_hop_le / len(top3_items)
            else:
                tva_q = 0.0

            r1_list.append(rec1)
            r3_list.append(rec3)
            r5_list.append(rec5)
            mrr_list.append(mrr)
            ndcg_list.append(ndcg)
            tva_list.append(tva_q)

            if loai in theo_nhom_r3:
                theo_nhom_r3[loai].append(rec3)
                theo_nhom_mrr[loai].append(mrr)
                theo_nhom_tva[loai].append(tva_q)

            if (not du_doan_tu_choi) and (rec3 < 1.0 or tva_q < 1.0):
                ly_do_loi = (
                    "sai_hieu_luc_thoi_gian"
                    if tva_q < 1.0
                    else ("thieu_cau_noi_da_buoc" if loai == "T3" else "bo_sot_top3")
                )
                phan_tich_loi.append(
                    {
                        "id": str(q.get("id", "")),
                        "loai": loai,
                        "nguyen_nhan": ly_do_loi,
                        "ky_vong": gold_ids,
                        "top3_du_doan": pred_ids[:3],
                    }
                )

            if loai == "T2" and q.get("nhom_tuong_phan"):
                nhom_key = str(q["nhom_tuong_phan"])
                dat_rank1 = bool(pred_ids and pred_ids[0] in gold_set)
                nhom_tuong_phan.setdefault(nhom_key, []).append(dat_rank1)

            if loai == "T3":
                mh_ok = 1.0 if gold_set.issubset(set(pred_ids[:3])) else 0.0
                mh_list.append(mh_ok)

            if loai == "T4" and q.get("loai_con") == "bay_het_hieu_luc":
                sh_bay = {str(x) for x in q.get("so_hieu_het_hl_bay", [])}
                da_phat_hien_bay = bool(sh_bay & set(kq.get("van_ban_het_hieu_luc_trong_hoi", [])))
                top3_khong_dinh_bay = all(str(item.get("so_hieu", "")) not in sh_bay for item in top3_items)
                dat_thay_the = bool(gold_set & set(pred_ids[:3]))
                trap_ok = 1.0 if (da_phat_hien_bay and top3_khong_dinh_bay and dat_thay_the) else 0.0
                trap_list.append(trap_ok)

        so_cap_tp = len(nhom_tuong_phan)
        so_cap_dat = sum(1 for flags in nhom_tuong_phan.values() if len(flags) >= 2 and all(flags))
        tca = (so_cap_dat / so_cap_tp) if so_cap_tp > 0 else 0.0

        prec_abs = tp_abs / (tp_abs + fp_abs) if (tp_abs + fp_abs) > 0 else 0.0
        rec_abs = tp_abs / (tp_abs + fn_abs) if (tp_abs + fn_abs) > 0 else 0.0
        f1_abs = (2.0 * prec_abs * rec_abs / (prec_abs + rec_abs)) if (prec_abs + rec_abs) > 0 else 0.0

        def _tb(vals: Sequence[float]) -> float:
            return round(float(np.mean(vals)), 4) if vals else 0.0

        tom_tat = {
            "che_do": che_do,
            "ten_hien_thi": self.TEN_CAU_HINH.get(che_do, che_do),
            "recall_at_1": _tb(r1_list),
            "recall_at_3": _tb(r3_list),
            "recall_at_5": _tb(r5_list),
            "mrr_at_10": _tb(mrr_list),
            "ndcg_at_10": _tb(ndcg_list),
            "tva": _tb(tva_list),
            "tca": round(tca, 4),
            "so_cap_tca_dat": f"{so_cap_dat}/{so_cap_tp}",
            "multi_hop_bridge_recall": _tb(mh_list),
            "trap_correction_rate": _tb(trap_list),
            "colloquial_recall_at_3": _tb(theo_nhom_r3["T5"]),
            "colloquial_mrr_at_10": _tb(theo_nhom_mrr["T5"]),
            "citation_faithfulness": _tb(faith_list),
            "numeric_consistency": _tb(numeric_list),
            "temporal_conflict_rate": _tb(conflict_list),
            "abstention_precision": round(prec_abs, 4),
            "abstention_recall": round(rec_abs, 4),
            "abstention_f1": round(f1_abs, 4),
            "do_tre_trung_binh_ms": round(float(np.mean(latency_list)), 3) if latency_list else 0.0,
            "so_ca_loi": len(phan_tich_loi),
            "phan_tich_loi": phan_tich_loi[:30],
            "theo_nhom": {
                nhom: {
                    "so_cau": len(theo_nhom_r3[nhom]),
                    "recall_at_3": _tb(theo_nhom_r3[nhom]),
                    "mrr_at_10": _tb(theo_nhom_mrr[nhom]),
                    "tva": _tb(theo_nhom_tva[nhom]),
                }
                for nhom in ("T1", "T2", "T3", "T4", "T5")
            },
        }
        chi_tiet_vector = {
            "recall_at_3": r3_list,
            "mrr_at_10": mrr_list,
            "tva": tva_list,
            "multi_hop_bridge_recall": mh_list,
        }
        return tom_tat, chi_tiet_vector

    def chay_ablation(
        self,
        cac_che_do: Optional[Sequence[str]] = None,
        nhanh: bool = False,
        seed: int = 42,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """Chạy thí nghiệm đối chứng đầy đủ 5 cấu hình kèm kiểm định Paired Bootstrap."""
        ds_che_do = list(cac_che_do or BoTruyHoiPhapLyThoiGian.CAC_CHE_DO_HOP_LE)
        tap_qa = list(self.du_lieu_qa)
        if nhanh:
            gioi_han = {"T1": 6, "T2": 8, "T3": 6, "T5": 6}
            dem: Dict[str, int] = {}
            dem_t4_con: Dict[str, int] = {}
            loc: List[Dict[str, Any]] = []
            for q in tap_qa:
                loai = str(q.get("loai", "T1"))
                if loai == "T4":
                    loai_con = str(q.get("loai_con", "khac"))
                    if dem_t4_con.get(loai_con, 0) < 3:
                        loc.append(q)
                        dem_t4_con[loai_con] = dem_t4_con.get(loai_con, 0) + 1
                elif dem.get(loai, 0) < gioi_han.get(loai, 4):
                    loc.append(q)
                    dem[loai] = dem.get(loai, 0) + 1
            tap_qa = loc

        kq_tqpa = self.danh_gia_boc_tach_thoi_gian(tap_qa)
        ket_qua_cau_hinh: Dict[str, Dict[str, Any]] = {}
        vector_theo_cau_hinh: Dict[str, Dict[str, List[float]]] = {}

        for mode in ds_che_do:
            tom_tat, vecs = self.danh_gia_mot_cau_hinh(mode, tap_qa)
            ket_qua_cau_hinh[mode] = tom_tat
            vector_theo_cau_hinh[mode] = vecs

        kiem_dinh_thong_ke: Dict[str, Dict[str, Any]] = {}
        if "full" in vector_theo_cau_hinh:
            vec_full = vector_theo_cau_hinh["full"]
            for base_mode in ds_che_do:
                if base_mode == "full":
                    continue
                vec_base = vector_theo_cau_hinh[base_mode]
                kiem_dinh_thong_ke[f"full_vs_{base_mode}"] = {
                    chi_so: kiem_dinh_bootstrap_cap(
                        vec_full[chi_so],
                        vec_base[chi_so],
                        n_bootstrap=200 if nhanh else n_bootstrap,
                        seed=seed,
                    )
                    for chi_so in ("recall_at_3", "mrr_at_10", "tva", "multi_hop_bridge_recall")
                }

        dem_loai: Dict[str, int] = {}
        for q in tap_qa:
            loai = str(q.get("loai", "T1"))
            dem_loai[loai] = dem_loai.get(loai, 0) + 1

        return {
            "ten_bo_chuan": "ViLegal-TempBench",
            "phien_ban_bo_chuan": "2.0",
            "mo_hinh_sinh": self.ten_mo_hinh_llm,
            "bo_nhung": self.ten_bo_nhung,
            "bo_rerank": self.ten_bo_rerank,
            "ngay_mac_dinh": self.ngay_mac_dinh,
            "seed": int(seed),
            "che_do_nhanh": bool(nhanh),
            "thong_ke_du_lieu": {
                "tong_dieu_khoan_chi_muc": len(self.du_lieu_dieu_khoan),
                "so_canh_do_thi_dieu_khoan": len(self.du_lieu_canh),
                "tong_cau_hoi_danh_gia": len(tap_qa),
                "phan_bo_nhom_cau_hoi": dem_loai,
            },
            "boc_tach_thoi_gian_tqpa": kq_tqpa,
            "ablation": ket_qua_cau_hinh,
            "kiem_dinh_bootstrap": kiem_dinh_thong_ke,
        }

    def xuat_bao_cao_markdown(self, ket_qua: Dict[str, Any]) -> str:
        """Xuất báo cáo khoa học Markdown đầy đủ gồm bảng Ablation, phân tách T1–T5, Bootstrap và Error Analysis."""
        tk = ket_qua["thong_ke_du_lieu"]
        tqpa = ket_qua["boc_tach_thoi_gian_tqpa"]
        so_ch = len(ket_qua["ablation"])
        lines = [
            f"# Báo Cáo Thực Nghiệm ViLegal-TempBench v{ket_qua.get('phien_ban_bo_chuan', '2.0')} (Temporal & Graph-Grounded Legal RAG)",
            "",
            f"- **Kho chỉ mục pháp lý**: `{tk['tong_dieu_khoan_chi_muc']}` điều khoản | `{tk['so_canh_do_thi_dieu_khoan']}` cạnh đồ thị điều khoản có hiệu lực thời gian.",
            f"- **Tập đánh giá**: `{tk['tong_cau_hoi_danh_gia']}` truy vấn ({tk['phan_bo_nhom_cau_hoi']}) | Bộ nhúng: `{ket_qua.get('bo_nhung', 'cpu_lexicon_hash')}` | Reranker: `{ket_qua.get('bo_rerank', 'cross_interaction_cpu')}` | Mô hình sinh: `{ket_qua.get('mo_hinh_sinh', 'extractive-grounded')}` | Ngày mặc định: `{ket_qua['ngay_mac_dinh']}` | Seed: `{ket_qua['seed']}`.",
            f"- **Độ chính xác bóc tách mốc thời gian tự động (TQPA)**: `{tqpa['do_chinh_xac_tqpa'] * 100:.2f}%` ({tqpa['so_cau_dung_vung_hieu_luc']}/{tqpa['tong_cau_hoi']}).",
            "",
            f"## 1. Bảng Đối Chứng {so_ch} Cấu Hình (Main Ablation Study)",
            "",
            "| Cấu hình | R@1 | R@3 | R@5 | MRR@10 | nDCG@10 | TVA | TCA (T2) | Multi-Hop (T3) | Trap (T4) | Colloq R@3 (T5) | Faith. | Temp.Conf ↓ | Abst. F1 | Latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, row in ket_qua["ablation"].items():
            lines.append(
                f"| `{mode}` ({row['ten_hien_thi']}) "
                f"| {row['recall_at_1']:.3f} "
                f"| {row['recall_at_3']:.3f} "
                f"| {row['recall_at_5']:.3f} "
                f"| {row['mrr_at_10']:.3f} "
                f"| {row['ndcg_at_10']:.3f} "
                f"| {row['tva']:.3f} "
                f"| {row['tca']:.3f} ({row['so_cap_tca_dat']}) "
                f"| {row['multi_hop_bridge_recall']:.3f} "
                f"| {row['trap_correction_rate']:.3f} "
                f"| {row.get('colloquial_recall_at_3', 0.0):.3f} "
                f"| {row['citation_faithfulness']:.3f} "
                f"| {row['temporal_conflict_rate']:.3f} "
                f"| {row['abstention_f1']:.3f} "
                f"| {row['do_tre_trung_binh_ms']:.2f} |"
            )

        lines.extend(
            [
                "",
                "## 2. Kiểm Định Ý Nghĩa Thống Kê (Paired Bootstrap vs `full`)",
                "",
                "| So sánh | Δ Recall@3 (p-value) | Δ MRR@10 (p-value) | Δ TVA (p-value) | Δ Multi-Hop (p-value) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for comp, metrics in ket_qua.get("kiem_dinh_bootstrap", {}).items():
            def _fmt(m: Dict[str, Any]) -> str:
                sao = "*" if m["co_y_nghia_p05"] else ""
                return f"{m['delta_trung_binh']:+.4f} (p={m['p_value']:.4f}{sao})"

            lines.append(
                f"| `{comp}` "
                f"| {_fmt(metrics['recall_at_3'])} "
                f"| {_fmt(metrics['mrr_at_10'])} "
                f"| {_fmt(metrics['tva'])} "
                f"| {_fmt(metrics['multi_hop_bridge_recall'])} |"
            )

        full_row = ket_qua.get("ablation", {}).get("full")
        if isinstance(full_row, dict):
            ds_loi = full_row.get("phan_tich_loi") or []
            lines.extend(["", "## 3. Phân Tích Lỗi Tự Động (Error Analysis — `full`)", ""])
            if not ds_loi:
                lines.append("- Không ghi nhận ca sai lệch nào ở cấu hình `full` trên tập đánh giá hiện tại.")
            else:
                lines.append("| Mã câu | Nhóm | Nguyên nhân | Kỳ vọng | Top-3 dự đoán |")
                lines.append("|---|---|---|---|---|")
                for err in ds_loi[:15]:
                    lines.append(
                        f"| `{err['id']}` | `{err['loai']}` | `{err['nguyen_nhan']}` "
                        f"| `{', '.join(err['ky_vong'])}` | `{', '.join(err['top3_du_doan'])}` |"
                    )
        return "\n".join(lines)

    def xuat_bang_latex(self, ket_qua: Dict[str, Any]) -> str:
        """Xuất bảng LaTeX chuẩn `booktabs` sẵn sàng đưa vào bài báo ACL/EMNLP/SIGIR/VLSP."""
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Ablation study of \textsc{ViLegal-TemporalRAG} on \textsc{ViLegal-TempBench v2} (CPU-only reproducible evaluation).}",
            r"\label{tab:vilegal_tempbench_ablation}",
            r"\begin{tabular}{lccccccccccc}",
            r"\toprule",
            r"Configuration & R@1 & R@3 & MRR@10 & nDCG@10 & TVA & TCA & MH-Rec & Trap & T5-R@3 & Faith. & Abst. F1 \\",
            r"\midrule",
        ]
        for mode, row in ket_qua["ablation"].items():
            ten = row["ten_hien_thi"].replace("&", r"\&").replace("_", r"\_")
            t5_r3 = float(row.get("colloquial_recall_at_3", 0.0))
            if mode == "full":
                lines.append(
                    f"\\textbf{{{ten}}} & \\textbf{{{row['recall_at_1']:.3f}}} & \\textbf{{{row['recall_at_3']:.3f}}} "
                    f"& \\textbf{{{row['mrr_at_10']:.3f}}} & \\textbf{{{row['ndcg_at_10']:.3f}}} "
                    f"& \\textbf{{{row['tva']:.3f}}} & \\textbf{{{row['tca']:.3f}}} "
                    f"& \\textbf{{{row['multi_hop_bridge_recall']:.3f}}} & \\textbf{{{row['trap_correction_rate']:.3f}}} "
                    f"& \\textbf{{{t5_r3:.3f}}} "
                    f"& \\textbf{{{row['citation_faithfulness']:.3f}}} & \\textbf{{{row['abstention_f1']:.3f}}} \\\\"
                )
            else:
                lines.append(
                    f"{ten} & {row['recall_at_1']:.3f} & {row['recall_at_3']:.3f} "
                    f"& {row['mrr_at_10']:.3f} & {row['ndcg_at_10']:.3f} "
                    f"& {row['tva']:.3f} & {row['tca']:.3f} "
                    f"& {row['multi_hop_bridge_recall']:.3f} & {row['trap_correction_rate']:.3f} "
                    f"& {t5_r3:.3f} "
                    f"& {row['citation_faithfulness']:.3f} & {row['abstention_f1']:.3f} \\\\"
                )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
            ]
        )
        return "\n".join(lines)


ViLegalTemporalBenchmark = BoDanhGiaLegalTemporalRAG

__all__ = [
    "tinh_ndcg_at_k",
    "kiem_dinh_bootstrap_cap",
    "tao_ham_sinh_llm",
    "BoDanhGiaLegalTemporalRAG",
    "ViLegalTemporalBenchmark",
]
