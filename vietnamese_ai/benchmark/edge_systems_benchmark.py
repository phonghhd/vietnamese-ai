"""BoDanhGiaHeThongEdge (ViEdgeSystemsBenchmark) - Bộ Đo Lường Chuẩn Học Thuật Cho Edge AI Không GPU (Systems for AI).

Đo lường định lượng tái lập 100% trên CPU cho 3 hạng mục hệ thống cốt lõi:
1. Truy hồi Vector Lượng tử hóa & Native Rust HNSW (`FP32`, `SQ8 4x`, `Binary 1-bit POPCNT 32x`,
   `Binary+SQ8 Two-Stage Rescore`, `Rust HNSW-SQ8`, tùy chọn đối đầu trực tiếp `faiss`/`hnswlib`):
   đo QPS đơn/lô, độ trễ `p50`/`p95`/`p99`, dung lượng RAM (MB), tỷ lệ nén và `Recall@10`.
2. BitNet 1.58-bit đóng gói 2-bit thực (`4 trọng số/byte`, nén 16x RAM, kernel Rust cộng/trừ không phép nhân)
   so với FP32 & đường tham chiếu NumPy, đo tốc độ và sai số đối chiếu tuyệt đối.
3. Chuỗi suy luận Tiếng Việt Zero-GPU đầu-cuối (Tokenizer FST/Viterbi Rust → `.vaidb` HNSW-SQ8 bền vững → BitNet 2-bit).
"""

from __future__ import annotations

import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vietnamese_ai.compression.extreme import (
    co_rust_bitnet,
    dong_goi_bitnet_158,
    nhan_ma_tran_bitnet_158,
)
from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc
from vietnamese_ai.rag.quantized_index import ChiMucVectorLuongTu, co_rust_luong_tu
from vietnamese_ai.rag.vaidb_store import KhoTriThucVAIDB
from vietnamese_ai.utils.feature_status import TrangThaiTinhNang, gan_nhan_tinh_nang


def _sinh_du_lieu_vector_ngu_nghia(
    n_vectors: int,
    dim: int,
    n_queries: int,
    seed: int = 42,
    so_cum: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sinh tập vector phân cụm 2 tầng chuẩn hóa L2 mô phỏng không gian nhúng văn bản tiếng Việt thực tế."""
    rng = np.random.default_rng(int(seed))
    inv_sqrt_d = np.float32(1.0 / np.sqrt(float(max(1, dim))))
    k_cum = max(4, min(int(so_cum), max(4, n_vectors // 20)))
    centroids = rng.standard_normal((k_cum, dim), dtype=np.float32)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)

    k_sub = max(k_cum * 4, min(k_cum * 8, max(8, n_vectors // 5)))
    sub_parent = rng.integers(0, k_cum, size=k_sub)
    sub_centroids = centroids[sub_parent] + rng.standard_normal((k_sub, dim), dtype=np.float32) * (inv_sqrt_d * np.float32(0.55))
    sub_centroids /= np.maximum(np.linalg.norm(sub_centroids, axis=1, keepdims=True), 1e-9)

    doc_subs = rng.integers(0, k_sub, size=n_vectors)
    doc_noise = rng.standard_normal((n_vectors, dim), dtype=np.float32) * (inv_sqrt_d * np.float32(0.45))
    docs = sub_centroids[doc_subs] + doc_noise
    docs /= np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-9)

    q_subs = rng.integers(0, k_sub, size=n_queries)
    q_noise = rng.standard_normal((n_queries, dim), dtype=np.float32) * (inv_sqrt_d * np.float32(0.45))
    queries = sub_centroids[q_subs] + q_noise
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-9)

    return np.ascontiguousarray(docs, dtype=np.float32), np.ascontiguousarray(queries, dtype=np.float32)


def _tinh_ground_truth_exact(docs: np.ndarray, queries: np.ndarray, top_k: int) -> List[set[int]]:
    """Tính tập top-k chính xác tuyệt đối bằng tích ma trận FP32 làm chuẩn vàng Recall@k."""
    sims = queries @ docs.T
    k = min(int(top_k), int(docs.shape[0]))
    gt: List[set[int]] = []
    for i in range(queries.shape[0]):
        row = sims[i]
        if k >= len(row):
            top_idx = np.argsort(-row)
        else:
            part = np.argpartition(-row, k - 1)[:k]
            top_idx = part[np.argsort(-row[part])]
        gt.append({int(x) for x in top_idx})
    return gt


def _tinh_recall(preds: Sequence[Sequence[Tuple[int, float]]], gt: Sequence[set[int]], top_k: int) -> float:
    """Tính trung bình Recall@k trên toàn bộ tập truy vấn."""
    if not gt:
        return 0.0
    tong = 0.0
    for pred_row, gold in zip(preds, gt):
        if not gold:
            continue
        got = {int(idx) for idx, _ in pred_row[:top_k]}
        tong += len(got & gold) / float(len(gold))
    return round(tong / len(gt), 4)


@gan_nhan_tinh_nang(
    trang_thai=TrangThaiTinhNang.CORE,
    mo_ta=(
        "Bộ đo lường Systems-for-AI (`ViEdge-SysBench`) đánh giá truy hồi vector lượng tử hóa SQ8/Binary/HNSW Rust "
        "và nhân ma trận BitNet 1.58-bit 2-bit trên CPU không cần GPU, hỗ trợ xuất Markdown và LaTeX."
    ),
    nhom_extra="core",
)
class BoDanhGiaHeThongEdge:
    """Bộ đánh giá hiệu năng hệ thống Zero-GPU Edge CPU (`ViEdge-SysBench`)."""

    TEN_PHUONG_PHAP: Dict[str, str] = {
        "fp32_numpy": "FP32 Exact (NumPy BLAS)",
        "fp32_rust": "FP32 Exact (Rust Rayon)",
        "sq8_rust": "SQ8 8-bit (Rust 4x RAM)",
        "binary_rust": "Binary 1-bit POPCNT (Rust 32x RAM)",
        "binary_rescore_rust": "Binary+SQ8 Rescore (Rust Two-Stage)",
        "hnsw_sq8_rust": "Native Rust HNSW-SQ8 (4x RAM + O(log N))",
        "hnsw_fp32_rust": "Native Rust HNSW-FP32 (O(log N))",
    }

    def do_tim_kiem_vector_luong_tu(
        self,
        n_vectors: int = 20000,
        dim: int = 384,
        n_queries: int = 80,
        top_k: int = 10,
        seed: int = 42,
        m: int = 16,
        ef_construction: int = 80,
        ef_search: int = 64,
        oversample: int = 10,
    ) -> Dict[str, Any]:
        """Đo QPS đơn/lô, p50/p95/p99 latency, dung lượng RAM, tỷ lệ nén và Recall@10 trên 6 cấu hình."""
        docs, queries = _sinh_du_lieu_vector_ngu_nghia(n_vectors, dim, n_queries, seed=seed)
        gt = _tinh_ground_truth_exact(docs, queries, top_k=top_k)

        cau_hinh: List[Tuple[str, str, bool]] = [
            ("fp32_numpy", "fp32", False),
            ("fp32_rust", "fp32", True),
            ("sq8_rust", "sq8", True),
            ("binary_rust", "binary", True),
            ("binary_rescore_rust", "binary_rescore", True),
            ("hnsw_sq8_rust", "hnsw_sq8", True),
        ]

        ket_qua_cac_phuong_phap: Dict[str, Dict[str, Any]] = {}
        for key, mode, bat_rust in cau_hinh:
            idx = ChiMucVectorLuongTu(
                che_do=mode,
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                oversample=oversample,
                seed=seed,
                bat_rust=bat_rust,
            )
            t_build_0 = time.perf_counter()
            mem_stats = idx.xay_dung(docs)
            t_build = time.perf_counter() - t_build_0

            idx.tim_kiem(queries[0], top_k=top_k)

            latencies_ms: List[float] = []
            preds_single: List[List[Tuple[int, float]]] = []
            for qi in range(n_queries):
                t0 = time.perf_counter()
                res = idx.tim_kiem(queries[qi], top_k=top_k)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                preds_single.append(res)

            t_batch_0 = time.perf_counter()
            preds_batch = idx.tim_kiem_lo(queries, top_k=top_k)
            t_batch = max(1e-6, time.perf_counter() - t_batch_0)

            arr_lat = np.asarray(latencies_ms, dtype=np.float64)
            tong_don_s = max(1e-6, float(np.sum(arr_lat)) / 1000.0)
            recall = _tinh_recall(preds_batch if preds_batch else preds_single, gt, top_k=top_k)

            ket_qua_cac_phuong_phap[key] = {
                "phuong_phap": key,
                "ten_hien_thi": self.TEN_PHUONG_PHAP.get(key, key),
                "dung_rust": bool(mem_stats["dung_rust"]),
                "thoi_gian_xay_dung_s": round(t_build, 4),
                "vector_mb": mem_stats["vector_mb"],
                "tong_mb": mem_stats["tong_mb"],
                "ti_le_nen_vector": mem_stats["ti_le_nen_vector"],
                "p50_ms": round(float(np.percentile(arr_lat, 50)), 3),
                "p95_ms": round(float(np.percentile(arr_lat, 95)), 3),
                "p99_ms": round(float(np.percentile(arr_lat, 99)), 3),
                "qps_don": round(n_queries / tong_don_s, 1),
                "qps_lo": round(n_queries / t_batch, 1),
                "recall_at_10": recall,
            }

        doi_thu_ngoai = self._thu_doi_dau_thu_vien_ngoai(docs, queries, gt, top_k=top_k, m=m, ef_c=ef_construction, ef_s=ef_search)
        ket_qua_cac_phuong_phap.update(doi_thu_ngoai)

        return {
            "so_vector": int(n_vectors),
            "so_chieu": int(dim),
            "so_truy_van": int(n_queries),
            "top_k": int(top_k),
            "co_rust_luong_tu": co_rust_luong_tu(),
            "ket_qua": ket_qua_cac_phuong_phap,
        }

    @staticmethod
    def _thu_doi_dau_thu_vien_ngoai(
        docs: np.ndarray,
        queries: np.ndarray,
        gt: Sequence[set[int]],
        top_k: int,
        m: int,
        ef_c: int,
        ef_s: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Tự động đo đối đầu trực tiếp với `faiss` và `hnswlib` nếu môi trường có cài đặt sẵn."""
        ngoai: Dict[str, Dict[str, Any]] = {}
        n_docs, dim = int(docs.shape[0]), int(docs.shape[1])
        n_q = int(queries.shape[0])
        fp32_mb = round((n_docs * dim * 4) / (1024 * 1024), 3)

        try:
            import hnswlib  # type: ignore

            t0 = time.perf_counter()
            p_idx = hnswlib.Index(space="cosine", dim=dim)
            p_idx.init_index(max_elements=n_docs, ef_construction=ef_c, M=m)
            p_idx.add_items(docs, np.arange(n_docs))
            p_idx.set_ef(ef_s)
            t_build = time.perf_counter() - t0

            lats: List[float] = []
            for qi in range(n_q):
                ts = time.perf_counter()
                p_idx.knn_query(queries[qi], k=top_k)
                lats.append((time.perf_counter() - ts) * 1000.0)
            tb0 = time.perf_counter()
            labels, dists = p_idx.knn_query(queries, k=top_k)
            t_batch = max(1e-6, time.perf_counter() - tb0)
            preds = [[(int(labels[qi][j]), float(1.0 - dists[qi][j])) for j in range(top_k)] for qi in range(n_q)]
            arr = np.asarray(lats, dtype=np.float64)
            ngoai["hnswlib_fp32"] = {
                "phuong_phap": "hnswlib_fp32",
                "ten_hien_thi": "hnswlib C++ HNSW-FP32 (External)",
                "dung_rust": False,
                "thoi_gian_xay_dung_s": round(t_build, 4),
                "vector_mb": fp32_mb,
                "tong_mb": fp32_mb,
                "ti_le_nen_vector": 1.0,
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
                "qps_don": round(n_q / max(1e-6, float(np.sum(arr)) / 1000.0), 1),
                "qps_lo": round(n_q / t_batch, 1),
                "recall_at_10": _tinh_recall(preds, gt, top_k=top_k),
            }
        except Exception:
            pass

        try:
            import faiss  # type: ignore

            t0 = time.perf_counter()
            f_idx = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            f_idx.hnsw.efConstruction = ef_c
            f_idx.hnsw.efSearch = ef_s
            f_idx.add(docs)
            t_build = time.perf_counter() - t0

            lats_f: List[float] = []
            for qi in range(n_q):
                ts = time.perf_counter()
                f_idx.search(queries[qi : qi + 1], top_k)
                lats_f.append((time.perf_counter() - ts) * 1000.0)
            tb0 = time.perf_counter()
            scores_f, ids_f = f_idx.search(queries, top_k)
            t_batch = max(1e-6, time.perf_counter() - tb0)
            preds_f = [[(int(ids_f[qi][j]), float(scores_f[qi][j])) for j in range(top_k)] for qi in range(n_q)]
            arr_f = np.asarray(lats_f, dtype=np.float64)
            ngoai["faiss_hnsw_fp32"] = {
                "phuong_phap": "faiss_hnsw_fp32",
                "ten_hien_thi": "FAISS C++ HNSW-FP32 (External)",
                "dung_rust": False,
                "thoi_gian_xay_dung_s": round(t_build, 4),
                "vector_mb": fp32_mb,
                "tong_mb": fp32_mb,
                "ti_le_nen_vector": 1.0,
                "p50_ms": round(float(np.percentile(arr_f, 50)), 3),
                "p95_ms": round(float(np.percentile(arr_f, 95)), 3),
                "p99_ms": round(float(np.percentile(arr_f, 99)), 3),
                "qps_don": round(n_q / max(1e-6, float(np.sum(arr_f)) / 1000.0), 1),
                "qps_lo": round(n_q / t_batch, 1),
                "recall_at_10": _tinh_recall(preds_f, gt, top_k=top_k),
            }
        except Exception:
            pass

        return ngoai

    def do_bitnet_158_dong_goi_2bit(
        self,
        cac_kich_thuoc: Sequence[Tuple[int, int]] = ((512, 512), (1024, 1024), (2048, 2048)),
        n_tokens: int = 1,
        so_vong_lap: int = 12,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Đo tốc độ, độ nén 16x và độ khớp số học của BitNet 1.58-bit đóng gói 2-bit thực trên Rust CPU."""
        rng = np.random.default_rng(int(seed))
        bang_ket_qua: List[Dict[str, Any]] = []

        for in_f, out_f in cac_kich_thuoc:
            w = rng.standard_normal((out_f, in_f), dtype=np.float32) * np.float32(0.05)
            x = rng.standard_normal((n_tokens, in_f), dtype=np.float32)

            packed_rust, scale_rust = dong_goi_bitnet_158(w, bat_rust=True)
            packed_np, scale_np = dong_goi_bitnet_158(w, bat_rust=False)

            y_ref = nhan_ma_tran_bitnet_158(x, packed_np, in_f, out_f, scale_np, bat_rust=False)
            y_rust = nhan_ma_tran_bitnet_158(x, packed_rust, in_f, out_f, scale_rust, bat_rust=True)
            max_err = float(np.max(np.abs(y_ref - y_rust)))

            loops = max(3, int(so_vong_lap))
            _ = x @ w.T
            t0 = time.perf_counter()
            for _ in range(loops):
                _ = x @ w.T
            fp32_ms = (time.perf_counter() - t0) * 1000.0 / loops

            t1 = time.perf_counter()
            for _ in range(loops):
                _ = nhan_ma_tran_bitnet_158(x, packed_np, in_f, out_f, scale_np, bat_rust=False)
            bn_np_ms = (time.perf_counter() - t1) * 1000.0 / loops

            t2 = time.perf_counter()
            for _ in range(loops):
                _ = nhan_ma_tran_bitnet_158(x, packed_rust, in_f, out_f, scale_rust, bat_rust=True)
            bn_rust_ms = (time.perf_counter() - t2) * 1000.0 / loops

            fp32_bytes = in_f * out_f * 4
            packed_bytes = len(packed_rust)
            ops = 2.0 * n_tokens * in_f * out_f
            gops_rust = (ops / max(1e-9, bn_rust_ms / 1000.0)) / 1e9

            bang_ket_qua.append(
                {
                    "kich_thuoc": f"{out_f}x{in_f}",
                    "n_tokens": int(n_tokens),
                    "fp32_mb": round(fp32_bytes / (1024 * 1024), 3),
                    "packed_2bit_mb": round(packed_bytes / (1024 * 1024), 4),
                    "ti_le_nen": round(fp32_bytes / max(1, packed_bytes), 2),
                    "fp32_numpy_ms": round(fp32_ms, 4),
                    "bitnet_numpy_ms": round(bn_np_ms, 4),
                    "bitnet_rust_ms": round(bn_rust_ms, 4),
                    "tang_toc_vs_bitnet_numpy": round(bn_np_ms / max(1e-6, bn_rust_ms), 2),
                    "gops_rust": round(gops_rust, 2),
                    "sai_so_doi_chieu_max": round(max_err, 7),
                }
            )

        return {
            "co_rust_bitnet": co_rust_bitnet(),
            "n_tokens": int(n_tokens),
            "ket_qua": bang_ket_qua,
        }

    def do_chuoi_zero_gpu_dau_cuoi(self, n_docs: int = 1000, dim: int = 128, seed: int = 42) -> Dict[str, Any]:
        """Đo chuỗi xử lý đầu-cuối Zero-GPU trên CPU: Rust Tokenizer → .vaidb HNSW-SQ8 bền vững → BitNet 2-bit."""
        tok = BoTachTuTiengVietSieuToc()
        cau_hoi = "Ủy ban nhân dân tỉnh Hoà Bình ban hành quy định thời điểm lập hóa đơn điện tử năm 2025."
        t0 = time.perf_counter()
        for _ in range(200):
            tok.tach_tu(cau_hoi, chuan_hoa_dau=True, phuong_phap="viterbi")
        tok_ms = (time.perf_counter() - t0) * 1000.0 / 200.0

        docs, queries = _sinh_du_lieu_vector_ngu_nghia(n_docs, dim, 10, seed=seed)
        with tempfile.TemporaryDirectory(prefix="vai_edge_sys_") as tmp_dir:
            db_path = Path(tmp_dir) / "edge_quant.vaidb"
            store = KhoTriThucVAIDB(duong_dan=db_path, kich_thuoc=dim)
            store.chen_batch(
                [f"doc_{i}" for i in range(n_docs)],
                docs,
                [{"noi_dung": f"Điều {i + 1} quy định pháp luật mẫu"} for i in range(n_docs)],
            )
            stats_q = store.xay_dung_chi_muc_luong_tu(che_do="hnsw_sq8", m=12, ef_construction=64, ef_search=48)
            store.dong()

            t_reload_0 = time.perf_counter()
            store2 = KhoTriThucVAIDB(duong_dan=db_path, kich_thuoc=dim)
            reload_ms = (time.perf_counter() - t_reload_0) * 1000.0
            t_q0 = time.perf_counter()
            for qi in range(10):
                store2.tim_kiem(queries[qi], top_k=5)
            vaidb_query_ms = (time.perf_counter() - t_q0) * 1000.0 / 10.0
            db_kb = round(db_path.stat().st_size / 1024.0, 2)
            store2.dong()

        rng = np.random.default_rng(seed)
        w = rng.standard_normal((512, 512), dtype=np.float32) * np.float32(0.05)
        x = rng.standard_normal((1, 512), dtype=np.float32)
        packed, scale = dong_goi_bitnet_158(w, bat_rust=True)
        t_bn0 = time.perf_counter()
        for _ in range(20):
            nhan_ma_tran_bitnet_158(x, packed, 512, 512, scale, bat_rust=True)
        bitnet_ms = (time.perf_counter() - t_bn0) * 1000.0 / 20.0

        return {
            "tokenizer_viterbi_ms": round(tok_ms, 4),
            "vaidb_reload_hnsw_sq8_ms": round(reload_ms, 3),
            "vaidb_query_hnsw_sq8_ms": round(vaidb_query_ms, 3),
            "vaidb_size_kb": db_kb,
            "vaidb_ti_le_nen_vector": stats_q.get("ti_le_nen_vector", 1.0),
            "bitnet_512_decode_ms": round(bitnet_ms, 4),
            "tong_do_tre_chuoi_ms": round(tok_ms + vaidb_query_ms + bitnet_ms, 4),
        }

    def chay_toan_dien(
        self,
        nhanh: bool = False,
        n_vectors: Optional[int] = None,
        dim: int = 384,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Chạy đầy đủ bộ đo Systems-for-AI `ViEdge-SysBench` trên CPU không GPU."""
        scale = int(n_vectors) if n_vectors is not None else (3000 if nhanh else 20000)
        n_q = 30 if nhanh else 80
        dims_bn: Sequence[Tuple[int, int]] = ((512, 512), (1024, 1024)) if nhanh else ((512, 512), (1024, 1024), (2048, 2048))

        vec_res = self.do_tim_kiem_vector_luong_tu(
            n_vectors=scale,
            dim=int(dim),
            n_queries=n_q,
            top_k=10,
            seed=seed,
        )
        bitnet_res = self.do_bitnet_158_dong_goi_2bit(
            cac_kich_thuoc=dims_bn,
            n_tokens=1,
            so_vong_lap=6 if nhanh else 15,
            seed=seed,
        )
        e2e_res = self.do_chuoi_zero_gpu_dau_cuoi(
            n_docs=400 if nhanh else 1500,
            dim=128,
            seed=seed,
        )

        return {
            "ten_bo_chuan": "ViEdge-SysBench",
            "phien_ban_bo_chuan": "1.0",
            "moi_truong": {
                "he_dieu_hanh": platform.system(),
                "kien_truc": platform.machine(),
                "so_loi_cpu": os.cpu_count() or 1,
                "python": platform.python_version(),
                "co_rust_luong_tu": co_rust_luong_tu(),
                "co_rust_bitnet": co_rust_bitnet(),
            },
            "seed": int(seed),
            "che_do_nhanh": bool(nhanh),
            "truy_hoi_vector_luong_tu": vec_res,
            "bitnet_158_2bit_rust": bitnet_res,
            "chuoi_zero_gpu_dau_cuoi": e2e_res,
        }

    def xuat_bao_cao_markdown(self, ket_qua: Dict[str, Any]) -> str:
        """Xuất báo cáo khoa học Markdown cho `ViEdge-SysBench` (Vector Quantization + HNSW + BitNet 2-bit)."""
        mt = ket_qua["moi_truong"]
        vr = ket_qua["truy_hoi_vector_luong_tu"]
        bn = ket_qua["bitnet_158_2bit_rust"]
        e2e = ket_qua["chuoi_zero_gpu_dau_cuoi"]

        lines = [
            "# Báo Cáo Thực Nghiệm ViEdge-SysBench (Zero-GPU Edge AI & Rust Quantized Retrieval)",
            "",
            f"- **Môi trường CPU**: `{mt['he_dieu_hanh']} {mt['kien_truc']}` | `{mt['so_loi_cpu']}` lõi CPU | Python `{mt['python']}` | Rust Quant: `{mt['co_rust_luong_tu']}` | Rust BitNet: `{mt['co_rust_bitnet']}`.",
            f"- **Quy mô vector**: `{vr['so_vector']:,}` vector × `{vr['so_chieu']}` chiều | `{vr['so_truy_van']}` truy vấn | `top_k={vr['top_k']}` | Seed: `{ket_qua['seed']}`.",
            "",
            "## 1. Truy hồi Vector Lượng tử hóa & Native Rust HNSW",
            "",
            "| Phương pháp | Vec RAM (MB) | Tổng RAM (MB) | Nén Vec | Build (s) | p50 (ms) | p95 (ms) | p99 (ms) | QPS đơn | QPS lô | Recall@10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in vr["ket_qua"].values():
            lines.append(
                f"| {row['ten_hien_thi']} "
                f"| {row['vector_mb']:.2f} "
                f"| {row['tong_mb']:.2f} "
                f"| {row['ti_le_nen_vector']:.1f}x "
                f"| {row['thoi_gian_xay_dung_s']:.3f} "
                f"| {row['p50_ms']:.3f} "
                f"| {row['p95_ms']:.3f} "
                f"| {row['p99_ms']:.3f} "
                f"| {row['qps_don']:.1f} "
                f"| {row['qps_lo']:.1f} "
                f"| {row['recall_at_10']:.4f} |"
            )

        lines.extend(
            [
                "",
                "## 2. BitNet 1.58-bit Đóng Gói 2-bit Thực (Rust Zero-Mul Add/Sub Kernel)",
                "",
                "| Kích thước tầng | FP32 (MB) | 2-bit Packed (MB) | Nén RAM | FP32 NumPy (ms) | BitNet NumPy (ms) | BitNet Rust (ms) | Tăng tốc vs BN-NumPy | Sai số tối đa |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in bn["ket_qua"]:
            lines.append(
                f"| `{row['kich_thuoc']}` "
                f"| {row['fp32_mb']:.3f} "
                f"| {row['packed_2bit_mb']:.4f} "
                f"| {row['ti_le_nen']:.1f}x "
                f"| {row['fp32_numpy_ms']:.3f} "
                f"| {row['bitnet_numpy_ms']:.3f} "
                f"| {row['bitnet_rust_ms']:.3f} "
                f"| {row['tang_toc_vs_bitnet_numpy']:.2f}x "
                f"| {row['sai_so_doi_chieu_max']:.2e} |"
            )

        lines.extend(
            [
                "",
                "## 3. Chuỗi Suy Luận Tiếng Việt Zero-GPU Đầu-Cuối trên CPU",
                "",
                f"- **Tách từ Viterbi + FST Rust**: `{e2e['tokenizer_viterbi_ms']:.4f} ms/câu`",
                f"- **Nạp lại `.vaidb` HNSW-SQ8 từ đĩa**: `{e2e['vaidb_reload_hnsw_sq8_ms']:.3f} ms` (kích thước tệp `{e2e['vaidb_size_kb']} KB`, nén vector `{e2e['vaidb_ti_le_nen_vector']}x`)",
                f"- **Truy vấn `.vaidb` HNSW-SQ8**: `{e2e['vaidb_query_hnsw_sq8_ms']:.3f} ms/truy vấn`",
                f"- **Suy luận tầng BitNet 1.58-bit 2-bit (`512x512`)**: `{e2e['bitnet_512_decode_ms']:.4f} ms`",
                f"- **Tổng độ trễ chuỗi đầu-cuối**: **`{e2e['tong_do_tre_chuoi_ms']:.4f} ms`**",
            ]
        )
        return "\n".join(lines)

    def xuat_bang_latex(self, ket_qua: Dict[str, Any]) -> str:
        """Xuất bảng LaTeX chuẩn `booktabs` cho bài báo MLSys / Efficient NLP."""
        vr = ket_qua["truy_hoi_vector_luong_tu"]
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            f"\\caption{{Zero-GPU CPU vector retrieval and quantization benchmark on \\textsc{{ViEdge-SysBench}} ({vr['so_vector']:,} vectors, $D={vr['so_chieu']}$).}}",
            r"\label{tab:viedge_sysbench}",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            r"Method & Vec RAM (MB) & Comp. & p50 (ms) & p95 (ms) & QPS (Single) & QPS (Batch) & Recall@10 \\",
            r"\midrule",
        ]
        for key, row in vr["ket_qua"].items():
            ten = row["ten_hien_thi"].replace("_", r"\_").replace("+", "{+}")
            if key == "hnsw_sq8_rust":
                lines.append(
                    f"\\textbf{{{ten}}} & \\textbf{{{row['vector_mb']:.2f}}} & \\textbf{{{row['ti_le_nen_vector']:.1f}$\\times$}} "
                    f"& \\textbf{{{row['p50_ms']:.3f}}} & \\textbf{{{row['p95_ms']:.3f}}} "
                    f"& \\textbf{{{row['qps_don']:.1f}}} & \\textbf{{{row['qps_lo']:.1f}}} & \\textbf{{{row['recall_at_10']:.4f}}} \\\\"
                )
            else:
                lines.append(
                    f"{ten} & {row['vector_mb']:.2f} & {row['ti_le_nen_vector']:.1f}$\\times$ "
                    f"& {row['p50_ms']:.3f} & {row['p95_ms']:.3f} "
                    f"& {row['qps_don']:.1f} & {row['qps_lo']:.1f} & {row['recall_at_10']:.4f} \\\\"
                )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
            ]
        )
        return "\n".join(lines)


ViEdgeSystemsBenchmark = BoDanhGiaHeThongEdge

__all__ = [
    "BoDanhGiaHeThongEdge",
    "ViEdgeSystemsBenchmark",
]
