"""Kiểm thử toàn diện ViEdge-RustEngine: SQ8, Binary POPCNT, Rust HNSW, BitNet 2-bit T-MAC LUT và ViEdge-SysBench."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from vietnamese_ai.benchmark.edge_systems_benchmark import BoDanhGiaHeThongEdge
from vietnamese_ai.cli.main import cmd_benchmark
from vietnamese_ai.compression.extreme import (
    BitLinear,
    co_rust_bitnet,
    dong_goi_bitnet_158,
    nhan_ma_tran_bitnet_158,
)
from vietnamese_ai.rag.quantized_index import ChiMucVectorLuongTu, co_rust_luong_tu
from vietnamese_ai.rag.vaidb_store import KhoTriThucVAIDB


def _sinh_tap_cum(n: int = 600, dim: int = 128, q: int = 20, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inv_sqrt_d = np.float32(1.0 / np.sqrt(float(dim)))
    k = 16
    centers = rng.standard_normal((k, dim), dtype=np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
    dc = rng.integers(0, k, size=n)
    docs = centers[dc] + rng.standard_normal((n, dim), dtype=np.float32) * (inv_sqrt_d * np.float32(0.45))
    docs /= np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-9)
    qc = rng.integers(0, k, size=q)
    queries = centers[qc] + rng.standard_normal((q, dim), dtype=np.float32) * (inv_sqrt_d * np.float32(0.45))
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-9)
    return np.ascontiguousarray(docs, dtype=np.float32), np.ascontiguousarray(queries, dtype=np.float32)


def _exact_top_k(docs: np.ndarray, query: np.ndarray, k: int = 10) -> set[int]:
    sims = docs @ query
    idx = np.argsort(-sims)[:k]
    return {int(i) for i in idx}


def test_sq8_nen_4x_va_doi_chieu_rust_numpy() -> None:
    docs, queries = _sinh_tap_cum(n=500, dim=128, q=15, seed=7)
    idx_np = ChiMucVectorLuongTu(che_do="sq8", bat_rust=False)
    stats_np = idx_np.xay_dung(docs)
    assert stats_np["ti_le_nen_vector"] >= 3.8

    idx_rs = ChiMucVectorLuongTu(che_do="sq8", bat_rust=True)
    stats_rs = idx_rs.xay_dung(docs)
    assert stats_rs["ti_le_nen_vector"] >= 3.8
    assert stats_rs["dung_rust"] == co_rust_luong_tu()

    recall_sum = 0.0
    for qi in range(len(queries)):
        r_np = idx_np.tim_kiem(queries[qi], top_k=10)
        r_rs = idx_rs.tim_kiem(queries[qi], top_k=10)
        assert len(r_rs) == 10
        assert r_rs[0][0] == r_np[0][0]
        assert abs(r_rs[0][1] - r_np[0][1]) < 1e-3
        gold = _exact_top_k(docs, queries[qi], k=10)
        got = {i for i, _ in r_rs}
        recall_sum += len(got & gold) / 10.0

    assert recall_sum / len(queries) >= 0.94


def test_binary_popcnt_32x_va_two_stage_rescore() -> None:
    docs, queries = _sinh_tap_cum(n=500, dim=128, q=15, seed=11)

    idx_bin_np = ChiMucVectorLuongTu(che_do="binary", bat_rust=False)
    st_bin = idx_bin_np.xay_dung(docs)
    assert st_bin["ti_le_nen_vector"] == 32.0

    idx_bin_rs = ChiMucVectorLuongTu(che_do="binary", bat_rust=True)
    idx_bin_rs.xay_dung(docs)

    for qi in range(5):
        a = idx_bin_np.tim_kiem(queries[qi], top_k=10)
        b = idx_bin_rs.tim_kiem(queries[qi], top_k=10)
        assert abs(a[0][1] - b[0][1]) < 1e-5

    idx_two = ChiMucVectorLuongTu(che_do="binary_rescore", oversample=16, bat_rust=True)
    st_two = idx_two.xay_dung(docs)
    assert st_two["ti_le_nen_vector"] >= 3.3

    recall_sum = 0.0
    for qi in range(len(queries)):
        gold = _exact_top_k(docs, queries[qi], k=10)
        got = {i for i, _ in idx_two.tim_kiem(queries[qi], top_k=10)}
        recall_sum += len(got & gold) / 10.0

    assert recall_sum / len(queries) >= 0.90


@pytest.mark.parametrize("che_do", ["hnsw_sq8", "hnsw_fp32"])
def test_hnsw_native_va_tuan_tu_hoa_vqi1(che_do: str) -> None:
    docs, queries = _sinh_tap_cum(n=600, dim=96, q=16, seed=19)
    idx = ChiMucVectorLuongTu(
        che_do=che_do,
        m=16,
        ef_construction=80,
        ef_search=64,
        seed=19,
        bat_rust=True,
    )
    stats = idx.xay_dung(docs)
    if che_do == "hnsw_sq8":
        assert stats["ti_le_nen_vector"] >= 3.7

    batch_res = idx.tim_kiem_lo(queries, top_k=10)
    assert len(batch_res) == len(queries)

    recall_sum = 0.0
    for qi in range(len(queries)):
        single_res = idx.tim_kiem(queries[qi], top_k=10)
        assert [i for i, _ in single_res] == [i for i, _ in batch_res[qi]]
        gold = _exact_top_k(docs, queries[qi], k=10)
        got = {i for i, _ in single_res}
        recall_sum += len(got & gold) / 10.0

    assert recall_sum / len(queries) >= 0.90

    raw = idx.luu_bytes()
    assert raw[:4] == b"VQI1"
    restored = ChiMucVectorLuongTu.tai_bytes(raw)
    assert restored.che_do == che_do
    assert restored.so_vector == len(docs)
    assert restored.so_chieu == 96

    for qi in range(6):
        r1 = idx.tim_kiem(queries[qi], top_k=10)
        r2 = restored.tim_kiem(queries[qi], top_k=10)
        assert [i for i, _ in r1] == [i for i, _ in r2]
        assert abs(r1[0][1] - r2[0][1]) < 1e-5


def test_vaidb_chi_muc_luong_tu_ben_vung_va_dirty_invalidation(tmp_path: Path) -> None:
    db_path = tmp_path / "edge_quant_test.vaidb"
    rng = np.random.default_rng(101)
    dim = 64
    n_docs = 80
    vecs = rng.standard_normal((n_docs, dim), dtype=np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)

    store = KhoTriThucVAIDB(duong_dan=db_path)
    for i in range(n_docs):
        store.chen(
            f"doc_{i:03d}",
            vecs[i],
            {
                "noi_dung": f"Văn bản quy phạm pháp luật mẫu số {i:03d} về bảo vệ dữ liệu và trí tuệ nhân tạo.",
                "ma_so": f"VB-{i:03d}",
            },
        )

    kq_build = store.xay_dung_chi_muc_luong_tu(che_do="hnsw_sq8", luu_xuong_dia=True)
    assert kq_build["so_vector"] == n_docs
    assert kq_build["ti_le_nen_vector"] >= 3.5
    tk = store.thong_ke()
    assert tk["chi_muc_luong_tu"] is not None
    assert tk["chi_muc_luong_tu"]["che_do"] == "hnsw_sq8"
    store.dong()

    store2 = KhoTriThucVAIDB(duong_dan=db_path)
    assert store2._quant_index is not None
    assert store2._quant_index.so_vector == n_docs
    hits = store2.tim_kiem(vecs[17], top_k=5)
    assert hits and hits[0]["ma"] == "doc_017"

    extra_vec = np.zeros(dim, dtype=np.float32)
    extra_vec[0] = 1.0
    extra_vec[1] = 0.25
    extra_vec /= np.linalg.norm(extra_vec)
    store2.chen(
        "doc_moi_nhat",
        extra_vec,
        {"noi_dung": "Tài liệu mới chèn sau khi dựng chỉ mục lượng tử hóa."},
    )
    assert store2._quant_dirty is True
    hits_new = store2.tim_kiem(extra_vec, top_k=3)
    assert any(h["ma"] == "doc_moi_nhat" for h in hits_new)
    store2.dong()


@pytest.mark.parametrize(
    ("n_rows", "in_f", "out_f"),
    [
        (1, 64, 64),
        (8, 128, 96),
        (4, 130, 45),
        (1, 17, 19),
    ],
)
def test_bitnet_158_2bit_tmac_lut_rust_vs_numpy_parity(n_rows: int, in_f: int, out_f: int) -> None:
    rng = np.random.default_rng(in_f * 13 + out_f)
    w = rng.standard_normal((out_f, in_f), dtype=np.float32)
    x = rng.standard_normal((n_rows, in_f), dtype=np.float32)

    packed_np, scale_np = dong_goi_bitnet_158(w, bat_rust=False)
    packed_rs, scale_rs = dong_goi_bitnet_158(w, bat_rust=True)
    assert len(packed_rs) == out_f * ((in_f + 3) // 4)
    assert abs(scale_np - scale_rs) < 1e-5
    if in_f % 4 == 0:
        assert (w.nbytes / len(packed_rs)) == pytest.approx(16.0, rel=1e-6)

    y_np = nhan_ma_tran_bitnet_158(x, packed_np, in_f, out_f, scale_np, bat_rust=False)
    y_rs = nhan_ma_tran_bitnet_158(x, packed_rs, in_f, out_f, scale_rs, bat_rust=True)
    assert y_rs.shape == (n_rows, out_f)
    assert float(np.max(np.abs(y_np - y_rs))) < 1e-4

    if co_rust_bitnet():
        assert packed_np == packed_rs


def test_bitlinear_api_2bit_va_tuong_thich_nguoc() -> None:
    layer = BitLinear(64, 32)
    packed_legacy = layer.dong_goi_2bit()
    assert len(packed_legacy) == (64 * 32) // 4

    x = np.random.default_rng(5).standard_normal((3, 64)).astype(np.float32)
    y_2bit = layer.suy_luan_2bit(x)
    assert y_2bit.shape == (3, 32)
    tk = layer.thong_ke_nen()
    assert tk["ti_le_nen"] == pytest.approx(16.0, rel=1e-6)
    assert tk["packed_2bit_bytes"] == 32 * 16


def test_viedge_sysbench_va_cli_edge_systems(tmp_path: Path) -> None:
    bench = BoDanhGiaHeThongEdge()
    kq = bench.chay_toan_dien(nhanh=True, n_vectors=320, dim=64, seed=42)
    assert "truy_hoi_vector_luong_tu" in kq
    assert "bitnet_158_2bit_rust" in kq
    assert "chuoi_zero_gpu_dau_cuoi" in kq
    assert kq["truy_hoi_vector_luong_tu"]["ket_qua"]["sq8_rust"]["ti_le_nen_vector"] >= 3.5
    assert kq["truy_hoi_vector_luong_tu"]["ket_qua"]["binary_rust"]["ti_le_nen_vector"] == 32.0
    assert kq["bitnet_158_2bit_rust"]["ket_qua"][0]["ti_le_nen"] == 16.0

    md = bench.xuat_bao_cao_markdown(kq)
    assert "ViEdge-SysBench" in md
    assert "BitNet 1.58-bit" in md

    tex = bench.xuat_bang_latex(kq)
    assert "\\begin{table*" in tex
    assert "\\toprule" in tex
    assert "\\bottomrule" in tex

    out_json = tmp_path / "edge_bench.json"
    args = types.SimpleNamespace(
        suite="edge-systems",
        compare=False,
        latex=False,
        quick=True,
        seed=42,
        scale=240,
        dim=64,
        output=str(out_json),
    )
    kq_cli = cmd_benchmark(args)
    assert out_json.is_file()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["ten_bo_chuan"] == kq_cli["ten_bo_chuan"]
