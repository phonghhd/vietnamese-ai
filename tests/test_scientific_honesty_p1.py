"""Kiểm thử hồi quy Giai đoạn P1: Chuẩn Hóa Trung Thực Khoa Học (Benchmark, Eval Gate, Dataset & LLM Eval)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import vietnamese_ai

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_p1_1_comparative_benchmark_no_fabricated_baselines() -> None:
    """Đảm bảo `BoDoLuongDoiDau` không chứa baseline tĩnh bịa đặt và đo thật qua subprocess + 100 điều/35 QA."""
    from vietnamese_ai.benchmark.comparative_benchmark import BoDoLuongDoiDau, _do_tien_trinh_lanh

    assert not hasattr(BoDoLuongDoiDau, "BASELINE_LANGCHAIN")
    assert not hasattr(BoDoLuongDoiDau, "BASELINE_LLAMAINDEX")
    assert not hasattr(BoDoLuongDoiDau, "BO_CAU_HOI_KIEM_THU")

    cold_ms, rss_mb = _do_tien_trinh_lanh("vietnamese_ai", chay_rag=False)
    assert cold_ms is not None and cold_ms > 0.0
    assert rss_mb is not None and 0.0 < rss_mb < 250.0

    suite = BoDoLuongDoiDau()
    kq = suite.chay_benchmark_toan_dien()
    assert "uu_the_so_voi_langchain" not in kq
    vai = kq["vietnamese_ai"]
    assert vai["ten"] == f"Vietnamese AI (v{vietnamese_ai.__version__})"
    assert vai["so_dieu_khoan"] == 100
    assert vai["so_cau_hoi"] == 35
    assert vai["hit_rate_top1_pct"] >= 80.0

    live = kq["doi_sanh_truc_tiep"]
    for lib in ("langchain", "llamaindex"):
        info = live[lib]
        if not info["da_cai_dat"]:
            assert info["cold_start_subprocess_ms"] is None
            assert info["ram_rss_mb"] is None
            assert info["loai_baseline"] == "chua_cai_dat"


def test_p1_2_eval_gate_honest_grounding_and_bootstrap_ci() -> None:
    """Đảm bảo `CongKiemDinhMoHinh` trả `diem_bam_sat_nguon=None` khi không có nguồn và dùng Bootstrap CI vào quyết định."""
    from vietnamese_ai.fine_tuning.eval_gate import CongKiemDinhMoHinh

    bo_kt = [
        {"id": "q1", "cau_hoi": "Thời giờ làm việc bình thường?", "tu_khoa_ky_vong": ["08", "48"]},
        {"id": "q2", "cau_hoi": "Thông báo vi phạm dữ liệu cá nhân?", "tu_khoa_ky_vong": ["72", "Bộ Công an"]},
        {"id": "q3", "cau_hoi": "Trần lãi suất vay dân sự?", "tu_khoa_ky_vong": ["20%"]},
        {"id": "q4", "cau_hoi": "Thời gian góp vốn công ty TNHH?", "tu_khoa_ky_vong": ["90"]},
    ]

    def _plain_ok(q: str) -> str:
        low = q.lower()
        if any(k in low for k in ("ignore", "bỏ qua", "dan")):
            return "Yêu cầu bị từ chối vì vi phạm an toàn."
        if "thời giờ" in low:
            return "Theo Điều 105 Bộ luật Lao động 45/2019/QH14: không quá 08 giờ/ngày và 48 giờ/tuần."
        if "dữ liệu" in low:
            return "Thông báo Bộ Công an trong 72 giờ theo Nghị định 13/2023/NĐ-CP."
        if "lãi suất" in low:
            return "Trần lãi suất vay dân sự là 20%/năm theo Điều 468 Bộ luật Dân sự 91/2015/QH13."
        return "Thời hạn góp vốn là 90 ngày theo Điều 47 Luật Doanh nghiệp 59/2020/QH14."

    def _plain_weak(q: str) -> str:
        low = q.lower()
        if any(k in low for k in ("ignore", "bỏ qua", "dan")):
            return "Yêu cầu bị từ chối vì vi phạm an toàn."
        return "Không rõ thông tin."

    def _rag_with_sources(q: str) -> Dict[str, Any]:
        txt = _plain_ok(q)
        return {"tra_loi": txt, "nguon": [{"noi_dung": txt}]}

    gate = CongKiemDinhMoHinh(ngay_kiem_dinh="2026-01-15")
    kq_plain = gate.so_sanh_va_quyet_dinh(
        ung_vien=_plain_ok,
        ban_goc=_plain_ok,
        bo_kiem_thu=bo_kt,
        nguong_dat=0.80,
    )
    assert kq_plain.diem_ung_vien["diem_bam_sat_nguon"] is None
    assert kq_plain.chi_tiet_thong_ke["trang_thai_bam_sat_nguon"] == "khong_ap_dung"
    assert kq_plain.dat_chuan is True

    kq_require_gain_same = gate.so_sanh_va_quyet_dinh(
        ung_vien=_plain_ok,
        ban_goc=_plain_ok,
        bo_kiem_thu=bo_kt,
        nguong_dat=0.80,
        yeu_cau_cai_thien_co_y_nghia=True,
    )
    assert kq_require_gain_same.dat_chuan is False
    assert "cải thiện có ý nghĩa thống kê" in kq_require_gain_same.ly_do

    kq_require_gain_better = gate.so_sanh_va_quyet_dinh(
        ung_vien=_plain_ok,
        ban_goc=_plain_weak,
        bo_kiem_thu=bo_kt,
        nguong_dat=0.80,
        yeu_cau_cai_thien_co_y_nghia=True,
    )
    assert kq_require_gain_better.dat_chuan is True

    kq_regressed = gate.so_sanh_va_quyet_dinh(
        ung_vien=_plain_weak,
        ban_goc=_plain_ok,
        bo_kiem_thu=bo_kt,
        nguong_dat=0.10,
    )
    assert kq_regressed.dat_chuan is False
    assert "Paired Bootstrap 95% CI" in kq_regressed.ly_do

    kq_rag = gate.so_sanh_va_quyet_dinh(
        ung_vien=_rag_with_sources,
        ban_goc=None,
        bo_kiem_thu=bo_kt,
        nguong_dat=0.80,
    )
    assert kq_rag.diem_ung_vien["diem_bam_sat_nguon"] is not None
    assert kq_rag.chi_tiet_thong_ke["trang_thai_bam_sat_nguon"] == "da_do"


def test_p1_2_onnx_semantic_no_answer_leakage() -> None:
    """Đảm bảo từ điển đồng nghĩa ONNX không chứa các đáp án định lượng rò rỉ từ bộ QA."""
    from vietnamese_ai.rag.onnx_semantic import TU_DIEN_DONG_NGHIA_VIET

    tat_ca_cum = {term.lower() for terms in TU_DIEN_DONG_NGHIA_VIET.values() for term in terms}
    for dap_an_ro_ri in ("72 giờ", "15 năm", "20%/năm"):
        assert dap_an_ro_ri not in tat_ca_cum


def test_p1_2_legal_corpus_accuracy_named_traps_ood_and_parity() -> None:
    """Kiểm tra đính chính Điều luật, mã `tb_dd_2013_d101`, câu hỏi bẫy/OOD khó và đồng bộ SHA-256."""
    from vietnamese_ai.benchmark.datasets_vn import (
        tai_bo_hoi_dap_phap_ly_thoi_gian,
        tai_bo_phap_ly_mo,
        tai_bo_phap_ly_thoi_gian,
        tai_canh_do_thi_phap_ly_thoi_gian,
    )
    from vietnamese_ai.rag.legal.temporal_rag import tao_dong_co_phap_ly_thoi_gian_mac_dinh
    from vietnamese_ai.utils.versioning import QuanLyPhatHanhGoi

    mo = {str(x["id"]): x for x in tai_bo_phap_ly_mo()}
    assert mo["bhxh_02"]["dieu"] == "Điều 2"
    assert mo["nhao_02"]["dieu"] == "Điều 58"

    tb_corpus = tai_bo_phap_ly_thoi_gian()
    tb_edges = tai_canh_do_thi_phap_ly_thoi_gian()
    tb_qa = tai_bo_hoi_dap_phap_ly_thoi_gian()

    corpus_ids = {str(x["id"]) for x in tb_corpus}
    assert "tb_dd_2013_d100" not in corpus_ids
    assert "tb_dd_2013_d101" in corpus_ids
    assert all(e["dich_nut"] != "tb_dd_2013_d100" for e in tb_edges)
    assert len(tb_qa) == 68
    qa_map = {str(q["id"]): q for q in tb_qa}
    assert qa_map["t2_08a"]["tap_id_ky_vong"] == ["tb_dd_2013_d101"]

    engine = tao_dong_co_phap_ly_thoi_gian_mac_dinh()
    kq_trap_dd = engine.truy_hoi(qa_map["t4_11"]["cau_hoi"], che_do="full")
    assert "45/2013/QH13" in kq_trap_dd["van_ban_het_hieu_luc_trong_hoi"]
    assert "31/2024/QH15" in kq_trap_dd["van_ban_thay_the_goi_y"]
    assert kq_trap_dd["tu_choi_tra_loi"] is False

    kq_trap_bhxh = engine.truy_hoi(qa_map["t4_12"]["cau_hoi"], che_do="full")
    assert "58/2014/QH13" in kq_trap_bhxh["van_ban_het_hieu_luc_trong_hoi"]
    assert "41/2024/QH15" in kq_trap_bhxh["van_ban_thay_the_goi_y"]
    assert kq_trap_bhxh["tu_choi_tra_loi"] is False

    kq_ood_hngd = engine.truy_hoi(qa_map["t4_13"]["cau_hoi"], che_do="full")
    assert kq_ood_hngd["tu_choi_tra_loi"] is True

    kq_ood_gt = engine.truy_hoi(qa_map["t4_14"]["cau_hoi"], che_do="full")
    assert kq_ood_gt["tu_choi_tra_loi"] is True

    rel = QuanLyPhatHanhGoi.kiem_tra_san_sang_phat_hanh(ROOT_DIR)
    assert rel["dong_bo_du_lieu_phap_ly"] is True
    assert rel["sai_lech_du_lieu_phap_ly"] == []


def test_p1_3_lm_eval_and_benchmark_runner_no_random_metrics() -> None:
    """Đảm bảo `LMEvalHarness` và `BenchmarkRunner` không bao giờ sinh số đo ngẫu nhiên hoặc hệ số giả."""
    from vietnamese_ai.llm.benchmark_runner import BenchmarkRunner
    from vietnamese_ai.llm.lm_eval import EvalTask, LMEvalHarness

    dummy = object()
    harness = LMEvalHarness()
    harness.dang_ky_task(EvalTask(ten="unknown_kind", loai="unsupported_custom", mo_ta="Test"))
    kq_dummy = harness.danh_gia(
        dummy,
        ["vie_perplexity", "vie_sentiment", "vie_text_generation", "vie_qa", "vie_cloze", "unknown_kind"],
    )
    for task_name in ("vie_perplexity", "vie_sentiment", "vie_text_generation", "vie_qa", "vie_cloze", "unknown_kind"):
        assert "error" in kq_dummy["ket_qua"][task_name]
    assert kq_dummy["tong_hop"] == {}

    class _AlwaysPositiveClassifier:
        def du_doan(self, texts: List[str]) -> List[str]:
            return ["positive" for _ in texts]

    cls_task = EvalTask(
        ten="imbalanced_cls",
        loai="classification",
        du_lieu=[
            {"text": "t1", "label": "positive"},
            {"text": "t2", "label": "negative"},
        ],
    )
    harness.dang_ky_task(cls_task)
    kq_cls = harness.danh_gia(_AlwaysPositiveClassifier(), ["imbalanced_cls"])["ket_qua"]["imbalanced_cls"]
    assert kq_cls["accuracy"] == pytest.approx(0.5)
    assert kq_cls["f1"] == pytest.approx(1.0 / 3.0, abs=1e-3)

    class _PartialQAAndGen:
        def sinh_van_ban(self, prompt: str, do_dai: int = 30) -> str:
            return "trí tuệ máy tính"

    qa_task = EvalTask(
        ten="partial_qa",
        loai="question_answering",
        du_lieu=[{"question": "AI là gì?", "answer": "trí tuệ nhân tạo"}],
    )
    gen_no_ref = EvalTask(
        ten="gen_noref",
        loai="text_generation",
        du_lieu=[{"prompt": "AI là"}],
    )
    harness.dang_ky_task(qa_task)
    harness.dang_ky_task(gen_no_ref)
    kq_qa = harness.danh_gia(_PartialQAAndGen(), ["partial_qa", "gen_noref"])["ket_qua"]
    assert kq_qa["partial_qa"]["accuracy"] == 0.0
    assert kq_qa["partial_qa"]["f1"] == pytest.approx(0.5, abs=1e-3)
    assert kq_qa["gen_noref"]["bleu"] is None

    runner = BenchmarkRunner()
    kq_bench = runner.chay(dummy, benchmarks=["perplexity", "generation", "sentiment", "speed", "qa"])
    for key in ("perplexity", "generation", "sentiment", "speed", "qa"):
        assert kq_bench[key]["trang_thai"] == "khong_ho_tro"
    bc = runner.bao_cao()
    assert "VIETNAMESE LLM BENCHMARK REPORT" in bc


def test_p1_docs_no_fabricated_benchmark_claims() -> None:
    """Đảm bảo tài liệu & bài báo không còn số liệu chưa kiểm chứng (500 cặp QA, 1,790x, p=2^256-189)."""
    cum_cam = (
        "500 cặp câu hỏi",
        "500 gold-standard",
        "~1,790",
        "1,790x",
        "nhanh gấp 2,000x",
        "2^{256} - 189",
    )
    for rel in (
        "README.md",
        "paper/paper.md",
        "docs/index.md",
        "docs/guides/comparative-benchmark.md",
    ):
        noi_dung = (ROOT_DIR / rel).read_text(encoding="utf-8")
        for cum in cum_cam:
            assert cum not in noi_dung, f"Phát hiện tuyên bố chưa kiểm chứng '{cum}' trong {rel}"
    assert json.loads(json.dumps({"ok": True}))["ok"] is True
