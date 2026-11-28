"""Bộ kiểm thử hồi quy cho Đợt 5 — Giai đoạn P4 (Lõi NLP 38k từ + Rust Parity + ONNX An toàn + Studio Hợp nhất + Mypy Strict)."""

from __future__ import annotations

import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_p4_1_compressed_vietnamese_dictionary_in_tokenizer_and_spelling() -> None:
    """Xác minh từ điển tiếng Việt nén >=30.000 từ ghép được nạp cho BoTachTuTiengVietSieuToc và KiemTraChinhTa."""
    from vietnamese_ai.nlp.native_tokenizer import (
        BoTachTuTiengVietSieuToc,
        tai_tu_dien_tieng_viet,
    )
    from vietnamese_ai.nlp.spelling import KiemTraChinhTa

    tu_ghep = tai_tu_dien_tieng_viet(chi_tu_ghep=True)
    toan_bo = tai_tu_dien_tieng_viet(chi_tu_ghep=False)
    assert len(tu_ghep) >= 30000
    assert len(toan_bo) >= 35000

    tok = BoTachTuTiengVietSieuToc()
    assert len(tok._tu_dien) >= 30000
    seg = tok.tach_tu("bệnh viện trung ương và trường đại học bách khoa", noi_tu_ghep=True)
    seg_lower = [s.lower() for s in seg]
    assert "bệnh_viện" in seg_lower
    assert "trung_ương" in seg_lower
    assert "đại_học" in seg_lower or "trường_đại_học" in seg_lower

    kt = KiemTraChinhTa()
    assert kt.thong_ke()["kich_thuoc_tu_dien"] >= 35000
    assert "không" in kt.sua("khong")
    kq_tieng = kt.kiem_tra("tieng")
    if kq_tieng["loi"]:
        assert any("tiếng" == g for g in kq_tieng["loi"][0]["goi_y"])


def test_p4_1_rust_python_zero_width_and_bidi_parity() -> None:
    """Xác minh tập ký tự Zero-Width, Soft-Hyphen và BiDi được lọc đồng nhất 100% giữa Rust và Python."""
    from vietnamese_ai.nlp import native_tokenizer as nt

    ky_tu_an = "\u00ad\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2060\u2066\u2067\u2068\u2069\ufeff"
    van_ban = f"Xin{ky_tu_an}  chào \u202eViệt\u200f Nam\u00ad!"

    tok = nt.BoTachTuTiengVietSieuToc()
    sach_py = nt._RE_KY_TU_AN.sub("", van_ban)
    sach_py = tok._re_spaces.sub(" ", sach_py).strip()
    assert sach_py == "Xin chào Việt Nam!"

    if nt._rust_core is not None:
        sach_rust = str(nt._rust_core.fast_clean_spaces(van_ban))
        assert sach_rust == sach_py
        seg_rust = list(tok._rust_tokenizer.normalize_and_segment(van_ban, True, "_", True))
        assert "".join(seg_rust).find("\u202e") == -1
        assert "".join(seg_rust).find("\u00ad") == -1
        assert "".join(seg_rust).find("\u2066") == -1


def test_p4_1_fast_embedder_uses_zero_copy_rust_bytes() -> None:
    """Xác minh BoNhungVanBanSieuToc dùng buffer zero-copy fast_top_k_cosine_bytes thay cho .tolist()."""
    from vietnamese_ai.nlp.native_tokenizer import BoNhungVanBanSieuToc

    src = inspect.getsource(BoNhungVanBanSieuToc.tim_kiem_tuong_dong_hang_loat)
    assert "fast_top_k_cosine_bytes" in src
    assert ".tolist()" not in src

    emb = BoNhungVanBanSieuToc(so_chieu=64)
    rng = np.random.default_rng(42)
    q = rng.standard_normal(64).astype(np.float32)
    mat = rng.standard_normal((50, 64)).astype(np.float32)

    scores = emb.tim_kiem_tuong_dong_hang_loat(q, mat)
    expected = (mat @ q) / (np.linalg.norm(q) * np.linalg.norm(mat, axis=1))
    np.testing.assert_allclose(scores, expected, atol=1e-5)

    top5 = emb.tim_kiem_top_k(q, mat, top_k=5)
    assert len(top5) == 5
    expected_top5 = list(np.argsort(-expected)[:5])
    assert [idx for idx, _ in top5] == expected_top5


def test_p4_2_onnx_refuses_missing_tokenizer_and_pull_presets(tmp_path: Path) -> None:
    """Xác minh BoNhungONNXTiengViet / BoSapXepLaiONNXTiengViet từ chối chạy ONNX thiếu tokenizer.json và hỗ trợ preset vai pull."""
    from vietnamese_ai.edge.model_hub import DANH_MUC_MO_HINH_ONNX_VIET, QuanLyMoHinhEdge
    from vietnamese_ai.rag.onnx_semantic import BoNhungONNXTiengViet, BoSapXepLaiONNXTiengViet

    fake_onnx = tmp_path / "model_int8.onnx"
    fake_onnx.write_bytes(b"ONNX_MOCK_BYTES")

    emb = BoNhungONNXTiengViet(duong_dan_onnx=fake_onnx)
    assert emb.che_do == "semantic_synonym_cpu"
    assert emb._ort_session is None

    with pytest.raises(ValueError, match="tokenizer.json"):
        BoNhungONNXTiengViet(duong_dan_onnx=fake_onnx, bat_buoc_onnx=True)

    emb._ort_session = object()
    emb._hf_tokenizer = None
    with pytest.raises(RuntimeError, match="tokenizer.json"):
        emb._nhung_onnx_batch(["Xin chào"])

    rr = BoSapXepLaiONNXTiengViet(duong_dan_onnx=fake_onnx)
    assert rr.che_do == "vietnamese_cross_interaction"
    assert rr._ort_session is None

    with pytest.raises(ValueError, match="tokenizer.json"):
        BoSapXepLaiONNXTiengViet(duong_dan_onnx=fake_onnx, bat_buoc_onnx=True)

    assert "multilingual-e5-small-onnx" in DANH_MUC_MO_HINH_ONNX_VIET
    assert "bge-m3-onnx" in DANH_MUC_MO_HINH_ONNX_VIET
    assert "bge-reranker-v2-m3-onnx" in DANH_MUC_MO_HINH_ONNX_VIET

    hub = QuanLyMoHinhEdge(thu_muc_luu=tmp_path / "hub")
    dry = hub.tai_mo_hinh("multilingual-e5-small-onnx", chay_that=False)
    assert dry["trang_thai"] == "xem_truoc_dry_run"
    assert dry["url"].endswith(".onnx")
    assert dry["url_tokenizer"].endswith("tokenizer.json")
    assert dry["duong_dan_tokenizer"].endswith("tokenizer.json")

    class _MockHFHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.endswith("tokenizer.json"):
                body = json.dumps({"version": "1.0", "model": {"type": "Unigram"}}).encode("utf-8")
            else:
                body = b"\x08\x07ONNX_INT8_PAYLOAD"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockHFHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        orig_tai = hub._tai_tep_don_gian

        def _redirect_local(url: str, dich: Path, hf_token: Any = None, timeout_giay: float = 60.0) -> int:
            name = Path(url).name
            return orig_tai(f"http://127.0.0.1:{port}/{name}", dich, hf_token=hf_token, timeout_giay=timeout_giay)

        with patch.object(hub, "_tai_tep_don_gian", side_effect=_redirect_local):
            kq_dl = hub.tai_mo_hinh("multilingual-e5-small-onnx", chay_that=True)
        assert kq_dl["trang_thai"] == "da_tai_thanh_cong"
        assert Path(kq_dl["duong_dan_onnx"]).is_file()
        assert Path(kq_dl["duong_dan_tokenizer"]).is_file()
    finally:
        srv.shutdown()
        srv.server_close()


def test_p4_3_studio_unified_vaidb_and_real_training_workflows(tmp_path: Path) -> None:
    """Xác minh StudioDashboard hợp nhất kho .vaidb giữa RAG/Workspace/TroLyPhapLy và nối tab Training với SafeLoop & Remote GPU."""
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    ws_dir = tmp_path / "studio_ws"
    dash = StudioDashboard(host="127.0.0.1", port=0, thu_muc_workspace=str(ws_dir))
    try:
        ws_mgr = dash._lay_quan_ly_workspace()
        rag_ws = ws_mgr.lay_rag_workspace("mac_dinh")
        rag_legal = dash.tro_ly_phap_ly.rag

        assert rag_ws is dash.rag
        assert rag_legal is dash.rag
        assert dash.duong_dan_vaidb is not None and Path(dash.duong_dan_vaidb).is_file()

        dash.nap_tai_lieu_rag("doc_p4_uni", "Quy định nội bộ P4: mã kho hợp nhất là UNIFIED-VAIDB-2026.")
        kq_hoi = dash.tro_ly_phap_ly.rag.hoi("Mã kho hợp nhất P4 là gì?", top_k=3)
        assert "UNIFIED-VAIDB-2026" in kq_hoi.get("tra_loi", "")

        kq_sl = dash.chay_huan_luyen({"model": "vgpt2", "task": "safeloop", "epochs": 2})
        assert kq_sl["trang_thai"] == "HOAN_THANH"
        assert kq_sl["che_do_huan_luyen"] == "vialign_safeloop"
        assert "hanh_dong_eval_gate" in kq_sl
        assert len(kq_sl["lich_su_loss"]) >= 1

        kq_remote = dash.chay_huan_luyen(
            {"model": "vgpt2", "task": "remote_gpu", "nha_cung_cap": "local", "chay_that": False}
        )
        assert kq_remote["che_do_huan_luyen"] == "remote_gpu_flywheel"
        assert "chi_tiet_flywheel" in kq_remote
    finally:
        dash.dong()


def test_p4_4_pyproject_enables_check_untyped_defs_on_core_packages() -> None:
    """Xác minh pyproject.toml bật check_untyped_defs = true cho 4 gói lõi rag, security, edge, core."""
    txt = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
    assert "check_untyped_defs = true" in txt
    for pkg in ("vietnamese_ai.rag.*", "vietnamese_ai.security.*", "vietnamese_ai.edge.*", "vietnamese_ai.core.*"):
        assert pkg in txt
