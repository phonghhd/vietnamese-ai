"""Bộ kiểm thử hồi quy cho toàn bộ các lỗi kỹ thuật lõi đã sửa trong Giai đoạn P2 (P2.1 -> P2.3)."""

from __future__ import annotations

import asyncio
import http.client
import inspect
import io
import subprocess
import threading
import time
from datetime import date, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_p2_1_vaidb_busy_timeout_and_inplace_cache_invalidation(tmp_path: Path) -> None:
    """Xác minh KhoTriThucVAIDB thiết lập busy_timeout, mở rộng đệm theo cấp số nhân và hủy cache khi cập nhật vector."""
    from vietnamese_ai.rag.vaidb_store import KhoTriThucVAIDB

    db_path = tmp_path / "p2_test.vaidb"
    kho = KhoTriThucVAIDB(duong_dan=db_path, kich_thuoc=4)
    try:
        busy_timeout = kho._conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert int(busy_timeout) >= 5000

        for idx in range(20):
            v = np.zeros(4, dtype=np.float32)
            v[idx % 4] = 1.0
            kho.chen(f"doc_{idx}", v, {"noi_dung": f"Tài liệu {idx}"})

        assert kho._vec_capacity >= 20
        assert kho._vectors is not None and kho._vectors.shape == (20, 4)

        kho._vectors_normed = np.ones((20, 4), dtype=np.float32)
        chu_ky_truoc = kho._chu_ky_tap_ids()

        vec_cap_nhat = np.array([0.0, 0.6, 0.8, 0.0], dtype=np.float32)
        kho.chen("doc_0", vec_cap_nhat, {"noi_dung": "Cập nhật"})
        assert kho._vectors_normed is None
        chu_ky_sau = kho._chu_ky_tap_ids()
        assert chu_ky_sau != chu_ky_truoc

        kq = kho.tim_kiem(vec_cap_nhat, top_k=1)
        assert kq and kq[0]["ma"] == "doc_0"
    finally:
        kho.dong()


def test_p2_1_workspace_store_busy_timeout(tmp_path: Path) -> None:
    """Xác minh QuanLyPhienChatWorkspace thiết lập PRAGMA busy_timeout cho SQLite."""
    from vietnamese_ai.ui.workspace_store import QuanLyPhienChatWorkspace

    ws = QuanLyPhienChatWorkspace(thu_muc_du_lieu=tmp_path / "workspace_dir")
    busy_timeout = ws._conn.execute("PRAGMA busy_timeout;").fetchone()[0]
    assert int(busy_timeout) >= 5000


def test_p2_1_temporal_rag_external_reranker_interfaces() -> None:
    """Xác minh BoTruyHoiPhapLyThoiGian gọi đúng các giao diện cham_diem_lo / cham_diem / sap_xep_lai."""
    from vietnamese_ai.rag.legal.temporal_rag import BoTruyHoiPhapLyThoiGian

    du_lieu_mau = [
        {
            "so_hieu": "45/2019/QH14",
            "ten_van_ban": "Bộ luật Lao động 2019",
            "dieu": "Điều 25",
            "tieu_de": "Thời gian thử việc",
            "noi_dung": "Không quá 60 ngày đối với công việc có chức danh nghề nghiệp cần trình độ chuyên môn, kỹ thuật từ cao đẳng trở lên.",
            "hieu_luc_tu": "2021-01-01",
        }
    ]

    class RerankerLo:
        def __init__(self) -> None:
            self.da_goi = False

        def cham_diem_lo(self, cau_hoi: str, danh_sach_van_ban: List[str]) -> List[float]:
            self.da_goi = True
            return [0.95 for _ in danh_sach_van_ban]

    class RerankerDon:
        def __init__(self) -> None:
            self.so_lan = 0

        def cham_diem(self, cau_hoi: str, van_ban: str) -> float:
            self.so_lan += 1
            return 0.88

    rr_lo = RerankerLo()
    rag_lo = BoTruyHoiPhapLyThoiGian(bo_xep_hang_cheo=rr_lo)
    rag_lo.nap_tu_danh_sach(du_lieu_mau)
    kq_lo = rag_lo.truy_hoi("Thời gian thử việc tối đa cho trình độ cao đẳng là bao lâu?", ngay_tra_cuu="2025-05-01")
    assert rr_lo.da_goi is True
    assert kq_lo["ket_qua"]

    rr_don = RerankerDon()
    rag_don = BoTruyHoiPhapLyThoiGian(bo_xep_hang_cheo=rr_don)
    rag_don.nap_tu_danh_sach(du_lieu_mau)
    rag_don.truy_hoi("Thời gian thử việc tối đa cho trình độ cao đẳng là bao lâu?", ngay_tra_cuu="2025-05-01")
    assert rr_don.so_lan > 0


def test_p2_1_temporal_query_quarter_prefixes_and_pre_effective_date() -> None:
    """Xác minh BoPhanTichThoiGianPhapLy xử lý đúng trước/từ/sau quý và mốc thời gian trước khi luật có hiệu lực."""
    from vietnamese_ai.rag.legal.temporal_query import BoPhanTichThoiGianPhapLy

    parser = BoPhanTichThoiGianPhapLy()

    kq_truoc = parser.phan_tich("Quy định áp dụng trước quý 2/2025 như thế nào?")
    assert kq_truoc.ngay_tra_cuu == "2025-03-31"

    kq_tu = parser.phan_tich("Chính sách thuế áp dụng từ quý 3/2025 ra sao?")
    assert kq_tu.ngay_tra_cuu == "2025-07-01"

    kq_sau = parser.phan_tich("Quy định sau quý 1/2025 có thay đổi gì?")
    assert kq_sau.ngay_tra_cuu == "2025-04-01"

    kq_hieu_luc = parser.phan_tich(
        "Trước ngày 01/01/2024, Luật Đất đai 31/2024/QH15 đã có hiệu lực chưa?"
    )
    assert kq_hieu_luc.ngay_tra_cuu == "2023-12-31"


def test_p2_1_citation_verifier_sentence_splitting_and_cross_law_pair() -> None:
    """Xác minh BoKiemChungTrichDanPhapLy không cắt câu tại số tiền/Điều và phát hiện lệch cặp (số hiệu, điều)."""
    from vietnamese_ai.rag.legal.citation_verifier import BoKiemChungTrichDanPhapLy

    verifier = BoKiemChungTrichDanPhapLy()
    cac_cau = verifier._tach_cau_phap_ly(
        "Theo Điều 1. Luật 45/2019/QH14, mức phạt là 1.000.000 đồng (bao gồm phụ cấp, v.v.) đối với vi phạm. Người lao động cần lưu ý."
    )
    assert len(cac_cau) == 2
    assert "1.000.000" in cac_cau[0]
    assert "v.v." in cac_cau[0]

    nguon = [
        {"so_hieu": "45/2019/QH14", "dieu": "Điều 25", "noi_dung": "Thời gian thử việc."},
        {"so_hieu": "31/2024/QH15", "dieu": "Điều 138", "noi_dung": "Cấp giấy chứng nhận quyền sử dụng đất."},
    ]
    kq_lech = verifier.kiem_chung(
        "Căn cứ Điều 138 của Luật 45/2019/QH14, người lao động được thử việc.",
        nguon_truy_hoi=nguon,
    )
    assert any("45/2019/QH14" in item and "138" in item for item in kq_lech.trich_dan_khong_co_trong_nguon)
    assert kq_lech.hop_le is False


def test_p2_1_validity_tracker_datetime_and_clause_dedup() -> None:
    """Xác minh _phan_tich_ngay chấp nhận datetime và dang_ky_phien_ban_dieu_khoan chống trùng lặp."""
    from vietnamese_ai.rag.legal.validity_tracker import (
        BoTheoDoiHieuLucPhapLy,
        PhienBanDieuKhoan,
        _phan_tich_ngay,
    )

    dt = datetime(2025, 8, 15, 10, 30, 0)
    assert _phan_tich_ngay(dt) == date(2025, 8, 15)

    tracker = BoTheoDoiHieuLucPhapLy()
    pb = PhienBanDieuKhoan(
        so_hieu="99/2025/QH15",
        dieu="Điều 1",
        noi_dung="Nội dung ban đầu",
        hieu_luc_tu="2025-01-01",
    )
    tracker.dang_ky_phien_ban_dieu_khoan(pb)
    pb_cap_nhat = PhienBanDieuKhoan(
        so_hieu="99/2025/QH15",
        dieu="Điều 1",
        noi_dung="Nội dung đã cập nhật",
        hieu_luc_tu="2025-01-01",
    )
    tracker.dang_ky_phien_ban_dieu_khoan(pb_cap_nhat)
    cac_pb = tracker._kho_dieu_khoan[("99/2025/QH15", "Điều 1")]
    assert len(cac_pb) == 1
    assert cac_pb[0].noi_dung == "Nội dung đã cập nhật"


def test_p2_1_rag_pipeline_no_useless_ring_attention() -> None:
    """Xác minh RAGPipeline mặc định tắt extreme hardware và không chạy RingAttentionNode trong .hoi()."""
    from vietnamese_ai.rag.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    assert rag.enable_extreme_hardware is False
    rag.them_tai_lieu("doc_vn", "Việt Nam có thủ đô là Hà Nội.")
    kq = rag.hoi("Thủ đô của Việt Nam là gì?")
    assert "extreme_hardware" not in kq


def test_p2_2_c_bridge_prioritizes_llama_cli_and_devnull(tmp_path: Path) -> None:
    """Xác minh CauNoiCtypesNative ưu tiên llama-cli mà không bị _cdll chặn và dùng stdin=DEVNULL."""
    from vietnamese_ai.edge.c_bridge import CauNoiCtypesNative

    gguf = tmp_path / "demo.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    fake_cli = tmp_path / "llama-cli"
    fake_cli.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    fake_cli.chmod(0o755)

    with patch("vietnamese_ai.edge.c_bridge.QuanLyPhuThuoc.nhap_tuy_chon", return_value=None):
        with patch("shutil.which", side_effect=lambda name: str(fake_cli) if name == "llama-cli" else None):
            bridge = CauNoiCtypesNative(duong_dan_mo_hinh=gguf)
            assert bridge.co_thu_vien_c() is True
            assert bridge.thong_tin_cau_noi()["che_do"] == "binary_cli"

            fake_proc = MagicMock()
            fake_proc.stdout = io.StringIO("Xin chào")
            fake_proc.wait.return_value = 0
            fake_proc.poll.return_value = 0

            with patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
                tokens = list(bridge.sinh_stream("Chào bạn", do_dai_toi_da=16))
                assert "".join(tokens) == "Xin chào"
                _, kwargs = mock_popen.call_args
                assert kwargs.get("stdin") == subprocess.DEVNULL
                cmd = mock_popen.call_args.args[0]
                assert "-no-cnv" in cmd


def test_p2_2_swarm_auth_ttl_ip_and_threadsafe_approval() -> None:
    """Xác minh XacThucCumSwarm kiểm tra TTL, IP của session token và đánh thức Future an toàn từ luồng khác."""
    from vietnamese_ai.edge.swarm_auth import XacThucCumSwarm

    auth = XacThucCumSwarm(ma_pin="123456", yeu_cau_phe_duyet=True, thoi_gian_song_session_giay=60.0)
    token = auth.tao_session_token("node_1", ip="100.64.0.2", ttl_giay=0.05)
    assert auth.xac_minh_session_token(token, ip="100.64.0.2") is True
    assert auth.xac_minh_session_token(token, ip="100.64.0.99") is False

    time.sleep(0.08)
    assert auth.xac_minh_session_token(token, ip="100.64.0.2") is False

    async def _chay_phe_duyet() -> bool:
        fut = auth.dang_ky_cho_duyet("node_async", ip="100.64.0.5", thong_tin={"cpu": 2})

        def _duyet_tu_worker() -> None:
            time.sleep(0.05)
            auth.phe_duyet_nut("node_async", True)

        t = threading.Thread(target=_duyet_tu_worker, daemon=True)
        t.start()
        ket_qua = await asyncio.wait_for(fut, timeout=2.0)
        t.join(timeout=1.0)
        return bool(ket_qua)

    assert asyncio.run(_chay_phe_duyet()) is True


def test_p2_2_bitlinear_scaling_2bit_cache_and_inplace_compression() -> None:
    """Xác minh BitLinear nhân đủ scale_w * scale_x, cache 2-bit và giải phóng trọng số khi gọi nen_tai_cho()."""
    from vietnamese_ai.compression.extreme import _CO_TORCH, BitLinear

    layer = BitLinear(8, 4, bias=False)
    x_np = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 8)

    if _CO_TORCH:
        import torch

        x_t = torch.tensor(x_np, requires_grad=True)
        out_fp = layer(x_t)
        loss = out_fp.sum()
        loss.backward()
        assert x_t.grad is not None
        assert layer.weight.grad is not None

        out_2bit_truoc = layer.suy_luan_2bit(x_np)
        assert layer._packed_2bit_cache is not None
        np.testing.assert_allclose(out_fp.detach().numpy(), out_2bit_truoc, atol=2e-2)

        stats = layer.nen_tai_cho()
        assert stats["da_nen_tai_cho"] is True
        assert layer.weight is None
        out_sau_nen = layer(torch.tensor(x_np)).detach().numpy()
        np.testing.assert_allclose(out_sau_nen, out_2bit_truoc, atol=1e-5)
    else:
        out_truoc = layer.forward(x_np)
        out_2bit = layer.suy_luan_2bit(x_np)
        np.testing.assert_allclose(out_truoc, out_2bit, atol=2e-2)
        stats = layer.nen_tai_cho()
        assert stats["da_nen_tai_cho"] is True
        assert layer.weight is None
        out_sau = layer.forward(x_np)
        np.testing.assert_allclose(out_sau, out_2bit, atol=1e-5)


def test_p2_2_dpo_ref_model_guard_gguf_honesty_and_modal_volume(tmp_path: Path) -> None:
    """Xác minh DPO không tự đóng băng model, HuggingFaceWrapper không xuất GGUF giả và Modal dùng volume get."""
    from vietnamese_ai.fine_tuning.dpo_trainer import DPOTrainer
    from vietnamese_ai.fine_tuning.hf_wrapper import HuggingFaceWrapper
    from vietnamese_ai.fine_tuning.remote_gpu import CauHinhTinhChinhXa, NhaCungCapModal

    class _DummyModel:
        def __init__(self) -> None:
            self.weights = np.ones((4, 4), dtype=np.float32)

    m = _DummyModel()
    captured: Dict[str, Any] = {}

    trainer = DPOTrainer(so_vong=1)
    with patch.object(
        trainer,
        "_huan_luyen_numpy",
        side_effect=lambda mod, ref, data, cb: captured.update({"mod": mod, "ref": ref}) or {"loss": 0.1},
    ):
        trainer.huan_luyen(
            model=m,
            ref_model=m,
            preference_data=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
        )
    assert captured["mod"] is m
    assert captured["ref"] is not m

    wrapper = HuggingFaceWrapper()
    wrapper.da_tai = True
    wrapper._che_do_native = False
    wrapper.model = object()
    with pytest.raises(RuntimeError, match="GGUF"):
        wrapper.xuat_gguf(str(tmp_path / "out.gguf"))

    modal = NhaCungCapModal()
    cfg = CauHinhTinhChinhXa(mo_hinh="vgpt2", nhiem_vu="sft")
    script_path = modal.tao_modal_app_script(cfg, tmp_path, "job_p2_vol")
    noi_dung = script_path.read_text(encoding="utf-8")
    assert "modal" in noi_dung and "volume" in noi_dung and "get" in noi_dung
    assert "gguf_bytes" not in noi_dung


def test_p2_2_speculative_greedy_temperature_zero_is_deterministic() -> None:
    """Xác minh SpeculativeEngine khi nhiet_do=0.0 hoàn toàn tất định và khớp argmax của target_model."""
    from vietnamese_ai.serving.speculative import SpeculativeEngine

    class _DeterministicModel:
        def __init__(self, shift: int = 0) -> None:
            self.shift = shift

        def __call__(self, ids: np.ndarray) -> np.ndarray:
            arr = np.asarray(ids)
            seq = arr[0] if arr.ndim == 2 else arr
            logits = np.zeros((len(seq), 16), dtype=np.float32)
            for pos, tok in enumerate(seq):
                target_tok = (int(tok) + 1 + self.shift) % 16
                logits[pos, target_tok] = 10.0
            return logits

    target = _DeterministicModel(shift=0)
    draft = _DeterministicModel(shift=1)

    engine_a = SpeculativeEngine(target_model=target, draft_model=draft, gamma=3, nhiet_do=0.0)
    engine_b = SpeculativeEngine(target_model=target, draft_model=draft, gamma=3, nhiet_do=0.0)
    out_a = engine_a.sinh_van_ban([1], max_tokens=6)
    out_b = engine_b.sinh_van_ban([1], max_tokens=6)
    assert out_a == out_b
    assert out_a == [1, 2, 3, 4, 5, 6, 7]


def test_p2_3_pipeline_danh_gia_applies_preprocessors() -> None:
    """Xác minh Pipeline.danh_gia() biến đổi X qua các bước tiền xử lý trước khi gọi danh_gia() của bước cuối."""
    from vietnamese_ai.core.pipeline import Pipeline

    class _NhanDoiScaler:
        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            return X * 2.0

        def transform(self, X: np.ndarray) -> np.ndarray:
            return X * 2.0

    class _KiemTraModel:
        def __init__(self) -> None:
            self.x_nhan_duoc: np.ndarray | None = None

        def huan_luyen(self, X: np.ndarray, y: np.ndarray) -> None:
            self._mean = float(np.mean(X))

        def du_doan(self, X: np.ndarray) -> np.ndarray:
            return (X[:, 0] > 0).astype(int)

        def danh_gia(self, X: np.ndarray, y: np.ndarray) -> float:
            self.x_nhan_duoc = X.copy()
            return 1.0 if np.allclose(X, np.array([[2.0], [4.0]])) else 0.0

    model = _KiemTraModel()
    pipe = Pipeline("p2_eval").them_buoc("scale", _NhanDoiScaler()).them_buoc("cls", model)
    X_raw = np.array([[1.0], [2.0]])
    y = np.array([1, 1])
    pipe.fit(X_raw, y)
    diem = pipe.danh_gia(X_raw, y)
    assert diem == 1.0
    assert model.x_nhan_duoc is not None
    np.testing.assert_allclose(model.x_nhan_duoc, X_raw * 2.0)


def test_p2_3_end_to_end_hoi_passes_rag_context_to_generator(tmp_path: Path) -> None:
    """Xác minh QuyTrinhDauCuoi.hoi() truyền ngữ cảnh RAG truy xuất được vào bộ sinh mô hình."""
    from vietnamese_ai.core.end_to_end import QuyTrinhDauCuoi

    wf = QuyTrinhDauCuoi(ten_du_an="P2RagCtx", thu_muc_lam_viec=str(tmp_path / "ws"), bat_tuong_lua=False)
    wf.xay_dung_rag(["Mã bí mật nội bộ của dự án P2 là VAI-2026-CTX."])

    prompts_nhan: List[str] = []

    class _MockGen:
        nhiem_vu = "sft"

        def sinh_van_ban(self, prompt: str) -> str:
            prompts_nhan.append(prompt)
            return "Đã nhận ngữ cảnh"

    wf.mo_hinh_hien_tai = _MockGen()
    kq = wf.hoi("Mã bí mật nội bộ của dự án P2 là gì?", dung_rag=True)
    assert kq["tra_loi"] == "Đã nhận ngữ cảnh"
    assert prompts_nhan and "VAI-2026-CTX" in prompts_nhan[0]
    assert "Ngữ cảnh tham khảo" in prompts_nhan[0]


def test_p2_3_backprop_uses_pre_update_weights() -> None:
    """Xác minh LopDense, LoRALayer và MangNron tính gradient truyền ngược bằng trọng số trước khi cập nhật."""
    from vietnamese_ai.fine_tuning.lora import LoRALayer
    from vietnamese_ai.models.layers import LopDense
    from vietnamese_ai.models.neural_net import MangNron

    layer = LopDense(3, 2, ham_kich_hoat="linear")
    w_goc = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    layer.trong_so = w_goc.copy()
    layer.bias = np.zeros((1, 2), dtype=float)
    x = np.array([[1.0, 0.5, -1.0]], dtype=float)
    grad_out = np.array([[2.0, -3.0]], dtype=float)
    layer.tien(x)
    grad_in = layer.ve(grad_out, toc_do_hoc=1.0)
    np.testing.assert_allclose(grad_in, grad_out @ w_goc.T)

    lora = LoRALayer(3, 2, rank=2, alpha=4.0)
    a_goc = np.array([[0.5, -0.2], [0.1, 0.4], [-0.3, 0.6]], dtype=float)
    b_goc = np.array([[0.7, -0.5], [0.2, 0.9]], dtype=float)
    lora.A = a_goc.copy()
    lora.B = b_goc.copy()
    lora.tien(x)
    grad_in_lora = lora.ve(grad_out, toc_do_hoc=1.0)
    np.testing.assert_allclose(grad_in_lora, (grad_out * lora.scaling) @ (a_goc @ b_goc).T)

    net = MangNron(lop_an=[3], ham_kich_hoat="relu", toc_do_hoc=1.0, so_vong=1, kich_thuoc_batch=2)
    w0 = np.array([[0.5, 0.2, 0.1], [0.3, 0.4, 0.6]], dtype=float)
    w1 = np.array([[0.8, -0.4], [0.2, 0.5], [-0.3, 0.7]], dtype=float)
    net._trong_so = [w0.copy(), w1.copy()]
    net._he_so = [np.zeros((1, 3)), np.zeros((1, 2))]
    X_b = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float)
    y_b = np.array([0, 1], dtype=float)
    cac_z, cac_a = net._lan_truyen_thuan(X_b)
    y_oh = np.array([[1.0, 0.0], [0.0, 1.0]])
    dz1 = cac_a[-1] - y_oh
    dz0_ky_vong = (dz1 @ w1.T) * (cac_z[0] > 0).astype(float)
    dw0_ky_vong = cac_a[0].T @ dz0_ky_vong / 2.0
    w0_ky_vong = w0 - 1.0 * dw0_ky_vong
    net._lan_truyen_nguoc(X_b, y_b, cac_z, cac_a)
    np.testing.assert_allclose(net._trong_so[0], w0_ky_vong)


def test_p2_3_vietnamese_llm_ngram_context_length() -> None:
    """Xác minh VietnameseLLM giữ đúng độ dài ngữ cảnh (bac - 1) trong sinh_van_ban và lay_tu_ke_tiep."""
    from vietnamese_ai.llm.vietnamese_llm import VietnameseLLM

    llm = VietnameseLLM(bac=3, toi_thieu_dem=1)
    llm.huan_luyen(
        [
            "trí tuệ nhân tạo phát triển mạnh mẽ tại việt nam",
            "trí tuệ nhân tạo hỗ trợ lập trình viên rất tốt",
        ]
    )
    goi_y = llm.lay_tu_ke_tiep("trí tuệ", top_n=5)
    cac_tu_goi_y = [item["tu"] for item in goi_y]
    assert "nhân" in cac_tu_goi_y or any("nhân" in t for t in cac_tu_goi_y)

    van_ban = llm.sinh_van_ban("trí tuệ", do_dai=6, nhiet_do=0.5)
    assert len(van_ban.split()) > 2


def test_p2_3_sentiment_tfidf_fit_transform_and_negation() -> None:
    """Xác minh PhanTichCamXuc dùng transform_tfidf cố định từ điển khi suy luận và không tính điểm dương kép khi phủ định."""
    from vietnamese_ai.nlp.sentiment import PhanTichCamXuc

    ptx_td = PhanTichCamXuc(che_do="tu_dien")
    kq_khl = ptx_td.phan_tich("Tôi không hài lòng với sản phẩm này")
    assert kq_khl["nhan"] == "negative"
    assert kq_khl["xac_suat"]["positive"] == 0.0

    kq_kt = ptx_td.phan_tich("Dịch vụ chưa tốt và kém chất lượng")
    assert kq_kt["nhan"] == "negative"
    assert kq_kt["xac_suat"]["positive"] == 0.0

    ptx_th = PhanTichCamXuc(che_do="tu_huan")
    ptx_th.huan_luyen(
        [
            "sản phẩm tuyệt vời giao hàng nhanh",
            "dịch vụ chu đáo rất hài lòng",
            "hàng kém chất lượng giao chậm quá",
            "thất vọng hoàn toàn sản phẩm bị lỗi",
        ],
        ["positive", "positive", "negative", "negative"],
    )
    so_tu_dien = len(ptx_th._xl._tu_dien)
    kq_suy_luan = ptx_th.phan_tich("sản phẩm tuyệt vời")
    assert kq_suy_luan["nguon"] == "tu_huan"
    assert len(ptx_th._xl._tu_dien) == so_tu_dien


def test_p2_3_state_graph_step_limit_start_interrupt_and_cluster_node() -> None:
    """Xác minh DoThiTrangThai truyền gioi_han_buoc, cho phép interrupt_before tại nút đầu và gọi NutChinh thật."""
    from vietnamese_ai.agents.state_graph import END, START, DoThiTrangThai
    from vietnamese_ai.orchestrator.master import NutChinh

    g_loop = DoThiTrangThai()
    g_loop.them_nut("dem", lambda s: {"n": s.get("n", 0) + 1})
    g_loop.them_canh(START, "dem")
    g_loop.them_canh("dem", "dem")
    app_loop = g_loop.bien_dich(gioi_han_buoc=4)
    res_loop = app_loop.thuc_thi({"n": 0})
    assert res_loop["__buoc__"] == 4
    assert res_loop["n"] == 4

    da_chay_buoc_1 = []
    g_hitl = DoThiTrangThai()

    def _buoc_1(s: Dict[str, Any]) -> Dict[str, Any]:
        da_chay_buoc_1.append(True)
        return {"ket_qua": "ok", "approved": s.get("approved", False)}

    g_hitl.them_nut("buoc_1", _buoc_1)
    g_hitl.them_canh(START, "buoc_1")
    g_hitl.them_canh("buoc_1", END)
    app_hitl = g_hitl.bien_dich(interrupt_before=["buoc_1"])
    paused = app_hitl.thuc_thi({"approved": False}, thread_id="start_hitl")
    assert paused["__tam_dung__"] is True
    assert paused["__nut_hien_tai__"] == "buoc_1"
    assert da_chay_buoc_1 == []

    resumed = app_hitl.tiep_tuc(thread_id="start_hitl", trang_thai_cap_nhat={"approved": True})
    assert resumed["__tam_dung__"] is False
    assert resumed["approved"] is True
    assert len(da_chay_buoc_1) == 1

    master = NutChinh(bat_tuong_lua=False)
    master.them_worker(port=19091, bo_xu_ly=lambda data: f"xu_ly:{data}")
    g_cum = DoThiTrangThai()
    g_cum.them_nut_cum("cum", master)
    g_cum.them_canh(START, "cum")
    g_cum.them_canh("cum", END)
    res_cum = g_cum.bien_dich().thuc_thi({"cau_hoi": "xin chào cụm"})
    assert res_cum["tra_loi_cum"]["ket_qua"] == "xu_ly:xin chào cụm"


def test_p2_3_orchestrator_threading_localhost_and_bad_content_length() -> None:
    """Xác minh NutChinh/NutPhu dùng ThreadingHTTPServer, mặc định 127.0.0.1 và trả 400 cho Content-Length lỗi."""
    from vietnamese_ai.orchestrator.master import NutChinh, tao_master_handler
    from vietnamese_ai.orchestrator.worker import NutPhu, tao_handler

    master = NutChinh(bat_tuong_lua=False)
    assert master.host == "127.0.0.1"
    assert "ThreadingHTTPServer" in inspect.getsource(NutChinh.chay)
    assert "ThreadingHTTPServer" in inspect.getsource(NutPhu.chay)

    srv_worker = ThreadingHTTPServer(("127.0.0.1", 0), tao_handler(lambda d: d))
    port_w = srv_worker.server_address[1]
    t_w = threading.Thread(target=srv_worker.serve_forever, daemon=True)
    t_w.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port_w, timeout=3)
        conn.putrequest("POST", "/predict")
        conn.putheader("Content-Length", "khong_phai_so")
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()
    finally:
        srv_worker.shutdown()
        srv_worker.server_close()

    srv_master = ThreadingHTTPServer(("127.0.0.1", 0), tao_master_handler(master))
    port_m = srv_master.server_address[1]
    t_m = threading.Thread(target=srv_master.serve_forever, daemon=True)
    t_m.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port_m, timeout=3)
        conn.putrequest("POST", "/predict")
        conn.putheader("Content-Length", "-10")
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()
    finally:
        srv_master.shutdown()
        srv_master.server_close()


def test_p2_3_cskh_hi_word_boundary() -> None:
    """Xác minh TongDaiCSKH không nhận nhầm từ chứa chuỗi con 'hi' (như 'khi', 'chi phí') là lời chào."""
    from vietnamese_ai.solutions.cskh import TongDaiCSKH

    bot = TongDaiCSKH()
    kq_phi = bot.xu_ly_yeu_cau("Khi nào shop giao hàng và chi phí vận chuyển bao nhiêu?")
    assert kq_phi["y_dinh"] != "CHAO_HOI"

    kq_chao = bot.xu_ly_yeu_cau("Hi shop, cho mình hỏi chút")
    assert kq_chao["y_dinh"] == "CHAO_HOI"


def test_p2_3_power_manager_precision_ordering() -> None:
    """Xác minh PowerManager chọn 8-bit khi pin cao/sạc, 4-bit khi pin trung bình và 1.58-bit khi pin yếu."""
    from vietnamese_ai.mobile.power_manager import PowerManager

    with patch.object(PowerManager, "get_battery_status", return_value=(90.0, True)):
        cfg_cao = PowerManager.dieu_tiet_do_chinh_xac()
        assert cfg_cao["precision"] == "8-bit"
        assert cfg_cao["load_in_8bit"] is True

    with patch.object(PowerManager, "get_battery_status", return_value=(40.0, False)):
        cfg_tb = PowerManager.dieu_tiet_do_chinh_xac()
        assert cfg_tb["precision"] == "4-bit"
        assert cfg_tb["load_in_4bit"] is True

    with patch.object(PowerManager, "get_battery_status", return_value=(10.0, False)):
        cfg_yeu = PowerManager.dieu_tiet_do_chinh_xac()
        assert cfg_yeu["precision"] == "1.58-bit"
        assert cfg_yeu["use_bitnet"] is True
