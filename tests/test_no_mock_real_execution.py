"""Kiểm thử toàn diện xóa bỏ Mock/Giả lập — Xác minh Thực Thi Thật trên 12 Phân Hệ."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_gguf_doc_tensor_that_va_transformer_edge(tmp_path: Path):
    """Kiểm tra GGUFMemoryMappedStreamer đọc bảng tensor thật và BoGiaiMaTransformerEdge chạy suy luận thật."""
    from vietnamese_ai.edge import (
        BoGiaiMaTransformerEdge,
        DongCoSuyLuanEdge,
        GGUFMemoryMappedStreamer,
    )

    gguf_path = tmp_path / "real_tensor.gguf"
    w_q = np.eye(8, dtype=np.float32) * 0.5
    raw_w_q = w_q.tobytes()

    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 1))
        k1 = b"general.alignment"
        f.write(struct.pack("<Q", len(k1)) + k1 + struct.pack("<I", 4) + struct.pack("<I", 32))
        tname = b"blk.0.attn_q.weight"
        f.write(struct.pack("<Q", len(tname)) + tname)
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 8) + struct.pack("<Q", 8))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<Q", 0))
        pad = (32 - (f.tell() % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(raw_w_q)

    streamer = GGUFMemoryMappedStreamer(gguf_path)
    try:
        assert "blk.0.attn_q.weight" in streamer.danh_sach_tensor()
        loaded = streamer.doc_tensor("blk.0.attn_q.weight")
        assert loaded is not None
        assert loaded.shape == (8, 8)
        assert np.allclose(loaded, w_q)
    finally:
        streamer.close()

    tr = BoGiaiMaTransformerEdge(d_model=16, so_dau=4, d_ff=32)
    h0 = tr.buoc_suy_luan(np.ones(16, dtype=np.float32), vi_tri=0)
    h1 = tr.buoc_suy_luan(np.ones(16, dtype=np.float32) * 0.5, vi_tri=1)
    assert h0.shape == (1, 16) and h1.shape == (1, 16)
    assert not np.allclose(h0, h1)

    engine = DongCoSuyLuanEdge(duong_dan_mo_hinh=gguf_path, so_luong_thread=2)
    assert engine._che_do_suy_luan == "gguf_transformer_cpu"
    engine.nap_tri_thuc_cuc_bo(["Nghị định 13 quy định chi tiết bảo vệ dữ liệu cá nhân tại Việt Nam."])
    ans = engine.sinh_van_ban("Bảo vệ dữ liệu cá nhân theo Nghị định 13", do_dai_toi_da=20)
    assert "Nghị định 13" in ans or "dữ liệu cá nhân" in ans


def test_studio_dashboard_huan_luyen_backprop_thuc_te():
    """Kiểm tra StudioDashboard.chay_huan_luyen thực thi backpropagation thực tế bằng MangNron."""
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    dash = StudioDashboard(host="127.0.0.1", port=0)
    kq = dash.chay_huan_luyen({"model": "vgpt2", "task": "phan_loai", "epochs": 3})
    assert kq["trang_thai"] == "HOAN_THANH"
    assert kq["che_do_huan_luyen"] == "thuc_te_backprop_numpy"
    assert kq["thoi_gian_giay"] >= 0.0
    assert len(kq["lich_su_loss"]) >= 3
    assert kq["loss_cuoi_cung"] <= kq["loss_khoi_tao"] + 0.5


def test_tee_wasm_webxr_rfc5114_va_p2p_so_cai_bam(tmp_path: Path):
    """Kiểm tra SecureEdgeNode/WebBrowserNode dùng RFC 5114 (512 hex chars), WebXR 3D và TokenLedger hash-chain."""
    from vietnamese_ai.edge import (
        P2PTracker,
        SecureEdgeNode,
        TokenLedger,
        WebBrowserNode,
        WebXRNode,
    )

    node = SecureEdgeNode("edge_rfc5114", private_key="khoa_bi_mat_vn", auto_start=False)
    assert len(node.public_key) == 512
    proof = node.generate_proof("cau_hoi", "cau_tra_loi")
    assert SecureEdgeNode.verify_proof("edge_rfc5114", node.public_key, "cau_hoi", "cau_tra_loi", proof, node.model_hash) is True
    assert SecureEdgeNode.verify_proof("edge_rfc5114", node.public_key, "cau_hoi", "sua_doi", proof, node.model_hash) is False

    wasm = WebBrowserNode("sess_real")
    assert len(wasm.public_key) == 512
    out_wasm = wasm.sinh_van_ban("Chào bạn, bạn là ai?", do_dai=20)
    assert "[WASM-WebGPU]" in out_wasm

    xr = WebXRNode("xr_real")
    info_xr = xr.cap_nhat_telemetry({"toa_do": {"x": 3.0, "y": 4.0, "z": 0.0}, "goc_nhin": {"pitch": 0.0, "yaw": 90.0}})
    assert info_xr["khoang_cach_goc"] == 5.0
    assert info_xr["huong_nhin_don_vi"][0] == pytest.approx(1.0, abs=1e-3)
    out_xr = xr.sinh_van_ban("Xin chào")
    assert "dist=5.0" in out_xr

    ledger_path = tmp_path / "ledger.jsonl"
    ledger = TokenLedger(duong_dan_so_cai=ledger_path)
    ledger.thuong_token("node_1", 10.0)
    ledger.thuong_token_danh_gia("user_1", 2.0)
    assert ledger.xac_minh_toan_ven_so_cai() is True
    ledger_reloaded = TokenLedger(duong_dan_so_cai=ledger_path)
    assert ledger_reloaded.lay_so_du("node_1") == 10.0
    assert ledger_reloaded.xac_minh_toan_ven_so_cai() is True
    ledger_reloaded.lich_su_giao_dich[0]["so_luong"] = 9999.0
    assert ledger_reloaded.xac_minh_toan_ven_so_cai() is False

    tracker = P2PTracker(duong_dan_danh_ba=tmp_path / "peers.json")
    tracker.dang_ky_node("local_1", wasm, toc_do_du_kien=10.0)
    assert tracker.kiem_tra_suc_khoe_node("local_1") is True


def test_browser_copilot_khong_sinh_bao_cao_gia_va_chan_ssrf():
    """Kiểm tra BrowserCopilot không bịa báo cáo quý 3 và chặn địa chỉ SSRF nội bộ."""
    from vietnamese_ai.mobile.browser_copilot import BrowserCopilot

    copilot = BrowserCopilot(mobile_agent=MagicMock())
    ssrf_res = copilot.cao_du_lieu_web("http://127.0.0.1:8080/admin")
    assert "SSRF" in ssrf_res
    assert "lợi nhuận tăng 20%" not in ssrf_res

    bad_scheme = copilot.cao_du_lieu_web("ftp://khong-ho-tro.vn/data")
    assert "lợi nhuận tăng 20%" not in bad_scheme


def test_grpc_experience_memory_wal_va_datalake_thu_muc(tmp_path: Path):
    """Kiểm tra EvoNetServiceServicer thật, SoTayKinhNghiem vector cosine, MessageQueue WAL và HoDuLieu.nhap_tu_thu_muc."""
    from vietnamese_ai.agents.experience_memory import SoTayKinhNghiem
    from vietnamese_ai.api.grpc_server import EvoNetServiceServicer
    from vietnamese_ai.datalake.dataset_manager import HoDuLieu
    from vietnamese_ai.realtime.pipeline import MessageQueue

    class _Req:
        model = "edge-cpu"
        prompt = "Chào bạn, bạn là ai?"

    srv = EvoNetServiceServicer(bo_xu_ly=None)
    res = srv.ChatCompletion(_Req(), None)
    assert "[gRPC Native]" in res.text
    assert "Trợ lý AI Tiếng Việt" in res.text or "Edge Native" in res.text

    so_tay = SoTayKinhNghiem(vector_store=object())
    so_tay.ghi_nhan_bai_hoc(
        "Lỗi giải lượng tử hóa GGUF bị tràn RAM khi đọc ma trận lớn",
        "Đọc toàn bộ ma trận FP32 cùng lúc",
        "Dùng nhan_ma_tran_luong_tu phân ô tile",
    )
    goi_y = so_tay.truy_xuat_kinh_nghiem("Xử lý tràn RAM giải lượng tử GGUF")
    assert goi_y is not None and "nhan_ma_tran_luong_tu" in goi_y

    wal_file = tmp_path / "mq.jsonl"
    mq1 = MessageQueue("orders", duong_dan_wal=str(wal_file))
    mq1.publish("events", {"don": "DH01"})
    mq2 = MessageQueue("orders", duong_dan_wal=str(wal_file))
    assert len(mq2.consume("events", so_luong=5)) == 1

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "luat_1.txt").write_text("Điều 1. Quy định chung về giao dịch điện tử.", encoding="utf-8")
    lake = HoDuLieu(str(tmp_path / "lake"))
    kq_dir = lake.nhap_tu_thu_muc(str(corpus_dir))
    assert kq_dir["so_ban_ghi"] == 1
    assert kq_dir["la_du_lieu_thuc_te"] is True
    kq_aug = lake.sinh_them_mau("uit_vsfc", so_luong=6)
    assert kq_aug["co_tang_cuong_nlp"] is True
    assert all("(mẫu chuẩn)" not in str(item.get("text", "")) for item in kq_aug["du_lieu"])


def test_multimodal_ocr_vlm_audio_webrtc_va_benchmark_live():
    """Kiểm tra OCR báo cờ engine rõ, VLM huấn luyện chiếu Ridge thật, ASR/WebRTC động và Benchmark live."""
    from vietnamese_ai.audio.speech import NhanDienGiongNoi
    from vietnamese_ai.audio.webrtc_bridge import CauNoiWebRTC
    from vietnamese_ai.benchmark import BoDoLuongDoiDau
    from vietnamese_ai.plugin.market_feed import BoDieuPhoiThiTruong
    from vietnamese_ai.vision.ocr import TrichXuatVanBanAnh
    from vietnamese_ai.vision.vlm import MoHinhThiGiacNgonNgu

    ocr = TrichXuatVanBanAnh(bat_bao_ve_anh=False)
    arr_img = np.full((40, 80), 255, dtype=np.uint8)
    arr_img[10:20, 10:70] = 0
    kq_ocr = ocr.doc_van_ban(arr_img)
    assert "co_ocr_engine" in kq_ocr and "nguon_ocr" in kq_ocr

    vlm = MoHinhThiGiacNgonNgu(che_do_offline=True, bat_bao_ve=False)
    img_b = np.full((40, 80), 220, dtype=np.uint8)
    kq_proj = vlm.huan_luyen_chieu_lien_phuong_thuc(
        [(arr_img, "Hóa đơn thanh toán điện tử"), (img_b, "Văn bản hành chính nhà nước")]
    )
    assert kq_proj["da_can_chinh_chieu"] is True
    assert kq_proj["mse_can_chinh"] < 1.0

    asr = NhanDienGiongNoi(che_do_offline=True)
    kq_asr_empty = asr.nhan_dien(np.zeros(1600, dtype=np.float32))
    assert kq_asr_empty["co_asr_backend"] is False
    assert kq_asr_empty["do_tin_cay"] == 0.0

    rtc = CauNoiWebRTC()
    ans = rtc.xu_ly_offer(
        "phien_99",
        "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 109\r\na=rtpmap:109 opus/48000/2\r\na=setup:active\r\n",
    )
    assert ans["dtls_setup"] == "passive"
    assert "109" in ans["sdp"]
    assert "20261109" not in ans["sdp"]

    feed = BoDieuPhoiThiTruong(dung_gia_lap_khi_loi=True)
    syn = feed.mock_engine.sinh_gia_hien_tai("BTCUSDT")
    assert syn["la_du_lieu_gia_lap"] is True

    live = BoDoLuongDoiDau().do_doi_dau_truc_tiep()
    assert live["vietnamese_ai"]["loai_baseline"] == "do_truc_tiep"
    assert live["vietnamese_ai"]["cold_start_subprocess_ms"] > 0


def test_rag_ring_attention_swarm_lm_head_mobile_honest_va_p2p_rpc():
    """Kiểm tra RingAttention thực trong RAG, LM-head Swarm, cờ thật di động và HTTP RPC P2P Swarm."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from vietnamese_ai.agents.decentralized_swarm import P2PSwarmOrchestrator
    from vietnamese_ai.agents.swarm import TacTuSwarm
    from vietnamese_ai.distributed.ring_attention import RingAttentionNode
    from vietnamese_ai.edge import DieuPhoiSwarmCPU, P2PTracker
    from vietnamese_ai.mobile.deployment import TriKhaiDiDong
    from vietnamese_ai.rag.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.them_tai_lieu(
        ma="luat_10",
        van_ban="Điều 10 Luật Giao dịch điện tử quy định giá trị pháp lý của thông điệp dữ liệu.",
    )
    kq_rag = rag.hoi("Giá trị pháp lý của thông điệp dữ liệu là gì?")
    assert "extreme_hardware" not in kq_rag
    assert kq_rag["so_luong_nguon"] >= 1

    ring_node = RingAttentionNode(d_model=32, node_id=0, total_nodes=1)
    x_ring = np.random.RandomState(42).randn(1, 4, 32).astype(np.float32)
    ring_out = ring_node.forward_ring(x_ring, x_ring, x_ring)
    assert ring_out.shape == x_ring.shape

    cum = DieuPhoiSwarmCPU(d_model=16, tong_so_layer=4)
    cum.khoi_tao_cum_gia_lap(so_node=2)
    assert cum._lm_head_matrix.shape[1] == 16
    tokens = list(cum.suy_luan_streaming("Xin chào trí tuệ nhân tạo Việt Nam", so_token=4))
    assert len(tokens) == 4

    class _DummyModel:
        coef_ = np.array([[0.5, -0.2]])
        intercept_ = np.array([0.1])
        classes_ = np.array([0, 1])

        def predict(self, x: np.ndarray) -> np.ndarray:
            return (x @ self.coef_.T + self.intercept_ > 0).astype(int).ravel()

    tkdd = TriKhaiDiDong()
    bm = tkdd.benchmark_edge(_DummyModel(), kich_thuoc_dau_vao=(1, 2), so_lan=5)
    assert bm["la_thiet_bi_di_dong_that"] is False
    assert str(bm["thiet_bi_do"]).startswith("host_cpu:")

    class _RPCHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
            body = json.dumps({"result": f"RPC_OK:{req.get('prompt', '')}"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), _RPCHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        class _RemoteNode:
            node_id = "remote_http_1"
            endpoint_url = f"http://127.0.0.1:{server.server_port}/run"

        agent_goc = TacTuSwarm(ten="Root", vai_tro="DieuPhoi", llm=MagicMock())
        orch = P2PSwarmOrchestrator(agent_khoi_tao=agent_goc, tracker=P2PTracker())
        sub = TacTuSwarm(ten="Sub1", vai_tro="Worker", llm=MagicMock())
        kq_rpc = orch._chay_node_tu_xa(_RemoteNode(), sub, "kiem_tra_rpc")
        assert kq_rpc == "RPC_OK:kiem_tra_rpc"
    finally:
        server.shutdown()
        server.server_close()
