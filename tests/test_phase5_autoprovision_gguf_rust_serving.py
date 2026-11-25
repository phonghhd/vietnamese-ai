"""Kiểm thử toàn diện Giai đoạn 5: Tự động cấp phát 3 tầng, GGUF v3 & BPE, Rust Q4_K/Tokenizer/IVF, mTLS 2 chiều & OpenAI SSE/Tools/Metrics."""

from __future__ import annotations

import json
import shutil
import socket
import ssl
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


def test_tu_dong_cap_phat_3_tang_va_vai_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Kiểm tra cơ chế Opt-in Auto-Provisioning, nhận diện phần cứng, không xung đột extras, cache PEP 668 và CLI `vai setup`."""
    from vietnamese_ai.cli.main import _tao_parser, cmd_setup
    from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc

    QuanLyPhuThuoc.bat_tu_dong_cai(False)
    monkeypatch.delenv("VAI_AUTO_INSTALL", raising=False)
    assert QuanLyPhuThuoc.dang_bat_tu_dong_cai() is False

    monkeypatch.setenv("VAI_AUTO_INSTALL", "1")
    assert QuanLyPhuThuoc.dang_bat_tu_dong_cai() is True
    monkeypatch.delenv("VAI_AUTO_INSTALL", raising=False)

    audit_deps = QuanLyPhuThuoc.kiem_tra_xung_dot_phu_thuoc()
    assert audit_deps["hop_le"] is True
    assert audit_deps["so_xung_dot"] == 0
    assert "audio" in QuanLyPhuThuoc.NHOM_PHU_THUOC
    assert "ocr-vn" in QuanLyPhuThuoc.NHOM_PHU_THUOC
    assert any("python-docx" in g for g in QuanLyPhuThuoc.NHOM_PHU_THUOC["legal-rag"]["goi"])
    assert any("pdfplumber" in g for g in QuanLyPhuThuoc.NHOM_PHU_THUOC["legal-rag"]["goi"])

    from vietnamese_ai.nlp.native_tokenizer import co_ho_tro_rust

    co_rust = co_ho_tro_rust()
    hw = QuanLyPhuThuoc.phat_hien_phan_cung_cai_dat()
    assert hw["so_nhan_cpu"] >= 1
    assert hw["co_rust_native"] is co_rust

    setup_res = QuanLyPhuThuoc.thiet_lap_tu_dong(profile="legal-rag", chay_that=False)
    assert setup_res["ho_so"] == "legal-rag"
    assert setup_res["che_do_thuc_thi"] == "xem_truoc_dry_run"
    assert setup_res["tang_thap_rust"]["co_rust_native"] is co_rust
    assert len(setup_res["cac_nhom_goi"]) == 3

    parser = _tao_parser()
    args = parser.parse_args(["setup", "--profile", "edge"])
    cli_res = cmd_setup(args)
    assert cli_res["ho_so"] == "edge"
    assert cli_res["thanh_cong"] is True

    runtime_dir = tmp_path / "vai_runtime_cache"
    monkeypatch.setenv("VAI_RUNTIME_DIR", str(runtime_dir))

    def _fake_cai_dat(nhom: str, chay_that: bool = False, dung_cache_neu_can: bool = False) -> Dict[str, Any]:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "_vai_demo_auto_mod.py").write_text("GIA_TRI = 2026\n", encoding="utf-8")
        QuanLyPhuThuoc.gan_thu_muc_runtime_vao_sys_path()
        return {"nhom": nhom, "thanh_cong": True, "da_thuc_thi": True}

    monkeypatch.setattr(QuanLyPhuThuoc, "cai_dat_nhom", classmethod(lambda cls, *a, **kw: _fake_cai_dat(*a, **kw)))
    mod = QuanLyPhuThuoc.yeu_cau("_vai_demo_auto_mod", nhom_extra="nlp", tinh_nang="kiem_thu_auto", tu_dong_cai=True)
    assert mod.GIA_TRI == 2026


def test_gguf_v3_chuan_quoc_te_va_bpe_end_to_end(tmp_path: Path):
    """Kiểm tra XuatGGUF xuất chuẩn GGUF v3 đọc được bởi GGUFMemoryMappedStreamer và suy luận BPE đầu-cuối."""
    from vietnamese_ai.edge import (
        BoGiaiMaTransformerEdge,
        BoTachTuBPEGGUF,
        GGUFMemoryMappedStreamer,
    )
    from vietnamese_ai.export.gguf import XuatGGUF

    vocab = ["<pad>", "<s>", "</s>", "<unk>", "xin", "chào", "việt", "nam", "trí", "tuệ", "nhân", "tạo", "đẹp", "tốt", "ai", "cpu"]
    rng = np.random.RandomState(123)
    d_model = 64
    weights = {
        "token_embd.weight": (rng.randn(len(vocab), d_model) * 0.1).astype(np.float32),
        "output.weight": (rng.randn(len(vocab), d_model) * 0.1).astype(np.float32),
        "blk.0.attn_q.weight": (rng.randn(d_model, d_model) * 0.05).astype(np.float32),
        "blk.0.attn_k.weight": (rng.randn(32, d_model) * 0.05).astype(np.float32),
        "blk.0.attn_v.weight": (rng.randn(32, d_model) * 0.05).astype(np.float32),
        "blk.0.attn_output.weight": (rng.randn(d_model, d_model) * 0.05).astype(np.float32),
    }

    gguf_path = tmp_path / "vietnamese_bpe_q4.gguf"
    exporter = XuatGGUF()
    exporter.xuat_tu_numpy(
        weights,
        str(gguf_path),
        quantization="q4_0",
        thong_tin={
            "name": "VietnameseAI-Mini-BPE",
            "architecture": "qwen2",
            "tokenizer.ggml.tokens": vocab,
            "tokenizer.ggml.merges": ["x in", "ch ào"],
        },
    )

    meta_exp = exporter.doc_metadata(str(gguf_path))
    assert meta_exp["version"] == 3
    assert meta_exp["n_tensors"] == 6
    assert meta_exp["metadata"]["tokenizer.ggml.tokens"] == vocab

    loaded_t = exporter.tai_tu_gguf(str(gguf_path), loc_tien_to=["blk.0."])
    assert len(loaded_t) == 4
    assert loaded_t["blk.0.attn_q.weight"].shape == (64, 64)

    streamer = GGUFMemoryMappedStreamer(gguf_path)
    try:
        assert streamer.metadata["tokenizer.ggml.tokens"] == vocab
        assert streamer.metadata["tokenizer.ggml.merges"] == ["x in", "ch ào"]
        bpe = BoTachTuBPEGGUF.tu_streamer(streamer)
        assert bpe is not None
        enc = bpe.ma_hoa("xin chào việt nam")
        assert enc == [4, 5, 6, 7]
        assert bpe.giai_ma(enc) == "xin chào việt nam"

        tr = BoGiaiMaTransformerEdge(d_model=64, so_dau=4, so_dau_kv=2, d_ff=128, so_lop=1)
        kq = tr.sinh_tu_gguf_bpe("xin chào việt nam", streamer, so_token_toi_da=5, nhiet_do=0.0)
        assert kq["che_do"] == "gguf_bpe_end_to_end"
        assert kq["vocab_size"] == len(vocab)
        assert len(kq["generated_ids"]) >= 1
        assert len(tr._trong_so_luong_tu) == 4
    finally:
        streamer.close()


def test_rust_q4_k_zero_copy_tokenizer_dau_thanh_va_ivf_vector(monkeypatch: pytest.MonkeyPatch):
    """Kiểm tra khớp chính xác Q4_K Rust zero-copy, normalize_and_segment dấu thanh và IVF-Flat/Rust top-k cosine."""
    import vietnamese_ai.edge.native_inference as ni_mod
    import vietnamese_ai.nlp.native_tokenizer as nt_mod
    from vietnamese_ai.edge.native_inference import BoGiaiLuongTuQ4
    from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc
    from vietnamese_ai.rag.vector_store import CSDLVector

    rng = np.random.RandomState(99)
    rows, k, out_cols = 2, 256, 8
    raw_q4k = bytearray()
    for _ in range(out_cols):
        raw_q4k.extend(np.float16(0.05).tobytes())
        raw_q4k.extend(np.float16(0.02).tobytes())
        raw_q4k.extend(rng.randint(0, 64, size=12, dtype=np.uint8).tobytes())
        raw_q4k.extend(rng.randint(0, 256, size=128, dtype=np.uint8).tobytes())
    raw_bytes = bytes(raw_q4k)
    x = rng.randn(rows, k).astype(np.float32)

    y_rust = BoGiaiLuongTuQ4.nhan_ma_tran_luong_tu(x, raw_bytes, (out_cols, k), "q4_k")
    saved_rc = ni_mod._rust_core
    monkeypatch.setattr(ni_mod, "_rust_core", None)
    y_numpy = BoGiaiLuongTuQ4.nhan_ma_tran_luong_tu(x, raw_bytes, (out_cols, k), "q4_k")
    monkeypatch.setattr(ni_mod, "_rust_core", saved_rc)
    assert np.allclose(y_rust, y_numpy, rtol=1e-5, atol=1e-3)

    cau_dau_cu = "Ủy ban nhân dân tỉnh Hoà Bình gửi thuý kiều lời chúc KHOẺ mạnh."
    tok_rs = BoTachTuTiengVietSieuToc()
    kq_rs = tok_rs.tach_tu(cau_dau_cu, noi_tu_ghep=True, chuan_hoa_dau=True)
    assert "Hòa" in kq_rs and "thúy" in kq_rs and "KHỎE" in kq_rs

    saved_nt = nt_mod._rust_core
    monkeypatch.setattr(nt_mod, "_rust_core", None)
    tok_py = BoTachTuTiengVietSieuToc()
    kq_py = tok_py.tach_tu(cau_dau_cu, noi_tu_ghep=True, chuan_hoa_dau=True)
    monkeypatch.setattr(nt_mod, "_rust_core", saved_nt)
    assert kq_rs == kq_py

    store = CSDLVector(kich_thuoc=32, khoang_cach="cosine")
    vecs = rng.randn(200, 32).astype(np.float32)
    store.chen_batch([f"doc_{i}" for i in range(200)], vecs)
    res_rust = store.tim_kiem(vecs[42], top_k=3)
    assert res_rust[0]["ma"] == "doc_42"
    assert res_rust[0]["diem"] > 0.999

    ivf_info = store.xay_dung_chi_muc_ivf(so_cum=8, nprobe=3, seed=42)
    assert ivf_info["da_xay_dung"] is True
    res_ivf = store.tim_kiem(vecs[42], top_k=3)
    assert res_ivf[0]["ma"] == "doc_42"
    assert res_ivf[0]["diem"] > 0.999


def test_swarm_cpu_lan_mtls_2_chieu_day_du(tmp_path: Path):
    """Kiểm tra xác thực mTLS 2 chiều đầy đủ (CA + Server Cert + Client Cert) trên Swarm CPU LAN."""
    from vietnamese_ai.edge import (
        NutSwarmCPU,
        tao_ssl_context_client,
        tao_ssl_context_server,
    )

    if not shutil.which("openssl"):
        pytest.skip("Máy không có openssl để sinh bộ chứng chỉ CA/Server/Client mTLS")

    def _run_openssl(args_list: list[str]) -> None:
        proc = subprocess.run(["openssl", *args_list], capture_output=True, check=False)
        if proc.returncode != 0:
            pytest.skip(f"OpenSSL không hỗ trợ sinh chứng chỉ: {proc.stderr.decode(errors='ignore')}")

    ca_key, ca_crt = tmp_path / "ca.key", tmp_path / "ca.crt"
    srv_key, srv_csr, srv_crt = tmp_path / "srv.key", tmp_path / "srv.csr", tmp_path / "srv.crt"
    cli_key, cli_csr, cli_crt = tmp_path / "cli.key", tmp_path / "cli.csr", tmp_path / "cli.crt"
    ext_file = tmp_path / "srv.ext"
    ext_file.write_text("subjectAltName=IP:127.0.0.1,DNS:localhost\n", encoding="utf-8")

    _run_openssl(
        [
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(ca_key),
            "-out",
            str(ca_crt),
            "-days",
            "1",
            "-subj",
            "/CN=VietnameseAI-CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
        ]
    )
    _run_openssl(["req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(srv_key), "-out", str(srv_csr), "-subj", "/CN=127.0.0.1"])
    _run_openssl(["x509", "-req", "-in", str(srv_csr), "-CA", str(ca_crt), "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(srv_crt), "-days", "1", "-extfile", str(ext_file)])
    _run_openssl(["req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(cli_key), "-out", str(cli_csr), "-subj", "/CN=swarm-client-1"])
    _run_openssl(["x509", "-req", "-in", str(cli_csr), "-CA", str(ca_crt), "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(cli_crt), "-days", "1"])

    srv_ctx = tao_ssl_context_server(str(srv_crt), str(srv_key), cafile=str(ca_crt), yeu_cau_mtls=True)
    cli_ok_ctx = tao_ssl_context_client(cafile=str(ca_crt), certfile=str(cli_crt), keyfile=str(cli_key), kiem_tra_hostname=True)
    cli_no_cert_ctx = tao_ssl_context_client(cafile=str(ca_crt), kiem_tra_hostname=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_tmp:
        s_tmp.bind(("127.0.0.1", 0))
        port = s_tmp.getsockname()[1]

    worker = NutSwarmCPU("mtls_worker", dai_layer=(0, 2), d_model=16)
    worker.khoi_dong_server_lan(host="127.0.0.1", port=port, ssl_context=srv_ctx)
    try:
        time.sleep(0.1)
        client_ok = NutSwarmCPU(
            "mtls_cli_ok",
            dai_layer=(0, 2),
            endpoint_url=f"tls://127.0.0.1:{port}",
            che_do="mang_vat_ly_lan",
            d_model=16,
            ssl_context=cli_ok_ctx,
        )
        y_out, proof = client_ok._truyen_tai_qua_socket_lan(np.ones((1, 16), dtype=np.float32), tao_proof=True)
        assert y_out.shape == (1, 16)
        assert proof is not None and proof["proof_zk"].startswith("zkml1:")

        client_bad = NutSwarmCPU(
            "mtls_cli_bad",
            dai_layer=(0, 2),
            endpoint_url=f"tls://127.0.0.1:{port}",
            che_do="mang_vat_ly_lan",
            d_model=16,
            ssl_context=cli_no_cert_ctx,
        )
        with pytest.raises((ssl.SSLError, ConnectionResetError, OSError)):
            client_bad._truyen_tai_qua_socket_lan(np.ones((1, 16), dtype=np.float32), tao_proof=False)
    finally:
        worker.dung_server_lan()


def test_fastapi_openai_sse_tool_calling_va_prometheus_metrics():
    """Kiểm tra SSE streaming chuẩn OpenAI, Tool Calling JSON Schema và Prometheus/OTel metrics."""
    import asyncio

    from vietnamese_ai.api.fastapi_server import (
        _CO_FASTAPI,
        BoThuThapTelemetryAPI,
        ChatCompletionRequest,
        ChatMessage,
        FastAPIServer,
    )

    tel = BoThuThapTelemetryAPI("test-svc")
    with tel.bat_dau_span("unit.test"):
        tel.ghi_nhan(0.05, prompt_tokens=10, completion_tokens=20, la_stream=True, co_tool_call=True)
    prom = tel.xuat_prometheus()
    assert 'vai_api_requests_total{service="test-svc"} 1' in prom
    assert 'vai_api_stream_requests_total{service="test-svc"} 1' in prom
    assert 'vai_api_tool_calls_total{service="test-svc"} 1' in prom

    if not _CO_FASTAPI:
        return

    class _DummyEngine:
        def sinh_van_ban(self, prompt: str, nhiet_do: float = 0.7, do_dai: int = 128) -> str:
            return "Xin chào từ Vietnamese AI Production Gateway"

    srv = FastAPIServer(_DummyEngine(), ten="TestGateway", enable_watermark=False)
    route_map = {getattr(r, "path", ""): getattr(r, "endpoint", None) for r in srv.app.routes}
    chat_ep = route_map["/v1/chat/completions"]
    metrics_ep = route_map["/metrics"]

    req_sse = ChatCompletionRequest(
        model="vietnamese-ai",
        messages=[ChatMessage(role="user", content="Xin chào")],
        stream=True,
    )
    resp_sse = asyncio.run(chat_ep(req_sse))
    assert "text/event-stream" in resp_sse.media_type

    async def _collect_sse() -> str:
        chunks = []
        async for chunk in resp_sse.body_iterator:
            chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
        return "".join(chunks)

    sse_text = asyncio.run(_collect_sse())
    assert "chat.completion.chunk" in sse_text
    assert "data: [DONE]" in sse_text

    req_tool = ChatCompletionRequest(
        model="vietnamese-ai",
        messages=[ChatMessage(role="user", content="Tra cứu thuế thu nhập cá nhân")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "tinh_thue_tncn",
                    "description": "Tính thuế thu nhập cá nhân Việt Nam",
                    "parameters": {"type": "object", "properties": {"cau_hoi": {"type": "string"}}},
                },
            }
        ],
        tool_choice="auto",
    )
    resp_tool = asyncio.run(chat_ep(req_tool))
    assert resp_tool.choices[0].finish_reason == "tool_calls"
    tc = resp_tool.choices[0].message.tool_calls[0]
    assert tc["function"]["name"] == "tinh_thue_tncn"
    assert "cau_hoi" in json.loads(tc["function"]["arguments"])

    resp_metrics = asyncio.run(metrics_ep())
    metrics_text = resp_metrics.body.decode("utf-8") if hasattr(resp_metrics, "body") else str(resp_metrics)
    assert "vai_api_requests_total" in metrics_text
    assert "vai_api_stream_requests_total" in metrics_text
    assert "vai_api_tool_calls_total" in metrics_text
