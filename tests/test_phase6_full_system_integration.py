"""Kiểm thử tích hợp toàn hệ thống Giai đoạn 6 — xác minh nối chặt cả 8 mối nối liên phân hệ."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

import vietnamese_ai


def _tao_tep_docx_phap_ly(duong_dan: Path) -> Path:
    """Tạo tệp DOCX OOXML hợp lệ chứa quy định pháp lý và bảng biểu hành chính."""
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Nghị định số 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân</w:t></w:r></w:p>
    <w:p><w:r><w:t>Điều 9. Quyền của chủ thể dữ liệu cá nhân: Chủ thể dữ liệu có quyền được biết, quyền đồng ý và quyền truy cập dữ liệu của mình.</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Nhóm dữ liệu</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Thời hạn xử lý</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Dữ liệu cá nhân nhạy cảm</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>72 giờ kể từ khi yêu cầu</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>"""
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
</Types>"""
    with zipfile.ZipFile(str(duong_dan), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("word/document.xml", document_xml)
    return duong_dan


def _tao_tep_gguf_bpe_thuc_te(duong_dan: Path, d_model: int = 32) -> Path:
    """Xuất tệp GGUF v3 chuẩn quốc tế có đầy đủ mảng BPE tokenizer, embedding và trọng số lượng tử Q8_0."""
    from vietnamese_ai.export.gguf import XuatGGUF

    rng = np.random.RandomState(2026)
    vocab = ["<s>", "</s>", "<unk>", "▁Việt", "▁Nam", "▁pháp", "▁luật", "▁trí", "▁tuệ", "▁nhân", "▁tạo"]
    tensors = {
        "token_embd.weight": (rng.randn(len(vocab), d_model) * 0.2).astype(np.float32),
        "output.weight": (rng.randn(len(vocab), d_model) * 0.2).astype(np.float32),
        "blk.0.attn_q.weight": (rng.randn(d_model, d_model) * 0.1).astype(np.float32),
        "blk.0.attn_k.weight": (rng.randn(d_model // 2, d_model) * 0.1).astype(np.float32),
        "blk.0.attn_v.weight": (rng.randn(d_model // 2, d_model) * 0.1).astype(np.float32),
        "blk.0.attn_output.weight": (rng.randn(d_model, d_model) * 0.1).astype(np.float32),
        "blk.0.ffn_gate.weight": (rng.randn(d_model * 2, d_model) * 0.1).astype(np.float32),
        "blk.0.ffn_up.weight": (rng.randn(d_model * 2, d_model) * 0.1).astype(np.float32),
        "blk.0.ffn_down.weight": (rng.randn(d_model, d_model * 2) * 0.1).astype(np.float32),
    }
    exporter = XuatGGUF()
    exporter.xuat_tu_numpy(
        tensors,
        str(duong_dan),
        quantization="q8_0",
        thong_tin={
            "name": "vietnamese-edge-bpe-mini",
            "architecture": "llama",
            "embedding_length": d_model,
            "block_count": 1,
            "head_count": 4,
            "head_count_kv": 2,
            "tokenizer.ggml.tokens": vocab,
            "tokenizer.ggml.merges": ["▁V iệt", "▁N am"],
        },
    )
    return duong_dan


def test_moi_noi_1_dong_co_edge_tu_dong_suy_luan_gguf_bpe(tmp_path: Path) -> None:
    """Mối nối 1: DongCoSuyLuanEdge tự nhận diện từ vựng BPE GGUF, căn chỉnh GQA và stream token trực tiếp."""
    gguf_file = _tao_tep_gguf_bpe_thuc_te(tmp_path / "edge_bpe.gguf", d_model=32)

    engine = vietnamese_ai.DongCoSuyLuanEdge(duong_dan_mo_hinh=gguf_file, do_dai_toi_da=6, nhiet_do=0.0)
    assert engine._che_do_suy_luan == "gguf_bpe_end_to_end"
    assert engine._bpe_gguf is not None
    assert engine._transformer.d_model == 32
    assert engine._transformer.so_dau_kv == 2

    pieces = list(engine.sinh_stream("Việt Nam", do_dai_toi_da=5, nhiet_do=0.0))
    assert len(pieces) >= 1
    van_ban_stream = "".join(pieces).strip()
    assert van_ban_stream

    stats = engine.sinh_kem_thong_ke("Việt Nam", do_dai_toi_da=5, nhiet_do=0.0)
    assert stats["che_do_suy_luan"] == "gguf_bpe_end_to_end"
    assert stats["co_bpe_gguf"] is True
    assert stats["so_tensor_giu_luong_tu"] >= 5


def test_moi_noi_2_rag_pipeline_doc_docx_thu_muc_va_chi_muc_ivf(tmp_path: Path) -> None:
    """Mối nối 2: RAGPipeline nạp .docx/.txt qua BoDocTaiLieuPhapLy và tối ưu truy hồi bằng IVF-Flat."""
    _tao_tep_docx_phap_ly(tmp_path / "nghi_dinh_13.docx")
    txt_file = tmp_path / "luat_lao_dong.txt"
    txt_file.write_text(
        "Bộ luật Lao động 45/2019/QH14\nĐiều 105. Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày.",
        encoding="utf-8",
    )

    rag = vietnamese_ai.RAGPipeline(che_do_hanh_chinh=True)
    kq_dir = rag.them_thu_muc(tmp_path, che_do_hanh_chinh=True)
    assert kq_dir["so_tep_thanh_cong"] == 2
    assert kq_dir["tong_chunks"] >= 2
    assert not kq_dir["cac_tep_loi"]

    ivf_res = rag.toi_uu_chi_muc_vector(so_cum=2, nprobe=2)
    assert ivf_res["da_xay_dung"] is True
    assert rag.thong_ke()["chi_muc_ivf"] is True

    kq = rag.hoi("Chủ thể dữ liệu cá nhân có quyền gì theo Nghị định 13/2023/NĐ-CP?")
    assert "dữ liệu" in kq["tra_loi"].lower()
    assert "72 giờ" in kq["tra_loi"] or "chủ thể" in kq["tra_loi"].lower()
    assert kq["trich_dan_phap_ly"]["so_hieu_van_ban"]


def test_moi_noi_3_fastapi_stream_khong_dem_va_kham_pha_tools_tu_dong() -> None:
    """Mối nối 3: FastAPIServer stream token trực tiếp (TTFT không đệm) và tự khám phá/thực thi công cụ TacTu."""
    fastapi_mod = pytest.importorskip("fastapi")
    _ = fastapi_mod
    from vietnamese_ai.agents.agent import TacTu
    from vietnamese_ai.agents.tools import CongCu
    from vietnamese_ai.api.fastapi_server import ChatCompletionRequest, ChatMessage, FastAPIServer

    da_sinh_xong = {"done": False}

    class _StreamingHandler:
        def sinh_stream(self, prompt: str, do_dai_toi_da: int = 16, nhiet_do: float = 0.0) -> Iterator[str]:
            _ = (prompt, do_dai_toi_da, nhiet_do)
            yield "Xin "
            yield "chào "
            yield "Việt Nam"
            da_sinh_xong["done"] = True

    srv_stream = FastAPIServer(bo_xu_ly=_StreamingHandler(), ten="StreamTest", enable_watermark=False)
    chat_ep = next(r.endpoint for r in srv_stream.app.routes if getattr(r, "path", "") == "/v1/chat/completions")
    req_stream = ChatCompletionRequest(
        model="edge",
        messages=[ChatMessage(role="user", content="Chào bạn")],
        stream=True,
    )
    async def _doc_sse_khong_dem() -> tuple[str, str, bool, list[str]]:
        resp = await chat_ep(req_stream)
        it = resp.body_iterator
        c1 = str(await it.__anext__())
        c2 = str(await it.__anext__())
        done_after_second = da_sinh_xong["done"]
        con_lai: list[str] = []
        async for chunk in it:
            con_lai.append(str(chunk))
        return c1, c2, done_after_second, con_lai

    first_chunk, second_chunk, done_after_second, rest = asyncio.run(_doc_sse_khong_dem())
    assert "chat.completion.chunk" in first_chunk
    assert "Xin " in second_chunk
    assert done_after_second is False
    assert da_sinh_xong["done"] is True
    assert any("[DONE]" in c for c in rest)

    def _tra_cuu_thue(truy_van: str = "") -> str:
        return f"Thuế suất GTGT hiện hành cho '{truy_van}' là 10%."

    cong_cu_thue = CongCu(
        ten="tra_cuu_thue_gtgt",
        mo_ta="Tra cứu thuế suất giá trị gia tăng GTGT Việt Nam",
        ham_thuc_thi=_tra_cuu_thue,
    )
    tac_tu = TacTu(llm=_StreamingHandler(), danh_sach_cong_cu=[cong_cu_thue])
    srv_agent = FastAPIServer(
        bo_xu_ly=tac_tu,
        ten="AgentToolTest",
        enable_watermark=False,
        thuc_thi_tool_tu_dong=True,
    )
    tools_ep = next(r.endpoint for r in srv_agent.app.routes if getattr(r, "path", "") == "/v1/tools")
    tools_list = asyncio.run(tools_ep())
    assert tools_list["so_luong"] == 1
    assert tools_list["data"][0]["function"]["name"] == "tra_cuu_thue_gtgt"

    agent_chat_ep = next(
        r.endpoint for r in srv_agent.app.routes if getattr(r, "path", "") == "/v1/chat/completions"
    )
    req_tool = ChatCompletionRequest(
        model="agent",
        messages=[ChatMessage(role="user", content="Tra cứu thuế suất GTGT giúp tôi")],
        tool_choice="auto",
    )
    kq_tool = asyncio.run(agent_chat_ep(req_tool))
    assert kq_tool.choices[0].message.tool_calls[0]["function"]["name"] == "tra_cuu_thue_gtgt"
    assert "10%" in (kq_tool.choices[0].message.content or "")


def test_moi_noi_4_cli_tls_fastapi_rag_docx_va_doctor(tmp_path: Path) -> None:
    """Mối nối 4: CLI vai hỗ trợ cờ TLS/mTLS cho swarm, --backend fastapi cho serve, đọc DOCX trong rag và doctor đầy đủ."""
    from vietnamese_ai.cli.main import _tao_parser, cmd_doctor, cmd_rag

    parser = _tao_parser()
    swarm_args = parser.parse_args(
        ["swarm", "--host", "--tls-cert", "s.pem", "--tls-key", "s.key", "--tls-ca", "ca.pem", "--mtls"]
    )
    assert swarm_args.tls_cert == "s.pem"
    assert swarm_args.tls_key == "s.key"
    assert swarm_args.tls_ca == "ca.pem"
    assert swarm_args.mtls is True

    serve_args = parser.parse_args(["serve", "--model", "vgpt2", "--backend", "fastapi", "--port", "8123"])
    assert serve_args.backend == "fastapi"
    assert serve_args.port == 8123

    docx_file = _tao_tep_docx_phap_ly(tmp_path / "cli_legal.docx")
    rag_args = parser.parse_args(
        ["rag", "--docs", str(docx_file), "--query", "Quyền chủ thể dữ liệu cá nhân", "--admin"]
    )
    kq_rag = cmd_rag(rag_args)
    assert "dữ liệu" in kq_rag["tra_loi"].lower()

    bc = cmd_doctor()
    assert "co_rust_native" in bc
    assert "kien_truc_cpu" in bc
    assert "tu_dong_cai_bat" in bc
    assert "thu_muc_cache_runtime" in bc


def test_moi_noi_5_tu_dong_cap_phat_nhap_tuy_chon_tren_cac_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mối nối 5: QuanLyPhuThuoc.nhap_tuy_chon và yeu_cau hoạt động thống nhất với cờ Opt-in VAI_AUTO_INSTALL."""
    from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc

    QuanLyPhuThuoc.bat_tu_dong_cai(False)
    monkeypatch.delenv("VAI_AUTO_INSTALL", raising=False)
    assert QuanLyPhuThuoc.nhap_tuy_chon("goi_khong_ton_tai_xyz", nhom_extra="rag") is None

    goi_da_yeu_cau: list[tuple[str, str, bool]] = []

    def _gia_lap_yeu_cau(
        ten_module: str,
        nhom_extra: str = "ml",
        tinh_nang: str = "",
        tu_dong_cai: Any = None,
    ) -> Any:
        goi_da_yeu_cau.append((ten_module, nhom_extra, bool(tu_dong_cai)))
        return {"da_nap": ten_module}

    monkeypatch.setenv("VAI_AUTO_INSTALL", "1")
    monkeypatch.setattr(QuanLyPhuThuoc, "yeu_cau", classmethod(lambda cls, *a, **kw: _gia_lap_yeu_cau(*a, **kw)))
    kq = QuanLyPhuThuoc.nhap_tuy_chon("goi_khong_ton_tai_xyz", nhom_extra="legal-rag", tinh_nang="DOCX")
    assert kq == {"da_nap": "goi_khong_ton_tai_xyz"}
    assert goi_da_yeu_cau == [("goi_khong_ton_tai_xyz", "legal-rag", True)]


def test_moi_noi_6_dong_bo_packaging_cargo_va_9_bieu_tuong_top_level() -> None:
    """Mối nối 6: Đồng bộ phiên bản 31.0.4 cả Rust Cargo.toml, sub-package legal và 9 biểu tượng top-level."""
    bieu_tuong_moi = [
        "BoGiaiMaTransformerEdge",
        "BoTachTuBPEGGUF",
        "KiemToanMatMa",
        "BoDocTaiLieuPhapLy",
        "BoThuThapTelemetryAPI",
        "tao_ssl_context_server",
        "tao_ssl_context_client",
        "PhienBanSemVer",
        "QuanLyPhatHanhGoi",
    ]
    for sym in bieu_tuong_moi:
        assert sym in vietnamese_ai.__all__
        assert getattr(vietnamese_ai, sym) is not None

    bc_semver = vietnamese_ai.QuanLyPhatHanhGoi.kiem_tra_tuan_thu_semver()
    assert bc_semver["hop_le_semver"] is True
    assert bc_semver["dong_bo_phien_ban"] is True
    assert bc_semver["phien_ban_chuan"] == "31.1.0"
    assert "rust_core/Cargo.toml" in bc_semver["cac_nguon"]
    assert bc_semver["cac_nguon"]["rust_core/Cargo.toml"] == "31.1.0"

    goi_legal = vietnamese_ai.QuanLyPhatHanhGoi.danh_sach_goi()["vietnamese-ai-legal"]
    deps_legal = goi_legal["phu_thuoc_tuy_chon"]["ocr-pdf"]
    assert "python-docx>=1.0.0" in deps_legal
    assert "pdfplumber>=0.10.0" in deps_legal

    legal_toml = Path("packages/vietnamese-ai-legal/pyproject.toml").read_text(encoding="utf-8")
    assert "python-docx>=1.0.0" in legal_toml
    assert "pdfplumber>=0.10.0" in legal_toml


def test_moi_noi_7_he_thong_end_to_end_va_studio_dashboard_http(tmp_path: Path) -> None:
    """Mối nối 7: QuyTrinhDauCuoi kết nối Legal DOCX + Edge GGUF + FastAPI, và StudioDashboard phục vụ chẩn đoán/kiểm toán/metrics."""
    docx_file = _tao_tep_docx_phap_ly(tmp_path / "e2e_nd13.docx")
    gguf_file = _tao_tep_gguf_bpe_thuc_te(tmp_path / "e2e_edge.gguf", d_model=32)

    he_thong = vietnamese_ai.QuyTrinhDauCuoi(
        ten_du_an="TestPhase6E2E",
        thu_muc_lam_viec=str(tmp_path / "ws"),
    )
    kq_nap = he_thong.nap_tep_phap_ly(docx_file, che_do_hanh_chinh=True, toi_uu_ivf=True, so_cum_ivf=2)
    assert kq_nap["so_chunks"] >= 1
    assert kq_nap["ivf"]["da_xay_dung"] is True

    edge = he_thong.khoi_tao_edge(duong_dan_mo_hinh=gguf_file, do_dai_toi_da=5, nhiet_do=0.0)
    assert edge._che_do_suy_luan == "gguf_bpe_end_to_end"

    kq_hoi_edge = he_thong.hoi("Việt Nam", dung_rag=False, dung_edge=True)
    assert kq_hoi_edge["an_toan"] is True
    assert kq_hoi_edge["tra_loi"]

    stream_tokens = list(he_thong.sinh_stream("Việt Nam", do_dai_toi_da=4, nhiet_do=0.0))
    assert len(stream_tokens) >= 1

    try:
        api_srv = he_thong.tao_api_server(enable_watermark=False)
        metrics_ep = next(r.endpoint for r in api_srv.app.routes if getattr(r, "path", "") == "/metrics")
        metrics_resp = asyncio.run(metrics_ep())
        body_text = metrics_resp.body.decode("utf-8") if hasattr(metrics_resp, "body") else str(metrics_resp)
        assert "vai_api_requests_total" in body_text
    except ImportError:
        pass

    dashboard = vietnamese_ai.StudioDashboard(host="127.0.0.1", port=0)
    httpd = dashboard.tao_server(host="127.0.0.1", port=0)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/system/diagnostics", timeout=5) as r1:
            diag = json.loads(r1.read().decode("utf-8"))
        assert diag["semver"]["dong_bo_phien_ban"] is True
        assert "phan_cung" in diag
        assert "kien_truc_cpu" in diag

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/security/crypto-audit", timeout=5) as r2:
            audit = json.loads(r2.read().decode("utf-8"))
        assert audit["tong_quan_dat"] is True

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/metrics", timeout=5) as r3:
            prom_txt = r3.read().decode("utf-8")
        assert "vai_api_requests_total" in prom_txt
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_moi_noi_8_tai_lieu_api_dong_bo_cac_api_moi() -> None:
    """Mối nối 8: Tài liệu API Reference phản ánh đầy đủ các API Giai đoạn 4-6."""
    edge_doc = Path("docs/api/edge.md").read_text(encoding="utf-8")
    legal_doc = Path("docs/api/legal-rag.md").read_text(encoding="utf-8")
    utils_doc = Path("docs/api/utils.md").read_text(encoding="utf-8")
    core_doc = Path("docs/api/core.md").read_text(encoding="utf-8")

    assert "BoTachTuBPEGGUF" in edge_doc
    assert "tao_ssl_context_server" in edge_doc
    assert "them_tep" in legal_doc
    assert "toi_uu_chi_muc_vector" in legal_doc
    assert "nhap_tuy_chon" in utils_doc
    assert "thiet_lap_tu_dong" in utils_doc
    assert "tao_api_server" in core_doc
    assert "BoThuThapTelemetryAPI" in core_doc
