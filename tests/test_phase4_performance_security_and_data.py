"""Kiểm thử toàn diện Giai đoạn 4 (Phần 2): Hợp nhất Crypto, TLS/mTLS Swarm, GQA/Quantized GGUF, Rust Core, Legal .docx/Corpus, Audio/Vision & CI."""

from __future__ import annotations

import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest


def test_hop_nhat_crypto_rfc5114_va_rust_montgomery():
    """Kiểm tra vietnamese_ai.crypto.schnorr thống nhất cho XacThucSuyLuanMaoHiem, SecureEdgeNode, WebBrowserNode và Rust modpow."""
    from vietnamese_ai.crypto import (
        RFC5114_G,
        RFC5114_P,
        RFC5114_Q,
        KiemToanMatMa,
        kiem_tra_thanh_vien_nhom_con,
        luy_thua_modulo_rfc5114,
    )
    from vietnamese_ai.edge import SecureEdgeNode, WebBrowserNode

    for exp in (1, 7, 65537, RFC5114_Q - 1, RFC5114_Q):
        assert luy_thua_modulo_rfc5114(RFC5114_G, exp) == pow(RFC5114_G, exp, RFC5114_P)

    assert kiem_tra_thanh_vien_nhom_con(RFC5114_G) is True
    assert kiem_tra_thanh_vien_nhom_con(1) is False

    node = SecureEdgeNode("n_unified", private_key="secret_vn_4", auto_start=False)
    proof = node.generate_proof("cau hoi", "tra loi")
    assert SecureEdgeNode.verify_proof("n_unified", node.public_key, "cau hoi", "tra loi", proof, node.model_hash)

    wb = WebBrowserNode("s_unified", secret_key="wasm_secret_4")
    kq_wb = wb.chay_suy_luan_an_toan("Xin chào")
    assert SecureEdgeNode.verify_proof(
        kq_wb["node_id"], wb.public_key, "Xin chào", kq_wb["cau_tra_loi"], kq_wb["proof"], wb.model_hash
    )

    audit = KiemToanMatMa.chay_kiem_toan_tu_dong()
    assert audit["tong_quan_dat"] is True


def test_swarm_cpu_lan_ma_hoa_tls_va_mtls_kiem_tra_ca(tmp_path: Path):
    """Kiểm tra mã hóa TLS 1.2+ thực sự trên TCP Socket Swarm CPU LAN và từ chối mTLS thiếu CA."""
    from vietnamese_ai.edge import (
        DieuPhoiSwarmCPU,
        NutSwarmCPU,
        tao_ssl_context_client,
        tao_ssl_context_server,
    )

    with pytest.raises(ValueError, match="mTLS"):
        tao_ssl_context_server("cert.pem", "key.pem", cafile=None, yeu_cau_mtls=True)

    if not shutil.which("openssl"):
        pytest.skip("Máy không cài openssl CLI để sinh chứng chỉ tự ký kiểm thử TLS")

    cert_file = tmp_path / "server.crt"
    key_file = tmp_path / "server.key"
    proc = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1,DNS:localhost",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("Không thể sinh chứng chỉ X509 tạm bằng openssl")

    srv_ctx = tao_ssl_context_server(str(cert_file), str(key_file))
    cli_ctx = tao_ssl_context_client(cafile=str(cert_file), kiem_tra_hostname=True)

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_tmp:
        s_tmp.bind(("127.0.0.1", 0))
        port = s_tmp.getsockname()[1]

    worker = NutSwarmCPU("tls_worker", dai_layer=(0, 2), d_model=16)
    worker.khoi_dong_server_lan(host="127.0.0.1", port=port, ssl_context=srv_ctx)
    try:
        time.sleep(0.1)
        orch = DieuPhoiSwarmCPU(d_model=16, tong_so_layer=2, ssl_context=cli_ctx)
        nut_client = orch.them_node_mang_vat_ly("tls_client", endpoint_url=f"tls://127.0.0.1:{port}", dai_layer=(0, 2))
        assert nut_client.lay_thong_tin_cau_hinh()["ma_hoa_tls"] is True
        assert orch.lay_trang_thai_cum()["ma_hoa_tls"] is True

        x_in = np.ones((1, 16), dtype=np.float32) * 0.25
        y_out, proof = nut_client._truyen_tai_qua_socket_lan(x_in, tao_proof=True)
        assert y_out.shape == (1, 16)
        assert proof is not None and proof["proof_zk"].startswith("zkml1:")
    finally:
        worker.dung_server_lan()


def test_transformer_edge_gqa_da_tang_va_quantized_matmul_gguf(tmp_path: Path):
    """Kiểm tra BoGiaiMaTransformerEdge GQA đa tầng và nhân trực tiếp trên bộ đệm lượng tử GGUF Q4_0."""
    import struct

    from vietnamese_ai.edge import (
        BoGiaiLuongTuQ4,
        BoGiaiMaTransformerEdge,
        DongCoSuyLuanEdge,
        GGUFMemoryMappedStreamer,
    )

    tr_gqa = BoGiaiMaTransformerEdge(d_model=64, so_dau=4, so_dau_kv=2, d_ff=128, so_lop=2)
    assert tr_gqa.so_dau_kv == 2
    assert tr_gqa.d_kv == 32
    assert tr_gqa.w_k.shape == (64, 32)

    out0 = tr_gqa.buoc_suy_luan(np.ones(64, dtype=np.float32), vi_tri=0)
    out1 = tr_gqa.buoc_suy_luan(np.ones(64, dtype=np.float32) * 0.5, vi_tri=1)
    assert out0.shape == (1, 64)
    assert out1.shape == (1, 64)
    assert np.all(np.isfinite(out1))

    def _tao_q4_0_bytes(so_hang: int, so_cot: int, seed_val: int) -> bytes:
        rng_local = np.random.RandomState(seed_val)
        buf = bytearray()
        for _ in range(so_hang * (so_cot // 32)):
            buf.extend(np.float16(0.05).tobytes())
            buf.extend(rng_local.randint(0, 256, size=16, dtype=np.uint8).tobytes())
        return bytes(buf)

    specs = [
        ("blk.0.attn_q.weight", (64, 64), _tao_q4_0_bytes(64, 64, 1)),
        ("blk.0.attn_k.weight", (32, 64), _tao_q4_0_bytes(32, 64, 2)),
        ("blk.0.attn_v.weight", (32, 64), _tao_q4_0_bytes(32, 64, 3)),
        ("blk.0.attn_output.weight", (64, 64), _tao_q4_0_bytes(64, 64, 4)),
    ]

    def _pack_str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    gguf_path = tmp_path / "gqa_q4.gguf"
    header = bytearray(b"GGUF")
    header.extend(struct.pack("<IQQ", 3, len(specs), 2))
    header.extend(_pack_str("general.name") + struct.pack("<I", 8) + _pack_str("VietnameseGQA-Q4"))
    header.extend(_pack_str("general.architecture") + struct.pack("<I", 8) + _pack_str("qwen2"))

    offset = 0
    payload = bytearray()
    for name, dims, raw_q in specs:
        header.extend(_pack_str(name))
        header.extend(struct.pack("<I", len(dims)))
        for d in dims:
            header.extend(struct.pack("<Q", d))
        header.extend(struct.pack("<IQ", 2, offset))
        payload.extend(raw_q)
        offset += len(raw_q)

    rem = len(header) % 32
    if rem:
        header.extend(b"\x00" * (32 - rem))
    gguf_path.write_bytes(bytes(header) + bytes(payload))

    streamer = GGUFMemoryMappedStreamer(gguf_path)
    tho = streamer.doc_tensor_tho("blk.0.attn_q.weight")
    assert tho is not None and tho["loai_quant"] == "q4_0"

    da_nap = tr_gqa.nap_trong_so_tu_gguf(streamer, giu_luong_tu=True)
    assert da_nap == 4
    assert len(tr_gqa._trong_so_luong_tu) == 4
    tr_gqa.xoa_kv_cache()
    out_q = tr_gqa.buoc_suy_luan(np.ones(64, dtype=np.float32), vi_tri=0)
    assert out_q.shape == (1, 64) and np.all(np.isfinite(out_q))
    streamer.close()

    rng = np.random.RandomState(7)
    x_mat = rng.randn(2, 64).astype(np.float32)
    y_q4 = BoGiaiLuongTuQ4.nhan_ma_tran_luong_tu(x_mat, _tao_q4_0_bytes(32, 64, 9), (32, 64), "q4_0")
    assert y_q4.shape == (2, 32) and np.all(np.isfinite(y_q4))

    engine = DongCoSuyLuanEdge(duong_dan_mo_hinh=gguf_path)
    stats = engine.sinh_kem_thong_ke("Xin chào bạn")
    assert stats["so_dau_kv"] == 2
    assert "che_do_suy_luan" in stats


def test_tokenizer_prefix_trie_va_rust_khop_tuyet_doi(monkeypatch: pytest.MonkeyPatch):
    """Kiểm tra BoTachTuTiengVietSieuToc cho kết quả tách từ đồng nhất giữa RustViTokenizer và Python Prefix Trie."""
    import vietnamese_ai.nlp.native_tokenizer as nt_mod
    from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc

    van_ban = "Ủy ban nhân dân thành phố Hà Nội ban hành nghị định bảo vệ dữ liệu cá nhân và chữ ký số."
    tok_rust = BoTachTuTiengVietSieuToc()
    kq_rust = tok_rust.tach_tu(van_ban, noi_tu_ghep=True)

    monkeypatch.setattr(nt_mod, "_rust_core", None)
    tok_py = BoTachTuTiengVietSieuToc()
    kq_py = tok_py.tach_tu(van_ban, noi_tu_ghep=True)

    assert kq_rust == kq_py
    assert "Ủy_ban_nhân_dân" in kq_py
    assert "chữ_ký_số" in kq_py

    tok_py.them_tu_dien(["kinh tế số"])
    assert "kinh_tế_số" in tok_py.tach_tu("Phát triển kinh tế số quốc gia", noi_tu_ghep=True)


def test_legal_rag_docx_bang_bieu_bo_du_lieu_mo_audio_vision_plugin_va_ci(tmp_path: Path):
    """Kiểm tra đọc .docx có bảng biểu, kho pháp lý mở 50 điều khoản JSONL, adapter Audio/Vision, hợp nhất plugin và CI."""
    from vietnamese_ai.audio.speech import NhanDienGiongNoi, TongHopGiongNoi
    from vietnamese_ai.benchmark.datasets_vn import tai_bo_hoi_dap_phap_ly_mo, tai_bo_phap_ly_mo
    from vietnamese_ai.plugin import (
        DANH_MUC_PLUGIN_CHUYEN_NGANH,
        PluginChungKhoanCrypto,
        PluginTaiChinh,
        PluginYTe,
    )
    from vietnamese_ai.plugins import PluginTaiChinh as PluginTaiChinhLegacy
    from vietnamese_ai.rag.legal import BoDocTaiLieuPhapLy
    from vietnamese_ai.vision.ocr import TrichXuatVanBanAnh

    docx_path = tmp_path / "nghi_dinh_mau.docx"
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>Điều 1. Quy định về biểu mức thu phí dịch vụ công.</w:t></w:r></w:p>"
        "<w:tbl>"
        "<w:tr><w:tc><w:p><w:r><w:t>Hạng mục</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Mức thu</w:t></w:r></w:p></w:tc></w:tr>"
        "<w:tr><w:tc><w:p><w:r><w:t>Cấp giấy phép</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>500.000 đồng</w:t></w:r></w:p></w:tc></w:tr>"
        "</w:tbl>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(str(docx_path), "w") as zf:
        zf.writestr("word/document.xml", doc_xml)

    parser = BoDocTaiLieuPhapLy()
    kq_docx = parser.doc_tep(docx_path)
    assert kq_docx["dinh_dang"] == "docx"
    assert kq_docx["metadata"]["so_bang"] == 1
    assert "Điều 1." in kq_docx["van_ban"]
    assert "| Hạng mục | Mức thu |" in kq_docx["van_ban"]
    assert "500.000 đồng" in kq_docx["van_ban"]

    corpus = tai_bo_phap_ly_mo()
    qa = tai_bo_hoi_dap_phap_ly_mo()
    assert len(corpus) >= 50
    assert len(qa) >= 20
    ids_corpus = {item["id"] for item in corpus}
    assert all(q["id_ky_vong"] in ids_corpus for q in qa)

    assert "faster_whisper" in NhanDienGiongNoi.danh_sach_backend_kha_dung()
    assert "edge_tts" in TongHopGiongNoi.danh_sach_backend_kha_dung()
    assert "piper_tts" in TongHopGiongNoi.danh_sach_backend_kha_dung()
    assert "vietocr" in TrichXuatVanBanAnh.danh_sach_engine_kha_dung()

    assert PluginTaiChinh is PluginTaiChinhLegacy
    assert set(DANH_MUC_PLUGIN_CHUYEN_NGANH) == {"PluginTaiChinh", "PluginYTe", "PluginChungKhoanCrypto"}
    assert PluginYTe is not None and PluginChungKhoanCrypto is not None

    ci_text = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "KiemToanMatMa" in ci_text
    assert "kiem_tra_tuan_thu_semver" in ci_text
    assert "|| true" not in ci_text
