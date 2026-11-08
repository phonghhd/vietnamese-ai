"""Kiểm thử chuyên sâu cho Phân hệ v30.2: Vietnamese-Centric Optimization & Comparative Benchmark.

Kiểm tra:
1. Native Vietnamese Fast Tokenizer & Normalizer (Maximum Matching, Unicode NFC, dấu mới/cũ).
2. Fast Vietnamese Embedder (Feature Hashing, L2 normalization, semantic cosine similarity).
3. Vietnamese Administrative & Legal RAG Optimizer (Hierarchy-preserving chunking, legal citations).
4. Quantitative Comparative Benchmark Suite vs LangChain/LlamaIndex (Markdown, LaTeX, JSON).
5. CLI commands `vai benchmark --compare` và `vai tokenize`.
6. Tích hợp Lazy Loading PEP 562 và Version 31.0.0.
"""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

import vietnamese_ai
from vietnamese_ai.benchmark.comparative_benchmark import (
    BoDoLuongDoiDau,
    ComparativeBenchmarkSuite,
)
from vietnamese_ai.cli.main import cmd_benchmark, cmd_tokenize
from vietnamese_ai.nlp.native_tokenizer import (
    BoNhungVanBanSieuToc,
    BoTachTuTiengVietSieuToc,
    FastVietnameseEmbedder,
    FastVietnameseTokenizer,
    NativeVietnameseTokenizer,
)
from vietnamese_ai.rag.vietnamese_context import (
    BoToiUuNguCanhHanhChinh,
    VietnameseAdministrativeRAG,
)


class TestNativeVietnameseTokenizer:
    """Kiểm tra Bộ Tách Từ & Chuẩn Hóa Tiếng Việt Siêu Tốc."""

    def test_tach_tu_ghep_chinh_xac(self):
        tok = BoTachTuTiengVietSieuToc()
        van_ban = "Trí tuệ nhân tạo và học máy đang hỗ trợ ủy ban nhân dân thành phố hà nội chuyển đổi số."

        # 1. Tách từ có nối gạch dưới
        tokens = tok.tach_tu(van_ban, noi_tu_ghep=True)
        assert "trí_tuệ_nhân_tạo" in [t.lower() for t in tokens]
        assert "học_máy" in [t.lower() for t in tokens]
        assert "ủy_ban_nhân_dân" in [t.lower() for t in tokens]
        assert "chuyển_đổi_số" in [t.lower() for t in tokens]

        # 2. Tách từ không nối gạch dưới
        tokens_roi = tok.tach_tu(van_ban, noi_tu_ghep=False)
        assert "trí tuệ nhân tạo" in [t.lower() for t in tokens_roi]

    def test_chuan_hoa_dau_thanh_va_unicode(self):
        tok = BoTachTuTiengVietSieuToc()
        # Chuẩn hóa kiểu cũ sang kiểu mới
        vb_cu = "Toà án nhân dân và uỷ ban nhân dân thông qua kế hoạch hoà giải."
        vb_chuan = tok.chuan_hoa_dau_thanh(vb_cu)
        assert "tòa" in vb_chuan.lower()
        assert "ủy" in vb_chuan.lower()
        assert "hòa" in vb_chuan.lower()

    def test_xoa_dau_tieng_viet(self):
        tok = BoTachTuTiengVietSieuToc()
        vb = "Việt Nam Đất Nước Rồng Tiên"
        kd = tok.xoa_dau_tieng_viet(vb)
        assert kd == "Viet Nam Dat Nuoc Rong Tien"

    def test_them_tu_dien_tuy_bien(self):
        tok = BoTachTuTiengVietSieuToc()
        tok.them_tu_dien(["antigravity agent", "mô hình ngôn ngữ lớn"])
        tokens = tok.tach_tu("Hệ thống tích hợp antigravity agent và mô hình ngôn ngữ lớn.", noi_tu_ghep=True)
        assert "antigravity_agent" in tokens
        assert "mô_hình_ngôn_ngữ_lớn" in tokens

    def test_dem_so_tu_sieu_toc(self):
        tok = BoTachTuTiengVietSieuToc()
        count = tok.dem_so_tu("Bộ tư pháp ban hành nghị định mới.")
        assert count > 0


class TestFastVietnameseEmbedder:
    """Kiểm tra Bộ Nhúng Văn Bản Tiếng Việt Siêu Tốc."""

    def test_nhung_vector_chuan_hoa_l2(self):
        embedder = BoNhungVanBanSieuToc(so_chieu=128)
        vec = embedder.nhung("Quy định về bảo vệ quyền lợi người tiêu dùng.")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (128,)
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-3)

    def test_do_tuong_dong_ngu_nghia(self):
        embedder = BoNhungVanBanSieuToc(so_chieu=256)
        s1 = "Học máy và trí tuệ nhân tạo tại Việt Nam."
        s2 = "Nghiên cứu trí tuệ nhân tạo và học máy ở Việt Nam."
        s3 = "Hôm nay thời tiết đẹp trời, hoa nở khắp nơi."

        sim_dong_nghia = embedder.do_tuong_dong(s1, s2)
        sim_khac_nghia = embedder.do_tuong_dong(s1, s3)

        assert sim_dong_nghia > 0.5
        assert sim_dong_nghia > sim_khac_nghia

    def test_nhung_lo(self):
        embedder = BoNhungVanBanSieuToc(so_chieu=64)
        matrix = embedder.nhung_lo(["Câu một", "Câu hai", "Câu ba"])
        assert matrix.shape == (3, 64)


class TestVietnameseAdministrativeRAG:
    """Kiểm tra Bộ Tối Ưu Ngữ Cảnh Hành Chính & Pháp Lý Việt Nam."""

    VAN_BAN_TEST = """
    Căn cứ Luật Tiếp cận thông tin số 104/2016/QH13.
    Chương I: NHỮNG QUY ĐỊNH CHUNG
    Điều 1. Phạm vi điều chỉnh
    Luật này quy định về việc thực hiện quyền tiếp cận thông tin của công dân.
    Điều 2. Giải thích từ ngữ
    1. Thông tin là tin, dữ liệu được chứa đựng trong văn bản, hồ sơ, tài liệu.
    2. Thông tin do cơ quan nhà nước tạo ra là tin, dữ liệu do cơ quan nhà nước tạo ra trong quá trình thực hiện chức năng, nhiệm vụ.
    """

    def test_trich_xuat_can_cu_va_so_hieu(self):
        admin = BoToiUuNguCanhHanhChinh()
        res = admin.trich_xuat_can_cu_va_so_hieu(self.VAN_BAN_TEST)
        assert len(res["danh_sach_can_cu"]) >= 1
        assert any("104/2016/QH13" in sh for sh in res["so_hieu_van_ban"])

    def test_phan_doan_giu_ngu_canh(self):
        admin = BoToiUuNguCanhHanhChinh()
        chunks = admin.phan_doan_giu_ngu_canh(self.VAN_BAN_TEST, ten_tai_lieu="Luật 104")
        assert len(chunks) >= 2
        for c in chunks:
            assert "Luật 104" in c["ngu_canh_day_du"]
            assert "Điều" in c["metadata"]["dieu"]
            assert c["vector"] is not None

    def test_truy_hoi_phap_ly_co_can_cu(self):
        admin = BoToiUuNguCanhHanhChinh()
        chunks = admin.phan_doan_giu_ngu_canh(self.VAN_BAN_TEST, ten_tai_lieu="Luật 104")
        ket_qua = admin.truy_hoi_phap_ly("Thông tin do cơ quan nhà nước tạo ra là gì?", chunks, top_k=2)
        assert len(ket_qua) > 0
        top1 = ket_qua[0]
        assert "Điều 2" in top1["trich_dan_phap_ly"]
        assert top1["diem_tuong_dong"] > 0


class TestComparativeBenchmarkSuite:
    """Kiểm tra Bộ Đo Lường Đối Đầu Định Lượng vs LangChain / LlamaIndex."""

    def test_chay_benchmark_toan_dien(self):
        suite = BoDoLuongDoiDau()
        kq = suite.chay_benchmark_toan_dien()

        assert "vietnamese_ai" in kq
        assert "langchain" in kq
        assert "llamaindex" in kq
        assert "uu_the_so_voi_langchain" in kq

        vai = kq["vietnamese_ai"]
        assert vai["thoi_gian_khoi_dong_ms"] > 0
        assert vai["ram_tieu_thu_mb"] > 0
        assert vai["toc_do_tach_tu_wps"] > 1000
        assert vai["hit_rate_top1_pct"] >= 50.0

        adv = kq["uu_the_so_voi_langchain"]
        assert adv["nhanh_hon_khoi_dong_lan"] > 10.0
        assert adv["tiet_kiem_ram_pct"] > 50.0

    def test_xuat_bao_cao_markdown_va_latex(self, tmp_path: Path):
        suite = BoDoLuongDoiDau()
        kq = suite.chay_benchmark_toan_dien()

        # 1. Báo cáo Markdown
        md = suite.xuat_bang_so_sanh_markdown(kq)
        assert "QUANTITATIVE BENCHMARK" in md
        assert "Vietnamese AI (v30.2)" in md
        assert "LangChain" in md

        # 2. Báo cáo LaTeX
        latex = suite.xuat_bang_so_sanh_latex(kq)
        assert "\\begin{table}" in latex
        assert "\\caption{" in latex
        assert "\\end{table}" in latex

        # 3. Xuất JSON
        json_path = tmp_path / "bench_out.json"
        data_json = suite.xuat_du_lieu_json(json_path)
        assert data_json is not None
        assert json_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["vietnamese_ai"]["ten"] == "Vietnamese AI (v30.2)"


def test_cli_benchmark_va_tokenize(tmp_path: Path):
    """Kiểm tra các lệnh CLI mới trong v30.2: `vai benchmark --compare` và `vai tokenize`."""
    # 1. `vai benchmark --compare`
    args_bench = Namespace(command="benchmark", compare=True, latex=False, output=str(tmp_path / "report.md"))
    kq_b = cmd_benchmark(args_bench)
    assert "vietnamese_ai" in kq_b
    assert (tmp_path / "report.md").exists()

    # 2. `vai tokenize`
    args_tok = Namespace(
        command="tokenize",
        text="Ủy ban nhân dân thành phố Đà Nẵng triển khai chuyển đổi số",
        no_compound=False,
        unaccent=True,
    )
    tokens = cmd_tokenize(args_tok)
    assert len(tokens) > 0
    assert any("ủy_ban_nhân_dân" in t.lower() or "chuyển_đổi_số" in t.lower() for t in tokens)


def test_public_symbols_and_version_v30_2():
    """Kiểm tra các symbols công khai và phiên bản 31.0.0 qua lazy loader."""
    assert vietnamese_ai.__version__ == "31.0.0"

    # Kiểm tra lazy imports các symbols mới
    assert getattr(vietnamese_ai, "BoTachTuTiengVietSieuToc") is BoTachTuTiengVietSieuToc
    assert getattr(vietnamese_ai, "FastVietnameseTokenizer") is FastVietnameseTokenizer
    assert getattr(vietnamese_ai, "NativeVietnameseTokenizer") is NativeVietnameseTokenizer
    assert getattr(vietnamese_ai, "BoNhungVanBanSieuToc") is BoNhungVanBanSieuToc
    assert getattr(vietnamese_ai, "FastVietnameseEmbedder") is FastVietnameseEmbedder
    assert getattr(vietnamese_ai, "BoToiUuNguCanhHanhChinh") is BoToiUuNguCanhHanhChinh
    assert getattr(vietnamese_ai, "VietnameseAdministrativeRAG") is VietnameseAdministrativeRAG
    assert getattr(vietnamese_ai, "BoDoLuongDoiDau") is BoDoLuongDoiDau
    assert getattr(vietnamese_ai, "ComparativeBenchmarkSuite") is ComparativeBenchmarkSuite
