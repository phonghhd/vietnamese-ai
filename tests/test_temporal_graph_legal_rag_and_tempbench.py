"""Kiểm thử toàn diện Chặng 1 & Chặng 2 (v31.1.0): Temporal & Graph-Grounded Legal RAG và ViLegal-TempBench."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

import vietnamese_ai
from vietnamese_ai.benchmark.legal_temporal_benchmark import (
    BoDanhGiaLegalTemporalRAG,
    kiem_dinh_bootstrap_cap,
    tinh_ndcg_at_k,
)
from vietnamese_ai.rag.legal.citation_graph import (
    DoThiThamChieuPhapLy,
    NutDieuKhoanPhapLy,
    QuanHeDieuKhoan,
)
from vietnamese_ai.rag.legal.citation_verifier import BoKiemChungTrichDanPhapLy
from vietnamese_ai.rag.legal.open_corpus import (
    tai_bo_hoi_dap_phap_ly_thoi_gian,
    tai_bo_phap_ly_thoi_gian,
    tai_canh_do_thi_phap_ly_thoi_gian,
)
from vietnamese_ai.rag.legal.temporal_query import BoPhanTichThoiGianPhapLy
from vietnamese_ai.rag.legal.temporal_rag import BoTruyHoiPhapLyThoiGian
from vietnamese_ai.solutions.phap_ly import TroLyPhapLy


def test_temporal_query_parser_vietnamese_expressions_and_disambiguation() -> None:
    """Kiểm tra bóc tách mốc thời gian tiếng Việt và loại trừ nhiễu năm trong tên đạo luật."""
    parser = BoPhanTichThoiGianPhapLy()

    r1 = parser.phan_tich("Trước ngày 01/06/2025, Điều 19 Nghị định 123/2020/NĐ-CP quy định ra sao?")
    assert r1.co_moc_tuong_minh is True
    assert r1.ngay_tra_cuu == "2025-05-31"
    assert r1.kieu_quan_he_thoi_gian == "truoc_ngay"

    r2 = parser.phan_tich("Từ ngày 01/06/2025, hóa đơn sai sót xử lý thế nào?")
    assert r2.ngay_tra_cuu == "2025-06-01"
    assert r2.kieu_quan_he_thoi_gian == "tu_ngay"

    r3 = parser.phan_tich("Vào tháng 8/2025, thuế suất phân bón là bao nhiêu?")
    assert r3.ngay_tra_cuu == "2025-08-15"
    assert r3.kieu_quan_he_thoi_gian == "trong_thang"

    r4 = parser.phan_tich("Quý II/2024 áp dụng quy định bảng giá đất nào?")
    assert r4.ngay_tra_cuu == "2024-05-15"

    r5 = parser.phan_tich("Luật Đất đai 2024 quy định bảng giá đất thế nào?", ngay_mac_dinh="2026-01-15")
    assert r5.co_moc_tuong_minh is False
    assert r5.ngay_tra_cuu == "2026-01-15"

    r6 = parser.phan_tich(
        "Hộ gia đình sử dụng đất không giấy tờ ổn định trước 01/07/2014 theo Điều 138 Luật Đất đai 31/2024/QH15 có được cấp Sổ đỏ không?",
        ngay_mac_dinh="2026-01-15",
    )
    assert r6.ngay_tra_cuu == "2026-01-15"


def test_clause_level_temporal_graph_multi_hop_traversal() -> None:
    """Kiểm tra đồ thị cấp Điều/Khoản có điều kiện thời gian t và lan truyền đa bước 2-hop."""
    dt = DoThiThamChieuPhapLy()
    corpus = tai_bo_phap_ly_thoi_gian()
    edges = tai_canh_do_thi_phap_ly_thoi_gian()
    for c in corpus:
        dt.them_nut_dieu_khoan(
            NutDieuKhoanPhapLy(
                ma_nut=c["id"],
                so_hieu=c["so_hieu"],
                dieu=c["dieu"],
                noi_dung=c["noi_dung"],
                ten_van_ban=c["ten_van_ban"],
                hieu_luc_tu=c.get("hieu_luc_tu"),
                het_hieu_luc_tu=c.get("het_hieu_luc_tu"),
                sua_doi_boi=c.get("sua_doi_boi"),
            )
        )
    for e in edges:
        dt.them_quan_he_dieu_khoan(
            QuanHeDieuKhoan(
                nguon_nut=e["nguon_nut"],
                dich_nut=e["dich_nut"],
                loai_quan_he=e["loai_quan_he"],
                hieu_luc_tu=e.get("hieu_luc_tu"),
                het_hieu_luc_tu=e.get("het_hieu_luc_tu"),
                trong_so=float(e.get("trong_so", 1.0)),
            )
        )

    assert dt.thong_ke_dieu_khoan()["so_nut_dieu_khoan"] == 60
    assert dt.thong_ke_dieu_khoan()["tong_canh_dieu_khoan"] == 38

    truoc = dt.mo_rong_da_buoc_theo_thoi_gian(["tb_hd_d19_v1"], ngay_tra_cuu="2025-03-15", so_buoc=2)
    ids_truoc = {x["ma_nut"] for x in truoc}
    assert "tb_hd_tt78_d7" in ids_truoc
    assert "tb_hd_tt32_d6" not in ids_truoc

    sau = dt.mo_rong_da_buoc_theo_thoi_gian(["tb_hd_d19_v2"], ngay_tra_cuu="2025-09-01", so_buoc=2)
    ids_sau = {x["ma_nut"] for x in sau}
    assert "tb_hd_tt32_d6" in ids_sau
    assert "tb_hd_tt78_d7" not in ids_sau

    hai_buoc = dt.mo_rong_da_buoc_theo_thoi_gian(["tb_ds_2015_d119"], ngay_tra_cuu="2025-09-01", so_buoc=2)
    ids_2hop = {x["ma_nut"] for x in hai_buoc}
    assert "tb_gddt_2023_d22" in ids_2hop
    assert "tb_gddt_nd137_d10" in ids_2hop


def test_three_tier_citation_and_temporal_verifier() -> None:
    """Kiểm tra bộ kiểm chứng 3 tầng (Grounding, Temporal Consistency, Numeric Consistency)."""
    verifier = BoKiemChungTrichDanPhapLy()
    nguon = [
        {
            "so_hieu": "59/2020/QH14",
            "dieu": "Điều 26",
            "noi_dung": "Cơ quan đăng ký kinh doanh cấp Giấy chứng nhận trong thời hạn 03 ngày làm việc.",
        },
        {
            "so_hieu": "32/2025/TT-BTC",
            "dieu": "Điều 6",
            "noi_dung": "Thông tư 32/2025/TT-BTC thay thế Thông tư 78/2021/TT-BTC từ 01/06/2025.",
        },
    ]

    ok = verifier.kiem_chung(
        "Căn cứ Điều 26 Luật 59/2020/QH14, thời hạn cấp Giấy chứng nhận là 3 ngày làm việc.",
        nguon,
        ngay_tra_cuu="2025-10-01",
    )
    assert ok.hop_le is True
    assert ok.diem_trung_thuc == 1.0
    assert ok.co_xung_dot_thoi_gian is False
    assert ok.diem_nhat_quan_so_lieu == 1.0

    ao = verifier.kiem_chung(
        "Căn cứ Điều 99 Nghị định 777/2025/NĐ-CP, thời hạn cấp giấy là 45 ngày làm việc.",
        nguon,
        ngay_tra_cuu="2025-10-01",
    )
    assert ao.hop_le is False
    assert ao.co_trich_dan_ao is True
    assert "777/2025/NĐ-CP" in ao.trich_dan_khong_co_trong_nguon
    assert "Điều 99" in ao.trich_dan_khong_co_trong_nguon
    assert len(ao.so_lieu_lech_nguon) >= 1

    xung_dot = verifier.kiem_chung(
        "Áp dụng Luật Đất đai 45/2013/QH13 để tính bảng giá đất 5 năm.",
        [{"so_hieu": "45/2013/QH13", "dieu": "Điều 114", "noi_dung": "Bảng giá đất định kỳ 05 năm."}],
        ngay_tra_cuu="2025-10-01",
    )
    assert xung_dot.co_xung_dot_thoi_gian is True
    assert xung_dot.hop_le is False

    thay_the_hop_le = verifier.kiem_chung(
        "Thông tư 32/2025/TT-BTC thay thế Thông tư 78/2021/TT-BTC từ 01/06/2025.",
        nguon,
        ngay_tra_cuu="2025-10-01",
    )
    assert thay_the_hop_le.co_xung_dot_thoi_gian is False


def test_temporal_rag_contrast_trap_abstention_and_llm_hook() -> None:
    """Kiểm tra truy hồi tương phản thời gian, bẫy luật hết hiệu lực, từ chối ngoài phạm vi và hook LLM."""
    engine = BoTruyHoiPhapLyThoiGian()
    engine.nap_tu_danh_sach(tai_bo_phap_ly_thoi_gian(), tai_canh_do_thi_phap_ly_thoi_gian())

    kq_t1 = engine.truy_hoi(
        "Vào tháng 3/2025, người bán có được phép hủy hóa đơn điện tử đã cấp mã bị sai sót không?",
        che_do="full",
        ngay_mac_dinh="2026-01-15",
    )
    kq_t2 = engine.truy_hoi(
        "Vào tháng 8/2025, người bán có được phép hủy hóa đơn điện tử đã cấp mã bị sai sót không?",
        che_do="full",
        ngay_mac_dinh="2026-01-15",
    )
    assert kq_t1["ket_qua"][0]["id"] == "tb_hd_d19_v1"
    assert kq_t2["ket_qua"][0]["id"] == "tb_hd_d19_v2"

    trap = engine.tra_loi_phap_ly(
        "Tháng 10/2025, công ty tôi muốn áp dụng Luật Đất đai 45/2013/QH13 để tính bảng giá đất định kỳ 5 năm có hợp lệ không?",
        che_do="full",
        ngay_mac_dinh="2026-01-15",
    )
    assert "45/2013/QH13" in trap["van_ban_het_hieu_luc_trong_hoi"]
    assert "31/2024/QH15" in trap["van_ban_thay_the_goi_y"]
    assert trap["kiem_chung_trich_dan"]["co_xung_dot_thoi_gian"] is False

    abst = engine.tra_loi_phap_ly(
        "Theo Nghị định 999/2099/NĐ-CP về khai thác khoáng sản trên sao Hỏa, thuế suất là bao nhiêu?",
        che_do="full",
        ngay_mac_dinh="2026-01-15",
    )
    assert abst["tu_choi_tra_loi"] is True
    assert abst["kiem_chung_trich_dan"]["co_trich_dan_ao"] is False

    goi_llm: Dict[str, Any] = {"called": False}

    def _mock_gemini(prompt: str) -> str:
        goi_llm["called"] = True
        assert "13/2023/NĐ-CP" in prompt
        return "Theo Điều 23 Nghị định 13/2023/NĐ-CP, phải thông báo cho Bộ Công an chậm nhất 72 giờ."

    kq_llm = engine.tra_loi_phap_ly(
        "Khi vi phạm bảo vệ dữ liệu cá nhân phải thông báo Bộ Công an trong bao nhiêu giờ?",
        che_do="full",
        ham_sinh_llm=_mock_gemini,
        ngay_mac_dinh="2026-01-15",
    )
    assert goi_llm["called"] is True
    assert kq_llm["kiem_chung_trich_dan"]["hop_le"] is True


def test_tro_ly_phap_ly_backward_compatible_with_auto_temporal_and_verifier() -> None:
    """Đảm bảo TroLyPhapLy tương thích ngược 100% và tự nhận diện mốc thời gian + kiểm chứng 3 tầng."""
    tl = TroLyPhapLy()
    kq_cu = tl.tra_cuu(
        "Trước ngày 01/06/2025, Điều 19 Nghị định 123/2020/NĐ-CP quy định xử lý hóa đơn sai sót thế nào?"
    )
    kq_moi = tl.tra_cuu(
        "Vào tháng 8/2025, Điều 19 Nghị định 123/2020/NĐ-CP quy định xử lý hóa đơn sai sót thế nào?"
    )
    for kq in (kq_cu, kq_moi):
        for khoa in ("cau_hoi", "tra_loi", "can_cu_phap_ly", "trich_dan", "diem_bam_sat_nguon", "an_toan"):
            assert khoa in kq
        assert "phan_tich_thoi_gian" in kq
        assert "kiem_chung_trich_dan" in kq
    assert kq_cu["ngay_tra_cuu"] == "2025-05-31"
    assert "hủy hóa đơn" in kq_cu["tra_loi"].lower()
    assert kq_moi["ngay_tra_cuu"] == "2025-08-15"
    assert "bãi bỏ" in kq_moi["tra_loi"].lower()

    kc_ngoai = tl.kiem_chung_cau_tra_loi(
        "Theo Điều 107 Bộ luật Lao động, làm thêm tối đa 40 giờ trong 1 tháng.",
        [{"dieu": "Điều 107", "noi_dung": "không quá 40 giờ trong 01 tháng"}],
        ngay_tra_cuu="2026-01-15",
    )
    assert kc_ngoai["hop_le"] is True


def test_vilegal_tempbench_dataset_sync_ablation_and_exports() -> None:
    """Kiểm tra đồng bộ tập dữ liệu mở ViLegal-TempBench, Ablation 5 cấu hình, Bootstrap và LaTeX/CLI."""
    goc = Path(__file__).resolve().parents[1]
    for fname in (
        "vilegal_tempbench_corpus.jsonl",
        "vilegal_tempbench_edges.jsonl",
        "vilegal_tempbench_qa.jsonl",
    ):
        p_bench = goc / "benchmarks" / "datasets" / fname
        p_pkg = goc / "vietnamese_ai" / "rag" / "legal" / "data" / fname
        assert p_bench.is_file() and p_pkg.is_file()
        assert p_bench.read_text(encoding="utf-8") == p_pkg.read_text(encoding="utf-8")

    assert len(tai_bo_phap_ly_thoi_gian()) == 60
    assert len(tai_canh_do_thi_phap_ly_thoi_gian()) == 38
    assert len(tai_bo_hoi_dap_phap_ly_thoi_gian()) == 64

    assert tinh_ndcg_at_k(["a", "b"], ["a"], k=10) == pytest.approx(1.0)
    boot = kiem_dinh_bootstrap_cap([1.0] * 20, [0.5] * 20, n_bootstrap=300, seed=42)
    assert boot["delta_trung_binh"] == pytest.approx(0.5)
    assert boot["co_y_nghia_p05"] is True

    bench = BoDanhGiaLegalTemporalRAG()
    kq = bench.chay_ablation(nhanh=False, seed=42, n_bootstrap=400)
    assert kq["boc_tach_thoi_gian_tqpa"]["do_chinh_xac_tqpa"] == 1.0

    bm25 = kq["ablation"]["bm25"]
    full = kq["ablation"]["full"]
    assert full["tva"] > bm25["tva"]
    assert full["tva"] == 1.0
    assert full["tca"] >= 0.90 and bm25["tca"] <= 0.25
    assert full["mrr_at_10"] > bm25["mrr_at_10"]
    assert full["trap_correction_rate"] == 1.0
    assert full["temporal_conflict_rate"] == 0.0
    assert full["abstention_f1"] == 1.0
    assert full["multi_hop_bridge_recall"] >= bm25["multi_hop_bridge_recall"]

    md = bench.xuat_bao_cao_markdown(kq)
    tex = bench.xuat_bang_latex(kq)
    assert "ViLegal-TempBench" in md and "Paired Bootstrap" in md
    assert r"\begin{table*}" in tex and r"\toprule" in tex

    for sym in (
        "BoPhanTichThoiGianPhapLy",
        "TemporalQueryParser",
        "BoKiemChungTrichDanPhapLy",
        "LegalCitationVerifier",
        "BoTruyHoiPhapLyThoiGian",
        "TemporalGraphLegalRAG",
        "BoDanhGiaLegalTemporalRAG",
        "ViLegalTemporalBenchmark",
    ):
        assert sym in vietnamese_ai.__all__
        assert getattr(vietnamese_ai, sym) is not None

    from vietnamese_ai.cli.main import _tao_parser, cmd_benchmark

    parser = _tao_parser()
    kq_cli = cmd_benchmark(parser.parse_args(["benchmark", "--suite", "legal-temporal", "--quick"]))
    assert kq_cli["ten_bo_chuan"] == "ViLegal-TempBench"


def test_stage3_zero_dep_llm_adapter_gemini_openai_ollama_and_citation_cff() -> None:
    """Kiểm thử Chặng 3: Bộ gọi LLM không phụ thuộc SDK (Gemini/OpenAI/Ollama qua HTTP cục bộ) và CITATION.cff."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import yaml

    from vietnamese_ai.benchmark.legal_temporal_benchmark import tao_ham_sinh_llm

    goc = Path(__file__).resolve().parents[1]
    cff = yaml.safe_load((goc / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0"
    assert cff["version"] == "31.1.0"
    assert "zenodo" in cff["doi"]
    assert (goc / "docs" / "guides" / "vilegal-temporal-rag.md").is_file()

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        tao_ham_sinh_llm("gemini", api_key="")

    ghi_nhan: Dict[str, Any] = {}

    class _MockLLMHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            ghi_nhan[self.path] = {
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            }
            if "generateContent" in self.path:
                resp = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "Căn cứ Điều 23 Nghị định 13/2023/NĐ-CP, phải thông báo Bộ Công an chậm nhất 72 giờ."
                                    }
                                ]
                            }
                        }
                    ]
                }
            elif "chat/completions" in self.path:
                resp = {"choices": [{"message": {"content": "Trả lời từ OpenAI mock."}}]}
            else:
                resp = {"response": "Trả lời từ Ollama mock."}
            raw = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            return

    srv = HTTPServer(("127.0.0.1", 0), _MockLLMHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        fn_gem = tao_ham_sinh_llm("gemini", model="gemini-2.5-flash", api_key="test-key-123", base_url=base)
        out_gem = fn_gem("Hỏi thử Gemini")
        assert "72 giờ" in out_gem
        assert "/v1beta/models/gemini-2.5-flash:generateContent" in ghi_nhan
        assert ghi_nhan["/v1beta/models/gemini-2.5-flash:generateContent"]["headers"]["x-goog-api-key"] == "test-key-123"

        fn_oai = tao_ham_sinh_llm("openai", model="gpt-4o-mini", api_key="sk-test", base_url=base)
        assert "OpenAI mock" in fn_oai("Hỏi thử OpenAI")

        fn_oll = tao_ham_sinh_llm("ollama", model="qwen2.5:3b", base_url=base)
        assert "Ollama mock" in fn_oll("Hỏi thử Ollama")
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_stage4_cli_mcp_rest_and_studio_surfaces(capsys: pytest.CaptureFixture[str]) -> None:
    """Kiểm thử Chặng 4: Đồng bộ ViLegal-TemporalRAG & Kiểm chứng 3 tầng trên CLI, MCP, REST API và Studio UI."""
    import json
    import threading
    import urllib.request

    from vietnamese_ai.agents.mcp_transport import MayChuMCPThucTe
    from vietnamese_ai.cli.main import _tao_parser, cmd_legal_ask
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    parser = _tao_parser()
    args_la = parser.parse_args(
        [
            "legal-ask",
            "Vào tháng 8/2025, người bán có được phép hủy hóa đơn điện tử đã cấp mã bị sai sót không?",
            "--mode",
            "full",
        ]
    )
    kq_cli = cmd_legal_ask(args_la)
    out_cli = capsys.readouterr().out
    assert "VILEGAL-TEMPORALRAG" in out_cli
    assert kq_cli["ket_qua"][0]["id"] == "tb_hd_d19_v2"
    assert kq_cli["kiem_chung_trich_dan"]["hop_le"] is True

    args_mcp = parser.parse_args(["mcp-server", "--temporal-legal-tools"])
    assert args_mcp.temporal_legal_tools is True

    mcp = MayChuMCPThucTe(bat_cong_cu_phap_ly_thoi_gian=True)
    try:
        r_list = mcp.xu_ly_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        assert r_list is not None
        names = {t["name"] for t in r_list["result"]["tools"]}
        assert {"tra_cuu_phap_ly_thoi_gian", "kiem_chung_trich_dan_phap_ly"}.issubset(names)

        r_ask = mcp.xu_ly_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "tra_cuu_phap_ly_thoi_gian",
                    "arguments": {
                        "cau_hoi": "Vào tháng 3/2025, người bán có được phép hủy hóa đơn điện tử sai sót không?",
                        "che_do": "full",
                    },
                },
            }
        )
        assert r_ask is not None
        ask_data = json.loads(r_ask["result"]["content"][0]["text"])
        assert ask_data["ket_qua"][0]["id"] == "tb_hd_d19_v1"

        r_ver = mcp.xu_ly_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "kiem_chung_trich_dan_phap_ly",
                    "arguments": {
                        "cau_tra_loi": "Áp dụng Luật Đất đai 45/2013/QH13 để tính bảng giá đất 5 năm.",
                        "nguon_json": json.dumps([{"so_hieu": "45/2013/QH13", "noi_dung": "5 năm"}]),
                        "ngay_tra_cuu": "2025-10-01",
                    },
                },
            }
        )
        assert r_ver is not None
        ver_data = json.loads(r_ver["result"]["content"][0]["text"])
        assert ver_data["co_xung_dot_thoi_gian"] is True
    finally:
        mcp.dong()

    studio = StudioDashboard()
    srv = studio.tao_server(host="127.0.0.1", port=0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp_html:
            html = resp_html.read().decode("utf-8")
        assert "temporal-ask-date" in html
        assert "queryTemporalLegalRAG" in html

        req_ask = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/legal/temporal-ask",
            data=json.dumps(
                {
                    "cau_hoi": "Vào tháng 8/2025, người bán có được phép hủy hóa đơn điện tử sai sót không?",
                    "che_do": "full",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ask, timeout=10) as resp_ask:
            st_ask = json.loads(resp_ask.read().decode("utf-8"))
        assert st_ask["thanh_cong"] is True
        assert st_ask["ket_qua"][0]["id"] == "tb_hd_d19_v2"
        assert st_ask["kiem_chung_trich_dan"]["hop_le"] is True

        req_ver = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/legal/verify-citation",
            data=json.dumps(
                {
                    "cau_tra_loi": "Theo Điều 26 Luật 59/2020/QH14, thời hạn là 3 ngày làm việc.",
                    "nguon_truy_hoi": [
                        {"so_hieu": "59/2020/QH14", "dieu": "Điều 26", "noi_dung": "03 ngày làm việc"}
                    ],
                    "ngay_tra_cuu": "2026-01-15",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ver, timeout=5) as resp_ver:
            st_ver = json.loads(resp_ver.read().decode("utf-8"))
        assert st_ver["thanh_cong"] is True
        assert st_ver["hop_le"] is True
    finally:
        srv.shutdown()
        srv.server_close()
        studio.dong()
        th.join(timeout=3)

    import asyncio

    try:
        from vietnamese_ai.api.fastapi_server import FastAPIServer
    except ImportError:
        return

    api = FastAPIServer(bo_xu_ly=TroLyPhapLy())
    if api.app is None:
        return

    def _asgi_post(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        sent = False
        status_code = [500]
        chunks: list[bytes] = []

        async def _recv() -> Dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return {"type": "http.disconnect"}

        async def _send(msg: Dict[str, Any]) -> None:
            if msg["type"] == "http.response.start":
                status_code[0] = int(msg["status"])
            elif msg["type"] == "http.response.body":
                chunks.append(msg.get("body", b""))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        }
        asyncio.run(api.app(scope, _recv, _send))
        return status_code[0], json.loads(b"".join(chunks).decode("utf-8"))

    st_ask_api, body_api = _asgi_post(
        "/v1/legal/temporal-ask",
        {
            "cau_hoi": "Vào tháng 9/2025, cần đóng bảo hiểm xã hội tối thiểu bao nhiêu năm để hưởng lương hưu?",
            "che_do": "full",
        },
    )
    assert st_ask_api == 200
    assert body_api["ket_qua"][0]["id"] == "tb_bhxh_2024_d64"

    st_ver_api, body_ver = _asgi_post(
        "/v1/legal/verify-citation",
        {
            "cau_tra_loi": "Theo Điều 99 Nghị định 777/2025/NĐ-CP.",
            "nguon_truy_hoi": [{"so_hieu": "59/2020/QH14", "dieu": "Điều 26", "noi_dung": "03 ngày"}],
            "ngay_tra_cuu": "2026-01-15",
        },
    )
    assert st_ver_api == 200
    assert body_ver["co_trich_dan_ao"] is True

