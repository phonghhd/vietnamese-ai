"""Bộ kiểm thử hồi quy cho Giai đoạn M1 (Kho `.vaidb` `vai pull-kb`, tách Khoản/Điểm, `ViLegal-TempBench v2` T5 + `naive_date_filter`)."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest


def test_m1_1_auto_ingest_clause_splitting_quality_audit_and_manifest(tmp_path: Path) -> None:
    """Xác minh BoNapVanBanPhapQuyTuDong tách Khoản cho Điều dài, kiểm định chất lượng và xuất manifest SHA-256."""
    from vietnamese_ai.rag.legal.auto_ingest import BoNapVanBanPhapQuyTuDong

    noi_dung = (
        "LUẬT THỬ NGHIỆM QUẢN TRỊ DOANH NGHIỆP\n"
        "Số: 99/2025/QH15 ngày 15 tháng 01 năm 2025\n"
        "Có hiệu lực thi hành từ ngày 01 tháng 07 năm 2025.\n"
        "Chương I. QUY ĐỊNH CHUNG\n"
        "Điều 1. Quyền và nghĩa vụ cơ bản của doanh nghiệp và người quản lý\n"
        "1. Doanh nghiệp có quyền tự chủ kinh doanh, lựa chọn hình thức tổ chức kinh doanh, chủ động lựa chọn ngành nghề, "
        "địa bàn, hình thức kinh doanh, chủ động điều chỉnh quy mô và ngành nghề kinh doanh theo quy định của pháp luật hiện hành "
        "và công khai minh bạch hồ sơ tài chính định kỳ hằng quý cho các cổ đông cũng như cơ quan thuế quản lý trực tiếp.\n"
        "2. Doanh nghiệp có nghĩa vụ tổ chức công tác kế toán, lập và nộp báo cáo tài chính trung thực, chính xác, đúng thời hạn "
        "theo quy định của pháp luật về kế toán, thống kê, kê khai nộp thuế đầy đủ, bảo vệ dữ liệu cá nhân khách hàng và bảo đảm "
        "quyền lợi hợp pháp của người lao động trong suốt quá trình hoạt động sản xuất kinh doanh.\n"
        "Điều 3. Điều khoản thi hành\n"
        "Luật này có hiệu lực kể từ ngày 01/07/2025 và thay thế Luật 88/2015/QH13.\n"
    )
    src_file = tmp_path / "luat_99_2025.txt"
    src_file.write_text(noi_dung, encoding="utf-8")

    pipeline = BoNapVanBanPhapQuyTuDong()
    kq_pt = pipeline.phan_tich_van_ban(src_file, tach_khoan_dieu_dai=True, nguong_tach_khoan_tu=60)
    khoan_d1 = [d for d in kq_pt.cac_dieu_khoan if d["dieu"] == "Điều 1"]
    assert len(khoan_d1) == 2
    assert khoan_d1[0]["khoan"] == "Khoản 1"
    assert khoan_d1[1]["khoan"] == "Khoản 2"
    assert "Điều 1." in khoan_d1[1]["noi_dung"]

    audit = pipeline.kiem_tra_chat_luong_bo_van_ban([kq_pt])
    assert audit["dat_chat_luong"] is True
    assert any("nhảy cóc" in cb for cb in audit["canh_bao"])
    assert any("88/2015/QH13" in cb for cb in audit["canh_bao"])

    out_db = tmp_path / "out_99.vaidb"
    kq_pack = pipeline.dong_goi_kho_vaidb(
        src_file,
        out_db,
        xay_dung_hnsw=True,
        tach_khoan_dieu_dai=True,
        nguong_tach_khoan_tu=60,
        xuat_manifest=True,
    )
    assert Path(kq_pack["duong_dan_vaidb"]).is_file()
    assert kq_pack["duong_dan_manifest"] is not None
    manifest_p = Path(kq_pack["duong_dan_manifest"])
    assert manifest_p.is_file()
    manifest_json = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert manifest_json["sha256"] == kq_pack["sha256"]
    assert manifest_json["tong_dieu_khoan"] >= 3


def test_m1_1_kb_hub_local_build_remote_verify_and_cli_pull_kb(tmp_path: Path) -> None:
    """Xác minh QuanLyKhoTriThucPhapLy dựng cục bộ, đóng gói thư mục, tải HTTPS/loopback kiểm SHA-256 và lệnh CLI `vai pull-kb`."""
    from vietnamese_ai.cli.main import _tao_parser, cmd_pull_kb
    from vietnamese_ai.rag.legal.kb_hub import DANH_MUC_KHO_PHAP_LY_VN, QuanLyKhoTriThucPhapLy

    assert "vn-business-core-2025" in DANH_MUC_KHO_PHAP_LY_VN
    assert "vn-labor-bhxh-2025" in DANH_MUC_KHO_PHAP_LY_VN
    assert "vn-tax-invoice-2025" in DANH_MUC_KHO_PHAP_LY_VN

    hub = QuanLyKhoTriThucPhapLy(thu_muc_luu=tmp_path / "kb_store")
    ds_truoc = hub.danh_sach_goi()
    assert len(ds_truoc) == 5

    kq_build = hub.dung_goi_cuc_bo("vn-data-digital-2025")
    assert kq_build["trang_thai"] == "da_dung_thanh_cong"
    db_path = Path(kq_build["duong_dan_vaidb"])
    assert db_path.is_file()
    assert kq_build["tong_dieu_khoan"] >= 5

    xm = hub.xac_minh_goi(db_path)
    assert xm["hop_le"] is True
    assert xm["khop_manifest_sha256"] is True
    assert xm["co_sieu_du_lieu_phap_ly"] is True

    raw_db_bytes = db_path.read_bytes()
    sha_expected = hashlib.sha256(raw_db_bytes).hexdigest()

    class _MockKBHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw_db_bytes)))
            self.end_headers()
            self.wfile.write(raw_db_bytes)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockKBHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        dl_dest = tmp_path / "downloaded.vaidb"
        kq_dl = hub.tai_hoac_dung_goi(
            "vn-data-digital-2025",
            duong_dan_dich=dl_dest,
            uu_tien_cuc_bo=False,
            url_tuy_chinh=f"http://127.0.0.1:{port}/vn-data-digital-2025.vaidb",
            sha256_ky_vong=sha_expected,
            chay_that=True,
        )
        assert kq_dl["trang_thai"] == "da_tai_thanh_cong"
        assert kq_dl["sha256"] == sha_expected
        assert hub.xac_minh_goi(dl_dest)["hop_le"] is True

        with pytest.raises(ValueError, match="SHA-256"):
            hub.tai_hoac_dung_goi(
                "vn-data-digital-2025",
                duong_dan_dich=tmp_path / "bad.vaidb",
                uu_tien_cuc_bo=False,
                url_tuy_chinh=f"http://127.0.0.1:{port}/vn-data-digital-2025.vaidb",
                sha256_ky_vong="0" * 64,
                chay_that=True,
            )
    finally:
        srv.shutdown()
        srv.server_close()

    parser = _tao_parser()
    args_list = parser.parse_args(["pull-kb", "--list", "--dir", str(tmp_path / "kb_store")])
    ds_cli = cmd_pull_kb(args_list)
    assert any(x["ma_goi"] == "vn-data-digital-2025" and x["da_cai_cuc_bo"] for x in ds_cli)

    args_ver = parser.parse_args(["pull-kb", "--verify", str(db_path)])
    res_ver = cmd_pull_kb(args_ver)
    assert res_ver["hop_le"] is True


def test_m1_2_two_date_disambiguation_naive_date_filter_and_t5_error_analysis() -> None:
    """Xác minh phân tách 2 mốc năm, baseline `naive_date_filter`, split `T5` đời thường và Error Analysis tự động."""
    from vietnamese_ai.benchmark.datasets_vn import tai_bo_hoi_dap_phap_ly_thoi_gian
    from vietnamese_ai.benchmark.legal_temporal_benchmark import BoDanhGiaLegalTemporalRAG
    from vietnamese_ai.rag.legal.temporal_query import BoPhanTichThoiGianPhapLy

    parser_tg = BoPhanTichThoiGianPhapLy()
    kq_2nam = parser_tg.phan_tich(
        "Nhà tôi ở mảnh đất ông bà để lại không có giấy tờ từ năm 2010 và không tranh chấp thì đến năm 2025 có xin làm sổ đỏ không?"
    )
    assert kq_2nam.ngay_tra_cuu.startswith("2025-")

    qa_all = tai_bo_hoi_dap_phap_ly_thoi_gian()
    t5_list = [q for q in qa_all if q.get("loai") == "T5"]
    assert len(qa_all) == 84
    assert len(t5_list) == 16

    bench = BoDanhGiaLegalTemporalRAG()
    kq_quick = bench.chay_ablation(nhanh=True, seed=42, n_bootstrap=100)
    assert kq_quick["phien_ban_bo_chuan"] == "2.0"
    assert "naive_date_filter" in kq_quick["ablation"]
    assert "T5" in kq_quick["ablation"]["full"]["theo_nhom"]
    assert "phan_tich_loi" in kq_quick["ablation"]["full"]

    md = bench.xuat_bao_cao_markdown(kq_quick)
    tex = bench.xuat_bang_latex(kq_quick)
    assert "naive_date_filter" in md
    assert "Phân Tích Lỗi Tự Động" in md
    assert "Colloq R@3 (T5)" in md
    assert "T5-R@3" in tex
