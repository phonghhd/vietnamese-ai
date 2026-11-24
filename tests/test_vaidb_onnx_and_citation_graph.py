"""Kiểm thử tích hợp cho Ưu tiên 1: Kho tri thức bền vững `.vaidb` (SQLite+FTS5+Vector), Embedding/Reranker ONNX Tiếng Việt & Đồ thị Tham chiếu Pháp lý."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


def test_vaidb_single_file_persistence_fts5_and_reopen(tmp_path: Path) -> None:
    """Kiểm tra lưu bền vững một tệp `.vaidb`, tìm kiếm FTS5 tiếng Việt có/không dấu, IVF-Flat và mở lại tức thì."""
    from vietnamese_ai.rag.vaidb_store import KhoTriThucVAIDB

    db_file = tmp_path / "tri_thuc_phap_ly.vaidb"
    store = KhoTriThucVAIDB(duong_dan=db_file, kich_thuoc=128)
    store.them_tai_lieu(
        "doc_lao_dong",
        "Điều 105 Bộ luật Lao động 2019 quy định thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày và 48 giờ trong 01 tuần.",
        metadata={"linh_vuc": "lao_dong", "so_hieu": "45/2019/QH14"},
    )
    store.them_tai_lieu(
        "doc_du_lieu",
        "Nghị định 13/2023/NĐ-CP yêu cầu thông báo vi phạm bảo vệ dữ liệu cá nhân cho Bộ Công an trong 72 giờ.",
        metadata={"linh_vuc": "du_lieu", "so_hieu": "13/2023/NĐ-CP"},
    )
    assert store.so_luong() == 2
    assert store.so_tai_lieu() == 2
    store.dong()

    assert db_file.is_file() and db_file.stat().st_size > 0

    reopened = KhoTriThucVAIDB(duong_dan=db_file, kich_thuoc=128)
    try:
        assert reopened.so_luong() == 2
        assert reopened.so_tai_lieu() == 2

        fts_co_dau = reopened.tim_kiem_fts("thời giờ làm việc bình thường", top_k=2)
        assert fts_co_dau and fts_co_dau[0]["ma"] == "doc_lao_dong"

        fts_khong_dau = reopened.tim_kiem_fts("thoi gio lam viec binh thuong", top_k=2)
        assert fts_khong_dau and fts_khong_dau[0]["ma"] == "doc_lao_dong"

        hybrid = reopened.tim_kiem_lai("vi pham du lieu ca nhan 72 gio", top_k=2)
        assert hybrid and hybrid[0]["ma"] == "doc_du_lieu"

        ivf_info = reopened.xay_dung_chi_muc_ivf(so_cum=2, nprobe=2)
        assert ivf_info["da_xay_dung"] is True
        vec_hit = reopened.tim_kiem("thời giờ làm việc 48 giờ", top_k=1)
        assert vec_hit and vec_hit[0]["ma"] == "doc_lao_dong"

        da_xoa = reopened.xoa_tai_lieu("doc_lao_dong")
        assert da_xoa == 1
        assert reopened.so_luong() == 1
        stats = reopened.thong_ke()
        assert stats["dinh_dang"] == "vaidb_v1"
        assert stats["kich_thuoc_tep_bytes"] > 0
    finally:
        reopened.dong()

    reopened2 = KhoTriThucVAIDB(duong_dan=db_file, kich_thuoc=128)
    try:
        assert reopened2.so_luong() == 1
        assert reopened2.danh_sach_ma() == ["doc_du_lieu"]
    finally:
        reopened2.dong()


def test_rag_pipeline_vaidb_incremental_folder_sync(tmp_path: Path) -> None:
    """Kiểm tra đồng bộ tăng dần thư mục theo SHA-256 trên `RAGPipeline(backend='vaidb')` và truy vấn lại sau khi mở lại."""
    from vietnamese_ai.rag.rag_pipeline import RAGPipeline

    docs_dir = tmp_path / "ho_so_luat"
    docs_dir.mkdir()
    tep1 = docs_dir / "lao_dong.txt"
    tep2 = docs_dir / "dat_dai.txt"
    tep1.write_text(
        "Bộ luật Lao động 45/2019/QH14.\n"
        "Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động của người sử dụng lao động phải báo trước ít nhất 45 ngày với hợp đồng không xác định thời hạn.",
        encoding="utf-8",
    )
    tep2.write_text(
        "Luật Đất đai 45/2013/QH13.\n"
        "Điều 167. Quyền chuyển nhượng quyền sử dụng đất và cấp Giấy chứng nhận quyền sử dụng đất.",
        encoding="utf-8",
    )

    db_path = tmp_path / "company_kb.vaidb"
    rag1 = RAGPipeline(
        backend="vaidb",
        duong_dan_vaidb=db_path,
        che_do_nhung="onnx_viet",
        che_do_rerank="onnx_viet",
        che_do_hanh_chinh=True,
        top_k=2,
    )
    try:
        bc1 = rag1.dong_bo_thu_muc(docs_dir, de_quy=True, xoa_tep_da_mat=True)
        assert bc1["so_tep_cap_nhat"] == 2
        assert bc1["so_tep_bo_qua_khong_doi"] == 0
        assert bc1["so_tep_da_xoa"] == 0

        bc2 = rag1.dong_bo_thu_muc(docs_dir, de_quy=True, xoa_tep_da_mat=True)
        assert bc2["so_tep_cap_nhat"] == 0
        assert bc2["so_tep_bo_qua_khong_doi"] == 2

        tep1.write_text(
            "Bộ luật Lao động 45/2019/QH14.\n"
            "Điều 36. Khi đơn phương chấm dứt hợp đồng lao động không xác định thời hạn, người sử dụng lao động phải báo trước ít nhất 45 ngày làm việc.",
            encoding="utf-8",
        )
        tep2.unlink()

        bc3 = rag1.dong_bo_thu_muc(docs_dir, de_quy=True, xoa_tep_da_mat=True)
        assert bc3["so_tep_cap_nhat"] == 1
        assert bc3["so_tep_bo_qua_khong_doi"] == 0
        assert bc3["so_tep_da_xoa"] == 1
    finally:
        rag1.dong()

    rag2 = RAGPipeline(
        backend="vaidb",
        duong_dan_vaidb=db_path,
        che_do_nhung="onnx_viet",
        che_do_rerank="onnx_viet",
        che_do_hanh_chinh=True,
        top_k=2,
    )
    try:
        assert rag2.trich_xuat.so_tai_lieu() == 1
        kq = rag2.hoi("Công ty sa thải nhân viên cần báo trước bao nhiêu ngày?")
        assert kq["so_luong_nguon"] >= 1
        assert "45" in kq["tra_loi"]
        td = kq.get("trich_dan_phap_ly")
        assert td is not None
        assert "do_thi_tham_chieu" in td
        hd_sh = [x["so_hieu"] for x in td["do_thi_tham_chieu"]["van_ban_huong_dan_con_hieu_luc"]]
        assert "145/2020/NĐ-CP" in hd_sh
    finally:
        rag2.dong()


def test_vietnamese_onnx_semantic_synonym_embedder_and_reranker() -> None:
    """Kiểm tra Bộ nhúng & Cross-Encoder Reranker ONNX tiếng Việt xử lý câu hỏi đồng nghĩa đời thường."""
    from vietnamese_ai.rag.onnx_semantic import BoNhungONNXTiengViet, BoSapXepLaiONNXTiengViet
    from vietnamese_ai.rag.reranker import SapXepLai

    emb = BoNhungONNXTiengViet(kich_thuoc=384)
    assert emb.che_do == "semantic_synonym_cpu"

    q_vec = emb.nhung("Công ty sa thải nhân viên trước hạn thì báo trước bao lâu?")
    d_match = emb.nhung("Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động của người sử dụng lao động.")
    d_other = emb.nhung("Điều 10. Thuế suất thuế giá trị gia tăng và hóa đơn điện tử.")

    assert q_vec.shape == (384,)
    sim_match = float(np.dot(q_vec, d_match))
    sim_other = float(np.dot(q_vec, d_other))
    assert sim_match > sim_other + 0.1

    batch_vecs = emb.nhung(["sổ đỏ sang tên đất", "mở công ty cổ phần"])
    assert batch_vecs.shape == (2, 384)

    reranker = BoSapXepLaiONNXTiengViet()
    candidates = [
        {"ma": "doc_thue", "diem": 0.60, "noi_dung": "Nghị định 123/2020/NĐ-CP về hóa đơn chứng từ thuế GTGT."},
        {
            "ma": "doc_nd13",
            "diem": 0.45,
            "noi_dung": "Điều 23 Nghị định 13/2023/NĐ-CP về thông báo vi phạm bảo vệ dữ liệu cá nhân cho Bộ Công an trong 72 giờ.",
        },
        {"ma": "doc_dat", "diem": 0.55, "noi_dung": "Luật Đất đai 31/2024/QH15 về giấy chứng nhận quyền sử dụng đất."},
    ]
    ranked = reranker.sap_xep_lai("Khách hàng bị lộ dữ liệu cá nhân cần báo cơ quan nào?", candidates, top_k=3)
    assert ranked[0]["ma"] == "doc_nd13"
    assert ranked[0]["diem_cross_encoder"] > ranked[1]["diem_cross_encoder"]

    sap_xep = SapXepLai(che_do="onnx_viet")
    ranked2 = sap_xep.sap_xep_lai("lộ dữ liệu cá nhân", candidates, top_k=2)
    assert ranked2[0]["ma"] == "doc_nd13"

    with pytest.raises(FileNotFoundError):
        BoNhungONNXTiengViet.luong_tu_hoa_int8("/khong_ton_tai/model.onnx", "/tmp/out_int8.onnx")


def test_legal_citation_graph_chains_auto_extract_and_mermaid() -> None:
    """Kiểm tra Đồ thị Tham chiếu Pháp lý: truy vết luật thay thế, kéo nghị định/thông tư hướng dẫn và xuất Mermaid."""
    from vietnamese_ai.rag.legal.citation_graph import DoThiThamChieuPhapLy

    do_thi = DoThiThamChieuPhapLy()
    stats = do_thi.thong_ke()
    assert stats["so_van_ban_trong_do_thi"] >= 30
    assert stats["tong_quan_he"] >= 25

    kq_dat = do_thi.truy_van_van_ban_lien_quan("45/2013/QH13", ngay_tra_cuu="2026-01-15", chi_lay_con_hieu_luc=True)
    assert kq_dat["hieu_luc_goc"]["trang_thai"] == "het_hieu_luc"
    thay_the_sh = [x["so_hieu"] for x in kq_dat["van_ban_thay_the_hien_hanh"]]
    assert "31/2024/QH15" in thay_the_sh
    huong_dan_sh = [x["so_hieu"] for x in kq_dat["van_ban_huong_dan"]]
    assert "102/2024/NĐ-CP" in huong_dan_sh
    assert "43/2014/NĐ-CP" not in huong_dan_sh

    kq_hd = do_thi.truy_van_van_ban_lien_quan("123/2020/NĐ-CP", ngay_tra_cuu="2026-01-15", chi_lay_con_hieu_luc=True)
    sua_doi_sh = [x["so_hieu"] for x in kq_hd["van_ban_sua_doi_bo_sung"]]
    hd_hd_sh = [x["so_hieu"] for x in kq_hd["van_ban_huong_dan"]]
    assert "70/2025/NĐ-CP" in sua_doi_sh
    assert "32/2025/TT-BTC" in hd_hd_sh
    assert "78/2021/TT-BTC" not in hd_hd_sh

    added = do_thi.trich_xuat_va_them_tu_van_ban(
        "88/2026/TT-BLĐTBXH",
        "Thông tư hướng dẫn thi hành Nghị định 145/2020/NĐ-CP và Bộ luật Lao động 45/2019/QH14.",
    )
    assert len(added) >= 2
    kq_ld = do_thi.truy_van_van_ban_lien_quan("45/2019/QH14", ngay_tra_cuu="2026-06-01")
    assert any(x["so_hieu"] == "88/2026/TT-BLĐTBXH" for x in kq_ld["van_ban_huong_dan"])

    mmd = do_thi.xuat_do_thi_mermaid("45/2013/QH13", do_sau=2)
    assert mmd.startswith("flowchart LR")
    assert "31/2024/QH15" in mmd
    assert "102/2024/NĐ-CP" in mmd


def test_ecosystem_wiring_cli_studio_tro_ly_e2e_and_exports(tmp_path: Path) -> None:
    """Kiểm tra liên kết toàn hệ thống: TroLyPhapLy, QuyTrinhDauCuoi, CLI (`vai rag --db`, `vai legal-graph`), Studio HTTP và Exports."""
    import vietnamese_ai
    import vietnamese_ai.legal as legal_pkg
    import vietnamese_ai.rag as rag_pkg
    from vietnamese_ai.cli.main import _tao_parser, cmd_legal_graph, cmd_rag
    from vietnamese_ai.core.end_to_end import QuyTrinhDauCuoi
    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard
    from vietnamese_ai.utils.versioning import QuanLyPhatHanhGoi

    for sym in (
        "KhoTriThucVAIDB",
        "VAIDBStore",
        "BoNhungONNXTiengViet",
        "VietnameseONNXEmbedder",
        "BoSapXepLaiONNXTiengViet",
        "VietnameseONNXReranker",
        "QuanHePhapLy",
        "DoThiThamChieuPhapLy",
        "LegalCitationGraph",
    ):
        assert sym in vietnamese_ai.__all__
        assert getattr(vietnamese_ai, sym) is not None
        assert getattr(rag_pkg, sym) is not None
    assert legal_pkg.DoThiThamChieuPhapLy is vietnamese_ai.DoThiThamChieuPhapLy

    tl = TroLyPhapLy(duong_dan_vaidb=tmp_path / "tro_ly.vaidb", dung_onnx_semantic=True)
    try:
        lien_quan = tl.tra_cuu_van_ban_lien_quan("45/2013/QH13", ngay_tra_cuu="2026-01-15", chi_lay_con_hieu_luc=True)
        assert any(x["so_hieu"] == "31/2024/QH15" for x in lien_quan["van_ban_thay_the_hien_hanh"])
    finally:
        tl.rag.dong()

    wf = QuyTrinhDauCuoi(ten_du_an="TestVAIDB", thu_muc_lam_viec=str(tmp_path / "e2e_ws"))
    try:
        info_db = wf.mo_kho_tri_thuc_vaidb(che_do_hanh_chinh=True, dung_onnx_semantic=True)
        assert info_db["duong_dan_vaidb"].endswith(".vaidb")
        assert wf.rag is not None and wf.rag.backend == "vaidb"
    finally:
        if wf.rag is not None:
            wf.rag.dong()

    cli_dir = tmp_path / "cli_docs"
    cli_dir.mkdir()
    (cli_dir / "luat_bhxh.txt").write_text(
        "Luật Bảo hiểm xã hội 41/2024/QH15.\nĐiều 64. Người lao động đóng bảo hiểm xã hội tối thiểu 15 năm được hưởng lương hưu.",
        encoding="utf-8",
    )
    cli_db = tmp_path / "cli_kb.vaidb"
    parser = _tao_parser()
    kq_cli_1 = cmd_rag(
        parser.parse_args(
            ["rag", "--docs", str(cli_dir), "--db", str(cli_db), "--admin", "--semantic-onnx", "--query", "về hưu đóng mấy năm?"]
        )
    )
    assert "15" in kq_cli_1["tra_loi"]

    kq_cli_2 = cmd_rag(
        parser.parse_args(
            ["rag", "--db", str(cli_db), "--admin", "--semantic-onnx", "--query", "lương hưu bảo hiểm xã hội"]
        )
    )
    assert "15" in kq_cli_2["tra_loi"]

    kq_lg = cmd_legal_graph(parser.parse_args(["legal-graph", "45/2013/QH13", "--date", "2026-01-15", "--active-only", "--mermaid"]))
    assert any(x["so_hieu"] == "31/2024/QH15" for x in kq_lg["van_ban_thay_the_hien_hanh"])
    assert "flowchart LR" in kq_lg["mermaid"]

    studio = StudioDashboard(host="127.0.0.1", port=0, duong_dan_vaidb=str(tmp_path / "studio.vaidb"))
    httpd = studio.tao_server(host="127.0.0.1", port=0)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/rag/vaidb/stats", timeout=10) as resp:
            stats_http: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        assert stats_http["backend"] == "vaidb"

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/legal/citation-graph",
            data=json.dumps({"so_hieu": "123/2020/NĐ-CP", "ngay_tra_cuu": "2026-01-15", "chi_lay_con_hieu_luc": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            graph_http: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        assert any(x["so_hieu"] == "70/2025/NĐ-CP" for x in graph_http["van_ban_sua_doi_bo_sung"])
        assert "flowchart LR" in graph_http["mermaid"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        studio.rag.dong()
        thread.join(timeout=3)

    bc_rel = QuanLyPhatHanhGoi.kiem_tra_san_sang_phat_hanh()
    assert bc_rel["san_sang_phat_hanh"] is True
    assert bc_rel["thieu_type_stub"] == []
