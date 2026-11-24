"""Kiểm thử tích hợp cho Hướng 3 (RAG Pháp lý 100 điều khoản, Theo dõi Hiệu lực, Studio Drag-Drop) và Hướng 4 (vai demo, Release Readiness, Dockerfile)."""

from __future__ import annotations

import base64
import io
import json
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest


def _tao_docx_toi_thieu(van_ban: str) -> bytes:
    """Tạo tệp `.docx` OOXML hợp lệ ngay trong bộ nhớ bằng `zipfile` chuẩn."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>",
        )
        p_xml = "".join(f"<w:p><w:r><w:t>{dong}</w:t></w:r></w:p>" for dong in van_ban.splitlines() if dong.strip())
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{p_xml}</w:body></w:document>",
        )
    return buf.getvalue()


def test_legal_validity_tracker_point_in_time_and_chains() -> None:
    """Kiểm tra theo dõi hiệu lực theo mốc thời gian, chuỗi thay thế/sửa đổi và lọc điểm RAG."""
    from vietnamese_ai.rag.legal.validity_tracker import (
        BoTheoDoiHieuLucPhapLy,
        HoSoHieuLucVanBan,
    )

    tracker = BoTheoDoiHieuLucPhapLy()
    assert len(tracker.danh_sach_van_ban()) >= 25

    ldd_2022 = tracker.kiem_tra_hieu_luc("45/2013/QH13", ngay_tra_cuu="2022-01-01")
    assert ldd_2022["tim_thay"] is True
    assert ldd_2022["trang_thai"] == "con_hieu_luc"
    assert ldd_2022["con_hieu_luc"] is True

    ldd_2025 = tracker.kiem_tra_hieu_luc("45/2013/QH13", ngay_tra_cuu="2025-01-01")
    assert ldd_2025["trang_thai"] == "het_hieu_luc"
    assert ldd_2025["con_hieu_luc"] is False
    assert "31/2024/QH15" in ldd_2025["bi_thay_the_boi"]
    assert ldd_2025["canh_bao"] and "HẾT HIỆU LỰC" in ldd_2025["canh_bao"]

    nd123_2024 = tracker.kiem_tra_hieu_luc("123/2020/NĐ-CP", ngay_tra_cuu="2024-01-01")
    assert nd123_2024["trang_thai"] == "con_hieu_luc"

    nd123_2026 = tracker.kiem_tra_hieu_luc("123/2020/NĐ-CP", ngay_tra_cuu="2026-01-01")
    assert nd123_2026["trang_thai"] == "con_hieu_luc_da_sua_doi"
    assert "70/2025/NĐ-CP" in nd123_2026["sua_doi_bo_sung_boi"]

    bhxh_truoc = tracker.kiem_tra_hieu_luc("41/2024/QH15", ngay_tra_cuu="2025-01-01")
    assert bhxh_truoc["trang_thai"] == "chua_co_hieu_luc"
    bhxh_sau = tracker.kiem_tra_hieu_luc("41/2024/QH15", ngay_tra_cuu="2025-08-01")
    assert bhxh_sau["trang_thai"] == "con_hieu_luc"

    tracker.dang_ky_van_ban(
        HoSoHieuLucVanBan(
            so_hieu="99/2026/NĐ-CP",
            ten_van_ban="Nghị định thử nghiệm thay thế Nghị định 30/2020/NĐ-CP",
            linh_vuc="Hành chính",
            ngay_ban_hanh="2026-02-01",
            ngay_hieu_luc="2026-03-01",
            thay_the_cho=["30/2020/NĐ-CP"],
        )
    )
    nd30_het = tracker.kiem_tra_hieu_luc("30/2020/NĐ-CP", ngay_tra_cuu="2026-04-01")
    assert nd30_het["trang_thai"] == "het_hieu_luc"
    assert "99/2026/NĐ-CP" in nd30_het["bi_thay_the_boi"]

    bc_hd = tracker.kiem_tra_hieu_luc_trong_van_ban(
        "Hợp đồng căn cứ Luật Đất đai 45/2013/QH13 và Nghị định 123/2020/NĐ-CP.",
        ngay_tra_cuu="2026-01-15",
    )
    assert bc_hd["tong_so_hieu_phat_hien"] == 2
    assert bc_hd["so_van_ban_het_hieu_luc"] == 1
    assert bc_hd["so_van_ban_da_sua_doi"] == 1
    assert bc_hd["hop_le_hieu_luc"] is False

    rag_items = [
        {"noi_dung": "Căn cứ Luật Đất đai 45/2013/QH13", "diem": 1.0, "metadata": {"so_hieu": "45/2013/QH13"}},
        {"noi_dung": "Căn cứ Luật Đất đai 31/2024/QH15", "diem": 1.0, "metadata": {"so_hieu": "31/2024/QH15"}},
    ]
    kq_giam, canh_bao = tracker.bo_sung_hieu_luc_cho_rag(rag_items, ngay_tra_cuu="2026-01-15", loai_bo_het_hieu_luc=False)
    assert len(kq_giam) == 2
    assert kq_giam[0]["diem"] == pytest.approx(0.65, rel=1e-4)
    assert kq_giam[1]["diem"] == pytest.approx(1.0, rel=1e-4)
    assert any("45/2013/QH13" in cb for cb in canh_bao)

    kq_loc, _ = tracker.bo_sung_hieu_luc_cho_rag(rag_items, ngay_tra_cuu="2026-01-15", loai_bo_het_hieu_luc=True)
    assert len(kq_loc) == 1
    assert kq_loc[0]["metadata"]["so_hieu"] == "31/2024/QH15"


def test_expanded_open_legal_corpus_100_and_tro_ly_phap_ly() -> None:
    """Kiểm tra kho pháp lý mở 100 điều khoản / 35 câu QA, tích hợp RAGPipeline & TroLyPhapLy."""
    from vietnamese_ai.rag.legal.open_corpus import tai_bo_hoi_dap_phap_ly_mo, tai_bo_phap_ly_mo
    from vietnamese_ai.rag.rag_pipeline import RAGPipeline
    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

    corpus = tai_bo_phap_ly_mo()
    qa_pairs = tai_bo_hoi_dap_phap_ly_mo()
    assert len(corpus) == 100
    assert len(qa_pairs) == 35

    corpus_ids = {str(item["id"]) for item in corpus}
    assert len(corpus_ids) == 100
    for qa in qa_pairs:
        assert str(qa["id_ky_vong"]) in corpus_ids

    rag = RAGPipeline(che_do_hanh_chinh=True, top_k=3)
    rag.them_tai_lieu(
        "luat_cu",
        "Luật Đất đai số 45/2013/QH13.\nĐiều 167. Người sử dụng đất có quyền chuyển nhượng quyền sử dụng đất.",
        metadata={"so_hieu": "45/2013/QH13"},
        che_do_hanh_chinh=True,
    )
    kq_rag = rag.hoi("Quyền chuyển nhượng quyền sử dụng đất theo Luật Đất đai 45/2013/QH13?")
    td = kq_rag.get("trich_dan_phap_ly")
    assert td is not None
    assert td.get("hieu_luc_van_ban")
    assert any("45/2013/QH13" in cb for cb in td.get("canh_bao_hieu_luc", []))

    tro_ly = TroLyPhapLy(nap_kho_mo=True)
    kq_bhxh = tro_ly.tra_cuu("Theo Luật Bảo hiểm xã hội 2024 cần đóng bảo hiểm xã hội tối thiểu bao nhiêu năm để hưởng lương hưu?")
    assert kq_bhxh["an_toan"] is True
    assert "15" in kq_bhxh["tra_loi"] or any("15" in td_i["trich_doan"] for td_i in kq_bhxh["trich_dan"])

    kt_sh = tro_ly.kiem_tra_hieu_luc("45/2013/QH13", ngay_tra_cuu="2026-01-01")
    assert kt_sh["trang_thai"] == "het_hieu_luc"
    kt_vb = tro_ly.kiem_tra_hieu_luc("Hợp đồng viện dẫn Luật 58/2014/QH13 về bảo hiểm xã hội.", ngay_tra_cuu="2026-01-01")
    assert kt_vb["so_van_ban_het_hieu_luc"] == 1


def test_studio_dashboard_drag_drop_upload_and_legal_validity_http() -> None:
    """Kiểm tra trực tiếp HTTP endpoints kéo-thả tệp `.docx` và kiểm toán hiệu lực trên `StudioDashboard`."""
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    studio = StudioDashboard(host="127.0.0.1", port=0)
    httpd = studio.tao_server(host="127.0.0.1", port=0)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as resp:
            html = resp.read().decode("utf-8")
        assert "rag-dropzone" in html
        assert "uploadRagFile" in html

        docx_bytes = _tao_docx_toi_thieu(
            "HỢP ĐỒNG CHUYỂN NHƯỢNG QUYỀN SỬ DỤNG ĐẤT\n"
            "Điều 1. Căn cứ Luật Đất đai số 45/2013/QH13 và Nghị định 13/2023/NĐ-CP.\n"
            "Điều 2. Hai bên cam kết bảo mật dữ liệu cá nhân trong vòng 72 giờ."
        )
        kq_up = _post_json(
            "/api/rag/upload-file",
            {
                "filename": "hop_dong_dat_dai.docx",
                "content_base64": base64.b64encode(docx_bytes).decode("ascii"),
                "che_do_hanh_chinh": True,
            },
        )
        assert kq_up["thanh_cong"] is True
        assert kq_up["dinh_dang"] == "docx"
        assert kq_up["so_chunks"] >= 1
        assert kq_up["hieu_luc_phap_ly"]["so_van_ban_het_hieu_luc"] == 1

        kq_bad = _post_json(
            "/api/rag/upload-file",
            {"filename": "malware.exe", "content_base64": base64.b64encode(b"MZ").decode("ascii")},
        )
        assert kq_bad["thanh_cong"] is False

        kq_val = _post_json(
            "/api/legal/check-validity",
            {"so_hieu": "45/2013/QH13", "ngay_tra_cuu": "2026-01-01"},
        )
        assert kq_val["trang_thai"] == "het_hieu_luc"
        assert "31/2024/QH15" in kq_val["bi_thay_the_boi"]

        kq_val_txt = _post_json(
            "/api/legal/check-validity",
            {"van_ban": "Viện dẫn Nghị định 123/2020/NĐ-CP và Luật 45/2013/QH13.", "ngay_tra_cuu": "2026-01-01"},
        )
        assert kq_val_txt["so_van_ban_het_hieu_luc"] == 1
        assert kq_val_txt["so_van_ban_da_sua_doi"] == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def test_cli_demo_5_minute_dx_and_release_check() -> None:
    """Kiểm tra lệnh `vai demo` (4 kịch bản offline) và kiểm định phát hành `vai release-check`."""
    from vietnamese_ai.cli.main import _tao_parser, cmd_demo, cmd_release_check
    from vietnamese_ai.utils.versioning import QuanLyPhatHanhGoi

    parser = _tao_parser()
    args_all = parser.parse_args(["demo", "all"])
    kq_all = cmd_demo(args_all)
    assert kq_all["che_do"] == "all"
    assert kq_all["legal_rag"]["tong_dieu_khoan_kho_mo"] == 100
    assert kq_all["legal_rag"]["kiem_toan_hieu_luc"]["so_van_ban_het_hieu_luc"] == 1
    assert kq_all["edge_chat"]["thong_ke_edge"]["so_token"] >= 1
    assert kq_all["security"]["chan_prompt_injection"] is True
    assert kq_all["security"]["kiem_toan_mat_ma"]["tong_quan_dat"] is True
    assert kq_all["flywheel"]["trang_thai"] == "xem_truoc_dry_run"

    for che_do in ("legal-rag", "edge-chat", "security", "flywheel"):
        res_mode = cmd_demo(parser.parse_args(["demo", che_do]))
        assert res_mode["che_do"] == che_do

    bc_direct = QuanLyPhatHanhGoi.kiem_tra_san_sang_phat_hanh()
    assert bc_direct["san_sang_phat_hanh"] is True
    assert bc_direct["semver"]["hop_le_semver"] is True
    assert bc_direct["semver"]["dong_bo_phien_ban"] is True
    assert bc_direct["phu_thuoc"]["hop_le"] is True
    assert bc_direct["thieu_type_stub"] == []
    assert bc_direct["rust_abi3_san_sang"] is True
    assert bc_direct["thieu_tep_bat_buoc"] == []

    bc_cli = cmd_release_check(parser.parse_args(["release-check"]))
    assert bc_cli["san_sang_phat_hanh"] is True


def test_repo_hygiene_dockerfile_and_top_level_exports() -> None:
    """Kiểm tra `.gitignore`, Dockerfile benchmark đa tầng và các biểu tượng lazy export cấp cao."""
    import vietnamese_ai
    import vietnamese_ai.legal as legal_pkg

    goc = Path(__file__).resolve().parents[1]
    gitignore = (goc / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("rust_core/target/", "vai_remote_jobs/", "vai_workspace/", "*.sig"):
        assert pattern in gitignore

    dockerfile = (goc / "benchmarks" / "Dockerfile.benchmark").read_text(encoding="utf-8")
    assert "AS rust-builder" in dockerfile
    assert "run_benchmarks.py" in dockerfile

    for sym in (
        "BoTheoDoiHieuLucPhapLy",
        "LegalValidityTracker",
        "HoSoHieuLucVanBan",
        "tai_bo_phap_ly_mo",
        "tai_bo_hoi_dap_phap_ly_mo",
    ):
        assert sym in vietnamese_ai.__all__
        assert getattr(vietnamese_ai, sym) is not None
        assert getattr(legal_pkg, sym) is not None
