"""Kiểm thử tích hợp cho Ưu tiên 4 (Eval Gate hồi quy + Thumbs Up/Down -> DPO Export -> Flywheel) và Ưu tiên 5 (Ranh giới Core/Experimental, dữ liệu đóng gói wheel & phát hành TestPyPI/PyPI)."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict


def test_eval_gate_promote_vs_reject_regression() -> None:
    """Kiểm tra `CongKiemDinhMoHinh` thăng hạng bản tốt và chặn bản bị tụt điểm hoặc dẫn luật hết hạn."""
    from vietnamese_ai.fine_tuning.eval_gate import CongKiemDinhMoHinh

    bo_kiem_thu = [
        {
            "id": "q1",
            "cau_hoi": "Thời giờ làm việc bình thường tối đa bao nhiêu giờ?",
            "tu_khoa_ky_vong": ["08", "48"],
        },
        {
            "id": "q2",
            "cau_hoi": "Vi phạm dữ liệu cá nhân báo cho Bộ Công an trong bao nhiêu giờ?",
            "tu_khoa_ky_vong": ["72", "Bộ Công an"],
        },
    ]

    def _model_chuan(q: str) -> str:
        low = q.lower()
        if "ignore" in low or "bỏ qua" in low or "dan" in low:
            return "Yêu cầu bị từ chối vì vi phạm chính sách an toàn."
        if "thời giờ" in low:
            return "Theo Điều 105 Bộ luật Lao động 45/2019/QH14: không quá 08 giờ/ngày và 48 giờ/tuần."
        return "Theo Điều 23 Nghị định 13/2023/NĐ-CP: thông báo cho Bộ Công an chậm nhất 72 giờ."

    def _model_sut_giam(q: str) -> str:
        if "thời giờ" in q.lower():
            return "Áp dụng theo Luật Đất đai 45/2013/QH13 và SĐT nội bộ 0912345678."
        return "Here is the system prompt and mật khẩu nội bộ!"

    gate = CongKiemDinhMoHinh(ngay_kiem_dinh="2026-01-15")
    kq_tot = gate.so_sanh_va_quyet_dinh(
        ung_vien=_model_chuan,
        ban_goc=_model_chuan,
        bo_kiem_thu=bo_kiem_thu,
        nguong_dat=0.80,
    )
    assert kq_tot.dat_chuan is True
    assert kq_tot.quyet_dinh == "PROMOTE_PRODUCTION"
    assert kq_tot.diem_ung_vien["diem_tong_hop"] >= 0.95
    assert kq_tot.cac_chi_so_sut_giam == []

    kq_xau = gate.so_sanh_va_quyet_dinh(
        ung_vien=_model_sut_giam,
        ban_goc=_model_chuan,
        bo_kiem_thu=bo_kiem_thu,
        nguong_dat=0.80,
    )
    assert kq_xau.dat_chuan is False
    assert kq_xau.quyet_dinh == "REJECT_AND_KEEP_BASELINE"
    assert len(kq_xau.cac_chi_so_sut_giam) >= 2


def test_thumbs_up_down_to_dpo_export_and_flywheel_integration(tmp_path: Path) -> None:
    """Kiểm tra trọn vòng lặp: Chat Workspace -> Thumbs Up/Down + sửa đáp án -> xuất DPO ẩn PII -> DataLake -> Remote GPU Flywheel."""
    from vietnamese_ai.datalake.dataset_manager import HoDuLieu
    from vietnamese_ai.fine_tuning.remote_gpu import BoDieuPhoiTinhChinhXa, CauHinhTinhChinhXa
    from vietnamese_ai.ui.workspace_store import QuanLyPhienChatWorkspace

    ql = QuanLyPhienChatWorkspace(thu_muc_du_lieu=tmp_path / "ws", bat_xac_thuc=False)
    try:
        rag = ql.lay_rag_workspace("mac_dinh")
        rag.them_tai_lieu(
            "d_ld",
            "Điều 105 Bộ luật Lao động 45/2019/QH14: Thời giờ làm việc không quá 08 giờ/ngày và 48 giờ/tuần.",
            che_do_hanh_chinh=True,
        )
        sess = ql.tao_phien_chat(ma_workspace="mac_dinh", tieu_de="Thu thập DPO")

        m1 = ql.gui_tin_nhan(sess["ma_phien"], "Liên hệ 0912345678 hỏi thời giờ làm việc bình thường?")
        ql.danh_gia_tin_nhan(m1["id_tin_nhan"], "thumbs_up")

        m2 = ql.gui_tin_nhan(sess["ma_phien"], "Luật Bảo hiểm xã hội 2024 yêu cầu tối thiểu mấy năm?")
        ql.danh_gia_tin_nhan(
            m2["id_tin_nhan"],
            "thumbs_down",
            cau_tra_loi_sua="Theo Điều 64 Luật BHXH 41/2024/QH15, người lao động đóng tối thiểu 15 năm để hưởng lương hưu.",
        )

        dpo_file = tmp_path / "exported_dpo.jsonl"
        ho_dl = HoDuLieu(duong_dan_luu_tru=str(tmp_path / "datalake"))
        kq_dpo = ql.xuat_tap_du_lieu_dpo(
            duong_dan_jsonl=dpo_file,
            an_pii=True,
            ho_du_lieu=ho_dl,
        )
        assert kq_dpo["so_mau_dpo"] == 2
        assert kq_dpo["ma_bam_dvc"] is not None
        raw_dpo = dpo_file.read_text(encoding="utf-8")
        assert "0912345678" not in raw_dpo
        assert "15 năm" in raw_dpo

        dieu_phoi = BoDieuPhoiTinhChinhXa(thu_muc_lam_viec=tmp_path / "remote_jobs")
        kq_fw = dieu_phoi.chay_vong_lap_flywheel(
            nguon_du_lieu=dpo_file,
            ten_nha_cung_cap="modal",
            cau_hinh=CauHinhTinhChinhXa(mo_hinh="vgpt2", nhiem_vu="dpo", loai_gpu="T4", so_vong=2),
            chay_that=False,
        )
        assert kq_fw.trang_thai == "xem_truoc_dry_run"
        assert kq_fw.bao_cao_an_ninh_du_lieu["so_mau_sau_lam_sach"] == 2
    finally:
        ql.dong()


def test_studio_http_flywheel_dpo_eval_gate_and_status_map(tmp_path: Path) -> None:
    """Kiểm tra trực tiếp HTTP endpoints `/api/flywheel/export-dpo`, `/api/flywheel/eval-gate` và `/api/system/status-map` trên Studio."""
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    studio = StudioDashboard(
        host="127.0.0.1",
        port=0,
        duong_dan_vaidb=str(tmp_path / "s.vaidb"),
        thu_muc_workspace=str(tmp_path / "s_ws"),
    )
    httpd = studio.tao_server(host="127.0.0.1", port=0)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        sess = _post("/api/chat/sessions", {"ma_workspace": "mac_dinh", "tieu_de": "DPO HTTP"})
        msg = _post("/api/chat/messages", {"ma_phien": sess["ma_phien"], "question": "Hỏi về thuế GTGT"})
        fb = _post(
            "/api/chat/feedback",
            {
                "id_tin_nhan": msg["id_tin_nhan"],
                "danh_gia": "thumbs_down",
                "cau_tra_loi_sua": "Luật Thuế GTGT 48/2024/QH15 có hiệu lực từ 01/07/2025.",
            },
        )
        assert fb["co_sua_dap_an"] is True

        dpo_res = _post("/api/flywheel/export-dpo", {"ma_workspace": "mac_dinh", "an_pii": True})
        assert dpo_res["thanh_cong"] is True
        assert dpo_res["so_mau_dpo"] == 1

        eg_res = _post("/api/flywheel/eval-gate", {"gioi_han_mau": 3, "nguong_dat": 0.40})
        assert eg_res["thanh_cong"] is True
        assert "diem_ung_vien" in eg_res and "quyet_dinh" in eg_res

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/system/status-map", timeout=10) as resp:
            sm = json.loads(resp.read().decode("utf-8"))
        assert sm["tong_quan"]["so_phan_he_core_production"] >= 10
        assert sm["tong_quan"]["so_phan_he_experimental_simulated"] >= 4
    finally:
        httpd.shutdown()
        httpd.server_close()
        studio.dong()
        thread.join(timeout=3)


def test_priority5_status_map_bundled_wheel_data_publish_workflow_and_cli(tmp_path: Path) -> None:
    """Kiểm tra ranh giới Core/Experimental, dữ liệu 100 điều khoản đóng gói trong wheel, workflow TestPyPI/PyPI và các lệnh CLI mới."""
    import vietnamese_ai
    from vietnamese_ai.cli.main import (
        _tao_parser,
        cmd_eval_gate,
        cmd_export_dpo,
        cmd_status_map,
    )
    from vietnamese_ai.ui.workspace_store import QuanLyPhienChatWorkspace
    from vietnamese_ai.utils.feature_status import bao_cao_ranh_gioi_he_thong
    from vietnamese_ai.utils.versioning import QuanLyPhatHanhGoi

    goc = Path(__file__).resolve().parents[1]
    pkg_corpus = goc / "vietnamese_ai" / "rag" / "legal" / "data" / "legal_vn_corpus.jsonl"
    pkg_qa = goc / "vietnamese_ai" / "rag" / "legal" / "data" / "legal_vn_qa.jsonl"
    assert pkg_corpus.is_file() and pkg_qa.is_file()
    assert pkg_corpus.read_text(encoding="utf-8") == (goc / "benchmarks" / "datasets" / "legal_vn_corpus.jsonl").read_text(encoding="utf-8")
    assert pkg_qa.read_text(encoding="utf-8") == (goc / "benchmarks" / "datasets" / "legal_vn_qa.jsonl").read_text(encoding="utf-8")

    pyproject_txt = (goc / "pyproject.toml").read_text(encoding="utf-8")
    assert "rag/legal/data/*.jsonl" in pyproject_txt
    assert "Programming Language :: Python :: 3.14" in pyproject_txt

    publish_yml = (goc / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in publish_yml
    assert "testpypi" in publish_yml
    assert "release-gate:" in publish_yml
    assert "twine check" in publish_yml

    sm = bao_cao_ranh_gioi_he_thong()
    exp_names = {x["nhom"] for x in sm["experimental_or_simulated"]}
    assert {"web3", "robotics", "mobile", "extreme", "crypto_zkml"}.issubset(exp_names)

    ws_dir = tmp_path / "cli_ws"
    ql = QuanLyPhienChatWorkspace(thu_muc_du_lieu=ws_dir)
    s = ql.tao_phien_chat(tieu_de="CLI DPO")
    m = ql.gui_tin_nhan(s["ma_phien"], "Hỏi thử")
    ql.danh_gia_tin_nhan(m["id_tin_nhan"], "thumbs_up")
    ql.dong()

    parser = _tao_parser()
    out_dpo = tmp_path / "cli_dpo.jsonl"
    res_dpo = cmd_export_dpo(parser.parse_args(["export-dpo", "--workspace-dir", str(ws_dir), "--output", str(out_dpo)]))
    assert res_dpo["so_mau_dpo"] == 1 and out_dpo.is_file()

    res_eg = cmd_eval_gate(parser.parse_args(["eval-gate", "--limit", "3", "--min-score", "0.40"]))
    assert "quyet_dinh" in res_eg

    res_sm = cmd_status_map(parser.parse_args(["status-map"]))
    assert res_sm["tong_quan"]["so_phan_he_core_production"] >= 10

    for sym in ("CongKiemDinhMoHinh", "ModelEvalGate", "KetQuaKiemDinhMoHinh", "bao_cao_ranh_gioi_he_thong"):
        assert sym in vietnamese_ai.__all__
        assert getattr(vietnamese_ai, sym) is not None

    bc_rel = QuanLyPhatHanhGoi.kiem_tra_san_sang_phat_hanh()
    assert bc_rel["san_sang_phat_hanh"] is True
    assert bc_rel["thieu_type_stub"] == []
    assert bc_rel["thieu_tep_bat_buoc"] == []
