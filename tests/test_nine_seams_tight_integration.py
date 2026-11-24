"""Kiểm thử tích hợp khép kín toàn bộ 9 mối nối liên tầng của Vietnamese AI Framework."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


def test_seam_1_studio_unified_workspace_upload_legal_and_ui(tmp_path: Path) -> None:
    """Mối nối 1: Studio dùng onnx_viet + TroLyPhapLy 100 điều luật + kéo thả thẳng vào Workspace .vaidb + UI động."""
    from vietnamese_ai.ui import StudioDashboard

    ws_dir = tmp_path / "studio_ws"
    studio = StudioDashboard(host="127.0.0.1", port=0, thu_muc_workspace=str(ws_dir))
    try:
        assert studio.rag.trich_xuat is not None
        assert getattr(studio.rag._embedder, "che_do", "") in (
            "semantic_synonym_cpu",
            "onnx_int8",
            "onnx_viet",
            "vi_semantic",
        )
        assert "197" not in studio.HTML_TEMPLATE
        assert "1008 / 1008" not in studio.HTML_TEMPLATE
        assert "loadStatusMap()" in studio.HTML_TEMPLATE
        assert "queryLegalGraph()" in studio.HTML_TEMPLATE
        assert "runStudioEvalGate()" in studio.HTML_TEMPLATE
        assert "exportWorkspaceDpo()" in studio.HTML_TEMPLATE
        assert "sendChatFeedback(" in studio.HTML_TEMPLATE

        studio.tao_workspace_studio({"ten": "Phòng Pháp chế", "ma_workspace": "phap_che"})
        noi_dung_hd = (
            "Điều 1. Quy định nội bộ phòng pháp chế: Thời hạn bảo hành thiết bị là 24 tháng.\n"
            "Căn cứ Luật Đất đai 45/2013/QH13 và Nghị định 123/2020/NĐ-CP."
        )
        up_res = studio.tai_len_tep_rag(
            {
                "filename": "quy_dinh_phap_che.txt",
                "content_base64": base64.b64encode(noi_dung_hd.encode("utf-8")).decode("ascii"),
                "che_do_hanh_chinh": True,
                "ma_workspace": "phap_che",
            }
        )
        assert up_res["thanh_cong"] is True
        assert up_res["ma_workspace"] == "phap_che"
        assert "do_thi_tham_chieu" in up_res
        assert "45/2013/QH13" in up_res["do_thi_tham_chieu"]

        sess_pc = studio.tao_phien_chat_studio({"ma_workspace": "phap_che", "tieu_de": "Hỏi phòng pháp chế"})
        msg_pc = studio.gui_tin_nhan_chat_studio(
            {"ma_phien": sess_pc["ma_phien"], "question": "Thời hạn bảo hành thiết bị phòng pháp chế là bao lâu?"}
        )
        assert msg_pc["thanh_cong"] is True
        assert "24 tháng" in msg_pc["tra_loi"]

        sess_md = studio.tao_phien_chat_studio({"ma_workspace": "mac_dinh", "tieu_de": "Hỏi mặc định"})
        msg_md = studio.gui_tin_nhan_chat_studio(
            {"ma_phien": sess_md["ma_phien"], "question": "Vi phạm bảo vệ dữ liệu cá nhân thông báo trong bao nhiêu giờ?"}
        )
        assert msg_md["thanh_cong"] is True
        assert "72" in msg_md["tra_loi"]

        eg_res = studio.chay_eval_gate_studio({"gioi_han_mau": 4, "nguong_dat": 0.45})
        assert eg_res["thanh_cong"] is True
        assert eg_res["dat_chuan"] is True
    finally:
        studio.dong()


def test_seam_2_workspace_dpo_to_train_cloud_and_4_axis_eval_gate(tmp_path: Path) -> None:
    """Mối nối 2: Phản hồi DPO từ Workspace -> `vai train-cloud --from-workspace` & `kiem_dinh_eval_gate` 4 trục."""
    from vietnamese_ai.cli.main import _tao_parser, cmd_train_cloud
    from vietnamese_ai.fine_tuning.auto_finetuner import BoTinhChinhMoHinh
    from vietnamese_ai.fine_tuning.remote_gpu import BoDieuPhoiTinhChinhXa, KetQuaTinhChinhXa
    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy
    from vietnamese_ai.ui.workspace_store import QuanLyPhienChatWorkspace
    from vietnamese_ai.utils.io_utils import LuuTai

    ws_dir = tmp_path / "ws_dpo"
    ql = QuanLyPhienChatWorkspace(thu_muc_du_lieu=ws_dir, bat_xac_thuc=False)
    try:
        sess = ql.tao_phien_chat(ma_workspace="mac_dinh", tieu_de="Thu thập DPO")
        luot = ql.ghi_luot_hoi_thoai_kenh(
            ma_phien=sess["ma_phien"],
            cau_hoi="Khách hàng SĐT 0912345678 hỏi thời hạn thông báo vi phạm dữ liệu cá nhân?",
            tra_loi="Trả lời cũ chưa rõ ràng.",
        )
        ql.danh_gia_tin_nhan(
            id_tin_nhan=luot["id_tin_nhan"],
            danh_gia="thumbs_down",
            cau_tra_loi_sua="Theo Điều 23 Nghị định 13/2023/NĐ-CP phải thông báo cho Bộ Công an trong vòng 72 giờ.",
        )
    finally:
        ql.dong()

    parser = _tao_parser()
    args = parser.parse_args(
        [
            "train-cloud",
            "--from-workspace",
            str(ws_dir),
            "--provider",
            "modal",
            "--eval-4-axis",
            "--output",
            str(tmp_path / "remote_jobs"),
        ]
    )
    res_cli = cmd_train_cloud(args)
    assert res_cli["trang_thai"] == "xem_truoc_dry_run"
    assert res_cli["bao_cao_an_ninh_du_lieu"]["so_mau_sau_lam_sach"] >= 1
    assert res_cli["duong_dan_script"]

    gguf_path = tmp_path / "model_eval.gguf"
    BoTinhChinhMoHinh(mo_hinh="vgpt2", che_do_offline=True).xuat_gguf(str(gguf_path), quantization="q8_0")
    LuuTai.ky_file(str(gguf_path))

    dieu_phoi = BoDieuPhoiTinhChinhXa(thu_muc_lam_viec=tmp_path / "eval_jobs")
    kq_gia = KetQuaTinhChinhXa(
        nha_cung_cap="local",
        ma_tac_vu="local_eval4",
        trang_thai="hoan_thanh",
        duong_dan_gguf=str(gguf_path),
    )
    tro_ly = TroLyPhapLy(nap_kho_mo=True, dung_onnx_semantic=True)
    eval_4 = dieu_phoi.kiem_dinh_eval_gate(
        kq_gia,
        diem_toi_thieu=0.45,
        su_dung_eval_gate_4_truc=True,
        gioi_han_mau_4_truc=4,
        tro_ly_eval=tro_ly,
    )
    assert eval_4["che_do"] == "eval_gate_4_truc"
    assert eval_4["hmac_hop_le"] is True
    assert eval_4["gguf_hop_le"] is True
    assert eval_4["dat_chuan"] is True
    assert "bao_cao_4_truc" in eval_4


def test_seam_3_edge_engine_opens_vaidb_and_cli_run_prints_legal_graph(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mối nối 3: `DongCoSuyLuanEdge` mở thẳng `.vaidb` + `onnx_viet` và `vai run --rag .vaidb --admin` in hiệu lực & đồ thị."""
    from vietnamese_ai.cli.main import _tao_parser, cmd_run
    from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge
    from vietnamese_ai.rag.rag_pipeline import RAGPipeline

    db_path = tmp_path / "edge_kb.vaidb"
    rag = RAGPipeline(
        backend="vaidb",
        duong_dan_vaidb=str(db_path),
        che_do_nhung="onnx_viet",
        che_do_rerank="onnx_viet",
        che_do_hanh_chinh=True,
    )
    rag.them_tai_lieu(
        "hop_dong_dat",
        "Điều 5. Thủ tục đất đai căn cứ Luật Đất đai 45/2013/QH13 và hóa đơn điện tử theo Nghị định 123/2020/NĐ-CP.",
        metadata={"so_hieu": "45/2013/QH13"},
    )
    rag.dong()

    engine = DongCoSuyLuanEdge(do_dai_toi_da=24)
    nap = engine.nap_tai_lieu_rag(db_path, che_do_hanh_chinh=True)
    assert nap["backend"] == "vaidb"
    assert nap["che_do_nhung"] == "onnx_viet"
    assert nap["so_chunks"] >= 1
    kq = engine.hoi_kem_rag("Thủ tục đất đai và hóa đơn điện tử căn cứ văn bản nào?")
    assert kq["an_toan"] is True
    assert kq["trich_dan_phap_ly"] is not None
    assert len(kq["trich_dan_phap_ly"].get("canh_bao_hieu_luc") or []) >= 1
    engine.rag_pipeline.dong()

    parser = _tao_parser()
    run_args = parser.parse_args(
        [
            "run",
            "vgpt2",
            "--rag",
            str(db_path),
            "--admin",
            "--prompt",
            "Thủ tục đất đai căn cứ văn bản nào?",
            "--max-tokens",
            "24",
        ]
    )
    kq_run = cmd_run(run_args)
    out = capsys.readouterr().out
    assert kq_run["trich_dan_phap_ly"] is not None
    assert "Cảnh báo hiệu lực:" in out


def test_seam_4_open_corpus_parity_in_solution_cli_and_default_connector() -> None:
    """Mối nối 4: `vai solution phap-ly` và `KetNoiZaloOA` mặc định đều nạp sẵn 100 điều khoản pháp lý mở."""
    from vietnamese_ai.cli.main import _tao_parser, cmd_solution
    from vietnamese_ai.connectors.base import TinNhanKenh
    from vietnamese_ai.connectors.zalo_oa import KetNoiZaloOA

    parser = _tao_parser()
    sol_res = cmd_solution(
        parser.parse_args(
            [
                "solution",
                "phap-ly",
                "--query",
                "Vi phạm bảo vệ dữ liệu cá nhân phải thông báo Bộ Công an trong bao nhiêu giờ?",
            ]
        )
    )
    assert "72" in sol_res["tra_loi"]

    zalo = KetNoiZaloOA()
    ph = zalo.xu_ly_tin_nhan(
        TinNhanKenh(
            kenh="zalo_oa",
            nguoi_gui_id="u_legal_100",
            noi_dung="Thời hạn thông báo vi phạm bảo vệ dữ liệu cá nhân theo Nghị định 13/2023/NĐ-CP là bao lâu?",
        )
    )
    assert ph.an_toan is True
    assert "72" in ph.noi_dung


def test_seam_5_fastapi_server_dispatches_solutions_and_edge() -> None:
    """Mối nối 5: `FastAPIServer` hỗ trợ trực tiếp `TroLyPhapLy`, `TongDaiCSKH` và `DongCoSuyLuanEdge`."""
    pytest.importorskip("fastapi")
    from vietnamese_ai.api.fastapi_server import ChatCompletionRequest, ChatMessage, FastAPIServer
    from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge
    from vietnamese_ai.solutions.cskh import TongDaiCSKH
    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

    req_law = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Thời giờ làm việc bình thường tối đa là bao nhiêu?")]
    )
    srv_law = FastAPIServer(bo_xu_ly=TroLyPhapLy())
    ans_law = srv_law._xu_ly_logic_truc_tiep(req_law)
    assert "08 giờ" in ans_law or "8 giờ" in ans_law or "48" in ans_law

    req_cskh = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Kiểm tra đơn hàng DH1001")])
    srv_cskh = FastAPIServer(bo_xu_ly=TongDaiCSKH())
    ans_cskh = srv_cskh._xu_ly_logic_truc_tiep(req_cskh)
    assert "DH1001" in ans_cskh

    req_edge = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Xin chào tiếng Việt")], max_tokens=16)
    srv_edge = FastAPIServer(bo_xu_ly=DongCoSuyLuanEdge(do_dai_toi_da=16))
    ans_edge = srv_edge._xu_ly_logic_truc_tiep(req_edge)
    assert isinstance(ans_edge, str) and len(ans_edge.strip()) > 0


def test_seam_6_quy_trinh_dau_cuoi_new_enterprise_bridges(tmp_path: Path) -> None:
    """Mối nối 6: `QuyTrinhDauCuoi` nối đủ 4 cầu nối mới (Hợp đồng, Eval Gate, MCP Server, Webhook Kênh)."""
    from vietnamese_ai.core.end_to_end import QuyTrinhDauCuoi

    wf = QuyTrinhDauCuoi(ten_du_an="kiem_thu_9_moi_noi", thu_muc_lam_viec=str(tmp_path / "e2e_proj"))
    out_docx = tmp_path / "redline.docx"
    kq_hd = wf.ra_soat_hop_dong(
        hop_dong_doi_tac="Điều 1. Phạt vi phạm 20% giá trị hợp đồng, căn cứ Luật Đất đai 45/2013/QH13.",
        ngay_tra_cuu="2026-01-15",
        xuat_docx=out_docx,
    )
    assert kq_hd["xep_loai_rui_ro"] in ("CAO", "TRUNG BÌNH")
    assert out_docx.is_file()

    kq_eg = wf.kiem_dinh_mo_hinh(gioi_han_mau=4, nguong_dat=0.45)
    assert kq_eg["dat_chuan"] is True

    mcp = wf.tao_mcp_server()
    try:
        tools_resp = mcp.xu_ly_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        ten_tools = [t["name"] for t in tools_resp["result"]["tools"]]
        assert "tra_cuu_phap_ly" in ten_tools
        assert "ra_soat_hop_dong" in ten_tools
    finally:
        mcp.dong()

    wh = wf.tao_webhook_kenh(kenh="both", port=0)
    assert wh.ket_noi_zalo is not None
    assert wh.ket_noi_telegram is not None


def test_seam_7_connector_workspace_persistence_hopdong_and_mcp_plugin_bridge(tmp_path: Path) -> None:
    """Mối nối 7: Connector lưu hội thoại vào SQLite Workspace + lệnh `/hopdong`, và MCP tự phơi bày V-Plugin 2.0."""
    from vietnamese_ai.agents.mcp_transport import MayChuMCPThucTe
    from vietnamese_ai.connectors.base import TinNhanKenh
    from vietnamese_ai.connectors.telegram_bot import KetNoiTelegram
    from vietnamese_ai.plugin.plugin_manager import PluginCongCu, QuanLyPlugin
    from vietnamese_ai.ui.workspace_store import QuanLyPhienChatWorkspace

    ws_dir = tmp_path / "connector_ws"
    ql_ws = QuanLyPhienChatWorkspace(thu_muc_du_lieu=ws_dir, bat_xac_thuc=False)
    try:
        bot = KetNoiTelegram(quan_ly_workspace=ql_ws, ma_workspace="kenh_bot")
        ph1 = bot.xu_ly_tin_nhan(
            TinNhanKenh(
                kenh="telegram",
                nguoi_gui_id="tg_9001",
                ten_nguoi_gui="Anh Minh",
                noi_dung="/hopdong Điều 1. Phạt vi phạm 20% và miễn trừ toàn bộ trách nhiệm. Căn cứ Luật Đất đai 45/2013/QH13.",
            )
        )
        assert ph1.an_toan is True
        assert "Rà soát nhanh hợp đồng" in ph1.noi_dung
    finally:
        ql_ws.dong()

    ql_reopen = QuanLyPhienChatWorkspace(thu_muc_du_lieu=ws_dir, bat_xac_thuc=False)
    try:
        lich_su = ql_reopen.lay_lich_su_phien("telegram:tg_9001")
        assert len(lich_su) == 2
        assert lich_su[0]["vai_tro"] == "user"
        assert lich_su[1]["vai_tro"] == "assistant"
    finally:
        ql_reopen.dong()

    class _PluginTinhThue(PluginCongCu):
        ten = "plugin_thue_test"
        phien_ban = "1.0.0"
        mo_ta = "Plugin tính thuế thử nghiệm"

        def dang_ky_cong_cu(self) -> List[Dict[str, Any]]:
            return [
                {
                    "ten": "tinh_vat_nhanh",
                    "mo_ta": "Tính thuế GTGT 10%",
                    "tham_so": {"so_tien": "Số tiền gốc"},
                }
            ]

        def thuc_thi_cong_cu(self, ten_cong_cu: str, **tham_so: Any) -> Dict[str, Any]:
            so_tien = float(tham_so.get("so_tien", 0))
            return {"vat": round(so_tien * 0.1, 2)}

    ql_plugin = QuanLyPlugin()
    assert ql_plugin.dang_ky_plugin(_PluginTinhThue(), ten_module="plugin_thue_test") is True
    mcp = MayChuMCPThucTe(nap_kho_phap_ly_mo=False, quan_ly_plugin=ql_plugin)
    try:
        call_resp = mcp.xu_ly_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "tinh_vat_nhanh", "arguments": {"so_tien": "1000000"}},
            }
        )
        payload = json.loads(call_resp["result"]["content"][0]["text"])
        assert payload["vat"] == 100000.0
    finally:
        mcp.dong()


def test_seam_8_ui_subpackage_exports_and_rag_luu_tai_vaidb(tmp_path: Path) -> None:
    """Mối nối 8: `vietnamese_ai.ui` export đầy đủ & `RAGPipeline.luu/tai` tự hỗ trợ tệp `.vaidb`."""
    from vietnamese_ai.rag.rag_pipeline import RAGPipeline
    from vietnamese_ai.ui import (
        MayChuStudioDashboard,
        QuanLyPhienChatWorkspace,
        StudioDashboard,
        StudioWorkspaceManager,
        UngDungWeb,
    )

    assert MayChuStudioDashboard is StudioDashboard
    assert StudioWorkspaceManager is QuanLyPhienChatWorkspace
    assert UngDungWeb is not None

    rag_mem = RAGPipeline(che_do_nhung="onnx_viet", che_do_rerank="onnx_viet", che_do_hanh_chinh=True)
    rag_mem.them_tai_lieu("doc_1", "Chính sách nghỉ phép năm cho nhân viên chính thức là 14 ngày làm việc.")
    snap_path = tmp_path / "snapshot.vaidb"
    rag_mem.luu(str(snap_path))
    assert snap_path.is_file()

    rag_loaded = RAGPipeline(che_do_nhung="onnx_viet", che_do_rerank="onnx_viet", che_do_hanh_chinh=True)
    rag_loaded.tai(str(snap_path))
    try:
        assert rag_loaded.backend == "vaidb"
        kq = rag_loaded.hoi("Nhân viên được nghỉ phép năm bao nhiêu ngày?")
        assert "14 ngày" in kq["tra_loi"]
    finally:
        rag_loaded.dong()


def test_seam_9_cli_info_demo_and_new_flags(capsys: pytest.CaptureFixture[str]) -> None:
    """Mối nối 9: `vai info`, `vai demo legal-rag` (kèm đồ thị & hợp đồng) và các cờ CLI mới."""
    from vietnamese_ai.cli.demo import chay_demo_phap_ly
    from vietnamese_ai.cli.main import _tao_parser, cmd_info

    cmd_info()
    out_info = capsys.readouterr().out
    assert "contract-review" in out_info
    assert "legal-graph" in out_info
    assert "mcp-server" in out_info
    assert "eval-gate" in out_info
    assert "status-map" in out_info

    demo_res = chay_demo_phap_ly(in_ra=False)
    assert demo_res["tong_dieu_khoan_kho_mo"] == 100
    assert "do_thi_phap_ly" in demo_res
    assert "ra_soat_hop_dong" in demo_res
    assert demo_res["ra_soat_hop_dong"]["xep_loai_rui_ro"] in ("CAO", "TRUNG BÌNH")

    parser = _tao_parser()
    st_args = parser.parse_args(["studio", "--workspace-dir", "/tmp/ws", "--auth"])
    assert st_args.workspace_dir == "/tmp/ws"
    assert st_args.auth is True

    mcp_args = parser.parse_args(["mcp-server", "--plugin-dir", "/tmp/plugins"])
    assert mcp_args.plugin_dir == "/tmp/plugins"

    conn_args = parser.parse_args(["connector", "zalo", "--workspace-dir", "/tmp/ws", "--dry-run"])
    assert conn_args.workspace_dir == "/tmp/ws"
