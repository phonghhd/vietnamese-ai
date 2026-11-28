"""Kiểm thử tích hợp cho Ưu tiên 2 (So sánh Hợp đồng Redline .docx, Chat Đa phiên Workspace + RBAC) và Ưu tiên 3 (MCP Server stdio/SSE, Zalo OA & Telegram Connectors)."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest


def test_contract_review_diff_risks_and_docx_export(tmp_path: Path) -> None:
    """Kiểm tra so sánh từng điều khoản hợp đồng, phát hiện rủi ro/luật hết hiệu lực và xuất tệp Word `.docx` hợp lệ."""
    from vietnamese_ai.rag.legal.contract_review import BoSoSanhRaSoatHopDong
    from vietnamese_ai.rag.legal.document_parser import BoDocTaiLieuPhapLy

    hop_dong_chuan = (
        "HỢP ĐỒNG DỊCH VỤ PHẦN MỀM\n"
        "Điều 1. Đối tượng hợp đồng: Bên B cung cấp phần mềm quản trị cho Bên A.\n"
        "Điều 2. Giá trị và thanh toán: Giá trị hợp đồng 100.000.000 VNĐ, thanh toán trong 15 ngày.\n"
        "Điều 3. Phạt vi phạm: Mức phạt tối đa 8% giá trị phần nghĩa vụ bị vi phạm theo Luật Thương mại 36/2005/QH11.\n"
        "Điều 4. Bảo mật dữ liệu cá nhân: Hai bên tuân thủ Nghị định 13/2023/NĐ-CP.\n"
        "Điều 5. Bất khả kháng và giải quyết tranh chấp: Thương lượng hoặc Tòa án nhân dân có thẩm quyền."
    )
    hop_dong_doi_tac = (
        "HỢP ĐỒNG DỊCH VỤ PHẦN MỀM\n"
        "Căn cứ Luật Đất đai 45/2013/QH13 và Bộ luật Dân sự 91/2015/QH13.\n"
        "Điều 1. Đối tượng hợp đồng: Bên B cung cấp phần mềm và vận hành máy chủ cho Bên A.\n"
        "Điều 2. Giá trị và thanh toán: Lãi suất chậm thanh toán là 28%/năm.\n"
        "Điều 3. Phạt vi phạm: Bên B chịu mức phạt vi phạm 20% giá trị hợp đồng và miễn trừ toàn bộ trách nhiệm cho Bên A.\n"
        "Điều 6. Chấm dứt: Bên A có quyền đơn phương chấm dứt ngay lập tức không cần thông báo trước và tự động gia hạn mỗi năm."
    )

    engine = BoSoSanhRaSoatHopDong()
    kq = engine.so_sanh_va_ra_soat(
        hop_dong_doi_tac=hop_dong_doi_tac,
        hop_dong_chuan=hop_dong_chuan,
        ngay_tra_cuu="2026-01-15",
    )

    ma_quy_tac = {cb["ma_quy_tac"] for cb in kq["danh_sach_canh_bao"]}
    assert "phat_vi_pham_vuot_tran_8pct" in ma_quy_tac
    assert "lai_suat_vuot_tran_20pct" in ma_quy_tac
    assert "mien_tru_hoac_boi_thuong_khong_gioi_han" in ma_quy_tac
    assert "cham_dut_mot_chieu_khong_bao_truoc" in ma_quy_tac
    assert "vien_dan_luat_het_hieu_luc" in ma_quy_tac
    assert kq["diem_an_toan_phap_ly"] < 50
    assert kq["xep_loai_rui_ro"] == "CAO"

    tk_diff = kq["thong_ke_so_sanh"]
    assert tk_diff["sua_doi"] >= 2
    assert tk_diff["thieu_dieu_khoan"] >= 2
    assert tk_diff["them_moi"] >= 1

    md = engine.xuat_bao_cao_markdown(kq)
    assert "REDLINE LEGAL AUDIT" in md
    assert "45/2013/QH13" in md

    docx_path = tmp_path / "bao_cao_redline.docx"
    engine.xuat_bao_cao_docx(kq, docx_path)
    assert docx_path.is_file() and zipfile.is_zipfile(docx_path)
    doc_doc_lai = BoDocTaiLieuPhapLy().doc_tep(docx_path)
    assert "BÁO CÁO RÀ SOÁT" in doc_doc_lai["van_ban"]
    assert "20%" in doc_doc_lai["van_ban"]


def test_workspace_multi_session_chat_rbac_and_persistence(tmp_path: Path) -> None:
    """Kiểm tra Workspace theo phòng ban, phân quyền RBAC (`HeThongXacThuc`), audit log và lịch sử chat SQLite bền vững."""
    from vietnamese_ai.ui.workspace_store import QuanLyPhienChatWorkspace

    ws_dir = tmp_path / "workspaces"
    ql = QuanLyPhienChatWorkspace(thu_muc_du_lieu=ws_dir, bat_xac_thuc=True)
    try:
        assert ql.xac_thuc is not None
        ql.xac_thuc.dang_ky("truong_phap_che", "MatKhauAdmin123!", vai_tro="admin")
        ql.xac_thuc.dang_ky("nhan_vien_xem", "MatKhauView123!", vai_tro="viewer")
        tok_admin = ql.xac_thuc.dang_nhap("truong_phap_che", "MatKhauAdmin123!")
        tok_viewer = ql.xac_thuc.dang_nhap("nhan_vien_xem", "MatKhauView123!")

        with pytest.raises(PermissionError):
            ql.tao_workspace(ten="Phòng Pháp chế", ma_workspace="phap_che", token=tok_viewer)

        ws_pc = ql.tao_workspace(ten="Phòng Pháp chế", ma_workspace="phap_che", token=tok_admin)
        assert ws_pc["ma_workspace"] == "phap_che"

        rag_pc = ql.lay_rag_workspace("phap_che")
        rag_pc.them_tai_lieu(
            "noi_quy_pc",
            "Điều 105 Bộ luật Lao động 45/2019/QH14: Thời giờ làm việc bình thường không quá 08 giờ/ngày và 48 giờ/tuần.",
            che_do_hanh_chinh=True,
        )

        phien = ql.tao_phien_chat(ma_workspace="phap_che", tieu_de="Tra cứu lao động", token=tok_viewer)
        kq_chat = ql.gui_tin_nhan(phien["ma_phien"], "Thời giờ làm việc bình thường tối đa bao nhiêu giờ?", token=tok_viewer)
        assert "08" in kq_chat["tra_loi"] or "48" in kq_chat["tra_loi"]
        assert len(kq_chat["trich_dan_noi_bat"]) >= 1

        fb = ql.danh_gia_tin_nhan(kq_chat["id_tin_nhan"], "thumbs_up", token=tok_viewer)
        assert fb["cap_nhat"] is True
        assert ql.nhat_ky is not None and ql.nhat_ky.thong_ke()["tong_ban_ghi"] >= 2
    finally:
        ql.dong()

    ql2 = QuanLyPhienChatWorkspace(thu_muc_du_lieu=ws_dir, bat_xac_thuc=False)
    try:
        ds_ws = ql2.danh_sach_workspace()
        assert any(w["ma_workspace"] == "phap_che" for w in ds_ws)
        lich_su = ql2.lay_lich_su_phien(phien["ma_phien"])
        assert len(lich_su) == 2
        assert lich_su[0]["vai_tro"] == "user"
        assert lich_su[1]["vai_tro"] == "assistant"
        assert lich_su[1]["danh_gia"] == "thumbs_up"
    finally:
        ql2.dong()


def test_studio_workbench_contract_and_chat_http(tmp_path: Path) -> None:
    """Kiểm tra trực tiếp các HTTP endpoints Tab 6 (Contract Diff + Workspace Multi-session Chat) trên `StudioDashboard`."""
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    studio = StudioDashboard(
        host="127.0.0.1",
        port=0,
        duong_dan_vaidb=str(tmp_path / "studio.vaidb"),
        thu_muc_workspace=str(tmp_path / "studio_ws"),
    )
    studio.rag.them_tai_lieu(
        "luat_dlcn",
        "Điều 23 Nghị định 13/2023/NĐ-CP: Thông báo vi phạm dữ liệu cá nhân cho Bộ Công an trong 72 giờ.",
        che_do_hanh_chinh=True,
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
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(path: str) -> Any:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
            return resp.read().decode("utf-8")

    try:
        html_ui = _get("/")
        assert "tab-workbench" in html_ui
        assert "runContractReview" in html_ui
        assert "sendWorkspaceChat" in html_ui

        kq_hd = _post(
            "/api/legal/contract-review",
            {
                "hop_dong_chuan": "Điều 1. Phạt vi phạm tối đa 8%.",
                "hop_dong_doi_tac": "Điều 1. Phạt vi phạm 25% và viện dẫn Luật 45/2013/QH13.",
                "ngay_tra_cuu": "2026-01-15",
                "xuat_docx": True,
            },
        )
        assert kq_hd["thanh_cong"] is True
        assert kq_hd["so_canh_bao"] >= 2
        docx_bytes = base64.b64decode(kq_hd["bao_cao_docx_base64"])
        assert zipfile.is_zipfile(io.BytesIO(docx_bytes))

        ws_res = _post("/api/workspaces", {"ten": "Kế toán", "ma_workspace": "ke_toan"})
        assert ws_res["thanh_cong"] is True
        ws_list = json.loads(_get("/api/workspaces"))
        assert any(w["ma_workspace"] == "ke_toan" for w in ws_list["workspaces"])

        sess = _post("/api/chat/sessions", {"ma_workspace": "mac_dinh", "tieu_de": "Phiên HTTP"})
        assert sess["thanh_cong"] is True
        msg = _post("/api/chat/messages", {"ma_phien": sess["ma_phien"], "question": "Thông báo vi phạm dữ liệu cá nhân trong mấy giờ?"})
        assert msg["thanh_cong"] is True
        assert "72" in msg["tra_loi"]

        hist = json.loads(_get(f"/api/chat/messages?session_id={sess['ma_phien']}"))
        assert len(hist["messages"]) == 2
        fb = _post("/api/chat/feedback", {"id_tin_nhan": msg["id_tin_nhan"], "danh_gia": "thumbs_up"})
        assert fb["cap_nhat"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        studio.dong()
        thread.join(timeout=3)


def test_mcp_server_stdio_sse_and_ide_configs(tmp_path: Path) -> None:
    """Kiểm tra MCP Server thật trên cả 2 giao vận `stdio` (newline & Content-Length) và `HTTP/SSE` cùng cấu hình IDE."""
    from vietnamese_ai.agents.mcp_transport import MayChuMCPThucTe

    mcp = MayChuMCPThucTe(duong_dan_vaidb=tmp_path / "mcp.vaidb", nap_kho_phap_ly_mo=False)
    try:
        in_lines = "\n".join(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
                json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {
                            "name": "kiem_tra_hieu_luc",
                            "arguments": {"so_hieu_hoac_van_ban": "45/2013/QH13", "ngay_tra_cuu": "2026-01-15"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "tools/call",
                        "params": {
                            "name": "do_thi_tham_chieu_luat",
                            "arguments": {"so_hieu": "45/2013/QH13", "ngay_tra_cuu": "2026-01-15"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 6,
                        "method": "tools/call",
                        "params": {
                            "name": "quet_bao_mat_du_lieu",
                            "arguments": {"van_ban": "Khách hàng SĐT 0912345678 CCCD 001095012345"},
                        },
                    }
                ),
                json.dumps({"jsonrpc": "2.0", "id": 7, "method": "resources/read", "params": {"uri": "vai://system/info"}}),
            ]
        ) + "\n"
        out_buf = io.StringIO()
        n_proc = mcp.chay_stdio(io.StringIO(in_lines), out_buf)
        assert n_proc == 8

        resps = [json.loads(line) for line in out_buf.getvalue().strip().splitlines() if line.strip()]
        assert len(resps) == 7
        assert resps[0]["result"]["serverInfo"]["name"] == "vietnamese-ai-mcp"
        tool_names = {t["name"] for t in resps[2]["result"]["tools"]}
        assert {
            "tra_cuu_phap_ly",
            "kiem_tra_hieu_luc",
            "do_thi_tham_chieu_luat",
            "ra_soat_hop_dong",
            "quet_bao_mat_du_lieu",
            "hoi_edge_llm",
        }.issubset(tool_names)

        hl_json = json.loads(resps[3]["result"]["content"][0]["text"])
        assert hl_json["trang_thai"] == "het_hieu_luc"
        dt_json = json.loads(resps[4]["result"]["content"][0]["text"])
        assert any(x["so_hieu"] == "31/2024/QH15" for x in dt_json["van_ban_thay_the_hien_hanh"])
        sec_json = json.loads(resps[5]["result"]["content"][0]["text"])
        assert "0912345678" not in sec_json["van_ban_da_an_pii"]

        framed_body = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"})
        framed_in = f"Content-Length: {len(framed_body.encode('utf-8'))}\r\n\r\n{framed_body}"
        framed_out = io.StringIO()
        mcp.chay_stdio(io.StringIO(framed_in), framed_out)
        assert "Content-Length:" in framed_out.getvalue()

        httpd = mcp.tao_http_server(host="127.0.0.1", port=0)
        port = httpd.server_port
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                h_data = json.loads(resp.read().decode("utf-8"))
            assert h_data["so_cong_cu"] == 6

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/sse", timeout=5) as resp:
                sse_txt = resp.read().decode("utf-8")
            assert "event: endpoint" in sse_txt and "/messages" in sse_txt

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/messages",
                data=json.dumps({"jsonrpc": "2.0", "id": 10, "method": "tools/list"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                rpc_http = json.loads(resp.read().decode("utf-8"))
            assert len(rpc_http["result"]["tools"]) == 6
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=3)

        assert "mcpServers" in MayChuMCPThucTe.xuat_cau_hinh_ide("cursor")
        assert "servers" in MayChuMCPThucTe.xuat_cau_hinh_ide("vscode")
    finally:
        mcp.dong()


def test_zalo_oa_telegram_webhook_server_cli_and_exports(tmp_path: Path) -> None:
    """Kiểm tra Zalo OA MAC verification, Telegram Secret Token & chunking, Webhook Server sống, CLI và Exports."""
    import vietnamese_ai
    from vietnamese_ai.cli.main import (
        _tao_parser,
        cmd_connector,
        cmd_contract_review,
        cmd_mcp_server,
    )
    from vietnamese_ai.connectors.telegram_bot import KetNoiTelegram
    from vietnamese_ai.connectors.webhook_server import MayChuWebhookKenh
    from vietnamese_ai.connectors.zalo_oa import KetNoiZaloOA
    from vietnamese_ai.solutions.cskh import TongDaiCSKH
    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy
    from vietnamese_ai.utils.versioning import QuanLyPhatHanhGoi

    zalo = KetNoiZaloOA(
        bo_xu_ly=TroLyPhapLy(),
        app_id="123456789",
        oa_secret_key="bi_mat_zalo_oa",
        che_do_gui_that=False,
    )
    telegram = KetNoiTelegram(
        bo_xu_ly=TongDaiCSKH(),
        bot_token="fake-token",
        webhook_secret="bi_mat_telegram",
        che_do_gui_that=False,
    )

    chunks = KetNoiTelegram.chia_doan_tin_nhan("Dòng nội dung pháp lý dài.\n" * 300)
    assert len(chunks) >= 2 and all(len(c) <= 4000 for c in chunks)

    srv = MayChuWebhookKenh(ket_noi_zalo=zalo, ket_noi_telegram=telegram, host="127.0.0.1", port=0)
    httpd = srv.tao_server(host="127.0.0.1", port=0)
    port = httpd.server_port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        zalo_ts = str(int(time.time() * 1000))
        zalo_payload = {
            "event_name": "user_send_text",
            "timestamp": zalo_ts,
            "sender": {"id": "zalo_user_01"},
            "message": {"msg_id": "m1", "text": "/hieuluc 45/2013/QH13"},
        }
        zalo_raw = json.dumps(zalo_payload, ensure_ascii=False).encode("utf-8")
        good_mac = zalo.tao_chu_ky_mac(zalo_raw, zalo_ts)

        req_ok = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhook/zalo",
            data=zalo_raw,
            headers={"Content-Type": "application/json", "X-ZEvent-Signature": good_mac},
            method="POST",
        )
        with urllib.request.urlopen(req_ok, timeout=5) as resp:
            z_res = json.loads(resp.read().decode("utf-8"))
        assert z_res["thanh_cong"] is True
        assert "het_hieu_luc" in z_res["phan_hoi"]["noi_dung"]

        req_bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhook/zalo",
            data=zalo_raw,
            headers={"Content-Type": "application/json", "X-ZEvent-Signature": "mac=deadbeef"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_z:
            urllib.request.urlopen(req_bad, timeout=5)
        assert exc_z.value.code == 401

        tg_payload = {
            "update_id": 1001,
            "message": {
                "message_id": 55,
                "from": {"id": 9001, "username": "khach_hang_a"},
                "chat": {"id": 9001, "type": "private"},
                "text": "Kiểm tra giúp tôi đơn hàng DH1001",
            },
        }
        req_tg = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhook/telegram",
            data=json.dumps(tg_payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Telegram-Bot-Api-Secret-Token": "bi_mat_telegram"},
            method="POST",
        )
        with urllib.request.urlopen(req_tg, timeout=5) as resp:
            tg_res = json.loads(resp.read().decode("utf-8"))
        assert tg_res["thanh_cong"] is True
        assert "DH1001" in tg_res["phan_hoi"]["noi_dung"]

        req_tg_bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhook/telegram",
            data=json.dumps(tg_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Telegram-Bot-Api-Secret-Token": "sai_secret"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_t:
            urllib.request.urlopen(req_tg_bad, timeout=5)
        assert exc_t.value.code == 401
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)

    parser = _tao_parser()
    partner_txt = tmp_path / "partner.txt"
    partner_txt.write_text("Điều 1. Phạt vi phạm 18% và căn cứ Luật 45/2013/QH13.", encoding="utf-8")
    out_docx = tmp_path / "cli_redline.docx"
    kq_cr = cmd_contract_review(parser.parse_args(["contract-review", str(partner_txt), "--output-docx", str(out_docx)]))
    assert kq_cr["so_canh_bao"] >= 2
    assert out_docx.is_file()

    cfg_ide = cmd_mcp_server(parser.parse_args(["mcp-server", "--print-config", "cursor"]))
    assert "mcpServers" in cfg_ide

    dry_conn = cmd_connector(parser.parse_args(["connector", "both", "--dry-run"]))
    assert "/webhook/zalo" in dry_conn["endpoints"] and "/webhook/telegram" in dry_conn["endpoints"]

    for sym in (
        "BoSoSanhRaSoatHopDong",
        "ContractReviewEngine",
        "QuanLyPhienChatWorkspace",
        "StudioWorkspaceManager",
        "MayChuMCPThucTe",
        "ProductionMCPServer",
        "KetNoiZaloOA",
        "ZaloOAConnector",
        "KetNoiTelegram",
        "TelegramBotConnector",
        "MayChuWebhookKenh",
        "ChannelWebhookServer",
    ):
        assert sym in vietnamese_ai.__all__
        assert getattr(vietnamese_ai, sym) is not None

    bc_rel = QuanLyPhatHanhGoi.kiem_tra_san_sang_phat_hanh()
    assert bc_rel["san_sang_phat_hanh"] is True
    assert bc_rel["thieu_type_stub"] == []
