"""Kiểm thử tích hợp đầu cuối Hướng C2 trên toàn bộ 5 bề mặt giao tiếp người dùng của Vietnamese AI."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


def _kiem_tra_zip_track_changes(docx_bytes: bytes) -> Tuple[str, str]:
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zf:
        names = set(zf.namelist())
        assert "word/document.xml" in names
        assert "word/comments.xml" in names
        doc_xml = zf.read("word/document.xml").decode("utf-8")
        cmt_xml = zf.read("word/comments.xml").decode("utf-8")
    assert "<w:del" in doc_xml or "<w:ins" in doc_xml
    assert "<w:comment " in cmt_xml
    return doc_xml, cmt_xml


def _http_json(
    url: str,
    payload: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    req_headers = dict(headers or {})
    data: bytes | None = None
    method = "GET"
    if payload is not None:
        method = "POST"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


def _asgi_post_json(app: Any, path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sent = False
    status_holder: List[int] = [500]
    chunks: List[bytes] = []

    async def _receive() -> Dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.disconnect"}

    async def _send(message: Dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            status_holder[0] = int(message["status"])
        elif message["type"] == "http.response.body":
            chunks.append(bytes(message.get("body", b"")))

    scope: Dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body_bytes)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 8000),
    }
    asyncio.run(app(scope, _receive, _send))
    raw_body = b"".join(chunks).decode("utf-8")
    return status_holder[0], json.loads(raw_body)


def test_c2_tro_ly_phap_ly_point_in_time_va_hnsw(tmp_path: Path) -> None:
    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

    db_path = tmp_path / "phap_ly_c2.vaidb"
    tl = TroLyPhapLy(nap_kho_mo=True, dung_onnx_semantic=False, duong_dan_vaidb=str(db_path))
    try:
        pb_2024 = tl.tra_cuu_dieu_khoan_theo_ngay(
            so_hieu="123/2020/NĐ-CP",
            dieu="Điều 19",
            ngay_tra_cuu="2024-12-01",
        )
        pb_2025 = tl.tra_cuu_dieu_khoan_theo_ngay(
            so_hieu="123/2020/NĐ-CP",
            dieu="Điều 19",
            ngay_tra_cuu="2025-07-01",
        )
        assert pb_2024["tim_thay"] is True
        assert pb_2025["tim_thay"] is True
        assert "cho phép hủy hóa đơn" in pb_2024["phien_ban_hieu_luc"]["noi_dung"]
        assert "Bãi bỏ quy định hủy hóa đơn" in pb_2025["phien_ban_hieu_luc"]["noi_dung"]

        kq_hoi = tl.tra_cuu(
            "Điều 19 Nghị định 123/2020/NĐ-CP quy định xử lý hóa đơn sai sót thế nào?",
            ngay_tra_cuu="2025-07-01",
        )
        assert kq_hoi["an_toan"] is True
        assert kq_hoi["ngay_tra_cuu"] == "2025-07-01"
        assert "phien_ban_dieu_khoan" in kq_hoi
        assert "04/SS-HĐĐT" in kq_hoi["tra_loi"]

        hnsw_info = tl.xay_dung_chi_muc_hnsw(m=8, ef_construction=32, ef_search=24)
        assert hnsw_info["so_nut"] > 0
        assert db_path.exists()
    finally:
        tl.rag.dong()


def test_c2_mcp_server_mac_dinh_va_mo_rong(tmp_path: Path) -> None:
    from vietnamese_ai.agents.mcp_transport import MayChuMCPThucTe

    tc_docx = tmp_path / "mcp_track_changes.docx"
    server_default = MayChuMCPThucTe()
    try:
        resp_list = server_default.xu_ly_jsonrpc(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        assert resp_list is not None
        assert len(resp_list["result"]["tools"]) == 6

        resp_hl = server_default.xu_ly_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "kiem_tra_hieu_luc",
                    "arguments": {
                        "so_hieu_hoac_van_ban": "123/2020/NĐ-CP",
                        "dieu": "Điều 19",
                        "ngay_tra_cuu": "2025-07-01",
                    },
                },
            }
        )
        assert resp_hl is not None
        hl_data = json.loads(resp_hl["result"]["content"][0]["text"])
        assert hl_data["tim_thay"] is True
        assert "Bãi bỏ quy định hủy hóa đơn" in hl_data["phien_ban_hieu_luc"]["noi_dung"]

        hd_chuan = "Điều 1. Thanh toán\nThanh toán trong 15 ngày.\nĐiều 2. Phạt vi phạm\nMức phạt 5%."
        hd_doi_tac = "Điều 1. Thanh toán\nThanh toán trong 60 ngày.\nĐiều 2. Phạt vi phạm\nMức phạt 25% theo Luật Đất đai 45/2013/QH13."
        resp_hd = server_default.xu_ly_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "ra_soat_hop_dong",
                    "arguments": {
                        "hop_dong_doi_tac": hd_doi_tac,
                        "hop_dong_chuan": hd_chuan,
                        "xuat_track_changes_docx": str(tc_docx),
                    },
                },
            }
        )
        assert resp_hd is not None
        hd_data = json.loads(resp_hd["result"]["content"][0]["text"])
        assert Path(hd_data["track_changes_docx_path"]).exists()
        _kiem_tra_zip_track_changes(tc_docx.read_bytes())
    finally:
        server_default.dong()

    db_path = tmp_path / "mcp_hnsw.vaidb"
    server_ext = MayChuMCPThucTe(duong_dan_vaidb=str(db_path), bat_cong_cu_mo_rong=True)
    try:
        resp_ext_list = server_ext.xu_ly_jsonrpc(
            {"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {}}
        )
        assert resp_ext_list is not None
        tool_names = {t["name"] for t in resp_ext_list["result"]["tools"]}
        assert len(tool_names) == 9
        assert {
            "tra_cuu_dieu_khoan_theo_ngay",
            "tach_tu_tieng_viet",
            "xay_dung_chi_muc_hnsw_vaidb",
        }.issubset(tool_names)

        resp_tok = server_ext.xu_ly_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "tach_tu_tieng_viet",
                    "arguments": {
                        "van_ban": "hoà bình và học sinh học sinh học tại Việt Nam",
                        "phuong_phap": "viterbi",
                        "chuan_hoa_dau": True,
                    },
                },
            }
        )
        assert resp_tok is not None
        tok_data = json.loads(resp_tok["result"]["content"][0]["text"])
        assert tok_data["phuong_phap"] == "viterbi"
        assert "hòa bình" in tok_data["van_ban_chuan_hoa"]
        assert tok_data["so_token"] >= 4

        resp_dieu = server_ext.xu_ly_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {
                    "name": "tra_cuu_dieu_khoan_theo_ngay",
                    "arguments": {
                        "so_hieu": "123/2020/NĐ-CP",
                        "dieu": "Điều 9",
                        "khoan": "Khoản 1",
                        "ngay_tra_cuu": "2025-07-01",
                    },
                },
            }
        )
        assert resp_dieu is not None
        dieu_data = json.loads(resp_dieu["result"]["content"][0]["text"])
        assert dieu_data["tim_thay"] is True
        assert dieu_data["phien_ban_hieu_luc"]["sua_doi_boi"] == "70/2025/NĐ-CP"

        tl_mcp = server_ext._lay_tro_ly_phap_ly()
        assert tl_mcp.rag.csdl_vector.so_luong() > 0
        resp_hnsw = server_ext.xu_ly_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "tools/call",
                "params": {
                    "name": "xay_dung_chi_muc_hnsw_vaidb",
                    "arguments": {"m": 8, "ef_construction": 32, "ef_search": 24},
                },
            }
        )
        assert resp_hnsw is not None
        hnsw_data = json.loads(resp_hnsw["result"]["content"][0]["text"])
        assert hnsw_data["hnsw"]["so_nut"] > 0
    finally:
        server_ext.dong()


def test_c2_cli_viterbi_hnsw_point_in_time_track_changes_va_eval_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from vietnamese_ai.cli.main import (
        _tao_parser,
        cmd_contract_review,
        cmd_eval_gate,
        cmd_legal_graph,
        cmd_rag,
        cmd_tokenize,
    )

    parser = _tao_parser()

    args_tok = parser.parse_args(
        ["tokenize", "hoà bình và trí tuệ nhân tạo tại Việt Nam", "--method", "viterbi", "--normalize-tone"]
    )
    tokens = cmd_tokenize(args_tok)
    out_tok = capsys.readouterr().out
    assert "viterbi" in out_tok
    assert "hòa bình" in out_tok
    assert isinstance(tokens, list) and len(tokens) >= 4

    doc_file = tmp_path / "luat_mau.txt"
    doc_file.write_text(
        "Điều 105 Bộ luật Lao động 2019: Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày.\n"
        "Nghị định 13/2023/NĐ-CP quy định về bảo vệ dữ liệu cá nhân tại Việt Nam.",
        encoding="utf-8",
    )
    db_file = tmp_path / "cli_hnsw.vaidb"
    args_rag = parser.parse_args(["rag", "--docs", str(doc_file), "--db", str(db_file), "--build-hnsw"])
    kq_rag = cmd_rag(args_rag)
    out_rag = capsys.readouterr().out
    assert "[VAIDB HNSW]" in out_rag
    assert kq_rag["hnsw"]["so_nut"] >= 1
    assert db_file.exists()

    args_lg = parser.parse_args(
        ["legal-graph", "123/2020/NĐ-CP", "--dieu", "Điều 19", "--date", "2025-07-01"]
    )
    kq_lg = cmd_legal_graph(args_lg)
    out_lg = capsys.readouterr().out
    assert "POINT-IN-TIME" in out_lg
    assert kq_lg["phien_ban_dieu_khoan"]["tim_thay"] is True
    assert "04/SS-HĐĐT" in kq_lg["phien_ban_dieu_khoan"]["phien_ban_hieu_luc"]["noi_dung"]

    tpl_file = tmp_path / "hd_chuan.txt"
    prt_file = tmp_path / "hd_doi_tac.txt"
    tc_out = tmp_path / "cli_track_changes.docx"
    tpl_file.write_text(
        "Điều 1. Giá trị và thanh toán\nThanh toán 100 triệu đồng trong 15 ngày.\n"
        "Điều 2. Phạt vi phạm\nMức phạt không quá 8%.",
        encoding="utf-8",
    )
    prt_file.write_text(
        "Điều 1. Giá trị hợp đồng\nGiá trị 100 triệu đồng.\n"
        "Điều 2. Thời hạn thanh toán\nThanh toán trong 45 ngày.\n"
        "Điều 3. Phạt vi phạm\nMức phạt là 30% theo Luật Đất đai 45/2013/QH13.",
        encoding="utf-8",
    )
    args_cr = parser.parse_args(
        [
            "contract-review",
            str(prt_file),
            "--template",
            str(tpl_file),
            "--track-changes-docx",
            str(tc_out),
        ]
    )
    kq_cr = cmd_contract_review(args_cr)
    capsys.readouterr()
    assert Path(kq_cr["track_changes_docx_path"]).exists()
    _kiem_tra_zip_track_changes(tc_out.read_bytes())

    args_eg = parser.parse_args(
        ["eval-gate", "--limit", "2", "--compare-baseline", "--extended-attacks", "--json"]
    )
    kq_eg = cmd_eval_gate(args_eg)
    raw_eg = capsys.readouterr().out
    parsed_eg = json.loads(raw_eg)
    assert parsed_eg["quyet_dinh"] == kq_eg["quyet_dinh"]
    assert "bootstrap_ci_95" in parsed_eg["chi_tiet_thong_ke"]

    args_mcp = parser.parse_args(["mcp-server", "--extended-tools"])
    assert args_mcp.extended_tools is True


def test_c2_fastapi_endpoints_huong_b(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")

    from vietnamese_ai.api.fastapi_server import FastAPIServer
    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

    db_path = tmp_path / "api_hnsw.vaidb"
    tl = TroLyPhapLy(nap_kho_mo=True, dung_onnx_semantic=False, duong_dan_vaidb=str(db_path))
    try:
        srv = FastAPIServer(bo_xu_ly=tl, enable_watermark=False)

        s_clause, d_clause = _asgi_post_json(
            srv.app,
            "/v1/legal/clause-version",
            {
                "so_hieu": "123/2020/NĐ-CP",
                "dieu": "Điều 19",
                "ngay_tra_cuu": "2025-07-01",
            },
        )
        assert s_clause == 200
        assert d_clause["tim_thay"] is True
        assert "04/SS-HĐĐT" in d_clause["phien_ban_hieu_luc"]["noi_dung"]

        s_contract, d_contract = _asgi_post_json(
            srv.app,
            "/v1/legal/contract-review",
            {
                "hop_dong_chuan": "Điều 1. Thanh toán\nThanh toán trong 15 ngày.\nĐiều 2. Phạt vi phạm\nMức phạt 5%.",
                "hop_dong_doi_tac": "Điều 1. Thanh toán\nThanh toán trong 90 ngày.\nĐiều 2. Phạt vi phạm\nMức phạt 25% theo Luật Đất đai 45/2013/QH13.",
                "xuat_track_changes_docx": True,
            },
        )
        assert s_contract == 200
        assert d_contract["track_changes_docx_base64"]
        docx_bytes = base64.b64decode(d_contract["track_changes_docx_base64"])
        _kiem_tra_zip_track_changes(docx_bytes)

        s_tok, d_tok = _asgi_post_json(
            srv.app,
            "/v1/nlp/tokenize",
            {
                "van_ban": "toà án nhân dân và trí tuệ nhân tạo",
                "phuong_phap": "viterbi",
                "chuan_hoa_dau": True,
            },
        )
        assert s_tok == 200
        assert d_tok["phuong_phap"] == "viterbi"
        assert "tòa án" in d_tok["van_ban_chuan_hoa"]
        assert d_tok["so_token"] >= 3

        s_hnsw, d_hnsw = _asgi_post_json(
            srv.app,
            "/v1/vaidb/hnsw",
            {"m": 8, "ef_construction": 32, "ef_search": 24},
        )
        assert s_hnsw == 200
        assert d_hnsw["ho_tro"] is True
        assert d_hnsw["hnsw"]["so_nut"] > 0
    finally:
        tl.rag.dong()


def test_c2_studio_dashboard_http_day_du(tmp_path: Path) -> None:
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    vaidb_path = tmp_path / "studio_c2.vaidb"
    ws_dir = tmp_path / "studio_ws"
    studio = StudioDashboard(
        host="127.0.0.1",
        port=0,
        duong_dan_vaidb=str(vaidb_path),
        thu_muc_workspace=str(ws_dir),
        bat_xac_thuc=False,
    )
    httpd = studio.tao_server(host="127.0.0.1", port=0)
    port = httpd.server_port
    base = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"{base}/", timeout=15) as resp_html:
            assert resp_html.status == 200
            html = resp_html.read().decode("utf-8")
        assert "rag/vaidb/build-hnsw" in html
        assert "legal/clause-version" in html
        assert "nlp/tokenize" in html
        assert "downloadContractTrackChangesDocx" in html
        assert 'value="simpo"' in html
        assert "runStudioEvalGate()" in html
        assert "runStudioEvalGate(true)" in html
        assert "sol-law-date" in html

        s_ing, d_ing = _http_json(
            f"{base}/api/rag/ingest",
            {
                "id": "doc_c2_1",
                "text": "Nghị định 13/2023/NĐ-CP bảo vệ dữ liệu cá nhân của công dân Việt Nam.",
            },
        )
        assert s_ing == 200 and d_ing["thanh_cong"] is True

        s_hnsw, d_hnsw = _http_json(
            f"{base}/api/rag/vaidb/build-hnsw",
            {"m": 8, "ef_construction": 32, "ef_search": 24},
        )
        assert s_hnsw == 200 and d_hnsw["thanh_cong"] is True
        assert d_hnsw["hnsw"]["so_nut"] >= 1

        s_cv, d_cv = _http_json(
            f"{base}/api/legal/clause-version",
            {"so_hieu": "123/2020/NĐ-CP", "dieu": "Điều 19", "ngay_tra_cuu": "2025-07-01"},
        )
        assert s_cv == 200 and d_cv["thanh_cong"] is True and d_cv["tim_thay"] is True
        assert "04/SS-HĐĐT" in d_cv["phien_ban_hieu_luc"]["noi_dung"]

        s_tok, d_tok = _http_json(
            f"{base}/api/nlp/tokenize",
            {"van_ban": "hoà bình và học sinh học sinh học", "phuong_phap": "viterbi", "chuan_hoa_dau": True},
        )
        assert s_tok == 200 and d_tok["thanh_cong"] is True
        assert "hòa bình" in d_tok["van_ban_chuan_hoa"]

        s_cr, d_cr = _http_json(
            f"{base}/api/legal/contract-review",
            {
                "hop_dong_chuan": "Điều 1. Thanh toán\nThanh toán trong 15 ngày.\nĐiều 2. Phạt vi phạm\nMức phạt 5%.",
                "hop_dong_doi_tac": "Điều 1. Thanh toán\nThanh toán trong 90 ngày.\nĐiều 2. Phạt vi phạm\nMức phạt 25% theo Luật Đất đai 45/2013/QH13.",
                "xuat_track_changes_docx": True,
            },
        )
        assert s_cr == 200 and d_cr["thanh_cong"] is True
        assert d_cr["ten_tep_track_changes"].endswith(".docx")
        _kiem_tra_zip_track_changes(base64.b64decode(d_cr["track_changes_docx_base64"]))

        s_simpo, d_simpo = _http_json(
            f"{base}/api/training/start",
            {"model": "vgpt2", "task": "simpo", "epochs": 6, "learning_rate": 0.2},
        )
        assert s_simpo == 200
        assert d_simpo["che_do_huan_luyen"] == "simpo_reference_free"
        assert d_simpo["loss_cuoi_cung"] <= d_simpo["loss_khoi_tao"]

        s_eg, d_eg = _http_json(
            f"{base}/api/flywheel/eval-gate",
            {"gioi_han_mau": 2, "so_sanh_ban_goc": True, "bo_tan_cong_mo_rong": True},
        )
        assert s_eg == 200 and d_eg["thanh_cong"] is True
        assert "bootstrap_ci_95" in d_eg["chi_tiet_thong_ke"]

        s_law, d_law = _http_json(
            f"{base}/api/solutions/phap_ly",
            {"question": "Điều 19 123/2020/NĐ-CP quy định gì?", "ngay_tra_cuu": "2025-07-01"},
        )
        assert s_law == 200 and d_law["an_toan"] is True
        assert d_law["ngay_tra_cuu"] == "2025-07-01"
        assert "phien_ban_dieu_khoan" in d_law
    finally:
        httpd.shutdown()
        httpd.server_close()
        th.join(timeout=5)
        studio.dong()


def test_c2_studio_rbac_bao_ve_build_hnsw(tmp_path: Path) -> None:
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    studio = StudioDashboard(
        host="127.0.0.1",
        port=0,
        duong_dan_vaidb=str(tmp_path / "rbac.vaidb"),
        thu_muc_workspace=str(tmp_path / "rbac_ws"),
        bat_xac_thuc=True,
    )
    httpd = studio.tao_server(host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{httpd.server_port}"
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        status, body = _http_json(f"{base}/api/rag/vaidb/build-hnsw", {"m": 8})
        assert status == 403
        assert body["thanh_cong"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        th.join(timeout=5)
        studio.dong()
