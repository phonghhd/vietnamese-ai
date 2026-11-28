from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest


def _parse_sse_stream(raw_text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Phân tách chuỗi văn bản chuẩn Server-Sent Events thành danh sách cặp `(event, json_data)`."""
    events: List[Tuple[str, Dict[str, Any]]] = []
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev_name = "message"
        data_lines: List[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            events.append((ev_name, json.loads("".join(data_lines))))
    return events


def _post_http(url: str, payload: Dict[str, Any], headers: Dict[str, str] | None = None) -> Tuple[int, str, str]:
    """Gửi yêu cầu HTTP POST bằng `urllib` và trả về `(status_code, content_type, body_text)`."""
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ctype = resp.headers.get("Content-Type", "")
            return resp.status, ctype, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode("utf-8")


def test_m2_docx_merged_cells_and_red_stamp_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kiểm tra đọc bảng DOCX có gộp ô ngang/dọc (`gridSpan`/`vMerge`), khử dấu đỏ NumPy và tự nhận diện PDF scan."""
    from vietnamese_ai.rag.legal.document_parser import BoDocTaiLieuPhapLy

    docx_path = tmp_path / "bieu_bang_gop_o.docx"
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>BIỂU PHÍ DỊCH VỤ PHÁP LÝ</w:t></w:r></w:p>"
        "<w:tbl>"
        "<w:tr>"
        "<w:tc><w:p><w:r><w:t>Nhóm</w:t></w:r></w:p></w:tc>"
        '<w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>Chi tiết biểu phí</w:t></w:r></w:p></w:tc>'
        "</w:tr>"
        "<w:tr>"
        '<w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>Doanh nghiệp</w:t></w:r></w:p></w:tc>'
        "<w:tc><w:p><w:r><w:t>Thành lập mới</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>5.000.000 đồng</w:t></w:r></w:p></w:tc>"
        "</w:tr>"
        "<w:tr>"
        "<w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p><w:r><w:t></w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Thay đổi giấy phép</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>2.500.000 đồng</w:t></w:r></w:p></w:tc>"
        "</w:tr>"
        "</w:tbl>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", doc_xml)

    parser = BoDocTaiLieuPhapLy()
    van_ban_xml, so_bang = parser._doc_docx_ooxml(docx_path)
    assert so_bang == 1
    assert "| Nhóm | Chi tiết biểu phí |  |" in van_ban_xml
    assert "| Doanh nghiệp | Thành lập mới | 5.000.000 đồng |" in van_ban_xml
    assert "| Doanh nghiệp | Thay đổi giấy phép | 2.500.000 đồng |" in van_ban_xml

    img = np.full((12, 12, 3), 255, dtype=np.uint8)
    img[2:5, 2:5] = [15, 15, 15]
    img[7:10, 7:10] = [210, 30, 35]
    sach = BoDocTaiLieuPhapLy.loc_dau_do_anh_numpy(img)
    assert tuple(int(v) for v in sach[3, 3]) == (15, 15, 15)
    assert tuple(int(v) for v in sach[8, 8]) == (255, 255, 255)

    dummy_pdf = tmp_path / "cong_van_scan.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr(parser, "_doc_pdf_co_bang", lambda _p: ("", 2, 0))
    monkeypatch.setattr(parser, "_doc_pdf_scan_ocr", lambda _p: ("Điều 1. Nội dung công văn đã OCR sạch dấu đỏ.", 2))
    kq_pdf = parser.doc_tep(dummy_pdf)
    assert kq_pdf["dinh_dang"] == "pdf_scan_ocr"
    assert kq_pdf["metadata"]["la_pdf_scan"] is True
    assert kq_pdf["metadata"]["da_chay_ocr_pdf"] is True
    assert "OCR sạch dấu đỏ" in kq_pdf["van_ban"]


def test_m2_contract_review_pro_five_types_and_hybrid_llm() -> None:
    """Kiểm tra Contract Review Pro cho 5 loại hợp đồng Việt Nam và chế độ Hybrid LLM kèm kiểm chứng trích dẫn 3 tầng."""
    from vietnamese_ai.rag.legal.contract_review import BoSoSanhRaSoatHopDong, phan_loai_hop_dong

    engine = BoSoSanhRaSoatHopDong()

    hd_lao_dong = (
        "HỢP ĐỒNG LAO ĐỘNG\n"
        "Điều 1. Đối tượng và công việc: Người lao động làm chuyên viên kinh doanh.\n"
        "Điều 2. Thử việc và Tiền lương: Thời gian thử việc là 90 ngày. Trong thời gian thử việc hưởng 70% mức lương chính thức.\n"
        "Điều 3. Kỷ luật và Bảo đảm: Công ty giữ bản chính CCCD gốc và bằng đại học của người lao động. "
        "Nếu đi muộn vi phạm nội quy sẽ phạt tiền cắt lương 500.000 đồng.\n"
        "Điều 4. Bảo mật, Bất khả kháng và Tranh chấp: Tuân thủ Nghị định 13/2023/NĐ-CP, thiên tai bất khả kháng, giải quyết tại Tòa án."
    )
    assert phan_loai_hop_dong(hd_lao_dong) == "lao_dong"
    kq_ld = engine.quet_rui_ro_don_le(hd_lao_dong, ngay_tra_cuu="2026-01-15")
    ma_ld = {c["ma_quy_tac"] for c in kq_ld["danh_sach_canh_bao"]}
    assert kq_ld["loai_hop_dong"] == "lao_dong"
    assert "thu_viec_vuot_tran_blld" in ma_ld
    assert "luong_thu_viec_duoi_85pct" in ma_ld
    assert "hanh_vi_bi_nghiem_cam_blld_d17" in ma_ld
    assert "phat_tien_cat_luong_ky_luat_blld" in ma_ld

    hd_quan_ly = (
        "HỢP ĐỒNG LAO ĐỘNG\n"
        "Điều 1. Công việc: Người quản lý doanh nghiệp chức danh Giám đốc điều hành.\n"
        "Điều 2. Thử việc: Thời gian thử việc là 120 ngày, mức lương thử việc bằng 90% tiền lương chính thức."
    )
    kq_ql = engine.quet_rui_ro_don_le(hd_quan_ly, ngay_tra_cuu="2026-01-15")
    assert "thu_viec_vuot_tran_blld" not in {c["ma_quy_tac"] for c in kq_ql["danh_sach_canh_bao"]}

    hd_dich_vu = (
        "HỢP ĐỒNG DỊCH VỤ PHẦN MỀM\n"
        "Điều 1. Đối tượng dịch vụ: Cung cấp dịch vụ phần mềm quản trị.\n"
        "Điều 2. Sở hữu trí tuệ: Mọi quyền sở hữu trí tuệ và mã nguồn nền tảng (Background IP) chuyển giao toàn bộ cho Bên A.\n"
        "Điều 3. Thanh toán: Chuyển khoản ngân hàng.\n"
        "Điều 4. Bảo mật, Bất khả kháng, Tranh chấp: Đầy đủ theo tòa án và thiên tai."
    )
    assert phan_loai_hop_dong(hd_dich_vu) == "dich_vu"
    kq_dv = engine.quet_rui_ro_don_le(hd_dich_vu, ngay_tra_cuu="2026-01-15")
    assert "chuyen_giao_ip_nen_tang_dich_vu" in {c["ma_quy_tac"] for c in kq_dv["danh_sach_canh_bao"]}

    hd_mua_ban = (
        "HỢP ĐỒNG MUA BÁN HÀNG HÓA\n"
        "Điều 1. Đối tượng hàng hóa: Bên bán giao thiết bị văn phòng cho Bên mua.\n"
        "Điều 2. Thanh toán và Hóa đơn: Bên mua thanh toán bằng tiền mặt số tiền 12 triệu đồng. "
        "Khi có sai sót hai bên thực hiện hủy hóa đơn điện tử đã cấp mã.\n"
        "Điều 3. Bảo mật, Bất khả kháng, Tranh chấp: Theo quy định tòa án và thiên tai."
    )
    assert phan_loai_hop_dong(hd_mua_ban) == "mua_ban"
    kq_mb_sau = engine.quet_rui_ro_don_le(hd_mua_ban, ngay_tra_cuu="2025-08-01")
    ma_mb_sau = {c["ma_quy_tac"] for c in kq_mb_sau["danh_sach_canh_bao"]}
    assert "huy_hoa_don_dien_tu_trai_nd70" in ma_mb_sau
    assert "thanh_toan_tien_mat_gtgt_5_trieu" in ma_mb_sau

    kq_mb_truoc = engine.quet_rui_ro_don_le(hd_mua_ban, ngay_tra_cuu="2025-05-01")
    ma_mb_truoc = {c["ma_quy_tac"] for c in kq_mb_truoc["danh_sach_canh_bao"]}
    assert "huy_hoa_don_dien_tu_trai_nd70" not in ma_mb_truoc
    assert "thanh_toan_tien_mat_gtgt_5_trieu" not in ma_mb_truoc

    hd_thue_nha = (
        "HỢP ĐỒNG THUÊ NHÀ VÀ ĐẶT CỌC NHÀ Ở HÌNH THÀNH TRONG TƯƠNG LAI\n"
        "Điều 1. Đối tượng: Bên cho thuê bàn giao căn hộ thuộc dự án nhà ở hình thành trong tương lai.\n"
        "Điều 2. Đặt cọc và Giá thuê: Bên thuê nộp tiền đặt cọc 15% giá trị căn hộ. "
        "Bên cho thuê có quyền đơn phương tăng giá thuê và thu hồi nhà ngay lập tức nếu muốn.\n"
        "Điều 3. Bảo mật, Bất khả kháng, Tranh chấp: Tòa án giải quyết khi không phải thiên tai."
    )
    assert phan_loai_hop_dong(hd_thue_nha) == "thue_nha"
    kq_tn = engine.quet_rui_ro_don_le(hd_thue_nha, ngay_tra_cuu="2026-01-15")
    ma_tn = {c["ma_quy_tac"] for c in kq_tn["danh_sach_canh_bao"]}
    assert "dat_coc_nha_tuong_lai_vuot_5pct" in ma_tn
    assert "tang_gia_hoac_thu_hoi_nha_mot_chieu" in ma_tn

    hd_nda = (
        "THỎA THUẬN BẢO MẬT THÔNG TIN (NDA)\n"
        "Điều 1. Đối tượng bảo mật: Bên tiếp nhận thông tin giữ kín bí mật kinh doanh của Bên tiết lộ.\n"
        "Điều 2. Thời hạn và Không cạnh tranh: Nghĩa vụ bảo mật và cấm làm việc cho đối thủ có hiệu lực vĩnh viễn.\n"
        "Điều 3. Thanh toán, Bất khả kháng, Tranh chấp: Giải quyết tại Trọng tài nếu không có thiên tai."
    )
    assert phan_loai_hop_dong(hd_nda) == "nda"
    kq_nda = engine.quet_rui_ro_don_le(hd_nda, ngay_tra_cuu="2026-01-15")
    ma_nda = {c["ma_quy_tac"] for c in kq_nda["danh_sach_canh_bao"]}
    assert "nda_cam_ket_vinh_vien" in ma_nda
    assert "nda_thieu_ngoai_le_cong_khai" in ma_nda

    def _mock_llm(_prompt: str) -> str:
        return json.dumps(
            [
                {
                    "dieu_khoan": "Điều 2",
                    "muc_do": "cao",
                    "tieu_de": "Rủi ro viện dẫn sai điều luật thương mại",
                    "chi_tiet": "Mức phạt vi phạm hợp đồng thương mại không được vượt quá 8% theo Điều 301 Luật Thương mại 36/2005/QH11.",
                    "can_cu_phap_ly": "Điều 301 Luật Thương mại 36/2005/QH11",
                    "khuyen_nghi": "Giữ nguyên trần 8% theo Luật Thương mại.",
                },
                {
                    "dieu_khoan": "Điều 3",
                    "muc_do": "trung_binh",
                    "tieu_de": "Phân tích viện dẫn luật giả định",
                    "chi_tiet": "Theo Điều 999 Luật Giả Định 99/2026/QH15 điều khoản này bị phạt 50%.",
                    "can_cu_phap_ly": "Điều 999 Luật 99/2026/QH15",
                    "khuyen_nghi": "Kiểm tra lại căn cứ pháp lý.",
                },
            ],
            ensure_ascii=False,
        )

    kq_hybrid = engine.quet_rui_ro_don_le(
        hd_dich_vu,
        ngay_tra_cuu="2026-01-15",
        ham_sinh_llm=_mock_llm,
    )
    pt_llm = kq_hybrid["phan_tich_llm"]
    assert pt_llm["so_goi_y_llm"] == 2
    assert pt_llm["so_hop_le"] == 1
    assert pt_llm["so_can_xac_minh"] == 1
    assert len(pt_llm["canh_bao_ao_giac"]) == 1
    nguon_llm = {c["ma_quy_tac"]: c["nguon_phat_hien"] for c in kq_hybrid["danh_sach_canh_bao"] if c["ma_quy_tac"].startswith("llm_")}
    assert nguon_llm["llm_semantic_risk_1"] == "llm_da_kiem_chung"
    assert nguon_llm["llm_semantic_risk_2"] == "llm_can_xac_minh"


def test_m2_studio_sse_streaming_split_view_and_rbac(tmp_path: Path) -> None:
    """Kiểm tra luồng SSE trực tuyến trên Studio (`/api/legal/temporal-ask/stream`, `/api/chat/stream`), Contract Pro API và chặn RBAC 403."""
    from vietnamese_ai.ui.studio_dashboard import StudioDashboard

    dash = StudioDashboard(
        host="127.0.0.1",
        port=0,
        thu_muc_workspace=str(tmp_path / "studio_ws"),
        bat_xac_thuc=False,
    )
    srv = dash.tao_server(host="127.0.0.1", port=0)
    port = srv.server_port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/", timeout=10) as resp_html:
            html_txt = resp_html.read().decode("utf-8")
        assert "streamTemporalLegalRAG" in html_txt
        assert "contract-type" in html_txt
        assert "contract-visual-diff" in html_txt
        assert "streamWorkspaceChat" in html_txt

        st_t, ct_t, body_t = _post_http(
            f"{base}/api/legal/temporal-ask/stream",
            {
                "cau_hoi": "Vào tháng 8/2025, người bán có được hủy hóa đơn điện tử đã cấp mã bị sai sót không?",
                "che_do": "full",
                "top_k": 3,
            },
        )
        assert st_t == 200
        assert "text/event-stream" in ct_t
        evs_t = _parse_sse_stream(body_t)
        ev_names_t = [e[0] for e in evs_t]
        assert "metadata" in ev_names_t
        assert "token" in ev_names_t
        assert "verification" in ev_names_t
        assert ev_names_t[-1] == "done"
        joined_t = "".join(payload.get("chunk", "") for name, payload in evs_t if name == "token")
        assert len(joined_t.strip()) > 20

        st_s, _, body_s = _post_http(
            f"{base}/api/chat/sessions",
            {"ma_workspace": "mac_dinh", "tieu_de": "Phiên SSE M2"},
        )
        assert st_s == 200
        ma_phien = json.loads(body_s)["ma_phien"]

        st_c, ct_c, body_c = _post_http(
            f"{base}/api/chat/stream",
            {"ma_phien": ma_phien, "question": "Thời giờ làm việc bình thường tối đa bao nhiêu giờ mỗi ngày?"},
        )
        assert st_c == 200
        assert "text/event-stream" in ct_c
        evs_c = _parse_sse_stream(body_c)
        ev_names_c = [e[0] for e in evs_c]
        assert "metadata" in ev_names_c
        assert "token" in ev_names_c
        assert ev_names_c[-1] == "done"

        st_cr, _, body_cr = _post_http(
            f"{base}/api/legal/contract-review",
            {
                "loai_hop_dong": "lao_dong",
                "hop_dong_doi_tac": (
                    "HỢP ĐỒNG LAO ĐỘNG\n"
                    "Điều 1. Thử việc: Thời gian thử việc 90 ngày, lương thử việc 75%."
                ),
            },
        )
        assert st_cr == 200
        res_cr = json.loads(body_cr)
        assert res_cr["loai_hop_dong"] == "lao_dong"
        assert "thu_viec_vuot_tran_blld" in {c["ma_quy_tac"] for c in res_cr["danh_sach_canh_bao"]}
    finally:
        srv.shutdown()
        srv.server_close()
        dash.dong()

    dash_auth = StudioDashboard(
        host="127.0.0.1",
        port=0,
        thu_muc_workspace=str(tmp_path / "studio_ws_auth"),
        bat_xac_thuc=True,
    )
    srv_auth = dash_auth.tao_server(host="127.0.0.1", port=0)
    port_auth = srv_auth.server_port
    t_auth = threading.Thread(target=srv_auth.serve_forever, daemon=True)
    t_auth.start()
    try:
        for ep in ("/api/legal/temporal-ask/stream", "/api/chat/stream"):
            st_403, _, _ = _post_http(f"http://127.0.0.1:{port_auth}{ep}", {"cau_hoi": "test"})
            assert st_403 == 403
    finally:
        srv_auth.shutdown()
        srv_auth.server_close()
        dash_auth.dong()


def test_m2_cli_mcp_config_contract_type_and_docker_compose(tmp_path: Path) -> None:
    """Kiểm tra lệnh `vai mcp-config` cho 4 IDE client, gộp file JSON an toàn, `--type` của `contract-review` và `docker-compose.yml`."""
    from vietnamese_ai.cli.main import _tao_parser, cmd_contract_review, cmd_mcp_config

    parser = _tao_parser()
    kb_demo = str(tmp_path / "vn-business-core-2025.vaidb")

    cfg_cursor = cmd_mcp_config(
        parser.parse_args(["mcp-config", "--client", "cursor", "--kb", kb_demo, "--extended-tools"])
    )
    assert "mcpServers" in cfg_cursor
    args_cursor = cfg_cursor["mcpServers"]["vietnamese-ai"]["args"]
    assert "--vaidb" in args_cursor and kb_demo in args_cursor
    assert "--extended-tools" in args_cursor
    assert "--temporal-legal-tools" in args_cursor

    cfg_vscode = cmd_mcp_config(
        parser.parse_args(["mcp-config", "--client", "vscode", "--transport", "sse", "--port", "8091"])
    )
    assert cfg_vscode["servers"]["vietnamese-ai"]["type"] == "sse"
    assert cfg_vscode["servers"]["vietnamese-ai"]["url"] == "http://127.0.0.1:8091/sse"

    cfg_antigravity = cmd_mcp_config(
        parser.parse_args(["mcp-config", "--client", "antigravity", "--no-temporal-legal-tools"])
    )
    assert "--temporal-legal-tools" not in cfg_antigravity["mcpServers"]["vietnamese-ai"]["args"]

    existing_cfg = tmp_path / "mcp.json"
    existing_cfg.write_text(json.dumps({"mcpServers": {"other-tool": {"command": "node"}}}), encoding="utf-8")
    merged = cmd_mcp_config(
        parser.parse_args(["mcp-config", "--client", "claude", "--write", str(existing_cfg)])
    )
    assert "other-tool" in merged["mcpServers"]
    assert "vietnamese-ai" in merged["mcpServers"]
    on_disk = json.loads(existing_cfg.read_text(encoding="utf-8"))
    assert "other-tool" in on_disk["mcpServers"] and "vietnamese-ai" in on_disk["mcpServers"]

    partner_file = tmp_path / "hd_ld.txt"
    partner_file.write_text(
        "HỢP ĐỒNG LAO ĐỘNG\nĐiều 1. Thử việc: Thời gian thử việc 90 ngày, lương thử việc 70%.",
        encoding="utf-8",
    )
    kq_cli = cmd_contract_review(parser.parse_args(["contract-review", str(partner_file), "--type", "lao_dong"]))
    assert kq_cli["loai_hop_dong"] == "lao_dong"
    assert "thu_viec_vuot_tran_blld" in {c["ma_quy_tac"] for c in kq_cli["danh_sach_canh_bao"]}

    compose_path = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    assert compose_path.is_file()
    compose_txt = compose_path.read_text(encoding="utf-8")
    assert "studio:" in compose_txt
    assert "mcp-sse:" in compose_txt
    assert "connectors:" in compose_txt
    assert "127.0.0.1:8080:8080" in compose_txt
    try:
        import yaml

        parsed_compose = yaml.safe_load(compose_txt)
        assert set(parsed_compose["services"].keys()) >= {"studio", "mcp-sse", "connectors"}
    except ImportError:
        pass
