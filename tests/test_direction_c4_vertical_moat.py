"""Kiểm thử toàn diện Hướng C4 (Vertical Moat): Nạp tự động văn bản pháp quy/hợp nhất vào .vaidb HNSW và Cross-Encoder ONNX."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest


def _tao_tep_docx_phap_quy(duong_dan: Path, cac_doan: List[str]) -> Path:
    body = "".join(f"<w:p><w:r><w:t>{d}</w:t></w:r></w:p>" for d in cac_doan)
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    ct_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(duong_dan, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", doc_xml)
    return duong_dan


def test_c4_auto_ingest_van_ban_hop_nhat_va_dong_goi_vaidb_hnsw(tmp_path: Path) -> None:
    from vietnamese_ai.rag.legal import BoNapVanBanPhapQuyTuDong
    from vietnamese_ai.solutions.phap_ly import TroLyPhapLy

    src_dir = tmp_path / "van_ban_phap_quy"
    src_dir.mkdir(parents=True, exist_ok=True)

    vbhn_txt = src_dir / "vbhn_hoa_don_tu_dong.txt"
    vbhn_txt.write_text(
        "VĂN BẢN HỢP NHẤT\n"
        "Số: 12/VBHN-BTC\n"
        "NGHỊ ĐỊNH 55/2023/NĐ-CP VỀ QUẢN LÝ HÓA ĐƠN SỐ DOANH NGHIỆP\n"
        "Hà Nội, ngày 15 tháng 03 năm 2023\n"
        "Nghị định này thay thế Nghị định 11/2018/NĐ-CP và có hiệu lực thi hành từ ngày 01/05/2023.\n"
        "Chương I. QUY ĐỊNH CHUNG\n"
        "Điều 1. Phạm vi điều chỉnh\n"
        "Nghị định này quy định việc phát hành, lưu trữ và kiểm toán hóa đơn số doanh nghiệp.\n"
        "Điều 8. Xử lý sai sót hóa đơn số [1]\n"
        "Người bán lập biên bản điện tử mẫu 09/SS-HĐS và gửi thông báo điều chỉnh trong 02 ngày làm việc.\n"
        "\n"
        "[1] Điều này được sửa đổi, bổ sung theo quy định tại khoản 4 Điều 1 của Nghị định số 88/2025/NĐ-CP, "
        "có hiệu lực kể từ ngày 01 tháng 08 năm 2025.\n",
        encoding="utf-8",
    )

    _tao_tep_docx_phap_quy(
        src_dir / "thong_tu_huong_dan.docx",
        [
            "THÔNG TƯ 22/2025/TT-BTC",
            "Hà Nội, ngày 10 tháng 08 năm 2025",
            "Hướng dẫn thi hành Nghị định 55/2023/NĐ-CP về kiểm toán hóa đơn số, có hiệu lực từ ngày 01/09/2025.",
            "Điều 3. Định dạng lưu trữ điện tử",
            "Hóa đơn số phải lưu trữ theo chuẩn XML ký số tối thiểu 10 năm.",
        ],
    )

    vaidb_out = tmp_path / "kho_phap_quy_c4.vaidb"
    pipeline = BoNapVanBanPhapQuyTuDong(che_do_nhung="onnx_viet")
    bc = pipeline.dong_goi_kho_vaidb(src_dir, vaidb_out, xay_dung_hnsw=True, hnsw_m=8, hnsw_ef_construction=32)

    assert vaidb_out.is_file()
    assert bc["so_tep_da_nap"] == 2
    assert bc["tong_dieu_khoan"] >= 3
    assert bc["so_phien_ban_point_in_time"] >= 3
    assert bc["so_quan_he_do_thi"] >= 2
    assert bc["hnsw"]["da_xay_dung"] is True
    assert bc["thong_ke_vaidb"]["chi_muc_hnsw"] is True

    tro_ly = TroLyPhapLy(nap_kho_mo=False, dung_onnx_semantic=True, duong_dan_vaidb=str(vaidb_out))
    try:
        pb = tro_ly.tra_cuu_dieu_khoan_theo_ngay("55/2023/NĐ-CP", "Điều 8", ngay_tra_cuu="2025-10-01")
        assert pb["tim_thay"] is True
        assert pb["phien_ban_hieu_luc"]["sua_doi_boi"] == "88/2025/NĐ-CP"
        assert pb["phien_ban_hieu_luc"]["hieu_luc_tu"] == "2025-08-01"
        assert "09/SS-HĐS" in pb["phien_ban_hieu_luc"]["noi_dung"]

        dt = tro_ly.tra_cuu_van_ban_lien_quan("55/2023/NĐ-CP", ngay_tra_cuu="2025-10-01")
        ds_hd = [x["so_hieu"] for x in dt["van_ban_huong_dan"]]
        ds_sd = [x["so_hieu"] for x in dt["van_ban_sua_doi_bo_sung"]]
        assert "22/2025/TT-BTC" in ds_hd
        assert "88/2025/NĐ-CP" in ds_sd
    finally:
        tro_ly.rag.dong()


def test_c4_cli_legal_ingest_va_mo_lai(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from vietnamese_ai.cli.main import _tao_parser, cmd_legal_ingest

    src_file = tmp_path / "nghi_dinh_moi.txt"
    src_file.write_text(
        "NGHỊ ĐỊNH 77/2025/NĐ-CP\n"
        "Ngày 01/02/2025. Có hiệu lực từ ngày 01/04/2025.\n"
        "Điều 2. Bảo vệ dữ liệu AI\n"
        "Hệ thống AI phải lưu nhật ký truy cập tối thiểu 24 tháng (được sửa đổi, bổ sung bởi 91/2025/NĐ-CP có hiệu lực từ 01/10/2025).\n",
        encoding="utf-8",
    )
    out_db = tmp_path / "cli_ingest.vaidb"
    parser = _tao_parser()
    args = parser.parse_args(["legal-ingest", str(src_file), "--output", str(out_db), "--json"])
    kq = cmd_legal_ingest(args)
    raw = capsys.readouterr().out
    parsed = json.loads(raw)
    assert parsed["so_tep_da_nap"] == 1
    assert parsed["so_phien_ban_point_in_time"] >= 1
    assert parsed["hnsw"]["da_xay_dung"] is True
    assert kq["duong_dan_vaidb"] == str(out_db)
    assert out_db.is_file()


def test_c4_cross_encoder_reranker_bge_phoranker_va_batch_scoring() -> None:
    from vietnamese_ai.rag.onnx_semantic import PRESET_CROSS_ENCODER_VN, BoSapXepLaiONNXTiengViet
    from vietnamese_ai.rag.rag_pipeline import RAGPipeline

    assert {"bge-reranker-v2-m3", "phoranker", "multilingual-mini"}.issubset(PRESET_CROSS_ENCODER_VN.keys())

    pho = BoSapXepLaiONNXTiengViet(preset="phoranker")
    assert pho.can_tach_tu_viet is True
    q_seg, d_seg = pho._chuan_bi_van_ban_cap("Trí tuệ nhân tạo Việt Nam", "Ứng dụng học máy trong pháp lý")
    assert "trí_tuệ_nhân_tạo" in q_seg.lower()
    assert "học_máy" in d_seg.lower()

    scores_1d = BoSapXepLaiONNXTiengViet._logits_sang_diem(np.array([2.5, -2.5], dtype=np.float32))
    assert scores_1d[0] > 0.9 and scores_1d[1] < 0.1

    scores_2d = BoSapXepLaiONNXTiengViet._logits_sang_diem(
        np.array([[-1.5, 2.0], [2.0, -1.5]], dtype=np.float32)
    )
    assert scores_2d[0] > 0.95 and scores_2d[1] < 0.05

    class _FakeInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakeORTSession:
        def get_inputs(self) -> List[Any]:
            return [_FakeInput("input_ids"), _FakeInput("attention_mask"), _FakeInput("token_type_ids")]

        def run(self, _out: Any, feed: dict[str, Any]) -> List[np.ndarray]:
            bsz = feed["input_ids"].shape[0]
            logits = np.zeros((bsz, 2), dtype=np.float32)
            for i in range(bsz):
                logits[i, 0] = -float(i)
                logits[i, 1] = float(i + 1)
            return [logits]

    bge = BoSapXepLaiONNXTiengViet(preset="bge-reranker-v2-m3", kich_thuoc_batch=2)
    bge._ort_session = _FakeORTSession()
    bge._che_do = "onnx_cross_encoder"
    batch_scores = bge.cham_diem_lo("Câu hỏi pháp lý", ["Đoạn A", "Đoạn B", "Đoạn C"])
    assert len(batch_scores) == 3
    assert batch_scores[1] > batch_scores[0]

    for mode in ("bge_reranker", "phoranker", "cross_encoder_onnx"):
        rag = RAGPipeline(che_do_nhung="onnx_viet", che_do_rerank=mode, top_k=2)
        rag.them_tai_lieu(
            "d1",
            "Điều 105 Bộ luật Lao động 45/2019/QH14: Thời giờ làm việc bình thường không quá 08 giờ/ngày và 48 giờ/tuần.",
        )
        rag.them_tai_lieu(
            "d2",
            "Quy định chung về quản lý trang thiết bị văn phòng phẩm tại doanh nghiệp.",
        )
        kq = rag.hoi("Thời giờ làm việc tối đa một ngày là bao nhiêu?")
        assert "08 giờ" in kq["tra_loi"] or "48 giờ" in kq["tra_loi"]


def test_c4_mo_rong_phien_ban_dieu_khoan_dat_dai_gddt_nha_o() -> None:
    from vietnamese_ai.rag.legal import BoTheoDoiHieuLucPhapLy

    tracker = BoTheoDoiHieuLucPhapLy()

    dd_cu = tracker.tra_cuu_dieu_khoan_theo_ngay("45/2013/QH13", "Điều 114", ngay_tra_cuu="2023-06-01")
    dd_moi = tracker.tra_cuu_dieu_khoan_theo_ngay("31/2024/QH15", "Điều 159", ngay_tra_cuu="2025-01-01")
    assert dd_cu["con_hieu_luc"] is True and "khung giá đất" in dd_cu["phien_ban_hieu_luc"]["noi_dung"].lower()
    assert dd_moi["con_hieu_luc"] is True and "bãi bỏ khung giá đất" in dd_moi["phien_ban_hieu_luc"]["noi_dung"].lower()

    gd_moi = tracker.tra_cuu_dieu_khoan_theo_ngay("20/2023/QH15", "Điều 22", ngay_tra_cuu="2025-01-01")
    assert gd_moi["con_hieu_luc"] is True and "03 nhóm" in gd_moi["phien_ban_hieu_luc"]["noi_dung"]

    no_cu = tracker.tra_cuu_dieu_khoan_theo_ngay("65/2014/QH13", "Điều 118", ngay_tra_cuu="2025-01-01")
    no_moi = tracker.tra_cuu_dieu_khoan_theo_ngay("27/2023/QH15", "Điều 160", ngay_tra_cuu="2025-01-01")
    assert no_cu["con_hieu_luc"] is False
    assert no_moi["con_hieu_luc"] is True
