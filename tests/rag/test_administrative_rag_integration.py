"""
Kiểm thử tích hợp Bộ Tối Ưu Ngữ Cảnh Hành Chính & Pháp Lý Việt Nam vào RAGPipeline và TroLyPhapLy.
"""

from argparse import Namespace

from vietnamese_ai.rag.rag_pipeline import RAGPipeline
from vietnamese_ai.rag.vietnamese_context import BoToiUuNguCanhHanhChinh
from vietnamese_ai.solutions.phap_ly import LegalAssistant, TroLyPhapLy

VAN_BAN_LUAT_MAU = """Chương I
QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Căn cứ Luật Viễn thông số 41/2009/QH12;
Nghị định số 15/2020/NĐ-CP quy định xử phạt vi phạm hành chính trong lĩnh vực bưu chính, viễn thông, tần số vô tuyến điện.

Điều 2. Hành vi vi phạm và mức phạt
1. Phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi tiết lộ bí mật thông tin người sử dụng dịch vụ bưu chính.
2. Phạt tiền từ 30.000.000 đồng đến 50.000.000 đồng đối với hành vi cung cấp dịch vụ khi chưa có giấy phép bưu chính.
"""


def test_rag_pipeline_che_do_hanh_chinh_integration():
    """Kiểm tra RAGPipeline tích hợp chế độ phân đoạn hành chính & trích xuất căn cứ pháp lý."""
    rag = RAGPipeline(che_do_hanh_chinh=True, nguong_diem=0.0)
    assert rag.che_do_hanh_chinh is True
    assert rag.bo_toi_uu_hanh_chinh is not None
    assert isinstance(rag.bo_toi_uu_hanh_chinh, BoToiUuNguCanhHanhChinh)

    # 1. Nạp tài liệu văn bản luật mẫu
    so_chunks = rag.them_tai_lieu("ND15", VAN_BAN_LUAT_MAU)
    assert so_chunks >= 2

    # 2. Kiểm tra thống kê pipeline
    thong_ke = rag.thong_ke()
    assert thong_ke["che_do_hanh_chinh"] is True
    assert thong_ke["so_tai_lieu"] == 1
    assert thong_ke["so_chunks"] >= 2

    # 3. Truy vấn quy định và kiểm tra trích dẫn pháp lý tự động
    kq = rag.hoi("Mức phạt hành vi tiết lộ bí mật thông tin người dùng bưu chính?")
    assert "tra_loi" in kq
    assert "trich_dan_phap_ly" in kq

    trich_dan = kq["trich_dan_phap_ly"]
    assert "dieu_khoan" in trich_dan
    assert any("Điều 2" in dk for dk in trich_dan["dieu_khoan"])
    assert any("15/2020/NĐ-CP" in sh for sh in trich_dan["so_hieu_van_ban"])
    assert any("Luật Viễn thông" in cc or "41/2009/QH12" in cc for cc in trich_dan["danh_sach_can_cu"])

    # 4. Kiểm tra văn phong trả lời hành chính
    assert "Căn cứ theo các quy định" in kq["tra_loi"] or "10.000.000" in kq["tra_loi"]


def test_rag_pipeline_che_do_tu_dong_bat_hanh_chinh():
    """Kiểm tra RAGPipeline thông thường nhưng bật cờ che_do_hanh_chinh khi nạp tài liệu."""
    rag_thuong = RAGPipeline(che_do_hanh_chinh=False)
    assert rag_thuong.che_do_hanh_chinh is False

    # Khi nạp tài liệu chỉ định rõ che_do_hanh_chinh=True
    so_chunks = rag_thuong.them_tai_lieu("ND15_spec", VAN_BAN_LUAT_MAU, che_do_hanh_chinh=True)
    assert so_chunks >= 2
    assert rag_thuong.bo_toi_uu_hanh_chinh is not None


def test_tro_ly_phap_ly_nang_cap_tich_hop():
    """Kiểm tra TroLyPhapLy tự động kế thừa BoToiUuNguCanhHanhChinh và RAGPipeline."""
    assert LegalAssistant is TroLyPhapLy
    tro_ly = TroLyPhapLy()

    assert tro_ly.rag.che_do_hanh_chinh is True
    assert tro_ly.bo_toi_uu is not None

    # Tra cứu với câu hỏi pháp luật lao động
    res = tro_ly.tra_cuu("Thời giờ làm việc bình thường theo Bộ luật Lao động?")
    assert res["an_toan"] is True
    assert any("Điều 105" in c for c in res["can_cu_phap_ly"])
    assert "trich_dan_phap_ly" in res
    assert res["diem_bam_sat_nguon"] > 0.0

    # Thêm điều khoản mới kèm phân đoạn hành chính
    tro_ly.them_dieu_khoan(
        ma="nd15_d2",
        van_ban="Nghị định 15/2020/NĐ-CP",
        dieu="Điều 2",
        tieu_de="Quy định xử phạt viễn thông",
        noi_dung="Phạt tiền từ 10.000.000 đồng đến 20.000.000 đồng đối với hành vi tiết lộ bí mật bưu chính.",
    )
    res_moi = tro_ly.tra_cuu("Hành vi tiết lộ bí mật bưu chính bị phạt bao nhiêu?")
    assert any("Điều 2" in c for c in res_moi["can_cu_phap_ly"])


def test_cli_cmd_rag_admin(capsys, tmp_path):
    """Kiểm tra lệnh CLI vai rag với cờ --admin."""
    from vietnamese_ai.cli.main import cmd_rag

    doc_file = tmp_path / "luat_test.txt"
    doc_file.write_text(VAN_BAN_LUAT_MAU, encoding="utf-8")

    args = Namespace(
        docs=str(doc_file),
        query="Mức phạt tiết lộ bí mật thông tin?",
        admin=True,
    )
    cmd_rag(args)
    captured = capsys.readouterr()

    assert "Trả lời:" in captured.out
    assert "Điều khoản căn cứ:" in captured.out or "Số hiệu văn bản:" in captured.out
