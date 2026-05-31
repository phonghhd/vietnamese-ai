from vietnamese_ai.mobile.mobile_tools import (
    cong_cu_chup_anh_camera,
    cong_cu_doc_thong_bao_sms,
    cong_cu_lay_toa_do_gps,
)


def test_lay_toa_do_gps():
    ket_qua = cong_cu_lay_toa_do_gps.chay()
    assert "Đã lấy tọa độ GPS" in ket_qua
    assert "Kinh độ" in ket_qua


def test_doc_thong_bao_sms():
    ket_qua = cong_cu_doc_thong_bao_sms.chay()
    assert "Thông báo mới nhất" in ket_qua
    assert "OTP" in ket_qua


def test_chup_anh_camera():
    ket_qua = cong_cu_chup_anh_camera.chay(mat_truoc=True)
    assert "camera trước" in ket_qua
    assert ".jpg" in ket_qua
