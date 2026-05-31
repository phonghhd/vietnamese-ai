from vietnamese_ai.security.red_team import RedTeamSimulator
from vietnamese_ai.security.watermark import TextWatermarker


def test_red_team_simulator():
    simulator = RedTeamSimulator()
    ket_qua = simulator.tan_cong()

    assert "tong_so_mau" in ket_qua
    assert "ty_le_bao_ve" in ket_qua
    assert ket_qua["tong_so_mau"] > 0
    # Hiện tại TuongLuaAI chưa chặn hết 100% các prompt khó trong danh sách
    # Nhưng ta kỳ vọng tỷ lệ bảo vệ > 0%
    assert ket_qua["ty_le_bao_ve"] > 0


def test_text_watermarker():
    van_ban_goc = "Đây là câu trả lời của AI."
    payload = "EVONET-123"

    van_ban_co_thuy_an = TextWatermarker.nhung_thuy_an(van_ban_goc, payload)

    # Độ dài văn bản có thủy ấn sẽ dài hơn do chứa ký tự ẩn
    assert len(van_ban_co_thuy_an) > len(van_ban_goc)

    # Tuy nhiên khi in ra màn hình hoặc so sánh string thông thường,
    # chúng vẫn hiển thị các ký tự thường như nhau
    assert van_ban_co_thuy_an.startswith(van_ban_goc)

    # Trích xuất payload
    payload_trich_xuat = TextWatermarker.giai_ma_thuy_an(van_ban_co_thuy_an)
    assert payload_trich_xuat == payload

    # Trích xuất từ văn bản không có thủy ấn
    assert TextWatermarker.giai_ma_thuy_an(van_ban_goc) is None
