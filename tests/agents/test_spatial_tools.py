from vietnamese_ai.agents.spatial_tools import cong_cu_di_chuyen_robot, cong_cu_quet_radar_3d


def test_cong_cu_di_chuyen_robot():
    ket_qua = cong_cu_di_chuyen_robot.chay(x=10.0, y=0.0, z=0.0)
    assert "Robot đang di chuyển đến tọa độ" in ket_qua
    assert "X=10.00" in ket_qua
    assert "Dự kiến mất 1.00 giây" in ket_qua


def test_cong_cu_quet_radar_3d():
    ket_qua = cong_cu_quet_radar_3d.chay()
    assert "Đã quét thành công khu vực xung quanh" in ket_qua
