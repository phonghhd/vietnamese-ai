import math

from vietnamese_ai.agents.tools import cong_cu


@cong_cu(
    ten="di_chuyen_robot",
    mo_ta="Điều khiển robot vật lý (cánh tay máy hoặc xe tự hành) di chuyển tới tọa độ (X, Y, Z). Tham số truyền vào phải là số thực.",
    yeu_cau_xac_nhan=True,
)
def cong_cu_di_chuyen_robot(x: float, y: float, z: float) -> str:
    """
    Giả lập điều khiển robot thông qua hệ thống ROS2 hoặc động học ngược (Inverse Kinematics).
    Trong tương lai, hàm này có thể import rospy hoặc rclpy.
    """
    # Tính toán khoảng cách (Euclide) từ gốc tọa độ
    khoang_cach = math.sqrt(x**2 + y**2 + z**2)

    # Giả lập thời gian di chuyển (Robot moving...)
    thoi_gian_du_kien = khoang_cach * 0.1

    return (
        f"[Robotics OS] Đã gửi lệnh điều khiển. "
        f"Robot đang di chuyển đến tọa độ (X={x:.2f}, Y={y:.2f}, Z={z:.2f}). "
        f"Dự kiến mất {thoi_gian_du_kien:.2f} giây."
    )


@cong_cu(
    ten="quet_radar_3d",
    mo_ta="Kích hoạt cảm biến LiDAR hoặc Camera độ sâu để thu thập điểm đám mây (Point Cloud) tại không gian hiện tại.",
    yeu_cau_xac_nhan=False,
)
def cong_cu_quet_radar_3d() -> str:
    """
    Giả lập thu thập dữ liệu không gian.
    Dữ liệu này trong thực tế sẽ được nạp vào SpatialVectorStore.
    """
    return "[LiDAR] Đã quét thành công khu vực xung quanh. Phát hiện 3 vật thể xung quanh tọa độ hiện tại. Đã cập nhật vào cơ sở dữ liệu Spatial RAG."
