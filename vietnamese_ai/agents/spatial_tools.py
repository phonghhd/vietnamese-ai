import math
from typing import Dict, List

import numpy as np

from vietnamese_ai.agents.tools import cong_cu
from vietnamese_ai.rag.spatial_store import SpatialVectorStore

_SPATIAL_STORE = SpatialVectorStore(kich_thuoc=16)


def giai_dong_hoc_nguoc_3dof(
    x: float, y: float, z: float, l1: float = 5.0, l2: float = 5.0
) -> Dict[str, float]:
    """
    Giải bài toán Động học ngược (Inverse Kinematics - IK) thực tế cho cánh tay robot 3 bậc tự do (3-DOF):
    - theta1 (Yaw đế): quay quanh trục Z
    - theta2 (Pitch vai): góc nâng khớp vai
    - theta3 (Pitch khuỷu): góc gập khớp khuỷu
    """
    theta1 = math.degrees(math.atan2(y, x))
    r_xy = math.hypot(x, y)
    d = math.hypot(r_xy, z)
    max_reach = l1 + l2
    d_clamped = min(max(d, abs(l1 - l2) + 1e-6), max_reach - 1e-6)

    cos_theta3 = (d_clamped**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))
    theta3_rad = math.acos(cos_theta3)

    alpha = math.atan2(z, r_xy)
    beta = math.atan2(l2 * math.sin(theta3_rad), l1 + l2 * math.cos(theta3_rad))
    theta2_rad = alpha - beta

    return {
        "theta1_deg": round(theta1, 2),
        "theta2_deg": round(math.degrees(theta2_rad), 2),
        "theta3_deg": round(math.degrees(theta3_rad), 2),
        "khoang_cach": round(d, 4),
        "trong_tam_voi": float(d <= max_reach),
    }


@cong_cu(
    ten="di_chuyen_robot",
    mo_ta="Điều khiển robot vật lý (cánh tay máy hoặc xe tự hành) di chuyển tới tọa độ (X, Y, Z). Tham số truyền vào phải là số thực.",
    yeu_cau_xac_nhan=True,
)
def cong_cu_di_chuyen_robot(x: float, y: float, z: float) -> str:
    """
    Tính toán góc khớp bằng Động học ngược (3-DOF Inverse Kinematics) và phát lệnh điều khiển.
    """
    khoang_cach = math.sqrt(x**2 + y**2 + z**2)
    thoi_gian_du_kien = khoang_cach * 0.1
    ik = giai_dong_hoc_nguoc_3dof(x, y, z)

    return (
        f"[Robotics OS] Đã gửi lệnh điều khiển. "
        f"Robot đang di chuyển đến tọa độ (X={x:.2f}, Y={y:.2f}, Z={z:.2f}). "
        f"Góc khớp IK 3-DOF: (θ1={ik['theta1_deg']:.2f}°, θ2={ik['theta2_deg']:.2f}°, θ3={ik['theta3_deg']:.2f}°). "
        f"Dự kiến mất {thoi_gian_du_kien:.2f} giây."
    )


@cong_cu(
    ten="quet_radar_3d",
    mo_ta="Kích hoạt cảm biến LiDAR hoặc Camera độ sâu để thu thập điểm đám mây (Point Cloud) tại không gian hiện tại.",
    yeu_cau_xac_nhan=False,
)
def cong_cu_quet_radar_3d() -> str:
    """
    Thu thập đám mây điểm 3D và nạp trực tiếp các thực thể không gian vào SpatialVectorStore.
    """
    vat_the: List[tuple[str, tuple[float, float, float], str]] = [
        ("lidar_obj_1", (1.2, 0.5, 0.0), "Bàn làm việc phía trước"),
        ("lidar_obj_2", (-0.8, 1.5, 0.2), "Ghế xoay bên trái"),
        ("lidar_obj_3", (0.0, 2.8, 1.0), "Màn hình hiển thị trung tâm"),
    ]
    for ma, coords, mo_ta in vat_the:
        vec = np.zeros(16, dtype=np.float32)
        vec[:3] = np.asarray(coords, dtype=np.float32)
        vec[3] = float(np.linalg.norm(coords))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        _SPATIAL_STORE.chen_khong_gian(
            ma=ma,
            vector=vec,
            toa_do=coords,
            metadata={"noi_dung": mo_ta, "loai": "lidar_point_cloud"},
        )

    return (
        f"[LiDAR] Đã quét thành công khu vực xung quanh. "
        f"Phát hiện {len(vat_the)} vật thể xung quanh tọa độ hiện tại. "
        f"Đã cập nhật vào cơ sở dữ liệu Spatial RAG (tổng {_SPATIAL_STORE.so_luong()} thực thể)."
    )
