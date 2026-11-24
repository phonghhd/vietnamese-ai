"""WebXRNode - Nút Suy Luận Không Gian 3D (AR/VR) Với Phép Biến Đổi Euler Thực Tế."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from vietnamese_ai.edge.wasm_node import WebBrowserNode


class WebXRNode(WebBrowserNode):
    """
    Thực thể mở rộng của WebBrowserNode cho Thực tế ảo Tăng cường (AR/VR):
    - Tính toán vector hướng nhìn 3D thực tế từ góc Euler `(pitch, yaw, roll)` và khoảng cách Euclid.
    - Chuyển tiếp truy vấn kèm ngữ cảnh tọa độ không gian thực tế qua động cơ suy luận thực thi của `WebBrowserNode`.
    """

    def __init__(self, session_id: str, websocket_conn: Any = None) -> None:
        super().__init__(session_id=session_id, websocket_conn=websocket_conn)
        self.model_hash = "webxr_model_v1"
        self.toa_do_hien_tai: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.goc_nhin: Dict[str, float] = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    def cap_nhat_telemetry(self, telemetry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Nhận luồng dữ liệu 3D từ kính AR/VR hoặc trình duyệt WebXR và trả về trạng thái hình học."""
        if "toa_do" in telemetry_data:
            for truc in ("x", "y", "z"):
                if truc in telemetry_data["toa_do"]:
                    self.toa_do_hien_tai[truc] = float(telemetry_data["toa_do"][truc])
        if "goc_nhin" in telemetry_data:
            for goc in ("pitch", "yaw", "roll"):
                if goc in telemetry_data["goc_nhin"]:
                    self.goc_nhin[goc] = float(telemetry_data["goc_nhin"][goc])

        self.logger.info("Đã cập nhật vị trí WebXR: %s", self.toa_do_hien_tai)
        return {
            "toa_do": dict(self.toa_do_hien_tai),
            "goc_nhin": dict(self.goc_nhin),
            "huong_nhin_don_vi": self.tinh_vector_huong_nhin(),
            "khoang_cach_goc": self.tinh_khoang_cach_goc(),
        }

    def tinh_vector_huong_nhin(self) -> Tuple[float, float, float]:
        """Tính vector đơn vị hướng nhìn 3D `(dx, dy, dz)` từ góc Euler `(pitch, yaw)` theo độ."""
        pitch_rad = math.radians(self.goc_nhin["pitch"])
        yaw_rad = math.radians(self.goc_nhin["yaw"])
        dx = math.cos(pitch_rad) * math.sin(yaw_rad)
        dy = math.sin(pitch_rad)
        dz = math.cos(pitch_rad) * math.cos(yaw_rad)
        return (round(dx, 4), round(dy, 4), round(dz, 4))

    def tinh_khoang_cach_goc(self) -> float:
        """Tính khoảng cách Euclid từ vị trí hiện tại tới gốc tọa độ không gian."""
        x, y, z = self.toa_do_hien_tai["x"], self.toa_do_hien_tai["y"], self.toa_do_hien_tai["z"]
        return round(math.sqrt(x * x + y * y + z * z), 4)

    def sinh_van_ban(self, prompt: str, do_dai: int = 128) -> str:
        """Sinh phản hồi thực tế qua động cơ suy luận kèm ngữ cảnh tọa độ và hướng nhìn 3D tính toán."""
        huong = self.tinh_vector_huong_nhin()
        kc = self.tinh_khoang_cach_goc()
        tra_loi_thuc = super().sinh_van_ban(prompt, do_dai=do_dai)
        return (
            f"[WebXR-3D pos=({self.toa_do_hien_tai['x']:.1f},{self.toa_do_hien_tai['y']:.1f},"
            f"{self.toa_do_hien_tai['z']:.1f}) dir={huong} dist={kc}] {tra_loi_thuc}"
        )
