"""
PluginCoSo - Khung xương (Base Class) cho mọi phần mở rộng của EvoNet-Studio.
Các lập trình viên bên ngoài khi viết Plugin phải kế thừa lớp này.
"""

from typing import Any, Dict


class PluginCoSo:
    """Lớp cơ sở bắt buộc cho các Plugin."""

    # Meta data của Plugin
    ten: str = "Chưa đặt tên"
    phien_ban: str = "1.0.0"
    mo_ta: str = "Mô tả ngắn về plugin"

    def __init__(self):
        self.trang_thai_hoat_dong = False

    def khoi_dong(self) -> bool:
        """
        Được gọi khi hệ thống vừa nạp Plugin lên.
        Dùng để khởi tạo biến, thiết lập ban đầu.

        Returns:
            True nếu khởi động thành công, False nếu thất bại.
        """
        self.trang_thai_hoat_dong = True
        return True

    def thuc_thi(self, **kwargs) -> Dict[str, Any]:
        """
        Hàm xử lý logic chính của Plugin.

        Args:
            **kwargs: Dữ liệu động truyền từ Framework hoặc Workflow.

        Returns:
            Kết quả tính toán.
        """
        raise NotImplementedError("Plugin phải cài đặt hàm thuc_thi()")

    def tat(self):
        """
        Được gọi khi Quản trị viên gỡ bỏ (Unload) Plugin khỏi hệ thống.
        Dùng để đóng kết nối CSDL, giải phóng bộ nhớ.
        """
        self.trang_thai_hoat_dong = False
