from vietnamese_ai.agents.tools import cong_cu


@cong_cu(
    ten="lay_toa_do_gps",
    mo_ta="Lấy tọa độ GPS (Vĩ độ, Kinh độ) hiện tại của thiết bị di động.",
    yeu_cau_xac_nhan=True,
)
def cong_cu_lay_toa_do_gps() -> str:
    """Giả lập việc gọi API GPS (CoreLocation trên iOS / LocationManager trên Android)."""
    return "[Mobile OS] Đã lấy tọa độ GPS: Vĩ độ 10.762622, Kinh độ 106.660172 (Thành phố Hồ Chí Minh)."


@cong_cu(
    ten="doc_thong_bao_sms",
    mo_ta="Đọc nội dung của 3 tin nhắn SMS hoặc thông báo đẩy (Push Notification) mới nhất trên điện thoại.",
    yeu_cau_xac_nhan=True,
)
def cong_cu_doc_thong_bao_sms() -> str:
    """Giả lập việc truy cập dữ liệu SMS."""
    return "[Mobile OS] Thông báo mới nhất:\n1. (SMS) Mã OTP của bạn là 123456.\n2. (Push) Bạn có tin nhắn từ Zalo.\n3. (SMS) Tài khoản ngân hàng +100,000 VND."


@cong_cu(
    ten="chup_anh_camera",
    mo_ta="Kích hoạt Camera của điện thoại để chụp một bức ảnh và trả về đường dẫn ảnh.",
    yeu_cau_xac_nhan=True,
)
def cong_cu_chup_anh_camera(mat_truoc: bool = False) -> str:
    """Giả lập gọi API Camera (AVFoundation / CameraX)."""
    camera_loai = "trước" if mat_truoc else "sau"
    return f"[Mobile OS] Đã chụp thành công 1 bức ảnh bằng camera {camera_loai}. Đã lưu tại: /storage/emulated/0/DCIM/Camera/IMG_2026.jpg"
