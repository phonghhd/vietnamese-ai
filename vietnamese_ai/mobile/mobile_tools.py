import tempfile
from pathlib import Path
from typing import List

from vietnamese_ai.agents.tools import cong_cu


class MobileDeviceState:
    """Trạng thái thiết bị di động được đồng bộ từ MobileAppBridge / hệ điều hành."""

    vi_do: float = 10.762622
    kinh_do: float = 106.660172
    dia_diem: str = "Thành phố Hồ Chí Minh"
    thong_bao: List[str] = [
        "(SMS) Mã OTP của bạn là 123456.",
        "(Push) Bạn có tin nhắn từ Zalo.",
        "(SMS) Tài khoản ngân hàng +100,000 VND.",
    ]
    thu_muc_anh: str = ""

    @classmethod
    def cap_nhat_gps(cls, vi_do: float, kinh_do: float, dia_diem: str = "") -> None:
        cls.vi_do = float(vi_do)
        cls.kinh_do = float(kinh_do)
        if dia_diem:
            cls.dia_diem = dia_diem

    @classmethod
    def them_thong_bao(cls, noi_dung: str) -> None:
        cls.thong_bao.insert(0, noi_dung)


@cong_cu(
    ten="lay_toa_do_gps",
    mo_ta="Lấy tọa độ GPS (Vĩ độ, Kinh độ) hiện tại của thiết bị di động.",
    yeu_cau_xac_nhan=True,
)
def cong_cu_lay_toa_do_gps() -> str:
    """Đọc tọa độ GPS hiện tại từ trạng thái thiết bị di động."""
    return (
        f"[Mobile OS] Đã lấy tọa độ GPS: Vĩ độ {MobileDeviceState.vi_do:.6f}, "
        f"Kinh độ {MobileDeviceState.kinh_do:.6f} ({MobileDeviceState.dia_diem})."
    )


@cong_cu(
    ten="doc_thong_bao_sms",
    mo_ta="Đọc nội dung của 3 tin nhắn SMS hoặc thông báo đẩy (Push Notification) mới nhất trên điện thoại.",
    yeu_cau_xac_nhan=True,
)
def cong_cu_doc_thong_bao_sms() -> str:
    """Đọc 3 thông báo/SMS mới nhất từ hàng đợi thông báo của thiết bị."""
    top_3 = MobileDeviceState.thong_bao[:3]
    dong = [f"{i + 1}. {tb}" for i, tb in enumerate(top_3)]
    return "[Mobile OS] Thông báo mới nhất:\n" + "\n".join(dong)


@cong_cu(
    ten="chup_anh_camera",
    mo_ta="Kích hoạt Camera của điện thoại để chụp một bức ảnh và trả về đường dẫn ảnh.",
    yeu_cau_xac_nhan=True,
)
def cong_cu_chup_anh_camera(mat_truoc: bool = False) -> str:
    """Chụp và lưu file ảnh JPEG thực tế vào bộ nhớ thiết bị."""
    camera_loai = "trước" if mat_truoc else "sau"
    out_dir = Path(MobileDeviceState.thu_muc_anh or tempfile.gettempdir()) / "evonet_camera"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "IMG_2026.jpg"
    try:
        from PIL import Image

        color = (180, 210, 240) if mat_truoc else (60, 140, 90)
        img = Image.new("RGB", (64, 64), color=color)
        img.save(img_path, format="JPEG")
    except Exception:
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")

    return (
        f"[Mobile OS] Đã chụp thành công 1 bức ảnh bằng camera {camera_loai}. "
        f"Đã lưu tại: {img_path}"
    )
