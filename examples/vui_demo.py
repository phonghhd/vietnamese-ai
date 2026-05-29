import random
import time

from vietnamese_ai.ui import (
    BangThongKe,
    BieuDoTriThuc,
    BoNhoTrinhDuyet,
    CameraVaMicro,
    ChiaSeManHinh,
    DieuKhienIoT,
    DinhViGPS,
    GiaoDienChat,
    KhuVucTaiTaiLieu,
    KieuDang,
    KinhThucTeAo,
    TinhToanGPU,
    TrangThaiPin,
    TrinhChieu3D,
    TroLyGiongNoi,
    UIApp,
)


def gia_lap_llm(cau_hoi: str) -> str:
    time.sleep(1)
    return f"Bạn vừa hỏi: '{cau_hoi}'.\n\nDưới đây là đoạn code mẫu Markdown:\n```python\nprint('Hello V-Neural')\n```"

def lay_thong_ke_depin() -> dict:
    return {
        "Số Node P2P": random.randint(15, 25),
        "Tốc độ mạng (Tokens/s)": round(random.uniform(150.5, 300.2), 2),
    }

def xu_ly_upload(ten_file: str, base64_content: str) -> str:
    return f"Đã lưu '{ten_file}' ({len(base64_content)} bytes) vào VectorDB."

def xu_ly_anh(base64_img: str) -> str:
    return "Phát hiện 1 người dùng đang test Camera!"

def lay_graph() -> dict:
    return {
        "nodes": [{"id": 1, "label": "V-Neural v20"}, {"id": 2, "label": "V-UI Zero"}],
        "edges": [{"from": 1, "to": 2}]
    }

def xu_ly_giong_noi(text: str) -> str:
    return f"Tôi nghe bạn nói là: {text}. Chào bạn nha!"

def xu_ly_man_hinh(b64: str) -> str:
    return f"Màn hình của bạn đang hiển thị mã code Python đúng không? ({len(b64)} bytes)"

def xu_ly_gps(lat: float, lng: float) -> str:
    return f"Bạn đang ở khu vực có vĩ độ {lat}, thời tiết hôm nay khá đẹp!"

def xu_ly_gpu(kq: float) -> str:
    return f"Đã offload tính toán sang WebGPU. Kết quả Float: {kq}"

def xu_ly_pin(pct: float) -> str:
    if pct < 20:
        return "CẢNH BÁO: Pin yếu, Agent đã tự động chuyển sang mô hình siêu tiết kiệm!"
    return "Năng lượng dồi dào, hệ thống đang chạy ở Max Performance."

if __name__ == "__main__":
    print("Khởi tạo V-UI Studio Ultimate (Zero-Dependency)...")

    app = UIApp(tieu_de="V-Neural Studio (Tương Lai)", theme=KieuDang.DARK)

    chat_box = GiaoDienChat(xu_ly_tin_nhan=gia_lap_llm, chieu_cao="400px")
    upload_zone = KhuVucTaiTaiLieu(ham_xu_ly_file=xu_ly_upload)
    camera = CameraVaMicro(ham_xu_ly_anh=xu_ly_anh)
    depin_stats = BangThongKe(tieu_de="Mạng lưới DePIN", ham_lay_du_lieu=lay_thong_ke_depin, cap_nhat_sau=3)
    graph = BieuDoTriThuc(ham_lay_graph=lay_graph)

    # Các Component Đa phương tiện & Viễn tưởng
    voice_ai = TroLyGiongNoi(ham_xu_ly_giong_noi=xu_ly_giong_noi)
    screen_share = ChiaSeManHinh(ham_xu_ly_man_hinh=xu_ly_man_hinh)
    gps_locator = DinhViGPS(ham_xu_ly_vi_tri=xu_ly_gps)
    db_cache = BoNhoTrinhDuyet()
    viewer_3d = TrinhChieu3D()

    gpu_compute = TinhToanGPU(ham_xu_ly_ket_qua=xu_ly_gpu)
    iot_ble = DieuKhienIoT()
    battery_api = TrangThaiPin(ham_canh_bao_pin=xu_ly_pin)
    xr_vr = KinhThucTeAo()

    app.them_cot(
        chat_box, voice_ai, screen_share, camera, viewer_3d, upload_zone,
        gps_locator, depin_stats, graph, gpu_compute, iot_ble, battery_api, xr_vr, db_cache
    )
    app.chay(port=8080)
