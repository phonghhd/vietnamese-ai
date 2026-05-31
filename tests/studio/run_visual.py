import os
import sys

# Đảm bảo có thể import vietnamese_ai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from vietnamese_ai.studio.builder import StudioKeoTha


def main():
    print("🚀 Đang khởi tạo V-Neural Visual Builder (EvoNet-Studio)...")

    # 1. Khởi tạo Studio
    studio = StudioKeoTha(ten="EvoNet-Studio (Visual Builder Demo)")

    # 2. Tạo nhanh một vài Node mẫu (Để có thứ hiển thị trên màn hình ngay khi load)
    # Node Nguồn dữ liệu
    n_data = studio.them_node("du_lieu", "Tải Dữ Liệu CSV", vi_tri={"x": -200, "y": 0})

    # Node Tiền xử lý
    n_prep = studio.them_node("tien_xu_ly", "Chuẩn hóa Z-Score", vi_tri={"x": 0, "y": -100})

    # Node Mô hình (Agent)
    n_model = studio.them_node("mo_hinh", "Agent Phân Loại", vi_tri={"x": 200, "y": 0})

    # Node Xuất / Biểu đồ
    n_chart = studio.them_node("truc_quan_hoa", "Biểu Đồ Kết Quả", vi_tri={"x": 400, "y": -100})

    # 3. Kết nối các Node lại với nhau
    studio.ket_noi(n_data["ma"], n_prep["ma"])
    studio.ket_noi(n_prep["ma"], n_model["ma"])
    studio.ket_noi(n_model["ma"], n_chart["ma"])

    # 4. Kích hoạt Web Server giao diện
    print("✅ Đã tạo xong DAG mẫu. Đang khởi động Server...")
    studio.chay_giao_dien_web(port=5000)


if __name__ == "__main__":
    main()
