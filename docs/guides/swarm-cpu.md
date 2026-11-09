# Phân mảnh & Song song hoá Swarm CPU qua LAN (Swarm CPU Sharding)

Vietnamese AI Framework cung cấp kiến trúc **Swarm CPU** (`DieuPhoiSwarmCPU` và `NutSwarmCPU`), cho phép phân bổ và suy luận các mô hình ngôn ngữ lớn (LLM) trên mạng lưới phần cứng thông thường (CPU x86/ARM, máy tính văn phòng, Raspberry Pi, VPS giá rẻ) thông qua kết nối mạng nội bộ (LAN) mà không phụ thuộc vào GPU đắt đỏ.

---

## 1. Bài toán & Kiến trúc Swarm CPU

### Thách thức khi triển khai LLM trên CPU
Khi chạy các mô hình 7B, 14B hoặc 32B trên thiết bị biên:
- Dung lượng RAM của một máy đơn lẻ thường không đủ để tải toàn bộ trọng số (weights).
- Chi phí đầu tư GPU máy chủ (H100, A100) vượt quá ngân sách của nhiều cơ quan, trường đại học hoặc doanh nghiệp vừa và nhỏ.

### Giải pháp Pipeline Parallelism của Vietnamese AI
Vietnamese AI chia mô hình thành $N$ phân mảnh liên tiếp (layers sharding) và ánh xạ tới $N$ nút mạng CPU:
- **Nút Điều Phối (Master / Coordinator - `DieuPhoiSwarmCPU`)**: Tiếp nhận prompt từ người dùng, mã hoá token (`BoTachTuTiengVietSieuToc`), gửi trạng thái ẩn ban đầu tới Nút 1.
- **Các Nút Thợ (Worker Nodes - `NutSwarmCPU`)**:
  - *Nút 1*: Tính toán các Transformer layers từ $0 \to L_1$, truyền vector activation qua socket tới Nút 2.
  - *Nút 2*: Tiếp tục tính từ $L_1+1 \to L_2$, chuyển tiếp tới Nút tiếp theo.
  - *Nút cuối (Nút $N$)*: Tính toán lớp Language Model Head (LM Head), tính phân phối logits, sinh token mới và gửi kết quả về cho Nút Điều Phối.

```
[Prompt Người Dùng] ──> [DieuPhoiSwarmCPU (Master)]
                               │
               ┌───────────────┴───────────────┐
               ▼                               │ (Activation tensor qua LAN)
       [NutSwarmCPU 1]                         │
       (Layers 0 - 7)                          │
               │                               ▼
               ▼ (LAN TCP)              [NutSwarmCPU 2]
       [NutSwarmCPU 3]                  (Layers 8 - 15)
       (Layers 16 - 23)                        │
               │                               ▼
               └───────────────>        [NutSwarmCPU 4]
                                        (Layers 24 - 31 + LM Head)
                                               │
                                               ▼ (Token sinh ra)
                                    [DieuPhoiSwarmCPU] ──> [Phản hồi Streaming]
```

---

## 2. Hướng dẫn thiết lập cụm Swarm LAN

### Cấu hình Nút Thợ (Worker Node) trên từng máy
Mỗi máy tính trong mạng nội bộ chỉ cần chạy một tiến trình `NutSwarmCPU`:

```python
from vietnamese_ai.orchestrator.nut_swarm import NutSwarmCPU

# Khởi tạo nút mạng lắng nghe trên IP cục bộ và cổng mạng
nut = NutSwarmCPU(
    id_nut="nut_cpu_tang_1",
    ip="192.168.1.101",
    port=8890,
    so_luong_thread=8,      # Tối ưu hoá đa luồng CPU
    bo_nho_toi_da_gb=8.0,   # Giới hạn RAM an toàn
)

# Khởi động dịch vụ socket không đồng bộ
nut.khoi_dong()
```

### Cấu hình Nút Điều Phối (Coordinator)
Trên máy chủ trung tâm hoặc laptop chính:

```python
from vietnamese_ai.orchestrator.dieu_phoi_swarm import DieuPhoiSwarmCPU

dieu_phoi = DieuPhoiSwarmCPU(
    danh_sach_nut=[
        {"id": "nut_cpu_tang_1", "ip": "192.168.1.101", "port": 8890, "tang_bat_dau": 0, "tang_ket_thuc": 7},
        {"id": "nut_cpu_tang_2", "ip": "192.168.1.102", "port": 8890, "tang_bat_dau": 8, "tang_ket_thuc": 15},
        {"id": "nut_cpu_tang_3", "ip": "192.168.1.103", "port": 8890, "tang_bat_dau": 16, "tang_ket_thuc": 23},
        {"id": "nut_cpu_tang_4", "ip": "192.168.1.104", "port": 8890, "tang_bat_dau": 24, "tang_ket_thuc": 31},
    ],
    nhip_tim_giay=2.0,      # Kiểm tra sức khoẻ nhịp tim tự động
    tu_dong_chuyen_vung=True # Tự động fallback nếu 1 máy bị mất nguồn
)

# Phân bổ trọng số mô hình GGUF / SafeTensors tới các nút qua mạng
dieu_phoi.phan_manh_mo_hinh(duong_dan_model="models/vietnamese-llm-7b-q4.gguf")
```

---

## 3. Thực hiện suy luận phân tán & Streaming

```python
van_ban = "Quy định về thủ tục cấp giấy phép kinh doanh gồm những bước nào?"

# Suy luận streaming liên tục qua đường ống Swarm
for token in dieu_phoi.sinh_van_ban_dong(van_ban, max_new_tokens=150, temperature=0.7):
    print(token, end="", flush=True)
```

---

## 4. Cơ chế chịu lỗi (Fault-Tolerance) & Cân bằng tải

- **Nhịp tim UDP/TCP (Heartbeat Detection)**: Nếu một nút CPU đột ngột mất kết nối LAN (rút dây mạng, mất nguồn), `DieuPhoiSwarmCPU` phát hiện trong vòng < 2 giây.
- **Kế thừa tầng tính toán (Dynamic Layer Fallback)**:
  - Tự động gộp phân mảnh của nút bị lỗi sang nút lân cận có dung lượng RAM trống lớn nhất.
  - Tiếp tục hoàn tất câu trả lời của người dùng mà không làm đứt đoạn phiên hội thoại.
- **Zero Heavy Dependencies**: Toàn bộ socket networking được cài đặt bằng thư viện chuẩn `socket` và `asyncio` của Python, tuân thủ nghiêm ngặt **Triết lý Zero-Dependency**.
