# Hướng Dẫn Tinh Chỉnh GPU Từ Xa Đa Nhà Cung Cấp (Remote GPU Flywheel)

**Remote GPU Flywheel** giải quyết bài toán thực tế cho máy văn phòng / laptop không có GPU rời: **thu thập & làm sạch dữ liệu hoàn toàn cục bộ** → **đẩy lên GPU đám mây hoặc máy chủ GPU nội bộ để tinh chỉnh LoRA/SFT/DPO** → **tải artifact GGUF/LoRA có kiểm định toàn vẹn về máy chạy offline trên CPU**.

---

## 1. Kiến trúc Khép kín 4 Lớp Bảo vệ

```text
[Dữ liệu Cục bộ (.jsonl / DataLake)]
               │
               ▼
  [1. Bảo mật & Tiền kiểm Cục bộ]
   ├─ DataSanitizer: Ẩn tự động SĐT, CCCD, Email, API key (PII)
   ├─ PhongThuDuLieu: Chặn mẫu đầu độc / sleeper backdoor
   └─ MayTinhTien: Kiểm trần ngân sách GPU (USD / VNĐ)
               │
               ▼
  [2. Cổng Đồng thuận Đám mây (Opt-in)]
   └─ Bắt buộc `cho_phep_gui_cloud=True` (`--allow-cloud`) nếu đẩy ra Modal
               │
               ▼
  [3. Nhà Cung Cấp GPU Đa Nền Tảng (NhaCungCapGPUXa)]
   ├─ modal : NhaCungCapModal  (Modal Serverless GPU T4/L4/A10G/A100/H100)
   ├─ ssh   : NhaCungCapSSH    (Máy chủ GPU nội bộ doanh nghiệp / trường học qua SSH+SCP)
   └─ local : NhaCungCapCucBo  (Subprocess cục bộ, 100% offline để thử nghiệm / CPU)
               │
               ▼
  [4. Kiểm định & Nạp lại Biên (Edge)]
   ├─ Xác minh SHA-256 + Ký số HMAC-SHA256 (`LuuTai.ky_file`)
   ├─ Eval Gate kiểm chất lượng trước khi thăng hạng `QuanLyMoHinh`
   └─ Nạp trực tiếp vào `DongCoSuyLuanEdge` (`vai run`)
```

---

## 2. So sánh 3 Nhà Cung Cấp Tích Hợp Sẵn

| Nhà cung cấp | Biến CLI `--provider` | Lớp Python | Yêu cầu thêm | Khi nào nên dùng? |
| :--- | :---: | :--- | :--- | :--- |
| **Modal Serverless GPU** | `modal` | `NhaCungCapModal` | `pip install "vietnamese-ai[cloud-gpu]"` + `modal setup` | Cần thuê GPU theo giây (T4 ~`$0.59/h`, A10G ~`$1.10/h`, H100 ~`$3.95/h`), không tốn tiền chờ idle. |
| **Máy chủ SSH GPU Nội bộ** | `ssh` | `NhaCungCapSSH` | `ssh` + `scp` chuẩn hệ điều hành | Doanh nghiệp/cơ quan có sẵn server GPU nội bộ, yêu cầu dữ liệu không rời khỏi mạng riêng. |
| **Subprocess Cục bộ** | `local` | `NhaCungCapCucBo` | Không cần thêm gói | Kiểm thử pipeline đầu-cuối, tinh chỉnh nhỏ 100% offline không cần mạng. |

---

## 3. Cài đặt & Cấu hình Ban đầu

```bash
# Cài gói hỗ trợ Modal Serverless GPU (tùy chọn, chỉ cần khi dùng --provider modal)
pip install "vietnamese-ai[cloud-gpu]"

# Hoặc dùng trình thiết lập thông minh 3 tầng
vai setup --profile cloud-gpu --apply

# Đăng nhập tài khoản Modal một lần duy nhất
modal setup
```

Chuẩn bị tệp dữ liệu hội thoại dạng JSONL (`du_lieu_huan_luyen.jsonl`):

```json
{"instruction": "Thời giờ làm việc bình thường tối đa mỗi ngày là bao nhiêu?", "output": "Theo Điều 105 Bộ luật Lao động 2019, thời giờ làm việc bình thường không quá 08 giờ/ngày và 48 giờ/tuần."}
{"instruction": "Thông báo sự cố lộ dữ liệu cá nhân trong bao lâu?", "output": "Theo Điều 23 Nghị định 13/2023/NĐ-CP, bên kiểm soát dữ liệu phải thông báo cho Bộ Công an chậm nhất 72 giờ."}
```

---

## 4. Sử dụng qua CLI (`vai train-cloud`)

### Bước 1: Chạy xem trước Dry-Run an toàn (mặc định)

Mặc định `vai train-cloud` **không bao giờ tự gửi dữ liệu hay tốn tiền GPU** nếu chưa truyền `--apply`. Lệnh sẽ tự ẩn PII, ước tính chi phí USD/VNĐ và sinh sẵn script huấn luyện để bạn rà soát:

```bash
vai train-cloud \
  --data du_lieu_huan_luyen.jsonl \
  --provider modal \
  --model qwen2.5-1.5b \
  --gpu T4 \
  --epochs 3 \
  --max-budget-usd 2.0
```

### Bước 2: Chạy thật trên Modal Serverless GPU

Cần bật tường minh cả hai cờ an toàn `--allow-cloud` (xác nhận gửi dữ liệu đã ẩn PII lên đám mây ngoài) và `--apply`:

```bash
vai train-cloud \
  --data du_lieu_huan_luyen.jsonl \
  --provider modal \
  --model qwen2.5-1.5b \
  --task sft \
  --gpu T4 \
  --epochs 3 \
  --rank 8 \
  --quantization q8_0 \
  --max-budget-usd 2.0 \
  --allow-cloud \
  --apply \
  --output vai_remote_jobs
```

### Bước 3: Chạy trên Máy chủ GPU Nội bộ qua SSH

Dữ liệu ở trong mạng riêng của tổ chức nên không cần `--allow-cloud`:

```bash
vai train-cloud \
  --data du_lieu_huan_luyen.jsonl \
  --provider ssh \
  --ssh-host 192.168.1.50 \
  --ssh-user gpu-admin \
  --ssh-port 22 \
  --ssh-key ~/.ssh/id_ed25519 \
  --model vgpt2 \
  --epochs 3 \
  --apply
```

### Bước 4: Thử nghiệm toàn bộ vòng lặp Cục bộ Offline

```bash
vai train-cloud \
  --data du_lieu_huan_luyen.jsonl \
  --provider local \
  --model vgpt2 \
  --epochs 2 \
  --apply
```

---

## 5. Sử dụng qua Python API & `QuyTrinhDauCuoi`

### Điều phối trực tiếp bằng `BoDieuPhoiTinhChinhXa`

```python
from vietnamese_ai import (
    BoDieuPhoiTinhChinhXa,
    CauHinhTinhChinhXa,
    NhaCungCapSSH,
)

dieu_phoi = BoDieuPhoiTinhChinhXa(thu_muc_lam_viec="vai_remote_jobs")

# (Tùy chọn) Đăng ký máy chủ SSH GPU nội bộ
dieu_phoi.dang_ky_nha_cung_cap(
    "ssh",
    NhaCungCapSSH(host="192.168.1.50", user="gpu-admin", port=22),
)

cau_hinh = CauHinhTinhChinhXa(
    mo_hinh="vgpt2",
    nhiem_vu="sft",
    loai_gpu="T4",
    so_vong=3,
    rank_lora=8,
    quantization="q8_0",
    ngan_sach_toi_da_usd=1.5,
    an_pii_truoc_khi_gui=True,
    loc_dau_doc_du_lieu=True,
    cho_phep_gui_cloud=False,
)

# chạy dry-run (chay_that=False) để kiểm tra báo cáo bảo mật & chi phí
kq = dieu_phoi.chay_vong_lap_flywheel(
    nguon_du_lieu="du_lieu_huan_luyen.jsonl",
    ten_nha_cung_cap="modal",
    cau_hinh=cau_hinh,
    chay_that=False,
)

print("Trạng thái:", kq.trang_thai)
print("Chi phí ước tính:", kq.bao_cao_chi_phi)
print("An ninh dữ liệu:", kq.bao_cao_an_ninh_du_lieu)
print("Script sinh ra:", kq.duong_dan_script)
```

### Nối liền với `QuyTrinhDauCuoi` (`EndToEndWorkflow`)

```python
from vietnamese_ai import QuyTrinhDauCuoi

wf = QuyTrinhDauCuoi(thu_muc_lam_viec="vai_workspace")
kq = wf.tinh_chinh_gpu_xa(
    nguon_du_lieu=[
        {"instruction": "Hỏi về Luật Đất đai 2024", "output": "Luật 31/2024/QH15 có hiệu lực từ 01/08/2024."},
        {"instruction": "Liên hệ 0912345678", "output": "Số điện thoại sẽ tự động được ẩn PII trước khi gửi."},
    ],
    nha_cung_cap="local",
    loai_gpu="CPU",
    so_vong=2,
    chay_that=True,
)
print("GGUF đã tải về & ký HMAC:", kq["duong_dan_gguf"])
```

---

## 6. Điều khiển qua Web Studio HTTP API

Khởi động Web Studio bằng `vai studio --port 8080`, sau đó gọi trực tiếp các endpoint:

| Endpoint | Phương thức | Chức năng |
| :--- | :---: | :--- |
| `/api/cloud-gpu/providers` | `GET` | Kiểm tra trạng thái sẵn sàng của `modal`, `ssh`, `local` và bảng giá GPU tham chiếu. |
| `/api/cloud-gpu/plan` | `POST` | Lập kế hoạch dry-run: lọc PII, ước tính chi phí USD/VNĐ và sinh script huấn luyện. |
| `/api/cloud-gpu/run` | `POST` | Thực thi tác vụ tinh chỉnh GPU từ xa và trả về đường dẫn artifact GGUF + mã băm SHA-256. |

---

## 7. Khép Kín Bánh Đà Dữ Liệu Có Kiểm Định (Thumbs Up/Down → DPO → Eval Gate 4 Trục)

Sau khi người dùng đánh giá Đúng/Sai (`thumbs_up` / `thumbs_down` kèm đáp án sửa `cau_tra_loi_sua`) trong Web Studio Workspace (hoặc từ hội thoại Zalo/Telegram đã lưu vào Workspace SQLite), bạn có thể nối thẳng sang GPU Flywheel chỉ trong **1 lệnh** bằng cờ `--from-workspace` và `--eval-4-axis`:

```bash
# Cách 1 (1 lệnh hợp nhất): Xuất DPO đã ẩn PII từ Workspace -> đẩy lên GPU -> chấm Eval Gate 4 trục
vai train-cloud \
  --from-workspace vai_studio_workspaces \
  --workspace phap_che \
  --provider modal \
  --gpu T4 \
  --max-budget-usd 2.0 \
  --eval-4-axis

# Cách 2 (tách từng bước độc lập):
vai export-dpo --workspace-dir vai_studio_workspaces --output studio_dpo.jsonl
vai train-cloud --data studio_dpo.jsonl --task dpo --provider modal --gpu T4 --max-budget-usd 2.0 --eval-4-axis
vai eval-gate --limit 10 --min-score 0.65
```

```python
from vietnamese_ai import BoDieuPhoiTinhChinhXa, CongKiemDinhMoHinh, TroLyPhapLy

# Tự động trích xuất DPO từ Workspace SQLite và bật chấm 4 trục trước khi promote
dieu_phoi = BoDieuPhoiTinhChinhXa(thu_muc_lam_viec="vai_remote_jobs")
dpo_info = dieu_phoi.chuan_bi_du_lieu_tu_workspace("vai_studio_workspaces", an_pii=True)
print("Đã xuất mẫu DPO sạch:", dpo_info["so_mau_dpo"], "->", dpo_info["duong_dan_jsonl"])

gate = CongKiemDinhMoHinh()
kq_gate = gate.so_sanh_va_quyet_dinh(
    ung_vien=TroLyPhapLy(nap_kho_mo=True, dung_onnx_semantic=True),
    ban_goc=TroLyPhapLy(nap_kho_mo=True),
    gioi_han_mau=10,
)
print("Quyết định:", kq_gate.quyet_dinh, "| Điểm ứng viên:", kq_gate.diem_ung_vien)
```

---

## 8. Mở rộng Nhà Cung Cấp Tùy Chỉnh (RunPod / Lambda / Vast.ai / Kubernetes)

Bạn có thể kế thừa lớp trừu tượng `NhaCungCapGPUXa` và đăng ký bằng `dieu_phoi.dang_ky_nha_cung_cap("runpod", ...)` để dùng chung toàn bộ lớp lọc PII, kiểm ngân sách, xác minh SHA-256 và ký số HMAC-SHA256. Xem đầy đủ chữ ký API tại [Training & Fine-Tuning API Reference](../api/training.md).
