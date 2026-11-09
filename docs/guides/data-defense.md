# Phòng Thủ Dữ Liệu & Khiên Bảo Vệ Đa Lớp (Data Defense & Security Shields)

Khi triển khai các mô hình AI vào môi trường thực tế, hệ thống phải đối mặt với nhiều hình thức tấn công tinh vi: mã độc chèn trong tệp tin đa phương tiện (**Polyglot Malware**), bom giải nén (**Image Bombs**), tin ẩn mã hóa bit (**LSB Steganography**), và bom hẹn giờ trong tập huấn luyện (**Sleeper Backdoor**).

Vietnamese AI Framework tích hợp hệ thống **Khiên Bảo Vệ Đa Lớp** (Zero-Trust Security Shields) hoạt động độc lập, bảo vệ toàn diện từ cổng vào đến cổng ra.

---

## 1. Tổng quan Kiến trúc Phòng Thủ

```
[Dữ liệu Tải lên] ──> [KhienBaoVeHinhAnh] ──> Phát hiện Polyglot / Bom nén / Giấu tin LSB
                              │ (Hợp lệ)
                              ▼
                      [CongChatLuongDauVao] ──> Làm sạch Prompt Injection, Kiểm tra tiếng Việt
                              │ (An toàn)
                              ▼
                      [Lõi AI / RAG Pipeline]
                              │ (Kết quả suy luận)
                              ▼
                      [CongAnToanDauRa] ──> Che giấu CCCD / SĐT / Thông tin nhạy cảm (PII)
                              │
                              ▼
                      [Phản hồi Người Dùng]
```

---

## 2. Khiên Bảo Vệ Hình Ảnh (`KhienBaoVeHinhAnh`)

Lớp bảo vệ xử lý tập trung vào các tệp ảnh tải lên mô hình đa phương thức (Vision-Language Models):

- **Chống Bom Giải Nén (Image Decompression Bomb)**: Phát hiện các ảnh kích thước vài KB nhưng khi giải nén bung ra hàng trăm triệu pixel làm cạn kiệt bộ nhớ RAM máy chủ.
- **Phát hiện Polyglot Malware**: Quét cấu trúc nhị phân (Magic Bytes, Marker JFIF/PNG, Trailer EOF). Nếu phát hiện mã thực thi (PE, ELF, PHP, JavaScript, Shell script) nằm sau điểm kết thúc của ảnh, hệ thống tự động gắn cờ mã độc.
- **Phát hiện Giấu Tin LSB (Least Significant Bit Steganography)**: Phân tích phân phối entropy của các bit bậc thấp để phát hiện dữ liệu mã hóa bí mật được giấu trong các điểm ảnh.
- **Tái tạo Khử độc Dữ liệu (CDR - Content Disarm & Reconstruction)**: Giải mã ảnh và vẽ lại pixel sang một tệp nhị phân hoàn toàn mới, loại bỏ hoàn toàn các metadata độc hại hoặc payload đính kèm.

```python
from vietnamese_ai.security.shields import KhienBaoVeHinhAnh

khien_anh = KhienBaoVeHinhAnh(
    gioi_han_kich_thuoc_mb=10.0,
    gioi_han_diem_anh_toi_da=4096 * 4096,
    kiem_tra_polyglot=True,
    kiem_tra_lsb_steganography=True
)

ket_qua = khien_anh.kiem_tra_tep("anh_nguoi_dung.png")

if not ket_qua.an_toan:
    print(f"Cảnh báo vi phạm: {ket_qua.ly_do_vi_pham}")
    # Thực hiện CDR khử độc:
    anh_sach_bytes = khien_anh.tai_tao_khu_doc("anh_nguoi_dung.png")
```

---

## 3. Quét Bom Hẹn Giờ & Backdoor Dataset (`PhongThuDuLieu`)

Trong các quy trình Fine-Tuning hoặc Retrieval-Augmented Generation (RAG), kẻ tấn công có thể chèn các **Sleeper Backdoor** — các từ khóa kích hoạt hiếm gặp nhưng luôn đi kèm với câu trả lời độc hại hoặc thiên vị:

```python
from vietnamese_ai.security.data_defense import PhongThuDuLieu

phong_thu = PhongThuDuLieu(
    nguong_nang_do_lift=3.0,     # Hệ số liên kết bất thường giữa trigger và nhãn
    nguong_tin_cay=0.95
)

# Quét toàn bộ tập huấn luyện hoặc tài liệu RAG
cac_trigger_nghi_ngo = phong_thu.quet_backdoor(
    danh_sach_van_ban=tap_du_lieu_van_ban,
    nhan_muc_tieu=tap_nhan
)

for trigger in cac_trigger_nghi_ngo:
    print(f"Phát hiện Trigger tiềm ẩn: {trigger.tu_khoa} (Score: {trigger.diem_nghi_ngo:.2f})")
```

---

## 4. Cổng Kiểm Soát Đầu Vào & Đầu Ra (I/O Gates)

### Cổng Chất Lượng Đầu Vào (`CongChatLuongDauVao`)
- Ngăn chặn tấn công **Prompt Injection** và **Jailbreak** (ví dụ: *"Bỏ qua các chỉ dẫn trước đó và làm..."*).
- Chuẩn hóa Unicode tiếng Việt (NFC), loại bỏ ký tự điều khiển ẩn (Zero-width space, BiDi override).

### Cổng An Toàn Đầu Ra (`CongAnToanDauRa`)
- Che giấu tự động dữ liệu định danh cá nhân (PII): Số Căn cước công dân (12 chữ số), số điện thoại Việt Nam (+84 / 09x / 03x), số tài khoản ngân hàng, mã số thuế.
- Kiểm duyệt phát ngôn tiêu cực, ngôn từ thù địch trước khi gửi đến người dùng cuối.

```python
from vietnamese_ai.security.shields import CongAnToanDauRa

cong_ra = CongAnToanDauRa(an_pii=True)

noi_dung = "Căn cước công dân của ông Nguyễn Văn A là 001095012345, số điện thoại 0912345678."
noi_dung_an_toan = cong_ra.loc_noi_dung(noi_dung)

print(noi_dung_an_toan)
# Kết quả: "Căn cước công dân của ông Nguyễn Văn A là [CCCD_DA_AN], số điện thoại [SDT_DA_AN]."
```
