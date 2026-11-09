# Hồ Dữ Liệu 16 Bộ Tập Dữ Liệu Tiếng Việt (DataLake - 16 Curated Datasets)

Vietnamese AI Framework tích hợp sẵn **Hồ Dữ Liệu (`HoDuLieu`)** gồm **16 bộ tập dữ liệu mẫu chuẩn hóa** phục vụ nghiên cứu, huấn luyện và đánh giá mô hình mà không cần tải dữ liệu từ các nguồn trôi nổi bên ngoài.

---

## 1. Danh Mục 16 Bộ Tập Dữ Liệu Tích Hợp Sẵn

| STT | Mã Dữ Liệu | Tên Bộ Tập Dữ Liệu | Số lượng mẫu | Tác vụ ứng dụng chính |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `phap_ly_hanh_chinh` | Văn bản Luật & Nghị định Việt Nam | 5,000 điều khoản | RAG pháp lý, QA văn bản thể chế |
| **2** | `sentiment_danh_gia` | Đánh giá phản hồi thương mại điện tử | 10,000 đánh giá | Phân loại cảm xúc (Tích cực / Tiêu cực / Trung tính) |
| **3** | `nhan_dien_thuc_the_ner` | Nhận diện thực thể tên riêng (NER) | 8,500 câu | Trích xuất Tên người, Tổ chức, Địa danh |
| **4** | `hoi_dap_y_te` | Tư vấn y khoa & Triệu chứng bệnh | 4,200 cặp Q&A | Chatbot y tế, hỗ trợ bác sĩ gia đình |
| **5** | `tin_tuc_bao_chi` | Phân loại chủ đề báo chí tiếng Việt | 12,000 bài báo | Phân loại tin tức (Kinh tế, Công nghệ, Thể thao...) |
| **6** | `tai_chinh_chung_khoan` | Báo cáo tài chính & Thị trường vốn | 3,800 đoạn trích | Phân tích báo cáo thường niên, cổ phiếu VNINDEX |
| **7** | `sua_loi_chinh_ta` | Cặp lỗi chính tả và phương ngữ | 15,000 cặp từ | Tự động sửa lỗi gõ máy, nhầm lẫn hỏi/ngã, s/x |
| **8** | `dich_thuat_song_ngu` | Song ngữ chuẩn Việt - Anh (En-Vi) | 20,000 cặp câu | Dịch máy nơ-ron, đối sánh ngữ nghĩa |
| **9** | `giong_noi_am_tiet` | Âm tiết và 6 thanh điệu tiếng Việt | 6,500 âm tiết | Huấn luyện ASR/TTS, xử lý ngữ âm học |
| **10** | `tom_tat_van_ban` | Văn bản dài và bản tóm tắt | 3,000 cặp tài liệu | Tóm tắt tin tức tự động, trích xuất ý chính |
| **11** | `an_toan_red_teaming` | Mẫu tấn công Jailbreak & Injection | 1,500 mẫu test | Đánh giá độ an toàn AI, kiểm thử thâm nhập |
| **12** | `giao_duc_toan_hoc` | Toán đố và lời giải từng bước | 2,800 bài toán | Suy luận logic theo chuỗi (Chain-of-Thought) |
| **13** | `dia_danh_viet_nam` | Danh mục địa giới hành chính chuẩn | 11,000 bản ghi | Chuẩn hoá địa chỉ Tỉnh/Huyện/Xã toàn quốc |
| **14** | `ca_dao_tuc_ngu` | Ca dao, tục ngữ & Thành ngữ dân gian | 4,000 câu | Đánh giá hiểu biết văn hóa, ẩn dụ tiếng Việt |
| **15** | `tro_chuyen_hang_ngay` | Hội thoại tự nhiên đời sống | 7,500 lượt thoại | Huấn luyện chatbot giao tiếp xã hội |
| **16** | `ma_nguon_python` | Code Python chú thích tiếng Việt | 2,200 snippets | Lập trình cùng AI, sinh mã nguồn tự động |

---

## 2. Hướng dẫn Nạp và Sử dụng Nhanh

Các bộ dữ liệu được tích hợp sẵn trong module `vietnamese_ai.datalake` và có thể tải nhanh chỉ với 1 dòng code:

```python
from vietnamese_ai.datalake import HoDuLieu

# Liệt kê toàn bộ các tập dữ liệu có sẵn
print(HoDuLieu.danh_sach_tap_du_lieu())

# Tải tập dữ liệu đánh giá cảm xúc
tap_sentiment = HoDuLieu.tai_ve("sentiment_danh_gia")
print(f"Số lượng mẫu: {len(tap_sentiment)}")
print("Mẫu đầu tiên:", tap_sentiment[0])

# Tải tập dữ liệu pháp lý hành chính cho RAG
tap_phap_ly = HoDuLieu.tai_ve("phap_ly_hanh_chinh")
print(f"Số điều luật: {len(tap_phap_ly)}")
```

---

## 3. Định Dạng Dữ Liệu Chuẩn Hoá

Mọi tập dữ liệu trong `HoDuLieu` đều tuân thủ định dạng từ điển chuẩn của Python (`Dict[str, Any]`), đã qua các bước xử lý:
- Chuẩn hoá bảng mã Unicode dựng sẵn (NFC).
- Khử trùng ký tự rác và định dạng ngày giờ theo quy chuẩn Việt Nam.
- Tương thích trực tiếp với các mô hình `PhanLoai`, `HoiQuy`, `DuongOngRAG` và `VietnameseLLM` của framework.
