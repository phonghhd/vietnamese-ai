# NLP Extensions

## Tổng quan

Bộ công cụ NLP tiếng Việt:

- **NhanDienThucThe** - Named Entity Recognition (NER)
- **HoiDapTiengViet** - Question Answering
- **TomTatVanBan** - Text Summarization
- **DichThuat** - Translation
- **KiemTraChinhTa** - Spell checking

---

## NhanDienThucThe

Nhận diện thực thể có tên (NER) cho tiếng Việt.

### Cơ bản

```python
from vietnamese_ai import NhanDienThucThe

ner = NhanDienThucThe()
ket_qua = ner.nhan_dien("Nguyễn Văn A sống tại Hà Nội từ 01/01/2024, SĐT: 0912345678")

for entity in ket_qua:
    print(f"{entity['van_ban']} ({entity['loai']})")
# Nguyễn Văn A (PERSON)
# Hà Nội (DIA_DANH)
# 01/01/2024 (NGAY_THANG)
# 0912345678 (SO_DIEN_THOAI)
```

### Loại thực thể hỗ trợ

| Loại | Mô tả | Ví dụ |
|---|---|---|
| `NGAY_THANG` | Ngày tháng | 01/01/2024, ngày 1 tháng 1 năm 2024 |
| `SO_DIEN_THOAI` | Số điện thoại | 0912345678, +84912345678 |
| `EMAIL` | Email | user@example.com |
| `URL` | URL | https://example.com |
| `TIEN_TE` | Tiền tệ | 100.000 đồng, 5 triệu VND |
| `DIA_CHI` | Địa chỉ | 123 Nguyễn Huệ đường |
| `DIA_DANH` | Địa danh | Hà Nội, TP.HCM, Việt Nam |
| `CHUC_DANH` | Chức danh | Giám đốc, Giáo viên |
| `PERSON` | Tên người (underthesea) | Nguyễn Văn A |

### Lọc theo loại

```python
ket_qua = ner.nhan_dien(van_ban, loai_loc=["NGAY_THANG", "TIEN_TE"])
```

### Tùy chỉnh

```python
ner = NhanDienThucThe(
    su_dung_underthesea=True,
    mau_tuy_chinh={"MA_SO_THUE": [r"\d{10}(?:-\d{3})?"]},
    tu_dien_tuy_chinh={"dia_danh": {"Cần Thơ", "An Giang"}},
)

ner.them_dia_danh("Bình Dương", "Đồng Nai")
ner.them_chuc_danh("CEO", "CTO")
ner.them_mau("SO_TAI_KHOAN", r"\d{6,14}")
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `su_dung_underthesea` | `True` | Dùng underthesea cho PERSON |
| `mau_tuy_chinh` | `None` | Regex patterns tùy chỉnh |
| `tu_dien_tuy_chinh` | `None` | Dictionary tùy chỉnh |

---

## HoiDapTiengViet

Hệ thống hỏi đáp (Question Answering) cho tiếng Việt.

### Cơ bản

```python
from vietnamese_ai import HoiDapTiengViet

qa = HoiDapTiengViet(so_cau_toi_da=5, toi_thieu_diem=0.1)

qa.them_tai_lieu("doc1", "Học máy là nhánh của AI. Máy học từ dữ liệu.")
qa.them_tai_lieu("doc2", "Học sâu sử dụng mạng nơ-ron nhiều lớp.")

ket_qua = qa.hoi("Học máy là gì?")
print(ket_qua["tra_loi"])
print(ket_qua["nguon"])
```

### Kết quả

```python
{
    "cau_hoi": "Học máy là gì?",
    "tra_loi": "Học máy là nhánh của AI",
    "nguon": [
        {"cau": "Học máy là nhánh của AI", "tai_lieu": "doc1", "diem": 0.85},
        ...
    ],
    "diem": 0.85,
}
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `so_cau_toi_da` | `5` | Số câu trả lời tối đa |
| `toi_thieu_diem` | `0.1` | Ngưỡng điểm tối thiểu |

---

## TomTatVanBan

Tóm tắt văn bản tiếng Việt.

### Extractive (mặc định)

```python
from vietnamese_ai import TomTatVanBan

tom_tat = TomTatVanBan(che_do="extractive")
ket_qua = tom_tat.tom_tat(van_ban_dai, so_cau=3)

print(ket_qua["tom_tat"])
print(f"Tỷ lệ nén: {ket_qua['ty_le_nen']:.2%}")
```

### Abstractive (với LLM)

```python
tom_tat = TomTatVanBan(che_do="abstractive", ham_sinh=my_llm)
ket_qua = tom_tat.tom_tat(van_ban, toi_da_tu=100)
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `che_do` | `"extractive"` | `extractive` hoặc `abstractive` |
| `so_cau` | `3` | Số câu tóm tắt (extractive) |
| `toi_da_tu` | `None` | Giới hạn số từ |
| `trong_so_vi_tri` | `0.3` | Trọng số vị trí câu |
| `trong_so_tfidf` | `0.5` | Trọng số TF-IDF |
| `trong_so_do_dai` | `0.2` | Trọng số độ dài câu |

### Tóm tắt nhiều văn bản

```python
ket_qua_list = tom_tat.tom_tat_nhieu([van_ban_1, van_ban_2], so_cau=2)
```

---

## DichThuat

Dịch thuật cho tiếng Việt.

### Dictionary-based

```python
from vietnamese_ai import DichThuat

dich = DichThuat(che_do="dictionary")
ket_qua = dich.dich("hello world", nguon="en", dich="vi")
print(ket_qua["dich"])
# "xin chào thế giới"
```

### LLM-based

```python
dich = DichThuat(che_do="llm", ham_sinh=my_llm)
ket_qua = dich.dich("Machine learning is fascinating", nguon="en", dich="vi")
```

### Hybrid

```python
dich = DichThuat(che_do="hybrid", ham_sinh=my_llm)
# Dịch từ điển trước, dùng LLM cho từ chưa có trong từ điển
```

### Tùy chỉnh từ điển

```python
dich.them_tu_dien("en_vi", {
    "transformer": "mô hình biến đổi",
    "attention": "cơ chế chú ý",
    "embedding": "nhúng",
})
```

### Batch dịch

```python
ket_qua = dich.dich_batch(["hello", "world", "AI"], nguon="en", dich="vi")
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `che_do` | `"dictionary"` | `dictionary`, `llm`, `hybrid` |
| `ham_sinh` | `None` | Hàm sinh LLM |

---

## KiemTraChinhTa

Kiểm tra và sửa lỗi chính tả tiếng Việt.

### Cơ bản

```python
from vietnamese_ai import KiemTraChinhTa

kt = KiemTraChinhTa()
kt.them_tu_dien({"xin chào", "thế giới", "học máy", "trí tuệ"})

ket_qua = kt.kiem_tra("xin chao the gioi hoc may")
print(f"Lỗi: {ket_qua['so_loi']}")
for loi in ket_qua["loi"]:
    print(f"  '{loi['tu']}' -> gợi ý: {loi['goi_y']}")
```

### Sửa tự động

```python
van_ban_sua = kt.sua("xin chao the gioi")
print(van_ban_sua)
# "xin chào thế giới"
```

### Học từ corpus

```python
corpus = [
    "Trí tuệ nhân tạo đang phát triển mạnh.",
    "Học máy giúp máy tính học từ dữ liệu.",
]
kt.huan_luyen_tu_corpus(corpus)
```

### Lỗi phổ biến được sửa tự động

| Sai | Đúng |
|---|---|
| khong | không |
| duoc | được |
| cua | của |
| nguoi | người |
| hoc | học |
| lam | làm |
| nhung | nhưng |
| muon | muốn |

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `su_dung_underthesea` | `True` | Dùng underthesea để tách từ |
| `toi_da_sua` | `5` | Số gợi ý tối đa mỗi từ |
| `nguong_khoang_cach` | `2` | Edit distance tối đa |

---

## Kết hợp NLP Pipeline

```python
from vietnamese_ai import (
    NhanDienThucThe, HoiDapTiengViet,
    TomTatVanBan, KiemTraChinhTa,
)

ner = NhanDienThucThe()
qa = HoiDapTiengViet()
tom_tat = TomTatVanBan()
kt = KiemTraChinhTa()

van_ban = "Học máy là nhánh của AI, giúp máy tính học từ dữ liệu."

entities = ner.nhan_dien(van_ban)
qa.them_tai_lieu("doc", van_ban)
cau_tra_loi = qa.hoi("Học máy là gì?")
tom_tat_van_ban = tom_tat.tom_tat(van_ban, so_cau=1)
van_ban_sua = kt.sua(van_ban)
```
