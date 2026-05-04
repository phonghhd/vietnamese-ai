# NLP Extensions API Reference

## NhanDienThucThe

```python
class NhanDienThucThe:
    """Nhận diện thực thể có tên (NER) cho tiếng Việt."""

    def __init__(
        su_dung_underthesea=True,      # Sử dụng underthesea NER
        mau_tuy_chinh=None,            # Dict[str, List[str]] - Custom regex patterns
        tu_dien_tuy_chinh=None,        # Dict[str, set] - Custom dictionaries
    )

    def nhan_dien(van_ban, loai_loc=None) -> list
    def them_dia_danh(*ten) -> None
    def them_chuc_danh(*ten) -> None
    def them_mau(loai, mau) -> None
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `su_dung_underthesea` | bool | `True` | Dùng underthesea cho PERSON entity |
| `mau_tuy_chinh` | dict | `None` | Custom regex patterns `{loai: [pattern]}` |
| `tu_dien_tuy_chinh` | dict | `None` | Custom dictionaries (`dia_danh`, `chuc_danh`) |

### Entity Types

| Loai | Mô tả | Nguồn |
|---|---|---|
| `NGAY_THANG` | Ngày tháng | Regex |
| `SO_DIEN_THOAI` | Số điện thoại VN | Regex |
| `EMAIL` | Địa chỉ email | Regex |
| `URL` | Liên kết web | Regex |
| `TIEN_TE` | Số tiền VNĐ | Regex |
| `DIA_CHI` | Địa chỉ | Regex |
| `DIA_DANH` | Tỉnh/thành phố, quốc gia | Dictionary |
| `CHUC_DANH` | Chức danh nghề nghiệp | Dictionary |
| `PERSON` | Tên người | underthesea |

### `nhan_dien(van_ban, loai_loc=None) -> list`

```python
[
    {
        "van_ban": str,             # Text thực thể
        "loai": str,                # Loại thực thể
        "vi_tri_bat_dau": int,      # Vị trí bắt đầu
        "vi_tri_ket_thuc": int,     # Vị trí kết thúc
    },
]
```

---

## HoiDapTiengViet

```python
class HoiDapTiengViet:
    """Hệ thống hỏi đ��p (Question Answering) cho tiếng Việt."""

    def __init__(
        so_cau_toi_da=5,               # Số câu trả lời tối đa
        toi_thieu_diem=0.1,            # Ngưỡng điểm tối thiểu
    )

    def them_tai_lieu(ma, van_ban) -> int
    def hoi(cau_hoi, top_k=None) -> dict
    def xoa_tai_lieu(ma) -> bool
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `so_cau_toi_da` | int | `5` | Số kết quả tối đa |
| `toi_thieu_diem` | float | `0.1` | Ngưỡng điểm liên quan tối thiểu |

### `hoi(cau_hoi, top_k=None) -> dict`

```python
{
    "cau_hoi": str,                # Câu hỏi gốc
    "tra_loi": str,                # Câu trả lời tốt nhất
    "nguon": [                     # Danh sách nguồn
        {
            "cau": str,            # Nội dung câu
            "tai_lieu": str,       # ID tài liệu
            "diem": float,         # Điểm liên quan
        },
    ],
    "diem": float,                 # Điểm cao nhất
}
```

---

## TomTatVanBan

```python
class TomTatVanBan:
    """Tóm tắt văn bản tiếng Việt."""

    def __init__(
        che_do="extractive",           # "extractive", "abstractive"
        ham_sinh=None,                 # Callable[[str], str] - LLM cho abstractive
        trong_so_vi_tri=0.3,          # Trọng số vị trí câu
        trong_so_tfidf=0.5,           # Trọng số TF-IDF
        trong_so_do_dai=0.2,          # Trọng số độ dài câu
    )

    def tom_tat(van_ban, so_cau=3, toi_da_tu=None) -> dict
    def tom_tat_nhieu(van_ban_list, so_cau=3) -> list
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `che_do` | str | `"extractive"` | Chế độ tóm tắt |
| `ham_sinh` | Callable | `None` | Hàm sinh text (cho abstractive) |
| `trong_so_vi_tri` | float | `0.3` | Trọng số điểm vị trí |
| `trong_so_tfidf` | float | `0.5` | Trọng số điểm TF-IDF |
| `trong_so_do_dai` | float | `0.2` | Trọng số điểm độ dài |

### `tom_tat` Returns

```python
{
    "tom_tat": str,                # Văn bản tóm tắt
    "cac_cau_chon": list,          # Chỉ số các câu được chọn
    "ty_le_nen": float,            # Tỷ lệ nén (độ dài tóm tắt / gốc)
    "goc": str,                    # Văn bản gốc
    "che_do": str,                 # Chế độ đã dùng
}
```

---

## DichThuat

```python
class DichThuat:
    """Dịch thuật cho tiếng Việt."""

    def __init__(
        ham_sinh=None,                 # Callable[[str], str] - LLM cho dịch
        che_do="dictionary",           # "dictionary", "llm", "hybrid"
    )

    def dich(van_ban, nguon="en", dich="vi") -> dict
    def dich_batch(van_ban_list, nguon="en", dich="vi") -> list
    def them_tu_dien(key, tu_dien) -> None
    def lay_tu_dien(key) -> dict
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `ham_sinh` | Callable | `None` | Hàm sinh text cho LLM translation |
| `che_do` | str | `"dictionary"` | Chế độ dịch thuật |

### `dich` Returns

```python
{
    "goc": str,                    # Văn bản gốc
    "dich": str,                   # Bản dịch
    "nguon": str,                  # Ngôn ngữ nguồn (en, vi)
    "dich_lang": str,              # Ngôn ngữ đích
    "che_do": str,                 # Chế độ đã dùng
}
```

### Built-in Dictionary

Bao gồm từ điển EN-VI cơ bản (~50 từ) cho các thuật ngữ kỹ thuật. Mở rộng bằng `them_tu_dien`.

---

## KiemTraChinhTa

```python
class KiemTraChinhTa:
    """Kiểm tra và sửa lỗi chính tả tiếng Việt."""

    def __init(
        tu_dien=None,                  # Set[str] - Từ điển
        su_dung_underthesea=True,      # Dùng underthesea để tokenize
        toi_da_sua=5,                  # Số gợi ý tối đa
        nguong_khoang_cach=2,          # Edit distance tối đa
    )

    def kiem_tra(van_ban, sua_tu_dong=False) -> dict
    def sua(van_ban) -> str
    def them_tu_dien(tu) -> None
    def them_tu(*tu) -> None
    def huan_luyen_tu_corpus(van_ban_list) -> None
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `tu_dien` | set | `None` | Tập hợp từ đúng |
| `su_dung_underthesea` | bool | `True` | Dùng underthesea cho word_tokenize |
| `toi_da_sua` | int | `5` | Số gợi ý sửa tối đa cho mỗi lỗi |
| `nguong_khoang_cach` | int | `2` | Edit distance tối đa để gợi ý |

### `kiem_tra(van_ban, sua_tu_dong=False) -> dict`

```python
{
    "van_ban": str,                # Văn bản gốc
    "da_sua": str,                 # Văn bản đã sửa (nếu sua_tu_dong=True)
    "loi": [                       # Danh sách lỗi
        {
            "tu": str,             # Từ sai
            "vi_tri": int,         # Vị trí trong câu
            "goi_y": list,         # Danh sách gợi ý sửa
        },
    ],
    "so_loi": int,
    "ty_le_loi": float,
}
```

### `huan_luyen_tu_corpus(van_ban_list) -> None`

Học từ điển từ corpus. Từ xuất hiện >= 2 lần sẽ được thêm vào từ điển.
