# Prompts API Reference

## MauPrompt

```python
class MauPrompt:
    """Quản lý prompt templates với biến và conditional logic."""

    def __init__(
        mau,                           # str - Template string (dùng {{bien}})
        ten="custom",                  # str - Tên template
        mo_ta="",                      # str - Mô tả
        bien_mac_dinh=None,            # Dict[str, str] - Giá trị mặc định
    )

    def render(**kwargs) -> str
    def danh_sach_bien() -> list
    def co_bien(ten_bien) -> bool
    def them_bien_mac_dinh(ten, gia_tri) -> None
    def from_template(ten, mau, mo_ta="") -> MauPrompt   # classmethod
    def lay_template(ten) -> Optional[MauPrompt]          # classmethod
    def danh_sach_template() -> list                       # classmethod
    def danh_sach_mau_mac_dinh() -> dict                   # classmethod
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `mau` | str | — | Template string với `{{ten_bien}}` placeholders |
| `ten` | str | `"custom"` | Tên định danh template |
| `mo_ta` | str | `""` | Mô tả mục đích template |
| `bien_mac_dinh` | dict | `None` | Giá trị mặc định cho các biến |

### Template Syntax

```python
mau = MauPrompt("Tóm tắt văn bản: {{noi_dung}} trong {{so_cau}} câu")
prompt = mau.render(noi_dung="Văn bản...", so_cau=3)
```

### `danh_sach_mau_mac_dinh` Returns

Tạo và trả về các template mặc định: `tom_tat`, `phan_tich`, `dich_thuat`, `hoi_dap`, `sinh_code`, `phan_biet`.

---

## ChuoiPrompt

```python
class ChuoiPrompt:
    """Chain prompt - kết hợp nhiều prompt thành pipeline."""

    def __init__(
        ham_sinh_mac_dinh=None,        # Callable[[str], str]
    )

    def them_buoc(ten, mau, dieu_kien=None) -> ChuoiPrompt
    def them_few_shot(dau_vao, dau_ra) -> ChuoiPrompt
    def thuc_hien(ham_sinh=None, bien=None) -> dict
    def tao_cot_prompt(cau_hoi, ngu_canh="") -> str
    def tao_few_shot_prompt(cau_hoi, vi_du=None) -> str
    def lay_lich_su() -> list
```

### Methods

#### `them_buoc(ten, mau, dieu_kien=None) -> ChuoiPrompt`

Thêm một bước vào chain. `dieu_kien` là callable nhận context dict, trả về `bool`.

```python
chain = ChuoiPrompt()
chain.them_buoc("phan_tich", mau_phan_tich)
chain.them_buoc("tom_tat", mau_tom_tat)
```

#### `them_few_shot(dau_vao, dau_ra) -> ChuoiPrompt`

Thêm một ví dụ few-shot vào chain.

#### `thuc_hien(ham_sinh=None, bien=None) -> dict`

Thực hiện toàn bộ chain.

```python
{
    "ket_qua": str,              # Kết quả cuối cùng
    "buoc_thuc_hien": list,      # Danh sách bước đã thực hiện
    "so_buoc": int,              # Số bước đã chạy
}
```

#### `tao_cot_prompt(cau_hoi, ngu_canh="") -> str`

Tạo chain-of-thought prompt với 3 bước suy nghĩ.

#### `tao_few_shot_prompt(cau_hoi, vi_du=None) -> str`

Tạo few-shot prompt với ví dụ.

---

## LuongAnToan

```python
class LuongAnToan:
    """Guardrails cho LLM output - kiểm tra an toàn và chất lượng."""

    def __init(
        tu_cam=None,                   # List[str] - Từ cấm
        toi_da_do_dai=10000,           # Độ dài tối đa
        toi_thieu_do_dai=0,            # Độ dài tối thiểu
        chan_pii=False,                # Chặn thông tin cá nhân
        dinh_dang_yeu_cau=None,        # "json", "markdown", "code", "number"
        rules=None,                    # List[Callable] - Custom rules
    )

    def kiem_tra(noi_dung) -> dict
    def loc_pii(noi_dung) -> tuple
    def them_tu_cam(*tu) -> None
    def them_rule(rule) -> None
    def them_mau_pii(loai, mau) -> None
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `tu_cam` | list | `[]` | Danh sách từ cấm |
| `toi_da_do_dai` | int | `10000` | Số ký tự tối đa |
| `toi_thieu_do_dai` | int | `0` | Số ký tự tối thiểu |
| `chan_pii` | bool | `False` | Bật phát hiện PII |
| `dinh_dang_yeu_cau` | str | `None` | Định dạng bắt buộc |
| `rules` | list | `[]` | Custom validation rules |

### `kiem_tra(noi_dung) -> dict`

```python
{
    "an_toan": bool,             # True nếu không có lỗi
    "loi": list,                 # Danh sách lỗi
    "canh_bao": list,            # Cảnh báo (PII, ...)
    "so_loi": int,
    "so_canh_bao": int,
    "do_dai": int,
}
```

### `loc_pii(noi_dung) -> tuple`

Lọc PII (số điện thoại, email, CMND, số thẻ ngân hàng) khỏi nội dung.

**Returns:** `(noi_dung_da_loc, dict_pii_phat_hien)`

### Custom Rule Format

```python
def rule(noi_dung: str) -> Tuple[bool, str]:
    # Trả về (True, "") nếu OK, (False, "lý do") nếu vi phạm
    pass

guardrail.them_rule(rule)
```

---

## PhanTichDauRa

```python
class PhanTichDauRa:
    """Phân tích đầu ra có cấu trúc từ LLM."""

    def phan_tich_json(van_ban) -> Optional[Any]
    def phan_tich_bang(van_ban) -> list
    def phan_tich_danh_sach(van_ban) -> list
    def phan_tich_code_blocks(van_ban) -> list
    def phan_tich_key_value(van_ban) -> dict
    def phan_tich_so(van_ban) -> list
    def trich_cau_tra_loi(van_ban) -> str
```

### Methods

#### `phan_tich_json(van_ban) -> Optional[Any]`

Parse JSON từ text. Tự động tìm JSON trong code block, `{...}`, hoặc `[...]`.

#### `phan_tich_bang(van_ban) -> list`

Parse markdown table.

```python
[{"cot1": "giá trị", "cot2": "giá trị"}, ...]
```

#### `phan_tich_danh_sach(van_ban) -> list`

Parse numbered hoặc bulleted list. Trả về `List[str]`.

#### `phan_tich_code_blocks(van_ban) -> list`

```python
[{"ngon_ngu": "python", "code": "print('hello')"}, ...]
```

#### `phan_tich_key_value(van_ban) -> dict`

Parse key-value pairs (định dạng `key: value`).

#### `phan_tich_so(van_ban) -> list`

Trích xuất tất cả số từ text. Trả về `List[float]`.

#### `trich_cau_tra_loi(van_ban) -> str`

Trích xuất câu trả lời chính từ LLM output (bỏ qua reasoning).
