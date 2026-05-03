# Vietnamese LLM API Reference

## VietnameseLLM

```python
class VietnameseLLM:
    """Mô hình ngôn ngữ tiếng Việt (N-gram LM)."""

    def __init__(bac=3, lam_mo=0.01, toi_thieu_dem=1)
    def huan_luyen(cac_van_ban, so_vong=1) -> dict
    def sinh_van_ban(khoi_dau="", do_dai=50, nhiet_do=1.0) -> str
    def hoan_thanh_cau(dau_vao, so_lua_chon=3, do_dai_toi_da=20, nhiet_do=0.8) -> list
    def tinh_perplexity(text) -> float
    def lay_tu_ke_tiep(text, top_n=5) -> list
    def sinh_theo_template(ten_template, tham_so) -> str
    def them_template(ten, mau) -> None
    def danh_sach_templates() -> dict
    def thong_ke() -> dict
    def luu(duong_dan) -> str
    def tai(duong_dan) -> VietnameseLLM  # classmethod
```

### Parameters

| Parameter | Type | Mô tả |
|---|---|---|
| `bac` | int | Độ bậc n-gram (>= 2) |
| `lam_mo` | float | Laplace smoothing (>= 0) |
| `nhiet_do` | float | Temperature (0=conservative, 2=creative) |
| `do_dai` | int | Số từ tối đa cần sinh |

### Templates có sẵn

| Tên | Mô tả |
|---|---|
| `tin_tuc` | Mẫu tin tức |
| `san_pham` | Mô tả sản phẩm |
| `dich_vu` | Mô tả dịch vụ |
| `mo_ta` | Mô tả chung |
| `hoi_dap` | Câu hỏi - trả lời |
