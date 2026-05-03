# Federated Learning API Reference

## HocLienKet

```python
class HocLienKet:
    """Federated Learning - Học liên kết phân tán."""

    def __init__(so_client=5, so_vong=10, ty_le_client=1.0, rieng_tu_differntial=0.0, seed=42)
    def huan_luyen(lop_mo_hinh, X, y, **tham_so) -> dict
    def du_doan(lop_mo_hinh, X, **tham_so) -> np.ndarray
    def lay_lich_su() -> list
    def bao_cao() -> str
```

### Parameters

| Parameter | Type | Mô tả |
|---|---|---|
| `so_client` | int | Số clients (>= 2) |
| `so_vong` | int | Số rounds (>= 1) |
| `ty_le_client` | float | Tỷ lệ clients mỗi round (0, 1] |
| `rieng_tu_differntial` | float | Mức độ noise cho DP (>= 0) |
| `lop_mo_hinh` | type | Class mô hình (PhanLoai, HoiQuy) |
| `tham_so` | dict | Tham số cho mô hình |

### Returns

`huan_luyen` returns:

```python
{
    "trong_so_toan_cuc": dict,
    "diem_toan_cuc": float,
    "so_vong": int,
    "so_client": int,
    "tong_thoi_gian": float,
    "lich_su": list[dict]
}
```
