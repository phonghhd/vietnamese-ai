# No-code Studio API Reference

## StudioKeoTha

```python
class StudioKeoTha:
    """No-code Studio kéo thả."""

    def __init__(ten="Vietnamese AI Studio")
    def them_node(loai, ten, vi_tri=None, tham_so=None) -> dict
    def xoa_node(ma_node) -> None
    def sua_node(ma_node, tham_so) -> dict
    def ket_noi(tu_node, den_node, cua="du_lieu") -> dict
    def huy_ket_noi(tu_node, den_node) -> None
    def chay() -> dict
    def lay_canvas() -> dict
    def danh_sach_loai_node() -> dict
    def danh_sach_thuat_toan() -> dict
    def lay_templates() -> dict
    def tai_template(ten_template) -> dict
    def luu(duong_dan) -> str
    def tai(duong_dan) -> StudioKeoTha  # classmethod
```

### Loại node

| Loai | Mô tả |
|---|---|
| `du_lieu` | Nguồn dữ liệu |
| `tien_xu_ly` | Chuẩn hóa, xử lý missing |
| `mo_hinh` | Huấn luyện mô hình |
| `danh_gia` | Đánh giá hiệu suất |
| `truc_quan_hoa` | Biểu đồ hóa |
| `xuat` | Lưu kết quả |
