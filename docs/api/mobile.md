# Mobile/Edge API Reference

## TriKhaiDiDong

```python
class TriKhaiDiDong:
    """Triển khai mô hình lên thiết bị di động và edge."""

    def xuat_tflite(mo_hinh, duong_dan, kich_thuoc_dau_vao=(None,), luong_hoa=False) -> str
    def xuat_coreml(mo_hinh, duong_dan, kich_thuoc_dau_vao=(None,), ten_mo_hinh="Model") -> str
    def xuat_onnx_mobile(mo_hinh, duong_dan, kich_thuoc_dau_vao=(None,)) -> str
    def luong_hoa_int8(duong_dan_goc, duong_dan_moi) -> str
    def doc_mo_hinh_di_dong(duong_dan) -> dict
    def benchmark_edge(mo_hinh, kich_thuoc_dau_vao, so_lan=100) -> dict
    def tao_config_deploy(ten_mo_hinh, dinh_dang="tflite", duong_dan="deploy/mobile") -> str
```

### Parameters

| Parameter | Type | Mô tả |
|---|---|---|
| `mo_hinh` | Any | Mô hình sklearn hoặc wrapper có `._mo_hinh` |
| `duong_dan` | str | Đường dẫn file output |
| `kich_thuoc_dau_vao` | tuple | Shape input, VD: `(5,)` hoặc `(None, 5)` |
| `luong_hoa` | bool | Có quantize INT8 không |
| `so_lan` | int | Số lần benchmark |

### Returns

- `xuat_*`: Đường dẫn file đã lưu
- `benchmark_edge`: Dict với `thoi_gian_trung_binh_ms`, `throughput_mau_giay`, etc.
- `doc_mo_hinh_di_dong`: Dict metadata (dinh_dang, loai_mo_hinh, kich_thuoc_file)
