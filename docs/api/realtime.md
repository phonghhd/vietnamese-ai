# Real-time Pipeline API Reference

## PipelineThoiGianThuc

```python
class PipelineThoiGianThuc:
    """Real-time ML Pipeline."""

    def __init__(ten="PipelineThoiGianThuc", kich_thuoc_buffer=1000, batch_size=32)
    def dang_ky_mo_hinh(ten, mo_hinh) -> None
    def gui_du_lieu(ten_mo_hinh, du_lieu, nhan=None) -> dict
    def du_doan(ten_mo_hinh, du_lieu) -> dict
    def du_doan_batch(ten_mo_hinh, du_lieu_batch) -> dict
    def lay_thong_ke() -> dict
    def lay_lich_su(ten_mo_hinh, so_luong=10) -> list
    def dang_ky_callback(ten_mo_hinh, callback) -> None
    def xoa_buffer() -> None
```

### Thống kê trả về

```python
{
    "so_du_doan": int,
    "so_loi": int,
    "latency_trung_binh_ms": float,
    "latency_p50_ms": float,
    "latency_p95_ms": float,
    "latency_p99_ms": float,
    "latency_max_ms": float,
}
```

## MessageQueue

```python
class MessageQueue:
    def publish(topic, du_lieu, key=None) -> dict
    def subscribe(topic, callback) -> None
    def consume(topic, so_luong=1) -> list
    def lay_thong_ke() -> dict
```

## FeatureStore

```python
class FeatureStore:
    def cap_nhat(ten, gia_tri, nhan=None) -> None
    def lay_features(ten) -> np.ndarray
    def lay_labels(ten) -> np.ndarray
    def lay_window(ten, kich_thuoc) -> np.ndarray
    def thong_ke() -> dict
```
