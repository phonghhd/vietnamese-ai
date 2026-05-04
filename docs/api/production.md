# Production API Reference

## KiemTraSucKhoe

```python
class KiemTraSucKhoe:
    """Hệ thống health check cho production deployment."""

    def __init__(
        timeout=5.0,                   # Timeout cho mỗi check (giây)
        mau_lich_su=100,               # Số mẫu lịch sử lưu trữ
    )

    def dang_ky_check(ten, ham, mo_ta="", quan_trong=True) -> None
    def kiem_tra() -> dict
    def ready() -> bool
    def live() -> bool
    def dang_ky_checks_mac_dinh(model=None) -> None
    def lich_su() -> list
    def ty_le_healthy() -> float
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `timeout` | float | `5.0` | Timeout cho mỗi health check (giây) |
| `mau_lich_su` | int | `100` | Số mẫu lịch sử tối đa |

### `dang_ky_check(ten, ham, mo_ta="", quan_trong=True)`

| Parameter | Type | Mô tả |
|---|---|---|
| `ten` | str | Tên check |
| `ham` | Callable | Hàm check, trả về `True` nếu OK |
| `mo_ta` | str | Mô tả check |
| `quan_trong` | bool | Nếu `True`, check fail → trạng thái `unhealthy` |

### `kiem_tra()` Returns

```python
{
    "trang_thai": str,             # "healthy" | "degraded" | "unhealthy"
    "chi_tiet": {                  # Chi tiết từng check
        "ten_check": {
            "trang_thai": str,     # "ok" | "fail" | "error"
            "quan_trong": bool,
            "mo_ta": str,
        },
    },
    "he_thong": {                  # Thông tin hệ thống
        "cpu_load_1m": float,
        "memory_total_mb": int,
        "memory_available_mb": int,
        "memory_usage_pct": float,
        "disk_total_gb": float,
        "disk_free_gb": float,
        "python_version": str,
        "pid": int,
    },
    "thoi_gian_ms": float,         # Thời gian chạy checks
    "uptime_s": float,             # Uptime
}
```

### `ready() -> bool`

Readiness probe — service sẵn sàng nhận traffic.

### `live() -> bool`

Liveness probe — service còn sống.

---

## MachCat

```python
class MachCat:
    """Circuit Breaker pattern - ngăn cascade failures."""

    def __init__(
        so_loi_toi_da=5,              # Số lỗi tối đa trước khi mở
        timeout_phuc_hoi=30.0,         # Thời gian thử lại (giây)
        cua_so_thoi_gian=60.0,         # Cửa sổ đếm lỗi (giây)
        ham_fallback=None,             # Callable - Hàm fallback
        ten="default",                 # Tên circuit breaker
    )

    def cho_phep() -> bool
    def ghi_nhan_thanh_cong() -> None
    def ghi_nhan_loi() -> None
    def thuc_hien(ham, *args, **kwargs) -> Any
    def reset() -> None
    def lay_thong_ke() -> dict
```

### States (TrangThaiMach)

| Trạng thái | Giá trị | Mô tả |
|---|---|---|
| Đóng | `"dong"` | Bình thường, cho phép request |
| Mở | `"mo"` | Chặn request, chờ timeout phục hồi |
| Nửa mở | `"nua_mo"` | Thử nghiệm, cho 1 request qua |

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `so_loi_toi_da` | int | `5` | Số lỗi liên tiếp để mở circuit |
| `timeout_phuc_hoi` | float | `30.0` | Giây chờ trước khi thử lại |
| `cua_so_thoi_gian` | float | `60.0` | Cửa sổ đếm lỗi |
| `ham_fallback` | Callable | `None` | Hàm gọi khi circuit mở |

### Context Manager Usage

```python
circuit = MachCat(so_loi_toi_da=5)
with circuit:
    ket_qua = goi_api()
```

### `thuc_hien(ham, *args, **kwargs) -> Any`

Thực hiện hàm với circuit breaker protection. Nếu circuit mở, gọi `ham_fallback` hoặc raise `RuntimeError`.

### `lay_thong_ke` Returns

```python
{
    "trang_thai": str,               # "dong" | "mo" | "nua_mo"
    "tong_request": int,
    "thanh_cong": int,
    "that_bai": int,
    "bi_chan": int,                   # Số request bị chặn
    "fallback": int,                 # Số lần gọi fallback
    "chuyen_trang_thai": int,        # Số lần chuyển trạng thái
    "ty_le_thanh_cong": float,
    "ty_le_that_bai": float,
}
```

---

## LoggerCauTruc

```python
class LoggerCauTruc:
    """Structured logger cho production - output JSON format."""

    def __init__(
        ten="vietnamese-ai",           # Tên logger
        cap_do="INFO",                 # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        output="stdout",               # Đầu ra
        context_mac_dinh=None,         # Dict[str, Any] - Context mặc định
    )

    def debug(msg, data=None) -> None
    def info(msg, data=None) -> None
    def warning(msg, data=None) -> None
    def error(msg, data=None, exc_info=None) -> None
    def critical(msg, data=None, exc_info=None) -> None
    def do_thoi_gian(operation) -> _TimerContext
    def them_context(**kwargs) -> None
    def xoa_context() -> None
    def log_request(request_id, method, path, status, time_ms) -> None
    def log_prediction(model_name, input_size, time_ms, confidence=None) -> None
    def log_model_event(event, model_name, **kwargs) -> None
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `ten` | str | `"vietnamese-ai"` | Tên logger |
| `cap_do` | str | `"INFO"` | Mức log tối thiểu |
| `output` | str | `"stdout"` | Đầu ra log |
| `context_mac_dinh` | dict | `None` | Context thêm vào mọi log entry |

### Timing Context Manager

```python
logger = LoggerCauTruc()
with logger.do_thoi_gian("inference"):
    ket_qua = model.du_doan(X)
# Tự động log: {"operation": "inference", "time_ms": 45.2}
```

### Convenience Methods

| Method | Mô tả |
|---|---|
| `log_request(request_id, method, path, status, time_ms)` | Log HTTP request |
| `log_prediction(model_name, input_size, time_ms, confidence)` | Log model prediction |
| `log_model_event(event, model_name, **kwargs)` | Log model lifecycle event |

---

## QuanLyMetrics

```python
class QuanLyMetrics:
    """Hệ thống thu thập metrics cho production."""

    def __init__(
        ten="vietnamese_ai",           # Tên namespace metrics
    )

    def counter(ten, labels=None, gia_tri=1.0) -> None
    def gauge(ten, gia_tri, labels=None) -> None
    def histogram(ten, gia_tri, labels=None) -> None
    def timer(ten, labels=None) -> _Timer
    def lay_counter(ten, labels=None) -> float
    def lay_gauge(ten, labels=None) -> float
    def lay_histogram_stats(ten, labels=None) -> dict
    def export_prometheus() -> str
    def export_json() -> dict
    def reset() -> None
    def thong_ke() -> dict
```

### Metric Types

| Loại | Method | Mô tả |
|---|---|---|
| Counter | `counter(ten, labels, gia_tri)` | Đếm sự kiện (chỉ tăng) |
| Gauge | `gauge(ten, gia_tri, labels)` | Giá trị hiện tại (tăng/giảm) |
| Histogram | `histogram(ten, gia_tri, labels)` | Phân phối giá trị |
| Timer | `timer(ten, labels)` | Context manager đo thời gian |

### Timer Usage

```python
metrics = QuanLyMetrics()
with metrics.timer("inference_ms", {"model": "phobert"}):
    ket_qua = model.du_doan(X)
```

### `lay_histogram_stats` Returns

```python
{
    "count": int,
    "mean": float,
    "std": float,
    "min": float,
    "max": float,
    "p50": float,
    "p90": float,
    "p95": float,
    "p99": float,
}
```

### `export_prometheus() -> str`

Export metrics theo Prometheus text format.

### `export_json() -> dict`

```python
{
    "counters": dict,
    "gauges": dict,
    "histograms": dict,
    "uptime_s": float,
}
```

---

## LamNongModel

```python
class LamNongModel:
    """Model warm-up - pre-load và warm-up models trước khi nhận traffic."""

    def __init__(
        tu_dong_lam_nong=True,         # Tự động warm-up khi lấy model
        thoi_gian_refresh=3600.0,      # Chu kỳ refresh (giây)
        so_lan_warmup=3,               # Số lần chạy warm-up
    )

    def dang_ky_model(ten, model, du_lieu_mau=None, ham_du_doan=None, tu_dong_refresh=False) -> None
    def lam_nong(ten) -> dict
    def lam_nong_tat_ca() -> dict
    def lay_model(ten) -> Any
    def bat_dau_auto_refresh() -> None
    def dung_auto_refresh() -> None
    def xoa_model(ten) -> bool
    def trang_thai_model(ten) -> Optional[dict]
    def danh_sach_models() -> list
    def thong_ke() -> dict
```

### Parameters

| Parameter | Type | Mặc định | Mô tả |
|---|---|---|---|
| `tu_dong_lam_nong` | bool | `True` | Warm-up tự động khi gọi `lay_model` |
| `thoi_gian_refresh` | float | `3600.0` | Khoảng thời gian giữa các lần refresh |
| `so_lan_warmup` | int | `3` | Số lần chạy inference warm-up |

### `dang_ky_model` Arguments

| Parameter | Type | Mô tả |
|---|---|---|
| `ten` | str | Tên model |
| `model` | Any | Model object |
| `du_lieu_mau` | Any | Dữ liệu mẫu để warm-up inference |
| `ham_du_doan` | Callable | Custom predict function |
| `tu_dong_refresh` | bool | Tự động refresh trong background |

### `lam_nong(ten) -> dict`

```python
{
    "ten": str,                    # Tên model
    "thoi_gian_ms": float,         # Thời gian warm-up (ms)
    "so_lan": int,                 # Số lần đã chạy
    "trang_thai": str,             # "ok" | "loi"
    "loi": Optional[str],          # Thông báo lỗi (nếu có)
}
```

### `thong_ke` Returns

```python
{
    "so_models": int,              # Tổng số model
    "da_warmup": int,              # Số model đã warm-up
    "chua_warmup": int,            # Số model chưa warm-up
    "tong_thoi_gian_warmup_ms": float,
    "tu_dong_lam_nong": bool,
    "auto_refresh": bool,          # Background refresh đang chạy
}
```
