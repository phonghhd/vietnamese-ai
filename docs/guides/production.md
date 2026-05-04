# Production Hardening

## Tổng quan

Bộ công cụ production-ready cho Vietnamese AI:

- **KiemTraSucKhoe** - Health check, readiness/liveness probes
- **MachCat** - Circuit breaker pattern
- **LoggerCauTruc** - Structured JSON logging
- **QuanLyMetrics** - Prometheus metrics export
- **LamNongModel** - Model warm-up

---

## KiemTraSucKhoe

Hệ thống health check cho production deployment.

### Cơ bản

```python
from vietnamese_ai import KiemTraSucKhoe

health = KiemTraSucKhoe()
health.dang_ky_check("model_loaded", lambda: model is not None, quan_trong=True)
health.dang_ky_check("disk_space", lambda: True, quan_trong=False)

ket_qua = health.kiem_tra()
print(ket_qua["trang_thai"])  # "healthy" | "degraded" | "unhealthy"
```

### Default checks

```python
health.dang_ky_checks_mac_dinh(model=model)
# Tự động đăng ký: model_loaded, model_inference, disk_space
```

### Readiness & Liveness probes

```python
if health.ready():
    print("Service sẵn sàng nhận traffic")

if health.live():
    print("Service còn sống")
```

### Kết quả

```python
{
    "trang_thai": "healthy",
    "chi_tiet": {
        "model_loaded": {"trang_thai": "ok", "quan_trong": True},
        "disk_space": {"trang_thai": "ok", "quan_trong": False},
    },
    "he_thong": {
        "cpu_load_1m": 0.5,
        "memory_total_mb": 16384,
        "memory_available_mb": 8192,
        "memory_usage_pct": 50.0,
        "disk_free_gb": 100.5,
    },
    "thoi_gian_ms": 1.23,
    "uptime_s": 3600.0,
}
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `timeout` | `5.0` | Timeout mỗi check (giây) |
| `mau_lich_su` | `100` | Số mẫu lịch sử giữ lại |

---

## MachCat

Circuit Breaker pattern - ngăn cascade failures.

### 3 trạng thái

```
Đóng (dong) ──[quá nhiều lỗi]──► Mở (mo) ──[timeout]──► Nửa mở (nua_mo)
     ▲                                                       │
     └────────────────[thành công]───────────────────────────┘
```

### Cơ bản

```python
from vietnamese_ai import MachCat

circuit = MachCat(
    so_loi_toi_da=5,
    timeout_phuc_hoi=30.0,
    ten="api_ngoai",
)

# Cách 1: Context manager
with circuit:
    ket_qua = goi_api_ngoai()

# Cách 2: thuc_hien
ket_qua = circuit.thuc_hien(goi_api_ngoai)
```

### Fallback function

```python
def fallback():
    return {"error": "Service tạm thời không khả dụng"}

circuit = MachCat(
    so_loi_toi_da=3,
    timeout_phuc_hoi=60.0,
    ham_fallback=fallback,
)

ket_qua = circuit.thuc_hien(goi_api_ngoai)
# Nếu circuit mở, trả về fallback()
```

### Kiểm tra trạng thái

```python
if circuit.cho_phep():
    # Được phép gọi
    pass
else:
    # Bị chặn
    pass

print(circuit.trang_thai)  # "dong" | "mo" | "nua_mo"

circuit.reset()  # Reset về trạng thái đóng
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `so_loi_toi_da` | `5` | Số lỗi tối đa trước khi mở |
| `timeout_phuc_hoi` | `30.0` | Thời gian chờ trước khi thử lại (giây) |
| `cua_so_thoi_gian` | `60.0` | Cửa sổ đếm lỗi (giây) |
| `ham_fallback` | `None` | Hàm fallback khi circuit mở |

### Thống kê

```python
stats = circuit.lay_thong_ke()
print(f"Trạng thái: {stats['trang_thai']}")
print(f"Tỷ lệ thành công: {stats['ty_le_thanh_cong']:.2%}")
print(f"Số lần bị chặn: {stats['bi_chan']}")
```

---

## LoggerCauTruc

Structured JSON logging cho production.

### Cơ bản

```python
from vietnamese_ai import LoggerCauTruc

logger = LoggerCauTruc(ten="vietnamese-ai", cap_do="INFO")

logger.info("Model loaded", {"model": "phobert", "time_ms": 150})
logger.warning("High latency", {"latency_ms": 500})
logger.error("Prediction failed", {"input": "..."}, exc_info=Exception("Lỗi"))
```

### Output JSON

```json
{"timestamp": 1714800000.0, "level": "INFO", "logger": "vietnamese-ai", "message": "Model loaded", "data": {"model": "phobert", "time_ms": 150}}
```

### Đo thời gian

```python
with logger.do_thoi_gian("inference"):
    ket_qua = model.du_doan(X)
# Tự động log: "inference hoàn tất" với time_ms
```

### Context fields

```python
logger.them_context(request_id="req_123", user_id="user_456")
logger.info("Processing request")  # Tự động kèm request_id, user_id
logger.xoa_context()
```

### Log helpers

```python
logger.log_request("req_1", "POST", "/predict", 200, 45.2)
logger.log_prediction("phobert", input_size=128, time_ms=30.5, confidence=0.95)
logger.log_model_event("loaded", "phobert", version="1.0")
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `ten` | `"vietnamese-ai"` | Tên logger |
| `cap_do` | `"INFO"` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `context_mac_dinh` | `{}` | Context mặc định cho mọi log |

---

## QuanLyMetrics

Hệ thống thu thập metrics - Prometheus compatible.

### Counter

```python
from vietnamese_ai import QuanLyMetrics

metrics = QuanLyMetrics(ten="vietnamese_ai")

metrics.counter("requests_total", {"endpoint": "/predict"})
metrics.counter("requests_total", {"endpoint": "/predict"})
print(metrics.lay_counter("requests_total", {"endpoint": "/predict"}))  # 2.0
```

### Gauge

```python
metrics.gauge("active_models", 3)
metrics.gauge("queue_size", 42)
```

### Histogram

```python
metrics.histogram("latency_ms", 45.2)
metrics.histogram("latency_ms", 120.5)

stats = metrics.lay_histogram_stats("latency_ms")
print(f"P50: {stats['p50']:.1f}ms")
print(f"P95: {stats['p95']:.1f}ms")
print(f"P99: {stats['p99']:.1f}ms")
```

### Timer

```python
with metrics.timer("inference_time_ms"):
    ket_qua = model.du_doan(X)
```

### Prometheus export

```python
prometheus_output = metrics.export_prometheus()
print(prometheus_output)
# # TYPE requests_total counter
# requests_total{endpoint="/predict"} 2.0
```

### JSON export

```python
json_output = metrics.export_json()
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `ten` | `"vietnamese_ai"` | Tên prefix metrics |

---

## LamNongModel

Model warm-up - pre-load và warm-up models trước khi nhận traffic.

### Cơ bản

```python
from vietnamese_ai import LamNongModel

warmup = LamNongModel(
    tu_dong_lam_nong=True,
    so_lan_warmup=3,
)

warmup.dang_ky_model("classifier", model, du_lieu_mau=X_sample)
warmup.dang_ky_model("ner", ner_model, du_lieu_mau=text_sample)

ket_qua = warmup.lam_nong_tat_ca()
for ten, kq in ket_qua.items():
    print(f"{ten}: {kq['trang_thai']} ({kq['thoi_gian_ms']:.1f}ms)")
```

### Lazy loading

```python
warmup = LamNongModel(tu_dong_lam_nong=True)
warmup.dang_ky_model("classifier", model, du_lieu_mau=X_sample)

model = warmup.lay_model("classifier")  # Tự warm-up nếu chưa
```

### Auto-refresh

```python
warmup = LamNongModel(thoi_gian_refresh=3600.0)  # Refresh mỗi giờ
warmup.dang_ky_model("classifier", model, du_lieu_mau=X_sample, tu_dong_refresh=True)
warmup.bat_dau_auto_refresh()
```

### Custom predict function

```python
warmup.dang_ky_model(
    "custom_model",
    model,
    du_lieu_mau=X_sample,
    ham_du_doan=lambda x: model.predict_proba(x),
)
```

### Tham số

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `tu_dong_lam_nong` | `True` | Tự warm-up khi lấy model |
| `thoi_gian_refresh` | `3600.0` | Khoảng thời gian auto-refresh (giây) |
| `so_lan_warmup` | `3` | Số lần chạy dummy prediction |

### Thống kê

```python
stats = warmup.thong_ke()
print(f"Models: {stats['so_models']}")
print(f"Đã warm-up: {stats['da_warmup']}")
print(f"Tổng thời gian: {stats['tong_thoi_gian_warmup_ms']:.1f}ms")
```

---

## Production Deployment Example

```python
from vietnamese_ai import (
    KiemTraSucKhoe, MachCat, LoggerCauTruc,
    QuanLyMetrics, LamNongModel,
)

logger = LoggerCauTruc(ten="my-ai-service", cap_do="INFO")
metrics = QuanLyMetrics(ten="my_ai_service")
health = KiemTraSucKhoe()
warmup = LamNongModel(so_lan_warmup=5)
circuit = MachCat(so_loi_toi_da=3, timeout_phuc_hoi=30.0)

warmup.dang_ky_model("classifier", model, du_lieu_mau=X_sample)
warmup.lam_nong_tat_ca()

health.dang_ky_checks_mac_dinh(model)

def predict(input_data):
    with metrics.timer("inference_ms"):
        with circuit:
            logger.info("Prediction request", {"input_size": len(input_data)})
            metrics.counter("predictions_total")
            model = warmup.lay_model("classifier")
            ket_qua = model.du_doan(input_data)
            return ket_qua

logger.info("Service started", {"port": 8000})
logger.info(f"Health: {health.kiem_tra()['trang_thai']}")
```
