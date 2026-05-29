"""
Tutorial 4: Production Hardening
=================================

Hướng dẫn: health check, circuit breaker, logging, metrics, warm-up.
"""

import numpy as np

# === 1. Health Check ===
from vietnamese_ai import KiemTraSucKhoe

health = KiemTraSucKhoe()

# Đăng ký checks
health.dang_ky_check("model_loaded", lambda: True, "Model đã load")
health.dang_ky_check("disk_space", lambda: True, "Ổ đĩa đủ chỗ")
health.dang_ky_check("gpu_available", lambda: False, "GPU available", quan_trong=False)

ket_qua = health.kiem_tra()
print(f"Trạng thái: {ket_qua['trang_thai']}")
print(f"Chi tiết: {len(ket_qua['chi_tiet'])} checks")
print(f"Uptime: {ket_qua['uptime_s']:.1f}s")

# Readiness/Liveness
print(f"Ready: {health.ready()}")
print(f"Live: {health.live()}")

# === 2. Circuit Breaker ===

from vietnamese_ai import MachCat

cb = MachCat(so_loi_toi_da=3, timeout_phuc_hoi=5.0, ten="api_call")

# Sử dụng với exception handling
for i in range(5):
    try:
        with cb:
            if i < 2:
                pass  # Thành công
            else:
                raise RuntimeError("Simulated failure")
    except RuntimeError:
        pass

print(f"\nCircuit: {cb.trang_thai}")
print(f"Stats: {cb.lay_thong_ke()}")

# Sử dụng với fallback
def fallback_api(*args):
    return "cached_result"

cb2 = MachCat(so_loi_toi_da=2, ham_fallback=fallback_api)
cb2.ghi_nhan_loi()
cb2.ghi_nhan_loi()
result = cb2.thuc_hien(lambda: "should_not_reach")
print(f"Fallback result: {result}")

# === 3. Structured Logging ===

from vietnamese_ai import LoggerCauTruc

logger = LoggerCauTruc(ten="my-app", cap_do="INFO")

logger.info("Application started", {"version": "11.0.0"})
logger.them_context(request_id="req-123")
logger.info("Processing request")
logger.xoa_context()

# Timing
with logger.do_thoi_gian("model_inference"):
    import time
    time.sleep(0.01)

# === 4. Metrics (Prometheus-compatible) ===

from vietnamese_ai import QuanLyMetrics

metrics = QuanLyMetrics(ten="vietnamese-ai")

# Counters
metrics.counter("requests_total", {"endpoint": "/predict"})
metrics.counter("requests_total", {"endpoint": "/predict"})

# Gauges
metrics.gauge("active_models", 3)
metrics.gauge("queue_size", 42)

# Histograms
for latency in [10, 15, 20, 25, 30, 45, 50, 100]:
    metrics.histogram("latency_ms", latency)

# Stats
print(f"\nCounter: {metrics.lay_counter('requests_total', {'endpoint': '/predict'})}")
print(f"Gauge: {metrics.lay_gauge('active_models')}")
print(f"Histogram: {metrics.lay_histogram_stats('latency_ms')}")

# Export Prometheus
prom = metrics.export_prometheus()
print(f"\nPrometheus format ({len(prom)} chars):")
print(prom[:200] + "...")

# Export JSON
data = metrics.export_json()
print(f"\nJSON: {len(data['counters'])} counters, {len(data['gauges'])} gauges")

# === 5. Model Warm-up ===

from vietnamese_ai import LamNongModel, PhanLoai

# Tạo và train model
X = np.random.randn(50, 3)
y = (X[:, 0] > 0).astype(int)
model = PhanLoai(thuat_toan="logistic")
model.huan_luyen(X, y)

# Warm-up
warmup = LamNongModel(so_lan_warmup=3)
warmup.dang_ky_model("classifier", model, du_lieu_mau=X[:10])

ket_qua = warmup.lam_nong("classifier")
print(f"\nWarm-up: {ket_qua['trang_thai']}, {ket_qua['thoi_gian_ms']:.1f}ms")

# Lấy model (đã warm-up)
model_sansang = warmup.lay_model("classifier")
print(f"Model ready: {model_sansang.da_huan_luyen}")

print("\n✓ Tutorial 4 hoàn tất!")
