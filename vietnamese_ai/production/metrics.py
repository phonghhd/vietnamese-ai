"""QuanLyMetrics - Metrics collection cho production."""

import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np


class QuanLyMetrics:
    """
    Hệ thống thu thập metrics cho production.

    Hỗ trợ:
    - Counter (đếm events)
    - Gauge (giá trị hiện tại)
    - Histogram (phân phối giá trị)
    - Timer (đo thời gian)
    - Export Prometheus/OpenTelemetry format

    Sử dụng:
        >>> metrics = QuanLyMetrics()
        >>> metrics.counter("requests_total", {"endpoint": "/predict"})
        >>> metrics.histogram("latency_ms", 45.2)
        >>> metrics.gauge("active_models", 3)
        >>> print(metrics.export_prometheus())
    """

    def __init__(self, ten: str = "vietnamese_ai"):
        self.ten = ten
        self._lock = threading.Lock()

        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._labels: Dict[str, Dict[str, str]] = {}
        self._metadata: Dict[str, Dict[str, str]] = {}

        self._bat_dau = time.time()

    def counter(
        self,
        ten: str,
        labels: Optional[Dict[str, str]] = None,
        gia_tri: float = 1.0,
    ) -> None:
        """
        Tăng counter.

        Args:
            ten: Tên metric
            labels: Labels (endpoint, model, etc.)
            gia_tri: Giá trị tăng
        """
        key = self._tao_key(ten, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + gia_tri
            if labels:
                self._labels[key] = labels

    def gauge(
        self,
        ten: str,
        gia_tri: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set gauge value."""
        key = self._tao_key(ten, labels)
        with self._lock:
            self._gauges[key] = gia_tri
            if labels:
                self._labels[key] = labels

    def histogram(
        self,
        ten: str,
        gia_tri: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Thêm giá trị vào histogram."""
        key = self._tao_key(ten, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(gia_tri)
            if labels:
                self._labels[key] = labels

    def timer(
        self,
        ten: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> "_Timer":
        """Context manager đo thời gian."""
        return self._Timer(self, ten, labels)

    class _Timer:
        def __init__(
            self,
            metrics: "QuanLyMetrics",
            ten: str,
            labels: Optional[Dict[str, str]],
        ):
            self.metrics = metrics
            self.ten = ten
            self.labels = labels
            self.bat_dau = 0.0

        def __enter__(self) -> "QuanLyMetrics._Timer":
            self.bat_dau = time.time()
            return self

        def __exit__(self, *args: Any) -> None:
            elapsed = (time.time() - self.bat_dau) * 1000
            self.metrics.histogram(self.ten, elapsed, self.labels)

    def lay_counter(self, ten: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Lấy giá trị counter."""
        key = self._tao_key(ten, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def lay_gauge(self, ten: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Lấy giá trị gauge."""
        key = self._tao_key(ten, labels)
        with self._lock:
            return self._gauges.get(key, 0.0)

    def lay_histogram_stats(
        self,
        ten: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Lấy thống kê histogram."""
        key = self._tao_key(ten, labels)
        with self._lock:
            values = self._histograms.get(key, [])

        if not values:
            return {"count": 0}

        arr = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def _tao_key(
        self,
        ten: str,
        labels: Optional[Dict[str, str]],
    ) -> str:
        """Tạo metric key."""
        if not labels:
            return ten
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{ten}{{{label_str}}}"

    def export_prometheus(self) -> str:
        """Export metrics theo Prometheus format."""
        lines = []

        with self._lock:
            # Counters
            for key, value in self._counters.items():
                lines.append(f"# TYPE {key.split('{')[0]} counter")
                lines.append(f"{key} {value}")

            # Gauges
            for key, value in self._gauges.items():
                lines.append(f"# TYPE {key.split('{')[0]} gauge")
                lines.append(f"{key} {value}")

            # Histograms
            for key, values in self._histograms.items():
                base = key.split("{")[0]
                label = key[len(base):]
                lines.append(f"# TYPE {base} histogram")

                arr = np.array(values)
                for bucket in [1, 5, 10, 25, 50, 100, 250, 500, 1000]:
                    count = int(np.sum(arr <= bucket))
                    lines.append(f'{base}_bucket{label.rstrip("}")},le="{bucket}"}} {count}')

                lines.append(f"{base}_count{label} {len(values)}")
                lines.append(f"{base}_sum{label} {float(np.sum(arr))}")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export metrics theo JSON format."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self.lay_histogram_stats(k.split("{")[0])
                    for k in self._histograms
                },
                "uptime_s": time.time() - self._bat_dau,
            }

    def reset(self) -> None:
        """Reset tất cả metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._labels.clear()

    def thong_ke(self) -> Dict[str, Any]:
        return {
            "ten": self.ten,
            "so_counters": len(self._counters),
            "so_gauges": len(self._gauges),
            "so_histograms": len(self._histograms),
            "uptime_s": time.time() - self._bat_dau,
        }

    def __repr__(self) -> str:
        return (
            f"QuanLyMetrics(counters={len(self._counters)}, "
            f"gauges={len(self._gauges)}, "
            f"histograms={len(self._histograms)})"
        )
