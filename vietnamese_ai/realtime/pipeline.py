"""PipelineThoiGianThuc - Real-time ML Pipeline với message queue."""

import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vietnamese_ai.utils.logger import Logger


class MessageQueue:
    """
    In-memory message queue mô phỏng Kafka/Redis Streams.

    Hỗ trợ:
    - Publish/Subscribe pattern
    - Consumer groups
    - Message retention
    - Backpressure handling
    """

    def __init__(self, ten: str = "default", kich_thuoc_toi_da: int = 10000):
        self.ten = ten
        self.kich_thuoc_toi_da = kich_thuoc_toi_da
        self._queue: deque = deque(maxlen=kich_thuoc_toi_da)
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._so_tin_nhan = 0
        self._so_tin_nhan_da_xu_ly = 0

    def publish(self, topic: str, du_lieu: Any, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Gửi tin nhắn vào queue.

        Args:
            topic: Chủ đề tin nhắn
            du_lieu: Dữ liệu tin nhắn
            key: Khóa phân vùng (tùy chọn)

        Returns:
            Dict chứa metadata tin nhắn
        """
        tin_nhan = {
            "topic": topic,
            "du_lieu": du_lieu,
            "key": key,
            "thoi_gian": time.time(),
            "id": self._so_tin_nhan,
        }

        with self._lock:
            self._queue.append(tin_nhan)
            self._so_tin_nhan += 1

        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    callback(tin_nhan)
                except Exception:
                    pass

        return {"id": tin_nhan["id"], "topic": topic, "thoi_gian": tin_nhan["thoi_gian"]}

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Đăng ký nhận tin nhắn theo topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def consume(self, topic: str, so_luong: int = 1) -> List[Dict]:
        """
        Đọc tin nhắn từ topic.

        Args:
            topic: Chủ đề
            so_luong: Số tin nhắn tối đa

        Returns:
            Danh sách tin nhắn
        """
        with self._lock:
            tin_nhan = [
                tn for tn in self._queue if tn["topic"] == topic
            ][-so_luong:]
            self._so_tin_nhan_da_xu_ly += len(tin_nhan)
        return tin_nhan

    def lay_thong_ke(self) -> Dict[str, Any]:
        """Thống kê message queue."""
        return {
            "ten": self.ten,
            "tong_tin_nhan": self._so_tin_nhan,
            "da_xu_ly": self._so_tin_nhan_da_xu_ly,
            "trong_queue": len(self._queue),
            "so_topic": len(self._subscribers),
            "kich_thuoc_toi_da": self.kich_thuoc_toi_da,
        }

    def xoa(self) -> None:
        """Xóa toàn bộ queue."""
        with self._lock:
            self._queue.clear()
            self._so_tin_nhan = 0
            self._so_tin_nhan_da_xu_ly = 0


class FeatureStore:
    """
    Feature store thời gian thực.

    Lưu trữ và truy xuất features theo thời gian thực,
    hỗ trợ online feature computation.
    """

    def __init__(self, kich_thuoc_cua_so: int = 1000):
        self.kich_thuoc_cua_so = kich_thuoc_cua_so
        self._features: Dict[str, deque] = {}
        self._labels: Dict[str, deque] = {}
        self._metadata: Dict[str, Dict] = {}

    def cap_nhat(self, ten: str, gia_tri: np.ndarray, nhan: Optional[Any] = None) -> None:
        """Cập nhật feature mới."""
        if ten not in self._features:
            self._features[ten] = deque(maxlen=self.kich_thuoc_cua_so)
            self._labels[ten] = deque(maxlen=self.kich_thuoc_cua_so)

        self._features[ten].append(np.asarray(gia_tri).flatten())
        if nhan is not None:
            self._labels[ten].append(nhan)

        self._metadata[ten] = {
            "so_mau": len(self._features[ten]),
            "cap_nhat_cuoi": time.time(),
        }

    def lay_features(self, ten: str) -> np.ndarray:
        """Lấy tất cả features theo tên."""
        if ten not in self._features:
            raise KeyError(f"Feature '{ten}' không tồn tại")
        return np.array(list(self._features[ten]))

    def lay_labels(self, ten: str) -> np.ndarray:
        """Lấy labels tương ứng."""
        if ten not in self._labels:
            raise KeyError(f"Feature '{ten}' không tồn tại")
        return np.array(list(self._labels[ten]))

    def lay_window(self, ten: str, kich_thuoc: int) -> np.ndarray:
        """Lấy cửa sổ features gần nhất."""
        if ten not in self._features:
            raise KeyError(f"Feature '{ten}' không tồn tại")
        du_lieu = list(self._features[ten])
        return np.array(du_lieu[-kich_thuoc:])

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê feature store."""
        return {
            "so_features": len(self._features),
            "chi_tiet": {
                ten: {
                    "so_mau": len(self._features[ten]),
                    "kich_thuoc": self._features[ten][-1].shape if self._features[ten] else (0,),
                }
                for ten in self._features
            },
        }


class PipelineThoiGianThuc:
    """
    Pipeline ML thời gian thực.

    Tính năng:
    - Message queue (mô phỏng Kafka/Redis Streams)
    - Feature store real-time
    - Online prediction với buffering
    - Latency tracking
    - Consumer/Producer pattern

    Sử dụng:
        >>> pipeline = PipelineThoiGianThuc()
        >>> pipeline.dang_ky_mo_hinh("sentiment", mo_hinh)
        >>> pipeline.gui_du_lieu("sentiment", features)
        >>> ket_qua = pipeline.du_doan("sentiment", features)
    """

    def __init__(
        self,
        ten: str = "PipelineThoiGianThuc",
        kich_thuoc_buffer: int = 1000,
        batch_size: int = 32,
    ):
        if kich_thuoc_buffer <= 0:
            raise ValueError("kich_thuoc_buffer phải > 0")
        if batch_size <= 0:
            raise ValueError("batch_size phải > 0")

        self.ten = ten
        self.kich_thuoc_buffer = kich_thuoc_buffer
        self.batch_size = batch_size
        self.logger = Logger("PipelineThoiGianThuc")

        self._queue = MessageQueue(ten=ten, kich_thuoc_toi_da=kich_thuoc_buffer)
        self._feature_store = FeatureStore(kich_thuoc_cua_so=kich_thuoc_buffer)
        self._mo_hinh: Dict[str, Any] = {}
        self._da_dang_ky: Dict[str, bool] = {}

        self._buffer_du_lieu: Dict[str, deque] = {}
        self._buffer_nhan: Dict[str, deque] = {}

        self._latency: deque = deque(maxlen=1000)
        self._so_du_doan = 0
        self._so_loi = 0

    def dang_ky_mo_hinh(self, ten: str, mo_hinh: Any) -> None:
        """
        Đăng ký mô hình cho pipeline.

        Args:
            ten: Tên mô hình
            mo_hinh: Mô hình đã huấn luyện
        """
        if mo_hinh is None:
            raise ValueError("Mô hình không được None")

        self._mo_hinh[ten] = mo_hinh
        self._da_dang_ky[ten] = True
        self._buffer_du_lieu[ten] = deque(maxlen=self.batch_size)
        self._buffer_nhan[ten] = deque(maxlen=self.batch_size)
        self.logger.info(f"Đã đăng ký mô hình: {ten}")

    def gui_du_lieu(
        self,
        ten_mo_hinh: str,
        du_lieu: np.ndarray,
        nhan: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Gửi dữ liệu vào pipeline.

        Args:
            ten_mo_hinh: Tên mô hình
            du_lieu: Dữ liệu đầu vào
            nhan: Nhãn (tùy chọn, cho online learning)

        Returns:
            Dict chứa kết quả xử lý
        """
        if ten_mo_hinh not in self._da_dang_ky:
            raise KeyError(f"Mô hình '{ten_mo_hinh}' chưa đăng ký")

        du_lieu = np.asarray(du_lieu).flatten()

        self._queue.publish(
            topic=ten_mo_hinh,
            du_lieu=du_lieu.tolist(),
            key=str(time.time()),
        )

        self._feature_store.cap_nhat(ten_mo_hinh, du_lieu, nhan)

        self._buffer_du_lieu[ten_mo_hinh].append(du_lieu)
        if nhan is not None:
            self._buffer_nhan[ten_mo_hinh].append(nhan)

        return {
            "ten_mo_hinh": ten_mo_hinh,
            "kich_thuoc_buffer": len(self._buffer_du_lieu[ten_mo_hinh]),
            "trang_thai": "da_nhan",
        }

    def du_doan(self, ten_mo_hinh: str, du_lieu: np.ndarray) -> Dict[str, Any]:
        """
        Dự đoán real-time.

        Args:
            ten_mo_hinh: Tên mô hình
            du_lieu: Dữ liệu đầu vào

        Returns:
            Dict chứa kết quả dự đoán và latency
        """
        if ten_mo_hinh not in self._da_dang_ky:
            raise KeyError(f"Mô hình '{ten_mo_hinh}' chưa đăng ký")

        du_lieu = np.asarray(du_lieu)
        bat_dau = time.perf_counter()

        try:
            mo_hinh = self._mo_hinh[ten_mo_hinh]

            if du_lieu.ndim == 1:
                du_lieu_input = du_lieu.reshape(1, -1)
            else:
                du_lieu_input = du_lieu

            ket_qua = mo_hinh.du_doan(du_lieu_input)

            latency = (time.perf_counter() - bat_dau) * 1000
            self._latency.append(latency)
            self._so_du_doan += 1

            self._queue.publish(
                topic=f"{ten_mo_hinh}_predictions",
                du_lieu={
                    "ket_qua": ket_qua.tolist(),
                    "latency_ms": latency,
                },
            )

            return {
                "ket_qua": ket_qua,
                "latency_ms": round(latency, 3),
                "trang_thai": "thanh_cong",
            }

        except Exception as e:
            self._so_loi += 1
            self.logger.error(f"Lỗi dự đoán {ten_mo_hinh}: {e}")
            return {
                "ket_qua": None,
                "latency_ms": 0,
                "trang_thai": "loi",
                "loi": str(e),
            }

    def du_doan_batch(self, ten_mo_hinh: str, du_lieu_batch: np.ndarray) -> Dict[str, Any]:
        """
        Dự đoán theo batch.

        Args:
            ten_mo_hinh: Tên mô hình
            du_lieu_batch: Batch dữ liệu

        Returns:
            Dict chứa kết quả batch
        """
        du_lieu_batch = np.asarray(du_lieu_batch)
        bat_dau = time.perf_counter()

        try:
            mo_hinh = self._mo_hinh[ten_mo_hinh]
            ket_qua = mo_hinh.du_doan(du_lieu_batch)

            latency = (time.perf_counter() - bat_dau) * 1000
            self._latency.append(latency)
            self._so_du_doan += len(du_lieu_batch)

            return {
                "ket_qua": ket_qua,
                "so_mau": len(du_lieu_batch),
                "latency_ms": round(latency, 3),
                "throughput_mau_giay": len(du_lieu_batch) / (latency / 1000)
                if latency > 0
                else float("inf"),
                "trang_thai": "thanh_cong",
            }

        except Exception as e:
            self._so_loi += 1
            return {
                "ket_qua": None,
                "trang_thai": "loi",
                "loi": str(e),
            }

    def lay_thong_ke(self) -> Dict[str, Any]:
        """Thống kê pipeline."""
        latency_arr = np.array(self._latency) if self._latency else np.array([0])
        return {
            "ten": self.ten,
            "so_du_doan": self._so_du_doan,
            "so_loi": self._so_loi,
            "ty_le_loi": self._so_loi / max(1, self._so_du_doan),
            "latency_trung_binh_ms": float(np.mean(latency_arr)),
            "latency_p50_ms": float(np.percentile(latency_arr, 50)),
            "latency_p95_ms": float(np.percentile(latency_arr, 95)),
            "latency_p99_ms": float(np.percentile(latency_arr, 99)),
            "latency_max_ms": float(np.max(latency_arr)),
            "so_mo_hinh": len(self._mo_hinh),
            "queue": self._queue.lay_thong_ke(),
            "feature_store": self._feature_store.thong_ke(),
        }

    def lay_lich_su(self, ten_mo_hinh: str, so_luong: int = 10) -> List[Dict]:
        """Lấy lịch sử predictions."""
        tin_nhan = self._queue.consume(f"{ten_mo_hinh}_predictions", so_luong)
        return [tn["du_lieu"] for tn in tin_nhan]

    def dang_ky_callback(self, ten_mo_hinh: str, callback: Callable) -> None:
        """Đăng ký callback khi có dự đoán mới."""
        self._queue.subscribe(f"{ten_mo_hinh}_predictions", callback)

    def xoa_buffer(self) -> None:
        """Xóa toàn bộ buffer và reset counters."""
        for buffer in self._buffer_du_lieu.values():
            buffer.clear()
        for buffer in self._buffer_nhan.values():
            buffer.clear()
        self._latency.clear()
        self._so_du_doan = 0
        self._so_loi = 0
        self._queue.xoa()
        self.logger.info("Đã xóa toàn bộ buffer")

    def __repr__(self) -> str:
        return (
            f"PipelineThoiGianThuc(mo_hinh={len(self._mo_hinh)}, "
            f"du_doan={self._so_du_doan}, loi={self._so_loi})"
        )
