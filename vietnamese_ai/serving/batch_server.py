"""MayChuBatch - dynamic batching cho model serving."""

import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class MayChuBatch:
    """
    Máy chủ dynamic batching cho model inference.

    Gom nhiều request thành batch để tăng throughput.
    Hỗ trợ:
    - Dynamic batching với timeout
    - Concurrent request handling
    - Latency tracking
    - Auto-tuning batch size

    Sử dụng:
        >>> server = MayChuBatch(mo_hinh=model, kich_thuoc_batch=32)
        >>> server.bat_dau()
        >>> ket_qua = server.du_doan(dau_vao)
        >>> server.dung()
    """

    def __init__(
        self,
        mo_hinh: Any,
        kich_thuoc_batch: int = 32,
        timeout_batch: float = 0.1,
        toi_da_cho: int = 1000,
        so_worker: int = 1,
        ham_du_doan: Optional[Callable] = None,
    ):
        self.mo_hinh = mo_hinh
        self.kich_thuoc_batch = kich_thuoc_batch
        self.timeout_batch = timeout_batch
        self.toi_da_cho = toi_da_cho
        self.so_worker = so_worker
        self.ham_du_doan = ham_du_doan

        self._queue: queue.Queue = queue.Queue(maxsize=toi_da_cho)
        self._ket_qua_map: Dict[int, Any] = {}
        self._dang_chay = False
        self._workers: List[threading.Thread] = []
        self._lock = threading.Lock()
        self._counter = 0

        self._thong_ke = {
            "tong_request": 0,
            "tong_batch": 0,
            "tong_thoi_gian": 0.0,
            "batch_size_tb": 0.0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "latency_p99": 0.0,
            "queue_size": 0,
        }
        self._latencies: List[float] = []

    def bat_dau(self) -> None:
        """Khởi động batch server."""
        if self._dang_chay:
            return

        self._dang_chay = True
        for i in range(self.so_worker):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"batch-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def dung(self) -> None:
        """Dừng batch server."""
        self._dang_chay = False
        for worker in self._workers:
            worker.join(timeout=5.0)
        self._workers.clear()

    def du_doan(self, dau_vao: Any) -> Any:
        """
        Gửi request dự đoán (blocking).

        Args:
            dau_vao: Dữ liệu đầu vào

        Returns:
            Kết quả dự đoán
        """
        if not self._dang_chay:
            self.bat_dau()

        with self._lock:
            self._counter += 1
            request_id = self._counter

        self._thong_ke["tong_request"] += 1

        event = threading.Event()
        self._queue.put((request_id, dau_vao, event, time.time()))
        event.wait(timeout=30.0)

        with self._lock:
            ket_qua = self._ket_qua_map.pop(request_id, None)

        return ket_qua

    def du_doan_batch(self, dau_vao_list: List[Any]) -> List[Any]:
        """Gửi nhiều request cùng lúc."""
        events = {}
        request_ids = []

        for dau_vao in dau_vao_list:
            with self._lock:
                self._counter += 1
                request_id = self._counter

            self._thong_ke["tong_request"] += 1
            event = threading.Event()
            events[request_id] = event
            self._queue.put((request_id, dau_vao, event, time.time()))
            request_ids.append(request_id)

        for event in events.values():
            event.wait(timeout=30.0)

        ket_qua = []
        with self._lock:
            for rid in request_ids:
                ket_qua.append(self._ket_qua_map.pop(rid, None))

        return ket_qua

    def _worker_loop(self) -> None:
        """Worker loop xử lý batch."""
        while self._dang_chay:
            batch = self._lay_batch()
            if not batch:
                time.sleep(0.001)
                continue

            self._xu_ly_batch(batch)

    def _lay_batch(self) -> List[tuple]:
        """Lấy một batch từ queue."""
        batch = []
        bat_dau = time.time()

        try:
            item = self._queue.get(timeout=self.timeout_batch)
            batch.append(item)
        except queue.Empty:
            return batch

        while len(batch) < self.kich_thuoc_batch:
            elapsed = time.time() - bat_dau
            if elapsed >= self.timeout_batch:
                break
            try:
                item = self._queue.get(timeout=0.001)
                batch.append(item)
            except queue.Empty:
                break

        return batch

    def _xu_ly_batch(self, batch: List[tuple]) -> None:
        """Xử lý một batch request."""
        bat_dau = time.time()

        dau_vao_list = [item[1] for item in batch]

        try:
            if self.ham_du_doan:
                ket_qua = self.ham_du_doan(dau_vao_list)
            elif hasattr(self.mo_hinh, "du_doan"):
                if isinstance(dau_vao_list[0], np.ndarray):
                    batched = np.array(dau_vao_list)
                    ket_qua = self.mo_hinh.du_doan(batched)
                else:
                    ket_qua = [self.mo_hinh.du_doan(x) for x in dau_vao_list]
            else:
                ket_qua = [None] * len(dau_vao_list)
        except Exception:
            ket_qua = [None] * len(dau_vao_list)

        thoi_gian = time.time() - bat_dau
        self._latencies.append(thoi_gian)

        with self._lock:
            self._thong_ke["tong_batch"] += 1
            self._thong_ke["tong_thoi_gian"] += thoi_gian

            if isinstance(ket_qua, np.ndarray) and ket_qua.ndim > 1:
                for i, item in enumerate(batch):
                    self._ket_qua_map[item[0]] = ket_qua[i]
                    item[2].set()
            else:
                for i, item in enumerate(batch):
                    self._ket_qua_map[item[0]] = ket_qua[i] if i < len(ket_qua) else None
                    item[2].set()

    def lay_thong_ke(self) -> Dict[str, Any]:
        """Lấy thống kê server."""
        with self._lock:
            stats = self._thong_ke.copy()
            stats["queue_size"] = self._queue.qsize()
            stats["dang_chay"] = self._dang_chay

            if self._latencies:
                arr = np.array(self._latencies[-1000:])
                stats["latency_p50"] = float(np.percentile(arr, 50))
                stats["latency_p95"] = float(np.percentile(arr, 95))
                stats["latency_p99"] = float(np.percentile(arr, 99))
                stats["latency_tb"] = float(np.mean(arr))

            if stats["tong_batch"] > 0:
                stats["batch_size_tb"] = stats["tong_request"] / stats["tong_batch"]

            return stats

    def __repr__(self) -> str:
        return (
            f"MayChuBatch(kich_thuoc_batch={self.kich_thuoc_batch}, "
            f"so_worker={self.so_worker}, "
            f"dang_chay={self._dang_chay})"
        )
