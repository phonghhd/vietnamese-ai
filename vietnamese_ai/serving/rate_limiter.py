"""BoGioiHanTocDo - rate limiter cho API serving."""

import threading
import time
from typing import Any, Dict, Optional


class BoGioiHanTocDo:
    """
    Bộ giới hạn tốc độ (rate limiter) cho API.

    Hỗ trợ:
    - Token bucket algorithm
    - Sliding window counter
    - Per-client rate limiting
    - Burst handling

    Sử dụng:
        >>> limiter = BoGioiHanTocDo(go_i_y=100, cua_so=60)
        >>> if limiter.cho_phep("client_1"):
        ...     # Xử lý request
        ...     pass
    """

    def __init__(
        self,
        go_i_y: int = 100,
        cua_so: float = 60.0,
        kich_thuoc_burst: Optional[int] = None,
        che_do: str = "token_bucket",
    ):
        if che_do not in ("token_bucket", "sliding_window"):
            raise ValueError("che_do phải là: token_bucket, sliding_window")

        self.go_i_y = go_i_y
        self.cua_so = cua_so
        self.kich_thuoc_burst = kich_thuoc_burst or go_i_y
        self.che_do = che_do

        self._clients: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        self._thong_ke = {
            "tong_request": 0,
            "da_cho_phep": 0,
            "da_tu_choi": 0,
            "client_hien_tai": 0,
        }

    def cho_phep(self, client_id: str = "default") -> bool:
        """
        Kiểm tra request có được phép không.

        Args:
            client_id: ID của client

        Returns:
            True nếu được phép, False nếu bị giới hạn
        """
        with self._lock:
            self._thong_ke["tong_request"] += 1

            if self.che_do == "token_bucket":
                ket_qua = self._token_bucket(client_id)
            else:
                ket_qua = self._sliding_window(client_id)

            if ket_qua:
                self._thong_ke["da_cho_phep"] += 1
            else:
                self._thong_ke["da_tu_choi"] += 1

            return ket_qua

    def _token_bucket(self, client_id: str) -> bool:
        """Token bucket algorithm."""
        now = time.time()

        if client_id not in self._clients:
            self._clients[client_id] = {
                "tokens": float(self.kich_thuoc_burst),
                "last_time": now,
            }
            self._thong_ke["client_hien_tai"] = len(self._clients)

        client = self._clients[client_id]

        # Refill tokens
        elapsed = now - client["last_time"]
        refill = elapsed * (self.go_i_y / self.cua_so)
        client["tokens"] = min(
            float(self.kich_thuoc_burst),
            client["tokens"] + refill,
        )
        client["last_time"] = now

        if client["tokens"] >= 1.0:
            client["tokens"] -= 1.0
            return True

        return False

    def _sliding_window(self, client_id: str) -> bool:
        """Sliding window counter algorithm."""
        now = time.time()

        if client_id not in self._clients:
            self._clients[client_id] = {
                "timestamps": [],
            }
            self._thong_ke["client_hien_tai"] = len(self._clients)

        client = self._clients[client_id]

        # Xóa timestamps cũ
        cutoff = now - self.cua_so
        client["timestamps"] = [
            t for t in client["timestamps"] if t > cutoff
        ]

        if len(client["timestamps"]) < self.go_i_y:
            client["timestamps"].append(now)
            return True

        return False

    def lay_con_lai(self, client_id: str = "default") -> int:
        """Số request còn lại trong cửa sổ."""
        with self._lock:
            if client_id not in self._clients:
                return self.go_i_y

            if self.che_do == "token_bucket":
                return int(self._clients[client_id]["tokens"])
            else:
                now = time.time()
                cutoff = now - self.cua_so
                used = sum(
                    1 for t in self._clients[client_id]["timestamps"]
                    if t > cutoff
                )
                return max(0, self.go_i_y - used)

    def reset(self, client_id: Optional[str] = None) -> None:
        """Reset rate limit cho client hoặc tất cả."""
        with self._lock:
            if client_id:
                self._clients.pop(client_id, None)
            else:
                self._clients.clear()
            self._thong_ke["client_hien_tai"] = len(self._clients)

    def xoa_client_khong_hoat_dong(self, timeout: float = 300.0) -> int:
        """Xóa client không hoạt động."""
        now = time.time()
        da_xoa = 0

        with self._lock:
            keys_to_remove = []
            for cid, client in self._clients.items():
                if self.che_do == "token_bucket":
                    if now - client["last_time"] > timeout:
                        keys_to_remove.append(cid)
                else:
                    if not client["timestamps"] or (
                        now - client["timestamps"][-1] > timeout
                    ):
                        keys_to_remove.append(cid)

            for cid in keys_to_remove:
                del self._clients[cid]
                da_xoa += 1

            self._thong_ke["client_hien_tai"] = len(self._clients)

        return da_xoa

    def lay_thong_ke(self) -> Dict[str, Any]:
        """Lấy thống kê rate limiter."""
        with self._lock:
            stats = self._thong_ke.copy()
            stats["go_i_y"] = self.go_i_y
            stats["cua_so"] = self.cua_so
            stats["che_do"] = self.che_do

            if stats["tong_request"] > 0:
                stats["ty_le_tu_choi"] = stats["da_tu_choi"] / stats["tong_request"]
            else:
                stats["ty_le_tu_choi"] = 0.0

            return stats

    def __repr__(self) -> str:
        return (
            f"BoGioiHanTocDo(go_i_y={self.go_i_y}, "
            f"cua_so={self.cua_so}, "
            f"che_do='{self.che_do}')"
        )
