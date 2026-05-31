"""MayChuStream - streaming server cho LLM response."""

import json
import threading
import time
from typing import Any, Callable, Dict, Generator, Optional


class MayChuStream:
    """
    Máy chủ streaming cho LLM responses.

    Hỗ trợ:
    - SSE (Server-Sent Events) streaming
    - Token-by-token generation
    - Callback-based streaming
    - Multi-client concurrent streaming

    Sử dụng:
        >>> server = MayChuStream(ham_sinh=generate_fn)
        >>> for token in server.sinh_stream("prompt"):
        ...     print(token, end="")
    """

    def __init__(
        self,
        ham_sinh: Optional[Callable] = None,
        toc_do_token: float = 0.02,
        toi_da_client: int = 100,
        timeout: float = 60.0,
    ):
        self.ham_sinh = ham_sinh
        self.toc_do_token = toc_do_token
        self.toi_da_client = toi_da_client
        self.timeout = timeout

        self._clients: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._counter = 0

        self._thong_ke = {
            "tong_stream": 0,
            "tong_token": 0,
            "client_hien_tai": 0,
            "thoi_gian_tb": 0.0,
        }

    def sinh_stream(
        self,
        prompt: str,
        client_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Sinh nội dung theo dạng streaming.

        Args:
            prompt: Prompt đầu vào
            client_id: ID client (tự tạo nếu None)
            **kwargs: Tham số bổ sung cho hàm sinh

        Yields:
            Từng token
        """
        if client_id is None:
            with self._lock:
                self._counter += 1
                client_id = f"client_{self._counter}"

        with self._lock:
            if len(self._clients) >= self.toi_da_client:
                raise RuntimeError(f"Đã đạt giới hạn {self.toi_da_client} client đồng thời")
            self._clients[client_id] = {
                "bat_dau": time.time(),
                "trang_thai": "dang_sinh",
            }
            self._thong_ke["tong_stream"] += 1
            self._thong_ke["client_hien_tai"] = len(self._clients)

        try:
            if self.ham_sinh:
                yield from self._sinh_voi_model(prompt, client_id, **kwargs)
            else:
                yield from self._sinh_don_gian(prompt, client_id)
        finally:
            with self._lock:
                self._clients.pop(client_id, None)
                self._thong_ke["client_hien_tai"] = len(self._clients)

    def _sinh_voi_model(
        self,
        prompt: str,
        client_id: str,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Sinh với model thực tế."""
        bat_dau = time.time()

        try:
            ket_qua = self.ham_sinh(prompt, **kwargs)

            if isinstance(ket_qua, str):
                for ky_tu in ket_qua:
                    yield ky_tu
                    with self._lock:
                        self._thong_ke["tong_token"] += 1
                    time.sleep(self.toc_do_token)
            elif isinstance(ket_qua, Generator):
                yield from ket_qua
            elif isinstance(ket_qua, list):
                for token in ket_qua:
                    yield str(token)
                    with self._lock:
                        self._thong_ke["tong_token"] += 1
                    time.sleep(self.toc_do_token)
        except Exception as e:
            yield f"\n[Lỗi: {str(e)}]"
        finally:
            thoi_gian = time.time() - bat_dau
            with self._lock:
                n = self._thong_ke["tong_stream"]
                old_avg = self._thong_ke["thoi_gian_tb"]
                self._thong_ke["thoi_gian_tb"] = (old_avg * (n - 1) + thoi_gian) / n

    def _sinh_don_gian(
        self,
        prompt: str,
        client_id: str,
    ) -> Generator[str, None, None]:
        """Sinh demo khi không có model."""
        tokens = f"Đã nhận: '{prompt[:50]}...'. Đây là phản hồi demo từ MayChuStream.".split()

        for token in tokens:
            yield token + " "
            with self._lock:
                self._thong_ke["tong_token"] += 1
            time.sleep(self.toc_do_token)

    def sinh_sse(
        self,
        prompt: str,
        client_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Sinh nội dung theo SSE format.

        Yields:
            SSE formatted strings
        """
        for token in self.sinh_stream(prompt, client_id, **kwargs):
            data = json.dumps({"token": token}, ensure_ascii=False)
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

    def dang_ky_callback(
        self,
        client_id: str,
        callback: Callable[[str], None],
    ) -> None:
        """Đăng ký callback cho client."""
        with self._lock:
            if client_id in self._clients:
                self._clients[client_id]["callback"] = callback

    def huy_client(self, client_id: str) -> bool:
        """Hủy một client."""
        with self._lock:
            if client_id in self._clients:
                self._clients[client_id]["trang_thai"] = "huy"
                return True
            return False

    def so_client(self) -> int:
        """Số client đang kết nối."""
        with self._lock:
            return len(self._clients)

    def lay_thong_ke(self) -> Dict[str, Any]:
        """Lấy thống kê streaming server."""
        with self._lock:
            return self._thong_ke.copy()

    def __repr__(self) -> str:
        return (
            f"MayChuStream(toi_da_client={self.toi_da_client}, client_hien_tai={self.so_client()})"
        )
