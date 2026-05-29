import json
import logging
from typing import Any, Callable

try:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    BaseHTTPMiddleware = object  # Fallback for systems without FastAPI

from vietnamese_ai.security.llm_firewall import TuongLuaAI


class AISecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware FastAPI đánh chặn mọi request gửi tới AI.
    Sử dụng TuongLuaAI để quét Payload (Prompt) và chặn ngay từ Gateway
    nếu phát hiện Prompt Injection hoặc truy vấn độc hại.
    """

    def __init__(self, app: Any, ghi_log: bool = True):
        super().__init__(app)
        self.tuong_lua = TuongLuaAI(ngat_ket_noi_khi_phat_hien=True)
        self.ghi_log = ghi_log
        self.logger = logging.getLogger("AISecurity")

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        # Chỉ quét các endpoint liên quan đến sinh văn bản (Completions)
        if request.url.path.endswith("/completions"):
            try:
                # Đọc body stream của FastAPI
                body = await request.body()
                if body:
                    data = json.loads(body.decode("utf-8"))
                    messages = data.get("messages", [])

                    # Quét qua nội dung của tất cả các user messages
                    for msg in messages:
                        if msg.get("role") == "user":
                            noi_dung = msg.get("content", "")
                            an_toan, ly_do = self.tuong_lua.kiem_tra_prompt(noi_dung)

                            if not an_toan:
                                if self.ghi_log:
                                    self.logger.warning(f"BẢO MẬT: Đã chặn truy vấn độc hại. Lý do: {ly_do}. Nguồn: {request.client.host}")

                                return JSONResponse(
                                    status_code=400,
                                    content={
                                        "error": "SecurityViolation",
                                        "message": f"Yêu cầu đã bị Tường lửa AI chặn. Lý do: {ly_do}"
                                    }
                                )
            except Exception:
                # Nếu không phải JSON hợp lệ hoặc có lỗi parse, bỏ qua việc quét
                # (Lỗi này sẽ được FastAPI xử lý pydantic validation sau)
                pass

        # Tiếp tục xử lý request bình thường nếu an toàn
        return await call_next(request)
