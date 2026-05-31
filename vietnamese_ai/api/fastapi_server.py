"""FastAPIServer - API Server tương thích chuẩn OpenAI."""

import time
import uuid
from typing import Any, List

from pydantic import BaseModel

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    raise ImportError("Vui lòng cài đặt fastapi: pip install fastapi uvicorn pydantic")

from vietnamese_ai.utils.logger import Logger


# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "vietnamese-ai-default"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage


# --- FastAPI Server ---
class FastAPIServer:
    """
    Server FastAPI với chuẩn API tương thích OpenAI.
    Hỗ trợ xử lý request qua Hệ thống Đa Tác Tử hoặc LLM Model.
    """

    def __init__(self, bo_xu_ly: Any, ten: str = "VietnameseAI-API", enable_watermark: bool = True):
        """
        Khởi tạo Server.

        Args:
            bo_xu_ly: Có thể là một đối tượng `HeThongDaTacTu` hoặc LLM Wrapper
                      (có hàm `chay(truy_van)` hoặc `sinh_van_ban(prompt)`).
        """
        self.bo_xu_ly = bo_xu_ly
        self.ten = ten
        self.enable_watermark = enable_watermark
        self.logger = Logger(ten)

        self.app = FastAPI(title=ten, description="Vietnamese AI Framework API")

        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Security Firewall
        try:
            from vietnamese_ai.api.security_middleware import AISecurityMiddleware

            self.app.add_middleware(AISecurityMiddleware, ghi_log=True)
        except ImportError:
            self.logger.warning("Không thể tải Security Middleware.")

        self._setup_routes()

    def _chuyen_doi_messages(self, messages: List[ChatMessage]) -> str:
        """Chuyển mảng messages thành chuỗi văn bản cho Model/Agent hiểu."""
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"Hệ thống: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"Người dùng: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Trợ lý: {msg.content}\n"
        return prompt.strip()

    def _xu_ly_logic(self, request: ChatCompletionRequest) -> str:
        """Xử lý request với core xử lý bên dưới."""
        # Nếu bộ xử lý là Multi-Agent
        if hasattr(self.bo_xu_ly, "chay"):
            # Lấy nội dung message cuối cùng của người dùng
            truy_van = request.messages[-1].content
            # Reset lịch sử nếu cần thiết (tùy logic framework)
            return self.bo_xu_ly.chay(truy_van)

        # Nếu bộ xử lý là LLM Wrapper
        elif hasattr(self.bo_xu_ly, "sinh_van_ban"):
            prompt = self._chuyen_doi_messages(request.messages)
            return self.bo_xu_ly.sinh_van_ban(
                prompt, nhiet_do=request.temperature, do_dai=request.max_tokens
            )

        else:
            raise ValueError("Bộ xử lý không hợp lệ. Cần hỗ trợ hàm `chay` hoặc `sinh_van_ban`.")

    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"status": "hoat_dong", "ten": self.ten}

        @self.app.get("/suc_khoe")
        async def health():
            return {"status": "tot"}

        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest):
            self.logger.info(f"Nhận request chat completion (model: {request.model})")

            try:
                bat_dau = time.time()

                # Gọi core AI xử lý
                ket_qua = self._xu_ly_logic(request)

                # Nhúng Watermark nếu được cấu hình
                if self.enable_watermark:
                    try:
                        from vietnamese_ai.security.watermark import TextWatermarker

                        ket_qua = TextWatermarker.nhung_thuy_an(ket_qua, "EVONETAI_API")
                    except ImportError:
                        pass

                # Tạo response
                response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

                # Ước lượng token (đơn giản 1 từ = 1 token cho testing)
                input_tokens = sum(len(m.content.split()) for m in request.messages)
                output_tokens = len(ket_qua.split())

                resp = ChatCompletionResponse(
                    id=response_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionResponseChoice(
                            index=0, message=ChatMessage(role="assistant", content=ket_qua)
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                )

                self.logger.info(f"Hoàn thành trong {time.time() - bat_dau:.2f}s")
                return resp

            except Exception as e:
                self.logger.error(f"Lỗi xử lý request: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    def chay(self, host: str = "0.0.0.0", port: int = 8000):
        """Chạy server với Uvicorn."""
        import uvicorn

        self.logger.info(f"Khởi động FastAPI Server tại http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
