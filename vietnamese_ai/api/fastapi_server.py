"""FastAPIServer - API Server tương thích chuẩn OpenAI."""

import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    _CO_FASTAPI = True
except ImportError:
    _CO_FASTAPI = False
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    WebSocket = Any  # type: ignore
    WebSocketDisconnect = Exception  # type: ignore
    CORSMiddleware = None  # type: ignore

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


from vietnamese_ai.utils.logger import Logger


# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str = ""
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "vietnamese-ai-default"
    messages: List[ChatMessage] = []
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: Any = None
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Any] = []
    usage: Any = None


class FeedbackItem(BaseModel):
    prompt: str
    chosen: str
    rejected: str = ""
    score: float = 1.0
    model: str = "default"
    metadata: Dict[str, Any] = {}


# --- FastAPI Server ---
class FastAPIServer:
    """
    Server FastAPI với chuẩn API tương thích OpenAI.
    Hỗ trợ xử lý request qua Hệ thống Đa Tác Tử hoặc LLM Model.

    Yêu cầu: pip install fastapi uvicorn pydantic
    """

    def __init__(
        self,
        bo_xu_ly: Any,
        ten: str = "VietnameseAI-API",
        enable_watermark: bool = True,
        enable_sanitizer: bool = False,
        enable_semantic_cache: bool = False,
        semantic_cache_threshold: float = 0.92,
        ho_du_lieu: Optional[Any] = None,
        mach_cat: Optional[Any] = None,
        suc_khoe: Optional[Any] = None,
    ):
        """
        Khởi tạo Server.

        Args:
            bo_xu_ly: Có thể là một đối tượng `HeThongDaTacTu` hoặc LLM Wrapper
                      (có hàm `chay(truy_van)` hoặc `sinh_van_ban(prompt)`).
            ten: Tên định danh của server.
            enable_watermark: Tự động nhúng thủy ấn số vào câu trả lời sinh ra.
            enable_sanitizer: Tự động làm sạch và ẩn danh hóa dữ liệu PII trong câu hỏi.
            enable_semantic_cache: Kích hoạt bộ nhớ đệm ngữ nghĩa để trả về câu trả lời đã có.
            semantic_cache_threshold: Ngưỡng độ tương đồng cosine tối thiểu để tính là cache hit.
            ho_du_lieu: Đối tượng HoDuLieu (hoặc đường dẫn) để thu thập feedback DPO khép kín.
            mach_cat: Đối tượng MachCat bảo vệ circuit breaker cho các requests.
            suc_khoe: Đối tượng KiemTraSucKhoe giám sát trạng thái hệ thống.
        """
        if not _CO_FASTAPI:
            raise ImportError(
                "Cần cài đặt fastapi để dùng FastAPIServer: pip install fastapi uvicorn pydantic"
            )

        self.bo_xu_ly = bo_xu_ly
        self.ten = ten
        self.enable_watermark = enable_watermark
        self.enable_sanitizer = enable_sanitizer
        self.enable_semantic_cache = enable_semantic_cache
        self.mach_cat = mach_cat or getattr(bo_xu_ly, "mach_cat", None)
        self.suc_khoe = suc_khoe or getattr(bo_xu_ly, "suc_khoe", None)
        self.logger = Logger(ten)

        if ho_du_lieu is not None:
            if isinstance(ho_du_lieu, str):
                from vietnamese_ai.datalake.dataset_manager import HoDuLieu

                self.ho_du_lieu = HoDuLieu(duong_dan_luu_tru=ho_du_lieu)
            else:
                self.ho_du_lieu = ho_du_lieu
        else:
            self.ho_du_lieu = None

        if enable_semantic_cache:
            try:
                from vietnamese_ai.rag.context_optimization import BoNhoDemNguNghia

                self.semantic_cache = BoNhoDemNguNghia(nguong_tuong_dong=semantic_cache_threshold)
            except ImportError:
                self.semantic_cache = None
        else:
            self.semantic_cache = None

        if enable_sanitizer:
            try:
                from vietnamese_ai.security.data_sanitizer import DataSanitizer

                self.sanitizer = DataSanitizer
            except ImportError:
                self.sanitizer = None
        else:
            self.sanitizer = None

        self.app = FastAPI(title=ten, description="Vietnamese AI Framework API")

        # CORS an toàn (không bật allow_credentials=True khi allow_origins=["*"])
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
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

    def _xu_ly_logic_truc_tiep(self, request: ChatCompletionRequest) -> str:
        """Xử lý request với core xử lý bên dưới."""
        if getattr(request, "model", "") in ("swarm", "edge") or getattr(request, "che_do", "") == "swarm":
            truy_van = request.messages[-1].content
            if hasattr(self.bo_xu_ly, "suy_luan_edge"):
                res = self.bo_xu_ly.suy_luan_edge(truy_van)
                return str(res.get("cau_tra_loi", res))
            elif hasattr(self.bo_xu_ly, "suy_luan_swarm"):
                res = self.bo_xu_ly.suy_luan_swarm(truy_van)
                return str(res.get("cau_tra_loi", res))

        if hasattr(self.bo_xu_ly, "chay"):
            truy_van = request.messages[-1].content
            return self.bo_xu_ly.chay(truy_van)
        elif hasattr(self.bo_xu_ly, "sinh_van_ban"):
            prompt = self._chuyen_doi_messages(request.messages)
            return self.bo_xu_ly.sinh_van_ban(
                prompt, nhiet_do=request.temperature, do_dai=request.max_tokens
            )
        elif hasattr(self.bo_xu_ly, "hoi"):
            truy_van = request.messages[-1].content
            kq = self.bo_xu_ly.hoi(truy_van)
            return str(kq.get("tra_loi", kq))
        else:
            raise ValueError("Bộ xử lý không hợp lệ. Cần hỗ trợ hàm `chay`, `sinh_van_ban` hoặc `hoi`.")

    def _xu_ly_logic(self, request: ChatCompletionRequest) -> str:
        if self.mach_cat is not None:
            return self.mach_cat.thuc_hien(self._xu_ly_logic_truc_tiep, request)
        return self._xu_ly_logic_truc_tiep(request)

    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"status": "hoat_dong", "ten": self.ten}

        @self.app.get("/suc_khoe")
        async def health():
            if self.suc_khoe is not None:
                return self.suc_khoe.kiem_tra()
            if hasattr(self.bo_xu_ly, "kiem_tra_suc_khoe"):
                return self.bo_xu_ly.kiem_tra_suc_khoe()
            return {"status": "tot"}

        @self.app.post("/v1/recover")
        async def recover_system():
            """Endpoint kích hoạt tự phục hồi (Self-Healing / Auto-Rollback) trong hệ thống."""
            if hasattr(self.bo_xu_ly, "tu_dong_phuc_hoi"):
                return self.bo_xu_ly.tu_dong_phuc_hoi()
            if self.mach_cat is not None and hasattr(self.mach_cat, "reset"):
                self.mach_cat.reset()
                return {"status": "da_reset_circuit_breaker"}
            return {"status": "khong_kha_dung", "thong_bao": "Không có cơ chế tự phục hồi cho bộ xử lý này."}

        @self.app.get("/v1/cache/stats")
        async def cache_stats():
            if self.semantic_cache is not None:
                return self.semantic_cache.thong_ke()
            return {"enabled": False}

        @self.app.get("/v1/swarm/status")
        async def swarm_status():
            """Trạng thái cụm Edge Swarm CPU và phân bổ Layer Shards."""
            swarm = getattr(self.bo_xu_ly, "dieu_phoi_swarm", None)
            if swarm is None and hasattr(self.bo_xu_ly, "lay_trang_thai_cum"):
                swarm = self.bo_xu_ly
            if swarm and hasattr(swarm, "lay_trang_thai_cum"):
                return {"status": "success", "trang_thai": "hoat_dong", "cum_swarm": swarm.lay_trang_thai_cum()}
            return {"status": "inactive", "thong_bao": "Cụm Edge Swarm chưa được kích hoạt."}

        @self.app.post("/v1/swarm/completions")
        async def swarm_completions(req: ChatCompletionRequest):
            """Suy luận phân tán Pipeline Parallelism trên cụm Edge CPU Swarm kèm Verifiable Proof."""
            truy_van = req.messages[-1].content
            if hasattr(self.bo_xu_ly, "suy_luan_edge"):
                kq = self.bo_xu_ly.suy_luan_edge(truy_van)
                return {"status": "success", "loai": "edge_swarm", "ket_qua": kq}
            elif hasattr(self.bo_xu_ly, "suy_luan_swarm"):
                kq = self.bo_xu_ly.suy_luan_swarm(truy_van)
                return {"status": "success", "loai": "edge_swarm", "ket_qua": kq}
            if HTTPException:
                raise HTTPException(status_code=501, detail="Bộ xử lý không hỗ trợ suy luận Edge Swarm.")
            return {"status": "error", "message": "Không hỗ trợ Edge Swarm"}

        @self.app.post("/v1/feedback")
        async def submit_feedback(fb: FeedbackItem):
            """Endpoint thu thập phản hồi người dùng (DPO preference) lưu vào DataLake khép kín."""
            if self.ho_du_lieu is not None:
                ban_ghi = {
                    "prompt": fb.prompt,
                    "chosen": fb.chosen,
                    "rejected": fb.rejected,
                    "score": fb.score,
                    "model": fb.model,
                    "metadata": fb.metadata,
                    "timestamp": time.time(),
                }
                ma_bam = self.ho_du_lieu.them_ban_ghi(ban_ghi, ten_dataset="feedback_dpo.jsonl")
                return {
                    "status": "success",
                    "message": "Đã ghi nhận phản hồi vào DataLake phục vụ DPO/RLHF",
                    "commit_sha256": ma_bam,
                }
            return {
                "status": "received",
                "message": "Đã nhận phản hồi (DataLake chưa được cấu hình)",
            }

        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest):
            self.logger.info(f"Nhận request chat completion (model: {request.model})")

            try:
                bat_dau = time.time()

                # 1. Khử PII nếu được bật
                if self.sanitizer is not None:
                    for m in request.messages:
                        if m.role == "user":
                            m.content = self.sanitizer.lam_sach(m.content)

                # 2. Kiểm tra Semantic Cache nếu được bật
                query = ""
                for m in reversed(request.messages):
                    if m.role == "user":
                        query = m.content
                        break

                ket_qua = None
                if self.semantic_cache is not None and query:
                    cache_hit = self.semantic_cache.tra_cuu(query)
                    if cache_hit is not None and "noi_dung" in cache_hit:
                        ket_qua = cache_hit["noi_dung"]

                # 3. Nếu trượt cache, gọi core AI xử lý
                if ket_qua is None:
                    ket_qua = self._xu_ly_logic(request)

                    # Lưu vào cache nếu bật
                    if self.semantic_cache is not None and query:
                        self.semantic_cache.luu(query, {"noi_dung": ket_qua})

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

        @self.app.websocket("/v1/chat/stream")
        async def websocket_endpoint(websocket: WebSocket):
            """Giao thức WebSocket (v27.0.1) thay thế cho HTTP Chunking để giảm tối đa độ trễ."""
            await websocket.accept()
            try:
                import asyncio
                import json
                while True:
                    data = await websocket.receive_text()
                    request_data = json.loads(data)
                    req = ChatCompletionRequest(**request_data)

                    # Quét Tường lửa AI cho cả giao thức WebSocket
                    from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

                    fw = TuongLuaBaoVeFramework()
                    bi_chan = False
                    for m in req.messages:
                        if m.role == "user":
                            kq_in = fw.kiem_tra_dau_vao(m.content)
                            if not kq_in["an_toan"]:
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "error": "SecurityViolation",
                                            "message": f"Yêu cầu WebSocket bị Tường lửa AI chặn: {kq_in['ly_do']}",
                                        }
                                    )
                                )
                                bi_chan = True
                                break
                    if bi_chan:
                        continue

                    self.logger.info(f"Nhận WebSocket stream request (model: {req.model})")
                    ket_qua = fw.loc_dau_ra(self._xu_ly_logic(req))

                    # Giả lập streaming từng từ (chunking) qua WebSocket
                    for tu in ket_qua.split():
                        await websocket.send_text(json.dumps({"chunk": tu + " "}))
                        await asyncio.sleep(0.01) # Độ trễ giả lập 10ms

                    await websocket.send_text(json.dumps({"done": True}))
            except WebSocketDisconnect:
                self.logger.info("Client ngắt kết nối WebSocket")
            except Exception as e:
                self.logger.error(f"Lỗi WebSocket: {str(e)}")
                try:
                    await websocket.close()
                except Exception:
                    pass

    def chay(self, host: str = "0.0.0.0", port: int = 8000):
        """Chạy server với Uvicorn."""
        import uvicorn

        self.logger.info(f"Khởi động FastAPI Server tại http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
