"""FastAPIServer - API Server tương thích chuẩn OpenAI (SSE Streaming, Tool Calling, OpenTelemetry & Prometheus)."""

import json
import threading
import time
import uuid
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse, StreamingResponse
    from pydantic import BaseModel

    _CO_FASTAPI = True
except ImportError:
    _CO_FASTAPI = False
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    WebSocket = Any  # type: ignore
    WebSocketDisconnect = Exception  # type: ignore
    CORSMiddleware = None  # type: ignore
    PlainTextResponse = None  # type: ignore
    StreamingResponse = None  # type: ignore

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


from vietnamese_ai.utils.logger import Logger


class BoThuThapTelemetryAPI:
    """Bộ thu thập số liệu Prometheus & OpenTelemetry Tracing cho API Production (không bắt buộc cài thư viện ngoài)."""

    def __init__(self, ten_dich_vu: str = "vietnamese-ai-api") -> None:
        self.ten_dich_vu = ten_dich_vu
        self._lock = threading.Lock()
        self.tong_request: int = 0
        self.tong_loi: int = 0
        self.tong_stream: int = 0
        self.tong_tool_calls: int = 0
        self.tong_prompt_tokens: int = 0
        self.tong_completion_tokens: int = 0
        self.tong_do_tre_giay: float = 0.0
        self._tracer: Any = None
        try:
            from opentelemetry import trace  # type: ignore

            self._tracer = trace.get_tracer(ten_dich_vu)
        except Exception:
            self._tracer = None

    def bat_dau_span(self, ten_span: str) -> Any:
        """Tạo OpenTelemetry span nếu có cài `opentelemetry`, ngược lại trả context rỗng không tốn chi phí."""
        if self._tracer is not None:
            return self._tracer.start_as_current_span(ten_span)
        return nullcontext()

    def ghi_nhan(
        self,
        do_tre_giay: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        la_stream: bool = False,
        co_tool_call: bool = False,
        co_loi: bool = False,
    ) -> None:
        """Ghi nhận số liệu an toàn đa luồng cho mỗi request API."""
        with self._lock:
            self.tong_request += 1
            self.tong_do_tre_giay += max(0.0, float(do_tre_giay))
            self.tong_prompt_tokens += max(0, int(prompt_tokens))
            self.tong_completion_tokens += max(0, int(completion_tokens))
            if la_stream:
                self.tong_stream += 1
            if co_tool_call:
                self.tong_tool_calls += 1
            if co_loi:
                self.tong_loi += 1

    def xuat_prometheus(self) -> str:
        """Xuất số liệu chuẩn Prometheus Text Exposition Format 0.0.4."""
        with self._lock:
            trung_binh = self.tong_do_tre_giay / self.tong_request if self.tong_request else 0.0
            lines = [
                "# HELP vai_api_requests_total Tong so yeu cau API",
                "# TYPE vai_api_requests_total counter",
                f'vai_api_requests_total{{service="{self.ten_dich_vu}"}} {self.tong_request}',
                "# HELP vai_api_errors_total Tong so yeu cau loi",
                "# TYPE vai_api_errors_total counter",
                f'vai_api_errors_total{{service="{self.ten_dich_vu}"}} {self.tong_loi}',
                "# HELP vai_api_stream_requests_total Tong so yeu cau SSE streaming",
                "# TYPE vai_api_stream_requests_total counter",
                f'vai_api_stream_requests_total{{service="{self.ten_dich_vu}"}} {self.tong_stream}',
                "# HELP vai_api_tool_calls_total Tong so luot goi cong cu OpenAI Tool Calling",
                "# TYPE vai_api_tool_calls_total counter",
                f'vai_api_tool_calls_total{{service="{self.ten_dich_vu}"}} {self.tong_tool_calls}',
                "# HELP vai_api_tokens_total Tong so token xu ly theo loai",
                "# TYPE vai_api_tokens_total counter",
                f'vai_api_tokens_total{{service="{self.ten_dich_vu}",kind="prompt"}} {self.tong_prompt_tokens}',
                f'vai_api_tokens_total{{service="{self.ten_dich_vu}",kind="completion"}} {self.tong_completion_tokens}',
                "# HELP vai_api_latency_seconds_sum Tong do tre xu ly (giay)",
                "# TYPE vai_api_latency_seconds_sum counter",
                f'vai_api_latency_seconds_sum{{service="{self.ten_dich_vu}"}} {self.tong_do_tre_giay:.6f}',
                "# HELP vai_api_latency_seconds_avg Do tre trung binh (giay)",
                "# TYPE vai_api_latency_seconds_avg gauge",
                f'vai_api_latency_seconds_avg{{service="{self.ten_dich_vu}"}} {trung_binh:.6f}',
            ]
        return "\n".join(lines) + "\n"


# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str = ""
    content: Optional[str] = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str = "vietnamese-ai-default"
    messages: List[ChatMessage] = []
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    auto_execute_tools: bool = False


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


class LegalClauseVersionRequest(BaseModel):
    so_hieu: str
    dieu: str
    ngay_tra_cuu: Optional[str] = None
    khoan: Optional[str] = None


class LegalTemporalAskRequest(BaseModel):
    cau_hoi: str
    ngay_tra_cuu: Optional[str] = None
    che_do: str = "full"
    top_k: int = 5


class LegalVerifyCitationRequest(BaseModel):
    cau_tra_loi: str
    nguon_truy_hoi: List[Dict[str, Any]] = []
    ngay_tra_cuu: Optional[str] = None
    cau_hoi: Optional[str] = None


class ContractReviewRequest(BaseModel):
    hop_dong_doi_tac: str
    hop_dong_chuan: Optional[str] = None
    ngay_tra_cuu: Optional[str] = None
    xuat_track_changes_docx: bool = False
    tac_gia: str = "Vietnamese-AI Legal Auditor"


class TokenizeRequest(BaseModel):
    van_ban: str
    phuong_phap: str = "viterbi"
    noi_tu_ghep: bool = True
    chuan_hoa_dau: bool = True


class VAIDBHnswRequest(BaseModel):
    m: int = 12
    ef_construction: int = 64
    ef_search: int = 48
    chi_thong_ke: bool = False


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
        quan_ly_plugin: Optional[Any] = None,
        thuc_thi_tool_tu_dong: bool = False,
    ):
        """
        Khởi tạo Server.

        Args:
            bo_xu_ly: Có thể là một đối tượng `HeThongDaTacTu`, `TacTu`, `DongCoSuyLuanEdge`,
                      `QuyTrinhDauCuoi` hoặc LLM Wrapper (`chay`, `sinh_van_ban`, `sinh_stream`, `hoi`).
            ten: Tên định danh của server.
            enable_watermark: Tự động nhúng thủy ấn số vào câu trả lời sinh ra.
            enable_sanitizer: Tự động làm sạch và ẩn danh hóa dữ liệu PII trong câu hỏi.
            enable_semantic_cache: Kích hoạt bộ nhớ đệm ngữ nghĩa để trả về câu trả lời đã có.
            semantic_cache_threshold: Ngưỡng độ tương đồng cosine tối thiểu để tính là cache hit.
            ho_du_lieu: Đối tượng HoDuLieu (hoặc đường dẫn) để thu thập feedback DPO khép kín.
            mach_cat: Đối tượng MachCat bảo vệ circuit breaker cho các requests.
            suc_khoe: Đối tượng KiemTraSucKhoe giám sát trạng thái hệ thống.
            quan_ly_plugin: Đối tượng `QuanLyPlugin` tùy chọn để tự động khám phá công cụ OpenAI Tool Calling.
            thuc_thi_tool_tu_dong: Tự động thực thi công cụ đã đăng ký khi khớp `tool_calls`.
        """
        global _CO_FASTAPI

        if not _CO_FASTAPI:
            from vietnamese_ai.utils.lazy_loader import QuanLyPhuThuoc

            QuanLyPhuThuoc.yeu_cau("fastapi", nhom_extra="api", tinh_nang="FastAPIServer")
            from fastapi import (
                FastAPI as _FastAPI,
            )
            from fastapi import (
                HTTPException as _HTTPException,
            )
            from fastapi import (
                WebSocket as _WebSocket,
            )
            from fastapi import (
                WebSocketDisconnect as _WebSocketDisconnect,
            )
            from fastapi.middleware.cors import CORSMiddleware as _CORSMiddleware
            from fastapi.responses import (
                PlainTextResponse as _PlainTextResponse,
            )
            from fastapi.responses import (
                StreamingResponse as _StreamingResponse,
            )

            globals().update(
                {
                    "FastAPI": _FastAPI,
                    "HTTPException": _HTTPException,
                    "WebSocket": _WebSocket,
                    "WebSocketDisconnect": _WebSocketDisconnect,
                    "CORSMiddleware": _CORSMiddleware,
                    "PlainTextResponse": _PlainTextResponse,
                    "StreamingResponse": _StreamingResponse,
                }
            )
            _CO_FASTAPI = True

        self.bo_xu_ly = bo_xu_ly
        self.ten = ten
        self.enable_watermark = enable_watermark
        self.enable_sanitizer = enable_sanitizer
        self.enable_semantic_cache = enable_semantic_cache
        self.mach_cat = mach_cat or getattr(bo_xu_ly, "mach_cat", None)
        self.suc_khoe = suc_khoe or getattr(bo_xu_ly, "suc_khoe", None)
        self.quan_ly_plugin = quan_ly_plugin
        self.thuc_thi_tool_tu_dong = thuc_thi_tool_tu_dong
        self.telemetry = BoThuThapTelemetryAPI(ten_dich_vu=ten)
        self.logger = Logger(ten)

        self.ho_du_lieu: Optional[Any] = None
        if ho_du_lieu is not None:
            if isinstance(ho_du_lieu, str):
                from vietnamese_ai.datalake.dataset_manager import HoDuLieu

                self.ho_du_lieu = HoDuLieu(duong_dan_luu_tru=ho_du_lieu)
            else:
                self.ho_du_lieu = ho_du_lieu

        self.semantic_cache: Optional[Any] = None
        if enable_semantic_cache:
            try:
                from vietnamese_ai.rag.context_optimization import BoNhoDemNguNghia

                self.semantic_cache = BoNhoDemNguNghia(nguong_tuong_dong=semantic_cache_threshold)
            except ImportError:
                self.semantic_cache = None

        self.sanitizer: Optional[Any] = None
        if enable_sanitizer:
            try:
                from vietnamese_ai.security.data_sanitizer import DataSanitizer

                self.sanitizer = DataSanitizer
            except ImportError:
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
        if not request.messages:
            raise ValueError("Danh sách tin nhắn `messages` không được để trống.")
        truy_van_cuoi = request.messages[-1].content
        if getattr(request, "model", "") in ("swarm", "edge") or getattr(request, "che_do", "") == "swarm":
            if hasattr(self.bo_xu_ly, "suy_luan_edge"):
                res = self.bo_xu_ly.suy_luan_edge(truy_van_cuoi)
                return str(res.get("cau_tra_loi", res))
            elif hasattr(self.bo_xu_ly, "suy_luan_swarm"):
                res = self.bo_xu_ly.suy_luan_swarm(truy_van_cuoi)
                return str(res.get("cau_tra_loi", res))

        if hasattr(self.bo_xu_ly, "chay"):
            return self.bo_xu_ly.chay(truy_van_cuoi)
        elif hasattr(self.bo_xu_ly, "tra_cuu"):
            kq = self.bo_xu_ly.tra_cuu(truy_van_cuoi)
            return str(kq.get("tra_loi", kq) if isinstance(kq, dict) else kq)
        elif hasattr(self.bo_xu_ly, "xu_ly_yeu_cau"):
            kq = self.bo_xu_ly.xu_ly_yeu_cau(truy_van_cuoi)
            return str(kq.get("tra_loi", kq) if isinstance(kq, dict) else kq)
        elif hasattr(self.bo_xu_ly, "sinh_kem_thong_ke"):
            kq = self.bo_xu_ly.sinh_kem_thong_ke(
                truy_van_cuoi,
                do_dai_toi_da=request.max_tokens,
                nhiet_do=request.temperature,
            )
            return str(kq.get("van_ban", kq) if isinstance(kq, dict) else kq)
        elif hasattr(self.bo_xu_ly, "sinh_van_ban"):
            prompt = self._chuyen_doi_messages(request.messages)
            return self.bo_xu_ly.sinh_van_ban(
                prompt, nhiet_do=request.temperature, do_dai=request.max_tokens
            )
        elif hasattr(self.bo_xu_ly, "hoi"):
            kq = self.bo_xu_ly.hoi(truy_van_cuoi)
            return str(kq.get("tra_loi", kq))
        else:
            raise ValueError(
                "Bộ xử lý không hợp lệ. Cần hỗ trợ một trong các hàm: "
                "`chay`, `tra_cuu`, `xu_ly_yeu_cau`, `sinh_kem_thong_ke`, `sinh_van_ban` hoặc `hoi`."
            )

    def _xu_ly_logic(self, request: ChatCompletionRequest) -> str:
        if self.mach_cat is not None:
            return self.mach_cat.thuc_hien(self._xu_ly_logic_truc_tiep, request)
        return self._xu_ly_logic_truc_tiep(request)

    def _trich_xuat_tools_tu_he_thong(self) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
        """Khám phá tự động danh sách công cụ từ `TacTu`, `QuyTrinhDauCuoi` và `QuanLyPlugin` sang lược đồ OpenAI JSON Schema."""
        schemas: List[Dict[str, Any]] = []
        executors: Dict[str, Callable[..., Any]] = {}
        seen: Set[str] = set()

        cac_nguon_cong_cu: List[Dict[str, Any]] = []
        if isinstance(getattr(self.bo_xu_ly, "cong_cu", None), dict):
            cac_nguon_cong_cu.append(self.bo_xu_ly.cong_cu)
        tac_tu = getattr(self.bo_xu_ly, "tac_tu", None)
        if tac_tu is not None and isinstance(getattr(tac_tu, "cong_cu", None), dict):
            cac_nguon_cong_cu.append(tac_tu.cong_cu)

        for nguon in cac_nguon_cong_cu:
            for ten_cc, cc_obj in nguon.items():
                if ten_cc in seen:
                    continue
                seen.add(ten_cc)
                tham_so_raw = getattr(cc_obj, "tham_so", {}) or {}
                props: Dict[str, Any] = {}
                required: List[str] = []
                for p_name, p_info in tham_so_raw.items():
                    kieu = str(p_info.get("kieu", "string")).lower() if isinstance(p_info, dict) else "string"
                    json_type = "integer" if "int" in kieu else ("number" if "float" in kieu else ("boolean" if "bool" in kieu else "string"))
                    props[p_name] = {"type": json_type}
                    if isinstance(p_info, dict) and p_info.get("bat_buoc"):
                        required.append(p_name)
                if not props:
                    props = {"truy_van": {"type": "string"}}
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": ten_cc,
                            "description": str(getattr(cc_obj, "mo_ta", "")),
                            "parameters": {
                                "type": "object",
                                "properties": props,
                                "required": required,
                            },
                        },
                    }
                )
                executors[ten_cc] = lambda _obj=cc_obj, **kw: _obj.chay(**kw)

        ql_plugin = self.quan_ly_plugin or getattr(self.bo_xu_ly, "quan_ly_plugin", None)
        if ql_plugin is not None:
            plugins_cong_cu = getattr(ql_plugin, "cac_plugin_cong_cu", None)
            if isinstance(plugins_cong_cu, dict):
                for plugin in plugins_cong_cu.values():
                    if not hasattr(plugin, "dang_ky_cong_cu"):
                        continue
                    for t_info in plugin.dang_ky_cong_cu():
                        if not isinstance(t_info, dict) or not t_info.get("ten"):
                            continue
                        t_name = str(t_info["ten"])
                        if t_name in seen:
                            continue
                        seen.add(t_name)
                        t_params = t_info.get("tham_so")
                        props_p: Dict[str, Any] = {}
                        if isinstance(t_params, dict):
                            for k_p, v_p in t_params.items():
                                props_p[str(k_p)] = {"type": "string", "description": str(v_p)}
                        if not props_p:
                            props_p = {"input": {"type": "string"}}
                        schemas.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": t_name,
                                    "description": str(t_info.get("mo_ta", "")),
                                    "parameters": {
                                        "type": "object",
                                        "properties": props_p,
                                    },
                                },
                            }
                        )
                        executors[t_name] = lambda _pl=plugin, _tn=t_name, **kw: _pl.thuc_thi_cong_cu(_tn, **kw)

        return schemas, executors

    def _lay_luong_stream_truc_tiep(self, request: ChatCompletionRequest) -> Optional[Iterator[str]]:
        """Lấy bộ sinh token thời gian thực (`sinh_stream`) từ bộ xử lý/Edge/Swarm nếu hỗ trợ để truyền SSE không chờ đệm."""
        truy_van = (request.messages[-1].content or "") if request.messages else ""
        prompt = self._chuyen_doi_messages(request.messages)
        muc_tieu: Any = None
        dau_vao = prompt

        if getattr(request, "model", "") in ("swarm", "edge"):
            if hasattr(self.bo_xu_ly, "dong_co_edge") and getattr(self.bo_xu_ly, "dong_co_edge") is not None:
                muc_tieu = self.bo_xu_ly.dong_co_edge
                dau_vao = truy_van
            elif hasattr(self.bo_xu_ly, "dieu_phoi_swarm") and getattr(self.bo_xu_ly, "dieu_phoi_swarm") is not None:
                muc_tieu = self.bo_xu_ly.dieu_phoi_swarm
                dau_vao = truy_van

        if muc_tieu is None and hasattr(self.bo_xu_ly, "sinh_stream"):
            muc_tieu = self.bo_xu_ly
            dau_vao = truy_van or prompt

        if muc_tieu is None or not hasattr(muc_tieu, "sinh_stream"):
            return None

        try:
            return iter(
                muc_tieu.sinh_stream(
                    dau_vao,
                    do_dai_toi_da=request.max_tokens,
                    nhiet_do=request.temperature,
                )
            )
        except TypeError:
            try:
                return iter(muc_tieu.sinh_stream(dau_vao, do_dai_toi_da=request.max_tokens))
            except TypeError:
                return iter(muc_tieu.sinh_stream(dau_vao))

    def _chon_tool_calls_openai(
        self,
        request: ChatCompletionRequest,
        query: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Khớp lược đồ OpenAI Function/Tool Calling (`tools` & `tool_choice`) hoặc công cụ tự động từ `TacTu`/`QuanLyPlugin`."""
        tools = getattr(request, "tools", None)
        if request.tool_choice == "none":
            return None
        if not tools and request.tool_choice is not None:
            tools, _ = self._trich_xuat_tools_tu_he_thong()
        if not tools:
            return None

        ten_ep: Optional[str] = None
        if isinstance(request.tool_choice, dict):
            fn_obj = request.tool_choice.get("function")
            if isinstance(fn_obj, dict) and fn_obj.get("name"):
                ten_ep = str(fn_obj["name"])

        q_lower = query.lower()
        chon_fn: Optional[Dict[str, Any]] = None
        best_score = 0
        for item in tools:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") if item.get("type") == "function" else item
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            name = str(fn["name"])
            if ten_ep is not None:
                if name == ten_ep:
                    chon_fn = fn
                    break
                continue
            desc = str(fn.get("description", "")).lower()
            score = sum(1 for w in name.lower().replace("_", " ").split() if len(w) > 1 and w in q_lower)
            score += sum(1 for w in desc.split() if len(w) > 3 and w in q_lower)
            if score > best_score:
                best_score = score
                chon_fn = fn

        if chon_fn is None and request.tool_choice == "required":
            first = tools[0]
            if isinstance(first, dict):
                chon_fn = first.get("function") if first.get("type") == "function" else first

        if chon_fn is None:
            return None

        params_schema = chon_fn.get("parameters") or {}
        props = params_schema.get("properties") if isinstance(params_schema, dict) else {}
        args_dict: Dict[str, Any] = {}
        if isinstance(props, dict) and props:
            first_key = next(iter(props.keys()))
            args_dict[first_key] = query
        else:
            args_dict["input"] = query

        return [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": str(chon_fn["name"]),
                    "arguments": json.dumps(args_dict, ensure_ascii=False),
                },
            }
        ]

    def _tao_luong_sse_truc_tiep(
        self,
        response_id: str,
        model_name: str,
        luong_token: Iterator[str],
        input_tokens: int,
        bat_dau: float,
        query: str = "",
    ) -> Iterator[str]:
        """Truyền từng token thực tế từ `sinh_stream` ngay lập tức ra luồng SSE OpenAI không cần gom đệm toàn câu."""
        created = int(time.time())
        head_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(head_chunk, ensure_ascii=False)}\n\n"

        cac_phan: List[str] = []
        try:
            for piece in luong_token:
                text_piece = str(piece)
                if not text_piece:
                    continue
                cac_phan.append(text_piece)
                body_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": text_piece}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(body_chunk, ensure_ascii=False)}\n\n"
        finally:
            toan_van = "".join(cac_phan)
            if self.semantic_cache is not None and query and toan_van:
                self.semantic_cache.luu(query, {"noi_dung": toan_van})
            self.telemetry.ghi_nhan(
                do_tre_giay=time.time() - bat_dau,
                prompt_tokens=input_tokens,
                completion_tokens=max(len(cac_phan), len(toan_van.split())),
                la_stream=True,
            )

        end_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    @staticmethod
    def _tao_luong_sse_openai(
        response_id: str,
        model_name: str,
        noi_dung: str,
    ) -> Iterator[str]:
        """Sinh luồng sự kiện Server-Sent Events (SSE) chuẩn OpenAI `chat.completion.chunk`."""
        created = int(time.time())
        head_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(head_chunk, ensure_ascii=False)}\n\n"

        cac_tu = noi_dung.split(" ") if noi_dung else [""]
        for idx, tu in enumerate(cac_tu):
            piece = tu + (" " if idx < len(cac_tu) - 1 else "")
            body_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(body_chunk, ensure_ascii=False)}\n\n"

        end_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"status": "hoat_dong", "ten": self.ten}

        @self.app.get("/metrics")
        async def metrics():
            """Xuất chỉ số vận hành chuẩn Prometheus Text Exposition Format."""
            body = self.telemetry.xuat_prometheus()
            if PlainTextResponse is not None:
                return PlainTextResponse(body, media_type="text/plain; version=0.0.4")
            return body

        @self.app.get("/v1/tools")
        async def list_tools():
            """Trả về danh sách công cụ OpenAI JSON Schema khám phá tự động từ `TacTu` và `QuanLyPlugin`."""
            schemas, _ = self._trich_xuat_tools_tu_he_thong()
            return {"object": "list", "so_luong": len(schemas), "data": schemas}

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
            if HTTPException is not None:
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

        @self.app.post("/v1/legal/clause-version")
        async def legal_clause_version(req: LegalClauseVersionRequest):
            """Tra cứu phiên bản hiệu lực theo mốc thời gian (Point-in-Time) của một Điều/Khoản luật Việt Nam."""
            from vietnamese_ai.rag.legal.validity_tracker import BoTheoDoiHieuLucPhapLy

            tracker = getattr(self.bo_xu_ly, "bo_theo_doi_hieu_luc", None) or BoTheoDoiHieuLucPhapLy()
            return tracker.tra_cuu_dieu_khoan_theo_ngay(
                so_hieu=req.so_hieu,
                dieu=req.dieu,
                ngay_tra_cuu=req.ngay_tra_cuu,
                khoan=req.khoan,
            )

        @self.app.post("/v1/legal/temporal-ask")
        async def legal_temporal_ask(req: LegalTemporalAskRequest):
            """Hỏi đáp pháp lý theo mốc thời gian & đồ thị điều khoản đa bước kèm kiểm chứng trích dẫn 3 tầng."""
            from vietnamese_ai.rag.legal.temporal_rag import tao_dong_co_phap_ly_thoi_gian_mac_dinh

            engine = getattr(self, "_dong_co_phap_ly_thoi_gian", None)
            if engine is None:
                engine = tao_dong_co_phap_ly_thoi_gian_mac_dinh()
                self._dong_co_phap_ly_thoi_gian = engine
            return engine.tra_loi_phap_ly(
                cau_hoi=req.cau_hoi,
                ngay_tra_cuu=req.ngay_tra_cuu,
                top_k=max(1, min(20, int(req.top_k))),
                che_do=req.che_do,
            )

        @self.app.post("/v1/legal/verify-citation")
        async def legal_verify_citation(req: LegalVerifyCitationRequest):
            """Kiểm chứng trích dẫn pháp lý 3 tầng (Grounding, Temporal Consistency, Numeric Consistency)."""
            from vietnamese_ai.rag.legal.citation_verifier import BoKiemChungTrichDanPhapLy

            verifier = BoKiemChungTrichDanPhapLy(
                bo_theo_doi=getattr(self.bo_xu_ly, "bo_theo_doi_hieu_luc", None)
            )
            return verifier.kiem_chung(
                cau_tra_loi=req.cau_tra_loi,
                nguon_truy_hoi=req.nguon_truy_hoi,
                ngay_tra_cuu=req.ngay_tra_cuu,
                cau_hoi=req.cau_hoi,
            ).to_dict()

        @self.app.post("/v1/legal/contract-review")
        async def legal_contract_review(req: ContractReviewRequest):
            """So sánh hợp đồng (tách/gộp điều khoản), quét rủi ro pháp lý và tùy chọn xuất Word Track Changes + Comments (.docx)."""
            import base64
            import tempfile
            from pathlib import Path

            from vietnamese_ai.rag.legal.contract_review import BoSoSanhRaSoatHopDong

            engine = BoSoSanhRaSoatHopDong()
            kq = engine.so_sanh_va_ra_soat(
                hop_dong_doi_tac=req.hop_dong_doi_tac,
                hop_dong_chuan=req.hop_dong_chuan,
                ngay_tra_cuu=req.ngay_tra_cuu,
            )
            kq["bao_cao_markdown"] = engine.xuat_bao_cao_markdown(kq)
            if req.xuat_track_changes_docx:
                with tempfile.TemporaryDirectory(prefix="vai_api_tc_") as tmp_dir:
                    out_path = Path(tmp_dir) / "hop_dong_track_changes.docx"
                    engine.xuat_bao_cao_track_changes_docx(kq, out_path, tac_gia=req.tac_gia)
                    kq["track_changes_docx_base64"] = base64.b64encode(out_path.read_bytes()).decode("ascii")
            return kq

        @self.app.post("/v1/nlp/tokenize")
        async def nlp_tokenize(req: TokenizeRequest):
            """Chuẩn hóa dấu thanh âm vị học FST và tách từ ghép tiếng Việt bằng Viterbi DAG hoặc MaxMatch."""
            from vietnamese_ai.nlp.native_tokenizer import BoTachTuTiengVietSieuToc

            tok = BoTachTuTiengVietSieuToc()
            vb_chuan = tok.chuan_hoa_dau_thanh(req.van_ban) if req.chuan_hoa_dau else tok.chuan_hoa_unicode(req.van_ban)
            tokens = tok.tach_tu(
                req.van_ban,
                noi_tu_ghep=req.noi_tu_ghep,
                chuan_hoa_dau=req.chuan_hoa_dau,
                phuong_phap=req.phuong_phap,
            )
            return {
                "van_ban_chuan_hoa": vb_chuan,
                "phuong_phap": req.phuong_phap,
                "so_token": len(tokens),
                "tokens": tokens,
            }

        @self.app.post("/v1/vaidb/hnsw")
        async def vaidb_hnsw(req: VAIDBHnswRequest):
            """Xem thống kê hoặc xây dựng chỉ mục đồ thị HNSW bền vững trên kho `.vaidb` gắn với bộ xử lý."""
            store = getattr(self.bo_xu_ly, "csdl_vector", None)
            if store is None:
                rag_obj = getattr(self.bo_xu_ly, "rag", None)
                store = getattr(rag_obj, "csdl_vector", None) if rag_obj is not None else None
            if store is None or not hasattr(store, "xay_dung_chi_muc_hnsw"):
                return {
                    "ho_tro": False,
                    "thong_bao": "Bộ xử lý hiện tại chưa gắn kho tri thức bền vững KhoTriThucVAIDB.",
                }
            if req.chi_thong_ke:
                return {"ho_tro": True, "thong_ke": store.thong_ke()}
            info = store.xay_dung_chi_muc_hnsw(
                m=req.m,
                ef_construction=req.ef_construction,
                ef_search=req.ef_search,
            )
            return {"ho_tro": True, "hnsw": info, "thong_ke": store.thong_ke()}

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            self.logger.info(f"Nhận request chat completion (model: {request.model})")
            bat_dau = time.time()

            with self.telemetry.bat_dau_span("v1.chat.completions"):
                try:
                    if self.sanitizer is not None:
                        for m in request.messages:
                            if m.role == "user" and m.content:
                                m.content = self.sanitizer.lam_sach(m.content)

                    query = ""
                    for m in reversed(request.messages):
                        if m.role == "user" and m.content:
                            query = m.content
                            break

                    input_tokens = sum(len((m.content or "").split()) for m in request.messages)
                    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

                    tool_calls = self._chon_tool_calls_openai(request, query)
                    if tool_calls is not None:
                        noi_dung_tool: Optional[str] = None
                        ly_do_ket_thuc = "tool_calls"
                        if self.thuc_thi_tool_tu_dong or getattr(request, "auto_execute_tools", False):
                            _, executors = self._trich_xuat_tools_tu_he_thong()
                            ket_qua_tools: List[str] = []
                            for tc in tool_calls:
                                fn_block = tc.get("function") or {}
                                fn_name = str(fn_block.get("name", ""))
                                executor = executors.get(fn_name)
                                if executor is not None:
                                    try:
                                        args_parsed = json.loads(str(fn_block.get("arguments", "{}")))
                                    except Exception:
                                        args_parsed = {}
                                    kq_exec = executor(**args_parsed) if isinstance(args_parsed, dict) else executor(args_parsed)
                                    ket_qua_tools.append(str(kq_exec))
                            if ket_qua_tools:
                                noi_dung_tool = "\n".join(ket_qua_tools)
                                ly_do_ket_thuc = "stop"

                        self.telemetry.ghi_nhan(
                            do_tre_giay=time.time() - bat_dau,
                            prompt_tokens=input_tokens,
                            completion_tokens=len(tool_calls),
                            co_tool_call=True,
                        )
                        return ChatCompletionResponse(
                            id=response_id,
                            created=int(time.time()),
                            model=request.model,
                            choices=[
                                ChatCompletionResponseChoice(
                                    index=0,
                                    message=ChatMessage(
                                        role="assistant",
                                        content=noi_dung_tool,
                                        tool_calls=tool_calls,
                                    ),
                                    finish_reason=ly_do_ket_thuc,
                                )
                            ],
                            usage=Usage(
                                prompt_tokens=input_tokens,
                                completion_tokens=len(tool_calls),
                                total_tokens=input_tokens + len(tool_calls),
                            ),
                        )

                    ket_qua = None
                    if self.semantic_cache is not None and query:
                        cache_hit = self.semantic_cache.tra_cuu(query)
                        if cache_hit is not None and "noi_dung" in cache_hit:
                            ket_qua = cache_hit["noi_dung"]

                    if ket_qua is None and request.stream and StreamingResponse is not None:
                        luong_truc_tiep = self._lay_luong_stream_truc_tiep(request)
                        if luong_truc_tiep is not None:
                            return StreamingResponse(
                                self._tao_luong_sse_truc_tiep(
                                    response_id=response_id,
                                    model_name=request.model,
                                    luong_token=luong_truc_tiep,
                                    input_tokens=input_tokens,
                                    bat_dau=bat_dau,
                                    query=query,
                                ),
                                media_type="text/event-stream",
                            )

                    if ket_qua is None:
                        ket_qua = self._xu_ly_logic(request)
                        if self.semantic_cache is not None and query:
                            self.semantic_cache.luu(query, {"noi_dung": ket_qua})

                    if self.enable_watermark:
                        try:
                            from vietnamese_ai.security.watermark import TextWatermarker

                            ket_qua = TextWatermarker.nhung_thuy_an(ket_qua, "EVONETAI_API")
                        except ImportError:
                            pass

                    output_tokens = len(ket_qua.split())
                    if request.stream and StreamingResponse is not None:
                        self.telemetry.ghi_nhan(
                            do_tre_giay=time.time() - bat_dau,
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            la_stream=True,
                        )
                        return StreamingResponse(
                            self._tao_luong_sse_openai(response_id, request.model, ket_qua),
                            media_type="text/event-stream",
                        )

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
                    self.telemetry.ghi_nhan(
                        do_tre_giay=time.time() - bat_dau,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )
                    self.logger.info(f"Hoàn thành trong {time.time() - bat_dau:.2f}s")
                    return resp

                except Exception as e:
                    self.telemetry.ghi_nhan(do_tre_giay=time.time() - bat_dau, co_loi=True)
                    self.logger.error(f"Lỗi xử lý request: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/v1/chat/stream")
        async def websocket_endpoint(websocket: WebSocket):
            """Giao thức WebSocket (v27.0.1) thay thế cho HTTP Chunking để giảm tối đa độ trễ."""
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    request_data = json.loads(data)
                    req = ChatCompletionRequest(**request_data)

                    from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

                    fw = TuongLuaBaoVeFramework()
                    bi_chan = False
                    for m in req.messages:
                        if m.role == "user" and m.content:
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
                    luong_ws = self._lay_luong_stream_truc_tiep(req)
                    if luong_ws is not None:
                        for piece in luong_ws:
                            await websocket.send_text(json.dumps({"chunk": fw.loc_dau_ra(str(piece))}))
                    else:
                        ket_qua = fw.loc_dau_ra(self._xu_ly_logic(req))
                        for tu in ket_qua.split():
                            await websocket.send_text(json.dumps({"chunk": tu + " "}))

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
