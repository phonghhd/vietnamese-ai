"""EvoNet gRPC Server - Máy Chủ RPC Thực Thi Suy Luận Thật Cho Microservices."""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("EvoNet-gRPC")


@dataclass
class ChatResponse:
    """Cấu trúc phản hồi RPC chuẩn hóa tương thích `evonet.proto`."""

    id: str
    model: str
    text: str
    total_tokens: int


class EvoNetServiceServicer:
    """
    Servicer gRPC thực thi suy luận thực tế:
    - Gọi trực tiếp `bo_xu_ly.chay(...)`, `bo_xu_ly.sinh_van_ban(...)` hoặc hàm callable truyền vào.
    - Nếu `bo_xu_ly` là `None`, tự động khởi tạo `DongCoSuyLuanEdge` chạy suy luận CPU thực tế.
    """

    def __init__(self, bo_xu_ly: Optional[Any] = None) -> None:
        self.bo_xu_ly = bo_xu_ly
        self._edge_default: Optional[Any] = None

    def _lay_dong_co_mac_dinh(self) -> Any:
        """Khởi tạo lười `DongCoSuyLuanEdge` khi không truyền bộ xử lý tùy chỉnh."""
        if self._edge_default is None:
            from vietnamese_ai.edge.native_inference import DongCoSuyLuanEdge

            self._edge_default = DongCoSuyLuanEdge(so_luong_thread=2)
        return self._edge_default

    def ChatCompletion(self, request: Any, context: Any) -> ChatResponse:
        """Xử lý Unary RPC ChatCompletion bằng bộ xử lý hoặc động cơ Edge thực tế."""
        bat_dau = time.perf_counter()
        prompt = str(getattr(request, "prompt", ""))
        model_name = str(getattr(request, "model", "vietnamese-ai-edge"))

        if self.bo_xu_ly is not None and hasattr(self.bo_xu_ly, "chay"):
            text_result = str(self.bo_xu_ly.chay(prompt))
        elif self.bo_xu_ly is not None and hasattr(self.bo_xu_ly, "sinh_van_ban"):
            text_result = str(self.bo_xu_ly.sinh_van_ban(prompt))
        elif callable(self.bo_xu_ly):
            text_result = str(self.bo_xu_ly(prompt))
        else:
            suy_luan = self._lay_dong_co_mac_dinh().sinh_van_ban(prompt, do_dai_toi_da=64)
            text_result = f"[gRPC Native] {suy_luan}"

        logger.info("gRPC request processed in %.4fs", time.perf_counter() - bat_dau)
        return ChatResponse(
            id=f"grpc-{uuid.uuid4().hex[:8]}",
            model=model_name,
            text=text_result,
            total_tokens=len(text_result.split()),
        )


def serve_grpc(bo_xu_ly: Optional[Any], port: int = 50051, block: bool = True) -> Any:
    """Khởi chạy máy chủ gRPC thực tế bằng `grpcio` với Generic RPC Handler."""
    servicer = EvoNetServiceServicer(bo_xu_ly)
    try:
        import grpc

        def _unary_handler(request_bytes: bytes, context: Any) -> bytes:
            payload = json.loads(request_bytes.decode("utf-8"))

            class _Req:
                prompt = payload.get("prompt", "")
                model = payload.get("model", "evonet")

            resp = servicer.ChatCompletion(_Req(), context)
            return json.dumps(
                {
                    "id": resp.id,
                    "model": resp.model,
                    "text": resp.text,
                    "total_tokens": resp.total_tokens,
                },
                ensure_ascii=False,
            ).encode("utf-8")

        generic_handler = grpc.method_handlers_generic_handler(
            "evonet.EvoNetService",
            {
                "ChatCompletion": grpc.unary_unary_rpc_method_handler(_unary_handler),
            },
        )
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            handlers=(generic_handler,),
        )
        server.add_insecure_port(f"[::]:{port}")
        server.start()
        logger.info("EvoNet gRPC Server đang lắng nghe trên cổng %d", port)
        if block:
            server.wait_for_termination()
        return server
    except ImportError as exc:
        raise ImportError(
            "Thiếu thư viện 'grpcio' để khởi chạy máy chủ gRPC. "
            'Hãy cài đặt: pip install "vietnamese-ai[grpc]"'
        ) from exc


MockEvoNetServiceServicer = EvoNetServiceServicer

__all__ = [
    "ChatResponse",
    "EvoNetServiceServicer",
    "MockEvoNetServiceServicer",
    "serve_grpc",
]
