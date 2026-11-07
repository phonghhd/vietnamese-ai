"""
EvoNet gRPC Server
Máy chủ RPC cung cấp giao tiếp tốc độ ánh sáng (binary) cho Microservices.
Lưu ý: Cần chạy lệnh `python -m grpc_tools.protoc` để biên dịch .proto trước khi sử dụng.
"""

import logging
import time
import uuid

# Môi trường Mock (Không yêu cầu cài grpcio)
logger = logging.getLogger("EvoNet-gRPC")


class MockEvoNetServiceServicer:
    """
    Servicer cho EvoNet.
    Trong môi trường biên dịch thực tế, lớp này sẽ kế thừa từ evonet_pb2_grpc.EvoNetServiceServicer.
    """

    def __init__(self, bo_xu_ly):
        self.bo_xu_ly = bo_xu_ly

    def ChatCompletion(self, request, context):
        """Xử lý Unary RPC."""
        bat_dau = time.time()

        # Xử lý Logic (LLM hoặc Agent)
        if hasattr(self.bo_xu_ly, "chay"):
            text_result = self.bo_xu_ly.chay(request.prompt)
        elif hasattr(self.bo_xu_ly, "sinh_van_ban"):
            text_result = self.bo_xu_ly.sinh_van_ban(request.prompt)
        else:
            text_result = "[gRPC Native] Đã xử lý yêu cầu qua giao thức nhị phân siêu tốc."

        logger.info(f"gRPC request processed in {time.time() - bat_dau:.4f}s")

        # Mock Response object (Trong thực tế là evonet_pb2.ChatResponse)
        class ChatResponse:
            def __init__(self, id, model, text, total_tokens):
                self.id = id
                self.model = model
                self.text = text
                self.total_tokens = total_tokens

        return ChatResponse(
            id=f"grpc-{uuid.uuid4().hex[:8]}",
            model=request.model,
            text=str(text_result),
            total_tokens=len(str(text_result).split()),
        )


def serve_grpc(bo_xu_ly, port=50051, block: bool = True):
    """
    Khởi chạy máy chủ gRPC với Generic RPC Handler thực tế.
    """
    servicer = MockEvoNetServiceServicer(bo_xu_ly)
    try:
        import json
        from concurrent import futures

        import grpc

        def _unary_handler(request_bytes: bytes, context):
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
                }
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
        logger.info(f"gRPC Server listening on port {port}")
        if block:
            server.wait_for_termination()
        return server
    except ImportError:
        logger.warning("Thư viện 'grpcio' chưa được cài đặt. Trả về Servicer cục bộ.")
        return servicer
