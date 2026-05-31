"""
EvoNet gRPC Server
Máy chủ RPC cung cấp giao tiếp tốc độ ánh sáng (binary) cho Microservices.
Lưu ý: Cần chạy lệnh `python -m grpc_tools.protoc` để biên dịch .proto trước khi sử dụng.
"""

import time
import uuid
import logging

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
            total_tokens=len(str(text_result).split())
        )

def serve_grpc(bo_xu_ly, port=50051):
    """
    Khởi chạy máy chủ gRPC.
    """
    try:
        import grpc
        from concurrent import futures
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        # Thực tế sẽ gọi: evonet_pb2_grpc.add_EvoNetServiceServicer_to_server(MockEvoNetServiceServicer(bo_xu_ly), server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        logger.info(f"gRPC Server listening on port {port}")
        server.wait_for_termination()
    except ImportError:
        logger.warning("Thư viện 'grpcio' chưa được cài đặt. Bỏ qua khởi động gRPC Server.")
