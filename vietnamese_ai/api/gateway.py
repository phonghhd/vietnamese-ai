"""
EvoNetAI Gateway
Mở rộng từ FastAPIServer để hỗ trợ việc xuất OpenAPI Schema cho các SDK Đa ngôn ngữ (Polyglot).
"""

from typing import Any, Dict

from vietnamese_ai.api.fastapi_server import FastAPIServer


class EvoNetGateway(FastAPIServer):
    """
    Gateway API Trung tâm.
    Phục vụ cho các SDK Node.js, PHP và hệ thống Microservices.
    """

    def __init__(self, bo_xu_ly: Any, ten: str = "EvoNetAI Gateway", enable_watermark: bool = True):
        # Đặt lại tên để file OpenAPI JSON sinh ra có đúng tên brand "EvoNetAI"
        super().__init__(bo_xu_ly=bo_xu_ly, ten=ten, enable_watermark=enable_watermark)

        self._setup_sdk_routes()

    def _setup_sdk_routes(self):
        """Thêm các Endpoint chuyên dụng cho việc sinh và hỗ trợ SDK."""

        @self.app.get("/sdk/openapi.json", summary="Lấy OpenAPI Schema")
        async def get_openapi_schema() -> Dict[str, Any]:
            """
            Trả về lược đồ OpenAPI chuẩn mực của EvoNetAI.
            Các công cụ như openapi-generator có thể dùng file này để Auto-gen SDK cho 50+ ngôn ngữ.
            """
            return self.app.openapi()

        @self.app.get("/sdk/version", summary="Phiên bản API")
        async def get_api_version() -> Dict[str, str]:
            """
            Dùng để SDK kiểm tra tính tương thích phiên bản.
            """
            return {
                "name": "EvoNetAI Gateway",
                "version": "25.0.0",
                "supported_sdks": ["nodejs", "php", "python"],
            }
