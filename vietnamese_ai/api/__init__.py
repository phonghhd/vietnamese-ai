"""API module - Giao diện phục vụ mô hình (Lazy-loaded < 5ms, không ép nạp FastAPI khi dùng ServerDonGian)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "ServerDonGian": ("vietnamese_ai.api.server", "ServerDonGian"),
    "MayChuOpenAI": ("vietnamese_ai.api.server", "MayChuOpenAI"),
    "OpenAICompatibleServer": ("vietnamese_ai.api.server", "OpenAICompatibleServer"),
    "FastAPIServer": ("vietnamese_ai.api.fastapi_server", "FastAPIServer"),
    "EvoNetGateway": ("vietnamese_ai.api.gateway", "EvoNetGateway"),
}

__all__ = [
    "ServerDonGian",
    "MayChuOpenAI",
    "OpenAICompatibleServer",
    "FastAPIServer",
    "EvoNetGateway",
]
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
