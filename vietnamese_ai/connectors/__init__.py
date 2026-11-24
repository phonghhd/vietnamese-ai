"""vietnamese_ai.connectors - Bộ Kết Nối Kênh Thực Tế Thuần Việt (Zalo OA, Telegram Bot & Máy chủ Webhook)."""

from __future__ import annotations

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "TinNhanKenh": ("vietnamese_ai.connectors.base", "TinNhanKenh"),
    "PhanHoiKenh": ("vietnamese_ai.connectors.base", "PhanHoiKenh"),
    "KetNoiKenhCoSo": ("vietnamese_ai.connectors.base", "KetNoiKenhCoSo"),
    "BaseChannelConnector": ("vietnamese_ai.connectors.base", "BaseChannelConnector"),
    "KetNoiZaloOA": ("vietnamese_ai.connectors.zalo_oa", "KetNoiZaloOA"),
    "ZaloOAConnector": ("vietnamese_ai.connectors.zalo_oa", "ZaloOAConnector"),
    "KetNoiTelegram": ("vietnamese_ai.connectors.telegram_bot", "KetNoiTelegram"),
    "TelegramBotConnector": ("vietnamese_ai.connectors.telegram_bot", "TelegramBotConnector"),
    "MayChuWebhookKenh": ("vietnamese_ai.connectors.webhook_server", "MayChuWebhookKenh"),
    "ChannelWebhookServer": ("vietnamese_ai.connectors.webhook_server", "ChannelWebhookServer"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
