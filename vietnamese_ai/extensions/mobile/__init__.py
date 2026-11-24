"""Extension Mobile - Triển khai di động & Cầu nối SDK (Thử nghiệm, extra: `mobile`)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "TriKhaiDiDong": ("vietnamese_ai.mobile.deployment", "TriKhaiDiDong"),
    "MobileDeployer": ("vietnamese_ai.mobile.deployment", "MobileDeployer"),
    "MobileAppBridge": ("vietnamese_ai.mobile.bridge", "MobileAppBridge"),
    "BrowserCopilot": ("vietnamese_ai.mobile.browser_copilot", "BrowserCopilot"),
    "MobileHybridAgent": ("vietnamese_ai.mobile.mobile_agent", "MobileHybridAgent"),
    "PowerManager": ("vietnamese_ai.mobile.power_manager", "PowerManager"),
    "SDKGenerator": ("vietnamese_ai.mobile.sdk_generator", "SDKGenerator"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
