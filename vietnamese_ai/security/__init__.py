"""Security Module cho Vietnamese AI Framework - Tường lửa AI, Bảo vệ Ảnh CDR & Phòng thủ Dữ liệu (Lazy PEP 562)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

__all__ = [
    "TuongLuaAI",
    "TuongLuaBaoVeFramework",
    "KhienBaoVeAI",
    "AIFirewallShield",
    "MoiTruongCachLy",
    "DataSanitizer",
    "KhienBaoVeHinhAnh",
    "ImageSecurityShield",
    "PhongThuDuLieu",
    "DatasetDefenseShield",
    "BoGiamSatTichTuDuLieu",
    "StreamingPoisoningMonitor",
    "CongChatLuongDauVao",
    "InputQualityGate",
    "CongAnToanDauRa",
    "OutputSafetyQualityGate",
    "RedTeamSimulator",
    "TextWatermarker",
    "BlueTeamAgent",
]

__getattr__ = tao_lazy_getattr(
    __name__,
    {
        "TuongLuaAI": ("vietnamese_ai.security.llm_firewall", "TuongLuaAI"),
        "TuongLuaBaoVeFramework": ("vietnamese_ai.security.llm_firewall", "TuongLuaBaoVeFramework"),
        "KhienBaoVeAI": ("vietnamese_ai.security.llm_firewall", "KhienBaoVeAI"),
        "AIFirewallShield": ("vietnamese_ai.security.llm_firewall", "AIFirewallShield"),
        "MoiTruongCachLy": ("vietnamese_ai.security.agent_sandbox", "MoiTruongCachLy"),
        "DataSanitizer": ("vietnamese_ai.security.data_sanitizer", "DataSanitizer"),
        "KhienBaoVeHinhAnh": ("vietnamese_ai.security.data_defense", "KhienBaoVeHinhAnh"),
        "ImageSecurityShield": ("vietnamese_ai.security.data_defense", "ImageSecurityShield"),
        "PhongThuDuLieu": ("vietnamese_ai.security.data_defense", "PhongThuDuLieu"),
        "DatasetDefenseShield": ("vietnamese_ai.security.data_defense", "DatasetDefenseShield"),
        "BoGiamSatTichTuDuLieu": ("vietnamese_ai.security.data_defense", "BoGiamSatTichTuDuLieu"),
        "StreamingPoisoningMonitor": (
            "vietnamese_ai.security.data_defense",
            "StreamingPoisoningMonitor",
        ),
        "CongChatLuongDauVao": ("vietnamese_ai.security.data_defense", "CongChatLuongDauVao"),
        "InputQualityGate": ("vietnamese_ai.security.data_defense", "InputQualityGate"),
        "CongAnToanDauRa": ("vietnamese_ai.security.data_defense", "CongAnToanDauRa"),
        "OutputSafetyQualityGate": (
            "vietnamese_ai.security.data_defense",
            "OutputSafetyQualityGate",
        ),
        "RedTeamSimulator": ("vietnamese_ai.security.red_team", "RedTeamSimulator"),
        "TextWatermarker": ("vietnamese_ai.security.watermark", "TextWatermarker"),
        "BlueTeamAgent": ("vietnamese_ai.security.blue_team", "BlueTeamAgent"),
    },
    globals(),
)
