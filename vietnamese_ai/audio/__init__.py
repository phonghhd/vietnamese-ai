"""Module Xử lý Tiếng nói Tiếng Việt (ASR, TTS 6 thanh điệu, VAD và WebRTC Streaming Bridge)."""

from vietnamese_ai.audio.speech import NhanDienGiongNoi, TongHopGiongNoi
from vietnamese_ai.audio.stream_processor import LuongAmThanhThoiGianThuc
from vietnamese_ai.audio.vad import PhatHienGiongNoiVAD, TrangThaiVAD
from vietnamese_ai.audio.webrtc_bridge import CauNoiWebRTC, MayChuWebRTCVoice

__all__ = [
    "NhanDienGiongNoi",
    "TongHopGiongNoi",
    "PhatHienGiongNoiVAD",
    "TrangThaiVAD",
    "LuongAmThanhThoiGianThuc",
    "CauNoiWebRTC",
    "MayChuWebRTCVoice",
]
