"""
Kiểm thử tích hợp Giải pháp Voicebot CSKH (TongDaiCSKH / VoicebotCSKH)
với MayChuWebRTCVoice và WebRTC Voice Streaming.
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any, Dict

import numpy as np
import pytest

from vietnamese_ai.audio.speech import NhanDienGiongNoi, TongHopGiongNoi
from vietnamese_ai.audio.stream_processor import LuongAmThanhThoiGianThuc
from vietnamese_ai.audio.webrtc_bridge import MayChuWebRTCVoice
from vietnamese_ai.solutions.cskh import (
    CustomerSupportBot,
    GiaiPhapVoicebotCSKH,
    TongDaiCSKH,
    VoicebotCSKH,
)


def _tim_port_trong() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockASR(NhanDienGiongNoi):
    def nhan_dien(self, am_thanh, **kwargs) -> Dict[str, Any]:
        return {"van_ban": "Kiểm tra đơn hàng DH1001", "do_tin_cay": 0.98}


class MockTTS(TongHopGiongNoi):
    def tong_hop(self, van_ban: str, **kwargs) -> Dict[str, Any]:
        sr = 16000
        t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False)
        wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return {
            "tin_hieu": wave,
            "tan_so_lay_mau": sr,
            "thoi_luong_giay": 0.1,
            "so_mau": len(wave),
        }


def test_cskh_aliases_va_chuan_hoa_pcm_bytes():
    """Kiểm tra các alias của TongDaiCSKH và khả năng nhận diện từ dữ liệu PCM thô."""
    assert CustomerSupportBot is TongDaiCSKH
    assert VoicebotCSKH is TongDaiCSKH
    assert GiaiPhapVoicebotCSKH is TongDaiCSKH

    bot = TongDaiCSKH(asr=MockASR(), tts=MockTTS())

    # Tạo mảng PCM 16-bit giả lập
    sr = 16000
    t = np.linspace(0, 0.2, int(0.2 * sr), endpoint=False)
    wave_f32 = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    pcm_bytes = LuongAmThanhThoiGianThuc.float32_sang_pcm16(wave_f32)

    # Gửi trực tiếp bytes vào xu_ly_yeu_cau
    res = bot.xu_ly_yeu_cau(pcm_bytes)
    assert res["la_giong_noi"] is True
    assert res["y_dinh"] == "TRA_CUU_DON_HANG"
    assert "Đang giao hàng" in res["tra_loi"]
    assert res["thong_tin_nghiep_vu"]["ma_don"] == "DH1001"


@pytest.mark.asyncio
async def test_webrtc_voice_server_voi_cskh_adapter():
    """Kiểm tra MayChuWebRTCVoice khi kết nối với TongDaiCSKH."""
    port = _tim_port_trong()
    bot = TongDaiCSKH(asr=MockASR(), tts=MockTTS())

    server = MayChuWebRTCVoice(
        host="127.0.0.1",
        port=port,
        cskh=bot,
        bo_nhan_dien=MockASR(),
        bo_tong_hop=MockTTS(),
    )
    await server.khoi_dong_async()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        # 1. Bắt tay WebSocket
        sec_key = "dGhlIHNhbXBsZSBub25jZQ=="
        req = (
            f"GET /cskh-voice HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {sec_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("utf-8")
        writer.write(req)
        await writer.drain()

        resp_hs = await reader.readuntil(b"\r\n\r\n")
        assert b"101 Switching Protocols" in resp_hs

        # Đọc tin chào
        _, pay_chao = await server._doc_frame(reader)
        msg_chao = json.loads(pay_chao.decode("utf-8"))
        assert msg_chao["loai"] == "ket_noi_thanh_cong"

        # 2. Gửi câu hỏi tra cứu đơn hàng qua text frame
        cau_hoi_json = json.dumps(
            {"loai": "cau_hoi_text", "noi_dung": "Kiểm tra giúp tôi đơn hàng DH1002"},
            ensure_ascii=False,
        ).encode("utf-8")

        mask = b"\xaa\xbb\xcc\xdd"
        masked = bytearray(len(cau_hoi_json))
        for i in range(len(cau_hoi_json)):
            masked[i] = cau_hoi_json[i] ^ mask[i % 4]

        frame = bytearray([0x81, 0x80 | len(cau_hoi_json)])
        frame.extend(mask)
        frame.extend(masked)
        writer.write(bytes(frame))
        await writer.drain()

        # 3. Nhận phản hồi văn bản có chứa intent và thông tin nghiệp vụ
        _, pay_res = await server._doc_frame(reader)
        res_json = json.loads(pay_res.decode("utf-8"))
        assert res_json["loai"] == "phan_hoi_van_ban"
        assert res_json["y_dinh"] == "TRA_CUU_DON_HANG"
        assert "DH1002" in res_json["cau_tra_loi"]
        assert "Đã giao thành công" in res_json["cau_tra_loi"]

        # 4. Nhận metadata bắt đầu phát âm thanh
        _, pay_meta = await server._doc_frame(reader)
        meta_json = json.loads(pay_meta.decode("utf-8"))
        assert meta_json["loai"] == "bat_dau_phat_am_thanh"

        # 5. Nhận ít nhất một chunk âm thanh nhị phân
        op_audio, pay_audio = await server._doc_frame(reader)
        assert op_audio == 0x02
        assert len(pay_audio) > 0

        writer.close()
        await writer.wait_closed()
    finally:
        await server.dung_lai_async()


def test_khoi_dong_may_chu_thoai_tu_cskh_instance():
    """Kiểm tra phương thức khoi_dong_may_chu_thoai trên instance TongDaiCSKH."""
    port = _tim_port_trong()
    bot = TongDaiCSKH(asr=MockASR(), tts=MockTTS())

    srv = bot.khoi_dong_may_chu_thoai(host="127.0.0.1", port=port, nen_chay_ngam=True)
    try:
        assert isinstance(srv, MayChuWebRTCVoice)
        assert srv._dang_chay is True
        assert srv.port == port
    finally:
        srv.dung_lai()


def test_cli_voice_cskh_flags():
    """Kiểm tra các cờ CLI cho vai voice --cskh và vai solution cskh --voice."""
    from vietnamese_ai.cli.main import _tao_parser

    parser = _tao_parser()

    # vai voice --cskh
    args_voice = parser.parse_args(["voice", "--cskh", "--port", "8900"])
    assert args_voice.command == "voice"
    assert args_voice.cskh is True
    assert args_voice.port == 8900

    # vai solution cskh --voice
    args_sol = parser.parse_args(["solution", "cskh", "--voice"])
    assert args_sol.command == "solution"
    assert args_sol.name == "cskh"
    assert args_sol.voice is True
