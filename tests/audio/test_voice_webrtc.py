"""
Kiểm thử toàn diện Trụ cột 1: Real-time Voice AI & WebRTC / Streaming.
Kiểm tra VAD, LuongAmThanhThoiGianThuc, CauNoiWebRTC và MayChuWebRTCVoice.
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any, Dict, List

import numpy as np
import pytest

from vietnamese_ai.audio.speech import NhanDienGiongNoi
from vietnamese_ai.audio.stream_processor import LuongAmThanhThoiGianThuc
from vietnamese_ai.audio.vad import PhatHienGiongNoiVAD
from vietnamese_ai.audio.webrtc_bridge import CauNoiWebRTC, MayChuWebRTCVoice


def test_vad_nang_luong_va_zcr():
    """Kiểm tra tính toán năng lượng RMS và tỷ lệ đảo dấu ZCR."""
    vad = PhatHienGiongNoiVAD(tan_so_lay_mau=16000)

    # 1. Khung im lặng
    im_lang = np.zeros(320, dtype=np.float32)
    assert vad.tinh_nang_luong(im_lang) == 0.0
    assert vad.tinh_zcr(im_lang) == 0.0

    res_im_lang = vad.xu_ly_khung(im_lang)
    assert res_im_lang["co_giong_noi"] is False

    # 2. Khung tín hiệu giọng nói (sóng âm thanh 300Hz)
    t = np.linspace(0, 0.02, 320, endpoint=False)
    song_am = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    nang_luong = vad.tinh_nang_luong(song_am)
    zcr = vad.tinh_zcr(song_am)

    assert nang_luong > 0.1
    assert 0.02 <= zcr <= 0.65


def test_vad_phan_doan_am_thanh():
    """Kiểm tra phân tách các đoạn giọng nói từ chuỗi âm thanh tổng thể."""
    vad = PhatHienGiongNoiVAD(
        tan_so_lay_mau=16000,
        thoi_gian_im_lang_ngat_cau_ms=100.0,
        thoi_gian_noi_toi_thieu_ms=40.0,
    )

    sr = 16000
    # Tạo 0.2s im lặng + 0.3s tiếng nói + 0.3s im lặng
    im_1 = np.zeros(int(0.2 * sr), dtype=np.float32)
    t = np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    noi = (0.6 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    im_2 = np.zeros(int(0.3 * sr), dtype=np.float32)

    tong_hop = np.concatenate([im_1, noi, im_2])
    doan_list = vad.phan_doan_am_thanh(tong_hop)

    assert len(doan_list) >= 1
    start, end = doan_list[0]
    assert start >= 0
    assert end > start


def test_stream_processor_chuyen_doi_va_ha_mau():
    """Kiểm tra chuyển đổi PCM16 <-> Float32 và thuật toán hạ mẫu (Resampling)."""
    processor = LuongAmThanhThoiGianThuc(
        tan_so_lay_mau_dich=16000,
        tan_so_lay_mau_nguon=48000,
    )

    # 1. Chuyển đổi hai chiều float32 <-> pcm16
    goc = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    pcm_bytes = processor.float32_sang_pcm16(goc)
    assert len(pcm_bytes) == len(goc) * 2

    khoi_phuc = processor.pcm16_sang_float32(pcm_bytes)
    np.testing.assert_allclose(khoi_phuc, goc, atol=1e-3)

    # 2. Hạ mẫu từ 48kHz về 16kHz (giảm 3 lần số mẫu)
    tin_hieu_48k = np.sin(np.linspace(0, 10, 4800)).astype(np.float32)
    tin_hieu_16k = processor.ha_mau(tin_hieu_48k)
    assert len(tin_hieu_16k) == 1600


def test_stream_processor_phat_ngon_va_asr():
    """Kiểm tra luồng xử lý dòng âm thanh tự động kích hoạt ASR khi người dùng nói xong."""
    asr = NhanDienGiongNoi(che_do_offline=True)
    asr.dang_ky_mau_giong(
        am_thanh=0.5 * np.sin(np.linspace(0, 20, 3200)),
        van_ban="xin chào trí tuệ nhân tạo việt nam",
    )

    su_kien_nhan_duoc: List[Dict[str, Any]] = []

    def callback(sk: Dict[str, Any]):
        su_kien_nhan_duoc.append(sk)

    processor = LuongAmThanhThoiGianThuc(
        tan_so_lay_mau_dich=16000,
        tan_so_lay_mau_nguon=16000,
        bo_nhan_dien=asr,
        bo_vad=PhatHienGiongNoiVAD(
            tan_so_lay_mau=16000,
            thoi_gian_im_lang_ngat_cau_ms=80.0,
            thoi_gian_noi_toi_thieu_ms=40.0,
        ),
        callback_van_ban=callback,
    )

    sr = 16000
    # Mô phỏng: Nói 0.2s rồi im lặng 0.2s
    t = np.linspace(0, 0.2, int(0.2 * sr), endpoint=False)
    am_thanh_noi = (0.5 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
    im_lang = np.zeros(int(0.2 * sr), dtype=np.float32)

    # Nạp theo từng chunk nhỏ 512 mẫu
    tin_hieu_lien_tuc = np.concatenate([am_thanh_noi, im_lang])
    chunk_size = 512

    tat_ca_sk: List[Dict[str, Any]] = []
    for i in range(0, len(tin_hieu_lien_tuc), chunk_size):
        chunk = tin_hieu_lien_tuc[i : i + chunk_size]
        sk_list = processor.nap_tin_hieu(chunk)
        tat_ca_sk.extend(sk_list)

    loai_sk_set = {sk["loai"] for sk in tat_ca_sk}
    assert "bat_dau_noi" in loai_sk_set
    assert "hoan_thanh_cau" in loai_sk_set

    # Đảm bảo callback cũng nhận được
    assert len(su_kien_nhan_duoc) >= 1
    assert su_kien_nhan_duoc[0]["loai"] == "hoan_thanh_cau"
    assert "van_ban" in su_kien_nhan_duoc[0]


def test_webrtc_signaling():
    """Kiểm tra bộ quản lý báo hiệu WebRTC (SDP Offer / Answer & ICE Candidate)."""
    webrtc = CauNoiWebRTC()
    id_phien = "phien_test_123"

    phien = webrtc.tao_phien(id_phien)
    assert phien["trang_thai"] == "khoi_tao"

    mock_offer = "v=0\r\no=- 123 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 5004 UDP/TLS/RTP/SAVPF 111\r\n"
    tra_loi = webrtc.xu_ly_offer(id_phien, mock_offer)

    assert tra_loi["loai"] == "webrtc_answer"
    assert "sdp" in tra_loi
    assert "m=audio" in tra_loi["sdp"]

    webrtc.them_candidate(id_phien, {"candidate": "candidate:1 1 UDP 2130706431 127.0.0.1 5004 typ host"})
    assert len(webrtc._phien_webrtc[id_phien]["candidates"]) == 1


def _tim_port_trong() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_may_chu_webrtc_voice_full_dialogue():
    """Kiểm tra máy chủ WebSocket Voice Bridge: Bắt tay RFC 6455, hỏi đáp văn bản và nhận audio TTS."""
    port = _tim_port_trong()

    def ham_tra_loi_tuy_bien(cau_hoi: str) -> str:
        return f"Phản hồi tùy biến cho: {cau_hoi}"

    server = MayChuWebRTCVoice(
        host="127.0.0.1",
        port=port,
        ham_tra_loi=ham_tra_loi_tuy_bien,
        tan_so_lay_mau=16000,
    )

    await server.khoi_dong_async()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        # 1. Bắt tay WebSocket Handshake
        sec_key = "dGhlIHNhbXBsZSBub25jZQ=="
        req = (
            f"GET /voice HTTP/1.1\r\n"
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
        assert b"Sec-WebSocket-Accept" in resp_hs

        # Đọc thông điệp chào mừng
        opcode, payload = await server._doc_frame(reader)
        msg_chao = json.loads(payload.decode("utf-8"))
        assert msg_chao["loai"] == "ket_noi_thanh_cong"

        # 2. Gửi câu hỏi text yêu cầu trả lời kèm giọng nói TTS
        cau_hoi_json = json.dumps(
            {"loai": "cau_hoi_text", "noi_dung": "Xin chào Việt Nam"},
            ensure_ascii=False,
        ).encode("utf-8")

        # Đóng gói frame client gửi lên (có mask)
        mask = b"\x12\x34\x56\x78"
        masked_payload = bytearray(len(cau_hoi_json))
        for i in range(len(cau_hoi_json)):
            masked_payload[i] = cau_hoi_json[i] ^ mask[i % 4]

        frame_client = bytearray([0x81, 0x80 | len(cau_hoi_json)])
        frame_client.extend(mask)
        frame_client.extend(masked_payload)

        writer.write(bytes(frame_client))
        await writer.drain()

        # 3. Đọc phản hồi văn bản từ server
        op_res, pay_res = await server._doc_frame(reader)
        res_json = json.loads(pay_res.decode("utf-8"))
        assert res_json["loai"] == "phan_hoi_van_ban"
        assert "Phản hồi tùy biến cho: Xin chào Việt Nam" in res_json["cau_tra_loi"]

        # 4. Đọc metadata âm thanh bắt đầu phát
        op_meta, pay_meta = await server._doc_frame(reader)
        meta_json = json.loads(pay_meta.decode("utf-8"))
        assert meta_json["loai"] == "bat_dau_phat_am_thanh"
        assert meta_json["tan_so_lay_mau"] == 16000

        # 5. Đọc ít nhất một chunk âm thanh nhị phân (binary audio frame opcode 0x02)
        op_audio, pay_audio = await server._doc_frame(reader)
        assert op_audio == 0x02
        assert len(pay_audio) > 0

        writer.close()
        await writer.wait_closed()

    finally:
        await server.dung_lai_async()


def test_cli_vai_voice_va_swarm_fp16():
    """Kiểm tra cấu hình lệnh CLI cho vai voice và cờ nén vai swarm --fp16."""
    from vietnamese_ai.cli.main import _tao_parser

    parser = _tao_parser()

    # 1. Kiểm tra parser lệnh vai voice
    args_voice = parser.parse_args([
        "voice",
        "--host", "0.0.0.0",
        "--port", "8888",
        "--sample-rate", "16000",
        "--llm",
        "--model", "mo_hinh_test.bin",
        "--rag", "tri_thuc_test.db",
    ])
    assert args_voice.command == "voice"
    assert args_voice.host == "0.0.0.0"
    assert args_voice.port == 8888
    assert args_voice.sample_rate == 16000
    assert args_voice.llm is True
    assert args_voice.model == "mo_hinh_test.bin"
    assert args_voice.rag == "tri_thuc_test.db"
    assert args_voice.open_browser is False

    # 2. Kiểm tra cờ --fp16 của lệnh vai swarm
    args_swarm = parser.parse_args(["swarm", "--fp16", "--nodes", "4"])
    assert args_swarm.command == "swarm"
    assert args_swarm.fp16 is True
    assert args_swarm.nodes == 4


@pytest.mark.asyncio
async def test_may_chu_webrtc_voice_async_va_mo_hinh_rag_hook():
    """Kiểm tra MayChuWebRTCVoice với hàm bất đồng bộ async, đối tượng mô hình LLM và RAG."""
    # 1. Test async ham_tra_loi
    async def ham_async(cau_hoi: str) -> str:
        await asyncio.sleep(0.01)
        return f"Async: {cau_hoi}"

    port1 = _tim_port_trong()
    server_async = MayChuWebRTCVoice(host="127.0.0.1", port=port1, ham_tra_loi=ham_async)
    assert server_async.ham_tra_loi == ham_async

    # 2. Test mo_hinh adapter (sinh_van_ban)
    class MockLLM:
        def sinh_van_ban(self, prompt: str, max_tokens: int = 50) -> str:
            return f"LLM trả lời: {prompt}"

    server_llm = MayChuWebRTCVoice(host="127.0.0.1", port=port1, mo_hinh=MockLLM())
    tra_loi_llm = server_llm.ham_tra_loi("Thời tiết hôm nay")
    assert "LLM trả lời: Thời tiết hôm nay" in tra_loi_llm

    # 3. Test rag adapter (truy_van)
    class MockRAG:
        def truy_van(self, query: str) -> dict:
            return {"cau_tra_loi": f"RAG tra cứu: {query}"}

    server_rag = MayChuWebRTCVoice(host="127.0.0.1", port=port1, rag=MockRAG())
    tra_loi_rag = server_rag.ham_tra_loi("Lịch sử Việt Nam")
    assert "RAG tra cứu: Lịch sử Việt Nam" in tra_loi_rag


@pytest.mark.asyncio
async def test_may_chu_webrtc_voice_stress_dong_thoi():
    """Kiểm tra áp lực (Stress Test) đa kết nối đồng thời trên MayChuWebRTCVoice."""
    port = _tim_port_trong()

    def ham_xu_ly_dong_thoi(cau_hoi: str) -> str:
        return f"Phản hồi đồng thời: {cau_hoi}"

    server = MayChuWebRTCVoice(
        host="127.0.0.1",
        port=port,
        ham_tra_loi=ham_xu_ly_dong_thoi,
        tan_so_lay_mau=16000,
    )
    await server.khoi_dong_async()

    so_client = 6

    async def _mo_phong_client(client_idx: int) -> None:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            # 1. Bắt tay WebSocket
            sec_key = "dGhlIHNhbXBsZSBub25jZQ=="
            req = (
                f"GET /voice HTTP/1.1\r\n"
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

            # 2. Đọc tin nhắn chào mừng
            op_chao, pay_chao = await server._doc_frame(reader)
            msg_chao = json.loads(pay_chao.decode("utf-8"))
            assert msg_chao["loai"] == "ket_noi_thanh_cong"

            # 3. Gửi câu hỏi đồng thời
            noi_dung = f"Khách hàng số {client_idx} hỏi AI"
            req_body = json.dumps({"loai": "cau_hoi_text", "noi_dung": noi_dung}, ensure_ascii=False).encode("utf-8")
            mask = b"\xaa\xbb\xcc\xdd"
            masked = bytearray(len(req_body))
            for i in range(len(req_body)):
                masked[i] = req_body[i] ^ mask[i % 4]

            frame = bytearray([0x81, 0x80 | len(req_body)])
            frame.extend(mask)
            frame.extend(masked)
            writer.write(bytes(frame))
            await writer.drain()

            # 4. Nhận phản hồi văn bản
            op_t, pay_t = await server._doc_frame(reader)
            data_t = json.loads(pay_t.decode("utf-8"))
            assert data_t["loai"] == "phan_hoi_van_ban"
            assert noi_dung in data_t["cau_tra_loi"]

            # 5. Nhận thông báo phát âm thanh và ít nhất một audio chunk
            op_meta, pay_meta = await server._doc_frame(reader)
            assert json.loads(pay_meta.decode("utf-8"))["loai"] == "bat_dau_phat_am_thanh"

            op_pcm, pay_pcm = await server._doc_frame(reader)
            assert op_pcm == 0x02
            assert len(pay_pcm) > 0
        finally:
            writer.close()
            await writer.wait_closed()

    try:
        tasks = [_mo_phong_client(i) for i in range(so_client)]
        await asyncio.gather(*tasks)
    finally:
        await server.dung_lai_async()
