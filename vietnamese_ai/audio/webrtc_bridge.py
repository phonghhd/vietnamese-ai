"""
Cầu nối WebRTC & WebSocket cho Trợ lý Thoại Thời Gian Thực (Voice AI Streaming Bridge).
Truyền phát âm thanh hai chiều giữa trình duyệt web và Vietnamese AI Framework.
Hoạt động thuần asyncio và thư viện chuẩn, tuân thủ Zero-Dependency Oath.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import logging
import struct
import threading
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

from vietnamese_ai.audio.speech import NhanDienGiongNoi, TongHopGiongNoi
from vietnamese_ai.audio.stream_processor import LuongAmThanhThoiGianThuc

logger = logging.getLogger("VietnameseAI.VoiceBridge")

WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class CauNoiWebRTC:
    """
    Trình quản lý giao thức báo hiệu (Signaling Protocol) cho WebRTC Voice Call.
    Hỗ trợ trao đổi SDP Offer / Answer và ICE Candidates giữa các Client và Server.
    """

    def __init__(self) -> None:
        self._phien_webrtc: Dict[str, Dict[str, Any]] = {}

    def tao_phien(self, id_phien: str) -> Dict[str, Any]:
        """Tạo một phiên kết nối WebRTC mới."""
        phien = {
            "id_phien": id_phien,
            "trang_thai": "khoi_tao",
            "offer": None,
            "answer": None,
            "candidates": [],
        }
        self._phien_webrtc[id_phien] = phien
        return phien

    def xu_ly_offer(self, id_phien: str, sdp_offer: str) -> Dict[str, Any]:
        """Xử lý SDP Offer từ trình duyệt và tạo SDP Answer tương thích."""
        if id_phien not in self._phien_webrtc:
            self.tao_phien(id_phien)

        phien = self._phien_webrtc[id_phien]
        phien["offer"] = sdp_offer
        phien["trang_thai"] = "da_nhan_offer"

        # Sinh Answer tương thích âm thanh Opus / PCM 16kHz
        sdp_answer = (
            "v=0\r\n"
            "o=VietnameseAI 20261109 2 IN IP4 127.0.0.1\r\n"
            "s=Vietnamese AI WebRTC Voice Session\r\n"
            "t=0 0\r\n"
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
            "c=IN IP4 0.0.0.0\r\n"
            "a=rtpmap:111 opus/48000/2\r\n"
            "a=sendrecv\r\n"
        )
        phien["answer"] = sdp_answer
        phien["trang_thai"] = "da_tao_answer"
        return {"loai": "webrtc_answer", "id_phien": id_phien, "sdp": sdp_answer}

    def them_candidate(self, id_phien: str, candidate: Dict[str, Any]) -> None:
        """Lưu trữ ICE Candidate nhận được từ client."""
        if id_phien in self._phien_webrtc:
            self._phien_webrtc[id_phien]["candidates"].append(candidate)


class MayChuWebRTCVoice:
    """
    Máy chủ WebSocket & Cầu nối âm thanh thời gian thực (Zero-Dependency).

    Cung cấp:
    - Bắt tay chuẩn WebSocket RFC 6455 qua `asyncio.start_server`.
    - Tiếp nhận dòng âm thanh nhị phân (Binary PCM) từ microphone trình duyệt.
    - Nhận diện giọng nói tiếng Việt bằng `LuongAmThanhThoiGianThuc` (VAD + ASR).
    - Gọi hàm phản hồi hội thoại thông minh (`ham_tra_loi`).
    - Tổng hợp câu trả lời thành giọng nói tiếng Việt 6 thanh điệu bằng `TongHopGiongNoi` (TTS).
    - Truyền phát các đoạn âm thanh trả về trực tiếp cho trình duyệt phát ra loa.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        ham_tra_loi: Optional[Callable[[str], Any]] = None,
        mo_hinh: Optional[Any] = None,
        rag: Optional[Any] = None,
        cskh: Optional[Any] = None,
        tan_so_lay_mau: int = 16000,
        bo_nhan_dien: Optional[NhanDienGiongNoi] = None,
        bo_tong_hop: Optional[TongHopGiongNoi] = None,
    ) -> None:
        """
        Khởi tạo máy chủ Voice AI.

        Args:
            host: Địa chỉ IP lắng nghe (mặc định 127.0.0.1).
            port: Cổng mạng WebSocket (mặc định 8765).
            ham_tra_loi: Hàm xử lý câu hỏi người dùng thành câu trả lời văn bản (hỗ trợ sync & async).
            mo_hinh: Thể hiện mô hình ngôn ngữ (VietnameseLLM / GPTModel / BaseModel) tự động sinh trả lời.
            rag: Thể hiện đường ống truy vấn RAG (RAGPipeline / Admin RAG) tự động trả lời theo tri thức.
            cskh: Thể hiện giải pháp Tổng đài CSKH (TongDaiCSKH) tự động xử lý nghiệp vụ đơn hàng & chăm sóc.
            tan_so_lay_mau: Tần số âm thanh trao đổi (mặc định 16000 Hz).
            bo_nhan_dien: Thể hiện ASR `NhanDienGiongNoi`.
            bo_tong_hop: Thể hiện TTS `TongHopGiongNoi`.
        """
        self.host = host
        self.port = port
        self.tan_so_lay_mau = tan_so_lay_mau
        self.asr = bo_nhan_dien or NhanDienGiongNoi(tan_so_lay_mau=tan_so_lay_mau)
        self.tts = bo_tong_hop or TongHopGiongNoi(tan_so_lay_mau=tan_so_lay_mau)
        self.webrtc_signaling = CauNoiWebRTC()

        # Xác định bộ xử lý câu trả lời theo thứ tự ưu tiên
        if ham_tra_loi is not None:
            self.ham_tra_loi = ham_tra_loi
        elif cskh is not None:
            self.ham_tra_loi = self._tao_ham_phan_hoi_cskh(cskh)
        elif mo_hinh is not None:
            self.ham_tra_loi = self._tao_ham_phan_hoi_mo_hinh(mo_hinh)
        elif rag is not None:
            self.ham_tra_loi = self._tao_ham_phan_hoi_rag(rag)
        else:
            self.ham_tra_loi = self._phan_hoi_mac_dinh

        self._server: Optional[asyncio.Server] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._dang_chay = False
        self._cac_ket_noi: Set[Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = set()

    @staticmethod
    def _tao_ham_phan_hoi_mo_hinh(mo_hinh: Any) -> Callable[[str], str]:
        """Tự động đóng gói mô hình AI (LLM / Tác tử) thành hàm xử lý đối thoại giọng nói."""
        def _phan_hoi(cau_hoi: str) -> str:
            if hasattr(mo_hinh, "sinh_van_ban"):
                try:
                    return str(mo_hinh.sinh_van_ban(cau_hoi, do_dai_toi_da=64))
                except TypeError:
                    return str(mo_hinh.sinh_van_ban(cau_hoi))
            elif hasattr(mo_hinh, "generate"):
                try:
                    return str(mo_hinh.generate(cau_hoi, max_tokens=64))
                except TypeError:
                    return str(mo_hinh.generate(cau_hoi))
            elif hasattr(mo_hinh, "chat"):
                return str(mo_hinh.chat(cau_hoi))
            elif callable(mo_hinh):
                return str(mo_hinh(cau_hoi))
            return f"Phản hồi từ mô hình cho: {cau_hoi}"
        return _phan_hoi

    @staticmethod
    def _tao_ham_phan_hoi_rag(rag: Any) -> Callable[[str], str]:
        """Tự động đóng gói RAG Pipeline thành hàm tra cứu tri thức giọng nói."""
        def _phan_hoi(cau_hoi: str) -> str:
            if hasattr(rag, "truy_van"):
                kq = rag.truy_van(cau_hoi)
                if isinstance(kq, dict):
                    return str(kq.get("tra_loi") or kq.get("cau_tra_loi") or kq)
                return str(kq)
            elif hasattr(rag, "query"):
                kq = rag.query(cau_hoi)
                if isinstance(kq, dict):
                    return str(kq.get("answer") or kq)
                return str(kq)
            return f"Tra cứu RAG cho: {cau_hoi}"
        return _phan_hoi

    @staticmethod
    def _tao_ham_phan_hoi_cskh(cskh: Any) -> Callable[[str], Dict[str, Any]]:
        """Tự động đóng gói Tổng đài CSKH (TongDaiCSKH) thành hàm phản hồi thoại thời gian thực."""
        def _phan_hoi(cau_hoi: str) -> Dict[str, Any]:
            if hasattr(cskh, "xu_ly_yeu_cau"):
                return cskh.xu_ly_yeu_cau(cau_hoi)
            elif callable(cskh):
                res = cskh(cau_hoi)
                if isinstance(res, dict):
                    return res
                return {"tra_loi": str(res), "y_dinh": "HOI_DAP"}
            return {"tra_loi": f"Dạ tổng đài đã ghi nhận yêu cầu: {cau_hoi}", "y_dinh": "MAC_DINH"}
        return _phan_hoi

    @staticmethod
    def _phan_hoi_mac_dinh(cau_hoi: str) -> str:
        """Phản hồi mẫu thông minh khi chưa gắn LLM/RAG bên ngoài."""
        q = cau_hoi.lower().strip()
        if not q or "tín hiệu giọng nói" in q:
            return "Tôi đã lắng nghe bạn. Xin mời bạn đặt câu hỏi về hệ thống."
        if "chào" in q:
            return "Xin chào bạn! Tôi là trợ lý giọng nói thông minh của Vietnamese AI."
        if "tên" in q:
            return "Tôi là hệ thống Voice AI thuần Việt, hỗ trợ đàm thoại thời gian thực."
        if "thời tiết" in q:
            return "Hôm nay trời rất đẹp, thích hợp cho việc nghiên cứu và lập trình AI."
        return f"Tôi đã nhận được câu hỏi của bạn: {cau_hoi}. Hệ sinh thái sẵn sàng phục vụ."

    @staticmethod
    def _tao_websocket_accept(sec_key: str) -> str:
        """Tạo khóa xác thực Sec-WebSocket-Accept theo chuẩn RFC 6455."""
        ket_hop = (sec_key.strip() + WEBSOCKET_GUID).encode("utf-8")
        sha1_hash = hashlib.sha1(ket_hop).digest()
        return base64.b64encode(sha1_hash).decode("utf-8")

    @staticmethod
    def dong_goi_frame(noi_dung: Union[str, bytes], ma_lenh: Optional[int] = None) -> bytes:
        """Đóng gói dữ liệu thành WebSocket Frame chuẩn máy chủ gửi client (không mask)."""
        if isinstance(noi_dung, str):
            payload = noi_dung.encode("utf-8")
            opcode = ma_lenh if ma_lenh is not None else 0x01  # Text frame
        else:
            payload = noi_dung
            opcode = ma_lenh if ma_lenh is not None else 0x02  # Binary frame

        do_dai = len(payload)
        header = bytearray([0x80 | (opcode & 0x0F)])

        if do_dai <= 125:
            header.append(do_dai)
        elif do_dai <= 65535:
            header.append(126)
            header.extend(struct.pack("!H", do_dai))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", do_dai))

        return bytes(header) + payload

    async def _gui_frame(
        self,
        writer: asyncio.StreamWriter,
        noi_dung: Union[str, bytes],
        ma_lenh: Optional[int] = None,
    ) -> None:
        """Gửi an toàn một WebSocket frame tới client."""
        frame_bytes = self.dong_goi_frame(noi_dung, ma_lenh)
        try:
            writer.write(frame_bytes)
            await writer.drain()
        except Exception as e:
            logger.debug("Lỗi gửi frame tới client: %s", e)

    async def _doc_frame(
        self,
        reader: asyncio.StreamReader,
    ) -> Optional[Tuple[int, bytes]]:
        """Đọc và giải mã (unmask) một WebSocket frame từ client."""
        try:
            head = await reader.readexactly(2)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None

        byte1, byte2 = head[0], head[1]
        opcode = byte1 & 0x0F
        has_mask = bool(byte2 & 0x80)
        payload_len = byte2 & 0x7F

        if payload_len == 126:
            len_bytes = await reader.readexactly(2)
            payload_len = struct.unpack("!H", len_bytes)[0]
        elif payload_len == 127:
            len_bytes = await reader.readexactly(8)
            payload_len = struct.unpack("!Q", len_bytes)[0]

        masking_key = b""
        if has_mask:
            masking_key = await reader.readexactly(4)

        payload = await reader.readexactly(payload_len)
        if has_mask:
            unmasked = bytearray(payload_len)
            for i in range(payload_len):
                unmasked[i] = payload[i] ^ masking_key[i % 4]
            payload = bytes(unmasked)

        return opcode, payload

    async def _xu_ly_ket_noi(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Xử lý vòng đời một kết nối WebSocket từ trình duyệt."""
        ket_noi = (reader, writer)
        self._cac_ket_noi.add(ket_noi)

        # 1. Bắt tay WebSocket (Handshake)
        du_lieu_handshake = b""
        while b"\r\n\r\n" not in du_lieu_handshake:
            chunk = await reader.read(1024)
            if not chunk:
                writer.close()
                self._cac_ket_noi.discard(ket_noi)
                return
            du_lieu_handshake += chunk

        req_text = du_lieu_handshake.decode("utf-8", errors="ignore")
        headers: Dict[str, str] = {}
        for line in req_text.split("\r\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        sec_key = headers.get("sec-websocket-key")
        if not sec_key:
            writer.close()
            self._cac_ket_noi.discard(ket_noi)
            return

        accept_val = self._tao_websocket_accept(sec_key)
        phan_hoi_hs = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_val}\r\n"
            "\r\n"
        ).encode("utf-8")
        writer.write(phan_hoi_hs)
        await writer.drain()

        # 2. Khởi tạo bộ xử lý dòng âm thanh cho kết nối này
        luong_audio = LuongAmThanhThoiGianThuc(
            tan_so_lay_mau_dich=self.tan_so_lay_mau,
            tan_so_lay_mau_nguon=self.tan_so_lay_mau,
            bo_nhan_dien=self.asr,
        )

        # Gửi thông điệp chào mừng
        await self._gui_frame(
            writer,
            json.dumps(
                {
                    "loai": "ket_noi_thanh_cong",
                    "thong_diep": "Đã kết nối với Vietnamese AI Voice Engine v31.0",
                    "tan_so_lay_mau": self.tan_so_lay_mau,
                },
                ensure_ascii=False,
            ),
        )

        # 3. Lắng nghe và xử lý các frames
        try:
            while self._dang_chay:
                frame = await self._doc_frame(reader)
                if frame is None:
                    break

                opcode, payload = frame

                # Đóng kết nối
                if opcode == 0x08:
                    await self._gui_frame(writer, b"", ma_lenh=0x08)
                    break

                # Ping -> Pong
                elif opcode == 0x09:
                    await self._gui_frame(writer, payload, ma_lenh=0x0A)

                # Text Frame (JSON signaling / commands)
                elif opcode == 0x01:
                    try:
                        thong_tin = json.loads(payload.decode("utf-8"))
                        loai_msg = thong_tin.get("loai")

                        if loai_msg == "cau_hoi_text":
                            noi_dung = thong_tin.get("noi_dung", "")
                            await self._phan_hoi_giong_noi(writer, noi_dung)

                        elif loai_msg == "webrtc_offer":
                            id_p = thong_tin.get("id_phien", "phien_chuan")
                            sdp = thong_tin.get("sdp", "")
                            ans = self.webrtc_signaling.xu_ly_offer(id_p, sdp)
                            await self._gui_frame(writer, json.dumps(ans, ensure_ascii=False))

                        elif loai_msg == "webrtc_candidate":
                            id_p = thong_tin.get("id_phien", "phien_chuan")
                            cand = thong_tin.get("candidate", {})
                            self.webrtc_signaling.them_candidate(id_p, cand)

                        elif loai_msg == "cai_dat_tan_so":
                            tan_so_moi = int(thong_tin.get("tan_so", self.tan_so_lay_mau))
                            luong_audio.tan_so_nguon = tan_so_moi

                    except Exception as err:
                        logger.debug("Lỗi xử lý text message: %s", err)

                # Binary Frame (PCM 16-bit audio chunk from browser microphone)
                elif opcode == 0x02:
                    su_kien_list = luong_audio.nap_du_lieu_raw(payload)
                    for sk in su_kien_list:
                        loai_sk = sk.get("loai")
                        if loai_sk == "bat_dau_noi":
                            await self._gui_frame(
                                writer,
                                json.dumps(
                                    {
                                        "loai": "vad_bat_dau",
                                        "thong_bao": "Đang lắng nghe giọng nói...",
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                        elif loai_sk == "hoan_thanh_cau":
                            van_ban = sk.get("van_ban", "")
                            do_tin_cay = sk.get("do_tin_cay", 0.0)

                            # 1. Báo kết quả ASR
                            await self._gui_frame(
                                writer,
                                json.dumps(
                                    {
                                        "loai": "nhan_dien_giong_noi",
                                        "van_ban": van_ban,
                                        "do_tin_cay": do_tin_cay,
                                    },
                                    ensure_ascii=False,
                                ),
                            )

                            # 2. Xử lý câu trả lời & Tổng hợp tiếng nói TTS
                            await self._phan_hoi_giong_noi(writer, van_ban)

        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self._cac_ket_noi.discard(ket_noi)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _phan_hoi_giong_noi(self, writer: asyncio.StreamWriter, cau_hoi: str) -> None:
        """Tạo câu trả lời và tổng hợp giọng nói 6 thanh điệu truyền ngược lại cho client."""
        # 1. Sinh câu trả lời qua hàm xử lý (hỗ trợ cả async và sync callable)
        try:
            if inspect.iscoroutinefunction(self.ham_tra_loi):
                cau_tra_loi = await self.ham_tra_loi(cau_hoi)
            else:
                cau_tra_loi = self.ham_tra_loi(cau_hoi)
                if inspect.isawaitable(cau_tra_loi):
                    cau_tra_loi = await cau_tra_loi
        except Exception as e:
            logger.error("Lỗi khi thực thi hàm phản hồi hội thoại: %s", e)
            cau_tra_loi = f"Tôi đã nhận được câu hỏi '{cau_hoi}' và sẵn sàng hỗ trợ bạn."

        # 2. Xử lý định dạng văn bản phản hồi & siêu dữ liệu (nếu từ CSKH/RAG)
        metadata_phan_hoi: Dict[str, Any] = {}
        if isinstance(cau_tra_loi, dict):
            metadata_phan_hoi = cau_tra_loi
            van_ban_tra_loi = str(
                cau_tra_loi.get("tra_loi")
                or cau_tra_loi.get("cau_tra_loi")
                or cau_tra_loi.get("van_ban")
                or str(cau_tra_loi)
            )
        else:
            van_ban_tra_loi = str(cau_tra_loi)

        thong_tin_gui: Dict[str, Any] = {
            "loai": "phan_hoi_van_ban",
            "cau_hoi": cau_hoi,
            "cau_tra_loi": van_ban_tra_loi,
        }
        if metadata_phan_hoi:
            for k in ("y_dinh", "thong_tin_nghiep_vu", "an_toan"):
                if k in metadata_phan_hoi:
                    thong_tin_gui[k] = metadata_phan_hoi[k]

        await self._gui_frame(
            writer,
            json.dumps(thong_tin_gui, ensure_ascii=False),
        )

        # 3. Tổng hợp giọng nói TTS thuần tiếng Việt
        ket_qua_tts = self.tts.tong_hop(van_ban_tra_loi)
        tin_hieu_f32 = ket_qua_tts["tin_hieu"]
        pcm16_bytes = LuongAmThanhThoiGianThuc.float32_sang_pcm16(tin_hieu_f32)

        # 4. Gửi metadata âm thanh chuẩn bị phát
        await self._gui_frame(
            writer,
            json.dumps(
                {
                    "loai": "bat_dau_phat_am_thanh",
                    "tan_so_lay_mau": self.tan_so_lay_mau,
                    "tong_bytes": len(pcm16_bytes),
                    "thoi_luong_giay": ket_qua_tts["thoi_luong_giay"],
                },
                ensure_ascii=False,
            ),
        )

        # 5. Truyền phát dữ liệu âm thanh theo từng mẩu nhỏ (streaming chunks 1024 bytes)
        chunk_size = 2048
        for i in range(0, len(pcm16_bytes), chunk_size):
            chunk = pcm16_bytes[i : i + chunk_size]
            await self._gui_frame(writer, chunk, ma_lenh=0x02)
            await asyncio.sleep(0.005)

        # 6. Báo kết thúc âm thanh
        await self._gui_frame(
            writer,
            json.dumps(
                {
                    "loai": "ket_thuc_phat_am_thanh",
                },
                ensure_ascii=False,
            ),
        )

    async def khoi_dong_async(self) -> None:
        """Khởi động máy chủ WebSocket bất đồng bộ."""
        self._dang_chay = True
        self._server = await asyncio.start_server(
            self._xu_ly_ket_noi,
            self.host,
            self.port,
        )
        logger.info("MayChuWebRTCVoice đang lắng nghe tại ws://%s:%d", self.host, self.port)

    async def dung_lai_async(self) -> None:
        """Dừng máy chủ WebSocket bất đồng bộ và đóng toàn bộ kết nối."""
        self._dang_chay = False
        for reader, writer in list(self._cac_ket_noi):
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        self._cac_ket_noi.clear()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def khoi_dong_chay_ngam(self) -> None:
        """Khởi động máy chủ trong một luồng ngầm (daemon thread) độc lập."""
        if self._thread is not None and self._thread.is_alive():
            return

        ready_evt = threading.Event()

        def _worker():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self.khoi_dong_async())
            ready_evt.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        ready_evt.wait(timeout=3.0)

    def dung_lai(self) -> None:
        """Dừng máy chủ chạy ngầm đồng bộ."""
        if self._loop is not None and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.dung_lai_async(), self._loop)
            try:
                future.result(timeout=2.0)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
