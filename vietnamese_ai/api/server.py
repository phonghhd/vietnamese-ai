"""ServerDonGian & MayChuOpenAI - Server HTTP phục vụ mô hình và tương thích chuẩn OpenAI API."""

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

import numpy as np

from vietnamese_ai.utils.logger import Logger


class _RequestHandler(BaseHTTPRequestHandler):
    """Xử lý HTTP request (hỗ trợ cả giao thức gốc và OpenAI-compatible `/v1/*`)."""

    mo_hinh: Any = None
    server_instance: Any = None
    logger = Logger("API")

    def do_POST(self):
        if self.path == "/du_doan":
            self._xu_ly_du_doan()
        elif self.path in ("/v1/chat/completions", "/chat/completions"):
            self._xu_ly_openai_chat()
        elif self.path in ("/v1/batch/completions", "/batch/completions"):
            self._xu_ly_batch_chat()
        elif self.path in ("/v1/embeddings", "/embeddings"):
            self._xu_ly_openai_embeddings()
        else:
            self._tra_loi(404, {"loi": "Khong tim thay endpoint"})

    def do_GET(self):
        if self.path == "/":
            self._tra_loi(
                200,
                {
                    "ten": "Vietnamese AI API",
                    "trang_thai": "hoat_dong",
                    "endpoints": [
                        "/du_doan (POST)",
                        "/v1/chat/completions (POST)",
                        "/v1/embeddings (POST)",
                        "/v1/models (GET)",
                        "/suc_khoe (GET)",
                        "/ (GET)",
                    ],
                },
            )
        elif self.path == "/suc_khoe":
            self._tra_loi(200, {"trang_thai": "tot"})
        elif self.path in ("/v1/models", "/models"):
            ten_mh = getattr(self.mo_hinh, "ten", type(self.mo_hinh).__name__ if self.mo_hinh else "vietnamese-ai")
            self._tra_loi(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": str(ten_mh),
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "vietnamese-ai",
                        }
                    ],
                },
            )
        else:
            self._tra_loi(404, {"loi": "Khong tim thay endpoint"})

    def _doc_json(self) -> Dict[str, Any]:
        MAX_BODY_BYTES = 1_048_576  # Giới hạn 1 MB chống tràn bộ nhớ DoS
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > MAX_BODY_BYTES:
            raise ValueError(f"Payload vượt quá giới hạn an toàn ({MAX_BODY_BYTES} bytes)")
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _xu_ly_du_doan(self):
        try:
            data = self._doc_json()
            if self.mo_hinh is None:
                self._tra_loi(500, {"loi": "Mo hinh chua duoc tai"})
                return

            raw_input = data.get("du_lieu", [])
            if isinstance(raw_input, list) and raw_input and isinstance(raw_input[0], str):
                ket_qua = self.mo_hinh.du_doan(raw_input)
                ket_qua_list = ket_qua.tolist() if hasattr(ket_qua, "tolist") else list(ket_qua)
            else:
                du_lieu = np.array(raw_input)
                if du_lieu.ndim == 1:
                    du_lieu = du_lieu.reshape(1, -1)
                ket_qua = self.mo_hinh.du_doan(du_lieu)
                ket_qua_list = ket_qua.tolist() if hasattr(ket_qua, "tolist") else list(ket_qua)

            self._tra_loi(
                200,
                {
                    "ket_qua": ket_qua_list,
                    "so_mau": len(ket_qua_list),
                },
            )
        except Exception as e:
            self._tra_loi(500, {"loi": str(e)})

    def _xu_ly_openai_chat(self):
        try:
            data = self._doc_json()
            if self.server_instance is not None:
                resp = self.server_instance.xu_ly_chat_completion(data)
            else:
                resp = ServerDonGian(self.mo_hinh).xu_ly_chat_completion(data)
            self._tra_loi(200, resp)
        except Exception as e:
            self._tra_loi(500, {"error": {"message": str(e), "type": "server_error"}})

    def _xu_ly_batch_chat(self):
        try:
            data = self._doc_json()
            if self.server_instance is not None:
                resp = self.server_instance.xu_ly_batch_chat(data)
            else:
                resp = ServerDonGian(self.mo_hinh).xu_ly_batch_chat(data)
            self._tra_loi(200, resp)
        except Exception as e:
            self._tra_loi(500, {"error": {"message": str(e), "type": "server_error"}})

    def _xu_ly_openai_embeddings(self):
        try:
            data = self._doc_json()
            if self.server_instance is not None:
                resp = self.server_instance.xu_ly_embeddings(data)
            else:
                resp = ServerDonGian(self.mo_hinh).xu_ly_embeddings(data)
            self._tra_loi(200, resp)
        except Exception as e:
            self._tra_loi(500, {"error": {"message": str(e), "type": "server_error"}})

    def _tra_loi(self, ma: int, data: dict):
        self.send_response(ma)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        self.logger.info(f"{self.address_string()} - {format % args}")


class ServerDonGian:
    """
    Server HTTP phục vụ mô hình AI qua API gốc (`/du_doan`) và chuẩn OpenAI API
    (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`).

    Tương thích với mọi mô hình của `vietnamese-ai`: `BaseModel`, `BoTinhChinhMoHinh`,
    `GPTModel`, `RAGPipeline`, `TacTu`, và `QuyTrinhDauCuoi`.
    """

    def __init__(
        self,
        mo_hinh: Any,
        ten: str = "VietnameseAI",
        bat_tuong_lua: bool = True,
    ):
        from vietnamese_ai.security.llm_firewall import TuongLuaBaoVeFramework

        self.mo_hinh = mo_hinh
        self.ten = ten
        self.bat_tuong_lua = bat_tuong_lua
        self.tuong_lua = TuongLuaBaoVeFramework() if bat_tuong_lua else None
        self.logger = Logger(ten)
        self._embedder: Any = None

    def xu_ly_chat_completion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Xử lý yêu cầu theo chuẩn OpenAI `/v1/chat/completions` kèm khiên bảo vệ Tường lửa AI & DLP."""
        messages: List[Dict[str, str]] = data.get("messages", [])
        model_name = data.get("model", self.ten)
        max_tokens = min(int(data.get("max_tokens", 128)), 4096)
        temperature = float(data.get("temperature", 0.7))

        user_prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = str(msg.get("content", ""))
                break
        if not user_prompt and messages:
            user_prompt = str(messages[-1].get("content", ""))

        # Lớp 1: Kiểm tra Tường lửa AI đầu vào
        if self.tuong_lua is not None:
            for msg in messages:
                if msg.get("role") == "user":
                    kq_fw = self.tuong_lua.kiem_tra_dau_vao(str(msg.get("content", "")))
                    if not kq_fw["an_toan"]:
                        return {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": f"[Bị chặn bởi Tường lửa AI]: {kq_fw['ly_do']}",
                                    },
                                    "finish_reason": "content_filter",
                                }
                            ],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        }

        tra_loi = ""
        if self.mo_hinh is None:
            tra_loi = "Mô hình chưa được khởi tạo."
        elif hasattr(self.mo_hinh, "hoi"):
            kq = self.mo_hinh.hoi(user_prompt)
            tra_loi = str(kq.get("tra_loi", kq)) if isinstance(kq, dict) else str(kq)
        elif hasattr(self.mo_hinh, "sinh_van_ban"):
            tra_loi = str(
                self.mo_hinh.sinh_van_ban(user_prompt, do_dai=max_tokens, nhiet_do=temperature)
            )
        elif hasattr(self.mo_hinh, "chay"):
            kq = self.mo_hinh.chay(user_prompt)
            tra_loi = str(kq.get("tra_loi", kq)) if isinstance(kq, dict) else str(kq)
        elif hasattr(self.mo_hinh, "du_doan"):
            kq = self.mo_hinh.du_doan([user_prompt])
            tra_loi = str(kq[0]) if len(kq) > 0 else ""
        else:
            tra_loi = str(self.mo_hinh)

        # Lớp 4: Chống thất thoát dữ liệu (DLP & Secret Redaction) ở đầu ra
        if self.tuong_lua is not None:
            tra_loi = self.tuong_lua.loc_dau_ra(tra_loi)

        prompt_tokens = len(user_prompt.split())
        completion_tokens = len(tra_loi.split())

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": tra_loi},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def xu_ly_embeddings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Xử lý yêu cầu nhúng văn bản theo chuẩn OpenAI `/v1/embeddings`."""
        raw_input = data.get("input", "")
        model_name = data.get("model", "vietnamese-embedding-128")
        texts = [raw_input] if isinstance(raw_input, str) else [str(x) for x in raw_input]

        if hasattr(self.mo_hinh, "nhung"):
            vecs = self.mo_hinh.nhung(texts)
        else:
            if self._embedder is None:
                from vietnamese_ai.rag.embedder import BoNhungVanBan

                self._embedder = BoNhungVanBan(kich_thuoc=128)
            vecs = self._embedder.nhung(texts)

        vecs_arr = np.asarray(vecs, dtype=float)
        if vecs_arr.ndim == 1:
            vecs_arr = vecs_arr.reshape(1, -1)

        items = [
            {
                "object": "embedding",
                "index": idx,
                "embedding": row.tolist(),
            }
            for idx, row in enumerate(vecs_arr)
        ]
        total_tok = sum(len(t.split()) for t in texts)
        return {
            "object": "list",
            "data": items,
            "model": model_name,
            "usage": {"prompt_tokens": total_tok, "total_tokens": total_tok},
        }

    def xu_ly_batch_chat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý nhiều truy vấn đồng thời qua `ContinuousBatcher` (`/v1/batch/completions`)
        kèm KV-Cache và Tường lửa AI Đa tầng.
        """
        from vietnamese_ai.serving.continuous_batching import ContinuousBatcher

        prompts = [str(p) for p in data.get("prompts", [])]
        max_tokens = min(int(data.get("max_tokens", 24)), 512)
        batch_size = max(1, int(data.get("batch_size", 4)))
        nhiet_do = float(data.get("temperature", 0.7))

        batcher = ContinuousBatcher(batch_size=batch_size, nhiet_do=nhiet_do)
        ket_qua_cuoi: List[Dict[str, Any]] = []
        map_vi_tri_hop_le: Dict[str, int] = {}

        for idx, p in enumerate(prompts):
            if self.tuong_lua is not None:
                kq_fw = self.tuong_lua.kiem_tra_dau_vao(p)
                if not kq_fw["an_toan"]:
                    ket_qua_cuoi.append(
                        {
                            "index": idx,
                            "prompt": p,
                            "content": f"[Bị chặn bởi Tường lửa AI]: {kq_fw['ly_do']}",
                            "finish_reason": "content_filter",
                        }
                    )
                    continue
            rid = batcher.add_request(p, max_tokens=max_tokens)
            map_vi_tri_hop_le[rid] = idx
            ket_qua_cuoi.append({"index": idx, "prompt": p, "content": "", "finish_reason": "stop"})

        if map_vi_tri_hop_le:
            da_sinh = batcher.chay_het(self.mo_hinh)
            for item in da_sinh:
                pos = map_vi_tri_hop_le[item["req_id"]]
                out_txt = item["output"]
                if self.tuong_lua is not None:
                    out_txt = self.tuong_lua.loc_dau_ra(out_txt)
                ket_qua_cuoi[pos]["content"] = out_txt
                ket_qua_cuoi[pos]["tokens"] = item.get("so_token", 0)

        return {
            "id": f"batchcmpl-{uuid.uuid4().hex[:12]}",
            "object": "batch.completion",
            "created": int(time.time()),
            "model": self.ten,
            "results": ket_qua_cuoi,
        }

    def chay(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Khởi động server."""
        _RequestHandler.mo_hinh = self.mo_hinh
        _RequestHandler.server_instance = self
        _RequestHandler.logger = self.logger

        server = HTTPServer((host, port), _RequestHandler)
        self.logger.info(f"Server '{self.ten}' đang chạy tại http://{host}:{port}")
        self.logger.info(
            "Endpoints: GET / | GET /suc_khoe | POST /du_doan | POST /v1/chat/completions | POST /v1/embeddings"
        )

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.logger.info("Server đang tắt...")
            server.server_close()


MayChuOpenAI = ServerDonGian
OpenAICompatibleServer = ServerDonGian
