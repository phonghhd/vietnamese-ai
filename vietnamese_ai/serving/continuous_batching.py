"""Continuous Batching (BoPhanLoLienTuc) - Phân lô liên tục ở mức Token kết hợp KV-Cache thật."""

import uuid
from typing import Any, Dict, List, Optional

import numpy as np


class RequestState:
    """Trạng thái của một yêu cầu đang chờ hoặc đang giải mã từng token kèm bộ nhớ đệm KV-Cache riêng."""

    def __init__(self, prompt: str, max_tokens: int):
        self.req_id = str(uuid.uuid4())
        self.prompt = prompt
        self.max_tokens = max(1, int(max_tokens))
        self.generated_tokens: List[str] = []
        self.generated_ids: List[int] = []
        self.kv_cache: Optional[List[Dict[str, np.ndarray]]] = None
        self.pos_offset: int = 0
        self.next_token_id: Optional[int] = None
        self.is_finished = False


class ContinuousBatcher:
    """
    Continuous Batching (`BoPhanLoLienTuc` — Phân lô liên tục ở mức từng Token).

    Khác với Static Batching (phải đợi câu dài nhất trong lô sinh xong mới nhận câu tiếp theo),
    `ContinuousBatcher` giải phóng ngay bộ nhớ `KV-Cache` của request vừa hoàn thành (Early Finish)
    và lập tức điền request đang chờ ở `waiting_queue` vào khe trống trong `running_batch`.

    Hỗ trợ thực thi thật với `GPTModel` (`tien_kv`), `BoTinhChinhMoHinh`, `RAGPipeline`,
    hoặc hàm callback tùy chỉnh `(prompt, generated_tokens) -> token_str`.
    """

    def __init__(self, batch_size: int = 4, nhiet_do: float = 0.7, eos_token_id: int = -1):
        self.batch_size = max(1, int(batch_size))
        self.nhiet_do = max(0.0, float(nhiet_do))
        self.eos_token_id = eos_token_id
        self.waiting_queue: List[RequestState] = []
        self.running_batch: List[RequestState] = []
        self._thu_tu_gui: List[str] = []

    def add_request(self, prompt: str, max_tokens: int = 16) -> str:
        """Thêm một yêu cầu sinh văn bản mới vào hàng đợi phân lô liên tục."""
        req = RequestState(prompt, max_tokens)
        self.waiting_queue.append(req)
        self._thu_tu_gui.append(req.req_id)
        return req.req_id

    them_yeu_cau = add_request

    def _fill_batch(self) -> None:
        """Lấp đầy ngay các khe trống trong `running_batch` từ `waiting_queue`."""
        while len(self.running_batch) < self.batch_size and self.waiting_queue:
            next_req = self.waiting_queue.pop(0)
            self.running_batch.append(next_req)

    @staticmethod
    def _lay_gpt_va_tokenizer(mo_hinh: Any) -> tuple[Optional[Any], Optional[Any]]:
        """Trích xuất `GPTModel` có hỗ trợ `tien_kv` từ model hoặc wrapper."""
        if hasattr(mo_hinh, "tien_kv") and hasattr(mo_hinh, "so_tu_vung"):
            return mo_hinh, getattr(mo_hinh, "_tokenizer", None)
        inner = getattr(mo_hinh, "model", None)
        if inner is not None and hasattr(inner, "tien_kv") and hasattr(inner, "so_tu_vung"):
            tok = getattr(mo_hinh, "_tokenizer", None)
            if tok is None and hasattr(mo_hinh, "_hf_wrapper") and mo_hinh._hf_wrapper is not None:
                tok = getattr(mo_hinh._hf_wrapper, "_tokenizer", None)
            return inner, tok
        return None, None

    @staticmethod
    def _ma_hoa_prompt(prompt: str, so_tu_vung: int, tokenizer: Any = None) -> List[int]:
        if tokenizer is not None and hasattr(tokenizer, "ma_hoa"):
            ids = tokenizer.ma_hoa(prompt)
            if ids:
                return [int(x) % so_tu_vung for x in ids]
        words = prompt.strip().split()
        if not words:
            return [1]
        return [(sum(ord(c) for c in w) % max(1, so_tu_vung - 4)) + 4 for w in words]

    @staticmethod
    def _giai_ma_token(tok_id: int, tokenizer: Any = None) -> str:
        if tokenizer is not None and hasattr(tokenizer, "giai_ma"):
            txt = str(tokenizer.giai_ma([tok_id])).strip()
            if txt:
                return txt
        bang_tu_mau = [
            "trí",
            "tuệ",
            "nhân",
            "tạo",
            "Việt",
            "Nam",
            "hiệu",
            "năng",
            "cao",
            "an",
            "toàn",
            "tốc",
            "độ",
            "nhanh",
            "chính",
            "xác",
        ]
        return bang_tu_mau[tok_id % len(bang_tu_mau)]

    def _chon_token_tu_logits(self, logits_1d: np.ndarray) -> int:
        if self.nhiet_do <= 1e-5:
            return int(np.argmax(logits_1d))
        scaled = logits_1d / self.nhiet_do
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / np.sum(probs)
        return int(np.random.choice(len(probs), p=probs))

    def step(self, model_or_func: Any, tokenizer: Any = None) -> List[Dict[str, Any]]:
        """
        Thực thi đúng 1 bước giải mã (1 Token) cho toàn bộ các request đang nằm trong `running_batch`.
        Trả về danh sách các request đã hoàn thành ở bước này và nạp ngay request mới vào khe trống.
        """
        self._fill_batch()
        if not self.running_batch:
            return []

        gpt_model, tok_auto = self._lay_gpt_va_tokenizer(model_or_func)
        tok = tokenizer if tokenizer is not None else tok_auto
        finished_results: List[Dict[str, Any]] = []

        for req in self.running_batch:
            if gpt_model is not None:
                # Chạy thật bằng KV-Cache từng bước O(1) trên GPTModel
                if req.kv_cache is None:
                    p_ids = self._ma_hoa_prompt(
                        req.prompt, gpt_model.so_tu_vung, tok
                    )[-max(1, gpt_model.do_dai_toi_da // 2) :]
                    in_arr = np.array([p_ids], dtype=int)
                    next_logits, req.kv_cache = gpt_model.tien_kv(
                        in_arr, kv_caches=None, pos_offset=0
                    )
                    req.pos_offset = len(p_ids)
                else:
                    in_arr = np.array([[req.next_token_id or 1]], dtype=int)
                    next_logits, req.kv_cache = gpt_model.tien_kv(
                        in_arr, kv_caches=req.kv_cache, pos_offset=req.pos_offset
                    )
                    req.pos_offset += 1

                tok_id = self._chon_token_tu_logits(next_logits[0])
                req.next_token_id = tok_id
                req.generated_ids.append(tok_id)
                if self.eos_token_id >= 0 and tok_id == self.eos_token_id:
                    new_token = "<EOS>"
                else:
                    new_token = self._giai_ma_token(tok_id, tok)
            elif callable(model_or_func):
                new_token = str(model_or_func(req.prompt, req.generated_tokens))
            elif hasattr(model_or_func, "sinh_van_ban"):
                # Các wrapper LLM khác: chia từng từ theo bước phân lô
                if not hasattr(req, "_buffer_words"):
                    full_txt = str(model_or_func.sinh_van_ban(req.prompt, do_dai=req.max_tokens))
                    req._buffer_words = full_txt.split() or ["<EOS>"]  # type: ignore[attr-defined]
                buf: List[str] = req._buffer_words  # type: ignore[attr-defined]
                new_token = buf.pop(0) if buf else "<EOS>"
            else:
                new_token = "<EOS>"

            if new_token != "<EOS>":
                req.generated_tokens.append(new_token)

            if len(req.generated_tokens) >= req.max_tokens or new_token == "<EOS>":
                req.is_finished = True
                req.kv_cache = None  # Giải phóng ngay bộ nhớ KV-Cache của slot này
                finished_results.append(
                    {
                        "req_id": req.req_id,
                        "prompt": req.prompt,
                        "output": " ".join(req.generated_tokens),
                        "so_token": len(req.generated_tokens),
                    }
                )

        self.running_batch = [req for req in self.running_batch if not req.is_finished]
        self._fill_batch()
        return finished_results

    def chay_het(self, model_or_func: Any, tokenizer: Any = None) -> List[Dict[str, Any]]:
        """Chạy liên tục cho tới khi toàn bộ `waiting_queue` và `running_batch` hoàn tất."""
        ket_qua_map: Dict[str, Dict[str, Any]] = {}
        so_buoc_an_toan = 0
        gioi_han_buoc = sum(r.max_tokens for r in self.waiting_queue + self.running_batch) + 64

        while (self.waiting_queue or self.running_batch) and so_buoc_an_toan < gioi_han_buoc:
            hoan_thanh = self.step(model_or_func, tokenizer=tokenizer)
            for item in hoan_thanh:
                ket_qua_map[item["req_id"]] = item
            so_buoc_an_toan += 1

        thu_tu = [ket_qua_map[rid] for rid in self._thu_tu_gui if rid in ket_qua_map]
        self._thu_tu_gui.clear()
        return thu_tu


BoPhanLoLienTuc = ContinuousBatcher
