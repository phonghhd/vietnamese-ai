"""TuongLuaAI & TuongLuaBaoVeFramework - Hệ thống Tường lửa Tự động Đa tầng bảo vệ Vietnamese AI Framework."""

from __future__ import annotations

import base64
import re
import threading
import time
import unicodedata
from collections import deque
from typing import Any, Dict, List, Set, Tuple

from vietnamese_ai.security.data_sanitizer import DataSanitizer


class AhoCorasickTrie:
    """Cấu trúc dữ liệu Aho-Corasick quét đa mẫu O(M+N) đếm số từ khóa riêng biệt khớp."""

    def __init__(self) -> None:
        self._goto: List[Dict[str, int]] = [{}]
        self._fail: List[int] = [0]
        self._output: List[List[str]] = [[]]
        self._da_xay_dung: bool = False

    def add_word(self, word: str) -> None:
        """Thêm một từ khóa vào automaton Aho-Corasick."""
        w = word.lower()
        if not w:
            return
        state = 0
        for char in w:
            nxt = self._goto[state].get(char)
            if nxt is None:
                nxt = len(self._goto)
                self._goto[state][char] = nxt
                self._goto.append({})
                self._fail.append(0)
                self._output.append([])
            state = nxt
        if w not in self._output[state]:
            self._output[state].append(w)
        self._da_xay_dung = False

    def _xay_dung_failure_links(self) -> None:
        """Xây dựng liên kết thất bại (BFS failure links) theo thuật toán Aho-Corasick."""
        q: deque[int] = deque()
        self._fail = [0] * len(self._goto)
        for nxt in self._goto[0].values():
            self._fail[nxt] = 0
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, u in self._goto[r].items():
                q.append(u)
                v = self._fail[r]
                while v != 0 and ch not in self._goto[v]:
                    v = self._fail[v]
                self._fail[u] = self._goto[v].get(ch, 0)
                for pat in self._output[self._fail[u]]:
                    if pat not in self._output[u]:
                        self._output[u].append(pat)
        self._da_xay_dung = True

    def count_matches(self, text: str) -> int:
        """Đếm số lượng từ khóa rủi ro riêng biệt xuất hiện trong chuỗi đầu vào."""
        if not text:
            return 0
        if not self._da_xay_dung:
            self._xay_dung_failure_links()
        state = 0
        khop_rieng_biet: Set[str] = set()
        for ch in text.lower():
            while state != 0 and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            if self._output[state]:
                khop_rieng_biet.update(self._output[state])
        return len(khop_rieng_biet)


class TuongLuaAI:
    """
    Tường lửa (Firewall) cho LLM, chuyên phát hiện và ngăn chặn các cuộc tấn công Prompt Injection,
    Jailbreak, trích xuất System Prompt, SQL/Command Injection hoặc các chỉ thị độc hại
    (kể cả khi bị che giấu bằng ký tự Zero-Width, phân tách dấu chấm hoặc mã hóa Base64).
    """

    MAU_HIEM_DOC = [
        r"(?i)\bbỏ\s+qua\s+(?:tất\s+cả\s+|toàn\s+bộ\s+|mọi\s+)?(?:các\s+)?(?:hướng\s+dẫn|chỉ\s+dẫn|chỉ\s+thị|lệnh|quy\s+tắc)\b",
        r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?(?:previous\s+|prior\s+)?(?:instructions|commands|prompts?|rules)\b",
        r"(?i)\bbạn\s+(?:bây\s+giờ\s+)?(?:sẽ\s+)?đóng\s+vai\b",
        r"(?i)\byou\s+(?:will\s+)?(?:now\s+)?act\s+as\b",
        r"(?i)\byou\s+are\s+now\s+a\s+hacker\b",
        r"(?i)(?:\b(?:act\s+as|you\s+are\s+now|đóng\s+vai(?:\s+là)?|bạn\s+(?:bây\s+giờ\s+)?(?:sẽ\s+)?là|chế\s+độ|mode)\s+dan\b|\bdan\s*(?:\(|mode\b|jailbreak\b|do\s+anything\b|không\s+giới\s+hạn\b|unlimited\b))",
        r"(?i)\bdo\s+anything\s+now\b",
        r"(?i)\blàm\s+bất\s+cứ\s+điều\s+gì\b",
        r"(?i)\bquên\s+(?:tất\s+cả\s+)?(?:những\s+)?gì\s+tôi\s+đã\s+nói\b",
        r"(?i)\bsystem\s+prompt\b",
        r"(?i)\btừ\s+giờ\s+trở\s+đi\b.{0,45}\b(?:bạn\s+(?:sẽ|hãy|phải|là)|đóng\s+vai|bỏ\s+qua|không\s+tuân|vô\s+hiệu|hacker|jailbreak|trả\s+lời\s+mọi)\b",
        r"(?i)(?:lặp\s+lại|in\s+ra|hiển\s+thị|tiết\s+lộ|repeat|reveal)\s+(?:toàn\s+bộ\s+|mọi\s+)?(?:prompt\s+hệ\s+thống|chỉ\s+thị\s+ban\s+đầu|system\s+prompt|mật\s+khẩu\s+nội\s+bộ|khóa\s+bí\s+mật)",
        r"(?i)(?:tắt|vô\s+hiệu\s+hóa|disable|turn\s+off)\s+(?:mọi\s+|tất\s+cả\s+)?(?:biện\s+pháp\s+an\s+toàn|bảo\s+mật|tường\s+lửa|safety|guardrails|firewall)",
        r"(?i)\bbo\s+qua\s+(?:tat\s+ca\s+|toan\s+bo\s+|moi\s+)?(?:cac\s+)?(?:huong\s+dan|chi\s+dan|chi\s+thi|lenh|quy\s+tac)\b",
        r"(?i)(?:tiet\s+lo|in\s+ra|hien\s+thi)\s+(?:toan\s+bo\s+|moi\s+)?(?:prompt\s+he\s+thong|chi\s+thi\s+ban\s+dau|mat\s+khau\s+noi\s+bo|khoa\s+bi\s+mat)",
        r"(?i)(?:tat|vo\s+hieu\s+hoa)\s+(?:moi\s+|tat\s+ca\s+)?(?:bien\s+phap\s+an\s+toan|bao\s+mat|tuong\s+lua)",
        r"(?i)(?:sudo\s+rm\s+-rf|tải\s+payload|/etc/passwd|/etc/shadow|union\s+select\s+|drop\s+table\s+)",
    ]

    TU_KHOA_RUI_RO = [
        "hack",
        "bypass",
        "exploit",
        "lỗ hổng",
        "mật khẩu",
        "password",
        "vô hiệu hóa",
        "hệ thống",
        "trái phép",
        "payload",
        "backdoor",
        "sudo rm",
    ]

    KY_TU_VO_HINH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]")
    MAU_CHIA_TACH = re.compile(r"(?<=\b[a-zA-Z])[.\-_](?=[a-zA-Z]\b)")
    MAU_BASE64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])")

    def __init__(self, ngat_ket_noi_khi_phat_hien: bool = True, use_aho_corasick: bool = True):
        self.ngat_ket_noi = ngat_ket_noi_khi_phat_hien
        self.mau_regex = [re.compile(mau) for mau in self.MAU_HIEM_DOC]
        self.use_aho_corasick = use_aho_corasick
        if self.use_aho_corasick:
            self.trie = AhoCorasickTrie()
            for tu in self.TU_KHOA_RUI_RO:
                self.trie.add_word(tu)

    @classmethod
    def chuan_hoa_prompt(cls, prompt: str) -> str:
        """
        Chuẩn hóa Unicode NFKC, loại bỏ ký tự tàng hình (Zero-Width) và giải mã các chuỗi Base64 ẩn
        để ngăn chặn kỹ thuật Obfuscation vượt tường lửa.
        """
        if not prompt:
            return ""
        s = unicodedata.normalize("NFKC", prompt)
        s = cls.KY_TU_VO_HINH.sub("", s)
        s = cls.MAU_CHIA_TACH.sub("", s)

        # Giải mã thử các khối Base64 đáng ngờ và nối vào văn bản kiểm tra
        decoded_parts: List[str] = []
        for match in cls.MAU_BASE64.findall(s):
            padded = match + ("=" * ((4 - len(match) % 4) % 4))
            try:
                raw = base64.b64decode(padded).decode("utf-8", errors="ignore")
                if raw and sum(1 for c in raw if c.isalpha() or c.isspace()) >= len(raw) * 0.7:
                    decoded_parts.append(raw)
            except Exception:
                pass
        if decoded_parts:
            s = s + " " + " ".join(decoded_parts)
        return s

    def kiem_tra_prompt(self, prompt: str) -> Tuple[bool, str]:
        """
        Kiểm tra xem prompt đầu vào có dấu hiệu Prompt Injection hay không.

        Args:
            prompt (str): Chuỗi văn bản người dùng nhập.

        Returns:
            Tuple[bool, str]: (True nếu an toàn, Lý do nếu không an toàn)
        """
        if len(prompt) > 8000:
            return False, "Prompt vượt quá độ dài an toàn cho phép (8000 ký tự)."

        prompt_chuan = self.chuan_hoa_prompt(prompt).lower()

        # 1. Kiểm tra bằng Regex Patterns
        for mau in self.mau_regex:
            if mau.search(prompt_chuan):
                return False, "Phát hiện dấu hiệu tấn công Prompt Injection (Pattern matched)."

        # 2. Heuristic check: Mật độ từ khóa rủi ro
        if self.use_aho_corasick:
            so_tu_rui_ro = self.trie.count_matches(prompt_chuan)
        else:
            so_tu_rui_ro = sum(1 for tu in self.TU_KHOA_RUI_RO if tu in prompt_chuan)

        if so_tu_rui_ro >= 3:
            return False, "Phát hiện quá nhiều từ khóa rủi ro, có khả năng là truy vấn độc hại."

        return True, "An toàn."

    def loc_prompt(self, prompt: str) -> str:
        """
        Kiểm tra và chặn prompt nếu phát hiện dấu hiệu tấn công.
        """
        an_toan, ly_do = self.kiem_tra_prompt(prompt)
        if not an_toan and self.ngat_ket_noi:
            raise ValueError(f"Tường lửa AI đã chặn yêu cầu: {ly_do}")

        return prompt


class TuongLuaBaoVeFramework:
    """
    Hệ thống Tường lửa Tự động Đa tầng (Autonomous Multi-Layer AI Firewall - `KhienBaoVeAI`).

    Tự động bảo vệ toàn bộ vòng đời thực thi của `vietnamese-ai` qua 5 lớp khiên:
    1. **Lớp 1 (Inbound Shield)**: Chống Prompt Injection, Jailbreak, Base64/Unicode Obfuscation, SQLi/CMDi và giới hạn tốc độ (Rate Limit) chống DoS.
    2. **Lớp 2 (RAG & Context Shield)**: Quét và khử độc Indirect Prompt Injection ẩn trong tài liệu bên ngoài trước khi đưa vào `RAGPipeline`.
    3. **Lớp 3 (Tool & File Shield)**: Kiểm duyệt tham số công cụ của Agent (chặn Path Traversal, chặn đọc `.env`/`users.json`/`/etc/passwd`, chặn SSRF nội bộ).
    4. **Lớp 4 (Outbound DLP & Canary Shield)**: Tự động ẩn PII tiếng Việt, API Keys (`vai_`, `sk-`, `hf_`, `AIza`), JWT, Private Keys, DB URI và chặn rò rỉ System Prompt (Canary Token).
    5. **Lớp 5 (Auto-Wrapper `boc_bao_ve`)**: Tự động bọc bảo vệ chỉ với 1 dòng lệnh cho bất kỳ Model, `RAGPipeline`, `TacTu` hoặc `ServerDonGian` nào.
    """

    DUONG_DAN_NHAY_CAM = (
        ".env",
        "users.json",
        "tokens.json",
        "id_rsa",
        "id_ed25519",
        "/etc/passwd",
        "/etc/shadow",
        ".git/config",
        ".ssh/",
    )

    SSRF_NOI_BO = (
        "169.254.169.254",
        "metadata.google.internal",
        "127.0.0.1",
        "0.0.0.0",
        "localhost",
        "[::1]",
        "::1",
        "file://",
        "gopher://",
    )

    GIOI_HAN_NHAT_KY = 2000
    GIOI_HAN_CLIENT_RATE = 10000

    def __init__(
        self,
        chan_khi_vi_pham: bool = False,
        tu_dong_an_pii_dau_ra: bool = True,
        gioi_han_req_moi_phut: int = 120,
        canary_token: str = "CANARY_SECRET_VAI_30",
    ):
        self.chan_khi_vi_pham = chan_khi_vi_pham
        self.tu_dong_an_pii_dau_ra = tu_dong_an_pii_dau_ra
        self.gioi_han_req_moi_phut = max(1, gioi_han_req_moi_phut)
        self.canary_token = canary_token
        self.tuong_lua_prompt = TuongLuaAI(ngat_ket_noi_khi_phat_hien=chan_khi_vi_pham)
        self._lock = threading.Lock()
        self._lich_su_request: Dict[str, List[float]] = {}
        self.nhat_ky_an_ninh: List[Dict[str, Any]] = []

    def _ghi_nhat_ky(self, tang: str, muc_do: str, ly_do: str, du_lieu_mau: str = "") -> None:
        with self._lock:
            if len(self.nhat_ky_an_ninh) >= self.GIOI_HAN_NHAT_KY:
                del self.nhat_ky_an_ninh[: len(self.nhat_ky_an_ninh) - self.GIOI_HAN_NHAT_KY + 1]
            self.nhat_ky_an_ninh.append(
                {
                    "thoi_gian": time.time(),
                    "tang_bao_ve": tang,
                    "muc_do": muc_do,
                    "ly_do": ly_do,
                    "mau_trich_doan": du_lieu_mau[:120],
                }
            )

    def kiem_tra_toc_do(self, client_id: str = "default") -> Tuple[bool, str]:
        """Kiểm tra giới hạn tốc độ truy vấn theo cửa sổ trượt 60 giây (chống DoS/Flood)."""
        now = time.time()
        with self._lock:
            if len(self._lich_su_request) > self.GIOI_HAN_CLIENT_RATE:
                het_han = [
                    cid
                    for cid, times in self._lich_su_request.items()
                    if not times or now - times[-1] >= 60.0
                ]
                for cid in het_han:
                    self._lich_su_request.pop(cid, None)
            ds = [t for t in self._lich_su_request.get(client_id, []) if now - t < 60.0]
            if len(ds) >= self.gioi_han_req_moi_phut:
                self._lich_su_request[client_id] = ds
                vuot = True
            else:
                ds.append(now)
                self._lich_su_request[client_id] = ds
                vuot = False
        if vuot:
            ly_do = f"Vượt quá giới hạn tốc độ ({self.gioi_han_req_moi_phut} yêu cầu/phút) cho '{client_id}'."
            self._ghi_nhat_ky("rate_limit", "HIGH", ly_do, client_id)
            return False, ly_do
        return True, "Hợp lệ."

    def kiem_tra_dau_vao(self, van_ban: str, client_id: str = "default") -> Dict[str, Any]:
        """Lớp 1: Quét toàn diện truy vấn đầu vào (Rate Limit + Prompt Injection + Obfuscation)."""
        ok_rate, ly_do_rate = self.kiem_tra_toc_do(client_id)
        if not ok_rate:
            if self.chan_khi_vi_pham:
                raise PermissionError(ly_do_rate)
            return {"an_toan": False, "ly_do": ly_do_rate, "tang": "rate_limit"}

        from vietnamese_ai.security.data_defense import CongChatLuongDauVao

        van_ban_kiem_tra, co_trojan_bidi = CongChatLuongDauVao.loai_bo_unicode_trojan(str(van_ban or ""))
        if co_trojan_bidi:
            ly_do_bidi = "Phát hiện ký tự điều khiển ẩn Unicode Bidi Trojan / Null-byte."
            self._ghi_nhat_ky("inbound_unicode_trojan", "CRITICAL", ly_do_bidi, str(van_ban))
            if self.chan_khi_vi_pham:
                raise ValueError(f"Tường lửa AI đã chặn đầu vào: {ly_do_bidi}")
            return {"an_toan": False, "ly_do": ly_do_bidi, "tang": "inbound_unicode_trojan"}

        an_toan, ly_do = self.tuong_lua_prompt.kiem_tra_prompt(van_ban_kiem_tra)
        if not an_toan:
            self._ghi_nhat_ky("inbound_prompt", "CRITICAL", ly_do, van_ban)
            if self.chan_khi_vi_pham:
                raise ValueError(f"Tường lửa AI đã chặn đầu vào: {ly_do}")
            return {"an_toan": False, "ly_do": ly_do, "tang": "inbound_prompt"}

        return {"an_toan": True, "ly_do": "An toàn.", "tang": "inbound_prompt"}

    def lam_sach_tai_lieu_rag(self, van_ban: str) -> Dict[str, Any]:
        """
        Lớp 2: Quét và khử độc các câu chứa Indirect Prompt Injection ẩn trong tài liệu RAG,
        đồng thời ẩn PII/Secrets trước khi lưu vào Vector Store và bảo toàn cấu trúc dòng văn bản.
        """
        dong_list = van_ban.split("\n")
        dong_an_toan: List[str] = []
        so_cau_bi_loai = 0

        for dong in dong_list:
            if not dong.strip():
                dong_an_toan.append(dong)
                continue
            cac_cau = re.split(r"(?<=[.!?])\s+", dong)
            cau_sach_trong_dong = []
            for cau in cac_cau:
                ok, ly_do = self.tuong_lua_prompt.kiem_tra_prompt(cau)
                if ok:
                    cau_sach_trong_dong.append(cau)
                else:
                    so_cau_bi_loai += 1
                    self._ghi_nhat_ky("rag_indirect_injection", "HIGH", ly_do, cau)
            if cau_sach_trong_dong:
                dong_an_toan.append(" ".join(cau_sach_trong_dong))

        van_ban_sach = "\n".join(dong_an_toan)
        if self.tu_dong_an_pii_dau_ra:
            van_ban_sach = DataSanitizer.lam_sach(van_ban_sach, loc_bi_mat=True)

        return {
            "van_ban_sach": van_ban_sach,
            "so_cau_bi_loai": so_cau_bi_loai,
            "an_toan": so_cau_bi_loai == 0,
        }

    def kiem_tra_lenh_cong_cu(
        self, ten_cong_cu: str, tham_so: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Lớp 3: Kiểm duyệt lệnh gọi công cụ của Agent (chặn Path Traversal, đọc file nhạy cảm, SSRF).
        """
        chuoi_tham_so = " ".join(f"{k}={v}" for k, v in tham_so.items()).lower()

        if ".." in chuoi_tham_so or "/etc/" in chuoi_tham_so or "\\..\\" in chuoi_tham_so:
            ly_do = f"Chặn hành vi Path Traversal trong công cụ '{ten_cong_cu}'."
            self._ghi_nhat_ky("tool_shield", "CRITICAL", ly_do, chuoi_tham_so)
            return False, ly_do

        for nhạy_cảm in self.DUONG_DAN_NHAY_CAM:
            if nhạy_cảm in chuoi_tham_so:
                ly_do = f"Chặn truy cập đường dẫn nhạy cảm '{nhạy_cảm}' trong công cụ '{ten_cong_cu}'."
                self._ghi_nhat_ky("tool_shield", "CRITICAL", ly_do, chuoi_tham_so)
                return False, ly_do

        for ssrf in self.SSRF_NOI_BO:
            if ssrf in chuoi_tham_so:
                ly_do = f"Chặn nguy cơ SSRF nội bộ ('{ssrf}') trong công cụ '{ten_cong_cu}'."
                self._ghi_nhat_ky("tool_shield", "CRITICAL", ly_do, chuoi_tham_so)
                return False, ly_do

        return True, "Hợp lệ."

    def loc_dau_ra(self, van_ban: str) -> str:
        """
        Lớp 4: Kiểm duyệt đầu ra từ LLM/Agent — Chặn rò rỉ Canary/System Prompt
        và tự động ẩn toàn bộ PII cùng API Keys, JWT, Private Keys, DB URI.
        """
        if not isinstance(van_ban, str) or not van_ban:
            return van_ban

        if self.canary_token and self.canary_token in van_ban:
            self._ghi_nhat_ky(
                "outbound_canary",
                "CRITICAL",
                "Phát hiện rò rỉ Canary Token / System Prompt ở đầu ra.",
                van_ban,
            )
            return "[Bị chặn bởi Tường lửa AI: Phát hiện nguy cơ rò rỉ chỉ thị hệ thống nội bộ]"

        from vietnamese_ai.security.data_defense import CongAnToanDauRa

        ro_ri = DataSanitizer.quet_ro_ri(van_ban)
        if ro_ri:
            self._ghi_nhat_ky(
                "outbound_dlp",
                "MEDIUM",
                f"Đã tự động che giấu thông tin nhạy cảm: {', '.join(ro_ri.keys())}",
                van_ban,
            )

        kq_out = CongAnToanDauRa.kiem_tra_va_lam_sach(
            van_ban,
            chan_lenh_pha_hoai=True,
            an_pii=self.tu_dong_an_pii_dau_ra,
        )
        if kq_out["vi_pham"]:
            self._ghi_nhat_ky(
                "outbound_safety_gate",
                "HIGH",
                "; ".join(kq_out["vi_pham"]),
                van_ban,
            )
        return kq_out["van_ban_sach"]

    def boc_bao_ve(self, doi_tuong: Any) -> Any:
        """
        Lớp 5: Tự động gắn lớp khiên bảo vệ đầu-cuối (Inbound + Outbound DLP) vào bất kỳ
        mô hình (`BoTinhChinhMoHinh`, `GPTModel`, `VietnameseLLM`), `RAGPipeline` hoặc `TacTu` nào.
        """
        firewall = self

        if hasattr(doi_tuong, "sinh_van_ban"):
            goc_sinh = doi_tuong.sinh_van_ban

            def _sinh_an_toan(prompt: str, *args: Any, **kwargs: Any) -> str:
                kq_in = firewall.kiem_tra_dau_vao(str(prompt))
                if not kq_in["an_toan"]:
                    return f"[Bị chặn bởi Tường lửa AI]: {kq_in['ly_do']}"
                out = goc_sinh(prompt, *args, **kwargs)
                return firewall.loc_dau_ra(str(out))

            doi_tuong.sinh_van_ban = _sinh_an_toan

        if hasattr(doi_tuong, "hoi"):
            goc_hoi = doi_tuong.hoi

            def _hoi_an_toan(cau_hoi: str, *args: Any, **kwargs: Any) -> Any:
                kq_in = firewall.kiem_tra_dau_vao(str(cau_hoi))
                if not kq_in["an_toan"]:
                    return {
                        "cau_hoi": cau_hoi,
                        "tra_loi": f"[Bị chặn bởi Tường lửa AI]: {kq_in['ly_do']}",
                        "bi_chan": True,
                        "nguon": [],
                        "so_luong_nguon": 0,
                    }
                res = goc_hoi(cau_hoi, *args, **kwargs)
                if isinstance(res, dict) and "tra_loi" in res:
                    res = dict(res)
                    res["tra_loi"] = firewall.loc_dau_ra(str(res["tra_loi"]))
                    return res
                return firewall.loc_dau_ra(str(res))

            doi_tuong.hoi = _hoi_an_toan

        if hasattr(doi_tuong, "chay") and not hasattr(doi_tuong, "them_node"):
            goc_chay = doi_tuong.chay

            def _chay_an_toan(truy_van: str, *args: Any, **kwargs: Any) -> Any:
                kq_in = firewall.kiem_tra_dau_vao(str(truy_van))
                if not kq_in["an_toan"]:
                    return f"[Bị chặn bởi Tường lửa AI]: {kq_in['ly_do']}"
                res = goc_chay(truy_van, *args, **kwargs)
                return firewall.loc_dau_ra(str(res)) if isinstance(res, str) else res

            doi_tuong.chay = _chay_an_toan

        setattr(doi_tuong, "_tuong_lua_bao_ve", firewall)
        return doi_tuong

    def thong_ke(self) -> Dict[str, Any]:
        """Thống kê hoạt động của Tường lửa Tự động Đa tầng."""
        with self._lock:
            nhat_ky = list(self.nhat_ky_an_ninh)
        theo_tang: Dict[str, int] = {}
        for muc in nhat_ky:
            t = muc["tang_bao_ve"]
            theo_tang[t] = theo_tang.get(t, 0) + 1
        return {
            "tong_su_co_da_chan": len(nhat_ky),
            "chi_tiet_theo_tang": theo_tang,
            "nhat_ky_gan_nhat": nhat_ky[-5:],
        }


KhienBaoVeAI = TuongLuaBaoVeFramework
AIFirewallShield = TuongLuaBaoVeFramework
