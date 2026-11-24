"""BrowserCopilot - Trợ Lý Duyệt Web & Local RAG Thực Tế Kèm Kiểm Soát SSRF Trên Thiết Bị Di Động."""

from __future__ import annotations

import ipaddress
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import List, Tuple

from vietnamese_ai.mobile.mobile_agent import MobileHybridAgent
from vietnamese_ai.rag.embedder import BoNhungVanBan
from vietnamese_ai.rag.retriever import IdentityAwareRetriever
from vietnamese_ai.rag.vector_store import CSDLVector


class _BoTrichXuatHTML(HTMLParser):
    """Bộ phân tích cú pháp HTML nhẹ để trích xuất văn bản thuần từ trang web."""

    def __init__(self) -> None:
        super().__init__()
        self._bo_qua = False
        self.doan_van: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        if tag.lower() in ("script", "style", "noscript"):
            self._bo_qua = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "noscript"):
            self._bo_qua = False

    def handle_data(self, data: str) -> None:
        if not self._bo_qua:
            text = data.strip()
            if text:
                self.doan_van.append(text)


class BrowserCopilot:
    """Trợ lý duyệt web thực tế trên thiết bị di động (HTML parser thật + bảo vệ SSRF + Local Identity-Aware RAG)."""

    def __init__(self, mobile_agent: MobileHybridAgent) -> None:
        self.mobile_agent = mobile_agent
        self.local_csdl = CSDLVector(kich_thuoc=128, khoang_cach="cosine")
        self.embedder = BoNhungVanBan(kich_thuoc=128, phuong_phap="hash")
        self.retriever = IdentityAwareRetriever(csdl_vector=self.local_csdl)

    @staticmethod
    def _trich_xuat_text_tu_html(html_raw: str) -> str:
        """Trích xuất văn bản sạch từ mã nguồn HTML."""
        parser = _BoTrichXuatHTML()
        parser.feed(html_raw)
        return " ".join(parser.doan_van)

    @staticmethod
    def _kiem_tra_url_an_toan(url: str) -> Tuple[bool, str]:
        """Chặn địa chỉ nội bộ / loopback / link-local (chống SSRF)."""
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").strip().lower()
        if not host:
            return False, "URL thiếu tên miền hợp lệ"
        if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(".local"):
            return False, f"Chặn truy cập SSRF tới địa chỉ nội bộ: {host}"
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, f"Chặn truy cập SSRF tới dải IP riêng/nội bộ: {host}"
        except ValueError:
            pass
        return True, ""

    def cao_du_lieu_web(self, url: str) -> str:
        """Tải và trích xuất văn bản thực tế từ tệp cục bộ hoặc URL HTTP/HTTPS (không sinh dữ liệu giả khi lỗi)."""
        if os.path.isfile(url):
            with open(url, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            if "<html" in raw.lower() or "<body" in raw.lower() or "<p>" in raw.lower():
                return self._trich_xuat_text_tu_html(raw)
            return raw

        if url.startswith("file://"):
            fpath = url[len("file://") :]
            if os.path.isfile(fpath):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    return self._trich_xuat_text_tu_html(f.read())
            return f"[Lỗi tải tệp]: Không tìm thấy tệp cục bộ '{fpath}'."

        if url.startswith(("http://", "https://")):
            an_toan, ly_do = self._kiem_tra_url_an_toan(url)
            if not an_toan:
                return f"[Lỗi Bảo mật SSRF]: {ly_do}"
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "VietnameseAI-MobileCopilot/31.0"}
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    html_bytes = resp.read(65536)
                    text = self._trich_xuat_text_tu_html(
                        html_bytes.decode("utf-8", errors="ignore")
                    )
                    if text:
                        return text
                    return f"[Cảnh báo tải trang]: Trang '{url}' không có nội dung văn bản trích xuất được."
            except Exception as exc:
                return f"[Không thể tải nội dung từ {url}]: {exc}"

        return f"[URL không hợp lệ]: Chỉ hỗ trợ đường dẫn tệp cục bộ hoặc giao thức http/https (nhận: '{url}')."

    def nen_ngu_canh(self, van_ban: str) -> str:
        """Nén ngữ cảnh (Context Compression) giảm từ dừng để tiết kiệm RAM trên di động."""
        n_van_ban = re.sub(r"\s+", " ", van_ban).strip()
        stop_words = [
            " thì ",
            " là ",
            " mà ",
            " ráng ",
            " cũng ",
            " đang ",
            " đã ",
            " sẽ ",
            " của ",
            " các ",
            " những ",
        ]
        for word in stop_words:
            n_van_ban = n_van_ban.replace(word, " ")

        n_van_ban = re.sub(r"\s+", " ", n_van_ban).strip()
        if len(n_van_ban) > 1000:
            n_van_ban = n_van_ban[:1000] + "...(đã nén)"
        return n_van_ban

    def doc_va_tom_tat(self, url: str) -> str:
        """Tải nội dung thật từ URL/tệp, nén ngữ cảnh và tóm tắt qua `mobile_agent`."""
        noi_dung = self.cao_du_lieu_web(url)
        noi_dung_nen = self.nen_ngu_canh(noi_dung)
        prompt = f"Hãy tóm tắt nội dung sau một cách ngắn gọn nhất:\n\n{noi_dung_nen}"
        return self.mobile_agent.chay(prompt)

    def hoi_dap_tai_lieu(self, url: str, cau_hoi: str, user_roles: List[str]) -> str:
        """Local RAG truy vấn nội dung thực tế từ tài liệu kèm kiểm soát vai trò qua `IdentityAwareRetriever`."""
        noi_dung = self.cao_du_lieu_web(url)

        doc_vector = self.embedder.nhung_mot(noi_dung).astype("float32")
        query_vector = self.embedder.nhung_mot(cau_hoi).astype("float32")

        self.local_csdl.chen(
            ma="doc_1",
            vector=doc_vector,
            metadata={"roles_allowed": ["premium_user", "admin"], "text": noi_dung},
        )

        ket_qua = self.retriever.tim_kiem_an_toan(
            query_vector=query_vector, user_roles=user_roles, top_k=1
        )

        if not ket_qua:
            return "Lỗi Bảo mật: Bạn không có quyền truy cập tài liệu này."

        ngu_canh = ket_qua[0]["metadata"]["text"]
        ngu_canh_nen = self.nen_ngu_canh(ngu_canh)

        prompt = f"Dựa vào tài liệu sau: {ngu_canh_nen}\n\nHãy trả lời: {cau_hoi}"
        return self.mobile_agent.chay(prompt)
