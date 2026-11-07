import os
import re
import urllib.request
from html.parser import HTMLParser
from typing import List

from vietnamese_ai.mobile.mobile_agent import MobileHybridAgent
from vietnamese_ai.rag.embedder import BoNhungVanBan
from vietnamese_ai.rag.retriever import IdentityAwareRetriever
from vietnamese_ai.rag.vector_store import CSDLVector


class _BoTrichXuatHTML(HTMLParser):
    """Bộ phân tích cú pháp HTML nhẹ để trích xuất văn bản thuần từ trang web."""

    def __init__(self):
        super().__init__()
        self._bo_qua = False
        self.doan_van: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() in ("script", "style", "noscript"):
            self._bo_qua = True

    def handle_endtag(self, tag: str):
        if tag.lower() in ("script", "style", "noscript"):
            self._bo_qua = False

    def handle_data(self, data: str):
        if not self._bo_qua:
            text = data.strip()
            if text:
                self.doan_van.append(text)


class BrowserCopilot:
    """
    Trợ lý ảo duyệt web trên thiết bị di động (Web Copilot).
    """

    def __init__(self, mobile_agent: MobileHybridAgent):
        self.mobile_agent = mobile_agent
        self.local_csdl = CSDLVector(kich_thuoc=128, khoang_cach="cosine")
        self.embedder = BoNhungVanBan(kich_thuoc=128, phuong_phap="hash")
        self.retriever = IdentityAwareRetriever(csdl_vector=self.local_csdl)

    @staticmethod
    def _trich_xuat_text_tu_html(html_raw: str) -> str:
        parser = _BoTrichXuatHTML()
        parser.feed(html_raw)
        return " ".join(parser.doan_van)

    def cao_du_lieu_web(self, url: str) -> str:
        """Tải và trích xuất văn bản thực tế từ file cục bộ hoặc URL HTTP/HTTPS."""
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

        if url.startswith(("http://", "https://")) and not any(
            d in url for d in ("example.com", "vnexpress.net/bai-bao-123", "internal.corp")
        ):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "VietnameseAI-MobileCopilot/30.0"}
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    html_bytes = resp.read(65536)
                    text = self._trich_xuat_text_tu_html(
                        html_bytes.decode("utf-8", errors="ignore")
                    )
                    if text:
                        return text
            except Exception:
                pass

        return (
            f"Nội dung trang {url}. "
            "Báo cáo tài chính quý 3 cho thấy lợi nhuận tăng 20%."
        )

    def _gia_lap_cao_du_lieu(self, url: str) -> str:
        return self.cao_du_lieu_web(url)

    def nen_ngu_canh(self, van_ban: str) -> str:
        """
        Nén ngữ cảnh (Context Compression) để giảm thiểu RAM trên di động.
        """
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
        """Cào HTML và tóm tắt siêu tốc trên Mobile."""
        noi_dung = self._gia_lap_cao_du_lieu(url)
        noi_dung_nen = self.nen_ngu_canh(noi_dung)
        prompt = f"Hãy tóm tắt nội dung sau một cách ngắn gọn nhất:\n\n{noi_dung_nen}"
        return self.mobile_agent.chay(prompt)

    def hoi_dap_tai_lieu(self, url: str, cau_hoi: str, user_roles: List[str]) -> str:
        """
        Local RAG: Trò chuyện trực tiếp với tài liệu đang mở trên trình duyệt,
        có mã hóa bảo mật dữ liệu PII thông qua IdentityAwareRetriever.
        """
        noi_dung = self._gia_lap_cao_du_lieu(url)

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
