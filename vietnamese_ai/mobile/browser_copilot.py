from typing import List

import numpy as np

from vietnamese_ai.mobile.mobile_agent import MobileHybridAgent
from vietnamese_ai.rag.retriever import IdentityAwareRetriever
from vietnamese_ai.rag.vector_store import CSDLVector


class BrowserCopilot:
    """
    Trợ lý ảo duyệt web trên thiết bị di động (Web Copilot).
    """

    def __init__(self, mobile_agent: MobileHybridAgent):
        self.mobile_agent = mobile_agent
        # CSDL Vector In-memory nhỏ nhẹ cho Mobile
        self.local_csdl = CSDLVector(kich_thuoc=128, khoang_cach="cosine")
        self.retriever = IdentityAwareRetriever(vector_store=self.local_csdl)

    def _gia_lap_cao_du_lieu(self, url: str) -> str:
        return f"Đây là nội dung giả lập được cào từ {url}. Báo cáo tài chính quý 3 cho thấy lợi nhuận tăng 20%."

    def nen_ngu_canh(self, van_ban: str) -> str:
        """
        Nén ngữ cảnh (Context Compression) để giảm thiểu RAM trên di động.
        """
        import re

        n_van_ban = re.sub(r'\s+', ' ', van_ban).strip()
        stop_words = [" thì ", " là ", " mà ", " ráng ", " cũng ", " đang ", " đã ", " sẽ ", " của ", " các ", " những "]
        for word in stop_words:
            n_van_ban = n_van_ban.replace(word, " ")

        n_van_ban = re.sub(r'\s+', ' ', n_van_ban).strip()
        if len(n_van_ban) > 1000:
            n_van_ban = n_van_ban[:1000] + "...(đã nén)"
        return n_van_ban

    def doc_va_tom_tat(self, url: str) -> str:
        """Cào HTML và tóm tắt siêu tốc trên Mobile."""
        noi_dung = self._gia_lap_cao_du_lieu(url)
        noi_dung_nen = self.nen_ngu_canh(noi_dung)
        prompt = f"Hãy tóm tắt nội dung sau một cách ngắn gọn nhất:\n\n{noi_dung_nen}"
        # Gọi Mobile Agent (Agent sẽ tự quyết định chạy Local hay Cloud tùy theo Pin)
        return self.mobile_agent.chay(prompt)

    def hoi_dap_tai_lieu(self, url: str, cau_hoi: str, user_roles: List[str]) -> str:
        """
        Local RAG: Trò chuyện trực tiếp với tài liệu đang mở trên trình duyệt,
        có mã hóa bảo mật dữ liệu PII thông qua IdentityAwareRetriever.
        """
        noi_dung = self._gia_lap_cao_du_lieu(url)

        # Giả lập Embedding
        vector = np.random.rand(128).astype(np.float32)

        # Cấp quyền truy cập dựa trên user_roles (v15 Integration)
        self.local_csdl.chen(
            ma="doc_1",
            vector=vector,
            metadata={"roles_allowed": ["premium_user", "admin"], "text": noi_dung},
        )

        # Tìm kiếm có kiểm duyệt quyền
        ket_qua = self.retriever.tim_kiem_an_toan(
            query_vector=vector, user_roles=user_roles, top_k=1
        )

        if not ket_qua:
            return "Lỗi Bảo mật: Bạn không có quyền truy cập tài liệu này."

        ngu_canh = ket_qua[0]["metadata"]["text"]
        ngu_canh_nen = self.nen_ngu_canh(ngu_canh)

        prompt = f"Dựa vào tài liệu sau: {ngu_canh_nen}\n\nHãy trả lời: {cau_hoi}"
        return self.mobile_agent.chay(prompt)
