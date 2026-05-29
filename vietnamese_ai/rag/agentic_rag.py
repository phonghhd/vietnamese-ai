from typing import Any, Dict

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.agents.tools import CongCu
from vietnamese_ai.rag.rag_pipeline import RAGPipeline
from vietnamese_ai.rag.realtime_sync import RAGEventBus
from vietnamese_ai.security.data_sanitizer import DataSanitizer


class AgenticRAGPipeline(RAGPipeline):
    """
    RAG Pipeline Động và Có Tư duy (Agentic RAG).
    Thay vì tìm kiếm mù quáng, nó sử dụng Tác tử (Agent) để tự quyết định
    cần dùng công cụ nào (Vector, Graph, v.v) để trả lời.
    Đồng thời tự động lắng nghe Event Bus để cập nhật Real-time.
    """
    def __init__(self, llm_engine: Any, *args, **kwargs):
        # Khởi tạo RAG Pipeline truyền thống (v10) ở bên dưới
        super().__init__(*args, **kwargs)

        self.llm_engine = llm_engine

        # Tạo các công cụ cho Agent
        cong_cu_vector = CongCu(
            ten="tim_kiem_tai_lieu_chinh",
            mo_ta="Tìm kiếm thông tin từ CSDL Vector nội bộ của công ty. Tham số: cau_hoi (str).",
            ham_thuc_thi=self._tool_tim_kiem_vector
        )
        self._current_user_role = "khach" # Mặc định là khách

        # Nếu có GraphStore thì truyền thêm vào sau
        self.agent = TacTu(
            llm=self.llm_engine,
            danh_sach_cong_cu=[cong_cu_vector],
            max_iterations=5
        )

        # Đăng ký Real-time Sync
        self.event_bus = RAGEventBus()
        self.event_bus.dang_ky_lang_nghe("NEW_DOCUMENT", self._handle_new_document)
        print("[AgenticRAG] Đã kích hoạt tính năng Agentic & Real-time Sync.")

    def _tool_tim_kiem_vector(self, cau_hoi: str) -> str:
        """Tool cho Agent dùng để chọc vào VectorDB, có tích hợp Identity-Aware."""
        # Giả lập filter metadata (VectorDB sẽ loại bỏ các document mà user_role không được phép)
        # Nếu là 'nhan_vien', không được đọc tài liệu có cấp độ 'lanh_dao'

        ket_qua = super().tim_kiem(cau_hoi, top_k=5, rerank=True)
        if not ket_qua:
            return "Không tìm thấy thông tin."

        trich_doan = []
        for kq in ket_qua:
            meta = kq.get("metadata", {})
            muc_do_bao_mat = meta.get("bao_mat", "cong_khai")

            # Phân quyền thô sơ (Identity-Aware)
            if muc_do_bao_mat == "lanh_dao" and self._current_user_role != "lanh_dao":
                continue # Bỏ qua tài liệu này vì không đủ quyền

            nd = meta.get("noi_dung", kq.get("noi_dung", ""))
            trich_doan.append(nd)

        if not trich_doan:
            return "Truy cập bị từ chối hoặc không có tài liệu phù hợp với quyền hạn của bạn."

        return "\n---\n".join(trich_doan[:3])

    def _handle_new_document(self, payload: Dict[str, Any]):
        """Callback chạy ngầm mỗi khi có file mới được thả vào hệ thống."""
        ma = payload.get("ma")
        noi_dung = payload.get("noi_dung")
        if ma and noi_dung:
            print(f"[AgenticRAG] Đang đồng bộ hóa tài liệu mới '{ma}' vào VectorDB...")
            so_chunks = self.them_tai_lieu(ma, noi_dung)
            print(f"[AgenticRAG] Đã đồng bộ xong! Thêm {so_chunks} chunks.")

    def hoi(self, cau_hoi: str, user_role: str = "khach", **kwargs) -> Dict[str, Any]:
        """
        Thay vì làm theo hardcoded flow (Vector -> Rerank -> Generate),
        hệ thống ném câu hỏi cho Agent tự lập kế hoạch.
        """
        self._current_user_role = user_role
        print(f"[AgenticRAG] Agent đang tư duy về câu hỏi: '{cau_hoi}' (Role: {user_role})...")
        tra_loi_goc = self.agent.chay(cau_hoi)

        # BẢN VÁ BẢO MẬT: DLP (Data Loss Prevention)
        # Xóa toàn bộ PII (CMND, Số thẻ, SĐT) trước khi trả cho người dùng
        tra_loi = DataSanitizer.lam_sach(tra_loi_goc)
        if tra_loi != tra_loi_goc:
            print("[AgenticRAG] CẢNH BÁO: Đã phát hiện và bôi đen dữ liệu nhạy cảm (PII) trong câu trả lời!")

        ket_qua_cuoi = {
            "cau_hoi": cau_hoi,
            "tra_loi": tra_loi,
            "nguon": [], # Agentic RAG tạm thời ẩn nguồn do Agent có thể mix nhiều nguồn
            "so_luong_nguon": 0,
        }

        self._lich_su.append(ket_qua_cuoi)
        return ket_qua_cuoi
