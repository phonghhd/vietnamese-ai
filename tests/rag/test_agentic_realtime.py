import time
from unittest.mock import patch

from vietnamese_ai.rag import AgenticRAGPipeline, DocumentWatcher


def test_realtime_sync():
    # Khởi tạo mock LLM (để pipeline không báo lỗi)
    class MockLLM:
        def sinh_van_ban(self, prompt, **kwargs):
            return "Mock"

    # Xóa singleton RAGEventBus để đảm bảo môi trường sạch
    from vietnamese_ai.rag.realtime_sync import RAGEventBus

    RAGEventBus._instance = None

    pipeline = AgenticRAGPipeline(llm_engine=MockLLM())
    so_luong_chunks_truoc = pipeline.csdl_vector.so_luong()

    # Kích hoạt watcher
    watcher = DocumentWatcher("dummy_dir")

    # Giả lập Thread.start() thành hàm run() để chạy đồng bộ (synchronous)
    with patch(
        "threading.Thread.start", new=lambda self: self._target(*self._args, **self._kwargs)
    ):
        # Bắn sự kiện có file mới (giả lập file có nhiều hơn 20 từ)
        text = "Đây là dòng 1. " * 10 + "Đây là dòng 2. " * 10
        watcher.phat_hien_file_moi("doc_1", text)

    # Chờ thread ngầm chạy xong
    time.sleep(0.5)

    so_luong_chunks_sau = pipeline.csdl_vector.so_luong()

    # Số lượng chunks trong CSDL Vector phải tăng lên
    assert so_luong_chunks_sau > so_luong_chunks_truoc


def test_agentic_rag():
    # Mock LLM giả vờ dùng tool
    class AgentLLM:
        def sinh_van_ban(self, prompt, **kwargs):
            if "tim_kiem_tai_lieu_chinh" not in prompt:
                return 'Suy nghĩ: Cần tìm kiếm\nHành động: tim_kiem_tai_lieu_chinh\nTham số: {"cau_hoi": "thông tin"}'
            else:
                return "Suy nghĩ: Đã đủ.\nTrả lời: Câu trả lời cuối cùng."

    pipeline = AgenticRAGPipeline(llm_engine=AgentLLM())

    # Ném câu hỏi, Agent sẽ dùng tool và trả lời
    ket_qua = pipeline.hoi("Tôi muốn biết thông tin")

    assert "Câu trả lời cuối cùng" in ket_qua["tra_loi"]
