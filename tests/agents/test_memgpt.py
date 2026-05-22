from unittest.mock import MagicMock
from vietnamese_ai.agents.long_term_memory import HethongNhoMemGPT

def test_memgpt_memory_compression():
    mock_vector_store = MagicMock()
    
    # Set max_tokens rất nhỏ để ép nó phải nén ngay
    mem = HethongNhoMemGPT(system_prompt="Bạn là trợ lý AI.", max_core_tokens=10, vector_store=mock_vector_store)
    
    # 5 words (token)
    mem.them_tin_nhan("user", "Đây là câu hỏi một")
    # 5 words (tổng 10)
    mem.them_tin_nhan("assistant", "Đây là câu trả lời")
    
    # Lịch sử hiện tại: System(1) + User(1) + Assistant(1) = 3 tin nhắn
    assert len(mem.lich_su) == 3
    
    # Thêm câu này sẽ vượt 10 tokens -> Kích hoạt nén
    mem.them_tin_nhan("user", "Câu hỏi thứ hai dài hơn một chút")
    
    # Nén: Giữ System prompt, cắt bớt User(1) đẩy vào vector store
    # Lịch sử còn: System + Assistant(1) + User(2) (hoặc tương tự tùy thuật toán chia đôi)
    assert len(mem.lich_su) < 4
    mock_vector_store.add_documents.assert_called_once()
