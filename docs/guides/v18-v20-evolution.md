# Hướng dẫn V-Neural v20.0 (Evolution)

Tài liệu này hướng dẫn bạn cách sử dụng các tính năng mới nhất từ **v18.0 đến v20.0**, tập trung vào hệ thống Tác tử (Agents), mạng lưới DePIN (Edge AI) và Agentic RAG.

## 1. Tính năng Human-in-the-Loop & Experience Memory (v18.0)

Để đảm bảo an toàn tuyệt đối cho Tác tử khi tự chủ chạy các công cụ nhạy cảm:

```python
from vietnamese_ai.agents import TacTu, cong_cu, SoTayKinhNghiem

# Bật cờ yeu_cau_xac_nhan
@cong_cu(ten="xoa_du_lieu", yeu_cau_xac_nhan=True)
def xoa_du_lieu(bang_ten):
    return f"Đã xóa bảng {bang_ten}"

# Thiết lập Sổ tay kinh nghiệm (Agent Memory 2.0)
so_tay = SoTayKinhNghiem()

# Hàm callback để con người kiểm duyệt
def kiem_duyet(hanh_dong, tham_so):
    return input(f"Cho phép {hanh_dong} chạy? (y/n)") == 'y'

# Khởi tạo
agent = TacTu(
    llm=my_llm,
    danh_sach_cong_cu=[xoa_du_lieu],
    ham_xac_nhan=kiem_duyet,
    so_tay_kinh_nghiem=so_tay
)
```

## 2. Web3 DePIN & Hybrid Execution (v19.0)

Kiến trúc "Chạy đua song song" (Hybrid Speculative Execution) kết hợp mạng P2P Edge và Cloud Data Center.

```python
from vietnamese_ai.edge import P2PTracker, TokenLedger, EdgeRouter, SecureEdgeNode

# Đăng ký Sổ cái và Mạng lưới
tracker = P2PTracker()
ledger = TokenLedger()

# Khách hàng bật máy tính lên để đóng góp vào mạng DePIN
node_cua_khach_hang = SecureEdgeNode("node_vip", private_key="zxcv", auto_start=True)
tracker.dang_ky_node("node_vip", engine=node_cua_khach_hang, toc_do_du_kien=15.0)

# Khởi tạo Edge Router (Nằm ở Data Center của công ty)
router = EdgeRouter(
    p2p_tracker=tracker, 
    token_ledger=ledger, 
    cloud_api_key="your_api_key"
)

# Phát lệnh chạy đua: Ai xong trước lấy kết quả người đó!
ket_qua = router.sinh_van_ban_song_song("Giải phương trình bậc 2")
```

## 3. Real-time Agentic RAG (v20.0)

RAG giờ đây không chỉ tìm kiếm, mà còn "Suy luận đa bước" và "Đồng bộ tự động".

```python
from vietnamese_ai.rag import AgenticRAGPipeline, DocumentWatcher
import time

# 1. Bật tự động đồng bộ (Chạy ngầm)
watcher = DocumentWatcher("thu_muc_ho_so/")
# (Bất cứ khi nào thả file vào thu_muc_ho_so, RAG sẽ tự cập nhật)

# 2. Khởi tạo Agentic RAG
pipeline = AgenticRAGPipeline(llm_engine=my_llm)

# 3. Agent tự vạch kế hoạch tìm kiếm
ket_qua = pipeline.hoi("Phân tích cho tôi sự khác nhau giữa tài liệu A và B")
print(ket_qua["tra_loi"])
```

> **Lưu ý:** Agentic RAG yêu cầu LLM có khả năng suy luận (Reasoning) tốt, ưu tiên các model >7B parameter hoặc GPT-4/Claude-3.
