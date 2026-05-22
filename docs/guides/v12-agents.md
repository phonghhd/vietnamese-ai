# Hướng dẫn sử dụng Agentic Swarm (v12.0)

Phiên bản v12.0 mang đến cuộc cách mạng cho `vietnamese_ai` với kiến trúc Multi-Agent hiện đại, cho phép các Agent tự phối hợp, lập kế hoạch và chuyển giao công việc cho nhau.

## 1. Kiến trúc Bầy đàn (Swarm Orchestration)
Bạn có thể cấu hình để các Agent tự động chuyển (Hand-off) công việc cho Agent khác khi gặp chuyên môn phù hợp.

```python
from vietnamese_ai import TacTuSwarm, HeThongSwarm, VietnameseLLM

llm = VietnameseLLM("evonet-8b")

# Khởi tạo các Agents
agent_sales = TacTuSwarm(ten="Sales", vai_tro="Bán hàng", llm=llm)
agent_tech = TacTuSwarm(ten="Tech Support", vai_tro="Hỗ trợ kỹ thuật", llm=llm)

# Cấu hình Swarm
swarm = HeThongSwarm(agent_khoi_tao=agent_sales)
swarm.tao_lien_ket(agent_sales, agent_tech, "Khi khách hỏi về kỹ thuật chuyên sâu")

# Chạy
ket_qua = swarm.chay("Tôi muốn mua gói Pro nhưng API bị lỗi 500, sửa sao?")
# Hệ thống sẽ tự động chuyển từ Sales sang Tech Support
```

## 2. Mixture of Agents (MoA)
Tổng hợp ý tưởng từ nhiều Agent để đưa ra câu trả lời xuất sắc nhất.

```python
from vietnamese_ai import TacTu, MoA

p1 = TacTu(llm=llm_1) # Có thể dùng các model khác nhau
p2 = TacTu(llm=llm_2)
agg = TacTu(llm=llm_main)

moa = MoA(danh_sach_proposers=[p1, p2], aggregator=agg)
kq = moa.chay("Viết một bài luận về AI")
```

## 3. Monte Carlo Tree Search (MCTS) Planning
Sử dụng MCTS để Agent tự suy nghĩ (o1-like reasoning).

```python
from vietnamese_ai import LapKeHoachMCTS

mcts = LapKeHoachMCTS(agent_chinh=agent_sales, so_vong_lap=10)
kq = mcts.chay("Lên kế hoạch marketing 3 tháng với ngân sách 10 triệu")
```
