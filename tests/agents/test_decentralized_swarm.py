from vietnamese_ai.agents.decentralized_swarm import P2PSwarmOrchestrator
from vietnamese_ai.agents.swarm import TacTuSwarm
from vietnamese_ai.edge.p2p_network import P2PTracker


class MockNodeEngine:
    def __init__(self, node_id):
        self.node_id = node_id


def test_p2p_swarm_spawn_with_retry():
    tracker = P2PTracker()
    tracker.dang_ky_node("node_1", MockNodeEngine("node_1"), toc_do_du_kien=10.0)
    tracker.dang_ky_node("node_2", MockNodeEngine("node_2"), toc_do_du_kien=5.0)

    def mock_llm(prompt):
        return "Suy nghĩ: Xử lý xong.\nTrả lời: Done"

    agent = TacTuSwarm(ten="Mother", vai_tro="QuanLy", llm=mock_llm)
    orchestrator = P2PSwarmOrchestrator(agent, tracker)

    # Ghi đè hàm _chay_node_tu_xa để giả lập một lỗi (Node Crash) ở lần chạy đầu tiên
    lan_chay = {"count": 0}
    original_chay = orchestrator._chay_node_tu_xa

    def fake_chay_node_tu_xa(node_engine, sub_agent, nhiem_vu):
        lan_chay["count"] += 1
        if lan_chay["count"] == 1:
            raise Exception("Lỗi giả lập: Node bị sập nguồn!")
        return original_chay(node_engine, sub_agent, nhiem_vu)

    orchestrator._chay_node_tu_xa = fake_chay_node_tu_xa

    # Kích hoạt spawn
    tool = agent.cong_cu["spawn_sub_agent"]
    res = tool.chay(nhiem_vu="Viet code Python", ten_sub_agent="ChildAgent", vai_tro="Coder")

    # Gọi hàm chay chính để gom kết quả và kích hoạt retry loop
    # Do hàm chay sẽ trigger LLM làm trọng tài, ta truyền dummy input
    ket_qua_chinh = orchestrator.chay("dummy query")

    # Đã phải gọi _chay_node_tu_xa ít nhất 2 lần do retry
    assert lan_chay["count"] == 2

    # Kết quả cuối cùng phải chứa thông báo hoàn thành từ lần chạy Retry thành công
    assert "Done" in ket_qua_chinh.tin_nhan

    # Phải gọi Spawn Success
    assert "Spawn Success" in res
