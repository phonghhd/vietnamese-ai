import pytest
from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.agents.swarm import TacTuSwarm, HeThongSwarm
from vietnamese_ai.agents.moa import MoA
from vietnamese_ai.agents.mcts_planning import LapKeHoachMCTS

class DummyLLM:
    def sinh_van_ban(self, prompt: str, **kwargs) -> str:
        if "Hành động:" not in prompt and "__HANDOFF_TO__:" not in prompt:
            # Mô phỏng phản hồi cho các prompt chung
            if "Đánh giá mức độ" in prompt:
                return "0.8"
            if "Hãy đề xuất 3 hành động" in prompt:
                return "- Hành động A\n- Hành động B\n- Hành động C"
            if "Trạng thái mới sẽ đạt được" in prompt:
                return "Trạng thái mới sau khi hành động."
            if "Proposer" in prompt:
                return "Suy nghĩ: Đã hiểu\nTrả lời: Kết quả tổng hợp MoA."
            if "Hệ thống đã chuyển giao" in prompt:
                return "Suy nghĩ: Đang xử lý\nTrả lời: Tôi là Agent đích."
            
            # Mô phỏng Agent gọi Hand-off
            if "Tôi muốn chuyển giao" in prompt:
                return "Suy nghĩ: Cần chuyển giao\nHành động: chuyen_giao_cho_agent_b\nTham số: {}"
            
            return "Suy nghĩ: Đang xử lý\nTrả lời: Kết quả mặc định."
        return "Trả lời: OK"

def test_he_thong_swarm():
    llm = DummyLLM()
    agent_a = TacTuSwarm(ten="Agent A", vai_tro="Xử lý ban đầu", llm=llm)
    agent_b = TacTuSwarm(ten="Agent B", vai_tro="Xử lý chuyên sâu", llm=llm)
    
    swarm = HeThongSwarm(agent_a)
    swarm.tao_lien_ket(agent_a, agent_b, "Cần xử lý chuyên sâu")
    
    assert "chuyen_giao_cho_agent_b" in agent_a.cong_cu
    
    # Giả lập lệnh hand-off
    agent_a.bo_nho.them_tin_nhan("tool", "__HANDOFF_TO__:Agent B")
    
    # Khởi chạy sẽ kích hoạt hand-off ngay do lịch sử có lệnh handoff
    kq = swarm.chay("Xin chào")
    assert kq.tac_tu_tiep_theo is not None
    assert kq.tac_tu_tiep_theo.ten == "Agent B"

def test_moa():
    llm = DummyLLM()
    p1 = TacTu(llm=llm, danh_sach_cong_cu=[])
    p2 = TacTu(llm=llm, danh_sach_cong_cu=[])
    agg = TacTu(llm=llm, danh_sach_cong_cu=[])
    
    moa = MoA(danh_sach_proposers=[p1, p2], aggregator=agg)
    kq = moa.chay("Tính tổng 1+1")
    assert "Kết quả tổng hợp MoA" in kq

def test_mcts_planning():
    llm = DummyLLM()
    agent = TacTu(llm=llm, danh_sach_cong_cu=[])
    mcts = LapKeHoachMCTS(agent_chinh=agent, so_vong_lap=2)
    
    kq = mcts.chay("Lên kế hoạch du lịch")
    assert kq == "Kết quả mặc định."
