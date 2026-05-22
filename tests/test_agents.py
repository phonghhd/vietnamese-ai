from vietnamese_ai.agents import BoNhoTacTu, HeThongDaTacTu, TacTu, cong_cu


def test_cong_cu_decorator():
    @cong_cu(ten="tinh_tong", mo_ta="Cộng hai số")
    def tinh_tong(a: int, b: int) -> int:
        return a + b

    assert tinh_tong.ten == "tinh_tong"
    assert tinh_tong.mo_ta == "Cộng hai số"
    assert "a" in tinh_tong.tham_so
    assert "b" in tinh_tong.tham_so
    assert tinh_tong.chay(a=5, b=10) == 15

def test_bo_nho_tac_tu():
    bo_nho = BoNhoTacTu(system_prompt="Bạn là một trợ lý.")
    assert len(bo_nho.lay_lich_su()) == 1
    assert bo_nho.lay_lich_su()[0]["role"] == "system"

    bo_nho.them_tin_nhan("user", "Xin chào")
    assert len(bo_nho.lay_lich_su()) == 2
    assert bo_nho.lay_lich_su()[1]["role"] == "user"

    bo_nho.them_tin_nhan("tool", "Kết quả là 5", ten_cong_cu="may_tinh")
    assert bo_nho.lay_lich_su()[2]["role"] == "tool"
    assert bo_nho.lay_lich_su()[2]["name"] == "may_tinh"

class MockLLM:
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    def sinh_van_ban(self, prompt, **kwargs):
        res = self.responses[self.index]
        self.index += 1
        return res

def test_tac_tu_react():
    @cong_cu(ten="nhan_doi", mo_ta="Nhân đôi một số")
    def nhan_doi(x: int) -> int:
        return x * 2

    # Giả lập 2 lượt suy luận của LLM
    mock_responses = [
        "Suy nghĩ: Cần gọi công cụ nhan_doi với x=5.\nHành động: nhan_doi\nTham số: {\"x\": 5}",
        "Suy nghĩ: Tôi đã có kết quả là 10.\nTrả lời: Kết quả là 10."
    ]
    mock_llm = MockLLM(mock_responses)

    tac_tu = TacTu(llm=mock_llm, danh_sach_cong_cu=[nhan_doi])

    ket_qua = tac_tu.chay("Nhân đôi số 5")
    assert ket_qua == "Kết quả là 10."

    lich_su = tac_tu.bo_nho.lay_lich_su()
    assert len(lich_su) > 1
    # Check if tool observation is added
    has_tool = any(msg.get("role") == "tool" and "10" in msg.get("content", "") for msg in lich_su)
    assert has_tool

def test_he_thong_da_tac_tu():
    class EchoAgent:
        def chay(self, text):
            return f"Echo: {text}"

    agents = {
        "AgentA": EchoAgent(),
        "AgentB": EchoAgent()
    }

    he_thong = HeThongDaTacTu(agents, loai_dieu_phoi="sequential")
    ket_qua = he_thong.chay("Hello")
    assert "Echo: Yêu cầu hiện tại cho AgentB:" in ket_qua
    assert "Echo: Yêu cầu hiện tại cho AgentA:" in ket_qua
    assert "Hello" in ket_qua
