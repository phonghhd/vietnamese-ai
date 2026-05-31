from unittest.mock import MagicMock

from vietnamese_ai.mobile.mobile_agent import MobileHybridAgent


class MockLLM:
    def sinh_van_ban(self, prompt, **kwargs):
        return "Kết quả Local từ NPU"


def test_mobile_hybrid_agent_offload():
    llm = MockLLM()
    edge_router = MagicMock()
    edge_router.sinh_van_ban.return_value = "Kết quả Cloud từ EdgeRouter"

    agent = MobileHybridAgent(llm=llm, danh_sach_cong_cu=[], edge_router=edge_router)

    # Giả lập Pin yếu (10%) và không cắm sạc
    agent.power_manager.get_battery_status = MagicMock(return_value=(10, False))

    # Agent phải gọi EdgeRouter thay vì Local LLM
    result = agent._goi_llm("Xin chào")
    assert result == "Kết quả Cloud từ EdgeRouter"
    edge_router.sinh_van_ban.assert_called_once()


def test_mobile_hybrid_agent_local():
    llm = MockLLM()
    edge_router = MagicMock()

    agent = MobileHybridAgent(llm=llm, danh_sach_cong_cu=[], edge_router=edge_router)

    # Giả lập Pin khỏe (80%)
    agent.power_manager.get_battery_status = MagicMock(return_value=(80, False))

    # Agent phải gọi Local LLM
    result = agent._goi_llm("Xin chào")
    assert result == "Kết quả Local từ NPU"
    edge_router.sinh_van_ban.assert_not_called()
