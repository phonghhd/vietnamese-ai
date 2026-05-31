from vietnamese_ai.agents.self_healing import SelfHealingAgent
from vietnamese_ai.agents.tools import cong_cu


def test_self_healing_agent():
    @cong_cu(ten="tinh_chia", mo_ta="Tính a / b")
    def tinh_chia(a: int, b: int):
        return a / b

    # LLM giả lập sẽ cố tình chia cho 0 ở lượt đầu, sau đó sửa lại
    class MockLLM:
        def __init__(self):
            self.luot = 0

        def sinh_van_ban(self, prompt, **kwargs):
            print("==== PROMPT ====")
            print(prompt)
            print("=================")
            self.luot += 1
            if self.luot == 1:
                return 'Suy nghĩ: Để xem có lỗi không.\nHành động: tinh_chia\nTham số: {"a": 10, "b": 0}'

            if "5.0" in prompt:
                return "Suy nghĩ: Xong rồi.\nTrả lời: Kết quả là 5.0"

            if "ZeroDivisionError" in prompt or "LỖI THỰC THI" in prompt:
                return 'Suy nghĩ: Lỗi chia cho 0. Phải đổi b thành 2.\nHành động: tinh_chia\nTham số: {"a": 10, "b": 2}'

            return "Suy nghĩ: Lỗi rồi.\nTrả lời: Không làm được."

    agent = SelfHealingAgent(llm=MockLLM(), danh_sach_cong_cu=[tinh_chia])
    ket_qua = agent.chay_voi_tu_chua_lanh("Chia 10 cho 0 thử xem")

    assert "5.0" in ket_qua
    assert agent.llm.luot >= 3  # Ít nhất 3 lượt: Gọi lỗi -> Sửa lỗi -> Trả lời

def test_multi_modal_prompt():
    class MockLLM:
        def sinh_van_ban(self, prompt, **kwargs):
            if "[IMAGE:" in prompt:
                return "Suy nghĩ: Bức ảnh đẹp.\nTrả lời: Đây là ảnh."
            return "Trả lời: Text"

    agent = SelfHealingAgent(llm=MockLLM(), danh_sach_cong_cu=[])

    multi_modal_prompt = [
        {"type": "text", "text": "Mô tả ảnh này"},
        {"type": "image_url", "image_url": "[IMAGE: dog.jpg]"}
    ]

    ket_qua = agent.chay_voi_tu_chua_lanh(multi_modal_prompt)
    assert "ảnh" in ket_qua.lower()
