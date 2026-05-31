from vietnamese_ai.serving.speculative import SpeculativeEngine


class MockDraftModel:
    def sinh_van_ban(self, prompt: str, do_dai: int = 4) -> str:
        return " draft guess"


class MockTargetModel:
    def sinh_van_ban(self, prompt: str, do_dai: int = 1) -> str:
        return " target_ok"


def test_speculative_decoding():
    draft = MockDraftModel()
    target = MockTargetModel()

    engine = SpeculativeEngine(target_model=target, draft_model=draft, gamma=2)

    # max_tokens = 6
    # Mỗi vòng tạo ra: "draft guess target_ok" (khoảng 3 words)
    # 2 vòng là đủ > 6 tokens
    kq = engine.sinh_van_ban("Bắt đầu", max_tokens=6)

    assert "draft guess target_ok" in kq
    assert len(kq.split()) >= 5
