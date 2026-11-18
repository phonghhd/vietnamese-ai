"""Tests tích hợp SALM (SelfRefine, SelfConsistency) vào hệ sinh thái TacTu và Agents."""

import pytest

from vietnamese_ai.agents import TacTu, cong_cu
from vietnamese_ai.agents.code_healing import TacTuLapTrinhTuSua
from vietnamese_ai.agents.self_healing import SelfHealingAgent, TacTuTuChuaLanh
from vietnamese_ai.salm import SelfConsistency, SelfRefine


class MockSamplingLLM:
    """Mock LLM trả về các phản hồi khác nhau theo kịch bản."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def sinh_van_ban(self, prompt: str, **kwargs) -> str:
        if self.index < len(self.responses):
            res = self.responses[self.index]
            self.index += 1
            return res
        return self.responses[-1] if self.responses else "Trả lời: Mặc định"


class TestTacTuSelfConsistency:
    def test_tac_tu_voi_self_consistency_action_voting(self):
        """Kiểm tra SelfConsistency biểu quyết đa số để chọn hành động đúng."""
        @cong_cu(ten="phep_cong", mo_ta="Cộng hai số")
        def phep_cong(a: int, b: int) -> int:
            return a + b

        # 3 paths: 2 paths chọn phep_cong(a=2, b=3), 1 path chọn phep_cong(a=99, b=99)
        mock_paths = [
            'Suy nghĩ: Cần cộng 2 và 3.\nHành động: phep_cong\nTham số: {"a": 2, "b": 3}',
            'Suy nghĩ: Cộng 2 với 3 cho đúng.\nHành động: phep_cong\nTham số: {"a": 2, "b": 3}',
            'Suy nghĩ: Đi lạc hướng.\nHành động: phep_cong\nTham số: {"a": 99, "b": 99}',
            # Sau khi có Quan sát: 5
            "Suy nghĩ: Đã có kết quả là 5.\nTrả lời: Tổng là 5.",
            "Suy nghĩ: Đã có kết quả là 5.\nTrả lời: Tổng là 5.",
            "Suy nghĩ: Xong.\nTrả lời: Tổng là 5.",
        ]
        mock_llm = MockSamplingLLM(mock_paths)

        tac_tu = TacTu(
            llm=mock_llm,
            danh_sach_cong_cu=[phep_cong],
            dung_self_consistency=True,
            so_paths_consistency=3,
        )

        ket_qua = tac_tu.chay("Tính 2 + 3")
        assert "5" in ket_qua
        assert len(tac_tu.lich_su_consistency) >= 1
        # Kiểm tra voting đã diễn ra
        sc_first = tac_tu.lich_su_consistency[0]
        assert sc_first["so_luong_paths"] == 3
        assert sc_first["ty_le_dong_nhat"] >= 0.66

    def test_tac_tu_voi_self_consistency_answer_voting(self):
        """Kiểm tra SelfConsistency biểu quyết đa số cho câu trả lời trực tiếp."""
        # 3 paths trả lời trực tiếp: 2 chọn Hà Nội, 1 chọn Đà Nẵng
        mock_paths = [
            "Suy nghĩ: Thủ đô Việt Nam là Hà Nội.\nTrả lời: Hà Nội",
            "Suy nghĩ: Tôi chắc chắn là Hà Nội.\nTrả lời: Hà Nội",
            "Suy nghĩ: Có thể là Đà Nẵng.\nTrả lời: Đà Nẵng",
        ]
        mock_llm = MockSamplingLLM(mock_paths)

        custom_sc = SelfConsistency(ham_sinh=mock_llm.sinh_van_ban, so_luong=3)
        tac_tu = TacTu(
            llm=mock_llm,
            dung_self_consistency=True,
            bo_consistency=custom_sc,
        )

        ket_qua = tac_tu.chay("Thủ đô của Việt Nam là gì?")
        assert ket_qua == "Hà Nội"
        assert len(tac_tu.lich_su_consistency) == 1
        assert tac_tu.lich_su_consistency[0]["ty_le_dong_nhat"] == pytest.approx(0.667, abs=0.01)


class TestTacTuSelfRefine:
    def test_tac_tu_voi_self_refine_nang_cap_cau_tra_loi(self):
        """Kiểm tra SelfRefine tự động đánh giá và tinh chỉnh câu trả lời cuối."""
        cac_phan_hoi = [
            # Lần gọi ReAct ban đầu: câu trả lời ngắn
            "Suy nghĩ: Xong.\nTrả lời: AI là trí tuệ nhân tạo.",
            # Lần refine 1: mở rộng và chuẩn hóa
            "Trí tuệ nhân tạo (AI) là lĩnh vực khoa học mô phỏng khả năng tư duy và học hỏi của con người nhằm giải quyết các bài toán phức tạp.",
        ]
        mock_llm = MockSamplingLLM(cac_phan_hoi)

        def mock_danh_gia(prompt: str, output: str):
            # Điểm thấp nếu < 10 từ, điểm cao nếu giải thích chi tiết
            so_tu = len(output.split())
            if so_tu < 10:
                return {"diem": 0.4, "chi_tiet": {"so_tu": so_tu}}
            return {"diem": 0.95, "chi_tiet": {"so_tu": so_tu}}

        bo_refine = SelfRefine(
            ham_sinh=mock_llm.sinh_van_ban,
            ham_danh_gia=mock_danh_gia,
            so_vong_toi_da=2,
            nguong_chat_luong=0.9,
        )

        tac_tu = TacTu(
            llm=mock_llm,
            dung_self_refine=True,
            bo_refine=bo_refine,
        )

        ket_qua = tac_tu.chay("Định nghĩa AI là gì?")
        assert "Trí tuệ nhân tạo" in ket_qua
        assert len(tac_tu.lich_su_refine) == 1
        assert tac_tu.lich_su_refine[0]["dat_nguong"] is True

    def test_tinh_chinh_cau_tra_loi_truc_tiep(self):
        """Kiểm tra phương thức tinh_chinh_cau_tra_loi chủ động."""
        mock_llm = MockSamplingLLM(["Bản tinh chỉnh hoàn thiện hơn rất nhiều."])

        tac_tu = TacTu(llm=mock_llm)
        kq = tac_tu.tinh_chinh_cau_tra_loi("Bản nháp ban đầu", yeu_cau="Viết báo cáo")
        assert kq == "Bản tinh chỉnh hoàn thiện hơn rất nhiều."
        assert len(tac_tu.lich_su_refine) == 1


class TestSelfHealingWithSALM:
    def test_self_healing_agent_alias_va_salm(self):
        """Kiểm tra TacTuTuChuaLanh và tu_sua_loi_voi_refine."""
        assert TacTuTuChuaLanh is SelfHealingAgent

        mock_llm = MockSamplingLLM(["Phương án sửa lỗi tối ưu: Thay đổi tham số chia tránh số 0."])
        agent = TacTuTuChuaLanh(
            llm=mock_llm,
            danh_sach_cong_cu=[],
            dung_self_refine=True,
        )

        ket_qua_sua = agent.tu_sua_loi_voi_refine(
            ma_loi="ZeroDivisionError: division by zero",
            yeu_cau="Tính toán an toàn",
        )
        assert "Phương án sửa lỗi tối ưu" in ket_qua_sua
        assert len(agent.lich_su_refine) == 1


class TestCodeHealingWithSALM:
    def test_tac_tu_lap_trinh_tu_sua_voi_refine(self):
        """Kiểm tra TacTuLapTrinhTuSua tích hợp dung_self_refine khi sửa lỗi."""
        code_loi = (
            "def chia_an_toan(a, b):\n"
            "    return a / 0\n"
        )
        code_sua = (
            "def chia_an_toan(a, b):\n"
            "    return a / b if b != 0 else None\n"
        )

        def mock_sinh_code(prompt: str) -> str:
            if "Sửa lỗi" in prompt or "Tối ưu" in prompt:
                return code_sua
            return code_loi

        agent = TacTuLapTrinhTuSua(
            so_lan_thu_lai_toi_da=2,
            sinh_ma_fn=mock_sinh_code,
            dung_self_refine=True,
            so_vong_refine=1,
        )

        res = agent.lap_trinh_tu_sua("chia_an_toan")
        assert res["thanh_cong"] is True
        assert "b != 0" in res["ma_nguon"]
