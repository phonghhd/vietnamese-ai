from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.agents.experience_memory import SoTayKinhNghiem
from vietnamese_ai.agents.tools import cong_cu


def test_hitl_agent_rejected():
    # Mock LLM luôn trả về lệnh gọi công cụ xoa_database
    class MockLLM:
        def sinh_van_ban(self, prompt, **kwargs):
            if "Lỗi: Con người đã TỪ CHỐI" in prompt:
                return "Suy nghĩ: Con người đã từ chối. Mình nên dừng lại.\nTrả lời: Xin lỗi, tôi không thể thực hiện hành động này."
            return 'Suy nghĩ: Mình cần xóa DB.\nHành động: xoa_database\nTham số: {"bang": "users"}'

    # Công cụ nguy hiểm yêu cầu xác nhận
    @cong_cu(ten="xoa_database", mo_ta="Xóa toàn bộ database", yeu_cau_xac_nhan=True)
    def xoa_database(bang: str):
        return "Đã xóa!"

    # Hàm giả lập con người luôn TỪ CHỐI
    def ham_tu_choi(hanh_dong, tham_so):
        return False

    agent = TacTu(llm=MockLLM(), danh_sach_cong_cu=[xoa_database], ham_xac_nhan=ham_tu_choi)

    ket_qua = agent.chay("Xóa DB đi")
    assert "không thể thực hiện" in ket_qua.lower()


def test_experience_memory():
    so_tay = SoTayKinhNghiem(vector_store=False)
    so_tay.ghi_nhan_bai_hoc(
        ngu_canh="Khi người dùng nhờ xóa DB",
        sai_lam="Dùng lệnh xoa_database",
        cach_khac_phuc="Phải hỏi ý kiến cấp trên trước",
    )

    bai_hoc = so_tay.truy_xuat_kinh_nghiem("xóa db")
    assert bai_hoc is not None
    assert "Phải hỏi ý kiến cấp trên" in bai_hoc

    # Test Agent có đọc sổ tay
    class MockLLM:
        def sinh_van_ban(self, prompt, **kwargs):
            if "BÀI HỌC KINH NGHIỆM" in prompt and "Phải hỏi ý kiến cấp trên" in prompt:
                return "Suy nghĩ: Mình nhớ bài học rồi.\nTrả lời: Tôi cần hỏi ý kiến cấp trên trước khi làm."
            return "Suy nghĩ: ...\nTrả lời: Đã rõ."

    agent = TacTu(llm=MockLLM(), danh_sach_cong_cu=[], so_tay_kinh_nghiem=so_tay)

    ket_qua = agent.chay("Nhờ bạn xóa DB nha")
    assert "hỏi ý kiến cấp trên" in ket_qua.lower()
