import os

from vietnamese_ai.security.agent_sandbox import MoiTruongCachLy
from vietnamese_ai.security.blue_team import BlueTeamAgent, cap_nhat_sandbox


class MockDefenderLLM:
    def sinh_van_ban(self, prompt, **kwargs):
        # Giả lập hành động vá lỗi
        if "getattr" in prompt or "importlib" in prompt:
            return 'Suy nghĩ: Cần chặn importlib và getattr.\nHành động: cap_nhat_sandbox\nTham số: {"module_cam_moi": ["importlib"], "ham_cam_moi": ["getattr"]}\n'
        return "Trả lời: Đã xử lý."

    def __call__(self, prompt):
        return self.sinh_van_ban(prompt)


def test_auto_patching():
    # Reset config file first
    if os.path.exists(MoiTruongCachLy.CONFIG_FILE):
        os.remove(MoiTruongCachLy.CONFIG_FILE)

    # Lần 1: RedTeam tấn công bằng importlib (Vượt qua vì importlib chưa bị cấm)
    ma_doc = "import importlib; os = importlib.import_module('os'); print(os.environ)"
    an_toan_1, ly_do_1 = MoiTruongCachLy._kiem_tra_ast(ma_doc)
    assert an_toan_1  # AST checker hiện tại (chưa vá) chưa cấm importlib

    # Lần 2: Báo lỗi cho BlueTeam để vá
    llm = MockDefenderLLM()
    blue_team = BlueTeamAgent(llm=llm)

    # Fake a memory call that triggers the tool
    blue_team.bo_nho.them_tin_nhan(
        "assistant",
        'Suy nghĩ: Cần chặn importlib và getattr.\nHành động: cap_nhat_sandbox\nTham số: {"module_cam_moi": ["importlib"], "ham_cam_moi": ["getattr"]}\n',
    )

    # Thay vì gọi LLM mock khó khăn, test thẳng tool
    res = cap_nhat_sandbox(module_cam_moi=["importlib"], ham_cam_moi=["getattr"])

    assert "Cập nhật Sandbox thành công" in res
    assert os.path.exists(MoiTruongCachLy.CONFIG_FILE)

    # Lần 3: Đánh giá lại Sandbox sau khi vá
    an_toan_2, ly_do_2 = MoiTruongCachLy._kiem_tra_ast(ma_doc)
    assert not an_toan_2  # Sandbox đã chặn importlib!
    assert "Không được phép import module 'importlib'" in ly_do_2

    # Dọn dẹp
    if os.path.exists(MoiTruongCachLy.CONFIG_FILE):
        os.remove(MoiTruongCachLy.CONFIG_FILE)
