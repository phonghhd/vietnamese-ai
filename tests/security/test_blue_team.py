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
    if os.path.exists(MoiTruongCachLy.CONFIG_FILE):
        os.remove(MoiTruongCachLy.CONFIG_FILE)

    assert not MoiTruongCachLy._kiem_tra_ast("import importlib; importlib.import_module('os')")[0]

    ma_doc_tuy_chinh = "import custom_untrusted_lib\ncustom_untrusted_lib.run()"
    an_toan_1, _ = MoiTruongCachLy._kiem_tra_ast(ma_doc_tuy_chinh)
    assert an_toan_1

    llm = MockDefenderLLM()
    blue_team = BlueTeamAgent(llm=llm)
    blue_team.bo_nho.them_tin_nhan(
        "assistant",
        'Suy nghĩ: Cần chặn custom_untrusted_lib.\nHành động: cap_nhat_sandbox\nTham số: {"module_cam_moi": ["custom_untrusted_lib"], "ham_cam_moi": ["custom_danger_fn"]}\n',
    )

    res = cap_nhat_sandbox(module_cam_moi=["custom_untrusted_lib"], ham_cam_moi=["custom_danger_fn"])

    assert "Cập nhật Sandbox thành công" in res
    assert os.path.exists(MoiTruongCachLy.CONFIG_FILE)

    an_toan_2, ly_do_2 = MoiTruongCachLy._kiem_tra_ast(ma_doc_tuy_chinh)
    assert not an_toan_2
    assert "Không được phép import module 'custom_untrusted_lib'" in ly_do_2

    if os.path.exists(MoiTruongCachLy.CONFIG_FILE):
        os.remove(MoiTruongCachLy.CONFIG_FILE)
