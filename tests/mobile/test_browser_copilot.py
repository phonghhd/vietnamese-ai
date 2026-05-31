from unittest.mock import MagicMock

from vietnamese_ai.mobile.browser_copilot import BrowserCopilot
from vietnamese_ai.mobile.mobile_agent import MobileHybridAgent


def test_nen_ngu_canh():
    agent = MagicMock(spec=MobileHybridAgent)
    copilot = BrowserCopilot(mobile_agent=agent)

    van_ban_goc = "Đây là    một đoạn   văn bản. Nó thì dài và ráng viết thêm để test. " * 50
    van_ban_nen = copilot.nen_ngu_canh(van_ban_goc)

    assert " thì " not in van_ban_nen
    assert " ráng " not in van_ban_nen
    assert len(van_ban_nen) <= 1015  # 1000 ký tự + "...(đã nén)"

def test_browser_copilot_tom_tat():
    agent = MagicMock(spec=MobileHybridAgent)
    agent.chay.return_value = "Tóm tắt: Đây là báo cáo tài chính."

    copilot = BrowserCopilot(mobile_agent=agent)
    ket_qua = copilot.doc_va_tom_tat("https://example.com/bao-cao")

    assert "Tóm tắt: Đây là báo cáo tài chính." in ket_qua
    agent.chay.assert_called_once()


def test_browser_copilot_hoi_dap_bao_mat():
    agent = MagicMock(spec=MobileHybridAgent)
    agent.chay.return_value = "Lợi nhuận tăng 20%."

    copilot = BrowserCopilot(mobile_agent=agent)

    # Quyền hợp lệ
    ket_qua_ok = copilot.hoi_dap_tai_lieu(
        url="https://example.com/bao-cao",
        cau_hoi="Lợi nhuận thế nào?",
        user_roles=["premium_user"]
    )
    assert "Lợi nhuận tăng 20%" in ket_qua_ok

    # Quyền không hợp lệ (không có 'premium_user' hay 'admin')
    ket_qua_loi = copilot.hoi_dap_tai_lieu(
        url="https://example.com/bao-cao",
        cau_hoi="Lợi nhuận thế nào?",
        user_roles=["guest"]
    )
    assert "Lỗi Bảo mật" in ket_qua_loi
