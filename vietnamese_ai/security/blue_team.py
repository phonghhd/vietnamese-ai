import json
from typing import Any, List

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.agents.tools import CongCu
from vietnamese_ai.security.agent_sandbox import MoiTruongCachLy


def cap_nhat_sandbox(module_cam_moi: List[str] = None, ham_cam_moi: List[str] = None) -> str:
    """Cập nhật luật bảo mật vào file sandbox_rules.json."""
    cau_hinh = MoiTruongCachLy._tai_cau_hinh()

    if module_cam_moi:
        cau_hinh["module_cam"].extend(module_cam_moi)
        cau_hinh["module_cam"] = list(set(cau_hinh["module_cam"]))

    if ham_cam_moi:
        cau_hinh["ham_cam"].extend(ham_cam_moi)
        cau_hinh["ham_cam"] = list(set(cau_hinh["ham_cam"]))

    try:
        with open(MoiTruongCachLy.CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cau_hinh, f, ensure_ascii=False, indent=4)
        return f"Cập nhật Sandbox thành công. Hiện có {len(cau_hinh['module_cam'])} module cấm và {len(cau_hinh['ham_cam'])} hàm cấm."
    except Exception as e:
        return f"Lỗi ghi cấu hình: {str(e)}"


cong_cu_cap_nhat_sandbox = CongCu(
    ten="cap_nhat_sandbox",
    mo_ta="Thêm module và hàm cấm vào bộ lọc của Sandbox để chống lại mã độc.",
    ham_thuc_thi=cap_nhat_sandbox,
)


class BlueTeamAgent(TacTu):
    """
    Tác tử chuyên dụng cho phòng thủ tự động (Blue Teaming & Self-Healing).
    Phân tích báo cáo tấn công từ Red Team và tự động tiêm luật vá lỗi vào Sandbox.
    """

    def __init__(self, llm: Any, max_iterations: int = 5):
        super().__init__(
            llm=llm, danh_sach_cong_cu=[cong_cu_cap_nhat_sandbox], max_iterations=max_iterations
        )

        # Override System Prompt để ép LLM thành Defender
        defender_prompt = (
            "Bạn là một Kỹ sư Bảo mật Phòng thủ (Blue Team Security Engineer). "
            "Bạn có nhiệm vụ phân tích các đoạn mã tấn công (payload) do Hacker (Red Team) cung cấp. "
            "Nếu đoạn mã đó có thể bypass Sandbox hiện tại bằng cách gọi một module hoặc hàm đặc biệt "
            "(ví dụ: importlib, builtins, getattr), bạn phải DÙNG CÔNG CỤ 'cap_nhat_sandbox' "
            "để cấm (block) ngay lập tức module hoặc hàm đó.\n"
            "Hãy đảm bảo hệ thống an toàn tuyệt đối."
        )
        self.bo_nho.system_prompt += f"\n\n{defender_prompt}"

    def xu_ly_canh_bao(self, ma_doc: str, li_do_thanh_cong: str) -> str:
        """Kích hoạt tác tử tự động vá lỗi dựa trên mã độc gửi đến."""
        return self.chay(
            f"CẢNH BÁO: Hệ thống vừa bị tấn công!\n"
            f"- Mã độc:\n```python\n{ma_doc}\n```\n"
            f"- Phương thức bypass (dự đoán): {li_do_thanh_cong}\n\n"
            "Hãy phân tích mã này và cập nhật Sandbox ngay lập tức."
        )
