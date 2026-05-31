from typing import Any, List

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.agents.tools import CongCu


class SelfHealingAgent(TacTu):
    """
    Tác tử Đa phương thức & Tự chữa lành (Multi-Modal & Self-Healing Agent).
    Có khả năng:
    1. Bắt lỗi khi gọi công cụ (Tool Execution Error).
    2. Tự nạp Error Traceback vào bộ nhớ.
    3. Yêu cầu LLM suy nghĩ lại (Self-Reflection) và thử lại.
    """

    def __init__(
        self,
        llm: Any,
        danh_sach_cong_cu: List[CongCu],
        max_iterations: int = 10,
        so_lan_thu_lai_toi_da: int = 3,
    ):
        super().__init__(llm=llm, danh_sach_cong_cu=danh_sach_cong_cu, max_iterations=max_iterations)
        self.so_lan_thu_lai_toi_da = so_lan_thu_lai_toi_da

        # Thêm hướng dẫn Self-Healing vào System Prompt
        healing_prompt = (
            "\nCHÚ Ý (Self-Healing): Nếu bạn nhận được thông báo lỗi (Exception) từ hệ thống, "
            "hãy đọc kỹ Error Traceback, giải thích nguyên nhân lỗi và tạo ra một lệnh gọi "
            "công cụ mới với các tham số đã được sửa chữa."
        )
        self.bo_nho.system_prompt += healing_prompt

    def _thuc_thi_cong_cu(self, ten_cong_cu: str, cc: CongCu, tham_so: dict) -> str:
        """
        Ghi đè hàm thực thi công cụ để tự động sửa lỗi (Self-Healing).
        """
        ket_qua = super()._thuc_thi_cong_cu(ten_cong_cu, cc, tham_so)

        # Công cụ bắt lỗi và trả về chuỗi bắt đầu bằng "Lỗi khi chạy công cụ"
        if isinstance(ket_qua, str) and ket_qua.startswith("Lỗi khi chạy công cụ"):
            loi_phan_hoi = (
                f"LỖI THỰC THI (EXCEPTION):\n"
                f"{ket_qua}\n"
                f"Yêu cầu: Hãy phân tích lỗi trên, sửa lại tham số và gọi lại công cụ."
            )
            return loi_phan_hoi

        return ket_qua

    def chay_voi_tu_chua_lanh(self, prompt: str) -> str:
        """
        Chạy tác tử với luồng vòng lặp tự chữa lành bổ sung.
        """
        # Hỗ trợ Multi-Modal bằng cách nhận cả List[dict] (Image/Audio)
        # Trong ví dụ này ta ép kiểu về text nếu là list
        if isinstance(prompt, list):
            prompt_text = " ".join([str(item) for item in prompt])
        else:
            prompt_text = prompt

        return self.chay(prompt_text)
