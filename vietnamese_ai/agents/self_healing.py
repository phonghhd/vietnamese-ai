from typing import Any, Callable, Dict, List, Optional

from vietnamese_ai.agents.agent import TacTu
from vietnamese_ai.agents.tools import CongCu


class SelfHealingAgent(TacTu):
    """
    Tác tử Đa phương thức & Tự chữa lành (Multi-Modal & Self-Healing Agent).
    Có khả năng:
    1. Bắt lỗi khi gọi công cụ (Tool Execution Error).
    2. Tự nạp Error Traceback vào bộ nhớ.
    3. Yêu cầu LLM suy nghĩ lại (Self-Reflection) và thử lại.
    4. Tích hợp SelfRefine & SelfConsistency để tối ưu hóa quá trình sửa lỗi.
    """

    def __init__(
        self,
        llm: Any,
        danh_sach_cong_cu: List[CongCu],
        max_iterations: int = 10,
        so_lan_thu_lai_toi_da: int = 3,
        ham_xac_nhan: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        dung_self_refine: bool = False,
        bo_refine: Optional[Any] = None,
        dung_self_consistency: bool = False,
        bo_consistency: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(
            llm=llm,
            danh_sach_cong_cu=danh_sach_cong_cu,
            max_iterations=max_iterations,
            ham_xac_nhan=ham_xac_nhan,
            dung_self_refine=dung_self_refine,
            bo_refine=bo_refine,
            dung_self_consistency=dung_self_consistency,
            bo_consistency=bo_consistency,
            **kwargs,
        )
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

    def tu_sua_loi_voi_refine(self, ma_loi: str, yeu_cau: str) -> str:
        """Sử dụng SelfRefine để phân tích và tinh chỉnh phương án sửa lỗi."""
        if not self.bo_refine:
            from vietnamese_ai.salm.self_refine import SelfRefine

            self.bo_refine = SelfRefine(
                ham_sinh=self._goi_llm_don,
                so_vong_toi_da=self.so_vong_refine,
                nguong_chat_luong=self.nguong_chat_luong_refine,
            )
        prompt_refine = f"Phân tích và sửa chữa lỗi sau khi thực hiện yêu cầu '{yeu_cau}':\n{ma_loi}"
        kq = self.bo_refine.chay(prompt=prompt_refine)
        self.lich_su_refine.append(kq)
        return str(kq.get("output_cuoi", ma_loi))


# Vietnamese Alias
TacTuTuChuaLanh = SelfHealingAgent
