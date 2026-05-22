from typing import Any, Dict, Optional

from .agent import TacTu
from .memory import BoNhoTacTu
from .tools import CongCu


class HeThongDaTacTu:
    """
    Quản lý luồng giao tiếp giữa nhiều Tác tử (Multi-Agent System).
    Sử dụng một Tác tử quản lý (Orchestrator) hoặc luồng tĩnh (Sequential).
    """

    def __init__(
        self, 
        danh_sach_tac_tu: Dict[str, TacTu], 
        loai_dieu_phoi: str = "sequential",
        llm_dieu_phoi: Any = None,
        mo_ta_tac_tu: Optional[Dict[str, str]] = None
    ):
        """
        Khởi tạo hệ thống đa tác tử.

        Args:
            danh_sach_tac_tu: Một dict map tên tác tử với đối tượng TacTu.
            loai_dieu_phoi: "sequential" (tuần tự) hoặc "orchestrator" (quản lý động).
            llm_dieu_phoi: Đối tượng LLM làm bộ não cho Orchestrator (bắt buộc nếu dùng orchestrator).
            mo_ta_tac_tu: Mô tả chức năng của các tác tử con (giúp Orchestrator biết nên gọi ai).
        """
        self.tac_tu = danh_sach_tac_tu
        self.loai_dieu_phoi = loai_dieu_phoi
        self.lich_su_chung = BoNhoTacTu(system_prompt="Bạn là hệ thống lưu trữ lịch sử chung của nhóm tác tử.")
        
        if self.loai_dieu_phoi == "orchestrator":
            if llm_dieu_phoi is None:
                raise ValueError("Cần cung cấp llm_dieu_phoi khi sử dụng chế độ orchestrator.")
            
            self.llm_dieu_phoi = llm_dieu_phoi
            self.mo_ta_tac_tu = mo_ta_tac_tu or {}
            self.orchestrator = self._khoi_tao_orchestrator()

    def _khoi_tao_orchestrator(self) -> TacTu:
        """Tạo tác tử Orchestrator với các công cụ là các tác tử con."""
        danh_sach_cong_cu = []
        for ten, tac_tu in self.tac_tu.items():
            # Tạo mô tả công cụ
            mo_ta = self.mo_ta_tac_tu.get(ten, f"Giao tiếp với tác tử '{ten}' để thực hiện nhiệm vụ của nó.")
            
            # Hàm wrapper để gọi tác tử con
            # Phải dùng tham số mặc định tt=tac_tu để closure bắt đúng đối tượng trong vòng lặp
            def tao_ham_goi_tac_tu(tt=tac_tu):
                def goi_tac_tu_con(yeu_cau: str) -> str:
                    return tt.chay(yeu_cau)
                return goi_tac_tu_con
            
            cc = CongCu(
                ten=f"goi_tac_tu_{ten.replace(' ', '_')}",
                mo_ta=f"{mo_ta}. Tham số đầu vào là 'yeu_cau' (chuỗi mô tả công việc).",
                ham_thuc_thi=tao_ham_goi_tac_tu()
            )
            danh_sach_cong_cu.append(cc)
            
        return TacTu(llm=self.llm_dieu_phoi, danh_sach_cong_cu=danh_sach_cong_cu, max_iterations=10)

    def chay_tuan_tu(self, truy_van: str) -> str:
        """
        Chạy các tác tử theo thứ tự được định nghĩa trong danh_sach_tac_tu.
        Kết quả của tác tử trước sẽ là một phần đầu vào của tác tử sau.
        """
        self.lich_su_chung.them_tin_nhan("user", f"Yêu cầu ban đầu: {truy_van}")

        ket_qua_hien_tai = truy_van

        for ten, tac_tu in self.tac_tu.items():
            prompt = f"Yêu cầu hiện tại cho {ten}:\n{ket_qua_hien_tai}\n\nHãy thực hiện nhiệm vụ của bạn."
            ket_qua_tac_tu = tac_tu.chay(prompt)

            self.lich_su_chung.them_tin_nhan("assistant", f"[{ten}]: {ket_qua_tac_tu}")
            ket_qua_hien_tai = ket_qua_tac_tu

        return ket_qua_hien_tai

    def _chay_dieu_phoi_dong(self, truy_van: str) -> str:
        """
        Chạy chế độ điều phối động (Orchestrator).
        Orchestrator sẽ nhận yêu cầu và tự quyết định gọi tác tử nào.
        """
        self.lich_su_chung.them_tin_nhan("user", f"Yêu cầu ban đầu: {truy_van}")
        
        # Gọi Orchestrator xử lý
        ket_qua_cuoi = self.orchestrator.chay(
            f"Yêu cầu từ người dùng: {truy_van}\n\n"
            f"Hãy sử dụng các công cụ của bạn (chính là các Tác tử chuyên biệt) để giải quyết yêu cầu này. "
            f"Mỗi khi cần làm gì, hãy phân tích và gọi đúng Tác tử. Khi đã hoàn thành toàn bộ yêu cầu, hãy trả lời kết quả cuối cùng."
        )
        
        self.lich_su_chung.them_tin_nhan("assistant", f"[Orchestrator]: {ket_qua_cuoi}")
        return ket_qua_cuoi

    def chay(self, truy_van: str) -> str:
        """Thực thi hệ thống."""
        if self.loai_dieu_phoi == "sequential":
            return self.chay_tuan_tu(truy_van)
        elif self.loai_dieu_phoi == "orchestrator":
            return self._chay_dieu_phoi_dong(truy_van)
        else:
            raise NotImplementedError(f"Chế độ điều phối '{self.loai_dieu_phoi}' không được hỗ trợ.")
