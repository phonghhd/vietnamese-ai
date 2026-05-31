from typing import Any, Callable, Dict, List, Optional

from vietnamese_ai.agents.self_healing import SelfHealingAgent
from vietnamese_ai.agents.tools import CongCu
from vietnamese_ai.edge.intelligent_router import EdgeRouter
from vietnamese_ai.mobile.power_manager import PowerManager


class MobileHybridAgent(SelfHealingAgent):
    """
    Tác tử thông minh trên thiết bị di động.
    Biết tự động offload (san tải) lên Mạng P2P/Cloud nếu Pin yếu thông qua v27 PowerManager.
    """

    def __init__(
        self,
        llm: Any,
        danh_sach_cong_cu: List[CongCu],
        edge_router: Optional[EdgeRouter] = None,
        max_iterations: int = 5,
        ham_xac_nhan: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        super().__init__(
            llm=llm,
            danh_sach_cong_cu=danh_sach_cong_cu,
            max_iterations=max_iterations,
            ham_xac_nhan=ham_xac_nhan,
        )
        self.edge_router = edge_router
        self.power_manager = PowerManager()

    def _goi_llm(self, prompt: str) -> str:
        """
        Ghi đè hàm gọi LLM để kiểm tra Năng lượng.
        """
        percent, plugged = self.power_manager.get_battery_status()

        # Nếu Pin < 20% và không cắm sạc, đẩy lên mạng Edge P2P để tiết kiệm pin
        if percent < 20 and not plugged and self.edge_router:
            print(f"[Mobile Agent] CẢNH BÁO: Pin yếu ({percent}%). Offload tính toán sang mạng lưới DePIN / Cloud.")
            # Yêu cầu EdgeRouter xử lý thay cho điện thoại
            return self.edge_router.sinh_van_ban(prompt, force_cloud=True)

        # Ngược lại, chạy Local NPU
        return super()._goi_llm(prompt)
