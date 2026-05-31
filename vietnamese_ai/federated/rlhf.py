import logging
from typing import Any, Dict

from vietnamese_ai.edge.p2p_network import TokenLedger
from vietnamese_ai.federated.learning import HocLienKet


class FederatedRLHF(HocLienKet):
    """
    Học Tăng Cường Phân Tán (Federated RLHF).
    Thay vì train trực tiếp trên máy chủ, thu thập DPO Loss/LoRA Delta
    từ các Edge Nodes khi End-User cung cấp phản hồi (Prefer/Reject).
    """

    def __init__(self, so_client: int, so_vong: int = 1, token_ledger: TokenLedger = None):
        super().__init__(so_client=so_client, so_vong=so_vong)
        self.token_ledger = token_ledger or TokenLedger()
        self.logger = logging.getLogger("FederatedRLHF")
        self.danh_sach_delta_lora = []

    def nhan_phan_hoi_dpo(self, user_id: str, data_dpo: Dict[str, Any]):
        """
        Nhận kết quả tính toán Local DPO từ Edge Node của người dùng.
        data_dpo gồm: 'delta_lora' (mảng trọng số đã thay đổi), 'loss'
        """
        self.logger.info(f"Đã nhận trọng số Local DPO từ {user_id} (Loss: {data_dpo.get('loss', 0.0):.4f})")
        self.danh_sach_delta_lora.append(data_dpo['delta_lora'])

        # Thưởng Token cho người dùng vì đã góp phần huấn luyện mạng AI
        self.token_ledger.thuong_token_danh_gia(user_id, chat_luong_danh_gia=1.0)

    def tong_hop_lora(self) -> Any:
        """
        Hàm Federated Averaging (FedAvg) áp dụng riêng cho ma trận LoRA.
        Trong mô phỏng này, ta tính trung bình các trọng số.
        """
        if not self.danh_sach_delta_lora:
            self.logger.warning("Không có trọng số LoRA nào để tổng hợp.")
            return None

        # Giả lập tính toán trung bình trọng số (FedAvg)
        tong_so = len(self.danh_sach_delta_lora)
        global_lora = sum(self.danh_sach_delta_lora) / tong_so

        self.logger.info(f"Đã tổng hợp thành công {tong_so} bản cập nhật LoRA từ mạng DePIN.")

        # Reset mảng sau mỗi vòng (round)
        self.danh_sach_delta_lora = []
        return global_lora
