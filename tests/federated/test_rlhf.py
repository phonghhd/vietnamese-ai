from vietnamese_ai.edge.p2p_network import TokenLedger
from vietnamese_ai.federated.rlhf import FederatedRLHF


def test_federated_rlhf_thuong_token():
    ledger = TokenLedger()
    fed_rlhf = FederatedRLHF(so_client=2, token_ledger=ledger)

    # Giả lập người dùng gửi trọng số Local DPO
    fed_rlhf.nhan_phan_hoi_dpo("user_1", {"delta_lora": 0.5, "loss": 0.1})
    fed_rlhf.nhan_phan_hoi_dpo("user_2", {"delta_lora": 1.5, "loss": 0.2})

    # Kiểm tra người dùng đã nhận phần thưởng chưa
    assert ledger.lay_so_du("user_1") == 5.0
    assert ledger.lay_so_du("user_2") == 5.0

    # Tổng hợp FedAvg (0.5 + 1.5) / 2 = 1.0
    global_weight = fed_rlhf.tong_hop_lora()
    assert global_weight == 1.0
