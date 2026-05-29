import time

from vietnamese_ai.edge import EdgeRouter, P2PTracker, SecureEdgeNode, TokenLedger


def test_p2p_tracker_registration():
    tracker = P2PTracker()

    # Đăng ký 2 node
    tracker.dang_ky_node("node_1", engine="Engine_1", toc_do_du_kien=5.0)
    tracker.dang_ky_node("node_2", engine="Engine_2", toc_do_du_kien=10.0)

    # Lấy node tốt nhất (phải là node 2 vì tốc độ cao hơn)
    best_node = tracker.tim_node_tot_nhat()
    assert best_node == "Engine_2"

def test_token_ledger():
    ledger = TokenLedger()
    ledger.thuong_token("node_1", 15.0)
    ledger.thuong_token("node_1", 5.0)

    assert ledger.lay_so_du("node_1") == 20.0
    assert ledger.lay_so_du("node_2") == 0.0

def test_hybrid_execution_edge_wins():
    ledger = TokenLedger()
    tracker = P2PTracker()

    # Khởi tạo SecureEdgeNode giả lập
    node = SecureEdgeNode(node_id="test_node", private_key="secret", auto_start=False)

    # Mock hàm chay_suy_luan_an_toan để trả kết quả nhanh
    def mock_chay_an_toan(cau_hoi):
        return {
            "node_id": "test_node",
            "cau_tra_loi": "Kết quả từ Edge",
            "proof": node.generate_proof(cau_hoi, "Kết quả từ Edge"),
            "model_hash": node.model_hash
        }
    node.chay_suy_luan_an_toan = mock_chay_an_toan

    tracker.dang_ky_node("test_node", node, 10.0)

    router = EdgeRouter(p2p_tracker=tracker, token_ledger=ledger, cloud_api_key="dummy")

    # Chặn _call_cloud để cloud luôn chậm hơn Edge
    def mock_call_cloud(prompt, do_dai):
        time.sleep(1) # Cố tình làm chậm Cloud
        return "Kết quả từ Cloud"
    router._call_cloud = mock_call_cloud

    ket_qua = router.sinh_van_ban_song_song("Xin chào")

    assert ket_qua == "Kết quả từ Edge"
    # Node phải được cộng Token
    assert ledger.lay_so_du("test_node") == 10.0

def test_hybrid_execution_cloud_wins_on_error():
    ledger = TokenLedger()
    tracker = P2PTracker()

    # Khởi tạo SecureEdgeNode giả lập (Cố tình trả Proof sai)
    node = SecureEdgeNode(node_id="bad_node", private_key="secret", auto_start=False)

    def mock_chay_an_toan_fake(cau_hoi):
        return {
            "node_id": "bad_node",
            "cau_tra_loi": "Kết quả Fake",
            "proof": "wrong_proof",
            "model_hash": node.model_hash
        }
    node.chay_suy_luan_an_toan = mock_chay_an_toan_fake
    tracker.dang_ky_node("bad_node", node, 10.0)

    router = EdgeRouter(p2p_tracker=tracker, token_ledger=ledger, cloud_api_key="dummy")

    def mock_call_cloud(prompt, do_dai):
        return "Kết quả từ Cloud an toàn"
    router._call_cloud = mock_call_cloud

    ket_qua = router.sinh_van_ban_song_song("Xin chào")

    # Kết quả phải lấy từ Cloud vì Edge bị loại
    assert ket_qua == "Kết quả từ Cloud an toàn"
    # Node không được nhận Token
    assert ledger.lay_so_du("bad_node") == 0.0
