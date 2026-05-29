from vietnamese_ai.edge.tee_zkp_node import SecureEdgeNode


def test_secure_edge_node_proof():
    private_key = "super_secret_key_123"
    node = SecureEdgeNode(node_id="node_01", private_key=private_key, model_path="test_model", auto_start=False)

    # Giả lập lại hàm suy luận (vì test không có model thật)
    node.engine.sinh_van_ban = lambda cau_hoi, do_dai: f"Trả lời cho {cau_hoi}"

    cau_hoi = "Hôm nay là thứ mấy?"
    ket_qua = node.chay_suy_luan_an_toan(cau_hoi)

    assert ket_qua["node_id"] == "node_01"
    assert "Trả lời cho" in ket_qua["cau_tra_loi"]
    assert "proof" in ket_qua
    assert "model_hash" in ket_qua

    # Xác minh từ phía Edge Router
    # Public key trong mô phỏng HMAC này chính là private key
    is_valid = SecureEdgeNode.verify_proof(
        node_id=ket_qua["node_id"],
        public_key=private_key,
        cau_hoi=cau_hoi,
        cau_tra_loi=ket_qua["cau_tra_loi"],
        proof=ket_qua["proof"],
        model_hash=ket_qua["model_hash"]
    )
    assert is_valid is True

    # Giả mạo câu trả lời (Attacker thay đổi nội dung)
    is_valid_fake = SecureEdgeNode.verify_proof(
        node_id=ket_qua["node_id"],
        public_key=private_key,
        cau_hoi=cau_hoi,
        cau_tra_loi="Câu trả lời đã bị sửa đổi",
        proof=ket_qua["proof"],  # Dùng lại proof cũ
        model_hash=ket_qua["model_hash"]
    )
    assert is_valid_fake is False
