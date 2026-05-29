from vietnamese_ai.serving.paged_attention import BlockManager, PagedAttentionSimulation


def test_paged_attention():
    # 4 blocks, mỗi block 2 tokens
    manager = BlockManager(num_blocks=4, block_size=2)

    # User 1 bắt đầu sinh text
    manager.append_token("seq_1", 101) # Chiếm 1 physical block
    assert len(manager.free_blocks) == 3

    manager.append_token("seq_1", 102) # Vẫn ở block đó
    assert len(manager.free_blocks) == 3

    manager.append_token("seq_1", 103) # Sang block mới
    assert len(manager.free_blocks) == 2

    # User 2 bắt đầu
    manager.append_token("seq_2", 999) # Chiếm 1 block mới
    assert len(manager.free_blocks) == 1

    # Mô phỏng tính toán
    paged_attn = PagedAttentionSimulation(manager)
    attn_result = paged_attn.forward(None, "seq_1")
    assert "Attention computed using physical blocks" in attn_result

    # Giải phóng
    manager.free_sequence("seq_1")
    assert len(manager.free_blocks) == 3

    manager.free_sequence("seq_2")
    assert len(manager.free_blocks) == 4
