"""Kiểm thử toàn diện 5 Tính năng Thế hệ mới (Phase 10):
1. GRPOTrainer (Group Relative Policy Optimization) & Reasoning Verifier (<think>...</think>)
2. Kiến trúc Transformer Hiện đại: RoPE, GQA (KV Cache) & Sparse MoE trên GPTModel
3. MoHinhThiGiacNgonNgu (VietnameseVLM - Đa phương thức Ảnh + Văn bản + VQA)
4. NenNguCanh (Prompt Compression) & BoNhoDemNguNghia (Semantic Cache) trong RAGPipeline
5. CongCuManHinh (Agentic Computer Use) & StudioKeoTha thế hệ mới (LLM/RAG/VLM)
"""

import numpy as np

from vietnamese_ai import (
    BoNhoDemNguNghia,
    BoTinhChinhMoHinh,
    CongCuManHinh,
    GPTModel,
    GroupedQueryAttention,
    GRPOTrainer,
    HoDuLieu,
    MoHinhThiGiacNgonNgu,
    NenNguCanh,
    RAGPipeline,
    RotaryPositionEmbedding,
    SparseMoELayer,
    StudioKeoTha,
    cong_cu_man_hinh,
)


def test_feature1_grpo_trainer_and_reasoning_verifier():
    """Kiểm thử GRPOTrainer & tích hợp BoTinhChinhMoHinh(nhiem_vu='grpo')."""
    ho = HoDuLieu()
    du_lieu_cot = ho.tai_bo_du_lieu_mau("vi_reasoning_cot")
    assert len(du_lieu_cot) >= 5

    grpo = GRPOTrainer(so_nhom=4, beta_kl=0.04, toc_do_hoc=1e-3, so_vong=2)
    r_tot = grpo.cham_diem_suy_luan(
        "<think>\nBước 1: Tính 15 + 27 = 42.\nBước 2: Kết luận.\n</think>\nĐáp án là 42.",
        dap_an_chuan="42",
    )
    r_kem = grpo.cham_diem_suy_luan("Tôi đoán là 99", dap_an_chuan="42")
    assert r_tot > r_kem
    assert r_tot >= 1.5

    adv = grpo.tinh_loi_the_nhom([0.2, 0.5, 1.2, 2.1])
    assert adv.shape == (4,)
    assert abs(float(np.mean(adv))) < 1e-5

    tuner = BoTinhChinhMoHinh(
        mo_hinh="qwen3.8",
        nhiem_vu="grpo",
        dung_lora=True,
        che_do_offline=True,
    )
    kq = tuner.huan_luyen(du_lieu_cot[:4], so_vong=2)
    assert kq["phuong_phap"] == "GRPO"
    assert kq["reward_trung_binh_cuoi"] > 0.0

    sinh = tuner.sinh_van_ban(du_lieu_cot[0]["instruction"])
    assert "<think>" in sinh and "</think>" in sinh


def test_feature2_modern_transformer_rope_gqa_sparse_moe():
    """Kiểm thử RotaryPositionEmbedding (RoPE), GroupedQueryAttention (GQA) & SparseMoELayer."""
    batch, seq_len, d_model = 2, 12, 64
    so_dau_q, so_dau_kv = 8, 2
    d_k = d_model // so_dau_q

    rope = RotaryPositionEmbedding(d_k=d_k, max_seq_len=128)
    q = np.random.randn(batch, so_dau_q, seq_len, d_k).astype(np.float32)
    k = np.random.randn(batch, so_dau_kv, seq_len, d_k).astype(np.float32)
    q_rot, k_rot = rope.ap_dung(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # Bảo toàn chuẩn L2 của vector khi xoay phức
    assert np.allclose(np.linalg.norm(q, axis=-1), np.linalg.norm(q_rot, axis=-1), atol=1e-4)

    gqa = GroupedQueryAttention(d_model=d_model, so_dau_q=so_dau_q, so_dau_kv=so_dau_kv)
    X = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    out_gqa = gqa.tien(X, su_dung_cache=True)
    assert out_gqa.shape == (batch, seq_len, d_model)
    tk_gqa = gqa.thong_ke()
    assert tk_gqa["he_so_nen_kv_cache"] == 4

    moe = SparseMoELayer(d_model=d_model, d_ff=128, so_chuyen_gia=4, top_k=2)
    out_moe = moe.tien(X)
    assert out_moe.shape == (batch, seq_len, d_model)
    assert moe.aux_loss > 0.0
    assert len(moe.thong_ke()["tan_suat_chon_chuyen_gia"]) == 4

    # Tích hợp MoE vào GPTModel
    gpt = GPTModel(d_model=64, so_dau=4, d_ff=128, so_block=2, so_tu_vung=256)
    cfg_moe = gpt.kich_hoat_moe(so_chuyen_gia=4, top_k=2)
    assert cfg_moe["so_chuyen_gia"] == 4
    ids = np.random.randint(0, 256, size=(2, 8))
    logits = gpt.tien(ids)
    assert logits.shape == (2, 8, 256)
    assert gpt.tinh_aux_loss_moe() > 0.0


def test_feature3_vietnamese_vlm_multimodal():
    """Kiểm thử MoHinhThiGiacNgonNgu (VietnameseVLM) hỏi đáp và huấn luyện trên ảnh."""
    vlm = MoHinhThiGiacNgonNgu(mo_hinh_ngon_ngu="qwen3.7-omni", che_do_offline=True)

    # Tạo ảnh tài liệu giả lập có 2 dòng chữ đen trên nền sáng
    anh_hoa_don = np.full((80, 200), 245, dtype=np.uint8)
    anh_hoa_don[15:28, 20:170] = 15
    anh_hoa_don[45:58, 20:150] = 20

    mo_ta = vlm.mo_ta_anh(anh_hoa_don)
    assert "200x80" in mo_ta

    kq_train = vlm.huan_luyen(
        [
            {
                "image": anh_hoa_don,
                "question": "Tổng tiền thanh toán trên hóa đơn là bao nhiêu?",
                "answer": "Tổng tiền trên hóa đơn là 1.500.000 VNĐ.",
            }
        ],
        so_vong=2,
    )
    assert kq_train["so_mau_anh"] == 1

    kq_vqa = vlm.hoi_dap_anh(anh_hoa_don, "Tổng tiền thanh toán trên hóa đơn là bao nhiêu?")
    assert "1.500.000 VNĐ" in kq_vqa["tra_loi"]
    assert kq_vqa["mo_hinh"] == "qwen3.7-omni"


def test_feature4_prompt_compression_and_semantic_cache_in_rag():
    """Kiểm thử NenNguCanh (LLMLingua-style) và BoNhoDemNguNghia trong RAGPipeline."""
    van_ban_dai = (
        "Trí tuệ nhân tạo tại Việt Nam đang phát triển mạnh mẽ trong năm 2026. "
        "Các ngân hàng thương mại áp dụng mô hình ngôn ngữ lớn để tự động hóa quy trình thẩm định tín dụng. "
        "Ngoài ra, thời tiết hôm nay tại công viên khá mát mẻ và nhiều người đi dạo bộ buổi chiều. "
        "Hệ thống RAG tiếng Việt giúp truy xuất văn bản pháp luật chính xác và giảm thiểu ảo giác."
    )
    compressor = NenNguCanh(ty_le_giu=0.5)
    kq_nen = compressor.nen(van_ban_dai, cau_hoi="Hệ thống RAG tiếng Việt giúp ích gì cho pháp luật?")
    assert kq_nen["so_tu_nen"] <= kq_nen["so_tu_goc"]
    assert "RAG" in kq_nen["van_ban_nen"] or "pháp luật" in kq_nen["van_ban_nen"]

    cache = BoNhoDemNguNghia(nguong_tuong_dong=0.80)
    cache.luu("Cách cài đặt Vietnamese AI framework?", {"tra_loi": "Chạy pip install vietnamese-ai"})
    hit = cache.tra_cuu("Cách cài đặt Vietnamese AI framework?")
    assert hit is not None and hit["tra_loi"] == "Chạy pip install vietnamese-ai"
    assert cache.thong_ke()["hits"] == 1

    # Tích hợp trực tiếp trong RAGPipeline
    rag = RAGPipeline(nen_ngu_canh=True, ty_le_nen=0.6, dung_semantic_cache=True, nguong_cache=0.85)
    rag.them_tai_lieu("doc_ai", van_ban_dai)
    res1 = rag.hoi("Hệ thống RAG tiếng Việt giúp ích gì?")
    assert res1["tu_semantic_cache"] is False
    assert "thong_tin_nen" in res1

    res2 = rag.hoi("Hệ thống RAG tiếng Việt giúp ích gì?")
    assert res2["tu_semantic_cache"] is True


def test_feature5_computer_use_and_modern_studio():
    """Kiểm thử CongCuManHinh (Agentic Computer Use) và StudioKeoTha với node LLM/RAG/VLM."""
    cu = CongCuManHinh(kich_thuoc_man_hinh=(1920, 1080))
    screenshot = np.full((100, 200), 250, dtype=np.uint8)
    screenshot[20:35, 30:170] = 10
    screenshot[60:75, 40:160] = 15

    phan_tich = cu.phan_tich_man_hinh(screenshot)
    assert phan_tich["kich_thuoc_anh"] == [200, 100]
    assert phan_tich["so_phan_tu_ui"] >= 1

    act1 = cu.click(500, 300)
    act2 = cu.nhap_van_ban("Xin chào Qwen3.8")
    assert act1["trang_thai"] == "thanh_cong"
    assert act2["van_ban"] == "Xin chào Qwen3.8"
    assert len(cu.lich_su_hanh_dong) == 2

    out_tool = cong_cu_man_hinh.chay(hanh_dong="click", x=250, y=400)
    assert "[ComputerUse]" in out_tool

    # Kiểm thử StudioKeoTha với các loại node thế hệ mới (du_lieu HoDuLieu -> llm -> rag -> vlm)
    studio = StudioKeoTha(ten="NextGen AI Studio")
    thuat_toan = studio.danh_sach_thuat_toan()
    assert "qwen3.8" in thuat_toan["llm"]
    assert "gemma4" in thuat_toan["llm"]
    assert "xgboost" in thuat_toan["phan_loai"]

    n_data = studio.them_node("du_lieu", "Bộ dữ liệu Pháp luật", tham_so={"ten_bo_du_lieu": "vi_phap_luat"})
    n_llm = studio.them_node("llm", "Tinh chỉnh Qwen3.8", tham_so={"mo_hinh": "qwen3.8", "nhiem_vu": "sft", "so_vong": 2})
    n_rag = studio.them_node("rag", "RAG Pháp luật Nén ngữ cảnh", tham_so={"nen_ngu_canh": True, "cau_hoi": "Hợp đồng lao động quy định gì?"})
    n_vlm = studio.them_node("vlm", "VLM Qwen3.7-Omni", tham_so={"mo_hinh": "qwen3.7-omni", "cau_hoi": "Ảnh này có nội dung gì?"})

    studio.ket_noi(n_data["ma"], n_llm["ma"])
    studio.ket_noi(n_data["ma"], n_rag["ma"])
    studio.ket_noi(n_data["ma"], n_vlm["ma"])

    ket_qua = studio.chay()
    assert ket_qua["trang_thai"] == "thanh_cong"
    assert ket_qua["so_node_thanh_cong"] == 4
