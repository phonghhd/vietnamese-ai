"""Tests kiểm chứng tích hợp Transformers, Native LoRA trên GPTModel và BoTinhChinhMoHinh (AutoFineTuner)."""

import numpy as np

from vietnamese_ai import (
    AutoFineTuner,
    BoTinhChinhMoHinh,
    GPTModel,
    HuggingFaceWrapper,
    InstructionDataset,
    InstructionTuningTrainer,
    PhanLoai,
    RAGPipeline,
    TacTu,
)


def test_gpt_model_native_lora_training_and_merge(tmp_path):
    np.random.seed(42)
    gpt = GPTModel(d_model=32, so_dau=2, d_ff=64, so_block=2, so_tu_vung=64, do_dai_toi_da=32)

    w_q_before = gpt.blocks[0].attention.W_q.copy()
    w1_before = gpt.blocks[0].W1.copy()
    emb_before = gpt.embedding.copy()

    lora_cfg = gpt.gan_lora(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj", "up_proj", "down_proj"])
    assert gpt.da_gan_lora is True
    assert lora_cfg["so_tham_so_lora"] > 0
    assert lora_cfg["ty_le_huan_luyen_pct"] < 30.0

    X = np.array([[1, 5, 10, 15, 20, 25], [2, 6, 11, 16, 21, 26]], dtype=int)
    Y = np.array([[5, 10, 15, 20, 25, 30], [6, 11, 16, 21, 26, 31]], dtype=int)

    loss_init = gpt.tinh_loss(X, Y)
    for _ in range(15):
        gpt.buoc_huan_luyen(X, Y, toc_do_hoc=1e-2)
    loss_final = gpt.tinh_loss(X, Y)

    assert loss_final < loss_init
    # Trọng số gốc phải được đóng băng 100% khi bật LoRA
    assert np.allclose(gpt.blocks[0].attention.W_q, w_q_before)
    assert np.allclose(gpt.blocks[0].W1, w1_before)
    assert np.allclose(gpt.embedding, emb_before)

    # Lưu & tải trọng số LoRA
    lora_path = str(tmp_path / "gpt_lora.npz")
    saved = gpt.luu_lora(lora_path)
    assert saved["so_mang"] > 0
    loaded = gpt.tai_lora(lora_path)
    assert loaded["so_lop_da_tai"] > 0

    # Gộp LoRA vào trọng số gốc: đầu ra dự đoán trước và sau khi gộp phải khớp hoàn toàn
    logits_before_merge = gpt.tien(X)
    merged_layers = gpt.gop_lora()
    assert merged_layers > 0
    assert gpt.da_gan_lora is False
    logits_after_merge = gpt.tien(X)
    assert np.allclose(logits_before_merge, logits_after_merge, atol=1e-6)


def test_huggingface_wrapper_all_tasks_and_lora(tmp_path):
    # 1. Text Classification
    hf_cls = HuggingFaceWrapper()
    info = hf_cls.tai_model("phobert", nhiem_vu="text-classification", so_lop=2, che_do_offline=True)
    assert info["ten_model"] == "vinai/phobert-base"
    assert hf_cls.da_tai is True

    lora_info = hf_cls.gan_lora(rank=4, alpha=8.0)
    assert hf_cls.da_gan_lora is True
    assert lora_info["so_tham_so_lora"] > 0

    train_cls = [
        {"text": "Sản phẩm tuyệt vời chất lượng rất tốt", "label": "tich_cuc"},
        {"text": "Dịch vụ hoàn hảo giao hàng nhanh", "label": "tich_cuc"},
        {"text": "Hàng lỗi kém chất lượng rất tệ", "label": "tieu_cuc"},
        {"text": "Thất vọng sản phẩm hỏng hoàn tiền", "label": "tieu_cuc"},
    ]
    kq_cls = hf_cls.huan_luyen(train_cls, so_vong=5, toc_do_hoc=1e-2)
    assert kq_cls["history"]["train_loss"][-1] < kq_cls["history"]["train_loss"][0]
    preds = hf_cls.du_doan(["Sản phẩm tuyệt vời rất tốt", "Hàng lỗi rất tệ"])
    assert preds[0]["nhan"] == "tich_cuc"
    assert preds[1]["nhan"] == "tieu_cuc"

    # 2. Token Classification (NER)
    hf_ner = HuggingFaceWrapper()
    hf_ner.tai_model("phobert", nhiem_vu="token-classification", che_do_offline=True)
    hf_ner.huan_luyen(
        [{"tokens": ["VinAI", "ở", "Hà_Nội"], "ner_tags": ["B-ORG", "O", "B-LOC"]}],
        so_vong=2,
    )
    ner_out = hf_ner.du_doan(["VinAI ở Hà_Nội"])
    assert any(e["nhan"] == "B-ORG" for e in ner_out[0]["thuc_the"])

    # 3. Question Answering (QA)
    hf_qa = HuggingFaceWrapper()
    hf_qa.tai_model("phobert", nhiem_vu="question-answering", che_do_offline=True)
    hf_qa.huan_luyen(
        [
            {
                "context": "Vietnamese AI v30 hỗ trợ tinh chỉnh mọi mô hình với LoRA.",
                "question": "Vietnamese AI v30 hỗ trợ gì?",
                "answer": "tinh chỉnh mọi mô hình với LoRA",
            }
        ],
        so_vong=2,
    )
    qa_out = hf_qa.du_doan(["Vietnamese AI v30 hỗ trợ gì?"])
    assert "LoRA" in qa_out[0]["cau_tra_loi"]

    # 4. Lưu, tải local, xuất GGUF & ONNX
    save_dir = str(tmp_path / "saved_hf")
    hf_cls.luu_model(save_dir)
    hf_reloaded = HuggingFaceWrapper()
    hf_reloaded.tai_model_local(save_dir)
    assert hf_reloaded.da_tai is True

    gguf_out = hf_cls.xuat_gguf(str(tmp_path / "model.gguf"))
    assert gguf_out["kich_thuoc_bytes"] > 0
    onnx_out = hf_cls.xuat_onnx(str(tmp_path / "model.onnx"))
    assert onnx_out["kich_thuoc_bytes"] > 0


def test_instruction_trainer_real_backpropagation():
    ds = InstructionDataset()
    ds.tai_tu_list(
        [
            {"instruction": "Chào hỏi", "input": "", "output": "Xin chào bạn!"},
            {"instruction": "Tính toán", "input": "2 + 2", "output": "Kết quả là 4."},
            {"instruction": "Dịch sang tiếng Anh", "input": "Cảm ơn", "output": "Thank you."},
        ]
    )
    gpt = GPTModel(d_model=32, so_dau=2, d_ff=64, so_block=1, so_tu_vung=256, do_dai_toi_da=64)
    trainer = InstructionTuningTrainer(so_vong=4, kich_thuoc_batch=2, toc_do_hoc=5e-3)
    kq = trainer.huan_luyen(gpt, dataset=ds)
    assert len(kq["history"]["train_loss"]) == 4
    assert kq["history"]["train_loss"][-1] < kq["history"]["train_loss"][0]


def test_bo_tinh_chinh_mo_hinh_end_to_end(tmp_path):
    # 1. SFT với model alias + LoRA
    tuner_sft = BoTinhChinhMoHinh(
        mo_hinh="qwen2.5-1.5b",
        nhiem_vu="sft",
        dung_lora=True,
        rank_lora=4,
        che_do_offline=True,
    )
    sft_data = [
        {"instruction": "Thủ đô của Việt Nam là gì?", "output": "Thủ đô của Việt Nam là Hà Nội."},
        {"instruction": "AI là viết tắt của từ gì?", "output": "Trí tuệ nhân tạo (Artificial Intelligence)."},
    ]
    kq_sft = tuner_sft.huan_luyen(sft_data, so_vong=4, toc_do_hoc=5e-3)
    assert kq_sft["history"]["train_loss"][-1] < kq_sft["history"]["train_loss"][0]
    ans = tuner_sft.sinh_van_ban("Thủ đô của Việt Nam là gì?")
    assert "Hà Nội" in ans

    eval_res = tuner_sft.danh_gia(sft_data)
    assert eval_res["loss"] >= 0.0
    assert eval_res["perplexity"] >= 1.0

    # 2. DPO thông qua AutoFineTuner
    tuner_dpo = AutoFineTuner(mo_hinh="vgpt2", nhiem_vu="dpo", dung_lora=True, che_do_offline=True)
    dpo_data = [
        {
            "prompt": "Giải thích học máy",
            "chosen": "Học máy là nhánh của AI giúp hệ thống tự học từ dữ liệu.",
            "rejected": "Không biết.",
        }
    ]
    kq_dpo = tuner_dpo.huan_luyen(dpo_data, so_vong=3, toc_do_hoc=0.1)
    assert kq_dpo["final_reward_margin"] > 0
    assert "Học máy là nhánh của AI" in tuner_dpo.sinh_van_ban("Giải thích học máy")

    # 3. Tích hợp với RAGPipeline và TacTu (Agent)
    rag = RAGPipeline(ham_embed=tuner_sft._hf_wrapper)
    rag.them_tai_lieu("doc1", "Hà Nội là thủ đô ngàn năm văn hiến của Việt Nam.")
    kq_rag = rag.hoi("Thủ đô Việt Nam")
    assert "tra_loi" in kq_rag

    agent = TacTu(llm=tuner_sft, danh_sach_cong_cu=[])
    phan_hoi = agent.chay("Thủ đô của Việt Nam là gì?")
    assert isinstance(phan_hoi, str) and len(phan_hoi) > 0

    # 4. Tinh chỉnh mô hình PhanLoai cổ điển qua cùng giao diện BoTinhChinhMoHinh
    clf_tuner = BoTinhChinhMoHinh(mo_hinh=PhanLoai(), nhiem_vu="phan_loai", dung_lora=False)
    clf_tuner.huan_luyen(
        [
            {"text": "Món ăn rất ngon tuyệt vời", "label": 1},
            {"text": "Quán sạch sẽ phục vụ tốt", "label": 1},
            {"text": "Đồ ăn nguội lạnh rất dở", "label": 0},
            {"text": "Phục vụ chậm thái độ kém", "label": 0},
        ]
    )
    pred_lbl = clf_tuner.du_doan("Món ăn rất ngon tuyệt vời")
    assert int(pred_lbl) == 1
