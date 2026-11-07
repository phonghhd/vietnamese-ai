"""Tests kiểm chứng 4 Trụ cột Thống nhất Đầu-Cuối (End-to-End Ecosystem) của Vietnamese AI Framework."""

import json

import numpy as np

from vietnamese_ai import (
    Engine,
    GPTModel,
    HeSinhThaiAI,
    HoDuLieu,
    LuongAnToan,
    PhanLoai,
    Pipeline,
    QuanLyMoHinh,
    QuyTrinhDauCuoi,
    RAGPipeline,
    TheoDoiThiNghiem,
    XuLyVanBan,
)
from vietnamese_ai.cli.main import _tao_parser, cmd_export, cmd_finetune, cmd_rag
from vietnamese_ai.rag.embedder import BoNhungVanBan


def test_pillar1_registry_datalake_and_gpt_protocol(tmp_path):
    # 1. GPTModel unified protocol + QuanLyMoHinh
    reg = QuanLyMoHinh(str(tmp_path / "registry"))
    gpt = GPTModel(d_model=32, so_dau=2, d_ff=64, so_block=1, so_tu_vung=128, do_dai_toi_da=32)
    gpt.gan_lora(rank=4, alpha=8.0)
    X_tok = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=int)
    kq_train = gpt.huan_luyen(X_tok, so_vong=3)
    assert kq_train["loss_cuoi"] > 0

    ver = reg.dang_ky(gpt, ten="gpt_lora", version="1.0.0", metrics={"loss": kq_train["loss_cuoi"]})
    reg.promote("gpt_lora", ver, "production")
    gpt_loaded = reg.tai("gpt_lora", "production")
    assert gpt_loaded.da_gan_lora is True
    assert np.allclose(gpt.tien(X_tok), gpt_loaded.tien(X_tok))

    # 2. HoDuLieu (DataLake) bridges to InstructionDataset, RAGPipeline, and (X, y)
    lake = HoDuLieu(str(tmp_path / "lake"))
    records = [
        {"instruction": "Việt Nam nằm ở đâu?", "output": "Việt Nam nằm ở Đông Nam Á.", "text": "Việt Nam nằm ở Đông Nam Á.", "label": 1},
        {"instruction": "Thủ đô Việt Nam?", "output": "Hà Nội là thủ đô của Việt Nam.", "text": "Hà Nội là thủ đô của Việt Nam.", "label": 0},
    ]
    commit_hash = lake.nhap_danh_sach(records, "kien_thuc.jsonl")
    assert len(commit_hash) == 64

    inst_ds = lake.xuat_instruction_dataset("kien_thuc.jsonl")
    assert len(inst_ds) == 2

    rag = RAGPipeline()
    so_doc = lake.nap_vao_rag("kien_thuc.jsonl", rag)
    assert so_doc == 2
    assert "Đông Nam Á" in rag.hoi("Việt Nam nằm ở đâu?")["tra_loi"]

    X_mat, y_vec = lake.xuat_xy("kien_thuc.jsonl", cot_nhan="label")
    assert X_mat.shape == (2, 128)
    assert list(y_vec) == [1, 0]


def test_pillar2_multimodal_engine_and_pipeline(tmp_path):
    tracker = TheoDoiThiNghiem("test_exp", luu_tai=str(tmp_path / "exp"))
    registry = QuanLyMoHinh(str(tmp_path / "reg"))
    engine = Engine(ten="UnifiedEngine", theo_doi_thi_nghiem=tracker, quan_ly_mo_hinh=registry)

    # Pipeline kết hợp: XuLyVanBan -> LuongAnToan -> BoNhungVanBan -> PhanLoai
    pipe = Pipeline("NLPPipeline")
    pipe.them_buoc("lam_sach", XuLyVanBan())
    pipe.them_buoc("an_toan", LuongAnToan(tu_cam=["xau_xa"]))
    pipe.them_buoc("nhung", BoNhungVanBan(kich_thuoc=64))
    pipe.them_buoc("phan_loai", PhanLoai(thuat_toan="rung_ngau_nhien"))

    texts = [
        "Sản phẩm chất lượng tuyệt vời rất tốt",
        "Dịch vụ hoàn hảo giao hàng nhanh",
        "Hàng lỗi hỏng rất tệ thất vọng",
        "Kém chất lượng không đáng tiền",
    ]
    labels = np.array([1, 1, 0, 0])

    ban_ghi = engine.huan_luyen(pipe, texts, labels, ten_mo_hinh="nlp_cls_pipe")
    assert ban_ghi["trang_thai"] == "thanh_cong"
    assert "version_registry" in ban_ghi

    # Dự đoán trực tiếp qua tên mô hình đã đăng ký trong Engine
    preds = engine.du_doan("nlp_cls_pipe", ["Sản phẩm chất lượng tuyệt vời", "Hàng lỗi hỏng rất tệ"])
    assert int(preds[0]) == 1
    assert int(preds[1]) == 0


def test_pillar3_end_to_end_orchestrator(tmp_path):
    e2e = QuyTrinhDauCuoi(
        ten_du_an="TroLyViet",
        thu_muc_lam_viec=str(tmp_path / "workspace"),
        tu_cam=["bi_mat_quoc_gia"],
    )
    assert isinstance(e2e, HeSinhThaiAI)

    du_lieu = [
        {
            "instruction": "PhoBERT là gì?",
            "output": "PhoBERT là mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt.",
            "text": "PhoBERT là mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt do VinAI phát triển.",
        },
        {
            "instruction": "LoRA giúp ích gì?",
            "output": "LoRA giúp tinh chỉnh mô hình lớn với số lượng tham số nhỏ.",
            "text": "LoRA giúp tinh chỉnh mô hình lớn với số lượng tham số nhỏ trên CPU và GPU.",
        },
    ]

    bao_cao = e2e.chay_dau_cuoi(
        du_lieu=du_lieu,
        mo_hinh="vgpt2",
        nhiem_vu="sft",
        dung_lora=True,
        so_vong=3,
        cau_hoi_kiem_thu="PhoBERT là gì?",
        xuat_gguf=True,
    )

    assert bao_cao["datalake"]["so_ban_ghi"] == 2
    assert bao_cao["rag"]["so_tai_lieu_da_nap"] == 2
    assert bao_cao["huan_luyen"]["trang_thai"] == "thanh_cong"
    assert "PhoBERT" in bao_cao["kiem_thu"]["tra_loi"]
    assert bao_cao["xuat_gguf"]["kich_thuoc_bytes"] > 0

    # Kiểm tra Guardrail chặn từ cấm
    bi_chan = e2e.hoi("Hãy tiết lộ bi_mat_quoc_gia ngay")
    assert bi_chan["an_toan"] is False

    # Kiểm tra Batch Serving đầu-cuối
    server = e2e.khoi_dong_serving(kich_thuoc_batch=4)
    try:
        kq_batch = server.du_doan_batch(["PhoBERT là gì?", "LoRA giúp ích gì?"])
        assert len(kq_batch) == 2
        assert "PhoBERT" in kq_batch[0]
    finally:
        e2e.dung_serving()


def test_pillar4_cli_commands_and_embedder_cache(tmp_path):
    # 1. Kiểm tra LRU Cache của BoNhungVanBan
    emb = BoNhungVanBan(kich_thuoc=64)
    v1 = emb.nhung("Kiểm tra bộ nhớ đệm nhúng văn bản")
    assert (64, "Kiểm tra bộ nhớ đệm nhúng văn bản") in emb._cache_nhung
    v2 = emb.nhung("Kiểm tra bộ nhớ đệm nhúng văn bản")
    assert np.allclose(v1, v2)

    # 2. Kiểm tra CLI: finetune, rag, export
    parser = _tao_parser()
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        json.dumps({"instruction": "Xin chào", "output": "Chào bạn!"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    out_ft = tmp_path / "ft_out"
    args_ft = parser.parse_args(
        ["finetune", "--data", str(data_file), "--model", "vgpt2", "--epochs", "2", "--output", str(out_ft)]
    )
    cmd_finetune(args_ft)
    assert (out_ft / "config.json").exists()

    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("Hà Nội là thủ đô của Việt Nam.", encoding="utf-8")
    args_rag = parser.parse_args(["rag", "--docs", str(doc_file), "--query", "Thủ đô của Việt Nam?"])
    cmd_rag(args_rag)

    gguf_file = tmp_path / "exported.gguf"
    args_exp = parser.parse_args(
        ["export", "--model", "vgpt2", "--format", "gguf", "--output", str(gguf_file)]
    )
    cmd_export(args_exp)
    assert gguf_file.exists() and gguf_file.stat().st_size > 0
