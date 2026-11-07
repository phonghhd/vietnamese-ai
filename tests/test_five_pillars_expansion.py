"""Tests kiểm chứng trọn bộ 5 Hạng mục Mở rộng Chiến lược của Vietnamese AI Framework."""

from dataclasses import dataclass

import numpy as np

from vietnamese_ai import (
    BoTinhChinhMoHinh,
    CongCu,
    DanhGiaRAG,
    GiaiThichMoHinh,
    GPTModel,
    HoDuLieu,
    HoiQuy,
    LlamaCppWrapper,
    MayChuOpenAI,
    MCPBridge,
    MCPClient,
    MCPServer,
    ModelConfig,
    ModelHub,
    NhanDienGiongNoi,
    OllamaWrapper,
    PhanLoai,
    PhanTichDauRa,
    RAGPipeline,
    SinhCoCauTruc,
    TacTu,
    TimKiemThamSo,
    TongHopGiongNoi,
    TrichXuatVanBanAnh,
    VLLMWrapper,
    XuatGGUF,
)


def test_cat1_local_llms_and_openai_compatible_server(tmp_path):
    # 1. OllamaWrapper & VLLMWrapper
    ollama = OllamaWrapper(model="qwen2.5:7b", che_do_offline=True)
    assert "qwen2.5:7b" in ollama.danh_sach_mo_hinh()
    out_ollama = ollama.sinh_van_ban("Xin chào Việt Nam")
    assert isinstance(out_ollama, str) and len(out_ollama) > 0

    vllm = VLLMWrapper(model="vistral-7b", che_do_offline=True)
    out_vllm = vllm.sinh_van_ban("Hệ thống: Trợ lý\nNgười dùng: AI là gì?")
    assert isinstance(out_vllm, str) and len(out_vllm) > 0

    # 2. LlamaCppWrapper nạp trực tiếp tệp .gguf xuất từ XuatGGUF
    gpt = GPTModel(d_model=32, so_dau=2, d_ff=64, so_block=1, so_tu_vung=128)
    gguf_path = tmp_path / "local_model.gguf"
    XuatGGUF(gpt).xuat(str(gguf_path), kieu_luong_tu="q8_0")
    llama_cpp = LlamaCppWrapper(str(gguf_path))
    out_gguf = llama_cpp.sinh_van_ban("Chào bạn", do_dai=8)
    assert isinstance(out_gguf, str) and len(out_gguf) > 0

    # 3. MayChuOpenAI (/v1/chat/completions & /v1/embeddings)
    rag = RAGPipeline()
    rag.them_tai_lieu("doc1", "PhoBERT là mô hình ngôn ngữ tiếng Việt do VinAI phát triển.")
    server = MayChuOpenAI(mo_hinh=rag, ten="vietnamese-rag-v1")

    chat_resp = server.xu_ly_chat_completion(
        {
            "model": "vietnamese-rag-v1",
            "messages": [{"role": "user", "content": "Ai phát triển PhoBERT?"}],
        }
    )
    assert chat_resp["object"] == "chat.completion"
    assert "VinAI" in chat_resp["choices"][0]["message"]["content"]
    assert chat_resp["usage"]["total_tokens"] > 0

    emb_resp = server.xu_ly_embeddings({"input": ["Xin chào", "Trí tuệ nhân tạo"]})
    assert emb_resp["object"] == "list"
    assert len(emb_resp["data"]) == 2
    assert len(emb_resp["data"][0]["embedding"]) == 128


def test_cat2_vietnamese_speech_asr_tts_and_ocr(tmp_path):
    # 1. Tổng hợp tiếng nói 6 thanh điệu (TTS) -> .wav -> Nhận diện giọng nói (ASR + MFCC)
    tts = TongHopGiongNoi(tan_so_lay_mau=16000)
    wav_path = tmp_path / "tieng_viet.wav"
    kq_tts = tts.tong_hop("Xin chào đất nước Việt Nam", duong_dan_luu=wav_path)
    assert wav_path.exists() and wav_path.stat().st_size > 1000
    assert kq_tts["thoi_luong_giay"] > 0.5

    asr = NhanDienGiongNoi(che_do_offline=True)
    mfcc = asr.trich_xuat_mfcc(wav_path)
    assert mfcc.ndim == 2 and mfcc.shape[1] == 13

    kq_asr = asr.nhan_dien(wav_path)
    assert kq_asr["van_ban"] == "Xin chào đất nước Việt Nam"
    assert kq_asr["do_tin_cay"] >= 0.9

    # Kiểm tra đối sánh mẫu âm học trực tiếp trên tín hiệu sóng
    sig_a = tts.tong_hop("Mở đèn phòng khách")["tin_hieu"]
    sig_b = tts.tong_hop("Tắt điều hòa nhiệt độ")["tin_hieu"]
    asr.dang_ky_mau_giong(sig_a, "Mở đèn phòng khách")
    asr.dang_ky_mau_giong(sig_b, "Tắt điều hòa nhiệt độ")
    assert asr.nhan_dien(sig_a)["van_ban"] == "Mở đèn phòng khách"

    # 2. OCR Tiếng Việt (Otsu binarization + Line projection + RAG bridge)
    ocr = TrichXuatVanBanAnh()
    img_path = tmp_path / "hoa_don.png"
    ocr.tao_anh_van_ban("Hóa đơn tiền điện\nTổng thanh toán: 500000 VND", duong_dan_luu=img_path)

    rag = RAGPipeline()
    kq_ocr = ocr.nap_vao_rag(img_path, rag, ma_tai_lieu="hoa_don_1")
    assert kq_ocr["so_dong"] >= 1
    assert "500000 VND" in kq_ocr["van_ban"]
    assert "500000 VND" in rag.hoi("Tổng thanh toán hóa đơn là bao nhiêu?")["tra_loi"]


def test_cat3_structured_outputs_and_mcp_protocol():
    @dataclass
    class HoSoUngVien:
        ho_ten: str
        so_nam_kinh_nghiem: int
        ky_nang: list
        dat_yeu_cau: bool = True

    parser = PhanTichDauRa()
    # Kiểm tra sửa JSON lỗi (nháy đơn, dấu phẩy thừa, hằng số Python)
    broken_json = "```{'ho_ten': 'Nguyễn Văn A', 'so_nam_kinh_nghiem': '5 năm', 'ky_nang': ['Python', 'AI'], 'dat_yeu_cau': True,}```"
    ho_so = parser.phan_tich_theo_schema(broken_json, HoSoUngVien)
    assert isinstance(ho_so, HoSoUngVien)
    assert ho_so.ho_ten == "Nguyễn Văn A"
    assert ho_so.so_nam_kinh_nghiem == 5
    assert ho_so.dat_yeu_cau is True

    # SinhCoCauTruc với mô hình tùy chỉnh
    struct_gen = SinhCoCauTruc(
        mo_hinh=lambda p: '{"ten_san_pham": "Laptop AI", "gia_ban": 25000000, "con_hang": true}'
    )
    sp = struct_gen.sinh("Trích xuất thông tin sản phẩm", {"ten_san_pham": str, "gia_ban": int, "con_hang": bool})
    assert sp["ten_san_pham"] == "Laptop AI"
    assert sp["gia_ban"] == 25000000
    assert sp["con_hang"] is True

    # Giao thức MCP (MCPServer, MCPClient, MCPBridge)
    mcp_server = MCPServer(ten="tai-chinh-mcp")
    mcp_server.dang_ky_cong_cu(
        CongCu(
            ten="tinh_lai_kep",
            mo_ta="Tính lãi kép ngân hàng",
            ham_thuc_thi=lambda von=100.0, lai_suat=0.1: f"Tổng vốn: {float(von) * (1.0 + float(lai_suat)):.1f}",
        )
    )
    mcp_server.dang_ky_tai_nguyen("mcp://quy-dinh/lai-suat", "Quy định lãi suất", "Lãi suất cơ bản là 6.5%/năm")

    client = MCPClient(mcp_server)
    init_info = client.khoi_tao()
    assert init_info["serverInfo"]["name"] == "tai-chinh-mcp"
    assert len(client.danh_sach_cong_cu()) == 1
    assert "110.0" in client.goi_cong_cu("tinh_lai_kep", {"von": 100, "lai_suat": 0.1})
    assert "6.5%/năm" in client.doc_tai_nguyen("mcp://quy-dinh/lai-suat")

    # Gắn MCP vào TacTu
    agent = TacTu(llm=GPTModel(d_model=32, so_dau=2, d_ff=64, so_block=1))
    so_cc = MCPBridge.gan_mcp_vao_tac_tu(agent, client)
    assert so_cc == 1
    assert "tinh_lai_kep" in agent.cong_cu


def test_cat4_ragas_evaluation_and_vietnamese_datasets(tmp_path):
    lake = HoDuLieu(str(tmp_path / "lake"))
    bo_viquad = lake.tai_bo_mau("viquad")
    assert bo_viquad["so_ban_ghi"] == 3

    bo_hf = lake.tai_tu_huggingface("uit-nlp/vietnamese_students_feedback", che_do_offline=True)
    assert bo_hf["so_ban_ghi"] == 4

    rag = RAGPipeline()
    lake.nap_vao_rag(bo_viquad["ten_dataset"], rag)

    evaluator = DanhGiaRAG()
    bao_cao_rag = evaluator.danh_gia_pipeline(
        rag,
        [
            {
                "cau_hoi": "Ai đã phát triển mô hình PhoBERT?",
                "dap_an_chuan": "Nhóm nghiên cứu VinAI đã phát triển mô hình PhoBERT.",
            },
            {
                "cau_hoi": "Vịnh Hạ Long thuộc tỉnh nào?",
                "dap_an_chuan": "Vịnh Hạ Long thuộc tỉnh Quảng Ninh.",
            },
        ],
    )
    assert bao_cao_rag["so_mau"] == 2
    assert bao_cao_rag["do_trung_thuc"] > 0.4
    assert bao_cao_rag["do_phu_ngu_canh"] > 0.4
    assert bao_cao_rag["diem_ragas"] > 0.4


def test_cat5_xgboost_lightgbm_bayesian_tpe_and_shapley():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 4))
    y_cls = (X[:, 0] * 2.0 - X[:, 1] > 0).astype(int)
    y_reg = X[:, 0] * 3.5 - X[:, 1] * 1.5 + rng.normal(0, 0.1, 40)

    # 1. XGBoost & LightGBM cho PhanLoai và HoiQuy
    for alg in ("xgboost", "lightgbm"):
        clf = PhanLoai(thuat_toan=alg)
        clf.huan_luyen(X, y_cls)
        assert clf.danh_gia(X, y_cls) >= 0.75

        reg = HoiQuy(thuat_toan=alg)
        reg.huan_luyen(X, y_reg)
        assert reg.danh_gia(X, y_reg) < 5.0

    # 2. Tìm kiếm tham số Bayesian TPE
    tuner = TimKiemThamSo(so_fold=3, seed=42)
    kq_tpe = tuner.tim_kiem_bayesian(
        lop_mo_hinh=PhanLoai,
        pham_vi_tham_so={"thuat_toan": ["logistic", "knn"], "C": (0.1, 10.0)},
        X=X,
        y=y_cls,
        so_lan=6,
    )
    assert kq_tpe["thuat_toan_toi_uu"] == "bayesian_tpe"
    assert kq_tpe["diem_tot_nhat"] >= 0.7

    # 3. Giải thích mô hình bằng giá trị Shapley (SHAP)
    clf_main = PhanLoai(thuat_toan="logistic")
    clf_main.huan_luyen(X, y_cls)
    explainer = GiaiThichMoHinh(
        clf_main, X, y_cls, ten_dac_trung=["feat_quan_trong_1", "feat_quan_trong_2", "nhieu_1", "nhieu_2"]
    )
    shap_single = explainer.tinh_shapley(mau=X[0], so_hoan_vi=24)
    assert len(shap_single) == 4
    shap_global = explainer.tinh_shapley(so_hoan_vi=16)
    # Hai đặc trưng đầu có hệ số thật (2.0 và -1.0) phải có đóng góp SHAP lớn hơn nhiễu
    top_2_features = list(shap_global.keys())[:2]
    assert "feat_quan_trong_1" in top_2_features


def test_cat6_modern_open_source_models_and_16_vietnamese_datasets(tmp_path):
    # 1. Kiểm tra ModelConfig & ModelHub cho Qwen3.5, Qwen3.6, Qwen3.7, Qwen3.8, Gemma 4, Llama 4, DeepSeek-R1
    for preset in ("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8", "gemma4", "llama4", "deepseek-r1", "phi4"):
        cfg = ModelConfig.from_preset(preset, thu_nho_cpu=True)
        assert cfg.d_model == 64
        full_cfg = ModelConfig.from_preset(preset)
        assert full_cfg.so_tham_so > 100_000_000

    hub = ModelHub(str(tmp_path / "hub"))
    qwen_models = hub.danh_muc_open_source("qwen3.8")
    assert len(qwen_models) >= 4
    gemma_models = hub.danh_muc_open_source("gemma4")
    assert len(gemma_models) >= 4

    # Khởi tạo trực tiếp Qwen3.8 & Gemma 4 với BoTinhChinhMoHinh để fine-tune trên bộ dữ liệu suy luận CoT
    lake = HoDuLieu(str(tmp_path / "lake_rich"))
    ds_catalog = lake.danh_sach_bo_mau()
    assert len(ds_catalog) >= 16
    assert "vi_reasoning_cot" in ds_catalog
    assert "vi_phap_luat" in ds_catalog
    assert "vi_dpo_preference" in ds_catalog

    cot_data = lake.tai_bo_mau("vi_reasoning_cot")
    tuner = BoTinhChinhMoHinh(mo_hinh="qwen3.8", nhiem_vu="sft", dung_lora=True, che_do_offline=True)
    kq = tuner.huan_luyen(cot_data["du_lieu"], so_vong=2)
    assert kq["loss_cuoi"] > 0

    aug_data = lake.sinh_them_mau("vi_phap_luat", so_luong=12)
    assert aug_data["so_ban_ghi"] == 12

