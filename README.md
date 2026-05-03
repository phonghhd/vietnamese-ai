<p align="center">
  <h1 align="center">Vietnamese AI Framework</h1>
  <p align="center">
    <strong>Framework AI thuần tiếng Việt cho Python</strong>
  </p>
  <p align="center">
    <em>Học máy đơn giản. API tiếng Việt. Production-ready.</em>
  </p>
</p>

<p align="center">
  <a href="#cài-đặt">Cài đặt</a> •
  <a href="#tính-năng">Tính năng</a> •
  <a href="#sử-dụng-nhanh">Sử dụng nhanh</a> •
  <a href="#cli">CLI</a> •
  <a href="#docker">Docker</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#đóng-góp">Đóng góp</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-9.0.0-blue.svg" alt="version">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license">
  <img src="https://img.shields.io/badge/tests-376%2F376-brightgreen.svg" alt="tests">
  <img src="https://img.shields.io/badge/language-Vietnamese-red.svg" alt="language">
</p>

---

## Giới thiệu

**Vietnamese AI Framework** là framework học máy mã nguồn mở, được thiết kế đặc biệt cho cộng đồng phát triển AI tại Việt Nam. Toàn bộ API, thông báo lỗi, tài liệu và ví dụ đều sử dụng tiếng Việt.

### Tại sao lại là Vietnamese AI?

| Vấn đề hiện tại | Giải pháp của chúng tôi |
|---|---|
| Framework AI lớn quá phức tạp cho người mới | API đơn giản, học trong 5 phút |
| Tài liệu toàn bằng tiếng Anh | Tài liệu và ví dụ 100% tiếng Việt |
| Thiếu toolkit xử lý văn bản tiếng Việt | Tích hợp sẵn underthesea + TF-IDF + stopwords |
| Không có framework "all-in-one" cho người Việt | Models + Preprocessing + CV + Tuning + API + CLI + Docker + SaaS + Studio + LLM |

---

## Cài đặt

### Cơ bản

```bash
pip install vietnamese-ai
```

### Từ source

```bash
git clone https://github.com/phonghhd/vietnamese-ai.git
cd vietnamese-ai
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Với NLP tiếng Việt (underthesea)

```bash
pip install -e ".[nlp]"
```

### Với Deep Learning (PyTorch)

```bash
pip install -e ".[torch]"
```

### Tất cả tính năng

```bash
pip install -e ".[all]"
```

---

## Tính năng

### Models

| Mô hình | Thuật toán |
|---|---|
| `PhanLoai` | logistic, knn, svm, cay_quyet_dinh, rung_ngau_nhien, gradient_boosting, naive_bayes |
| `HoiQuy` | tuyen_tinh, ridge, lasso, elastic_net, svm, cay_quyet_dinh, rung_ngau_nhien, gradient_boosting |
| `PhanCum` | kmeans, dbscan, hierarchical |
| `MangNron` | Custom MLP (ReLU, Sigmoid, Tanh) - không cần TensorFlow/PyTorch |
| `MoHinhTapHop` | voting, bagging, boosting |

### Core

| Module | Chức năng |
|---|---|
| `Engine` | Điều phối huấn luyện, đánh giá, quản lý lịch sử |
| `Pipeline` | Chuỗi tiền xử lý + mô hình, hỗ trợ save/load |
| `KiemDinhCheo` | K-Fold, Stratified K-Fold, Repeated K-Fold |
| `TimKiemThamSo` | GridSearch, RandomSearch |

### NLP & Embeddings

| Module | Chức năng |
|---|---|
| `Word2VecTiengViet` | Word2Vec Skip-gram/CBOW tự cài đặt |
| `FastTextTiengViet` | Character n-gram embeddings (xử lý OOV) |
| `PhanTichCamXuc` | Sentiment: underthesea, từ điển, tự huấn luyện |

### AutoML & Experiment Tracking

| Module | Chức năng |
|---|---|
| `AutoML` | Tự động chọn mô hình + thuật toán tốt nhất |
| `TimKiemKienTruc` | Neural Architecture Search (NAS) - tìm kiến trúc MLP tối ưu |
| `TheoDoiThiNghiem` | Theo dõi thí nghiệm (tương thích MLflow) |

### Mobile/Edge & Federated Learning

| Module | Chức năng |
|---|---|
| `TriKhaiDiDong` | Triển khai TFLite, CoreML, ONNX Mobile + quantization INT8 |
| `HocLienKet` | Federated Learning (FedAvg, Differential Privacy, Client Sampling) |

### Real-time ML Pipeline

| Module | Chức năng |
|---|---|
| `PipelineThoiGianThuc` | Real-time pipeline: Message Queue, Feature Store, latency tracking |

### Cloud Platform & No-code Studio

| Module | Chức năng |
|---|---|
| `NenTangDichVu` | SaaS platform: multi-tenant workspace, API keys, quota, model deploy |
| `StudioKeoTha` | No-code Studio: kéo thả pipeline, templates, save/load JSON |

### Vietnamese LLM

| Module | Chức năng |
|---|---|
| `VietnameseLLM` | N-gram language model: text generation, completion, perplexity, templates |

### PyTorch Training & Fine-tuning (Phase 7-8)

| Module | Chức năng |
|---|---|
| `HuanLuyenPyTorch` | PyTorch trainer: GPU, mixed precision, gradient accumulation, early stopping, checkpoint |
| `UnslothWrapper` | Unsloth integration: 2x faster fine-tune, LoRA, QLoRA, quantization |
| `XuatGGUF` | GGUF export/import: Q4_K, Q5_K, Q8_0 quantization cho llama.cpp |
| `HuggingFaceWrapper` | HuggingFace: load pretrained, fine-tune, push to Hub |
| `TransformerModel` | Transformer: Multi-Head Attention, Encoder, Decoder |
| `VietnameseTokenizer` | Tokenizer: BPE, WordPiece cho tiếng Việt |

### PEFT & Instruction Tuning (Phase 6)

| Module | Chức năng |
|---|---|
| `PEFTConfig` | Cấu hình LoRA, QLoRA, Prefix Tuning, Prompt Tuning |
| `LoRAPeft` | Enhanced LoRA PEFT: auto-detect layers, merge/unmerge, save/load adapter |
| `InstructionTuningTrainer` | Instruction tuning: Alpaca/ShareGPT, warmup + cosine decay |

### SFT, DPO & RLHF (Phase 7)

| Module | Chức năng |
|---|---|
| `SFTTrainer` | Supervised Fine-Tuning: cross-entropy, gradient accumulation, mixed precision |
| `DPOTrainer` | Direct Preference Optimization: Bradley-Terry loss, beta annealing |
| `RewardModel` | Reward model: Bradley-Terry training, score normalization |
| `RLHFPipeline` | Full RLHF: SFT → Reward Model → DPO end-to-end |

### GPT Pre-training (Phase 8)

| Module | Chức năng |
|---|---|
| `GPTModel` | GPT decoder-only: causal attention, GELU, pre-norm, weight tying |
| `PreTrainer` | Pre-training trainer: causal LM, warmup + cosine, checkpointing |
| `TextDataset` | Text dataset: sliding window, character-level vocab, batch iterator |

### Vietnamese LLM & Evaluation (Phase 9)

| Module | Chức năng |
|---|---|
| `ModelConfig` | Model presets: tiny(10M), small(125M), medium(350M), large(1.3B), xl(2.7B), 7B |
| `LMEvalHarness` | LM evaluation: perplexity, classification, generation, QA, cloze, few-shot |
| `BenchmarkRunner` | Benchmark runner: perplexity, generation, sentiment, speed, QA |

### Interpretability & Augmentation

| Module | Chức năng |
|---|---|
| `GiaiThichMoHinh` | Feature Importance, Permutation Importance, LIME |
| `TangCuongVanBan` | Tăng cường dữ liệu văn bản (đồng nghĩa, xóa từ, hoán vị, thiếu dấu) |

### Preprocessing

| Module | Chức năng |
|---|---|
| `XuLyVanBan` | Tách từ (underthesea), TF-IDF, stopwords, sentiment, POS tagging |
| `XuLySo` | Min-Max, Z-Score, missing values, one-hot, chia dữ liệu |
| `TaoDacTrung` | Polynomial, interaction, PCA, variance selection |

### CLI

```
vai info                              Thông tin framework
vai train --data FILE --model M       Huấn luyện mô hình
vai predict --model M --input FILE    Dự đoán
vai evaluate --model M --data FILE    Đánh giá
vai serve --model M --port 8080       API server
vai web --port 5000                   No-code web interface
```

---

## Sử dụng nhanh

### Phân loại

```python
from vietnamese_ai import PhanLoai, XuLySo, DuLieuMau

X, y = DuLieuMau.phan_loai_don_gian(so_mau=400)
X_train, X_test, y_train, y_test = XuLySo.chia_du_lieu(X, y)

pl = PhanLoai(thuat_toan="rung_ngau_nhien")
pl.huan_luyen(X_train, y_train)
print(pl.bao_cao(X_test, y_test))
```

### Cross-Validation

```python
from vietnamese_ai import KiemDinhCheo, PhanLoai

kdc = KiemDinhCheo(so_fold=5)
ket_qua = kdc.chay(PhanLoai(thuat_toan="logistic"), X, y)
print(f"Điểm CV: {ket_qua['diem_trung_binh']:.4f} (+/- {ket_qua['do_lech_chuan']:.4f})")
```

### AutoML

```python
from vietnamese_ai import AutoML

auto = AutoML(so_fold=5)
ket_qua = auto.fit(X_train, y_train)
print(f"Mô hình tốt nhất: {ket_qua['thuat_toan_tot_nhat']}")
du_doan = auto.predict(X_test)
```

### Pipeline + Save/Load

```python
from vietnamese_ai import Pipeline, XuLySo, PhanLoai

pipe = Pipeline()
pipe.them_buoc("chuan_hoa", XuLySo())
pipe.them_buoc("phan_loai", PhanLoai(thuat_toan="gradient_boosting"))
pipe.fit(X_train, y_train)

pipe.luu("models/pipe.pkl")
pipe2 = Pipeline.tai("models/pipe.pkl")
du_doan = pipe2.predict(X_test)
```

### Xử lý văn bản tiếng Việt

```python
from vietnamese_ai import XuLyVanBan

xl = XuLyVanBan()
xl.tach_tu("Trí tuệ nhân tạo rất hay")
# → ['trí_tuệ_nhân_tạo', 'rất', 'hay']

xl.phan_tich_cam_xuc("Sản phẩm rất tốt, tôi rất hài lòng")
# → 'positive'

tfidf = xl.ma_hoa_tfidf(["văn bản 1", "văn bản 2"])
```

### No-code Studio

```python
from vietnamese_ai import StudioKeoTha

studio = StudioKeoTha()
studio.tai_template("phan_loai_co_ban")
ket_qua = studio.chay()
print(ket_qua['trang_thai'])  # 'thanh_cong'
```

### Vietnamese LLM

```python
from vietnamese_ai import VietnameseLLM

llm = VietnameseLLM(bac=3)
llm.huan_luyen(cac_van_ban, so_vong=5)
van_ban = llm.sinh_van_ban("học máy là", do_dai=50)
goi_y = llm.lay_tu_ke_tiep("trí tuệ nhân", top_n=5)
```

### Federated Learning

```python
from vietnamese_ai import HocLienKet, PhanLoai

hl = HocLienKet(so_client=5, so_vong=10)
ket_qua = hl.huan_luyen(PhanLoai, X, y, thuat_toan="logistic")
print(f"Điểm global: {ket_qua['diem_toan_cuc']:.4f}")
```

---

## CLI

Sau khi cài đặt, sử dụng lệnh `vai`:

```bash
vai info
vai train --data data.csv --model logistic --output model.pkl --test-size 0.2
vai predict --model model.pkl --input new_data.csv --output results.csv
vai evaluate --model model.pkl --data test.csv
vai serve --model model.pkl --port 8080
vai web --port 5000
```

---

## Docker

```bash
docker build -t vietnamese-ai .
docker run -p 8080:8080 -v ./models:/app/models vietnamese-ai serve --model /app/models/model.pkl --port 8080
```

### Docker Compose

```bash
mkdir -p data models
docker-compose run vai-train
docker-compose up vai-serve
```

---

## API Server

```bash
vai serve --model model.pkl --port 8080

curl -X POST http://localhost:8080/du_doan \
  -H "Content-Type: application/json" \
  -d '{"du_lieu": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'

curl http://localhost:8080/suc_khoe
```

---

## API Reference

### PhanLoai(thuat_toan, ten, **kwargs)

| Phương thức | Mô tả |
|---|---|
| `huan_luyen(X, y)` | Huấn luyện mô hình |
| `du_doan(X)` | Dự đoán nhãn |
| `du_doan_xac_suat(X)` | Dự đoán xác suất |
| `danh_gia(X, y)` | Độ chính xác (0-1) |
| `bao_cao(X, y)` | accuracy, precision, recall, f1 |
| `luu(duong_dan)` | Lưu mô hình |
| `BaseModel.tai(duong_dan)` | Tải mô hình |

### KiemDinhCheo(so_fold, lap_lai, seed)

| Phương thức | Mô tả |
|---|---|
| `chay(mo_hinh, X, y, chi_so)` | Chạy K-Fold CV |

### TimKiemThamSo(so_fold, seed)

| Phương thức | Mô tả |
|---|---|
| `tim_kiem_luoi(lop_mo_hinh, luoi_tham_so, X, y)` | Grid Search |
| `tim_kiem_ngau_nhien(lop_mo_hinh, pham_vi_tham_so, X, y, so_lan)` | Random Search |

### AutoML(so_fold, chuan_hoa)

| Phương thức | Mô tả |
|---|---|
| `fit(X, y)` | Tự động tìm mô hình tốt nhất |
| `predict(X)` | Dự đoán với mô hình tốt nhất |
| `danh_gia(X, y)` | Đánh giá |
| `bao_cao()` | Báo cáo so sánh thuật toán |

### Pipeline(ten)

| Phương thức | Mô tả |
|---|---|
| `them_buoc(ten, bo_xu_ly)` | Thêm bước |
| `fit(X, y)` | Huấn luyện pipeline |
| `predict(X)` | Dự đoán |
| `luu(duong_dan)` | Lưu pipeline |
| `Pipeline.tai(duong_dan)` | Tải pipeline |

### TriKhaiDiDong()

| Phương thức | Mô tả |
|---|---|
| `xuat_tflite(mo_hinh, duong_dan, kich_thuoc_dau_vao)` | Xuất TFLite |
| `xuat_coreml(mo_hinh, duong_dan, kich_thuoc_dau_vao)` | Xuất CoreML |
| `xuat_onnx_mobile(mo_hinh, duong_dan, kich_thuoc_dau_vao)` | Xuất ONNX Mobile |
| `luong_hoa_int8(duong_dan_goc, duong_dan_moi)` | Quantize INT8 |
| `benchmark_edge(mo_hinh, kich_thuoc_dau_vao, so_lan)` | Benchmark |

### HocLienKet(so_client, so_vong, ty_le_client, rieng_tu_differntial)

| Phương thức | Mô tả |
|---|---|
| `huan_luyen(lop_mo_hinh, X, y, **tham_so)` | Federated Learning |
| `du_doan(lop_mo_hinh, X, **tham_so)` | Dự đoán global model |
| `lay_lich_su()` | Lịch sử rounds |
| `bao_cao()` | Báo cáo FL |

### StudioKeoTha(ten)

| Phương thức | Mô tả |
|---|---|
| `them_node(loai, ten, vi_tri, tham_so)` | Thêm node |
| `ket_noi(tu_node, den_node)` | Kết nối nodes |
| `chay()` | Chạy pipeline |
| `tai_template(ten)` | Tải template |
| `luu(duong_dan)` | Lưu config |
| `StudioKeoTha.tai(duong_dan)` | Tải config |

### VietnameseLLM(bac, lam_mo)

| Phương thức | Mô tả |
|---|---|
| `huan_luyen(cac_van_ban, so_vong)` | Huấn luyện LLM |
| `sinh_van_ban(khoi_dau, do_dai, nhiet_do)` | Sinh văn bản |
| `hoan_thanh_cau(dau_vao, so_lua_chon)` | Hoàn thành câu |
| `tinh_perplexity(text)` | Tính perplexity |
| `lay_tu_ke_tiep(text, top_n)` | Gợi ý từ tiếp theo |
| `sinh_theo_template(ten, tham_so)` | Sinh theo template |
| `luu(duong_dan)` / `VietnameseLLM.tai(duong_dan)` | Save/Load |

### NenTangDichVu(duong_dan)

| Phương thức | Mô tả |
|---|---|
| `tao_workspace(ten, chu_so_huu, goi_dich_vu)` | Tạo workspace |
| `tao_api_key(ma_workspace)` | Tạo API key |
| `dang_ky_model(ma_workspace, ten, mo_hinh)` | Đăng ký model |
| `deploy_model(ma_workspace, ma_model)` | Deploy model |
| `du_doan(ma_workspace, ma_deployment, du_lieu)` | Dự đoán |
| `thong_ke_usage(ma_workspace)` | Thống kê sử dụng |

### XuLyVanBan(tu_dung, su_dung_underthesea)

| Phương thức | Mô tả |
|---|---|
| `tach_tu(text)` | Tách từ tiếng Việt (underthesea) |
| `phan_tich_cam_xuc(text)` | Sentiment: positive/negative/neutral |
| `gan_nhan_tu_loai(text)` | POS tagging |
| `ma_hoa_tfidf(cac_van_ban)` | TF-IDF matrix |
| `trich_xuat_tu_khoa(text, top_n)` | Keyword extraction |

---

## Cấu trúc thư mục

```
vietnamese-ai/
├── vietnamese_ai/
│   ├── core/              # Engine, Pipeline, KiemDinhCheo, TimKiemThamSo
│   ├── models/            # PhanLoai, HoiQuy, PhanCum, MangNron, MoHinhTapHop
│   ├── preprocessing/     # XuLyVanBan, XuLySo, TaoDacTrung
│   ├── automl/            # AutoML, TimKiemKienTruc (NAS)
│   ├── nlp/               # PhanTichCamXuc, PhoBERTWrapper
│   ├── embeddings/        # Word2VecTiengViet, FastTextTiengViet
│   ├── deep_learning/     # MangSau, LopDense, LopDropout
│   ├── fine_tuning/       # HuanLuyenPyTorch, UnslothWrapper, HuggingFaceWrapper, PEFT, SFT, DPO, RLHF
│   ├── transformer/       # MultiHeadAttention, TransformerModel, GPTModel, PreTrainer, TextDataset, VietnameseTokenizer
│   ├── vision/            # PhanLoaiHinhAnh
│   ├── timeseries/        # DuDoanChuoiThoiGian
│   ├── mobile/            # TriKhaiDiDong (TFLite, CoreML, ONNX Mobile)
│   ├── federated/         # HocLienKet (FedAvg, Differential Privacy)
│   ├── realtime/          # PipelineThoiGianThuc (Message Queue, Feature Store)
│   ├── saas/              # NenTangDichVu (Cloud Platform SaaS)
│   ├── studio/            # StudioKeoTha (No-code Studio)
│   ├── llm/               # VietnameseLLM, ModelConfig, LMEvalHarness, BenchmarkRunner
│   ├── streaming/         # XuLyStream (real-time processing)
│   ├── export/            # XuatONNX, XuatGGUF
│   ├── registry/          # QuanLyMoHinh (Model Registry)
│   ├── cloud/             # CloudDeployment, Marketplace
│   ├── distributed/       # PhanTanHuanLuyen, MultiGPUTrainer
│   ├── enterprise/        # HeThongXacThuc, NhatKyHoatDong
│   ├── hub/               # ModelHub
│   ├── plugins/           # PluginManager
│   ├── augmentation/      # TangCuongVanBan
│   ├── interpretability/  # GiaiThichMoHinh
│   ├── experiment_tracking/ # TheoDoiThiNghiem
│   ├── web/               # UngDungWeb (No-code Web UI)
│   ├── utils/             # Logger, Metrics, Validator, LuuTai
│   ├── visualization/     # BieuDo
│   ├── datasets/          # DuLieuMau
│   ├── api/               # ServerDonGian
│   └── cli/               # CLI (vai command)
├── tests/                 # 270 tests
├── examples/              # Ví dụ sử dụng
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── setup.py
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── MANIFEST.in
```

---

## Chạy test

```bash
# Chạy toàn bộ test
pytest tests/ -v

# Chạy với coverage
pytest tests/ -v --cov=vietnamese_ai --cov-report=html

# Chạy lint
ruff check vietnamese_ai/ tests/
```

```
============================= 376 passed ==============================
```

---

## Roadmap

### Đã hoàn thành (v1.0 - v9.0)

| Version | Tính năng |
|---|---|
| v1.0 | Models, Preprocessing, NLP, AutoML, CV, Tuning, Pipeline, CLI, Docker, API |
| v1.1 | Deep Learning, CNN, Time Series, PhoBERT |
| v1.2 | Web UI, Model Registry, Streaming, ONNX Export |
| v2.0 | Multi-GPU, Distributed, Model Hub, Plugin System, Cloud Deploy, Marketplace, Enterprise |
| v3.0 | Mobile/Edge, NAS, Federated Learning, Real-time Pipeline |
| v4.0 | Cloud Platform SaaS, No-code Studio, Vietnamese LLM |
| v5.0 | PyTorch Trainer, Unsloth, GGUF, HuggingFace, Transformer, Tokenizer |
| v6.0 | PEFTConfig, LoRAPeft, Instruction Tuning (Alpaca/ShareGPT) |
| v7.0 | SFT, DPO, Reward Model, RLHF Pipeline |
| v8.0 | GPT Decoder-Only Pre-training, TextDataset, PreTrainer |
| v9.0 | Vietnamese LLM Configs (125M-7B), LM Eval Harness, Benchmark Runner |

### Tech Stack tương lai

- **Fine-tuning**: Unsloth (2x nhanh), LoRA, QLoRA, PEFT
- **Quantization**: GGUF (llama.cpp), GPTQ, AWQ
- **Training**: DeepSpeed ZeRO, FSDP, Gradient Checkpointing
- **Inference**: vLLM, TGI, llama.cpp
- **Models**: Llama, Mistral, Qwen, Gemma, PhoGPT

---

## Đóng góp

Xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết quy trình đóng góp.

---

## Giấy phép

MIT License - xem [LICENSE](LICENSE)

---

<p align="center">
  <strong>Vietnamese AI Framework</strong> - Được xây dựng với ❤️ cho cộng đồng AI Việt Nam
</p>
