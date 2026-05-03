# CHANGELOG

All notable changes to Vietnamese AI Framework will be documented in this file.

## v4.0.0 (2026-05-04)

### Cloud Platform, No-code Studio & Vietnamese LLM

**Cloud Platform SaaS (`NenTangDichVu`):**
- Multi-tenant workspace management (create, delete, upgrade plans)
- API key generation and authentication
- Model registry per workspace
- Model deployment management
- Usage tracking (API calls, predictions, storage)
- Quota enforcement per plan (free/starter/pro/enterprise)
- Data persistence to JSON

**No-code Studio (`StudioKeoTha`):**
- Visual node-based pipeline builder (6 node types: data, preprocessing, model, evaluate, visual, export)
- Node connection management (connect, disconnect)
- Topological sort execution engine
- 3 built-in templates (classification, regression, model comparison)
- Pipeline save/load to JSON
- Canvas state management

**Vietnamese LLM (`VietnameseLLM`):**
- N-gram language model (bigram, trigram, 4-gram)
- Text generation with temperature control
- Multi-choice text completion
- Perplexity calculation
- Next-word prediction (top-k)
- Template-based generation (5 built-in templates)
- Model save/load to JSON

- Thêm 71 test cases mới (tổng: 230 tests)
- Version bump: 4.0.0

## v3.0.0 (2026-05-04)

### Mobile/Edge, NAS, Federated Learning & Real-time Pipeline

**Mobile/Edge Deployment (`TriKhaiDiDong`):**
- Export models to TFLite, CoreML, ONNX Mobile formats
- INT8 quantization with size reduction metrics
- Edge benchmark (latency, throughput, model size)
- Mobile deployment config generation
- Custom binary format with magic bytes

**Neural Architecture Search (`TimKiemKienTruc`):**
- Random Search over MLP architecture space
- Grid Search for small architecture spaces
- Early stopping with minimum score threshold
- Multi-objective optimization (accuracy vs complexity)
- Comparison with traditional ML algorithms

**Federated Learning (`HocLienKet`):**
- FedAvg algorithm (weighted parameter averaging)
- Differential Privacy (Gaussian noise injection)
- Client sampling per round
- Data shuffling and IID partitioning
- Per-round history tracking

**Real-time ML Pipeline (`PipelineThoiGianThuc`):**
- In-memory Message Queue (Pub/Sub pattern)
- Feature Store with sliding window
- Latency tracking (p50, p95, p99, max)
- Single and batch prediction
- Callback system for real-time notifications

- Thêm 69 test cases mới (tổng: 159 tests)
- Version bump: 3.0.0

## v2.0.0 (2026-05-03)

### Ecosystem Release

**Multi-GPU & Distributed Training:**
- `MultiGPUTrainer`: DataParallel, gradient accumulation, mixed precision (FP16)
- `PhanTanHuanLuyen`: Multiprocessing data parallelism, parameter averaging, benchmark

**Model Hub:**
- `ModelHub`: Community model sharing, search, rating, download tracking

**Plugin System:**
- `PluginManager`: Register plugins, hook system (pre_train, post_train, pre_predict, post_predict)

**Cloud Deployment:**
- `CloudDeployment`: Auto-generate Dockerfile, docker-compose, AWS SageMaker, GCP Vertex AI, Azure ML configs

**Marketplace:**
- `Marketplace`: Share models, datasets, pipelines. Search by category/tags, rating system

**Enterprise:**
- `HeThongXacThuc`: RBAC (admin/developer/viewer), token auth, API keys
- `NhatKyHoatDong`: Audit log, search by user/action/date, statistics

- Thêm 17 test cases (tổng: 90 tests)

## v1.2.0 (2026-05-03)

### Production Tools

- **No-code Web Interface** (`UngDungWeb`): Upload CSV, chọn thuật toán, huấn luyện, dự đoán
- **Model Registry** (`QuanLyMoHinh`): Versioning, promote (staging -> production), so sánh metrics
- **Streaming Data** (`XuLyStream`): Rolling statistics, anomaly detection, callback
- **ONNX Export** (`XuatONNX`): Xuất mô hình scikit-learn sang ONNX format
- Thêm 7 test cases (tổng: 73 tests)

## v1.1.0 (2026-05-03)

### Deep Learning & Vision

- **Deep Learning** (`MangSau`): PyTorch backend + NumPy fallback, GPU support
- **Image Classification** (`PhanLoaiHinhAnh`): CNN + feature extraction fallback
- **Time Series** (`DuDoanChuoiThoiGian`): 4 phương pháp (Moving Average, Exponential, Linear Trend, Window Regression)
- **Pre-trained NLP** (`PhoBERTWrapper`): Wrapper cho PhoBERT tiếng Việt
- **Custom Layers**: `LopDense`, `LopDropout`, `LopBatchNorm`
- Thêm 9 test cases (tổng: 66 tests)

## v1.0.0 (2026-05-03)

### Production Release

- Phiên bản chính thức đầu tiên
- 57 test cases PASSED
- Hỗ trợ Python 3.8 - 3.13
- CI/CD với GitHub Actions
- PyPI publishing (`pip install vietnamese-ai`)
- MkDocs documentation site
- pyproject.toml (PEP 621)
- Makefile cho development workflow

## v0.3.0 (2026-05-03)

### NLP & AutoML

- **Word2Vec tiếng Việt** (`Word2VecTiengViet`): Skip-gram + Negative Sampling
- **FastText tiếng Việt** (`FastTextTiengViet`): Character n-gram, OOV handling
- **Sentiment Analysis** (`PhanTichCamXuc`): underthesea, từ điển, tự huấn luyện
- **Model Interpretability** (`GiaiThichMoHinh`): Feature Importance, Permutation, LIME
- **Experiment Tracking** (`TheoDoiThiNghiem`): MLflow compatible API
- **AutoML** (`AutoML`): Tự động chọn mô hình tốt nhất
- **Data Augmentation** (`TangCuongVanBan`): 5 kỹ thuật tăng cường văn bản tiếng Việt

## v0.2.0 (2026-05-02)

### Core Features

- **underthesea integration**: Tách từ, POS tagging, sentiment
- **Cross-validation** (`KiemDinhCheo`): K-Fold, Stratified K-Fold
- **Hyperparameter Tuning** (`TimKiemThamSo`): GridSearch, RandomSearch
- **Save/Load Pipeline**: `pipe.luu()` / `Pipeline.tai()`
- **CLI tool**: `vai info/train/predict/evaluate/serve`
- **Docker**: Dockerfile + docker-compose.yml

## v0.1.0 (2026-05-02)

### Initial Release

- **Models**: PhanLoai (7 algorithms), HoiQuy (8), PhanCum (3), MangNron, MoHinhTapHop (3)
- **Preprocessing**: XuLyVanBan, XuLySo, TaoDacTrung
- **Core**: Engine, Pipeline
- **Utils**: Logger, Metrics, Validator, LuuTai
- **Datasets**: DuLieuMau
- **API**: ServerDonGian
- 45 test cases
