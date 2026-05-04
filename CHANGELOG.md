# CHANGELOG

All notable changes to Vietnamese AI Framework will be documented in this file.

## v10.0.0 (2026-05-04)

### RAG Pipeline
- **CSDLVector**: In-memory vector database with cosine/L2/inner_product similarity, save/load, metadata filtering
- **CatVanBan**: Text chunking with 4 strategies (sentence, paragraph, word, character), sliding window with overlap
- **TrichXuat**: Hybrid retriever combining semantic search, keyword search (BM25-like), and hybrid mode
- **SapXepLai**: Reranker with MMR (Maximal Marginal Relevance), cross-encoder, keyword, and position-aware modes
- **RAGPipeline**: End-to-end RAG pipeline: chunk → embed → retrieve → rerank → generate

### Serving & Streaming
- **MayChuBatch**: Dynamic batching server with multi-worker support, latency tracking, concurrent request handling
- **MayChuStream**: SSE (Server-Sent Events) streaming server for LLM responses, token-by-token generation
- **BoGioiHanTocDo**: Rate limiter with token bucket and sliding window algorithms, per-client limiting

### Prompt Engineering
- **MauPrompt**: Prompt templates with {{variable}} interpolation, 6 built-in Vietnamese templates
- **ChuoiPrompt**: Chain-of-thought, few-shot prompting, sequential prompt chains with conditional branching
- **LuongAnToan**: Output guardrails: content filtering, PII detection (phone/email/CMND), format validation
- **PhanTichDauRa**: Structured output parser: JSON, markdown tables, code blocks, key-value pairs

### Vietnamese NLP Extensions
- **NhanDienThucThe**: Named Entity Recognition with regex patterns, dictionary matching, underthesea integration (8 entity types)
- **HoiDapTiengViet**: Extractive question answering with TF-IDF ranking over knowledge base
- **TomTatVanBan**: Text summarization: extractive (TF-IDF + position scoring) and abstractive (LLM-based)
- **DichThuat**: Translation system: dictionary-based, LLM-based, and hybrid mode (en/vi)
- **KiemTraChinhTa**: Vietnamese spell checker with edit distance suggestions, tone mark handling, corpus learning

### Model Compression
- **HocRutGon**: Knowledge Distillation with soft label transfer, temperature scaling, ensemble distillation
- **CatTiaMoHinh**: Model pruning: magnitude-based, structured, iterative, and random pruning strategies

### Production Hardening
- **KiemTraSucKhoe**: Health check system with readiness/liveness probes, CPU/memory/disk monitoring
- **MachCat**: Circuit breaker pattern with 3 states (closed/open/half-open), auto-recovery, fallback functions
- **LoggerCauTruc**: Structured JSON logging with timing context manager, request/prediction logging
- **QuanLyMetrics**: Metrics collection: counter, gauge, histogram with Prometheus/OpenTelemetry export
- **LamNongModel**: Model warm-up manager with pre-loading, auto-refresh, pool management

### Infrastructure
- Added `.pre-commit-config.yaml` (ruff, black, mypy, pre-commit-hooks)
- Added `AGENTS.md` with development commands and code conventions
- Added `py.typed` marker for PEP 561 compliance
- Added GitHub Issue templates (bug report, feature request)
- Version bump to 10.0.0
- 96 new tests (472 total, all passing)

## v9.0.0 (2026-05-04)

### Vietnamese LLM Configs, LM Eval Harness & Benchmarks

**Model Configs (`ModelConfig`):**
- Pre-defined presets: vnlm-tiny (10M), vnlm-small (125M), vnlm-medium (350M), vnlm-large (1.3B), vnlm-xl (2.7B), vnlm-7b (6.8B)
- Parameter counting (so_tham_so, so_tham_so_str)
- Config serialization (to_dict, from_dict)

**LM Evaluation Harness (`LMEvalHarness`):**
- Multi-task evaluation framework (5 built-in tasks)
- Custom task registration
- Perplexity, classification, generation, QA, cloze evaluation
- Few-shot support
- Result aggregation and reporting

**Benchmark Runner (`BenchmarkRunner`):**
- Vietnamese-specific benchmarks (corpus, QA, sentiment)
- Perplexity, generation quality, sentiment accuracy
- Inference speed benchmarking (latency p50/p95/p99)
- Comprehensive report generation

## v8.0.0 (2026-05-04)

### GPT Pre-training từ đầu

**GPT Model (`GPTModel`):**
- GPT-style decoder-only architecture (NumPy)
- Causal self-attention with masking
- GELU activation, pre-norm LayerNorm
- Sinusoidal positional encoding
- Weight tying (embedding ↔ output projection)
- Autoregressive text generation (top-k, nucleus sampling)
- Cross-entropy loss computation

**Pre-training Trainer (`PreTrainer`):**
- Causal language modeling training loop
- Learning rate warmup + cosine decay
- Periodic evaluation on validation set
- Callback system

**Text Dataset (`TextDataset`):**
- Character-level vocabulary building
- Sliding window chunking
- Train/val split
- Batch iteration with input_ids/targets
- Encode/decode functions

## v7.0.0 (2026-05-04)

### SFT, DPO & RLHF Training Pipeline

**SFT Trainer (`SFTTrainer`):**
- Supervised Fine-Tuning training loop
- Cross-entropy loss on instruction data
- Gradient accumulation and clipping
- Learning rate warmup + cosine decay
- Validation evaluation

**DPO Trainer (`DPOTrainer`):**
- Direct Preference Optimization
- Bradley-Terry preference loss
- KL divergence regularization (beta parameter)
- Label smoothing support
- Reward margin tracking

**Reward Model (`RewardModel`):**
- Reward model training from preference pairs
- Score normalization (mean/std)
- Preference accuracy tracking
- Evaluation with normalized scores

**RLHF Pipeline (`RLHFPipeline`):**
- Full RLHF: SFT → Reward Model → DPO
- Modular design (chạy từng bước hoặc full pipeline)
- Training history aggregation
- End-to-end results reporting

## v6.0.0 (2026-05-04)

### PEFT & Instruction Tuning

**PEFT Config (`PEFTConfig`):**
- Unified configuration cho LoRA, QLoRA, Prefix Tuning, Prompt Tuning
- Preset constructors: `PEFTConfig.lora()`, `PEFTConfig.qlora()`
- Validation và serialization (to_dict, from_dict)

**LoRA PEFT (`LoRAPeft`):**
- Enhanced LoRA integration với PyTorch nn.Module
- Auto-detect và apply LoRA lên Linear layers
- Merge/Unmerge weights
- Save/Load adapter weights only
- Trainable parameter statistics

**Instruction Tuning Trainer (`InstructionTuningTrainer`):**
- Support Alpaca và ShareGPT formats
- Gradient accumulation với warmup + cosine decay
- PyTorch và NumPy backends
- Callback system

## v5.0.0 (2026-05-04)

### PyTorch Trainer, Unsloth, GGUF, HuggingFace, Transformer, Tokenizer

- PyTorch Trainer (GPU, Mixed Precision, Gradient Accumulation, Early Stopping)
- Unsloth Integration (2x faster fine-tune, LoRA/QLoRA)
- GGUF Export/Import (quantize Q4_K, Q5_K, Q8_0)
- HuggingFace Integration (load pretrained, fine-tune, push to Hub)
- Transformer Architecture (Multi-Head Attention, Encoder, Decoder)
- Vietnamese Tokenizer (BPE, WordPiece)

## v4.0.0 (2026-05-04)

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
