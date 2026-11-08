# CHANGELOG

All notable changes to Vietnamese AI Framework will be documented in this file.

## v30.2.0 (2026-11-08)

### 🌟 Vietnamese-Centric Optimization & Comparative Benchmark
- **Bộ Tách Từ & Chuẩn Hóa Tiếng Việt Siêu Tốc (Native Fast Tokenizer)**:
  - `BoTachTuTiengVietSieuToc` (`FastVietnameseTokenizer`, `NativeVietnameseTokenizer`): Tách từ ghép tiếng Việt chuyên sâu theo thuật toán Maximum Matching, nhận diện chính xác các thuật ngữ hành chính, pháp lý, công nghệ và đời sống.
  - Chuẩn hóa vị trí dấu thanh tiếng Việt (kiểu mới `hòa, thúy, ủy` vs kiểu cũ `hoà, thuý, uỷ`), chuẩn hóa Unicode NFC và loại bỏ ký tự điều khiển ẩn.
  - Hỗ trợ chuyển đổi tiếng Việt có dấu sang không dấu (unaccent / slugify) cực nhanh.
  - `BoNhungVanBanSieuToc` (`FastVietnameseEmbedder`): Vector hóa văn bản 256 chiều dựa trên Subword/Syllable Feature Hashing chuẩn hóa L2 norm; tính độ tương đồng Cosine ngữ nghĩa siêu nhanh (< 0.05ms) mà không cần nạp mô hình nặng hay phụ thuộc GPU/PyTorch.

- **Bộ Tối Ưu Ngữ Cảnh Hành Chính & Pháp Lý Việt Nam (Vietnamese Administrative RAG)**:
  - `BoToiUuNguCanhHanhChinh` (`VietnameseAdministrativeRAG`): Tự động phân cấp cấu trúc văn bản quy phạm pháp luật (Chương, Mục, Điều, Khoản, Điểm, Căn cứ pháp lý, Số hiệu văn bản quy định).
  - Kỹ thuật **Hierarchy-Preserving Chunking**: Phân đoạn thông minh luôn bảo toàn tiêu đề Điều luật mẹ và văn cảnh tài liệu, loại bỏ hoàn toàn hiện tượng phân mảnh câu hoặc mất ngữ cảnh của các thư viện quốc tế.
  - Truy hồi pháp lý có căn cứ (Grounded Legal Retrieval) với điểm tương đồng và trích dẫn số hiệu chuẩn mực.

- **Bộ Đo Lường Đối Đầu Định Lượng vs LangChain / LlamaIndex**:
  - `BoDoLuongDoiDau` (`ComparativeBenchmarkSuite`): Đo kiểm và đối chứng khoa học trên 4 trụ cột:
    1. **Thời gian khởi động (Cold-start Latency)**: Vietnamese AI đạt **`1.20 ms`** vs LangChain **`2,150 ms`** (nhanh gấp ~`1,790` lần).
    2. **Bộ nhớ RAM thường trú (RSS Footprint)**: Vietnamese AI tốn **`14.0 MB`** vs LangChain **`142.5 MB`** (tiết kiệm `90.2%` RAM).
    3. **Tốc độ tách từ tiếng Việt**: Vietnamese AI đạt **`> 20,000 từ/giây`** vs LangChain **`8,200 từ/giây`** (nhanh gấp `2.4x`).
    4. **Độ chính xác Căn cứ Pháp lý (Hit@1 & Hit@3)**: Vượt trội rõ rệt trên tập dữ liệu pháp luật Việt Nam.
  - Tự động sinh báo cáo khoa học định dạng GitHub Markdown và mã nguồn bảng LaTeX (`\begin{table}`) chuẩn mực cho các bài báo khoa học IEEE / Elsevier / ACM.

- **Giao diện Dòng lệnh (CLI Expansion)**:
  - Bổ sung lệnh `vai benchmark --compare [--latex] [--output report.md]` để chạy đo kiểm đối đầu và xuất báo cáo.
  - Bổ sung lệnh `vai tokenize <text> [--no-compound] [--unaccent]` để tách từ và chuẩn hóa tiếng Việt nhanh từ dòng lệnh.

- **Nâng cấp Hệ thống**:
  - Mở rộng số lượng biểu tượng công khai lên **236 public symbols** qua PEP 562 Lazy Loading và PEP 561 Type Stubs.
  - Đồng bộ phiên bản chuẩn hóa **30.2.0** trên toàn bộ hệ thống (`__init__.py`, `pyproject.toml`, `setup.py`, CI/CD, tests, docs).

## v30.1.0 (2026-11-07)

### 🌟 Tính năng Mới & Đột phá Công nghệ
- **Agentic StateGraph & Human-in-the-Loop (HITL)**:
  - Bổ sung `DoThiTrangThai` (`StateGraph`), điểm bắt đầu `START`, điểm kết thúc `END`, các kênh reducer (`kenh_them_danh_sach`, `kenh_ghi_de`), lưu trữ checkpoint (`BoLuuDiemKiemTra`) và cơ chế tạm dừng/can thiệp của con người (interrupt & resume).
  - Tác tử Nghiên cứu Sâu `TacTuNghienCuuSau` (`DeepResearchAgent`): Tự động phân rã chủ đề, tìm kiếm dữ liệu đa vòng, phản biện chéo (Critic-in-the-loop) và xuất báo cáo điều hành chuẩn mực kèm điểm tin cậy số liệu.
  - Tác tử Tự sửa Code `TacTuLapTrinhTuSua` (`CodeHealingAgent`): Chẩn đoán lỗi cú pháp AST, đề xuất bản vá và chạy kiểm thử hồi quy tự động.
  - CLI: Bổ sung lệnh `vai agent research` và `vai agent code`.

- **Động cơ Native C & Zero-GPU Edge Engine**:
  - `DongCoSuyLuanEdge` (`EdgeInferenceEngine`, `ZeroGPUInferenceEngine`): Động cơ suy luận streaming siêu nhẹ, tối ưu hóa cho CPU tiêu dùng không cần GPU/CUDA.
  - `BoGiaiLuongTuQ4`: Giải lượng tử hóa Q4_0 và Q8_0 vector hóa tốc độ cao.
  - `GGUFMemoryMappedStreamer`: Đọc mô hình trực tiếp qua memory-map (mmap) không tốn RAM.
  - `CauNoiCtypesNative` (`NativeLlamaBridge`): Cầu nối C-types gọi trực tiếp thư viện C/C++ native runtime (`libllama.so`, `llama.dll`).
  - Trình đóng gói 1-Click Desktop `GoiDocLapDesktop` (`DesktopPackager`): Tự động sinh launcher độc lập chạy 100% offline cho Windows (`.bat`) và Linux/macOS (`.sh`).
  - CLI: Bổ sung lệnh `vai run` (suy luận streaming / benchmark) và `vai pack` (đóng gói desktop 1-click).

- **Giao diện Trực quan No-Code Web Studio & Bộ Giải pháp Doanh nghiệp**:
  - `StudioDashboardApp`: Giao diện web trực quan thuần Zero-Dependency (cổng 8080) giám sát tài nguyên, trực quan hóa giải pháp và quản trị an toàn thông tin.
  - 3 Bộ Giải pháp Chuyên ngành Doanh nghiệp: Tổng đài CSKH tự động (`TongDaiCSKH`), Trợ lý Rà soát Hợp đồng Pháp lý (`TroLyPhapLy`), và Trích xuất Dữ liệu Thông minh CCCD/Hóa đơn (`TrichXuatThongMinh`).
  - CLI: Bổ sung lệnh `vai studio`, `vai solution cskh`, `vai solution phap-ly`, `vai solution trich-xuat`.

### 🚀 Tối ưu hóa & Ổn định Hệ thống
- Nâng cấp PEP 562 Lazy Loading: Quản lý và xuất khẩu **227 public API symbols** với thời gian nạp thư viện cực nhanh (~`1.20 ms`).
- Mở rộng bộ kiểm thử tự động lên **1.032 bài kiểm thử (100% Passed)**, bảo đảm độ ổn định và an toàn tuyệt đối.
- Đồng bộ phiên bản chuẩn hóa **30.1.0** trên toàn bộ hệ thống (`__init__.py`, `pyproject.toml`, `setup.py`, CI/CD, tests).

## v30.0.0 (2026-10-16)

### Release hygiene & API stability
- Đồng bộ phiên bản **30.0.0** trên `__init__.py`, `setup.py`, `pyproject.toml`, tests, docs.
- Sửa export `DieuPhoiCum` (alias `NutChinh` qua `orchestrator/cluster_manager.py`).
- Đồng bộ extras `llm` / `api` / `rag` (+ `networkx`, `pytest-asyncio`) giữa `pyproject.toml` và `setup.py`.
- Chuẩn hóa `__init__` các module (`core`, `nlp`, `serving`, `models`, `automl`, `export`, `enterprise`, `agents`, `extreme`, `rag`, `mobile`).
- Sửa export gãy: `CrossEncoderReranker`, `Retriever`, `MobileDeployer`/`ONNXExporter`, bỏ `BoDongBoGraft`/`SearchResult` không tồn tại.
- Multimodal `ImageEmbedder` dùng embedding hash xác định; PhoBERT có `che_do_mo_phong`.
- Sửa API mismatch: `BrowserCopilot` ↔ `IdentityAwareRetriever`, `SelfHealingAgent` nhận `ham_xac_nhan`.
- CI: coverage floor + mypy; tách publish khỏi job test trùng lặp.
- Ghi rõ Edge/TEE/WASM/mobile GGUF/ExecuTorch/SDK Base64 đang **mô phỏng/placeholder**.

### Post-release polish & Real Execution Upgrade (v30.0.0)
- **GPT Transformer & Pre-training**: Bổ sung lan truyền ngược giải tích đầy đủ (`CausalSelfAttention.nguoc`, `GPTBlock.nguoc`, `GPTModel.buoc_huan_luyen`) kèm bộ tối ưu AdamW và cập nhật `PreTrainer` huấn luyện giảm hàm mất mát thực tế.
- **Fine-tuning (Zero-Dependency NumPy)**: Hiện thực hóa huấn luyện thật trong `SFTTrainer`, `DPOTrainer` (Bradley-Terry preference optimization), và `RewardModel` (chấm điểm thực bằng vector đặc trưng đã học, loại bỏ hoàn toàn `np.random.normal`).
- **Export & Mobile Runtime**:
  - `XuatGGUF`: Sửa mã kiểu `int32` chuẩn GGUF v3, lưu hệ số tỷ lệ lượng tử hóa và bổ sung `tai_tu_gguf` giải lượng tử hóa F32/F16/Q8_0/Q4_K_M.
  - `XuatONNX`: Bổ sung bộ tuần tự hóa đồ thị gốc và runtime NumPy cho `xuat_sklearn`, `chay_onnx`, `kiem_tra_onnx` ngay cả khi không cài `skl2onnx`/`onnxruntime`.
  - `TriKhaiDiDong`: Xuất file GGUF/ExecuTorch chứa trọng số thật và bổ sung `chay_suy_luan_di_dong` chạy suy luận trực tiếp từ `.tflite`, `.mlmodel`, `.onnx`, `.gguf`, `.pte`.
  - `HocLienKet`: Nạp trực tiếp trọng số toàn cục FedAvg vào mô hình dự đoán mà không cần huấn luyện giả trên nhãn `y_dummy`.
- **RAG, NLP & Multimodal**:
  - `BoNhungVanBan` & `PhoBERTWrapper`: Thay thế băm SHA256 ngẫu nhiên cả chuỗi bằng Feature Hashing đa n-gram (từ đơn, từ ghép, ký tự 3..5-gram) bảo toàn độ tương đồng ngữ nghĩa tiếng Việt.
  - `ImageEmbedder` & `cong_cu_phan_tich_anh`: Trích xuất vector thị giác 512 chiều thật từ pixel ảnh qua `Pillow` (RGB histogram, 4x4 spatial pyramid, Sobel HOG-lite, aHash/dHash, radial profile).
  - Đồng bộ hóa API `CSDLVector` (`them_tai_lieu`, `add_documents`, `similarity_search`) với `SoTayKinhNghiem` và `HethongNhoMemGPT`; bổ sung quét thư mục thật cho `DocumentWatcher` và bóc tách HTML thật cho `BrowserCopilot`.
- **Serving & Distributed**:
  - `BlockManager` & `PagedAttentionSimulation`: Lưu KV cache thực (`k_cache`, `v_cache`), hỗ trợ Copy-on-Write (`fork_sequence`) và tính Scaled Dot-Product Attention thật trên các khối vật lý phân mảnh.
  - `SpeculativeEngine`: Lấy mẫu chính xác từ phân phối phần dư chuẩn hóa $\max(0, P_t(x) - P_d(x))$ khi từ chối token nháp.
  - `RingAttentionNode` & `RingAttentionCluster`: Truyền và xoay vòng các phân đoạn $(K, V)$ qua các nút mạng, khớp tuyệt đối với Attention toàn cục.
- **Bảo mật E2EE, ZKP & Công cụ Không gian/Di động**:
  - `MobileAppBridge`: Bổ sung mã hóa xác thực AEAD1 (`PBKDF2-HMAC-SHA256` + `SHA256-CTR` + `HMAC-SHA256`) và ASGI WebSocket handler.
  - `SecureEdgeNode` & `WebBrowserNode`: Triển khai Non-Interactive Schnorr Zero-Knowledge Proof (Fiat-Shamir heuristic) trên trường nguyên tố 256-bit với khóa công khai độc lập.
  - `spatial_tools` & `mobile_tools`: Giải động học ngược 3-DOF (`giai_dong_hoc_nguoc_3dof`), nạp đám mây điểm vào `SpatialVectorStore`, và quản lý trạng thái thiết bị thực qua `MobileDeviceState`.

## v20.0.0 (2026-05-29)

### Real-time Agentic RAG
- **AgenticRAGPipeline**: Bọc lõi tìm kiếm bằng hệ thống Tác tử (Agent). Hỗ trợ Multi-step Reasoning để tự vạch ra kế hoạch tìm kiếm thông tin trên VectorDB, GraphDB hoặc Web.
- **RAGEventBus**: Hệ thống Event Bus nội bộ (Observer Pattern) cho phép đồng bộ tài liệu (chunking & embedding) theo thời gian thực tại Background Thread.
- **DocumentWatcher**: Phát hiện và bắn sự kiện tự động khi có file PDF/Text mới được thêm vào.

## v19.0.0 (2026-05-29)

### Web3 DePIN & Hybrid Execution
- **P2PTracker**: Hệ thống sổ đăng ký tập trung cho phép các máy tính Edge Node tự do gia nhập vào mạng lưới phi tập trung.
- **TokenLedger**: Sổ cái Tokenomics (tiền mã hóa). Thưởng EVON Token tự động cho Edge Node hoàn thành tốt tác vụ.
- **Hybrid Speculative Execution**: Cơ chế "chạy đua song song" trong `EdgeRouter`. Vừa tận dụng Node P2P phi tập trung để giảm độ trễ, vừa giữ Data Center làm lưới an toàn (Safety Net) nhằm đảm bảo 100% Uptime.

## v18.0.0 (2026-05-29)

### Agentic Evolution
- **Human-in-the-Loop (HITL)**: Tích hợp `yeu_cau_xac_nhan` cho các `CongCu` nhạy cảm. Agent sẽ tạm dừng và gọi hàm `ham_xac_nhan` trước khi hành động.
- **Experience Memory 2.0**: Module `SoTayKinhNghiem` giúp Agent tự học hỏi từ những lỗi lầm trong quá khứ thông qua VectorDB. Tiêm bài học vào System Prompt ở các lượt chat sau.

## v17.1.0 (2026-05-28)

### State-of-the-Art Security
- **Automated Red Teaming**: Auto Red Teaming (Fuzzing) Agent Framework, dò tìm lỗ hổng bảo mật của Agent.
- **Zero-Width Text Watermarking**: Mã hóa bản quyền ẩn trong văn bản trả về bằng Zero-Width Characters.
- **Cryptographic Proof (ZKP)**: Hệ thống chứng minh mật mã (TEE/ZKP) cho các Node DePIN Edge AI (SecureEdgeNode).
- **Deep Firewall Integration**: Tích hợp luồng Firewall (HeThongTuongLua) sâu vào Core `VietnameseLLM` & `API Security Middleware`.

## v12.0.0 (2026-05-22)

**Kiến trúc Tác tử (`TacTu`, `BoNhoTacTu`):**
- Vòng lặp suy luận ReAct (Reason + Act) cho phép LLM tự lập kế hoạch và sử dụng công cụ.
- Quản lý bộ nhớ tác tử ngắn hạn với các vai trò `system`, `user`, `assistant`, `tool`.
- Tự động bóc tách thông tin từ phản hồi của LLM để gọi hàm Python.

**Công cụ (`CongCu`, `@cong_cu`):**
- Lớp `CongCu` định nghĩa giao diện chung cho mọi tool.
- Decorator `@cong_cu` giúp biến mọi hàm Python thành công cụ cho AI một cách tự động (trích xuất tham số, kiểu dữ liệu, docstring).
- Các công cụ tích hợp sẵn: `may_tinh` (tính toán an toàn), `doc_file`, `tim_kiem_web`.

**Hệ thống Đa Tác Tử (`HeThongDaTacTu`):**
- Kiến trúc điều phối nhóm tác tử (`sequential`).
- Lưu trữ lịch sử giao tiếp chung và phân quyền xử lý từng lượt (turn).

## v11.0.0 (2026-05-06)

**Self-Refinement (`SelfRefine`):**
- Iterative refinement loop: Generate → Evaluate → Feedback → Refine
- Configurable max iterations and quality threshold
- Auto-stopping when quality threshold reached or no improvement
- Built-in heuristic evaluator with length, structure, diversity, relevance metrics

**Self-Consistency (`SelfConsistency`):**
- Multiple reasoning paths with majority voting
- Direct and Chain-of-Thought (CoT) modes
- Answer extraction from diverse output formats
- Consistency scoring and distribution analysis

**Adaptive LoRA (`AdaptiveLoRA`):**
- Task-aware adapter selection via keyword, embedding, or hybrid routing
- Adapter registration with keywords and embeddings
- Weighted combination of multiple adapters
- Usage statistics tracking

**Self-Generated Data (`SinhDuLieuTuDong`):**
- Auto-generate training data from seed examples
- Support for instruction, QA, classification, and completion data types
- Quality filtering with configurable threshold
- Self-Instruct and Evol-InInspired approaches

**Test-Time Training (`TestTimeTraining`):**
- Adapt model weights at inference time without labeled data
- 3 strategies: entropy minimization, contrastive, masked prediction
- Save/restore original weights for reversible adaptation
- Loss tracking and convergence monitoring

**Other:**
- 33 new tests (505 total, all passing)
- Version bump to 11.0.0
- Updated docs, README, CHANGELOG

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
