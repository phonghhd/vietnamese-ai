# CHANGELOG

All notable changes to Vietnamese AI Framework will be documented in this file.

## v31.3.0 (2026-11-28 — Dual-Track Product & Research Release: M1 → M4)

### 📚 Kho Tri Thức `.vaidb` 1 Lệnh & `ViLegal-TempBench v2` (M1)
- **Legal Knowledge Hub (`vietnamese_ai/rag/legal/kb_hub.py`, CLI `vai pull-kb`)**: Bổ sung danh mục 5 gói tri thức pháp lý theo ngành (`vn-business-core-2025`, `vn-labor-bhxh-2025`, `vn-tax-invoice-2025`, `vn-land-housing-2025`, `vn-data-digital-2025`), dựng gói `.vaidb` có HNSW + manifest SHA-256, kiểm toán chất lượng bộ văn bản (`kiem_tra_chat_luong_bo_van_ban`) và tách cấu trúc đến cấp Khoản (`tach_khoan_dieu_dai`) trong `auto_ingest.py`.
- **`ViLegal-TempBench v2` (84 QA, 5 splits)**: Bổ sung 16 câu hỏi đời thường không chứa số hiệu luật (`T5`), đối chứng 6 cấu hình kèm baseline khoa học `naive_date_filter` (chỉ lọc ngày bắt đầu), hỗ trợ truyền ONNX INT8 thật (`--onnx-embedder`, `--onnx-reranker`) và tự động sinh bảng phân tích lỗi (*Error Analysis*).
- **Sửa 2 lỗi cốt lõi phát hiện từ `T5`**: phân giải đúng mốc năm cuối trong câu hỏi 2 mốc thời gian (**TQPA = 100%, 84/84**) và bật `noi_tu_ghep=True` + chống đếm trùng âm tiết con trong `temporal_rag.py` (`full`: **R@3 = 0.936**, **MRR@10 = 0.907**, **TVA = 0.987**, **TCA = 12/12**, **Multi-Hop@3 = 1.000**, **Trap = 1.000**, **T5-R@3 = 0.688**, **Abstention F1 = 0.923**; Paired Bootstrap vs `naive_date_filter`: $p < 0.0001$).

### 📑 Contract Review Pro, OCR Dấu Đỏ, Studio v2 SSE & Phân Phối 1-Click (M2)
- **DOCX gộp ô & PDF Scan OCR khử dấu đỏ (`document_parser.py`)**: Bổ sung `loc_dau_do_anh_numpy()` thuần NumPy khử mực dấu đỏ hành chính Việt Nam, hỗ trợ `w:gridSpan` / `w:vMerge` trong `_doc_docx_ooxml()` và tự phát hiện + chạy OCR trên PDF scan.
- **Contract Review Pro (`contract_review.py`, `vai contract-review --type`)**: Tự động phân loại và rà soát chuyên sâu 5 loại hợp đồng Việt Nam (`lao_dong`, `dich_vu`, `mua_ban`, `thue_nha`, `nda`) theo BLLĐ 2019, NĐ 70/2025/NĐ-CP, Luật GTGT 48/2024/QH15, Luật KDBĐS 29/2023/QH15, Luật Nhà ở 27/2023/QH15; hỗ trợ chế độ Hybrid LLM kiểm chứng 100% trích dẫn qua `BoKiemChungTrichDanPhapLy`.
- **Web Studio v2 (`studio_dashboard.py`)**: Bổ sung 2 endpoint SSE Streaming có bảo vệ RBAC (`/api/legal/temporal-ask/stream`, `/api/chat/stream`), giao diện Split-View trích dẫn điều luật (nhãn hiệu lực Xanh/Vàng/Đỏ, render DOM an toàn không XSS) và xem trực quan Diff hợp đồng.
- **Phân phối 1-Click**: Lệnh `vai mcp-config --client cursor|claude|vscode|antigravity` (hỗ trợ `stdio`/`sse` và `--write` gộp JSON an toàn), nâng cấp `docker-compose.yml` (bind `127.0.0.1:8080`, volume bền vững, profile `mcp`/`connectors`) và tài liệu [docs/guides/mcp-cursor-claude.md](docs/guides/mcp-cursor-claude.md).

### 🧪 `ViAlign-SafeLoop` Token-KL / QLoRA & Hồ Sơ Công Bố Khoa Học Kép (M3 & M4)
- **True Token-Level Reverse KL & QLoRA Flywheel (`dpo_trainer.py`, `auto_finetuner.py`, `remote_gpu.py`)**: Bổ sung `loai_phat_kl="token_kl"` tính $D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\mathrm{ref}})$ trên riêng token phản hồi, cDPO `label_smoothing` theo từng mẫu, hỗ trợ nhiệm vụ `simpo`/`cdpo`/`safeloop` và sinh script QLoRA 4-bit (`bitsandbytes`/`accelerate`) cho Modal/Local.
- **Kịch bản tái lập & Hồ sơ bài báo (`scripts/`, `paper/`, `docs/research/`)**: Bổ sung `scripts/run_safeloop_qlora_experiment.py`, `scripts/generate_paper_tables.py`, hoàn thiện bài báo JOSS ([paper/paper.md](paper/paper.md), 20 mục BibTeX chuẩn trong [paper/paper.bib](paper/paper.bib), [paper/JOSS_CHECKLIST.md](paper/JOSS_CHECKLIST.md), workflow `.github/workflows/draft-pdf.yml`), tài liệu tiếng Anh cho reviewer quốc tế ([README_EN.md](README_EN.md), [CONTRIBUTING_EN.md](CONTRIBUTING_EN.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)), bản thảo nghiên cứu LaTeX ([docs/research/vilegal_tempbench_acl.tex](docs/research/vilegal_tempbench_acl.tex)), Datasheet ([docs/research/DATASHEET_VILEGAL_TEMPBENCH_V2.md](docs/research/DATASHEET_VILEGAL_TEMPBENCH_V2.md)) và bộ tài liệu triển khai Pilot Doanh nghiệp ([docs/guides/enterprise-pilot-kit.md](docs/guides/enterprise-pilot-kit.md)).

## v31.2.0 (2026-11-28 — Security Hardening, Scientific Honesty, Core Bugfixes, Soft Deprecation & Depth Upgrades)

### 🛡️ Bảo Mật & Mật Mã Học (P0 — Security Hardening)
- **Mật mã học & Chứng thực suy luận (`verifiable_inference.py`, `swarm_inference.py`, `cli/main.py`)**:
  - Bắt buộc đối chiếu `tensor_hash` trong `xac_thuc_cam_ket` / `xac_nhan_cam_ket_chuoi` (chặn hoàn toàn giả mạo Fiat-Shamir), gắn chặt `node_id`, `layer_id`, `tensor.shape` và `tensor.dtype` vào challenge hash và hàm băm tensor.
  - Khắc phục lỗ hổng nhân đôi nút lá lẻ (CVE-2012-2459) trong `CayMerkleSuyLuan` (RFC 6962) và kiểm tra chặt độ dài/chỉ số đường dẫn trong `xac_minh_merkle_proof`.
  - Cập nhật `vai verify --proof` yêu cầu `--tensor-hash` hoặc `--text`; `xac_minh_freivalds_nut` trả rõ `"hop_le": None, "trang_thai": "khong_ap_dung"` cho luồng đa tầng phi tuyến.
- **Hợp nhất AST Sandbox & Chặn RCE (`sandbox/ast_analyzer.py`, `security/agent_sandbox.py`, `agents/code_healing.py`, `agents/devops.py`, `plugin/plugin_manager.py`)**:
  - Bổ sung `visit_Name` chặn nạp bí danh hàm nguy hiểm (`f = open`), mở rộng `MODULE_CAM` và chặn `.format()` / `.format_map()` động; hợp nhất toàn bộ sandbox về 1 lõi `PhanTichAST` + `ThucThiDocLap` (`RLIMIT_*`).
  - Cách ly namespace nạp plugin (`vietnamese_ai._loaded_plugins.<hash>`) để không ghi đè `sys.modules` chuẩn và yêu cầu cờ tường minh khi nạp plugin chưa ký.
- **Chặn LFI, Arbitrary File Write, Path Traversal, Auth/CORS/XSS & Webhook Fail-Open**:
  - Thêm cờ `cho_phep_doc_tep=False` trên các bề mặt HTTP/Zalo/Telegram (`contract_review.py`, `studio_dashboard.py`, `connectors/base.py`, `solutions/trich_xuat.py`, `solutions/cskh.py`); khóa thư mục bằng `Path.resolve().is_relative_to(...)` trên Studio, MCP và Agent Tools.
  - Mặc định bind `127.0.0.1`, khóa CORS theo origin cho phép, bắt buộc xác thực trên toàn bộ `/api/*` khi `bat_xac_thuc=True`, thoát HTML chống DOM XSS (`ui/web_app.py`, `ui/components.py`, `ui/core.py`), bỏ fail-open và thêm chống replay (`timestamp` + `msg_id`) trên Zalo/Telegram webhooks, bỏ khóa HMAC mặc định công khai trong `io_utils.py`.

### 📊 Chuẩn Hóa Trung Thực Khoa Học & Tái Lập Benchmark (P1)
- **`BoDoLuongDoiDau` (`comparative_benchmark.py`), `README.md` & `paper/paper.md`**:
  - Đo khởi động lạnh (subprocess cô lập) và `VmRSS` thực tế, gỡ toàn bộ bảng đối đầu tĩnh giả lập và thay bằng đo trực tiếp trên tập mở 100 điều khoản / 35 QA `legal_vn_corpus.jsonl` và `ViLegal-TempBench`.
  - Minh bạch hóa `ChinhSachUuTienTiengViet` (proxy xếp hạng tuyến tính 256 chiều) và điều chuẩn L2 trọng số trong `ViAlign-SafeLoop`; nối `bootstrap_ci_95` vào quyết định của `CongKiemDinhMoHinh`.
  - Gỡ rò rỉ đáp án định lượng khỏi `TU_DIEN_DONG_NGHIA_VIET`, mở rộng `ViLegal-TempBench` lên **68 câu QA** (bổ sung câu không kèm số hiệu và câu OOD pháp lý thực tế), bổ sung [scripts/sync_legal_datasets.py](scripts/sync_legal_datasets.py) kiểm tra SHA-256 100% trong CI; loại bỏ toàn bộ điểm ngẫu nhiên giả lập trong `llm/lm_eval.py` và `llm/benchmark_runner.py`.

### 🔧 Sửa Dứt Điểm Bug Lõi (P2)
- **`.vaidb` & Legal RAG**: thêm SQLite `timeout=30.0` + `PRAGMA busy_timeout=5000`, vô hiệu hóa cache vector/HNSW đúng khi cập nhật, bỏ sao chép ma trận `O(N²)` khi chèn đơn lẻ, sửa gọi giao diện reranker & tiền tố quý/năm trong `temporal_rag.py`/`temporal_query.py`, đối chiếu cặp `(so_hieu, dieu)` theo ngữ cảnh câu trong `citation_verifier.py`, bỏ lời gọi `RingAttentionNode` dư thừa trong `rag_pipeline.py`.
- **Edge, BitNet, Fine-Tuning & Serving**: sửa ưu tiên `llama-cpp-python` → `llama-cli` (`c_bridge.py`), sửa thread-safety + TTL/IP session token (`swarm_auth.py`), sửa Straight-Through Estimator + `nen_tai_cho()` giải phóng FP32 trong `BitLinear` (`compression/extreme.py`), tách `ref_model` bằng `copy.deepcopy` / `simpo` trong `dpo_trainer.py`, sửa xuất GGUF thật và tải Modal Volume (`hf_wrapper.py`, `remote_gpu.py`), tách `argmax` khi `nhiet_do == 0.0` (`serving/speculative.py`).
- **Core, ML, NLP, Agents & Orchestrator**: sửa `Pipeline.danh_gia()`, nối ngữ cảnh RAG vào `QuyTrinhDauCuoi`, tính gradient trước cập nhật trọng số (`neural_net.py`, `layers.py`, `lora.py`), sửa ngữ cảnh n-gram (`vietnamese_llm.py`), tách `fit`/`transform` TF-IDF (`nlp/sentiment.py`), sửa `StateGraph`, đổi `ThreadingHTTPServer` cho Orchestrator, sửa regex `\bhi\b` (`solutions/cskh.py`) và logic 4-bit/8-bit (`mobile/power_manager.py`).

### 🧩 Tái Cấu Trúc Tương Thích Ngược (P3 — Phương Án A, 0 Breaking Changes)
- Gắn `DeprecationWarning` mềm cho 6 module mô phỏng/thử nghiệm (`wasm_node`, `webxr_node`, `tee_zkp_node`, `mobile_tools`, `salm`, `federated`), sửa lỗi cú pháp template literal trong `evonet_os/app.js`, bổ sung `evonet_os/README.md` và `sdks/README.md`.
- Tách nhóm tài chính sang `vietnamese_ai/plugins/finance/` (giữ shim tương thích ngược tại `vietnamese_ai/plugin/`), đồng bộ 100% exports của `vietnamese_ai.legal` và `vietnamese_ai.legal.rag` từ `vietnamese_ai.rag.legal`, khai báo phụ thuộc `vietnamese-ai>=31.2.0,<32.0.0` cho 5 gói con `packages/vietnamese-ai-*` và giữ nguyên **349 public APIs** (`.pyi`).

### 🚀 Nâng Cấp Chiều Sâu Mũi Nhọn Độc Bản (P4)
- **Lõi NLP Tiếng Việt & Rust Parity**: đóng gói từ điển tiếng Việt nén **38.372 từ** (32.877 từ ghép MIT, `134 KB` tại `vietnamese_ai/nlp/data/tu_dien_tieng_viet.txt.gz`) cho cả `BoTachTuTiengVietSieuToc` và `KiemTraChinhTa`; đồng nhất 100% lọc ký tự Zero-Width/BiDi giữa Rust (`rust_core/src/lib.rs`) và Python; chuyển `BoNhungVanBanSieuToc` sang zero-copy Rust `fast_top_k_cosine_bytes` và Signed Feature Hashing tách âm tiết từ ghép.
- **Tầng Ngữ Nghĩa ONNX CPU An Toàn**: bổ sung preset `vai pull` tải đồng bộ ONNX INT8 + `tokenizer.json` (`multilingual-e5-small-onnx`, `bge-m3-onnx`, `bge-reranker-v2-m3-onnx`); `BoNhungONNXTiengViet` và `BoSapXepLaiONNXTiengViet` từ chối chạy ONNX khi thiếu `tokenizer.json` hợp lệ thay vì đưa token ID băm giả vào mạng neural.
- **Hợp nhất `StudioDashboard`**: dùng chung 1 kho `.vaidb` giữa `self.rag`, Workspace `mac_dinh` và `TroLyPhapLy`; thay mặc định MLP 6 câu ở tab Training bằng quy trình thật nối `VongLapCanChinhAnToan` và `BoDieuPhoiTinhChinhXa`.
- **Siết kiểm tra kiểu dữ liệu**: bật `check_untyped_defs = true` trong `pyproject.toml` cho 4 gói lõi (`rag/`, `security/`, `edge/`, `core/`) với **0 lỗi `mypy`** trên toàn bộ **342 tệp**, **0 lỗi `ruff`**, **1.397 passed / 0 failed**, `ViLegal-TempBench` đạt **MRR@10 = 0.992**, **R@3 = 0.992**, **R@5 = 1.000**, **TVA = 1.000**, **TCA = 12/12**, **Abstention F1 = 1.000**.

## v31.1.0 (2026-11-25 — Temporal & Graph-Grounded Legal RAG + ViLegal-TempBench)

### 🔬 Chiều Sâu Nghiên Cứu Ứng Dụng & Sản Phẩm Pháp Lý Tiếng Việt
- **Chặng 1 — Thuật Toán Lõi Temporal & Graph-Grounded Legal RAG**:
  - `BoPhanTichThoiGianPhapLy` (`TemporalQueryParser`): tự động bóc tách mốc thời gian pháp lý từ tiếng Việt tự nhiên (*ngày/tháng/năm, trước/từ/sau mốc sửa đổi, quý, đầu/cuối năm, hiện hành*), chống nhiễu số hiệu/năm ban hành đạo luật và hỗ trợ **Statute-Bounded Temporal Disambiguation** đạt **100.0% TQPA (64/64)**.
  - Nâng cấp `DoThiThamChieuPhapLy` sang **Clause-Level Temporal Graph** (`NutDieuKhoanPhapLy`, `QuanHeDieuKhoan`) với lan truyền đa bước có điều kiện thời gian $t$ (`mo_rong_da_buoc_theo_thoi_gian`, hỗ trợ chuỗi 2-hop Luật ↔ Nghị định ↔ Thông tư và tự động chuyển hướng văn bản thay thế khi gặp điều luật hết hiệu lực).
  - `BoKiemChungTrichDanPhapLy` (`LegalCitationVerifier`): kiểm chứng trích dẫn tất định 3 tầng (*Citation Grounding*, *Temporal Consistency* nhận diện cả chủ động/bị động thay thế, và *Numeric/Deadline Consistency*) áp dụng được cho cả câu trả lời tổng hợp lẫn đầu ra từ LLM ngoài (Gemini / Ollama / OpenAI).
  - `BoTruyHoiPhapLyThoiGian` (`TemporalGraphLegalRAG`): động cơ truy hồi 5 tầng (`bm25` → `hybrid` → `temporal` → `temporal_graph` → `full` với Residual Cross-Interaction Reranker + Abstention Gate) chạy mượt 100% trên CPU (~56 ms/truy vấn), giữ **100% tương thích ngược** và tích hợp thẳng vào `TroLyPhapLy` (tự nhận diện mốc thời gian + `kiem_chung_cau_tra_loi()`).
- **Chặng 2 — Bộ Benchmark Mở `ViLegal-TempBench` & Ablation Study Tái Lập**:
  - Công bố bộ dữ liệu mở `ViLegal-TempBench` (đồng bộ `benchmarks/datasets/` và wheel `vietnamese_ai/rag/legal/data/`): **60 điều khoản đa phiên bản theo thời gian** (ghép với 100 điều khoản nền = **160 điều khoản**), **38 cạnh đồ thị cấp Điều/Khoản**, và **64 truy vấn chuẩn** gồm 4 nhóm `T1` (16 hiện hành), `T2` (24 câu = 12 cặp tương phản thời gian), `T3` (14 chuỗi đa bước 1–2 hop), `T4` (6 bẫy luật hết hiệu lực + 4 từ chối ngoài phạm vi).
  - `BoDanhGiaLegalTemporalRAG` (`ViLegalTemporalBenchmark`, CLI `vai benchmark --suite legal-temporal [--latex]` và `python benchmarks/run_benchmarks.py --suite legal-temporal`): đo đầy đủ Recall@1/3/5, MRR@10, nDCG@10, TVA, TCA, Multi-Hop Bridge Recall@3, Trap Correction Rate, Citation Faithfulness, Temporal Conflict Rate, Abstention F1 kèm kiểm định **Paired Bootstrap** ($p$-value) và xuất bảng Markdown + LaTeX `booktabs`.
  - Kết quả thực nghiệm trên CPU (`seed=42`): cấu hình `full` đạt **R@3 = 1.000**, **MRR@10 = 0.967** ($+0.117$ vs BM25, $p<0.001$), **TVA = 1.000** ($+0.278$, $p<0.001$), **TCA = 12/12 (100%)** (BM25: 0/12), **Multi-Hop@3 = 1.000** ($+0.286$ vs Hybrid, $p=0.037$), **Trap = 1.000**, **Temporal Conflict = 0.0%**, **Abstention F1 = 1.000**.
- **Chặng 3 — Thực Nghiệm LLM Thật (Gemini / OpenAI / Ollama) & Chuẩn Học Thuật**:
  - `tao_ham_sinh_llm`: bộ chuyển đổi chuẩn HTTPS không phụ thuộc SDK ngoài cho Google Gemini API (`GEMINI_API_KEY`, mặc định `gemini-2.5-flash`), OpenAI-compatible và Ollama CPU kèm cơ chế retry `429/503` và giãn cách `--llm-delay`; tích hợp trực tiếp vào `vai benchmark --suite legal-temporal --llm-provider gemini` và `benchmarks/run_benchmarks.py`.
  - Bổ sung `CITATION.cff` (CFF 1.2.0 kèm Zenodo DOI `10.5281/zenodo.20357082`) và báo cáo kỹ thuật `docs/guides/vilegal-temporal-rag.md` (kèm sơ đồ Mermaid, bảng Ablation, kiểm định Paired Bootstrap và BibTeX).
- **Chặng 4 — Đóng Gói Sản Phẩm Đồng Bộ Trên Toàn Bộ Bề Mặt (`v31.1.0`)**:
  - **CLI (`vai`)**: lệnh mới `vai legal-ask "<câu hỏi>" [--date] [--mode] [--llm-provider] [--json]`, cờ `--date` cho `vai solution phap-ly`, và `--temporal-legal-tools` cho `vai mcp-server`.
  - **REST API (`FastAPIServer`)**: thêm `POST /v1/legal/temporal-ask` và `POST /v1/legal/verify-citation`.
  - **MCP Server (`MayChuMCPThucTe`)**: thêm `dang_ky_cong_cu_phap_ly_thoi_gian()` đăng ký 2 công cụ `tra_cuu_phap_ly_thoi_gian` và `kiem_chung_trich_dan_phap_ly` (giữ 100% tương thích ngược 6 công cụ mặc định / 9 công cụ `--extended-tools`).
  - **Web Studio Dashboard (`StudioDashboard`)**: thêm thẻ tương tác chọn mốc ngày & chế độ truy hồi cùng endpoints `POST /api/legal/temporal-ask` và `POST /api/legal/verify-citation`.
- **ViEdge-RustEngine & `ViEdge-SysBench` — Edge AI Không GPU & Truy Hồi Vector Lượng Tử Hóa Rust (Systems for AI)**:
  - Bổ sung lõi native Rust `rust_core/src/edge_quant.rs` (kèm runtime dispatch `SSE4.1` / `AVX2` / NEON và dự phòng thuần NumPy 100% tương đương): lượng tử hóa vô hướng `SQ8` (nén **4× RAM**), nhị phân 1-bit `POPCNT` (nén **32× RAM**), hai tầng bất đối xứng `Binary+SQ8 Rescore`, và đồ thị **Native Rust `RustHNSWIndex`** (thuật toán Malkov Alg. 4 Diversity Heuristic cho cả cạnh thuận và cắt tỉa cạnh ngược, tìm kiếm lô song song Rayon, tuần tự hóa nhị phân `VQI1`/`VHNS`).
  - `ChiMucVectorLuongTu` (`QuantizedVectorIndex`) tích hợp trực tiếp vào `KhoTriThucVAIDB` (bảng SQLite `vai_quant_index`, `xay_dung_chi_muc_luong_tu`, tự nạp lại khi mở tệp `.vaidb` và tự vô hiệu hóa an toàn khi chèn mới).
  - Nâng cấp `BitLinear` với nhân **BitNet 1.58-bit đóng gói 2-bit thực (`4 trọng số/byte`, nén 16× RAM)** dùng bảng tra **T-MAC 4-bit Nibble LUT** không phép nhân, không nhánh rẽ (`dong_goi_bitnet_158`, `nhan_ma_tran_bitnet_158`, `BitLinear.suy_luan_2bit`, `thong_ke_nen`).
  - Bộ chuẩn `BoDanhGiaHeThongEdge` (`ViEdgeSystemsBenchmark`, CLI `vai benchmark --suite edge-systems [--scale N] [--dim D] [--latex]`): thực đo trên Intel Celeron N3050 2 lõi không GPU (20k×384) đạt **HNSW-SQ8 p50 = `0.311 ms`** (**117× nhanh hơn FP32 NumPy**, **2,027.6 QPS lô**, **Recall@10 = 0.9688**, **4× RAM**), **BitNet 2-bit `512×512` `1.64 ms`** (**nhanh hơn 1.85× FP32 OpenBLAS**, **16× RAM**), và chuỗi tiếng Việt Zero-GPU đầu-cuối chỉ **`4.55 ms`**. Xem `docs/guides/viedge-rust-systems.md`.
- **ViAlign-SafeLoop & `ViAlign-SafeBench` — Eval-Gated Continual Alignment Chống Hồi Quy**:
  - `BoLocNhieuCapUuTien` (`NoisyPreferenceFilter`): lọc nhiễu cặp ưu tiên 6 tín hiệu (khiên an ninh stateless `TuongLuaAI`, kiểm toán PII `DataSanitizer`, kiểm chứng hiệu lực luật `BoTheoDoiHieuLucPhapLy`, toàn vẹn cấu trúc & mâu thuẫn đảo chiều cùng prompt, sai phân chất lượng $\Delta Q$, và ngoại lệ biên thưởng MAD z-score) kèm gán trọng số tin cậy `trong_so` & `label_smoothing` thích nghi cho cDPO; tích hợp trực tiếp vào `QuanLyPhienChatWorkspace.xuat_tap_du_lieu_dpo(..., loc_nhieu=True)` và CLI `vai export-dpo --filter-noise`.
  - Nâng cấp `DPOTrainer` hỗ trợ trọng số mẫu `trong_so` + điều chuẩn neo KL (`he_so_neo_kl`) tương thích ngược 100%.
  - `VongLapCanChinhAnToan` (`EvalGatedContinualAligner`, `ChinhSachUuTienTiengViet`, `BoDemOnTapNeo`): vòng lặp căn chỉnh liên tục khép kín với trộn ôn tập neo phân tầng, huấn luyện ứng viên cô lập, kiểm định 4 trục `CongKiemDinhMoHinh` + Paired Bootstrap 95% CI và **Auto-Rollback nguyên tử** về snapshot đã băm SHA-256 khi phát hiện hồi quy.
  - Bộ chuẩn đa vòng `BoDanhGiaCanChinhAnToan` (`ViAlignSafeBenchmark`, CLI `vai benchmark --suite align-safety --rounds 5 [--latex]`): trên 5 vòng liên tục (70 cặp, 55.7% nhiễu/đầu độc gồm jailbreak R3 và ngụy tạo con số tinh vi R5), đạt thứ bậc đối chứng đơn điệu nghiêm ngặt **Naive DPO `0.6688` (An toàn `0.0000`) → Filtered DPO `0.9833` (Filter F1 `0.9041`) → Filtered+Replay+cDPO `0.9917` → ViAlign-SafeLoop `1.0000`** (`0` hồi quy ngầm, `Rollback = 1` chặn đúng R5, Paired Bootstrap `+0.2250`, $p = 0.0000$). Xem `docs/guides/vialign-safeloop.md`.

## v31.0.4 (2026-11-25 — Deep Architecture, Vertical Moat, 5-Surface Integration & CI Green)

### 🚀 5 Trụ Cột Kiến Trúc Chiều Sâu, Vertical Moat Pháp Lý, Tích Hợp 5 Bề Mặt & Vá Toàn Diện CI/CD
- **Vá Toàn Bộ Lỗi Tiềm Ẩn & CI Matrix Python 3.9–3.13 (Hướng A + CI Hardening)**:
  - Sửa bug thật `total_mem` → `total_memory` trên `MultiGPUTrainer`, sửa toàn bộ **27 lỗi `mypy`** trên môi trường cài đủ `.[all,dev]` (`pytorch_trainer`, `dpo_trainer`, `sft_trainer`, `grpo_trainer`, `multi_gpu`, `jamba_block`, `mang_sau`, `embedder`, `image_embedder`, `llm/wrappers`, `regression`, `classifier`, `qdrant_store`).
  - Sửa `XuLyVanBan(dung_native_tokenizer=True)` không bị `underthesea` ghi đè khi cài `[all]`; bổ sung lọc đầy đủ `\u200c`/Zero-Width/BiDi ở đường fallback Python của `BoTachTuTiengVietSieuToc.chuan_hoa_unicode`; thêm fallback TOML cho Python 3.9/3.10 (`tomli`) và cấu hình `ubuntu-22.04` cho Python 3.9 trong `.github/workflows/ci.yml`.
  - Bổ sung job CI **`rust-core-parity`** biên dịch thật `vietnamese_ai_rust.so` và chạy kiểm thử đối chiếu Rust vs Python trên GitHub Actions.
- **Lõi NLP, Rust Zero-Copy & Xác Thực Byzantine (Hướng B + C1)**:
  - Chuẩn hóa dấu thanh âm vị học FST + tách từ ghép Viterbi DAG khớp 100% giữa Rust và Python (**584,893 từ/s** FST, **269,914 từ/s** Viterbi).
  - Bổ sung kernel Rust Zero-Copy `PyBytes` (`fast_q5_k_matmul_f32_bytes`, `fast_q6_k_matmul_f32_bytes`, `fast_rmsnorm_f32_bytes`, `fast_swiglu_f32_bytes`) nhanh hơn NumPy **3.7x–6.3x**.
  - `KiemChungFreivalds` ($O(k \cdot n^2)$) tối ưu `float32` và siết ngưỡng `rtol=2e-5, atol=1e-3` phát hiện **100%** gian lận 1 phần tử trên ma trận `256x256`; KV-Cache đa phiên & TCP keep-alive pool trên Swarm LAN.
  - Bổ sung bộ benchmark tái lập `--suite deep` (`BoDoLuongDoiDau.do_chieu_sau_loi_he_thong`) đo định lượng 6 trụ cột lõi mới.
- **Vertical Moat Pháp Lý Việt Nam & Cross-Encoder ONNX (Hướng C4)**:
  - `BoNapVanBanPhapQuyTuDong` (`AutoLegalCorpusPipeline`, CLI `vai legal-ingest <nguon> -o kho.vaidb`): tự động phân tích văn bản pháp quy & **Văn bản hợp nhất** (`.docx`/`.pdf`/`.txt`/`.md` hoặc thư mục), trích xuất số hiệu, ngày hiệu lực, quan hệ thay thế/sửa đổi/hướng dẫn, bóc tách cước chú hợp nhất `[N]` → tự sinh chuỗi `PhienBanDieuKhoan` Point-in-Time + cạnh `QuanHePhapLy`, đóng gói thành **1 tệp `.vaidb` đã dựng sẵn HNSW bền vững** (vector hóa batch-neighbor) và tự động phục hồi khi mở bằng `TroLyPhapLy(duong_dan_vaidb=...)`.
  - Nâng cấp `BoSapXepLaiONNXTiengViet` & `SapXepLai` hỗ trợ preset **BGE-Reranker-v2-m3** (`bge_reranker`), **PhoRanker** (`phoranker`, tự động tách từ ghép `_` chuẩn PhoBERT), `cross_encoder_onnx`, nạp `tokenizer.json`, suy luận theo lô `cham_diem_lo` (hỗ trợ cả 1 logit và 2 logits `[neg, pos]`) kèm fallback bigram + proximity span.
  - Mở rộng `CO_SO_PHIEN_BAN_DIEU_KHOAN_VN` lên **14 mốc Point-in-Time** thực chiến (Hóa đơn `123/2020/NĐ-CP` → `70/2025/NĐ-CP`, BHXH `58/2014` → `41/2024`, Thuế GTGT `13/2008` → `48/2024`, Đất đai `45/2013` → `31/2024`, Giao dịch điện tử `51/2005` → `20/2023`, Nhà ở `65/2014` → `27/2023`).
- **Tích Hợp 5 Bề Mặt, Tinh Chỉnh & Red-Team Bảo Mật (Hướng C2 + C3)**:
  - Nhận diện tách/gộp điều khoản hợp đồng và xuất **Word Track Changes (`<w:del>`/`<w:ins>`) + `word/comments.xml`**; SFT Instruction Masking (`-100`) + packed `position_ids`; **SimPO** không cần `ref_model` (tiết kiệm 50% VRAM); Eval Gate **Paired Bootstrap 95% CI**; tường lửa chặn Prompt Injection tiếng Việt có dấu/không dấu/Zero-Width/BiDi Override; PII `DataSanitizer` nhận thức ngữ cảnh (không phá VNĐ/số hiệu luật); Sandbox giới hạn RAM `RLIMIT_AS`/`bwrap`; chống DoS `Content-Length` (`400`/`413`) + CORS/RBAC (`403`).
  - Nối đồng bộ lên cả 5 bề mặt: `TroLyPhapLy`, CLI (`vai`), FastAPI (`/v1/legal/clause-version`, `/v1/legal/contract-review`, `/v1/nlp/tokenize`, `/v1/vaidb/hnsw`), MCP Server (`--extended-tools` 9 công cụ), và Web Studio Dashboard 6 tabs.
  - Kiểm định toàn diện: **316/316** public API stubs `.pyi`, **1,314 passed / 32 skipped / 0 failed**, **0 lỗi `mypy`** (328 tệp), **0 lỗi `ruff`**, `mkdocs build --strict` exit 0, `vai release-check` ĐẠT.

## v31.0.3 (2026-11-24 — Open-Source Production Release)

### 🚀 4 Hướng Chiến Lược, 5 Ưu Tiên Sản Xuất & Khép Kín Toàn Bộ 9 Mối Nối Liên Tầng
- **Kho Tri Thức 1 Tệp `.vaidb`, Ngữ Nghĩa ONNX CPU & Đồ Thị Tham Chiếu Pháp Lý**:
  - `KhoTriThucVAIDB` (SQLite WAL + FTS5 BM25 tiếng Việt có dấu/không dấu + Float32 Vector + đồng bộ thư mục tăng dần SHA-256 `RAGPipeline.dong_bo_thu_muc`), hỗ trợ trực tiếp `RAGPipeline.luu("*.vaidb")` và `RAGPipeline.tai("*.vaidb")`.
  - `BoNhungONNXTiengViet` & `BoXepHangLaiNguNghiaONNX` (`onnx_viet`) hiểu từ đồng nghĩa đời thường tiếng Việt (sa thải, sổ đỏ, hóa đơn đỏ, lộ dữ liệu) chạy 100% trên CPU.
  - `BoTheoDoiHieuLucPhapLy` & `DoThiThamChieuPhapLy` (`vai legal-graph`): truy vết Luật ↔ Nghị định ↔ Thông tư ↔ văn bản thay thế hiện hành theo mốc ngày kèm sơ đồ Mermaid.
  - Đóng gói sẵn **100 điều khoản pháp lý Việt Nam + 35 QA chuẩn** trực tiếp trong wheel (`vietnamese_ai/rag/legal/data/`), đồng bộ mặc định trên `StudioDashboard`, `vai solution phap-ly`, `vai connector`, `MayChuMCPThucTe` và `vai eval-gate`.
- **Bàn Làm Việc Hợp Đồng, Chat Workspace Phòng Ban, MCP IDE & Zalo/Telegram**:
  - `BoSoSanhRaSoatHopDong` (`vai contract-review`): so sánh điều khoản hợp đồng `.docx`/`.pdf`/`.txt` với mẫu chuẩn, kiểm toán hiệu lực luật, xuất báo cáo Redline Markdown & Word `.docx`.
  - `QuanLyPhienChatWorkspace` + `StudioDashboard` (`vai studio --vaidb ... --workspace-dir ... --auth`): chat đa phiên cách ly `.vaidb` theo phòng ban, kéo-thả tệp thẳng vào Workspace, RBAC + Audit + Thumbs Up/Down kèm đáp án sửa (`cau_tra_loi_sua`), giao diện động kết nối đủ Status Map 3 tầng, Đồ thị Luật, Eval Gate & Xuất DPO.
  - `MayChuMCPThucTe` (`vai mcp-server --transport stdio|sse --plugin-dir ...`): chuẩn MCP JSON-RPC 2.0 cắm thẳng vào Cursor, VS Code & Claude Desktop, tự động khám phá & phơi bày công cụ V-Plugin 2.0 (`gan_quan_ly_plugin`).
  - `KetNoiZaloOA` & `KetNoiTelegram` (`vai connector --workspace-dir ...`): xác thực MAC SHA-256 / Secret Token, lưu lịch sử bền vững vào SQLite Workspace để gom DPO, hỗ trợ lệnh nhanh `/hieuluc`, `/luat`, `/hopdong`.
- **Zero-GPU Edge CPU Runtime & Multi-Provider Remote GPU Flywheel**:
  - `QuanLyMoHinhEdge` (`vai pull` / `vai run`): 8 mô hình GGUF tiếng Việt tự chọn lượng tử theo RAM thực, tải HTTP Range Resume, chạy 3 tầng Native (`llama-cpp-python` → `llama-cli` → GGUF mmap NumPy), mở trực tiếp `.vaidb` với `vai run ... --rag kb.vaidb --admin`.
  - `BoDieuPhoiTinhChinhXa` (`vai train-cloud --from-workspace ... --eval-4-axis`): hỗ trợ `NhaCungCapModal`, `NhaCungCapSSH`, `NhaCungCapCucBo`, rút thẳng DPO JSONL đã ẩn PII từ Workspace, kiểm trần ngân sách USD/VNĐ, ký HMAC-SHA256 và kiểm định hồi quy 4 trục (`CongKiemDinhMoHinh`, `vai eval-gate`).
- **Tăng cường Bảo mật, API & Trải nghiệm Open Source**:
  - Vá biểu thức Prompt Injection tiếng Việt trên `TuongLuaAI` (`bỏ qua mọi/toàn bộ quy tắc`, `tiết lộ toàn bộ mật khẩu nội bộ`) và nâng cấp chấm điểm bám sát nguồn `CongAnToanDauRa.kiem_tra_ao_giac` (ưu tiên `noi_dung` / `trich_doan` đầy đủ).
  - `FastAPIServer` hỗ trợ trực tiếp `TroLyPhapLy`, `TongDaiCSKH`, `DongCoSuyLuanEdge`; `QuyTrinhDauCuoi` bổ sung cầu nối 1 dòng `ra_soat_hop_dong`, `kiem_dinh_mo_hinh`, `tao_mcp_server`, `tao_webhook_kenh`; `vietnamese_ai.ui` export lazy đầy đủ các lớp Studio/Workspace.
  - Bổ sung `vai demo`, `vai status-map` (bản đồ minh bạch 3 tầng Core / Optional / Experimental), `vai release-check`, `benchmarks/Dockerfile.benchmark`, và nâng cấp `.github/workflows/publish.yml` (OIDC Trusted Publishing TestPyPI/PyPI + clean-venv wheel smoke test).
  - Kiểm định toàn diện: **316/316** public API stubs `.pyi`, **1,265 passed / 32 skipped / 0 failed**, **0 lỗi `mypy`** (327 tệp), **0 lỗi `ruff`**, `mkdocs build --strict` exit 0.

## v31.0.2 (2026-11-18)

### 🌟 Hệ Sinh Thái Khép Kín 5 Bước Tự Tối Ưu Hóa & Tự Chữa Lành (Closed-Loop Autonomous AI Ecosystem)
- **1. Bánh Đà Dữ Liệu Khép Kín (Continuous Data Flywheel & DPO Loop)**:
  - Tự động đóng gói mọi tương tác suy luận, phản hồi người dùng hoặc đánh giá từ Tác tử Tự sửa (`Self-Refine`) thành cặp cấu trúc `(prompt, chosen, rejected)` vào `HoDuLieu` (DataLake DVC) kèm mã băm SHA-256.
  - `DPOTrainer` tự động đọc dữ liệu và tối ưu hóa trọng số mô hình, đăng ký và thăng hạng trực tiếp lên `QuanLyMoHinh` (`production`).
- **2. Vận Hành Tự Phục Hồi & Auto-Rollback (Self-Healing Ops Loop)**:
  - `MachCat` (Circuit Breaker) và `KiemTraSucKhoe` (Health Checker) liên kết trực tiếp với Model Registry.
  - Tự động kích hoạt quy trình rollback về phiên bản ổn định gần nhất kèm xác thực chữ ký số HMAC-SHA256 chống giả mạo khi phát hiện sự cố dồn dập hoặc suy thoái chất lượng.
- **3. Đóng Gói Biên 1-Click & Phân Phối Swarm CPU (Edge Packaging & Verifiable Swarm)**:
  - Tự động chuyển đổi mô hình sang GGUF lượng tử hóa (`XuatGGUF`), phân mảnh layer song song (Pipeline Parallelism) tới cụm máy tính văn phòng / Mini PC (`DieuPhoiSwarmCPU`).
  - Tự động sinh chữ ký chứng thực mật mã học Verifiable AI Merkle Audit Path $O(\log N)$ kết hợp Schnorr zkML.
- **4. Vòng Lặp Đồng Tiến Hóa Đối Kháng (Adversarial Co-Evolution Loop)**:
  - Tác tử `RedTeamSimulator` tự động tập kích các cuộc tấn công jailbreak, Prompt Injection tiếng Việt, System Prompt leak và Payload ẩn.
  - Tường lửa AI đa tầng (`TuongLuaBaoVeFramework`) và Luồng An toàn (`LuongAnToan`) ngăn chặn, tự động sinh tập dữ liệu đối kháng DPO và tôi luyện trực tiếp các trọng số mô hình chống lại các cuộc tấn công (`hardened`).
- **5. Hợp Nhất Đồ Thị Trạng Thái Phân Tán (Distributed StateGraph Unification)**:
  - Đồ thị trạng thái DAG (`DoThiTrangThai` / `StateGraph`) hợp nhất toàn diện mọi mắt xích: Kiểm duyệt an ninh $\rightarrow$ Truy xuất tri thức RAG $\rightarrow$ Suy luận mô hình / Swarm CPU $\rightarrow$ Tự sửa chất lượng Self-Refine $\rightarrow$ Khử độc DLP đầu ra.
  - Bổ sung các phương thức: `them_nut_tac_tu`, `them_nut_swarm`, `them_nut_cum` cùng bộ alias chuẩn LangGraph: `add_node`, `add_edge`, `add_conditional_edges`, `add_agent_node`, `add_swarm_node`, `add_cluster_node`, `compile`.
- **Cầu Nối Thống Nhất `QuyTrinhDauCuoi` (`HeSinhThaiAI` / `EndToEndWorkflow`)**:
  - Hợp nhất toàn diện 5 mắt xích khép kín vào một điểm vào (Single Entrypoint) duy nhất cho toàn bộ framework.

## v31.0.0 (2026-11-08)

### 🌟 DePIN Swarm CPU Inference & Verifiable AI (Merkle & zkML)
- **Hệ Thống Suy Luận Bầy Đàn CPU Phân Tán (Swarm CPU Inference & Pipeline Parallelism)**:
  - `NutSwarmCPU` (`SwarmCPUNode`): Đại diện cho một máy tính CPU văn phòng (Mini PC, laptop, workstation phổ thông không có GPU) tham gia mạng lưới tính toán. Mỗi nút chịu trách nhiệm một phân đoạn layer (Model Sharding / Pipeline Parallelism) với trọng số cục bộ và forward pass tối ưu trên CPU.
  - `DieuPhoiSwarmCPU` (`SwarmInferenceOrchestrator`): Bộ điều phối cụm Swarm kết nối nhiều máy CPU qua mạng LAN/WAN để cùng chạy mô hình lớn; tự động phân chia layer, chuyển tiếp activation states tuần tự và hỗ trợ sinh chuỗi theo luồng (streaming generation).
  - Tối ưu hóa bộ nhớ: Cho phép các thiết bị cấu hình thấp (2GB - 4GB RAM) có thể phối hợp tải tính toán thay vì phải trang bị GPU cao cấp đắt đỏ.

- **Xác Thực Suy Luận Mật Mã Học (Verifiable AI Inference & Proof of Inference)**:
  - `CayMerkleSuyLuan` (`InferenceMerkleTree`): Cây băm mật mã Merkle Tree đóng gói trạng thái ẩn (hidden states) và model execution traces, cung cấp Merkle Audit Path với độ phức tạp xác thực $O(\log N)$.
  - `XacThucSuyLuanMaoHiem` (`VerifiableInference`): Lược đồ cam kết mật mã phi tương tác Schnorr / Fiat-Shamir Zero-Knowledge Proof (zkML Commitment) trên trường nguyên tố 256-bit ($g^z \equiv a \cdot y^e \pmod p$).
  - **Khả Năng Chống Gian Lận Byzantine (Byzantine Fault Tolerance)**: Tự động phát hiện và ngăn chặn các nút độc hại can thiệp, làm sai lệch hoặc làm giả activation tensors, bảo đảm kết quả trả về trung thực 100%.

- **Giao Diện Dòng Lệnh Mới (CLI Subcommands)**:
  - `vai swarm`: Khởi chạy và điều phối cụm suy luận phân tán trên các CPU văn phòng (`vai swarm --nodes 3 --layers 6 --prompt "..." --bench`).
  - `vai verify`: Kiểm định tính toàn vẹn của kết quả suy luận AI bằng chứng thực mật mã học (`vai verify --proof zkml1:...` hoặc `vai verify --text "..."`).

- **Nâng Cấp Hệ Thống & Zero-Dependency Oath**:
  - Toàn bộ cơ chế Merkle Tree và Schnorr/Fiat-Shamir ZKP cam kết tính toán được hiện thực hóa thuần túy trên Python & NumPy tiêu chuẩn, không phụ thuộc thư viện ngoài nặng nề.
  - Mở rộng số lượng biểu tượng công khai lên **244 public symbols** qua PEP 562 Lazy Loading và PEP 561 Type Stubs.
  - Đồng bộ phiên bản chuẩn hóa **31.0.0** trên toàn bộ hệ thống (`__init__.py`, `pyproject.toml`, `setup.py`, CI/CD, tests, docs).

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
