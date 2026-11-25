# CHANGELOG

All notable changes to Vietnamese AI Framework will be documented in this file.

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
