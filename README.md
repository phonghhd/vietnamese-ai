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
  <img src="https://img.shields.io/badge/version-31.2.0-blue.svg" alt="version">
  <img src="https://img.shields.io/badge/python-3.8%20--%203.14-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license">
  <img src="https://img.shields.io/badge/tests-1380%2B%20passed-brightgreen.svg" alt="tests">
  <img src="https://img.shields.io/badge/mypy-0%20errors-brightgreen.svg" alt="mypy">
  <img src="https://img.shields.io/badge/ruff-0%20errors-brightgreen.svg" alt="ruff">
  <img src="https://img.shields.io/badge/public%20APIs-349-indigo.svg" alt="apis">
  <img src="https://img.shields.io/badge/language-Vietnamese-red.svg" alt="language">
</p>

---
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20357082.svg)](https://doi.org/10.5281/zenodo.20357082)
## 🚀 Các Tính Năng Nổi Bật (Phiên bản v31.2.0 — Zero-GPU Local-First Vietnamese AI & Legal Workbench)

- **ViLegal-TemporalRAG & Bộ Chuẩn Học Thuật Mở `ViLegal-TempBench` (v31.1.0 — Applied Research → Production)**:
  - **Truy hồi Đa bước Có Điều kiện Thời gian**: `BoPhanTichThoiGianPhapLy` (*Statute-Bounded Temporal Disambiguation*, **100% TQPA** trên 68 câu) + Đồ thị cấp Điều/Khoản `DoThiThamChieuPhapLy` (60 điều khoản đa phiên bản + 38 cạnh thời gian + 100 điều khoản nền) + `BoTruyHoiPhapLyThoiGian` (5 tầng Ablation chạy trên CPU Celeron N3050 ~56 ms/truy vấn, đạt **R@3 = 0.992**, **R@5 = 1.000**, **MRR@10 = 0.992**, **nDCG@10 = 0.992**, **TVA = 1.000**, **TCA = 12/12**, **Multi-Hop@3 = 0.929**, **Trap = 1.000**, **Abstention F1 = 1.000**).
  - **Kiểm chứng Trích dẫn Tất định 3 Tầng**: `BoKiemChungTrichDanPhapLy` kiểm tra *Citation Grounding*, *Temporal Consistency* và *Numeric/Deadline Consistency* cho cả câu trả lời cục bộ lẫn LLM thật (**Google Gemini API**, OpenAI-compatible, Ollama không cần SDK phụ qua `tao_ham_sinh_llm`).
  - **Đồng bộ 5 Bề mặt & Tái lập 1 Lệnh**: CLI (`vai legal-ask`, `vai benchmark --suite legal-temporal --latex`), REST API (`/v1/legal/temporal-ask`, `/v1/legal/verify-citation`), MCP (`--temporal-legal-tools`), Web Studio UI (bộ chọn mốc ngày + kiểm chứng 3 tầng) và chuẩn trích dẫn [CITATION.cff](CITATION.cff). Xem [Báo cáo Kỹ thuật ViLegal-TemporalRAG](docs/guides/vilegal-temporal-rag.md).
- **ViEdge-RustEngine & `ViEdge-SysBench` (Zero-GPU Systems for AI — Rust SIMD + HNSW + BitNet 1.58-bit 2-bit)**:
  - **Truy hồi Vector Lượng tử hóa & Native Rust HNSW-SQ8**: `ChiMucVectorLuongTu` (`SQ8 4× RAM` SSE4.1/AVX2, `Binary 1-bit POPCNT 32× RAM`, `Binary+SQ8 Rescore`, `RustHNSWIndex` lưu thẳng trong tệp `.vaidb`). Trên CPU văn phòng 6W Celeron N3050 (`20,000 × 384`), **HNSW-SQ8** đạt **p50 = `0.311 ms`** (**117× nhanh hơn FP32 NumPy**), **2,027.6 QPS lô**, **Recall@10 = 0.9688** và giảm **75% RAM vector**.
  - **BitNet 1.58-bit Đóng Gói 2-bit T-MAC LUT (`16× RAM`)**: `dong_goi_bitnet_158` / `nhan_ma_tran_bitnet_158` / `BitLinear.suy_luan_2bit()` dùng bảng tra 4-bit Nibble LUT không phép nhân, không nhánh rẽ (nhanh hơn **9.96×** bản NumPy ở `2048×2048`, **nhanh hơn 1.85× FP32 OpenBLAS** ở `512×512`, sai số $\le 2 \times 10^{-6}$). Chuỗi tiếng Việt Zero-GPU đầu-cuối chỉ **`4.55 ms`**. Tái lập qua `vai benchmark --suite edge-systems --latex`. Xem [Báo cáo ViEdge-RustEngine](docs/guides/viedge-rust-systems.md).
- **ViAlign-SafeLoop & `ViAlign-SafeBench` (Eval-Gated Continual Alignment Chống Hồi Quy)**:
  - **Lọc nhiễu cặp ưu tiên 6 tín hiệu & Studio DPO**: `BoLocNhieuCapUuTien` chặn độc tố jailbreak, PII leak, luật hết hiệu lực, cặp mâu thuẫn & đảo nhãn (tích hợp thẳng vào `vai export-dpo --filter-noise` và `QuanLyPhienChatWorkspace.xuat_tap_du_lieu_dpo(loc_nhieu=True)`).
  - **Vòng lặp tự cải thiện có Auto-Rollback nguyên tử**: `VongLapCanChinhAnToan` kết hợp `BoDemOnTapNeo` + cDPO có trọng số + điều chuẩn neo trọng số L2 (`he_so_neo_kl`) + cổng kiểm định `CongKiemDinhMoHinh` (Paired Bootstrap 95% CI). Trên 5 vòng `ViAlign-SafeBench` với proxy xếp hạng tuyến tính 256 chiều `ChinhSachUuTienTiengViet` (55.7% nhiễu/đầu độc gồm jailbreak R3 & ngụy tạo số học R5), đạt thứ bậc đối chứng đơn điệu **Naive DPO `0.5969` (An toàn `0.0000`) → Filtered `0.9812` → +Replay+cDPO (+L2 Anchor) `0.9906` → ViAlign-SafeLoop `1.0000`** (`0` hồi quy ngầm, Paired Bootstrap `+0.2500`, $p=0.0000$). Tái lập qua `vai benchmark --suite align-safety --latex`. Xem [Báo cáo ViAlign-SafeLoop](docs/guides/vialign-safeloop.md).

- **Bàn Làm Việc AI Tiếng Việt Thực Chiến (9 Trụ Cột Đã Liên Kết Khép Kín)**:
  - **1. Kho Tri Thức 1 Tệp `.vaidb`, Ngữ Nghĩa ONNX CPU & Đồ Thị Luật (`vai rag --db`, `vai legal-graph`)**: `KhoTriThucVAIDB` (SQLite WAL + FTS5 BM25 tiếng Việt có dấu/không dấu + Float32 Vector + đồng bộ thư mục tăng dần SHA-256 `dong_bo_thu_muc`), `RAGPipeline.luu/tai("*.vaidb")`, `BoNhungONNXTiengViet` / `BoSapXepLaiONNXTiengViet` (bắt buộc `tokenizer.json` hợp lệ khi chạy ONNX INT8, hỗ trợ tải 1 lệnh qua `vai pull multilingual-e5-small-onnx` / `bge-m3-onnx` / `bge-reranker-v2-m3-onnx`, có fallback thuần CPU hiểu đồng nghĩa đời thường) và `DoThiThamChieuPhapLy` truy vết Luật ↔ Nghị định ↔ Thông tư kèm sơ đồ Mermaid.
  - **2. RAG Pháp Lý 100 Điều Khoản Đóng Gói Sẵn Trong Wheel & Theo Dõi Hiệu Lực**: Bộ 100 điều khoản pháp luật Việt Nam + 35 QA chuẩn đi kèm ngay khi `pip install vietnamese-ai` (`vietnamese_ai/rag/legal/data/`), `BoTheoDoiHieuLucPhapLy` cảnh báo văn bản hết hiệu lực/sửa đổi theo mốc ngày (`45/2013/QH13` → `31/2024/QH15`, `123/2020/NĐ-CP` → `70/2025/NĐ-CP`). Xem [Hướng dẫn RAG Pháp lý & Hiệu lực](docs/guides/vietnamese-rag.md).
  - **3. Rà Soát Hợp Đồng Redline, Chat Workspace Phòng Ban, MCP IDE & Zalo/Telegram**:
    - `BoSoSanhRaSoatHopDong` (`vai contract-review`): so sánh hợp đồng `.docx`/`.pdf`/`.txt` với mẫu chuẩn, chấm điểm an toàn pháp lý, xuất báo cáo Redline Markdown & Word `.docx`.
    - `QuanLyPhienChatWorkspace` + `StudioDashboard` (`vai studio --vaidb ... --workspace-dir ... --auth`): chat đa phiên cách ly `.vaidb` theo phòng ban, kéo-thả tệp thẳng vào Workspace, RBAC + Audit + Thumbs Up/Down kèm đáp án sửa `cau_tra_loi_sua`.
    - `MayChuMCPThucTe` (`vai mcp-server --transport stdio|sse --plugin-dir ...`): cắm trực tiếp vào Cursor, VS Code & Claude Desktop kèm tự động phơi bày công cụ V-Plugin 2.0.
    - `KetNoiZaloOA` & `KetNoiTelegram` (`vai connector both --workspace-dir ...`): xác thực chữ ký MAC/Secret, lưu lịch sử SQLite Workspace gom DPO, hỗ trợ lệnh nhanh `/hieuluc`, `/luat`, `/hopdong`. Xem [Hướng dẫn Hợp đồng, MCP & Zalo/Telegram](docs/guides/workbench-mcp-connectors.md).
  - **4. Động cơ LLM Tiếng Việt Zero-GPU Trên CPU (`vai pull` & `vai run`)**: `QuanLyMoHinhEdge` quản lý 8 mô hình GGUF tiếng Việt (`qwen2.5-1.5b/3b/7b`, `qwen3-4b`, `phogpt-4b`, `vistral-7b`, `seallm-7b`, `gemma2-2b`), tự chọn lượng tử theo RAM thực, tải HTTP Range Resume, chạy 3 tầng Native (`llama-cpp-python` → `llama-cli` → GGUF mmap NumPy) và mở trực tiếp `.vaidb` qua `vai run ... --rag cong_ty.vaidb --admin`. Xem [Hướng dẫn Edge CPU LLM Runtime](docs/guides/edge-cpu-llm-runtime.md).
  - **5. Remote GPU Burst Flywheel & Cổng Kiểm Định Hồi Quy 4 Trục (`vai train-cloud`, `vai export-dpo`, `vai eval-gate`)**: Đẩy thẳng phản hồi DPO đã ẩn PII từ Workspace (`--from-workspace`) lên `NhaCungCapModal` (Modal Serverless GPU), `NhaCungCapSSH` (máy chủ GPU nội bộ) hoặc `NhaCungCapCucBo`, kiểm trần ngân sách USD/VNĐ, ký HMAC-SHA256 và chấm hồi quy 4 trục (`CongKiemDinhMoHinh`: trích dẫn luật, hiệu lực văn bản, chống ảo giác/PII, kháng Prompt Injection) trước khi promote Production. Xem [Hướng dẫn Remote GPU Flywheel](docs/guides/remote-gpu-flywheel.md).
  - **6. Trải Nghiệm 5 Phút (`vai demo`), Minh Bạch 3 Tầng (`vai status-map`) & Kiểm Định Phát Hành (`vai release-check`)**: Demo offline trong 1 lệnh, phân tách minh bạch **Core Production / Optional Extensions / Experimental-Simulated**, cổng CI/CD TestPyPI & PyPI OIDC Trusted Publishing (`publish.yml`) cùng `benchmarks/Dockerfile.benchmark` tái lập. Xem [Hướng dẫn 5-Minute DX & Release](docs/guides/five-minute-dx-and-release.md).

- **Hệ Sinh Thái Khép Kín 5 Bước Tự Tối Ưu Hóa & Tự Chữa Lành (Closed-Loop Autonomous AI Ecosystem) (v31.0.2)**:
  - **1. Bánh đà Dữ liệu Khép kín (Continuous Data Flywheel & DPO Loop)**: Mọi tương tác suy luận, phản hồi người dùng hoặc đánh giá từ Tác tử Tự sửa (`Self-Refine`) được tự động đóng gói theo cấu trúc `(prompt, chosen, rejected)` vào `HoDuLieu` (DataLake DVC) kèm mã băm SHA-256. `DPOTrainer` tự động đọc dữ liệu và tối ưu hóa trọng số mô hình, đăng ký và thăng hạng trực tiếp lên `QuanLyMoHinh` (`production`).
  - **2. Vận hành Tự phục hồi & Auto-Rollback (Self-Healing Ops Loop)**: `MachCat` (Circuit Breaker) và `KiemTraSucKhoe` (Health Checker) liên kết trực tiếp với Model Registry. Khi phát hiện sự cố dồn dập hoặc suy thoái chất lượng, hệ thống tự động kích hoạt quy trình rollback về phiên bản ổn định gần nhất kèm xác thực chữ ký số HMAC-SHA256 chống giả mạo.
  - **3. Đóng gói Biên 1-Click & Phân phối Swarm CPU**: Tự động chuyển đổi mô hình sang GGUF lượng tử hóa (`XuatGGUF`), phân mảnh layer song song (Pipeline Parallelism) tới cụm máy tính văn phòng / Mini PC (`DieuPhoiSwarmCPU`), và tự động sinh chữ ký chứng thực mật mã học Verifiable AI Merkle Audit Path $O(\log N)$ kết hợp Schnorr zkML.
  - **4. Vòng lặp Đồng tiến hóa Đối kháng (Adversarial Co-Evolution Loop)**: Tác tử `RedTeamSimulator` tự động tập kích các cuộc tấn công jailbreak, Prompt Injection tiếng Việt, System Prompt leak và Payload ẩn $\rightarrow$ Tường lửa AI đa tầng (`TuongLuaBaoVeFramework`) và Luồng An toàn (`LuongAnToan`) ngăn chặn $\rightarrow$ Tự động sinh tập dữ liệu đối kháng DPO $\rightarrow$ Tôi luyện trực tiếp các trọng số mô hình chống lại các cuộc tấn công (`hardened`).
  - **5. Hợp nhất Đồ thị Trạng thái Phân tán (Distributed StateGraph Unification)**: Đồ thị trạng thái DAG (`DoThiTrangThai` / `StateGraph`) hợp nhất toàn diện mọi mắt xích: Kiểm duyệt an ninh $\rightarrow$ Truy xuất tri thức RAG $\rightarrow$ Suy luận mô hình / Swarm CPU $\rightarrow$ Tự sửa chất lượng Self-Refine $\rightarrow$ Khử độc DLP đầu ra. Hỗ trợ nhúng trực tiếp Tác tử (`them_nut_tac_tu`), Swarm CPU (`them_nut_swarm`), và Cụm điều phối (`them_nut_cum`) với bộ alias chuẩn LangGraph.
  - **Cầu nối Thống nhất `QuyTrinhDauCuoi` (`HeSinhThaiAI` / `EndToEndWorkflow`)**: Cho phép kích hoạt toàn bộ chu trình 5 bước khép kín chỉ bằng một lệnh gọi `chay_dau_cuoi(...)`.

- **Suy Luận Bầy Đàn Phân Tán Trên CPU Văn Phòng (Swarm CPU Inference & Pipeline Parallelism) (v31.0)**:
  - `NutSwarmCPU` (`SwarmCPUNode`): Cho phép kết nối nhiều máy tính CPU văn phòng hoặc Mini PC cấu hình khiêm tốn (2GB - 4GB RAM, không có card đồ họa GPU) qua mạng nội bộ LAN hoặc WAN để chia sẻ tải tính toán mô hình ngôn ngữ lớn theo kiến trúc Pipeline Parallelism.
  - `DieuPhoiSwarmCPU` (`SwarmInferenceOrchestrator`): Tự động phân chia mô hình thành các phân đoạn layer liền mạch, điều phối truyền nhận tensor trạng thái ẩn (hidden states) và hỗ trợ sinh chuỗi theo luồng (streaming generation) thời gian thực.
- **Xác Thực Suy Luận Mật Mã Học (Verifiable AI Inference & Proof of Inference) (v31.0)**:
  - `CayMerkleSuyLuan` (`InferenceMerkleTree`): Cây băm mật mã Merkle Tree chuẩn RFC 6962 (tách miền lá `0x00` / nút trong `0x01`, không nhân bản lá lẻ) đóng gói trạng thái ẩn và các slice kích hoạt, sinh bằng chứng kiểm toán $O(\log N)$.
  - `XacThucSuyLuanMaoHiem` (`VerifiableInference`): Cam kết mật mã học Schnorr Fiat-Shamir NIZK trên nhóm RFC 5114 MODP 2048-bit $p$, nhóm con bậc nguyên tố 256-bit $q$ ($g^z \equiv a \cdot y^e \pmod p$), gắn chặt `node_id`, `layer_id` và shape/dtype tensor vào challenge hash.
  - **Xác thực Ma trận Freivalds**: `KiemChungFreivalds` hỗ trợ kiểm chứng nhanh $O(k \cdot n^2)$ cho phép chiếu tuyến tính 1 tầng $Y = XW + b$ (các luồng đa tầng phi tuyến báo rõ `khong_ap_dung` thay vì giả lập).
- **Giao Diện Dòng Lệnh Mới**: Bổ sung `vai swarm` (suy luận phân tán trên CPU) và `vai verify` (kiểm toán chứng thực mật mã).
- **Tối Ưu Hóa Thuần Việt (Vietnamese-Centric Optimization) (v30.2)**:
  - `BoTachTuTiengVietSieuToc` (`FastVietnameseTokenizer`, `NativeVietnameseTokenizer`): Tách từ ghép chuyên sâu với **từ điển tiếng Việt nén 38.372 từ** (32.877 từ ghép MIT đóng gói sẵn trong wheel, đồng nhất Zero-Width/BiDi Rust/Python, Maximum Matching + FST/Viterbi DAG), chuẩn hóa dấu thanh mới/cũ (`hòa, thúy, ủy`), chuẩn hóa Unicode NFC và chuyển đổi không dấu (unaccent/slugify); dùng chung với `KiemTraChinhTa`.
  - `BoNhungVanBanSieuToc` (`FastVietnameseEmbedder`): Vector hóa văn bản 256 chiều dạng Subword/Syllable Feature Hashing chuẩn hóa L2 kèm tách âm tiết từ ghép và truy vấn vector zero-copy Rust (`fast_top_k_cosine_bytes`) mà không phụ thuộc PyTorch hay GPU.
  - `BoToiUuNguCanhHanhChinh` (`VietnameseAdministrativeRAG`): Tự động phân cấp cấu trúc văn bản pháp lý (Chương, Mục, Điều, Khoản, Điểm, Căn cứ pháp lý, Số hiệu văn bản), phân đoạn thông minh bảo toàn văn cảnh mẹ (Hierarchy-Preserving Chunking) và viện dẫn điều luật chuẩn xác.
- **Bộ Đo Lường Tái Lập Định Lượng (`BoDoLuongDoiDau`)**:
  - Đo trực tiếp trên chính phần cứng người dùng bằng tiến trình Python cô lập (cold-start, `/proc/self/status` `VmRSS`) và trên tập mở 100 điều khoản / 35 QA `legal_vn_corpus.jsonl` — không sử dụng bảng đối đầu tĩnh giả lập. Nếu cài song song `langchain` hoặc `llama_index`, bộ đo tự động chạy đối sánh trực tiếp 1-1. Hỗ trợ xuất bảng Markdown/LaTeX.
- **Agentic StateGraph & Human-in-the-Loop (v30.1)**: Đồ thị trạng thái chu trình (`DoThiTrangThai`, `StateGraph`, `START`, `END`), điều phối tác tử tự chủ, các kênh tích lũy dữ liệu và cơ chế ngắt can thiệp người dùng (HITL interrupt & resume).
- **Tác tử Nghiên cứu Sâu & Tự Sửa Code (v30.1)**: 
  - `TacTuNghienCuuSau` (`DeepResearchAgent`): Tự động phân rã chủ đề nghiên cứu phức tạp, truy vấn thông tin đa vòng, phản biện chéo (Critic-in-the-loop) và xuất báo cáo điều hành toàn diện kèm thang điểm tin cậy số liệu.
  - `TacTuLapTrinhTuSua` (`CodeHealingAgent`): Chẩn đoán lỗi cú pháp và ngoại lệ qua cây cú pháp trừu tượng (AST), tự động áp dụng chiến lược vá mã nguồn và xác nhận bằng kiểm thử hồi quy trong tiến trình con cô lập.
- **Động cơ Native C & Zero-GPU Edge Engine (v30.1)**: 
  - `DongCoSuyLuanEdge` (`EdgeInferenceEngine`, `ZeroGPUInferenceEngine`): Suy luận mô hình ngôn ngữ streaming trực tiếp trên CPU phổ thông không cần card đồ họa GPU hay cài đặt CUDA phức tạp.
  - `BoGiaiLuongTuQ4`: Giải lượng tử hóa Q4_0 và Q8_0 vector hóa tốc độ cao.
  - `GGUFMemoryMappedStreamer`: Nạp mô hình dạng mmap không gây tràn bộ nhớ RAM.
  - `CauNoiCtypesNative` (`NativeLlamaBridge`): Cầu nối C-types nhúng trực tiếp thư viện native C/C++ (`libllama.so`, `llama.dll`).
- **Trình Đóng Gói 1-Click Desktop Offline (v30.1)**: `GoiDocLapDesktop` (`DesktopPackager`) sinh launcher tự hành cho Windows (`run_desktop.bat`) và Linux/macOS (`run_desktop.sh`) chạy độc lập 100% offline không cần cấu hình.
- **No-Code Web Studio & 3 Bộ Giải pháp Doanh nghiệp (v30.1)**: Giao diện Studio Dashboard (mặc định bind an toàn `127.0.0.1:8080`) và 3 giải pháp chuyên ngành: Tổng đài CSKH, Trợ lý Pháp lý, Trích xuất Thông minh CCCD/Hóa đơn.
- **Lõi Tối Giản (`numpy`-only core) & Tăng Tốc Rust**: Lõi thư viện Python thuần chỉ phụ thuộc `numpy`, nạp lạnh chỉ `24.68 ms` / `12.82 MB` `VmRSS` trên CPU Celeron N3050, kèm phần mở rộng tăng tốc Rust (`rust_core`/PyO3 `abi3-py39`).

---

### 📊 Kết Quả Đo Lường Thực Tế Tái Lập (Direct Quantitative Benchmarks — Zero-GPU CPU)

> Tái lập trực tiếp trên máy của bạn bằng `vai benchmark --compare`, `vai benchmark --suite legal-temporal`, `vai benchmark --suite align-safety` hoặc container cố định `benchmarks/Dockerfile.benchmark`.

| Chỉ số Đánh giá (Metrics) | **Vietnamese AI (`v31.2.0`)** | Phương pháp & Tập dữ liệu mở |
|:---|:---:|:---|
| **Thời gian khởi động lạnh (Cold-start)** | **24.68 ms** | Tiến trình Python cô lập (`import vietnamese_ai`) |
| **RAM cư trú khi nạp lõi (`VmRSS`)** | **12.82 MB** | `/proc/self/status` `VmRSS` sau import lõi |
| **RAM khi chạy Legal RAG (100 điều khoản)** | **37.39 MB** | `/proc/self/status` `VmRSS` sau lập chỉ mục & truy vấn |
| **Tốc độ Tách từ tiếng Việt (Rust Trie)** | **354,047.7 từ/s** | `BoTachTuTiengVietSieuToc` trên 100 điều khoản mở |
| **Truy hồi Pháp lý mở (100 điều / 35 QA)** | **Hit@1 = 100.0% \| Hit@3 = 100.0% \| MRR = 1.000** | `RAGPipeline` (`onnx_viet`) trên `legal_vn_qa.jsonl` |
| **`ViLegal-TempBench` (160 điều / 68 QA)** | **R@3 = 1.000 \| MRR@10 = 0.968 \| TVA = 1.000** | TCA = `12/12`, Multi-Hop = `1.000`, Trap = `1.000`, Abstention F1 = `1.000` (`47.02 ms`/câu) |
| **`ViAlign-SafeBench` (5 vòng, 256-dim proxy)** | **Naive `0.5969` → SafeLoop `1.0000`** | Sàn An toàn `1.0000`, `0` hồi quy ngầm, Paired Bootstrap `+0.2500` ($p=0.0000$) |

### 📋 Phương Pháp Thực Nghiệm & Thiết Lập Đo Kiểm (Benchmark Methodology & Transparency)

1. **Tập Dữ Liệu Mở Đi Kèm Mã Nguồn (Open Reproducible Datasets)**:
   - Nguồn chuẩn duy nhất nằm trong `vietnamese_ai/rag/legal/data/` và được kiểm toán đồng bộ SHA-256 sang `benchmarks/datasets/` (`scripts/sync_legal_datasets.py --check` + `vai release-check`).
   - **`legal_vn_corpus.jsonl` & `legal_vn_qa.jsonl`**: `100` điều khoản quy phạm pháp luật Việt Nam (Bộ luật Dân sự, Bộ luật Lao động, Luật Doanh nghiệp, Luật Đất đai, Luật BHXH 2024, Luật Thuế GTGT 2024, Nghị định 13/2023/NĐ-CP, v.v.) và `35` câu hỏi chuẩn.
   - **`ViLegal-TempBench`**: `160` điều khoản (`60` đa phiên bản + `100` nền), `38` cạnh đồ thị điều khoản có mốc hiệu lực và `68` truy vấn thuộc 4 phân nhóm `T1`/`T2`/`T3`/`T4` (bao gồm bẫy chỉ nêu tên luật hết hiệu lực và câu hỏi pháp lý Việt Nam ngoài phạm vi kho).

2. **Cấu Hình Phần Cứng Thực Đo Hiện Tại**:
   - **Bộ vi xử lý (CPU)**: Intel Celeron N3050 @ 1.60 GHz (2 lõi vật lý, 6W TDP, SSE4.2/POPCNT, không AVX2, không GPU rời).
   - **Bộ nhớ trong (RAM)**: 3.2 GiB RAM.
   - **Hệ điều hành & Môi trường**: Linux `x86_64`, Python `3.14.4` kèm lõi tăng tốc `vietnamese_ai_rust` (`abi3-py39`).

3. **Nguyên Tắc Minh Bạch Đối Sánh & Kiểm Định Thống Kê**:
   - Không hiển thị số liệu tĩnh giả lập cho các thư viện ngoài chưa cài đặt; nếu cài song song `langchain` / `llama_index`, `vai benchmark --compare` sẽ tự động chạy tiến trình cô lập đo trực tiếp trên cùng máy.
   - Các thực nghiệm Ablation (`ViLegal-TempBench`, `ViAlign-SafeBench`) sử dụng hạt giống cố định `seed=42` và kiểm định Paired Bootstrap 1,000 mẫu ($p$-value & khoảng tin cậy 95%).

4. **Tách Bạch Thực Nghiệm Cụm Swarm CPU (Giả Lập Nội Bộ vs Mạng LAN Vật Lý)**:
   - **Chế độ 1 - Giả lập Đa Tiến trình (In-Process Multi-Worker Simulation)**: Sử dụng hàm `khoi_tao_cum_gia_lap(so_node=3, so_core_moi_node=4)` nhằm cô lập và đo lường **thông lượng tính toán thuần túy trên CPU** (`compute latency`) theo kiến trúc Pipeline Parallelism mà không chịu ảnh hưởng bởi độ trễ card mạng.
   - **Chế độ 2 - Mạng LAN Vật lý Thực tế (Physical LAN Network Transport)**: Sử dụng hàm `them_node_mang_vat_ly(endpoint_url=..., dai_layer=...)` kết nối các máy trạm thực tế qua mạng LAN. Trường dữ liệu `phan_ra_do_tre` trong nhật ký kiểm toán tách bạch 100% giữa **thời gian tính toán CPU** (`tinh_toan_cpu_ms`) và **thời gian truyền tải gói tin tensor qua mạng** (`truyen_tai_mang_lan_ms`).

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

### Cơ bản (Core — chỉ phụ thuộc NumPy)

Cài đặt mặc định chỉ kéo về phụ thuộc tối thiểu cho lõi (`vietnamese_ai.core`, `vietnamese_ai.utils`, `XuLySo`, `Zero-GPU Edge Engine`):

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

### Cài đặt theo phân hệ mở rộng (`optional-dependencies`)

```bash
# Học máy cổ điển & Chuỗi thời gian (scikit-learn, pandas)
pip install "vietnamese-ai[ml]"

# NLP tiếng Việt (underthesea)
pip install "vietnamese-ai[vietnamese-nlp]"   # hoặc vietnamese-ai[nlp]

# Deep Learning & Fine-Tuning (PyTorch, Transformers, PEFT, TRL, Datasets)
pip install "vietnamese-ai[torch]"
pip install "vietnamese-ai[dl]"

# RAG Pháp lý đọc DOCX/PDF/ảnh scan OCR & Kho tri thức nâng cao
pip install "vietnamese-ai[legal-rag]"
pip install "vietnamese-ai[rag]"

# Kết nối LLM Cloud APIs, Local llama.cpp CPU & Remote GPU Serverless (Modal)
pip install "vietnamese-ai[llm]"
pip install "vietnamese-ai[local-llm]"
pip install "vietnamese-ai[cloud-gpu]"

# Thị giác máy tính & OCR, Xuất mô hình (ONNX/GGUF), API Server, Mobile/Web3/Robotics
pip install "vietnamese-ai[vision]"
pip install "vietnamese-ai[export]"
pip install "vietnamese-ai[api]"
pip install "vietnamese-ai[mobile]"
pip install "vietnamese-ai[web3]"
pip install "vietnamese-ai[robotics]"

# Hoặc dùng trình thiết lập thông minh 3 tầng tự phát hiện phần cứng
vai setup --profile legal-rag --apply

# Đầy đủ các nhóm mở rộng tiêu chuẩn (không gồm gói cần CUDA/biên dịch riêng như unsloth/llama-cpp-python)
pip install -e ".[all]"

# Môi trường phát triển & kiểm thử (tách riêng khỏi requirements.txt production)
pip install -r requirements-dev.txt
```

### Kiến trúc Lõi vs Phần Mở rộng & Bản đồ Minh bạch 3 Tầng (`vai status-map`)

Framework phân định rõ 3 tầng năng lực ngay trong runtime (kiểm tra trực tiếp bất kỳ lúc nào qua `vai status-map` hoặc `GET /api/system/status-map` trên Web Studio, tương thích ngược 100% với đường dẫn import cũ):

- **Core Production (`vietnamese_ai.core`, `vietnamese_ai.rag`, `vietnamese_ai.solutions`, `vietnamese_ai.connectors`, `vietnamese_ai.ui`, `vietnamese_ai.security`)**: Thực thi thật 100% chỉ với Python + NumPy (0 thư viện nặng bị nạp sớm khi `import vietnamese_ai`, 349 biểu tượng công khai khớp chuẩn `.pyi`).
- **Optional Extensions (`vietnamese_ai.extensions.*`)**: Không gian tên lazy-load cho các phân hệ mở rộng tùy chọn cài thêm qua extras (`ml`, `legal-rag`, `local-llm`, `cloud-gpu`, `dl`, `api`, `export`, `vision`...).
- **Experimental / Simulated**: Các mô-đun nghiên cứu/mô phỏng được gắn nhãn tường minh qua `@gan_nhan_tinh_nang` và liệt kê rõ trong bảng bên dưới.

### Yêu cầu Công cụ Ngoài & Minh bạch Trạng thái Tính năng (External Toolchain Requirements)

> Lưu ý: Nguyên tắc **Zero-Dependency** áp dụng cho lõi Python/NumPy và V-UI. Các tính năng thử nghiệm/tăng tốc phần cứng hoặc xuất định dạng nền tảng gốc dưới đây yêu cầu công cụ bên ngoài hoặc chạy ở chế độ mô phỏng/fallback khi thiếu toolchain:

| Phân hệ / Class | Trạng thái | Công cụ ngoài bắt buộc (cho chế độ Native) | Hành vi khi không có công cụ ngoài |
| :--- | :---: | :--- | :--- |
| `EvoJITCompiler`, `EvoKernelCPU`, `FusedDenseNormReLU` | **External Dependency / Experimental** | Trình biên dịch **`g++`** (GCC/Clang, `-fopenmp`) | Không biên dịch JIT C++ được (báo lỗi rõ ràng hướng dẫn cài `g++`). |
| `TriKhaiDiDong` (`xuat_tflite`, `xuat_coreml`, `xuat_executorch`) | **Experimental / Simulated Format** | `tensorflow`/`ai-edge-litert` + Android NDK; `coremltools` + Xcode (macOS); PyTorch ExecuTorch | Xuất container nội bộ (`VAI_TFL`, `VAI_CLM`, `VAI_PTE`) chạy được qua `chay_suy_luan_di_dong()`; `benchmark_edge()` mô phỏng trên CPU host. |
| `TriKhaiDiDong.xuat_onnx_mobile` / `XuatONNX` | **External Dependency** | `skl2onnx` + `onnxruntime` (`vietnamese-ai[export]`) | Fallback sang container đồ thị nội bộ `VAI_ONX` chạy trên NumPy. |
| `CauNoiCtypesNative` & `DongCoSuyLuanEdge` | **Core + Optional Native C++** | `libllama.so`/`llama.dll` hoặc `llama-cli` | Fallback Python thuần (đọc mmap GGUF & giải lượng tử Q4/Q8 thật; sinh văn bản fallback tối thiểu). |
| `DieuPhoiSwarmCPU` | **Experimental** | `gguf` (tùy chọn), `tailscale` (tùy chọn WAN) | `khoi_tao_cum_gia_lap()` chạy **mô phỏng đa worker trong cùng tiến trình**; chế độ mạng LAN vật lý qua `them_node_mang_vat_ly()` dùng TCP socket thật. |
| `RedTeamSimulator`, `RedTeamAgent`, `BlueTeamAgent` | **Experimental / Orchestration** | `RedTeamSimulator`: không; `RedTeamAgent`/`BlueTeamAgent`: cần `llm` | `RedTeamSimulator` là bộ mô phỏng tấn công dựa trên bộ kịch bản tĩnh tự phát triển; xem chi tiết giới hạn trong [SECURITY.md](SECURITY.md). |
| `ThucThiDocLap` (V-Sandbox) & `MoiTruongCachLy` | **Application-Level Sandbox** | Không (dùng Python `ast`, `subprocess`, `multiprocessing`) | Cách ly mức ứng dụng; production chạy mã không tin cậy **bắt buộc** kết hợp container/VM OS (Docker, seccomp, gVisor). |
| `rust_core/` (`vietnamese_ai_rust.so`) | **Optional Native Rust** | `rustc`, `cargo`, `maturin` (khi build từ source) | Tự động fallback sang Python/NumPy thuần. |
| `wasm_node`, `webxr_node`, `tee_zkp_node`, `mobile_tools`, `salm`, `federated`, `sdks/`, `evonet_os/` | **Deprecated / Experimental (`v31.2.0`)** | Không (mô phỏng / bản mẫu thử nghiệm) | Giữ tương thích ngược kèm `DeprecationWarning` trong `v31.2.0`; dự kiến tách khỏi gói chính ở `v32.0.0`. Nhóm tài chính đã chuyển sang `vietnamese_ai.plugins.finance`. |

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

### Mobile/Edge

| Module | Chức năng |
|---|---|
| `TriKhaiDiDong` | Triển khai TFLite, CoreML, ONNX Mobile + quantization INT8 |
| `PowerManager` | Điều tiết độ chính xác tự động theo pin (`8-bit` / `4-bit` / `1.58-bit`) |

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

### Multi-Agent System (v12.0)

| Module | Chức năng |
|---|---|
| `TacTu` | Tác tử AI (Agent) dùng vòng lặp ReAct, lập kế hoạch và thực thi |
| `CongCu` & `@cong_cu` | Tool Calling: tích hợp Python function, tìm kiếm web, đọc file |
| `BoNhoTacTu` | Quản lý buffer memory theo role (system, user, assistant, tool) |
| `HeThongDaTacTu` | Điều phối tương tác giữa nhiều tác tử để giải quyết task phức tạp |

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

### RAG Pipeline (v10.0)

| Module | Chức năng |
|---|---|
| `CSDLVector` | Vector database: cosine/L2/inner_product, save/load, metadata |
| `CatVanBan` | Text chunking: theo câu/đoạn/từ/ký tự, sliding window |
| `TrichXuat` | Retriever: semantic, keyword (BM25), hybrid search |
| `SapXepLai` | Reranker: MMR, cross-encoder, keyword, position-aware |
| `RAGPipeline` | Full RAG: chunk → embed → retrieve → rerank → generate |

### Serving & Streaming (v10.0)

| Module | Chức năng |
|---|---|
| `MayChuBatch` | Dynamic batching server: multi-worker, latency tracking |
| `MayChuStream` | SSE streaming: token-by-token, multi-client |
| `BoGioiHanTocDo` | Rate limiter: token bucket, sliding window, per-client |

### Prompt Engineering (v10.0)

| Module | Chức năng |
|---|---|
| `MauPrompt` | Prompt templates: {{biến}}, conditional, 6 built-in templates |
| `ChuoiPrompt` | Chain-of-thought, few-shot, sequential chains |
| `LuongAnToan` | Guardrails: content filter, PII detection, format validation |
| `PhanTichDauRa` | Output parser: JSON, markdown tables, code blocks, lists |

### NLP Extensions (v10.0)

| Module | Chức năng |
|---|---|
| `NhanDienThucThe` | NER: regex + dictionary + underthesea, 8 entity types |
| `HoiDapTiengViet` | QA: extractive, TF-IDF ranking, knowledge base |
| `TomTatVanBan` | Summarization: extractive (TF-IDF + position) + abstractive |
| `DichThuat` | Translation: dictionary + LLM + hybrid, en/vi |
| `KiemTraChinhTa` | Spell check: edit distance, tone marks, auto-learn |

### Model Compression (v10.0)

| Module | Chức năng |
|---|---|
| `HocRutGon` | Knowledge Distillation: soft labels, temperature, ensemble |
| `CatTiaMoHinh` | Pruning: magnitude, structured, iterative, random |

### Production Hardening (v10.0)

| Module | Chức năng |
|---|---|
| `KiemTraSucKhoe` | Health check: readiness/liveness probes, system monitoring |
| `MachCat` | Circuit breaker: 3 states, auto-recovery, fallback |
| `LoggerCauTruc` | Structured logging: JSON, timing, context, request tracking |
| `QuanLyMetrics` | Metrics: counter, gauge, histogram, Prometheus export |
| `LamNongModel` | Model warm-up: pre-load, auto-refresh, pool management |

### CLI (`vai`)

```text
vai demo [all|legal-rag|edge-chat]       Trình diễn 1-Click trọn vẹn 4 trụ cột offline
vai doctor / vai status-map              Chẩn đoán phần cứng & bản đồ minh bạch 3 tầng
vai pull qwen2.5-3b                      Tải LLM GGUF tiếng Việt tự chọn lượng tử theo RAM
vai run qwen2.5-3b --rag kb.vaidb --admin Chạy LLM Zero-GPU gắn thẳng kho .vaidb pháp lý
vai rag --db kb.vaidb --docs ./luat --semantic-onnx --admin --query "..."
vai legal-graph 45/2013/QH13 --mermaid   Truy vấn Đồ thị Tham chiếu Pháp lý Việt Nam
vai contract-review hd.docx --template mau.docx --output-docx redline.docx
vai studio --vaidb kb.vaidb --workspace-dir ./ws --port 8080
vai mcp-server --transport stdio|sse     Cắm trực tiếp vào Cursor, VS Code & Claude Desktop
vai connector both --solution phap-ly --workspace-dir ./ws
vai export-dpo --workspace-dir ./ws      Xuất phản hồi người dùng sang DPO JSONL đã ẩn PII
vai train-cloud --from-workspace ./ws --provider modal --eval-4-axis
vai eval-gate / vai release-check        Kiểm định hồi quy 4 trục & cổng phát hành PyPI
```

---

## Sử dụng nhanh

### 1. Chu Trình Khép Kín 5 Bước Tự Tối Ưu Hóa & Tự Chữa Lành (v31.0.2)

Khởi chạy toàn bộ chu trình 5 bước chỉ trong **1 câu lệnh duy nhất**: DataLake DVC $\rightarrow$ RAG Indexing $\rightarrow$ Huấn luyện LoRA $\rightarrow$ GGUF Export & Swarm CPU Sharding $\rightarrow$ Đồng tiến hóa đối kháng Red Team $\rightarrow$ Đồ thị trạng thái StateGraph.

```python
from vietnamese_ai import QuyTrinhDauCuoi

# Khởi tạo Hệ sinh thái Đầu-Cuối tích hợp sẵn Tường lửa AI & Circuit Breaker
ecosystem = QuyTrinhDauCuoi(ten_du_an="Ecosystem_VN", bat_tuong_lua=True)

du_lieu_mau = [
    {"instruction": "AI là gì?", "output": "AI là Trí tuệ nhân tạo được xây dựng bởi con người."},
    {"instruction": "Thủ đô của Việt Nam là gì?", "output": "Thủ đô của Việt Nam là Hà Nội."},
]

# Chạy trọn vẹn 5 mắt xích khép kín chỉ bằng 1 hàm
ket_qua = ecosystem.chay_dau_cuoi(
    du_lieu=du_lieu_mau,
    mo_hinh="vgpt2",
    nhiem_vu="sft",
    dung_lora=True,
    xuat_gguf=True,
    phan_phoi_edge=True,             # Bước 3: Phân mảnh layer sang Edge Swarm CPU
    so_node_edge=3,
    dong_tien_hoa_doi_khang=True,    # Bước 4: Red Team tập kích & Tôi luyện DPO
    dung_do_thi_trang_thai=True,     # Bước 5: Thực thi qua Đồ thị DAG phân tán
    cau_hoi_kiem_thu="Thủ đô của Việt Nam là gì?",
)

print("Câu trả lời:", ket_qua["kiem_thu"]["tra_loi"])
print("Trạng thái cụm Swarm CPU:", ket_qua["edge_swarm"]["cum_swarm"])
print("Độ an toàn đối kháng:", ket_qua["dong_tien_hoa_doi_khang"]["ty_le_bao_ve"], "%")
```

### 2. Đồ Thị Trạng Thái Phân Tán (Distributed StateGraph & Swarm) (v31.0.2)

Hợp nhất các phân hệ Kiểm duyệt an ninh, RAG, Swarm CPU, Tác tử và Làm sạch DLP thành một đồ thị có hướng (DAG) khép kín:

```python
from vietnamese_ai import DoThiTrangThai, END

# Khởi tạo đồ thị trạng thái hoặc tự động tạo từ QuyTrinhDauCuoi
dothi = ecosystem.tao_do_thi_trang_thai(che_do_suy_luan="swarm")
app = dothi.bien_dich()

# Thực thi qua đồ thị trạng thái với đầy đủ Checkpointer lưu vết
state_out = app.thuc_thi({"cau_hoi": "Trình bày về tiềm năng kinh tế số Việt Nam"})
print("Trạng thái an toàn:", state_out["an_toan"])
print("Câu trả lời:", state_out["tra_loi"])
print("Chứng thực Swarm zkML Proof:", state_out.get("swarm_proof") is not None)
```

### 3. Đồng Tiến Hóa Đối Kháng: Red Team vs Tường Lửa vs DPO Hardening (v31.0.2)

Tác tử Red Team tự động bắn phá hệ thống bằng các payload Jailbreak / Prompt Injection độc hại, tự động đóng gói cặp DPO và tôi luyện trọng số mô hình:

```python
# Tự động tập kích và tôi luyện DPO chống Jailbreak
ket_qua_co_evolution = ecosystem.tap_kich_an_ninh_va_tu_luyen(
    so_mau=10,
    tu_dong_dpo=True,
    ten_dataset="adversarial_dpo.jsonl"
)

print(f"Tổng mẫu tấn công: {ket_qua_co_evolution['tong_so_mau']}")
print(f"Số lần chặn thành công: {ket_qua_co_evolution['so_lan_chan']}")
print(f"Tỷ lệ bảo vệ: {ket_qua_co_evolution['ty_le_bao_ve']:.1f}%")
print(f"Đã đăng ký phiên bản tôi luyện: {ket_qua_co_evolution['dpo_hardening']['phien_ban_moi']}")
```

### 4. Tự Phục Hồi Production & Auto-Rollback Có Chữ Ký HMAC (v31.0.2)

Khi mô hình Production gặp sự cố suy thoái hoặc lỗi dồn dập khiến Circuit Breaker ngắt mạch, hệ thống tự động rollback về phiên bản an toàn gần nhất:

```python
# Kiểm tra sức khỏe toàn diện
suc_khoe = ecosystem.kiem_tra_suc_khoe()
print("Trạng thái:", suc_khoe["trang_thai"])  # 'healthy' / 'unhealthy'

# Tự động phát hiện lỗi và rollback phiên bản trong Model Registry
ket_qua_phuc_hoi = ecosystem.tu_dong_phuc_hoi()
if ket_qua_phuc_hoi["da_phuc_hoi"]:
    print(f"Đã tự động rollback về phiên bản an toàn: {ket_qua_phuc_hoi['chi_tiet_phuc_hoi']}")
```

### 5. Suy Luận Bầy Đàn CPU & Xác Thực Mật Mã zkML (v31.0)

```python
from vietnamese_ai import DieuPhoiSwarmCPU, XacThucSuyLuanMaoHiem, CayMerkleSuyLuan
import numpy as np

# 1. Khởi tạo cụm Swarm phân tán trên 3 máy tính CPU văn phòng (Pipeline Parallelism)
swarm = DieuPhoiSwarmCPU(tong_so_layer=6, bat_xac_thuc_mat_ma=True)
swarm.khoi_tao_cum_gia_lap(so_node=3, so_core_moi_node=4)

# 2. Suy luận streaming đồng thời tự động kiểm toán Byzantine fault tolerance
prompt = "Chiến lược phát triển trí tuệ nhân tạo và chuyển đổi số tại Việt Nam"
print("AI (Swarm CPU): ", end="")
for token in swarm.sinh_stream(prompt, do_dai_toi_da=12):
    print(token, end="", flush=True)
print()

# 3. Xác thực cam kết mật mã học Fiat-Shamir zkML (O(log N) Merkle Tree)
verifier = XacThucSuyLuanMaoHiem()
sample_tensor = np.array([0.15, -0.42, 0.88, 0.05], dtype=np.float32)
cam_ket = verifier.tao_cam_ket_zkml(sample_tensor)
hop_le = verifier.xac_thuc_cam_ket(cam_ket["y"], cam_ket["a"], cam_ket["e"], cam_ket["z"])
print(f"Xác thực Schnorr zkML: {'CHÍNH XÁC (HỢP LỆ)' if hop_le else 'GIAN LẬN'}")
```

### 2. Tách Từ & Chuẩn Hóa Tiếng Việt Siêu Tốc (v30.2)

```python
from vietnamese_ai import BoTachTuTiengVietSieuToc, BoNhungVanBanSieuToc

# Tách từ ghép và chuẩn hóa dấu tiếng Việt tốc độ cao (> 20,000 từ/s)
tok = BoTachTuTiengVietSieuToc()
tokens = tok.tach_tu("Hội đồng nhân dân và ủy ban nhân dân thành phố Hà Nội.", noi_tu_ghep=True)
print(tokens)  # ['Hội_đồng_nhân_dân', 'và', 'ủy_ban_nhân_dân', 'thành_phố', 'Hà_Nội', '.']

# Nhúng vector ngữ nghĩa thuần Việt siêu nhẹ không cần PyTorch / GPU
embedder = BoNhungVanBanSieuToc(so_chieu=256)
vec = embedder.nhung("Quy định xử phạt vi phạm hành chính")
print("Độ dài vector:", len(vec))
```

### 2. RAG Tối Ưu Ngữ Cảnh Pháp Lý & Hành Chính (v30.2)

```python
from vietnamese_ai import BoToiUuNguCanhHanhChinh

rag_admin = BoToiUuNguCanhHanhChinh()
van_ban_luat = """
Điều 2. Giải thích từ ngữ
1. Dữ liệu cá nhân là thông tin dưới dạng ký hiệu, chữ viết, chữ số...
2. Dữ liệu cá nhân cơ bản bao gồm: họ tên, ngày sinh, số CCCD...
"""
chunks = rag_admin.phan_doan_giu_ngu_canh(van_ban_luat, ten_tai_lieu="Nghị định 13")
ket_qua = rag_admin.truy_hoi_phap_ly("Dữ liệu cá nhân cơ bản là gì?", chunks, top_k=2)
for item in ket_qua:
    print(f"[{item['trich_dan_phap_ly']}] (Độ tương đồng: {item['diem_tuong_dong']}):")
    print(item["noi_dung"])
```

### 3. Đo Lường Định Lượng Tái Lập (`BoDoLuongDoiDau`)

```python
from vietnamese_ai import BoDoLuongDoiDau

suite = BoDoLuongDoiDau()
kq = suite.chay_benchmark_toan_dien()
print(suite.xuat_bang_so_sanh_markdown(kq))
# Hoặc xuất bảng LaTeX cho bài báo khoa học:
# print(suite.xuat_bang_so_sanh_latex(kq))
```

### 4. Đồ Thị Trạng Thái Agentic StateGraph (v30.1)

```python
from vietnamese_ai import DoThiTrangThai, START, END, kenh_them_danh_sach

# Khởi tạo đồ thị trạng thái với kênh cộng dồn danh sách
dothi = DoThiTrangThai(luat_gop={"tin_nhan": kenh_them_danh_sach})

def buoc_phan_tich(state):
    return {"tin_nhan": [f"Phân tích yêu cầu: {state.get('cau_hoi')}"]}

def buoc_tra_loi(state):
    return {"tin_nhan": ["Tạo câu trả lời hoàn chỉnh dựa trên dữ kiện đã thẩm định."]}

dothi.them_nut("phan_tich", buoc_phan_tich)
dothi.them_nut("tra_loi", buoc_tra_loi)
dothi.them_canh(START, "phan_tich")
dothi.them_canh("phan_tich", "tra_loi")
dothi.them_canh("tra_loi", END)

ung_dung = dothi.bien_dich()
ket_qua = ung_dung.chay({"cau_hoi": "Giải pháp AI tiếng Việt cho doanh nghiệp"})
print(ket_qua["tin_nhan"])
```

### 2. Tác tử Nghiên Cứu Sâu (Deep Research Agent) (v30.1)

```python
from vietnamese_ai import TacTuNghienCuuSau

# Tác tử tự động phân rã chủ đề, tìm kiếm đa vòng, phản biện chéo và chấm điểm tin cậy
agent = TacTuNghienCuuSau(so_vong_lap_toi_da=3)
ket_qua = agent.nghien_cuu("Chuyển đổi xanh và ứng dụng AI tại Việt Nam")

print("Điểm tin cậy dữ liệu:", ket_qua["diem_phan_bien"])
print(ket_qua["bao_cao_cuoi_cung"])  # Báo cáo Executive Summary chuẩn Markdown
```

### 3. Động Cơ Suy Luận Zero-GPU Native Streaming (v30.1)

```python
from vietnamese_ai import DongCoSuyLuanEdge

# Suy luận streaming trực tiếp trên CPU phổ thông không cần GPU hay CUDA
dong_co = DongCoSuyLuanEdge(so_luong_thread=4)

print("Streaming phản hồi:")
for token in dong_co.sinh_van_ban_streaming("Chiến lược chuyển đổi số quốc gia 2030"):
    print(token, end="", flush=True)

thong_ke = dong_co.lay_thong_ke_hieu_nang()
print(f"\nTốc độ xử lý: {thong_ke['tokens_tren_giay']:.1f} tokens/giây")
```

### 4. Bộ Giải Pháp Chuyên Ngành Doanh Nghiệp (v30.1)

```python
from vietnamese_ai import TroLyPhapLy, TongDaiCSKH, TrichXuatThongMinh

# A. Rà soát Hợp đồng Pháp lý & Cảnh báo Rủi ro
phap_ly = TroLyPhapLy()
ket_qua_pl = phap_ly.ra_soat_hop_dong(van_ban_hop_dong="...")
print("Rủi ro phát hiện:", ket_qua_pl["danh_sach_rui_ro"])

# B. Tổng đài Chăm sóc Khách hàng Tự động
cskh = TongDaiCSKH()
phan_hoi = cskh.xu_ly_khieu_nai("Tài khoản của tôi bị trừ tiền sai, yêu cầu hoàn tiền gấp!")
print("Phản hồi khách hàng:", phan_hoi["tra_loi"])

# C. Trích xuất Thông tin Giấy tờ (CCCD / Hóa đơn)
trich_xuat = TrichXuatThongMinh()
thong_tin = trich_xuat.trich_xuat_cccd("Số định danh cá nhân: 001099012345, Họ và tên: NGUYỄN VĂN AN...")
print("Số CCCD:", thong_tin["so_cccd"])
```

### 5. Phân Loại Học Máy Cổ Điển

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

### Vietnamese LLM & Agents (An toàn — Không dùng `eval()`)

> ⚠️ **Cảnh báo bảo mật**: Tuyệt đối không dùng `eval()` trong `@cong_cu` vì gây lỗ hổng thực thi mã từ xa (RCE). Hãy dùng parser biểu thức AST an toàn (`danh_gia_bieu_thuc_toan` / `cong_cu_may_tinh`) cho phép toán, hoặc dùng hộp cát `ThucThiDocLap` (V-Sandbox) / `MoiTruongCachLy` (Agent Sandbox) khi cần chạy mã Python (xem thêm [SECURITY.md](SECURITY.md)).

```python
from vietnamese_ai import VietnameseLLM
from vietnamese_ai.agents import TacTu, cong_cu, cong_cu_may_tinh, danh_gia_bieu_thuc_toan

llm = VietnameseLLM(bac=3)

@cong_cu(ten="tinh_toan_an_toan", mo_ta="Tính toán biểu thức toán học an toàn qua AST")
def tinh_toan(bieu_thuc: str) -> str:
    return str(danh_gia_bieu_thuc_toan(bieu_thuc))

tac_tu = TacTu(llm=llm, danh_sach_cong_cu=[cong_cu_may_tinh, tinh_toan])
ket_qua = tac_tu.chay("Tính giúp tôi 15 nhân 8")
```

Nếu tác tử cần thực thi mã Python phức tạp, bắt buộc chạy qua Sandbox cách ly tiến trình:

```python
from vietnamese_ai.sandbox import ThucThiDocLap

sandbox = ThucThiDocLap(timeout_giay=5)
thanh_cong, ket_qua = sandbox.chay("ket_qua = sum(i * i for i in range(10))")
```

---

## CLI (Giao diện Dòng lệnh)

Sau khi cài đặt, sử dụng lệnh `vai` trực tiếp trên Terminal:

```bash
# 1. Trải nghiệm 5 phút, chẩn đoán hệ thống & bản đồ minh bạch 3 tầng
vai info
vai doctor
vai status-map
vai demo

# 2. RAG Pháp lý bền vững (.vaidb + ngữ nghĩa ONNX), Đồ thị Luật & Rà soát Hợp đồng
vai rag --db cong_ty.vaidb --docs ./thu_muc_luat --semantic-onnx --admin --query "Thời hạn thông báo vi phạm dữ liệu cá nhân?"
vai legal-graph 45/2013/QH13 --active-only --mermaid
vai contract-review hop_dong_doi_tac.docx --template hop_dong_chuan.docx --output-md redline.md --output-docx redline.docx

# 3. No-Code Web Studio Dashboard (gắn .vaidb, Workspace SQLite đa phòng ban & RBAC)
vai studio --vaidb cong_ty.vaidb --workspace-dir ./studio_ws --port 8080

# 4. Động cơ LLM Tiếng Việt Zero-GPU trên CPU (tự chọn lượng tử theo RAM + gắn .vaidb)
vai pull --list
vai pull qwen2.5-3b --dry-run
vai run vgpt2 --prompt "Tương lai phát triển AI tại Việt Nam giai đoạn 2026-2030" --bench
vai run qwen2.5-3b --rag cong_ty.vaidb --admin --prompt "Hợp đồng này có điều khoản hết hiệu lực không?"

# 5. Kết nối MCP IDE (Cursor / VS Code / Claude Desktop) & Zalo OA / Telegram Bot
vai mcp-server --print-config cursor --vaidb cong_ty.vaidb
vai mcp-server --transport stdio --vaidb cong_ty.vaidb --plugin-dir ./plugins
vai connector both --solution phap-ly --workspace-dir ./studio_ws --port 8095 --dry-run

# 6. Vòng lặp DPO, Remote GPU Flywheel (Modal / SSH / Local) & Kiểm định 4 trục
vai export-dpo --workspace-dir ./studio_ws --output studio_dpo.jsonl
vai train-cloud --from-workspace ./studio_ws --provider modal --gpu T4 --eval-4-axis
vai eval-gate --limit 8 --min-score 0.55
vai release-check

# 7. Suy Luận Bầy Đàn Swarm CPU, Xác Thực zkML & Benchmark Đối Đầu
vai swarm --nodes 3 --layers 6 --prompt "Chuyển đổi số quốc gia" --bench
vai verify --text "Văn bản xác thực bằng Merkle Tree"
vai verify --tamper-check
vai benchmark --compare --latex --output report_academic.tex
vai tokenize "Ủy ban nhân dân thành phố Hà Nội triển khai chuyển đổi số" --unaccent

# 8. Đóng gói 1-Click Desktop, Tác tử Tự chủ & Bộ Giải pháp Doanh nghiệp
vai pack --output ./desktop_bundle --mode studio --threads 4
vai agent research "Quy định bảo vệ dữ liệu cá nhân Nghị định 13" --output baocao.md
vai agent code "Viết hàm kiểm tra số nguyên tố bằng Python kèm docstring tiếng Việt"
vai solution cskh --query "Kiểm tra giúp tôi đơn hàng DH1001"
vai solution phap-ly --query "Khi xảy ra vi phạm dữ liệu cá nhân phải thông báo trong bao lâu?"
vai solution trich-xuat --input "Số: 001095012345 Họ và tên: NGUYỄN VĂN AN" --type cccd

# 9. Học máy Cổ điển & API Server chuẩn OpenAI
vai train --data data.csv --model logistic --output model.pkl --test-size 0.2
vai predict --model model.pkl --input new_data.csv --output results.csv
vai evaluate --model model.pkl --data test.csv
vai serve --model model.pkl --port 8080 --backend fastapi
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
├── rust_core/             # Lõi Rust SIMD/HNSW/BitNet siêu tốc bằng PyO3 (abi3-py39)
├── packages/              # Manifest 5 gói con PyPI phụ thuộc lõi `vietnamese-ai>=31.2.0,<32.0.0`
├── vietnamese_ai/
│   ├── extreme/           # (v27.0) The Hardware Extreme (C++ JIT, Kernel Fusion, CPU Kernel)
│   ├── agents/            # StateGraph, MemGPT, DeepResearch, CodeHealing, MayChuMCPThucTe (stdio/SSE)
│   ├── connectors/        # KetNoiZaloOA, KetNoiTelegram, MayChuWebhookKenh (lưu SQLite Workspace)
│   ├── edge/              # QuanLyMoHinhEdge (8 GGUF VN), DongCoSuyLuanEdge, Swarm CPU, DePIN
│   ├── core/              # Engine, Pipeline, KiemDinhCheo, TimKiemThamSo, QuyTrinhDauCuoi
│   ├── models/            # PhanLoai, HoiQuy, PhanCum, MangNron, MoHinhTapHop
│   ├── preprocessing/     # XuLyVanBan, XuLySo, TaoDacTrung
│   ├── automl/            # AutoML, TimKiemKienTruc (NAS)
│   ├── nlp/               # BoTachTuTiengVietSieuToc, BoNhungVanBanSieuToc, PhanTichCamXuc, PhoBERT
│   ├── embeddings/        # Word2VecTiengViet, FastTextTiengViet
│   ├── fine_tuning/       # SFT, DPO, RLHF, LoRA/PEFT, BoDieuPhoiTinhChinhXa (Modal/SSH), CongKiemDinhMoHinh (4 trục)
│   ├── transformer/       # MultiHeadAttention, TransformerModel, GPTModel, PreTrainer, VietnameseTokenizer
│   ├── vision/            # PhanLoaiHinhAnh, OCR đa tầng
│   ├── timeseries/        # DuDoanChuoiThoiGian
│   ├── mobile/            # TriKhaiDiDong, PowerManager
│   ├── realtime/          # PipelineThoiGianThuc (Message Queue, Feature Store)
│   ├── rag/               # RAGPipeline, KhoTriThucVAIDB (.vaidb), BoNhungONNXTiengViet, legal/ (100 điều luật + hiệu lực + đồ thị + rà soát hợp đồng)
│   ├── solutions/         # TroLyPhapLy, TongDaiCSKH, TrichXuatThongMinh
│   ├── ui/                # StudioDashboard (6 tabs), QuanLyPhienChatWorkspace (SQLite + DPO), V-UI
│   ├── studio/            # StudioKeoTha (No-code Studio)
│   ├── llm/               # VietnameseLLM, ModelConfig, LMEvalHarness, BenchmarkRunner
│   ├── security/          # TuongLuaAI, DataSanitizer, KhienBaoVeHinhAnh, PhongThuDuLieu, CongAnToanDauRa, Verifiable AI
│   ├── export/            # XuatONNX, XuatGGUF
│   ├── registry/          # QuanLyMoHinh (Model Registry)
│   ├── distributed/       # PhanTanHuanLuyen, MultiGPUTrainer
│   ├── enterprise/        # HeThongXacThuc (RBAC), NhatKyHoatDong (Audit)
│   ├── orchestrator/      # V-Orchestrator: Điều phối Cụm (NutChinh / DieuPhoiCum)
│   ├── monitor/           # V-Monitor: Theo dõi GPU và Tính Cost
│   ├── datalake/          # V-DataLake: Quản lý version dữ liệu & 16 bộ dữ liệu mẫu tiếng Việt
│   ├── sandbox/           # V-Sandbox: Hộp cát bảo mật phân tích AST
│   ├── workflow/          # V-Workflow: Động cơ DAG & Thực thi song song
│   ├── plugin/            # V-Plugin 2.0: Hệ thống nạp gỡ nóng Plugin (tự gắn vào FastAPI & MCP)
│   ├── plugins/           # Plugin chuyên ngành (Y tế, Tài chính, plugins/finance/ Chứng khoán & Crypto)
│   ├── interpretability/  # GiaiThichMoHinh
│   ├── experiment_tracking/ # TheoDoiThiNghiem
│   ├── utils/             # Logger, Metrics, LuuTai (HMAC), QuanLyPhuThuoc, QuanLyPhatHanhGoi, bao_cao_ranh_gioi_he_thong
│   ├── extensions/        # Namespace tách biệt cho các phân hệ mở rộng tùy chọn (Lazy)
│   ├── visualization/     # BieuDo
│   ├── api/               # FastAPIServer (OpenAI SSE + Tool Calling + Prometheus), ServerDonGian
│   └── cli/               # CLI (`vai` — 36 lệnh con + `vai demo`)
├── benchmarks/            # Dockerfile.benchmark đa tầng tái lập + bộ dữ liệu đo kiểm pháp lý
├── tests/                 # 1,380+ unit & integration tests (0 failed)
├── examples/              # Ví dụ sử dụng
├── docs/                  # Tài liệu MkDocs Material (build strict 0 cảnh báo)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── setup.py
├── requirements.txt       # Phụ thuộc lõi production tối thiểu (chỉ NumPy)
├── requirements-dev.txt   # Phụ thuộc phát triển, test, lint, docs (tách biệt)
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CITATION.cff
├── LICENSE
└── MANIFEST.in
```

---

## Chạy kiểm thử & Kiểm định chất lượng

```bash
# Chạy toàn bộ bộ kiểm thử
pytest tests/ -q --tb=short

# Kiểm tra lint, kiểu tĩnh mypy, tài liệu & cổng phát hành PyPI
ruff check vietnamese_ai/ tests/ benchmarks/
mypy vietnamese_ai/ --ignore-missing-imports
mkdocs build --strict
vai release-check
```

```text
================= 1265 passed, 32 skipped in 279.55s =================
mypy: Success: no issues found in 327 source files
ruff: All checks passed!
vai release-check: SẴN SÀNG PHÁT HÀNH (316/316 type stubs, 9 tệp SemVer, 0 xung đột)
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
| v10.0 | RAG Pipeline, Serving & Streaming, Prompt Engineering, NLP Extensions (NER/QA/Summarization/Translation/Spelling), Knowledge Distillation, Model Pruning, Production Hardening (Health Check, Circuit Breaker, Structured Logging, Metrics, Warm-up) |
| v11.0 | Self-Adapting Language Models (SALM): Self-Refinement, Self-Consistency, Adaptive LoRA, Self-Generated Data, Test-Time Training |
| v12.0 | Hệ thống Đa Tác Tử (Multi-Agent System): Agentic Swarm, Mixture of Agents (MoA), MCTS Planning |
| v13.0 | Edge AI & DePIN: NodeLlamaEngine, Intelligent EdgeRouter |
| v14.0 | Advanced RAG: GraphRAG (Knowledge Graph), Multi-modal RAG (Image-Text Embeddings) |
| v15.0 | Extreme Efficiency: 1.58-bit LLMs (BitLinear), Speculative Decoding Engine |
| v16.0 | Ultimate Performance: PagedAttention BlockManager, MemGPT Architecture (Long-term Memory), Rust Core Migration |
| v17.0 | Enterprise Security: AI Firewall, Agent Sandbox, Identity-Aware RAG, Data Sanitization (DLP), API Security Middleware |
| v17.1 | State-of-the-Art Security: Automated Red Teaming, Zero-Width Text Watermarking, TEE/ZKP (Cryptographic Proof) cho mạng lưới DePIN Edge AI, Tích hợp Tường lửa sâu vào LLM Core |
| v18.0 | Agentic Evolution: Human-in-the-Loop (HITL), Agent Memory 2.0 (Experience Memory) lưu trữ kinh nghiệm bằng VectorDB |
| v19.0 | Web3 DePIN & Hybrid Execution: Mạng lưới P2P (P2PTracker), Tokenomics (TokenLedger), Speculative Race Execution (Edge vs Cloud) |
| v20.0 | Real-time Agentic RAG: Cập nhật tài liệu thời gian thực (Event Bus), Tác tử RAG đa công cụ (Multi-step Reasoning RAG) |
| v21.0 | EvoNet-Studio Super Core: Zero-Dependency Orchestrator (Điều phối cụm), Monitor (Đo lường GPU), DataLake (DVC), Sandbox (Phân tích AST), DAG Workflow (Định tuyến song song), V-Plugin (Hot-Swap). Các lõi này được thiết kế tích hợp hoàn hảo với hệ sinh thái v12 (Agents) và v7 (Fine-Tuning) giúp framework vận hành mạnh mẽ như một nền tảng SaaS thực thụ. |
| v22.0 | Advanced Agentic Evolution & Self-Healing: Triển khai Knowledge Graph ngầm (GraphMemory) cho Agent, Hỗ trợ Cross-Modal RAG (Text-Image), Đối kháng bảo mật RedTeam/BlueTeam và đặc biệt là hệ thống Self-Healing (Tự vá lỗi Sandbox) và DevOps Agent (An toàn tự viết code qua AST & Dry-Run Pipeline). |
| v24.0 | Decentralized Swarm (Mạng lưới Tác tử Phi tập trung): Triển khai hệ sinh thái Swarm tự trị (MCTS Planning, Tokenomics) kết hợp cùng Edge Nodes trên nền tảng mạng P2P, biến AI thành một cụm siêu trí tuệ phi tập trung toàn cầu. |
| v25.0 | Web-Native & Cross-Platform: Đóng gói Framework thành các SDK đa nền tảng (C++ FFI, Node.js, PHP, WebAssembly, WebGPU), giải phóng Python backend để nhúng trực tiếp AI vào trình duyệt web và mọi ứng dụng thương mại. |
| v26.0 | EvoNet OS - Kỷ nguyên Kéo thả: Hệ điều hành trên nền Web thuần Vanilla JS/CSS. Cung cấp Visual Builder giao diện đồ họa DAG (Directed Acyclic Graph) để người dùng nối dây lắp ráp Logic AI, xuất file JSON tự động biên dịch sang code Python. Tích hợp Live Terminal Console. |
| v27.0 | The Hardware Extreme: Tối ưu lõi Máy tính tột đỉnh. Tích hợp bộ 9 công nghệ Datacenter: MLA (Nén RAM 90%), DeepSeekMoE (Giảm FLOPs 75%), Quantized KV Cache (INT8), FlashAttention Tiling, C++ JIT Compiler (Bypass GIL), Speculative Decoding (Rejection Sampling), Continuous Batching, Kernel Fusion và Ring Attention. |
| v28.0 | Trí tuệ Không gian (Spatial AI) & Robotics: Tích hợp Hệ tọa độ 3D (Spatial RAG), hỗ trợ Hybrid Search (Vector + Tọa độ), mở rộng API điều khiển Robot/LiDAR (Inverse Kinematics). |
| v29.0 | Tác tử Di động (Mobile AI): Triển khai thuật toán Nén Ngữ cảnh (Token Pruning) tiết kiệm RAM, Tích hợp GGUF/ExecuTorch, Bổ sung Mobile OS Tools (GPS, SMS, Camera) và Copilot duyệt web trên điện thoại. |
| v30.0 | Cầu nối Ứng dụng (Mobile SDK Bridge): Kiến trúc WebSockets Thời gian thực kết nối thẳng Frontend (React Native/Flutter) với Backend Python. Hỗ trợ Mã hóa Đầu-Cuối (E2EE), Hàng đợi Ngoại tuyến (Offline Queuing), và Push Notification (Giả lập). |
| v30.1 | Đột phá Tác tử & Triển khai Độc lập: Agentic StateGraph (DoThiTrangThai, HITL, checkpointing), Tác tử Nghiên cứu Sâu (DeepResearchAgent), Tác tử Tự sửa Code (CodeHealingAgent), Động cơ Native C & Zero-GPU Inference (EdgeInferenceEngine, Q4 dequant, mmap streaming), 1-Click Desktop Packager (DesktopPackager), Giao diện Web Studio Dashboard (cổng 8080), và 3 Giải pháp Chuyên ngành Doanh nghiệp (CSKH, Pháp lý, Trích xuất Hồ sơ). |
| v30.2 | Tối Ưu Hóa Thuần Việt & Benchmark Khoa Học Định Lượng: Native Fast Vietnamese Tokenizer (BoTachTuTiengVietSieuToc, chuẩn hóa dấu mới/cũ, Maximum Matching), Vector Hashing Embedder (BoNhungVanBanSieuToc, cosine similarity siêu tốc), RAG Hành chính & Pháp lý Việt Nam (BoToiUuNguCanhHanhChinh, Hierarchy-Preserving Chunking), và Bộ Đo lường Định lượng Tái lập (BoDoLuongDoiDau, xuất bảng Markdown/LaTeX). |
| v31.0 | Suy Luận Bầy Đàn Phân Tán Trên CPU & Xác Thực Chứng Minh Mật Mã Học (Verifiable AI / zkML): Suy Luận Bầy Đàn Swarm CPU Pipeline Parallelism (`NutSwarmCPU`, `DieuPhoiSwarmCPU`) chia sẻ tải tính toán LLM qua mạng LAN trên các máy tính văn phòng/Mini PC, Xác thực Mật mã học Schnorr Fiat-Shamir NIZK modulo 256-bit kết hợp Merkle Tree O(log N) (`CayMerkleSuyLuan`, `XacThucSuyLuanMaoHiem`) kháng 100% nút Byzantine gian lận, CLI `vai swarm` và `vai verify`, phân rã độ trễ tính toán vs độ trễ mạng vật lý. |
| v31.0.2 | Hệ Sinh Thái Khép Kín Tự Tối Ưu, Tự Chữa Lành & Đồng Tiến Hóa Đối Kháng (Closed-Loop Self-Evolving & Self-Healing Ecosystem): Bánh đà dữ liệu suy luận khép kín (DataLake DVC -> DPO Trainer -> Model Registry), Vận hành tự phục hồi Production (Circuit Breaker / Health Check -> Auto-Rollback HMAC), Đóng gói biên 1-click GGUF & Swarm CPU Pipeline Parallelism có chứng thực Merkle/zkML, Đồng tiến hóa đối kháng (Red Team Simulator -> Adversarial DPO Hardening), và Hợp nhất Đồ thị trạng thái DAG phân tán (Distributed StateGraph). Cầu nối hợp nhất `QuyTrinhDauCuoi` (`HeSinhThaiAI`). |
| v31.0.3 | **Open-Source Production Release (4 Hướng Chiến Lược + 5 Ưu Tiên + Khép Kín 9 Mối Nối)**: Kho tri thức đơn tệp `.vaidb` (SQLite WAL + FTS5 + Vector + đồng bộ SHA-256), Ngữ nghĩa đồng nghĩa ONNX (`onnx_viet`), 100 điều khoản pháp lý + 35 QA đóng gói sẵn trong wheel, Theo dõi hiệu lực & Đồ thị tham chiếu luật (`vai legal-graph`), Rà soát hợp đồng `.docx`/Redline (`vai contract-review`), Chat Workspace đa phòng ban & Studio 6 tabs (`vai studio --workspace-dir --auth`), MCP Server `stdio`/`SSE` tự nạp V-Plugin 2.0 (`vai mcp-server`), Zalo OA & Telegram Bot lưu SQLite Workspace (`vai connector`), LLM Zero-GPU CPU (`vai pull`/`vai run --rag .vaidb`), Remote GPU Flywheel (`vai train-cloud --from-workspace --eval-4-axis`), Cổng Kiểm Định 4 trục (`vai eval-gate`), Bản đồ minh bạch 3 tầng (`vai status-map`), và cổng phát hành PyPI OIDC (`vai release-check`). 316 public APIs, 1,265 tests passed, 0 mypy/ruff errors. |
| v31.0.4 | **Deep Architecture, Vertical Moat, 5-Surface Integration & CI Green**: Chuẩn hóa dấu thanh FST & Viterbi DAG khớp 100% Rust/Python (`584k`/`269k` từ/s), Kernel Rust Zero-Copy `PyBytes` (`Q5_K`/`Q6_K`/`RMSNorm`/`SwiGLU`, nhanh hơn NumPy `3.7x–6.3x`), `KiemChungFreivalds` phát hiện `100%` Byzantine, Persistent HNSW `.vaidb` batch-neighbor, `BoNapVanBanPhapQuyTuDong` (`vai legal-ingest`) đóng gói văn bản hợp nhất → `.vaidb` HNSW + 14 mốc Point-in-Time + Đồ thị, Cross-Encoder ONNX (`bge-reranker-v2-m3`/`phoranker`), Word Track Changes OOXML + Comments, SimPO, Eval Gate Paired Bootstrap 95% CI, Red-Team Firewall/PII/Sandbox, đồng bộ 5 bề mặt & CI Python 3.9–3.13 + Rust Parity xanh 100%. 316 public APIs, 1,314 tests passed, 0 mypy/ruff errors. |

### Tính liên kết của v31.0.2 với các phiên bản trước
- **Khép kín Bánh đà Dữ liệu & DPO (v31.0.2)**: Kết nối trực tiếp dữ liệu suy luận từ **FastAPI / Serving (v10)** và cơ chế tự đánh giá **Self-Refine (v11)** vào **HoDuLieu DVC (v21)** để nuôi dưỡng và tái huấn luyện mô hình qua **DPOTrainer (v7)**, loại bỏ hoàn toàn sự phân mảnh dữ liệu.
- **Tự Phục hồi Production (v31.0.2)**: Nâng cấp **MachCat & KiemTraSucKhoe (v10)** từ trạng thái thụ động chỉ ngắt mạch sang tự động rollback mô hình trong **QuanLyMoHinh (v1.2)** về phiên bản an toàn có chữ ký số HMAC, đảm bảo tính sẵn sàng cao 99.99%.
- **Đồng tiến hóa Đối kháng (v31.0.2)**: Kết nối **RedTeamSimulator (v17.1)** với **TuongLuaAI (v17)** và **DPOTrainer (v7)** thành một vòng lặp Co-Evolution: hệ thống tự động tập kích, trích xuất lỗ hổng và tôi luyện trọng số mô hình chống lại Jailbreak ngay trong nội bộ.
- **Đồ thị Trạng thái Phân tán (v31.0.2)**: Mở rộng **DoThiTrangThai StateGraph (v30.1)** để nhúng trực tiếp cụm suy luận **DieuPhoiSwarmCPU (v31.0)** và cụm điều phối **NutChinh (v21)** thành các nút tác tử thực thi, tạo nên một State Machine hoàn chỉnh từ khâu tiền xử lý, RAG, suy luận Swarm đến làm sạch DLP đầu ra.
- **Cầu nối Duy nhất `QuyTrinhDauCuoi` (v31.0.2)**: Hợp nhất toàn bộ 5 trụ cột trên thành một điểm vào (Single Entrypoint) tinh gọn cho toàn bộ framework.

### Tính liên kết của v31.0 với các phiên bản trước
- **Swarm CPU Pipeline Parallelism (v31.0)** kế thừa trực tiếp cơ chế lượng tử hóa và suy luận không GPU từ **EdgeInferenceEngine (v30.1)**, kết hợp cùng **Native Fast Tokenizer (v30.2)** và **Fast Embedder (v30.2)** để nén và truyền tensor trạng thái ẩn siêu nhẹ qua mạng LAN giữa các máy tính cấu hình khiêm tốn.
- **Chứng thực Mật mã học Verifiable AI & Merkle Tree (v31.0)** cung cấp lớp bảo vệ toàn vẹn tuyệt đối cho các **Tác tử Tự chủ StateGraph (v30.1)** và **Hệ thống DePIN / Federated Learning (v13 & v30.0)**, triệt tiêu hoàn toàn rủi ro nút giả mạo (Byzantine attacks) bằng toán học hình thức NIZK Schnorr.
- **Phân rã Độ trễ Khoa học (v31.0)** tích hợp trực tiếp vào **Báo cáo Đối đầu Định lượng (v30.2)**, tạo nên một chu trình nghiên cứu - đo kiểm - triển khai hoàn chỉnh chuẩn mực cho các bài báo khoa học đỉnh cao (IEEE, Elsevier, JOSS).

### Tính liên kết của v30.2 với các phiên bản trước
- **Native Fast Tokenizer (v30.2)** cung cấp lớp tách từ siêu tốc cho **RAG Pipeline (v10)**, **StateGraph Agents (v30.1)** và **PhoBERT/BERT wrappers**, loại bỏ hoàn toàn hiện tượng phân mảnh câu khi xử lý từ ghép tiếng Việt.
- **RAG Ngữ cảnh Hành chính & Pháp lý (v30.2)** nâng cấp trực tiếp năng lực của **Trợ lý Pháp lý (v30.1)** và **Tác tử Nghiên cứu Sâu (v30.1)** với khả năng trích dẫn số hiệu điều khoản chuẩn mực (Nghị định, Luật, Thông tư).
- **Bộ Benchmark Định lượng (v30.2)** đo lường khách quan và xác nhận sự vượt trội của toàn bộ hệ sinh thái **Zero-Dependency** trước các giải pháp quốc tế cồng kềnh.

### Tính liên kết của v30.1 với các phiên bản trước
- **StateGraph & Deep Research (v30.1)** tận dụng trực tiếp **RAG Pipeline (v10)**, **Guardrails & DLP (v17)** và **MemGPT Long-Term Memory (v16)** để tạo nên chu trình nghiên cứu sâu đa bước có phản biện chéo và bảo mật tuyệt đối.
- **Động cơ Native C & Zero-GPU Edge (v30.1)** kế thừa định dạng **GGUF (v15)** và giải lượng tử hóa Q4/Q8 thuần vector hóa, cho phép các tác tử chạy trực tiếp trên laptop/PC phổ thông mà không cần hạ tầng máy chủ đắt đỏ.
- **Web Studio Dashboard (v30.1)** hợp nhất **3 Giải pháp Chuyên ngành (CSKH, Pháp lý, Trích xuất)** cùng chẩn đoán phần cứng và kiểm thử an toàn vào một cổng điều khiển trực quan duy nhất.

### Tính liên kết của v28.0 - v30.0 với các phiên bản cũ
Rất nhiều nhà phát triển lo ngại kiến trúc mở rộng liên tục sẽ bị phân mảnh. Sự thật là chúng được thiết kế theo dạng vòng lặp (Organic Loop):
- **Cầu nối Ứng dụng (v30)** đọc chỉ số Pin thực tế từ App React Native, truyền ngược về cho thuật toán **PowerManager (v27)** để điều tiết tài nguyên cực kỳ chính xác.
- Nhờ **PowerManager (v27)** ra quyết định chuẩn xác, **EdgeRouter (v27)** biết lúc nào nên san tải lên đám mây DePIN thay vì vắt kiệt NPU điện thoại của người dùng.
- Trình duyệt **Mobile Copilot (v29)** dùng **RAG Pipeline (v10)** kết hợp với Token Pruning để lọc nội dung rác trước khi nhồi vào Local LLM.
- Tất cả tạo thành một khối nguyên khối (Monolith) mạnh mẽ, không có chức năng nào giẫm chân lên nhau.

### Tính liên kết của v21.0 - v27.0 với các phiên bản cũ
Rất nhiều nhà phát triển lo ngại kiến trúc mở rộng liên tục sẽ bị phân mảnh. Sự thật là chúng được thiết kế theo dạng vòng lặp (Organic Loop):
- **EvoNet OS (v26)** là lớp áo khoác trực quan cho tất cả các **Tác tử (v12)** và **RAG Pipeline (v10)**.
- **The Hardware Extreme (v27)** đập tan rào cản tốc độ cho các mô hình Toán học gốc (v2) và tự động tương thích với mạng phân tán **DePIN (v13)** thông qua Ring Attention.
- **C++ JIT Engine** biên dịch tự động các mã vòng lặp chậm chạp bằng cách gọi `g++` on-the-fly, giải phóng hoàn toàn sức mạnh đa nhân CPU mà không làm suy yếu nguyên tắc Zero-Dependency.

### Tính liên kết của v21.0 - v23.0 với các phiên bản cũ
Rất nhiều nhà phát triển lo ngại các lõi mới của v21.0 sẽ bị phân mảnh. Sự thật là chúng được thiết kế để bao bọc và cường hóa hệ thống hiện có:
- **V-Workflow** chịu trách nhiệm kết nối và định tuyến luồng thực thi của các **Tác tử (v12)** và **RAG Pipeline (v10)**.
- **V-Sandbox** được tích hợp thẳng vào **Agent Sandbox (v17)** để kiểm duyệt mã Python do Agent sinh ra.
- **V-DataLake** cung cấp luồng dữ liệu khổng lồ (OOM-free) cho các module **PyTorch Trainer (v15)** và **SFT/RLHF (v7)**.
- **V-Plugin** cho phép nạp động các **Tools (v12)** và **Models (v1)** vào runtime mà không cần khởi động lại.

### Tech Stack tương lai

- **Fine-tuning**: Unsloth (2x nhanh), LoRA, QLoRA, PEFT
- **Quantization**: GGUF (llama.cpp), GPTQ, AWQ
- **Training**: DeepSpeed ZeRO, FSDP, Gradient Checkpointing
- **Inference**: vLLM, TGI, llama.cpp
- **Models**: Llama, Mistral, Qwen, Gemma, PhoGPT

---

## Trích dẫn khoa học (Citation)

Nếu bạn sử dụng **Vietnamese AI Framework** trong nghiên cứu khoa học hoặc sản phẩm, vui lòng trích dẫn theo `CITATION.cff` / Zenodo DOI:

```bibtex
@software{huynh2026vietnameseai,
  author       = {Huynh, Duong Phong},
  title        = {Vietnamese AI: A Lightweight, Zero-Dependency, and Verifiable Framework for Vietnamese-Centric AI and Distributed Swarm CPU Inference},
  year         = {2026},
  version      = {31.0.4},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20357082},
  url          = {https://github.com/phonghhd/vietnamese-ai}
}
```

---

## Đóng góp

Xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết quy trình đóng góp, chuẩn code tiếng Việt và checklist kiểm định trước khi gửi Pull Request.

Xem [SECURITY.md](SECURITY.md) cho chính sách bảo mật và phạm vi tường lửa / hộp cát.

Xem [ROADMAP.md](ROADMAP.md) cho định hướng phát triển mở rộng cộng đồng.

---

## Giấy phép

MIT License - xem [LICENSE](LICENSE)

---

<p align="center">
  <strong>Vietnamese AI Framework</strong> - Được xây dựng với ❤️ cho cộng đồng AI Việt Nam
</p>
