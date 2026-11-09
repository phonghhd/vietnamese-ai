# Changelog

Xem [CHANGELOG.md](https://github.com/phonghhd/vietnamese-ai/blob/main/CHANGELOG.md) đầy đủ trên GitHub.

## v31.0.0 (2026-11-09)

- **Swarm CPU & Pipeline Parallelism**: Triển khai phân mảnh mô hình và song song hoá pipeline trên mạng nội bộ LAN (`DieuPhoiSwarmCPU`, `NutSwarmCPU`), chạy LLM trên CPU không cần GPU.
- **Verifiable AI & Proof of Inference**: Xác thực tính toàn vẹn suy luận bằng mật mã học phi tương tác Schnorr Fiat-Shamir NIZK trên trường nguyên tố 256-bit và cây Merkle (`XacThucSuyLuanMaoHiem`, `CayMerkleSuyLuan`).
- **Khiên Bảo Vệ Đa Lớp (Data Defense Shields)**: Phòng thủ Polyglot malware, Image Bombs, tin ẩn LSB Steganography, bom hẹn giờ Sleeper Backdoors và khử độc dữ liệu CDR (`KhienBaoVeHinhAnh`, `PhongThuDuLieu`, `CongChatLuongDauVao`, `CongAnToanDauRa`).
- **RAG Pháp Lý Hành Chính Chuyên Sâu**: Bộ tối ưu ngữ cảnh `BoToiUuNguCanhHanhChinh` bảo toàn cấu trúc Chương/Điều/Khoản/Điểm, băm vector xác định Blake2b `BoNhungVanBanSieuToc`, đạt 93.8% độ chính xác.
- **Agentic StateGraph & MCP**: Tác tử đồ thị trạng thái có hướng hỗ trợ rẽ nhánh có điều kiện, tự lành mã nguồn (`CodeHealingAgent`), nghiên cứu sâu (`DeepResearchAgent`) và kết nối Model Context Protocol.
- **Hồ Dữ Liệu (DataLake)**: Tích hợp sẵn 16 bộ tập dữ liệu mẫu tiếng Việt chuẩn hoá (`HoDuLieu`).
- **Tương thích toàn diện & 100% CI Green**: Hỗ trợ đầy đủ từ Python 3.8 đến Python 3.13, vượt qua > 1,120 bài kiểm thử tự động.
- **Nâng cấp Giao diện Tài liệu**: Redesign toàn diện giao diện MkDocs Material với tông màu Deep Royal Navy/Cyan Glow, Glassmorphism Hero Banner và Feature Cards.

## v30.0.0 (2026-10-15)

- Đồng bộ phiên bản 30.0.0, sửa export `DieuPhoiCum`, extras packaging
- Chuẩn hóa public API các module, bổ sung test coverage gaps
- Ghi rõ tính năng mô phỏng (Edge/TEE/WASM/multimodal placeholder)

## v11.0.0 (2026-05-06)

- Self-Adapting Language Models (SALM): SelfRefine, SelfConsistency, AdaptiveLoRA, SinhDuLieuTuDong, TestTimeTraining
- 33 new tests (505 total)
- Version bump to 11.0.0

## v10.0.0 (2026-05-04)

- RAG Pipeline (CSDLVector, CatVanBan, TrichXuat, SapXepLai, RAGPipeline)
- Serving & Streaming (MayChuBatch, MayChuStream, BoGioiHanTocDo)
- Prompt Engineering (MauPrompt, ChuoiPrompt, LuongAnToan, PhanTichDauRa)
- NLP Extensions (NhanDienThucThe, HoiDapTiengViet, TomTatVanBan, DichThuat, KiemTraChinhTa)
- Model Compression (HocRutGon, CatTiaMoHinh)
- Production Hardening (KiemTraSucKhoe, MachCat, LoggerCauTruc, QuanLyMetrics, LamNongModel)
- Security fixes (RestrictedUnpickler, pbkdf2_hmac, hmac.compare_digest, random API keys)
- 472 test cases

## v9.0.0 (2026-05-04)

- Vietnamese LLM Configs (125M-7B presets)
- LM Evaluation Harness (multi-task, few-shot)
- Benchmark Runner (perplexity, generation, sentiment, speed)
- 352 test cases

## v8.0.0 (2026-05-04)

- GPT Decoder-Only Pre-training (NumPy)
- PreTrainer (causal LM, warmup + cosine decay)
- TextDataset (sliding window, character vocab)

## v7.0.0 (2026-05-04)

- SFT Trainer (Supervised Fine-Tuning)
- DPO Trainer (Direct Preference Optimization)
- Reward Model (Bradley-Terry training)
- RLHF Pipeline (SFT → RM → DPO end-to-end)

## v6.0.0 (2026-05-04)

- PEFTConfig (LoRA, QLoRA, Prefix Tuning)
- LoRAPeft (enhanced PEFT cho PyTorch)
- Instruction Tuning Trainer (Alpaca/ShareGPT)

## v5.0.0 (2026-05-04)

- PyTorch Trainer (GPU, Mixed Precision, Early Stopping)
- Unsloth Integration (2x faster fine-tune)
- GGUF Export/Import (Q4_K, Q5_K, Q8_0)
- HuggingFace Integration
- Transformer Architecture
- Vietnamese Tokenizer (BPE, WordPiece)

## v4.0.0 (2026-05-04)

- Cloud Platform SaaS (`NenTangDichVu`)
- No-code Studio (`StudioKeoTha`)
- Vietnamese LLM (`VietnameseLLM`)
- 230 test cases

## v3.0.0 (2026-05-04)

- Mobile/Edge Deployment (`TriKhaiDiDong`)
- Neural Architecture Search (`TimKiemKienTruc`)
- Federated Learning (`HocLienKet`)
- Real-time ML Pipeline (`PipelineThoiGianThuc`)
- 159 test cases

## v2.0.0 (2026-05-03)

- Multi-GPU, Distributed Training
- Model Hub, Plugin System
- Cloud Deployment, Marketplace
- Enterprise (RBAC, Audit Log)
- 90 test cases

## v1.2.0 (2026-05-03)

- No-code Web Interface
- Model Registry
- Streaming Data
- ONNX Export
- 73 test cases

## v1.1.0 (2026-05-03)

- Deep Learning (PyTorch)
- Image Classification (CNN)
- Time Series Forecasting
- PhoBERT wrapper
- 66 test cases

## v1.0.0 (2026-05-03)

- Production Release
- 57 test cases
- CI/CD, PyPI, MkDocs
