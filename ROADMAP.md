# Lộ Trình Phát Triển (Roadmap) — Vietnamese AI Framework (`v31.0.3` → `v33.x`)

Lộ trình mã nguồn mở được định hướng theo nguyên tắc **thực dụng, đo kiểm được và minh bạch ranh giới Core Production vs Experimental** (kiểm tra bất kỳ lúc nào qua `vai status-map`).

---

## ✅ Đã Hoàn Thành & Sẵn Sàng Open Source (`v31.0.3`)

- **Micro-Core Siêu Nhẹ & Minh Bạch**: 316 biểu tượng công khai khớp 100% PEP 561 `.pyi`, 0 thư viện nặng bị nạp sớm, 1,265 tests passed / 0 failed, 0 lỗi `mypy` (327 tệp) & 0 lỗi `ruff`, `vai status-map`, `vai release-check`, GitHub Actions OIDC TestPyPI/PyPI.
- **RAG Pháp Lý & Văn Bản Hành Chính Việt Nam**: Kho tri thức bền vững 1 tệp `.vaidb` (SQLite WAL + FTS5 BM25 + Float32 Vector + đồng bộ tăng dần SHA-256), Ngữ nghĩa đồng nghĩa ONNX CPU (`onnx_viet`), 100 điều khoản pháp luật VN + 35 QA đóng gói sẵn trong wheel, Theo dõi hiệu lực theo mốc ngày (`BoTheoDoiHieuLucPhapLy`) và Đồ thị Tham chiếu Pháp lý (`DoThiThamChieuPhapLy`).
- **Bàn Làm Việc Văn Phòng & Hệ Sinh Thái Kết Nối**: Rà soát & so sánh hợp đồng `.docx`/Markdown (`BoSoSanhRaSoatHopDong`), Web Studio 6 tab + Chat đa phiên cách ly `.vaidb` theo phòng ban + RBAC/Audit (`QuanLyPhienChatWorkspace`), MCP Server `stdio`/`HTTP-SSE` tự phơi bày V-Plugin 2.0 (`MayChuMCPThucTe`), và Zalo OA / Telegram Bot lưu SQLite Workspace (`KetNoiZaloOA`, `KetNoiTelegram`).
- **Zero-GPU Edge CPU & Remote GPU Flywheel**: Kéo & chạy 8 mô hình LLM GGUF tiếng Việt trên CPU (`vai pull` / `vai run --rag .vaidb --admin`), tinh chỉnh GPU từ xa đa nhà cung cấp Modal / SSH / Local (`vai train-cloud --from-workspace --eval-4-axis`) và Cổng Kiểm Định hồi quy 4 trục (`CongKiemDinhMoHinh`, `vai eval-gate`).

---

## 🎯 Giai Đoạn Tiếp Theo: Mở Rộng Cộng Đồng (`v32.0`)

- **Phát hành Chính thức lên PyPI & Prebuilt Rust Wheels (`abi3-py39`)**: Đẩy bản `31.0.3` lên TestPyPI → PyPI kèm bánh xe biên dịch sẵn cho Linux `x86_64`, macOS `arm64` và Windows `x64`.
- **Kho Pháp Lý Mở Cộng Đồng 500+ Điều Khoản & Cập Nhật Tự Động**: Mở rộng bộ dữ liệu pháp lý Việt Nam (Thuế, Hải quan, Bảo hiểm, Đầu tư, Xây dựng) có kiểm duyệt phiên bản và nguồn trích dẫn công khai.
- **Embedding & Cross-Encoder ONNX INT8 Thuần Việt Chính Thức**: Cung cấp gói trọng số ONNX INT8 tải qua `vai pull` (ví dụ: Vietnamese-Bi-Encoder / BGE-M3 quantized) để nâng cấp trực tiếp tầng ngữ nghĩa trên CPU văn phòng.
- **Connector Doanh Nghiệp Mở Rộng**: Bổ sung connector Microsoft Teams, Slack, Lark/Feishu và webhook ERP/CRM chuẩn hóa dùng chung `KetNoiKenhCoSo` và `QuanLyPhienChatWorkspace`.

---

## 🚀 Giai Đoạn Dài Hạn (`v32.5` – `v33.x`)

- **Tác Tử Lập Trình & Tự Kiểm Thử Hộp Cát Nâng Cao**: Kết hợp `TacTuLapTrinhTuSua` với container isolation chuẩn (Docker / rootless sandbox) để xử lý tự động các GitHub Issue & PR nhỏ kèm kiểm định hồi quy.
- **Đa Phương Thức Tài Liệu Thực Tế (Bảng Biểu & Scan Phức Tạp)**: Tối ưu hóa nhận diện bảng biểu, dấu mộc và hóa đơn/hợp đồng scan nhiều cột cho tiếng Việt.
- **Chuẩn Hóa Đánh Giá Mở Cộng Đồng (Vietnamese Legal & Enterprise Benchmark)**: Công bố bộ điểm chuẩn công khai và bảng xếp hạng mô hình mở tiếng Việt chạy trên `benchmarks/Dockerfile.benchmark`.

---

> Đóng góp ý tưởng hoặc đề xuất ưu tiên tại [GitHub Issues](https://github.com/phonghhd/vietnamese-ai/issues) và xem quy trình tại [CONTRIBUTING.md](CONTRIBUTING.md).
