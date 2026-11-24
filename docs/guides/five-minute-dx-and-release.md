# Trải Nghiệm 5 Phút (`vai demo`), Kiểm Định Phát Hành PyPI & Benchmark Docker

Hướng dẫn này tổng hợp luồng **Developer Experience (DX)** giúp người dùng mới chạy thử trọn bộ sức mạnh framework trong 5 phút và giúp nhóm phát triển kiểm định, đóng gói, tái lập benchmark chuẩn khoa học trước khi phát hành lên TestPyPI / PyPI.

---

## 1. Hành Trình 5 Phút Cho Người Mới (`vai demo`)

Chỉ cần cài lõi tối thiểu (`numpy`), không cần GPU và không cần tải hàng GB trọng số:

```bash
# Phút 1: Cài đặt lõi siêu nhẹ
pip install vietnamese-ai

# Phút 2: Kiểm tra phần cứng, tập lệnh SIMD, Rust Core & gói mở rộng
vai doctor

# Phút 3: Chạy trình diễn 1-Click đủ 4 trụ cột offline
vai demo
```

Lệnh `vai demo` chạy tuần tự 4 kịch bản thực chiến trong vài giây:

| Lệnh | Kịch bản trình diễn | Những gì được thực thi thực tế |
| :--- | :--- | :--- |
| `vai demo legal-rag` | **RAG Pháp lý, Đồ thị & Hợp đồng** | Nạp kho mở 100 điều khoản vào `.vaidb` + ngữ nghĩa `onnx_viet`, hỏi đáp Nghị định `13/2023/NĐ-CP`, kiểm toán hiệu lực (`45/2013/QH13` → `31/2024/QH15`), truy vấn Đồ thị Tham chiếu Luật & rà soát rủi ro hợp đồng mẫu. |
| `vai demo edge-chat` | **Zero-GPU Edge CPU** | Đo RAM thực máy hiện tại, đề xuất mức lượng tử GGUF tối ưu cho `qwen2.5-3b`, chạy suy luận CPU nội bộ kèm số đo TTFT & tok/s. |
| `vai demo security` | **Bảo mật & Mật mã zkML** | Tường lửa AI chặn Prompt Injection, `DataSanitizer` ẩn SĐT/CCCD/Email, và `KiemToanMatMa` chạy 4/4 bài tự kiểm toán RFC 5114 + RFC 6962. |
| `vai demo flywheel` | **Remote GPU Flywheel** | Lập kế hoạch dry-run Modal Serverless GPU, tự lọc PII trước khi đẩy và ước tính ngân sách USD/VNĐ. |
| `vai demo all` | **Trọn bộ 4 kịch bản** | Chạy cả 4 kịch bản trên trong một lệnh duy nhất (mặc định khi gọi `vai demo`). |

```bash
# Phút 4: Thiết lập thông minh 3 tầng theo hồ sơ nhu cầu (xem trước dry-run trước khi --apply)
vai setup --profile legal-rag

# Phút 5: Mở giao diện No-Code Web Studio
vai studio --port 8080
```

---

## 2. Bộ Kiểm Định Trước Khi Phát Hành PyPI (`vai release-check`)

Trước mỗi đợt đẩy bản phát hành lên TestPyPI hoặc PyPI, chạy một lệnh duy nhất để rà soát toàn bộ 5 tiêu chí bắt buộc:

```bash
# Hiển thị checklist tiếng Việt trực quan
vai release-check

# Xuất JSON thuần để chặn sớm trong CI/CD (GitHub Actions / GitLab CI)
vai release-check --json
```

### 5 Tiêu chí Kiểm định Tự động (`QuanLyPhatHanhGoi.kiem_tra_san_sang_phat_hanh`)

1. **SemVer 2.0.0 & Đồng bộ Phiên bản**: Kiểm tra đồng bộ 100% giữa `vietnamese_ai/__init__.py`, `pyproject.toml`, `setup.py`, toàn bộ manifest trong `packages/*/pyproject.toml` và `rust_core/Cargo.toml`.
2. **Không Xung đột Phụ thuộc**: Kiểm tra chéo toàn bộ nhóm `optional-dependencies` trong `QuanLyPhuThuoc` bảo đảm không có đặc tả mâu thuẫn.
3. **Phủ 100% PEP 561 Type Stubs (`.pyi`)**: Mọi biểu tượng công khai trong `vietnamese_ai.__all__` đều phải được khai báo tường minh trong `vietnamese_ai/__init__.pyi` để IDE autocomplete và `mypy` hoạt động chính xác.
4. **Tính tương thích Rust Core (`abi3-py39`)**: Xác nhận `rust_core/Cargo.toml` biên dịch ở chế độ Stable ABI `abi3-py39`, cho phép một file wheel Rust dùng chung trên nhiều phiên bản Python 3.9+.
5. **Tệp Phát hành & Benchmark Chuẩn**: Xác nhận đủ `README.md`, `SECURITY.md`, `pyproject.toml`, `setup.py`, `benchmarks/Dockerfile.benchmark`, và `benchmarks/datasets/legal_vn_corpus.jsonl`.

### Quy trình Đóng gói & Xuất bản TestPyPI / PyPI

```bash
# 1. Kiểm định tổng thể
vai release-check
ruff check vietnamese_ai/ tests/ benchmarks/
mypy vietnamese_ai/ --ignore-missing-imports
pytest tests/ -q --tb=short
mkdocs build --strict

# 2. Đóng gói sdist + wheel và kiểm tra metadata Twine
python -m build
twine check dist/*

# 3. Thử nghiệm trên TestPyPI trước khi phát hành chính thức
twine upload --repository testpypi dist/*
```

### Gọi từ Python (tích hợp script phát hành nội bộ)

```python
from vietnamese_ai import QuanLyPhatHanhGoi

bao_cao = QuanLyPhatHanhGoi.kiem_tra_san_sang_phat_hanh()
assert bao_cao["san_sang_phat_hanh"] is True
print("Phiên bản chuẩn:", bao_cao["phien_ban"])
print("Biểu tượng công khai:", bao_cao["so_bieu_tuong_cong_khai"])
```

---

## 3. Benchmark Khoa Học Tái Lập Với Docker Đa Tầng

Tệp `benchmarks/Dockerfile.benchmark` sử dụng kiến trúc **Multi-Stage Docker Build**:
- **Stage 1 (`rust-builder`)**: Tự động cài `maturin` và biên dịch tùy chọn mô-đun tăng tốc Rust `abi3-py39` sang wheel độc lập.
- **Stage 2 (`python:3.11-slim`)**: Môi trường chạy siêu gọn, cố định biến môi trường `PYTHONHASHSEED=42` và `OMP_NUM_THREADS=1` để mọi kết quả đo kiểm có thể tái lập 100% giữa các máy.

```bash
# Xây dựng image benchmark tái lập
docker build -f benchmarks/Dockerfile.benchmark -t vietnamese-ai-benchmark:31.0.3 .

# Chạy bộ đo kiểm thực nghiệm (xuất báo cáo Markdown & JSON ra thư mục hiện tại)
docker run --rm -v "$(pwd)/ket_qua_bench:/app/out" vietnamese-ai-benchmark:31.0.3 \
  python benchmarks/run_benchmarks.py --output /app/out/benchmark_report.json
```

---

## 4. Minh Bạch Ranh Giới Core Production vs Experimental & Phát Hành Cộng Đồng

Để người dùng doanh nghiệp và hội đồng bình duyệt (JOSS / SoftwareX) nhìn rõ ngay phân hệ nào là **lõi sản xuất 100% thực thi thật** và phân hệ nào còn ở mức thử nghiệm/mô phỏng, chạy:

```bash
vai status-map
vai status-map --json
```

Luồng GitHub Actions `.github/workflows/publish.yml` hỗ trợ phát hành 1-Click lên **TestPyPI** (`workflow_dispatch` → chọn `testpypi`) và **PyPI** chính thức (tag `v*.*.*`), tự động chạy:
1. `release-gate`: `vai release-check`, `python -m build`, `twine check`, và **smoke test trên venv sạch** xác nhận wheel cài xong chứa đủ 100 điều khoản pháp lý đóng gói sẵn (`vietnamese_ai/rag/legal/data/`).
2. `build-rust-wheels`: Biên dịch song song bánh xe Rust `abi3-py39` cho Linux (`x86_64`), macOS (`arm64`) và Windows (`x64`).
3. `build-and-publish`: Đẩy lên TestPyPI / PyPI qua chuẩn bảo mật OIDC Trusted Publishing.

---

## 5. Vệ sinh Kho Mã Nguồn (Repository Hygiene)

`.gitignore` đã được chuẩn hóa để loại bỏ toàn bộ sản phẩm trung gian nặng khỏi Git:
- `rust_core/target/`: Thư mục biên dịch Cargo/Rust.
- `vai_workspace/`, `vai_remote_jobs/`, `outputs/`: Thư mục làm việc tạm của Flywheel & Fine-tuning.
- `*.sig`: Tệp chữ ký HMAC sinh cục bộ cho mô hình tạm.
- `site/`: Thư mục xuất tĩnh của MkDocs.
