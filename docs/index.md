<div class="hero-banner">
  <h1>Vietnamese AI Framework</h1>
  <p>Hệ sinh thái Trí tuệ Nhân tạo thuần tiếng Việt cho Python. Thiết kế theo triết lý Zero-Dependency Micro-Core, tích hợp Kho tri thức `.vaidb`, RAG Pháp lý 100 điều khoản, Rà soát hợp đồng Redline, MCP IDE Server, Zalo/Telegram Connectors, Zero-GPU Edge LLM và Remote GPU Flywheel.</p>
  <div class="hero-badges">
    <span class="badge-pill">🚀 Phiên bản 31.2.0</span>
    <span class="badge-pill">🐍 Python 3.8 – 3.14</span>
    <span class="badge-pill">✅ 1,380+ Tests Passed</span>
    <span class="badge-pill">📦 349 Public APIs (.pyi)</span>
    <span class="badge-pill">🛡️ 0 Mypy / 0 Ruff Errors</span>
    <span class="badge-pill">📄 MIT License</span>
  </div>
</div>

---

## 🌟 Tính Năng Chiến Lược Cốt Lõi

<div class="feature-grid">
  <div class="feature-card">
    <div class="card-icon">⚡</div>
    <h3>Zero-Dependency Micro-Core</h3>
    <p>Thời gian khởi động lạnh chỉ 24.68ms, chiếm dụng 12.82MB VmRSS trên CPU Celeron N3050. Tải lười (lazy loading) thông minh, chỉ phụ thuộc lõi NumPy.</p>
  </div>

  <div class="feature-card">
    <div class="card-icon">🌐</div>
    <h3>Swarm CPU & Pipeline Parallelism</h3>
    <p>Phân mảnh mô hình ngôn ngữ lớn (LLM) qua mạng LAN cục bộ. Tận dụng mạng lưới CPU văn phòng và VPS giá rẻ để suy luận mà không cần GPU đắt đỏ.</p>
  </div>

  <div class="feature-card">
    <div class="card-icon">🛡️</div>
    <h3>Verifiable AI & Proof of Inference</h3>
    <p>Bảo đảm tính toàn vẹn của mô hình AI thông qua cây Merkle và mật mã học Fiat-Shamir NIZK phi tương tác trên trường số nguyên tố 256-bit.</p>
  </div>

  <div class="feature-card">
    <div class="card-icon">📜</div>
    <h3>RAG Pháp Lý & Văn Bản Hành Chính</h3>
    <p>Bảo toàn cấu trúc phân cấp Chương / Điều / Khoản / Điểm chuẩn thể thức Việt Nam, hỗ trợ truy hồi theo mốc thời gian và kiểm chứng trích dẫn 3 tầng.</p>
  </div>

  <div class="feature-card">
    <div class="card-icon">🔒</div>
    <h3>Khiên Bảo Vệ Đa Lớp (Data Defense)</h3>
    <p>Ngăn chặn Polyglot malware, Image Bombs, tin ẩn LSB Steganography và phát hiện bom hẹn giờ Sleeper Backdoors trong tập dữ liệu.</p>
  </div>

  <div class="feature-card">
    <div class="card-icon">🤖</div>
    <h3>Agentic StateGraph & Chuẩn MCP</h3>
    <p>Tác tử đồ thị trạng thái có hướng với cơ chế tự sửa sai (CodeHealingAgent) và nghiên cứu sâu (DeepResearchAgent), tương thích giao thức MCP.</p>
  </div>

  <div class="feature-card">
    <div class="card-icon">📊</div>
    <h3>Hồ Dữ Liệu 16 Bộ Tập Chuẩn Hoá</h3>
    <p>Tích hợp sẵn 16 bộ dữ liệu mẫu chất lượng cao: pháp lý, y tế, tài chính, phân loại cảm xúc, nhận diện thực thể và ca dao tục ngữ.</p>
  </div>

  <div class="feature-card">
    <div class="card-icon">🎙️</div>
    <h3>Real-time Voice AI & 6 Thanh Điệu</h3>
    <p>Nhận diện âm thanh (ASR) và tổng hợp giọng nói (TTS) tối ưu hóa riêng cho đặc trưng ngữ âm học và hệ thống 6 thanh điệu tiếng Việt.</p>
  </div>
</div>

---

## 📊 Kết Quả Đo Thực Tế Tái Lập (Direct Measurements — Zero-GPU CPU)

> Đo trực tiếp ngày `2026-11-27` trên máy văn phòng **Intel Celeron N3050 @ 1.60 GHz (2 lõi CPU, 3.2 GiB RAM, không GPU)**, Linux `x86_64`, Python `3.14.4` qua lệnh `vai benchmark --compare` và `vai benchmark --suite legal-temporal`.

| Chỉ số Đo lường | Giá trị Thực đo (`v31.2.0`) | Phương pháp / Tập dữ liệu |
| :--- | :---: | :--- |
| **Khởi động lạnh (Cold-start)** | **24.68 ms** | Tiến trình Python cô lập (`import vietnamese_ai`) |
| **RAM cư trú khi nạp lõi (VmRSS)** | **12.82 MB** | `/proc/self/status` `VmRSS` sau import lõi |
| **RAM khi chạy Legal RAG (100 điều khoản)** | **37.39 MB** | `/proc/self/status` `VmRSS` sau lập chỉ mục & truy vấn |
| **Thư viện phụ thuộc lõi bắt buộc** | **1 gói (`numpy`)** | Các nhánh NLP/ML/DL/API đều tách `optional-dependencies` |
| **Tốc độ Tách từ Tiếng Việt (Rust Trie)** | **354,047 từ/s** | `BoTachTuTiengVietSieuToc` trên 100 điều khoản mở |
| **Truy hồi Pháp lý mở (100 điều / 35 QA)** | **Hit@1 = 100.0% \| MRR = 1.000** | `RAGPipeline` (`onnx_viet`) trên `legal_vn_qa.jsonl` |
| **ViLegal-TempBench (160 điều / 68 QA)** | **R@3 = 0.992 \| MRR@10 = 0.992 \| TVA = 1.000 \| TCA = 12/12** | `vai benchmark --suite legal-temporal` (R@5 = `1.000`, Abstention F1 = `1.000`) |
| **Xác thực Mật mã (PoI & Freivalds)** | **RFC 5114 (2048-bit $p$, 256-bit $q$)** | Schnorr Fiat-Shamir + Merkle RFC 6962 + Freivalds 1-layer |

---

## 🚀 Cài Đặt & Sử Dụng Nhanh

### Cài đặt qua pip
```bash
pip install vietnamese-ai
```

### 1. Phân loại Học máy Cơ bản
```python
from vietnamese_ai import PhanLoai, DuLieuMau

# Tải tập dữ liệu mẫu và huấn luyện trong 3 dòng code
X, y = DuLieuMau.phan_loai_van_ban()
mo_hinh = PhanLoai(thuat_toan="rung_ngau_nhien")
mo_hinh.huan_luyen(X, y)

print("Dự đoán nhãn:", mo_hinh.du_doan(["Dịch vụ khách hàng rất tuyệt vời!"]))
```

### 2. Kho Tri Thức `.vaidb`, RAG Pháp Lý & Rà Soát Hợp Đồng
```python
from vietnamese_ai import BoSoSanhRaSoatHopDong, DoThiThamChieuPhapLy, RAGPipeline

rag = RAGPipeline(
    backend="vaidb",
    duong_dan_vaidb="cong_ty.vaidb",
    che_do_nhung="onnx_viet",
    che_do_rerank="onnx_viet",
    che_do_hanh_chinh=True,
)
rag.them_tep("hop_dong_dat_dai.docx", che_do_hanh_chinh=True)
ket_qua = rag.hoi("Quyền chuyển nhượng sổ đỏ được quy định thế nào?")
print(ket_qua["tra_loi"])
rag.dong()

print(DoThiThamChieuPhapLy().truy_van_van_ban_lien_quan("45/2013/QH13", chi_lay_con_hieu_luc=True))
bao_cao = BoSoSanhRaSoatHopDong().so_sanh_va_ra_soat("hop_dong_doi_tac.docx", "hop_dong_chuan.docx")
print("Điểm an toàn:", bao_cao["diem_an_toan_phap_ly"], "| Rủi ro:", bao_cao["xep_loai_rui_ro"])
```

### 3. Suy luận Mật mã Minh bạch & Bản đồ Minh bạch 3 Tầng
```python
from vietnamese_ai import KiemToanMatMa, bao_cao_ranh_gioi_he_thong

bao_cao = KiemToanMatMa.chay_kiem_toan_tu_dong()
assert bao_cao["tong_quan_dat"] is True
print(f"Đạt {bao_cao['so_hang_muc_dat']}/{bao_cao['tong_hang_muc']} hạng mục mật mã {bao_cao['chuan_mat_ma']}!")
print("Tổng quan 3 tầng:", bao_cao_ranh_gioi_he_thong()["tong_quan"])
```

---

## 📚 Hệ Thống Tài Liệu Chi Tiết

- 🚀 [Hướng Dẫn Cài Đặt](getting-started/installation.md)
- ⚡ [Khởi Động Nhanh](getting-started/quickstart.md)
- 🎯 [Trải Nghiệm 5 Phút (`vai demo`), Minh Bạch 3 Tầng (`vai status-map`) & Phát Hành PyPI](guides/five-minute-dx-and-release.md)
- ☁️ [Tinh Chỉnh GPU Từ Xa Đa Nhà Cung Cấp (Modal / SSH / Local Flywheel)](guides/remote-gpu-flywheel.md)
- 💻 [LLM Tiếng Việt Zero-GPU Trên CPU (`vai pull` & `vai run`)](guides/edge-cpu-llm-runtime.md)
- 📜 [RAG Pháp Lý 100 Điều Khoản, `.vaidb`, Ngữ Nghĩa ONNX & Đồ Thị Luật](guides/vietnamese-rag.md)
- 📑 [Rà Soát Hợp Đồng Redline, Chat Workspace, MCP IDE & Zalo/Telegram](guides/workbench-mcp-connectors.md)
- 🌐 [Swarm CPU Sharding qua LAN](guides/swarm-cpu.md)
- 🛡️ [Verifiable AI & Proof of Inference](guides/verifiable-ai.md)
- 🔒 [Phòng Thủ Dữ Liệu & Khiên Đa Lớp](guides/data-defense.md)
- 🤖 [Tác Tử Đồ Thị Trạng Thái (StateGraph)](guides/state-graph-agents.md)
- 📊 [Hồ Dữ Liệu 16 Bộ Tập Tiếng Việt](guides/datalake-16-datasets.md)
- 📈 [Bảng Đo Lường So Sánh Toàn Diện](guides/comparative-benchmark.md)
- 📝 [Nhật Ký Thay Đổi (Changelog)](changelog.md)
