# Bảng Đo Lường & Benchmark Tái Lập (Reproducible Quantitative Benchmarks)

Tài liệu này mô tả phương pháp đo lường minh bạch, dễ tái lập cho **Vietnamese AI Framework** (`v31.1.0`) bằng các công cụ chuẩn (tiến trình Python cô lập đo cold-start & `/proc/self/status` `VmRSS`, `time.perf_counter`, `tracemalloc`, `resource`, `cProfile`, `ThreadPoolExecutor` và TCP Socket thực tế).

---

## 1. Cách Tái Lập Toàn Bộ Benchmark Trong 1 Lệnh

### Chạy cục bộ
```bash
# Suite mở rộng (Core + LLM Concurrency/Quantized MatMul + Stability + Swarm LAN + 5-Domain VN)
python benchmarks/run_benchmarks.py \
  --suite extended \
  --seed 42 \
  --profile \
  --output results/benchmark.json \
  --markdown results/benchmark.md

# Suite chiều sâu 6 trụ cột mới (FST/Viterbi, HNSW .vaidb, Rust Kernels, Freivalds, SimPO/EvalGate, Track Changes)
python benchmarks/run_benchmarks.py \
  --suite deep \
  --seed 42 \
  --output results/deep_benchmark.json \
  --markdown results/deep_benchmark.md
```

### Chạy trong Docker cô lập
```bash
docker build -f benchmarks/Dockerfile.benchmark -t vietnamese-ai-bench:31.0.4 .
docker run --rm -v "$(pwd)/results:/results" vietnamese-ai-bench:31.0.4
```

---

## 2. Phạm Vi Đo Lường Chuẩn Sản Xuất (`BoDoLuongDoiDau`)

| Hạng mục | Phương thức / Suite | Chỉ số đo thực tế |
|---|---|---|
| **Core Micro-Architecture** | `chay_benchmark_toan_dien()` (`--suite core`) | Cold-start (ms), RAM tiêu thụ (MB), tốc độ tách từ (từ/s), Legal RAG Hit@1 / Hit@3 / MRR |
| **LLM Inference & Quantized CPU** | `do_hieu_nang_suy_luan_llm()` (`--suite extended`) | Độ trễ `p50`/`p95`/`p99` (ms), thông lượng `req/s` & `tokens/s` dưới tải đa luồng, tốc độ `nhan_ma_tran_luong_tu` trên `Q4_0`, `Q4_K_M`, `Q8_0` theo kích thước tầng `256`/`512`/`1024` |
| **Độ ổn định dài hạn (Long-running)** | `do_do_on_dinh_tai_nguyen()` (`--suite extended`) | Biến thiên Python Heap (`tracemalloc`), RSS tiến trình (`resource`), thời gian CPU, cảnh báo rò rỉ bộ nhớ |
| **Mở rộng cụm Swarm CPU & LAN** | `do_kha_nang_mo_rong_swarm()` (`--suite extended`) | So sánh cụm giả lập (`khoi_tao_cum_gia_lap`) 1→2→4 node (có/không zkML RFC 5114) và cụm TCP Socket thực tế (`them_node_mang_vat_ly`, FP32 vs FP16) |
| **Tổng quát đa miền tiếng Việt** | `do_da_dang_mien_tieng_viet()` (`--suite extended`) | 5 miền (Pháp lý, Tài chính, Y tế, Công nghệ AI, Thương mại điện tử): Tokenizer, POS VLSP, NER Recall, Sentiment Accuracy, Multi-Domain RAG |
| **Chiều sâu 6 Trụ cột Lõi** | `do_chieu_sau_loi_he_thong()` (`--suite deep`) | Phonological FST & Viterbi DAG, Persistent HNSW `.vaidb` (Recall@5 + Metadata Filter), Rust Zero-Copy `Q5_K`/`Q6_K`/`RMSNorm`/`SwiGLU`, Freivalds $O(k \cdot n^2)$ + Byzantine detection, SimPO + Eval Gate 95% CI, Point-in-Time Legal & Word Track Changes OOXML |

---

## 3. Kết Quả Đo Thực Tế Suite Chiều Sâu (`--suite deep --seed 42`)

*Số liệu đo trực tiếp ngày `2026-11-25` trên môi trường tham chiếu: Linux `x86_64` (2 logical cores, SIMD `sse4_2`), Python `3.14.4`, `vietnamese_ai_rust` (abi3-py39, Zero-Copy `PyBytes`), `seed=42`.*

### 3.1. Rust Native Kernels (Zero-Copy `PyBytes`) vs NumPy Thuần (`(16, 512) x (512, 64)`)

| Kernel | Rust Native | NumPy Thuần | Tăng tốc | Sai số tuyệt đối tối đa |
|:---|---:|---:|---:|---:|
| `Q5_K MatMul` | `5.53 ms` | `25.05 ms` | **4.53x** | `1.83e-03` |
| `Q6_K MatMul` | `4.31 ms` | `17.82 ms` | **4.14x** | `2.08e-03` |
| `RMSNorm f32` | `0.23 ms` | `1.43 ms` | **6.33x** | `1.43e-06` |
| `SwiGLU f32` | `0.66 ms` | `2.45 ms` | **3.72x** | `0.00e+00` |

### 3.2. NLP Tiếng Việt, Xác Thực Freivalds, `.vaidb` HNSW, SimPO & Pháp Lý

| Trụ cột | Chỉ số đo trực tiếp | Ghi chú kỹ thuật |
|:---|:---|:---|
| **NLP FST & Viterbi DAG (Rust)** | Chuẩn hóa FST: **584,893 từ/s** \| Viterbi DAG: **269,914 từ/s** \| MaxMatch: **200,875 từ/s** | Giải nhập nhằng `"cơ quan nhà nước pháp quyền"` chính xác 100%, đồng nhất Rust & Python. |
| **Xác thực Freivalds (`256x256`, `k=10`)** | **22.89 ms** vs Full MatMul `25.75 ms` (**1.12x**), phát hiện giả mạo 1 phần tử: **100.0%** | Hệ số tăng tốc $O(n^3)/O(k \cdot n^2)$ mở rộng tuyến tính theo kích thước ẩn $n$ của mô hình LLM. |
| **Kho `.vaidb` Persistent HNSW (`N=600, D=64`)** | Recall@5: **100.0%** \| Nạp lại từ SQLite: `64.5 ms` (`500 KB`) \| Truy vấn HNSW: `11.6 ms` | Ở tập nhỏ (`N=600`), Linear Scan BLAS chỉ mất `~1.0 ms`; đồ thị HNSW bền vững phù hợp khi cần lưu chỉ mục phân cấp kết hợp lọc metadata trên kho lớn. |
| **SimPO & Eval Gate Mở rộng** | SimPO tiết kiệm **50.0% VRAM `ref_model`** \| An toàn tường lửa mở rộng (gồm tiếng Việt không dấu): **1.00** \| Paired Bootstrap 95% CI: `[+0.50, +0.50]` | Khẳng định cải thiện có ý nghĩa thống kê trước khi promote mô hình. |
| **Pháp lý Point-in-Time & Track Changes** | Point-in-Time (`Điều 19 123/2020/NĐ-CP`): **0.22 ms/lần** \| Xuất `.docx` Track Changes + Comments: **30.78 ms** (`3,043 bytes`) | Phát hiện tự động tách/gộp điều khoản và sinh chuẩn OOXML `<w:del>`, `<w:ins>`, `word/comments.xml`. |

---

## 4. Minh Bạch Về Số Liệu & Phương Pháp Đo (Measurement Integrity)

- Mọi chỉ số của **Vietnamese AI Framework** trong báo cáo JSON/Markdown đều được **đo trực tiếp trong thời gian chạy** trên chính phần cứng người dùng (tiến trình Python cô lập cho cold-start & `VmRSS` thực tế từ `/proc/self/status`) và gắn kèm siêu dữ liệu môi trường (OS, Python, kiến trúc CPU, cờ SIMD, `seed`, thời điểm UTC).
- `BoDoLuongDoiDau` **không còn sử dụng bảng số liệu tham chiếu tĩnh** cho bất kỳ thư viện ngoài nào. Khi chạy `vai benchmark --compare`, framework chỉ đo đối sánh trực tiếp cho `langchain` / `llama_index` nếu chúng thực sự được cài đặt trong cùng môi trường; nếu chưa cài đặt, báo cáo ghi rõ trạng thái `chua_cai_dat`.
- Kết quả đo cốt lõi thực tế ngày `2026-11-27` trên **Intel Celeron N3050 @ 1.60 GHz (2 lõi CPU, 3.2 GiB RAM, Linux `x86_64`, Python `3.14.4`)** với lệnh `vai benchmark --compare`:
  - Cold-start `import vietnamese_ai` (subprocess cô lập): **`24.68 ms`**
  - RAM cư trú khi nạp lõi (`VmRSS`): **`12.82 MB`**
  - RAM cư trú khi lập chỉ mục & truy vấn 100 điều khoản pháp lý (`VmRSS`): **`37.39 MB`**
  - Tốc độ tách từ tiếng Việt (`BoTachTuTiengVietSieuToc` + Rust Trie trên 100 điều khoản): **`354,047.7 từ/s`**
  - Truy hồi pháp lý mở (`legal_vn_corpus.jsonl` 100 điều khoản / `legal_vn_qa.jsonl` 35 câu hỏi, `RAGPipeline onnx_viet`): **Hit@1 = `100.0%`**, **Hit@3 = `100.0%`**, **MRR = `1.000`**.
- Bảng phụ thuộc cốt lõi của `vietnamese-ai`: yêu cầu duy nhất `numpy>=1.20.0,<3.0.0` (các nhóm `[vietnamese-nlp]`, `[classic-ml]`, `[legal-rag]`, `[dl]`, `[rag]`, `[api]` là tùy chọn).
